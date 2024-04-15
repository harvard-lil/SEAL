import os
import click
import traceback
import json
import csv

from context import IVANContext

OUTPUT_DIR = os.path.join(IVANContext.data_dir, "taxonomy-output")
""" Output path for this command. """


@click.command("process-batch")
@click.pass_obj
@click.option(
    "--batch-ids",
    required=True,
    type=str,
    help="Coma-separated list of OpenAI batch operation identifiers to retrieve and process.",
)
def process_batch(ctx: IVANContext, batch_ids: str):
    """
    In this experiment, GPT-X is used to generate an alignment taxonomy of the Anthropic/hh-rlhf dataset.
    This command allows for pulling a series of completed batches from the OpenAI API and:
    - Pull the raw results of the various batches into a single JSONL file
    - Validate and save processed results in a single CSV. This format allows for easier processing and for associating analysis results with their original prompts.

    This command allows for patching invalid entries on the fly by re-running the analysis against gpt-4-turbo (optional).
    """
    from taxonomy import ANALYSIS_KEYS, get_taxonomy_analysis_prompt, run_taxonomy_analysis

    batch_ids = batch_ids.split(",")

    output_jsonl_filepath = os.path.join(OUTPUT_DIR, f"{ctx.datetime_slug}.jsonl")
    output_csv_filepath = os.path.join(OUTPUT_DIR, f"{ctx.datetime_slug}.csv")

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #
    # For each batch id:
    # - Pull batch and check status
    # - Download JSONL file
    #
    for batch_id in batch_ids:
        batch = None
        jsonl_file = None

        #
        # Pull batch info and check status
        #
        click.echo(f"{batch_id} - Pulling info from batch")

        try:
            batch = ctx.openai_client.batches.retrieve(batch_id=batch_id)
            assert batch
        except Exception:
            click.echo(traceback.format_exc())
            click.echo(f"Error while pulling info about batch {batch_id}")

        try:
            assert batch.status == "completed"
        except AssertionError:
            click.echo(f"Batch {batch_id} is not complete ({batch.status})")
            exit(1)

        #
        # Download associated JSONL file
        #
        click.echo(f"{batch_id} - Pulling data from batch")

        try:
            with open(output_jsonl_filepath, "a+") as jsonl_file:
                jsonl_file.write(ctx.openai_client.files.retrieve_content(batch.output_file_id))
        except Exception:
            click.echo(traceback.format_exc())
            click.echo(f"Error while pulling data from batch {batch_id}")
            exit(1)

        click.echo(f"{batch_id} - Data added to {os.path.basename(output_jsonl_filepath)}")

    #
    # Validate JSONL and write to CSV
    #
    with open(output_jsonl_filepath, "r+") as jsonl_file:
        with open(output_csv_filepath, "w+") as csv_file:

            # Write CSV headers
            csv_headers = ["row_id", "preference", "text"]
            csv_headers += ANALYSIS_KEYS.keys()
            writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
            writer.writeheader()

            # Read JSON lines, parse, validate, and add to CSV
            for jsonl in jsonl_file:
                try:
                    completion = json.loads(jsonl)

                    # Check that we have a row_id and preference
                    row_id, preference = completion["custom_id"].split(".")
                    row_id = int(row_id)

                    assert row_id is not None
                    assert preference in ["chosen", "rejected"]

                    # Load analysis data and validate format
                    analysis_data = json.loads(
                        completion["response"]["body"]["choices"][0]["message"]["content"]
                    )

                    # If analysis data is invalid, offer to patch on the fly
                    for attempt in range(0, 5):

                        if set(analysis_data.keys()) == set(ANALYSIS_KEYS.keys()):
                            break

                        confirm_prompt = f"#{row_id}.{preference} is invalid. "
                        confirm_prompt += f"Run API call again? (Attempt {attempt+1} of 5)"

                        if click.confirm(confirm_prompt):
                            click.echo(f"#{row_id}.{preference} is being patched ...")

                            analysis_data = run_taxonomy_analysis(
                                ctx=ctx,
                                prompt=get_taxonomy_analysis_prompt(
                                    text=ctx.rlhf_dataset[row_id][preference]
                                ),
                            )
                        else:  # User rejected patching
                            break

                    assert set(analysis_data.keys()) == set(ANALYSIS_KEYS.keys())

                    # Save to CSV
                    csv_entry = {
                        "row_id": row_id,
                        "preference": preference,
                        "text": ctx.rlhf_dataset[row_id][preference],
                    }

                    csv_entry = csv_entry | analysis_data

                    writer.writerow(csv_entry)
                except Exception:
                    click.echo(traceback.format_exc())
                    click.echo("Error while validating JSON output. Skipping line.")
                    click.echo(f"Data:\n{jsonl}")
                    continue

    click.echo(f"Batches merged into {os.path.basename(output_csv_filepath)}")
