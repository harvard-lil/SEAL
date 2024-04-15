import os
import json
import locale
import traceback

import click
import tiktoken

from context import IVANContext

OUTPUT_DIR = os.path.join(IVANContext.data_dir, "taxonomy-input")
""" Output path for this command. """

MAX_FILESIZE = 100 * 1000 * 1000
""" Max size file size for every batch file, in bytes. Must allow for "wiggle room". """


@click.command("create-batch")
@click.pass_obj
@click.option(
    "--limit",
    default=0,
    type=int,
    help="If set and > 0, only processes up to X rows from the RLHF dataset. 1 row = 2 requests.",
)
def create_batch(ctx: IVANContext, limit: int):
    """
    In this experiment, GPT-X is used to generate an alignment taxonomy of the Anthropic/hh-rlhf dataset.
    This command generates a series of batch files that can be used by OpenAI's API to process a large amount of requests efficiently.
    The command will also perform a cost estimate and ask for confirmation before uploading the files for processing.
    """
    from taxonomy import get_taxonomy_analysis_prompt, TARGET_MODEL, ANALYSIS_KEYS

    tokenizer = tiktoken.encoding_for_model(TARGET_MODEL)
    locale.setlocale(locale.LC_ALL, "en_US")

    jsonl_filepaths = []
    batch_ids = []

    input_tokens = 0
    output_tokens = 0

    openai_file_upload = None
    openai_batch = None

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #
    # For up to X entries:
    # - Generate batch request and save it to files of up to ~MAX_FILESIZE
    # - Estimate input and output token counts
    #
    i = 0
    current_file = None

    for entry in ctx.rlhf_dataset:

        if limit and i >= limit:
            click.echo(f"Reached --limit ({limit}). Interrupting")
            break

        # The batch is broken into chunks of ~MAX_FILESIZE bytes.
        if not current_file or current_file.tell() >= MAX_FILESIZE:
            jsonl_filepath = os.path.join(OUTPUT_DIR, f"{ctx.datetime_slug}-{i}-input.jsonl")
            jsonl_filepaths.append(jsonl_filepath)
            current_file = open(jsonl_filepath, "w+")

        for preference in ["chosen", "rejected"]:
            task = {}
            task["custom_id"] = f"{entry['row_id']}.{preference}"
            task["method"] = "POST"
            task["url"] = "/v1/chat/completions"
            task["body"] = {
                "model": TARGET_MODEL,
                "temperature": 0.0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": get_taxonomy_analysis_prompt(entry[preference]),
                    }
                ],
            }

            # Write request to batch file
            current_file.write(json.dumps(task) + "\n")

            # Estimate input token count
            input_tokens += len(tokenizer.encode(task["body"]["messages"][0]["content"]))

            # Estimate output token count
            dummy_response_object = {key: True for key in ANALYSIS_KEYS.keys()}
            dummy_response_object["topics"] = ["Foo", "Bar", "Baz"]
            dummy_response_object["sentiment"] = "neutral"

            output_tokens += len(tokenizer.encode(json.dumps(dummy_response_object)))

        i += 1

    current_file.close()

    #
    # Print stats and cost estimate
    #
    input_cost = ((input_tokens / 1_000_000) * 10.0) / 2
    output_cost = ((output_tokens / 1_000_000) * 30.0) / 2

    click.echo(f"Batch input saved to disk.")
    click.echo(f"Estimated input tokens: {input_tokens}")
    click.echo(f"Estimated output tokens: {output_tokens}")
    click.echo(f"Estimated input cost: {locale.currency(input_cost, grouping=True)}")
    click.echo(f"Estimated output cost: {locale.currency(output_cost, grouping=True)}")
    click.echo(f"Estimated total cost: {locale.currency(input_cost + output_cost, grouping=True)}")

    #
    # Ask for confirmation before creating batches
    #
    if not click.confirm("Send to OpenAI Batch API for processing?"):
        click.echo("Cancelled.")
        exit(0)

    for filepath in jsonl_filepaths:
        try:
            batch_id = upload_file_and_create_batch(ctx, filepath)
            assert batch_id
            batch_ids.append(batch_id)
        except Exception:
            click.echo(traceback.format_exc())
            exit(1)

    #
    # Print list of batch_ids
    #
    click.echo("OpenAI batch operations created:")
    for batch_id in batch_ids:
        click.echo(f"- {batch_id}")


def upload_file_and_create_batch(ctx: IVANContext, input_filepath: str) -> str:
    """
    Sends a batch file to OpenAI, starts a batch operation and returns the batch id.
    """
    openai_file_upload = None
    openai_batch = None

    # Send file to Open AI
    click.echo(f"Sending {os.path.basename(input_filepath)} to OpenAI.")

    with open(input_filepath, "rb+") as file:
        try:
            openai_file_upload = ctx.openai_client.files.create(file=file, purpose="batch")
        except Exception as err:
            click.echo(f"Error while uploading {os.path.basename(input_filepath)} to OpenAI.")
            raise err

    # Create batch
    click.echo(f"Creating OpenAI batch for {os.path.basename(input_filepath)}.")

    try:
        openai_batch = ctx.openai_client.batches.create(
            input_file_id=openai_file_upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        assert openai_batch.id
    except Exception as err:
        click.echo(f"Error while creating OpenAI batch for {os.path.basename(input_filepath)}.")
        raise err

    return openai_batch.id
