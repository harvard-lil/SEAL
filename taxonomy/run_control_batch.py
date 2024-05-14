import os
import csv
import traceback
import random

import click
from slugify import slugify

from context import IVANContext

OUTPUT_DIR = os.path.join(IVANContext.data_dir, "taxonomy-control")
""" Output path for this command. """


@click.command("run-control-batch")
@click.pass_obj
@click.option(
    "--model",
    type=str,
    default="gemma:7b-instruct-fp16",
    help="Open Source model to run the control batch against.",
)
@click.option(
    "--limit",
    default=1608,
    type=int,
    help="If set and > 0, only processes up to X rows from the RLHF dataset. 1 row = 2 requests.",
)
@click.option(
    "--set-row-ids",
    type=str,
    default=None,
    required=False,
    help="Coma-separated list of row IDs. Will override limit and random selection of rows if set.",
)
def run_control_batch(ctx: IVANContext, model: str, limit: int, set_row_ids: str):
    """
    In this experiment, GPT-X is used to generate an alignment taxonomy of the Anthropic/hh-rlhf dataset.
    This command picks random entries from the dataset and runs the same experiment against an open source model, as a way to generate control data.
    """
    from taxonomy import get_taxonomy_analysis_prompt, run_taxonomy_analysis, ANALYSIS_KEYS

    output_format = ["row_id", "preference", "text"] + list(ANALYSIS_KEYS.keys())
    output = []  # Entries to process. List of output_format dictionaries.
    output_filepath = os.path.join(OUTPUT_DIR, f"{ctx.datetime_slug}-{slugify(model)}-control.csv")

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #
    # Initialize CSV with headers
    #
    with open(output_filepath, "w+") as file:
        writer = csv.DictWriter(file, fieldnames=output_format)
        writer.writeheader()

    #
    # Pick entries to analyze
    #
    if not limit:
        limit = len(ctx.rlhf_dataset)

    limit = int(limit)
    total_rows = len(ctx.rlhf_dataset)
    row_id_range = []

    # Pick entries at random between 0 and --limit if set_row_ids is not set
    if not set_row_ids:
        row_id_range = random.sample(range(total_rows), limit)
    # Parse and validate set_row_ids otherwise
    else:
        try:
            row_id_range = [int(row_id) for row_id in set_row_ids.split(",")]

            for row_id in row_id_range:
                assert type(row_id) is int
                assert row_id > 0 and row_id < total_rows - 1
        except Exception:
            click.echo(traceback.format_exc())
            click.echo("set_row_ids contain invalid values.")
            exit(1)

    for row_id in row_id_range:
        for preference in ["chosen", "rejected"]:
            new_entry = dict.fromkeys(output_format)
            new_entry["row_id"] = row_id
            new_entry["preference"] = preference
            new_entry["text"] = ctx.rlhf_dataset[row_id][preference]
            output.append(new_entry)

    #
    # Analyze entries, reject broken pairs, save to CSV
    #
    with open(output_filepath, "a+") as file:
        writer = csv.DictWriter(file, fieldnames=output_format)
        i = 1
        total_entries = len(row_id_range) * 2
        buffer = []  # Used to save once we have valid pairs and reject broken pairs
        interrupted_row_id = None  # Used to skip processing of broken pairs

        for entry in output:
            row_id = entry["row_id"]
            preference = entry["preference"]

            # Skip if "chosen" could not be processed
            if row_id == interrupted_row_id:
                continue

            click.echo(f"#{row_id}.{preference} Taxonomy analysis w/ {model} ({i}/{total_entries})")

            try:
                prompt = get_taxonomy_analysis_prompt(entry["text"])
                analysis_data = run_taxonomy_analysis(ctx, prompt, model)

                assert set(analysis_data.keys()) == set(ANALYSIS_KEYS.keys())

                for key, value in analysis_data.items():
                    entry[key] = value

                buffer.append(entry)
            except Exception:
                buffer = []
                interrupted_row_id = row_id
                click.echo(traceback.format_exc())
                click.echo(f"#{row_id}.{preference} Could not process entry, skipping pair.")

            # Save buffer to CSV whenever a pair is complete
            if len(buffer) >= 2 and buffer[0]["row_id"] == buffer[1]["row_id"]:
                for entry in buffer:
                    writer.writerow(entry)
                buffer = []

            i += 1

    click.echo(f"{os.path.basename(output_filepath)} written to disk.")
