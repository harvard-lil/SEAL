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
def run_control_batch(ctx: IVANContext, model: str, limit: int):
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
    # Pick random entries to analyze
    #
    if not limit:
        limit = len(ctx.rlhf_dataset)

    limit = int(limit)
    total_rows = len(ctx.rlhf_dataset)

    for row_id in random.sample(range(total_rows), limit):
        for preference in ["chosen", "rejected"]:
            new_entry = dict.fromkeys(output_format)
            new_entry["row_id"] = row_id
            new_entry["preference"] = preference
            new_entry["text"] = ctx.rlhf_dataset[row_id][preference]
            output.append(new_entry)

    #
    # Analyze entries
    #
    for entry in output:
        row_id = entry["row_id"]
        preference = entry["preference"]

        click.echo(f"#{row_id}.{preference} Taxonomsy analysis against {model}")

        try:
            prompt = get_taxonomy_analysis_prompt(entry["text"])
            analysis_data = run_taxonomy_analysis(ctx, prompt, model)

            print(analysis_data)

            assert set(analysis_data.keys()) == set(ANALYSIS_KEYS.keys())

            for key, value in analysis_data.items():
                entry[key] = value
        except Exception:
            click.echo(traceback.format_exc())
            click.echo(f"#{row_id}.{preference} Could not process entry, skipping.")

    #
    # Write to CSV
    #
    with open(output_filepath, "a+") as file:
        writer = csv.DictWriter(file, fieldnames=output_format)

        for entry in output:
            writer.writerow(entry)

    click.echo(f"{os.path.basename(output_filepath)} written to disk.")
