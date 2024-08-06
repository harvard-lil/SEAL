import os
import csv
import gc
import traceback

import click
import torch
from slugify import slugify

from context import SEALContext

OUTPUT_DIR = os.path.join(SEALContext.data_dir, "compare")
""" Output path for this command. """

RM_MODELS = [
    "OpenAssistant/reward-model-deberta-v3-large-v2",
    "OpenAssistant/reward-model-deberta-v3-large",
    "OpenAssistant/reward-model-deberta-v3-base",
    "OpenAssistant/reward-model-electra-large-discriminator",
]
""" List of models to evaluate. """

RM_MODELS_TRAINED_ON_DATASET = ["OpenAssistant/reward-model-deberta-v3-large-v2"]
""" Subset of models trained on the RLHF dataset analyzed here. """

RM_SCORE_FORMAT = {
    "row_id": None,
    "chosen_score": None,
    "rejected_score": None,
    "agreement": None,
}
""" Data format for RM scores collected by this pipeline. """

STATS_FORMAT = {
    "model_name": "",
    "model_trained_on_dataset": False,
    "rlfh_dataset_name": "",
    "rlhf_entries": "",
    "rlhf_entries_rated": 0,
    "rlhf_entries_rated_percentage": 0,
    "rlhf_entries_agreement": 0,
    "rlhf_entries_agreement_percentage": 0,
}
""" Data format for global statistics generated by this pipeline. """


@click.command("compare")
@click.pass_obj
@click.option(
    "--limit",
    default=0,
    type=int,
    help="If set and > 0, only processes up to X rows from the RLHF dataset.",
)
def compare(ctx: SEALContext, limit: int):
    """
    Evaluates a series of reward models against an RLHF dataset.
    Collects reward scores and assesses whether the RMs align with human preferences.
    """
    results_per_model = {}  # Results indexed by model name

    # Create output dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #
    # Run dataset against each reward model and:
    # - Collect scores in `results_per_model`
    # - Write scores to CSV
    #
    for model_name in RM_MODELS:
        click.echo(f"Evaluating {model_name}")

        csv_filename = f"{ctx.datetime_slug}-{slugify(model_name)}.csv"
        csv_filepath = os.path.join(OUTPUT_DIR, csv_filename)

        with open(csv_filepath, "w+") as csv_file:
            # Write CSV header
            csv_writer = csv.DictWriter(csv_file, fieldnames=RM_SCORE_FORMAT.keys())
            csv_writer.writeheader()

            # Create indexed entry in `results_per_model`
            results_per_model[model_name] = []
            results = results_per_model[model_name]

            # Run evaluation for each chosen / rejected pair
            total = limit if limit > 0 else len(ctx.rlhf_dataset)

            for entry in ctx.rlhf_dataset:
                if limit > 0 and len(results) >= limit:
                    click.echo(f"Reached --limit ({limit}). Interrupting")
                    break

                click.echo(f"#{entry['row_id']} -> {model_name} (sample size: {total})")

                try:
                    # Separate last response from the rest of the text.
                    # Output: {"chosen": [conv, last_response], "rejected": [conv, last_response]}
                    split = {}

                    for type in ["chosen", "rejected"]:
                        text = entry[type]
                        pivot = text.rfind("Assistant:")
                        split[type] = [text[0:pivot], text[pivot:]]

                    if not split:
                        raise Exception(f"#{entry['row_id']} - Could not process row, skipping.")

                    chosen_score = ctx.infer_with_reward_model(
                        model_name,
                        split["chosen"][0],
                        split["chosen"][1],
                    )

                    rejected_score = ctx.infer_with_reward_model(
                        model_name,
                        split["rejected"][0],
                        split["rejected"][1],
                    )

                    result = dict(RM_SCORE_FORMAT)
                    result["row_id"] = entry["row_id"]
                    result["chosen_score"] = chosen_score
                    result["rejected_score"] = rejected_score
                    result["agreement"] = True if chosen_score > rejected_score else False

                    results.append(result)
                    csv_writer.writerow(result)
                except Exception:
                    click.echo(traceback.format_exc())
                    click.echo(f"#{entry['row_id']} Could not process entry, skipping.")
                finally:
                    if ctx.transformers_device.startswith("cuda"):
                        gc.collect()
                        torch.cuda.empty_cache()

    #
    # Generate and saving stats to CSV
    #
    click.echo("Writing stats.csv to disk")

    with open(
        os.path.join(OUTPUT_DIR, f"{ctx.datetime_slug}-stats.csv"),
        "w+",
    ) as csv_file:
        # Write CSV header
        csv_writer = csv.DictWriter(csv_file, fieldnames=STATS_FORMAT.keys())
        csv_writer.writeheader()

        # Generate and save stats for each model tested
        for model_name, results in results_per_model.items():
            stats = dict(STATS_FORMAT)
            stats["rlfh_dataset_name"] = ctx.rlhf_dataset_name
            stats["rlhf_entries"] = len(ctx.rlhf_dataset)
            stats["model_name"] = model_name

            # Was this model trained on Anthropic/hh-rlhf?
            stats["model_trained_on_dataset"] = model_name in RM_MODELS_TRAINED_ON_DATASET

            # How many entries were rated?
            stats["rlhf_entries_rated"] = len(results)

            stats["rlhf_entries_rated_percentage"] = (
                stats["rlhf_entries_rated"] / stats["rlhf_entries"] * 100
            )

            # How many entries were agreement (reward model agrees w/ human preference)
            stats["rlhf_entries_agreement"] = len(
                [result["agreement"] for result in results if result["agreement"]]
            )

            stats["rlhf_entries_agreement_percentage"] = (
                stats["rlhf_entries_agreement"] / stats["rlhf_entries_rated"] * 100
            )

            csv_writer.writerow(stats)

    click.echo("Done.")
