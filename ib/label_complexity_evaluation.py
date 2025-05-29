"""Command-line tool for evaluating label complexity using Fourier Frequency metric."""

import json
from pathlib import Path
from typing import Annotated

import typer

from ib.metrics.label_complexity import LabelComplexity
from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


@app.command(no_args_is_help=True)
@measure_time
def evaluate_labels(
    input_path: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
) -> None:
    """Evaluate complexity of each label's subcloud using Fourier Frequency metric.
    
    Example:
    uv run label_complexity_evaluation \
        --input-path=<path_to_data>/pc_aligned_filtered_trimmed_both_01.ply
    """

    logging.stage("Running label complexity evaluation.")
    logging.info(f"Input pointcloud: {input_path}")

    # Initialize the evaluator
    evaluator = LabelComplexity()

    # Evaluate complexity
    results = evaluator.evaluate_from_file(input_path)

    logging.info("Label complexity results:")
    for label_name, complexity in results.items():
        logging.info(f"  {label_name}: {complexity}")

    output_path = input_path.parent / f"{input_path.stem}_label_complexity_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    app()
