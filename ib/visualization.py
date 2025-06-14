"""Visualize results."""

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from ib.utils.logging_module import logging
from ib.utils.pipeline import measure_time, resolve_and_expand_path

app = typer.Typer(add_completion=False)


class PlotType(str, Enum):
    LINE = "line"
    BAR = "bar"


def get_exp_name_from_full_name(full_experiment_name: str) -> str:
    """Extract experiment name without date prefix."""
    parts = full_experiment_name.split("_")
    if len(parts) > 1 and len(parts[0]) == 8:  # Date format: YY-MM-DD
        return "_".join(parts[1:])
    return full_experiment_name


def parse_results_from_directory(
    results_dir: Path,
    experiment_name: str,
) -> pd.DataFrame:
    """Parse results from a directory containing JSON files."""

    results = []

    if not results_dir.exists():
        logging.warning(f"Results directory does not exist: {results_dir}")
        return pd.DataFrame()

    # Find all JSON files matching the pattern
    json_files = list(
        results_dir.glob(f"results_{experiment_name}_epoch_*_res_1024.json")
    )

    if not json_files:
        logging.warning(f"No result files found in {results_dir}")
        return pd.DataFrame()

    for json_file in json_files:
        try:
            # Extract epoch from filename
            filename = json_file.stem
            parts = filename.split("_")
            epoch_idx = parts.index("epoch") + 1
            epoch = int(parts[epoch_idx])

            # Load metrics from JSON
            with open(json_file, "r") as f:
                metrics = json.load(f)

            # Add each metric as a row
            for metric_name, metric_value in metrics.items():
                results.append(
                    {
                        "experiment": experiment_name,
                        "epoch": epoch,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                    }
                )

        except (ValueError, KeyError, json.JSONDecodeError) as e:
            logging.warning(f"Error parsing {json_file}: {e}")
            continue

    return pd.DataFrame(results)


def create_metric_plots(
    data: pd.DataFrame,
    metrics_to_show: list[str],
    output_dir: Path,
) -> None:
    """Create separate line plots for each metric showing metric vs epoch."""
    # Set up seaborn style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    for metric in metrics_to_show:
        # Filter data for current metric
        metric_data = data[data["metric_name"] == metric].copy()

        if metric_data.empty:
            logging.warning(f"No data found for metric: {metric}")
            continue

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create line plot with different experiments as different lines
        sns.lineplot(
            data=metric_data,
            x="epoch",
            y="metric_value",
            hue="experiment",
            marker="o",
            linewidth=2,
            markersize=6,
        )

        # Customize the plot
        plt.title(
            f'{metric.replace("_", " ").replace("/", " / ").title()} vs Epoch',
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel(metric.replace("_", " ").replace("/", " / ").title(), fontsize=14)
        plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # Improve layout
        plt.tight_layout()

        # Save the plot
        plot_filename = f"{metric.replace('/', '_').replace('_', '-')}-vs-epoch.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved plot: {plot_path}")

        # Show the plot
        plt.show()
        plt.close()


def create_metric_barplots(
    data: pd.DataFrame,
    metrics_to_show: list[str],
    output_dir: Path,
) -> None:
    """Create separate bar plots for each metric showing last epoch results."""
    # Set up seaborn style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    # Get last epoch data for each experiment
    last_epoch_data = []
    for experiment in data["experiment"].unique():
        exp_data = data[data["experiment"] == experiment]
        last_epoch = exp_data["epoch"].max()
        last_epoch_exp_data = exp_data[exp_data["epoch"] == last_epoch]
        last_epoch_data.append(last_epoch_exp_data)

    if not last_epoch_data:
        logging.warning("No last epoch data found")
        return

    last_epoch_df = pd.concat(last_epoch_data, ignore_index=True)

    for metric in metrics_to_show:
        # Filter data for current metric
        metric_data = last_epoch_df[last_epoch_df["metric_name"] == metric].copy()

        if metric_data.empty:
            logging.warning(f"No data found for metric: {metric}")
            continue

        # Create the plot
        plt.figure(figsize=(12, 8))

        # Create bar plot
        sns.barplot(
            data=metric_data,
            x="experiment",
            y="metric_value",
            hue="experiment",
            palette="viridis",
            legend=False,
        )

        # Customize the plot
        plt.title(
            f'{metric.replace("_", " ").replace("/", " / ").title()} - Last Epoch Results',
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Experiment", fontsize=14)
        plt.ylabel(metric.replace("_", " ").replace("/", " / ").title(), fontsize=14)
        plt.xticks(rotation=45, ha="right")
        plt.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        ax = plt.gca()
        for container in ax.containers:
            ax.bar_label(container, fmt="%.4f", fontsize=10)

        # Improve layout
        plt.tight_layout()

        # Save the plot
        plot_filename = (
            f"{metric.replace('/', '_').replace('_', '-')}-last-epoch-bar.png"
        )
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved plot: {plot_path}")

        # Show the plot
        plt.show()
        plt.close()


@app.command(no_args_is_help=True)
@measure_time
def visualization(
    output_dir: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    plot_type: Annotated[PlotType, typer.Option()] = PlotType.LINE,
) -> None:
    """Visualize results."""
    # TODO(oleg): add normal input arguments instead of hardcoded values.

    logging.stage("Running visualization.")

    METRICS_TO_SHOW = [
        "metrics_main/lpips",
        # "metrics_main/completeness_high_freq_0002",
        "metrics_main/completeness_high_freq_0003",
        # "metrics_main/completeness_low_freq_0002",
        "metrics_main/completeness_low_freq_0003",
        "curvature_mean",
        "curvature_median",
    ]
    EXPERIMENTS = [
        "25-05-12_siren_newdata_o5",
        "25-05-13_siren_newdata_o15",
        "25-05-12_siren_newdata_o30_linter300",
        "25-05-14_siren_newdata_o45_linter200",
        "25-05-29_finer_o20_linter500",
        # "25-06-04_hosc_init",
        # "25-06-01_double_siren_pe_mlp",
        # "25-05-21_relu_pe_init_linter100",
        # "25-05-31_staf_init",
        "25-05-26_attn_ff_linter100_o20",
    ]

    # Collect all data
    all_data = []

    for experiment_full_name in EXPERIMENTS:
        results_dir = Path(
            f"/home/magnes/outputs/{experiment_full_name}/latest/results/"
        )
        experiment_name = get_exp_name_from_full_name(experiment_full_name)
        logging.info(
            f"Processing experiment: {experiment_full_name} -> {experiment_name}"
        )

        df = parse_results_from_directory(results_dir, experiment_name)
        if not df.empty:
            all_data.append(df)

    if not all_data:
        logging.error("No data found in any experiment directories!")
        return

    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logging.info(
        f"Loaded {len(combined_data)} data points from {combined_data['experiment'].nunique()} experiments"
    )

    # Print summary
    for experiment in combined_data["experiment"].unique():
        exp_data = combined_data[combined_data["experiment"] == experiment]
        epochs = sorted(exp_data["epoch"].unique().tolist())
        logging.info(f"  {experiment}: epochs {epochs}")

    # Create plots
    output_dir.mkdir(parents=True, exist_ok=True)

    if plot_type is PlotType.LINE:
        logging.stage("Creating line plots (metric vs epoch)...")
        create_metric_plots(combined_data, METRICS_TO_SHOW, output_dir)
    elif plot_type is PlotType.BAR:
        logging.stage("Creating bar plots (last epoch comparison)...")
        create_metric_barplots(combined_data, METRICS_TO_SHOW, output_dir)

    logging.stage(f"Visualization results saved to {output_dir}")


if __name__ == "__main__":
    app()
