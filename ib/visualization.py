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


class Scene(str, Enum):
    ROOM_1 = "1"  # 8b5caf3398
    ROOM_2 = "2"  # 5fb5d2dbf2
    # ROOM_3 = "3" # 5fb5d2dbf2


def get_exp_name_from_full_name(full_experiment_name: str) -> str:
    """Extract experiment name without date prefix."""
    parts = full_experiment_name.split("_")
    if len(parts) > 1 and len(parts[0]) == 8:  # Date format: YY-MM-DD
        return "_".join(parts[1:])
    return full_experiment_name


def parse_results_from_directory(
    results_dir: Path,
    experiment_name: str,
    resolution: int,
) -> pd.DataFrame:
    """Parse results from a directory containing JSON files."""

    results = []

    if not results_dir.exists():
        logging.warning(f"Results directory does not exist: {results_dir}")
        return pd.DataFrame()

    # Find all JSON files matching the pattern
    json_files = list(
        results_dir.glob(f"results_{experiment_name}_epoch_*_res_{resolution}.json")
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
    metrics_to_show: dict[str, str],
    output_dir: Path,
) -> None:
    """Create separate line plots for each metric showing metric vs epoch."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up seaborn style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)

    for readable_name, metric_name in metrics_to_show.items():
        # Filter data for current metric using the actual metric name
        metric_data = data[data["metric_name"] == metric_name].copy()

        if metric_data.empty:
            logging.warning(f"No data found for metric: {metric_name}")
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
            f"{readable_name} vs Epoch",
            fontsize=16,
            fontweight="bold",
        )
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel(readable_name, fontsize=14)
        plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)

        # Improve layout
        plt.tight_layout()

        # Save the plot using a safe filename based on the human-readable name
        plot_filename = f"{readable_name.replace('/', '_').replace(',', '').replace(' ', '-').lower()}-vs-epoch.png"
        plot_path = output_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logging.info(f"Saved plot: {plot_path}")

        # Show the plot
        plt.show()
        plt.close()


def create_metric_comparison_plot(
    data: pd.DataFrame,
    metrics_to_show: dict[str, str],
    output_dir: Path,
) -> None:
    """Create a grouped bar chart comparing all metrics across different experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_style("whitegrid")

    # Extract the last epoch data for each experiment/model
    logging.info("Extracting last epoch data for each experiment...")
    last_epoch_data = []
    for experiment in data["experiment"].unique():
        exp_data = data[data["experiment"] == experiment]
        last_epoch = exp_data["epoch"].max()
        last_epoch_exp_data = exp_data[exp_data["epoch"] == last_epoch]
        last_epoch_data.append(last_epoch_exp_data)
        logging.info(f"  {experiment}: using epoch {last_epoch}")

    if not last_epoch_data:
        logging.warning("No last epoch data found")
        return

    last_epoch_df = pd.concat(last_epoch_data, ignore_index=True)

    # Filter to only include the metrics we want to compare
    filtered_data = last_epoch_df[
        last_epoch_df["metric_name"].isin(metrics_to_show.values())
    ].copy()

    if filtered_data.empty:
        logging.warning("No data found for any of the specified metrics")
        return

    # Convert internal metric names to readable names
    metric_name_mapping = {v: k for k, v in metrics_to_show.items()}
    filtered_data["readable_metric"] = filtered_data["metric_name"].map(
        metric_name_mapping
    )

    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=filtered_data,
        x="readable_metric",  # Metrics on X-axis
        y="metric_value",  # Metric values on Y-axis
        hue="experiment",  # Different models as different colored bars
        palette="Blues",
    )

    plt.title(
        "Model Comparison",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Model/Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    plot_filename = "model-comparison-last-epoch-only.png"
    plot_path = output_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    logging.info(f"Saved comparison plot: {plot_path}")

    plt.show()
    plt.close()


@app.command(no_args_is_help=True)
@measure_time
def visualization(
    output_dir: Annotated[Path, typer.Option(callback=resolve_and_expand_path)],
    scene: Annotated[
        Scene, typer.Option(help="Scene to visualize (1: Room 1, 2: Room 2)")
    ],
    resolution: int = 1024,
) -> None:
    """Visualize results."""
    # TODO(oleg): add normal input arguments instead of hardcoded values.

    logging.stage("Running visualization.")

    if scene is Scene.ROOM_1:
        METRICS_TO_SHOW_CLASSES_T2P = {
            "Chair": "metrics_labels/chamfer_chair_t2p",
            "Heater": "metrics_labels/chamfer_heater_t2p",
            "Lamp": "metrics_labels/chamfer_lamp_t2p",
            "Laptop stand": "metrics_labels/chamfer_laptop stand_t2p",
            "Socket": "metrics_labels/chamfer_socket_t2p",
            "Wall": "metrics_labels/chamfer_wall_t2p",
            "Window": "metrics_labels/chamfer_window_t2p",
        }
        METRICS_TO_SHOW_CLASSES_GENERAL = {
            "Chair": "metrics_labels/chamfer_chair",
            "Heater": "metrics_labels/chamfer_heater",
            "Lamp": "metrics_labels/chamfer_lamp",
            "Laptop stand": "metrics_labels/chamfer_laptop stand",
            "Socket": "metrics_labels/chamfer_socket",
            "Wall": "metrics_labels/chamfer_wall",
            "Window": "metrics_labels/chamfer_window",
        }

    if scene is Scene.ROOM_2:
        METRICS_TO_SHOW_CLASSES_T2P = {
            "Compressor": "metrics_labels/chamfer_compressor_t2p",
            "Door": "metrics_labels/chamfer_door_t2p",
            "Door frame": "metrics_labels/chamfer_door frame_t2p",
            "Electrical box": "metrics_labels/chamfer_electrical box_t2p",
            "Floor": "metrics_labels/chamfer_floor_t2p",
            "Machine": "metrics_labels/chamfer_machine_t2p",
            "Power socket unit": "metrics_labels/chamfer_power socket unit_t2p",
            "Stairs": "metrics_labels/chamfer_stairs_t2p",
            "Window frame": "metrics_labels/chamfer_window frame_t2p",
        }
        METRICS_TO_SHOW_CLASSES_GENERAL = {
            "Compressor": "metrics_labels/chamfer_compressor",
            "Door": "metrics_labels/chamfer_door",
            "Door frame": "metrics_labels/chamfer_door frame",
            "Electrical box": "metrics_labels/chamfer_electrical box",
            "Floor": "metrics_labels/chamfer_floor",
            "Machine": "metrics_labels/chamfer_machine",
            "Power socket unit": "metrics_labels/chamfer_power socket unit",
            "Stairs": "metrics_labels/chamfer_stairs",
            "Window frame": "metrics_labels/chamfer_window frame",
        }

    METRICS_TO_SHOW_COMMON = {
        "Chamfer, High Freq": "metrics/chamfer_high_freq",
        "Chamfer, High Freq, P2T": "metrics/chamfer_high_freq_p2t",
        "Chamfer, High Freq, T2P": "metrics_main/chamfer_high_freq_t2p",
        "Chamfer, Low Freq": "metrics/chamfer_low_freq",
        "Chamfer, Low Freq, P2T": "metrics/chamfer_low_freq_p2t",
        "Chamfer, Low Freq, T2P": "metrics_main/chamfer_low_freq_t2p",
        "Chamfer, T2P": "metrics_main/chamfer_t2p",
        "Chamfer, P2T": "metrics/chamfer_p2t",
        "LPIPS, Low Freq": "metrics_main/lpips_low",
        "LPIPS, High Freq": "metrics_main/lpips_high",
        "LPIPS 0": "metrics_main/lpips_0",
        "LPIPS 1": "metrics_main/lpips_1",
        "LPIPS 2": "metrics_main/lpips_2",
        "LPIPS 3": "metrics_main/lpips_3",
        "LPIPS 4": "metrics_main/lpips_4",
        "LPIPS": "metrics_main/lpips",
        "Completeness, High Freq, 0002": "metrics_main/completeness_high_freq_0002",
        "Completeness, High Freq, 0003": "metrics_main/completeness_high_freq_0003",
        "Completeness, Low Freq, 0002": "metrics_main/completeness_low_freq_0002",
        "Completeness, Low Freq, 0003": "metrics_main/completeness_low_freq_0003",
    }

    EXPERIMENTS = {
        ##### ROOM 1
        ### Large
        # "SIREN HL 1": "25-06-14_siren_o30_linter200_hl1",
        # "SIREN HL 5": "25-06-07_siren_o30_hl_5",
        # "SIREN WL 512": "25-06-14_siren_o30_linter200_wl512",
        # "SIREN WL 1536": "25-06-09_siren_o30_wl_1536_linter200",
        # "SIREN weight coeff 3": "25-06-13_siren_o30_linter200_wm3",
        # "SIREN o30": "25-05-12_siren_newdata_o30_linter300",
        # "SIREN weight coeff 12": "25-06-12_weightm12_linter200",
        # "SIREN weight coeff 18": "25-06-13_siren_o30_linter200_wm18",
        # "SIREN o5": "25-05-12_siren_newdata_o5",
        # "SIREN o15": "25-05-13_siren_newdata_o15",
        # "SIREN o45": "25-05-14_siren_newdata_o45_linter200",
        # "FINER": "25-05-29_finer_o20_linter500",
        # "SIREN-Mod": "25-06-17_dsiren_tanh_ab_room1_mod500",
        # "HOSC": "25-06-04_hosc_init",
        # "SIREN-Mod-old": "25-06-01_double_siren_pe_mlp",
        # "ReLU PE": "25-05-21_relu_pe_init_linter100",
        # "STAFF": "25-05-31_staf_init",
        # "Attn FF": "25-05-26_attn_ff_linter100_o20",\
        ### Small
        # "SIREN o30": "25-06-19_siren_o30_small_room1",
        # "FINER": "25-06-19_finer_small_room1",
        # "SIREN-Mod AB": "25-06-20_dsiren_ab_small_room1",
        # "SIREN-Mod AB o10": "25-06-24_dsiren_small_room1_pretr256_o10",
        # "SIREN-Mod repeat": "25-06-24_dsiren_small_room1_pretr256_o10_l400",
        ### Medium
        # "SIREN-o30-med long": "25-06-22_siren_o30_medium_room1",
        # "FINER-med long": "25-06-22_finer_medium_room1",
        # "SIREN-Mod-pretrained-med": "25-06-22_dsiren_medium_room1_pretr256",
        # "SIREN-Mod-pretrained-med-long": "25-06-23_dsiren_medium_room1_pretr256_long",
        # Misc
        # "SIREN-o30-med": "25-06-20_siren_o30_medium_room1",
        # "FINER-med": "25-06-20_finer_medium_room1",
        ##### ROOM 2
        ### Small
        # "SIREN o30": "25-06-15_siren_o30_linter200_room2",
        # "SIREN-Mod A": "25-06-15_dsiren_linter200_room2",
        # "FINER": "25-06-15_finer_linter200_room2",
        # "SIREN omega up": "25-06-18_siren_omega_up",
        # "SIREN omega down": "25-06-18_siren_omega_down",
        # "SIREN o30 skips": "25-06-18_siren_skips_room2",
        # "SIREN-Mod AB": "25-06-15_dsiren_mlp_ab_linter200_room2",
        ### Medium
        "SIREN o30, medium": "25-06-23_siren_o30_medium_room2",
        "FINER, medium": "25-06-24_finer_medium_room2",
        # "SIREN-Mod AB": "25-06-15_dsiren_mlp_ab_linter200_room2",
        "SIREN-Mod AB long": "25-06-25_dsiren_medium_room2_pretr256",
        #### New archi
        # "SIREN o30": "25-06-15_siren_o30_linter200_room2",
        # "SIREN-Mod AB 512": "25-06-15_dsiren_mlp_ab_linter200_room2",
        # "DSiren Softplus ABC": "25-06-26_dsiren_small_room2_soft_abc",
        # "DSiren Softplus GELU ABC": "25-06-26_dsiren_small_room2_soft_gelu_abc",
        # "DSiren Softplus Sin ABC": "25-06-26_dsiren_small_room2_soft_sin_abc",
    }

    # Collect all data
    all_data = []

    for readable_name, experiment_full_name in EXPERIMENTS.items():
        results_dir = Path(
            f"/home/magnes/outputs/{experiment_full_name}/latest/results/"
        )
        experiment_name = get_exp_name_from_full_name(experiment_full_name)
        logging.info(
            f"Processing experiment: {readable_name} ({experiment_full_name}) -> {experiment_name}"
        )

        df = parse_results_from_directory(results_dir, experiment_name, resolution)
        if not df.empty:
            # Replace the experiment name with the readable name
            df["experiment"] = readable_name
            all_data.append(df)

    if not all_data:
        logging.warning("No data found in any experiment directories!")
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

    logging.stage("Creating line plots (metric vs epoch)...")
    create_metric_plots(
        combined_data,
        METRICS_TO_SHOW_COMMON,
        output_dir / "lineplots",
    )
    logging.stage("Creating comparison plot (all metrics across experiments)...")
    create_metric_comparison_plot(
        combined_data,
        METRICS_TO_SHOW_CLASSES_T2P,
        output_dir / "comparison_t2p",
    )
    create_metric_comparison_plot(
        combined_data,
        METRICS_TO_SHOW_CLASSES_GENERAL,
        output_dir / "comparison_general",
    )
    logging.stage(f"Visualization results saved to {output_dir}")


if __name__ == "__main__":
    app()
