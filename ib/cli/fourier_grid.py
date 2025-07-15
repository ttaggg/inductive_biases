"""Check random initializations for Fourier Frequency metric."""

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import typer

from ib.metrics.fourier_freq import FourierFrequency
from ib.models.inrs.finer import Finer
from ib.models.inrs.relu_pe import ReluPe
from ib.models.inrs.siren import Siren
from ib.utils.model import query_model
from ib.utils.pipeline import measure_time, resolve_and_expand_path, set_seed
from ib.utils.logging_module import logging


app = typer.Typer(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_completion=False,
)


def plot_heatmap(
    results: dict[tuple[int, int], float],
    lhs_label: str,
    rhs_label: str,
    output_path: Path,
    value_range: tuple[float, float] | None = None,
) -> None:
    """Plot the heatmap of the Fourier Frequency metric."""

    # Prepare axes values
    lhs = sorted({lhs for lhs, _ in results})
    rhs = sorted({rhs for _, rhs in results})

    # Build heatmap data array
    heatmap = np.zeros((len(lhs), len(rhs)))
    for (l, r), value in results.items():
        i = lhs.index(l)
        j = rhs.index(r)
        heatmap[i, j] = value

    # Plot heatmap
    vmin, vmax = (
        value_range if value_range is not None else (heatmap.min(), heatmap.max())
    )
    fig, ax = plt.subplots()
    cax = ax.imshow(
        heatmap,
        origin="lower",
        cmap="jet",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xticks(range(len(rhs)))
    ax.set_xticklabels(rhs)
    ax.set_xlabel(rhs_label)
    ax.set_yticks(range(len(lhs)))
    ax.set_yticklabels(lhs)
    ax.set_ylabel(lhs_label)
    fig.colorbar(cax, ax=ax, label="Complexity")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


@app.command()
@measure_time
def fourier_grid(
    resolution: int = 512,
    batch_size: int = 10000,
    device: str = "cuda",
) -> None:
    """Run the Fourier Frequency metric check."""
    logging.stage("Running Fourier Frequency metric check.")

    # Specify model, lhs, rhs labels.
    model_name = "ReLU"
    lhs_label = "Depth"
    rhs_label = "Weight_Magnitude"

    # Initialize metric and run.
    set_seed()
    metric_fn = FourierFrequency()
    results: dict[tuple[int, int], float] = {}
    for depth in range(1, 10, 1):
        for width in [256]:  # range(128, 768, 64):
            for weight_magnitude in range(1, 30, 2):
                for omega in [30]:  # range(5, 100, 5):
                    for num_frequencies in [10]:  # range(1, 12, 1):
                        # model = Siren(
                        #     in_features=3,
                        #     out_features=1,
                        #     hidden_features=width,
                        #     hidden_layers=depth,
                        #     first_omega=omega,
                        #     hidden_omega=omega,
                        #     weight_magnitude=weight_magnitude,
                        # )
                        model = ReluPe(
                            in_features=3,
                            out_features=1,
                            hidden_features=width,
                            hidden_layers=depth,
                            num_frequencies=num_frequencies,
                            weight_magnitude=weight_magnitude,
                        )
                        model.to(device)
                        model_outputs = query_model(
                            model, resolution, batch_size, device
                        )
                        value = list(metric_fn(model_outputs).values())[0]
                        results[(depth, weight_magnitude)] = value

    for k, v in results.items():
        logging.info(f"{k}: {v}")

    # Initialize directory.
    save_dir = Path("heatmaps")
    save_dir = resolve_and_expand_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save heatmap.
    heatmap_output_path = save_dir / f"{model_name}_{lhs_label}_{rhs_label}_heatmap.png"
    plot_heatmap(results, lhs_label, rhs_label, heatmap_output_path, (0, 0.5))
    logging.info(f"Saved heatmap to {heatmap_output_path}")

    # Save results.
    json_output_path = save_dir / f"{model_name}_{lhs_label}_{rhs_label}_results.json"
    with json_output_path.open("w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    logging.info(f"Saved results to {json_output_path}")


if __name__ == "__main__":
    app()
