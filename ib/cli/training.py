"""Training."""

import typer

from ib.utils.factory import create_loader, create_model, create_trainer
from ib.utils.pipeline import initialize_run, measure_time
from ib.utils.logging_module import logging

app = typer.Typer(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    add_completion=False,
)


@app.command(no_args_is_help=True)
@measure_time
def training(ctx: typer.Context) -> None:
    """Run the training.

    Command: uv run training [HYDRA-OVERRIDES] \n
    Example: uv run training scene=scannet_room_2 run_name=init

    The `scene` argument is required.
    """

    # Initialize: resolve configs, output directory and logging.
    cfg = initialize_run(ctx)

    # Load the data.
    data_loader = create_loader(cfg.scene.dataset, cfg.trainer)

    # Create the model.
    model = create_model(cfg.model)

    # Create the trainer.
    trainer = create_trainer(cfg.trainer, cfg.evaluator)

    # Run training.
    logging.stage("Running training.")
    trainer.fit(model, data_loader)


if __name__ == "__main__":
    app()
