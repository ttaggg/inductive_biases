# WIP: Neural Network Inductive Biases for High-fidelity 3D Reconstruction

## Installation

This project uses [uv](https://github.com/astral-sh/uv) to manage Python environment and dependencies. 
```bash
brew install uv   # macOS
# or
pip install uv    # linux, macOS, ...
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```
uv will automatically install all dependencies after the first `uv run ...` command.

## Training

```bash
# uv run training [HYDRA-OVERRIDES, scene is required]
uv run training scene=thai_statue inr=siren trainer.batch_size=250000 run_name=my_experiment_name
```
Note: `run_name` is useful for organizing experiments.

## Evaluation

```bash
# uv run evaluation --help
uv run evaluation --model-path=path/to/model.pt --metric=<metric_name> [--file-path=path/to/ground_truth.{ply,npy}] [--resolution=512] [--device=cuda]
```
This script evaluates a trained model using specified metrics (e.g., chamfer distance, IoU, normal distance, Fourier frequencies).

## Exporting INR to mesh

```bash
# uv run decoding --help
uv run decoding --model-path=path/to/model.pt [--resolution=512] [--batch-size=256000] [--device=cuda]
```
This script exports the shape encoded in the Implicit Neural Representation (INR) to a mesh file.

## Resampling pointcloud from the OBJ mesh

Will be deprecated or extended to work with .ply meshes.
OBJ files are currently not used in the pipeline.
```bash
# uv run resampling --help
uv run resampling --input-path=normalized_model.obj --num-samples=1000000
```
This script resamples vertices from an input mesh file to generate a point cloud.

## Hydra config's layout

```
ib/configs
├── config.yaml
├── inr
│   ├── finer.yaml
│   ├── relu_pe.yaml
│   └── siren.yaml
├── loss
│   ├── l1_loss.yaml
│   ├── l2_loss.yaml
│   └── siren_sdf_loss.yaml
├── scene
│   ├── dataset
│   │   ├── obj_dataset.yaml
│   │   ├── ply_dataset.yaml
│   │   ├── sdf_dataset.yaml
│   │   └── xyz_dataset.yaml
│   ├── interior_room.yaml
│   ├── scannet_room_1.yaml
│   ├── thai_statue.yaml
│   └── thai_statue_sdf.yaml
├── model
│   └── base_model.yaml
├── evaluator
│   └── base_evaluator.yaml
└── trainer
    └── base_trainer.yaml
```

## WandB Integration

The project also supports Weights & Biases for experiment tracking. To use WandB:

1. Create a `.env` file in the project root (copy from `.env.example`)
2. Add your WandB API key to the `.env` file:
   ```
   WANDB_API_KEY=your_wandb_api_key_here
   ```
If the WandB API key is not found, the pipeline will still use TensorBoard.
