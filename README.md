# Neural Network Inductive Biases for High-fidelity 3D Reconstruction

## Installation

This project uses [uv](https://github.com/astral-sh/uv) to manage Python environment and dependencies. 
```bash
brew install uv   # macOS
# or
pip install uv    # linux, macOS, ...
# or
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Dependencies are in the `pyproject.toml`

uv will automatically install all dependencies after the first `uv run ...` command.

## Overview
There are multiple commands provided by this project: training, evaluation, visualization, data preprocessing and special tools. They are listed in this README, and the scripts are in the `ib/cli/` directory.

## Training
Run the training. All possible Hydra config values, including defaults are listed in `configs/` directory.
```bash
# uv run training [HYDRA-OVERRIDES, scene is required]
uv run training scene=scannet_room_2 inr=siren trainer.batch_size=250000 run_name=my_experiment_name
```
Note: `run_name` is useful for organizing experiments.

## Evaluation

### Evaluate the model
This script evaluates a trained model using specified metrics (e.g., chamfer distance, normal distance, lpips, completeness).

```bash
# uv run evaluate_model --help
uv run evaluate_model \
--file-path=path/to/ground_truth/pc_aligned.ply \
--model-path=path/to/model/model.pt \
--resolution=1536 \
--batch-size=300000 \
```


### Evaluate the mesh
This script evaluates the generated mesh.
```bash
# uv run evaluate_mesh --help
uv run evaluate_mesh \
--file-path=path/to/ground_truth/pc_aligned.ply \
--mesh-path=path/to/mesh/mesh.ply
```

### Export INR to mesh
This script exports the 3D shape encoded in the Implicit Neural Representation to a mesh file.
```bash
# uv run decoding --help
uv run decoding \
--model-path=path/to/model/model.pt \
--resolution=512 \
--batch-size=100000
```

## Data
Preprocess the data: cut a portion of the scene, rotate if neccessary, annotate each point in a pointcloud with a label, save the corresponding ground truth mesh portion. Specific arguments are given in the YAML files for a corresponding sample.
```bash
# uv run preprocessing --help
uv run preprocessing \
--input-path=path/to/raw_data/pc_aligned.ply \
--x-range -0.61 0.6 \
--y-range -0.8 0.2 \
--z-range -1.0 0.1 \
--rotation-angle 1.0 \
--margin 0.005
```


## Visualization

### Choose camera views for the mesh and save them
```bash
# uv run capture_camera_params --help
uv run capture_camera_params \
--mesh-path=path/to/mesh/mesh.ply
```

### Take camera view and render
```bash
# uv run render_image_from_mesh --help
uv run render_image_from_mesh \
--mesh-path=path/to/mesh/mesh.ply \
--cam-params-path=path/to/cam_params/cam_params.json
```

### Make plots for a given experiments

```bash
# uv run visualization --help
uv run visualization \
--output-dir=output_dir \
--scene=2 \
--resolution=1536
```

### Visualize model's coefficients

```bash
# uv run coeffs_visualize --help
uv run coeffs_visualize \
--model-path=path/to/model/model.pt \
--output-dir=output_dir
```


## Miscellaneous

### Pretrain modulator

Pretrain the modulator to return dummy coefficients. The checkpoint is then used to initialize SIREN-FM, this makes the training more stable during the early stages.
```bash
# uv run pretrain_modulator --help
uv run pretrain_modulator \
--mod-hidden-size=320 \
--save-path=./modulator_320_softplus_relu_abc.pt
```

### Generate Fourier complexity heatmaps
Create images similar to the Neural Redshift paper.
```bash
# uv run fourier_grid --help
uv run fourier_grid \
--resolution=128
```


## Hydra config's layout

```
ib/configs
├── config.yaml
├── evaluator
│   └── base_evaluator.yaml
├── inr
│   ├── finer.yaml
│   ├── relu_pe.yaml
│   ├── siren.yaml
│   ├── sirenfm.yaml
│   └── staf.yaml
├── loss
│   ├── l1_loss.yaml
│   ├── l2_loss.yaml
│   └── siren_sdf_loss.yaml
├── model
│   └── base_model.yaml
├── scene
│   ├── dataset
│   │   ├── ply_dataset.yaml
│   │   ├── sdf_dataset.yaml
│   │   └── sparse_sdf_dataset.yaml
│   ├── scannet_room_1.yaml
│   ├── scannet_room_2.yaml
│   ├── scannet_sdf.yaml
│   └── thai_statue_sdf.yaml
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
