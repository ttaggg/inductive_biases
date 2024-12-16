# WIP: Neural Network Inductive Biases for High-fidelity 3D Reconstruction

## Installation

This project uses [uv](https://github.com/astral-sh/uv) to manage Python environment and dependencies. 
```bash
brew install uv   # macOS
pip install uv    # linux, macOS, ...
```
uv will automatically install all dependencies after the first `uv run ...` command.

## Training

```bash
# uv run training [HYDRA-OVERRIDES]
uv run training scene=thai_statue run_name=first_try # scene= is required
```


## Hydra config's layout

```
ib/conf
├── config.yaml
├── model
│   ├── inr
│   │   └── siren.yaml
│   ├── loss
│   │   └── siren_sdf_loss.yaml
│   └── original_siren.yaml
├── scene
│   ├── dataset
│   │   ├── obj_dataset.yaml
│   │   └── xyz_dataset.yaml
│   ├── lamp.yaml
│   ├── sphere.yaml
│   └── thai_statue.yaml
└── trainer
    └── base_trainer.yaml
```