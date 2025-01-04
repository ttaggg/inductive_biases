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
uv run training scene=thai_statue inr=siren trainer.batch_size=250000
```

## Exporting INR to mesh

```bash
# uv run decoding --help
uv run decoding  --model-path=model_epoch_9000.pt --device=cpu --resolution=64
```

## Resampling pointcloud from the OBJ mesh

```bash
# uv run resampling --help
uv run resampling  --input-path=normalized_model.obj --num-samples=1000000
```

## Hydra config's layout

```
ib/configs
├── config.yaml
├── inr
│   ├── relu_pe.yaml
│   └── siren.yaml
├── loss
│   └── siren_sdf_loss.yaml
├── model
│   └── base_model.yaml
├── scene
│   ├── dataset
│   │   ├── obj_dataset.yaml
│   │   └── xyz_dataset.yaml
│   ├── interior_room.yaml
│   └── thai_statue.yaml
└── trainer
    └── base_trainer.yaml
```