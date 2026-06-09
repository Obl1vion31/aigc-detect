# 488FP

Computer vision final project workspace.

## Repository Layout

```text
488FP/
├── projects/          # Project source code and experiments tracked by Git
├── configs/           # Training and evaluation configuration files tracked by Git
├── README.md          # Project notes and setup instructions
├── requirements.txt   # Python dependencies
├── datasets/          # Local/remote datasets, not tracked by Git
└── outputs/           # Generated outputs, logs, and checkpoints, not tracked by Git
```

## Dataset

The dataset is stored on the AutoDL server and should not be committed to Git.

```text
/root/autodl-tmp/488FP/datasets/GenImage
```

Current dataset subsets:

```text
datasets/GenImage/BigGAN/imagenet_ai_0419_biggan/{train,val}/{ai,nature}
datasets/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/{train,val}/{ai,nature}
```

## Git Policy

Track code, configs, documentation, and dependency files. Do not track datasets,
checkpoints, archives, generated outputs, logs, or caches.
