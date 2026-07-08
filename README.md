# 488FP

Computer vision final project workspace for AIGC image detection on GenImage.

## Repository Layout

```text
488FP/
|-- projects/          # Project source code and experiment scripts
|-- configs/           # Training/evaluation configuration files
|-- outputs/           # Selected milestone deliverables
|-- README.md          # Project notes and setup instructions
|-- requirements.txt   # Python dependencies
`-- datasets/          # Local/remote datasets, not tracked by Git
```

## Project Scripts

See `projects/README.md` for detailed script-level explanations.

`projects/curate_biggan_to_sd15_split.py` creates the original Milestone 1 split:
BigGAN is used for training/validation/seen testing, and Stable Diffusion v1.5
is held out as the unseen generator.

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/curate_biggan_to_sd15_split.py
```

`projects/curate_sd15_to_biggan_split.py` creates the reverse Milestone 2 split:
Stable Diffusion v1.5 is used for training/validation/seen testing, while BigGAN
is held out as the unseen generator.

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/curate_sd15_to_biggan_split.py
```

`projects/analyze_m1_resolutions.py` reads `outputs/M1/metadata.csv`, records
image sizes and formats, and exports Milestone 1 resolution plots.

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/analyze_m1_resolutions.py
```

`projects/train_resnet18_baseline.py` trains and evaluates a ResNet-18 binary
real/fake classifier with cross-entropy loss. It includes early stopping,
regularization, augmentation options, and writes metrics plus plots.

Example BigGAN-to-SD1.5 regularized run:

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/train_resnet18_baseline.py \
  --metadata-path datasets/curated_genimage/metadata.csv \
  --train-generator BigGAN \
  --seen-generator BigGAN \
  --unseen-generator stable_diffusion_v_1_5 \
  --output-dir outputs/M2/biggan_to_sd15_regularized \
  --weights none
```

Example SD1.5-to-BigGAN run:

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/train_resnet18_baseline.py \
  --metadata-path datasets/curated_genimage_sd15_train/metadata.csv \
  --train-generator stable_diffusion_v_1_5 \
  --seen-generator stable_diffusion_v_1_5 \
  --unseen-generator BigGAN \
  --output-dir outputs/M2/sd15_to_biggan \
  --epochs 3 \
  --batch-size 128 \
  --weights none
```

## Dataset

The original dataset is stored on the AutoDL/server instance and should not be
committed to Git.

```text
/root/autodl-tmp/488FP/datasets/GenImage
```

Current source dataset subsets:

```text
datasets/GenImage/BigGAN/imagenet_ai_0419_biggan/{train,val}/{ai,nature}
datasets/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/{train,val}/{ai,nature}
```

## Outputs

Milestone 1 deliverables are under `outputs/M1/`, including dataset metadata,
statistics, sample images, and resolution-analysis plots.

Milestone 2 deliverables are under `outputs/M2/`:

```text
outputs/M2/
|-- README.md
|-- biggan_to_sd15/
|-- biggan_to_sd15_regularized/
`-- sd15_to_biggan/
```

Each M2 experiment folder contains metrics CSVs and visualization PNGs. See
`outputs/M2/README.md` for interpretation of the training curves, confusion
matrices, seen/unseen gap, and regularization results.

## Git Policy

Track code, configs, documentation, and selected milestone outputs. Do not track
datasets, checkpoints, archives, logs, caches, or temporary experiment folders.
