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

## Project Scripts

```text
projects/curate_genimage.py
```

Creates the Milestone 1 curated GenImage dataset without changing the original
dataset folders. It splits BigGAN train into 90% train and 10% val with fixed
random seed `488`, maps BigGAN val to `test_seen`, maps Stable Diffusion 1.5 val
to `test_unseen`, writes metadata/statistics CSV files, and exports key M1
deliverables to `outputs/M1`.

Run full curation from scratch:

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/curate_genimage.py
```

Refresh only the M1 output folder from existing metadata:

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/curate_genimage.py --export-m1-only
```

## Dataset

The original dataset is stored on the AutoDL server and should not be committed
to Git.

```text
/root/autodl-tmp/488FP/datasets/GenImage
```

Current source dataset subsets:

```text
datasets/GenImage/BigGAN/imagenet_ai_0419_biggan/{train,val}/{ai,nature}
datasets/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/{train,val}/{ai,nature}
```

Curated dataset:

```text
datasets/curated_genimage/
├── train/BigGAN/{ai,nature}
├── val/BigGAN/{ai,nature}
├── test_seen/BigGAN/{ai,nature}
├── test_unseen/stable_diffusion_v_1_5/{ai,nature}
├── metadata.csv
└── dataset_statistics.csv
```

## M1 Outputs

Milestone 1 key output files are stored here:

```text
outputs/M1/
├── metadata.csv
├── dataset_statistics.csv
└── samples/
```

`metadata.csv` records each image path, source path, split, generator, class
name, real/fake label, binary fake label, and filename.

`dataset_statistics.csv` records real/fake counts by generator and by split.

`samples/` contains one real and one fake example image for each split.

## Image Formats

The curated dataset contains both PNG and JPEG/JPEG-style files. This is
expected for GenImage: generated BigGAN images are PNG, while real ImageNet
images often use JPEG.

This should not hurt training if the data loader uses an image library such as
PIL or OpenCV and converts every image to a consistent format before transforms,
for example RGB:

```python
image = Image.open(path).convert("RGB")
```

Do not filter by extension unless the code includes both `.png`, `.jpg`,
`.jpeg`, and `.JPEG`.

## Daily Git Push

Use this workflow to back up code changes to GitHub:

```bash
cd /root/autodl-tmp/488FP

git status
git add projects configs README.md requirements.txt .gitignore
git commit -m "Update project code"
git push
```

Before committing, check that `datasets/`, `outputs/`, checkpoints, archives,
and logs are not included in `git status`.

## Git Policy

Track code, configs, documentation, and dependency files. Do not track datasets,
checkpoints, archives, generated outputs, logs, or caches.
