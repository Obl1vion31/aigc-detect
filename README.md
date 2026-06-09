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
└── outputs/           # Milestone deliverables and generated outputs
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

Milestone 1 key output files are stored here and tracked by Git:

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

Other generated outputs, checkpoints, logs, and large experiment artifacts
should remain untracked unless they are intentionally selected as milestone
deliverables.

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

Use this workflow to back up code changes and milestone deliverables to GitHub:

```bash
cd /root/autodl-tmp/488FP

git status
git add projects configs outputs/M1 README.md requirements.txt .gitignore
git commit -m "Update project code and outputs"
git push
```

Before committing, check that `datasets/`, checkpoints, archives, logs, and
large non-milestone outputs are not included in `git status`.

## Git Policy

Track code, configs, documentation, dependency files, and selected milestone
outputs such as `outputs/M1`. Do not track datasets, checkpoints, archives,
logs, caches, or large intermediate experiment outputs.
