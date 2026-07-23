# 488FP Project: AI-Generated Image Detection

This repository contains the current final-project pipeline for AI-generated
image detection on **MS COCOAI**, a 2026 dataset hosted on Hugging Face as
`Rajarshi-Roy-research/Defactify_Image_Dataset`.

The model structure is kept simple and unchanged: a ResNet-18 binary classifier
predicts whether an image is `real` or `fake`. The current experiments focus on
cross-generator generalization.

## Current Experiment Logic

```text
Step 1: Single-generator bidirectional comparison
  1. stable_diffusion_3 -> midjourney_v6
  2. midjourney_v6 -> stable_diffusion_3

Step 2: Multi-generator training
  stable_diffusion_3 + sdxl -> midjourney_v6

Step 3: CLIP-linear comparison
  repeat all three protocols with frozen CLIP ViT-B/32 features
```

MidJourney v6 is treated as the held-out unseen generator in Step 2.

## Server Data

Raw parquet dataset:

```text
/root/autodl-tmp/488FP/datasets/Defactify_Image_Dataset/data
```

Curated image datasets:

```text
/root/autodl-tmp/488FP/datasets/current/
```

Current outputs:

```text
/root/autodl-tmp/488FP/outputs/current/
```

Old output folders are not used. The current project should be read from
`outputs/current` only.

## Output Structure

```text
outputs/current/
|-- report.md
|-- step1_bidirectional/
|   |-- report.md
|   |-- bidirectional_comparison.png
|   |-- sd3_to_midjourney/
|   |   |-- report.md
|   |   |-- curation/
|   |   `-- run/
|   `-- midjourney_to_sd3/
|       |-- report.md
|       |-- curation/
|       `-- run/
`-- step3_clip/
|   |-- report.md
|   |-- clip_vs_resnet_unseen.png
|   |-- sd3_to_midjourney/
|   |-- midjourney_to_sd3/
|   `-- sd3_sdxl_to_midjourney_anything/
`-- step2_multi_generator/
    `-- sd3_sdxl_to_midjourney_anything/
        |-- report.md
        |-- curation/
        `-- run/
```

Each ResNet `run/` folder contains:

```text
metrics.csv
training_log.csv
training_curves.png
split_metrics.png
confusion_matrix_seen.png
confusion_matrix_unseen.png
seen_unseen_comparison.png
run_config.json
checkpoints/best_resnet18.pt
```

Each CLIP `run/` folder contains:

```text
metrics.csv
training_log.csv
training_curves.png
split_metrics.png
confusion_matrix_seen.png
confusion_matrix_unseen.png
run_config.json
features/
checkpoints/best_clip_linear.pt
```

Step 3 summary:

```text
step3_clip/
    |-- report.md
    |-- clip_vs_resnet_unseen.png
    |-- sd3_to_midjourney/
    |-- midjourney_to_sd3/
    `-- sd3_sdxl_to_midjourney_anything/
```

## Latest Results

```text
Protocol                    ResNet AUC  CLIP AUC  ResNet F1  CLIP F1
SD3 -> MidJourney           0.5318      0.8719    0.1747     0.6633
MidJourney -> SD3           0.5482      0.9153    0.1673     0.6045
SD3 + SDXL -> MidJourney    0.6694      0.9018    0.3695     0.7096
```

## Main Conclusion

Step 1 shows that single-generator training does not generalize robustly in
either direction. Step 2 shows that adding generator diversity improves unseen
MidJourney performance. Step 3 shows that frozen CLIP features improve unseen
performance across all protocols, making CLIP-linear the strongest current
baseline.

## Reproduce Step 1

```bash
cd /root/autodl-tmp/488FP

/root/miniconda3/bin/python projects/curate_mscocoai_split.py \
  --train-generator stable_diffusion_3 \
  --unseen-generator midjourney_v6 \
  --curated-dir datasets/current/step1_bidirectional/sd3_to_midjourney \
  --output-dir outputs/current/step1_bidirectional/sd3_to_midjourney/curation \
  --overwrite

/root/miniconda3/bin/python projects/train_resnet18_baseline.py \
  --metadata-path datasets/current/step1_bidirectional/sd3_to_midjourney/metadata.csv \
  --train-generator stable_diffusion_3 \
  --seen-generator stable_diffusion_3 \
  --unseen-generator midjourney_v6 \
  --output-dir outputs/current/step1_bidirectional/sd3_to_midjourney/run \
  --weights none
```

Use the same command pattern with `midjourney_v6` and `stable_diffusion_3`
swapped for the reverse direction.

## Reproduce Step 2

```bash
cd /root/autodl-tmp/488FP

/root/miniconda3/bin/python projects/curate_mscocoai_split.py \
  --train-generator stable_diffusion_3,sdxl \
  --seen-generator stable_diffusion_3,sdxl \
  --unseen-generator midjourney_v6 \
  --curated-dir datasets/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything \
  --output-dir outputs/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything/curation \
  --overwrite

/root/miniconda3/bin/python projects/train_resnet18_baseline.py \
  --metadata-path datasets/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything/metadata.csv \
  --train-generator stable_diffusion_3,sdxl \
  --seen-generator stable_diffusion_3,sdxl \
  --unseen-generator midjourney_v6 \
  --output-dir outputs/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything/run \
  --weights none
```

## Reproduce Step 3

```bash
cd /root/autodl-tmp/488FP

HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python projects/train_clip_linear_detector.py \
  --metadata-path datasets/current/step1_bidirectional/sd3_to_midjourney/metadata.csv \
  --train-generator stable_diffusion_3 \
  --seen-generator stable_diffusion_3 \
  --unseen-generator midjourney_v6 \
  --output-dir outputs/current/step3_clip/sd3_to_midjourney/run
```

Use the same command pattern for `midjourney_to_sd3` and
`sd3_sdxl_to_midjourney_anything`.
