# Project Scripts

This folder contains the scripts for the current MS COCOAI experiments.

## `curate_mscocoai_split.py`

Converts local MS COCOAI parquet files into image folders plus metadata.

Input:

```text
/root/autodl-tmp/488FP/datasets/Defactify_Image_Dataset/data
```

The script reads parquet files with `pyarrow` in small batches and writes the
original image bytes to disk. It supports one generator or comma-separated
generator lists.

Example single-generator split:

```bash
/root/miniconda3/bin/python projects/curate_mscocoai_split.py \
  --train-generator stable_diffusion_3 \
  --unseen-generator midjourney_v6 \
  --curated-dir datasets/current/step1_bidirectional/sd3_to_midjourney \
  --output-dir outputs/current/step1_bidirectional/sd3_to_midjourney/curation \
  --overwrite
```

Example multi-generator split:

```bash
/root/miniconda3/bin/python projects/curate_mscocoai_split.py \
  --train-generator stable_diffusion_3,sdxl \
  --seen-generator stable_diffusion_3,sdxl \
  --unseen-generator midjourney_v6 \
  --curated-dir datasets/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything \
  --output-dir outputs/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything/curation \
  --overwrite
```

Metadata columns:

```text
relative_path, source_path, split, generator, source_generator,
class_name, label, is_fake, label_a, label_b, caption, filename
```

## `train_resnet18_baseline.py`

Trains and evaluates a ResNet-18 binary classifier. It accepts one generator or
a comma-separated generator list for `--train-generator`, `--seen-generator`,
and `--unseen-generator`.

Example:

```bash
/root/miniconda3/bin/python projects/train_resnet18_baseline.py \
  --metadata-path datasets/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything/metadata.csv \
  --train-generator stable_diffusion_3,sdxl \
  --seen-generator stable_diffusion_3,sdxl \
  --unseen-generator midjourney_v6 \
  --output-dir outputs/current/step2_multi_generator/sd3_sdxl_to_midjourney_anything/run \
  --weights none
```

Outputs:

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

## Current Results

```text
Protocol                    seen AUC  unseen AUC  unseen F1
SD3 -> MidJourney           0.8681    0.5318      0.1747
MidJourney -> SD3           0.9524    0.5482      0.1673
SD3 + SDXL -> MidJourney    0.8887    0.6694      0.3695
```

## `train_clip_linear_detector.py`

Trains a real/fake detector on frozen CLIP image features. The CLIP image
encoder is not fine-tuned; only a linear classification head is trained.

Example:

```bash
HF_ENDPOINT=https://hf-mirror.com /root/miniconda3/bin/python projects/train_clip_linear_detector.py \
  --metadata-path datasets/current/step1_bidirectional/sd3_to_midjourney/metadata.csv \
  --train-generator stable_diffusion_3 \
  --seen-generator stable_diffusion_3 \
  --unseen-generator midjourney_v6 \
  --output-dir outputs/current/step3_clip/sd3_to_midjourney/run
```

Current CLIP comparison on unseen generators:

```text
Protocol                    ResNet AUC  CLIP AUC  ResNet F1  CLIP F1
SD3 -> MidJourney           0.5318      0.8719    0.1747     0.6633
MidJourney -> SD3           0.5482      0.9153    0.1673     0.6045
SD3 + SDXL -> MidJourney    0.6694      0.9018    0.3695     0.7096
```
