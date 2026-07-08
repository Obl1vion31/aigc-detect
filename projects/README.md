# Project Scripts

This folder contains the data-curation, analysis, and model-training scripts for
the GenImage AIGC detection project.

## Split-Curation Scripts

### `curate_biggan_to_sd15_split.py`

Purpose: create the original split used in Milestone 1 and the first Milestone 2
baseline.

Protocol:

- Training generator: BigGAN.
- Validation generator: BigGAN.
- Seen test generator: BigGAN.
- Unseen test generator: Stable Diffusion v1.5.

Main output:

```text
datasets/curated_genimage/
|-- metadata.csv
`-- dataset_statistics.csv
```

This split is useful because it directly tests whether a detector trained on
BigGAN can transfer to a more realistic diffusion generator. The result shows
that this transfer is very weak.

### `curate_sd15_to_biggan_split.py`

Purpose: create the reverse split used in Milestone 2.

Protocol:

- Training generator: Stable Diffusion v1.5.
- Validation generator: Stable Diffusion v1.5.
- Seen test generator: Stable Diffusion v1.5.
- Unseen test generator: BigGAN.

Main output:

```text
datasets/curated_genimage_sd15_train/
|-- metadata.csv
`-- dataset_statistics.csv
```

This split tests the intuition that training on more realistic SD1.5 fake images
might learn more general fake-image evidence. The result is still weak on unseen
BigGAN, so single-generator training remains the main limitation.

## Analysis Script

### `analyze_m1_resolutions.py`

Purpose: inspect image validity, native image resolution, image format, and area
distribution for the Milestone 1 curated subset.

Main output:

```text
outputs/M1/resolution_analysis/
```

This script does not train a model. It only validates and describes the dataset.

## Training Script

### `train_resnet18_baseline.py`

Purpose: train and evaluate a ResNet-18 real/fake baseline.

The script performs the full Milestone 2 pipeline:

1. Read metadata CSV rows.
2. Filter rows by split and generator.
3. Load images with PIL and convert them to RGB.
4. Apply training or evaluation transforms.
5. Train a ResNet-18 binary classifier using cross-entropy loss.
6. Validate after each epoch.
7. Save the best checkpoint according to validation loss by default.
8. Apply early stopping when validation loss stops improving.
9. Evaluate the selected checkpoint on validation, seen test, and unseen test.
10. Export metrics, training curves, confusion matrices, and seen/unseen plots.

### Dataset Flow

The script uses `GenImageMetadataDataset`.

Each metadata row contains:

- relative image path,
- split,
- generator,
- label (`real` or `fake`),
- filename and related metadata.

The dataset class filters rows such as:

```text
split == "train" and generator == "BigGAN"
```

or:

```text
split == "test_unseen" and generator == "stable_diffusion_v_1_5"
```

This design lets the same training script run both generator directions simply
by changing command-line arguments.

### Model

The model is ResNet-18 with the final classification head replaced by a
two-class output layer:

```text
real -> class 0
fake -> class 1
```

The output before softmax is two logits:

```text
logit_real, logit_fake
```

During training, `nn.CrossEntropyLoss` consumes the raw logits directly. During
evaluation, the script applies softmax and uses `P(fake)` as the continuous fake
score for AUC.

### One Epoch

One epoch means the model sees every image in the training split once.

Inside one epoch:

1. A mini-batch of images and labels is loaded.
2. Images are passed through ResNet-18.
3. The model outputs two logits per image.
4. Cross-entropy loss compares logits with the true labels.
5. Backpropagation computes gradients.
6. AdamW updates the model weights.
7. The script accumulates average training loss.

After the epoch finishes, the model is evaluated on the validation split. The
validation step does not update model weights.

### Overfitting Control

The first BigGAN baseline overfit quickly: training loss decreased while
validation loss increased after the first epoch.

The current script includes several controls:

- lower default learning rate,
- larger weight decay,
- dropout before the final classifier,
- label smoothing,
- stronger training augmentation,
- cosine learning-rate schedule,
- checkpoint selection by validation loss,
- early stopping.

These controls prevent blindly training more epochs. However, the regularized
BigGAN run still selected epoch 1 as the best checkpoint and still failed on
unseen SD1.5. This indicates that the main limitation is not only ordinary
overfitting, but generator-specific learning from a single training generator.

### Outputs

For each run, the script writes:

```text
metrics.csv
training_log.csv
training_curves.png
confusion_matrix_seen.png
confusion_matrix_unseen.png
seen_unseen_comparison.png
run_config.json
checkpoints/best_resnet18.pt
```

The checkpoint file is useful on the server, but it is ignored by Git because it
is a large model artifact.
