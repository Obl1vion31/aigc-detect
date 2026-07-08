# 项目脚本说明

这个文件夹包含 GenImage AIGC 检测项目的数据整理、数据分析和模型训练脚本。

## 数据划分脚本

### `curate_biggan_to_sd15_split.py`

用途：创建 Milestone 1 使用的原始数据划分，也是第一个 Milestone 2 baseline 使用的
划分。

实验协议：

- 训练生成器：BigGAN。
- 验证生成器：BigGAN。
- seen 测试生成器：BigGAN。
- unseen 测试生成器：Stable Diffusion v1.5。

主要输出：

```text
datasets/curated_genimage/
|-- metadata.csv
`-- dataset_statistics.csv
```

这个划分用于测试：一个只在 BigGAN fake 图像上训练的检测器，能否迁移到更真实的
diffusion 生成图上。实验结果显示，这种迁移非常弱。

### `curate_sd15_to_biggan_split.py`

用途：创建 Milestone 2 的反向数据划分。

实验协议：

- 训练生成器：Stable Diffusion v1.5。
- 验证生成器：Stable Diffusion v1.5。
- seen 测试生成器：Stable Diffusion v1.5。
- unseen 测试生成器：BigGAN。

主要输出：

```text
datasets/curated_genimage_sd15_train/
|-- metadata.csv
`-- dataset_statistics.csv
```

这个划分用于验证一个直觉：如果用更真实的 SD1.5 fake 图像训练，模型是否会学到更
通用的 fake 图像证据。实验结果显示，unseen BigGAN 上仍然很弱，所以单一生成器训练
仍然是主要限制。

## 数据分析脚本

### `analyze_m1_resolutions.py`

用途：检查 Milestone 1 curated subset 中的图像有效性、原始分辨率、图像格式和图像
面积分布。

主要输出：

```text
outputs/M1/resolution_analysis/
```

这个脚本不训练模型，只用于验证和描述数据集。

## 训练脚本

### `train_resnet18_baseline.py`

用途：训练和评估 ResNet-18 real/fake 二分类 baseline。

该脚本完成完整的 Milestone 2 训练流程：

1. 读取 metadata CSV。
2. 按 split 和 generator 筛选样本。
3. 使用 PIL 读取图像并转换为 RGB。
4. 对训练集或评估集应用对应 transform。
5. 使用 cross-entropy loss 训练 ResNet-18 二分类器。
6. 每个 epoch 结束后在 validation split 上评估。
7. 默认根据 validation loss 保存最佳 checkpoint。
8. 如果 validation loss 不再改善，则触发 early stopping。
9. 使用最佳 checkpoint 评估 validation、seen test 和 unseen test。
10. 导出 metrics、training curves、confusion matrices 和 seen/unseen 对比图。

### 数据流

脚本使用 `GenImageMetadataDataset` 读取数据。

每一行 metadata 包含：

- 图像相对路径；
- 实验 split；
- generator 名称；
- 标签，`real` 或 `fake`；
- 文件名和其他 metadata。

Dataset 会根据命令行参数筛选样本，例如：

```text
split == "train" and generator == "BigGAN"
```

或者：

```text
split == "test_unseen" and generator == "stable_diffusion_v_1_5"
```

因此，同一个训练脚本可以通过不同命令行参数运行两个方向的实验。

### 模型

模型是 ResNet-18。原始最后一层分类头被替换为二分类输出层：

```text
real -> class 0
fake -> class 1
```

模型在 softmax 之前输出两个 logits：

```text
logit_real, logit_fake
```

训练时，`nn.CrossEntropyLoss` 直接接收 raw logits，不需要提前手动 softmax。

评估时，脚本会对 logits 做 softmax，并使用 `P(fake)` 作为 fake 类的连续置信分数，
用于计算 AUC。

### 一个 epoch 中发生什么

一个 epoch 表示模型完整看过一次训练 split 中的所有图像。

在一个 epoch 内部：

1. DataLoader 取出一个 mini-batch 的图像和标签。
2. 图像输入 ResNet-18。
3. 模型为每张图输出两个 logits。
4. Cross-entropy loss 比较 logits 和真实标签。
5. Backpropagation 计算梯度。
6. AdamW 根据梯度更新模型参数。
7. 脚本累计平均 training loss。

一个 epoch 结束后，模型会在 validation split 上评估。Validation 过程只计算指标，
不更新模型参数。

### 过拟合控制

最初的 BigGAN baseline 很快出现过拟合：training loss 持续下降，但 validation loss
在第 1 个 epoch 后开始上升，validation accuracy 和 F1 也下降。

当前脚本加入了以下控制：

- 更低的默认 learning rate；
- 更大的 weight decay；
- 分类头前的 dropout；
- label smoothing；
- 更强的数据增强；
- cosine learning-rate schedule；
- 根据 validation loss 保存 checkpoint；
- early stopping。

这些机制可以避免盲目训练更多 epoch。实际 regularized BigGAN 实验仍然选择 epoch 1
作为最佳 checkpoint，并且仍然不能泛化到 unseen SD1.5。这说明主要限制不只是普通
overfitting，而是单一训练生成器导致的 generator-specific learning。

### 输出文件

每次训练会输出：

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

checkpoint 文件在服务器上有用，但因为属于较大的模型 artifact，所以被 Git 忽略。
