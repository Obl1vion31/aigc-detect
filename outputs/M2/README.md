# Milestone 2 输出说明

本文件夹保存 Milestone 2 的正式实验结果。当前两个方向都使用同一套模型和训练策略：
ResNet-18、cross-entropy loss、strong augmentation、dropout、label smoothing、weight
decay、cosine learning-rate schedule、validation-loss checkpoint 和 early stopping。

也就是说，`biggan_to_sd15/` 和 `sd15_to_biggan/` 不是两套不同模型；它们只是训练源
生成器和 unseen 测试生成器不同。

## 文件夹结构

```text
outputs/M2/
|-- README.md
|-- biggan_to_sd15/      # BigGAN 训练，SD1.5 unseen 测试
`-- sd15_to_biggan/      # SD1.5 训练，BigGAN unseen 测试
```

每个实验目录包含：

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

其中 checkpoint 文件在服务器上保留，但被 Git 忽略，不作为代码仓库内容提交。

## 为什么不是固定训练 10 个 epoch

课程示例里常见 `10 epochs`，但 epoch 数不是一个必须固定的值。一个 epoch 表示模型
完整看过一次训练集。本项目每个 epoch 大约包含 28-29 万张训练图像，因此一个 epoch
本身已经是很大的训练量。

更重要的是，是否继续训练应该由 validation set 决定。如果 training loss 继续下降，
但 validation loss 上升，说明模型正在更好地拟合训练集，却更差地泛化到验证集，这
就是过拟合。

所以当前脚本采用：

- 最多训练 `10` 个 epoch；
- 每个 epoch 后验证一次；
- 默认按 `val_loss` 保存最佳 checkpoint；
- 如果 `val_loss` 连续 2 个 epoch 不改善，就 early stop；
- 最终测试使用最佳 checkpoint，而不是最后一个 epoch 的模型。

因此，BigGAN 方向最终选择 epoch 1，而 SD1.5 方向最终选择 epoch 3。这不是实验不
完整，而是两个数据源的过拟合速度不同。

## 过拟合为什么容易发生

本项目容易过拟合主要有几个原因。

第一，训练源是单一生成器。BigGAN fake 图像有明显的 generator-specific artifact，
模型很容易学到这些低层痕迹，而不是通用的 AI 图像特征。

第二，数据量虽然大，但分布并不丰富。训练 split 里 fake 图像来自同一个 generator，
所以增加样本数量并不等于增加 generator diversity。

第三，real/fake 的差异可能混入了分辨率、压缩、纹理、颜色统计等 shortcut。CNN 很
擅长利用这些 shortcut，因此 seen generator 上表现很好，但 unseen generator 上失效。

第四，ResNet-18 是从随机初始化训练的，没有 ImageNet 预训练特征作为先验，因此更
容易从当前训练数据中学习局部 artifact。

Regularization 可以减缓过拟合，但不能从根本上提供 unseen generator 的多样性。因此
单靠 dropout、weight decay、augmentation 不能完全解决 cross-generator generalization。

## 统一训练设置

两个方向都使用以下核心设置：

```text
model: ResNet-18
weights: none
max epochs: 10
learning rate: 1e-4
weight decay: 1e-3
dropout: 0.2
label smoothing: 0.05
augmentation: strong
random erasing probability: 0.15
checkpoint metric: val_loss
early stopping patience: 2
```

训练时使用 cross-entropy loss。模型输出两个 logits：`logit_real` 和 `logit_fake`。
训练阶段不需要手动 softmax，因为 `nn.CrossEntropyLoss` 内部会处理 logits。评估
阶段会显式 softmax，并使用 `P(fake)` 计算 AUC。

## 实验一：BigGAN 训练，SD1.5 unseen

文件夹：`biggan_to_sd15`

训练过程：

```text
epoch  train_loss  val_loss  val_accuracy  val_f1   val_auc
1      0.2226      0.1710    0.9786        0.9790   0.9926
2      0.1852      0.1818    0.9728        0.9729   0.9912
3      0.1628      0.5420    0.7715        0.7081   0.9858
```

BigGAN 方向在 epoch 1 后已经开始过拟合。虽然 training loss 持续下降，但 validation
loss 上升，validation accuracy/F1 下降。脚本在 epoch 3 触发 early stopping，并回滚
使用 epoch 1 的最佳 checkpoint。

最终测试结果：

```text
Seen BigGAN test:
accuracy = 0.9909
F1       = 0.9910
AUC      = 0.9972

Unseen SD1.5 test:
accuracy = 0.4944
F1       = 0.0095
AUC      = 0.4938
```

结论：模型在 BigGAN seen domain 上几乎完美，但遇到 SD1.5 fake 图像时几乎全部漏检。
这说明模型主要学到了 BigGAN-specific artifact，而不是通用 AI 图像检测特征。

## 实验二：SD1.5 训练，BigGAN unseen

文件夹：`sd15_to_biggan`

训练过程：

```text
epoch  train_loss  val_loss  val_accuracy  val_f1   val_auc
1      0.3961      0.3094    0.9003        0.9064   0.9643
2      0.3213      0.2816    0.9153        0.9165   0.9744
3      0.2869      0.2611    0.9271        0.9279   0.9820
4      0.2625      0.2611    0.9250        0.9314   0.9851
5      0.2447      0.4034    0.8579        0.8787   0.9765
```

SD1.5 方向可以稳定训练更多 epoch。它在 epoch 1 到 epoch 3 持续改善，epoch 4 的
F1/AUC 仍然较高，但 `val_loss` 没有继续改善；epoch 5 明显过拟合。脚本最终选择
epoch 3 的最佳 checkpoint。

最终测试结果：

```text
Seen SD1.5 test:
accuracy = 0.9244
F1       = 0.9217
AUC      = 0.9821

Unseen BigGAN test:
accuracy = 0.5113
F1       = 0.1131
AUC      = 0.5545
```

结论：SD1.5 训练方向经过 regularization 后，seen domain 表现比之前更稳定，但 unseen
BigGAN 仍然接近随机。AUC 只有 0.5545，说明有一点点排序信号，但远远不够成为可靠的
跨生成器检测器。

## 混淆矩阵怎么看

混淆矩阵的纵轴是真实标签，横轴是模型预测标签：

```text
                 Predicted real    Predicted fake
True real        TN                FP
True fake        FN                TP
```

在本项目中，左下角 `FN` 尤其重要，因为它表示 AI 图像被漏检为真实图像。

BigGAN 到 SD1.5 的 unseen 混淆矩阵中，大多数 SD1.5 fake 图像都落在左下角，因此
F1 极低。

SD1.5 到 BigGAN 的 unseen 混淆矩阵中，也有大量 BigGAN fake 图像被预测为 real，
说明跨生成器泛化仍然不足。

## 同一个脚本如何跑出不同 output

`train_resnet18_baseline.py` 是参数化脚本。它不会把 BigGAN 或 SD1.5 写死，而是根据
命令行参数决定：

- 从哪个 metadata CSV 读取数据；
- 哪个 generator 用作训练；
- 哪个 generator 用作 seen test；
- 哪个 generator 用作 unseen test；
- 输出结果写到哪个 output directory。

例如 BigGAN 到 SD1.5：

```bash
--metadata-path datasets/curated_genimage/metadata.csv
--train-generator BigGAN
--seen-generator BigGAN
--unseen-generator stable_diffusion_v_1_5
--output-dir outputs/M2/biggan_to_sd15
```

SD1.5 到 BigGAN：

```bash
--metadata-path datasets/curated_genimage_sd15_train/metadata.csv
--train-generator stable_diffusion_v_1_5
--seen-generator stable_diffusion_v_1_5
--unseen-generator BigGAN
--output-dir outputs/M2/sd15_to_biggan
```

所以同一个训练代码可以复用在不同 generator split 上，只要换参数和 output directory。

## 总结

两个方向现在都使用同一套 regularized/early-stopping 模型设置，因此结果是可比的。

主要结论是：

ResNet-18 能够学习 seen generator 上的 real/fake 分类，但单一 generator 训练仍然
无法得到稳健的 cross-generator AIGC detector。下一阶段更应该尝试 CLIP feature、
多生成器训练、ImageNet pretrained backbone 或频域特征，而不是继续单纯增加 epoch。
