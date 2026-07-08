# Milestone 2 输出说明

这个文件夹保存 Milestone 2 的基线实验结果。当前模型是一个
ResNet-18 二分类器，用来判断图像是 `real` 还是 `fake`，训练损失是
cross-entropy loss。

## 文件夹命名

```text
outputs/M2/
|-- biggan_to_sd15/      # 用 BigGAN 训练/验证，用 SD1.5 做 unseen 测试
`-- sd15_to_biggan/      # 用 SD1.5 训练/验证，用 BigGAN 做 unseen 测试
```

命名方式是 `训练生成器_to_unseen生成器`。前半部分表示模型主要从哪个生成器
学习 real/fake 区分方式，后半部分表示哪个生成器没有参与训练，用来测试跨生成
器泛化能力。

## 这个实验到底在看什么

这个实验不是只看模型能不能在训练过的生成器上分类正确。更重要的问题是：

模型学到的是通用的 AI 图像检测特征，还是只记住了某一个生成器自己的 artifact？

所以我们同时看两类结果：

- `seen test`：测试生成器和训练生成器一致，用来判断模型是否学会了当前训练域。
- `unseen test`：测试生成器没有出现在训练中，用来判断模型能不能跨生成器泛化。

如果 `seen test` 很高，但 `unseen test` 接近随机，说明模型不是完全没学会，而是
学到的东西太依赖训练生成器。

## 混淆矩阵怎么看

混淆矩阵的纵轴是 `True`，也就是真实标签；横轴是 `Predicted`，也就是模型预测。

```text
                 Predicted real    Predicted fake
True real        TN                FP
True fake        FN                TP
```

在本项目里：

- 左上角：真实 real，被预测成 real，正确。
- 右上角：真实 real，被预测成 fake，误报。
- 左下角：真实 fake，被预测成 real，漏检 AI 图。
- 右下角：真实 fake，被预测成 fake，正确检测 AI 图。

因为我们的目标是检测 AI 生成图，所以左下角特别关键。左下角越大，说明越多 AI
图被模型当成自然图，这是最严重的问题之一。

## 例子：`biggan_to_sd15` 的两张混淆矩阵

你截图里右边是 `confusion_matrix_seen.png`，也就是 seen BigGAN 测试结果：

```text
真实 real -> 预测 real: 5899
真实 real -> 预测 fake: 101
真实 fake -> 预测 real: 2
真实 fake -> 预测 fake: 5998
```

这说明模型在 BigGAN 测试集上几乎完全学会了分类：

- 6000 张真实图里，5899 张被正确识别为 real。
- 6000 张 BigGAN fake 图里，5998 张被正确识别为 fake。
- fake 漏检只有 2 张，说明在 seen domain 里模型非常会抓 BigGAN 的生成痕迹。

所以右图反映的是：模型在训练过的生成器分布上表现很好，训练本身是成功的。

你截图里左边是 `confusion_matrix_unseen.png`，也就是 unseen SD1.5 测试结果：

```text
真实 real -> 预测 real: 7876
真实 real -> 预测 fake: 124
真实 fake -> 预测 real: 7977
真实 fake -> 预测 fake: 23
```

这说明模型遇到 Stable Diffusion 1.5 后基本失效：

- 8000 张真实图里，7876 张仍然被预测成 real，这部分看起来还可以。
- 但 8000 张 SD1.5 fake 图里，只有 23 张被预测成 fake。
- 7977 张 SD1.5 fake 图被预测成 real，也就是几乎所有 AI 图都被漏检。

所以左图反映的是：模型没有学到能够迁移到 SD1.5 的通用 AI 检测特征。它更像是
学会了 BigGAN 的特定痕迹，而不是学会了“AI 图像”这个更一般的概念。

这就是为什么 `biggan_to_sd15` 的 seen 指标很高，但 unseen F1 极低。

## 指标怎么看

`accuracy` 表示整体预测正确率。因为测试集 real/fake 基本平衡，所以 accuracy
接近 `0.5` 时，通常意味着接近随机猜。

`F1` 是 fake 类 precision 和 recall 的综合指标。它回答的是：模型检测 AI 图
是否既准又全。对于本项目，F1 很重要，因为我们特别关心 fake 图有没有被抓出来。

`AUC` 看的是模型给 fake 图的分数排序能力。AUC 接近 `1.0` 表示模型通常会给
fake 图更高的 fake 概率；AUC 接近 `0.5` 表示模型基本没有区分 real/fake 的
有效排序能力。

简单说：

- F1 更接近“当前阈值下，模型真的检测得怎么样”。
- AUC 更接近“模型输出的 fake 分数本身有没有区分能力”。

## 实验一：BigGAN 训练，SD1.5 unseen

文件夹：`biggan_to_sd15`

```text
Seen BigGAN test:
accuracy = 0.9914
F1       = 0.9915
AUC      = 0.9977

Unseen SD1.5 test:
accuracy = 0.4937
F1       = 0.0056
AUC      = 0.5230
```

这个结果说明模型在 BigGAN 上训练得很好。Seen accuracy、F1、AUC 都接近 1，
说明模型可以非常稳定地区分 BigGAN fake 和 ImageNet real。

但是 unseen SD1.5 上几乎崩掉。F1 只有 `0.0056`，结合混淆矩阵可以看出，主要
原因不是模型把 real 误报成 fake，而是模型几乎把所有 SD1.5 fake 都预测成 real。

因此，这个实验的核心结论是：BigGAN 训练出来的模型学到的是 BigGAN-specific
artifact，不是足够通用的 AIGC 检测特征。

## BigGAN 方向的过拟合检查和调参结果

原始 BigGAN baseline 的训练日志显示出明显过拟合：

```text
epoch  train_loss  val_loss  val_accuracy  val_f1
1      0.0826      0.0767    0.9769        0.9775
2      0.0558      0.0936    0.9720        0.9720
3      0.0482      0.1181    0.9551        0.9543
```

也就是说，训练集 loss 持续下降，但验证集 loss 持续上升，验证 accuracy 和 F1
也下降。这不是只看 loss 的 calibration 问题，而是真正的验证集性能变差。

因此训练代码已经加入以下防过拟合机制：

- 使用更低学习率：`3e-4 -> 1e-4`。
- 使用更高 weight decay：`1e-4 -> 1e-3`。
- 在分类头前加入 dropout：`p = 0.2`。
- 使用 label smoothing：`0.05`。
- 加强训练增强：更大范围 random resized crop、color jitter、Gaussian blur、
  random grayscale、random erasing。
- 使用 cosine learning-rate schedule。
- 使用 `val_loss` 选择 best checkpoint。
- 加入 early stopping：如果 `val_loss` 连续 2 个 epoch 不改善，则停止训练。

重新训练后的 regularized 结果保存在 `biggan_to_sd15_regularized/`。训练日志为：

```text
epoch  train_loss  val_loss  val_accuracy  val_f1   val_auc
1      0.2226      0.1710    0.9786        0.9790   0.9926
2      0.1852      0.1818    0.9728        0.9729   0.9912
3      0.1628      0.5420    0.7715        0.7081   0.9858
```

该版本在 epoch 3 自动 early stop，并回滚使用 epoch 1 的最佳 checkpoint。最终测试
结果为：

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

这个调参结果说明两件事：

第一，过拟合确实存在，所以训练流程必须使用 early stopping，而不能无条件训练
更多 epoch。

第二，即使用更强正则化和增强，unseen SD1.5 仍然接近随机。这说明问题不只是
普通意义上的 overfitting，而是单一 BigGAN 训练源导致的 generator-specific
learning。模型可以在 BigGAN 上学得很好，但这些特征仍然不能迁移到 SD1.5。

## 实验二：SD1.5 训练，BigGAN unseen

文件夹：`sd15_to_biggan`

```text
Seen SD1.5 test:
accuracy = 0.8310
F1       = 0.8537
AUC      = 0.9681

Unseen BigGAN test:
accuracy = 0.5109
F1       = 0.4058
AUC      = 0.4977
```

这个实验说明 SD1.5 训练任务本身更难。Seen accuracy 没有 BigGAN 那么高，但
AUC 仍然很高，说明模型其实学到了一定的 real/fake 排序信号，只是默认阈值下
仍然会犯一些分类错误。

但是 unseen BigGAN 的 AUC 接近 `0.5`，说明模型对 BigGAN real/fake 几乎没有
可靠排序能力。虽然 F1 比上一个 unseen 结果高一些，但这并不代表真的泛化好了；
更可能只是默认阈值让模型多预测了一些 fake，碰巧提高了 F1。

因此，这个实验同样说明：只用单一生成器训练，还不足以得到稳定的跨生成器检测器。

## 每张图反映什么训练情况

`training_curves.png` 显示训练过程中 loss 和验证指标的变化。它主要用来判断模型
有没有学起来、是否过拟合、以及验证集性能是否稳定。

`confusion_matrix_seen.png` 显示 seen 测试集的错误类型。如果对角线很深，说明模型
在训练过的生成器分布上分类效果好。

`confusion_matrix_unseen.png` 显示 unseen 测试集的错误类型。这张图最关键，因为它
直接反映模型遇到新生成器时会不会失效。

`seen_unseen_comparison.png` 把 seen 和 unseen 的 accuracy、F1、AUC 放在一起比较。
如果 seen 高、unseen 低，说明模型存在明显 domain shift 问题。

## 表格文件说明

`metrics.csv` 保存 validation、seen test、unseen test 的 loss、accuracy、F1、
AUC 和混淆矩阵计数。

`training_log.csv` 保存每个 epoch 的训练和验证指标。

`run_config.json` 保存这次训练运行时使用的参数。

`curation/dataset_statistics.csv` 出现在 `sd15_to_biggan/` 里，因为这个实验需要重新
划分数据，让 SD1.5 作为训练生成器，BigGAN 作为 unseen 生成器。

## 总结

两个方向的实验都说明同一个问题：ResNet-18 可以学会单一生成器上的 real/fake
分类，但还不能成为真正稳健的通用 AIGC 检测器。

下一步应该围绕跨生成器泛化改进，例如：

- 用多个生成器一起训练。
- 加强数据增强，减少模型对固定生成器 artifact 的依赖。
- 尝试频域特征或 artifact-focused feature。
- 使用 ImageNet 预训练 backbone，而不是完全随机初始化。
- 比较不同生成器组合下的 seen/unseen 泛化差距。
