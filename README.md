# 488FP 项目仓库

这是 ECE4880J 计算机视觉课程项目的代码与实验输出仓库。项目主题是
GenImage 数据集上的 AIGC 图像检测，也就是判断输入图像是 `real` 还是 `fake`。

项目重点不是只追求 seen generator 上的高准确率，而是研究模型在
cross-generator distribution shift 下是否仍然可靠。

## 仓库结构

```text
488FP/
|-- projects/          # 数据整理、分析、训练脚本
|-- configs/           # 配置文件
|-- outputs/           # 课程 milestone 需要提交/保留的结果
|-- README.md          # 项目总说明
|-- requirements.txt   # Python 依赖
`-- datasets/          # 本地/服务器数据集目录，不提交到 Git
```

## 主要脚本

更详细的脚本说明见 `projects/README.md`。

`projects/curate_biggan_to_sd15_split.py` 用于创建最初的 Milestone 1 数据划分：
BigGAN 用作训练、验证和 seen 测试生成器，Stable Diffusion v1.5 作为 unseen 测试
生成器。

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/curate_biggan_to_sd15_split.py
```

`projects/curate_sd15_to_biggan_split.py` 用于创建反向 Milestone 2 数据划分：
Stable Diffusion v1.5 用作训练、验证和 seen 测试生成器，BigGAN 作为 unseen 测试
生成器。

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/curate_sd15_to_biggan_split.py
```

`projects/analyze_m1_resolutions.py` 读取 `outputs/M1/metadata.csv`，统计图像尺寸、格式
和无效图像，并导出 Milestone 1 的分辨率分析图表。

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/analyze_m1_resolutions.py
```

`projects/train_resnet18_baseline.py` 用于训练和评估 ResNet-18 real/fake 二分类基线。
脚本包含 early stopping、regularization、data augmentation、checkpoint 保存、
指标计算和绘图。

BigGAN 到 SD1.5 的训练示例：

```bash
cd /root/autodl-tmp/488FP
/root/miniconda3/bin/python projects/train_resnet18_baseline.py \
  --metadata-path datasets/curated_genimage/metadata.csv \
  --train-generator BigGAN \
  --seen-generator BigGAN \
  --unseen-generator stable_diffusion_v_1_5 \
  --output-dir outputs/M2/biggan_to_sd15 \
  --weights none
```

SD1.5 到 BigGAN 的训练示例：

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

## 数据集位置

原始数据集存放在 AutoDL/服务器实例上，不提交到 Git。

```text
/root/autodl-tmp/488FP/datasets/GenImage
```

当前使用的 GenImage 子集：

```text
datasets/GenImage/BigGAN/imagenet_ai_0419_biggan/{train,val}/{ai,nature}
datasets/GenImage/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5/{train,val}/{ai,nature}
```

其中：

- `ai` 表示 AI 生成图，训练标签为 `fake`。
- `nature` 表示真实自然图，训练标签为 `real`。

## 输出结果

Milestone 1 结果位于 `outputs/M1/`，包括数据 metadata、统计表、样例图和分辨率分析。

Milestone 2 结果位于 `outputs/M2/`：

```text
outputs/M2/
|-- README.md
|-- biggan_to_sd15/
`-- sd15_to_biggan/
```

每个 M2 实验文件夹包含：

- `metrics.csv`：validation、seen test、unseen test 的 loss、accuracy、F1、AUC 和混淆矩阵计数。
- `training_log.csv`：每个 epoch 的训练和验证指标。
- `training_curves.png`：训练曲线。
- `confusion_matrix_seen.png`：seen generator 上的混淆矩阵。
- `confusion_matrix_unseen.png`：unseen generator 上的混淆矩阵。
- `seen_unseen_comparison.png`：seen/unseen 指标对比图。

更详细的结果解释见 `outputs/M2/README.md`。

## 当前主要结论

ResNet-18 可以很好地学习单一 seen generator 上的 real/fake 分类，但不能自然泛化到
unseen generator。当前两个 M2 方向都使用同一套 regularized/early-stopping 训练设置，
因此结果可比。

BigGAN 到 SD1.5 的实验显示，模型在 BigGAN seen test 上几乎完美，但在 SD1.5 fake
图像上几乎全部漏检。加入更强 regularization 和 early stopping 后，过拟合过程得到
控制，但 unseen SD1.5 仍然接近随机。

这说明当前问题不只是普通 overfitting，而是 single-generator training 导致的
generator-specific learning。

## Git 管理原则

提交到 Git 的内容包括代码、配置、文档和选定的 milestone 输出。

以下内容不提交：

- 原始数据集；
- checkpoint 权重文件；
- 压缩包和下载中间产物；
- 训练日志；
- 临时实验目录；
- cache 文件。
