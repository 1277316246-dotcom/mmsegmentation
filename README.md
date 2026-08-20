# 无人机低空航拍图像语义分割

本项目基于 MMSegmentation 完成无人机低空航拍图像的八类别语义分割。最终采用单模型 **SegFormer-B3 + 高置信度伪标签自训练 + 24-TTA**，线上最高成绩为 **69.4**。

## 1. 任务定义

模型预测以下八个类别：

| 训练 ID | 原始/提交 ID | 类别 |
|---:|---:|---|
| 0 | 1 | Background |
| 1 | 2 | Building |
| 2 | 3 | Road |
| 3 | 4 | Water |
| 4 | 5 | Barren |
| 5 | 6 | Vegetation |
| 6 | 7 | Agricultural |
| 7 | 8 | Vehicle |

原始标签 ID 0 为 Ignore。数据集启用 `reduce_zero_label=True` 后，原始 ID 1-8 映射为模型训练 ID 0-7，Ignore 映射为 255。提交脚本会将模型输出重新加 1，生成比赛要求的 ID 1-8 单通道灰度 PNG。

## 2. 最终技术路线

```text
6296 张真标签训练图
        ↓
SegFormer-B3 监督训练（40k iter）
        ↓
线上 68.4 的基线模型
        ↓
对 500 张测试图执行 24-TTA，生成置信度不低于 0.90 的伪标签
        ↓
6296 张真标签 + 500 张伪标签联合微调（40k iter，lr=2e-5）
        ↓
最终模型：验证集 TTA mIoU 76.97，线上 69.4
        ↓
640×640 重叠滑窗 + 24-TTA 推理并输出提交标签
```

最终配置与权重：

```text
configs/segformer/segformer_b3_mydata_pseudo.py
work_dirs/segformer_b3_mydata_pseudo/best_mIoU_iter_40000.pth
```

## 3. 针对性改进

### 3.1 SegFormer-B3 多尺度特征建模

模型由 SegFormer-B0 基线升级为 MiT-B3：

- `embed_dims=64`；
- 注意力头数为 `[1, 2, 5, 8]`；
- 编码层数为 `[3, 4, 18, 3]`；
- 解码头输入通道为 `[64, 128, 320, 512]`；
- 输出类别数改为 8；
- 使用 MiT-B3 ImageNet 预训练权重初始化。

训练时对 MiT-B3 编码器和分割解码头进行全网络微调，没有冻结主干网络。项目保留了历史配置文件名，但实际网络是 SegFormer-B3，并非 DeepLabV3+。

### 3.2 航拍数据增强

训练阶段采用以下增强策略提升尺度和域泛化能力：

- 0.5-2.0 比例随机缩放；
- 640×640 随机裁剪；
- 90°、180°、270°随机旋转；
- 水平翻转；
- 亮度、对比度、饱和度和色调扰动；
- 小概率随机遮挡；
- 适度的 Barren 类别感知裁剪。

### 3.3 联合损失函数

训练损失由普通交叉熵与 DiceLoss 组成：

```text
Loss = CrossEntropyLoss + 0.5 × DiceLoss
```

交叉熵负责逐像素分类，DiceLoss增强区域重叠学习并缓解类别不均衡。最终方案未采用曾导致数值不稳定的 FocalLoss 组合。

### 3.4 高置信度伪标签自训练

第一阶段模型对 500 张无标签测试图执行 24-TTA，并对 softmax 概率进行平均。仅保留最大概率不低于 0.90 的像素，其余像素写为原始 ID 0，在训练加载后变为 Ignore 255。

伪标签统计：

- 平均保留像素比例：66.1%；
- 平均最大 softmax：0.888；
- 真标签图片：6296 张；
- 伪标签图片：500 张；
- 联合训练列表：6796 张。

自训练从 68.4 分模型的完整权重继续微调，学习率由 `6e-5` 降至 `2e-5`，而不是重新从 ImageNet 权重开始训练。

### 3.5 滑窗推理

验证和提交时保留 1024×1024 原始分辨率，使用：

```text
滑窗大小：640×640
滑窗步长：480×480
```

重叠区域由模型内部融合，可减少直接缩放对窄道路、小车辆和建筑边界造成的细节损失。

### 3.6 24-TTA

最终提交使用同一个 SegFormer-B3 模型进行 24 次测试时增强：

```text
3 个尺度（0.75、1.0、1.25）
× 4 个方向（0°、90°、180°、270°）
× 2 个翻转状态（原图、水平翻转）
= 24 个预测结果
```

所有预测恢复到原始方向和尺寸后，对 softmax 概率逐像素平均，再取最大概率类别。该方法属于单模型测试时增强，不属于模型融合。

## 4. 当前精度

最终模型在 700 张验证图上的 6-TTA 结果如下：

| 类别 | IoU | Acc |
|---|---:|---:|
| Background | 68.08 | 77.50 |
| Building | 82.39 | 92.00 |
| Road | 79.02 | 89.78 |
| Water | 87.75 | 93.46 |
| Barren | 51.66 | 75.16 |
| Vegetation | 86.79 | 93.26 |
| Agricultural | 80.21 | 94.03 |
| Vehicle | 79.90 | 89.72 |

汇总指标：

| 指标 | 结果 |
|---|---:|
| 验证集 aAcc | 87.86 |
| 验证集 mIoU | 76.97 |
| 验证集 mAcc | 88.11 |
| 线上成绩 | **69.4** |

## 5. 精度提升对比

| 方案 | 验证集 TTA mIoU | 线上结果 |
|---|---:|---:|
| SegFormer-B3 监督训练 | 75.54 | 68.4 |
| 高置信度伪标签自训练 | **76.97** | **69.4** |

高置信度伪标签自训练使验证集 mIoU 提升 1.43，线上成绩提升 1.0，表明利用置信度过滤后的测试域伪标签进行低学习率微调能够有效缓解域偏移。

## 6. 数据目录

```text
data/mydata/
├── JPEGImages/
├── SegmentationClass/
├── train.txt
├── val.txt
└── train_pseudo.txt

test_data/
└── test1_*.png
```

`train.txt` 和 `val.txt` 每行保存不带 `.png` 后缀的图片主文件名。`train_pseudo.txt` 为原始训练列表与 500 张测试图名称的组合。

## 7. 训练与验证

伪标签自训练模型训练：

```bash
python tools/train.py configs/segformer/segformer_b3_mydata_pseudo.py --work-dir work_dirs/segformer_b3_mydata_pseudo
```

使用配置内的 3 尺度 × 水平翻转（6-TTA）验证最终权重：

```bash
python tools/test.py configs/segformer/segformer_b3_mydata_pseudo.py work_dirs/segformer_b3_mydata_pseudo/best_mIoU_iter_40000.pth --tta
```

注意：`tools/test.py --tta` 使用的是配置文件内的 6-TTA；最终提交脚本使用的是 24-TTA。

## 8. 生成提交标签

```bash
python demo/infer_gray_label.py test_data configs/segformer/segformer_b3_mydata_pseudo.py work_dirs/segformer_b3_mydata_pseudo/best_mIoU_iter_40000.pth --out-dir outputs_final --zip-file outputs_final/submission.zip --device cuda:0 --scales 0.75 1.0 1.25 --rotations 0 1 2 3 --hflip
```

提交文件为：

```text
outputs_final/submission.zip
```

ZIP 根目录直接包含 500 张 1024×1024、单通道、`uint8` PNG标签图，像素值范围为 1-8。

## 9. 最终结论

项目最终采用单模型 SegFormer-B3，通过航拍方向与尺度增强、CE与Dice联合监督、640重叠滑窗、高置信度伪标签自训练和24-TTA，将线上成绩从68.4提升到69.4，验证集TTA mIoU达到76.97。
