m# MMSegmentation 自定义数据集训练教程

本教程以遥感场景（9类：Ignore, Background, Building, Road, Water, Barren, Vegetation, Agricultural, Vehicle）为例，演示如何基于 MMSegmentation 的 Pascal VOC 数据集模板，修改三个关键文件来训练自己的数据。

---

## 一、数据集目录结构

自定义数据集需按以下结构组织：

```
data/mydata/
├── JPEGImages/          # 原图（.png 格式）
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── SegmentationClass/   # 标注图（.png 格式，像素值为类别索引 0-8）
│   ├── 0000.png
│   ├── 0001.png
│   └── ...
├── train.txt            # 训练集文件名列表（不含后缀）
└── val.txt              # 验证集文件名列表（不含后缀）
```

`train.txt` / `val.txt` 内容示例：
```
0935
6386
0027
4280
1134
```

---

## 二、修改文件对比

### 2.1 数据集类定义：`mmseg/datasets/voc.py`

修改 `PascalVOCDataset` 的 `METAINFO` 和 `__init__` 参数。

**原始代码：**
```python
METAINFO = dict(
    classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
             'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
             'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
             'sofa', 'train', 'tvmonitor'),
    palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
             [0, 64, 128]])

def __init__(self,
             ann_file,
             img_suffix='.jpg',       # <-- VOC 原图是 .jpg
             seg_map_suffix='.png',
             **kwargs) -> None:
```

**修改后：**
```python
METAINFO = dict(
    classes=('Ignore', 'Background', 'Building', 'Road', 'Water',
             'Barren', 'Vegetation', 'Agricultural', 'Vehicle'),
    palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
             [64, 0, 0]])

def __init__(self,
             ann_file,
             img_suffix='.png',       # <-- 改为 .png
             seg_map_suffix='.png',
             **kwargs) -> None:
```

**修改要点：**
| 项目 | 原始 | 修改后 | 说明 |
|------|------|--------|------|
| `classes` | 21类（VOC物体类） | 9类（遥感场景类） | 类别名必须与标注像素值一一对应 |
| `palette` | 21组RGB | 9组RGB | 可视化颜色，数量需与 classes 一致 |
| `img_suffix` | `.jpg` | `.png` | 原图格式改为 png |

---

### 2.2 数据集配置：`configs/_base_/datasets/pascal_voc12.py`

**原始代码：**
```python
dataset_type = 'PascalVOCDataset'
data_root = 'data/VOCdevkit/VOC2012'          # <-- VOC 原始路径
...
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='ImageSets/Segmentation/train.txt',  # <-- VOC 子目录
        ...))
val_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='ImageSets/Segmentation/val.txt',    # <-- VOC 子目录
        ...))
```

**修改后：**
```python
dataset_type = 'PascalVOCDataset'
data_root = 'data/mydata'                      # <-- 自定义数据路径
...
train_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='train.txt',                   # <-- 简化路径
        ...))
val_dataloader = dict(
    ...
    dataset=dict(
        ...
        ann_file='val.txt',                     # <-- 简化路径
        ...))
```

**修改要点：**
| 项目 | 原始 | 修改后 | 说明 |
|------|------|--------|------|
| `data_root` | `data/VOCdevkit/VOC2012` | `data/mydata` | 指向自定义数据根目录 |
| `ann_file` (train) | `ImageSets/Segmentation/train.txt` | `train.txt` | 训练集列表文件 |
| `ann_file` (val) | `ImageSets/Segmentation/val.txt` | `val.txt` | 验证集列表文件 |

> `data_prefix` 中的 `img_path='JPEGImages'` 和 `seg_map_path='SegmentationClass'` 保持不变，因为自定义数据的目录结构与 VOC 一致。

---

### 2.3 模型配置：`configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py`

这是新建的配置文件，继承基础配置并覆盖类别数。

**新建文件：**
```python
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=9),       # <-- 原始为 19（Cityscapes）或 21（VOC）
    auxiliary_head=dict(num_classes=9))    # <-- 同上
```

**对比原始 VOC 配置（`deeplabv3plus_r50-d8_4xb4-40k_voc12aug-512x512.py`）：**
```python
_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/pascal_voc12_aug.py',   # <-- 使用 aug 版本
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=21),           # <-- VOC 21类
    auxiliary_head=dict(num_classes=21))        # <-- VOC 21类
```

**修改要点：**
| 项目 | 原始 VOC 配置 | 自定义配置 | 说明 |
|------|---------------|------------|------|
| 数据集引用 | `pascal_voc12_aug.py` | `pascal_voc12.py` | 不使用 VOC 数据增强 |
| `num_classes` | 21 | 9 | 必须与 voc.py 中的 classes 数量一致 |

---

## 三、修改总结

训练自定义数据只需修改 **3 处**：

1. **`mmseg/datasets/voc.py`** — 修改类别名、调色板、图片后缀
2. **`configs/_base_/datasets/pascal_voc12.py`** — 修改数据路径和标注文件路径
3. **模型配置文件** — 修改 `num_classes` 为实际类别数

---

## 四、启动训练

```bash
# 单 GPU 训练
python tools/train.py configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py

# 多 GPU 训练（例如 4 卡）
bash tools/dist_train.sh configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py 4
```

---

## 五、测试（评估模型精度）

训练完成后，使用验证集评估模型指标（mIoU 等）：

```bash
# 单 GPU 测试
python tools/test.py \
    configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py \
    work_dirs/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512/best_mIoU_iter_XXXX.pth

# 多 GPU 测试（例如 4 卡）
bash tools/dist_test.sh \
    configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py \
    work_dirs/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512/best_mIoU_iter_XXXX.pth \
    4
```

输出示例：
```
+------------+-----------+-------+
|   Class    |    IoU    |  Acc  |
+------------+-----------+-------+
|   Ignore   |   0.000   | 0.000 |
| Background |   0.852   | 0.923 |
|  Building  |   0.781   | 0.856 |
|    Road    |   0.763   | 0.832 |
|   Water    |   0.891   | 0.945 |
|   Barren   |   0.654   | 0.743 |
| Vegetation |   0.812   | 0.889 |
| Agricultural|  0.723   | 0.812 |
|  Vehicle   |   0.534   | 0.678 |
+------------+-----------+-------+
|    mIoU    |   0.668   |       |
|    mAcc    |           | 0.742 |
+------------+-----------+-------+
```

> `best_mIoU_iter_XXXX.pth` 是训练过程中自动保存的最优权重，位于 `work_dirs/` 目录下。也可使用 `iter_40000.pth`（最后一次迭代）。

---

## 六、推理（预测单张图片）

### 6.1 命令行推理

```bash
python demo/image_demo.py \
    data/mydata/JPEGImages/0000.png \
    configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py \
    work_dirs/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512/best_mIoU_iter_XXXX.pth \
    --out-file result.png \
    --opacity 0.5
```

参数说明：
- `--out-file`：输出可视化结果保存路径（分割掩码叠加在原图上）
- `--opacity`：掩码透明度，0~1 之间，0.5 表示半透明叠加

### 6.2 Python 脚本推理

```python
from mmengine.config import Config
from mmseg.apis import init_model, inference_model, show_result_pyplot

# 1. 加载模型
config_path = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py'
checkpoint_path = 'work_dirs/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512/best_mIoU_iter_XXXX.pth'
model = init_model(Config.fromfile(config_path), checkpoint_path, device='cuda:0')

# 2. 推理单张图片
img_path = 'data/mydata/JPEGImages/0000.png'
result = inference_model(model, img_path)

# 3. 可视化并保存
show_result_pyplot(model, img_path, result, out_file='result.png', opacity=0.5)

# 4. 获取预测的类别掩码（numpy 数组，形状 HxW，值为类别索引 0~8）
pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
print('预测类别:', pred_mask.unique())  # 查看预测包含哪些类别
```

### 6.3 批量推理

```python
import os
import numpy as np
from mmengine.config import Config
from mmseg.apis import init_model, inference_model
from PIL import Image

config_path = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py'
checkpoint_path = 'work_dirs/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512/best_mIoU_iter_XXXX.pth'
model = init_model(Config.fromfile(config_path), checkpoint_path, device='cuda:0')

img_dir = 'data/mydata/JPEGImages'
save_dir = 'data/mydata/Predictions'
os.makedirs(save_dir, exist_ok=True)

for img_name in sorted(os.listdir(img_dir)):
    if not img_name.endswith('.png'):
        continue
    img_path = os.path.join(img_dir, img_name)
    result = inference_model(model, img_path)
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
    Image.fromarray(pred_mask).save(os.path.join(save_dir, img_name))
    print(f'已保存: {img_name}')
```

---

## 七、注意事项

1. **标注图像素值**：标注图的像素值必须是 0~N-1 的整数（N 为类别数），255 通常表示忽略区域。本例中像素值 0=Ignore, 1=Background, ..., 8=Vehicle。
2. **类别数一致性**：`voc.py` 中的 `classes` 数量、模型配置中的 `num_classes`、标注图中的最大像素值 + 1，三者必须一致。
3. **图片后缀**：`img_suffix` 必须与实际图片格式匹配，本例为 `.png`。
4. **文件名列表**：`train.txt` / `val.txt` 中每行一个文件名（不含后缀和目录），对应 `JPEGImages/` 和 `SegmentationClass/` 下的文件。
