# check_my_config.py
import numpy as np
from PIL import Image
from mmengine.config import Config

from mmseg.registry import DATASETS
from mmseg.utils import register_all_modules


CONFIG_PATH = 'configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py'

EXPECTED_CLASSES = (
    'Background',
    'Building',
    'Road',
    'Water',
    'Barren',
    'Vegetation',
    'Agricultural',
    'Vehicle',
)


def check_dataset(name, dataset_cfg):
    dataset = DATASETS.build(dataset_cfg)

    print(f'\n===== {name} =====')
    print('数据集类型:', type(dataset).__name__)
    print('类别:', dataset.metainfo['classes'])
    print('类别数量:', len(dataset.metainfo['classes']))
    print('reduce_zero_label:', dataset.reduce_zero_label)
    print('ignore_index:', dataset.ignore_index)
    print('图片数量:', len(dataset))

    assert tuple(dataset.metainfo['classes']) == EXPECTED_CLASSES, (
        f'{name} 类别定义错误'
    )
    assert dataset.reduce_zero_label is True, (
        f'{name} reduce_zero_label 没有开启'
    )
    assert dataset.ignore_index == 255, (
        f'{name} ignore_index 应为 255'
    )

    return dataset


def check_raw_labels(dataset):
    """检查验证集原始标签是否为单通道的 0-8。"""
    pixel_counts = np.zeros(9, dtype=np.int64)
    invalid_values = set()

    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        label_path = info['seg_map_path']

        with Image.open(label_path) as image:
            label = np.asarray(image)

        if label.ndim != 2:
            raise ValueError(
                f'标签必须是单通道图像，但 {label_path} '
                f'的形状为 {label.shape}'
            )

        values, counts = np.unique(label, return_counts=True)

        for value, count in zip(values, counts):
            value = int(value)

            if 0 <= value <= 8:
                pixel_counts[value] += int(count)
            else:
                invalid_values.add(value)

    print('\n===== 验证集原始标签统计 =====')

    raw_names = (
        'Ignore',
        'Background',
        'Building',
        'Road',
        'Water',
        'Barren',
        'Vegetation',
        'Agricultural',
        'Vehicle',
    )

    for label_id, class_name in enumerate(raw_names):
        print(
            f'原始值 {label_id}: {class_name:<12} '
            f'像素数量={pixel_counts[label_id]}'
        )

    assert not invalid_values, (
        f'发现 0-8 以外的标签值: {sorted(invalid_values)}'
    )

    missing = [
        raw_names[label_id]
        for label_id in range(1, 9)
        if pixel_counts[label_id] == 0
    ]

    if missing:
        print(f'\n[警告] 验证集缺少这些类别: {missing}')
        print('缺少的类别在验证结果中会显示 nan。')
    else:
        print('\n验证集包含全部 8 个有效类别。')


def check_transformed_label(dataset):
    """检查经过 LoadAnnotations 后是否正确转换。"""
    sample = dataset[0]

    gt = (
        sample['data_samples']
        .gt_sem_seg.data
        .cpu()
        .numpy()
    )

    unique_values = np.unique(gt).tolist()

    print('\n===== 标签转换结果 =====')
    print('转换后标签值:', unique_values)
    print('正确范围应为: 0-7，以及可能存在的 255')

    allowed_values = set(range(8)) | {255}

    assert set(unique_values).issubset(allowed_values), (
        f'转换后出现异常标签值: {unique_values}'
    )


def main():
    register_all_modules()
    cfg = Config.fromfile(CONFIG_PATH)

    print('===== 模型配置 =====')
    print(
        'decode_head.num_classes =',
        cfg.model.decode_head.num_classes,
    )
    print(
        'auxiliary_head.num_classes =',
        cfg.model.auxiliary_head.num_classes,
    )

    assert cfg.model.decode_head.num_classes == 8
    assert cfg.model.auxiliary_head.num_classes == 8

    train_dataset = check_dataset(
        '训练集',
        cfg.train_dataloader.dataset,
    )
    val_dataset = check_dataset(
        '验证集',
        cfg.val_dataloader.dataset,
    )
    check_dataset(
        '测试集',
        cfg.test_dataloader.dataset,
    )

    check_raw_labels(val_dataset)
    check_transformed_label(train_dataset)

    print('\n================================')
    print('全部检查通过，配置修改正确。')
    print('================================')


if __name__ == '__main__':
    main()