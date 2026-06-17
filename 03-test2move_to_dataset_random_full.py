#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本功能说明：

本脚本用于将每天生成的多模态车窗 crop 数据整理成适合 pix2pixHD
或自定义多模态图像翻译模型训练的数据结构。

功能：
1. 支持将文件复制 / 剪切 / 硬链接 / 软链接到目标数据集目录。
2. 支持额外转移：
   - dahua_window.png
   - _dahua_paste_back_meta.json
3. 支持随机剪切指定数量的数据组，或者处理全部数据组。
4. 会生成 manifest.csv 和 skipped.csv，方便追踪成功和跳过的数据。

输入数据结构：
DAY_DATA_DIR/
├── 0001_xxx/
│   ├── _dahua_paste_back_meta.json
│   ├── dahua_window.png
│   ├── GT_Camera_usb.npy
│   ├── Nir_Camera_usb.npy
│   ├── NIR_Long_Camera_248.npy
│   ├── RGB_Camera_249.npy
│   └── RGB_Long_Camera_247.npy
├── 0002_xxx/
│   └── ...

输出数据结构：
DEST_ROOT/
├── GT/
│   └── 20260603_000001.npy
├── NIR/
│   └── 20260603_000001.npy
├── NIR_Long/
│   └── 20260603_000001.npy
├── RGB/
│   └── 20260603_000001.npy
├── RGB_Long/
│   └── 20260603_000001.npy
├── DAHUA_Window/
│   └── 20260603_000001.png
├── paste_back_meta/
│   └── 20260603_000001.json
└── manifests/
    ├── 20260603_manifest.csv
    └── 20260603_skipped.csv
"""

import csv
import shutil
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm

# =========================================================
# 1. 路径配置
# =========================================================
# 每天的数据路径：该目录下直接包含 0001_xxx、0002_xxx 这样的组文件夹
DAY_DATA_DIR = Path(
    r"/mnt/bigdata/ndy-5JAI/01_processed_data/processed_2026-06-05/_window_crop copy"
)

# 每日日期前缀，用于生成统一文件名
# 例如 DATE_PREFIX = "20260603"，输出文件名为 20260603_000001.npy
DATE_PREFIX = "20260605"

# 整理后的训练数据保存路径
DEST_ROOT = Path(r"/mnt/bigdata/ndy-5JAI/dataset-5JAI/test2")


# =========================================================
# 2. 处理选项配置
# =========================================================
# 是否只处理数字开头的组文件夹
# 例如 0002_205458_415 会处理，_checker_grid_gt 不会处理
ONLY_DIGIT_PREFIX_GROUP = True

# 是否检查五个模态的 H/W 是否一致
CHECK_SHAPE = True

# 是否要求 dahua_window.png 和 _dahua_paste_back_meta.json 必须存在
# True：缺少任意一个就跳过该组
# False：缺少时只记录为空，不影响五个 npy 转移
REQUIRE_EXTRA_FILES = True

# 是否覆盖目标路径下已经存在的同名文件
OVERWRITE = False

# 文件转移方式：
# "move"     ：剪切/移动，源文件会被移动到目标路径，源文件会消失；
# "copy"     ：普通复制，源文件保留；
# "hardlink" ：硬链接，速度快、省空间，但源和目标必须在同一磁盘分区；
# "symlink"  ：软链接，省空间，但原始数据移动后链接会失效。
COPY_MODE = "move"

# 编号起始值
START_INDEX = 1

# 是否自动跳过已经存在的编号
# True：如果 20260603_000001.npy 已存在，会自动尝试 20260603_000002.npy
AUTO_SKIP_EXISTING_SAMPLE_ID = True


# =========================================================
# 2.1 随机剪切数量配置
# =========================================================
# RANDOM_SAMPLE_COUNT = "all" 或 None：处理全部有效组
# RANDOM_SAMPLE_COUNT = 50：从有效组中随机剪切 50 组
RANDOM_SAMPLE_COUNT = "100"

# 随机种子：
# 固定为整数，例如 42，每次随机结果一样；
# 设置为 None，每次运行随机结果不同。
RANDOM_SEED = 42


# =========================================================
# 3. 文件名配置
# =========================================================
MODALITY_FILE_MAP = {
    "GT": "GT_Camera_usb.npy",
    "NIR": "Nir_Camera_usb.npy",
    "NIR_Long": "NIR_Long_Camera_248.npy",
    "RGB": "RGB_Camera_249.npy",
    "RGB_Long": "RGB_Long_Camera_247.npy",
}

MODALITY_ORDER = [
    "GT",
    "NIR",
    "NIR_Long",
    "RGB",
    "RGB_Long",
]

# 贴回相关文件
# key 用于 manifest 字段名；dst_dir 表示输出子文件夹；dst_ext 表示统一命名后的后缀
EXTRA_FILE_MAP = {
    "DAHUA_Window": {
        "src_name": "dahua_window.png",
        "dst_dir": "DAHUA_Window",
        "dst_ext": ".png",
    },
    "PasteBackMeta": {
        "src_name": "_dahua_paste_back_meta.json",
        "dst_dir": "paste_back_meta",
        "dst_ext": ".json",
    },
}

EXTRA_ORDER = [
    "DAHUA_Window",
    "PasteBackMeta",
]


# =========================================================
# 4. 工具函数
# =========================================================
def is_digit_prefix_folder(path: Path):
    return path.is_dir() and len(path.name) > 0 and path.name[0].isdigit()


def ensure_output_dirs():
    """
    检查输出文件夹是否存在。
    如果不存在，则自动创建。
    """
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for modality in MODALITY_ORDER:
        modality_dir = DEST_ROOT / modality
        modality_dir.mkdir(parents=True, exist_ok=True)

    for extra_key in EXTRA_ORDER:
        extra_cfg = EXTRA_FILE_MAP[extra_key]
        extra_dir = DEST_ROOT / extra_cfg["dst_dir"]
        extra_dir.mkdir(parents=True, exist_ok=True)

    manifest_dir = DEST_ROOT / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)


def get_group_folders(day_dir: Path):
    """
    获取每日数据路径下的所有组文件夹。
    """
    if not day_dir.exists():
        raise FileNotFoundError(f"每日数据路径不存在: {day_dir}")

    if not day_dir.is_dir():
        raise NotADirectoryError(f"每日数据路径不是文件夹: {day_dir}")

    folders = sorted([p for p in day_dir.iterdir() if p.is_dir()])

    if ONLY_DIGIT_PREFIX_GROUP:
        folders = [p for p in folders if is_digit_prefix_folder(p)]

    return folders


def get_npy_hw(path: Path):
    """
    只读取 npy header，不完整加载大数组。
    """
    arr = np.load(path, mmap_mode="r")

    if len(arr.shape) < 2:
        raise ValueError(f"npy 维度异常: {path}, shape={arr.shape}")

    h, w = arr.shape[0], arr.shape[1]

    return int(h), int(w), str(arr.shape)


def validate_group(group_dir: Path):
    """
    检查单组数据是否完整。
    """
    src_paths = {}

    # 1. 检查五个 npy 模态文件
    for modality, filename in MODALITY_FILE_MAP.items():
        src_path = group_dir / filename

        if not src_path.exists():
            return False, f"缺失文件: {filename}", {}

        src_paths[modality] = src_path

    # 2. 检查贴回相关文件
    extra_paths = {}

    for extra_key in EXTRA_ORDER:
        extra_cfg = EXTRA_FILE_MAP[extra_key]
        src_name = extra_cfg["src_name"]
        src_path = group_dir / src_name

        if not src_path.exists():
            if REQUIRE_EXTRA_FILES:
                return False, f"缺失贴回文件: {src_name}", {}
            else:
                extra_paths[extra_key] = None
                continue

        extra_paths[extra_key] = src_path

    # 3. 检查五个 npy 的 H/W 是否一致
    shape_info = {}

    if CHECK_SHAPE:
        hw_list = []

        try:
            for modality, src_path in src_paths.items():
                h, w, shape_str = get_npy_hw(src_path)
                hw_list.append((h, w))
                shape_info[modality] = shape_str

            first_hw = hw_list[0]

            for hw in hw_list[1:]:
                if hw != first_hw:
                    return False, f"五个模态 H/W 不一致: {shape_info}", {}

            height, width = first_hw

        except Exception as e:
            return False, f"读取 npy shape 失败: {e}", {}

    else:
        height, width = "", ""

    info = {
        "src_paths": src_paths,
        "extra_paths": extra_paths,
        "height": height,
        "width": width,
        "shape_info": shape_info,
    }

    return True, "OK", info


def sample_id_exists(sample_id: str):
    """
    判断某个 sample_id 是否已经在任意目标目录中存在。
    """
    npy_filename = f"{sample_id}.npy"

    for modality in MODALITY_ORDER:
        if (DEST_ROOT / modality / npy_filename).exists():
            return True

    for extra_key in EXTRA_ORDER:
        extra_cfg = EXTRA_FILE_MAP[extra_key]
        extra_filename = f"{sample_id}{extra_cfg['dst_ext']}"
        if (DEST_ROOT / extra_cfg["dst_dir"] / extra_filename).exists():
            return True

    return False


def make_sample_id(index: int):
    """
    生成统一样本名。
    """
    return f"{DATE_PREFIX}_{index:06d}"


def find_next_available_sample_id(index: int):
    """
    根据当前 index 生成 sample_id。
    如果开启 AUTO_SKIP_EXISTING_SAMPLE_ID，则自动跳过已经存在的编号。
    """
    while True:
        sample_id = make_sample_id(index)

        if not AUTO_SKIP_EXISTING_SAMPLE_ID:
            return sample_id, index

        if not sample_id_exists(sample_id):
            return sample_id, index

        index += 1


def transfer_one_file(src: Path, dst: Path):
    """
    按 COPY_MODE 转移文件。
    COPY_MODE = "move" 时表示剪切/移动，源文件会消失。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if OVERWRITE:
            dst.unlink()
        else:
            raise FileExistsError(f"目标文件已存在: {dst}")

    if COPY_MODE == "move":
        shutil.move(str(src), str(dst))

    elif COPY_MODE == "copy":
        shutil.copy2(src, dst)

    elif COPY_MODE == "hardlink":
        dst.hardlink_to(src)

    elif COPY_MODE == "symlink":
        dst.symlink_to(src.resolve())

    else:
        raise ValueError(f"不支持的 COPY_MODE: {COPY_MODE}")


def write_manifest_csv(records, csv_path: Path):
    """
    保存成功转移的数据记录。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_id",
        "source_group",
        "height",
        "width",
        "GT",
        "NIR",
        "NIR_Long",
        "RGB",
        "RGB_Long",
        "DAHUA_Window",
        "PasteBackMeta",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow(r)


def write_skipped_csv(records, csv_path: Path):
    """
    保存跳过的数据记录。
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_group",
        "reason",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in records:
            writer.writerow(r)


def select_items_by_random_count(valid_items):
    """
    根据 RANDOM_SAMPLE_COUNT 决定处理全部有效组，还是随机抽取部分有效组。
    """
    if RANDOM_SAMPLE_COUNT is None or RANDOM_SAMPLE_COUNT == "all":
        print(f"处理模式: 全部转移，共 {len(valid_items)} 组")
        return valid_items

    sample_count = int(RANDOM_SAMPLE_COUNT)

    if sample_count <= 0:
        print(f"处理模式: RANDOM_SAMPLE_COUNT={sample_count}，不转移任何数据")
        return []

    if sample_count > len(valid_items):
        print(
            f"警告：请求随机转移 {sample_count} 组，"
            f"但有效组只有 {len(valid_items)} 组，将全部转移。"
        )
        sample_count = len(valid_items)

    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)

    selected_items = random.sample(valid_items, sample_count)
    print(f"处理模式: 随机转移 {len(selected_items)} 组")

    return selected_items


# =========================================================
# 5. 主处理逻辑
# =========================================================
def copy_group_to_dataset(group_dir: Path, sample_id: str, info: dict):
    """
    将一组数据转移到 GT / NIR / NIR_Long / RGB / RGB_Long 平行目录中，
    同时转移 dahua_window.png 和 _dahua_paste_back_meta.json。
    """
    src_paths = info["src_paths"]
    extra_paths = info["extra_paths"]

    manifest_row = {
        "sample_id": sample_id,
        "source_group": group_dir.name,
        "height": info["height"],
        "width": info["width"],
    }

    # 1. 转移五个 npy 模态
    for modality in MODALITY_ORDER:
        src = src_paths[modality]
        dst = DEST_ROOT / modality / f"{sample_id}.npy"

        transfer_one_file(src, dst)

        # manifest 中保存相对路径，方便移动整个 DEST_ROOT
        rel_path = dst.relative_to(DEST_ROOT).as_posix()
        manifest_row[modality] = rel_path

    # 2. 转移贴回相关文件
    for extra_key in EXTRA_ORDER:
        src = extra_paths.get(extra_key)

        if src is None:
            manifest_row[extra_key] = ""
            continue

        extra_cfg = EXTRA_FILE_MAP[extra_key]
        dst = DEST_ROOT / extra_cfg["dst_dir"] / f"{sample_id}{extra_cfg['dst_ext']}"

        transfer_one_file(src, dst)

        rel_path = dst.relative_to(DEST_ROOT).as_posix()
        manifest_row[extra_key] = rel_path

    return manifest_row


def main():
    ensure_output_dirs()

    group_folders = get_group_folders(DAY_DATA_DIR)

    print(f"每日数据路径: {DAY_DATA_DIR}")
    print(f"输出数据路径: {DEST_ROOT}")
    print(f"日期前缀: {DATE_PREFIX}")
    print(f"发现组文件夹数量: {len(group_folders)}")
    print(f"是否要求贴回文件存在: {REQUIRE_EXTRA_FILES}")
    print(f"转移方式: {COPY_MODE}")
    print(f"随机转移数量: {RANDOM_SAMPLE_COUNT}")
    print(f"随机种子: {RANDOM_SEED}")

    manifest_records = []
    skipped_records = []

    # 1. 先检查所有组，筛选出有效组
    valid_items = []

    for group_dir in tqdm(group_folders, desc="检查数据完整性", unit="组", ncols=100):
        ok, reason, info = validate_group(group_dir)

        if not ok:
            skipped_records.append(
                {
                    "source_group": group_dir.name,
                    "reason": reason,
                }
            )
            continue

        valid_items.append((group_dir, info))

    print(f"有效组数量: {len(valid_items)}")

    # 2. 根据 RANDOM_SAMPLE_COUNT 决定处理全部，还是随机抽取一部分
    selected_items = select_items_by_random_count(valid_items)

    # 3. 正式转移数据
    current_index = START_INDEX

    for group_dir, info in tqdm(selected_items, desc="整理数据", unit="组", ncols=100):
        sample_id, used_index = find_next_available_sample_id(current_index)
        current_index = used_index + 1

        try:
            manifest_row = copy_group_to_dataset(
                group_dir=group_dir,
                sample_id=sample_id,
                info=info,
            )
            manifest_records.append(manifest_row)

        except Exception as e:
            skipped_records.append(
                {
                    "source_group": group_dir.name,
                    "reason": f"转移失败: {e}",
                }
            )
            continue

    # 4. 保存记录文件
    manifest_dir = DEST_ROOT / "manifests"

    manifest_csv = manifest_dir / f"{DATE_PREFIX}_manifest.csv"
    skipped_csv = manifest_dir / f"{DATE_PREFIX}_skipped.csv"

    write_manifest_csv(manifest_records, manifest_csv)
    write_skipped_csv(skipped_records, skipped_csv)

    print("\n整理完成")
    print(f"成功转移: {len(manifest_records)} 组")
    print(f"跳过: {len(skipped_records)} 组")
    print(f"manifest: {manifest_csv}")
    print(f"skipped : {skipped_csv}")

    print("\n输出目录结构:")
    for modality in MODALITY_ORDER:
        count = len(list((DEST_ROOT / modality).glob("*.npy")))
        print(f"{modality}: {count} 个文件")

    for extra_key in EXTRA_ORDER:
        extra_cfg = EXTRA_FILE_MAP[extra_key]
        pattern = f"*{extra_cfg['dst_ext']}"
        count = len(list((DEST_ROOT / extra_cfg["dst_dir"]).glob(pattern)))
        print(f"{extra_cfg['dst_dir']}: {count} 个文件")


if __name__ == "__main__":
    main()
