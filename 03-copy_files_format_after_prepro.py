"""
脚本功能说明：

本脚本用于将每天生成的多模态车窗 crop 数据整理成适合 pix2pixHD
或自定义多模态图像翻译模型训练的数据结构。

输入数据结构：
给定一个每日数据路径 DAY_DATA_DIR，该路径下包含多组数据子文件夹。
每个子文件夹中包含如下 npy 文件：
1. GT_Camera_usb.npy              作为 ground truth；
2. Nir_Camera_usb.npy             作为 NIR 输入模态；
3. NIR_Long_Camera_248.npy        作为 NIR_Long 输入模态；
4. RGB_Camera_249.npy             作为 RGB 输入模态；
5. RGB_Long_Camera_247.npy        作为 RGB_Long 输入模态。

输出数据结构：
DEST_ROOT/
├── GT/
├── NIR/
├── NIR_Long/
├── RGB/
├── RGB_Long/
└── manifests/

主要功能：
1. 检查 DEST_ROOT 下是否存在 GT、NIR、NIR_Long、RGB、RGB_Long 子文件夹；
2. 如果不存在，则自动创建；
3. 遍历 DAY_DATA_DIR 下的每个数据组文件夹；
4. 检查每组是否同时包含五个必需的 npy 文件；
5. 可选检查五个 npy 文件的 H/W 尺寸是否一致；
6. 将每组数据复制到对应模态子文件夹中；
7. 复制后的五个模态文件使用统一 sample_id 命名，方便后续配对读取；
8. sample_id 由 DATE_PREFIX + 数字编号组成，例如 20260522_000001.npy；
9. 生成 manifest.csv，记录 sample_id、原始组名、复制后的各模态路径；
10. 缺失文件或尺寸不一致的数据会被跳过，并记录到 skipped.csv。

适用场景：
- 每天有几百组车窗 crop 数据；
- 后续需要训练 pix2pixHD；
- 输入可以是 NIR、NIR_Long、RGB、RGB_Long 的任意组合；
- ground truth 固定为 GT_Camera_usb.npy。
"""

import csv
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# =========================================================
# 1. 路径配置
# =========================================================
# 每天的数据路径：该目录下直接包含 0001_xxx、0002_xxx 这样的组文件夹
DAY_DATA_DIR = Path(
    r"/mnt/bigdata/ndy-5JAI/01_processed_data/processed_2026-06-05/_window_crop copy"
    # r"/mnt/bigdata/ndy-5JAI/01_processed_data/processed_2026-05-27/_window_croptest"
)
# 每日日期前缀，用于生成统一文件名
# 例如 DATE_PREFIX = "20260522"，输出文件名为 20260522_000001.npy
DATE_PREFIX = "20260605"
# 整理后的训练数据保存路径
DEST_ROOT = Path(r"/mnt/bigdata/ndy-5JAI/dataset-5JAI/train")


# 是否只处理数字开头的组文件夹
# 例如 0002_205458_415 会处理，_checker_grid_gt 不会处理
ONLY_DIGIT_PREFIX_GROUP = True

# 是否检查五个模态的 H/W 是否一致
CHECK_SHAPE = True

# 是否覆盖目标路径下已经存在的同名文件
OVERWRITE = False

# 复制方式：
# "copy"     ：普通复制，最稳妥；
# "hardlink" ：硬链接，速度快、省空间，但源和目标必须在同一磁盘分区；
# "symlink"  ：软链接，省空间，但原始数据移动后链接会失效。
COPY_MODE = "copy"

# 编号起始值
START_INDEX = 1

# 是否自动跳过已经存在的编号
# True：如果 20260522_000001.npy 已存在，会自动尝试 20260522_000002.npy
AUTO_SKIP_EXISTING_SAMPLE_ID = True


# =========================================================
# 2. 文件名配置
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


# =========================================================
# 3. 工具函数
# =========================================================
def is_digit_prefix_folder(path: Path):
    return path.is_dir() and len(path.name) > 0 and path.name[0].isdigit()


def ensure_output_dirs():
    """
    检查 GT / NIR / NIR_Long / RGB / RGB_Long 文件夹是否存在。
    如果不存在，则自动创建。
    """
    DEST_ROOT.mkdir(parents=True, exist_ok=True)

    for modality in MODALITY_ORDER:
        modality_dir = DEST_ROOT / modality
        modality_dir.mkdir(parents=True, exist_ok=True)

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

    for modality, filename in MODALITY_FILE_MAP.items():
        src_path = group_dir / filename

        if not src_path.exists():
            return False, f"缺失文件: {filename}", {}

        src_paths[modality] = src_path

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
        "height": height,
        "width": width,
        "shape_info": shape_info,
    }

    return True, "OK", info


def sample_id_exists(sample_id: str):
    """
    判断某个 sample_id 是否已经在任意模态目录中存在。
    """
    filename = f"{sample_id}.npy"

    for modality in MODALITY_ORDER:
        if (DEST_ROOT / modality / filename).exists():
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


def copy_one_file(src: Path, dst: Path):
    """
    按 COPY_MODE 复制文件。
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        if OVERWRITE:
            dst.unlink()
        else:
            raise FileExistsError(f"目标文件已存在: {dst}")

    if COPY_MODE == "copy":
        shutil.copy2(src, dst)

    elif COPY_MODE == "hardlink":
        dst.hardlink_to(src)

    elif COPY_MODE == "symlink":
        dst.symlink_to(src.resolve())

    else:
        raise ValueError(f"不支持的 COPY_MODE: {COPY_MODE}")


def write_manifest_csv(records, csv_path: Path):
    """
    保存成功复制的数据记录。
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


# =========================================================
# 4. 主处理逻辑
# =========================================================
def copy_group_to_dataset(group_dir: Path, sample_id: str, info: dict):
    """
    将一组数据复制到 GT / NIR / NIR_Long / RGB / RGB_Long 平行目录中。
    """
    src_paths = info["src_paths"]

    manifest_row = {
        "sample_id": sample_id,
        "source_group": group_dir.name,
        "height": info["height"],
        "width": info["width"],
    }

    for modality in MODALITY_ORDER:
        src = src_paths[modality]
        dst = DEST_ROOT / modality / f"{sample_id}.npy"

        copy_one_file(src, dst)

        # manifest 中保存相对路径，方便移动整个 DEST_ROOT
        rel_path = dst.relative_to(DEST_ROOT).as_posix()
        manifest_row[modality] = rel_path

    return manifest_row


def main():
    ensure_output_dirs()

    group_folders = get_group_folders(DAY_DATA_DIR)

    print(f"每日数据路径: {DAY_DATA_DIR}")
    print(f"输出数据路径: {DEST_ROOT}")
    print(f"日期前缀: {DATE_PREFIX}")
    print(f"发现组文件夹数量: {len(group_folders)}")

    manifest_records = []
    skipped_records = []

    current_index = START_INDEX

    for group_dir in tqdm(group_folders, desc="整理数据", unit="组", ncols=100):
        ok, reason, info = validate_group(group_dir)

        if not ok:
            skipped_records.append(
                {
                    "source_group": group_dir.name,
                    "reason": reason,
                }
            )
            continue

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
                    "reason": f"复制失败: {e}",
                }
            )
            continue

    manifest_dir = DEST_ROOT / "manifests"

    manifest_csv = manifest_dir / f"{DATE_PREFIX}_manifest.csv"
    skipped_csv = manifest_dir / f"{DATE_PREFIX}_skipped.csv"

    write_manifest_csv(manifest_records, manifest_csv)
    write_skipped_csv(skipped_records, skipped_csv)

    print("\n整理完成")
    print(f"成功复制: {len(manifest_records)} 组")
    print(f"跳过: {len(skipped_records)} 组")
    print(f"manifest: {manifest_csv}")
    print(f"skipped : {skipped_csv}")

    print("\n输出目录结构:")
    for modality in MODALITY_ORDER:
        count = len(list((DEST_ROOT / modality).glob("*.npy")))
        print(f"{modality}: {count} 个文件")


if __name__ == "__main__":
    main()
