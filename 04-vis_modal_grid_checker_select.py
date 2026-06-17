#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python 04-vis_modal_grid_checker_select.py \
  --root /mnt/bigdata/ndy-5JAI/dataset-5JAI/test1 \
  --num 10 \
  --mode random 
  
#   --seed 123  如果增加随机性，可修改seed参数值

脚本功能说明：

本脚本用于对已经整理好的多模态 npy 数据集进行可视化检查。

支持功能：
1. 指定数据集根目录 DATASET_ROOT；
2. 指定可视化数量；
3. 可选择按顺序筛选或随机筛选；
4. 生成每组样本的 2 行 × 5 列可视化图；
5. 第一行展示 GT、NIR、NIR_Long、RGB、RGB_Long 原图；
6. 第二行展示 GT reference 以及 GT 和其他模态的棋盘格对齐结果。

数据结构要求：
DATASET_ROOT/
├── GT/
├── NIR/
├── NIR_Long/
├── RGB/
├── RGB_Long/
└── manifests/
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# =========================================================
# 1. 默认路径配置
# =========================================================
DEFAULT_DATASET_ROOT = Path(r"/mnt/bigdata/ndy-5JAI/dataset-5JAI/test1")
DEFAULT_VIS_DIR_NAME = "_vis_modal_grid_checker"

# 是否只处理五个模态都存在的同名文件
ONLY_COMMON_FILES = True


# =========================================================
# 2. 模态配置
# =========================================================
MODALITY_DIRS = {
    "GT": "GT",
    "NIR": "NIR",
    "NIR_Long": "NIR_Long",
    "RGB": "RGB",
    "RGB_Long": "RGB_Long",
}

VIS_ORDER = [
    "GT",
    "NIR",
    "NIR_Long",
    "RGB",
    "RGB_Long",
]

CHECKER_ORDER = [
    "GT",
    "NIR",
    "NIR_Long",
    "RGB",
    "RGB_Long",
]

# 如果 NIR / NIR_Long 是 H×W×2，这里控制怎么显示
# "first" ：显示第 0 个通道
# "second"：显示第 1 个通道
# "mean"  ：两个通道取平均
NIR_2CH_VIS_MODE = "first"


# =========================================================
# 3. 棋盘格配置
# =========================================================
CHECKER_TILE_SIZE = 64


# =========================================================
# 4. 可视化布局配置
# =========================================================
CELL_W = 420
CELL_H = 300
TITLE_H = 45
GAP = 10

BG_COLOR = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)


# =========================================================
# 5. 全局路径变量，由命令行参数赋值
# =========================================================
DATASET_ROOT = DEFAULT_DATASET_ROOT
VIS_DIR = DATASET_ROOT / DEFAULT_VIS_DIR_NAME


# =========================================================
# 6. 参数解析
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="对多模态 npy 数据集进行可视化，可选择随机或按顺序筛选样本。"
    )

    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="数据集根目录，里面应包含 GT、NIR、NIR_Long、RGB、RGB_Long 文件夹。",
    )

    parser.add_argument(
        "--num",
        "-n",
        type=int,
        default=None,
        help="可视化样本数量。None 或不填表示全部可视化。",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="seq",
        choices=["seq", "random"],
        help="筛选方式：seq 表示按文件名顺序筛选；random 表示随机筛选。",
    )

    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="按顺序筛选时的起始下标，默认 0。例如 --start 100 表示从第 101 个样本开始取。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机筛选的随机种子，保证每次随机结果可复现。",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="可视化结果保存目录。不填则默认保存到 DATASET_ROOT/_vis_modal_grid_checker。",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="只筛选指定前缀的样本。例如 --prefix 20260522 只处理 20260522_ 开头的 npy。",
    )

    parser.add_argument(
        "--tile",
        type=int,
        default=CHECKER_TILE_SIZE,
        help="棋盘格单块大小，默认 64。",
    )

    return parser.parse_args()


# =========================================================
# 7. 基础工具函数
# =========================================================
def check_modality_dirs():
    for modality, folder_name in MODALITY_DIRS.items():
        folder = DATASET_ROOT / folder_name
        if not folder.exists():
            raise FileNotFoundError(f"缺失模态文件夹: {modality} -> {folder}")
        if not folder.is_dir():
            raise NotADirectoryError(f"路径不是文件夹: {folder}")


def get_npy_names(folder: Path):
    return set(p.name for p in folder.glob("*.npy"))


def get_all_common_sample_names(prefix=None):
    """
    获取五个模态中同名的 npy 文件。
    """
    name_sets = []

    for modality, folder_name in MODALITY_DIRS.items():
        folder = DATASET_ROOT / folder_name
        names = get_npy_names(folder)

        if prefix is not None:
            names = {name for name in names if name.startswith(prefix)}

        name_sets.append(names)
        print(f"{modality}: {len(names)} 个 npy 文件")

    if ONLY_COMMON_FILES:
        common_names = set.intersection(*name_sets)
    else:
        common_names = name_sets[0]

    sample_names = sorted(list(common_names))
    return sample_names


def select_sample_names(sample_names, num=None, mode="seq", start=0, seed=42):
    """
    从全部样本中筛选指定数量：
    1. mode='seq'：按文件名顺序筛选；
    2. mode='random'：随机筛选。
    """
    total = len(sample_names)

    if total == 0:
        return []

    if num is None:
        return sample_names

    if num <= 0:
        raise ValueError(f"--num 必须大于 0，当前为: {num}")

    num = min(num, total)

    if mode == "seq":
        if start < 0:
            raise ValueError(f"--start 不能小于 0，当前为: {start}")

        if start >= total:
            raise ValueError(f"--start 超出样本总数，start={start}, total={total}")

        end = min(start + num, total)
        selected = sample_names[start:end]
        return selected

    if mode == "random":
        rng = random.Random(seed)
        selected = rng.sample(sample_names, num)
        selected = sorted(selected)
        return selected

    raise ValueError(f"不支持的筛选模式: {mode}")


def npy_to_uint8(arr):
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.round(arr * 255.0).astype(np.uint8)
    return arr


def convert_npy_to_bgr(arr, modality):
    arr = np.asarray(arr)

    if arr.ndim == 2:
        gray = npy_to_uint8(arr)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return bgr

    if arr.ndim == 3:
        c = arr.shape[2]

        if c == 1:
            gray = npy_to_uint8(arr[:, :, 0])
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return bgr

        if c == 2:
            if NIR_2CH_VIS_MODE == "first":
                gray_float = arr[:, :, 0]
            elif NIR_2CH_VIS_MODE == "second":
                gray_float = arr[:, :, 1]
            elif NIR_2CH_VIS_MODE == "mean":
                gray_float = arr.mean(axis=2)
            else:
                raise ValueError(f"不支持的 NIR_2CH_VIS_MODE: {NIR_2CH_VIS_MODE}")

            gray = npy_to_uint8(gray_float)
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return bgr

        if c >= 3:
            rgb = arr[:, :, :3]
            rgb_u8 = npy_to_uint8(rgb)
            bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
            return bgr

    raise ValueError(f"{modality} 的 npy shape 不支持: {arr.shape}")


def resize_keep_ratio(img, target_w, target_h):
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), BG_COLOR, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def make_cell(img_bgr, title):
    img_resized = resize_keep_ratio(img_bgr, CELL_W, CELL_H)

    cell = np.full((TITLE_H + CELL_H, CELL_W, 3), BG_COLOR, dtype=np.uint8)

    text_size, _ = cv2.getTextSize(title, FONT, FONT_SCALE, FONT_THICKNESS)
    text_w, text_h = text_size

    text_x = max(5, (CELL_W - text_w) // 2)
    text_y = (TITLE_H + text_h) // 2

    cv2.putText(
        cell,
        title,
        (text_x, text_y),
        FONT,
        FONT_SCALE,
        TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

    cell[TITLE_H : TITLE_H + CELL_H, :, :] = img_resized
    return cell


def make_checkerboard(gt_img_bgr, other_img_bgr, tile_size=64):
    gt = gt_img_bgr.copy()
    other = other_img_bgr.copy()

    gt_h, gt_w = gt.shape[:2]

    if other.shape[:2] != (gt_h, gt_w):
        other = cv2.resize(other, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)

    yy, xx = np.indices((gt_h, gt_w))
    board = ((yy // tile_size) + (xx // tile_size)) % 2

    checker = gt.copy()
    checker[board == 1] = other[board == 1]

    return checker


# =========================================================
# 8. 单组可视化
# =========================================================
def load_sample_images(sample_name):
    images = {}
    shapes = {}

    for modality in VIS_ORDER:
        folder_name = MODALITY_DIRS[modality]
        npy_path = DATASET_ROOT / folder_name / sample_name

        if not npy_path.exists():
            raise FileNotFoundError(f"缺失文件: {npy_path}")

        arr = np.load(npy_path, allow_pickle=False)
        img_bgr = convert_npy_to_bgr(arr, modality)

        images[modality] = img_bgr
        shapes[modality] = arr.shape

    return images, shapes


def visualize_one_sample(sample_name, tile_size=64):
    images, shapes = load_sample_images(sample_name)
    gt_img = images["GT"]

    rows_cells = []

    row1_cells = []
    for modality in VIS_ORDER:
        title = f"{modality} | {shapes[modality]}"
        cell = make_cell(images[modality], title)
        row1_cells.append(cell)
    rows_cells.append(row1_cells)

    row2_cells = []
    for modality in CHECKER_ORDER:
        if modality == "GT":
            title = "GT reference"
            cell = make_cell(gt_img, title)
        else:
            checker = make_checkerboard(
                gt_img_bgr=gt_img,
                other_img_bgr=images[modality],
                tile_size=tile_size,
            )
            title = f"GT / {modality}"
            cell = make_cell(checker, title)

        row2_cells.append(cell)

    rows_cells.append(row2_cells)

    rows = 2
    cols = len(VIS_ORDER)

    canvas_h = rows * (TITLE_H + CELL_H) + (rows - 1) * GAP
    canvas_w = cols * CELL_W + (cols - 1) * GAP

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    for row_idx, row_cells in enumerate(rows_cells):
        for col_idx, cell in enumerate(row_cells):
            y1 = row_idx * (TITLE_H + CELL_H + GAP)
            x1 = col_idx * (CELL_W + GAP)
            canvas[y1 : y1 + cell.shape[0], x1 : x1 + cell.shape[1]] = cell

    save_name = Path(sample_name).stem + "_grid_checker.png"
    save_path = VIS_DIR / save_name

    ok = cv2.imwrite(str(save_path), canvas)
    if not ok:
        raise ValueError(f"保存失败: {save_path}")

    return save_path


# =========================================================
# 9. 主函数
# =========================================================
def main():
    global DATASET_ROOT, VIS_DIR

    args = parse_args()

    DATASET_ROOT = Path(args.root).resolve()

    if args.out is None:
        VIS_DIR = DATASET_ROOT / DEFAULT_VIS_DIR_NAME
    else:
        VIS_DIR = Path(args.out).resolve()

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"DATASET_ROOT 不存在: {DATASET_ROOT}")

    if not DATASET_ROOT.is_dir():
        raise NotADirectoryError(f"DATASET_ROOT 不是文件夹: {DATASET_ROOT}")

    check_modality_dirs()

    print("=" * 80)
    print(f"数据集路径: {DATASET_ROOT}")
    print(f"可视化保存路径: {VIS_DIR}")
    print(f"筛选模式: {args.mode}")
    print(f"指定数量: {args.num}")
    print(f"顺序起始下标: {args.start}")
    print(f"随机种子: {args.seed}")
    print(f"样本前缀过滤: {args.prefix}")
    print(f"棋盘格大小: {args.tile}")
    print("=" * 80)

    all_sample_names = get_all_common_sample_names(prefix=args.prefix)

    if len(all_sample_names) == 0:
        print("没有找到五个模态都存在的同名 npy 文件。")
        return

    sample_names = select_sample_names(
        sample_names=all_sample_names,
        num=args.num,
        mode=args.mode,
        start=args.start,
        seed=args.seed,
    )

    print(f"全部可用样本数量: {len(all_sample_names)}")
    print(f"本次可视化样本数量: {len(sample_names)}")

    if len(sample_names) > 0:
        print(f"第一个样本: {sample_names[0]}")
        print(f"最后一个样本: {sample_names[-1]}")

    success_count = 0
    fail_count = 0

    for sample_name in tqdm(sample_names, desc="可视化进度", unit="组", ncols=100):
        try:
            visualize_one_sample(sample_name, tile_size=args.tile)
            success_count += 1
        except Exception as e:
            fail_count += 1
            print(f"处理失败: {sample_name}, 原因: {e}")

    print("\n全部可视化完成")
    print(f"成功: {success_count} 组")
    print(f"失败: {fail_count} 组")
    print(f"输出目录: {VIS_DIR}")


if __name__ == "__main__":
    main()
