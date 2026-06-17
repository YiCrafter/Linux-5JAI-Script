#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脚本功能说明：

本脚本用于批量可视化 _window_crop 目录下每个子文件夹中的多模态 npy 数据，
并把 dahua_window.png 一起加入拼接图中。

适用文件结构：

_window_crop/
├── 0001_205445_965/
│   ├── _dahua_paste_back_meta.json
│   ├── dahua_window.png
│   ├── GT_Camera_usb.npy
│   ├── Nir_Camera_usb.npy
│   ├── NIR_Long_Camera_248.npy
│   ├── RGB_Camera_249.npy
│   └── RGB_Long_Camera_247.npy
├── 0002_205458_415/
└── ...

数据说明：
1. 所有 npy 都是 float32，数值范围 0~1；
2. Nir_Camera_usb.npy 和 NIR_Long_Camera_248.npy 是 H×W×2；
3. GT_Camera_usb.npy、RGB_Camera_249.npy、RGB_Long_Camera_247.npy 是 H×W×3；
4. dahua_window.png 是普通 png 图像。

输出：
1. 普通拼接图：
   ROOT_DIR/_npy_dahua_window_grid/子文件夹名_npy_dahua_window_grid.png

2. 棋盘格检查图，可选：
   ROOT_DIR/_npy_dahua_window_checker/子文件夹名_checker_grid.png
"""

import cv2
import numpy as np
from pathlib import Path
from math import ceil
import random
from tqdm import tqdm

# =========================================================
# 1. 路径配置
# =========================================================
# 这里改成你的 _window_crop 路径
ROOT_DIR = Path(
    r"/mnt/bigdata/ndy-5JAI/03_processed_daily/processed_2026-06-09/_window_crop"
)

# 普通拼接图保存目录
SAVE_DIR = ROOT_DIR / "_npy_dahua_window_grid"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 是否生成棋盘格检查图
GENERATE_CHECKERBOARD = True

# 棋盘格保存目录
CHECKER_SAVE_DIR = ROOT_DIR / "_npy_dahua_window_checker"
if GENERATE_CHECKERBOARD:
    CHECKER_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. 筛选配置
# =========================================================
# None 表示全部处理
MAX_GROUPS = 10
# MAX_GROUPS = None

# 选择模式：
# "seq"    ：按文件夹名称顺序取前 MAX_GROUPS 个
# "random" ：随机取 MAX_GROUPS 个
# SELECT_MODE = "seq"
SELECT_MODE = "random"

# 随机种子，SELECT_MODE="random" 时生效
RANDOM_SEED = 42

# 是否只处理数字开头的子文件夹
# 例如 0001_205445_965 会处理，_xxx 输出目录不会处理
ONLY_DIGIT_PREFIX_GROUP = True


# =========================================================
# 3. 图像显示配置
# =========================================================
# 如果 NIR / NIR_Long 是 H×W×2，这里控制显示方式：
# "first"  ：显示第 0 通道
# "second" ：显示第 1 通道
# "mean"   ：两个通道取平均
NIR_2CH_VIS_MODE = "first"

# 棋盘格单块大小
CHECKER_TILE_SIZE = 64


# =========================================================
# 4. 拼接布局配置
# =========================================================
COLS = 3
ROWS = 2

CELL_W = 500
CELL_H = 360
TITLE_H = 45
GAP = 10

BG_COLOR = (255, 255, 255)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)


# =========================================================
# 5. 文件顺序配置
# =========================================================
# 普通拼接图显示顺序
VIS_ITEMS = [
    ("DAHUA", "dahua_window.png", "png"),
    ("GT", "GT_Camera_usb.npy", "npy"),
    ("NIR", "Nir_Camera_usb.npy", "npy"),
    ("NIR-L", "NIR_Long_Camera_248.npy", "npy"),
    ("RGB", "RGB_Camera_249.npy", "npy"),
    ("RGB-L", "RGB_Long_Camera_247.npy", "npy"),
]

# 棋盘格检查顺序：全部和 GT 做棋盘格
CHECKER_ITEMS = [
    ("DAHUA", "dahua_window.png", "png"),
    ("NIR", "Nir_Camera_usb.npy", "npy"),
    ("NIR-L", "NIR_Long_Camera_248.npy", "npy"),
    ("RGB", "RGB_Camera_249.npy", "npy"),
    ("RGB-L", "RGB_Long_Camera_247.npy", "npy"),
]


# =========================================================
# 6. 基础工具函数
# =========================================================
def is_digit_prefix_folder(path: Path):
    return path.is_dir() and len(path.name) > 0 and path.name[0].isdigit()


def get_group_folders(root_dir: Path):
    """
    获取 ROOT_DIR 下的所有组文件夹。
    """
    if not root_dir.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"ROOT_DIR 不是文件夹: {root_dir}")

    folders = sorted([p for p in root_dir.iterdir() if p.is_dir()])

    skip_names = {SAVE_DIR.name}
    if GENERATE_CHECKERBOARD:
        skip_names.add(CHECKER_SAVE_DIR.name)

    folders = [p for p in folders if p.name not in skip_names]

    if ONLY_DIGIT_PREFIX_GROUP:
        folders = [p for p in folders if is_digit_prefix_folder(p)]

    if MAX_GROUPS is not None:
        if SELECT_MODE == "seq":
            folders = folders[:MAX_GROUPS]

        elif SELECT_MODE == "random":
            rng = random.Random(RANDOM_SEED)
            sample_num = min(MAX_GROUPS, len(folders))
            folders = rng.sample(folders, sample_num)
            folders = sorted(folders, key=lambda x: x.name)

        else:
            raise ValueError(f"不支持的 SELECT_MODE: {SELECT_MODE}")

    return folders


def npy_to_uint8(arr):
    """
    将 float32 0~1 的 npy 转为 uint8 0~255。
    """
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.round(arr * 255.0).astype(np.uint8)
    return arr


def npy_to_bgr(arr, name=""):
    """
    将 npy 转成 OpenCV BGR 三通道图像。

    支持：
    1. H×W
    2. H×W×1
    3. H×W×2
    4. H×W×3
    """
    arr = np.asarray(arr)

    # H x W
    if arr.ndim == 2:
        gray = npy_to_uint8(arr)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # H x W x C
    if arr.ndim == 3:
        c = arr.shape[2]

        if c == 1:
            gray = npy_to_uint8(arr[:, :, 0])
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        if c >= 3:
            # 默认认为 npy 中前三通道是 RGB
            rgb = arr[:, :, :3]
            rgb_u8 = npy_to_uint8(rgb)

            # OpenCV 保存需要 BGR
            return cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    raise ValueError(f"{name} 的 npy shape 不支持: {arr.shape}")


def read_png_bgr(path: Path):
    """
    读取 png，统一转成 BGR 三通道。
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"图片读取失败: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    elif img.ndim == 3 and img.shape[2] == 3:
        pass

    else:
        raise ValueError(f"不支持的 png shape: {path}, shape={img.shape}")

    return img


def read_item_as_bgr(folder: Path, filename: str, file_type: str):
    """
    读取一个文件，并转成 BGR 图像。
    """
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"缺失文件: {path}")

    if file_type == "png":
        img = read_png_bgr(path)
        shape = img.shape
        return img, shape

    if file_type == "npy":
        arr = np.load(path, allow_pickle=False)
        img = npy_to_bgr(arr, filename)
        shape = arr.shape
        return img, shape

    raise ValueError(f"不支持的 file_type: {file_type}")


def format_shape(shape):
    """
    将 shape 转成简短字符串：
    (513, 1087, 3) -> (513,1087,3)
    """
    return "(" + ",".join(str(x) for x in shape) + ")"


def resize_keep_ratio(img, target_w, target_h):
    """
    保持比例缩放，并填充到固定大小。
    """
    h, w = img.shape[:2]

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.full((target_h, target_w, 3), BG_COLOR, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    return canvas


def make_cell_from_image(img, title: str):
    """
    生成一个带标题的 cell。
    """
    img_resized = resize_keep_ratio(img, CELL_W, CELL_H)

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


def make_checkerboard(gt_img, other_img, tile_size=64):
    """
    将 other_img 和 GT 图做棋盘格交替显示。
    """
    gt = gt_img.copy()
    other = other_img.copy()

    gt_h, gt_w = gt.shape[:2]

    if other.shape[:2] != (gt_h, gt_w):
        other = cv2.resize(other, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)

    yy, xx = np.indices((gt_h, gt_w))
    board = ((yy // tile_size) + (xx // tile_size)) % 2

    checker = gt.copy()
    checker[board == 1] = other[board == 1]

    return checker


# =========================================================
# 7. 单个子文件夹处理
# =========================================================
def visualize_one_folder(folder: Path):
    """
    对一个子文件夹生成普通拼接图。
    """
    cells = []

    for display_name, filename, file_type in VIS_ITEMS:
        try:
            img, shape = read_item_as_bgr(folder, filename, file_type)
        except Exception as e:
            print(f"跳过 {folder.name}: 读取 {filename} 失败，原因: {e}")
            return False

        title = f"{display_name} | {format_shape(shape)}"
        cell = make_cell_from_image(img, title)
        cells.append(cell)

    cols = COLS
    rows = max(ROWS, ceil(len(cells) / cols))

    canvas_h = rows * (TITLE_H + CELL_H) + (rows - 1) * GAP
    canvas_w = cols * CELL_W + (cols - 1) * GAP

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, cell in enumerate(cells):
        row = idx // cols
        col = idx % cols

        y1 = row * (TITLE_H + CELL_H + GAP)
        x1 = col * (CELL_W + GAP)

        canvas[y1 : y1 + cell.shape[0], x1 : x1 + cell.shape[1]] = cell

    save_path = SAVE_DIR / f"{folder.name}_npy_dahua_window_grid.png"

    ok = cv2.imwrite(str(save_path), canvas)

    if not ok:
        raise ValueError(f"保存失败: {save_path}")

    return True


def visualize_checkerboard_one_folder(folder: Path):
    """
    对一个子文件夹生成 GT 与其他模态的棋盘格图。
    """
    gt_path = folder / "GT_Camera_usb.npy"

    if not gt_path.exists():
        print(f"跳过棋盘格 {folder.name}: 缺失 GT_Camera_usb.npy")
        return False

    try:
        gt_arr = np.load(gt_path, allow_pickle=False)
        gt_img = npy_to_bgr(gt_arr, "GT_Camera_usb.npy")
    except Exception as e:
        print(f"跳过棋盘格 {folder.name}: 读取 GT 失败，原因: {e}")
        return False

    cells = []

    gt_title = f"GT reference | {format_shape(gt_arr.shape)}"
    cells.append(make_cell_from_image(gt_img, gt_title))

    for display_name, filename, file_type in CHECKER_ITEMS:
        try:
            other_img, shape = read_item_as_bgr(folder, filename, file_type)
        except Exception as e:
            print(f"跳过 {folder.name} 的棋盘格目标 {filename}: {e}")
            continue

        checker = make_checkerboard(
            gt_img=gt_img,
            other_img=other_img,
            tile_size=CHECKER_TILE_SIZE,
        )

        title = f"GT / {display_name} | {format_shape(shape)}"
        cell = make_cell_from_image(checker, title)
        cells.append(cell)

    if len(cells) <= 1:
        print(f"跳过棋盘格 {folder.name}: 没有可检查目标")
        return False

    cols = COLS
    rows = max(ROWS, ceil(len(cells) / cols))

    canvas_h = rows * (TITLE_H + CELL_H) + (rows - 1) * GAP
    canvas_w = cols * CELL_W + (cols - 1) * GAP

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, cell in enumerate(cells):
        row = idx // cols
        col = idx % cols

        y1 = row * (TITLE_H + CELL_H + GAP)
        x1 = col * (CELL_W + GAP)

        canvas[y1 : y1 + cell.shape[0], x1 : x1 + cell.shape[1]] = cell

    save_path = CHECKER_SAVE_DIR / f"{folder.name}_checker_grid.png"

    ok = cv2.imwrite(str(save_path), canvas)

    if not ok:
        raise ValueError(f"保存失败: {save_path}")

    return True


# =========================================================
# 8. 主函数
# =========================================================
def main():
    group_folders = get_group_folders(ROOT_DIR)

    print("=" * 80)
    print(f"输入目录: {ROOT_DIR}")
    print(f"普通拼接图输出目录: {SAVE_DIR}")
    print(f"是否生成棋盘格: {GENERATE_CHECKERBOARD}")
    if GENERATE_CHECKERBOARD:
        print(f"棋盘格输出目录: {CHECKER_SAVE_DIR}")
    print(f"筛选模式: {SELECT_MODE}")
    print(f"MAX_GROUPS: {MAX_GROUPS}")
    print(f"NIR 双通道显示方式: {NIR_2CH_VIS_MODE}")
    print(f"本次处理子文件夹数量: {len(group_folders)}")
    print("=" * 80)

    success_grid = 0
    fail_grid = 0
    success_checker = 0
    fail_checker = 0

    for folder in tqdm(group_folders, desc="拼接进度", unit="组", ncols=100):
        try:
            ok_grid = visualize_one_folder(folder)
            if ok_grid:
                success_grid += 1
            else:
                fail_grid += 1
        except Exception as e:
            fail_grid += 1
            print(f"普通拼接失败: {folder.name}, 原因: {e}")

        if GENERATE_CHECKERBOARD:
            try:
                ok_checker = visualize_checkerboard_one_folder(folder)
                if ok_checker:
                    success_checker += 1
                else:
                    fail_checker += 1
            except Exception as e:
                fail_checker += 1
                print(f"棋盘格失败: {folder.name}, 原因: {e}")

    print("\n全部处理完成")
    print(f"普通拼接成功: {success_grid} 组")
    print(f"普通拼接失败: {fail_grid} 组")

    if GENERATE_CHECKERBOARD:
        print(f"棋盘格成功: {success_checker} 组")
        print(f"棋盘格失败: {fail_checker} 组")

    print(f"普通拼接图目录: {SAVE_DIR}")
    if GENERATE_CHECKERBOARD:
        print(f"棋盘格目录: {CHECKER_SAVE_DIR}")


if __name__ == "__main__":
    main()
