"""
脚本功能说明：

本脚本用于批量生成多模态 PNG 图像的拼接预览图，并可选择额外生成
GT 与其他模态图像的棋盘格对齐检查图。程序会遍历指定根目录下的各个
子文件夹，读取其中的 PNG 图片，按照预设顺序进行排列、缩放、标注名称
并拼接保存，方便快速查看每组数据的图像内容和对齐效果。

主要功能：
1. 遍历 ROOT_DIR 下的各个子文件夹，可通过 MAX_GROUPS 控制最多处理的组数；
2. 按固定顺序读取每个子文件夹中的 PNG 图像，包括：
   大华车窗图/大华图、GT、NIR、NIR-Long、RGB、RGB-Long；
3. 自动适配新旧两种大华图结构：
   - 如果同时存在 dahua_window.png 和 dahua.png，则拼接时使用 dahua_window.png，
     并忽略原始大华全图 dahua.png；
   - 如果子文件夹中只有 dahua.png，则使用 dahua.png，兼容旧版本文件结构；
4. 将每张图片按比例缩放到统一尺寸，并在图片上方标注对应文件名；
5. 按设定的行列布局生成普通 PNG 拼接预览图；
6. 可通过 GENERATE_CHECKERBOARD 开关控制是否生成棋盘格检查图；
7. 棋盘格检查图以 GT_Camera_usb.png 为参考图，将其他模态图像与 GT 交替显示，
   用于观察图像边缘、车窗区域、道路白线等结构是否对齐；
8. 棋盘格检查中：
   - 如果存在 dahua_window.png，则包含 GT / dahua_window.png；
   - 如果只有 dahua.png，则不对 dahua.png 做棋盘格检查，因为它是原始大华全图，
     不一定与 GT 车窗 crop 尺寸和视角一致。

输出结果：
1. 普通拼接预览图：
   保存到 SAVE_DIR 目录下，文件名格式为：
   子文件夹名_png_grid.png

2. 棋盘格对齐检查图：
   当 GENERATE_CHECKERBOARD = True 时生成，
   保存到 CHECKER_SAVE_DIR 目录下，文件名格式为：
   子文件夹名_checker_grid.png

注意：
- 本脚本只处理 PNG 图片，不读取或处理 npy 文件；
- 普通拼接图主要用于查看每组数据中各模态图片是否生成完整；
- 棋盘格图主要用于检查 GT 与其他模态图像之间的空间对齐效果；
- 若子文件夹中存在额外 PNG 图片，可根据脚本中的 extra_pngs 逻辑决定是否追加显示；
- 若不需要生成棋盘格检查图，可将 GENERATE_CHECKERBOARD 设置为 False。
"""

import cv2
import numpy as np
from pathlib import Path
from math import ceil

# =========================
# 1. 路径配置
# =========================
# 这里改成你的 output 总目录，也就是图1中包含 0001、0002、0003... 的文件夹
ROOT_DIR = Path(r"/mnt/bigdata/gongyong/5JAI/processed_2026-05-22")
# ROOT_DIR = Path(r"/mnt/bigdata/gongyong/5JAI/processed_2026-05-22/_window_crop")
# ROOT_DIR = Path(r"D:\company-file-5-22\output-06-03")

# 拼接图保存目录，和各个子文件夹同级
SAVE_DIR = ROOT_DIR / "_png_grid_2x3"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 是否额外生成 GT 与其他图像的棋盘格检查图
GENERATE_CHECKERBOARD = True
# GENERATE_CHECKERBOARD = False

# 棋盘格结果保存目录，和各个子文件夹同级
CHECKER_SAVE_DIR = ROOT_DIR / "_checker_grid_gt"

if GENERATE_CHECKERBOARD:
    CHECKER_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# 棋盘格单块大小，越小交替越密集
CHECKER_TILE_SIZE = 64

# 指定最多处理多少组
# None 表示全部处理
# MAX_GROUPS = None
MAX_GROUPS = 10

# =========================
# 2. 拼接布局配置
# =========================
ROWS = 2
COLS = 3

# 每张图缩放到统一大小，方便拼接
CELL_W = 500
CELL_H = 400

# 每张图上方标题栏高度
TITLE_H = 45

# 图片之间的间隔
GAP = 10

# 背景颜色，白色
BG_COLOR = (255, 255, 255)

# 字体配置
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
TEXT_COLOR = (0, 0, 0)


# =========================
# 3. PNG 顺序配置
# =========================
# 按照你图1中的顺序显示
PREFERRED_ORDER = [
    # "dahua_window.png",
    "GT_Camera_usb.png",
    "Nir_Camera_usb.png",
    "NIR_Long_Camera_248.png",
    "RGB_Camera_249.png",
    "RGB_Long_Camera_247.png",
]

# 棋盘格检查顺序：
# 注意：
# 1. 如果存在 dahua_window.png，则会加入 GT / dahua_window.png 棋盘格；
# 2. 如果只有 dahua.png，不加入棋盘格检查，因为 dahua.png 是大华原图，不是车窗 crop；
# 3. 其他图像均与 GT_Camera_usb.png 做棋盘格。
CHECKER_ORDER = [
    "Nir_Camera_usb.png",
    "NIR_Long_Camera_248.png",
    "RGB_Camera_249.png",
    "RGB_Long_Camera_247.png",
]


def read_image_rgb(path: Path):
    """
    读取图片，统一转为 BGR 格式给 OpenCV 后续处理。
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"图片读取失败: {path}")

    # 如果是灰度图，转成三通道
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # 如果是四通道，去掉 alpha
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def resize_keep_ratio(img, target_w, target_h):
    """
    保持比例缩放，并填充到固定大小。
    """
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


def make_cell_from_image(img, title: str):
    """
    生成单个 cell：上方显示标题，下方显示图片。
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


def make_cell(img_path: Path):
    """
    生成单个 cell：上方显示文件名，下方显示图片。
    """
    img = read_image_rgb(img_path)
    return make_cell_from_image(img, img_path.name)


def get_ordered_pngs(folder: Path):
    """
    按指定顺序获取 PNG。

    大华图适配逻辑：
    1. 如果当前子文件夹同时存在 dahua_window.png 和 dahua.png：
       使用 dahua_window.png，忽略 dahua.png；
    2. 如果当前子文件夹只有 dahua.png：
       使用 dahua.png，兼容旧版本文件结构；
    3. 其他图像按 PREFERRED_ORDER 顺序读取；
    4. 额外 PNG 会按文件名排序追加，但会排除 dahua.png / dahua_window.png 的重复干扰。
    """
    all_pngs = list(folder.glob("*.png"))
    name_to_path = {p.name: p for p in all_pngs}

    ordered = []
    used_names = set()

    # -------------------------
    # 1. 大华图优先级处理
    # -------------------------
    if "dahua_window.png" in name_to_path:
        # 新结构：使用大华车窗 crop，不使用原始大华全图 dahua.png
        ordered.append(name_to_path["dahua_window.png"])

        # 两个都标记为已处理，避免 dahua.png 被作为 extra_pngs 追加
        used_names.add("dahua_window.png")
        used_names.add("dahua.png")

    elif "dahua.png" in name_to_path:
        # 旧结构：只有 dahua.png，就使用它
        ordered.append(name_to_path["dahua.png"])
        used_names.add("dahua.png")

    # -------------------------
    # 2. 其他指定图像按顺序加入
    # -------------------------
    for name in PREFERRED_ORDER:
        if name in name_to_path:
            ordered.append(name_to_path[name])
            used_names.add(name)

    # -------------------------
    # 3. 额外 PNG 追加
    # -------------------------
    extra_pngs = sorted(
        [p for p in all_pngs if p.name not in used_names], key=lambda x: x.name
    )

    ordered.extend(extra_pngs)

    return ordered


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


def get_checker_targets(folder: Path):
    """
    获取需要和 GT 做棋盘格检查的图片。

    规则：
    1. 只有存在 dahua_window.png 时，才生成棋盘格检查；
    2. 如果只有 dahua.png，没有 dahua_window.png，则不做棋盘格；
    3. dahua.png 是原始大华全图，不参与棋盘格检查；
    4. dahua_window.png 存在时，加入 GT / dahua_window.png；
    5. 其他模态图像正常加入 GT 棋盘格检查。
    """
    all_pngs = list(folder.glob("*.png"))
    name_to_path = {p.name: p for p in all_pngs}

    # 如果没有 dahua_window.png，说明只有原始大华图 dahua.png，
    # 此时不生成棋盘格检查图
    if "dahua_window.png" not in name_to_path:
        return []

    targets = []

    # 有 dahua_window.png 时，才加入大华车窗图棋盘格
    targets.append(name_to_path["dahua_window.png"])

    # 其他图像按固定顺序加入
    for name in CHECKER_ORDER:
        if name in name_to_path:
            targets.append(name_to_path[name])

    return targets


def concat_checkerboards_in_folder(folder: Path):
    """
    对当前子文件夹生成棋盘格检查图。

    规则：
    1. 如果当前子文件夹没有 dahua_window.png，只存在 dahua.png，则不生成棋盘格；
    2. 如果存在 dahua_window.png，则第 1 张显示 GT 原图；
    3. 后续图片为 GT 分别和 dahua_window、NIR、NIR_LONG、RGB、RGB_LONG 做棋盘格。
    """
    # 没有 dahua_window.png 时，不做棋盘格
    if not (folder / "dahua_window.png").exists():
        print(f"跳过棋盘格 {folder.name}: 只有 dahua.png，没有 dahua_window.png")
        return

    gt_path = folder / "GT_Camera_usb.png"

    if not gt_path.exists():
        print(f"跳过棋盘格 {folder.name}: 没有找到 GT_Camera_usb.png")
        return

    gt_img = read_image_rgb(gt_path)
    target_paths = get_checker_targets(folder)

    if len(target_paths) == 0:
        print(f"跳过棋盘格 {folder.name}: 没有可用于棋盘格检查的图片")
        return

    checker_cells = []

    # 第一张固定显示 GT 原图
    gt_cell = make_cell_from_image(gt_img, "GT_Camera_usb.png")
    checker_cells.append(gt_cell)

    # 后续显示 GT / 其他图棋盘格
    for img_path in target_paths:
        other_img = read_image_rgb(img_path)

        checker = make_checkerboard(
            gt_img=gt_img,
            other_img=other_img,
            tile_size=CHECKER_TILE_SIZE,
        )

        title = f"GT / {img_path.name}"
        cell = make_cell_from_image(checker, title)
        checker_cells.append(cell)

    cols = COLS
    rows = max(ROWS, ceil(len(checker_cells) / cols))

    canvas_h = rows * (TITLE_H + CELL_H) + (rows - 1) * GAP
    canvas_w = cols * CELL_W + (cols - 1) * GAP

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, cell in enumerate(checker_cells):
        row = idx // cols
        col = idx % cols

        y1 = row * (TITLE_H + CELL_H + GAP)
        x1 = col * (CELL_W + GAP)

        canvas[y1 : y1 + cell.shape[0], x1 : x1 + cell.shape[1]] = cell

    save_path = CHECKER_SAVE_DIR / f"{folder.name}_checker_grid.png"
    cv2.imwrite(str(save_path), canvas)

    print(f"已保存棋盘格: {save_path}")


def concat_pngs_in_folder(folder: Path):
    """
    拼接单个子文件夹中的 PNG。
    """
    png_paths = get_ordered_pngs(folder)

    if len(png_paths) == 0:
        print(f"跳过 {folder.name}: 没有找到 png 图片")
        return

    # 默认 3x2，如果图片数量超过 6，则自动增加行数
    cols = COLS
    rows = max(ROWS, ceil(len(png_paths) / cols))

    canvas_h = rows * (TITLE_H + CELL_H) + (rows - 1) * GAP
    canvas_w = cols * CELL_W + (cols - 1) * GAP

    canvas = np.full((canvas_h, canvas_w, 3), BG_COLOR, dtype=np.uint8)

    for idx, img_path in enumerate(png_paths):
        row = idx // cols
        col = idx % cols

        cell = make_cell(img_path)

        y1 = row * (TITLE_H + CELL_H + GAP)
        x1 = col * (CELL_W + GAP)

        canvas[y1 : y1 + cell.shape[0], x1 : x1 + cell.shape[1]] = cell

    save_path = SAVE_DIR / f"{folder.name}_png_grid.png"
    cv2.imwrite(str(save_path), canvas)

    print(f"已保存: {save_path}")

    # 根据开关决定是否额外生成棋盘格检查图
    if GENERATE_CHECKERBOARD:
        concat_checkerboards_in_folder(folder)


def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR 不存在: {ROOT_DIR}")

    subfolders = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])

    # 跳过输出文件夹自身
    skip_names = {SAVE_DIR.name}

    if GENERATE_CHECKERBOARD:
        skip_names.add(CHECKER_SAVE_DIR.name)

    subfolders = [p for p in subfolders if p.name not in skip_names]

    if MAX_GROUPS is not None:
        subfolders = subfolders[:MAX_GROUPS]

    print(f"输入目录: {ROOT_DIR}")
    print(f"输出目录: {SAVE_DIR}")
    print(f"本次处理子文件夹数量: {len(subfolders)}")

    for folder in subfolders:
        print(f"\n处理子文件夹: {folder.name}")
        concat_pngs_in_folder(folder)

    print("\n全部拼接完成")


if __name__ == "__main__":
    main()
