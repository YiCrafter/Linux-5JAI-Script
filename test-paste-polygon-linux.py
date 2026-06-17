"""
脚本功能说明：

本脚本用于根据每个数据子文件夹中的 _dahua_paste_back_meta.json，
将原始大华全图 dahua.png 上的车窗贴回区域可视化出来。

主要功能：
1. 读取 _dahua_paste_back_meta.json 中的 original_window_polygon_xy；
2. 在对应的 dahua.png 原图上绘制车窗贴回四边形；
3. 标注四个角点编号，方便检查贴回位置是否准确；
4. 支持两种输入路径：
   - 输入单个数字开头的数据子文件夹；
   - 输入多个数据子文件夹的上一级目录；
5. 当输入为上一级目录时，只处理名称以数字开头的子文件夹，
   自动跳过 _checker_grid_gt、_png_2x 等辅助目录；
6. 每个处理成功的子文件夹中会生成 _debug_polygon_on_dahua.png。
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================
# 可以填单个子文件夹：
# INPUT_PATH = Path(r"D:\company-file-5-22\output-06-03\_window_crop_aligned_keep_format-dahua-cor\0007_205612_106")

# 也可以填子文件夹的上一级目录：
INPUT_PATH = Path(r"D:\company-file-5-22\output-06-03\_window_crop_aligned_keep_format-dahua-cor")

# 原始大华图文件名
DAHUA_IMAGE_NAME = "dahua.png"

# meta 文件名
META_JSON_NAME = "_dahua_paste_back_meta.json"

# 输出可视化文件名
OUTPUT_DEBUG_NAME = "_debug_polygon_on_dahua.png"

# 是否只处理数字开头的子文件夹
ONLY_PROCESS_DIGIT_PREFIX_FOLDER = True


# =========================================================
# 2. 工具函数
# =========================================================
def is_digit_prefix_folder(folder: Path):
    """
    判断文件夹名称是否以数字开头。
    例如：
        0007_205612_106 -> True
        _checker_grid_gt -> False
        _png_2x -> False
    """
    return folder.is_dir() and len(folder.name) > 0 and folder.name[0].isdigit()


def is_valid_data_folder(folder: Path):
    """
    判断当前文件夹是否包含可处理所需文件。
    """
    img_path = folder / DAHUA_IMAGE_NAME
    json_path = folder / META_JSON_NAME

    return img_path.exists() and json_path.exists()


def get_folders_to_process(input_path: Path):
    """
    根据 INPUT_PATH 自动判断处理模式：
    1. 如果 INPUT_PATH 本身就是一个数据子文件夹，则直接处理它；
    2. 如果 INPUT_PATH 是上一级目录，则处理其下数字开头的数据子文件夹。
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    # 情况 1：INPUT_PATH 本身就是一个数据子文件夹
    if input_path.is_dir() and is_valid_data_folder(input_path):
        return [input_path]

    # 情况 2：INPUT_PATH 是上一级目录
    if input_path.is_dir():
        folders = []

        for p in sorted(input_path.iterdir()):
            if not p.is_dir():
                continue

            if ONLY_PROCESS_DIGIT_PREFIX_FOLDER and not is_digit_prefix_folder(p):
                continue

            if is_valid_data_folder(p):
                folders.append(p)

        return folders

    raise ValueError(f"输入路径不是有效文件夹: {input_path}")


def read_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 3. 单个文件夹处理
# =========================================================
def draw_polygon_on_dahua(folder: Path):
    img_path = folder / DAHUA_IMAGE_NAME
    json_path = folder / META_JSON_NAME
    save_path = folder / OUTPUT_DEBUG_NAME

    if not img_path.exists():
        raise FileNotFoundError(f"缺失大华原图: {img_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"缺失 meta json: {json_path}")

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"读取 dahua.png 失败: {img_path}")

    meta = read_json(json_path)

    if "original_window_polygon_xy" not in meta:
        raise KeyError(f"json 中缺少 original_window_polygon_xy 字段: {json_path}")

    pts = np.array(meta["original_window_polygon_xy"], dtype=np.int32)

    if pts.ndim != 2 or pts.shape[0] != 4 or pts.shape[1] != 2:
        raise ValueError(
            f"original_window_polygon_xy 格式不正确，应为 4x2，当前 shape={pts.shape}"
        )

    vis = img.copy()

    # 画四边形
    cv2.polylines(
        vis,
        [pts],
        isClosed=True,
        color=(0, 0, 255),
        thickness=4
    )

    # 画四个角点和编号
    for i, (x, y) in enumerate(pts):
        cv2.circle(
            vis,
            (int(x), int(y)),
            8,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            vis,
            str(i),
            (int(x) + 10, int(y) + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    ok = cv2.imwrite(str(save_path), vis)

    if not ok:
        raise ValueError(f"保存失败: {save_path}")

    return save_path


# =========================================================
# 4. 主函数
# =========================================================
def main():
    folders = get_folders_to_process(INPUT_PATH)

    if len(folders) == 0:
        print("没有找到可处理的数据文件夹。")
        print("请检查 INPUT_PATH 是否正确，或者子文件夹中是否包含 dahua.png 和 _dahua_paste_back_meta.json。")
        return

    print(f"输入路径: {INPUT_PATH}")
    print(f"待处理文件夹数量: {len(folders)}")

    success_count = 0
    fail_count = 0

    for folder in tqdm(folders, desc="绘制进度", unit="组", ncols=100):
        try:
            save_path = draw_polygon_on_dahua(folder)
            success_count += 1
            print(f"保存成功: {save_path}")
        except Exception as e:
            fail_count += 1
            print(f"处理失败: {folder.name}, 原因: {e}")

    print("\n全部处理完成")
    print(f"成功: {success_count} 组")
    print(f"失败: {fail_count} 组")


if __name__ == "__main__":
    main()