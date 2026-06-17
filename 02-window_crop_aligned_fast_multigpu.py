"""
脚本功能说明：

本脚本用于批量生成多相机车窗区域对齐裁剪数据。程序会遍历 ROOT_DIR 下的各个子文件夹，
读取 GT、NIR、NIR_LONG、RGB、RGB_LONG 和 DAHUA 图像，并基于 Homography、YOLO 车窗分割
和 ROI 内 ECC 微调，将各模态图像对齐到 GT 视角后裁剪出统一车窗区域。

主要功能：
1. 将各模态图像通过 Homography 变换到 GT 视角；
2. 优先在 NIR / NIR_LONG 上使用 YOLO 检测车窗 mask，并构造外接多边形 ROI；
3. 在车窗 ROI 内对非 GT 图像进行 ECC 微调配准；
4. 使用统一 bbox 裁剪 GT、NIR、NIR_LONG、RGB、RGB_LONG 和 DAHUA 车窗区域；
5. 对 .npy 车窗裁剪结果额外保存一份同名 .png 预览图，方便人工检查；
6. 从源子文件夹复制 dahua.png 原始大华全图到目标子文件夹；
7. 将裁剪得到的大华车窗图保存为 dahua_window.png；
8. 为每组数据保存精简版 _dahua_paste_back_meta.json，用于后续将 pix2pixHD 输出图贴回原始大华全图；
9. 终端仅显示整体处理进度，异常信息统一记录到输出目录下 txt/error_records.txt。
"""

import json
import csv
import shutil
import cv2
import numpy as np
import os
import traceback
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

#  python 02_window_crop_aligned_with_completeness_filter-linux.py  --root_dir /mnt/bigdata/gongyong/5JAI/processed_2026-05-22


# =========================================================
# 1. 路径配置
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="批量生成多相机车窗区域对齐裁剪数据")

    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="当天数据目录，例如 /mnt/bigdata/gongyong/5JAI/processed_2026-05-22",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default="_window_crop",
        help="输出文件夹名称，默认保存在 root_dir 下",
    )

    return parser.parse_args()


ARGS = parse_args()

ROOT_DIR = Path(ARGS.root_dir)

HOMO_JSON = Path(r"resources/02-homo_6camera_refer_gt.json")

YOLO_WEIGHT = Path(r"resources/02-best-v8-window-3k-26-5-28.pt")

OUTPUT_DIR = ROOT_DIR / ARGS.output_name
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 异常记录目录，与各输出子文件夹同级
TXT_DIR = OUTPUT_DIR / "txt"
TXT_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG_PATH = TXT_DIR / "error_records.txt"

# =========================================================
# 1.1 加速配置
# =========================================================
# 4 张 3090：默认每张 GPU 一个 worker；如果 GPU 利用率低，可把 NUM_WORKERS_PER_GPU 改成 2。
GPU_DEVICES = [0, 1, 2, 3]
NUM_WORKERS_PER_GPU = 1

# 多进程时建议每个 worker 只给 OpenCV 1 个线程，避免 CPU 线程过度竞争。
CV2_NUM_THREADS_PER_WORKER = 1

# 训练只需要 npy 时建议改为 False，能明显减少 PNG 编码和磁盘写入时间。
SAVE_NPY_PREVIEW_PNG = False

# 贴回需要 dahua.png；只训练 crop 时可设为 False。
COPY_ORIGINAL_DAHUA = False

# 是否保存每组 _saved_window_crop_file_stats.csv。关闭可以减少小文件写入。
SAVE_FOLDER_STATS_CSV = False

# 中断后续跑时可设为 True，已生成 _dahua_paste_back_meta.json 的组会跳过。
SKIP_ALREADY_PROCESSED = False

# Linux + CUDA 推荐 spawn，避免 fork 后 CUDA 状态异常。
MP_START_METHOD = "spawn"


# =========================================================
# 2. YOLO 配置
# =========================================================
YOLO_IMGSZ = 640
YOLO_CONF = 0.25

# 使用第 0 块 GPU；如果用 CPU，改成 "cpu"
YOLO_DEVICE = 0

# 如果模型是单类 window，一般为 0
# 如果模型是 0 face, 1 window，则改成 1
# None 表示自动识别
FORCE_WINDOW_CLASS_ID = None

# YOLO 权重是红外图训练的，所以优先在 NIR / NIR_LONG 上检测车窗
ROI_SOURCE_ORDER = ["NIR", "NIR_LONG"]


# =========================================================
# 3. ROI + ECC 配置
# =========================================================
# 推荐 euclidean：只做平移+旋转，比较稳
# 可选: "translation" / "euclidean" / "affine" / "homography"
ECC_MOTION_MODE = "euclidean"

ECC_MAX_ITER = 80
ECC_EPS = 1e-5

# 多模态图像差异大，用梯度图做 ECC 更稳
ECC_USE_GRADIENT = True

# 外接多边形模式:
# "hull"    : 最大轮廓凸包，默认推荐
# "approx"  : 轮廓近似多边形，更贴合 mask
# "minrect" : 最小外接旋转矩形
POLYGON_MODE = "hull"
POLYGON_APPROX_EPS_RATIO = 0.006

# ROI 膨胀像素，让 ECC 和最终裁剪区域保留一点上下文
POLYGON_ROI_DILATE = 8

# 排除 Homography 后的黑边
VALID_MASK_THRESHOLD = 5

# mask 最小有效像素
MIN_MASK_PIXELS = 1000

# 最终裁剪用哪个 mask 来计算 bbox:
# "polygon" : 用外接多边形 ROI 的 bbox，推荐，空间更稳定
# "yolo"    : 用 YOLO 原始 mask 的 bbox，更贴合分割结果
FINAL_MASK_MODE = "polygon"

# 最终只保存车窗区域 crop，不保留整图黑色背景
# 0 表示紧贴车窗 ROI bbox；如果担心裁太紧，可以改成 10、20
CROP_PADDING = 0


# =========================================================
# 4. 文件和 Homography 对应关系
# =========================================================
IMAGE_ITEMS = [
    {
        "title": "GT",
        "filename": "GT_Camera_usb.npy",
        "output_filename": "GT_Camera_usb.npy",
        "homo_key": "GT",
        "is_reference": True,
    },
    {
        "title": "NIR",
        "filename": "Nir_Camera_usb.npy",
        "output_filename": "Nir_Camera_usb.npy",
        "homo_key": "NIR",
        "is_reference": False,
    },
    {
        "title": "NIR_LONG",
        "filename": "NIR_Long_Camera_248.npy",
        "output_filename": "NIR_Long_Camera_248.npy",
        "homo_key": "NIR_LONG",
        "is_reference": False,
    },
    {
        "title": "RGB",
        "filename": "RGB_Camera_249.npy",
        "output_filename": "RGB_Camera_249.npy",
        "homo_key": "RGB",
        "is_reference": False,
    },
    {
        "title": "RGB_LONG",
        "filename": "RGB_Long_Camera_247.npy",
        "output_filename": "RGB_Long_Camera_247.npy",
        "homo_key": "RGB_LONG",
        "is_reference": False,
    },
    {
        "title": "DAHUA",
        "filename": "dahua.png",
        # dahua.png 在目标文件夹中保留为“原始大华全图”，因此车窗裁剪图另存为 dahua_window.png
        "output_filename": "dahua_window.png",
        "homo_key": "DAHUA",
        "is_reference": False,
    },
]


# =========================================================
# 5. 日志与异常记录
# =========================================================
def reset_error_log():
    with open(ERROR_LOG_PATH, "w", encoding="utf-8") as f:
        f.write("folder\treason\n")


def log_error(folder_name, reason):
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{folder_name}\t{reason}\n")


# =========================================================
# 6. 基础工具函数
# =========================================================
def load_homographies(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "homographies" not in data:
        raise KeyError("json 文件中没有 homographies 字段")

    return data["homographies"]


def get_homography(homo_dict, key, is_reference=False):
    if is_reference:
        return np.eye(3, dtype=np.float64)

    if key not in homo_dict:
        raise KeyError(f"json 中没有找到 homography key: {key}")

    H = np.array(homo_dict[key], dtype=np.float64)

    if H.shape != (3, 3):
        raise ValueError(f"{key} 的 homography 不是 3x3，当前 shape={H.shape}")

    return H


def load_image_keep_format(file_path: Path):
    """
    读取图像，同时保存原始格式信息。
    - npy: 保持原始 dtype、ndim、shape 信息
    - jpg/png: 读取后转 RGB/RGBA 方便处理，保存时再转回 BGR/BGRA
    """
    suffix = file_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(file_path, allow_pickle=False)
        meta = {
            "suffix": suffix,
            "format": "npy",
            "dtype": arr.dtype,
            "shape": arr.shape,
            "ndim": arr.ndim,
            "is_image_file": False,
        }
        return arr, meta

    if suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

        if img is None:
            raise ValueError(f"读取失败: {file_path}")

        if img.ndim == 3:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        meta = {
            "suffix": suffix,
            "format": suffix.replace(".", ""),
            "dtype": img.dtype,
            "shape": img.shape,
            "ndim": img.ndim,
            "is_image_file": True,
        }
        return img, meta

    raise ValueError(f"不支持的文件格式: {file_path}")


def restore_dtype(arr, meta):
    target_dtype = meta["dtype"]

    if np.issubdtype(target_dtype, np.floating):
        return arr.astype(target_dtype)

    if np.issubdtype(target_dtype, np.integer):
        info = np.iinfo(target_dtype)
        arr = np.rint(arr)
        arr = np.clip(arr, info.min, info.max)
        return arr.astype(target_dtype)

    return arr.astype(target_dtype)


def save_image_keep_format(save_path: Path, arr, meta):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    arr = restore_dtype(arr, meta)

    if not meta["is_image_file"]:
        np.save(save_path, arr)
        return arr

    out = arr

    if out.ndim == 3:
        if out.shape[2] == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        elif out.shape[2] == 4:
            out = cv2.cvtColor(out, cv2.COLOR_RGBA2BGRA)

    ok = cv2.imwrite(str(save_path), out)
    if not ok:
        raise ValueError(f"保存失败: {save_path}")

    return arr


def array_stats(arr, filename, fmt):
    return {
        "file": filename,
        "format": fmt,
        "shape": str(arr.shape),
        "ndim": int(arr.ndim),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr)) if arr.size > 0 else None,
        "max": float(np.max(arr)) if arr.size > 0 else None,
        "mean": float(np.mean(arr)) if arr.size > 0 else None,
    }


def write_stats_csv(records, save_path: Path):
    if not records:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "folder",
        "file",
        "format",
        "shape",
        "ndim",
        "dtype",
        "min",
        "max",
        "mean",
    ]

    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_json(data, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================================================
# 7. 图像格式转换，用于 YOLO/ECC/PNG 预览
# =========================================================
def to_uint8_plane(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size == 0:
        return x.astype(np.uint8)

    max_v = float(np.max(x))
    min_v = float(np.min(x))

    if max_v <= 1.5 and min_v >= -0.1:
        y = x * 255.0
    elif max_v <= 255.0:
        y = x
    else:
        if max_v > min_v:
            y = (x - min_v) / (max_v - min_v) * 255.0
        else:
            y = np.zeros_like(x, dtype=np.float32)

    return np.clip(y, 0, 255).astype(np.uint8)


def to_uint8_rgb(arr):
    if arr.ndim == 2:
        gray = to_uint8_plane(arr)
        return np.stack([gray, gray, gray], axis=-1)

    if arr.ndim == 3:
        if arr.shape[2] == 1:
            gray = to_uint8_plane(arr[:, :, 0])
            return np.stack([gray, gray, gray], axis=-1)

        if arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
            r = to_uint8_plane(rgb[:, :, 0])
            g = to_uint8_plane(rgb[:, :, 1])
            b = to_uint8_plane(rgb[:, :, 2])
            return np.stack([r, g, b], axis=-1)

    raise ValueError(f"无法转成 RGB，shape={arr.shape}")


def to_uint8_gray(arr):
    rgb = to_uint8_rgb(arr)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def save_png_preview_for_npy(npy_save_path: Path, arr):
    """
    每保存一个 npy crop 后，同目录额外保存一份同名 png 预览图。
    - 灰度 / 单通道：保存灰度 png；
    - RGB：按 RGB -> BGR 保存 png。
    """
    png_save_path = npy_save_path.with_suffix(".png")

    if arr.ndim == 2:
        out = to_uint8_plane(arr)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        out = to_uint8_plane(arr[:, :, 0])
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        out_rgb = to_uint8_rgb(arr[:, :, :3])
        out = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"无法保存 png 预览，shape={arr.shape}")

    ok = cv2.imwrite(str(png_save_path), out)
    if not ok:
        raise ValueError(f"png 预览保存失败: {png_save_path}")


def copy_original_dahua_to_output(source_folder: Path, save_folder: Path):
    """
    将源子文件夹中的 dahua.png 原始大华全图复制到目标子文件夹中。
    注意：目标中的 dahua.png 是原始大华全图；裁剪后的大华车窗图保存为 dahua_window.png。
    """
    src = source_folder / "dahua.png"
    dst = save_folder / "dahua.png"

    if not src.exists():
        raise FileNotFoundError("缺失 dahua.png，无法复制原始大华全图")

    save_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# =========================================================
# 8. Homography、裁剪、反投影相关函数
# =========================================================
def zero_border_value(arr):
    if arr.ndim == 2:
        return 0

    if arr.ndim == 3:
        return tuple([0] * arr.shape[2])

    return 0


def warp_to_gt_view(arr, H, target_w, target_h):
    warped = cv2.warpPerspective(
        arr,
        H,
        dsize=(target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=zero_border_value(arr),
    )
    return warped


def get_bbox_from_mask(mask_uint8, padding=0):
    if mask_uint8 is None:
        return None

    ys, xs = np.where(mask_uint8 > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask_uint8.shape[:2]

    x1 = max(0, int(xs.min()) - padding)
    x2 = min(w, int(xs.max()) + 1 + padding)
    y1 = max(0, int(ys.min()) - padding)
    y2 = min(h, int(ys.max()) + 1 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def crop_by_bbox(arr, bbox):
    if bbox is None:
        return arr

    x1, y1, x2, y2 = bbox

    if arr.ndim == 2:
        return arr[y1:y2, x1:x2]

    if arr.ndim == 3:
        return arr[y1:y2, x1:x2, :]

    raise ValueError(f"不支持的图像维度: {arr.shape}")


def motion_matrix_to_3x3(warp_matrix, motion_flag):
    """
    ECC 返回的 warp_matrix 转为 3x3 矩阵。

    在当前代码中，ECC 应用时使用了 cv2.WARP_INVERSE_MAP。
    因此这个 warp_matrix 表示：
    aligned GT 坐标 -> Homography 后 moving 图坐标
    """
    if warp_matrix is None or motion_flag is None:
        return np.eye(3, dtype=np.float64)

    M = np.array(warp_matrix, dtype=np.float64)

    if M.shape == (2, 3):
        M3 = np.eye(3, dtype=np.float64)
        M3[:2, :] = M
        return M3

    if M.shape == (3, 3):
        return M

    raise ValueError(f"不支持的 ECC 矩阵 shape: {M.shape}")


def transform_points_by_homography(points_xy, H):
    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(points, H)
    return out.reshape(-1, 2)


def bbox_from_points(points_xy):
    pts = np.asarray(points_xy, dtype=np.float64)
    x1 = float(np.min(pts[:, 0]))
    y1 = float(np.min(pts[:, 1]))
    x2 = float(np.max(pts[:, 0]))
    y2 = float(np.max(pts[:, 1]))
    return [x1, y1, x2, y2]


def clip_bbox_xyxy(bbox, width, height):
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))

    return [float(x1), float(y1), float(x2), float(y2)]


def matrix_to_list(M):
    M = np.asarray(M, dtype=np.float64)
    return [[float(v) for v in row] for row in M]


def points_to_list(points):
    return [[float(x), float(y)] for x, y in np.asarray(points, dtype=np.float64)]


def build_dahua_paste_back_meta(
    original_dahua_arr,
    H_dahua_to_gt,
    crop_bbox_gt_xyxy,
    saved_dahua_crop_arr,
    ecc_warp_matrix,
    ecc_motion_flag_value,
):
    """
    构建精简版 DAHUA crop 反贴回原始 DAHUA 全图所需信息。

    后续 pix2pixHD 输出图如果与 dahua_window.png 尺寸一致，可直接使用：
        canvas = cv2.warpPerspective(pred_crop, matrix_crop_to_original_dahua, (original_w, original_h))
    将预测车窗图映射回原始 dahua.png 全图坐标。
    """
    original_h, original_w = original_dahua_arr.shape[:2]

    x1, y1, x2, y2 = crop_bbox_gt_xyxy
    crop_w = int(x2 - x1)
    crop_h = int(y2 - y1)

    H_dahua_to_gt = np.asarray(H_dahua_to_gt, dtype=np.float64)
    H_gt_to_dahua = np.linalg.inv(H_dahua_to_gt)

    # crop 坐标 -> GT 全图坐标
    T_crop_to_full_gt = np.array(
        [
            [1.0, 0.0, float(x1)],
            [0.0, 1.0, float(y1)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # ECC 矩阵：GT aligned 全图坐标 -> Homography 后 DAHUA 图坐标
    M_ecc_aligned_to_homography = motion_matrix_to_3x3(
        ecc_warp_matrix,
        ecc_motion_flag_value,
    )

    # 模型输出 crop 坐标 -> 原始 DAHUA 全图坐标
    M_crop_to_original_dahua = (
        H_gt_to_dahua @ M_ecc_aligned_to_homography @ T_crop_to_full_gt
    )

    crop_corners_xy = np.array(
        [
            [0.0, 0.0],
            [float(crop_w), 0.0],
            [float(crop_w), float(crop_h)],
            [0.0, float(crop_h)],
        ],
        dtype=np.float64,
    )

    original_polygon_xy = transform_points_by_homography(
        crop_corners_xy,
        M_crop_to_original_dahua,
    )

    original_bbox_xyxy = bbox_from_points(original_polygon_xy)
    original_bbox_clipped_xyxy = clip_bbox_xyxy(
        original_bbox_xyxy,
        width=original_w,
        height=original_h,
    )

    return {
        "original_dahua_file": "dahua.png",
        "window_crop_file": "dahua_window.png",
        "original_dahua_size_hw": [int(original_h), int(original_w)],
        "window_crop_size_hw": [
            int(saved_dahua_crop_arr.shape[0]),
            int(saved_dahua_crop_arr.shape[1]),
        ],
        "paste_back_method": "cv2.warpPerspective(pred_crop, matrix_crop_to_original_dahua, (original_w, original_h))",
        "matrix_crop_to_original_dahua": matrix_to_list(M_crop_to_original_dahua),
        "original_window_polygon_xy": points_to_list(original_polygon_xy),
        "original_window_bbox_xyxy": [float(v) for v in original_bbox_xyxy],
        "original_window_bbox_clipped_xyxy": [
            float(v) for v in original_bbox_clipped_xyxy
        ],
    }


# =========================================================
# 9. YOLO 车窗 mask
# =========================================================
def get_window_class_id(model):
    if FORCE_WINDOW_CLASS_ID is not None:
        return int(FORCE_WINDOW_CLASS_ID)

    names = model.names

    for class_id, name in names.items():
        name = str(name).lower()
        if name in ["window", "windows", "car_window", "car window"]:
            return int(class_id)

    if len(names) == 1:
        return int(list(names.keys())[0])

    # 不再终端详细打印，返回 None 表示不指定 classes
    return None


def clean_mask(mask_uint8):
    mask = (mask_uint8 > 0).astype(np.uint8) * 255

    if mask.sum() == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels <= 1:
        return mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + np.argmax(areas)

    clean = np.zeros_like(mask)
    clean[labels == largest_label] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return clean


def predict_window_mask(model, img_rgb_uint8, window_class_id):
    h, w = img_rgb_uint8.shape[:2]

    classes = None
    if window_class_id is not None:
        classes = [window_class_id]

    results = model.predict(
        source=img_rgb_uint8,
        imgsz=YOLO_IMGSZ,
        conf=YOLO_CONF,
        device=YOLO_DEVICE,
        classes=classes,
        verbose=False,
        # 关键：关闭 Ultralytics 默认预测结果保存
        save=False,
        save_txt=False,
        save_conf=False,
        save_crop=False,
        show=False,
    )

    result = results[0]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return final_mask

    masks = result.masks.data.cpu().numpy()

    for m in masks:
        m = (m > 0.5).astype(np.uint8)

        if m.shape[:2] != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        final_mask = np.maximum(final_mask, m * 255)

    return clean_mask(final_mask)


def get_shared_yolo_window_mask(warped_dict, model, window_class_id):
    for key in ROI_SOURCE_ORDER:
        if key not in warped_dict:
            continue

        img_rgb = to_uint8_rgb(warped_dict[key])
        mask = predict_window_mask(model, img_rgb, window_class_id)

        area = int((mask > 0).sum())
        if area > MIN_MASK_PIXELS:
            return mask, key

    return None, None


# =========================================================
# 10. mask 转外接多边形 ROI
# =========================================================
def mask_to_external_polygon_roi(window_mask):
    if window_mask is None:
        return None, None

    mask = (window_mask > 0).astype(np.uint8) * 255

    if mask.sum() == 0:
        return None, None

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        return None, None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_MASK_PIXELS:
        return None, None

    mode = POLYGON_MODE.lower()

    if mode == "hull":
        hull = cv2.convexHull(contour)
        eps = POLYGON_APPROX_EPS_RATIO * cv2.arcLength(hull, True)
        polygon = cv2.approxPolyDP(hull, eps, True)

    elif mode == "approx":
        eps = POLYGON_APPROX_EPS_RATIO * cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, eps, True)

    elif mode == "minrect":
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        polygon = np.int32(box).reshape(-1, 1, 2)

    else:
        raise ValueError(f"不支持的 POLYGON_MODE: {POLYGON_MODE}")

    h, w = mask.shape[:2]
    polygon_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(polygon_roi, [polygon.astype(np.int32)], 255)

    if POLYGON_ROI_DILATE > 0:
        k = POLYGON_ROI_DILATE * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        polygon_roi = cv2.dilate(polygon_roi, kernel, iterations=1)

    return polygon_roi, polygon.astype(np.int32)


# =========================================================
# 11. ECC 配准
# =========================================================
def ecc_motion_flag(mode_name):
    mode_name = mode_name.lower()

    if mode_name == "translation":
        return cv2.MOTION_TRANSLATION, np.eye(2, 3, dtype=np.float32)

    if mode_name == "euclidean":
        return cv2.MOTION_EUCLIDEAN, np.eye(2, 3, dtype=np.float32)

    if mode_name == "affine":
        return cv2.MOTION_AFFINE, np.eye(2, 3, dtype=np.float32)

    if mode_name == "homography":
        return cv2.MOTION_HOMOGRAPHY, np.eye(3, 3, dtype=np.float32)

    raise ValueError(f"不支持的 ECC motion mode: {mode_name}")


def preprocess_for_ecc(arr):
    gray = to_uint8_gray(arr)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    if ECC_USE_GRADIENT:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
        return mag.astype(np.float32)

    return gray.astype(np.float32) / 255.0


def make_valid_overlap_mask(fixed_arr, moving_arr):
    fixed_gray = to_uint8_gray(fixed_arr)
    moving_gray = to_uint8_gray(moving_arr)

    valid = (
        (fixed_gray > VALID_MASK_THRESHOLD) & (moving_gray > VALID_MASK_THRESHOLD)
    ).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    valid = cv2.erode(valid, kernel, iterations=1)

    return valid


def combine_roi_and_valid(roi_mask, valid_mask):
    if roi_mask is None:
        return None

    if int((roi_mask > 0).sum()) < MIN_MASK_PIXELS:
        return None

    if valid_mask is None:
        return roi_mask

    out = ((roi_mask > 0) & (valid_mask > 0)).astype(np.uint8) * 255

    if int((out > 0).sum()) < MIN_MASK_PIXELS:
        return None

    return out


def estimate_ecc_transform(fixed_arr, moving_arr, ecc_mask, mode_name):
    if ecc_mask is None:
        return None, None, False, None

    ecc_mask = (ecc_mask > 0).astype(np.uint8) * 255

    if int((ecc_mask > 0).sum()) < MIN_MASK_PIXELS:
        return None, None, False, None

    fixed_ecc = preprocess_for_ecc(fixed_arr)
    moving_ecc = preprocess_for_ecc(moving_arr)

    motion_flag, warp_matrix = ecc_motion_flag(mode_name)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITER,
        ECC_EPS,
    )

    try:
        cc, warp_matrix = cv2.findTransformECC(
            templateImage=fixed_ecc,
            inputImage=moving_ecc,
            warpMatrix=warp_matrix,
            motionType=motion_flag,
            criteria=criteria,
            inputMask=ecc_mask,
            gaussFiltSize=5,
        )

        return warp_matrix, motion_flag, True, cc

    except cv2.error:
        return None, None, False, None


def apply_ecc_transform_to_array(arr, warp_matrix, motion_flag, target_w, target_h):
    if warp_matrix is None or motion_flag is None:
        return arr

    if motion_flag == cv2.MOTION_HOMOGRAPHY:
        aligned = cv2.warpPerspective(
            arr,
            warp_matrix,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=zero_border_value(arr),
        )
    else:
        aligned = cv2.warpAffine(
            arr,
            warp_matrix,
            (target_w, target_h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=zero_border_value(arr),
        )

    return aligned


# =========================================================
# 12. 单个文件夹处理
# =========================================================
def process_one_folder(folder: Path, homo_dict, model, window_class_id):
    gt_path = folder / "GT_Camera_usb.npy"

    if not gt_path.exists():
        raise FileNotFoundError("缺失 GT_Camera_usb.npy")

    loaded = {}

    # 读取六张图，保留原文件名和原格式信息
    for item in IMAGE_ITEMS:
        key = item["title"]
        file_path = folder / item["filename"]

        if not file_path.exists():
            raise FileNotFoundError(f"缺失 {item['filename']}")

        arr, meta = load_image_keep_format(file_path)

        loaded[key] = {
            "arr": arr,
            "meta": meta,
            "filename": item["filename"],
            "output_filename": item["output_filename"],
            "file_path": file_path,
            "homo_key": item["homo_key"],
            "is_reference": item["is_reference"],
        }

    gt_arr = loaded["GT"]["arr"]
    gt_h, gt_w = gt_arr.shape[:2]

    # -----------------------------------------------------
    # 12.1 所有图先 Homography 到 GT 视角
    # -----------------------------------------------------
    warped_dict = {}

    for key, data in loaded.items():
        H = get_homography(
            homo_dict=homo_dict,
            key=data["homo_key"],
            is_reference=data["is_reference"],
        )

        data["H_to_gt"] = H

        warped = warp_to_gt_view(
            arr=data["arr"],
            H=H,
            target_w=gt_w,
            target_h=gt_h,
        )

        warped_dict[key] = warped

    gt_warped = warped_dict["GT"]

    # -----------------------------------------------------
    # 12.2 YOLO 获取车窗 mask，并转外接多边形 ROI
    # -----------------------------------------------------
    yolo_mask, roi_source = get_shared_yolo_window_mask(
        warped_dict=warped_dict,
        model=model,
        window_class_id=window_class_id,
    )

    if roi_source is None:
        raise RuntimeError("NIR / NIR_LONG 均未检测到有效车窗 ROI")

    polygon_roi, polygon_points = mask_to_external_polygon_roi(yolo_mask)

    if FINAL_MASK_MODE.lower() == "yolo":
        final_roi = yolo_mask
    elif FINAL_MASK_MODE.lower() == "polygon":
        final_roi = polygon_roi
    else:
        raise ValueError(f"不支持的 FINAL_MASK_MODE: {FINAL_MASK_MODE}")

    if final_roi is None or int((final_roi > 0).sum()) < MIN_MASK_PIXELS:
        raise RuntimeError("没有有效车窗 ROI")

    # -----------------------------------------------------
    # 12.3 根据 ROI 计算公共裁剪框
    # -----------------------------------------------------
    crop_bbox = get_bbox_from_mask(final_roi, padding=CROP_PADDING)

    if crop_bbox is None:
        raise RuntimeError("无法根据 ROI 计算裁剪框")

    # -----------------------------------------------------
    # 12.4 输出目录，并复制原始大华全图
    # -----------------------------------------------------
    save_folder = OUTPUT_DIR / folder.name
    save_folder.mkdir(parents=True, exist_ok=True)

    if COPY_ORIGINAL_DAHUA:
        copy_original_dahua_to_output(folder, save_folder)

    records = []
    dahua_paste_back_meta = None

    # -----------------------------------------------------
    # 12.5 对每张图做 ECC，然后用同一个 bbox 裁剪保存
    # -----------------------------------------------------
    for key, data in loaded.items():
        output_filename = data["output_filename"]
        meta = data["meta"]
        before_arr = warped_dict[key]

        warp_matrix = None
        motion_flag = None
        ok = False
        cc = None

        if key == "GT":
            aligned_arr = before_arr
        else:
            valid_mask = make_valid_overlap_mask(gt_warped, before_arr)
            ecc_mask = combine_roi_and_valid(polygon_roi, valid_mask)

            warp_matrix, motion_flag, ok, cc = estimate_ecc_transform(
                fixed_arr=gt_warped,
                moving_arr=before_arr,
                ecc_mask=ecc_mask,
                mode_name=ECC_MOTION_MODE,
            )

            if ok:
                aligned_arr = apply_ecc_transform_to_array(
                    arr=before_arr,
                    warp_matrix=warp_matrix,
                    motion_flag=motion_flag,
                    target_w=gt_w,
                    target_h=gt_h,
                )
            else:
                # ECC 失败不作为致命异常，保留 Homography 结果继续保存
                aligned_arr = before_arr

        window_crop = crop_by_bbox(aligned_arr, crop_bbox)
        save_path = save_folder / output_filename

        saved_arr = save_image_keep_format(
            save_path=save_path,
            arr=window_crop,
            meta=meta,
        )

        # 对 npy 文件额外保存一份同名 png 预览图
        if SAVE_NPY_PREVIEW_PNG and save_path.suffix.lower() == ".npy":
            save_png_preview_for_npy(save_path, saved_arr)

        # 直接使用内存中的 saved_arr 做统计，避免保存后再从磁盘重新读取一次
        saved_arr_for_stats = saved_arr
        saved_meta = meta

        rec = array_stats(
            arr=saved_arr_for_stats,
            filename=output_filename,
            fmt=saved_meta["format"],
        )
        rec["folder"] = folder.name
        records.append(rec)

        # -------------------------------------------------
        # 12.6 记录 DAHUA window crop 贴回原视角所需的精简信息
        # -------------------------------------------------
        if key == "DAHUA":
            dahua_paste_back_meta = build_dahua_paste_back_meta(
                original_dahua_arr=data["arr"],
                H_dahua_to_gt=data["H_to_gt"],
                crop_bbox_gt_xyxy=crop_bbox,
                saved_dahua_crop_arr=saved_arr_for_stats,
                ecc_warp_matrix=warp_matrix,
                ecc_motion_flag_value=motion_flag,
            )

    if SAVE_FOLDER_STATS_CSV:
        folder_csv = save_folder / "_saved_window_crop_file_stats.csv"
        write_stats_csv(records, folder_csv)

    if dahua_paste_back_meta is not None:
        dahua_json = save_folder / "_dahua_paste_back_meta.json"
        write_json(dahua_paste_back_meta, dahua_json)
    else:
        raise RuntimeError("未生成 DAHUA 贴回信息 json")

    return records


# =========================================================
# 13. 多进程 worker
# =========================================================
_WORKER_MODEL = None
_WORKER_HOMO_DICT = None
_WORKER_WINDOW_CLASS_ID = None
_WORKER_ID = None


def _init_worker(gpu_queue):
    """
    每个进程只初始化一次：绑定 GPU、加载 Homography、加载 YOLO，并限制 OpenCV 线程数。
    """
    global YOLO_DEVICE
    global _WORKER_MODEL
    global _WORKER_HOMO_DICT
    global _WORKER_WINDOW_CLASS_ID
    global _WORKER_ID

    cv2.setNumThreads(CV2_NUM_THREADS_PER_WORKER)

    gpu_id = gpu_queue.get()
    YOLO_DEVICE = gpu_id
    _WORKER_ID = f"pid={os.getpid()}, gpu={gpu_id}"

    _WORKER_HOMO_DICT = load_homographies(HOMO_JSON)
    _WORKER_MODEL = YOLO(str(YOLO_WEIGHT))
    _WORKER_WINDOW_CLASS_ID = get_window_class_id(_WORKER_MODEL)


def _process_folder_task(folder_str):
    """单个文件夹任务。主进程负责收集结果和写 error log。"""
    folder = Path(folder_str)

    try:
        save_folder = OUTPUT_DIR / folder.name
        done_flag = save_folder / "_dahua_paste_back_meta.json"

        if SKIP_ALREADY_PROCESSED and done_flag.exists():
            return {
                "ok": True,
                "skipped_existing": True,
                "folder": folder.name,
                "records": [],
                "error": "",
                "worker": _WORKER_ID,
            }

        records = process_one_folder(
            folder=folder,
            homo_dict=_WORKER_HOMO_DICT,
            model=_WORKER_MODEL,
            window_class_id=_WORKER_WINDOW_CLASS_ID,
        )

        return {
            "ok": True,
            "skipped_existing": False,
            "folder": folder.name,
            "records": records,
            "error": "",
            "worker": _WORKER_ID,
        }

    except Exception as e:
        return {
            "ok": False,
            "skipped_existing": False,
            "folder": folder.name,
            "records": [],
            "error": str(e),
            "traceback": traceback.format_exc(),
            "worker": _WORKER_ID,
        }


# =========================================================
# 14. 主函数
# =========================================================
def main():
    reset_error_log()

    try:
        import torch

        cuda_info = torch.cuda.is_available()
    except Exception:
        cuda_info = "unknown"

    subfolders = sorted([p for p in ROOT_DIR.iterdir() if p.is_dir()])
    # 跳过所有辅助输出目录，避免把 _window、_checker、txt 等当作输入处理
    subfolders = [p for p in subfolders if not p.name.startswith("_")]

    worker_gpu_list = []
    for gpu in GPU_DEVICES:
        for _ in range(NUM_WORKERS_PER_GPU):
            worker_gpu_list.append(gpu)

    if len(worker_gpu_list) == 0:
        worker_gpu_list = ["cpu"]

    num_workers = len(worker_gpu_list)

    print(f"输入目录: {ROOT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"异常记录: {ERROR_LOG_PATH}")
    print(f"待处理子文件夹数量: {len(subfolders)}")
    print(f"CUDA available: {cuda_info}")
    print(f"GPU_DEVICES: {GPU_DEVICES}")
    print(f"NUM_WORKERS_PER_GPU: {NUM_WORKERS_PER_GPU}")
    print(f"总 worker 数: {num_workers}")
    print(f"CV2_NUM_THREADS_PER_WORKER: {CV2_NUM_THREADS_PER_WORKER}")
    print(f"SAVE_NPY_PREVIEW_PNG: {SAVE_NPY_PREVIEW_PNG}")
    print(f"COPY_ORIGINAL_DAHUA: {COPY_ORIGINAL_DAHUA}")
    print(f"SAVE_FOLDER_STATS_CSV: {SAVE_FOLDER_STATS_CSV}")
    print(f"SKIP_ALREADY_PROCESSED: {SKIP_ALREADY_PROCESSED}")

    if len(subfolders) == 0:
        print("没有找到待处理子文件夹。")
        return

    ctx = mp.get_context(MP_START_METHOD)
    gpu_queue = ctx.Queue()
    for gpu in worker_gpu_list:
        gpu_queue.put(gpu)

    all_records = []
    error_records = []
    success_count = 0
    fail_count = 0
    skipped_existing_count = 0

    folder_strs = [str(p) for p in subfolders]

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(gpu_queue,),
    ) as executor:
        futures = [executor.submit(_process_folder_task, s) for s in folder_strs]

        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="处理进度",
            unit="组",
            ncols=100,
        ):
            result = fut.result()

            if result["ok"]:
                if result.get("skipped_existing", False):
                    skipped_existing_count += 1
                else:
                    success_count += 1
                    all_records.extend(result["records"])
            else:
                fail_count += 1
                error_records.append((result["folder"], result["error"]))

    # 主进程统一写错误日志，避免多进程同时写同一个 txt
    with open(ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        for folder_name, reason in error_records:
            f.write(f"{folder_name}\t{reason}\n")

    global_csv = OUTPUT_DIR / "_all_saved_window_crop_file_stats.csv"
    write_stats_csv(all_records, global_csv)

    print("\n全部处理完成。")
    print(f"成功: {success_count} 组")
    print(f"跳过已存在: {skipped_existing_count} 组")
    print(f"异常: {fail_count} 组")
    print(f"全局统计信息: {global_csv}")
    print(f"异常记录文件: {ERROR_LOG_PATH}")


if __name__ == "__main__":
    main()
