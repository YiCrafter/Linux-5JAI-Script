"""
脚本功能说明：

本脚本用于批量生成多模态 PNG 图像的拼接预览图，并可选择生成
GT 与其他模态图像之间的棋盘格对齐检查图，方便快速查看每组数据
的图像完整性、模态内容差异以及空间对齐效果。

注意事项：
- 本脚本只处理 PNG 图片，不读取或处理 npy 文件；
- 普通拼接图主要用于检查每组数据中各模态图片是否完整生成；
- 棋盘格图主要用于检查 GT 与其他模态之间的空间对齐效果；
- 如果不需要生成棋盘格检查图，可将 GENERATE_CHECKERBOARD 设置为 False；
- 若子文件夹中存在额外 PNG 图片，脚本会在固定顺序图片之后按文件名排序追加显示。
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# =========================================================
# 1. 路径配置
# =========================================================
ROOT_DIR = Path(r"D:\company-file-5-22\output-test\_window_crop_aligned_keep_format-dahua-cor")

# 要贴回的图片
# 测试贴回效果用 dahua_window.png
# 模型输出可以改成 fake_B.png / pred.png
PASTE_IMAGE_NAME = "NIR_Camera_usb.png"
# PASTE_IMAGE_NAME = "dahua_window.png"

# 输出贴回后的大华全图
OUTPUT_IMAGE_NAME = "dahua_paste_back.png"

# 指定处理某一个子文件夹；批量处理则改成 None
# ONLY_FOLDER_NAME = "0001_205445_965"
ONLY_FOLDER_NAME = None

# 如果贴回图尺寸和 meta 记录尺寸不一致，是否自动 resize
AUTO_RESIZE_TO_META_SIZE = True


# =========================================================
# 2. 贴回自然融合配置
# =========================================================
# 是否启用边缘自然融合
FEATHER_EDGE = True

# 贴回区域向内收缩像素，避免 crop 四边线框被贴回去
# 你现在有明显边框，建议先用 20~30
MASK_INNER_MARGIN = 25

# 羽化半径，越大边缘越柔和
# 建议 25~45
FEATHER_RADIUS = 35

# 是否保存 mask 调试图
SAVE_DEBUG_MASK = False


# =========================================================
# 3. 图像读取工具
# =========================================================
def to_uint8_plane(x):
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if x.size == 0:
        return x.astype(np.uint8)

    min_v = float(np.min(x))
    max_v = float(np.max(x))

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


def read_paste_image_as_bgr(image_path: Path):
    """
    支持读取 png/jpg/bmp 以及 npy。
    返回：
        img_bgr: BGR 三通道 uint8 图像
        alpha: 如果是带透明通道的 png，则返回 alpha；否则返回 None
    """
    suffix = image_path.suffix.lower()

    if suffix == ".npy":
        arr = np.load(image_path, allow_pickle=False)

        if arr.ndim == 2:
            gray = to_uint8_plane(arr)
            img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return img_bgr, None

        if arr.ndim == 3 and arr.shape[2] == 1:
            gray = to_uint8_plane(arr[:, :, 0])
            img_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return img_bgr, None

        if arr.ndim == 3 and arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
            r = to_uint8_plane(rgb[:, :, 0])
            g = to_uint8_plane(rgb[:, :, 1])
            b = to_uint8_plane(rgb[:, :, 2])
            img_rgb = np.stack([r, g, b], axis=-1)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            return img_bgr, None

        raise ValueError(f"不支持的 npy 图像 shape: {arr.shape}")

    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"读取失败: {image_path}")

    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img_bgr, None

    if img.ndim == 3 and img.shape[2] == 3:
        return img, None

    if img.ndim == 3 and img.shape[2] == 4:
        img_bgr = img[:, :, :3]
        alpha = img[:, :, 3]
        return img_bgr, alpha

    raise ValueError(f"不支持的图像格式或通道数: {image_path}, shape={img.shape}")


def read_json(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# 4. mask 和 alpha 融合工具
# =========================================================
def make_inner_crop_mask(crop_h, crop_w, margin):
    """
    生成向内收缩后的 crop mask。
    不贴 crop 最外圈，避免贴回后出现四边线框。
    """
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)

    margin = int(margin)

    if margin <= 0:
        mask[:, :] = 255
        return mask

    x1 = margin
    y1 = margin
    x2 = crop_w - margin
    y2 = crop_h - margin

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"MASK_INNER_MARGIN={margin} 太大，超过 crop 尺寸: {(crop_h, crop_w)}"
        )

    mask[y1:y2, x1:x2] = 255
    return mask


def make_feather_alpha(mask_uint8, feather_radius):
    """
    基于距离变换生成更自然的 alpha。
    边缘从 0 平滑过渡到 1，比单纯 GaussianBlur 更稳。
    """
    mask = (mask_uint8 > 0).astype(np.uint8)

    if feather_radius <= 0:
        return mask.astype(np.float32)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    alpha = dist / float(feather_radius)
    alpha = np.clip(alpha, 0.0, 1.0)

    # 再轻微平滑一下，减少锯齿
    k = max(3, int(feather_radius // 2) * 2 + 1)
    alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    alpha = np.clip(alpha, 0.0, 1.0)

    return alpha.astype(np.float32)


# =========================================================
# 5. 单个文件夹贴回
# =========================================================
def paste_back_one_folder(folder: Path):
    meta_path = folder / "_dahua_paste_back_meta.json"
    dahua_path = folder / "dahua.png"
    paste_image_path = folder / PASTE_IMAGE_NAME

    if not meta_path.exists():
        raise FileNotFoundError(f"缺失 meta 文件: {meta_path}")

    if not dahua_path.exists():
        raise FileNotFoundError(f"缺失原始大华图: {dahua_path}")

    if not paste_image_path.exists():
        raise FileNotFoundError(f"缺失指定贴回图: {paste_image_path}")

    meta = read_json(meta_path)

    original = cv2.imread(str(dahua_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"读取原始大华图失败: {dahua_path}")

    original_h, original_w = original.shape[:2]

    paste_img, paste_alpha = read_paste_image_as_bgr(paste_image_path)

    # meta 中记录的是 dahua_window.png 的 crop 尺寸
    crop_h, crop_w = meta["window_crop_size_hw"]

    if paste_img.shape[:2] != (crop_h, crop_w):
        if AUTO_RESIZE_TO_META_SIZE:
            paste_img = cv2.resize(
                paste_img,
                (crop_w, crop_h),
                interpolation=cv2.INTER_LINEAR
            )

            if paste_alpha is not None:
                paste_alpha = cv2.resize(
                    paste_alpha,
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_LINEAR
                )
        else:
            raise ValueError(
                f"贴回图尺寸不一致: 当前 {paste_img.shape[:2]}, "
                f"meta 需要 {(crop_h, crop_w)}"
            )

    M = np.array(meta["matrix_crop_to_original_dahua"], dtype=np.float64)

    # -----------------------------------------------------
    # 5.1 将指定车窗图 warp 回原始 dahua 全图坐标
    # -----------------------------------------------------
    # BORDER_REPLICATE 可以减少边界黑色参与插值导致的边框问题
    warped_paste = cv2.warpPerspective(
        paste_img,
        M,
        (original_w, original_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # -----------------------------------------------------
    # 5.2 生成向内收缩后的 crop mask，再 warp 回原图坐标
    # -----------------------------------------------------
    crop_mask = make_inner_crop_mask(
        crop_h=crop_h,
        crop_w=crop_w,
        margin=MASK_INNER_MARGIN,
    )

    warped_mask = cv2.warpPerspective(
        crop_mask,
        M,
        (original_w, original_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    # 如果贴回图本身有 alpha 通道，则叠加 alpha
    if paste_alpha is not None:
        warped_alpha = cv2.warpPerspective(
            paste_alpha,
            M,
            (original_w, original_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = np.minimum(warped_mask, warped_alpha)

    # -----------------------------------------------------
    # 5.3 自然融合
    # -----------------------------------------------------
    if FEATHER_EDGE:
        alpha = make_feather_alpha(
            mask_uint8=warped_mask,
            feather_radius=FEATHER_RADIUS,
        )
        alpha = alpha[:, :, None]

        result = (
            warped_paste.astype(np.float32) * alpha
            + original.astype(np.float32) * (1.0 - alpha)
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

    else:
        result = original.copy()
        mask_bool = warped_mask > 0
        result[mask_bool] = warped_paste[mask_bool]

    save_path = folder / OUTPUT_IMAGE_NAME
    ok = cv2.imwrite(str(save_path), result)

    if not ok:
        raise ValueError(f"保存失败: {save_path}")

    if SAVE_DEBUG_MASK:
        debug_mask_path = folder / "_paste_back_mask.png"
        cv2.imwrite(str(debug_mask_path), warped_mask)

    return save_path


# =========================================================
# 6. 获取待处理文件夹
# =========================================================
def get_folders_to_process():
    """
    支持两种情况：
    1. ROOT_DIR 是总目录，里面有很多子文件夹；
    2. ROOT_DIR 本身就是一个具体子文件夹。
    """
    if (ROOT_DIR / "_dahua_paste_back_meta.json").exists():
        return [ROOT_DIR]

    if ONLY_FOLDER_NAME is not None:
        return [ROOT_DIR / ONLY_FOLDER_NAME]

    return sorted([
        p for p in ROOT_DIR.iterdir()
        if p.is_dir()
    ])


# =========================================================
# 7. 主函数
# =========================================================
def main():
    folders = get_folders_to_process()

    if len(folders) == 0:
        print("没有找到待处理文件夹，请检查 ROOT_DIR。")
        return

    success = 0
    fail = 0

    for folder in tqdm(folders, desc="贴回进度", unit="组", ncols=100):
        try:
            save_path = paste_back_one_folder(folder)
            success += 1
            print(f"保存成功: {save_path}")
        except Exception as e:
            fail += 1
            print(f"处理失败: {folder.name}, 原因: {e}")

    print("\n贴回完成")
    print(f"成功: {success} 组")
    print(f"失败: {fail} 组")


if __name__ == "__main__":
    main()