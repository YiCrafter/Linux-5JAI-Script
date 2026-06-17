"""
脚本功能说明：

本脚本用于批量处理多相机采集数据。程序会遍历输入根目录下的各个子文件夹，
对每组数据中的 GT、RGB、RGB-Long、NIR、NIR-Long 图像进行预处理，并将结果
保存到对应的输出子文件夹中。

主要处理流程如下：
1. 对 GT、RGB、RGB-Long 三类 Bayer 图像执行：
   减黑电平 → Bayer 去马赛克 → 灰度世界白平衡及额外 RGB 权重校正 → 去畸变 → 保存 npy(归一化) 和 png。

2. 对 NIR、NIR-Long 两类红外图像执行：
   减黑电平 → 去畸变 → 固定白电平归一化到 0~1 → 保存 npy 和 png。

3. 对每个子文件夹中的 dahua.jpg 执行：
   转换为 dahua.png，并保存到对应输出子文件夹中。

输出结果包括：
- 训练使用的 float32 格式 .npy 文件，数值范围为 0~1；
- 便于人工查看的 .png 可视化图像；
- 由 dahua.jpg 转换得到的 dahua.png 图像。

注意：
- 输入图像默认是 uint8 类型，原始范围为 0~255；
- 去畸变参数从当前脚本同级目录下的 01-undistort_params.json 中读取；
- 黑电平文件需要放置在 BLACK_LEVEL_DIR 指定目录下；
- 脚本会保留不同图像之间的真实亮度关系，不使用单张图像自适应最大值归一化。
"""

import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


def log(msg=""):
    tqdm.write(msg)


# =========================
# 1. 路径配置
# =========================
# 原始数据根目录：里面包含很多子文件夹，例如 0001_205806_247
# INPUT_ROOT = Path(r"/mnt/bigdata/ndy-5JAI/02_unzip_daily/2026-06-13/matched_groups")
INPUT_ROOT = Path(
    r"/mnt/bigdata/ndy-5JAI/99_Long_10ms_g80/02_unzip_daily/2026-06-14/matched_groups"
)  # 新参数（gain; exporsure） 数据路径

# 黑电平文件夹
BLACK_LEVEL_DIR = Path(
    r"/mnt/bigdata/dongyang/linux-script-5JAI/resources/blc/blc_gain20"
)

# 最终输出文件夹
# OUTPUT_ROOT = Path(r"/mnt/bigdata/ndy-5JAI/03_processed_daily/processed_2026-06-13")
OUTPUT_ROOT = Path(
    r"/mnt/bigdata/ndy-5JAI/99_Long_10ms_g80/03_processed_daily/processed_2026-06-14"
)  # 新参数（gain; exporsure） 数据路径

# 去畸变参数文件：建议和当前 py 文件放在同一个目录
# UNDISTORT_PARAM_PATH = Path(__file__).with_name("resources/01-undistort_params.json")
UNDISTORT_PARAM_PATH = Path(
    r"/mnt/bigdata/dongyang/linux-script-5JAI/resources/01-undistort_params.json"
)


# =========================
# 2. 数据范围配置
# =========================
# 原始 npy 是 uint8，范围 0~255，所以这里用 255
RAW_WHITE_LEVEL = 255.0


# =========================
# 3. 文件配置
# =========================
# 三个 Bayer RGB / GT 图：减黑电平 + 去马赛克 + 白平衡 + 去畸变
COLOR_BAYER_FILES = [
    "GT_Camera_usb.npy",
    "RGB_Camera_249.npy",
    "RGB_Long_Camera_247.npy",
]

# 两个 NIR 图：减黑电平 + 去畸变 + 归一化
NIR_FILES = [
    "Nir_Camera_usb.npy",
    "NIR_Long_Camera_248.npy",
]

# 原始图像文件名 -> 黑电平文件名
BLACK_LEVEL_FILE_MAP = {
    "GT_Camera_usb.npy": "GT_Camera-usb_black_level.npy",
    "Nir_Camera_usb.npy": "NIR_Camera_black_level.npy",
    "NIR_Long_Camera_248.npy": "Nir-Long_Camera-usb_black_level.npy",
    "RGB_Camera_249.npy": "RGB_Camera_black_level.npy",
    "RGB_Long_Camera_247.npy": "RGB-Long_Camera_black_level.npy",
}


# =========================
# 4. 去畸变参数读取与调用
# =========================
def load_undistort_params(json_path):
    """
    读取简化后的去畸变参数文件。

    JSON 格式：
    {
      "image_size": [2448, 2048],
      "files": {
        "GT_Camera_usb.npy": {
          "camera_matrix": [[...], [...], [...]],
          "distortion_coeffs": [k1, k2, p1, p2, k3]
        }
      }
    }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"去畸变参数文件不存在: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        params = json.load(f)

    if "files" not in params:
        raise ValueError("去畸变参数文件缺少字段: files")

    return params


_UNDISTORT_MAP_CACHE = {}


def undistort_image(img, filename, undistort_params):
    """
    按 filename 从参数文件中取 camera_matrix 和 distortion_coeffs，
    对 img 做去畸变，输出尺寸、通道数和输入保持一致。

    img 可以是：
        H x W      float32 灰度图，例如 NIR 减黑电平后的图
        H x W x 3  float32 RGB 图，例如白平衡后的 GT/RGB 图
    """
    file_params = undistort_params.get("files", {})
    if filename not in file_params:
        raise KeyError(f"去畸变参数中没有找到当前文件: {filename}")

    img = img.astype(np.float32)

    if img.ndim == 2:
        h, w = img.shape
    elif img.ndim == 3:
        h, w = img.shape[:2]
    else:
        raise ValueError(f"去畸变输入只支持 HxW 或 HxWxC，当前 shape={img.shape}")

    expected_size = undistort_params.get("image_size", None)
    if expected_size is not None:
        expected_w, expected_h = expected_size
        if (w, h) != (expected_w, expected_h):
            raise ValueError(
                f"图像尺寸和去畸变参数不一致: {filename}, "
                f"img_size={(w, h)}, expected={(expected_w, expected_h)}"
            )

    key = (filename, w, h)
    if key not in _UNDISTORT_MAP_CACHE:
        camera_matrix = np.array(
            file_params[filename]["camera_matrix"], dtype=np.float64
        )
        distortion_coeffs = np.array(
            file_params[filename]["distortion_coeffs"], dtype=np.float64
        ).reshape(-1)

        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coeffs,
            R=None,
            newCameraMatrix=camera_matrix,
            size=(w, h),
            m1type=cv2.CV_32FC1,
        )
        _UNDISTORT_MAP_CACHE[key] = (map1, map2)

    map1, map2 = _UNDISTORT_MAP_CACHE[key]
    undistorted = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return undistorted.astype(np.float32)


# =========================
# 5. 查找文件，忽略大小写
# =========================
def find_file_ignore_case(folder, filename):
    """
    在 folder 中查找 filename。
    先按原文件名查找；如果找不到，再忽略大小写查找。
    注意：这里只忽略大小写，不会自动处理 _ 和 - 的差异。
    """
    path = folder / filename
    if path.exists():
        return path

    target = filename.lower()
    for p in folder.iterdir():
        if p.name.lower() == target:
            return p

    raise FileNotFoundError(f"没有找到文件: {filename}，所在目录: {folder}")


# =========================
# 6. 减黑电平
# =========================
def subtract_black_level(raw, black_level, white_level=255.0):
    """
    输入:
        raw: 原始 raw 图像，H x W，通常是 uint8，范围 0~255
        black_level: 黑电平图像，H x W，float32
        white_level: 原始图像最大有效值，uint8 用 255

    输出:
        corrected: float32，范围 0~white_level
    """
    raw = raw.astype(np.float32)
    black_level = black_level.astype(np.float32)

    if raw.ndim != 2:
        raise ValueError(f"原始图像应该是二维 raw 图，但当前 shape={raw.shape}")

    if raw.shape != black_level.shape:
        raise ValueError(
            f"原始图像和黑电平尺寸不一致: raw={raw.shape}, black_level={black_level.shape}"
        )

    corrected = raw - black_level
    corrected = np.clip(corrected, 0.0, white_level)

    return corrected.astype(np.float32)


# =========================
# 7. float raw 转成 0~1
# =========================
def raw_float_to_float01(raw_float, white_level=255.0):
    """
    把减黑电平后的 raw float32 转成 0~1。
    不使用每张图自己的 max，而是固定除以 white_level。
    """
    img = raw_float.astype(np.float32) / float(white_level)
    img = np.clip(img, 0.0, 1.0)
    return img.astype(np.float32)


# =========================
# 8. 减黑后的 float raw 转 uint16 给 OpenCV 去马赛克
# =========================
def raw_float_to_uint16_for_demosaic(raw_float, white_level=255.0):
    """
    OpenCV 的 Bayer 去马赛克一般需要 uint8 或 uint16。
    这里把减黑电平后的 float32 raw 先归一化，再映射到 uint16。

    这样比直接转 uint8 更好，因为可以尽量保留减黑电平后的小数信息。
    """
    raw01 = raw_float_to_float01(raw_float, white_level=white_level)
    raw_u16 = np.round(raw01 * 65535.0).astype(np.uint16)
    return raw_u16


# =========================
# 9. RGGB Bayer 去马赛克
# =========================
def demosaic_rggb_from_corrected_raw(raw_corrected, white_level=255.0):
    """
    输入:
        raw_corrected: 已减黑电平的 Bayer raw，H x W，float32，范围 0~white_level

    输出:
        rgb: H x W x 3，float32，范围 0~1
    """
    if raw_corrected.ndim != 2:
        raise ValueError(
            f"输入应该是单通道 Bayer 图像，但当前 shape={raw_corrected.shape}"
        )

    raw_u16 = raw_float_to_uint16_for_demosaic(raw_corrected, white_level=white_level)

    # 保持你之前脚本的写法：RGGB Bayer -> RGB
    rgb_u16 = cv2.cvtColor(raw_u16, cv2.COLOR_BayerBG2RGB)

    rgb = rgb_u16.astype(np.float32) / 65535.0
    rgb = np.clip(rgb, 0.0, 1.0)

    return rgb.astype(np.float32)


# =========================
# 10. 灰度世界白平衡
# =========================
def gray_world_white_balance(rgb):
    """
    灰度世界白平衡。
    排除太暗和过曝区域，避免黑背景和饱和区域影响白平衡估计。
    """
    rgb = rgb.astype(np.float32)

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"白平衡输入应该是 H x W x 3 RGB 图，但当前 shape={rgb.shape}")

    intensity = rgb.mean(axis=2)

    valid_mask = (intensity > 0.03) & (intensity < 0.95)

    if valid_mask.sum() < 1000:
        valid_mask = np.ones(intensity.shape, dtype=bool)

    r_mean = rgb[:, :, 0][valid_mask].mean()
    g_mean = rgb[:, :, 1][valid_mask].mean()
    b_mean = rgb[:, :, 2][valid_mask].mean()

    channel_means = np.array([r_mean, g_mean, b_mean], dtype=np.float32)
    gray_mean = channel_means.mean()

    gains = gray_mean / np.maximum(channel_means, 1e-6)

    # 在原本计算出的白平衡 gains 基础上，再乘你手动给定的 RGB 权重
    extra_gains = np.array([0.960606, 1.0, 1.297619], dtype=np.float32)
    final_gains = gains * extra_gains

    rgb_wb = rgb * final_gains.reshape(1, 1, 3)
    rgb_wb = np.clip(rgb_wb, 0.0, 1.0)

    return rgb_wb.astype(np.float32), gains


# =========================
# 11. 保存 npy 和 png
# =========================
def save_npy_and_png(save_dir, name, img_float):
    """
    img_float:
        灰度图: H x W，float32，0~1
        RGB图 : H x W x 3，float32，0~1

    保存:
        .npy: float32，范围 0~1
        .png: uint8，范围 0~255，仅用于可视化
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(name).stem

    npy_path = save_dir / f"{stem}.npy"
    png_path = save_dir / f"{stem}.png"

    img_float = np.clip(img_float.astype(np.float32), 0.0, 1.0)

    np.save(npy_path, img_float)

    png_img = np.round(img_float * 255.0).astype(np.uint8)

    if png_img.ndim == 3 and png_img.shape[2] == 3:
        # npy 是 RGB，OpenCV 保存 png 需要 BGR
        png_img = cv2.cvtColor(png_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(str(png_path), png_img)

    log(
        f"    saved npy: {npy_path.name}, shape={img_float.shape}, dtype={img_float.dtype}, "
        f"min={img_float.min():.6f}, max={img_float.max():.6f}, mean={img_float.mean():.6f}"
    )
    log(f"    saved png: {png_path.name}")


def convert_dahua_jpg_to_png(folder, out_dir):
    """
    将当前子文件夹下的 dahua.jpg 转为 dahua.png，
    保存到对应输出子文件夹 out_dir 中。
    """
    dahua_jpg_path = folder / "dahua.jpg"

    if not dahua_jpg_path.exists():
        log(f"    跳过，dahua.jpg 不存在: {dahua_jpg_path}")
        return

    img = cv2.imread(str(dahua_jpg_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        log(f"    跳过，dahua.jpg 读取失败: {dahua_jpg_path}")
        return

    dahua_png_path = out_dir / "dahua.png"

    ok = cv2.imwrite(str(dahua_png_path), img)

    if not ok:
        raise ValueError(f"dahua.png 保存失败: {dahua_png_path}")

    log(f"    saved png: {dahua_png_path.name}, shape={img.shape}, dtype={img.dtype}")


# =========================
# 12. 处理单个子文件夹
# =========================
def process_one_folder(folder, undistort_params, pbar=None):
    log(f"\n处理文件夹: {folder.name}")

    out_dir = OUTPUT_ROOT / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 额外处理 dahua.jpg：转为 dahua.png 并保存到对应输出子文件夹
    convert_dahua_jpg_to_png(folder, out_dir)

    # -------------------------
    # 处理 RGB / GT Bayer 图
    # 流程：减黑电平 -> 去马赛克 -> 白平衡 -> 去畸变 -> 保存
    # -------------------------
    for filename in COLOR_BAYER_FILES:
        log("\n------------------------------")
        log(f"处理 Bayer 图像: {filename}")

        raw_path = folder / filename

        if not raw_path.exists():
            log(f"    跳过，原始图像不存在: {raw_path}")
            continue

        black_filename = BLACK_LEVEL_FILE_MAP[filename]
        black_path = find_file_ignore_case(BLACK_LEVEL_DIR, black_filename)

        raw = np.load(raw_path)
        black_level = np.load(black_path)

        log(
            f"    raw: shape={raw.shape}, dtype={raw.dtype}, "
            f"min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}"
        )

        log(
            f"    black_level: shape={black_level.shape}, dtype={black_level.dtype}, "
            f"min={black_level.min():.4f}, max={black_level.max():.4f}, mean={black_level.mean():.4f}"
        )

        # 1. 减黑电平，输出 float32，范围 0~255
        raw_corrected = subtract_black_level(
            raw, black_level, white_level=RAW_WHITE_LEVEL
        )

        log(
            f"    corrected raw: dtype={raw_corrected.dtype}, "
            f"min={raw_corrected.min():.4f}, max={raw_corrected.max():.4f}, mean={raw_corrected.mean():.4f}"
        )

        # 2. 去马赛克，输出 RGB float32，范围 0~1
        rgb = demosaic_rggb_from_corrected_raw(
            raw_corrected, white_level=RAW_WHITE_LEVEL
        )

        # 3. 白平衡，输出 RGB float32，范围 0~1
        rgb_wb, gains = gray_world_white_balance(rgb)

        # 4. 去畸变，输出 RGB float32，范围仍按 0~1 处理
        rgb_wb_undistorted = undistort_image(
            rgb_wb, filename=filename, undistort_params=undistort_params
        )
        rgb_wb_undistorted = np.clip(rgb_wb_undistorted, 0.0, 1.0).astype(np.float32)

        log(
            f"    undistorted RGB: dtype={rgb_wb_undistorted.dtype}, "
            f"min={rgb_wb_undistorted.min():.4f}, max={rgb_wb_undistorted.max():.4f}, "
            f"mean={rgb_wb_undistorted.mean():.4f}"
        )

        # 5. 保存最终结果，保存格式不变
        save_npy_and_png(out_dir, filename, rgb_wb_undistorted)

        log(
            f"    white balance gains: R={gains[0]:.4f}, G={gains[1]:.4f}, B={gains[2]:.4f}"
        )

        if pbar is not None:
            pbar.update(1)

    # -------------------------
    # 处理 NIR 图
    # 流程：减黑电平 -> 去畸变 -> 归一化 -> 保存
    # -------------------------
    for filename in NIR_FILES:
        log("\n------------------------------")
        log(f"处理 NIR 图像: {filename}")

        raw_path = folder / filename

        if not raw_path.exists():
            log(f"    跳过，原始图像不存在: {raw_path}")
            continue

        black_filename = BLACK_LEVEL_FILE_MAP[filename]
        black_path = find_file_ignore_case(BLACK_LEVEL_DIR, black_filename)

        raw = np.load(raw_path)
        black_level = np.load(black_path)

        log(
            f"    raw: shape={raw.shape}, dtype={raw.dtype}, "
            f"min={raw.min():.4f}, max={raw.max():.4f}, mean={raw.mean():.4f}"
        )

        log(
            f"    black_level: shape={black_level.shape}, dtype={black_level.dtype}, "
            f"min={black_level.min():.4f}, max={black_level.max():.4f}, mean={black_level.mean():.4f}"
        )

        # 1. 减黑电平，输出 float32，范围 0~255
        nir_corrected = subtract_black_level(
            raw, black_level, white_level=RAW_WHITE_LEVEL
        )

        log(
            f"    corrected NIR: dtype={nir_corrected.dtype}, "
            f"min={nir_corrected.min():.4f}, max={nir_corrected.max():.4f}, mean={nir_corrected.mean():.4f}"
        )

        # 2. 去畸变，仍保持 float32，范围按 0~255 处理
        nir_corrected_undistorted = undistort_image(
            nir_corrected, filename=filename, undistort_params=undistort_params
        )
        nir_corrected_undistorted = np.clip(
            nir_corrected_undistorted, 0.0, RAW_WHITE_LEVEL
        ).astype(np.float32)

        log(
            f"    undistorted NIR: dtype={nir_corrected_undistorted.dtype}, "
            f"min={nir_corrected_undistorted.min():.4f}, "
            f"max={nir_corrected_undistorted.max():.4f}, "
            f"mean={nir_corrected_undistorted.mean():.4f}"
        )

        # 3. 固定除以 255，转成 float32，范围 0~1
        nir_float = raw_float_to_float01(
            nir_corrected_undistorted, white_level=RAW_WHITE_LEVEL
        )

        # 4. 保存最终结果，保存格式不变
        save_npy_and_png(out_dir, filename, nir_float)

        if pbar is not None:
            pbar.update(1)


# =========================
# 13. 主程序
# =========================
def main():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"输入根目录不存在: {INPUT_ROOT}")

    if not BLACK_LEVEL_DIR.exists():
        raise FileNotFoundError(f"黑电平目录不存在: {BLACK_LEVEL_DIR}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    undistort_params = load_undistort_params(UNDISTORT_PARAM_PATH)

    # 如果 INPUT_ROOT 下面直接就是一组图像，也可以处理；
    # 如果 INPUT_ROOT 下面是多个子文件夹，就逐个处理子文件夹。
    direct_has_files = any(
        (INPUT_ROOT / f).exists() for f in COLOR_BAYER_FILES + NIR_FILES
    )

    if direct_has_files:
        folders = [INPUT_ROOT]
    else:
        folders = sorted([p for p in INPUT_ROOT.iterdir() if p.is_dir()])

    if len(folders) == 0:
        print("没有找到需要处理的文件夹")
        return

    print(f"输入目录: {INPUT_ROOT}")
    print(f"黑电平目录: {BLACK_LEVEL_DIR}")
    print(f"输出目录: {OUTPUT_ROOT}")
    print(f"去畸变参数: {UNDISTORT_PARAM_PATH}")
    print(f"共找到 {len(folders)} 个待处理文件夹")

    total_tasks = 0
    for folder in folders:
        for filename in COLOR_BAYER_FILES + NIR_FILES:
            if (folder / filename).exists():
                total_tasks += 1

    log(f"总共需要处理 {total_tasks} 张图像")

    with tqdm(total=total_tasks, desc="总处理进度", unit="张", ncols=100) as pbar:
        for folder in folders:
            process_one_folder(folder, undistort_params=undistort_params, pbar=pbar)

    print("\n全部处理完成")


if __name__ == "__main__":
    main()
