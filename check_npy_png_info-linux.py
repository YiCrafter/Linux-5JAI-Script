"""
脚本功能说明：

本脚本用于检查指定文件夹中 .npy 和 .png 文件的基本信息，便于确认图像数据
是否读取正常、尺寸是否一致、数值范围是否合理。

主要功能：
1. 遍历指定 FOLDER 文件夹下的所有 .npy 和 .png 文件；
2. 对 .npy 文件读取其数组信息，包括 shape、维度、数据类型、最小值、最大值和均值；
3. 对 .png 文件使用 cv2.IMREAD_UNCHANGED 方式读取，尽量保留原始通道数和位深信息；
4. 将所有文件的检查结果整理为表格并打印到控制台；
5. 将检查结果保存为 inspect_result.csv，存放在当前检查文件夹下。

输出结果包括：
- 控制台打印的文件检查表；
- inspect_result.csv 文件，记录每个 .npy / .png 文件的格式、shape、ndim、dtype、min、max、mean 等信息。

适用场景：
- 检查预处理后的训练数据是否正常；
- 判断 npy 和 png 的尺寸、通道数、数值范围是否符合预期；
- 快速发现全黑图、异常图、读取失败或数据范围不一致的问题。
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# 1. 修改为你的数据文件夹
# =========================
FOLDER = Path(r"/mnt/bigdata/gongyong/5JAI_dataset/train/GT")


def inspect_npy(file_path: Path):
    arr = np.load(file_path, allow_pickle=False)

    return {
        "file": file_path.name,
        "format": "npy",
        "shape": str(arr.shape),
        "ndim": arr.ndim,
        "dtype": str(arr.dtype),
        "min": float(arr.min()) if arr.size > 0 else None,
        "max": float(arr.max()) if arr.size > 0 else None,
        "mean": float(arr.mean()) if arr.size > 0 else None,
    }


def inspect_png(file_path: Path):
    # 关键：IMREAD_UNCHANGED 可以保留 png 原始位深和通道数
    img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        return {
            "file": file_path.name,
            "format": "png",
            "shape": "read failed",
            "ndim": None,
            "dtype": None,
            "min": None,
            "max": None,
            "mean": None,
        }

    return {
        "file": file_path.name,
        "format": "png",
        "shape": str(img.shape),
        "ndim": img.ndim,
        "dtype": str(img.dtype),
        "min": float(img.min()) if img.size > 0 else None,
        "max": float(img.max()) if img.size > 0 else None,
        "mean": float(img.mean()) if img.size > 0 else None,
    }


def main():
    if not FOLDER.exists():
        raise FileNotFoundError(f"文件夹不存在: {FOLDER}")

    results = []

    files = sorted(list(FOLDER.glob("*.npy")) + list(FOLDER.glob("*.png")))

    if not files:
        print("没有找到 .npy 或 .png 文件")
        return

    for file_path in files:
        if file_path.suffix.lower() == ".npy":
            info = inspect_npy(file_path)
        elif file_path.suffix.lower() == ".png":
            info = inspect_png(file_path)
        else:
            continue

        results.append(info)

    df = pd.DataFrame(results)

    print("\n====== NPY / PNG 文件检查结果 ======\n")
    print(df.to_string(index=False))

    save_csv = FOLDER / "inspect_result.csv"
    df.to_csv(save_csv, index=False, encoding="utf-8-sig")

    print(f"\n结果已保存到: {save_csv}")


if __name__ == "__main__":
    main()
