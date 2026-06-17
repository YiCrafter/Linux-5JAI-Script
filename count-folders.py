"""
脚本功能说明：

本脚本用于统计指定目录下的文件夹数量。

主要功能：
1. 指定一个根目录 ROOT_DIR；
2. 统计该目录下的一级子文件夹数量；
3. 可选择是否递归统计所有层级的子文件夹数量；
4. 可选择是否只统计数字开头的文件夹；
5. 在终端输出统计结果，并列出被统计的文件夹名称。

适用场景：
- 统计 processed_2026-05-22 等目录下有多少组数据；
- 统计某个输出目录下生成了多少个子文件夹；
- 快速检查数据处理前后文件夹数量是否一致。
"""

from pathlib import Path

# =========================================================
# 1. 路径配置
# =========================================================
# ROOT_DIR = Path(r"/mnt/bigdata/ndy-5JAI/00_raw_data/2026-06-08/matched_groups")
ROOT_DIR = Path(r"/mnt/bigdata/ndy-5JAI/dataset-5JAI/test1/DAHUA_Window")

# 是否递归统计所有层级文件夹
# False：只统计 ROOT_DIR 下面的一级子文件夹
# True ：统计 ROOT_DIR 下面所有层级的文件夹
RECURSIVE = False

# 是否只统计数字开头的文件夹
# 例如 0001_205612_106 会被统计，_png_2x 不会被统计
ONLY_DIGIT_PREFIX = True

# 是否打印每个被统计的文件夹名称
PRINT_FOLDER_NAMES = False


# =========================================================
# 2. 工具函数
# =========================================================
def is_digit_prefix_folder(path: Path):
    return path.is_dir() and len(path.name) > 0 and path.name[0].isdigit()


def get_folders(root_dir: Path):
    if not root_dir.exists():
        raise FileNotFoundError(f"指定路径不存在: {root_dir}")

    if not root_dir.is_dir():
        raise NotADirectoryError(f"指定路径不是文件夹: {root_dir}")

    if RECURSIVE:
        folders = [p for p in root_dir.rglob("*") if p.is_dir()]
    else:
        folders = [p for p in root_dir.iterdir() if p.is_dir()]

    if ONLY_DIGIT_PREFIX:
        folders = [p for p in folders if is_digit_prefix_folder(p)]

    folders = sorted(folders, key=lambda x: str(x))

    return folders


# =========================================================
# 3. 主函数
# =========================================================
def main():
    folders = get_folders(ROOT_DIR)

    print(f"统计路径: {ROOT_DIR}")
    print(f"是否递归统计: {RECURSIVE}")
    print(f"是否只统计数字开头文件夹: {ONLY_DIGIT_PREFIX}")
    print(f"文件夹数量: {len(folders)}")

    if PRINT_FOLDER_NAMES:
        print("\n被统计的文件夹：")
        for folder in folders:
            print(folder.name)


if __name__ == "__main__":
    main()
