# NDY-5JAI Dataset Processing Pipeline
## 2026-05-22至2026-06-09 手动代码执行使用脚本 从服务器端构建dataset流程 

本仓库用于管理 **NDY-5JAI 多模态图像数据集** 的数据处理脚本，主要包括原始数据预处理、多模态图像对齐、车窗区域裁剪、数据集格式整理、测试集构建、可视化检查以及大华图像贴回验证等流程。


## 1. 项目目录

```text
linux-script-5JAI/
├── resources/                                      # 标定文件、配置文件、模板等依赖资源
├── tools/                                          # 工具文件或公共辅助模块
├── 01-process_raw_black_demosic_white_undistort-linux.py
├── 02-window_crop_aligned_fast_multigpu.py
├── 03-copy_files_format_after_prepro.py
├── 03-test1copy_5jai_dataset_with_dahua_meta_03-test.py
├── 03-test2move_to_dataset_random_full.py
├── 04-vis_modal_grid_checker_select.py
├── check_npy_png_info-linux.py
├── count-folders.py
├── paste_back_to_dahua.py
├── pinjie-all-car-linux.py
├── pinjie-crop-linux.py
├── run_all_days.sh
├── test-paste-polygon-linux.py
├── 常用命令.txt
├── 数据集记录-06-09.txt
└── README.md
```

---

## 2. 数据处理流程

NDY-5JAI 数据集的标准处理流程如下：

```text
原始采集数据
    ↓
01 原始 Bayer 数据预处理
    ↓
02 多模态图像对齐与车窗区域裁剪
    ↓
03 数据整理并划分 train / val / test
    ↓
04 多模态可视化检查
    ↓
必要时进行大华图像贴回验证
```

对应主要脚本：

```text
01-process_raw_black_demosic_white_undistort-linux.py
02-window_crop_aligned_fast_multigpu.py
03-copy_files_format_after_prepro.py
03-test1copy_5jai_dataset_with_dahua_meta_03-test.py
03-test2move_to_dataset_random_full.py
04-vis_modal_grid_checker_select.py
```

---

## 3. 主要脚本说明

| 脚本                                                      | 功能                                     |
| ------------------------------------------------------- | -------------------------------------- |
| `01-process_raw_black_demosic_white_undistort-linux.py` | 对原始 Bayer 数据进行去黑电平、去马赛克、白平衡和去畸变处理      |
| `02-window_crop_aligned_fast_multigpu.py`               | 对多模态图像进行视角对齐，并裁剪车窗区域                   |
| `03-copy_files_format_after_prepro.py`                  | 将处理后的数据整理为标准数据集格式，并合并到 `train` / `val` |
| `03-test1copy_5jai_dataset_with_dahua_meta_03-test.py`  | 构建 `test1` 数据集，并保留大华图像贴回所需元信息          |
| `03-test2move_to_dataset_random_full.py`                | 随机抽取样本构建 `test2`，剩余数据按规则加入训练集          |
| `04-vis_modal_grid_checker_select.py`                   | 对 `train`、`val`、`test` 中的多模态数据进行可视化检查  |

---

## 4. 辅助脚本说明

| 脚本                            | 功能                                     |
| ----------------------------- | -------------------------------------- |
| `check_npy_png_info-linux.py` | 检查 `.npy` 和 `.png` 文件的尺寸、通道数、数据类型和数值范围 |
| `count-folders.py`            | 统计指定目录下的子文件夹数量                         |
| `paste_back_to_dahua.py`      | 将车窗区域图像贴回到大华全图中，用于结果还原和展示              |
| `pinjie-all-car-linux.py`     | 拼接全车区域的多模态图像，用于可视化对比                   |
| `pinjie-crop-linux.py`        | 拼接车窗区域的多模态图像，用于可视化对比                   |
| `run_all_days.sh`             | 批量执行多天数据处理流程                           |
| `test-paste-polygon-linux.py` | 检查车窗区域贴回大华图像时的四边形位置是否合理                |

---

## 5. 数据目录建议

服务器端建议采用如下数据结构：

```text
/mnt/bigdata/ndy-5JAI/
├── zip/                  # 百度网盘下载的原始压缩包
├── raw/                  # 解压后的原始采集数据
├── processed_daily/       # 每日处理后的中间结果
├── releases/              # 固定版本数据集索引，如 train.txt / val.txt / test.txt
├── metadata/              # 样本信息、数据划分、质量标签、处理记录
└── checkpoints/           # 模型权重，不上传 GitHub
```

推荐原则：

* 原始数据只新增，不覆盖；
* 每日数据按日期独立保存；
* 训练集、验证集和测试集尽量通过索引文件管理；
* 不同补光灯、不同路口、不同测试场景使用不同的 `test_xxx.txt` 管理；
* 正式实验前固定一个 dataset release，便于复现。

---

## 6. 使用方式

### 6.1 单日数据处理

处理单日数据时，建议按顺序执行：

```text
1. 运行 01 脚本，完成原始 Bayer 数据预处理
2. 运行 02 脚本，完成多模态图像对齐和车窗裁剪
3. 运行对应的 03 脚本，整理为标准数据集格式
4. 运行 04 脚本，进行多模态可视化检查
5. 如有需要，运行贴回脚本验证大华图像还原效果
```

### 6.2 多日数据批量处理

可使用：

```bash
bash run_all_days.sh
```

执行前请确认：

* 输入日期范围正确；
* 输入和输出路径正确；
* `01` 预处理结果已经生成；
* GPU 设置符合当前服务器环境；
* 目标数据集目录不会覆盖已有结果。

---

## 7. 数据集构建记录

`数据集记录-6-11.txt` 用于记录数据集构建过程，包括：

* 每日数据数量；
* 每日数据划分方式；
* `train` / `val` / `test1` / `test2` 样本数量；
* 测试集随机抽取规则；
* 特殊日期或特殊处理说明。

每次新增数据、修改划分方式或调整处理流程后，建议同步更新该记录文件。

---

## 8. 注意事项

1. 执行脚本前，请先检查脚本开头的路径配置、参数说明和使用方法。
2. `01`、`02`、`03`、`04` 脚本之间存在顺序依赖，不建议跳步执行。
3. `resources/` 和 `tools/` 中的依赖文件不要随意删除、重命名或移动。
4. 原始数据目录不建议手动修改。
5. 构建测试集时，需要严格避免测试样本混入训练集。
6. 修改数据划分规则后，应同步更新数据集记录文档。
7. GitHub 仓库中不要提交大规模数据、模型权重、日志文件和临时输出结果。
8. 如果脚本功能发生变化，应同步更新脚本开头说明和本文档。

---

## 9. 维护规范

| 操作     | 建议                                                              |
| ------ | --------------------------------------------------------------- |
| 新增脚本   | 在 README 或脚本说明文档中补充用途                                           |
| 修改脚本功能 | 同步更新脚本开头说明                                                      |
| 新增数据日期 | 更新数据集记录文件                                                       |
| 修改数据划分 | 记录修改原因、样本数量和划分规则                                                |
| 删除旧脚本  | 确认无依赖后再删除，必要时先移入 `archive/`                                     |
| 新增测试集  | 使用明确命名，如 `test_no_light`、`test_strong_light`、`test_rainy_night` |

---
