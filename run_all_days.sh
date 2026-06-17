#!/bin/bash
# chmod +x run_all_days.sh
# ./run_all_days.sh
# =========================================================
# 脚本功能说明：
# 本脚本用于按日期顺序批量执行车窗对齐裁剪脚本。
# 每次只处理一天的数据，当前日期处理完成后，再继续处理下一天。
# =========================================================

# 你的 Python 处理脚本路径
PY_SCRIPT="/mnt/bigdata/ndy-5JAI/00_linux-script-5JAI/02-window_crop_aligned_fast_multigpu.py"

# 数据总目录，也就是 processed_2026-05-22 等文件夹的上一级目录
BASE_DIR="/mnt/bigdata/ndy-5JAI/03_processed_daily"

# 输出文件夹名称，和 Python 脚本里的 --output_name 对应
OUTPUT_NAME="_window_crop"

# 日志保存目录
LOG_DIR="${BASE_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

# 需要处理的日期列表
DAYS=(
  "2026-06-12"
  "2026-06-13"
)

echo "=================================================="
echo "开始批量处理"
echo "Python脚本: ${PY_SCRIPT}"
echo "数据总目录: ${BASE_DIR}"
echo "日志目录: ${LOG_DIR}"
echo "=================================================="


for DAY in "${DAYS[@]}"; do
    ROOT_DIR="${BASE_DIR}/processed_${DAY}"
    LOG_FILE="${LOG_DIR}/${DAY}.log"

    echo ""
    echo "=================================================="
    echo "开始处理日期: ${DAY}"
    echo "输入目录: ${ROOT_DIR}"
    echo "日志文件: ${LOG_FILE}"
    echo "=================================================="

    if [ ! -d "${ROOT_DIR}" ]; then
        echo "目录不存在，跳过: ${ROOT_DIR}" | tee -a "${LOG_FILE}"
        continue
    fi

    python "${PY_SCRIPT}" \
        --root_dir "${ROOT_DIR}" \
        --output_name "${OUTPUT_NAME}" \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo ""
        echo "处理失败: ${DAY}"
        echo "退出码: ${EXIT_CODE}"
        echo "请查看日志: ${LOG_FILE}"
        echo "已停止后续日期处理。"
        exit ${EXIT_CODE}
    fi

    echo ""
    echo "完成处理日期: ${DAY}"
    echo "日志保存到: ${LOG_FILE}"
done

echo ""
echo "=================================================="
echo "全部日期处理完成"
echo "=================================================="