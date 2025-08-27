#!/bin/bash

# =============================================================================
# VSCode Extension 入口脚本
# =============================================================================

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 检查参数数量（不再接收Few-shot文件，全部从config.sh读取）
if [ $# -lt 2 ]; then
    echo "错误: 参数不足"
    echo "用法: $0 <算子名称> <源码路径...>"
    echo "示例: $0 AllGatherMatmul /path/to/source1 /path/to/source2"
    exit 1
fi

# 获取算子名称
OPERATOR_NAME="$1"
shift

# 剩余参数都是源码路径
SOURCE_PATHS=("$@")

# 尝试激活虚拟环境
VENV_FOUND=false

# 优先使用脚本所在目录的虚拟环境
if [ -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo "🔄 激活项目虚拟环境: $SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    VENV_FOUND=true
# 尝试父目录的虚拟环境（如果脚本在子目录中）
elif [ -f "$SCRIPT_DIR/../.venv/bin/activate" ]; then
    echo "🔄 激活父目录虚拟环境: $SCRIPT_DIR/../.venv"
    source "$SCRIPT_DIR/../.venv/bin/activate"
    VENV_FOUND=true
# 检查当前工作目录的虚拟环境
elif [ -f "$(pwd)/.venv/bin/activate" ]; then
    echo "🔄 激活当前目录虚拟环境: $(pwd)/.venv"
    source "$(pwd)/.venv/bin/activate"
    VENV_FOUND=true
fi

if [ "$VENV_FOUND" = true ]; then
    echo "✅ 虚拟环境已激活"
    echo "📍 Python: $(which python)"
    echo "📦 Pip: $(which pip)"
else
    echo "⚠️  未找到虚拟环境，使用系统Python"
    echo "📍 Python: $(which python3 || which python)"
fi

# 加载config.sh中的配置
source "$SCRIPT_DIR/config.sh"

# 执行完整流程
echo "🚀 开始执行测试用例生成流程"
echo "算子名称: $OPERATOR_NAME"
echo "源码路径: ${SOURCE_PATHS[@]}"
echo "API配置: ${BASE_URL} (${MODEL_NAME})"
echo "Few-shot: ${FEWSHOT_EXAMPLES_FILE}"
echo "=================================="
exit -1
# 调用 workflow.sh 执行完整流程
cd "$SCRIPT_DIR"
./workflow.sh gen-all "$OPERATOR_NAME" "${SOURCE_PATHS[@]}"

