#!/bin/bash

# 激活虚拟环境脚本
# 用法: source activate.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"

if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ 虚拟环境已激活"
    echo "📍 Python: $(which python)"
    echo "📦 Pip: $(which pip)"
    echo "📂 工作目录: $SCRIPT_DIR"
else
    echo "❌ 虚拟环境不存在: $VENV_PATH"
    echo "请先运行以下命令创建虚拟环境:"
    echo "  python3 -m venv .venv"
    echo "  source activate.sh"
    echo "  pip install -r requirements.txt"
fi
