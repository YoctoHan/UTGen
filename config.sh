#!/bin/bash

# 统一配置文件
# 所有脚本都使用这个配置文件，避免重复配置

# =============================================================================
# API配置 - 请根据实际情况修改
# =============================================================================
# 使用deepseek-v3-250324
# export API_KEY="aca64c03-034f-4002-8091-d63e6c127553"
# export BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
# export MODEL_NAME="ep-20250619095027-zv7m7"

export API_KEY="sk-lM7fPdhmk2hPzvdHuURPkQXQpB7KD9iHnLGpiwVD6XmrnU2X"
export BASE_URL="http://123.57.215.191:3000/v1/"
export MODEL_NAME="qwen3-coder-480b-a35b-instruct"

# =============================================================================
# 路径配置
# =============================================================================
export UT_TEMPLATE_DIR="./ut-template"
export UT_TEMPLATE_FILE="$UT_TEMPLATE_DIR/ut_template.cpp"
export EXAMPLES_DIR="./tiling-examples"
export TEST_EXAMPLES_DIR="./test-examples"
export RUNS_DIR="./runs"

# =============================================================================
# 工具脚本路径
# =============================================================================
export PYTHON_SCRIPTS_DIR="."
export PROMPT_GENERATOR="$PYTHON_SCRIPTS_DIR/prompt_generator.py"
export STAGE_1="$PYTHON_SCRIPTS_DIR/stage_1.py"
export MODEL_CALLER="$PYTHON_SCRIPTS_DIR/model_caller.py"
export POST_PROCESSOR="$PYTHON_SCRIPTS_DIR/post_processor.py"

# Few-shot示例文件
export FEWSHOT_EXAMPLES_FILE="$PYTHON_SCRIPTS_DIR/tiling-examples/fewshot_examples.txt"

# =============================================================================
# 系统配置
# =============================================================================
export MAX_FILE_SIZE="2097152"  # 2MB
export MAX_RETRIES="5"
export LOG_LEVEL="INFO"

# =============================================================================
# 功能开关
# =============================================================================
export ENABLE_TESTCASE_GENERATION="true"
export ENABLE_UT_GENERATION="true"
export ENABLE_AUTO_CSV_SEARCH="true"

# =============================================================================
# 辅助函数
# =============================================================================

# 检查必要的依赖
check_dependencies() {
    echo "📋 检查依赖..."
    
    # 检查Python依赖
    if ! python3 -c "import openai, pandas, openpyxl, pathlib" &>/dev/null; then
        echo "⚠️  缺少Python依赖库，正在安装..."
        pip3 install openai pandas openpyxl
    fi
    
    # 检查必要文件
    if [ ! -f "$UT_TEMPLATE_FILE" ]; then
        echo "⚠️  UT模板文件不存在: $UT_TEMPLATE_FILE"
    fi
    
    if [ ! -d "$EXAMPLES_DIR" ]; then
        echo "⚠️  示例目录不存在: $EXAMPLES_DIR"
    fi
    
    echo "✅ 依赖检查完成"
}

# 创建必要的目录
ensure_directories() {
    mkdir -p "$RUNS_DIR"
    mkdir -p "$UT_TEMPLATE_DIR"
    mkdir -p "$EXAMPLES_DIR"
    mkdir -p "$TEST_EXAMPLES_DIR"
}

# 显示配置信息
show_config() {
    echo "📋 当前配置:"
    echo "  API_KEY: ${API_KEY:0:10}..."
    echo "  BASE_URL: $BASE_URL"
    echo "  MODEL_NAME: $MODEL_NAME"
    echo "  UT_TEMPLATE: $UT_TEMPLATE_FILE"
    echo "  EXAMPLES_DIR: $EXAMPLES_DIR"
    echo "  RUNS_DIR: $RUNS_DIR"
    echo ""
}

# 验证配置
validate_config() {
    local errors=0
    
    if [ -z "$API_KEY" ]; then
        echo "❌ API_KEY未设置"
        ((errors++))
    fi
    
    if [ -z "$BASE_URL" ]; then
        echo "❌ BASE_URL未设置"
        ((errors++))
    fi
    
    if [ -z "$MODEL_NAME" ]; then
        echo "❌ MODEL_NAME未设置"
        ((errors++))
    fi
    
    return $errors
}

# 初始化配置
init_config() {
    echo "🔧 初始化配置..."
    ensure_directories
    
    if validate_config; then
        echo "✅ 配置验证通过"
        return 0
    else
        echo "❌ 配置验证失败"
        return 1
    fi
} 