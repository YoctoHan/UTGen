#!/bin/bash

# 算子单测自动生成工具 - 主入口脚本
# 简洁的面向过程设计，易于使用和维护

set -e

# 加载配置文件
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# 在脚本开头添加一个函数来转换小写
to_lower() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

# =============================================================================
# 显示帮助信息
# =============================================================================
show_help() {
    cat << EOF
🔧 算子单测自动生成工具 v2.0

用法: $0 <命令> [选项] <算子名称> <源码路径...>

命令:
  gen-ut          生成单元测试代码 (默认命令)
  gen-params      生成测试参数CSV文件
  gen-all         先生成参数，再生成单测 (完整流程)

选项:
  -h, --help      显示帮助信息
  -c, --config    显示当前配置
  --validate      验证项目配置
  --init          初始化项目结构
  -v, --verbose   详细输出模式
  --dry-run       模拟运行，不实际调用API

参数:
  算子名称        算子名称，如 AllGatherMatmul、MatmulReduceScatter
  源码路径        算子源码目录，支持多个路径

示例:
  # 生成单元测试 (默认命令)
  $0 AllGatherMatmul ../cann-ops-adv/src/mc2/all_gather_matmul

  # 生成测试参数
  $0 gen-params MatmulReduceScatter ../canndev/ops/built-in/op_tiling/runtime/matmul_reduce_scatter

  # 完整流程：先生成参数，再生成单测
  $0 gen-all MatmulAllReduce ../cann-ops-adv/src/mc2/matmul_all_reduce

配置:
  修改 config.sh 文件来调整API密钥、模型等配置

输出目录:
  runs/YYYYMMDD_HHMMSS_<算子名称>[_testcase]/
    ├── test_<算子名称>_tiling.cpp     # 生成的单测文件
    ├── test_params_<算子名称>.xlsx     # 测试参数文件
    ├── prompt_*.txt                   # 使用的prompt
    └── *.log                          # 运行日志

EOF
}

# =============================================================================
# 生成测试参数
# =============================================================================
stage_1() {
    local operator_name="$1"
    shift
    local source_paths=("$@")
    
    echo "📊 生成测试参数: $operator_name"
    echo "=================================="
    
    # 创建运行目录
    local timestamp=$(python3 -c "
import datetime
print(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
" 2>/dev/null)
    local operator_lower=$(to_lower "$operator_name")
    local run_dir="runs/${timestamp}_${operator_lower}_stage_1"
    
    mkdir -p "$run_dir"
    
    # 定义文件路径
    local output_file="$run_dir/test_params_${operator_lower}.xlsx"
    local prompt_file="$run_dir/prompt_testcase_${operator_lower}.txt"
    local log_file="$run_dir/testcase_generation.log"
    
    # 记录开始信息
    {
        echo "开始时间: $(date)"
        echo "算子名称: $operator_name"
        echo "源码路径: ${source_paths[*]}"
        echo "运行目录: $run_dir"
        echo "=================================="
        echo ""
    } > "$log_file"
    
    # 调用测试参数生成器
    echo "🚀 调用测试参数生成器..." | tee -a "$log_file"
    if python3 "$STAGE_1" "$operator_name" "$output_file" "$prompt_file" \
                "$FEWSHOT_STAGE1_FILE" "$API_KEY" "$BASE_URL" "$MODEL_NAME" "${source_paths[@]}" 2>&1 | tee -a "$log_file"; then
        
        if [ -f "$output_file" ]; then
            echo "✅ 测试参数生成成功: $output_file" | tee -a "$log_file"
            echo "生成的测试参数: $(wc -l < "$output_file") 行" | tee -a "$log_file"
            return 0
        else
            echo "❌ 测试参数生成失败：未生成输出文件" | tee -a "$log_file"
            return 1
        fi
    else
        echo "❌ 测试参数生成过程出错" | tee -a "$log_file"
        return 1
    fi
}

# =============================================================================
# 生成单元测试
# =============================================================================
stage_2() {
    local operator_name="$1"
    shift
    local source_paths=("$@")
    
    echo "🧪 生成单元测试: $operator_name"
    echo "=============================="
    
    # 创建运行目录
    local timestamp=$(python3 -c "
import datetime
print(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
" 2>/dev/null)
    local operator_lower=$(to_lower "$operator_name")
    local run_dir="runs/${timestamp}_${operator_lower}"
    
    mkdir -p "$run_dir"
    
    # 定义文件路径
    local prompt_file="$run_dir/prompt_${operator_lower}.txt"
    local raw_response_file="$run_dir/raw_response.txt"
    local output_file="$run_dir/test_${operator_lower}_tiling.cpp"
    local log_file="$run_dir/generation.log"
    
    # 查找参考参数文件（支持 .xlsx 和 .csv）
    local param_file=""
    if [ "$ENABLE_AUTO_CSV_SEARCH" = "true" ]; then
        echo "🔍 查找参考参数文件 (xlsx/csv)..." | tee -a "$log_file"
        local operator_lower=$(to_lower "$operator_name")

        local candidates=()
        # 找到最新的 xlsx
        if compgen -G "runs/*/test_params_${operator_lower}.xlsx" > /dev/null; then
            local latest_xlsx=$(ls -t runs/*/test_params_${operator_lower}.xlsx | head -n 1)
            candidates+=("$latest_xlsx")
        fi
        # 找到最新的 csv
        if compgen -G "runs/*/test_params_${operator_lower}.csv" > /dev/null; then
            local latest_csv=$(ls -t runs/*/test_params_${operator_lower}.csv | head -n 1)
            candidates+=("$latest_csv")
        fi

        if [ ${#candidates[@]} -gt 0 ]; then
            # 在候选中选择修改时间最新的一个
            param_file=$(ls -t "${candidates[@]}" | head -n 1)
            echo "找到参考参数文件: $param_file" | tee -a "$log_file"
        fi
    fi
    
    # 记录开始信息
    {
        echo "开始时间: $(date)"
        echo "算子名称: $operator_name"
        echo "源码路径: ${source_paths[*]}"
        echo "参数参考文件: ${param_file:-无}"
        echo "运行目录: $run_dir"
        echo "=============================="
        echo ""
    } >> "$log_file"
    
    # 步骤1: 生成prompt
    echo "📝 步骤1: 生成Prompt..." | tee -a "$log_file"
    local prompt_args=("${source_paths[@]}" "-t" "$UT_TEMPLATE_FILE" "-o" "$prompt_file")
    # 为Stage 2注入few-shot文件（可选）
    if [ -n "$FEWSHOT_STAGE2_FILE" ] && [ -f "$FEWSHOT_STAGE2_FILE" ]; then
        prompt_args+=("-f" "$FEWSHOT_STAGE2_FILE")
        echo "使用Stage2 Few-shot文件: $(basename "$FEWSHOT_STAGE2_FILE")" | tee -a "$log_file"
    fi
    if [ -n "$param_file" ]; then
        prompt_args+=("-c" "$param_file")
        echo "使用参考参数文件: $(basename "$param_file")" | tee -a "$log_file"
    fi
    
    if python3 "$PROMPT_GENERATOR" "${prompt_args[@]}" 2>&1 | tee -a "$log_file"; then
        echo "✅ Prompt生成完成" | tee -a "$log_file"
    else
        echo "❌ Prompt生成失败" | tee -a "$log_file"
        return 1
    fi
    
    # 步骤2: 调用模型
    echo "🤖 步骤2: 调用模型获取响应..." | tee -a "$log_file"
    if python3 "$MODEL_CALLER" "$prompt_file" "$raw_response_file" \
                "$API_KEY" "$BASE_URL" "$MODEL_NAME" 2>&1 | tee -a "$log_file"; then
        echo "✅ 模型响应获取完成" | tee -a "$log_file"
    else
        echo "❌ 模型调用失败" | tee -a "$log_file"
        return 1
    fi
    
    # 步骤3: 后处理
    echo "🛠️ 步骤3: 后处理生成单测代码..." | tee -a "$log_file"
    if python3 "$POST_PROCESSOR" "$raw_response_file" "$output_file" \
                "$operator_name" 2>&1 | tee -a "$log_file"; then
        echo "✅ 单测代码生成完成: $output_file" | tee -a "$log_file"
        
        # 显示结果统计
        if [ -f "$output_file" ]; then
            local output_lines=$(wc -l < "$output_file")
            echo "📊 生成统计: $output_lines 行代码" | tee -a "$log_file"
            echo "🎉 单元测试生成成功!"
            echo "📁 输出目录: $run_dir"
            echo "📄 单测文件: $output_file"
            return 0
        fi
    else
        echo "❌ 后处理失败" | tee -a "$log_file"
        return 1
    fi
}

# =============================================================================
# 完整流程：先生成参数，再生成单测
# =============================================================================
stage_all() {
    local operator_name="$1"
    shift
    local source_paths=("$@")
    
    echo "🚀 执行完整流程: $operator_name"
    echo "==============================="
    
    # 步骤1: 生成测试参数
    echo "第1步: 生成测试参数"
    if stage_1 "$operator_name" "${source_paths[@]}"; then
        echo "✅ 测试参数生成完成"
    else
        echo "❌ 测试参数生成失败，但继续执行单测生成"
    fi
    
    echo ""
    
    # 步骤2: 生成单元测试
    echo "第2步: 生成单元测试"
    if stage_2 "$operator_name" "${source_paths[@]}"; then
        echo "✅ 完整流程执行成功!"
        return 0
    else
        echo "❌ 完整流程执行失败"
        return 1
    fi
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    # 解析命令行参数
    local command="stage-all"  # 默认命令
    local verbose=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--config)
                show_config
                exit 0
                ;;
            --validate)
                echo "🔍 验证项目配置..."
                python3 "$SCRIPT_DIR/config_validator.py"
                exit $?
                ;;
            --init)
                echo "🚀 初始化项目..."
                python3 "$SCRIPT_DIR/config_validator.py" --init
                exit $?
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --dry-run)
                dry_run=true
                shift
                ;;
            stage-1|stage-2|stage-all)
                command="$1"
                shift
                ;;
            gen-params)
                command="stage-1"
                shift
                ;;
            gen-ut)
                command="stage-2"
                shift
                ;;
            gen-all)
                command="stage-all"
                shift
                ;;
            -*)
                echo "错误: 未知选项 $1"
                echo "使用 $0 --help 查看帮助"
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done
    
    # 检查参数
    if [ $# -lt 2 ]; then
        echo "错误: 缺少必要参数"
        echo ""
        show_help
        exit 1
    fi
    
    local operator_name="$1"
    shift
    local source_paths=("$@")
    
    # 初始化配置
    if ! init_config; then
        echo "❌ 配置初始化失败"
        exit 1
    fi
    
    if [ "$verbose" = true ]; then
        show_config
    fi
    
    # 检查依赖
    check_dependencies
    
    # 验证源码路径
    echo "🔍 验证源码路径..."
    local valid_paths=()
    for path in "${source_paths[@]}"; do
        if [ -e "$path" ]; then
            echo "✅ $path"
            valid_paths+=("$path")
        else
            echo "⚠️  路径不存在: $path"
        fi
    done
    
    if [ ${#valid_paths[@]} -eq 0 ]; then
        echo "❌ 没有有效的源码路径"
        exit 1
    fi
    
    echo ""
    
    # 执行对应命令
    case $command in
        stage-1)
            stage_1 "$operator_name" "${valid_paths[@]}"
            ;;
        stage-2)
            stage_2 "$operator_name" "${valid_paths[@]}"
            ;;
        stage-all)
            stage_all "$operator_name" "${valid_paths[@]}"
            ;;
        *)
            echo "错误: 未知命令 $command"
            exit 1
            ;;
    esac
}

# 运行主函数
main "$@" 