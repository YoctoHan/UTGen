#!/bin/bash

# 快速开始脚本
# 帮助用户快速配置和开始使用算子单测生成工具

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# 显示欢迎信息
show_welcome() {
    clear
    echo "================================================"
    echo "   🚀 算子单测自动生成工具 - 快速配置向导"
    echo "================================================"
    echo ""
    echo "本向导将帮助您："
    echo "  1. 检查系统环境"
    echo "  2. 安装必要的依赖"
    echo "  3. 配置API密钥"
    echo "  4. 初始化项目结构"
    echo "  5. 运行示例测试"
    echo ""
    echo "按 Enter 键继续，或 Ctrl+C 退出..."
    read
}

# 检查Python版本
check_python() {
    print_info "检查Python环境..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "找到 Python $PYTHON_VERSION"
        
        # 检查版本
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 8 ]; then
            print_success "Python版本符合要求 (>= 3.8)"
            return 0
        else
            print_warning "建议使用 Python 3.8 或更高版本"
        fi
    else
        print_error "未找到 Python 3，请先安装 Python"
        echo "安装方法："
        echo "  macOS: brew install python3"
        echo "  Ubuntu: sudo apt-get install python3 python3-pip"
        echo "  CentOS: sudo yum install python3 python3-pip"
        exit 1
    fi
}

# 安装Python依赖
install_dependencies() {
    print_info "安装Python依赖..."
    
    if [ -f "requirements.txt" ]; then
        print_info "使用 pip 安装依赖包..."
        pip3 install -r requirements.txt --user -q
        
        if [ $? -eq 0 ]; then
            print_success "Python依赖安装成功"
        else
            print_error "依赖安装失败，请手动运行: pip3 install -r requirements.txt"
            exit 1
        fi
    else
        print_warning "requirements.txt 不存在，跳过依赖安装"
    fi
}

# 配置API
configure_api() {
    print_info "配置API访问..."
    echo ""
    echo "请选择API提供商："
    echo "  1. DeepSeek"
    echo "  2. OpenAI"
    echo "  3. 自定义API"
    echo "  4. 跳过（稍后配置）"
    echo ""
    read -p "请输入选项 [1-4]: " api_choice
    
    case $api_choice in
        1)
            read -p "请输入 DeepSeek API Key: " api_key
            BASE_URL="https://api.deepseek.com/v1/"
            MODEL_NAME="deepseek-coder"
            ;;
        2)
            read -p "请输入 OpenAI API Key: " api_key
            BASE_URL="https://api.openai.com/v1/"
            read -p "请输入模型名称 [默认: gpt-4]: " model
            MODEL_NAME=${model:-gpt-4}
            ;;
        3)
            read -p "请输入 API Key: " api_key
            read -p "请输入 Base URL: " BASE_URL
            read -p "请输入模型名称: " MODEL_NAME
            ;;
        4)
            print_warning "跳过API配置，请稍后编辑 config.sh 文件"
            return
            ;;
        *)
            print_error "无效选项"
            configure_api
            return
            ;;
    esac
    
    # 更新config.sh
    if [ -n "$api_key" ]; then
        # 备份原配置
        cp config.sh config.sh.bak
        
        # 更新配置
        sed -i.tmp "s|export API_KEY=.*|export API_KEY=\"$api_key\"|" config.sh
        sed -i.tmp "s|export BASE_URL=.*|export BASE_URL=\"$BASE_URL\"|" config.sh
        sed -i.tmp "s|export MODEL_NAME=.*|export MODEL_NAME=\"$MODEL_NAME\"|" config.sh
        
        rm -f config.sh.tmp
        print_success "API配置已更新"
    fi
}

# 初始化项目
initialize_project() {
    print_info "初始化项目结构..."
    
    python3 config_validator.py --init
    
    if [ $? -eq 0 ]; then
        print_success "项目初始化成功"
    else
        print_error "项目初始化失败"
        exit 1
    fi
}

# 运行示例
run_example() {
    print_info "是否运行示例测试？"
    read -p "运行示例将调用API生成测试代码 [y/N]: " run_test
    
    if [[ "$run_test" == "y" || "$run_test" == "Y" ]]; then
        print_info "运行示例测试..."
        
        # 创建示例源码目录
        mkdir -p example_source
        cat > example_source/sample_operator.cpp << 'EOF'
#include <iostream>

class SampleOperator {
public:
    int add(int a, int b) {
        return a + b;
    }
    
    int multiply(int a, int b) {
        return a * b;
    }
};

// Tiling function
void sample_tiling(int m, int n, int k) {
    // Sample tiling implementation
    std::cout << "Tiling: " << m << "x" << n << "x" << k << std::endl;
}
EOF
        
        print_info "生成示例算子的单元测试..."
        ./workflow.sh gen-ut SampleOperator example_source
        
        if [ $? -eq 0 ]; then
            print_success "示例测试生成成功！"
            
            # 显示生成的文件
            latest_run=$(ls -t runs/ | head -1)
            if [ -n "$latest_run" ]; then
                echo ""
                print_info "生成的文件位于: runs/$latest_run/"
                ls -la "runs/$latest_run/"
                
                # 验证生成的代码
                if [ -f "runs/$latest_run/test_sampleoperator_tiling.cpp" ]; then
                    print_info "验证生成的代码..."
                    python3 test_validator.py "runs/$latest_run/test_sampleoperator_tiling.cpp"
                fi
            fi
        else
            print_warning "示例测试生成失败，请检查API配置"
        fi
    fi
}

# 显示下一步
show_next_steps() {
    echo ""
    echo "================================================"
    echo "   ✅ 配置完成！"
    echo "================================================"
    echo ""
    echo "🎯 下一步操作："
    echo ""
    echo "1. 生成单元测试（默认）："
    echo "   ./workflow.sh <算子名称> <源码路径>"
    echo ""
    echo "2. 仅生成测试参数："
    echo "   ./workflow.sh gen-params <算子名称> <源码路径>"
    echo ""
    echo "3. 完整流程（参数+单测）："
    echo "   ./workflow.sh gen-all <算子名称> <源码路径>"
    echo ""
    echo "4. 查看帮助："
    echo "   ./workflow.sh --help"
    echo ""
    echo "5. 验证项目配置："
    echo "   ./workflow.sh --validate"
    echo ""
    echo "================================================"
    echo ""
    echo "📚 更多信息请查看 README.md"
}

# 主流程
main() {
    show_welcome
    check_python
    install_dependencies
    configure_api
    initialize_project
    run_example
    show_next_steps
}

# 运行主流程
main
