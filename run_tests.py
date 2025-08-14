#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目测试脚本
验证各个模块是否正常工作
"""

import sys
import os
import importlib
import subprocess
from pathlib import Path

# 测试结果统计
tests_run = 0
tests_passed = 0
tests_failed = 0

def print_test_header(test_name):
    """打印测试头"""
    global tests_run
    tests_run += 1
    print(f"\n[测试 {tests_run}] {test_name}")
    print("-" * 50)

def test_passed(message=""):
    """标记测试通过"""
    global tests_passed
    tests_passed += 1
    print(f"✅ 通过: {message}")

def test_failed(message=""):
    """标记测试失败"""
    global tests_failed
    tests_failed += 1
    print(f"❌ 失败: {message}")

def test_python_imports():
    """测试Python模块导入"""
    print_test_header("Python模块导入测试")
    
    modules = [
        'utils',
        'stage_1',
        'prompt_generator',
        'model_caller',
        'post_processor',
        'config_validator',
        'test_validator'
    ]
    
    for module_name in modules:
        try:
            importlib.import_module(module_name)
            test_passed(f"成功导入 {module_name}")
        except ImportError as e:
            test_failed(f"无法导入 {module_name}: {str(e)}")

def test_dependencies():
    """测试依赖包"""
    print_test_header("Python依赖包测试")
    
    dependencies = [
        ('openai', 'OpenAI API客户端'),
        ('pandas', '数据处理'),
        ('openpyxl', 'Excel处理')
    ]
    
    for package, description in dependencies:
        try:
            __import__(package)
            test_passed(f"{package} ({description})")
        except ImportError:
            test_failed(f"{package} ({description}) - 请运行: pip install {package}")

def test_shell_scripts():
    """测试Shell脚本"""
    print_test_header("Shell脚本测试")
    
    scripts = [
        'workflow.sh',
        'config.sh',
        'entrypoint.sh',
        'quickstart.sh'
    ]
    
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            if os.access(script_path, os.X_OK):
                test_passed(f"{script} 存在且可执行")
            else:
                test_failed(f"{script} 存在但不可执行")
        else:
            test_failed(f"{script} 不存在")

def test_directory_structure():
    """测试目录结构"""
    print_test_header("目录结构测试")
    
    directories = [
        ('ut-template', '单测模板目录'),
        ('tiling-examples', '重要示例目录'),
        ('test-examples', '一般示例目录'),
        ('runs', '输出目录'),
        ('.cache', '缓存目录')
    ]
    
    for dir_name, description in directories:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            test_passed(f"{dir_name} ({description})")
        else:
            test_failed(f"{dir_name} ({description}) 不存在")
            # 尝试创建
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"  已自动创建: {dir_name}")
            except:
                pass

def test_config_files():
    """测试配置文件"""
    print_test_header("配置文件测试")
    
    # 检查config.sh
    config_path = Path('config.sh')
    if config_path.exists():
        with open(config_path, 'r') as f:
            content = f.read()
        
        required_vars = ['API_KEY', 'BASE_URL', 'MODEL_NAME']
        for var in required_vars:
            if f'export {var}=' in content:
                test_passed(f"配置变量 {var} 已定义")
            else:
                test_failed(f"配置变量 {var} 未定义")
    else:
        test_failed("config.sh 不存在")

def test_api_connectivity():
    """测试API连接（可选）"""
    print_test_header("API连接测试（可选）")
    
    try:
        # 尝试导入并获取配置
        import os
        api_key = os.environ.get('API_KEY', '')
        
        if api_key and not api_key.startswith('your_'):
            print("检测到API密钥配置")
            # 这里可以添加实际的API测试
            test_passed("API密钥已配置")
        else:
            print("⚠️  API密钥未配置或使用默认值")
    except:
        print("⚠️  无法测试API连接")

def test_utils_functions():
    """测试工具函数"""
    print_test_header("工具函数测试")
    
    try:
        from utils import validate_path, format_file_size, create_timestamped_dir
        
        # 测试路径验证
        test_path = validate_path('.')
        if test_path:
            test_passed("validate_path 函数正常")
        
        # 测试文件大小格式化
        size_str = format_file_size(1024)
        if size_str == "1.00 KB":
            test_passed("format_file_size 函数正常")
        
        # 测试时间戳目录创建
        test_dir = create_timestamped_dir("test", parent_dir="runs")
        if test_dir.exists():
            test_passed("create_timestamped_dir 函数正常")
            # 清理测试目录
            test_dir.rmdir()
        
    except Exception as e:
        test_failed(f"工具函数测试失败: {str(e)}")

def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("   🧪 算子单测生成工具 - 系统测试")
    print("=" * 60)
    
    # 运行各项测试
    test_python_imports()
    test_dependencies()
    test_shell_scripts()
    test_directory_structure()
    test_config_files()
    test_utils_functions()
    test_api_connectivity()
    
    # 打印测试结果
    print("\n" + "=" * 60)
    print("   测试结果汇总")
    print("=" * 60)
    print(f"总计运行: {tests_run} 个测试")
    print(f"✅ 通过: {tests_passed} 个")
    print(f"❌ 失败: {tests_failed} 个")
    
    if tests_failed == 0:
        print("\n🎉 所有测试通过！系统已准备就绪。")
        print("\n下一步：")
        print("1. 运行 ./quickstart.sh 进行快速配置")
        print("2. 或运行 ./workflow.sh --help 查看使用说明")
        return 0
    else:
        print(f"\n⚠️  有 {tests_failed} 个测试失败，请检查并修复问题。")
        print("\n建议：")
        print("1. 运行 pip install -r requirements.txt 安装依赖")
        print("2. 运行 python3 config_validator.py --init 初始化项目")
        print("3. 检查 config.sh 中的API配置")
        return 1

def main():
    """主函数"""
    # 切换到脚本所在目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 运行测试
    exit_code = run_all_tests()
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
