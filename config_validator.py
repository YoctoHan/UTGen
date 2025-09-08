#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置验证器
检查和初始化项目所需的目录结构和配置文件
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """配置验证和初始化器"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.errors = []
        self.warnings = []
        self.created_items = []
    
    def validate_all(self) -> bool:
        """
        执行完整的配置验证
        
        Returns:
            bool: 验证是否通过
        """
        logger.info("=" * 60)
        logger.info("开始配置验证...")
        logger.info(f"项目根目录: {self.project_root}")
        
        # 验证各个部分
        self._validate_directory_structure()
        self._validate_config_files()
        self._validate_python_scripts()
        self._validate_shell_scripts()
        self._validate_dependencies()
        self._validate_api_config()
        
        # 输出结果
        self._print_results()
        
        return len(self.errors) == 0
    
    def _validate_directory_structure(self):
        """验证目录结构"""
        logger.info("\n📁 检查目录结构...")
        
        required_dirs = [
            ("ut-template", "单测模板目录"),
            ("tiling-examples", "重要示例目录"),
            ("test-examples", "一般示例目录"),
            ("runs", "输出目录"),
            (".cache", "缓存目录")
        ]
        
        for dir_name, description in required_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                self.warnings.append(f"目录不存在: {dir_name} ({description})")
                # 尝试创建目录
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    self.created_items.append(f"创建目录: {dir_name}")
                    logger.info(f"  ✅ 创建目录: {dir_name}")
                except Exception as e:
                    self.errors.append(f"无法创建目录 {dir_name}: {str(e)}")
                    logger.error(f"  ❌ 无法创建目录: {dir_name}")
            else:
                logger.info(f"  ✅ {dir_name}")
    
    def _validate_config_files(self):
        """验证配置文件"""
        logger.info("\n📄 检查配置文件...")
        
        # 检查config.sh
        config_sh = self.project_root / "config.sh"
        if not config_sh.exists():
            self.errors.append("config.sh 不存在")
            logger.error("  ❌ config.sh 不存在")
        else:
            # 检查必要的配置项
            with open(config_sh, 'r') as f:
                content = f.read()
                
            required_vars = [
                "API_KEY",
                "BASE_URL", 
                "MODEL_NAME",
                "UT_TEMPLATE_FILE"
            ]
            
            for var in required_vars:
                if f"export {var}=" not in content:
                    self.warnings.append(f"config.sh 中可能缺少 {var} 配置")
                    logger.warning(f"  ⚠️  可能缺少配置: {var}")
            
            logger.info("  ✅ config.sh")
        
        # 检查示例配置
        examples_config = self.project_root / "examples_config.json"
        if not examples_config.exists():
            # 创建默认配置
            default_config = {
                "important_examples": [
                    {
                        "name": "AllGatherMatmul",
                        "cpp_file": "test_all_gather_matmul.cpp",
                        "excel_file": "AllgatherMatmulTilingCases.xlsx",
                        "priority": 1,
                        "tags": ["collective", "matmul"]
                    }
                ],
                "example_selection": {
                    "max_general_examples": 3,
                    "max_important_examples": 3,
                    "prefer_similar_operators": True
                }
            }
            
            try:
                with open(examples_config, 'w', encoding='utf-8') as f:
                    json.dump(default_config, f, indent=2, ensure_ascii=False)
                self.created_items.append("创建 examples_config.json")
                logger.info("  ✅ 创建默认 examples_config.json")
            except Exception as e:
                self.warnings.append(f"无法创建 examples_config.json: {str(e)}")
                logger.warning("  ⚠️  无法创建 examples_config.json")
        else:
            logger.info("  ✅ examples_config.json")
    
    def _validate_python_scripts(self):
        """验证Python脚本"""
        logger.info("\n🐍 检查Python脚本...")
        
        required_scripts = [
            "utils.py",
            "stage_1.py",
            "convert_ut_from_xlsx.py",
        ]
        
        for script in required_scripts:
            script_path = self.project_root / script
            if not script_path.exists():
                self.errors.append(f"Python脚本不存在: {script}")
                logger.error(f"  ❌ {script}")
            else:
                # 检查是否可执行
                try:
                    import ast
                    with open(script_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                    logger.info(f"  ✅ {script}")
                except SyntaxError as e:
                    self.errors.append(f"Python脚本语法错误 {script}: {str(e)}")
                    logger.error(f"  ❌ {script} (语法错误)")
    
    def _validate_shell_scripts(self):
        """验证Shell脚本"""
        logger.info("\n🔧 检查Shell脚本...")
        
        required_scripts = [
            "workflow.sh",
            "config.sh",
            "entrypoint.sh"
        ]
        
        for script in required_scripts:
            script_path = self.project_root / script
            if not script_path.exists():
                self.errors.append(f"Shell脚本不存在: {script}")
                logger.error(f"  ❌ {script}")
            else:
                # 检查执行权限
                if not os.access(script_path, os.X_OK):
                    self.warnings.append(f"Shell脚本没有执行权限: {script}")
                    # 尝试添加执行权限
                    try:
                        script_path.chmod(0o755)
                        self.created_items.append(f"添加执行权限: {script}")
                        logger.info(f"  ✅ {script} (已添加执行权限)")
                    except Exception as e:
                        logger.warning(f"  ⚠️  {script} (无执行权限)")
                else:
                    logger.info(f"  ✅ {script}")
    
    def _validate_dependencies(self):
        """验证Python依赖"""
        logger.info("\n📦 检查Python依赖...")
        
        required_packages = [
            "openai",
            "pandas",
            "openpyxl"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
                logger.info(f"  ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                logger.warning(f"  ⚠️  {package} (未安装)")
        
        if missing_packages:
            self.warnings.append(f"缺少Python包: {', '.join(missing_packages)}")
            logger.info(f"\n建议运行: pip install {' '.join(missing_packages)}")
    
    def _validate_api_config(self):
        """验证API配置"""
        logger.info("\n🔑 检查API配置...")
        
        # 从环境变量或config.sh读取配置
        api_key = os.environ.get('API_KEY', '')
        base_url = os.environ.get('BASE_URL', '')
        model_name = os.environ.get('MODEL_NAME', '')
        
        if not api_key or api_key.startswith('your_') or api_key.startswith('sk-'):
            self.warnings.append("API_KEY 未配置或使用默认值")
            logger.warning("  ⚠️  API_KEY 需要配置")
        else:
            logger.info(f"  ✅ API_KEY (已配置: {api_key[:10]}...)")
        
        if not base_url or 'example.com' in base_url:
            self.warnings.append("BASE_URL 未配置或使用默认值")
            logger.warning("  ⚠️  BASE_URL 需要配置")
        else:
            logger.info(f"  ✅ BASE_URL ({base_url})")
        
        if not model_name:
            self.warnings.append("MODEL_NAME 未配置")
            logger.warning("  ⚠️  MODEL_NAME 需要配置")
        else:
            logger.info(f"  ✅ MODEL_NAME ({model_name})")
    
    def _print_results(self):
        """输出验证结果"""
        logger.info("\n" + "=" * 60)
        logger.info("验证结果汇总\n")
        
        if self.created_items:
            logger.info("🔨 自动修复:")
            for item in self.created_items:
                logger.info(f"  - {item}")
        
        if self.warnings:
            logger.info("\n⚠️  警告:")
            for warning in self.warnings:
                logger.info(f"  - {warning}")
        
        if self.errors:
            logger.info("\n❌ 错误:")
            for error in self.errors:
                logger.info(f"  - {error}")
        
        logger.info("\n" + "=" * 60)
        
        if not self.errors:
            logger.info("✅ 配置验证通过！")
        else:
            logger.info("❌ 配置验证失败，请修复上述错误。")
    
    def create_sample_files(self):
        """创建示例文件"""
        logger.info("\n📝 创建示例文件...")
        
        # 创建UT模板
        ut_template_dir = self.project_root / "ut-template"
        ut_template_file = ut_template_dir / "ut_template.cpp"
        
        if not ut_template_file.exists():
            ut_template_content = """/**
 * 单元测试模板文件
 * 用于生成算子的单元测试代码
 */

#include <gtest/gtest.h>
#include <vector>
#include <memory>

// 测试类模板
class OperatorTiling : public testing::Test {
protected:
    void SetUp() override {
        // 初始化测试环境
    }
    
    void TearDown() override {
        // 清理测试环境
    }
};

// 测试用例模板
TEST_F(OperatorTiling, BasicTest) {
    // 准备测试数据
    
    // 调用算子
    
    // 验证结果
    EXPECT_EQ(expected, actual);
}

// 边界测试模板
TEST_F(OperatorTiling, BoundaryTest) {
    // 测试边界条件
}

// 性能测试模板
TEST_F(OperatorTiling, PerformanceTest) {
    // 测试性能
}
"""
            try:
                ut_template_dir.mkdir(parents=True, exist_ok=True)
                with open(ut_template_file, 'w', encoding='utf-8') as f:
                    f.write(ut_template_content)
                logger.info(f"  ✅ 创建UT模板: {ut_template_file}")
            except Exception as e:
                logger.error(f"  ❌ 无法创建UT模板: {str(e)}")
        
        # 创建示例测试文件
        test_examples_dir = self.project_root / "test-examples"
        sample_test_file = test_examples_dir / "test_sample_operator.cpp"
        
        if not sample_test_file.exists():
            sample_content = """#include <gtest/gtest.h>

class SampleOperatorTiling : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SampleOperatorTiling, BasicTest) {
    int result = 1 + 1;
    EXPECT_EQ(2, result);
}
"""
            try:
                test_examples_dir.mkdir(parents=True, exist_ok=True)
                with open(sample_test_file, 'w', encoding='utf-8') as f:
                    f.write(sample_content)
                logger.info(f"  ✅ 创建示例测试文件: {sample_test_file}")
            except Exception as e:
                logger.error(f"  ❌ 无法创建示例文件: {str(e)}")


def init_project(project_root: str = ".") -> bool:
    """
    初始化项目配置
    
    Args:
        project_root: 项目根目录
    
    Returns:
        bool: 初始化是否成功
    """
    validator = ConfigValidator(project_root)
    
    # 执行验证
    is_valid = validator.validate_all()
    
    # 创建示例文件
    validator.create_sample_files()
    
    return is_valid


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="配置验证和初始化工具")
    parser.add_argument(
        "--init", 
        action="store_true",
        help="初始化项目结构"
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="项目根目录路径"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="自动修复可修复的问题"
    )
    
    args = parser.parse_args()
    
    if args.init or args.fix:
        logger.info("🚀 初始化项目配置...")
        if init_project(args.project_root):
            logger.info("\n✅ 项目配置完成，可以开始使用！")
            logger.info("\n下一步:")
            logger.info("1. 编辑 config.sh 设置API配置")
            logger.info("2. 运行 ./workflow.sh --help 查看使用说明")
            sys.exit(0)
        else:
            logger.error("\n❌ 项目配置存在问题，请检查错误信息")
            sys.exit(1)
    else:
        # 仅验证
        validator = ConfigValidator(args.project_root)
        if validator.validate_all():
            sys.exit(0)
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
