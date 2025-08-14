#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单测结果验证工具
检查生成的单测代码的编译性和运行性
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import json
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CompilerChecker:
    """编译器检查器"""
    
    def __init__(self):
        self.compilers = ['g++', 'clang++', 'c++']
        self.compiler = None
        self.compiler_version = None
        self._find_compiler()
    
    def _find_compiler(self):
        """查找可用的C++编译器"""
        for compiler in self.compilers:
            if shutil.which(compiler):
                self.compiler = compiler
                try:
                    result = subprocess.run(
                        [compiler, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if result.returncode == 0:
                        self.compiler_version = result.stdout.split('\n')[0]
                        logger.info(f"找到编译器: {self.compiler_version}")
                        break
                except:
                    pass
        
        if not self.compiler:
            logger.warning("未找到C++编译器，编译检查将被跳过")
    
    def check_syntax(self, code: str) -> Tuple[bool, List[str]]:
        """
        检查代码语法
        
        Args:
            code: C++代码
        
        Returns:
            tuple: (是否通过, 错误信息列表)
        """
        if not self.compiler:
            return True, ["编译器不可用，跳过语法检查"]
        
        errors = []
        
        try:
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # 编译命令
            cmd = [
                self.compiler,
                '-fsyntax-only',  # 仅检查语法
                '-std=c++11',
                '-Wall',
                '-Wextra',
                temp_file
            ]
            
            # 添加gtest路径（如果存在）
            gtest_paths = [
                '/usr/include',
                '/usr/local/include',
                '/opt/local/include'
            ]
            for path in gtest_paths:
                if Path(path).exists():
                    cmd.extend(['-I', path])
            
            # 执行编译
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # 解析结果
            if result.returncode != 0:
                error_lines = result.stderr.split('\n')
                for line in error_lines:
                    if 'error:' in line or 'Error:' in line:
                        errors.append(line.strip())
                    elif 'warning:' in line and len(errors) < 10:
                        errors.append(f"[警告] {line.strip()}")
            
            # 清理临时文件
            os.unlink(temp_file)
            
            return result.returncode == 0, errors[:20]  # 最多返回20个错误
            
        except subprocess.TimeoutExpired:
            return False, ["编译超时"]
        except Exception as e:
            return False, [f"编译检查失败: {str(e)}"]
    
    def try_compile(self, code: str, output_dir: str) -> Tuple[bool, Optional[str], List[str]]:
        """
        尝试编译代码为可执行文件
        
        Args:
            code: C++代码
            output_dir: 输出目录
        
        Returns:
            tuple: (是否成功, 可执行文件路径, 错误信息)
        """
        if not self.compiler:
            return False, None, ["编译器不可用"]
        
        errors = []
        executable = None
        
        try:
            # 创建临时源文件
            source_file = Path(output_dir) / "test_program.cpp"
            with open(source_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            # 输出文件
            executable = Path(output_dir) / "test_program"
            
            # 编译命令
            cmd = [
                self.compiler,
                '-std=c++11',
                '-o', str(executable),
                str(source_file),
                '-lgtest',  # 链接gtest
                '-lgtest_main',
                '-pthread'
            ]
            
            # 执行编译
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                errors = result.stderr.split('\n')[:10]
                return False, None, errors
            
            return True, str(executable), []
            
        except Exception as e:
            return False, None, [f"编译失败: {str(e)}"]


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.test_results = {}
    
    def run_test(self, executable: str, timeout: int = 60) -> Dict:
        """
        运行测试程序
        
        Args:
            executable: 可执行文件路径
            timeout: 超时时间
        
        Returns:
            dict: 测试结果
        """
        results = {
            'success': False,
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'output': '',
            'errors': []
        }
        
        try:
            # 运行测试
            result = subprocess.run(
                [executable, '--gtest_brief=1'],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            results['output'] = result.stdout
            
            # 解析gtest输出
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'PASSED' in line and 'test' in line:
                    results['tests_passed'] += 1
                elif 'FAILED' in line and 'test' in line:
                    results['tests_failed'] += 1
                elif 'RUN' in line:
                    results['tests_run'] += 1
            
            results['success'] = result.returncode == 0
            
            if result.returncode != 0:
                results['errors'].append(result.stderr)
            
        except subprocess.TimeoutExpired:
            results['errors'].append("测试运行超时")
        except Exception as e:
            results['errors'].append(f"运行测试失败: {str(e)}")
        
        return results


class TestValidator:
    """测试验证器主类"""
    
    def __init__(self):
        self.compiler_checker = CompilerChecker()
        self.test_runner = TestRunner()
        self.validation_report = {
            'timestamp': datetime.now().isoformat(),
            'file': None,
            'operator_name': None,
            'syntax_check': {},
            'compilation': {},
            'runtime': {},
            'overall_status': 'unknown'
        }
    
    def validate_file(self, test_file: str, operator_name: Optional[str] = None) -> Dict:
        """
        验证测试文件
        
        Args:
            test_file: 测试文件路径
            operator_name: 算子名称
        
        Returns:
            dict: 验证报告
        """
        logger.info(f"开始验证测试文件: {test_file}")
        
        self.validation_report['file'] = test_file
        self.validation_report['operator_name'] = operator_name
        
        # 读取文件
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            logger.error(f"无法读取文件: {str(e)}")
            self.validation_report['overall_status'] = 'error'
            return self.validation_report
        
        # 步骤1: 语法检查
        logger.info("步骤1: 检查语法...")
        syntax_ok, syntax_errors = self.compiler_checker.check_syntax(code)
        
        self.validation_report['syntax_check'] = {
            'passed': syntax_ok,
            'errors': syntax_errors
        }
        
        if syntax_ok:
            logger.info("✅ 语法检查通过")
        else:
            logger.error("❌ 语法检查失败")
            for error in syntax_errors[:5]:
                logger.error(f"  {error}")
        
        # 步骤2: 尝试编译
        logger.info("步骤2: 尝试编译...")
        temp_dir = tempfile.mkdtemp(prefix='utgen_test_')
        
        compile_ok, executable, compile_errors = self.compiler_checker.try_compile(code, temp_dir)
        
        self.validation_report['compilation'] = {
            'passed': compile_ok,
            'executable': executable,
            'errors': compile_errors
        }
        
        if compile_ok:
            logger.info("✅ 编译成功")
        else:
            logger.warning("⚠️  编译失败（可能缺少依赖）")
            for error in compile_errors[:5]:
                logger.warning(f"  {error}")
        
        # 步骤3: 运行测试（如果编译成功）
        if compile_ok and executable:
            logger.info("步骤3: 运行测试...")
            test_results = self.test_runner.run_test(executable)
            
            self.validation_report['runtime'] = test_results
            
            if test_results['success']:
                logger.info(f"✅ 测试运行成功: {test_results['tests_passed']} 通过, "
                          f"{test_results['tests_failed']} 失败")
            else:
                logger.warning("⚠️  测试运行失败")
        
        # 清理临时文件
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
        
        # 确定整体状态
        if syntax_ok:
            if compile_ok:
                if executable and self.validation_report['runtime'].get('success'):
                    self.validation_report['overall_status'] = 'success'
                else:
                    self.validation_report['overall_status'] = 'partial'
            else:
                self.validation_report['overall_status'] = 'syntax_only'
        else:
            self.validation_report['overall_status'] = 'failed'
        
        return self.validation_report
    
    def save_report(self, output_file: str):
        """
        保存验证报告
        
        Args:
            output_file: 输出文件路径
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
            logger.info(f"验证报告已保存: {output_file}")
        except Exception as e:
            logger.error(f"保存报告失败: {str(e)}")
    
    def print_summary(self):
        """打印验证摘要"""
        logger.info("\n" + "=" * 60)
        logger.info("验证摘要")
        logger.info("=" * 60)
        
        status_emoji = {
            'success': '✅',
            'partial': '⚠️',
            'syntax_only': '📝',
            'failed': '❌',
            'unknown': '❓'
        }
        
        status = self.validation_report['overall_status']
        logger.info(f"整体状态: {status_emoji[status]} {status.upper()}")
        
        # 语法检查结果
        if self.validation_report['syntax_check']:
            if self.validation_report['syntax_check']['passed']:
                logger.info("语法检查: ✅ 通过")
            else:
                logger.info(f"语法检查: ❌ 失败 ({len(self.validation_report['syntax_check']['errors'])} 个错误)")
        
        # 编译结果
        if self.validation_report['compilation']:
            if self.validation_report['compilation']['passed']:
                logger.info("编译测试: ✅ 通过")
            else:
                logger.info("编译测试: ⚠️  失败（可能需要安装gtest）")
        
        # 运行结果
        if self.validation_report['runtime']:
            runtime = self.validation_report['runtime']
            if runtime['success']:
                logger.info(f"运行测试: ✅ {runtime['tests_passed']}/{runtime['tests_run']} 测试通过")
            else:
                logger.info("运行测试: ⚠️  执行失败")
        
        logger.info("=" * 60)


def validate_directory(directory: str) -> List[Dict]:
    """
    验证目录中的所有测试文件
    
    Args:
        directory: 目录路径
    
    Returns:
        list: 所有文件的验证结果
    """
    results = []
    test_files = list(Path(directory).glob("test_*.cpp"))
    
    if not test_files:
        logger.warning(f"目录中未找到测试文件: {directory}")
        return results
    
    logger.info(f"找到 {len(test_files)} 个测试文件")
    
    for test_file in test_files:
        logger.info(f"\n处理文件: {test_file.name}")
        validator = TestValidator()
        report = validator.validate_file(str(test_file))
        validator.print_summary()
        results.append(report)
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="单测代码验证工具")
    parser.add_argument(
        'input',
        help="要验证的测试文件或目录"
    )
    parser.add_argument(
        '--operator',
        help="算子名称"
    )
    parser.add_argument(
        '--output',
        help="验证报告输出文件",
        default="validation_report.json"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 验证单个文件
        validator = TestValidator()
        report = validator.validate_file(str(input_path), args.operator)
        validator.save_report(args.output)
        validator.print_summary()
        
        # 根据状态返回相应的退出码
        exit_codes = {
            'success': 0,
            'partial': 0,
            'syntax_only': 1,
            'failed': 2,
            'unknown': 3
        }
        sys.exit(exit_codes.get(report['overall_status'], 3))
        
    elif input_path.is_dir():
        # 验证目录
        results = validate_directory(str(input_path))
        
        # 保存汇总报告
        summary = {
            'timestamp': datetime.now().isoformat(),
            'directory': str(input_path),
            'total_files': len(results),
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n汇总报告已保存: {args.output}")
        
        # 统计结果
        success_count = sum(1 for r in results if r['overall_status'] == 'success')
        logger.info(f"成功: {success_count}/{len(results)}")
        
        sys.exit(0 if success_count == len(results) else 1)
    else:
        logger.error(f"输入路径不存在: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
