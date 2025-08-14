#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理器 - 增强版
处理模型生成的原始响应，提取、清理、验证和格式化代码
支持更智能的代码提取和验证
"""

import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodeExtractor:
    """代码提取器"""
    
    def __init__(self):
        self.cpp_patterns = [
            r'```(?:cpp|c\+\+|C\+\+|CPP)\s*(.*?)```',  # Markdown代码块
            r'```\s*(#include.*?)```',  # 通用代码块，以#include开头
            r'<code>\s*(.*?)</code>',  # HTML代码块
        ]
        
        self.code_indicators = [
            '#include',
            'namespace',
            'class',
            'TEST_F',
            'TEST(',
            'void',
            'int main'
        ]
    
    def extract(self, content: str) -> str:
        """
        从内容中提取C++代码
        
        Args:
            content: 原始内容
        
        Returns:
            str: 提取的C++代码
        """
        # 首先尝试从代码块中提取
        extracted_code = self._extract_from_blocks(content)
        if extracted_code:
            return extracted_code
        
        # 如果没有找到代码块，尝试智能识别
        extracted_code = self._smart_extract(content)
        if extracted_code:
            return extracted_code
        
        # 最后的备选：返回整个内容
        logger.warning("未能识别代码块，返回原始内容")
        return content.strip()
    
    def _extract_from_blocks(self, content: str) -> Optional[str]:
        """从代码块中提取"""
        for pattern in self.cpp_patterns:
            matches = re.findall(pattern, content, re.DOTALL | re.IGNORECASE)
            if matches:
                # 选择最长的匹配
                longest_match = max(matches, key=len)
                if self._is_valid_cpp_code(longest_match):
                    logger.info(f"从代码块中提取了 {len(longest_match)} 字符的C++代码")
                    return longest_match.strip()
        return None
    
    def _smart_extract(self, content: str) -> Optional[str]:
        """智能提取代码"""
        lines = content.split('\n')
        
        # 找到代码的开始和结束位置
        start_idx = -1
        end_idx = len(lines)
        
        # 查找开始位置
        for i, line in enumerate(lines):
            if any(indicator in line for indicator in self.code_indicators):
                start_idx = i
                # 向前查找可能的注释
                while start_idx > 0 and (
                    lines[start_idx - 1].strip().startswith('//') or 
                    lines[start_idx - 1].strip().startswith('/*') or
                    lines[start_idx - 1].strip().startswith('*')
                ):
                    start_idx -= 1
                break
        
        if start_idx == -1:
            return None
        
        # 查找结束位置
        brace_count = 0
        in_code = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            
            # 计算大括号
            brace_count += line.count('{') - line.count('}')
            
            if '{' in line or '}' in line:
                in_code = True
            
            # 找到代码结束位置
            if in_code and brace_count == 0:
                end_idx = i + 1
                break
        
        if start_idx < end_idx:
            extracted = '\n'.join(lines[start_idx:end_idx])
            if self._is_valid_cpp_code(extracted):
                logger.info(f"智能提取了 {len(extracted)} 字符的C++代码")
                return extracted
        
        return None
    
    def _is_valid_cpp_code(self, code: str) -> bool:
        """验证是否为有效的C++代码"""
        if not code or len(code) < 50:
            return False
        
        # 检查必要的代码元素
        required_elements = ['#include', 'TEST']
        return any(elem in code for elem in required_elements)


class CodeCleaner:
    """代码清理器"""
    
    def __init__(self):
        self.cleanup_patterns = [
            (r'\s*//\s*\.\.\.\s*省略.*?\n', ''),  # 移除省略标记
            (r'\s*//\s*内容已截断.*?\n', ''),  # 移除截断标记
            (r'\n{3,}', '\n\n'),  # 压缩多余空行
            (r'[ \t]+$', '', re.MULTILINE),  # 移除行尾空白
        ]
    
    def clean(self, code: str) -> str:
        """
        清理和格式化代码
        
        Args:
            code: 原始代码
        
        Returns:
            str: 清理后的代码
        """
        cleaned = code
        
        # 应用清理模式
        for pattern, replacement, *flags in self.cleanup_patterns:
            if flags:
                cleaned = re.sub(pattern, replacement, cleaned, flags=flags[0])
            else:
                cleaned = re.sub(pattern, replacement, cleaned)
        
        # 格式化代码结构
        cleaned = self._format_structure(cleaned)
        
        # 修复常见问题
        cleaned = self._fix_common_issues(cleaned)
        
        return cleaned.strip()
    
    def _format_structure(self, code: str) -> str:
        """格式化代码结构"""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # 跳过空行
            if not stripped:
                if formatted_lines and formatted_lines[-1] != '':
                    formatted_lines.append('')
                continue
            
            # 处理预处理指令
            if stripped.startswith('#'):
                formatted_lines.append(stripped)
                continue
            
            # 处理大括号缩进
            if stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
            
            # 添加适当的缩进
            if stripped.startswith('//') or stripped.startswith('/*'):
                # 注释保持原样
                formatted_lines.append('    ' * indent_level + stripped)
            else:
                formatted_lines.append('    ' * indent_level + stripped)
            
            # 更新缩进级别
            if stripped.endswith('{') and not stripped.startswith('}'):
                indent_level += 1
        
        return '\n'.join(formatted_lines)
    
    def _fix_common_issues(self, code: str) -> str:
        """修复常见问题"""
        # 确保头文件包含正确
        if '<gtest/gtest.h>' not in code:
            code = '#include <gtest/gtest.h>\n' + code
        
        # 修复TEST_F宏的格式
        code = re.sub(r'TEST_F\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)',
                     r'TEST_F(\1, \2)', code)
        
        # 确保文件末尾有换行
        if not code.endswith('\n'):
            code += '\n'
        
        return code


class CodeValidator:
    """代码验证器"""
    
    def __init__(self):
        self.required_elements = {
            'headers': ['<gtest/gtest.h>'],
            'macros': ['TEST_F', 'TEST'],
            'assertions': ['ASSERT_', 'EXPECT_'],
            'class': 'class.*:.*public.*testing::Test'
        }
    
    def validate(self, code: str, operator_name: Optional[str] = None) -> Dict[str, any]:
        """
        验证代码的完整性和正确性
        
        Args:
            code: 要验证的代码
            operator_name: 算子名称
        
        Returns:
            dict: 验证结果
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'metrics': {}
        }
        
        # 基本检查
        self._check_required_elements(code, results)
        
        # 结构检查
        self._check_structure(code, results)
        
        # 算子特定检查
        if operator_name:
            self._check_operator_specific(code, operator_name, results)
        
        # 代码度量
        self._calculate_metrics(code, results)
        
        # 编译性检查（可选）
        self._check_compilability(code, results)
        
        # 更新整体验证状态
        results['valid'] = len(results['errors']) == 0
        
        return results
    
    def _check_required_elements(self, code: str, results: Dict):
        """检查必需元素"""
        # 检查头文件
        for header in self.required_elements['headers']:
            if header not in code:
                results['errors'].append(f"缺少必需的头文件: {header}")
        
        # 检查测试宏
        has_test_macro = any(macro in code for macro in self.required_elements['macros'])
        if not has_test_macro:
            results['errors'].append("未找到测试宏 (TEST_F 或 TEST)")
        
        # 检查断言
        has_assertion = any(assertion in code for assertion in self.required_elements['assertions'])
        if not has_assertion:
            results['warnings'].append("未找到断言语句 (ASSERT_* 或 EXPECT_*)")
        
        # 检查测试类
        if not re.search(self.required_elements['class'], code):
            results['warnings'].append("未找到标准的测试类定义")
    
    def _check_structure(self, code: str, results: Dict):
        """检查代码结构"""
        # 检查大括号匹配
        open_braces = code.count('{')
        close_braces = code.count('}')
        if open_braces != close_braces:
            results['errors'].append(f"大括号不匹配: {open_braces} 个 '{{' vs {close_braces} 个 '}}'")
        
        # 检查括号匹配
        open_parens = code.count('(')
        close_parens = code.count(')')
        if open_parens != close_parens:
            results['warnings'].append(f"括号可能不匹配: {open_parens} 个 '(' vs {close_parens} 个 ')'")
    
    def _check_operator_specific(self, code: str, operator_name: str, results: Dict):
        """检查算子特定内容"""
        # 检查测试类名称
        expected_class = f"{operator_name}Tiling"
        if expected_class not in code:
            results['warnings'].append(f"测试类名称可能不正确，期望包含: {expected_class}")
        
        # 检查是否调用了tiling函数
        if 'tiling' not in code.lower():
            results['warnings'].append("未找到tiling函数调用")
    
    def _calculate_metrics(self, code: str, results: Dict):
        """计算代码度量"""
        lines = code.split('\n')
        
        results['metrics'] = {
            'total_lines': len(lines),
            'code_lines': len([l for l in lines if l.strip() and not l.strip().startswith('//')]),
            'test_count': len(re.findall(r'TEST_?F?\s*\(', code)),
            'assertion_count': len(re.findall(r'(ASSERT_|EXPECT_)\w+', code)),
            'function_count': len(re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*\{', code))
        }
        
        # 添加建议
        if results['metrics']['test_count'] < 3:
            results['suggestions'].append("建议增加更多测试用例以提高覆盖率")
        
        if results['metrics']['assertion_count'] < results['metrics']['test_count']:
            results['suggestions'].append("某些测试可能缺少断言")
    
    def _check_compilability(self, code: str, results: Dict):
        """检查代码的编译性（需要g++）"""
        try:
            # 创建临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # 尝试编译（仅语法检查）
            result = subprocess.run(
                ['g++', '-fsyntax-only', '-std=c++11', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode != 0:
                # 解析编译错误
                error_lines = result.stderr.split('\n')
                for line in error_lines[:5]:  # 只显示前5个错误
                    if 'error:' in line:
                        results['warnings'].append(f"编译警告: {line.strip()}")
            
            # 清理临时文件
            os.unlink(temp_file)
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # g++不可用或超时，跳过编译检查
            pass
        except Exception as e:
            logger.debug(f"编译检查失败: {str(e)}")


class PostProcessor:
    """主后处理器"""
    
    def __init__(self):
        self.extractor = CodeExtractor()
        self.cleaner = CodeCleaner()
        self.validator = CodeValidator()
    
    def process(self, raw_response: str, operator_name: Optional[str] = None) -> Tuple[str, Dict]:
        """
        处理模型的原始响应
        
        Args:
            raw_response: 模型的原始响应
            operator_name: 算子名称
        
        Returns:
            tuple: (处理后的代码, 验证结果)
        """
        logger.info("开始后处理生成的代码...")
        
        # 步骤1: 提取代码
        logger.info("步骤1: 提取C++代码")
        extracted_code = self.extractor.extract(raw_response)
        
        # 步骤2: 清理代码
        logger.info("步骤2: 清理和格式化代码")
        cleaned_code = self.cleaner.clean(extracted_code)
        
        # 步骤3: 验证代码
        logger.info("步骤3: 验证代码结构")
        validation_results = self.validator.validate(cleaned_code, operator_name)
        
        # 步骤4: 添加文件头
        if operator_name:
            final_code = self._add_header(cleaned_code, operator_name)
        else:
            final_code = cleaned_code
        
        # 输出验证结果
        self._print_validation_results(validation_results)
        
        return final_code, validation_results
    
    def _add_header(self, code: str, operator_name: str) -> str:
        """添加文件头注释"""
        header = f"""/**
 * 自动生成的{operator_name}算子单元测试
 * 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
 * 
 * 注意: 此代码由AI自动生成，使用前请进行人工审核和测试
 */

"""
        return header + code
    
    def _print_validation_results(self, results: Dict):
        """输出验证结果"""
        if results['errors']:
            logger.error("代码验证发现错误:")
            for error in results['errors']:
                logger.error(f"  ❌ {error}")
        
        if results['warnings']:
            logger.warning("代码验证发现警告:")
            for warning in results['warnings']:
                logger.warning(f"  ⚠️  {warning}")
        
        if results['suggestions']:
            logger.info("改进建议:")
            for suggestion in results['suggestions']:
                logger.info(f"  💡 {suggestion}")
        
        if results['metrics']:
            logger.info("代码统计:")
            for key, value in results['metrics'].items():
                logger.info(f"  📊 {key}: {value}")
        
        if results['valid']:
            logger.info("✅ 代码验证通过")
        else:
            logger.error("❌ 代码验证失败，需要手动修复")


def process_file(input_file: str, output_file: str, 
                operator_name: Optional[str] = None) -> bool:
    """
    处理输入文件，生成最终的单测代码文件
    
    Args:
        input_file: 输入的原始响应文件
        output_file: 输出的单测代码文件
        operator_name: 算子名称
    
    Returns:
        bool: 处理是否成功
    """
    try:
        # 读取输入文件
        input_path = Path(input_file)
        if not input_path.exists():
            logger.error(f"输入文件不存在: {input_file}")
            return False
        
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_response = f.read()
        
        if not raw_response.strip():
            logger.error("输入文件为空")
            return False
        
        logger.info(f"读取原始响应文件: {input_file}")
        logger.info(f"原始响应长度: {len(raw_response):,} 字符")
        
        # 处理响应
        processor = PostProcessor()
        final_code, validation_results = processor.process(raw_response, operator_name)
        
        # 保存结果
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_code)
        
        logger.info(f"最终代码已保存到: {output_file}")
        logger.info(f"最终代码长度: {len(final_code):,} 字符")
        
        # 保存验证报告
        report_file = output_path.with_suffix('.validation.json')
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        logger.info(f"验证报告已保存到: {report_file}")
        
        return validation_results['valid']
        
    except Exception as e:
        logger.error(f"处理文件失败: {str(e)}")
        return False


def main():
    """主函数"""
    if len(sys.argv) < 3:
        print("用法: python post_processor.py <原始响应文件> <输出文件> [算子名称]")
        print("示例: python post_processor.py raw_response.txt result.cpp AllGatherMatmul")
        return
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    operator_name = sys.argv[3] if len(sys.argv) > 3 else None
    
    # 处理文件
    success = process_file(input_file, output_file, operator_name)
    
    if success:
        logger.info("✅ 代码后处理完成")
        sys.exit(0)
    else:
        logger.error("❌ 代码后处理失败")
        sys.exit(1)


if __name__ == "__main__":
    main()