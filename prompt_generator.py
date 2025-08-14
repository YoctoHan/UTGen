#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt生成器 - 面向过程版本
收集C++源码文件并生成结构化的prompt
"""

import sys
from pathlib import Path
from utils import get_cpp_files, read_file_content, read_csv_content, save_file_content


def generate_prompt(source_paths, ut_template_path=None, csv_file_path=None, output_file=None):
    """
    生成prompt文本
    
    Args:
        source_paths: 源码路径列表
        ut_template_path: UT模板文件路径
        csv_file_path: 参考输入参数CSV文件路径
        output_file: 输出文件路径
    
    Returns:
        str: 生成的prompt内容
    """
    print("正在收集C++源码文件...")
    cpp_files = get_cpp_files(source_paths)
    
    if not cpp_files:
        print("未找到任何C++文件")
        return ""
    
    print(f"找到 {len(cpp_files)} 个C++文件")
    
    # 构建prompt内容
    prompt_lines = [
        "## 任务目标",
        "为这个算子类写一个完整的单元测试(UT)，要求：",
        "1. 使用gtest框架",
        "2. 包含必要的头文件和命名空间",
        "6. 使用UT模板中的代码结构",
        "7. 利用输入参数，生成多个UT，以提高覆盖率",
        ""
        "# 以下是目标算子的C++ 算子代码文件内容",
        ""
    ]
    
    total_lines = 0
    valid_files = 0
    
    for file_path in cpp_files:
        content = read_file_content(file_path)
        if not content.strip():
            continue
            
        valid_files += 1
        total_lines += len(content.splitlines())
        
        prompt_lines.extend([
            f"## 文件 {valid_files}: {file_path}",
            "",
            "```cpp",
            content,
            "```",
            ""
        ])
    
    # 添加统计信息
    prompt_lines.extend([
        "---",
        f"总计: {valid_files} 个文件, {total_lines} 行代码",
        "",
        "请为这个算子类写一个完整的单元测试(UT)，要求：",
        "1. 使用gtest框架",
        "2. 包含必要的头文件和命名空间",
        "6. 使用UT模板中的代码结构",
        "7. 尽可能多的生成不同的输入参数，生成多个UT，以提高覆盖率",
        ""
    ])
    
    # 添加UT模板
    if ut_template_path and Path(ut_template_path).exists():
        template_content = read_file_content(Path(ut_template_path))
        if template_content:
            prompt_lines.extend([
                "以下是UT的参考模板：",
                "",
                "```cpp",
                template_content,
                "```",
                ""
            ])
    
    # 添加CSV参考参数
    if csv_file_path and Path(csv_file_path).exists():
        csv_content = read_csv_content(csv_file_path)
        if csv_content:
            prompt_lines.extend([
                "以下是参考的输入参数信息（用于单测的入参）：",
                "",
                csv_content,
                "",
                "请参考以上参数信息设计测试用例的输入参数，确保生成的单测能够覆盖这些参数组合。",
                ""
            ])
    
    prompt_lines.append("请直接输出完整的UT代码，不需要额外的说明文字。")
    
    prompt_content = "\n".join(prompt_lines)
    
    # 保存到文件
    if output_file:
        save_file_content(prompt_content, output_file)
    
    print(f"生成完成: {valid_files} 个文件, {total_lines} 行代码")
    return prompt_content


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python prompt_generator.py <源码路径1> [源码路径2] ... [-t 模板路径] [-c CSV参数文件] [-o 输出文件]")
        print()
        print("参数说明:")
        print("  源码路径      - C++源码目录或文件路径，支持多个")
        print("  -t 模板路径   - UT模板文件路径")
        print("  -c CSV文件    - 参考输入参数CSV文件路径")
        print("  -o 输出文件   - prompt输出文件路径")
        print()
        print("示例:")
        print("  python prompt_generator.py ../cann-ops-adv/src/mc2/all_gather_matmul -o prompt.txt")
        print("  python prompt_generator.py ../src1 ../src2 -t template.cpp -c params.csv -o prompt.txt")
        return
    
    # 解析命令行参数
    source_paths = []
    ut_template = None
    csv_file = None
    output_file = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == '-t' and i + 1 < len(sys.argv):
            ut_template = sys.argv[i + 1]
            i += 2
        elif arg == '-c' and i + 1 < len(sys.argv):
            csv_file = sys.argv[i + 1]
            i += 2
        elif arg == '-o' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        elif not arg.startswith('-'):
            source_paths.append(arg)
            i += 1
        else:
            i += 1
    
    if not source_paths:
        print("错误: 未指定源码路径")
        return
    
    # 生成prompt
    generate_prompt(source_paths, ut_template, csv_file, output_file)


if __name__ == "__main__":
    main() 