#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt生成器 - 面向过程版本
收集C++源码文件并生成结构化的prompt
"""

import sys
from pathlib import Path
from utils import (
    get_cpp_files,
    read_file_content,
    read_csv_content,
    read_excel_content,
    save_file_content,
)


def generate_prompt(source_paths, ut_template_path=None, csv_file_path=None, output_file=None, fewshot_file: str = None, operator_name: str = None, reference_ut_files=None):
    """
    生成prompt文本
    
    Args:
        source_paths: 源码路径列表
        ut_template_path: UT模板文件路径
        csv_file_path: 参考输入参数CSV/XLSX文件路径
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
    ]

    if operator_name:
        prompt_lines.extend([
            f"目标算子名称: {operator_name}",
            ""
        ])

    prompt_lines.extend([
        "# 以下是目标算子的C++ 算子代码文件内容",
        ""
    ])
    
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

    # 添加Stage2 Few-shot参考
    if fewshot_file and Path(fewshot_file).exists():
        fewshot_content = read_file_content(Path(fewshot_file))
        if fewshot_content:
            prompt_lines.extend([
                "以下是与算子相关的Few-shot示例（用于指导UT风格与覆盖点）：",
                "",
                fewshot_content,
                ""
            ])
    
    # 添加参考参数（支持 CSV 或 XLSX）
    if csv_file_path and Path(csv_file_path).exists():
        suffix = Path(csv_file_path).suffix.lower()
        if suffix in {'.xlsx', '.xls'}:
            param_content = read_excel_content(csv_file_path)
        else:
            param_content = read_csv_content(csv_file_path)

        if param_content:
            prompt_lines.extend([
                "以下是参考的输入参数信息（用于单测的入参）：",
                "",
                param_content,
                "",
                "请参考以上参数信息设计测试用例的输入参数，确保生成的单测能够覆盖这些参数组合。",
                ""
            ])

    # 添加历史/参考UT代码
    if reference_ut_files:
        collected = []
        for ref in reference_ut_files:
            p = Path(ref)
            if p.exists() and p.is_file():
                content = read_file_content(p)
                if content:
                    collected.append((str(p), content))
        if collected:
            prompt_lines.extend([
                "以下是与该算子相关的历史/参考UT代码文件，可作为风格与覆盖点参考：",
                ""
            ])
            for file_path, content in collected:
                prompt_lines.extend([
                    f"// 参考文件: {file_path}",
                    "```cpp",
                    content,
                    "```",
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
        print("用法: python prompt_generator.py <源码路径1> [源码路径2] ... [-t 模板路径] [-c CSV/XLSX参数文件] [-f Few-shot文件] [-o 输出文件] [-n 算子名称] [-r 参考UT文件(可多次)]")
        print()
        print("参数说明:")
        print("  源码路径      - C++源码目录或文件路径，支持多个")
        print("  -t 模板路径   - UT模板文件路径")
        print("  -c 参数文件   - 参考输入参数CSV或XLSX文件路径")
        print("  -f Few-shot   - Stage2参考Few-shot文本文件")
        print("  -o 输出文件   - prompt输出文件路径")
        print("  -n 算子名称   - 目标算子名称（用于上下文与检索提示）")
        print("  -r 参考UT     - 历史/参考UT代码文件路径，可多次指定")
        print()
        print("示例:")
        print("  python prompt_generator.py ../cann-ops-adv/src/mc2/all_gather_matmul -o prompt.txt")
        print("  python prompt_generator.py ../src1 ../src2 -t template.cpp -c params.xlsx -o prompt.txt")
        return
    
    # 解析命令行参数
    source_paths = []
    ut_template = None
    csv_file = None
    fewshot_file = None
    output_file = None
    operator_name = None
    reference_ut_files = []
    
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
        elif arg == '-f' and i + 1 < len(sys.argv):
            fewshot_file = sys.argv[i + 1]
            i += 2
        elif arg == '-n' and i + 1 < len(sys.argv):
            operator_name = sys.argv[i + 1]
            i += 2
        elif arg == '-r' and i + 1 < len(sys.argv):
            reference_ut_files.append(sys.argv[i + 1])
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
    generate_prompt(source_paths, ut_template, csv_file, output_file, fewshot_file, operator_name, reference_ut_files)


if __name__ == "__main__":
    main() 