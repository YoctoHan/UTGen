import os
from numpy import outer
import pandas as pd
from pathlib import Path
import argparse

def read_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        # 如果 UTF-8 失败，尝试其他编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except:
            return f"[无法读取文件 {file_path}]"

def xlsx_to_csv_string(xlsx_path):
    """将 xlsx 文件转换为 CSV 格式的字符串"""
    try:
        df = pd.read_excel(xlsx_path, engine='openpyxl')
        return df.to_csv(index=False)
    except Exception as e:
        return f"[无法读取 xlsx 文件: {str(e)}]"

def process_folder_and_xlsx(folder_path, xlsx_path):
    """处理一个文件夹和对应的 xlsx 文件"""
    result = f"\n{'='*50}\n"
    result += f"{'='*50}\n\n"
    
    # 遍历文件夹中的所有文件
    folder = Path(folder_path)
    if not folder.exists():
        return f"[文件夹不存在: {folder_path}]\n"
    
    # 获取所有源码文件并排序
    source_files = []
    for file_path in folder.rglob('*'):
        if file_path.is_file():
            # 跳过一些常见的非源码文件
            skip_extensions = {'.pyc', '.pyo', '.dll', '.so', '.dylib', '.exe', '.bin'}
            if file_path.suffix.lower() not in skip_extensions:
                source_files.append(file_path)
    
    source_files.sort()
    
    # 添加源码文件
    result += "源码文件:\n"
    result += "-" * 30 + "\n"
    
    for file_path in source_files:
        relative_path = file_path.relative_to(folder)
        result += f"\n// {relative_path}\n"
        content = read_file_content(file_path)
        result += content
        if not content.endswith('\n'):
            result += '\n'
    
    # 添加对应的 CSV 输出
    result += "\n\n对应生成物:\n"
    result += "-" * 30 + "\n"
    result += "```csv\n"
    
    if Path(xlsx_path).exists():
        csv_content = xlsx_to_csv_string(xlsx_path)
        result += csv_content
    else:
        result += f"[xlsx 文件不存在: {xlsx_path}]"
    
    result += "\n```\n"
    
    return result

def main():
    # parser = argparse.ArgumentParser(description='将文件夹和 xlsx 文件整理为 few-shot 例子')
    # parser.add_argument('--folders', nargs=3, required=True, help='三个文件夹路径')
    # parser.add_argument('--xlsx', nargs=3, required=True, help='三个 xlsx 文件路径')
    # parser.add_argument('--output', default='fewshot_examples.txt', help='输出文件名 (默认: fewshot_examples.txt)')
    # parser.add_argument('--names', nargs=3, default=['Example 1', 'Example 2', 'Example 3'], 
    #                    help='三个例子的名称 (可选)')
    
    # args = parser.parse_args()

    folders = [
        "/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/all_gather_matmul",
        "/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/matmul_all_reduce",
        "/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/matmul_reduce_scatter"
    ]
    xlsx = [
        "/Users/edy/Desktop/华为/canndev-utgen/utgen/tiling-examples/AllgatherMatmulTilingCases.xlsx",
        "/Users/edy/Desktop/华为/canndev-utgen/utgen/tiling-examples/MatmulAllReduceTilingCases.xlsx",
        "/Users/edy/Desktop/华为/canndev-utgen/utgen/tiling-examples/MatmulReduceScatterTilingCases.xlsx",
    ]
    output = "/Users/edy/Desktop/华为/utgen-v2/tiling-examples/fewshot_examples.txt"
    
    # 确保文件夹和 xlsx 文件数量匹配
    if len(folders) != len(xlsx):
        print("错误: 文件夹数量和 xlsx 文件数量必须相同")
        return
    
    # 处理所有例子
    all_results = ""
    all_results += "Few-shot Examples\n"
    all_results += "=" * 70 + "\n"
    
    for i, (folder, xlsx) in enumerate(zip(folders, xlsx)):
        print(f"处理 {folder} -> {xlsx}")
        result = process_folder_and_xlsx(folder, xlsx)
        all_results += result
        all_results += "\n"
    
    # 写入输出文件
    with open(output, 'w', encoding='utf-8') as f:
        f.write(all_results)
    
    print(f"\n完成! 结果已保存到: {output}")

if __name__ == "__main__":
    main()