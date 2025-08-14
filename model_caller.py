#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型调用器 - 面向过程版本
调用API模型生成单测代码并后处理
"""

import sys
from pathlib import Path
from utils import call_model, read_file_content, save_file_content


def call_model_from_file(prompt_file, output_file, api_key, base_url, model_name):
    """
    从prompt文件调用模型，保存原始响应
    
    Args:
        prompt_file: 输入prompt文件
        output_file: 输出文件路径
        api_key: API密钥
        base_url: API基础URL
        model_name: 模型名称
    
    Returns:
        bool: 处理是否成功
    """
    # 读取prompt文件
    prompt = read_file_content(Path(prompt_file))
    if not prompt.strip():
        print("prompt文件为空或读取失败")
        return False
    
    print(f"读取prompt文件: {prompt_file}")
    print(f"Prompt长度: {len(prompt)} 字符")
    
    # 调用模型生成代码
    raw_response = call_model(prompt, api_key, base_url, model_name)
    if not raw_response:
        print("模型生成失败")
        return False
    
    # 保存原始响应到文件
    if save_file_content(raw_response, output_file):
        print(f"响应长度: {len(raw_response)} 字符")
        return True
    else:
        return False


def main():
    """主函数"""
    if len(sys.argv) < 6:
        print("用法: python model_caller.py <prompt文件> <输出文件> <API_KEY> <BASE_URL> <MODEL_NAME>")
        print("示例: python model_caller.py prompt.txt raw_response.txt your_api_key https://api.com/v3 deepseek-v3")
        return
    
    prompt_file = sys.argv[1]
    output_file = sys.argv[2]
    api_key = sys.argv[3]
    base_url = sys.argv[4]
    model_name = sys.argv[5]
    
    # 检查输入文件是否存在
    if not Path(prompt_file).exists():
        print(f"错误: prompt文件不存在: {prompt_file}")
        return
    
    # 调用模型并保存原始响应
    success = call_model_from_file(prompt_file, output_file, api_key, base_url, model_name)
    
    if success:
        print("✅ 模型调用完成，原始响应已保存")
    else:
        print("❌ 模型调用失败")
        sys.exit(1)


if __name__ == "__main__":
    main() 