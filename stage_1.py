#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试用例生成器 - Stage 1
基于few-shot示例生成目标算子的测试参数
"""

import sys
import json
import csv
import io
from pathlib import Path
from typing import List, Dict, Any, Optional
from utils import (
    get_cpp_files, read_file_content,
    ModelCaller, save_xlsx_content, save_file_content,
    logger, validate_path
)

def load_fewshot_examples(fewshot_file: str) -> str:
    """
    从文件加载few-shot示例
    
    Args:
        fewshot_file: few-shot示例文件路径
    
    Returns:
        str: 示例内容
    """
    fewshot_path = Path(fewshot_file)
    if not fewshot_path.exists():
        logger.warning(f"Few-shot示例文件不存在: {fewshot_file}")
        return ""
    
    try:
        with open(fewshot_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"成功加载few-shot示例文件: {fewshot_file}")
        logger.info(f"示例文件大小: {len(content)} 字符")
        return content
    except Exception as e:
        logger.error(f"读取few-shot示例文件失败: {str(e)}")
        return ""


class TestcasePromptGenerator:
    """测试用例提示词生成器"""
    
    def __init__(self, model_caller: ModelCaller):
        self.template = self._load_template()
        self.model_caller = model_caller
    
    def _load_template(self) -> str:
        """加载提示词模板"""
        return """# 为{operator_name}算子生成测试用例参数

## 任务目标
根据{operator_name}算子的源码和以下示例，生成一套完整的测试参数。
输出格式为CSV，包含测试用例名称和各种参数组合。

## 参考示例
{examples_section}

## 目标算子相关信息
### 算子完整源码
{source_code_section}

## 生成要求

### 1. 参数设计原则
- **覆盖性**: 确保测试用例覆盖算子的所有关键功能路径
- **边界测试**: 包含最小值、最大值、边界条件
- **性能测试**: 包含不同规模的数据测试
- **异常处理**: 包含可能触发异常的参数组合

### 2. 测试用例类型
请生成以下类型的测试用例：
- 请尝试理解源码和示例中的 tiling key，这是生成优质测试用例的关键
- 请参考示例中的输出形式，并生成类似的测试用例

### 3. 参数命名规范
- 使用清晰的参数名称，与源码中的变量名保持一致
- 测试用例名称应描述测试目的，请参考示例中的命名方式
- 数值参数使用合理的范围和步长

## 特殊要求

### tiling-key 取值范围说明

- **是否带 bias**  
  - 不带 bias：不加额外值  
  - 带 bias（存在第 3 个可选输入）：`+1`

- **是否 ND2NZ**  
  - 当前实现固定为 `1` （`SetSocParam` 将 `isND2NZ` 置为 `1`）：`+10`

- **通信算法**  
  - 当前实现固定为 `FULL_MESH`（`SetCommAlg`）：`+100`

### 计算规则
- **不带 bias** → `tilingKey = 110`  
- **带 bias** → `tilingKey = 111`


### 4. 输出格式
CSV格式，第一行为列名，格式示例：
```csv
test_name,param1,param2,param3,...
basic_small,64,128,256,...
boundary_min,1,1,1,...
```

请直接输出CSV内容，至少生成8-10个测试用例：
"""
    
    def generate(self, operator_name: str, source_paths: List[str], 
                fewshot_content: str, operator_info: Optional[Dict] = None) -> str:
        """
        生成测试用例提示词
        
        Args:
            operator_name: 算子名称
            source_paths: 源码路径列表
            fewshot_content: few-shot示例内容
            operator_info: 算子额外信息
        
        Returns:
            str: 生成的提示词
        """
        
        # 生成源码部分
        source_code_section = self._generate_source_section(source_paths)
       
        # 分析算子特征
        operator_analysis = self._analyze_operator(source_code_section)
        # breakpoint()
        # 生成示例部分
        examples_section = self._generate_examples_section(fewshot_content)

        # 生成特殊注意事项
        # operator_analysis = self._generate_special_notes(operator_name, operator_info)
        
        # 填充模板
        prompt = self.template.format(
            operator_name=operator_name,
            operator_analysis=operator_analysis,
            source_code_section=source_code_section,
            examples_section=examples_section,
        )
        return prompt
    
    def _analyze_operator(self, source_code_section: str) -> str:
        """分析算子特征"""
        
        system_message = """你是一位资深的AI算子开发专家，精通各类深度学习框架的CANN算子实现。
        你具备以下专业能力：
        - 深入理解算子的计算逻辑和优化策略
        - 熟悉tiling技术在算子性能优化中的应用
        - 能够准确识别和解释代码中的关键参数含义"""
        
        prompt = f"""请分析以下算子源代码，重点关注tiling相关的实现：
    源代码：
    {source_code_section}
    请做以下分析：
1. 识别代码中和tiling key相关的代码
2. 解释每个tiling key的具体含义和作用

输出格式要求：
- 枚举可能出现的 tiling key，并给出其含义。"""
        response = self.model_caller.call(prompt, system_message, temperature=0.3)  # 降低temperature以获得更准确的技术分析
        return response

    def _generate_source_section(self, source_paths: List[str]) -> str:
        """生成源码部分"""
        lines = []
        cpp_files = get_cpp_files(source_paths)
        
        if cpp_files:
            logger.info(f"收集到 {len(cpp_files)} 个源码文件")
            
            for i, file_path in enumerate(cpp_files, 1):
                content = read_file_content(file_path)
                if content.strip():
                    lines.extend([
                        f"#### 源码文件 {i}:",
                        "```cpp\n",
                        content,
                        "\n```",
                        ""
                    ])
            
        else:
            lines.append("未找到目标算子源码文件")
        
        return "\n".join(lines)
    
    def _generate_examples_section(self, fewshot_content: str) -> str:
        """生成示例部分"""
        if not fewshot_content:
            return "未提供few-shot示例"
        
        lines = []
        lines.extend([
            "以下是相关算子的实现示例，请参考其代码结构和参数设计模式：",
            fewshot_content,
            ""
        ])
        
        return "\n".join(lines)


def parse_csv_response(response: str) -> List[str]:
    """
    解析模型响应中的CSV内容
    
    Args:
        response: 模型响应
    
    Returns:
        list: CSV行列表
    """
    lines = response.split('\n')
    csv_lines = []
    in_csv_block = False
    
    for line in lines:
        # 检测CSV代码块
        if line.strip().startswith('```csv'):
            in_csv_block = True
            continue
        elif line.strip() == '```' and in_csv_block:
            in_csv_block = False
            continue
        elif line.strip().startswith('```'):
            in_csv_block = False
            continue
        
        # 收集CSV内容
        if in_csv_block and line.strip():  # 过滤空行
            csv_lines.append(line)
        elif not in_csv_block and line.strip() and is_likely_csv_line(line):
            csv_lines.append(line)
    
    # 验证CSV格式
    if csv_lines:
        validated_lines = validate_csv_format(csv_lines)
        if validated_lines:
            return validated_lines
    
    return csv_lines


def is_likely_csv_line(line: str) -> bool:
    """
    判断一行是否可能是CSV数据
    """
    line = line.strip()
    
    # 跳过空行和注释
    if not line or line.startswith('#') or line.startswith('//'):
        return False
    
    # 尝试用csv模块解析
    try:
        reader = csv.reader(io.StringIO(line))
        fields = next(reader)
        
        # 至少要有2个字段
        if len(fields) < 2:
            return False
        
        # 检查是否有有效内容（不全是空字段）
        if all(not field.strip() for field in fields):
            return False
            
        return True
        
    except:
        return False


def validate_csv_format(lines: List[str]) -> List[str]:
    """
    验证并返回有效的CSV行
    """
    if not lines:
        return []
    
    # 尝试解析第一行来确定列数
    try:
        reader = csv.reader(io.StringIO(lines[0]))
        first_row_fields = next(reader)
        expected_columns = len(first_row_fields)
        
        # 检查是否有表头
        if not any(char.isdigit() for char in lines[0]):
            logger.info(f"检测到CSV表头，共{expected_columns}列")
        
        # 验证所有行
        valid_lines = []
        for i, line in enumerate(lines):
            try:
                reader = csv.reader(io.StringIO(line))
                fields = next(reader)
                
                # 严格检查列数一致性
                if len(fields) == expected_columns:
                    valid_lines.append(line)
                else:
                    logger.warning(f"第{i+1}行列数不匹配: 期望{expected_columns}列，实际{len(fields)}列")
                    
            except Exception as e:
                logger.warning(f"第{i+1}行解析失败: {e}")
                
        return valid_lines
        
    except Exception as e:
        logger.error(f"CSV格式验证失败: {e}")
        return lines


def generate_testcase_params(operator_name: str, source_paths: List[str], 
                            output_file: str, prompt_file: str,
                            fewshot_file: str, api_key: str, 
                            base_url: str, model_name: str) -> bool:
    """
    生成测试用例参数的主函数
    
    Args:
        operator_name: 算子名称
        source_paths: 源码路径列表
        output_file: 输出Excel文件路径（XLSX格式）
        prompt_file: 提示词文件路径
        fewshot_file: few-shot示例文件路径
        api_key: API密钥
        base_url: API基础URL
        model_name: 模型名称
    
    Returns:
        bool: 是否成功
    """
    logger.info(f"🎯 开始为{operator_name}算子生成测试参数")
    logger.info("=" * 50)
    
    # 初始化模型调用器
    model_caller = ModelCaller(api_key, base_url, model_name, use_cache=True)

    # 加载few-shot示例
    logger.info("📚 加载few-shot示例...")
    fewshot_content = load_fewshot_examples(fewshot_file)
    if not fewshot_content:
        logger.warning("未能加载few-shot示例，将仅基于源码生成")
    
    # 初始化提示词生成器
    prompt_generator = TestcasePromptGenerator(model_caller)
    
    # 生成prompt
    logger.info("📝 生成测试参数生成prompt...")
    prompt = prompt_generator.generate(operator_name, source_paths, fewshot_content)
    
    # 保存prompt到文件
    logger.info("💾 保存prompt到文件...")
    if not save_file_content(prompt, prompt_file):
        logger.warning("prompt保存失败，但继续执行...")
    
    logger.info("🤖 调用模型生成测试参数...")
    
    system_message = """你是一个专业的C++测试工程师，专门为算子设计测试参数。
请根据提供的算子代码和示例，生成全面的测试参数集。
直接输出CSV格式的数据，确保参数覆盖各种测试场景。
第一行必须是列名，后续行是具体的测试数据。"""
    
    response = model_caller.call(prompt, system_message, temperature=0.7)
    
    if not response:
        logger.error("模型调用失败")
        return False
    
    # 解析CSV响应
    logger.info("📊 解析生成的测试参数...")
    csv_lines = parse_csv_response(response)
    
    if not csv_lines:
        logger.error("未能从响应中提取有效的CSV内容")
        logger.debug(f"原始响应前500字符: {response[:500]}")
        return False
    
    # 保存为Excel文件（XLSX格式）
    success = save_xlsx_content(csv_lines, output_file)
    
    if success:
        logger.info("✅ 测试参数生成完成!")
        logger.info(f"📄 输出文件: {output_file}")
        logger.info(f"📝 Prompt文件: {prompt_file}")
        
        # 显示预览
        if csv_lines:
            logger.info("\n📋 生成内容预览:")
            for i, line in enumerate(csv_lines[:5]):
                print(f"  {i+1}: {line}")
            if len(csv_lines) > 5:
                print(f"  ... 还有 {len(csv_lines) - 5} 行")
    
    return success


def main():
    """主函数"""
    if len(sys.argv) < 9:
        print("用法: python stage_1.py <算子名称> <输出Excel文件> <Prompt文件> <Few-shot文件> <API_KEY> <BASE_URL> <MODEL_NAME> <源码路径1> [源码路径2] ...")
        print()
        print("示例:")
        print("  python stage_1.py AllGatherMatmul test_params.xlsx prompt_testcase.txt \\")
        print("    tiling-examples/fewshot_examples.txt \\")
        print("    your_api_key https://api.com/v1 model-name \\")
        print("    ../cann-ops-adv/src/mc2/all_gather_matmul")
        return
    
    operator_name = sys.argv[1]
    output_file = sys.argv[2]
    prompt_file = sys.argv[3]
    fewshot_file = sys.argv[4]
    api_key = sys.argv[5]
    base_url = sys.argv[6]
    model_name = sys.argv[7]
    source_paths = sys.argv[8:]

    # 验证few-shot文件路径
    if not Path(fewshot_file).exists():
        logger.warning(f"Few-shot文件不存在: {fewshot_file}，将使用默认路径")
        # 尝试使用默认路径
        default_fewshot = "tiling-examples/fewshot_examples.txt"
        if Path(default_fewshot).exists():
            fewshot_file = default_fewshot
            logger.info(f"使用默认few-shot文件: {fewshot_file}")

    # 验证源码路径
    valid_paths = []
    for path in source_paths:
        validated = validate_path(path, must_exist=True)
        if validated:
            valid_paths.append(str(validated))
        else:
            logger.warning(f"源码路径不存在: {path}")
    
    if not valid_paths:
        logger.error("没有有效的源码路径")
        sys.exit(1)
    
    # 生成测试参数
    success = generate_testcase_params(
        operator_name, valid_paths, output_file, prompt_file,
        fewshot_file, api_key, base_url, model_name
    )
    
    if not success:
        logger.error("❌ 测试参数生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main()