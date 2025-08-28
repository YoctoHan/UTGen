#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 - 基于Stage1产物与历史UT，生成gtest单测代码

职责：
- 搜集输入：源码路径、算子名、stage1参数文件、历史UT
- 生成Prompt（复用 prompt_generator.py）
- 调用模型（复用 model_caller.py）
- 后处理生成 gtest（复用 post_processor.py）

依赖的环境变量（由 config.sh 注入）：
- RUNS_DIR, TEST_EXAMPLES_DIR
- UT_TEMPLATE_FILE, FEWSHOT_STAGE2_FILE
- PROMPT_GENERATOR, MODEL_CALLER, POST_PROCESSOR
- API_KEY, BASE_URL, MODEL_NAME
"""

import os
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from utils import (
    validate_path,
    create_timestamped_dir,
    logger,
    ModelCaller,
    save_file_content,
    read_file_content
)


class TemplateExtractor:
    """从现有UT文件中抽取模板"""

    def __init__(self, model_caller: ModelCaller):
        self.model_caller = model_caller

    def extract_template(self, ut_file_path: str, operator_name: str) -> str:
        """
        从UT文件中抽取模板

        Args:
            ut_file_path: UT文件路径
            operator_name: 算子名称

        Returns:
            str: 提取的模板内容
        """
        logger.info(f"📝 从 {ut_file_path} 中抽取模板...")

        # 读取UT文件内容
        ut_content = read_file_content(ut_file_path)
        if not ut_content:
            logger.warning("无法读取UT文件内容")
            return ""

        # 构建抽取模板的提示词
        system_message = """你是一个专业的C++测试工程师，精通gtest框架。
你的任务是从给定的单元测试文件中抽取模板部分，包括：
1. 头文件包含部分
2. 命名空间声明
3. 测试类的定义
4. 通用的setup和teardown代码
5. 所有不会随测试参数变化的固定代码

请移除具体的测试用例代码，只保留模板结构。"""

        prompt = f"""请从以下{operator_name}算子的单元测试文件中抽取模板：

{ut_content}

要求：
1. 保留所有头文件包含
2. 保留命名空间声明
3. 保留测试类定义
4. 保留通用的初始化和清理代码
5. 移除具体的TEST_F测试用例
6. 用占位符标记需要填充测试参数的位置
7. 确保模板可以重复使用

请直接输出抽取的模板代码："""

        response = self.model_caller.call(prompt, system_message, temperature=0.3)

        if response:
            logger.info("✅ 模板抽取完成")
            logger.debug(f"模板长度: {len(response)} 字符")
        else:
            logger.error("❌ 模板抽取失败")

        return response or ""


class TestGenerator:
    """基于模板生成单个测试用例"""

    def __init__(self, model_caller: ModelCaller):
        self.model_caller = model_caller

    def generate_test_case(self, template: str, operator_name: str,
                          test_params: dict, test_name: str) -> str:
        """
        生成单个测试用例

        Args:
            template: 模板内容
            operator_name: 算子名称
            test_params: 测试参数字典
            test_name: 测试用例名称

        Returns:
            str: 生成的测试用例代码
        """
        logger.info(f"🤖 生成测试用例: {test_name}")

        # 构建生成测试用例的提示词
        system_message = f"""你是一个专业的C++测试工程师，精通gtest框架。
你的任务是基于提供的模板和测试参数生成完整的TEST_F测试用例。

要求：
1. 使用提供的模板作为基础
2. 根据测试参数填充具体的数值
3. 生成完整的TEST_F函数
4. 确保代码语法正确
5. 保持与模板一致的代码风格"""

        # 格式化测试参数
        params_str = "\n".join([f"- {key}: {value}" for key, value in test_params.items()])

        prompt = f"""基于以下模板和测试参数生成完整的TEST_F测试用例：

## 模板：
{template}

## 测试参数：
{params_str}

## 算子名称：
{operator_name}

## 测试用例名称：
{test_name}

请生成完整的TEST_F函数代码，直接输出代码内容："""

        response = self.model_caller.call(prompt, system_message, temperature=0.5)

        if response:
            logger.info(f"✅ 测试用例 {test_name} 生成完成")
        else:
            logger.error(f"❌ 测试用例 {test_name} 生成失败")

        return response or ""


def parse_param_file(param_file: Path) -> List[dict]:
    """
    解析参数文件，返回参数列表

    Args:
        param_file: 参数文件路径

    Returns:
        List[dict]: 参数字典列表
    """
    import pandas as pd

    try:
        if param_file.suffix.lower() == '.xlsx':
            df = pd.read_excel(param_file)
        elif param_file.suffix.lower() == '.csv':
            df = pd.read_csv(param_file)
        else:
            logger.error(f"不支持的参数文件格式: {param_file.suffix}")
            return []

        # 转换为字典列表
        params_list = []
        for _, row in df.iterrows():
            params = {}
            for col in df.columns:
                params[col] = row[col]
            params_list.append(params)

        logger.info(f"📊 解析到 {len(params_list)} 组测试参数")
        return params_list

    except Exception as e:
        logger.error(f"解析参数文件失败: {e}")
        return []


def to_lower(name: str) -> str:
    return (name or "").lower()


def latest_file(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    try:
        return max(paths, key=lambda p: p.stat().st_mtime)
    except Exception:
        return None


def discover_param_file(runs_dir: Path, operator_lower: str) -> Optional[Path]:
    candidates: List[Path] = []
    candidates += list(runs_dir.glob(f"*/test_params_{operator_lower}.xlsx"))
    candidates += list(runs_dir.glob(f"*/test_params_{operator_lower}.csv"))
    return latest_file(candidates)


def camel_to_snake(name: str) -> str:
    """将 CamelCase 转换为 snake_case，支持连续大写的情况"""
    # 在 "小写或数字 + 大写" 之间加下划线
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    # 在 "大写 + 大写 + 小写" 之间加下划线（处理缩写+单词的情况）
    s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
    return s2.lower()

def discover_reference_ut(operator_name: str) -> List[Path]:
    base_dir = Path("/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/tests/ut/op_tiling_test")
    snake_name = camel_to_snake(operator_name)
    file_name = f"test_{snake_name}.cpp"
    target_path = base_dir / file_name
    return [target_path] if target_path.exists() else []


def extract_code_block(text: str) -> str:
    """从模型输出中提取```代码块```内容，若无代码块则返回原文本

    优先返回标注了语言的代码块（如cpp、c++、cc），否则返回最长的代码块。
    """
    if not text:
        return ""
    # 正则匹配三引号代码块
    pattern = re.compile(r"```(?:([a-zA-Z0-9_+\-]+))?\n([\s\S]*?)\n```", re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        return text

    # 优先选择C/C++相关语言块
    preferred_langs = {"cpp", "c++", "cc", "c"}
    for m in matches:
        lang = (m.group(1) or "").lower()
        if lang in preferred_langs:
            return m.group(2)

    # 否则返回最长的代码块内容
    longest = max(matches, key=lambda m: len(m.group(2)))
    return longest.group(2)


def sanitize_filename(name: str) -> str:
    """清理文件名中的非法字符"""
    if not name:
        return "case"
    # 替换空白为下划线
    name = re.sub(r"\s+", "_", str(name))
    # 仅保留字母、数字、下划线和中划线
    name = re.sub(r"[^0-9A-Za-z_\-]", "", name)
    # 截断过长文件名
    return name[:120] or "case"


def run_and_log(cmd: List[str], log_path: Path, step: str) -> bool:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = f"[{ts}] >>> {step}: {' '.join(cmd)}\n"
    print(header, end="")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(header)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                print(proc.stdout, end="")
                lf.write(proc.stdout)
            if proc.stderr:
                # 也记录stderr
                print(proc.stderr, end="")
                lf.write(proc.stderr)
            lf.flush()
            return proc.returncode == 0
        except Exception as e:
            err = f"执行失败: {e}\n"
            print(err, end="")
            lf.write(err)
            return False


def stage2(operator_name: str, source_paths: List[str]) -> int:
    # 环境变量
    runs_dir = Path(os.environ.get("RUNS_DIR", "runs"))
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    # 校验关键配置
    if not api_key or not base_url or not model_name:
        print("❌ 缺少模型配置(API_KEY/BASE_URL/MODEL_NAME)")
        return 1

    operator_lower = to_lower(operator_name)

    # 创建运行目录与关键文件
    run_dir = create_timestamped_dir(operator_lower, str(runs_dir))
    log_file = run_dir / "generation.log"
    combined_output = run_dir / f"test_{operator_lower}_tiling.cpp"
    template_file = run_dir / f"template_{operator_lower}.cpp"

    # 记录开头信息
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write("开始时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        lf.write(f"算子名称: {operator_name}\n")
        lf.write(f"源码路径: {' '.join(source_paths)}\n")
        lf.write(f"运行目录: {run_dir}\n")
        lf.write("==============================\n\n")

    # 查找参数文件
    param_file = discover_param_file(runs_dir, operator_lower)
    if param_file:
        print(f"🔍 参考参数文件: {param_file}")
        with open(log_file, "a", encoding="utf-8") as lf:
            lf.write(f"参考参数文件: {param_file}\n")
    else:
        print("❌ 未找到参数文件，无法生成测例")
        return 1

    # 收集参考UT，用于抽取模板
    reference_files = discover_reference_ut(operator_name)
    if not reference_files:
        print("❌ 未找到参考UT文件，无法抽取模板")
        return 1
    print(f"🔎 参考UT文件数: {len(reference_files)}")

    # 初始化模型调用器
    caller = ModelCaller(api_key, base_url, model_name, use_cache=True)
    # 抽取模板（仅取第一个参考UT）
    extractor = TemplateExtractor(caller)
    template_raw = extractor.extract_template(str(reference_files[0]), operator_name)
    template_code = extract_code_block(template_raw)
    if not template_code.strip():
        print("❌ 模板抽取结果为空")
        return 1
    save_file_content(template_code, str(template_file))
    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"模板文件: {template_file}\n")

    # 解析参数文件
    params_list = parse_param_file(param_file)
    if not params_list:
        print("❌ 参数列表为空")
        return 1

    # 逐行参数生成测例
    generator = TestGenerator(caller)
    generated_tests: List[str] = []
    success_count = 0
    for idx, params in enumerate(params_list, start=1):
        # 确定测例名
        test_name = str(params.get("test_name") or params.get("name") or f"case_{idx}")
        test_name_clean = sanitize_filename(test_name)

        # 调用模型生成TEST_F代码
        case_raw = generator.generate_test_case(template_code, operator_name, params, test_name)
        case_code = extract_code_block(case_raw)
        if not case_code.strip():
            logger.warning(f"跳过空测例: {test_name}")
            continue

        # 合成完整cpp（模板 + 单个TEST_F）
        full_code = f"{template_code}\n\n{case_code}\n"
        case_file = run_dir / f"{idx:02d}_{test_name_clean}.cpp"
        if save_file_content(full_code, str(case_file)):
            success_count += 1
            generated_tests.append(case_code)
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(f"生成测例: {case_file}\n")
        else:
            logger.error(f"保存测例失败: {case_file}")

    if success_count == 0:
        print("❌ 未成功生成任何测例文件")
        return 1

    # 生成合并文件（模板 + 全部TEST_F）
    combined_code = template_code + "\n\n" + "\n\n".join(generated_tests) + "\n"
    save_file_content(combined_code, str(combined_output))

    # 成功信息
    try:
        lines = sum(1 for _ in combined_output.open("r", encoding="utf-8")) if combined_output.exists() else 0
    except Exception:
        lines = 0
    print("✅ 单测代码生成完成:", combined_output)
    print(f"📊 生成统计: {lines} 行代码, 单文件数: {success_count}")
    return 0


def main():
    if len(sys.argv) < 3:
        print("用法: python stage_2.py <算子名称> <源码路径1> [源码路径2] ...")
        return 1
    operator_name = sys.argv[1]
    source_paths = sys.argv[2:]
    # 过滤存在的路径
    valid_paths: List[str] = []
    for p in source_paths:
        vp = validate_path(p, must_exist=True)
        if vp:
            valid_paths.append(str(vp))
        else:
            logger.warning(f"源码路径不存在: {p}")
    if not valid_paths:
        print("❌ 没有有效的源码路径")
        return 1
    return stage2(operator_name, valid_paths)


if __name__ == "__main__":
    sys.exit(main())


