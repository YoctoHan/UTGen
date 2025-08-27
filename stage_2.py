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
import sys
import subprocess
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from utils import (
    validate_path,
    create_timestamped_dir,
    logger,
)


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


def discover_reference_ut(runs_dir: Path, test_examples_dir: Optional[Path], operator_lower: str) -> List[Path]:
    results: List[Path] = []
    # 1) runs 目录最新 UT
    latest_ut = latest_file(list(runs_dir.glob(f"*/test_{operator_lower}_tiling.cpp")))
    if latest_ut:
        results.append(latest_ut)
    # 2) TEST_EXAMPLES_DIR 模糊匹配
    if test_examples_dir and test_examples_dir.exists():
        for p in sorted(test_examples_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".cpp", ".cc", ".cxx", ".hpp", ".h"}:
                if operator_lower in p.name.lower():
                    results.append(p)
    # 去重保持顺序
    seen = set()
    deduped: List[Path] = []
    for p in results:
        sp = str(p.resolve())
        if sp not in seen:
            seen.add(sp)
            deduped.append(p)
    return deduped


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
    test_examples_dir_env = os.environ.get("TEST_EXAMPLES_DIR")
    test_examples_dir = Path(test_examples_dir_env) if test_examples_dir_env else None

    ut_template = os.environ.get("UT_TEMPLATE_FILE")
    fewshot_file = os.environ.get("FEWSHOT_STAGE2_FILE")
    prompt_generator = os.environ.get("PROMPT_GENERATOR", "prompt_generator.py")
    model_caller = os.environ.get("MODEL_CALLER", "model_caller.py")
    post_processor = os.environ.get("POST_PROCESSOR", "post_processor.py")

    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("BASE_URL")
    model_name = os.environ.get("MODEL_NAME")

    # 校验关键配置
    if not api_key or not base_url or not model_name:
        print("❌ 缺少模型配置(API_KEY/BASE_URL/MODEL_NAME)")
        return 1
    if not ut_template or not Path(ut_template).exists():
        print(f"⚠️  UT模板文件不存在: {ut_template}")

    operator_lower = to_lower(operator_name)

    # 创建运行目录
    run_dir = create_timestamped_dir(operator_lower, str(runs_dir))
    prompt_file = run_dir / f"prompt_{operator_lower}.txt"
    raw_response_file = run_dir / "raw_response.txt"
    output_cpp = run_dir / f"test_{operator_lower}_tiling.cpp"
    log_file = run_dir / "generation.log"

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
        print("ℹ️  未找到参考参数文件")

    # 收集参考UT
    reference_files = discover_reference_ut(runs_dir, test_examples_dir, operator_lower)
    if reference_files:
        print(f"🔎 参考UT文件数: {len(reference_files)}")
    else:
        print("ℹ️  未找到参考UT文件")

    # 步骤1: 生成 Prompt
    prompt_args: List[str] = [
        "python3",
        prompt_generator,
        *source_paths,
        "-t", ut_template,
        "-o", str(prompt_file),
        "-n", operator_name,
    ]
    if fewshot_file and Path(fewshot_file).exists():
        prompt_args += ["-f", fewshot_file]
    if param_file and param_file.exists():
        prompt_args += ["-c", str(param_file)]
    for rf in reference_files:
        prompt_args += ["-r", str(rf)]

    if not run_and_log(prompt_args, log_file, step="生成Prompt"):
        print("❌ Prompt生成失败")
        return 1

    # 步骤2: 调用模型
    model_args = [
        "python3",
        model_caller,
        str(prompt_file),
        str(raw_response_file),
        api_key,
        base_url,
        model_name,
    ]
    if not run_and_log(model_args, log_file, step="调用模型"):
        print("❌ 模型调用失败")
        return 1

    # 步骤3: 后处理
    post_args = [
        "python3",
        post_processor,
        str(raw_response_file),
        str(output_cpp),
        operator_name,
    ]
    if not run_and_log(post_args, log_file, step="后处理代码"):
        print("❌ 后处理失败")
        return 1

    # 成功信息
    if output_cpp.exists():
        try:
            lines = sum(1 for _ in output_cpp.open("r", encoding="utf-8"))
        except Exception:
            lines = 0
        print("✅ 单测代码生成完成:", output_cpp)
        print(f"📊 生成统计: {lines} 行代码")
        return 0

    print("❌ 未生成输出文件")
    return 1


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


