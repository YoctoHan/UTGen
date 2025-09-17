#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于参考UT与xlsx参数，纯工程方式生成gtest单测：
- 从参考UT中提取公共部分（包含：头文件、using、测试夹具类等），并移除所有 TEST_F
- 读取 xlsx 每一行参数，渲染为 TEST_F 单测用例
- 输出到 runs/ 时间戳 目录下的 test_<op>_tiling.cpp（也同时生成每个独立用例文件可选）

用法：
  python convert_ut_from_xlsx.py --ref /path/to/test_xxx.cpp --xlsx /path/to/params.xlsx \
    --op AllGatherMatmul [--out /custom/output.cpp] [--name-col test_name]

说明：
- 若未指定 --op，会尝试从测试夹具类名或文件名推断（如 test_all_gather_matmul.cpp -> AllGatherMatmul）
- xlsx 推荐列名（不强制，脚本会尽量兼容）：
  - test_name | name
  - m,k,n（或 x1_shape/x2_shape/gather_output_shape/output_shape 形如 "[1024,2048]"）
  - dtype（float16/bfloat16/bf16）
  - is_trans_a/transpose_a, is_trans_b/transpose_b（True/False/1/0）
  - is_bias（True/False）
  - bias_len（可选，整数）或 bias_shape（可选，如 "[12288]")
  - world_size/rank_size（整数）
  - gather_output（True/False）、gather_index、comm_turn
  - expected_tiling_key/tiling_key（整数，可选）
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
import importlib.util
import inspect
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd

from utils import (
    create_timestamped_dir,
    save_file_content,
    logger,
)


TEST_F_PATTERN = re.compile(r"^\s*TEST_F\s*\(", re.MULTILINE)


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="utf-8", errors="ignore")


def extract_common_prefix(ut_content: str) -> str:
    """提取参考UT中 TEST_F 之前的公共部分（完全移除所有 TEST_F）。"""
    m = TEST_F_PATTERN.search(ut_content)
    if not m:
        # 没有 TEST_F，则整体视为公共部分
        return ut_content.rstrip() + "\n"
    prefix = ut_content[: m.start()]  # 不包含 TEST_F 自身
    return prefix.rstrip() + "\n"


def strip_all_testf_blocks(ut_content: str) -> str:
    """移除所有 TEST_F 块，保留其它代码（以防 TEST_F 中间穿插有辅助函数）。"""
    lines = ut_content.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if TEST_F_PATTERN.match(line):
            # 跳过到匹配的闭合花括号 '}'
            brace = 0
            # 找到第一个 '{'
            # 该行可能包含 '{'
            rest = line
            # 寻找 '{' 并开始计数
            while True:
                lb = rest.find('{')
                if lb != -1:
                    brace += 1
                    rest = rest[lb + 1 :]
                    break
                i += 1
                if i >= len(lines):
                    break
                rest = lines[i]
            # 消耗到配对 '}'
            while i < len(lines) and brace > 0:
                # 扫描整行的花括号计数
                for ch in lines[i]:
                    if ch == '{':
                        brace += 1
                    elif ch == '}':
                        brace -= 1
                i += 1
            # 已跳过一个 TEST_F 块
            continue
        else:
            out.append(line)
            i += 1
    return ''.join(out)


def camel_from_snake(snake: str) -> str:
    parts = [p for p in re.split(r"[_\-]", snake) if p]
    return ''.join(p.capitalize() for p in parts)


def infer_operator_name(ref_path: Path, content: str) -> Optional[str]:
    # 1) 从类名推断：class XxxTiling : public testing::Test
    m = re.search(r"class\s+([A-Za-z0-9_]+)\s*:\s*public\s+testing::Test", content)
    if m:
        name = m.group(1)
        # 去除结尾 Tiling/Something
        name = re.sub(r"Tiling$", "", name)
        if name:
            return name
    # 2) 从文件名推断 test_all_gather_matmul.cpp -> AllGatherMatmul
    base = ref_path.stem  # test_all_gather_matmul
    base = re.sub(r"^test_", "", base)
    if base:
        return camel_from_snake(base)
    return None


def parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return False


def parse_int(v: Any, default: int = 0) -> int:
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def parse_shape(value: Any) -> Optional[Tuple[int, int]]:
    """解析形如 "[1024,2048]"、"1024, 2048"、"1024x2048"、[1024,2048] 的二元形状。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        try:
            return (int(value[0]), int(value[1]))
        except Exception:
            return None
    s = str(value).strip()
    if not s:
        return None
    s = s.strip('[](){}')
    s = s.replace('x', ',').replace('X', ',')
    parts = [p.strip() for p in s.split(',') if p.strip()]
    if len(parts) != 2:
        return None
    try:
        return (int(float(parts[0])), int(float(parts[1])))
    except Exception:
        return None


def parse_shape1d(value: Any) -> Optional[int]:
    """解析一维形状，返回长度，如 "[12288]"、12288、[12288]。"""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            try:
                return int(float(value[0]))
            except Exception:
                return None
        return None
    s = str(value).strip()
    if not s:
        return None
    s = s.strip('[](){}')
    try:
        return int(float(s))
    except Exception:
        return None


def parse_shape_list(value: Any) -> List[Tuple[int, ...]]:
    """解析形如 "[[512,12288],[12288,3904],[3904]]" 的形状列表。
    返回形如 [(512,12288),(12288,3904),(3904,)]，对一维形状返回单元素元组。
    """
    if value is None:
        return []
    if isinstance(value, list):
        result: List[Tuple[int, ...]] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                try:
                    result.append(tuple(int(float(x)) for x in item))
                except Exception:
                    continue
        return result
    s = str(value).strip()
    if not s:
        return []
    # 宽松解析：去除外层括号，按 '],' 分段
    inner = s.strip().strip('[]')
    parts = [p.strip() for p in re.split(r"\],\s*\[", inner)]
    result: List[Tuple[int, ...]] = []
    for p in parts:
        p_clean = p.strip('[]')
        if not p_clean:
            continue
        nums = [x.strip() for x in p_clean.replace('x', ',').replace('X', ',').split(',') if x.strip()]
        try:
            tup = tuple(int(float(x)) for x in nums)
            if tup:
                result.append(tup)
        except Exception:
            continue
    return result


def parse_dtype_list(value: Any) -> List[str]:
    """解析形如 "[FLOAT16,FLOAT16,FLOAT16]" 或 "[DT_FLOAT16,DT_FLOAT16]" 的 dtype 列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().upper() for v in value if str(v).strip()]
    s = str(value).strip()
    if not s:
        return []
    s = s.strip('[]')
    tokens = [t.strip().upper() for t in s.split(',') if t.strip()]
    return tokens


def dtype_to_ge(dtype: str) -> Tuple[str, str]:
    s = (dtype or "").strip().lower()
    # 兼容多种写法
    if s in {"bf16", "bfloat16", "dt_bf16"}:
        return ("DT_BF16", "DT_BF16")
    if s in {"float16", "fp16", "dt_float16"}:
        return ("DT_FLOAT16", "DT_FLOAT16")
    # 默认 float16
    return ("DT_FLOAT16", "DT_FLOAT16")


@dataclass
class CaseSpec:
    name: str
    m: Optional[int]
    k: Optional[int]
    n: Optional[int]
    dtype: str
    is_trans_a: bool
    is_trans_b: bool
    has_bias: bool
    world_size: int
    gather_output: bool
    gather_index: int
    comm_turn: int
    expected_tiling_key: Optional[int]
    # 也允许直接提供形状（优先级高于 m,k,n）
    x1_shape: Optional[Tuple[int, int]] = None
    x2_shape: Optional[Tuple[int, int]] = None
    gather_output_shape: Optional[Tuple[int, int]] = None
    output_shape: Optional[Tuple[int, int]] = None
    # bias 长度（可选，若提供则覆盖默认 out_n）
    bias_len: Optional[int] = None
    # 额外字段（为各算子模板提供注入能力）
    group_ep: str = "ep_group"
    group_tp: str = "tp_group"
    ep_world_size: Optional[int] = None
    ep_rank_id: Optional[int] = None
    moe_expert_num: Optional[int] = None
    tp_world_size: Optional[int] = None
    tp_rank_id: Optional[int] = None
    expert_shard_type: Optional[int] = None
    shared_expert_num: Optional[int] = None
    shared_expert_rank_num: Optional[int] = None
    global_bs: Optional[int] = None
    out_dtype: Optional[int] = None
    comm_quant_mode: Optional[int] = None
    quant_mode: Optional[int] = None
    group_list_type: Optional[int] = None
    expert_token_nums_type: Optional[int] = None
    short_soc_version: Optional[str] = None


def row_to_case(row: Dict[str, Any], idx: int) -> CaseSpec:
    # 名称
    name = str(row.get("test_name") or row.get("name") or row.get("case_name") or f"case_{idx}")
    # m,k,n
    m = row.get("m") if "m" in row else row.get("M")
    k = row.get("k") if "k" in row else row.get("K")
    n = row.get("n") if "n" in row else row.get("N")
    m = parse_int(m, None) if m is not None else None
    k = parse_int(k, None) if k is not None else None
    n = parse_int(n, None) if n is not None else None

    # 形状（可选）
    # 兼容列别名：x_shape/expand_x_shape -> x1_shape，expert_ids_shape -> x2_shape
    x1_shape = parse_shape(row.get("x1_shape") or row.get("x_shape") or row.get("expand_x_shape")) or None
    x2_shape = parse_shape(row.get("x2_shape") or row.get("expert_ids_shape")) or None
    go_shape = parse_shape(row.get("gather_output_shape") or row.get("gather_out_shape")) or None
    out_shape = parse_shape(row.get("output_shape") or row.get("x_output_shape") or row.get("output_x_shape")) or None
    shared_expert_x = parse_shape(row.get("shared_expert_x_shape")) or None

    # 如果提供了 input_tensor_shape（形如 [[M,K],[K,N],[N]]），优先使用
    inputs = parse_shape_list(row.get("input_tensor_shape"))
    if inputs:
        if len(inputs) >= 1 and len(inputs[0]) >= 2:
            x1_shape = (int(inputs[0][0]), int(inputs[0][1]))
        if len(inputs) >= 2 and len(inputs[1]) >= 2:
            x2_shape = (int(inputs[1][0]), int(inputs[1][1]))
        # 第三个可能是一维 bias
        if len(inputs) >= 3 and len(inputs[2]) >= 1:
            # 在 ensure_shapes 中通过 has_bias 决定是否使用
            if out_shape is None:
                # 无法由 inputs 推导 output，这里交给显式列或 mkn
                pass

    # 布尔/标量
    # dtype 优先使用 expand_x_dtype，其次 input_tensor_dtype/output_dtype，最后 dtype 列
    dtype_list_out = parse_dtype_list(row.get("output_dtype"))
    dtype_list_in = parse_dtype_list(row.get("input_tensor_dtype"))
    dtype = (
        str(row.get("expand_x_dtype")).strip() if row.get("expand_x_dtype") else None
    ) or (
        dtype_list_in[0] if dtype_list_in else None
    ) or (
        dtype_list_out[0] if dtype_list_out else None
    ) or str(row.get("dtype") or row.get("DType") or "float16")
    is_trans_a = parse_bool(row.get("is_trans_a") or row.get("transpose_a") or False)
    is_trans_b = parse_bool(row.get("is_trans_b") or row.get("transpose_b") or False)
    # has_bias: 仅由 xlsx 的 is_bias 字段决定，不再进行额外推断
    has_bias = parse_bool(row.get("is_bias") or False)

    # 可选：bias_len 或 bias_shape 指定一维 bias 的长度
    bias_len: Optional[int] = None
    if "bias_len" in row:
        bias_len = parse_int(row.get("bias_len"), None)
    if bias_len is None and row.get("bias_shape") is not None:
        bias_len = parse_shape1d(row.get("bias_shape"))
    world_size = parse_int(row.get("world_size") or row.get("rank_size") or 8, 8)
    gather_output = parse_bool(row.get("gather_output") or (go_shape is not None) or True)
    gather_index = parse_int(row.get("gather_index") or 0, 0)
    comm_turn = parse_int(row.get("comm_turn") or 0, 0)
    expected_tiling_key = row.get("expected_tiling_key")
    if expected_tiling_key is None:
        expected_tiling_key = row.get("tiling_key")
    expected_tiling_key = (
        parse_int(expected_tiling_key, None) if expected_tiling_key is not None else None
    )

    # 额外字段注入：从常见列中提取（不存在则为 None）
    ep_world_size = parse_int(row.get("ep_world_size"), None)
    ep_rank_id = parse_int(row.get("ep_rank_id"), None)
    moe_expert_num = parse_int(row.get("moe_expert_num"), None)
    tp_world_size = parse_int(row.get("tp_world_size"), None)
    tp_rank_id = parse_int(row.get("tp_rank_id"), None)
    expert_shard_type = parse_int(row.get("expert_shard_type"), None)
    shared_expert_num = parse_int(row.get("shared_expert_num"), None)
    shared_expert_rank_num = parse_int(row.get("shared_expert_rank_num"), None)
    global_bs = parse_int(row.get("global_bs"), None)
    out_dtype = parse_int(row.get("out_dtype"), None)
    comm_quant_mode = parse_int(row.get("comm_quant_mode"), None)
    quant_mode = parse_int(row.get("quant_mode"), None)
    group_list_type = parse_int(row.get("group_list_type"), None)
    expert_token_nums_type = parse_int(row.get("expert_token_nums_type"), None)
    short_soc_version = str(row.get("soc_version")).strip() if row.get("soc_version") is not None else None

    spec = CaseSpec(
        name=name,
        m=m,
        k=k,
        n=n,
        dtype=dtype,
        is_trans_a=is_trans_a,
        is_trans_b=is_trans_b,
        has_bias=has_bias,
        world_size=world_size,
        gather_output=gather_output,
        gather_index=gather_index,
        comm_turn=comm_turn,
        expected_tiling_key=expected_tiling_key,
        x1_shape=x1_shape,
        x2_shape=x2_shape,
        gather_output_shape=go_shape,
        output_shape=out_shape,
        bias_len=bias_len,
        ep_world_size=ep_world_size,
        ep_rank_id=ep_rank_id,
        moe_expert_num=moe_expert_num,
        tp_world_size=tp_world_size,
        tp_rank_id=tp_rank_id,
        expert_shard_type=expert_shard_type,
        shared_expert_num=shared_expert_num,
        shared_expert_rank_num=shared_expert_rank_num,
        global_bs=global_bs,
        out_dtype=out_dtype,
        comm_quant_mode=comm_quant_mode,
        quant_mode=quant_mode,
        group_list_type=group_list_type,
        expert_token_nums_type=expert_token_nums_type,
        short_soc_version=short_soc_version,
    )
    # 附加可选字段：shared_expert_x（模板 MoeDistributeCombineV2 使用）
    if shared_expert_x is not None:
        try:
            setattr(spec, "shared_expert_x", shared_expert_x)
        except Exception:
            pass
    return spec


def ensure_shapes(spec: CaseSpec) -> Tuple[Tuple[int, int], Tuple[int, int], Optional[Tuple[int, int]], Tuple[int, int], Optional[Tuple[int]]]:
    """根据 m,k,n 和 flags 计算或补全形状。返回：(x1_shape, x2_shape, gather_output_shape, output_shape, bias_shape)"""
    # 优先：若给出 x1/x2，允许 output_shape 缺省，回退为 x1
    if spec.x1_shape and spec.x2_shape:
        x1 = spec.x1_shape
        x2 = spec.x2_shape
        out = spec.output_shape or x1
        go = spec.gather_output_shape or x1
    else:
        if spec.m is None or spec.k is None or spec.n is None:
            raise ValueError("参数缺失：无法从 m,k,n 推导形状，也未提供显式形状列")
        m, k, n = spec.m, spec.k, spec.n
        x1 = (m, k)
        # B 维度：is_trans_b=True 时 (N,K)，否则 (K,N)
        x2 = (n, k) if spec.is_trans_b else (k, n)
        out = spec.output_shape or (m, n)
        go = spec.gather_output_shape or (m, k)
    if spec.has_bias:
        if spec.bias_len is not None:
            bias_shape = (int(spec.bias_len),)
        else:
            # 回退：使用输出列维度
            bias_shape = (out[1],)
    else:
        bias_shape = None
    return x1, x2, (go if spec.gather_output else None), out, bias_shape


def render_test_case_default(op_name: str, spec: CaseSpec, idx: int) -> str:
    x1, x2, go, out, bias = ensure_shapes(spec)
    dt_in, dt_out = dtype_to_ge(spec.dtype)
    world_size = spec.world_size
    
    test_name = spec.name
    # 期望 tiling key 断言
    tiling_key_check = ""
    if spec.expected_tiling_key is not None:
        tiling_key_check = (
            "\n".join([
                "    // 11. Check tiling key",
                "    auto tiling_key = tiling_context->GetTilingKey();",
                f"    ASSERT_EQ(tiling_key, {spec.expected_tiling_key});",
            ])
        )
    
    # 组装 NodeAttrs 片段（参考UT：使用 ge::AnyValue，且不包含 rank_size 与 gather_output）
    attr_parts = [
        "<LB>\"group\", ge::AnyValue::CreateFrom<std::string>(group)<RB>",
        f"<LB>\"is_trans_a\", ge::AnyValue::CreateFrom<bool>({'true' if spec.is_trans_a else 'false'})<RB>",
        f"<LB>\"is_trans_b\", ge::AnyValue::CreateFrom<bool>({'true' if spec.is_trans_b else 'false'})<RB>",
        f"<LB>\"gather_index\", ge::AnyValue::CreateFrom<int64_t>({spec.gather_index})<RB>",
        f"<LB>\"comm_turn\", ge::AnyValue::CreateFrom<int64_t>({spec.comm_turn})<RB>",
    ]
    
    # 组装测试代码（用占位符规避花括号转义）
    lines: List[str] = []
    lines.append(f"TEST_F({op_name}Tiling, {test_name}) <LB>")
    lines.append("    // 1. Setup interfaces")
    lines.append(f"    std::string op_type(\"{op_name}\");")
    lines.append("    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);")
    lines.append("    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;")
    lines.append("    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;")
    lines.append("")
    lines.append("    // 2. Setup compile info and platform info")
    lines.append("    string compile_info_string = R\"(<LB>")
    lines.append("        \"hardware_info\": <LB>")
    lines.append("            \"BT_SIZE\": 1024,")
    lines.append("            \"load3d_constraints\": \"0\",")
    lines.append("            \"Intrinsic_fix_pipe_l0c2out\": true,")
    lines.append("            \"Intrinsic_data_move_l12ub\": false,")
    lines.append("            \"Intrinsic_data_move_l0c2ub\": false,")
    lines.append("            \"Intrinsic_data_move_out2l1_nd2nz\": true,")
    lines.append("            \"UB_SIZE\": 196608,")
    lines.append("            \"L2_SIZE\": 33554432,")
    lines.append("            \"L1_SIZE\": 524288,")
    lines.append("            \"L0A_SIZE\": 65536,")
    lines.append("            \"L0B_SIZE\": 65536,")
    lines.append("            \"L0C_SIZE\": 131072,")
    lines.append("            \"CORE_NUM\": 20")
    lines.append("        <RB>")
    lines.append("    <RB>)\";")
    lines.append("    map<string, string> soc_infos;")
    lines.append("    map<string, string> aicore_spec;")
    lines.append("    map<string, string> intrinsics;")
    lines.append("    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);")
    lines.append("")
    lines.append("    fe::PlatFormInfos platform_info;")
    lines.append("    platform_info.Init();")
    lines.append(f"    struct {op_name}CompileInfo <LB><RB> compile_info;")
    lines.append("")
    lines.append("    // tilingParseFunc simulate")
    lines.append("    auto kernel_holder =")
    lines.append("        gert::KernelRunContextFaker()")
    lines.append("            .KernelIONum(4, 2)")
    lines.append("            .Inputs(<LB>const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)<RB>)")
    lines.append("            .Outputs(<LB>&compile_info<RB>)")
    lines.append("            .Build();")
    lines.append("")
    lines.append("    // 3. Create context")
    lines.append("    auto param = gert::TilingData::CreateCap(4096);")
    lines.append("    ASSERT_NE(param, nullptr);")
    lines.append("")
    lines.append("    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);")
    lines.append("    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());")
    lines.append("")
    lines.append("    // 4. Define input/output shapes (dims 与 storage_dims 对齐)")
    lines.append(f"    gert::StorageShape x1_shape = <LB><LB>{x1[0]}, {x1[1]}<RB>, <LB>{x1[0]}, {x1[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape x2_shape = <LB><LB>{x2[0]}, {x2[1]}<RB>, <LB>{x2[0]}, {x2[1]}<RB><RB>;")
    if spec.has_bias:
        lines.append(f"    gert::StorageShape bias_shape = <LB><LB>{bias[0]}<RB>, <LB>{bias[0]}<RB><RB>;")
    go_m, go_n = (x1[0], x1[1])
    lines.append(f"    gert::StorageShape gather_output_shape = <LB><LB>{go_m}, {go_n}<RB>, <LB>{go_m}, {go_n}<RB><RB>;")
    lines.append(f"    gert::StorageShape output_shape = <LB><LB>{out[0]}, {out[1]}<RB>, <LB>{out[0]}, {out[1]}<RB><RB>;")
    lines.append("")
    lines.append("    // 5. Build fake context")
    lines.append("    string group(\"group\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                        .NodeIoNum(4, 2)")
    lines.append("                        .IrInstanceNum(<LB>1, 1, 1, 1<RB>)")
    if spec.has_bias:
        lines.append("                        .InputShapes(<LB>&x1_shape, &x2_shape, &bias_shape, nullptr<RB>)")
    else:
        lines.append("                        .InputShapes(<LB>&x1_shape, &x2_shape, nullptr, nullptr<RB>)")
    lines.append("                        .OutputShapes(<LB>&output_shape, &gather_output_shape<RB>)")
    lines.append(f"                        .NodeAttrs(<LB>{', '.join(attr_parts)}<RB>)")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    lines.append(f"                        .NodeInputTd(0, ge::{dt_in}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(1, ge::{dt_in}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(2, ge::{dt_in}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(0, ge::{dt_out}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(1, ge::{dt_out}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .TilingData(param.get())")
    lines.append("                        .Workspace(ws_size)")
    lines.append("                        .Build();")
    lines.append("")
    lines.append("    // 6. Init TilingContext pointer")
    lines.append("    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();")
    lines.append("    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);")
    lines.append("")
    lines.append("    // 7. Set Compile settings")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"SoCInfo\", soc_infos);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreSpec\", aicore_spec);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType(\"AICore\");")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreintrinsicDtypeMap\", intrinsics);")
    lines.append("")
    lines.append("    // 8. Set communication")
    lines.append("    ge::HcomTopoInfo::TopoInfo topoInfo;")
    lines.append(f"    topoInfo.rank_size = {world_size};")
    lines.append("    topoInfo.topo_level_descs[0].comm_sets = 0b1U;")
    lines.append("    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);")
    lines.append("")
    lines.append("    // 9. Call op function, check returns == GRAPH_SUCCESS")
    lines.append("    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);")
    lines.append("")
    lines.append("    // 10. Unset communication")
    lines.append("    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln != ""]).replace("<LB>", "{").replace("<RB>", "}")
    code = re.sub(r"\n\s*\n\s*\n+", "\n\n", code)
    return code.strip() + "\n"


def snake_from_camel(name: str) -> str:
    s1 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    return s2.lower()


def load_case_template_renderer(op_name: str) -> Any:
    """按算子名加载可插拔模板模块，返回可调用的 render(op_name, spec, idx)。

    模板搜索顺序：
    1) $CASE_TEMPLATE_DIR/<CamelCase>.py
    2) $CASE_TEMPLATE_DIR/<snake_case>.py
    3) $CASE_TEMPLATE_DIR/default.py
    若均不存在，回退内置默认实现。
    
    模板模块可导出：
    - 函数 render_test_case(op_name, spec, idx, helpers)
      或
    - 类 Template，包含方法 render_test_case(self, op_name, spec, idx, helpers)
    
    helpers 提供常用工具：ensure_shapes, dtype_to_ge, logger。
    """
    # 解析模板目录：优先使用环境变量；若为相对路径，兼容 cwd 与脚本所在目录；最终回退脚本目录下的 case-templates
    env_dir = os.environ.get("CASE_TEMPLATE_DIR")
    script_dir = Path(__file__).resolve().parent
    probe_dirs: List[Path] = []
    if env_dir:
        p = Path(env_dir)
        probe_dirs.append(p if p.is_absolute() else (Path(os.getcwd()) / p))
        # 兼容相对脚本目录
        if not p.is_absolute():
            probe_dirs.append(script_dir / p)
    else:
        probe_dirs.append(Path(os.getcwd()) / "case-templates")
    # 最终回退：脚本目录
    probe_dirs.append(script_dir / "case-templates")

    base: Optional[Path] = None
    for d in probe_dirs:
        try:
            if d.exists() and d.is_dir():
                base = d
                break
        except Exception:
            continue

    if base is None:
        logger.warning("未找到模板目录，使用内置默认模板")
        def _fallback(op: str, spec: CaseSpec, idx: int) -> str:
            return render_test_case_default(op, spec, idx)
        return _fallback

    candidates: List[Path] = []
    try:
        camel = f"{op_name}.py"
        snake = f"{snake_from_camel(op_name)}.py"
        for filename in (camel, snake, "default.py"):
            p = base / filename
            if p.exists() and p.is_file():
                candidates.append(p)
    except Exception:
        candidates = []

    if not candidates:
        # 返回内置默认
        def _fallback(op: str, spec: CaseSpec, idx: int) -> str:
            return render_test_case_default(op, spec, idx)
        return _fallback

    target = candidates[0]
    try:
        logger.info(f"模板目录: {base}")
        logger.info(f"算子: {op_name}，候选模板: {[str(p.name) for p in candidates]}")
        logger.info(f"选用模板: {target}")
    except Exception:
        pass
    try:
        spec_obj = importlib.util.spec_from_file_location(target.stem, str(target))
        if spec_obj and spec_obj.loader:
            module = importlib.util.module_from_spec(spec_obj)
            spec_obj.loader.exec_module(module)

            helpers = {
                "ensure_shapes": ensure_shapes,
                "dtype_to_ge": dtype_to_ge,
                "logger": logger,
            }

            # 函数导出
            func = getattr(module, "render_test_case", None)
            if callable(func):
                def _call(op: str, s: CaseSpec, i: int) -> str:
                    # 优先尝试四参
                    try:
                        return func(op, s, i, helpers)
                    except TypeError:
                        return func(op, s, i)
                return _call

            # 类导出
            cls = getattr(module, "Template", None)
            if cls is not None:
                try:
                    inst = cls()
                    method = getattr(inst, "render_test_case")
                    if callable(method):
                        def _call2(op: str, s: CaseSpec, i: int) -> str:
                            try:
                                return method(op, s, i, helpers)
                            except TypeError:
                                return method(op, s, i)
                        return _call2
                except Exception:
                    pass
    except Exception as e:
        logger.warning(f"加载模板失败，使用默认模板: {e}")

    def _fallback(op: str, spec: CaseSpec, idx: int) -> str:
        return render_test_case_default(op, spec, idx)
    return _fallback


def load_params(xlsx_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_excel(xlsx_path)
    # 转换为字典列表，并保留原始列名
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        row = {str(col): r[col] for col in df.columns}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="从参考UT和xlsx参数生成gtest单测（纯工程方案）")
    parser.add_argument("--ref", required=True, help="参考UT cpp文件路径（包含完整公共代码与若干TEST_F）")
    parser.add_argument("--xlsx", required=True, help="xlsx参数文件路径")
    parser.add_argument("--op", default=None, help="算子名称（如 AllGatherMatmul），可选，默认自动推断")
    parser.add_argument("--out", default=None, help="输出单文件路径，默认写入 runs/<ts>_<op>/test_<op>_tiling.cpp")
    parser.add_argument("--name-col", default=None, help="测试名称列名，默认自动在 test_name/name 中选择")
    args = parser.parse_args()

    ref_path = Path(args.ref).resolve()
    xlsx_path = Path(args.xlsx).resolve()
    if not ref_path.exists():
        print(f"❌ 参考UT不存在: {ref_path}")
        return 1
    if not xlsx_path.exists():
        print(f"❌ xlsx不存在: {xlsx_path}")
        return 1

    ref_content = read_text(ref_path)

    # 完整移除 TEST_F，以尽量保留所有公共辅助代码
    common_full = strip_all_testf_blocks(ref_content)
    common_prefix = extract_common_prefix(common_full)

    op_name = args.op or infer_operator_name(ref_path, ref_content)
    if not op_name:
        print("❌ 无法推断算子名称，请使用 --op 指定")
        return 1

    try:
        rows = load_params(xlsx_path)
    except Exception as e:
        print(f"❌ 读取xlsx失败: {e}")
        return 1

    if not rows:
        print("❌ xlsx为空，无测试参数")
        return 1

    # 选择模板渲染器
    renderer = load_case_template_renderer(op_name)

    # 生成测例
    cases: List[str] = []
    for idx, row in enumerate(rows, start=1):
        try:
            spec = row_to_case(row, idx)
            case_code = renderer(op_name, spec, idx)
            cases.append(case_code)
        except Exception as e:
            logger.warning(f"跳过第{idx}行: {e}")
            continue

    if not cases:
        print("❌ 未能生成任何测试用例")
        return 1

    combined = common_prefix + "\n\n" + "\n\n".join(cases) + "\n"

    # 输出目标
    if args.out:
        out_path = Path(args.out).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ok = save_file_content(combined, str(out_path))
        if not ok:
            print(f"❌ 写入失败: {out_path}")
            return 1
        print(f"✅ 写入完成: {out_path}")
        return 0

    # 默认 runs 目录
    runs_dir = Path("runs")
    run_dir = create_timestamped_dir(op_name.lower(), str(runs_dir))
    out_path = run_dir / f"test_{op_name.lower()}_tiling.cpp"
    ok = save_file_content(combined, str(out_path))
    if not ok:
        print(f"❌ 写入失败: {out_path}")
        return 1
    print(f"✅ 单测生成完成: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


