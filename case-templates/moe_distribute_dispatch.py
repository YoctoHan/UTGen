# -*- coding: utf-8 -*-

# MoeDistributeDispatch UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    ensure_shapes = helpers["ensure_shapes"] if helpers else None
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # 形状来源优先级：显式 x1/x2 -> ensure_shapes(spec) -> 兜底
    if getattr(spec, "x1_shape", None) and getattr(spec, "x2_shape", None):
        x1 = spec.x1_shape
        x2 = spec.x2_shape
        out = getattr(spec, "output_shape", None) or x1
    elif ensure_shapes:
        x1, x2, _go, out, _bias = ensure_shapes(spec)
    else:
        # 兜底：从 spec 读取或给出占位形状
        x1 = getattr(spec, "x1", (8, 7168))
        x2 = getattr(spec, "x2", (8, 8))
        out = getattr(spec, "out", (x1[0] * 8, x1[1]))

    dt_in, _dt_out_unused = dtype_to_ge(spec.dtype) if dtype_to_ge else ("ge::DT_FLOAT16", "ge::DT_FLOAT16")

    test_name = spec.name

    # SoC version 注入（可选）
    short_soc_version = getattr(spec, "short_soc_version", None)
    version_lines = []
    if short_soc_version:
        version_lines = [
            "    map<string, string> version = <LB><LB>\"Short_SoC_version\", \"" + short_soc_version + "\"<RB><RB>;",
            "    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"version\", version);",
        ]

    # tiling key 校验（可选，支持字符串或数值）
    tiling_key_check = ""
    key_str = getattr(spec, "expected_tiling_key_str", None)
    key_val = getattr(spec, "expected_tiling_key", None)
    if key_str is not None:
        tiling_key_check = "\n".join([
            "    // 11. Check tiling key",
            "    auto tiling_key = tiling_context->GetTilingKey();",
            f"    ASSERT_EQ(tiling_key, {key_str});",
        ])
    elif key_val is not None:
        tiling_key_check = "\n".join([
            "    // 11. Check tiling key",
            "    auto tiling_key = tiling_context->GetTilingKey();",
            f"    ASSERT_EQ(tiling_key, {key_val});",
        ])

    # Node attrs（与 UT 保持一致命名）
    ep_group = getattr(spec, "group_ep", "ep_group")
    tp_group = getattr(spec, "group_tp", "tp_group")
    ep_world_size = getattr(spec, "ep_world_size", 8)
    tp_world_size = getattr(spec, "tp_world_size", 1)
    ep_rank_id = getattr(spec, "ep_rank_id", 0)
    tp_rank_id = getattr(spec, "tp_rank_id", 0)
    expert_shard_type = getattr(spec, "expert_shard_type", 0)
    shared_expert_num = getattr(spec, "shared_expert_num", 1)
    shared_expert_rank_num = getattr(spec, "shared_expert_rank_num", 1)
    moe_expert_num = getattr(spec, "moe_expert_num", 8)
    quant_mode = getattr(spec, "quant_mode", 0)
    global_bs = getattr(spec, "global_bs", 0)
    expert_token_nums_type = getattr(spec, "expert_token_nums_type", 0)

    # 输出形状：按 UT 规则/经验推导，允许通过 spec 覆盖
    # 0: expand_x_output -> 使用 out 形状
    expand_x_out = getattr(spec, "expand_x_out", (out[0], out[1]))
    scales_shape = getattr(spec, "scales_shape", (out[0], out[1]))
    # 1: dynamic_scales_output -> 一维，长度与 expand_x_out 的第一维一致
    dynamic_scales_len = getattr(spec, "dynamic_scales_len", expand_x_out[0])
    # 2: expand_idx_output -> 一维，等于 expert_ids 的元素数（B * topk）
    expand_idx_len = getattr(spec, "expand_idx_len", x2[0] * x2[1])
    # 3: expert_token_nums_output -> 一维，默认 1，可由 spec 指定
    expert_token_nums_len = getattr(spec, "expert_token_nums_len", 1)
    # 4: ep_recv_count_output -> 一维，长度等于 ep_world_size
    ep_recv_count_len = getattr(spec, "ep_recv_count_len", ep_world_size if ep_world_size > 0 else 1)
    # 5: tp_recv_count_output -> 一维，长度等于 tp_world_size（若 <=0 则置 1）
    tp_recv_count_len = getattr(spec, "tp_recv_count_len", tp_world_size if tp_world_size > 0 else 1)

    # 期望返回值：默认 GRAPH_SUCCESS，可通过 spec.expect_success 或 expected_ret 覆盖
    expected_ret = getattr(spec, "expected_ret", None)
    if expected_ret is None:
        expect_success = getattr(spec, "expect_success", True)
        expected_ret = "ge::GRAPH_SUCCESS" if expect_success else "ge::GRAPH_FAILED"

    # 可选环境变量设置（例如 HCCL_INTRA_PCIE_ENABLE/HCCL_INTRA_ROCE_ENABLE）
    env_vars = getattr(spec, "env_vars", {}) or {}

    lines = []
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
    lines.append("    struct MoeDistributeDispatchCompileInfo <LB><RB> compile_info;")
    lines.append("")
    lines.append("    // tilingParseFunc simulate")
    lines.append("    auto kernel_holder =")
    lines.append("        gert::KernelRunContextFaker()")
    lines.append("            .KernelIONum(3, 6)")
    lines.append("            .Inputs(<LB>const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)<RB>)")
    lines.append("            .Outputs(<LB>&compile_info<RB>)")
    lines.append("            .Build();")
    lines.append("")
    lines.append("    // 3. Create context")
    lines.append("    auto param = gert::TilingData::CreateCap(4096);")
    lines.append("    ASSERT_NE(param, nullptr);")
    lines.append("    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);")
    lines.append("    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());")
    lines.append("")
    lines.append("    // 4. Define input/output shapes (dims 与 storage_dims 对齐)")
    lines.append(f"    gert::StorageShape expand_x_shape = <LB><LB>{x1[0]}, {x1[1]}<RB>, <LB>{x1[0]}, {x1[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape expert_ids_shape = <LB><LB>{x2[0]}, {x2[1]}<RB>, <LB>{x2[0]}, {x2[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape scales_shape = <LB><LB>{scales_shape[0]}, {scales_shape[1]}<RB>, <LB>{scales_shape[0]}, {scales_shape[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape expand_x_output_shape = <LB><LB>{expand_x_out[0]}, {expand_x_out[1]}<RB>, <LB>{expand_x_out[0]}, {expand_x_out[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape dynamic_scales_output_shape = <LB><LB>{dynamic_scales_len}<RB>, <LB>{dynamic_scales_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape expand_idx_output_shape = <LB><LB>{expand_idx_len}<RB>, <LB>{expand_idx_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape expert_token_nums_output_shape = <LB><LB>{expert_token_nums_len}<RB>, <LB>{expert_token_nums_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape ep_recv_count_output_shape = <LB><LB>{ep_recv_count_len}<RB>, <LB>{ep_recv_count_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape tp_recv_count_output_shape = <LB><LB>{tp_recv_count_len}<RB>, <LB>{tp_recv_count_len}<RB><RB>;")
    lines.append("")
    lines.append("    // 5. Build fake context")
    lines.append("    std::string ep_group(\"" + ep_group + "\");")
    lines.append("    std::string tp_group(\"" + tp_group + "\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                        .NodeIoNum(2, 6)")
    lines.append("                        .IrInstanceNum(<LB>1, 1<RB>)")
    lines.append("                        .InputShapes(<LB>&expand_x_shape, &expert_ids_shape<RB>)")
    lines.append("                        .OutputShapes(<LB>&expand_x_output_shape, &dynamic_scales_output_shape, &expand_idx_output_shape,\n                                       &expert_token_nums_output_shape, &ep_recv_count_output_shape,\n                                       &tp_recv_count_output_shape<RB>)")
    lines.append("                        .NodeAttrs(<LB>" + ", ".join([
        "<LB>\"group_ep\", ge::AnyValue::CreateFrom<std::string>(ep_group)<RB>",
        f"<LB>\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>({ep_world_size})<RB>",
        f"<LB>\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>({ep_rank_id})<RB>",
        f"<LB>\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>({moe_expert_num})<RB>",
        "<LB>\"group_tp\", ge::AnyValue::CreateFrom<std::string>(tp_group)<RB>",
        f"<LB>\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>({tp_world_size})<RB>",
        f"<LB>\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>({tp_rank_id})<RB>",
        f"<LB>\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>({expert_shard_type})<RB>",
        f"<LB>\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>({shared_expert_num})<RB>",
        f"<LB>\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>({shared_expert_rank_num})<RB>",
        f"<LB>\"quant_mode\", ge::AnyValue::CreateFrom<int64_t>({quant_mode})<RB>",
        f"<LB>\"global_bs\", ge::AnyValue::CreateFrom<int64_t>({global_bs})<RB>",
        f"<LB>\"expert_token_nums_type\", ge::AnyValue::CreateFrom<int64_t>({expert_token_nums_type})<RB>",
    ]) + "<RB>)")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    lines.append(f"                        .NodeInputTd(0, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(0, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeOutputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeOutputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeOutputTd(4, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeOutputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .TilingData(param.get())")
    lines.append("                        .Workspace(ws_size)")
    lines.append("                        .SetOpType(op_type)")
    lines.append("                        .Build();")
    lines.append("")
    # 环境变量设置
    for k, v in env_vars.items():
        lines.append(f"    setenv(\"{k}\", \"{v}\", 1);")
    lines.append("")
    lines.append("    // 6. Init TilingContext pointer")
    lines.append("    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();")
    lines.append("    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"SoCInfo\", soc_infos);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreSpec\", aicore_spec);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType(\"AICore\");")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreintrinsicDtypeMap\", intrinsics);")
    if version_lines:
        lines.extend(version_lines)
    lines.append("")
    lines.append("    // 7. Call op function")
    lines.append(f"    EXPECT_EQ(tiling_func(tiling_context), {expected_ret});")
    # 清理环境变量
    for k in env_vars.keys():
        lines.append(f"    unsetenv(\"{k}\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln != ""]).replace("<LB>", "{").replace("<RB>", "}")
    code = code.replace("{dt_in}", dt_in)
    return code.strip() + "\n"


