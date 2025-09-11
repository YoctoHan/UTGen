# -*- coding: utf-8 -*-

# MoeDistributeCombineV2 UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    ensure_shapes = helpers["ensure_shapes"] if helpers else None
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # 形状：优先使用 ensure_shapes 提供的 x1/x2/out
    # x1 -> expand_x，x2 -> expert_ids，out -> x_output
    if ensure_shapes:
        x1, x2, _go, out, _bias = ensure_shapes(spec)
    else:
        x1 = getattr(spec, "x1", (64, 7168))
        x2 = getattr(spec, "x2", (8, 8))
        out = getattr(spec, "out", (8, 7168))

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

    # tiling key 校验（可选）
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
    moe_expert_num = getattr(spec, "moe_expert_num", 7)
    global_bs = getattr(spec, "global_bs", 0)
    out_dtype = getattr(spec, "out_dtype", 0)
    comm_quant_mode = getattr(spec, "comm_quant_mode", 0)
    group_list_type = getattr(spec, "group_list_type", 0)

    # 输入/输出 shape 推导与覆盖
    expand_x = getattr(spec, "expand_x", (x1[0], x1[1]))
    expert_ids = getattr(spec, "expert_ids", (x2[0], x2[1]))
    expand_idx_len = getattr(spec, "expand_idx_len", expert_ids[0] * expert_ids[1])
    ep_send_counts_len = getattr(spec, "ep_send_counts_len", ep_world_size if ep_world_size > 0 else 1)
    tp_send_counts_len = getattr(spec, "tp_send_counts_len", tp_world_size if tp_world_size > 0 else 1)
    expert_scales = getattr(spec, "expert_scales", (expert_ids[0], expert_ids[1]))
    x_output = getattr(spec, "x_output", (out[0], out[1]))

    # V2 额外可选输入：shared_expert_x（支持 2D/3D），控制是否使用 12 输入形态
    shared_expert_x = getattr(spec, "shared_expert_x", None)
    use_shared_expert_path = shared_expert_x is not None

    # 期望返回值：默认 GRAPH_SUCCESS，可通过 spec.expect_success 或 expected_ret 覆盖
    expected_ret = getattr(spec, "expected_ret", None)
    if expected_ret is None:
        expect_success = getattr(spec, "expect_success", True)
        expected_ret = "ge::GRAPH_SUCCESS" if expect_success else "ge::GRAPH_FAILED"

    # 可选环境变量设置
    env_vars = getattr(spec, "env_vars", {}) or {}

    # compile info struct 名称（可覆盖）
    compile_info_name = getattr(spec, "compile_info_name", "MoeDistributeCombineCompileInfo")

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
    lines.append(f"    struct {compile_info_name} <LB><RB> compile_info;")
    lines.append("")
    lines.append("    // tilingParseFunc simulate")
    lines.append("    auto kernel_holder =")
    lines.append("        gert::KernelRunContextFaker()")
    if use_shared_expert_path:
        lines.append("            .KernelIONum(6, 1)")
    else:
        lines.append("            .KernelIONum(6, 1)")
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
    lines.append(f"    gert::StorageShape expand_x_shape = <LB><LB>{expand_x[0]}, {expand_x[1]}<RB>, <LB>{expand_x[0]}, {expand_x[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape expert_ids_shape = <LB><LB>{expert_ids[0]}, {expert_ids[1]}<RB>, <LB>{expert_ids[0]}, {expert_ids[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape expand_idx_shape = <LB><LB>{expand_idx_len}<RB>, <LB>{expand_idx_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape ep_send_counts_shape = <LB><LB>{ep_send_counts_len}<RB>, <LB>{ep_send_counts_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape tp_send_counts_shape = <LB><LB>{tp_send_counts_len}<RB>, <LB>{tp_send_counts_len}<RB><RB>;")
    lines.append(f"    gert::StorageShape expert_scales_shape = <LB><LB>{expert_scales[0]}, {expert_scales[1]}<RB>, <LB>{expert_scales[0]}, {expert_scales[1]}<RB><RB>;")
    if use_shared_expert_path:
        # shared_expert_x 支持 2D 或 3D，按给定元组长度生成
        if len(shared_expert_x) == 2:
            lines.append(f"    gert::StorageShape shared_expert_x_shape = <LB><LB>{shared_expert_x[0]}, {shared_expert_x[1]}<RB>, <LB>{shared_expert_x[0]}, {shared_expert_x[1]}<RB><RB>;")
        elif len(shared_expert_x) == 3:
            lines.append(f"    gert::StorageShape shared_expert_x_shape = <LB><LB>{shared_expert_x[0]}, {shared_expert_x[1]}, {shared_expert_x[2]}<RB>, <LB>{shared_expert_x[0]}, {shared_expert_x[1]}, {shared_expert_x[2]}<RB><RB>;")
        else:
            # 兜底：按二维处理
            lines.append(f"    gert::StorageShape shared_expert_x_shape = <LB><LB>{shared_expert_x[0]}, {shared_expert_x[1]}<RB>, <LB>{shared_expert_x[0]}, {shared_expert_x[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape x_output_shape = <LB><LB>{x_output[0]}, {x_output[1]}<RB>, <LB>{x_output[0]}, {x_output[1]}<RB><RB>;")
    lines.append("")
    lines.append("    // 5. Build fake context")
    lines.append("    std::string ep_group(\"" + ep_group + "\");")
    lines.append("    std::string tp_group(\"" + tp_group + "\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    if use_shared_expert_path:
        lines.append("                        .NodeIoNum(12, 1)")
        lines.append("                        .IrInstanceNum(<LB>1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1<RB>)")
        lines.append("                        .InputShapes(<LB>&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,\n                                       &expert_scales_shape, &tp_send_counts_shape, nullptr, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape<RB>)")
    else:
        lines.append("                        .NodeIoNum(6, 1)")
        lines.append("                        .IrInstanceNum(<LB>1, 1, 1, 1, 1, 1<RB>)")
        lines.append("                        .InputShapes(<LB>&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,\n                                       &expert_scales_shape, &tp_send_counts_shape<RB>)")
    lines.append("                        .OutputShapes(<LB>&x_output_shape<RB>)")
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
        f"<LB>\"global_bs\", ge::AnyValue::CreateFrom<int64_t>({global_bs})<RB>",
        f"<LB>\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>({out_dtype})<RB>",
        f"<LB>\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>({comm_quant_mode})<RB>",
        f"<LB>\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>({group_list_type})<RB>",
    ]) + "><RB>")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    # 输入/输出 dtype 映射（参考 UT）
    lines.append(f"                        .NodeInputTd(0, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
    if use_shared_expert_path:
        lines.append("                        .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append("                        .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append("                        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append("                        .NodeInputTd(9, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append("                        .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append(f"                        .NodeInputTd(11, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .TilingData(param.get())")
    lines.append("                        .Workspace(ws_size)")
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


