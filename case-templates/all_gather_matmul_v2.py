# -*- coding: utf-8 -*-

# AllGatherMatmulV2 UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    ensure_shapes = helpers["ensure_shapes"] if helpers else None
    # 注意：V2 场景通常输入与输出 dtype 不一致，因此不沿用单一 dtype_to_ge

    # 形状：x1, x2, go, out, bias(可忽略)
    x1, x2, go, out, _bias = ensure_shapes(spec)

    world_size = getattr(spec, "world_size", 8)
    test_name = spec.name

    # SoC version 注入（可选）
    short_soc_version = getattr(spec, "short_soc_version", None)
    version_lines = []
    if short_soc_version:
        version_lines = [
            "    map<string, string> socversions = <LB><LB>\"Short_SoC_version\", \"" + short_soc_version + "\"<RB><RB>;",
            "    tiling_context->GetPlatformInfo()->SetPlatformRes(\"version\", socversions);",
        ]

    # tiling key 校验（可选，支持字符串形式如 110UL、或数值）
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

    # V2 额外输入：x1Scale/x2Scale（标量），第三输入与第三输出常为可选（此处默认 nullptr）
    # dtype 默认：x1/x2 为 FP8(E4M3FN)，scale 为 FP32，输出为 FP16；允许通过 spec 覆盖
    dt_x1 = getattr(spec, "dt_x1", "ge::DT_FLOAT8_E4M3FN")
    dt_x2 = getattr(spec, "dt_x2", "ge::DT_FLOAT8_E4M3FN")
    dt_scale = getattr(spec, "dt_scale", "ge::DT_FLOAT")
    dt_out = getattr(spec, "dt_out", "ge::DT_FLOAT16")

    # Node attrs（与UT对齐）
    gather_index = getattr(spec, "gather_index", 0)
    comm_turn = getattr(spec, "comm_turn", 0)
    rank_size_attr = getattr(spec, "rank_size_attr", 0)
    block_size = getattr(spec, "block_size", 0)
    group_size = getattr(spec, "group_size", 0)
    is_gather_out = getattr(spec, "is_gather_out", False)
    is_amax_out = getattr(spec, "is_amax_out", False)
    y_dtype = getattr(spec, "y_dtype", 0)

    attr_parts = [
        "<LB>\"group\", ge::AnyValue::CreateFrom<std::string>(group)<RB>",
        f"<LB>\"is_trans_a\", ge::AnyValue::CreateFrom<bool>({'true' if spec.is_trans_a else 'false'})<RB>",
        f"<LB>\"is_trans_b\", ge::AnyValue::CreateFrom<bool>({'true' if spec.is_trans_b else 'false'})<RB>",
        f"<LB>\"gather_index\", ge::AnyValue::CreateFrom<int64_t>({gather_index})<RB>",
        f"<LB>\"comm_turn\", ge::AnyValue::CreateFrom<int64_t>({comm_turn})<RB>",
        f"<LB>\"rank_size\", ge::AnyValue::CreateFrom<int64_t>({rank_size_attr})<RB>",
        f"<LB>\"block_size\", ge::AnyValue::CreateFrom<int64_t>({block_size})<RB>",
        f"<LB>\"group_size\", ge::AnyValue::CreateFrom<int64_t>({group_size})<RB>",
        f"<LB>\"is_gather_out\", ge::AnyValue::CreateFrom<bool>({'true' if is_gather_out else 'false'})<RB>",
        f"<LB>\"is_amax_out\", ge::AnyValue::CreateFrom<bool>({'true' if is_amax_out else 'false'})<RB>",
        f"<LB>\"y_dtype\", ge::AnyValue::CreateFrom<int64_t>({y_dtype})<RB>",
    ]

    # 允许可选的 amax 输出 shape（若 is_amax_out = true 且提供元素个数）
    amax_elems = getattr(spec, "amax_elems", None)

    # compile_info struct 名称可覆盖
    compile_info_name = getattr(spec, "compile_info_name", f"{op_name}CompileInfo")

    lines = []
    lines.append(f"TEST_F({op_name}Tiling, {test_name}) <LB>")
    lines.append("    // 1. Setup interfaces")
    lines.append(f"    std::string op_type(\"{op_name}\");")
    lines.append("    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);")
    lines.append("    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;")
    lines.append("    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;")
    lines.append("    map<string, string> socversions = <LB><RB>;")
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
    lines.append("            \"UB_SIZE\": 262144,")
    lines.append("            \"L2_SIZE\": 134217728,")
    lines.append("            \"L1_SIZE\": 524288,")
    lines.append("            \"L0A_SIZE\": 65536,")
    lines.append("            \"L0B_SIZE\": 65536,")
    lines.append("            \"L0C_SIZE\": 262144,")
    lines.append("            \"CORE_NUM\": 32")
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
    lines.append("            .KernelIONum(4, 1)")
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
    lines.append(f"    gert::StorageShape gather_output_shape = <LB><LB>{go[0]}, {go[1]}<RB>, <LB>{go[0]}, {go[1]}<RB><RB>;")
    lines.append(f"    gert::StorageShape output_shape = <LB><LB>{out[0]}, {out[1]}<RB>, <LB>{out[0]}, {out[1]}<RB><RB>;")
    lines.append("    gert::StorageShape x1Scale_shape = <LB><LB>1<RB>, <LB>1<RB><RB>;")
    lines.append("    gert::StorageShape x2Scale_shape = <LB><LB>1<RB>, <LB>1<RB><RB>;")
    if is_amax_out and amax_elems:
        lines.append(f"    gert::StorageShape amax_shape = <LB><LB>{amax_elems}<RB>, <LB>{amax_elems}<RB><RB>;")
    lines.append("")
    lines.append("    // 5. Build fake context")
    lines.append("    string group(\"group\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                        .NodeIoNum(6, 3)")
    lines.append("                        .IrInstanceNum(<LB>1, 1, 1, 1, 1, 1<RB>)")
    if is_amax_out and amax_elems:
        lines.append("                        .InputShapes(<LB>&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr<RB>)")
        lines.append("                        .OutputShapes(<LB>&output_shape, &gather_output_shape, &amax_shape<RB>)")
    else:
        lines.append("                        .InputShapes(<LB>&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr<RB>)")
        lines.append("                        .OutputShapes(<LB>&output_shape, &gather_output_shape, nullptr<RB>)")
    lines.append(f"                        .NodeAttrs(<LB>{', '.join(attr_parts)}<RB>)")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    lines.append(f"                        .NodeInputTd(0, {dt_x1}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(1, {dt_x2}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(2, {dt_scale}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(3, {dt_scale}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(4, {dt_scale}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(0, {dt_out}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .TilingData(param.get())")
    lines.append("                        .Workspace(ws_size)")
    lines.append("                        .SetOpType(op_type)")
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
    if version_lines:
        lines.extend(version_lines)
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
    return code.strip() + "\n"


