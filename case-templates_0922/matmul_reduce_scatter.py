# -*- coding: utf-8 -*-

# 可选导出 1：函数形式

def render_test_case(op_name, spec, idx, helpers=None):
    ensure_shapes = helpers["ensure_shapes"] if helpers else None
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # MatmulReduceScatter 不需要 gather_output，忽略 ensure_shapes 返回的第三项
    x1, x2, _go, out, bias = ensure_shapes(spec)
    dt_in, dt_out = dtype_to_ge(spec.dtype)

    world_size = getattr(spec, "world_size", 8)
    test_name = spec.name

    tiling_key_check = ""
    if getattr(spec, "expected_tiling_key", None) is not None:
        tiling_key_check = "\n".join([
            "    // 11. Check tiling key",
            "    auto tiling_key = tiling_context->GetTilingKey();",
            f"    ASSERT_EQ(tiling_key, {spec.expected_tiling_key});",
        ])

    reduce_op = getattr(spec, "reduce_op", "sum")

    attr_parts = [
        "<LB>\"group\", ge::AnyValue::CreateFrom<std::string>(group)<RB>",
        f"<LB>\"reduce_op\", ge::AnyValue::CreateFrom<std::string>(std::string(\"{reduce_op}\"))<RB>",
        f"<LB>\"is_trans_a\", ge::AnyValue::CreateFrom<bool>({'true' if spec.is_trans_a else 'false'})<RB>",
        f"<LB>\"is_trans_b\", ge::AnyValue::CreateFrom<bool>({'true' if spec.is_trans_b else 'false'})<RB>",
        f"<LB>\"comm_turn\", ge::AnyValue::CreateFrom<int64_t>({spec.comm_turn})<RB>",
    ]

    # 可选：是否设置 HCCL_DETERMINISTIC 环境变量（例如 "FALSE"）
    hccl_det = getattr(spec, "hccl_deterministic", None)

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
    lines.append(f"    struct {op_name}CompileInfo <LB><RB> compile_info;")
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
    if spec.has_bias:
        lines.append(f"    gert::StorageShape bias_shape = <LB><LB>{bias[0]}<RB>, <LB>{bias[0]}<RB><RB>;")
    lines.append(f"    gert::StorageShape output_shape = <LB><LB>{out[0]}, {out[1]}<RB>, <LB>{out[0]}, {out[1]}<RB><RB>;")
    lines.append("")
    lines.append("    // 5. Build fake context")
    lines.append("    string group(\"group\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                        .NodeIoNum(4, 1)")
    lines.append("                        .IrInstanceNum(<LB>1, 1, 1, 1<RB>)")
    if spec.has_bias:
        lines.append("                        .InputShapes(<LB>&x1_shape, &x2_shape, &bias_shape, nullptr<RB>)")
    else:
        lines.append("                        .InputShapes(<LB>&x1_shape, &x2_shape, nullptr, nullptr<RB>)")
    lines.append("                        .OutputShapes(<LB>&output_shape<RB>)")
    lines.append(f"                        .NodeAttrs(<LB>{', '.join(attr_parts)}<RB>)")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    lines.append(f"                        .NodeInputTd(0, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(1, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeInputTd(2, ge::{{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(0, ge::{{dt_out}}, ge::FORMAT_ND, ge::FORMAT_ND)")
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
    if hccl_det is not None:
        lines.append(f"    setenv(\"HCCL_DETERMINISTIC\", \"{hccl_det}\", 1);")
    lines.append("    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);")
    lines.append("")
    lines.append("    // 9. Call op function, check returns == GRAPH_SUCCESS")
    lines.append("    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);")
    lines.append("")
    lines.append("    // 10. Unset communication")
    lines.append("    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());")
    if hccl_det is not None:
        lines.append("    unsetenv(\"HCCL_DETERMINISTIC\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln != ""]).replace("<LB>", "{").replace("<RB>", "}")
    code = code.replace("{dt_in}", dt_in).replace("{dt_out}", dt_out)
    return code.strip() + "\n"



