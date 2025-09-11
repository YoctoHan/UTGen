# -*- coding: utf-8 -*-

# AlltoAllvGroupedMatMul UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    # 不依赖 ensure_shapes，AlltoAllvGroupedMatMul 的输入/输出与其它算子不同
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # 数据类型：gmm/mm 张量默认 FP16；计数为 INT64；输出默认 FP16
    dt_fp16 = "ge::DT_FLOAT16"
    if dtype_to_ge and getattr(spec, "dtype", None) is not None:
        dt_in, _ = dtype_to_ge(spec.dtype)
        dt_fp16 = dt_in

    test_name = spec.name

    # 组网/属性
    group = getattr(spec, "group", "group")
    ep_world_size = getattr(spec, "ep_world_size", 8)
    send_counts = getattr(spec, "send_counts", [128] * 32)
    recv_counts = getattr(spec, "recv_counts", [128] * 32)
    trans_gmm_weight = getattr(spec, "trans_gmm_weight", False)
    trans_mm_weight = getattr(spec, "trans_mm_weight", False)
    permute_out_flag = getattr(spec, "permute_out_flag", False)
    is_need_mm = getattr(spec, "is_need_mm", True)

    # 形状（支持 2D/3D），若未给出，提供常见默认
    gmm_x = getattr(spec, "gmm_x", (4096, 7168))
    gmm_weight = getattr(spec, "gmm_weight", (4, 7168, 4096))
    mm_x = getattr(spec, "mm_x", (2048, 7168)) if is_need_mm else None
    mm_weight = getattr(spec, "mm_weight", (7168, 64)) if is_need_mm else None

    gmm_y = getattr(spec, "gmm_y", (4096, 4096))
    mm_y = getattr(spec, "mm_y", (2047, 64)) if is_need_mm else None
    permute_out = getattr(spec, "permute_out", None)

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

    # 期望返回值
    expected_ret = getattr(spec, "expected_ret", None)
    if expected_ret is None:
        expect_success = getattr(spec, "expect_success", True)
        expected_ret = "ge::GRAPH_SUCCESS" if expect_success else "ge::GRAPH_FAILED"

    # 可选环境变量设置
    env_vars = getattr(spec, "env_vars", {}) or {}

    # 编译信息 struct 名称
    compile_info_name = getattr(spec, "compile_info_name", "AlltoAllvGroupedMatMulCompileInfo")

    def shape_decl(var_name, dims):
        if dims is None:
            return None
        if len(dims) == 2:
            return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}<RB>, <LB>{dims[0]}, {dims[1]}<RB><RB>;"
        elif len(dims) == 3:
            return (
                f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}, {dims[2]}<RB>, "
                f"<LB>{dims[0]}, {dims[1]}, {dims[2]}<RB><RB>;"
            )
        else:
            # 非2/3维，按一维处理
            return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}<RB>, <LB>{dims[0]}<RB><RB>;"

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
    lines.append("            .KernelIONum(5, 4)")
    lines.append("            .Inputs(<LB>const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)<RB>)")
    lines.append("            .Outputs(<LB>&compile_info<RB>)")
    lines.append("            .Build();")
    lines.append("")
    lines.append("    // 3. Create context")
    lines.append("    auto param = gert::TilingData::CreateCap(8192);")
    lines.append("    ASSERT_NE(param, nullptr);")
    lines.append("    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);")
    lines.append("    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());")
    # 形状声明
    lines.append(shape_decl("gmmX_shape", gmm_x))
    lines.append(shape_decl("gmmWeight_shape", gmm_weight))
    if gmm_y is not None:
        lines.append(shape_decl("gmmY_shape", gmm_y))
    if mm_x is not None:
        lines.append(shape_decl("mmX_shape", mm_x))
    if mm_weight is not None:
        lines.append(shape_decl("mmWeight_shape", mm_weight))
    if mm_y is not None:
        lines.append(shape_decl("mmY_shape", mm_y))
    if permute_out_flag and (permute_out is not None):
        lines.append(shape_decl("permuteOut_shape", permute_out))
    lines.append("")
    lines.append("    // 4. Build fake context")
    lines.append("    std::string group(\"" + group + "\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                      .NodeIoNum(6, 3)")
    lines.append("                      .IrInstanceNum(<LB>1, 1, 1, 1, 1, 1<RB>)")
    # 输入顺序：gmmX, gmmWeight, nullptr, nullptr, mmX, mmWeight
    input_shapes_parts = []
    input_shapes_parts.append("&gmmX_shape")
    input_shapes_parts.append("&gmmWeight_shape")
    input_shapes_parts.append("nullptr")
    input_shapes_parts.append("nullptr")
    input_shapes_parts.append("&mmX_shape" if mm_x is not None else "nullptr")
    input_shapes_parts.append("&mmWeight_shape" if mm_weight is not None else "nullptr")
    lines.append("                      .InputShapes(<LB>" + ", ".join(input_shapes_parts) + "<RB>)")
    # 输出顺序：gmmY, mmY, permuteOut
    output_shapes_parts = []
    output_shapes_parts.append("&gmmY_shape" if gmm_y is not None else "nullptr")
    output_shapes_parts.append("&mmY_shape" if mm_y is not None else "nullptr")
    if permute_out_flag and (permute_out is not None):
        output_shapes_parts.append("&permuteOut_shape")
    else:
        output_shapes_parts.append("nullptr")
    lines.append("                      .OutputShapes(<LB>" + ", ".join(output_shapes_parts) + "<RB>)")
    # Node Attrs
    lines.append("                      .NodeAttrs(<LB>" + ", ".join([
        "<LB>\\\"group\\\", ge::AnyValue::CreateFrom<std::string>(group)<RB>",
        f"<LB>\\\"ep_world_size\\\", ge::AnyValue::CreateFrom<int64_t>({ep_world_size})<RB>",
        "<LB>\\\"send_counts\\\", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)<RB>",
        "<LB>\\\"recv_counts\\\", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)<RB>",
        f"<LB>\\\"trans_gmm_weight\\\", ge::AnyValue::CreateFrom<bool>({'true' if trans_gmm_weight else 'false'})<RB>",
        f"<LB>\\\"trans_mm_weight\\\", ge::AnyValue::CreateFrom<bool>({'true' if trans_mm_weight else 'false'})<RB>",
        f"<LB>\\\"permute_out_flag\\\", ge::AnyValue::CreateFrom<bool>({'true' if permute_out_flag else 'false'})<RB>",
    ]) + "><RB>)")
    lines.append("                      .CompileInfo(&compile_info)")
    lines.append("                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    # 输入/输出类型映射：0 FP16, 1 FP16, 2 INT64, 3 INT64, 4 FP16, 5 FP16；输出均 FP16
    lines.append(f"                      .NodeInputTd(0, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(1, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(4, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(5, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeOutputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .TilingData(param.get())")
    lines.append("                      .Workspace(ws_size)")
    lines.append("                      .Build();")
    lines.append("")
    # 环境变量设置
    for k, v in env_vars.items():
        lines.append(f"    setenv(\"{k}\", \"{v}\", 1);")
    lines.append("")
    lines.append("    // 5. Init TilingContext pointer")
    lines.append("    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();")
    lines.append("    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"SoCInfo\", soc_infos);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreSpec\", aicore_spec);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType(\"AICore\");")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreintrinsicDtypeMap\", intrinsics);")
    if version_lines:
        lines.extend(version_lines)
    lines.append("")
    # 通信拓扑
    lines.append("    ge::HcomTopoInfo::TopoInfo topoInfo;")
    lines.append(f"    topoInfo.rank_size = {ep_world_size};")
    lines.append("    topoInfo.topo_level_descs[0].comm_sets = 0b1U;")
    lines.append("    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);")
    lines.append("")
    lines.append("    // 6. Call op function")
    lines.append(f"    EXPECT_EQ(tiling_func(tiling_context), {expected_ret});")
    # 取消拓扑
    lines.append("    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());")
    # 清理环境变量
    for k in env_vars.keys():
        lines.append(f"    unsetenv(\"{k}\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln])
    code = code.replace("<LB>", "{").replace("<RB>", "}")
    return code.strip() + "\n"


