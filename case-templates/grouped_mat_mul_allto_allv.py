# -*- coding: utf-8 -*-

# GroupedMatMulAlltoAllv UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    # 数据类型映射
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None
    dt_fp16 = "ge::DT_FLOAT16"
    if dtype_to_ge and getattr(spec, "dtype", None) is not None:
        dt_in, _ = dtype_to_ge(spec.dtype)
        dt_fp16 = dt_in

    test_name = spec.name

    # 组/属性（支持 group/hcom 键名）
    group_attr_key = getattr(spec, "group_attr_key", "group")
    group_name = getattr(spec, "group", "group")
    ep_world_size = getattr(spec, "ep_world_size", 8)
    send_counts = getattr(spec, "send_counts", [128] * 32)
    recv_counts = getattr(spec, "recv_counts", [128] * 32)
    trans_gmm_weight = getattr(spec, "trans_gmm_weight", False)
    trans_mm_weight = getattr(spec, "trans_mm_weight", False)

    # 形状（支持 2D/3D），默认与参考 UT 对齐
    gmm_x = getattr(spec, "gmm_x", (4096, 7168))
    gmm_weight = getattr(spec, "gmm_weight", (4, 7168, 4096))
    mm_x = getattr(spec, "mm_x", None)  # 可选分支
    mm_weight = getattr(spec, "mm_weight", None)
    y = getattr(spec, "y", (4096, 4096))
    mm_y = getattr(spec, "mm_y", None)

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
        expect_success = getattr(spec, "expect_success", False)
        expected_ret = "ge::GRAPH_SUCCESS" if expect_success else "ge::GRAPH_FAILED"

    # 可选环境变量设置
    env_vars = getattr(spec, "env_vars", {}) or {}

    # 编译信息 struct 名称
    compile_info_name = getattr(spec, "compile_info_name", "AlltoAllvGroupedMatMulCompileInfo")

    # HcomTopoInfo rank size（默认与 ep_world_size 一致，可通过 spec.world_size 覆盖）
    world_size = getattr(spec, "world_size", ep_world_size)

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
    lines.append("    // platform info")
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
    if mm_x is not None:
        lines.append(shape_decl("mmX_shape", mm_x))
    if mm_weight is not None:
        lines.append(shape_decl("mmWeight_shape", mm_weight))
    lines.append(shape_decl("y_shape", y))
    if mm_y is not None:
        lines.append(shape_decl("mmY_shape", mm_y))
    lines.append("")
    lines.append("    // 4. Build fake context")
    lines.append("    std::string group_name(\"" + group_name + "\");")
    lines.append("")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                      .NodeIoNum(6, 2)")
    lines.append("                      .IrInstanceNum(<LB>1, 1, 1, 1, 1, 1<RB>)")
    # 输入：gmmX, gmmWeight, nullptr, nullptr, mmX, mmWeight
    input_shapes_parts = []
    input_shapes_parts.append("&gmmX_shape")
    input_shapes_parts.append("&gmmWeight_shape")
    input_shapes_parts.append("nullptr")
    input_shapes_parts.append("nullptr")
    input_shapes_parts.append("&mmX_shape" if mm_x is not None else "nullptr")
    input_shapes_parts.append("&mmWeight_shape" if mm_weight is not None else "nullptr")
    lines.append("                      .InputShapes(<LB>" + ", ".join(input_shapes_parts) + "<RB>)")
    # 输出：y, mm_y
    output_shapes_parts = []
    output_shapes_parts.append("&y_shape" if y is not None else "nullptr")
    output_shapes_parts.append("&mmY_shape" if mm_y is not None else "nullptr")
    lines.append("                      .OutputShapes(<LB>" + ", ".join(output_shapes_parts) + "<RB>)")
    # Node Attrs
    node_attrs = [
        f"<LB>\\\"{group_attr_key}\\\", ge::AnyValue::CreateFrom<std::string>(group_name)<RB>",
        f"<LB>\\\"ep_world_size\\\", ge::AnyValue::CreateFrom<int64_t>({ep_world_size})<RB>",
        "<LB>\\\"send_counts\\\", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)<RB>",
        "<LB>\\\"recv_counts\\\", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)<RB>",
        f"<LB>\\\"trans_gmm_weight\\\", ge::AnyValue::CreateFrom<bool>({'true' if trans_gmm_weight else 'false'})<RB>",
        f"<LB>\\\"trans_mm_weight\\\", ge::AnyValue::CreateFrom<bool>({'true' if trans_mm_weight else 'false'})<RB>",
    ]
    lines.append("                      .NodeAttrs(<LB>" + ", ".join(node_attrs) + "><RB>)")
    lines.append("                      .CompileInfo(&compile_info)")
    lines.append("                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    # 输入/输出类型：0 FP16, 1 FP16, 2 INT64, 3 INT64, 4 FP16, 5 FP16；输出均 FP16
    lines.append(f"                      .NodeInputTd(0, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(1, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeInputTd(2, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeInputTd(3, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(4, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(5, {dt_fp16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)")
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
    lines.append(f"    topoInfo.rank_size = {world_size};")
    lines.append("    topoInfo.topo_level_descs[0].comm_sets = 0b1U;")
    lines.append("    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group_name.c_str(), topoInfo);")
    lines.append("")
    lines.append("    // 6. Call op function")
    lines.append(f"    EXPECT_EQ(tiling_func(tiling_context), {expected_ret});")
    # 清理拓扑与环境变量
    # lines.append("    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group_name.c_str());")
    for k in env_vars.keys():
        lines.append(f"    unsetenv(\"{k}\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln])
    code = code.replace("<LB>", "{").replace("<RB>", "}")
    return code.strip() + "\n"


