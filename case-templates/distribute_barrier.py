# -*- coding: utf-8 -*-

# DistributeBarrier UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # 数据类型：输入/输出同 dtype
    if dtype_to_ge and getattr(spec, "dtype", None) is not None:
        dt_in, dt_out = dtype_to_ge(spec.dtype)
    else:
        dt_in, dt_out = "ge::DT_FLOAT16", "ge::DT_FLOAT16"

    # 形状（默认与参考 UT 对齐）
    x = getattr(spec, "x", (3, 4))
    y = getattr(spec, "y", x)

    # Node 属性
    group = getattr(spec, "group", "group")
    world_size = getattr(spec, "world_size", 16)

    # SoC 版本注入（可选）
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
            "    // 8. Check tiling key",
            "    auto tiling_key = tiling_context->GetTilingKey();",
            f"    ASSERT_EQ(tiling_key, {key_str});",
        ])
    elif key_val is not None:
        tiling_key_check = "\n".join([
            "    // 8. Check tiling key",
            "    auto tiling_key = tiling_context->GetTilingKey();",
            f"    ASSERT_EQ(tiling_key, {key_val});",
        ])

    # 期望返回值
    expected_ret = getattr(spec, "expected_ret", None)
    if expected_ret is None:
        expect_success = getattr(spec, "expect_success", True)
        expected_ret = "ge::GRAPH_SUCCESS" if expect_success else "ge::GRAPH_FAILED"

    # 环境变量（可选）
    env_vars = getattr(spec, "env_vars", {}) or {}

    # 编译信息 struct 名称
    compile_info_name = getattr(spec, "compile_info_name", "DistributeBarrierCompileInfo")

    test_name = spec.name

    def shape_decl(var_name, dims):
        if dims is None:
            return None
        if isinstance(dims, (tuple, list)):
            if len(dims) == 2:
                return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}<RB>, <LB>{dims[0]}, {dims[1]}<RB><RB>;"
            return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}<RB>, <LB>{dims[0]}<RB><RB>;"
        return f"    gert::StorageShape {var_name} = <LB><LB>{dims}<RB>, <LB>{dims}<RB><RB>;"

    lines = []
    lines.append(f"TEST_F({op_name}Tiling, {test_name}) <LB>")
    lines.append("    // 1. Setup interfaces")
    lines.append(f"    std::string op_type(\\\"{op_name}\\\");")
    lines.append("    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);")
    lines.append("    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;")
    lines.append("    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;")
    lines.append("")
    lines.append("    // 2. Setup compile info and platform info")
    lines.append("    string compile_info_string = R\"(<LB>")
    lines.append("        \\\"hardware_info\\\": <LB>")
    lines.append("            \\\"BT_SIZE\\\": 1024,")
    lines.append("            \\\"load3d_constraints\\\": \\\"0\\\",")
    lines.append("            \\\"Intrinsic_fix_pipe_l0c2out\\\": true,")
    lines.append("            \\\"Intrinsic_data_move_l12ub\\\": false,")
    lines.append("            \\\"Intrinsic_data_move_l0c2ub\\\": false,")
    lines.append("            \\\"Intrinsic_data_move_out2l1_nd2nz\\\": true,")
    lines.append("            \\\"UB_SIZE\\\": 196608,")
    lines.append("            \\\"L2_SIZE\\\": 33554432,")
    lines.append("            \\\"L1_SIZE\\\": 524288,")
    lines.append("            \\\"L0A_SIZE\\\": 65536,")
    lines.append("            \\\"L0B_SIZE\\\": 65536,")
    lines.append("            \\\"L0C_SIZE\\\": 131072,")
    lines.append("            \\\"CORE_NUM\\\": 20")
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
    lines.append("            .KernelIONum(1, 1)")
    lines.append("            .Inputs(<LB>const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)<RB>)")
    lines.append("            .Outputs(<LB>&compile_info<RB>)")
    lines.append("            .Build();")
    lines.append("")
    lines.append("    auto param = gert::TilingData::CreateCap(4096);")
    lines.append("    ASSERT_NE(param, nullptr);")
    lines.append("    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);")
    lines.append("    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());")
    lines.append(shape_decl("x_ref_shape", x))
    lines.append(shape_decl("x_ref_output_shape", y))
    lines.append("")
    lines.append("    // 3. Create context")
    lines.append("    std::string group(\"" + group + "\");")
    lines.append(f"    int64_t world_size = {world_size};")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                        .NodeIoNum(1, 1)")
    lines.append("                        .IrInstanceNum(<LB>1<RB>)")
    lines.append("                        .InputShapes(<LB>&x_ref_shape<RB>)")
    lines.append("                        .OutputShapes(<LB>&x_ref_output_shape<RB>)")
    lines.append("                        .NodeAttrs(<LB>" + ", ".join([
        "<LB>\\\"group\\\", ge::AnyValue::CreateFrom<std::string>(group)<RB>",
        "<LB>\\\"world_size\\\", ge::AnyValue::CreateFrom<int64_t>(world_size)<RB>",
    ]) + "><RB>)")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    lines.append(f"                        .NodeInputTd(0, {{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(0, {{dt_out}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .TilingData(param.get())")
    lines.append("                        .Workspace(ws_size)")
    lines.append("                        .Build();")
    lines.append("")
    # 环境变量设置
    for k, v in env_vars.items():
        lines.append(f"    setenv(\"{k}\", \"{v}\", 1);")
    lines.append("")
    lines.append("    // 4. Init TilingContext pointer")
    lines.append("    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();")
    lines.append("    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);")
    if version_lines:
        lines.extend(version_lines)
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"SoCInfo\", soc_infos);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreSpec\", aicore_spec);")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType(\"AICore\");")
    lines.append("    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes(\"AICoreintrinsicDtypeMap\", intrinsics);")
    lines.append("")
    lines.append("    // 5. Call op function")
    lines.append(f"    EXPECT_EQ(tiling_func(tiling_context), {expected_ret});")
    # 清理环境变量
    for k in env_vars.keys():
        lines.append(f"    unsetenv(\"{k}\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln])
    code = code.replace("<LB>", "{").replace("<RB>", "}")
    code = code.replace("{dt_in}", dt_in).replace("{dt_out}", dt_out)
    return code.strip() + "\n"


