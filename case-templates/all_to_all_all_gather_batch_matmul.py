# -*- coding: utf-8 -*-

# AlltoAllAllGatherBatchMatMul UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # 数据类型：x/weight 按 spec.dtype；输出同输入；bias 可通过 spec.bias_dtype/bias_dt_ge 覆盖
    if dtype_to_ge and getattr(spec, "dtype", None) is not None:
        dt_in, dt_out = dtype_to_ge(spec.dtype)
    else:
        dt_in, dt_out = "ge::DT_FLOAT16", "ge::DT_FLOAT16"

    # 形状（支持 2D/3D），若未给出，采用常见默认（来自参考 UT 的首例）
    x = getattr(spec, "x", (16, 128, 64))
    weight = getattr(spec, "weight", (4, 64, 128))
    bias = getattr(spec, "bias", None)
    y = getattr(spec, "y", (4, 512, 64))

    # bias dtype（优先使用 ge 字面量；否则从 bias_dtype 经 dtype_to_ge 推断；再否则与输入一致）
    bias_dt = getattr(spec, "bias_dt_ge", None)
    if bias_dt is None:
        b_dtype = getattr(spec, "bias_dtype", None)
        if b_dtype is not None and dtype_to_ge:
            _, bias_dt = dtype_to_ge(b_dtype)
        else:
            bias_dt = dt_in

    # Node 属性（与参考 UT 对齐）
    group_ep = getattr(spec, "group_ep", "ep_group")
    group_tp = getattr(spec, "group_tp", "tp_group")
    ep_world_size = getattr(spec, "ep_world_size", 4)
    tp_world_size = getattr(spec, "tp_world_size", 2)
    x_shard_type = getattr(spec, "x_shard_type", 1)
    act_type = getattr(spec, "act_type", 0)
    transpose_weight = getattr(spec, "transpose_weight", False)
    output_y2_flag = getattr(spec, "output_y2_flag", False)
    output_y3_flag = getattr(spec, "output_y3_flag", False)

    # Hcom topo 的 rank_size（默认 8，可通过 spec.world_size 覆盖）
    world_size = getattr(spec, "world_size", 8)

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

    # 编译信息 struct 名称（参考 UT 沿用 AllGatherMatmulCompileInfo）
    compile_info_name = getattr(spec, "compile_info_name", "AllGatherMatmulCompileInfo")

    # 是否将 x_shape 置空（用于构造失败用例）
    x_is_null = getattr(spec, "x_is_null", False)

    test_name = spec.name

    def shape_decl(var_name, dims):
        if dims is None:
            return None
        if isinstance(dims, tuple) or isinstance(dims, list):
            if len(dims) == 2:
                return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}<RB>, <LB>{dims[0]}, {dims[1]}<RB><RB>;"
            elif len(dims) == 3:
                return (
                    f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}, {dims[2]}<RB>, "
                    f"<LB>{dims[0]}, {dims[1]}, {dims[2]}<RB><RB>;"
                )
            elif len(dims) == 4:
                return (
                    f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}<RB>, "
                    f"<LB>{dims[0]}, {dims[1]}, {dims[2]}, {dims[3]}<RB><RB>;"
                )
            else:
                return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}<RB>, <LB>{dims[0]}<RB><RB>;"
        else:
            return f"    gert::StorageShape {var_name} = <LB><LB>{dims}<RB>, <LB>{dims}<RB><RB>;"

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
    # 固定 4,2 以对齐参考 UT（仅模拟 compile info 解析，不影响 IO 数）
    lines.append("            .KernelIONum(4, 2)")
    lines.append("            .Inputs(<LB>const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)<RB>)")
    lines.append("            .Outputs(<LB>&compile_info<RB>)")
    lines.append("            .Build();")
    lines.append("")
    lines.append("    // 3. Create context")
    lines.append("    auto param = gert::TilingData::CreateCap(4096);")
    lines.append("    ASSERT_NE(param, nullptr);")
    lines.append("    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);")
    lines.append("    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());")
    # 形状声明
    if not x_is_null:
        lines.append(shape_decl("x_shape", x))
    lines.append(shape_decl("weight_shape", weight))
    if bias is not None:
        lines.append(shape_decl("bias_shape", bias))
    lines.append(shape_decl("y1_output_shape", y))
    lines.append("")
    lines.append("    // 4. Build fake context")
    lines.append("    std::string ep_group(\"" + group_ep + "\");")
    lines.append("    std::string tp_group(\"" + group_tp + "\");")
    lines.append("")
    # 输入输出数量
    has_bias = bias is not None
    if has_bias:
        lines.append("    auto holder = gert::TilingContextFaker()")
        lines.append("                        .NodeIoNum(3, 1)")
        lines.append("                        .IrInstanceNum(<LB>1, 1, 1<RB>)")
        # 输入顺序：x, weight, bias
        input_shapes_parts = []
        input_shapes_parts.append("nullptr" if x_is_null else "&x_shape")
        input_shapes_parts.append("&weight_shape")
        input_shapes_parts.append("&bias_shape")
        lines.append("                        .InputShapes(<LB>" + ", ".join(input_shapes_parts) + "<RB>)")
    else:
        lines.append("    auto holder = gert::TilingContextFaker()")
        lines.append("                        .NodeIoNum(2, 1)")
        lines.append("                        .IrInstanceNum(<LB>1, 1<RB>)")
        lines.append("                        .InputShapes(<LB>" + ("nullptr" if x_is_null else "&x_shape") + ", &weight_shape<RB>)")

    lines.append("                        .OutputShapes(<LB>&y1_output_shape<RB>)")
    # Node Attrs
    lines.append("                        .NodeAttrs(<LB>" + ", ".join([
        "<LB>\\\"group_ep\\\", ge::AnyValue::CreateFrom<std::string>(ep_group)<RB>",
        "<LB>\\\"group_tp\\\", ge::AnyValue::CreateFrom<std::string>(tp_group)<RB>",
        f"<LB>\\\"ep_world_size\\\", ge::AnyValue::CreateFrom<int64_t>({ep_world_size})<RB>",
        f"<LB>\\\"tp_world_size\\\", ge::AnyValue::CreateFrom<int64_t>({tp_world_size})<RB>",
        f"<LB>\\\"x_shard_type\\\", ge::AnyValue::CreateFrom<int64_t>({x_shard_type})<RB>",
        f"<LB>\\\"act_type\\\", ge::AnyValue::CreateFrom<int64_t>({act_type})<RB>",
        f"<LB>\\\"transpose_weight\\\", ge::AnyValue::CreateFrom<bool>({'true' if transpose_weight else 'false'})<RB>",
        f"<LB>\\\"output_y2_flag\\\", ge::AnyValue::CreateFrom<bool>({'true' if output_y2_flag else 'false'})<RB>",
        f"<LB>\\\"output_y3_flag\\\", ge::AnyValue::CreateFrom<bool>({'true' if output_y3_flag else 'false'})<RB>",
    ]) + "><RB>)")
    lines.append("                        .CompileInfo(&compile_info)")
    lines.append("                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    # 输入/输出类型
    if bias is not None:
        lines.append(f"                        .NodeInputTd(0, {{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append(f"                        .NodeInputTd(1, {{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append(f"                        .NodeInputTd(2, {{bias_dt}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    else:
        lines.append(f"                        .NodeInputTd(0, {{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
        lines.append(f"                        .NodeInputTd(1, {{dt_in}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                        .NodeOutputTd(0, {{dt_out}}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append("                        .TilingData(param.get())")
    lines.append("                        .Workspace(ws_size)")
    lines.append("                        .Build();")
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
    # 通信拓扑（对 ep_group 设置）
    lines.append("    ge::HcomTopoInfo::TopoInfo topoInfo;")
    lines.append(f"    topoInfo.rank_size = {world_size};")
    lines.append("    topoInfo.topo_level_descs[0].comm_sets = 0b1U;")
    lines.append("    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(ep_group.c_str(), topoInfo);")
    lines.append("")
    lines.append("    // 6. Call op function")
    lines.append(f"    EXPECT_EQ(tiling_func(tiling_context), {expected_ret});")
    # 取消拓扑
    # lines.append("    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(ep_group.c_str());")
    # 清理环境变量
    for k in env_vars.keys():
        lines.append(f"    unsetenv(\"{k}\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln])
    code = code.replace("<LB>", "{").replace("<RB>", "}")
    code = code.replace("{dt_in}", dt_in).replace("{dt_out}", dt_out).replace("{bias_dt}", bias_dt)
    return code.strip() + "\n"


