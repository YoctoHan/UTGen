# -*- coding: utf-8 -*-

# MoeDistributeCombineAddRmsNorm UT 模板

def render_test_case(op_name, spec, idx, helpers=None):
    dtype_to_ge = helpers["dtype_to_ge"] if helpers else None

    # 数据类型：按参考UT固定各输入/输出类型；BF16 位置可由 spec.dtype 覆盖
    dt_bf16 = "ge::DT_BF16"
    if dtype_to_ge and getattr(spec, "dtype", None) is not None:
        dt_in, _ = dtype_to_ge(spec.dtype)
        dt_bf16 = dt_in
    dt_i32 = "ge::DT_INT32"
    dt_i64 = "ge::DT_INT64"
    dt_bool = "ge::DT_BOOL"
    dt_f32 = "ge::DT_FLOAT"

    # 形状默认（与参考用例一致，可通过 spec 覆盖）
    A = getattr(spec, "A", 64)
    BS = getattr(spec, "BS", 8)
    K = getattr(spec, "K", 8)
    H = getattr(spec, "H", 7168)
    ep_world_size = getattr(spec, "ep_world_size", 8)
    tp_world_size = getattr(spec, "tp_world_size", 1)

    expand_x = getattr(spec, "expand_x", (A, H))
    expert_ids = getattr(spec, "expert_ids", (BS, K))
    assist_info = getattr(spec, "assist_info", (A * 128,))
    ep_send_counts = getattr(spec, "ep_send_counts", (ep_world_size,))
    expert_scales = getattr(spec, "expert_scales", (BS, K))
    residual_x = getattr(spec, "residual_x", (BS, 1, H))
    gamma = getattr(spec, "gamma", (H,))
    tp_send_counts = getattr(spec, "tp_send_counts", (tp_world_size,))
    x_active_mask = getattr(spec, "x_active_mask", (BS,))
    activation_scale = getattr(spec, "activation_scale", None)
    weight_scale = getattr(spec, "weight_scale", None)
    group_list = getattr(spec, "group_list", None)
    expand_scales = getattr(spec, "expand_scales", None)
    shared_expert_x = getattr(spec, "shared_expert_x", (BS, H))

    y = getattr(spec, "y", (BS, 1, H))
    rstd = getattr(spec, "rstd", (BS, 1, 1))
    x = getattr(spec, "x", (BS, 1, H))

    # 节点属性
    group_ep = getattr(spec, "group_ep", "group_ep")
    ep_rank_id = getattr(spec, "ep_rank_id", 0)
    moe_expert_num = getattr(spec, "moe_expert_num", 8)
    group_tp = getattr(spec, "group_tp", "group_tp")
    tp_rank_id = getattr(spec, "tp_rank_id", 0)
    expert_shard_type = getattr(spec, "expert_shard_type", 0)
    shared_expert_num = getattr(spec, "shared_expert_num", 0)
    shared_expert_rank_num = getattr(spec, "shared_expert_rank_num", 0)
    global_bs = getattr(spec, "global_bs", 0)
    out_dtype = getattr(spec, "out_dtype", 0)
    comm_quant_mode = getattr(spec, "comm_quant_mode", 0)
    group_list_type = getattr(spec, "group_list_type", 0)
    comm_alg = getattr(spec, "comm_alg", "")
    norm_eps = getattr(spec, "norm_eps", 1e-6)

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
            "    // 10. Check tiling key",
            "    auto tiling_key = tiling_context->GetTilingKey();",
            f"    ASSERT_EQ(tiling_key, {key_str});",
        ])
    elif key_val is not None:
        tiling_key_check = "\n".join([
            "    // 10. Check tiling key",
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
    compile_info_name = getattr(spec, "compile_info_name", "MoeEPLBUpdateExpertCompileInfo")

    test_name = spec.name

    def shape_decl(var_name, dims):
        if dims is None:
            return None
        if isinstance(dims, (tuple, list)):
            if len(dims) == 1:
                return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}<RB>, <LB>{dims[0]}<RB><RB>;"
            if len(dims) == 2:
                return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}<RB>, <LB>{dims[0]}, {dims[1]}<RB><RB>;"
            if len(dims) == 3:
                return (
                    f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}, {dims[1]}, {dims[2]}<RB>, "
                    f"<LB>{dims[0]}, {dims[1]}, {dims[2]}<RB><RB>;"
                )
            return f"    gert::StorageShape {var_name} = <LB><LB>{dims[0]}<RB>, <LB>{dims[0]}<RB><RB>;"
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
    lines.append("            .KernelIONum(2, 1)")
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
    lines.append(shape_decl("expand_x_shape", expand_x))
    lines.append(shape_decl("expert_ids_shape", expert_ids))
    lines.append(shape_decl("assist_info_shape", assist_info))
    lines.append(shape_decl("ep_send_counts_shape", ep_send_counts))
    lines.append(shape_decl("expert_scales_shape", expert_scales))
    lines.append(shape_decl("residual_x_shape", residual_x))
    lines.append(shape_decl("gamma_shape", gamma))
    lines.append(shape_decl("tp_send_counts_shape", tp_send_counts))
    if x_active_mask is not None:
        lines.append(shape_decl("x_active_mask_shape", x_active_mask))
    if activation_scale is not None:
        lines.append(shape_decl("activation_scale_shape", activation_scale))
    if weight_scale is not None:
        lines.append(shape_decl("weight_scale_shape", weight_scale))
    if group_list is not None:
        lines.append(shape_decl("group_list_shape", group_list))
    if expand_scales is not None:
        lines.append(shape_decl("expand_scales_shape", expand_scales))
    lines.append(shape_decl("shared_expert_x_shape", shared_expert_x))
    lines.append(shape_decl("y_shape", y))
    lines.append(shape_decl("rstd_shape", rstd))
    lines.append(shape_decl("x_shape", x))
    lines.append("")
    lines.append("    // 4. Build fake context")
    lines.append("    auto holder = gert::TilingContextFaker()")
    lines.append("                      .NodeIoNum(14, 3)")
    lines.append("                      .IrInstanceNum(<LB>1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1<RB>)")
    input_shapes_parts = []
    input_shapes_parts.append("&expand_x_shape")
    input_shapes_parts.append("&expert_ids_shape")
    input_shapes_parts.append("&assist_info_shape")
    input_shapes_parts.append("&ep_send_counts_shape")
    input_shapes_parts.append("&expert_scales_shape")
    input_shapes_parts.append("&residual_x_shape")
    input_shapes_parts.append("&gamma_shape")
    input_shapes_parts.append("&tp_send_counts_shape")
    input_shapes_parts.append("&x_active_mask_shape" if x_active_mask is not None else "nullptr")
    input_shapes_parts.append("&activation_scale_shape" if activation_scale is not None else "nullptr")
    input_shapes_parts.append("&weight_scale_shape" if weight_scale is not None else "nullptr")
    input_shapes_parts.append("&group_list_shape" if group_list is not None else "nullptr")
    input_shapes_parts.append("&expand_scales_shape" if expand_scales is not None else "nullptr")
    input_shapes_parts.append("&shared_expert_x_shape")
    lines.append("                      .InputShapes(<LB>" + ", ".join(input_shapes_parts) + "<RB>)")
    lines.append("                      .OutputShapes(<LB>&y_shape, &rstd_shape, &x_shape<RB>)")
    # Node Attrs
    node_attrs = [
        "<LB>\\\"group_ep\\\", ge::AnyValue::CreateFrom<std::string>(std::string(\"" + group_ep + "\"))<RB>",
        f"<LB>\\\"ep_world_size\\\", ge::AnyValue::CreateFrom<int64_t>({ep_world_size})<RB>",
        f"<LB>\\\"ep_rank_id\\\", ge::AnyValue::CreateFrom<int64_t>({ep_rank_id})<RB>",
        f"<LB>\\\"moe_expert_num\\\", ge::AnyValue::CreateFrom<int64_t>({moe_expert_num})<RB>",
        "<LB>\\\"group_tp\\\", ge::AnyValue::CreateFrom<std::string>(std::string(\"" + group_tp + "\"))<RB>",
        f"<LB>\\\"tp_world_size\\\", ge::AnyValue::CreateFrom<int64_t>({tp_world_size})<RB>",
        f"<LB>\\\"tp_rank_id\\\", ge::AnyValue::CreateFrom<int64_t>({tp_rank_id})<RB>",
        f"<LB>\\\"expert_shard_type\\\", ge::AnyValue::CreateFrom<int64_t>({expert_shard_type})<RB>",
        f"<LB>\\\"shared_expert_num\\\", ge::AnyValue::CreateFrom<int64_t>({shared_expert_num})<RB>",
        f"<LB>\\\"shared_expert_rank_num\\\", ge::AnyValue::CreateFrom<int64_t>({shared_expert_rank_num})<RB>",
        f"<LB>\\\"global_bs\\\", ge::AnyValue::CreateFrom<int64_t>({global_bs})<RB>",
        f"<LB>\\\"out_dtype\\\", ge::AnyValue::CreateFrom<int64_t>({out_dtype})<RB>",
        f"<LB>\\\"comm_quant_mode\\\", ge::AnyValue::CreateFrom<int64_t>({comm_quant_mode})<RB>",
        f"<LB>\\\"group_list_type\\\", ge::AnyValue::CreateFrom<int64_t>({group_list_type})<RB>",
        "<LB>\\\"comm_alg\\\", ge::AnyValue::CreateFrom<std::string>(std::string(\"" + comm_alg + "\"))<RB>",
        f"<LB>\\\"norm_eps\\\", ge::AnyValue::CreateFrom<float>({norm_eps})<RB>",
    ]
    lines.append("                      .NodeAttrs(<LB>" + ", ".join(node_attrs) + "><RB>)")
    lines.append("                      .CompileInfo(&compile_info)")
    lines.append("                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))")
    # 输入类型（与参考UT一致）
    lines.append(f"                      .NodeInputTd(0, {dt_bf16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(1, {dt_i32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(2, {dt_i32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(3, {dt_i32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(4, {dt_f32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(5, {dt_bf16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(6, {dt_bf16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(7, {dt_i32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(8, {dt_bool}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(9, {dt_f32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(10, {dt_f32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(11, {dt_i64}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(12, {dt_f32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeInputTd(13, {dt_bf16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    # 输出类型：y BF16, rstd FLOAT, x BF16
    lines.append(f"                      .NodeOutputTd(0, {dt_bf16}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeOutputTd(1, {dt_f32}, ge::FORMAT_ND, ge::FORMAT_ND)")
    lines.append(f"                      .NodeOutputTd(2, {dt_bf16}, ge::FORMAT_ND, ge::FORMAT_ND)")
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
    lines.append("    // 6. Call op function")
    lines.append(f"    EXPECT_EQ(tiling_func(tiling_context), {expected_ret});")
    # 清理环境变量
    for k in env_vars.keys():
        lines.append(f"    unsetenv(\"{k}\");")
    if tiling_key_check:
        lines.append(tiling_key_check)
    lines.append("<RB>")

    code = "\n".join([ln for ln in lines if ln])
    code = code.replace("<LB>", "{").replace("<RB>", "}")
    return code.strip() + "\n"


