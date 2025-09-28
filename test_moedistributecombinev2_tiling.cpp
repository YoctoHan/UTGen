


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_basic_small) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{64, 1024}, {64, 1024}};
    gert::StorageShape expert_ids_shape = {{64, 2}, {64, 2}};
    gert::StorageShape expand_idx_shape = {{128}, {128}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{64, 2}, {64, 2}};
    gert::StorageShape x_output_shape = {{64, 1024}, {64, 1024}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(16)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(512)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_basic_medium) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{256, 4096}, {256, 4096}};
    gert::StorageShape expert_ids_shape = {{256, 4}, {256, 4}};
    gert::StorageShape expand_idx_shape = {{1024}, {1024}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{256, 4}, {256, 4}};
    gert::StorageShape x_output_shape = {{256, 4096}, {256, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_basic_large) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{512, 8192}, {512, 8192}};
    gert::StorageShape expert_ids_shape = {{512, 8}, {512, 8}};
    gert::StorageShape expand_idx_shape = {{4096}, {4096}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{512, 8}, {512, 8}};
    gert::StorageShape x_output_shape = {{512, 8192}, {512, 8192}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(2)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(4096)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_shared_expert) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{128, 2048}, {128, 2048}};
    gert::StorageShape expert_ids_shape = {{128, 2}, {128, 2}};
    gert::StorageShape expand_idx_shape = {{256}, {256}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{128, 2}, {128, 2}};
    gert::StorageShape shared_expert_x_shape = {{128, 2048}, {128, 2048}};
    gert::StorageShape x_output_shape = {{128, 2048}, {128, 2048}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(12, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape, nullptr, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(16)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(2)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(2)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(1024)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(6, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(7, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(8, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(9, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(11, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 2000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_tp_world_size_1) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{256, 4096}, {256, 4096}};
    gert::StorageShape expert_ids_shape = {{256, 4}, {256, 4}};
    gert::StorageShape expand_idx_shape = {{1024}, {1024}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{1}, {1}};
    gert::StorageShape expert_scales_shape = {{256, 4}, {256, 4}};
    gert::StorageShape x_output_shape = {{256, 4096}, {256, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_tp_world_size_2) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{256, 4096}, {256, 4096}};
    gert::StorageShape expert_ids_shape = {{256, 4}, {256, 4}};
    gert::StorageShape expand_idx_shape = {{1024}, {1024}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{256, 4}, {256, 4}};
    gert::StorageShape x_output_shape = {{256, 4096}, {256, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1100);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_comm_quant_int8) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{256, 4096}, {256, 4096}};
    gert::StorageShape expert_ids_shape = {{256, 4}, {256, 4}};
    gert::StorageShape expand_idx_shape = {{1024}, {1024}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{256, 4}, {256, 4}};
    gert::StorageShape x_output_shape = {{256, 4096}, {256, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(2)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1020);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_max_bs) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{512, 4096}, {512, 4096}};
    gert::StorageShape expert_ids_shape = {{512, 8}, {512, 8}};
    gert::StorageShape expand_idx_shape = {{4096}, {4096}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{512, 8}, {512, 8}};
    gert::StorageShape x_output_shape = {{512, 4096}, {512, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(2)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(4096)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_max_h) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{256, 8192}, {256, 8192}};
    gert::StorageShape expert_ids_shape = {{256, 4}, {256, 4}};
    gert::StorageShape expand_idx_shape = {{1024}, {1024}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{256, 4}, {256, 4}};
    gert::StorageShape x_output_shape = {{256, 8192}, {256, 8192}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_max_k) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 196608,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": 20
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct MoeDistributeCombineCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(6, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape expand_x_shape = {{256, 4096}, {256, 4096}};
    gert::StorageShape expert_ids_shape = {{256, 16}, {256, 16}};
    gert::StorageShape expand_idx_shape = {{4096}, {4096}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{256, 16}, {256, 16}};
    gert::StorageShape x_output_shape = {{256, 4096}, {256, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 7. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000);
}

