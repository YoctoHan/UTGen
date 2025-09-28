#include <iostream>
#include <map>
#include <vector>
#include <string>

#include <gtest/gtest.h>
#include "op_log.h"
#define private public

#include "kernel_run_context_facker.h"
#include "fusion_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "external/hcom/hcom_topo_info.h"
#include "test_cube_util.h"

class MoeDistributeCombineV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MoeDistributeCombineV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MoeDistributeCombineV2Tiling TearDown" << std::endl;
    }
};

// normal: share rank


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
    gert::StorageShape expand_x_shape = {{32, 1024}, {32, 1024}};
    gert::StorageShape expert_ids_shape = {{32, 4}, {32, 4}};
    gert::StorageShape expand_idx_shape = {{128}, {128}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{32, 4}, {32, 4}};
    gert::StorageShape x_output_shape = {{32, 1024}, {32, 1024}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(256)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10100);
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
    gert::StorageShape expand_x_shape = {{64, 2048}, {64, 2048}};
    gert::StorageShape expert_ids_shape = {{64, 3}, {64, 3}};
    gert::StorageShape expand_idx_shape = {{192}, {192}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{1}, {1}};
    gert::StorageShape expert_scales_shape = {{64, 3}, {64, 3}};
    gert::StorageShape shared_expert_x_shape = {{64, 2048}, {64, 2048}};
    gert::StorageShape x_output_shape = {{64, 2048}, {64, 2048}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(12, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape, nullptr, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(24)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(2)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(4)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(256)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 11000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_int8_quant) {
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
    gert::StorageShape expand_x_shape = {{16, 4096}, {16, 4096}};
    gert::StorageShape expert_ids_shape = {{16, 2}, {16, 2}};
    gert::StorageShape expand_idx_shape = {{32}, {32}};
    gert::StorageShape ep_send_counts_shape = {{4}, {4}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{16, 2}, {16, 2}};
    gert::StorageShape x_output_shape = {{16, 4096}, {16, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(2)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(16)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(128)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(2)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10120);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_active_mask) {
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
    gert::StorageShape expand_x_shape = {{32, 1024}, {32, 1024}};
    gert::StorageShape expert_ids_shape = {{32, 4}, {32, 4}};
    gert::StorageShape expand_idx_shape = {{128}, {128}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{1}, {1}};
    gert::StorageShape expert_scales_shape = {{32, 4}, {32, 4}};
    gert::StorageShape x_output_shape = {{32, 1024}, {32, 1024}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(3)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(256)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10000);
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
    gert::StorageShape expand_x_shape = {{64, 2048}, {64, 2048}};
    gert::StorageShape expert_ids_shape = {{64, 4}, {64, 4}};
    gert::StorageShape expand_idx_shape = {{256}, {256}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{64, 4}, {64, 4}};
    gert::StorageShape x_output_shape = {{64, 2048}, {64, 2048}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(4)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(512)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10100);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_min_expert) {
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
    gert::StorageShape expand_x_shape = {{8, 1024}, {8, 1024}};
    gert::StorageShape expert_ids_shape = {{8, 1}, {8, 1}};
    gert::StorageShape expand_idx_shape = {{8}, {8}};
    gert::StorageShape ep_send_counts_shape = {{2}, {2}};
    gert::StorageShape tp_send_counts_shape = {{1}, {1}};
    gert::StorageShape expert_scales_shape = {{8, 1}, {8, 1}};
    gert::StorageShape x_output_shape = {{8, 1024}, {8, 1024}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(2)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(64)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_large_hidden) {
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
    gert::StorageShape expand_x_shape = {{16, 8192}, {16, 8192}};
    gert::StorageShape expert_ids_shape = {{16, 4}, {16, 4}};
    gert::StorageShape expand_idx_shape = {{64}, {64}};
    gert::StorageShape ep_send_counts_shape = {{4}, {4}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{16, 4}, {16, 4}};
    gert::StorageShape x_output_shape = {{16, 8192}, {16, 8192}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(16)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(128)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10100);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_max_expert) {
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
    gert::StorageShape expert_ids_shape = {{32, 8}, {32, 8}};
    gert::StorageShape expand_idx_shape = {{256}, {256}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{1}, {1}};
    gert::StorageShape expert_scales_shape = {{32, 8}, {32, 8}};
    gert::StorageShape x_output_shape = {{32, 2048}, {32, 2048}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(5)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(256)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10000);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_shared_int8) {
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
    gert::StorageShape expand_x_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape expert_ids_shape = {{64, 4}, {64, 4}};
    gert::StorageShape expand_idx_shape = {{256}, {256}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{64, 4}, {64, 4}};
    gert::StorageShape shared_expert_x_shape = {{64, 4096}, {64, 4096}};
    gert::StorageShape x_output_shape = {{64, 4096}, {64, 4096}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(12, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape, nullptr, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(2)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(28)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(1)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(2)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(256)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(2)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
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
                        .NodeInputTd(11, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 11120);
}


TEST_F(MoeDistributeCombineV2Tiling, moe_distribute_combine_v2_boundary_max) {
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
    gert::StorageShape expand_x_shape = {{128, 8192}, {128, 8192}};
    gert::StorageShape expert_ids_shape = {{64, 8}, {64, 8}};
    gert::StorageShape expand_idx_shape = {{512}, {512}};
    gert::StorageShape ep_send_counts_shape = {{8}, {8}};
    gert::StorageShape tp_send_counts_shape = {{2}, {2}};
    gert::StorageShape expert_scales_shape = {{64, 8}, {64, 8}};
    gert::StorageShape x_output_shape = {{64, 8192}, {64, 8192}};
    // 5. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 1)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&expand_x_shape, &expert_ids_shape, &expand_idx_shape, &ep_send_counts_shape,
                                       &expert_scales_shape, &tp_send_counts_shape})
                        .OutputShapes({&x_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(6)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(512)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(0)}})
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
    ASSERT_EQ(tiling_key, 10100);
}

