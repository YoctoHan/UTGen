#include <iostream>
#include <vector>

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

class AllGatherMatmulV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulV2Tiling TearDown" << std::endl;
    }
};


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_basic_small_fp16) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{64, 128}, {64, 128}};
    gert::StorageShape x2_shape = {{128, 256}, {128, 256}};
    gert::StorageShape gather_output_shape = {{512, 128}, {512, 128}};
    gert::StorageShape output_shape = {{512, 256}, {512, 256}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_basic_small_bf16) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{64, 128}, {64, 128}};
    gert::StorageShape x2_shape = {{128, 256}, {128, 256}};
    gert::StorageShape gather_output_shape = {{512, 128}, {512, 128}};
    gert::StorageShape output_shape = {{512, 256}, {512, 256}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_large_shape_fp16) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 4096}, {8192, 4096}};
    gert::StorageShape x2_shape = {{4096, 2048}, {4096, 2048}};
    gert::StorageShape gather_output_shape = {{65536, 4096}, {65536, 4096}};
    gert::StorageShape output_shape = {{65536, 2048}, {65536, 2048}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_large_shape_bf16) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 4096}, {8192, 4096}};
    gert::StorageShape x2_shape = {{4096, 2048}, {4096, 2048}};
    gert::StorageShape gather_output_shape = {{65536, 4096}, {65536, 4096}};
    gert::StorageShape output_shape = {{65536, 2048}, {65536, 2048}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_boundary_min) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{1, 256}, {1, 256}};
    gert::StorageShape x2_shape = {{256, 1}, {256, 1}};
    gert::StorageShape gather_output_shape = {{8, 256}, {8, 256}};
    gert::StorageShape output_shape = {{8, 1}, {8, 1}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_boundary_max) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{65535, 65535}, {65535, 65535}};
    gert::StorageShape x2_shape = {{65535, 65535}, {65535, 65535}};
    gert::StorageShape gather_output_shape = {{524280, 65535}, {524280, 65535}};
    gert::StorageShape output_shape = {{524280, 65535}, {524280, 65535}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_transpose_b) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{1024, 2048}, {1024, 2048}};
    gert::StorageShape x2_shape = {{1024, 2048}, {1024, 2048}};
    gert::StorageShape gather_output_shape = {{8192, 2048}, {8192, 2048}};
    gert::StorageShape output_shape = {{8192, 1024}, {8192, 1024}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(true)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_small_m_large_nk) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{32, 16384}, {32, 16384}};
    gert::StorageShape x2_shape = {{16384, 8192}, {16384, 8192}};
    gert::StorageShape gather_output_shape = {{256, 16384}, {256, 16384}};
    gert::StorageShape output_shape = {{256, 8192}, {256, 8192}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_large_m_small_nk) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{32768, 256}, {32768, 256}};
    gert::StorageShape x2_shape = {{256, 512}, {256, 512}};
    gert::StorageShape gather_output_shape = {{262144, 256}, {262144, 256}};
    gert::StorageShape output_shape = {{262144, 512}, {262144, 512}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(AllGatherMatmulV2Tiling, all_gather_matmul_v2_medium_shape_no_bias) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmulV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions = {};
    // 2. Setup compile info and platform info
    string compile_info_string = R"({
        "hardware_info": {
            "BT_SIZE": 1024,
            "load3d_constraints": "0",
            "Intrinsic_fix_pipe_l0c2out": true,
            "Intrinsic_data_move_l12ub": false,
            "Intrinsic_data_move_l0c2ub": false,
            "Intrinsic_data_move_out2l1_nd2nz": true,
            "UB_SIZE": 262144,
            "L2_SIZE": 134217728,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 262144,
            "CORE_NUM": 32
        }
    })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{2048, 1024}, {2048, 1024}};
    gert::StorageShape x2_shape = {{1024, 512}, {1024, 512}};
    gert::StorageShape gather_output_shape = {{16384, 1024}, {16384, 1024}};
    gert::StorageShape output_shape = {{16384, 512}, {16384, 512}};
    gert::StorageShape x1Scale_shape = {{1}, {1}};
    gert::StorageShape x2Scale_shape = {{1}, {1}};
    // 5. Build fake context
    string group("group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(6, 3)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, &x1Scale_shape, &x2Scale_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"gather_index", ge::AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_gather_out", ge::AnyValue::CreateFrom<bool>(false)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(false)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT8_E4M3FN, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .SetOpType(op_type)
                        .Build();
    // 6. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Unset communication
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}

