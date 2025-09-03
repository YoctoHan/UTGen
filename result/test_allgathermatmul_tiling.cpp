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

class AllGatherMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulTiling TearDown" << std::endl;
    }
};


TEST_F(AllGatherMatmulTiling, all_gather_matmul_basic_float16) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{512, 1024}, {512, 1024}};
    StorageShape x2_shape = {{1024, 2048}, {1024, 2048}};
    StorageShape x3_shape = {{2048}, {2048}};
    StorageShape gather_output_shape = {{512, 1024}, {512, 1024}};
    StorageShape output_shape = {{512, 2048}, {512, 2048}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_basic_bfloat16) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{512, 1024}, {512, 1024}};
    StorageShape x2_shape = {{1024, 2048}, {1024, 2048}};
    StorageShape x3_shape = {{2048}, {2048}};
    StorageShape gather_output_shape = {{512, 1024}, {512, 1024}};
    StorageShape output_shape = {{512, 2048}, {512, 2048}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_large_shape) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{8192, 4096}, {8192, 4096}};
    StorageShape x2_shape = {{4096, 16384}, {4096, 16384}};
    StorageShape x3_shape = {{16384}, {16384}};
    StorageShape gather_output_shape = {{8192, 4096}, {8192, 4096}};
    StorageShape output_shape = {{8192, 16384}, {8192, 16384}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_small_shape) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{32, 64}, {32, 64}};
    StorageShape x2_shape = {{64, 128}, {64, 128}};
    StorageShape x3_shape = {{128}, {128}};
    StorageShape gather_output_shape = {{32, 64}, {32, 64}};
    StorageShape output_shape = {{32, 128}, {32, 128}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_large_k) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{1024, 32768}, {1024, 32768}};
    StorageShape x2_shape = {{32768, 2048}, {32768, 2048}};
    StorageShape x3_shape = {{2048}, {2048}};
    StorageShape gather_output_shape = {{1024, 32768}, {1024, 32768}};
    StorageShape output_shape = {{1024, 2048}, {1024, 2048}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_large_n) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{1024, 2048}, {1024, 2048}};
    StorageShape x2_shape = {{2048, 32768}, {2048, 32768}};
    StorageShape x3_shape = {{32768}, {32768}};
    StorageShape gather_output_shape = {{1024, 2048}, {1024, 2048}};
    StorageShape output_shape = {{1024, 32768}, {1024, 32768}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_min_boundary) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{256, 256}, {256, 256}};
    StorageShape x2_shape = {{256, 256}, {256, 256}};
    StorageShape x3_shape = {{256}, {256}};
    StorageShape gather_output_shape = {{256, 256}, {256, 256}};
    StorageShape output_shape = {{256, 256}, {256, 256}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_max_boundary) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{65535, 65534}, {65535, 65534}};
    StorageShape x2_shape = {{65534, 65535}, {65534, 65535}};
    StorageShape x3_shape = {{65535}, {65535}};
    StorageShape gather_output_shape = {{65535, 65534}, {65535, 65534}};
    StorageShape output_shape = {{65535, 65535}, {65535, 65535}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_gather_index_0) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{2048, 1024}, {2048, 1024}};
    StorageShape x2_shape = {{1024, 4096}, {1024, 4096}};
    StorageShape x3_shape = {{4096}, {4096}};
    StorageShape gather_output_shape = {{2048, 1024}, {2048, 1024}};
    StorageShape output_shape = {{2048, 4096}, {2048, 4096}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(AllGatherMatmulTiling, all_gather_matmul_no_gather_out) {
    // 1. Setup interfaces
    std::string op_type("AllGatherMatmul");
    auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    ASSERT_NE(op_impl, nullptr);
    auto tiling_func = op_impl->tiling;
    map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
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
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AllGatherMatmulCompileInfo {} compile_info;
    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    StorageShape x1_shape = {{1024, 2048}, {1024, 2048}};
    StorageShape x2_shape = {{2048, 4096}, {2048, 4096}};
    StorageShape x3_shape = {{4096}, {4096}};
    StorageShape gather_output_shape = {{1024, 2048}, {1024, 2048}};
    StorageShape output_shape = {{1024, 4096}, {1024, 4096}};
    // 5. Build fake context
    string group("group");
    auto holder = TilingContextFaker()
                        .NodeIoNum(4, 2)
                        .IrInstanceNum({1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, &x3_shape, nullptr})
                        .OutputShapes({&output_shape, &gather_output_shape})
                        .NodeAttrs({{"group", AnyValue::CreateFrom<string>(group)}, {"is_trans_a", AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", AnyValue::CreateFrom<bool>(false)}, {"gather_index", AnyValue::CreateFrom<int64_t>(0)}, {"comm_turn", AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 6. Init TilingContext pointer
    TilingContext* tiling_context = holder.GetContext<TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    // 7. Set Compile settings
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    // 8. Set communication
    HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}

