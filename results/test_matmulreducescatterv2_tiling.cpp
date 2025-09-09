#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "op_log.h"
#define private public

#include "kernel_run_context_facker.h"

#include "experiment_ops.h"
#include "fusion_ops.h"
#include "op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"
#include "common_unittest.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "external/hcom/hcom_topo_info.h"
#include "test_cube_util.h"

class MatmulReduceScatterV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "MatmulReduceScatterV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MatmulReduceScatterV2Tiling TearDown" << std::endl;
    }
};

// case_name: 4096_1024_8192_e4m3fn_e4m3fn_fp32_rank8_reducescatterv2_david_ID000


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_float16_1) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{1536, 12288}, {1536, 12288}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_float16_2) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{12288, 1536}, {12288, 1536}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(true)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_float16_3) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{16384, 4096}, {16384, 4096}};
    gert::StorageShape x2_shape = {{4096, 2752}, {4096, 2752}};
    gert::StorageShape output_shape = {{16384, 2752}, {16384, 2752}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_float16_4) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{16384, 4096}, {16384, 4096}};
    gert::StorageShape x2_shape = {{2752, 4096}, {2752, 4096}};
    gert::StorageShape output_shape = {{16384, 2752}, {16384, 2752}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(true)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_float16_5) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape x2_shape = {{2752, 4096}, {2752, 4096}};
    gert::StorageShape output_shape = {{4096, 2752}, {4096, 2752}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(true)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_bfloat16) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{1536, 12288}, {1536, 12288}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_fp8_1) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{1536, 12288}, {1536, 12288}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_fp8_2) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{12288, 1536}, {12288, 1536}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(true)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_perblock_1) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{1536, 12288}, {1536, 12288}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(false)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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


TEST_F(MatmulReduceScatterV2Tiling, matmul_reduce_scatter_v2_test_tiling_perblock_2) {
    // 1. Setup interfaces
    std::string op_type("MatmulReduceScatterV2");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;
    map<string, string> socversions={};
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
    struct MatmulReduceScatterV2CompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(7, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    // 4. Define input/output shapes (dims 与 storage_dims 对齐)
    gert::StorageShape x1_shape = {{8192, 1536}, {8192, 1536}};
    gert::StorageShape x2_shape = {{12288, 1536}, {12288, 1536}};
    gert::StorageShape output_shape = {{8192, 12288}, {8192, 12288}};
    // 5. Build fake context
    string group("group");
    string reduce_op("sum");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(7, 2)
                        .IrInstanceNum({1, 1, 1, 1, 1, 1, 1})
                        .InputShapes({&x1_shape, &x2_shape, nullptr, nullptr, nullptr, nullptr, nullptr})
                        .OutputShapes({&output_shape, nullptr})
                        .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"reduce_op", ge::AnyValue::CreateFrom<std::string>(std::string("sum"))}, {"is_trans_a", ge::AnyValue::CreateFrom<bool>(false)}, {"is_trans_b", ge::AnyValue::CreateFrom<bool>(true)}, {"comm_turn", ge::AnyValue::CreateFrom<int64_t>(0)}, {"rank_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"block_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_size", ge::AnyValue::CreateFrom<int64_t>(0)}, {"is_amax_out", ge::AnyValue::CreateFrom<bool>(true)}, {"y_dtype", ge::AnyValue::CreateFrom<int64_t>(0)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(5, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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

