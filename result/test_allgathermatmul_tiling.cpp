/**
 * 自动生成的AllGatherMatmul算子单元测试模板
 * 生成时间: 2025-08-06 16:16:36
 */

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

using namespace ge;
using namespace gert;

// 测试类模板
class AllGatherMatmulTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AllGatherMatmulTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AllGatherMatmulTiling TearDown" << std::endl;
    }
};

// TODO: 在此处添加具体的TEST_F测试用例
// 示例格式:
// TEST_F(AllGatherMatmulTiling, test_case_name) {
//     // 1. Setup interfaces
//     std::string op_type("AllGatherMatmul");
//     auto op_impl = OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
//     ASSERT_NE(op_impl, nullptr);
//     auto tiling_func = op_impl->tiling;
//     map<string, string> socversions = {{"Short_Soc_version", "ascend910b"}};
//
//     // 2. Setup compile info and platform info
//     string compile_info_string = R"({
//         "hardware_info": {
//             "BT_SIZE": 1024,
//             "load3d_constraints": "0",
//             "Intrinsic_fix_pipe_l0c2out": true,
//             "Intrinsic_data_move_l12ub": false,
//             "Intrinsic_data_move_l0c2ub": false,
//             "Intrinsic_data_move_out2l1_nd2nz": true,
//             "UB_SIZE": 196608,
//             "L2_SIZE": 33554432,
//             "L1_SIZE": 524288,
//             "L0A_SIZE": 65536,
//             "L0B_SIZE": 65536,
//             "L0C_SIZE": 131072,
//             "CORE_NUM": 20
//         }
//     })";
//     map<string, string> soc_infos;
//     map<string, string> aicore_spec;
//     map<string, string> intrinsics;
//
//     fe::PlatFormInfos platform_info;
//     platform_info.Init();
//     struct AllGatherMatmulCompileInfo {
//     } compile_info;
//
//     // 3. Create context
//     auto param = TilingData::CreateCap(4096);
//     ASSERT_NE(param, nullptr);
//
//     auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
//     auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());
//
//     // 4. Define input/output shapes (根据测试参数调整)
//     // StorageShape x1_shape = {{...}, {...}};
//     // StorageShape x2_shape = {{...}, {...}};
//     // StorageShape output_shape = {{...}, {...}};
//     // StorageShape gather_output_shape = {{...}, {...}};
//
//     // 5. Build fake context (根据测试参数调整)
//     string group("group");
//
//     auto holder = TilingContextFaker()
//                         .NodeIoNum(/*input_num*/, /*output_num*/)
//                         .IrInstanceNum({/*instance_nums*/})
//                         .InputShapes({/*input_shapes*/})
//                         .OutputShapes({/*output_shapes*/})
//                         .NodeAttrs({/*attributes*/})
//                         .CompileInfo(&compile_info)
//                         .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                         .NodeInputTd(/*index*/, /*data_type*/, /*format*/, /*storage_format*/)
//                         .NodeOutputTd(/*index*/, /*data_type*/, /*format*/, /*storage_format*/)
//                         .TilingData(param.get())
//                         .Workspace(ws_size)
//                         .Build();
//
//     // 6. Init TilingContext pointer
//     TilingContext* tiling_context = holder.GetContext<TilingContext>();
//     ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//
//     // 7. Set Compile settings
//     holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//     holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//     holder.GetContext<TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//     holder.GetContext<TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
//
//     // 8. Set communication
//     HcomTopoInfo::TopoInfo topoInfo;
//     topoInfo.rank_size = /*rank_size*/;
//     HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
//     tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);
//
//     // 9. Call op function, check returns == GRAPH_SUCCESS
//     EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);
//
//     // 10. Unset communication
//     HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
//
//     // 11. Check tiling key (可选)
//     // auto tiling_key = tiling_context->GetTilingKey();
//     // ASSERT_EQ(tiling_key, expected_key);
// }

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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{512, 1024}};
    StorageShape x2_shape = {{1024, 2048}};
    StorageShape bias_shape = {{2048}};
    StorageShape output_shape = {{512, 2048}};
    StorageShape gather_output_shape = {{512, 1024}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs, 2 outputs
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"}
                        })
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{512, 1024}};
    StorageShape x2_shape = {{1024, 2048}};
    StorageShape output_shape = {{512, 2048}};
    StorageShape gather_output_shape = {{512, 1024}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs (x1, x2, bias), 2 outputs (output, gather_output)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x2_shape, StorageShape({2048})})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"world_size", "8"},
                            {"gather_output", "True"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_BF16, FORMAT_ND, FORMAT_ND)   // x1
                        .NodeInputTd(1, DT_BF16, FORMAT_ND, FORMAT_ND)   // x2
                        .NodeInputTd(2, DT_BF16, FORMAT_ND, FORMAT_ND)   // bias
                        .NodeOutputTd(0, DT_BF16, FORMAT_ND, FORMAT_ND)  // output
                        .NodeOutputTd(1, DT_BF16, FORMAT_ND, FORMAT_ND)  // gather_output
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

TEST_F(AllGatherMatmulTiling, case_3) {
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{8192, 4096}};
    StorageShape x2_shape = {{4096, 16384}};
    StorageShape bias_shape = {{16384}};
    StorageShape output_shape = {{8192, 16384}};
    StorageShape gather_output_shape = {{8192, 4096}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs, 2 outputs
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"world_size", "8"},
                            {"gather_output", "True"}
                        })
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

TEST_F(AllGatherMatmulTiling, case_4) {
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{32, 64}, {64, 128}};
    StorageShape x2_shape = {{64, 128}, {}};
    StorageShape output_shape = {{32, 128}, {}};
    StorageShape gather_output_shape = {{32, 64}, {}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(2, 2)
                        .IrInstanceNum({1, 1})
                        .InputShapes({x1_shape, x2_shape})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"bias_shape", "[128]"},
                            {"world_size", "8"},
                            {"gather_output", "True"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
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

TEST_F(AllGatherMatmulTiling, case_5) {
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{1024, 32768}};
    StorageShape x2_shape = {{32768, 2048}};
    StorageShape output_shape = {{1024, 2048}};
    StorageShape gather_output_shape = {{1024, 32768}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs (x1, x2, bias), 2 outputs (output, gather_output)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x2_shape, StorageShape({2048})})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"}
                        })
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

TEST_F(AllGatherMatmulTiling, case_6) {
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{1024, 2048}};
    StorageShape x2_shape = {{2048, 32768}};
    StorageShape output_shape = {{1024, 32768}};
    StorageShape gather_output_shape = {{1024, 2048}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(2, 2)
                        .IrInstanceNum({1, 1})
                        .InputShapes({x1_shape, x2_shape})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
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

TEST_F(AllGatherMatmulTiling, case_7) {
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{256}, {256}};
    StorageShape x2_shape = {{256}, {256}};
    StorageShape output_shape = {{256}, {256}};
    StorageShape gather_output_shape = {{256}, {256}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(2, 2)
                        .IrInstanceNum({1, 1})
                        .InputShapes({x1_shape, x2_shape})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"world_size", "8"},
                            {"gather_output", "True"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
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

TEST_F(AllGatherMatmulTiling, case_8) {
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{65535, 65534}};
    StorageShape x2_shape = {{65534, 65535}};
    StorageShape output_shape = {{65535, 65535}};
    StorageShape gather_output_shape = {{65535, 65534}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs (x1, x2, bias), 2 outputs (output, gather_output)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x2_shape, StorageShape({65535})})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"world_size", "8"},
                            {"gather_output", "True"}
                        })
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{2048, 1024}, {1024, 4096}};
    StorageShape x2_shape = {{1024, 4096}, {}};
    StorageShape output_shape = {{2048, 4096}, {}};
    StorageShape gather_output_shape = {{2048, 1024}, {}};
    StorageShape bias_shape = {{4096}, {}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs (x1, x2, bias), 2 outputs (output, gather_output)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes({output_shape, gather_output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"world_size", "8"},
                            {"gather_output", "True"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)   // x1
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)   // x2
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)   // bias
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)  // output
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)  // gather_output
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{1024, 2048}, {2048, 4096}};
    StorageShape bias_shape = {{4096}, {}};
    StorageShape output_shape = {{1024, 4096}, {}};
    StorageShape gather_out_shape = {{1024, 2048}, {}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs (x1, x2, bias), 2 outputs (output, gather_out)
                        .IrInstanceNum({1, 1, 1})
                        .InputShapes({x1_shape, x1_shape, bias_shape})  // x1, x2, bias
                        .OutputShapes({output_shape, gather_out_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"world_size", "8"},
                            {"gather_output", "False"}
                        })
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
