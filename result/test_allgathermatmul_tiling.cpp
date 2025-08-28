/**
 * AllGatherMatmul算子单元测试模板
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

// 测试用例模板
TEST_F(AllGatherMatmulTiling, all_gather_matmul_test_template) {
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

    // 4. Define input/output shapes (需要根据具体测试用例设置)
    // StorageShape x1_shape = {{...}, {...}};
    // StorageShape x2_shape = {{...}, {...}};
    // StorageShape x3_shape = {{...}, {...}};  // 可选
    // StorageShape gather_output_shape = {{...}, {...}};
    // StorageShape output_shape = {{...}, {...}};

    // 5. Build fake context (需要根据具体测试用例调整参数)
    string group("group");

    auto holder = TilingContextFaker()
                        // .NodeIoNum(输入数量, 输出数量)
                        // .IrInstanceNum({...})
                        // .InputShapes({...})
                        // .OutputShapes({...})
                        // .NodeAttrs({...})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        // .NodeInputTd(索引, 数据类型, 格式, 格式)
                        // .NodeOutputTd(索引, 数据类型, 格式, 格式)
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
    // topoInfo.rank_size = ...;  // 根据测试需求设置
    HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    // 9. Call op function, check returns == GRAPH_SUCCESS
    EXPECT_EQ(tiling_func(tiling_context), GRAPH_SUCCESS);

    // 10. Unset communication
    HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());

    // 11. Check tiling key (根据实际期望值调整)
    // auto tiling_key = tiling_context->GetTilingKey();
    // ASSERT_EQ(tiling_key, 期望值);
}

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
    StorageShape x1_shape = {{1024, 2048}};
    StorageShape x2_shape = {{2048, 4096}};
    StorageShape bias_shape = {{4096}};
    StorageShape gather_output_shape = {{1024, 2048}};
    StorageShape output_shape = {{1024, 4096}};

    // 5. Build fake context
    string group("group");

    std::vector<StorageShape> input_shapes = {x1_shape, x2_shape, bias_shape};
    std::vector<StorageShape> output_shapes = {output_shape};
    if (true) {  // gather_output is True
        output_shapes.insert(output_shapes.begin(), gather_output_shape);
    }

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, output_shapes.size())
                        .IrInstanceNum({0, 1, 2})
                        .InputShapes(input_shapes)
                        .OutputShapes(output_shapes)
                        .NodeAttrs({
                            {"group", group},
                            {"rank_size", "8"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
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

TEST_F(AllGatherMatmulTiling, case_2) {
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
    StorageShape x1_shape = {{8192, 4096}};  // input[0]
    StorageShape x2_shape = {{4096, 8192}};  // input[1]
    StorageShape x3_shape = {{8192}};        // input[2], bias
    StorageShape gather_output_shape = {{8192, 4096}};  // gather output
    StorageShape output_shape = {{8192, 8192}};         // final output

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs, 2 outputs
                        .IrInstanceNum({0})
                        .InputShapes({x1_shape, x2_shape, x3_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"group", group},
                            {"rank_size", "8"},
                            {"gather_output", "True"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)  // gather output
                        .NodeOutputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)  // matmul output
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
    StorageShape x1_shape = {{32, 4096}};
    StorageShape x2_shape = {{4096, 2048}};
    StorageShape bias_shape = {{2048}};
    StorageShape gather_output_shape = {{32, 4096}};
    StorageShape output_shape = {{32, 2048}};

    // 5. Build fake context
    string group("group");

    std::vector<ge::TensorDesc> input_descs;
    input_descs.emplace_back(x1_shape.GetDims(), ge::FORMAT_ND, ge::DT_FLOAT16);
    input_descs.emplace_back(x2_shape.GetDims(), ge::FORMAT_ND, ge::DT_FLOAT16);
    input_descs.emplace_back(bias_shape.GetDims(), ge::FORMAT_ND, ge::DT_FLOAT16);

    std::vector<ge::TensorDesc> output_descs;
    if (true) { // gather_output is True
        output_descs.emplace_back(gather_output_shape.GetDims(), ge::FORMAT_ND, ge::DT_FLOAT16);
    }
    output_descs.emplace_back(output_shape.GetDims(), ge::FORMAT_ND, ge::DT_FLOAT16);

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, true ? 2 : 1)
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes(true ? std::vector<StorageShape>{gather_output_shape, output_shape}
                                          : std::vector<StorageShape>{output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"},
                            {"group", group}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    topoInfo.rank_size = 8;  // world_size
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

TEST_F(AllGatherMatmulTiling, all_gather_matmul_bfloat16) {
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
    StorageShape x1_shape = {{2048, 1024}};
    StorageShape x2_shape = {{1024, 2048}};
    StorageShape x3_shape = {{2048}};  // bias
    StorageShape gather_output_shape = {{2048, 1024}};
    StorageShape output_shape = {{2048, 2048}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)  // 3 inputs, 2 outputs
                        .IrInstanceNum({0})
                        .InputShapes({x1_shape, x2_shape, x3_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"group", group},
                            {"world_size", "8"},
                            {"gather_output", "True"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_BF16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_BF16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(2, DT_BF16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_BF16, FORMAT_ND, FORMAT_ND)  // gather output
                        .NodeOutputTd(1, DT_BF16, FORMAT_ND, FORMAT_ND)  // matmul output
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

TEST_F(AllGatherMatmulTiling, all_gather_matmul_transpose_b) {
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
    StorageShape x2_shape = {{4096, 2048}};
    StorageShape bias_shape = {{4096}};
    StorageShape gather_output_shape = {{1024, 2048}};
    StorageShape output_shape = {{1024, 4096}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)
                        .IrInstanceNum({0, 1, 2})
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"transpose_a", "False"},
                            {"transpose_b", "True"},
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
    struct AllGatherMatmulCompileInfo {
    } compile_info;

    // 3. Create context
    auto param = TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);

    auto workspace_size_holer = ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<ContinuousVector*>(workspace_size_holer.get());

    // 4. Define input/output shapes
    StorageShape x1_shape = {{1024, 8192}};
    StorageShape x2_shape = {{8192, 2048}};
    StorageShape x3_shape = {{2048}};
    StorageShape gather_output_shape = {{1024, 8192}};
    StorageShape output_shape = {{1024, 2048}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)
                        .IrInstanceNum({0, 1, 2})
                        .InputShapes({x1_shape, x2_shape, x3_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"bias_shape", "[2048]"},
                            {"world_size", "8"},
                            {"gather_output", "True"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"},
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

TEST_F(AllGatherMatmulTiling, all_gather_matmul_small_n) {
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
    StorageShape x1_shape = {{2048, 1024}};
    StorageShape x2_shape = {{1024, 32}};
    StorageShape x3_shape = {{32}};
    StorageShape gather_output_shape = {{2048, 1024}};
    StorageShape output_shape = {{2048, 32}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)
                        .IrInstanceNum({0, 1, 2})
                        .InputShapes({x1_shape, x2_shape, x3_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"bias_shape", "[32]"},
                            {"world_size", "8"},
                            {"gather_output", "True"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"},
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
    StorageShape x1_shape = {{1024, 2048}, {2048, 4096}};
    StorageShape x2_shape = {{2048, 4096}, {}};
    StorageShape gather_output_shape = {{1024, 2048}, {}};
    StorageShape output_shape = {{1024, 4096}, {}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({2, 1})
                        .InputShapes({x1_shape, x2_shape})
                        .OutputShapes({output_shape})
                        .NodeAttrs({
                            {"group", group},
                            {"world_size", "8"},
                            {"gather_output", "True"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"},
                            {"gather_index", "0"},
                            {"comm_turn", "0"}
                        })
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, FORMAT_ND, FORMAT_ND)
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
    ASSERT_EQ(tiling_key, 100);
}

TEST_F(AllGatherMatmulTiling, all_gather_matmul_boundary_k) {
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
    StorageShape x1_shape = {{1024, 256}};
    StorageShape x2_shape = {{256, 4096}};
    StorageShape bias_shape = {{4096}};
    StorageShape gather_output_shape = {{1024, 256}};
    StorageShape output_shape = {{1024, 4096}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)
                        .IrInstanceNum({0, 1, 2})
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"group", group},
                            {"world_size", "8"},
                            {"gather_output", "True"},
                            {"is_trans_a", "False"},
                            {"is_trans_b", "False"},
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

TEST_F(AllGatherMatmulTiling, case_10) {
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
    StorageShape x1_shape = {{2048, 1024}};
    StorageShape x2_shape = {{4096, 1024}};
    StorageShape bias_shape = {{4096}};
    StorageShape gather_output_shape = {{2048, 1024}};
    StorageShape output_shape = {{2048, 4096}};

    // 5. Build fake context
    string group("group");

    auto holder = TilingContextFaker()
                        .NodeIoNum(3, 2)
                        .IrInstanceNum({0, 1, 2})
                        .InputShapes({x1_shape, x2_shape, bias_shape})
                        .OutputShapes({gather_output_shape, output_shape})
                        .NodeAttrs({
                            {"group", group},
                            {"rank_size", "8"},
                            {"gather_index", "1"},
                            {"comm_turn", "0"},
                            {"transpose_a", "False"},
                            {"transpose_b", "True"},
                            {"has_bias", "True"}
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
