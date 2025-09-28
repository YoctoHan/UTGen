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

class AlltoAllAllGatherBmmTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "AlltoAllAllGatherBmmTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "AlltoAllAllGatherBmmTiling TearDown" << std::endl;
    }
};


TEST_F(AlltoAllAllGatherBmmTiling, alltoall_allgather_bmm_no_bias) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllAllGatherBatchMatMul");
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
    struct AllGatherMatmulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(4, 2)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(4096);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape x_shape = {{16, 128, 64}, {16, 128, 64}};
    gert::StorageShape weight_shape = {{4, 64, 128}, {4, 64, 128}};
    gert::StorageShape y1_output_shape = {{4, 512, 64}, {4, 512, 64}};
    // 4. Build fake context
    std::string ep_group("ep_group");
    std::string tp_group("tp_group");
    auto holder = gert::TilingContextFaker()
                        .NodeIoNum(2, 1)
                        .IrInstanceNum({1, 1})
                        .InputShapes({&x_shape, &weight_shape})
                        .OutputShapes({&y1_output_shape})
                        .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"x_shard_type", ge::AnyValue::CreateFrom<int64_t>(1)}, {"act_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"transpose_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"output_y2_flag", ge::AnyValue::CreateFrom<bool>(false)}, {"output_y3_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                        .CompileInfo(&compile_info)
                        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                        .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeInputTd(1, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                        .TilingData(param.get())
                        .Workspace(ws_size)
                        .Build();
    // 5. Init TilingContext pointer
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(ep_group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1001);
}

