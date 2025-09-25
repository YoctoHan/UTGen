#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>

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

namespace
{
struct TestParam {
    string test_name{};
    std::vector<std::pair<string, string>> tiling_params_str_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_dTypes_pair{};
    ge::graphStatus status;
};

struct TilingParams {
    int64_t A{64};
    int64_t BSK{192};
    int64_t BS{8};
    int64_t K{8};
    int64_t H{7168};
    int64_t ep_world_size{8};
    int64_t ep_rank_id{0};
    int64_t moe_expert_num{8};
    int64_t tp_world_size{1};
    int64_t tp_rank_id{0};
    int64_t expert_shard_type{0};
    int64_t shared_expert_num{0};
    int64_t shared_expert_rank_num{0};
    int64_t global_bs{0};
    int64_t out_dtype{0};
    int64_t comm_quant_mode{0};
    int64_t group_list_type{0};
    float norm_eps{1e-6};
    std::string comm_alg{""};
    std::string group_ep{"group_ep"};
    std::string group_tp{"group_tp"};
};

struct TilingShapes {
    gert::StorageShape expand_x_shape;
    gert::StorageShape expert_ids_shape;
    gert::StorageShape assist_info_shape;
    gert::StorageShape ep_send_counts_shape;
    gert::StorageShape expert_scales_shape;
    gert::StorageShape residual_x_shape;
    gert::StorageShape gamma_shape;
    gert::StorageShape tp_send_counts_shape;
    gert::StorageShape x_active_mask_shape;
    gert::StorageShape activation_scale_shape;
    gert::StorageShape weight_scale_shape;
    gert::StorageShape group_list_shape;
    gert::StorageShape expand_scales_shape;
    gert::StorageShape shared_expert_x_shape;

    gert::StorageShape y_shape;
    gert::StorageShape rstd_shape;
    gert::StorageShape x_shape;
};

struct TilingDTypes {
    std::vector<ge::DataType> dtypes{
        ge::DT_BF16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT,
        ge::DT_BF16, ge::DT_BF16, ge::DT_INT32, ge::DT_BOOL, ge::DT_FLOAT,
        ge::DT_FLOAT, ge::DT_INT64, ge::DT_FLOAT, ge::DT_BF16, 
        ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16,
    };
};

class MoeDistributeCombineAddRmsNormTilingTest : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
        std::cout << "MoeDistributeCombineAddRmsNormTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineAddRmsNormTilingTest TearDown" << std::endl;
    }

    void InitTilingParams(const TestParam& test_param)
    {
        this->tiling_params = TilingParams{};
        this->InitTilingStrParams(test_param.tiling_params_str_pair);
    }

    void InitTilingStrParams(const std::vector<std::pair<string, string>>& tiling_params_pair)
    {
        auto& tiling_params = this->tiling_params;
        auto& tiling_params_str_handlers = MoeDistributeCombineAddRmsNormTilingTest::tiling_params_str_handlers;
        for (auto& kv : tiling_params_pair) {
            if (tiling_params_str_handlers.count(kv.first) != 0) {
                tiling_params_str_handlers[kv.first](tiling_params, kv.second);
            }
        }
    }

    void InitTilingShape()
    {
        auto const& tiling_params = this->tiling_params;
        auto A = tiling_params.A;
        auto BSK = tiling_params.BSK;
        auto BS = tiling_params.BS;
        auto K = tiling_params.K;
        auto H = tiling_params.H;
        auto ep_world_size = tiling_params.ep_world_size;
        auto ep_rank_id = tiling_params.ep_rank_id;
        auto moe_expert_num = tiling_params.moe_expert_num;
        auto tp_world_size = tiling_params.tp_world_size;
        auto tp_rank_id = tiling_params.tp_rank_id;
        auto expert_shard_type = tiling_params.expert_shard_type;
        auto shared_expert_num = tiling_params.shared_expert_num;
        auto shared_expert_rank_num = tiling_params.shared_expert_rank_num;
        auto global_bs = tiling_params.global_bs;
        auto out_dtype = tiling_params.out_dtype;
        auto comm_quant_mode = tiling_params.comm_quant_mode;
        auto group_list_type = tiling_params.group_list_type;

        auto& tiling_shapes = this->tiling_shapes;
        tiling_shapes.expand_x_shape = {{A, H}, {A, H}};
        tiling_shapes.expert_ids_shape = {{BS, K}, {BS, K}};
        tiling_shapes.assist_info_shape = {{A * 128}, {A * 128}};
        tiling_shapes.ep_send_counts_shape = {{ep_world_size}, {ep_world_size}};
        tiling_shapes.expert_scales_shape = {{BS, K}, {BS, K}};
        tiling_shapes.residual_x_shape = {{BS, 1, H}, {BS, 1, H}};
        tiling_shapes.gamma_shape = {{H}, {H}};
        tiling_shapes.tp_send_counts_shape = {{tp_world_size}, {tp_world_size}};
        tiling_shapes.x_active_mask_shape = {{BS}, {BS}};
        tiling_shapes.activation_scale_shape = {};
        tiling_shapes.weight_scale_shape = {};
        tiling_shapes.group_list_shape = {};
        tiling_shapes.expand_scales_shape = {};
        tiling_shapes.shared_expert_x_shape = {{BS, H}, {BS, H}};

        tiling_shapes.y_shape = {{BS, 1, H}, {BS, 1, H}};
        tiling_shapes.rstd_shape = {{BS, 1, 1}, {BS, 1, 1}};
        tiling_shapes.x_shape = {{BS, 1, H}, {BS, 1, H}};
    }

    void InitTilingDTypes(const std::vector<std::pair<size_t, ge::DataType>>& tiling_dTypes_pair)
    {
        auto& tiling_dtypes = this->tiling_dtypes;
        tiling_dtypes = TilingDTypes{};
        for (auto& kv : tiling_dTypes_pair) {
            if (kv.first >= 0 && kv.first < tiling_dtypes.dtypes.size()) {
                tiling_dtypes.dtypes[kv.first] = kv.second;
            }
        }
    }

    void InitHolder(void* tilingData, gert::ContinuousVector* workspace, const TestParam& test_param)
    {
        auto& compile_info_string = MoeDistributeCombineAddRmsNormTilingTest::compile_info_string;
        auto& platform_info = this->platform_info;
        auto& compile_info = this->compile_info;
        platform_info.Init();

        auto input_num = this->input_num;
        auto output_num = this->output_num;

        this->kernel_faker =
            gert::KernelRunContextFaker()
                .KernelIONum(input_num, output_num)
                .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
                .Outputs({&compile_info});
        this->kernel_holder = this->kernel_faker.Build();

        this->InitTilingParams(test_param);
        auto const& tiling_params = this->tiling_params;

        this->InitTilingShape();
        auto& tiling_shapes = this->tiling_shapes;

        this->InitTilingDTypes(test_param.tiling_dTypes_pair);
        auto& tiling_dtypes = this->tiling_dtypes;

        std::string op_type("MoeDistributeCombineAddRmsNorm");

        this->tiling_faker =
            gert::TilingContextFaker()
                .NodeIoNum(input_num, output_num)
                .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                .InputShapes({&tiling_shapes.expand_x_shape,
                              &tiling_shapes.expert_ids_shape,
                              &tiling_shapes.assist_info_shape,
                              &tiling_shapes.ep_send_counts_shape,
                              &tiling_shapes.expert_scales_shape,
                              &tiling_shapes.residual_x_shape,
                              &tiling_shapes.gamma_shape,
                              &tiling_shapes.tp_send_counts_shape,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              nullptr,
                              &tiling_shapes.shared_expert_x_shape})
                .OutputShapes({&tiling_shapes.y_shape,
                               &tiling_shapes.rstd_shape,
                               &tiling_shapes.x_shape})
                .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(tiling_params.group_ep)},
                            {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
                            {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(tiling_params.ep_rank_id)},
                            {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(tiling_params.moe_expert_num)},
                            {"group_tp", ge::AnyValue::CreateFrom<std::string>(tiling_params.group_tp)},
                            {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(tiling_params.tp_world_size)},
                            {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(tiling_params.tp_rank_id)},
                            {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(tiling_params.expert_shard_type)},
                            {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(tiling_params.shared_expert_num)},
                            {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(tiling_params.shared_expert_rank_num)},
                            {"global_bs", ge::AnyValue::CreateFrom<int64_t>(tiling_params.global_bs)},
                            {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(tiling_params.out_dtype)},
                            {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(tiling_params.comm_quant_mode)},
                            {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(tiling_params.group_list_type)},
                            {"comm_alg", ge::AnyValue::CreateFrom<std::string>(tiling_params.comm_alg)},
                            {"norm_eps", ge::AnyValue::CreateFrom<float>(tiling_params.norm_eps)}})
                .CompileInfo(&compile_info)
                .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                .TilingData(tilingData)
                .Workspace(workspace)
                .SetOpType(op_type);

        for (int64_t i = 0; i < input_num; ++i) {
            this->tiling_faker =
                this->tiling_faker.NodeInputTd(i, tiling_dtypes.dtypes[i], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        for (int64_t i = 0; i < output_num; ++i) {
            this->tiling_faker =
                this->tiling_faker.NodeOutputTd(i, tiling_dtypes.dtypes[input_num + i], ge::FORMAT_ND, ge::FORMAT_ND);
        }
        this->tiling_holder = this->tiling_faker.Build();
    }

public:
    int64_t input_num{14};
    int64_t output_num{3};
    TilingParams tiling_params;
    TilingShapes tiling_shapes;
    TilingDTypes tiling_dtypes;
    struct AllGatherMatmulCompileInfo {
    } compile_info;
    fe::PlatFormInfos platform_info;
    gert::KernelRunContextFaker kernel_faker{};
    gert::KernelRunContextHolder kernel_holder{};
    gert::TilingContextFaker tiling_faker{};
    gert::KernelRunContextHolder tiling_holder{};
    static string compile_info_string;
    static std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>>
        tiling_params_str_handlers;
};

string MoeDistributeCombineAddRmsNormTilingTest::compile_info_string = R"({
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

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>>
    MoeDistributeCombineAddRmsNormTilingTest::tiling_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BSK = std::stoi(value_str); }}};

TEST_P(MoeDistributeCombineAddRmsNormTilingTest, common_test)
{
    auto test_param = GetParam();
    std::string op_type("MoeDistributeCombineAddRmsNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = MoeDistributeCombineAddRmsNormTilingTest::compile_info_string;
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> version = {{"Short_SoC_version", "Ascend910_93"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    auto tilingData = gert::TilingData::CreateCap(4096);
    ASSERT_NE(tilingData, nullptr);
    auto workspace_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspace_holer.get());

    InitHolder(tilingData.get(), workspace, test_param);

    auto& holder = this->tiling_holder;

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), test_param.status);
}

static TestParam test_params[] = {
    {"Test_sample", {}, {}, ge::GRAPH_SUCCESS}

};

INSTANTIATE_TEST_SUITE_P(MoeDistributeCombineAddRmsNormTilingTest, MoeDistributeCombineAddRmsNormTilingTest,
                         testing::ValuesIn(test_params),
                         [](const testing::TestParamInfo<MoeDistributeCombineAddRmsNormTilingTest::ParamType>& info) {
                             return info.param.test_name;
                         });
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_basic_fp16_1) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_basic_bf16) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_large_shape) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_small_shape) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_boundary_min) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_transposed_b) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_no_bias) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 110);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_large_k) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_large_n) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}


TEST_F(MoeDistributeCombineAddRMSNormTiling, moe_distribute_combine_add_rms_norm_4p_world_size) {
    // 1. Setup interfaces
    std::string op_type("MoeDistributeCombineAddRMSNorm");
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
    struct MoeEPLBUpdateExpertCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape expand_x_shape = {{64, 7168}, {64, 7168}};
    gert::StorageShape expert_ids_shape = {{8, 8}, {8, 8}};
    gert::StorageShape assist_info_shape = {{8192}, {8192}};
    gert::StorageShape ep_send_counts_shape = {{None}, {None}};
    gert::StorageShape expert_scales_shape = {{8, 8}, {8, 8}};
    gert::StorageShape residual_x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape gamma_shape = {{7168}, {7168}};
    gert::StorageShape tp_send_counts_shape = {{None}, {None}};
    gert::StorageShape x_active_mask_shape = {{8}, {8}};
    gert::StorageShape shared_expert_x_shape = {{8, 7168}, {8, 7168}};
    gert::StorageShape y_shape = {{8, 1, 7168}, {8, 1, 7168}};
    gert::StorageShape rstd_shape = {{8, 1, 1}, {8, 1, 1}};
    gert::StorageShape x_shape = {{8, 1, 7168}, {8, 1, 7168}};
    // 4. Build fake context
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(14, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1})
                      .InputShapes({&expand_x_shape, &expert_ids_shape, &assist_info_shape, &ep_send_counts_shape, &expert_scales_shape, &residual_x_shape, &gamma_shape, &tp_send_counts_shape, &x_active_mask_shape, nullptr, nullptr, nullptr, nullptr, &shared_expert_x_shape})
                      .OutputShapes({&y_shape, &rstd_shape, &x_shape})
                      .NodeAttrs({{\"group_ep\", ge::AnyValue::CreateFrom<std::string>(std::string("ep_group"))}, {\"ep_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"ep_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"moe_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_tp\", ge::AnyValue::CreateFrom<std::string>(std::string("tp_group"))}, {\"tp_world_size\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"tp_rank_id\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"expert_shard_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"shared_expert_rank_num\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"global_bs\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"out_dtype\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_quant_mode\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"group_list_type\", ge::AnyValue::CreateFrom<int64_t>(None)}, {\"comm_alg\", ge::AnyValue::CreateFrom<std::string>(std::string(""))}, {\"norm_eps\", ge::AnyValue::CreateFrom<float>(1e-06)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(5, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(6, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(7, ge::DT_INT32, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(8, ge::DT_BOOL, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(9, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(10, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(11, ge::DT_INT64, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(12, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(13, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
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
    map<string, string> version = {{"Short_SoC_version", "ascend910b"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // 10. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 111);
}

