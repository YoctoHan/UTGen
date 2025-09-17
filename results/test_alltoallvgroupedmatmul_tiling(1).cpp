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
    std::vector<std::pair<string, std::vector<int64_t>>> tiling_params_vec_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_dTypes_pair{};
    ge::graphStatus status;
};

struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{64};
    uint64_t ep_world_size{8};
    uint64_t e{4};
    uint64_t commOut;
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t totalUbSize{196608};
    uint64_t gmm_weight_dim1{7168};
    uint64_t gmm_y_dim1{4096};
    uint64_t mm_weight_dim0{7168};
    bool trans_gmm_weight{false};
    bool trans_mm_weight{false};
    bool permute_out_flag{false};
    bool is_Need_MM{true};
    std::string group{"group"};
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

struct TilingShapes {
    gert::StorageShape gmm_x_shape;
    gert::StorageShape gmm_weight_shape;
    gert::StorageShape send_counts_shape;
    gert::StorageShape recv_counts_shape;
    gert::StorageShape mm_x_shape;
    gert::StorageShape mm_weight_shape;

    gert::StorageShape gmm_y_shape;
    gert::StorageShape mm_y_shape;
    gert::StorageShape permute_out_shape;
};

struct TilingDTypes {
    std::vector<ge::DataType> dtypes{ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT64,   ge::DT_INT64,  ge::DT_FLOAT16,
                                     ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16};
};

class AlltoAllvGroupedMatMulTilingTest : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        setenv("ASCEND_SLOG_PRINT_TO_STDOUT", "1", 1);
        std::cout << "AlltoAllvGroupedMatMulTilingTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTilingTest TearDown" << std::endl;
    }

    void InitTilingParams(const TestParam& test_param)
    {
        this->tiling_params = TilingParams{};
        this->InitTilingStrParams(test_param.tiling_params_str_pair);
        this->InitTilingVecParams(test_param.tiling_params_vec_pair);
    }

    void InitTilingStrParams(const std::vector<std::pair<string, string>>& tiling_params_pair)
    {
        auto& tiling_params = this->tiling_params;
        auto& tiling_params_str_handlers = AlltoAllvGroupedMatMulTilingTest::tiling_params_str_handlers;
        for (auto& kv : tiling_params_pair) {
            if (tiling_params_str_handlers.count(kv.first) != 0) {
                tiling_params_str_handlers[kv.first](tiling_params, kv.second);
            }
        }
    }

    void InitTilingVecParams(const std::vector<std::pair<string, std::vector<int64_t>>>& tiling_params_pair)
    {
        auto& tiling_params = this->tiling_params;
        auto& tiling_params_vec_handlers = AlltoAllvGroupedMatMulTilingTest::tiling_params_vec_handlers;
        for (auto& kv : tiling_params_pair) {
            if (tiling_params_vec_handlers.count(kv.first) != 0) {
                tiling_params_vec_handlers[kv.first](tiling_params, kv.second);
            }
        }
    }

    void InitTilingShape()
    {
        auto& tiling_params = this->tiling_params;
        auto& BSK = tiling_params.BSK;
        auto& BS = tiling_params.BS;
        auto& K = tiling_params.K;
        auto& H1 = tiling_params.H1;
        auto& H2 = tiling_params.H2;
        auto& A = tiling_params.A;
        auto& N1 = tiling_params.N1;
        auto& N2 = tiling_params.N2;
        auto& ep_world_size = tiling_params.ep_world_size;
        auto& e = tiling_params.e;
        auto& gmm_weight_dim1 = tiling_params.gmm_weight_dim1;
        auto& gmm_y_dim1 = tiling_params.gmm_y_dim1;
        auto& mm_weight_dim0 = tiling_params.mm_weight_dim0;

        auto& tiling_shapes = this->tiling_shapes;
        tiling_shapes.gmm_x_shape = {{BSK, H1}, {BSK, H1}};
        tiling_shapes.gmm_weight_shape = {{e, gmm_weight_dim1, N1}, {e, gmm_weight_dim1, N1}};
        tiling_shapes.send_counts_shape = {{e * ep_world_size}, {e * ep_world_size}};
        tiling_shapes.recv_counts_shape = {{e * ep_world_size}, {e * ep_world_size}};
        tiling_shapes.mm_x_shape = {{BS, H2}, {BS, H2}};
        tiling_shapes.mm_weight_shape = {{mm_weight_dim0, N2}, {mm_weight_dim0, N2}};

        tiling_shapes.gmm_y_shape = {{A, gmm_y_dim1}, {A, gmm_y_dim1}};
        tiling_shapes.mm_y_shape = {{BS, N2}, {BS, N2}};
        tiling_shapes.permute_out_shape = {{A, H1}, {A, H1}};
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
        auto& compile_info_string = AlltoAllvGroupedMatMulTilingTest::compile_info_string;
        auto& platform_info = this->platform_info;
        auto& compile_info = this->compile_info;

        platform_info.Init();
        this->kernel_faker =
            gert::KernelRunContextFaker()
                .KernelIONum(5, 4)  // 这里5 4 对吗？ 啥意思？
                .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
                .Outputs({&compile_info});
        this->kernel_holder = this->kernel_faker.Build();

        this->InitTilingParams(test_param);
        auto& tiling_params = this->tiling_params;
        auto& group = tiling_params.group;
        auto& ep_world_size = tiling_params.ep_world_size;
        auto& send_counts = tiling_params.send_counts;
        auto& recv_counts = tiling_params.recv_counts;
        auto& trans_gmm_weight = tiling_params.trans_gmm_weight;
        auto& trans_mm_weight = tiling_params.trans_mm_weight;
        auto& permute_out_flag = tiling_params.permute_out_flag;
        auto& is_Need_MM = tiling_params.is_Need_MM;

        this->InitTilingShape();
        auto& tiling_shapes = this->tiling_shapes;
        auto& gmm_x_shape = tiling_shapes.gmm_x_shape;
        auto& gmm_weight_shape = tiling_shapes.gmm_weight_shape;
        auto& send_counts_shape = tiling_shapes.send_counts_shape;
        auto& recv_counts_shape = tiling_shapes.recv_counts_shape;
        auto& mm_x_shape = tiling_shapes.mm_x_shape;
        auto& mm_weight_shape = tiling_shapes.mm_weight_shape;

        auto& gmm_y_shape = tiling_shapes.gmm_y_shape;
        auto& mm_y_shape = tiling_shapes.mm_y_shape;
        auto& permute_out_shape = tiling_shapes.permute_out_shape;

        auto mm_x = &mm_x_shape;
        auto mm_weight = &mm_weight_shape;
        auto mm_y = &mm_y_shape;
        if (is_Need_MM == false) {
            mm_x = nullptr;
            mm_weight = nullptr;
            mm_y = nullptr;
        }

        this->InitTilingDTypes(test_param.tiling_dTypes_pair);
        auto& tiling_dtypes = this->tiling_dtypes;

        auto input_num = this->input_num;
        auto output_num = this->output_num;

        std::string op_type("AlltoAllvGroupedMatMul");

        this->tiling_faker =
            gert::TilingContextFaker()
                .NodeIoNum(input_num, output_num)
                .IrInstanceNum({1, 1, 1, 1, 1, 1})
                .InputShapes({&gmm_x_shape, &gmm_weight_shape, nullptr, nullptr, mm_x, mm_weight})
                .OutputShapes({&gmm_y_shape, mm_y, &permute_out_shape})
                .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)},
                            {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(ep_world_size)},
                            {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)},
                            {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)},
                            {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(trans_gmm_weight)},
                            {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(trans_mm_weight)},
                            {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(permute_out_flag)}})
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
    int64_t input_num{6};
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
    static std::unordered_map<string,
                              std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
        tiling_params_vec_handlers;
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

string AlltoAllvGroupedMatMulTilingTest::compile_info_string = R"({
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
    AlltoAllvGroupedMatMulTilingTest::tiling_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BSK = std::stoi(value_str); }},
        {"BS", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BS = std::stoi(value_str); }},
        {"K", [](TilingParams& tiling_params, const string& value_str) { tiling_params.K = std::stoi(value_str); }},
        {"H1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H1 = std::stoi(value_str); }},
        {"H2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H2 = std::stoi(value_str); }},
        {"A", [](TilingParams& tiling_params, const string& value_str) { tiling_params.A = std::stoi(value_str); }},
        {"N1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N1 = std::stoi(value_str); }},
        {"N2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N2 = std::stoi(value_str); }},
        {"ep_world_size", [](TilingParams& tiling_params,
                             const string& value_str) { tiling_params.ep_world_size = std::stoi(value_str); }},
        {"e", [](TilingParams& tiling_params, const string& value_str) { tiling_params.e = std::stoi(value_str); }},
        {"gmm_weight_dim1", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.gmm_weight_dim1 = std::stoi(value_str); }},
        {"gmm_y_dim1",
         [](TilingParams& tiling_params, const string& value_str) { tiling_params.gmm_y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParams& tiling_params,
                              const string& value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight", [](TilingParams& tiling_params,
                                const string& value_str) { tiling_params.trans_gmm_weight = value_str == "true"; }},
        {"trans_mm_weight", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.trans_mm_weight = value_str == "true"; }},
        {"permute_out_flag", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.permute_out_flag = value_str == "true";
         }},
        {"is_Need_MM", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.is_Need_MM = value_str == "true";
         }}
        };

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
    AlltoAllvGroupedMatMulTilingTest::tiling_params_vec_handlers = {
        {"send_counts", [](TilingParams& tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParams& tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

TEST_P(AlltoAllvGroupedMatMulTilingTest, shape_size)
{
    auto test_param = GetParam();
    std::string op_type("AlltoAllvGroupedMatMul");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    string compile_info_string = AlltoAllvGroupedMatMulTilingTest::compile_info_string;
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> version = {{"Short_SoC_version", "Ascend910_93"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    auto tilingData = gert::TilingData::CreateCap(8192);
    ASSERT_NE(tilingData, nullptr);
    auto workspace_holer = gert::ContinuousVector::Create<size_t>(8192);
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

    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = tiling_params.ep_world_size;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(tiling_params.group.c_str(), topoInfo);
    EXPECT_EQ(tiling_func(tiling_context), test_param.status);
}

static TestParam test_params[] = {
    {"Test_sample", {{"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_SUCCESS},
    {"Test_gmmWeight_size", {{"e", "64"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_ep_world_size", {{"ep_world_size", "4"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_e", {{"e", "64"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_e_multi_ep",
     {{"ep_world_size", "16"}, {"e", "32"}, {"permute_out_flag", "true"}},
     {{"send_counts", std::vector<int64_t>(512, 128)}, {"recv_counts", std::vector<int64_t>(512, 128)}},
     {},
     ge::GRAPH_FAILED},
    {"Test_send_counts_size",
     {{"ep_world_size", "16"}, {"e", "32"}, {"permute_out_flag", "true"}},
     {},
     {},
     ge::GRAPH_FAILED},
    {"Test_BSK_1", {{"BSK", "52428800"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_BS_1", {{"BS", "52428800"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H1", {{"H1", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H2", {{"H2", "12289"}, {"mm_weight_dim0", "12289"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N1", {{"N1", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_N2", {{"N2", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_1", {{"H1", "7168"}, {"gmm_weight_dim1", "7169"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_3", {{"H2", "7168"}, {"mm_weight_dim0", "7169"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_H_4", {{"H1", "65536"}, {"permute_out_flag", "true"}}, {}, {}, ge::GRAPH_FAILED},
    {"Test_send_counts_0",
     {{"BSK", "16386"}, {"permute_out_flag", "true"}},
     {{"send_counts",
       std::vector<int64_t>{
           3201, 3201, 3200, 3200, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
           128,  128,  128,  128,  128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
       }}},
     {},
     ge::GRAPH_FAILED},
    {"Test_recv_counts_0",
     {{"A", "16386"}, {"BS", "8193"}, {"permute_out_flag", "true"}},
     {{"recv_counts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 3200}}},
     {},
     ge::GRAPH_FAILED},
    {"Test_recv_counts_1",
     {{"A", "16386"}, {"BS", "8193"}, {"permute_out_flag", "true"}},
     {{"recv_counts",
       std::vector<int64_t>{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,  128,  128,  128,  128, 128,
                            128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 3201, 3201, 3200, 1600, 1600}}},
     {},
     ge::GRAPH_FAILED},
    {"Test_no_MM", {{"permute_out_flag", "true"}, {"is_Need_MM", "false"}}, {}, {}, ge::GRAPH_SUCCESS}
    };

INSTANTIATE_TEST_SUITE_P(AlltoAllvGroupedMatMulTilingTest, AlltoAllvGroupedMatMulTilingTest,
                         testing::ValuesIn(test_params),
                         [](const testing::TestParamInfo<AlltoAllvGroupedMatMulTilingTest::ParamType>& info) {
                             return info.param.test_name;
                         });


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_fp16_basic) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1111);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_fp16_trans_gmm_w) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(16)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 16;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1010);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_bf16_trans_mm_w) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(32)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 32;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 101);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_fp16_trans_both) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1011);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_bf16_large_mm) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(16)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 16;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_fp16_small_shapes) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1100);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_bf16_no_permute) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 8;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_fp16_a5_path) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(32)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    map<string, string> version = {{"Short_SoC_version", "ascend910p"}};
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 32;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 1000000000000001024);
}


TEST_F(AlltoAllvGroupedMatMulTiling, alltoallv_gmm_bf16_boundary) {
    // 1. Setup interfaces
    std::string op_type("AlltoAllvGroupedMatMul");
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
    struct AlltoAllvGroupedMatMulCompileInfo {} compile_info;
    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(5, 4)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    // 3. Create context
    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(8192);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    gert::StorageShape gmmX_shape = {{4096, 7168}, {4096, 7168}};
    gert::StorageShape gmmWeight_shape = {{4, 7168, 4096}, {4, 7168, 4096}};
    gert::StorageShape gmmY_shape = {{4096, 4096}, {4096, 4096}};
    gert::StorageShape mmX_shape = {{2048, 7168}, {2048, 7168}};
    gert::StorageShape mmWeight_shape = {{7168, 64}, {7168, 64}};
    gert::StorageShape mmY_shape = {{2047, 64}, {2047, 64}};
    // 4. Build fake context
    std::string group("group");
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(6, 3)
                      .IrInstanceNum({1, 1, 1, 1, 1, 1})
                      .InputShapes({&gmmX_shape, &gmmWeight_shape, nullptr, nullptr, &mmX_shape, &mmWeight_shape})
                      .OutputShapes({&gmmY_shape, &mmY_shape, nullptr})
                      .NodeAttrs({{"group", ge::AnyValue::CreateFrom<std::string>(group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"send_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(send_counts)}, {"recv_counts", ge::AnyValue::CreateFrom<vector<int64_t>>(recv_counts)}, {"trans_gmm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"trans_mm_weight", ge::AnyValue::CreateFrom<bool>(false)}, {"permute_out_flag", ge::AnyValue::CreateFrom<bool>(false)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
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
    ge::HcomTopoInfo::TopoInfo topoInfo;
    topoInfo.rank_size = 64;
    topoInfo.topo_level_descs[0].comm_sets = 0b1U;
    ge::HcomTopoInfo::Instance().SetGroupTopoInfo(group.c_str(), topoInfo);
    // 6. Call op function
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    ge::HcomTopoInfo::Instance().UnsetGroupTopoInfo(group.c_str());
    // 11. Check tiling key
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
}

}
