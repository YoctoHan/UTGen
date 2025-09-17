[2025-09-17 19:09:29] [ 60%] Building CXX object ops/built-in/tests/ut/op_tiling_test/CMakeFiles/ops_cpp_op_tiling_utest.dir/test_moe_distribute_combine.cpp.o
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_basic_small_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:93:562: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(32)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:93:562: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:93:868: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(32)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(32)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(2048)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:108:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:108:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:110:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:110:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:111:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:111:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:112:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:112:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:113:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:113:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:115:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:115:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_basic_large_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:183:566: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(256)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(128)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(256)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(32768)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:183:566: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:183:873: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(256)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(128)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(256)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(32768)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:198:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:198:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:200:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:200:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:201:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:201:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:202:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:202:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:203:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:203:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:205:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:205:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_tp_enabled_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:273:563: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(32)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(Non)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(4096)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ^~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:273:563: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:273:869: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(32)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(2)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(1)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(4096)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:288:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:288:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:290:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:290:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:291:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:291:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:292:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:292:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:293:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:293:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:295:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:295:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_shared_expert_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:363:565: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(128)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(64)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(128)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(8192)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:363:565: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:363:872: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(128)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(64)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(128)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(8192)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:378:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:378:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:380:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:380:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:381:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:381:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:382:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:382:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:383:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:383:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:385:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:385:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_max_bs_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:453:565: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(144)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(72)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(144)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(73728)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:453:565: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:453:872: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(144)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(72)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(144)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(73728)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:468:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:468:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:470:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:470:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:471:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:471:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:472:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:472:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:473:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:473:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:475:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:475:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_min_bs_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:543:560: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(8)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(8)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:543:560: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:543:863: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(8)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(8)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(8)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:558:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:558:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:560:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:560:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:561:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:561:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:562:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:562:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:563:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:563:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:565:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:565:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_boundary_k_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:633:563: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(32)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(Non)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(8192)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ^~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:633:563: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:633:869: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(32)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(8192)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:648:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:648:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:650:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:650:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:651:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:651:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:652:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:652:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:653:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:653:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:655:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:655:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_boundary_h_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:723:565: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(128)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(64)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(128)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(16384)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:723:565: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:723:872: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(128)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(64)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(128)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(16384)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:738:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:738:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:740:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:740:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:741:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:741:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:742:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:742:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:743:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:743:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:745:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:745:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_int8_quant_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:813:563: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(32)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(Non)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(4096)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(2)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   ^~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:813:563: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:813:869: error: no matching function for call to ‘gert::TilingContextFaker::NodeAttrs(<brace-enclosed initializer list>)’
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(64)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(32)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(64)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(4096)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(2)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     ^
In file included from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:10:
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note: candidate: ‘gert::TilingContextFaker& gert::TilingContextFaker::NodeAttrs(std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >)’
   TilingContextFaker &NodeAttrs(std::vector<std::pair<std::string, ge::AnyValue>> keys_to_value) {
                       ^~~~~~~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/common/utils/kernel_run_context_facker.h:273:23: note:   no known conversion for argument 1 from ‘<brace-enclosed initializer list>’ to ‘std::vector<std::pair<std::basic_string<char>, ge::AnyValue> >’
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:828:80: error: expected primary-expression before ‘>’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:828:82: error: expected primary-expression before ‘)’ token
     gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
                                                                                  ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:830:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:830:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:831:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:831:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:832:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:832:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:833:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:833:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:835:42: error: expected primary-expression before ‘>’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                          ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:835:44: error: expected primary-expression before ‘)’ token
     holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", version);
                                            ^
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp: In member function ‘virtual void MoeDistributeCombineTiling_moe_distribute_combine_david_scenario_Test::TestBody()’:
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:903:560: error: ‘None’ was not declared in this scope
                         .NodeAttrs({{"group_ep", ge::AnyValue::CreateFrom<std::string>(ep_group)}, {"ep_world_size", ge::AnyValue::CreateFrom<int64_t>(4)}, {"ep_rank_id", ge::AnyValue::CreateFrom<int64_t>(2)}, {"moe_expert_num", ge::AnyValue::CreateFrom<int64_t>(4)}, {"group_tp", ge::AnyValue::CreateFrom<std::string>(tp_group)}, {"tp_world_size", ge::AnyValue::CreateFrom<int64_t>(1)}, {"tp_rank_id", ge::AnyValue::CreateFrom<int64_t>(0)}, {"expert_shard_type", ge::AnyValue::CreateFrom<int64_t>(0)}, {"shared_expert_num", ge::AnyValue::CreateFrom<int64_t>(None)}, {"shared_expert_rank_num", ge::AnyValue::CreateFrom<int64_t>(0)}, {"global_bs", ge::AnyValue::CreateFrom<int64_t>(512)}, {"out_dtype", ge::AnyValue::CreateFrom<int64_t>(None)}, {"comm_quant_mode", ge::AnyValue::CreateFrom<int64_t>(0)}, {"group_list_type", ge::AnyValue::CreateFrom<int64_t>(None)}})
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ^~~~
/home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:903:560: note: suggested alternative:
In file included from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-internal.h:67,
                 from /home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/gtest.h:62,
                 from /home/hwx1446482/canndev-master/ops/built-in/tests/ut/op_tiling_test/test_moe_distribute_combine.cpp:6:
/home/hwx1446482/Ascend/latest/opensdk/opensdk/gtest/include/gtest/internal/gtest-type-util.h:112:8: note:   ‘testing::internal::None’
 struct None {};
        ^~~~
