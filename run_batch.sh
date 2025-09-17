# echo "1. AllGatherMatmul"
# # AllGatherMatmul
# OPERATOR_NAME=AllGatherMatmul
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/all_gather_matmul
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "2. AllGatherMatmulV2"
# # AllGatherMatmulV2
# OPERATOR_NAME=AllGatherMatmulV2
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/all_gather_matmul_v2
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "3. AllToAllGatherBatchMatmul"
# # AllToAllGatherBatchMatmul
# OPERATOR_NAME=AllToAllAllGatherBatchMatmul
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/all_to_all_all_gather_batch_matmul
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "4. BatchMatmulReduceScatterAllToAll"
# # BatchMatmulReduceScatterAllToAll
# OPERATOR_NAME=BatchMatmulReduceScatterAllToAll
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/batch_matmul_reduce_scatter_all_to_all
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "5. DistributeBarrier"
# # DistributeBarrier
# OPERATOR_NAME=DistributeBarrier
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/distribute_barrier
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "6. GroupedMatMulAlltoAllv"
# # GroupedMatMulAlltoAllv
# OPERATOR_NAME=GroupedMatMulAlltoAllv
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/grouped_mat_mul_allto_allv/
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "7. MoeDistributeCombineAddRmsNorm"
# # MoeDistributeCombineAddRmsNorm
# OPERATOR_NAME=MoeDistributeCombineAddRmsNorm
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/moe_distribute_combine_add_rms_norm
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "8. MoeDistributeCombineV2"
# # MoeDistributeCombineV2
# OPERATOR_NAME=MoeDistributeCombineV2
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/moe_distribute_combine_v2
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "9. MoeDistributeDispatchV2"
# # MoeDistributeDispatchV2
# OPERATOR_NAME=MoeDistributeDispatchV2
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/moe_distribute_dispatch_v2
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "10. MoeEplbUpdateExpert"
# # MoeEplbUpdateExpert
# OPERATOR_NAME=MoeEplbUpdateExpert
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/moe_eplb_update_expert
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "11. MatmulReduceScatter"
# OPERATOR_NAME=MatmulReduceScatter
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/matmul_reduce_scatter
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "12. MatmulReduceScatterV2"
# # MatmulReduceScatterV2 
# OPERATOR_NAME=MatmulReduceScatterV2
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/matmul_reduce_scatter_v2
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "13. MatmulAllReduce"
# # MatmulAllReduce
# OPERATOR_NAME=MatmulAllReduce
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/matmul_all_reduce
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------" 
# echo "14. MoeDistributeDispatch"
# # MoeDistributeDispatch
# OPERATOR_NAME=MoeDistributeDispatch
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/moe_distribute_dispatch
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
# echo "----------------------------------------------------------------------------------------"
# echo "15. MoeDistributeCombine"
# # MoeDistributeCombine
# OPERATOR_NAME=MoeDistributeCombine
# OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/moe_distribute_combine
# ./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
echo "----------------------------------------------------------------------------------------"
echo "16. AlltoAllvGroupedMatMulTiling"
# AlltoAllvGroupedMatMulTiling
OPERATOR_NAME=AlltoAllvGroupedMatMulTiling
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/allto_allv_grouped_mat_mul
./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH