# AllGatherMatmul
OPERATOR_NAME=AllGatherMatmul
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/all_gather_matmul
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# AllGatherMatmulV2
OPERATOR_NAME=AllGatherMatmulV2
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/all_gather_matmul_v2
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# MatmulReduceScatter
OPERATOR_NAME=MatmulReduceScatter
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/matmul_reduce_scatter
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# MatmulReduceScatterV2 
OPERATOR_NAME=MatmulReduceScatterV2
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/matmul_reduce_scatter_v2
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# MatmulAllReduce
OPERATOR_NAME=MatmulAllReduce
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/matmul_all_reduce
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# MoeDistributeDispatch
OPERATOR_NAME=MoeDistributeDispatch
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/moe_distribute_dispatch
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# MoeDistributeCombine
OPERATOR_NAME=MoeDistributeCombine
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/moe_distribute_combine
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH