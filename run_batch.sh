# AllGatherMatmul
OPERATOR_NAME=AllGatherMatmul
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/all_gather_matmul
./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH

# AllGatherMatmulV2
OPERATOR_NAME=AllGatherMatmulV2
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/all_gather_matmul_v2
./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH

# MatmulReduceScatter
OPERATOR_NAME=MatmulReduceScatter
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/matmul_reduce_scatter
./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH

# MatmulReduceScatterV2 
OPERATOR_NAME=MatmulReduceScatterV2
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/matmul_reduce_scatter_v2
./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH

# MatmulAllReduce
OPERATOR_NAME=MatmulAllReduce
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev/ops/built-in/op_tiling/runtime/matmul_all_reduce
./entrypoint.sh  $OPERATOR_NAME $OPERATOR_PATH
