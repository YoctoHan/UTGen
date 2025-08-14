# =============================================================================
# 配置区域 - 根据需要修改
# =============================================================================
OPERATOR_NAME=AllGatherMatmul
OPERATOR_PATH=/Users/edy/Desktop/华为/canndev-utgen/ops/built-in/op_tiling/runtime/all_gather_matmul

# =============================================================================
# 执行区域 - 选择要执行的操作
# =============================================================================

# 选项1: 仅生成单元测试 
# ./workflow.sh $OPERATOR_NAME $OPERATOR_PATH

# 选项2: 仅生成测试参数
# ./workflow.sh gen-params $OPERATOR_NAME $OPERATOR_PATH

# 选项3: 完整流程 (先生成参数，再生成单测)
./workflow.sh stage-1 $OPERATOR_NAME $OPERATOR_PATH

# 选项4: 查看配置
# ./workflow.sh --config

# 选项5: 查看帮助
# ./workflow.sh --help

