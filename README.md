# 算子单测自动生成工具 v2.0

## 📋 概述

这是一个基于大语言模型的算子单元测试自动生成工具，支持华为CANN算子的测试代码生成。工具采用两阶段设计：Stage 1 通过大模型生成参数；Stage 2 基于参考UT与xlsx参数进行工程化转换生成gtest代码。

### 核心特性

- 🎯 **两阶段解耦设计**：测试参数生成和单测代码生成相互独立
- 🤖 **AI驱动生成**：基于大模型理解算子代码并生成高质量测试
- 📊 **Few-shot学习**：通过示例学习生成符合规范的测试代码
- 🔄 **灵活的工作流**：支持单独运行各阶段或完整流程
- 📝 **自动化程度高**：从源码到测试代码全流程自动化

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone <repository_url>
cd utgen-v2

# 安装Python依赖
pip install -r requirements.txt

# 配置API密钥（编辑config.sh）
vim config.sh
```

### 2. 配置文件

编辑 `config.sh` 文件，设置您的API配置：

```bash
export API_KEY="your-api-key"
export BASE_URL="https://your-api-endpoint.com/v1/"
export MODEL_NAME="your-model-name"
```

### 3. 基本使用

```bash
# 查看帮助
./workflow.sh --help

# 完整流程：生成测试参数 + 生成单测代码
./workflow.sh stage-all AllGatherMatmul ../path/to/operator/source

# 仅生成测试参数（Stage 1）
./workflow.sh stage-1 AllGatherMatmul ../path/to/operator/source

# 仅生成单测代码（Stage 2）
./workflow.sh stage-2 AllGatherMatmul ../path/to/operator/source
```

## 📁 项目结构

```
utgen-v2/
├── workflow.sh          # 主入口脚本
├── config.sh           # 配置文件
├── entrypoint.sh       # 快速启动脚本
│
├── stage_1.py              # Stage 1: 测试参数生成器
├── convert_ut_from_xlsx.py # Stage 2: 工程化转换（参考UT+xlsx → gtest）
├── utils.py               # 通用工具函数
│
├── ut-template/       # 单测模板目录
│   └── ut_template.cpp
├── tiling-examples/   # Few-shot示例目录
│   ├── test_*.cpp
│   └── *.xlsx
├── test-examples/     # 其他测试示例
└── runs/             # 输出目录（自动生成）
    └── YYYYMMDD_HHMMSS_operatorname/
        ├── test_*.cpp
        ├── test_params_*.xlsx
        ├── prompt_*.txt
        └── *.log
```

## 🔄 工作流程

### Stage 1: 测试参数生成

1. **收集Few-shot示例（Stage1）**：从`tiling-examples`目录加载文本few-shot
2. **分析目标算子源码**：解析目标算子的接口和参数
3. **生成测试参数**：通过AI模型生成Excel（xlsx）格式的测试参数组合

```bash
./workflow.sh stage-1 MatmulAllReduce ../ops/matmul_all_reduce
```

输出：
- `test_params_matmulallreduce.xlsx` - 测试参数文件
- `prompt_testcase_matmulallreduce.txt` - 生成时使用的prompt

### Stage 2: 单测代码生成（工程化）

1. **选择参考UT**：从 `REFERENCE_UT_DIR` 中选择 `test_<snake>.cpp`
2. **抽取公共部分**：移除参考UT中的所有 `TEST_F`，保留公共代码
3. **渲染用例**：读取xlsx中每一行参数，渲染为 `TEST_F` 测例

```bash
./workflow.sh stage-2 MatmulAllReduce ../ops/matmul_all_reduce
```

输出：
- `test_matmulallreduce_tiling.cpp` - 完整的单测代码
- `generation.log` - 生成日志

## 🛠️ 高级配置

### 环境变量

在 `config.sh` 中可配置：

- `API_KEY` - API密钥
- `BASE_URL` - API端点地址
- `MODEL_NAME` - 使用的模型名称
- `MAX_FILE_SIZE` - 最大文件大小限制（默认2MB）
- `MAX_RETRIES` - API调用重试次数（默认5次）
- `ENABLE_AUTO_CSV_SEARCH` - 是否自动查找测试参数文件

### Few-shot示例管理

1. **重要示例**：放在 `tiling-examples/` 目录
   - 包含完整的测试代码（.cpp）
   - 包含测试参数文件（.xlsx或.csv）

2. **一般示例**：放在 `test-examples/` 目录
   - 提供测试代码结构参考

### 自定义模板

编辑 `ut-template/ut_template.cpp` 来定制生成的单测代码结构。

## 📊 输出说明

每次运行会在 `runs/` 目录下创建带时间戳的子目录：

```
runs/20250102_143022_allgathermatmul/
├── test_allgathermatmul_tiling.cpp    # 生成的单测代码
├── test_params_allgathermatmul.xlsx   # 测试参数（Stage 1）
├── prompt_testcase_allgathermatmul.txt# Stage1的prompt
└── generation.log                     # 生成日志
```

## 🐛 故障排除

### 常见问题

1. **API调用失败**
   - 检查API密钥是否正确
   - 确认网络连接正常
   - 查看日志文件了解详细错误

2. **生成代码不完整**
   - 增加 `MAX_RETRIES` 值
   - 检查源码文件是否过大
   - 确认模型支持的最大token数

3. **找不到示例文件**
   - 确保 `tiling-examples/` 目录存在
   - 检查文件命名是否符合规范

### 日志查看

```bash
# 查看最新的生成日志
ls -t runs/*/generation.log | head -1 | xargs cat

# 查看特定算子的所有运行记录
ls -la runs/*allgathermatmul*/
```

## 📝 最佳实践

1. **准备高质量的Few-shot示例**
   - 提供完整、规范的测试代码示例
   - 包含多样化的测试参数组合

2. **源码路径指定**
   - 提供完整的算子实现目录
   - 可以指定多个源码路径

3. **迭代优化**
   - 先运行Stage 1生成参数
   - 检查并调整参数后再运行Stage 2
   - 根据生成结果完善参考UT或参数列，提升覆盖率

## 🤝 贡献指南

欢迎提交Issue和Pull Request来改进工具。

### 开发环境设置

```bash
# 安装开发依赖
pip install -r requirements.txt

# 运行测试
pytest tests/

# 代码格式化
black *.py
isort *.py
```

## 📄 许可证

本项目采用 MIT 许可证。

## 📧 联系方式

如有问题或建议，请提交Issue或联系维护者。

---

**注意**：使用本工具生成的代码需要人工审核和测试，确保符合项目规范和质量要求。
