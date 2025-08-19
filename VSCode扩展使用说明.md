# VSCode 扩展使用说明

## 🎯 最新更新

### 简化界面设计 (NEW!)
- ✅ **移除了 API 配置输入框** - API Key、Base URL、Model Name 现在统一在 `config.sh` 中配置
- ✅ **简化了表单界面** - 只保留必要的输入：算子名称、源码目录、Few-shot文件
- ✅ **更安全的配置管理** - 敏感信息不再通过界面输入，避免泄露风险

### 之前的改动
1. ✅ **执行 Shell 脚本而非 Python 脚本** - 现在调用 `entrypoint.sh` 而不是 `stage_1.py`
2. ✅ **自动激活项目级虚拟环境** - 在多个层面实现了虚拟环境自动激活

## 修改内容详情

我已经完成了以下修改：

### 1. **extension.ts 修改**
- ✅ 从调用 Python 脚本改为调用 Shell 脚本 (`entrypoint.sh`)
- ✅ 添加了虚拟环境自动激活功能
- ✅ 支持通过配置选择是否使用虚拟环境

### 2. **entrypoint.sh 修改**
- ✅ 改为接收 VSCode 扩展传递的参数
- ✅ 自动激活项目级虚拟环境 (`.venv`)
- ✅ 将参数传递给 `workflow.sh` 执行实际任务

### 3. **配置文件更新**
- ✅ 更新了 `package.json` 添加虚拟环境配置项
- ✅ 创建了 `.vscode/settings.json` 示例配置

## 虚拟环境激活位置

虚拟环境会在**两个地方**自动激活：

1. **VSCode 扩展层面** (extension.ts)
   - 如果配置 `useVirtualEnv: true`，会在执行脚本前激活虚拟环境
   
2. **Shell 脚本层面** (entrypoint.sh)
   - 脚本开始时会检查并激活 `.venv` 虚拟环境
   - 这提供了双重保障

## 使用步骤

### 1. 配置 API（重要！）
编辑 `config.sh` 文件，设置您的 API 配置：
```bash
export API_KEY="your-api-key"
export BASE_URL="https://your-api-endpoint/v1"
export MODEL_NAME="your-model-name"
```

### 2. 创建虚拟环境（如果还没有）
```bash
python3 -m venv .venv
source .venv/bin/activate  # 或使用 source activate.sh
pip install -r requirements.txt
```

### 3. 编译扩展
```bash
cd extension
npm install
npm run build
```

### 4. 在 VSCode 中使用
1. 按 `F5` 启动扩展开发主机
2. 运行命令：`CANN 测试用例生成器: 打开面板`
3. 填写必要信息：
   - **算子名称**：如 AllGatherMatmul
   - **源码目录**：算子源码路径（支持多个）
   - **Few-shot文件**：可选，默认使用 tiling-examples/fewshot_examples.txt
4. 点击"开始生成"

## 配置选项

在 VSCode 设置中可以配置：

- `cannTestcaseGenerator.useVirtualEnv`: 是否使用虚拟环境（默认：true）
- `cannTestcaseGenerator.venvPath`: 虚拟环境路径（默认：.venv）
- `cannTestcaseGenerator.defaultScriptPath`: 入口脚本路径（默认：entrypoint.sh）

## 工作流程

```
VSCode 扩展
    ↓
激活虚拟环境（如果配置）
    ↓
执行 entrypoint.sh
    ↓
再次确认虚拟环境激活
    ↓
调用 workflow.sh
    ↓
执行实际的测试用例生成
```

## 注意事项

1. **必须先配置 API**：在 `config.sh` 中设置 API_KEY、BASE_URL 和 MODEL_NAME
2. 确保 `entrypoint.sh` 有执行权限：`chmod +x entrypoint.sh`
3. 虚拟环境路径默认为 `.venv`，如需更改请修改配置
4. API 配置统一在 `config.sh` 中管理，更安全且便于维护
5. Few-shot文件可选，不填则使用默认配置
