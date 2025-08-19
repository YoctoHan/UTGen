# CANN 测试用例生成器 VS Code 扩展

该扩展提供一个可视化界面，封装运行仓库中的 `stage_1.py`，基于 few-shot 示例与源码分析生成算子测试参数（导出为 xlsx）。

## 功能
- 在命令面板执行 “CANN 测试用例生成器: 打开面板”
- 在 Webview 中填写算子名称、输出 xlsx 文件、prompt 文件、few-shot 文件、API Key、Base URL、模型名与源码目录
- 一键运行，实时在面板里查看日志与状态

## 使用
1. 在 VS Code 中打开本仓库
2. 安装依赖并构建：
   ```bash
   cd extension
   npm install
   npm run build
   ```
3. F5 进入扩展开发主机，或通过 “运行扩展” 启动
4. 通过命令面板执行 “CANN 测试用例生成器: 打开面板”

## 配置
- `cannTestcaseGenerator.pythonPath`: Python 可执行文件（默认 `python3`）
- `cannTestcaseGenerator.defaultScriptPath`: `stage_1.py` 的默认路径（默认工作区根目录）

## 运行时依赖
- Python3 及脚本所需依赖（`stage_1.py` 会调用 `utils` 中的工具，请确保环境就绪）

## 注意
- 需要可访问的 API（Base URL、API Key、Model Name）
- 输出路径、Prompt 文件路径、Few-shot 文件路径与源码目录，支持绝对路径或工作区内相对路径

