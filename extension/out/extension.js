"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const vscode = __importStar(require("vscode"));
const path = __importStar(require("path"));
const child_process_1 = require("child_process");
// Util: get configuration values
function getConfig(key, defaultValue) {
    const config = vscode.workspace.getConfiguration('cannTestcaseGenerator');
    const value = config.get(key);
    return (value === undefined ? defaultValue : value);
}
function activate(context) {
    const disposable = vscode.commands.registerCommand('cannTestcaseGenerator.open', async () => {
        const panel = vscode.window.createWebviewPanel('cannTestcaseGenerator', 'CANN 测试用例生成器', vscode.ViewColumn.One, {
            enableScripts: true,
            retainContextWhenHidden: true,
            localResourceRoots: [
                vscode.Uri.file(path.join(context.extensionPath, 'media'))
            ]
        });
        const mediaUri = vscode.Uri.file(path.join(context.extensionPath, 'media'));
        const cssUri = panel.webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, 'ui.css'));
        const jsUri = panel.webview.asWebviewUri(vscode.Uri.joinPath(mediaUri, 'ui.js'));
        panel.webview.html = getWebviewContent(cssUri.toString(), jsUri.toString());
        panel.webview.onDidReceiveMessage(async (message) => {
            switch (message.type) {
                case 'pickFile': {
                    const uris = await vscode.window.showOpenDialog({ canSelectFiles: true, canSelectFolders: false, canSelectMany: false });
                    if (uris && uris.length > 0) {
                        panel.webview.postMessage({ type: 'pickedFile', path: uris[0].fsPath });
                    }
                    break;
                }
                case 'pickFolder': {
                    const uris = await vscode.window.showOpenDialog({ canSelectFiles: false, canSelectFolders: true, canSelectMany: true });
                    if (uris && uris.length > 0) {
                        panel.webview.postMessage({ type: 'pickedFolder', paths: uris.map(u => u.fsPath) });
                    }
                    break;
                }
                case 'run': {
                    runStage1(panel, context, message.payload).catch(err => {
                        vscode.window.showErrorMessage(String(err));
                    });
                    break;
                }
            }
        });
    });
    context.subscriptions.push(disposable);
}
function getWebviewContent(cssHref, jsSrc) {
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <link rel="stylesheet" href="${cssHref}" />
  <title>CANN 测试用例生成器</title>
  <style>
    .hidden { display: none; }
  </style>
</head>
<body>
  <div class="container">
    <h2>基于源码与 Few-shot 的测试用例生成</h2>
    <form id="form">
      <div class="row">
        <label>算子名称</label>
        <input id="operatorName" placeholder="如: AllGatherMatmul" />
      </div>
      <div class="row">
        <label>输出 Excel 文件 (.xlsx)</label>
        <input id="outputFile" placeholder="绝对路径，或工作区下相对路径" />
        <button type="button" data-action="pickFile" data-target="outputFile">选择</button>
      </div>
      <div class="row">
        <label>Prompt 文件 (.txt)</label>
        <input id="promptFile" placeholder="绝对路径，或工作区下相对路径" />
        <button type="button" data-action="pickFile" data-target="promptFile">选择</button>
      </div>
      <div class="row">
        <label>Few-shot 示例文件</label>
        <input id="fewshotFile" placeholder="如: tiling-examples/fewshot_examples.txt" />
        <button type="button" data-action="pickFile" data-target="fewshotFile">选择</button>
      </div>
      <div class="row">
        <label>API Key</label>
        <input id="apiKey" type="password" />
      </div>
      <div class="row">
        <label>Base URL</label>
        <input id="baseUrl" placeholder="如: https://api.com/v1" />
      </div>
      <div class="row">
        <label>Model Name</label>
        <input id="modelName" placeholder="模型名称" />
      </div>
      <div class="row">
        <label>源码目录(可多选)</label>
        <textarea id="sourcePaths" rows="2" placeholder="多行/逗号分隔"></textarea>
        <button type="button" data-action="pickFolder" data-target="sourcePaths">选择</button>
      </div>
      <div class="row">
        <button id="run" type="submit">开始生成</button>
      </div>
      <div id="status" class="status"></div>
      <pre id="log" class="log"></pre>
    </form>
  </div>
  <script src="${jsSrc}"></script>
</body>
</html>`;
}
async function runStage1(panel, context, payload) {
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath;
    const pythonCmd = getConfig('pythonPath', 'python3');
    const defaultScript = getConfig('defaultScriptPath', path.join(workspaceFolder || context.extensionPath, 'stage_1.py'));
    // Resolve script path
    const scriptPath = payload && payload.scriptPath ? payload.scriptPath : defaultScript;
    const args = [
        scriptPath,
        payload.operatorName,
        toAbsolute(payload.outputFile, workspaceFolder),
        toAbsolute(payload.promptFile, workspaceFolder),
        toAbsolute(payload.fewshotFile, workspaceFolder),
        payload.apiKey,
        payload.baseUrl,
        payload.modelName,
        ...payload.sourcePaths.map(p => toAbsolute(p, workspaceFolder))
    ];
    panel.webview.postMessage({ type: 'status', text: '开始执行 stage_1.py ...' });
    const proc = (0, child_process_1.spawn)(pythonCmd, args, { cwd: workspaceFolder || context.extensionPath, shell: process.platform === 'win32' });
    proc.stdout.on('data', (data) => {
        panel.webview.postMessage({ type: 'log', text: data.toString() });
    });
    proc.stderr.on('data', (data) => {
        panel.webview.postMessage({ type: 'log', text: data.toString() });
    });
    await new Promise((resolve) => {
        proc.on('close', (code) => {
            if (code === 0) {
                panel.webview.postMessage({ type: 'status', text: '完成 ✅' });
                vscode.window.showInformationMessage('测试参数生成完成');
            }
            else {
                panel.webview.postMessage({ type: 'status', text: `失败，退出码 ${code}` });
                vscode.window.showErrorMessage(`stage_1.py 运行失败，退出码 ${code}`);
            }
            resolve();
        });
    });
}
function toAbsolute(p, workspaceFolder) {
    if (!p)
        return p;
    if (path.isAbsolute(p))
        return p;
    if (workspaceFolder)
        return path.join(workspaceFolder, p);
    return path.resolve(p);
}
function deactivate() { }
//# sourceMappingURL=extension.js.map