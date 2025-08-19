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
        <label>算子名称 <span style="color: red;">*</span></label>
        <input id="operatorName" placeholder="如: AllGatherMatmul" required />
      </div>
      <div class="row">
        <label>源码目录 <span style="color: red;">*</span></label>
        <textarea id="sourcePaths" rows="3" placeholder="算子源码目录路径（支持多个，每行一个）" required></textarea>
        <button type="button" data-action="pickFolder" data-target="sourcePaths">选择目录</button>
      </div>
      <div class="row">
        <label>Few-shot 示例文件</label>
        <input id="fewshotFile" placeholder="默认: tiling-examples/fewshot_examples.txt" />
        <button type="button" data-action="pickFile" data-target="fewshotFile">选择文件</button>
      </div>
      <div class="row">
        <button id="run" type="submit">开始生成</button>
      </div>
      <div class="row" style="margin-top: 10px; font-size: 12px; color: #666;">
        <p>💡 提示：API配置请在 config.sh 中设置</p>
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
    // 输出调试信息
    panel.webview.postMessage({ type: 'log', text: `📂 工作区目录: ${workspaceFolder || '未设置'}\n` });
    panel.webview.postMessage({ type: 'log', text: `📂 扩展目录: ${context.extensionPath}\n` });
    // 配置项：是否使用虚拟环境
    const useVenv = getConfig('useVirtualEnv', true);
    const venvPath = getConfig('venvPath', '.venv');
    // 计算脚本路径：优先使用设置项；若为空则使用扩展父目录的 entrypoint.sh
    const extensionParentDir = path.dirname(context.extensionPath);
    const configuredScriptPath = getConfig('defaultScriptPath', '');
    const resolvedConfiguredScript = configuredScriptPath && configuredScriptPath.trim() !== ''
        ? toAbsolute(configuredScriptPath, extensionParentDir)
        : '';
    const autoDefaultScript = path.join(extensionParentDir, 'entrypoint.sh');
    const selectedDefaultScript = resolvedConfiguredScript || autoDefaultScript;
    let scriptPath = payload?.scriptPath;
    if (!scriptPath || String(scriptPath).trim() === '') {
        scriptPath = selectedDefaultScript;
    }
    panel.webview.postMessage({ type: 'log', text: `📜 执行脚本: ${scriptPath}\n` });
    // 校验脚本是否存在
    const fs = require('fs');
    if (!fs.existsSync(scriptPath)) {
        panel.webview.postMessage({ type: 'status', text: '失败 ❌' });
        panel.webview.postMessage({ type: 'log', text: `❌ 找不到脚本: ${scriptPath}\n` });
        vscode.window.showErrorMessage(`找不到脚本: ${scriptPath}`);
        return;
    }
    // 构建命令
    let command;
    let commandArgs = [];
    if (useVenv) {
        // 激活虚拟环境并执行脚本
        const isWindows = process.platform === 'win32';
        // 优先使用扩展父目录（utgen-v2）的虚拟环境，其次是工作区的虚拟环境
        const possibleVenvPaths = [
            path.join(extensionParentDir, venvPath), // utgen-v2/.venv
            workspaceFolder ? path.join(workspaceFolder, venvPath) : null // 工作区/.venv
        ].filter((p) => p !== null);
        let activateScript = '';
        let venvExists = false;
        for (const venvDir of possibleVenvPaths) {
            const testScript = isWindows
                ? path.join(venvDir, 'Scripts', 'activate.bat')
                : path.join(venvDir, 'bin', 'activate');
            if (fs.existsSync(testScript)) {
                activateScript = testScript;
                venvExists = true;
                panel.webview.postMessage({ type: 'log', text: `✅ 找到虚拟环境: ${venvDir}\n` });
                break;
            }
        }
        if (isWindows) {
            // Windows: 使用 cmd.exe
            command = 'cmd.exe';
            const cdCmd = workspaceFolder ? `cd /d "${workspaceFolder}" && ` : '';
            if (venvExists) {
                commandArgs = ['/c', `"${activateScript}" && ${cdCmd}"${scriptPath}" "${payload.operatorName}" ${payload.fewshotFile ? `"${toAbsolute(payload.fewshotFile, workspaceFolder)}"` : ''} ${payload.sourcePaths.map(p => `"${toAbsolute(p, workspaceFolder)}"`).join(' ')}`];
            }
            else {
                commandArgs = ['/c', `${cdCmd}"${scriptPath}" "${payload.operatorName}" ${payload.fewshotFile ? `"${toAbsolute(payload.fewshotFile, workspaceFolder)}"` : ''} ${payload.sourcePaths.map(p => `"${toAbsolute(p, workspaceFolder)}"`).join(' ')}`];
            }
        }
        else {
            // macOS/Linux: 使用 bash
            command = '/bin/bash';
            const cdCmd = workspaceFolder ? `cd "${workspaceFolder}" && ` : '';
            if (venvExists) {
                commandArgs = ['-c', `source "${activateScript}" && ${cdCmd}"${scriptPath}" "${payload.operatorName}" ${payload.fewshotFile ? `"${toAbsolute(payload.fewshotFile, workspaceFolder)}"` : ''} ${payload.sourcePaths.map(p => `"${toAbsolute(p, workspaceFolder)}"`).join(' ')}`];
            }
            else {
                // 如果虚拟环境不存在，直接执行脚本（脚本内部会尝试激活）
                commandArgs = ['-c', `${cdCmd}"${scriptPath}" "${payload.operatorName}" ${payload.fewshotFile ? `"${toAbsolute(payload.fewshotFile, workspaceFolder)}"` : ''} ${payload.sourcePaths.map(p => `"${toAbsolute(p, workspaceFolder)}"`).join(' ')}`];
            }
        }
        if (!venvExists) {
            panel.webview.postMessage({ type: 'log', text: `⚠️ 未找到虚拟环境，检查过以下位置:\n` });
            for (const venvDir of possibleVenvPaths) {
                panel.webview.postMessage({ type: 'log', text: `  - ${venvDir}\n` });
            }
            panel.webview.postMessage({ type: 'log', text: `将尝试使用系统Python环境...\n` });
        }
    }
    else {
        // 直接执行脚本
        command = scriptPath;
        commandArgs = [
            payload.operatorName,
            ...(payload.fewshotFile ? [toAbsolute(payload.fewshotFile, workspaceFolder)] : []),
            ...payload.sourcePaths.map(p => toAbsolute(p, workspaceFolder))
        ];
    }
    panel.webview.postMessage({ type: 'status', text: '开始执行脚本...' });
    // 使用扩展父目录作为工作目录，这样脚本可以找到config.sh等文件
    const proc = (0, child_process_1.spawn)(command, commandArgs, {
        cwd: extensionParentDir, // utgen-v2 目录
        shell: false,
        env: { ...process.env }
    });
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
                vscode.window.showInformationMessage('测试用例生成完成');
            }
            else {
                panel.webview.postMessage({ type: 'status', text: `失败，退出码 ${code}` });
                vscode.window.showErrorMessage(`脚本运行失败，退出码 ${code}`);
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