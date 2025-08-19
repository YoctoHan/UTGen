(function() {
  const vscode = acquireVsCodeApi();

  const form = document.getElementById('form');
  const logEl = document.getElementById('log');
  const statusEl = document.getElementById('status');

  function appendLog(text) {
    logEl.textContent += text;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function setStatus(text) {
    statusEl.textContent = text;
  }

  document.querySelectorAll('button[data-action]')
    .forEach(btn => {
      btn.addEventListener('click', (e) => {
        const action = btn.getAttribute('data-action');
        const target = btn.getAttribute('data-target');
        if (action === 'pickFile') {
          vscode.postMessage({ type: 'pickFile' });
          btn.dataset.activeTarget = target;
        } else if (action === 'pickFolder') {
          vscode.postMessage({ type: 'pickFolder' });
          btn.dataset.activeTarget = target;
        }
      });
    });

  window.addEventListener('message', (event) => {
    const msg = event.data;
    if (msg.type === 'pickedFile') {
      const input = document.getElementById('' + document.querySelector('button[data-active-target]')?.getAttribute('data-active-target'));
      const targetId = document.querySelector('button[data-active-target]')?.getAttribute('data-active-target');
      if (targetId) {
        document.getElementById(targetId).value = msg.path;
      }
    } else if (msg.type === 'pickedFolder') {
      const targetId = document.querySelector('button[data-active-target]')?.getAttribute('data-active-target');
      if (targetId) {
        const current = document.getElementById(targetId).value.trim();
        const list = msg.paths.join('\n');
        document.getElementById(targetId).value = current ? (current + '\n' + list) : list;
      }
    } else if (msg.type === 'log') {
      appendLog(String(msg.text || ''));
    } else if (msg.type === 'status') {
      setStatus(String(msg.text || ''));
    }
  });

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const payload = {
      operatorName: document.getElementById('operatorName').value.trim(),
      outputFile: document.getElementById('outputFile').value.trim(),
      promptFile: document.getElementById('promptFile').value.trim(),
      fewshotFile: document.getElementById('fewshotFile').value.trim(),
      apiKey: document.getElementById('apiKey').value.trim(),
      baseUrl: document.getElementById('baseUrl').value.trim(),
      modelName: document.getElementById('modelName').value.trim(),
      sourcePaths: (document.getElementById('sourcePaths').value || '')
        .split(/\n|,/).map(s => s.trim()).filter(Boolean)
    };
    logEl.textContent = '';
    setStatus('准备执行 ...');
    vscode.postMessage({ type: 'run', payload });
  });
})();


