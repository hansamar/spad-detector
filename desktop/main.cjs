const { app, BrowserWindow, Menu, dialog, shell } = require('electron');
const { spawn } = require('node:child_process');
const http = require('node:http');
const path = require('node:path');
const { selectBackendPython } = require('./backend-python.cjs');

const BACKEND_PORT = Number(process.env.SPAD_BACKEND_PORT || 8000);
const BACKEND_HOST = '127.0.0.1';
const BACKEND_URL = `http://${BACKEND_HOST}:${BACKEND_PORT}`;
const DEV_SERVER_URL = process.env.SPAD_FRONTEND_URL || 'http://127.0.0.1:3000';
const isDev = !app.isPackaged || process.env.SPAD_DESKTOP_DEV === '1';

let backendProcess = null;
let mainWindow = null;

function resolveAppRoot() {
  if (app.isPackaged) {
    return app.getAppPath();
  }
  return path.resolve(__dirname, '..');
}

function findBackendPython() {
  const selection = selectBackendPython({ appRoot: resolveAppRoot() });
  if (selection.selected?.cuda_available) {
    return selection.selected;
  }
  if (process.env.SPAD_REQUIRE_CUDA !== '0') {
    return null;
  }
  return selection.selected;
}

function isBackendHealthy() {
  return new Promise((resolve) => {
    const req = http.get(`${BACKEND_URL}/api/health`, { timeout: 800 }, (res) => {
      let body = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => {
        body += chunk;
      });
      res.on('end', () => {
        if (res.statusCode !== 200) {
          resolve(false);
          return;
        }
        try {
          const payload = JSON.parse(body);
          resolve(payload && payload.status === 'ok');
        } catch (_error) {
          resolve(false);
        }
      });
    });
    req.on('timeout', () => {
      req.destroy();
      resolve(false);
    });
    req.on('error', () => resolve(false));
  });
}

async function waitForBackend(timeoutMs = 30000) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (await isBackendHealthy()) return true;
    await new Promise((resolve) => setTimeout(resolve, 400));
  }
  return false;
}

async function startBackend() {
  if (await isBackendHealthy()) return;

  const root = resolveAppRoot();
  const python = findBackendPython();
  if (!python) {
    dialog.showErrorBox(
      'SPAD backend requires CUDA',
      'No CUDA-capable Python environment was found. Set SPAD_PYTHON_EXE to a Python with torch+CUDA, or set SPAD_REQUIRE_CUDA=0 to allow CPU fallback.',
    );
    return;
  }
  const args = [
    ...python.argsPrefix,
    '-m',
    'uvicorn',
    'backend.main:app',
    '--host',
    BACKEND_HOST,
    '--port',
    String(BACKEND_PORT),
  ];

  backendProcess = spawn(python.command, args, {
    cwd: root,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      SPAD_DESKTOP: '1',
      SPAD_BACKEND_PORT: String(BACKEND_PORT),
      SPAD_SELECTED_PYTHON: python.selected_python,
    },
    windowsHide: true,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  backendProcess.stdout.on('data', (chunk) => {
    console.log(`[backend] ${chunk.toString().trim()}`);
  });
  backendProcess.stderr.on('data', (chunk) => {
    console.error(`[backend] ${chunk.toString().trim()}`);
  });
  backendProcess.on('exit', (code) => {
    if (code !== 0 && mainWindow && !mainWindow.isDestroyed()) {
      mainWindow.webContents.send('spad-backend-exit', { code });
    }
    backendProcess = null;
  });

  const ready = await waitForBackend();
  if (!ready) {
    dialog.showErrorBox(
      'SPAD 后端启动失败',
      `无法启动 FastAPI 仿真后端。\n\nPython: ${python.selected_python}\n端口: ${BACKEND_PORT}\nCUDA: ${python.cuda_available ? '已启用' : '未启用'}\n\n请确认已安装 requirements.txt 中的 Python 依赖。`,
    );
  }
}

function frontendEntryUrl() {
  if (isDev) return DEV_SERVER_URL;
  return `file://${path.join(resolveAppRoot(), 'dist', 'index.html').replace(/\\/g, '/')}`;
}

function createMenu() {
  const template = [
    {
      label: '文件',
      submenu: [
        { label: '重新加载界面', accelerator: 'Ctrl+R', click: () => mainWindow?.reload() },
        { type: 'separator' },
        { label: '退出', accelerator: 'Alt+F4', click: () => app.quit() },
      ],
    },
    {
      label: '视图',
      submenu: [
        { role: 'togglefullscreen', label: '全屏' },
        { role: 'zoomIn', label: '放大' },
        { role: 'zoomOut', label: '缩小' },
        { role: 'resetZoom', label: '重置缩放' },
      ],
    },
    {
      label: '帮助',
      submenu: [
        {
          label: '打开数据目录',
          click: () => shell.openPath(path.join(resolveAppRoot(), 'output')),
        },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 920,
    minWidth: 1180,
    minHeight: 760,
    title: 'SPAD Detector Professional Simulator',
    backgroundColor: '#0b1020',
    show: false,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.once('ready-to-show', () => mainWindow.show());
  await mainWindow.loadURL(frontendEntryUrl());

  if (isDev) {
    mainWindow.webContents.openDevTools({ mode: 'detach' });
  }
}

app.whenReady().then(async () => {
  app.setAppUserModelId('com.spad.detector.simulator');
  createMenu();
  await startBackend();
  await createWindow();
});

app.on('window-all-closed', () => {
  app.quit();
});

app.on('before-quit', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
});
