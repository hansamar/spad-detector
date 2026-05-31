const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('spadDesktop', {
  platform: process.platform,
  backendBaseUrl: `http://127.0.0.1:${process.env.SPAD_BACKEND_PORT || 8000}/api`,
  versions: {
    electron: process.versions.electron,
    chrome: process.versions.chrome,
  },
  onBackendExit(callback) {
    ipcRenderer.on('spad-backend-exit', (_event, payload) => callback(payload));
  },
});
