const { spawn } = require('node:child_process');
const path = require('node:path');
const { selectBackendPython } = require('../desktop/backend-python.cjs');

const appRoot = path.resolve(__dirname, '..');
const backendHost = process.env.SPAD_BACKEND_HOST || '127.0.0.1';
const backendPort = process.env.SPAD_BACKEND_PORT || '8000';
const requireCuda = process.env.SPAD_REQUIRE_CUDA !== '0';
const selection = selectBackendPython({ appRoot });
const selected = selection.selected;

if (process.argv.includes('--probe')) {
  console.log(JSON.stringify({
    selected_python: selected?.selected_python || null,
    cuda_available: Boolean(selected?.cuda_available),
    torch_available: Boolean(selected?.torch_available),
    torch_version: selected?.torch_version || null,
    torch_cuda_version: selected?.torch_cuda_version || null,
    gpu_name: selected?.gpu_name || null,
    probes: selection.probes,
  }, null, 2));
  process.exit(selected ? 0 : 1);
}

if (!selected) {
  console.error('[backend] No runnable Python executable was found.');
  process.exit(1);
}

if (requireCuda && !selected.cuda_available) {
  console.error('[backend] CUDA is required at startup, but no CUDA-capable Python environment was found.');
  console.error('[backend] Set SPAD_PYTHON_EXE to a Python with torch+CUDA, or set SPAD_REQUIRE_CUDA=0 to allow CPU fallback.');
  console.error(JSON.stringify(selection.probes, null, 2));
  process.exit(1);
}

const cudaLabel = selected.cuda_available
  ? `CUDA enabled: ${selected.gpu_name || 'GPU detected'}, torch ${selected.torch_version || 'unknown'}`
  : 'CPU fallback';
console.log(`[backend] Python: ${selected.selected_python}`);
console.log(`[backend] ${cudaLabel}`);

const child = spawn(
  selected.command,
  [
    ...selected.argsPrefix,
    '-m',
    'uvicorn',
    'backend.main:app',
    '--host',
    backendHost,
    '--port',
    String(backendPort),
  ],
  {
    cwd: appRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      SPAD_BACKEND_PORT: String(backendPort),
      SPAD_SELECTED_PYTHON: selected.selected_python,
    },
    stdio: 'inherit',
  },
);

child.on('exit', (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(code ?? 0);
});
