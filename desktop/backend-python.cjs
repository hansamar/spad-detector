const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

function pythonCandidates(appRoot = path.resolve(__dirname, '..')) {
  const bundled = path.join(appRoot, 'python', 'python.exe');
  const userConda = path.join(process.env.USERPROFILE || '', '.conda', 'envs', 'spad-detector', 'python.exe');
  const condaPrefix = process.env.CONDA_PREFIX
    ? path.join(process.env.CONDA_PREFIX, 'python.exe')
    : null;

  return [
    process.env.SPAD_PYTHON_EXE,
    bundled,
    userConda,
    condaPrefix,
    'python',
    'py',
  ].filter(Boolean);
}

function candidateCommand(candidate) {
  if (candidate === 'py') {
    return { command: 'py', argsPrefix: ['-3'], label: 'py -3' };
  }
  return { command: candidate, argsPrefix: [], label: candidate };
}

function isRunnableCandidate(candidate) {
  return candidate === 'python' || candidate === 'py' || fs.existsSync(candidate);
}

function probePython(candidate) {
  const commandInfo = candidateCommand(candidate);
  const probeCode = [
    'import json, sys',
    'payload = {"python": sys.executable, "torch_available": False, "cuda_available": False, "torch_version": None, "torch_cuda_version": None, "gpu_name": None}',
    'try:',
    '    import torch',
    '    payload["torch_available"] = True',
    '    payload["torch_version"] = str(torch.__version__)',
    '    payload["torch_cuda_version"] = str(torch.version.cuda) if torch.version.cuda else None',
    '    payload["cuda_available"] = bool(torch.cuda.is_available())',
    '    if payload["cuda_available"]:',
    '        payload["gpu_name"] = torch.cuda.get_device_name(0)',
    'except Exception as exc:',
    '    payload["error"] = str(exc)',
    'print(json.dumps(payload, ensure_ascii=False))',
  ].join('\n');

  const result = spawnSync(
    commandInfo.command,
    [...commandInfo.argsPrefix, '-c', probeCode],
    { encoding: 'utf8', timeout: 10000 },
  );

  if (result.error || result.status !== 0) {
    return {
      candidate,
      runnable: false,
      cuda_available: false,
      error: result.error?.message || result.stderr || `exit ${result.status}`,
    };
  }

  try {
    const parsed = JSON.parse(result.stdout.trim());
    return {
      candidate,
      command: commandInfo.command,
      argsPrefix: commandInfo.argsPrefix,
      runnable: true,
      selected_python: parsed.python,
      torch_available: Boolean(parsed.torch_available),
      cuda_available: Boolean(parsed.cuda_available),
      torch_version: parsed.torch_version,
      torch_cuda_version: parsed.torch_cuda_version,
      gpu_name: parsed.gpu_name,
      error: parsed.error,
    };
  } catch (error) {
    return {
      candidate,
      runnable: false,
      cuda_available: false,
      error: `invalid probe output: ${error.message}`,
    };
  }
}

function selectBackendPython(options = {}) {
  const appRoot = options.appRoot || path.resolve(__dirname, '..');
  const candidates = pythonCandidates(appRoot).filter(isRunnableCandidate);
  const probes = candidates.map(probePython);
  const cudaProbe = probes.find((probe) => probe.runnable && probe.cuda_available);
  const fallbackProbe = probes.find((probe) => probe.runnable);
  const selected = cudaProbe || fallbackProbe || null;

  return {
    selected,
    probes,
  };
}

module.exports = {
  candidateCommand,
  probePython,
  pythonCandidates,
  selectBackendPython,
};
