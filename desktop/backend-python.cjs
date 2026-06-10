const fs = require('node:fs');
const path = require('node:path');
const { spawn } = require('node:child_process');

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

async function probePython(candidate) {
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

  return new Promise((resolve) => {
    const child = spawn(
      commandInfo.command,
      [...commandInfo.argsPrefix, '-c', probeCode],
      { windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'] },
    );

    const chunks = [];
    const errChunks = [];
    child.stdout.on('data', (data) => chunks.push(data));
    child.stderr.on('data', (data) => errChunks.push(data));

    const timeout = setTimeout(() => {
      child.kill();
      resolve({
        candidate,
        runnable: false,
        cuda_available: false,
        error: 'probe timed out after 10s',
      });
    }, 10000);

    child.on('close', (code) => {
      clearTimeout(timeout);
      const stdout = Buffer.concat(chunks).toString('utf8').trim();
      const stderr = Buffer.concat(errChunks).toString('utf8').trim();

      if (code !== 0) {
        resolve({
          candidate,
          runnable: false,
          cuda_available: false,
          error: stderr || `exit ${code}`,
        });
        return;
      }

      try {
        const parsed = JSON.parse(stdout);
        resolve({
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
        });
      } catch (error) {
        resolve({
          candidate,
          runnable: false,
          cuda_available: false,
          error: `invalid probe output: ${error.message}`,
        });
      }
    });

    child.on('error', (err) => {
      clearTimeout(timeout);
      resolve({
        candidate,
        runnable: false,
        cuda_available: false,
        error: err.message,
      });
    });
  });
}

async function selectBackendPython(options = {}) {
  const appRoot = options.appRoot || path.resolve(__dirname, '..');
  const candidates = pythonCandidates(appRoot).filter(isRunnableCandidate);
  const probes = [];
  let fallbackProbe = null;
  for (const candidate of candidates) {
    const probe = await probePython(candidate);
    probes.push(probe);
    if (probe.runnable && fallbackProbe === null) {
      fallbackProbe = probe;
    }
    if (probe.runnable && probe.cuda_available) {
      return {
        selected: probe,
        probes,
      };
    }
  }

  return {
    selected: fallbackProbe,
    probes,
  };
}

module.exports = {
  candidateCommand,
  probePython,
  pythonCandidates,
  selectBackendPython,
};
