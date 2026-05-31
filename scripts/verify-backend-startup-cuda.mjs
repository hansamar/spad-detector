import { execFileSync } from 'node:child_process';
import path from 'node:path';

const output = execFileSync(
  process.execPath,
  ['scripts/start-backend.cjs', '--probe'],
  {
    cwd: path.resolve(import.meta.dirname, '..'),
    encoding: 'utf8',
    timeout: 60000,
  },
);

const probe = JSON.parse(output);
if (!probe.selected_python) {
  throw new Error('backend startup did not select a Python executable');
}
if (!probe.cuda_available) {
  throw new Error(`backend startup did not select a CUDA-capable Python: ${JSON.stringify(probe)}`);
}
if (!String(probe.selected_python).toLowerCase().includes('spad-detector')) {
  throw new Error(`backend startup did not prefer the spad-detector environment: ${probe.selected_python}`);
}

console.log('backend startup CUDA probe passed');
