# SPAD Detector

SPAD Detector is an integrated single-photon active-imaging simulation platform for UAV scenes, PF32 SPAD array data generation, backend photon simulation, and professional desktop 3D visualization.

## Project Layout

- `src/`: Angular + Three.js frontend for professional UI, 3D scene rendering, UAV interaction, and result visualization.
- `backend/`: FastAPI service for capability checks, simulation jobs, summaries, and `.bin` downloads.
- `sim/`: Physics simulation core, including PF32 preset data, spectral response, background, imaging, geometry, atmospheric attenuation, and active SPAD count generation.
- `scripts/`: Frontend/backend smoke tests and physics consistency checks.
- `output/`: Local screenshots and backend job artifacts.

## Common Commands

```powershell
npm install
python -m pip install -r requirements.txt
npm run dev
npm run backend
npm run desktop
npm run desktop:dev
npm run desktop:pack
npm run desktop:dist
npm run verify:physics
npm run verify:backend
npm run verify:startup
npx tsc --noEmit --pretty false
npm run build
```

`npm run backend` probes available Python environments and starts the backend with
a CUDA-capable `torch` environment when available. By default it requires CUDA;
set `SPAD_REQUIRE_CUDA=0` only when an explicit CPU fallback is needed.

The backend listens on `http://127.0.0.1:8000`. The frontend connects to that backend under `/api`.

## Desktop Mode

The project includes an Electron desktop shell for professional local simulation use.

- `npm run desktop:dev`: starts the Angular dev server, launches the desktop window, and lets Electron start the FastAPI backend.
- `npm run desktop`: builds the Angular frontend and opens it as a local desktop app.
- `npm run desktop:pack`: creates an unpacked desktop app under `release/`.
- `npm run desktop:dist`: creates Windows installer/portable artifacts under `release/`.

The desktop shell starts the backend with `SPAD_PYTHON_EXE` when that environment variable is set. Otherwise it tries the local Conda environment `~\.conda\envs\spad-detector\python.exe`, then falls back to `python` / `py`.

## Verification

Run the following checks before packaging:

```powershell
npm run verify:backend
npm run verify:physics
npx tsc --noEmit --pretty false
npm run build
npm run verify:startup
```

`npm run verify:startup` confirms that the selected Python environment exposes a CUDA-enabled `torch` runtime.

## Release Artifacts

Generated frontend output, backend job artifacts, local caches, and Electron installers are intentionally excluded from the source repository. Run `npm run desktop:dist` to rebuild the Windows installer and portable executable under `release/`.
