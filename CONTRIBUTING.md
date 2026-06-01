# Contributing

Thank you for helping improve SPAD Detector.

## Before Opening an Issue

- Search existing issues before creating a new one.
- Include the operating system, Node.js version, Python version, and selected
  compute backend when reporting a bug.
- Include the simulation parameters and the smallest reproducible workflow.
- For modeling questions, describe the expected physical behavior and provide
  a reference when possible.
- Do not post sensitive data, private datasets, tokens, or credentials.

## Pull Requests

1. Keep each pull request focused on one change.
2. Explain the motivation, user-visible impact, and validation performed.
3. Update the README or model audit when assumptions, parameters, outputs, or
   data formats change.
4. Run the portable verification set:

```powershell
npm ci
python -m pip install -r requirements.txt
npm run verify:backend
npm run verify:physics
npx tsc --noEmit --pretty false
python -m compileall -q backend sim scripts
npm run build
```

5. When changing CUDA startup behavior, also run:

```powershell
npm run verify:startup
```

## Physics Model Changes

Changes to detector parameters, photon budgets, geometry, sampling, or
background terms should include:

- the physical assumption and source;
- the affected parameter range;
- a reproducible validation case;
- any compatibility impact on earlier results or exported artifacts;
- an update to `docs/physics-model-audit.md` when the documented model changes.

## Scope

This repository is a research-oriented simulation platform. Small, reviewable
changes that improve physical traceability, reproducibility, diagnostics, and
usability are preferred.
