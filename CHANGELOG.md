# Changelog

All notable changes to SPAD Detector will be documented in this file.

## Unreleased

### Added

- Added a manual signal-background-ratio background mode that is mutually
  exclusive with solar-irradiance-driven natural background photons.
- Added a fast interactive backend-job mode that skips export artifacts during
  ordinary simulation runs.

### Changed

- Extended backend and frontend verification for solar-environment background
  photons and manual SBR background control.
- Changed the frontend Run Simulation path to request summary-only backend jobs
  by default, avoiding event-list, TDC-cube, count-cube, and bundle generation
  unless an export path explicitly requests them.
- Changed summary-result rendering to use backend `preview_counts` directly
  without allocating an empty per-frame frontend dataset.
- Aligned the recorded-trajectory request limit with the backend frame budget
  so 100,000-frame frontend runs are not rejected before simulation starts.
- Replaced the hot per-line blur loop with vectorized SciPy convolution to
  speed up large summary-only runs.
- Vectorized custom propeller-shape projection for moving blade targets,
  avoiding per-point camera projection during large trajectory simulations.
- Restored blade pitch as an active backend return-strength parameter for
  custom propeller and blade-shape simulations.
- Changed summary-only downloads to launch a real backend export job instead
  of writing an empty local `.bin` from the preview dataset.
- Changed manual SBR background generation to distribute noise from an
  observation-level signal budget, preserving background photons when the
  target leaves the detector FOV and preventing target mechanical-frequency
  leakage into background noise.

## [0.1.1] - 2026-06-01

### Added

- MIT open-source license.
- Contribution guide for issue reports, pull requests, and physics-model
  changes.
- Public roadmap for validation, reproducibility, maintenance, and research
  distribution.
- Security policy for responsible reporting.

### Changed

- Updated the README with open-source project links, citation guidance, and
  license information.
- Patched Angular and Vite dependency lines to remove known high-severity
  advisories present in the initial lock file.

## [0.1.0] - 2026-05-30

### Added

- Initial public release of the SPAD Detector research simulation platform.
- Angular and Three.js user interface for ball, blade, and UAV workflows.
- FastAPI and Python simulation backend with CPU and optional CUDA sampling.
- PF32 detector preset, physics-model audit, reproducibility checks, desktop
  shell, and portable GitHub Actions verification.
