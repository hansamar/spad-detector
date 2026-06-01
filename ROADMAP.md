# Roadmap

This roadmap describes the current direction of SPAD Detector. It is
intentionally focused and may evolve as validation evidence and contributor
feedback grow.

## Physics Validation

- Add reference scenarios for signal, scene-stray, and dark-count budgets.
- Expand regression checks for detector presets and custom detector settings.
- Compare selected simulated observables with documented hardware figures and
  clearly label engineering approximations.

## Reproducibility and Data

- Document the exported `.bin` count-cube layout with Python and MATLAB loading
  examples.
- Add small example presets for ball, blade, and UAV workflows.
- Record reproducibility metadata such as random seed, backend, dependency
  versions, and commit SHA in exported experiment summaries.

## Platform Maintenance

- Keep the portable CPU verification workflow running on GitHub Actions.
- Add repeatable desktop release checks for packaged Windows artifacts.
- Improve contributor templates as the issue and pull request workflow grows.

## Research Distribution

- Publish versioned releases with changelog entries.
- Evaluate archiving stable releases in a research repository that can issue a
  DOI.
- Add citation metadata after the archival workflow is selected.
