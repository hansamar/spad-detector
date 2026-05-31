# Physics Model Audit

Audit date: 2026-05-31

## Scope

This platform simulates nearby ball, propeller, and UAV observations with a PF32 SPAD array. The default signal budget is:

```text
detected signal = laser-reflection signal + solar-reflection signal
detected noise  = solar-driven scene stray photons + detector dark counts
```

The transmitter and receiver use separate angular parameters. `transmitter_divergence_mrad` controls laser beam spreading. `detector_fov_urad` controls image projection and focal-plane clipping.

## Externally Anchored PF32 Values

The `pf32` preset uses the current Photon Force PF32-Core specification sheet as its hardware reference.

| Quantity | PF32 preset | External source |
| --- | ---: | --- |
| Array layout | `32 x 32` pixels | PF32-Core specification sheet |
| Pixel pitch | `50 um` | PF32-Core specification sheet |
| Optical fill factor | `1.5%` | PF32-Core specification sheet |
| Effective fill factor with MLA | up to `20%` | PF32-Core specification sheet |
| Dark count rate | `<100 cps` for more than 80% of pixels | PF32-Core specification sheet |
| Temporal bin | `55 ps` | PF32-Core specification sheet |
| IRF | approximately `200 ps FWHM` | PF32-Core specification sheet |
| Maximum count rate | `20 Mcps` per pixel | PF32-Core specification sheet |

The code uses a digitized wavelength-dependent PDP curve stored in `sim/data/pf32_pdp_digitized.csv`. The graph digitization is suitable for trade studies. Publication-grade absolute predictions should retain the original digitization record and verify the curve against the camera used for the experiment.

The simulated setup also contains optical-system engineering defaults: receiver efficiency, filter bandwidth, receiver FOV, wavelength, microlens gain, and transmitter divergence. These values describe the optical setup and are not PF32 camera specifications.

## Signal Model

### Laser Reflection

The laser path uses a centered Gaussian-beam engineering model:

```text
beam radius = range * tan(full-angle transmitter divergence / 2)
intercepted fraction = min(1, 2 * target area / beam area)
received power = transmitted average power
               * intercepted fraction
               * reflectivity
               * receiver aperture area / (pi * range^2)
```

Pulsed mode derives average power from pulse energy multiplied by repetition frequency. CW mode uses the configured average power. The intercepted fraction is capped at one so the target cannot reflect more optical power than the transmitter emits.

The current implementation assumes an on-axis target, Lambertian return, and a centered beam footprint. In attitude-driven mode, laser incidence and observation both use the transmitter-receiver line of sight, independently from the solar direction. Beam wander, laser pointing error, measured beam quality, measured optical throughput, target BRDF, partial overlap, and obscuration require experiment-specific calibration.

### Solar Reflection

The solar-reflection path uses spectral irradiance, filter bandwidth, target area, reflectivity, aperture area, range, receiver efficiency, PF32 PDP, and a lightweight phase-angle factor. It is computed independently from the laser path.

The bundled solar table is a sparse AM0 trend approximation anchored to ASTM E490. ASTM E490 defines zero-air-mass solar spectral irradiance for spacecraft-related applications. Nearby terrestrial observations should use site measurements or an appropriate terrestrial reference such as ASTM G173, then record the irradiance and spectral assumptions used for each experiment.

### Scene Stray Photons

Scene stray photons are detector-side counts per second per pixel. The frontend starts from an empirical `350 cps/pixel` reference and scales it with current solar irradiance, wavelength response, filter bandwidth, and a visibility-based haze factor. The backend applies spatial nonuniformity and temporal drift.

This term represents nearby-scene ambient photons. It intentionally contains no deep-space, Earth-albedo, or zodiacal-light parameters.

The `350 cps/pixel` reference is an engineering default. Calibrated studies should measure background counts with the target signal blocked under the relevant illumination, filter, aperture, FOV, and atmospheric conditions.

## Detector And Statistics

The detector path applies PF32 PDP, synthetic pixel nonuniformity, synthetic hot pixels, dark-count variation, a nonparalyzable dead-time rate approximation, optional afterpulsing, optional crosstalk, Poisson sampling, saturation clipping, timing jitter, and TDC quantization.

Frame counts are authoritative. Event-mode timestamps are synthesized from the sampled frame counts so the event list and frame cube remain consistent. This event output supports algorithm development. Full TCSPC transport studies should model pulse-indexed arrival times, per-event dead time, pile-up, and measured IRF directly.

Reported SNR uses the aggregate Poisson shot-noise approximation:

```text
SNR = S / sqrt(S + B + D)
```

The backend reports this value in decibels as `20 * log10(SNR)`.

## Geometry And Atmosphere

Receiver FOV clipping uses radial off-axis distance. Projected silhouettes drive focal-plane footprints for balls, blades, and UAVs. Their integrated photon-rate modulation still uses lightweight analytic terms, so projected image shape is stronger than absolute photometric fidelity.

Atmospheric transmission uses a visibility-based Kruse/Koschmieder-style extinction approximation with a molecular floor. The dense-atmosphere segment is capped at `6000 m` for long paths. Nearby experiments should replace this with measured transmission or a site-specific model when atmospheric loss materially affects conclusions.

## Correctness Fixes From This Audit

- Event timestamps are generated from final sampled counts, removing a second independent Poisson draw.
- Event timestamps remain inside their source frame after jitter and TDC quantization.
- Recorded-trajectory FOV tests use the same radial detector aperture as the simulator.
- Background spatial maps remain positive after gradients and nonuniformity.
- Explicit CUDA requests fail clearly when CUDA sampling is unavailable.
- PF32 timing defaults use `55 ps` TDC bins and approximately `200 ps FWHM` IRF.
- SNR now uses a photon shot-noise denominator.
- Laser reflection, solar reflection, and solar-driven scene stray photons use independent parameter paths.
- Attitude-driven laser reflection uses the transmitter-receiver line of sight instead of the solar direction.
- The transmitter beam divergence is independent from receiver FOV.
- Near-range defaults use a `2 m` target distance, `1 uW` average laser power, and a `5e-5` indoor solar scale so the initial weak-light bench scene starts outside detector saturation.

## Remaining Validation Work

Use measured data before treating absolute photon counts as calibrated predictions:

1. Capture dark frames across integration time and temperature to fit per-pixel DCR, hot-pixel fraction, and temporal stability.
2. Capture target-blocked background frames across solar irradiance, wavelength, filter bandwidth, aperture, and receiver FOV to replace the empirical ambient reference.
3. Measure a Lambertian reference panel across distance, laser power, divergence, and incidence angle to fit transmitter throughput and receiver efficiency.
4. Measure synchronized laser returns to fit IRF, timing jitter, pile-up, afterpulsing, and event dead-time behavior.
5. Measure tennis-ball, blade, and UAV surfaces across orientation to replace scalar reflectivity and analytic modulation with target-specific BRDF or empirical response tables.
6. Compare simulated and measured count histograms, per-pixel maps, range sweeps, and saturation curves with fixed random seeds and recorded software revisions.

## External References

- [Photon Force PF32 camera range](https://www.photon-force.com/products/pf32-camera-range/)
- [Photon Force PF32-Core specification sheet](https://www.photon-force.com/download/PF32Core_SpecSheet.pdf)
- [ASTM E490-22: Standard Solar Constant and Zero Air Mass Solar Spectral Irradiance Tables](https://store.astm.org/e0490-22.html)
- [ASTM G173-23: Reference Spectra for Terrestrial Direct Normal and Hemispherical Solar Irradiance](https://store.astm.org/g0173-23.html)
- [Cova et al., Avalanche photodiodes and quenching circuits for single-photon detection](https://doi.org/10.1364/AO.35.001956)
- [Degnan, Photon-counting multikilohertz microlaser altimeters for airborne and spaceborne topographic measurements](https://doi.org/10.1029/2001JD001085)
