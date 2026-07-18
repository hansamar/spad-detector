import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import ts from 'typescript';
import vm from 'node:vm';

const root = path.resolve(import.meta.dirname, '..');
const plainJson = (value) => JSON.parse(JSON.stringify(value));

function loadTsModule(relativePath, exportNames, context = {}) {
  const filePath = path.join(root, relativePath);
  const source = fs.readFileSync(filePath, 'utf8').replace(/^import .+;\r?\n/gm, '');
  const harness = `${source.replaceAll('export ', '')}
return { ${exportNames.join(', ')} };`;
  const js = ts.transpileModule(`(() => {\n${harness}\n})()`, {
    compilerOptions: { module: ts.ModuleKind.ES2022, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  return vm.runInNewContext(js, context, { filename: relativePath });
}

const spectral = loadTsModule('src/services/spectral-response.service.ts', [
  'pf32PdpFraction',
  'am0SolarIrradianceWM2Nm',
  'relativeChannelResponse',
  'spectralBackgroundScale',
  'applyDeadTimeRate',
]);

const budget = loadTsModule('src/services/simulation-budget.service.ts', [
  'estimateSimulationBudget',
]);

const physics = loadTsModule('src/services/physics.service.ts', [
  'atmosphericAttenuationCoefficientKm',
  'hazeScatterScaleFromVisibility',
  'photonEnergyJoule',
  'solarEnvironmentBackgroundRateCpsPerPixel',
], {
  ...spectral,
  Injectable: () => (target) => target,
});

const backendSimulation = loadTsModule('src/services/backend-simulation.service.ts', [
  'encodeCountsCubeFromDataset',
  'diagnosticFrequencyBand',
  'backendEnvironmentScales',
  'expectedSignalMapFromSummary',
  'backendErrorMessage',
  'hasLocalDatasetDownload',
  'BackendSimulationService',
], {
  ...spectral,
  ...physics,
  HttpClient: class {},
  Injectable: () => (target) => target,
  inject: () => ({ get: () => ({}), post: () => ({}) }),
  firstValueFrom: (value) => value,
  btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
  window: { spadDesktop: { backendBaseUrl: 'http://127.0.0.1:8000/api' } },
});

assert.equal(Number(spectral.pf32PdpFraction(850).toFixed(3)), 0.049);
assert.equal(Number(spectral.pf32PdpFraction(550).toFixed(3)), 0.274);
assert.equal(Number(spectral.am0SolarIrradianceWM2Nm(550).toFixed(2)), 1.86);
assert.equal(Number(spectral.pf32PdpFraction(200).toFixed(3)), 0.054);
assert.equal(Number(spectral.pf32PdpFraction(1200).toFixed(3)), 0.002);
assert.ok(spectral.relativeChannelResponse(850, 10) < spectral.relativeChannelResponse(550, 10));
assert.ok(spectral.spectralBackgroundScale('scene_stray', 850, 10) > 0);
assert.ok(spectral.applyDeadTimeRate(1_000_000, 20e-9) < 1_000_000);

assert.ok(physics.atmosphericAttenuationCoefficientKm(780, 5) > physics.atmosphericAttenuationCoefficientKm(780, 23));
assert.ok(physics.atmosphericAttenuationCoefficientKm(450, 23) > physics.atmosphericAttenuationCoefficientKm(1064, 23));
assert.ok(physics.hazeScatterScaleFromVisibility(5) > physics.hazeScatterScaleFromVisibility(23));
{
  const commonSolarBackground = {
    solarIrradiance: 1.35,
    atmosphericVisibilityKm: 23,
    laserWavelengthNm: 780,
    filterBandwidth: 10,
    apertureDiameter: 0.025,
    systemEfficiency: 0.05,
    quantumEfficiency: 0.07,
    detectorFov: 50,
    resolution: { width: 32, height: 32 },
  };
  const narrow = physics.solarEnvironmentBackgroundRateCpsPerPixel(commonSolarBackground);
  const wide = physics.solarEnvironmentBackgroundRateCpsPerPixel({ ...commonSolarBackground, filterBandwidth: 50 });
  const noSun = physics.solarEnvironmentBackgroundRateCpsPerPixel({ ...commonSolarBackground, solarIrradiance: 0 });
  assert.ok(narrow > 1e9);
  assert.ok(wide > narrow * 4.5);
  assert.equal(noSun, 0);
}

const lightBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 5000);
assert.equal(lightBudget.level, 'safe');
assert.ok(lightBudget.datasetMb < 20);
const heavyBudget = budget.estimateSimulationBudget({ width: 128, height: 128 }, 500000);
assert.equal(heavyBudget.level, 'blocked');
assert.ok(heavyBudget.datasetMb > 1000);
const hundredKBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 100000);
assert.equal(hundredKBudget.totalSamples, 102_400_000);
assert.equal(hundredKBudget.level, 'blocked');
const twoHundredKBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 200000);
assert.equal(twoHundredKBudget.totalSamples, 204_800_000);
assert.equal(twoHundredKBudget.level, 'blocked');
const overBackendFrameBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 200001);
assert.equal(overBackendFrameBudget.level, 'blocked');

{
  const empty = 99;
  const dataset = new Uint16Array([empty, 7, 8, empty, 3, empty, empty, 2]);
  const encoded = backendSimulation.encodeCountsCubeFromDataset(dataset, 2, 2, 2, empty);
  const [dtype, shape, payload] = encoded.split('|');
  assert.equal(dtype, 'uint16');
  assert.equal(shape, '(2,2,2)');
  const decoded = Buffer.from(payload, 'base64');
  const values = Array.from(new Uint16Array(decoded.buffer, decoded.byteOffset, decoded.byteLength / 2));
  assert.deepEqual(values, [0, 1, 1, 0, 1, 0, 0, 1]);
  const band = backendSimulation.diagnosticFrequencyBand({
    nFrames: 256,
    frameDurationUs: 20,
    rotationSpeed: 18000,
    pathRotationSpeeds1: [18000],
  });
  assert.ok(band.fmax >= 900);
  assert.ok(band.fmax < 25000);
  const clear = backendSimulation.backendEnvironmentScales({
    solarIrradiance: 1,
    atmosphericVisibilityKm: 23,
    atmosphericAttenuationEnabled: true,
    laserWavelengthNm: 850,
    filterBandwidth: 10,
  });
  const hazy = backendSimulation.backendEnvironmentScales({
    solarIrradiance: 1,
    atmosphericVisibilityKm: 5,
    atmosphericAttenuationEnabled: true,
    laserWavelengthNm: 850,
    filterBandwidth: 10,
  });
  assert.equal(clear.solar_irradiance, 1);
  assert.equal(clear.atmospheric_attenuation_enabled, true);
  assert.equal(clear.atmospheric_visibility_km, 23);
  assert.equal(hazy.atmospheric_visibility_km, 5);
  assert.ok(hazy.scene_stray_rate > clear.scene_stray_rate);
  assert.ok(clear.scene_stray_rate > 5);
  const dimScene = backendSimulation.backendEnvironmentScales({
    solarIrradiance: 0.001,
    atmosphericVisibilityKm: 50,
    atmosphericAttenuationEnabled: true,
    laserWavelengthNm: 850,
    filterBandwidth: 10,
  });
  assert.ok(dimScene.scene_stray_rate > 0);
  assert.ok(dimScene.scene_stray_rate < clear.scene_stray_rate);
  const noSunScene = backendSimulation.backendEnvironmentScales({
    solarIrradiance: 0,
    atmosphericVisibilityKm: 23,
    atmosphericAttenuationEnabled: true,
    laserWavelengthNm: 850,
    filterBandwidth: 10,
  });
  assert.equal(noSunScene.scene_stray_rate, 0);
  const manualSbrScene = backendSimulation.backendEnvironmentScales({
    solarIrradiance: 1,
    backgroundNoiseMode: 'manual_sbr',
    atmosphericVisibilityKm: 5,
    atmosphericAttenuationEnabled: true,
    laserWavelengthNm: 850,
    filterBandwidth: 10,
  });
  assert.equal(manualSbrScene.scene_stray_rate, 0);
  assert.equal(manualSbrScene.solar_environment_rate, 0);
  const backendService = new backendSimulation.BackendSimulationService();
  const manualPayload = backendService.toSummaryRequest({
    targetType: 'Ball',
    initialPos: { x: 0, y: 1.5, z: 4 },
    initialVel: { x: 0, y: 0, z: 0 },
    reflectivity: 0.1,
    restitution: 0.8,
    ballMotionType: 'Gravity',
    bladeMotionType: 'Fixed',
    waypoints: [],
    pathSpeeds: [],
    pathRotationSpeeds: [],
    rotationRadius: 0.5,
    rotationSpeed: 12000,
    rotationKeyframes: [],
    bladePitch: 90,
    rotationCenter: { x: 0, z: 0 },
    uploadedImage: null,
    droneScale: 1,
    resolution: { width: 32, height: 32 },
    detectorPresetId: 'pf32',
    detectorFov: 50,
    detectorYaw: 0,
    detectorPitch: 0,
    pixelPitchUm: 50,
    fillFactor: 0.015,
    microlensGain: 13.3,
    frameDurationUs: 20,
    quantumEfficiency: 0.3,
    apertureDiameter: 0.025,
    systemEfficiency: 0.05,
    filterBandwidth: 10,
    darkCountRate: 100,
    timeResolutionPs: 256,
    tdcMaxCount: 8191,
    solarIrradiance: 1.35,
    backgroundNoiseMode: 'manual_sbr',
    manualSignalBackgroundRatio: 7,
    atmosphericAttenuationEnabled: true,
    atmosphericVisibilityKm: 23,
    laserMode: 'CW',
    laserPulseEnergy: 1e-12,
    laserAveragePower: 1e-6,
    laserRepetitionFrequency: 1000000,
    laserPulseWidthNs: 1,
    laserWavelengthNm: 780,
    transmitterDivergenceMrad: 1,
    nFrames: 1000,
    cameraHeight: 1,
  });
  assert.equal(manualPayload.background_noise_mode, 'manual_sbr');
  assert.equal(manualPayload.manual_signal_background_ratio, 7);
  assert.equal(manualPayload.scene_stray_rate, 0);
  assert.equal(manualPayload.persist_artifacts, false);
  assert.equal(manualPayload.include_event_list, false);
  assert.equal(manualPayload.include_tdc_frame_cube, false);
  assert.equal(manualPayload.save_truth_series, false);
  const exportPayload = backendService.toSummaryRequest({
    targetType: 'Ball',
    initialPos: { x: 0, y: 1.5, z: 4 },
    initialVel: { x: 0, y: 0, z: 0 },
    reflectivity: 0.1,
    restitution: 0.8,
    ballMotionType: 'Gravity',
    bladeMotionType: 'Fixed',
    waypoints: [],
    pathSpeeds: [],
    pathRotationSpeeds: [],
    rotationRadius: 0.5,
    rotationSpeed: 12000,
    rotationKeyframes: [],
    bladePitch: 90,
    rotationCenter: { x: 0, z: 0 },
    uploadedImage: null,
    droneScale: 1,
    resolution: { width: 32, height: 32 },
    detectorPresetId: 'pf32',
    detectorFov: 50,
    detectorYaw: 0,
    detectorPitch: 0,
    pixelPitchUm: 50,
    fillFactor: 0.015,
    microlensGain: 13.3,
    frameDurationUs: 20,
    quantumEfficiency: 0.3,
    apertureDiameter: 0.025,
    systemEfficiency: 0.05,
    filterBandwidth: 10,
    darkCountRate: 100,
    timeResolutionPs: 256,
    tdcMaxCount: 8191,
    solarIrradiance: 1.35,
    backgroundNoiseMode: 'manual_sbr',
    manualSignalBackgroundRatio: 7,
    atmosphericAttenuationEnabled: true,
    atmosphericVisibilityKm: 23,
    laserMode: 'CW',
    laserPulseEnergy: 1e-12,
    laserAveragePower: 1e-6,
    laserRepetitionFrequency: 1000000,
    laserPulseWidthNs: 1,
    laserWavelengthNm: 780,
    transmitterDivergenceMrad: 1,
    nFrames: 1000,
    cameraHeight: 1,
  }, { persistArtifacts: true, includeEventList: true, includeTdcFrameCube: true });
  assert.equal(exportPayload.persist_artifacts, true);
  assert.equal(exportPayload.include_event_list, true);
  assert.equal(exportPayload.include_tdc_frame_cube, true);
  assert.equal(
    backendSimulation.backendErrorMessage({
      message: 'Http failure response for http://127.0.0.1:8000/api/simulate/jobs: 422 Unprocessable Entity',
      error: {
        detail: [
          { loc: ['body', 'manual_signal_background_ratio'], msg: 'Input should be greater than 0' },
          { loc: ['body'], msg: 'simulation frame count exceeds backend limit (200001 > 200000)' },
        ],
      },
    }),
    'manual_signal_background_ratio: Input should be greater than 0; request: simulation frame count exceeds backend limit (200001 > 200000)',
  );
  const summaryResult = backendService.summaryToSimulationResult({
    roi_h: 2,
    roi_w: 2,
    n_frames: 100000,
    sample_rate_hz: 50000,
    preview_counts: [
      [1, 2],
      [3, 4],
    ],
    total_noise_photons: 10,
    total_background_photons: 6,
    total_signal_photons: 7,
    expected_signal_map: [
      [0, 1],
      [2, 4],
    ],
    truth_freq_hz: 12,
    truth_row: 1,
    truth_col: 1,
  }, { tdcMaxCount: 8191 });
  assert.equal(summaryResult.dataset.length, 0);
  assert.equal(backendSimulation.hasLocalDatasetDownload(summaryResult), false);
  assert.equal(backendSimulation.hasLocalDatasetDownload({ dataset: new Uint16Array([1, 2]) }), true);
  assert.deepEqual(plainJson(summaryResult.photonCountMap), [
    [1, 2],
    [3, 4],
  ]);
  const variableFrequencyResult = backendService.summaryToSimulationResult({
    roi_h: 1,
    roi_w: 1,
    n_frames: 4,
    sample_rate_hz: 2,
    preview_counts: [[0]],
    total_noise_photons: 0,
    total_background_photons: 0,
    total_signal_photons: 0,
    expected_signal_map: [[0]],
    truth_freq_hz: 12,
    truth_frequency_series_hz: [10, 11, 12, 13],
    truth_propeller_frequency_series_hz: [
      [10, 20, 30, 40],
      [11, 21, 31, 41],
      [12, 22, 32, 42],
      [13, 23, 33, 43],
    ],
    truth_row: 0,
    truth_col: 0,
  }, { tdcMaxCount: 8191 });
  assert.deepEqual(variableFrequencyResult.groundTruthData.frequencies, [10, 11, 12, 13]);
  assert.deepEqual(variableFrequencyResult.groundTruthData.propellerFrequencies, [
    [10, 20, 30, 40],
    [11, 21, 31, 41],
    [12, 22, 32, 42],
    [13, 23, 33, 43],
  ]);
  const expectedMap = backendSimulation.expectedSignalMapFromSummary({
    roi_h: 2,
    roi_w: 3,
    truth_row: 1,
    truth_col: 2,
    total_signal_photons: 99,
    expected_signal_map: [
      [1.2, 0, 3],
      [0, 4.5, 6],
    ],
  });
  assert.deepEqual(plainJson(expectedMap), [
    [1.2, 0, 3],
    [0, 4.5, 6],
  ]);
  const fallbackMap = backendSimulation.expectedSignalMapFromSummary({
    roi_h: 2,
    roi_w: 3,
    truth_row: 1,
    truth_col: 2,
    total_signal_photons: 99,
    expected_signal_map: [],
  });
  assert.deepEqual(plainJson(fallbackMap), [
    [0, 0, 0],
    [0, 0, 99],
  ]);
}

const detector = loadTsModule('src/services/detector-preset.service.ts', [
  'DETECTOR_PRESETS',
  'getDetectorPreset',
  'resolveDetectorSettings',
  'uradToDeg',
], spectral);

const pf32 = detector.getDetectorPreset('pf32');
assert.equal(detector.DETECTOR_PRESETS.length, 1);
assert.equal(pf32.roi.width, 32);
assert.equal(pf32.roi.height, 32);
assert.equal(pf32.pixelPitchUm, 50);
assert.equal(Number(detector.uradToDeg(pf32.detectorFovUrad).toFixed(1)), 50.0);
assert.equal(pf32.irfFwhmPs, 200);
assert.equal(Number(pf32.timingJitterNs.toFixed(4)), Number((0.2 / 2.355).toFixed(4)));

const resolved = detector.resolveDetectorSettings('pf32', 850, 10);
assert.equal(Number(resolved.quantumEfficiency.toFixed(3)), 0.049);
assert.equal(Number(resolved.solarIrradiance.toFixed(2)), 1.16);
assert.equal(resolved.resolution.width, 32);
assert.equal(resolved.resolution.height, 32);
assert.equal(Number(resolved.detectorFovDeg.toFixed(1)), 50.0);

for (const relativePath of [
  'src/models/simulation-params.model.ts',
  'src/components/simulation-view/simulation-view.component.ts',
  'src/components/simulation-view/simulation-view.component.html',
  'backend/models.py',
  'sim/detector_presets.py',
]) {
  const source = fs.readFileSync(path.join(root, relativePath), 'utf8');
  assert.equal(source.includes('pf32_low_noise'), false, `${relativePath} still references pf32_low_noise`);
  assert.equal(source.includes('pf32_high_speed'), false, `${relativePath} still references pf32_high_speed`);
  assert.equal(source.includes('enableStochasticMotion'), false, `${relativePath} still references enableStochasticMotion`);
  assert.equal(source.includes('frequencyWalkSigma'), false, `${relativePath} still references frequencyWalkSigma`);
  assert.equal(source.includes('momentOfInertia'), false, `${relativePath} still references momentOfInertia`);
}

const detectorPanelTemplate = fs.readFileSync(path.join(root, 'src/components/simulation-view/simulation-view.component.html'), 'utf8');
assert.equal(detectorPanelTemplate.includes('detectorPresetCustom'), true);
assert.equal(detectorPanelTemplate.includes('value="custom"'), true);
for (const hiddenHardwareInput of [
  'pixelPitchInput',
  'fillFactorInput',
  'microlensGainInput',
  'deadTimeInput',
  'timingJitterInput',
  'irfFwhmInput',
]) {
  assert.equal(detectorPanelTemplate.includes(hiddenHardwareInput), false);
}
const backendModels = fs.readFileSync(path.join(root, 'backend/models.py'), 'utf8');
assert.equal(backendModels.includes('DetectorPreset = Literal["custom", "pf32"]'), true);
const backendSimulationSource = fs.readFileSync(path.join(root, 'src/services/backend-simulation.service.ts'), 'utf8');
assert.equal(backendSimulationSource.includes('activeLaserIrradianceWm2Nm'), false);
assert.equal(backendSimulationSource.includes("illumination_mode: 'laser_plus_solar'"), true);
assert.equal(backendSimulationSource.includes('transmitter_divergence_mrad: params.transmitterDivergenceMrad'), true);
const simulationViewSource = fs.readFileSync(path.join(root, 'src/components/simulation-view/simulation-view.component.ts'), 'utf8');
assert.equal(simulationViewSource.includes('Math.min(window.devicePixelRatio || 1, 2)'), true);

console.log('physics service checks passed');
