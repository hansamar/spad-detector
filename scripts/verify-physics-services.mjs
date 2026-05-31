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

const reflectance = loadTsModule('src/services/reflectance.service.ts', [
  'phaseAngleRad',
  'monostaticLambertFactor',
  'monostaticSurfaceReturnFactor',
  'normalizeVec3',
]);

const sceneBackground = loadTsModule('src/services/scene-background.service.ts', [
  'backgroundTemporalDriftFactor',
  'backgroundSpatialFactor',
  'makeBackgroundSpatialMap',
]);

const budget = loadTsModule('src/services/simulation-budget.service.ts', [
  'estimateSimulationBudget',
]);

const environmentPresets = loadTsModule('src/models/environment-presets.model.ts', [
  'ENVIRONMENT_PRESETS',
  'findEnvironmentPreset',
]);

const backendSimulation = loadTsModule('src/services/backend-simulation.service.ts', [
  'encodeCountsCubeFromDataset',
  'diagnosticFrequencyBand',
  'backendEnvironmentScales',
  'expectedSignalMapFromSummary',
], {
  ...spectral,
  Injectable: () => (target) => target,
  inject: () => ({}),
  firstValueFrom: (value) => value,
  btoa: (value) => Buffer.from(value, 'binary').toString('base64'),
});

const photonSampling = loadTsModule('src/services/photon-sampling.service.ts', [
  'samplePoissonCount',
  'firstPhotonTofUnits',
]);

const physics = loadTsModule('src/services/physics.service.ts', [
  'atmosphericAttenuationCoefficientKm',
  'hazeScatterScaleFromVisibility',
], {
  ...spectral,
  Injectable: () => (target) => target,
});

assert.equal(Number(spectral.pf32PdpFraction(850).toFixed(3)), 0.049);
assert.equal(Number(spectral.pf32PdpFraction(550).toFixed(3)), 0.274);
assert.equal(Number(spectral.am0SolarIrradianceWM2Nm(550).toFixed(2)), 1.86);
assert.equal(Number(spectral.pf32PdpFraction(200).toFixed(3)), 0.054);
assert.equal(Number(spectral.pf32PdpFraction(1200).toFixed(3)), 0.002);
assert.ok(spectral.relativeChannelResponse(850, 10) < spectral.relativeChannelResponse(550, 10));
assert.ok(spectral.spectralBackgroundScale('scene_stray', 850, 10) > 0);
assert.ok(spectral.applyDeadTimeRate(1_000_000, 20e-9) < 1_000_000);

assert.equal(photonSampling.samplePoissonCount(0), 0);
{
  const seq = [0.85, 0.85, 0.85, 0.85];
  const count = photonSampling.samplePoissonCount(0.5, () => seq.shift() ?? 0.85);
  assert.equal(count, 3);
}
assert.equal(photonSampling.samplePoissonCount(64, Math.random, () => 0), 64);
assert.equal(photonSampling.firstPhotonTofUnits(3, 120.6, 1, () => 0.5), 121);
assert.equal(photonSampling.firstPhotonTofUnits(0, 120.4, 1), null);

assert.ok(physics.atmosphericAttenuationCoefficientKm(780, 5) > physics.atmosphericAttenuationCoefficientKm(780, 23));
assert.ok(physics.atmosphericAttenuationCoefficientKm(450, 23) > physics.atmosphericAttenuationCoefficientKm(1064, 23));
assert.ok(physics.hazeScatterScaleFromVisibility(5) > physics.hazeScatterScaleFromVisibility(23));

const los = reflectance.normalizeVec3({ x: 0, y: 0, z: 1 });
assert.equal(Number(reflectance.phaseAngleRad(los, los).toFixed(6)), 0);
assert.equal(Number(reflectance.phaseAngleRad(los, { x: 0, y: 0, z: -1 }).toFixed(6)), Number(Math.PI.toFixed(6)));
assert.equal(Number(reflectance.monostaticLambertFactor({ x: 0, y: 0, z: 1 }, los).toFixed(3)), 1.000);
assert.equal(Number(reflectance.monostaticLambertFactor({ x: Math.sqrt(3) / 2, y: 0, z: 0.5 }, los).toFixed(3)), 0.250);
assert.equal(Number(reflectance.monostaticLambertFactor({ x: 0, y: 0, z: -1 }, los).toFixed(3)), 0.000);
assert.ok(
  reflectance.monostaticSurfaceReturnFactor({ x: 0, y: 0, z: 1 }, los, { specularGain: 6, specularWidthDeg: 5 })
  > reflectance.monostaticSurfaceReturnFactor({ x: Math.sqrt(3) / 2, y: 0, z: 0.5 }, los, { specularGain: 6, specularWidthDeg: 5 }),
);

assert.equal(Number(sceneBackground.backgroundTemporalDriftFactor(0, 0, 2).toFixed(3)), 1.000);
assert.ok(sceneBackground.backgroundTemporalDriftFactor(0, 0.2, 1) > sceneBackground.backgroundTemporalDriftFactor(0.5, 0.2, 1));
assert.notEqual(
  Number(sceneBackground.backgroundSpatialFactor(0, 0, 4, 4, 0.1, 0.2, -0.1).toFixed(4)),
  Number(sceneBackground.backgroundSpatialFactor(3, 3, 4, 4, 0.1, 0.2, -0.1).toFixed(4)),
);
const bgMap = sceneBackground.makeBackgroundSpatialMap(4, 4, 0.02, 0.1, -0.05);
const bgMean = bgMap.flat().reduce((sum, value) => sum + value, 0) / bgMap.flat().length;
assert.equal(Number(bgMean.toFixed(3)), 1.000);

const lightBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 5000);
assert.equal(lightBudget.level, 'safe');
assert.ok(lightBudget.datasetMb < 20);
const heavyBudget = budget.estimateSimulationBudget({ width: 128, height: 128 }, 500000);
assert.equal(heavyBudget.level, 'blocked');
assert.ok(heavyBudget.datasetMb > 1000);
const hundredKBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 100000);
assert.equal(hundredKBudget.totalSamples, 102_400_000);
assert.equal(hundredKBudget.level, 'caution');
const twoHundredKBudget = budget.estimateSimulationBudget({ width: 32, height: 32 }, 200000);
assert.equal(twoHundredKBudget.totalSamples, 204_800_000);
assert.equal(twoHundredKBudget.level, 'caution');
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
assert.equal(environmentPresets.findEnvironmentPreset('lab_dim').solarScale, 0.00005);

for (const relativePath of [
  'src/models/simulation-params.model.ts',
  'src/components/simulation-view/simulation-view.component.ts',
  'src/components/simulation-view/simulation-view.component.html',
  'src/services/simulation.service.ts',
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
const localSimulationSource = fs.readFileSync(path.join(root, 'src/services/simulation.service.ts'), 'utf8');
assert.equal(localSimulationSource.includes('transmitterDivergenceMrad * 1e-3 / 2'), true);

console.log('physics service checks passed');
