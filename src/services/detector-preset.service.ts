import { am0SolarIrradianceWM2Nm, pf32PdpFraction } from './spectral-response.service';

export interface IDetectorPreset {
  id: string;
  label: string;
  roi: { width: number; height: number };
  pixelPitchUm: number;
  fillFactor: number;
  microlensGain: number;
  detectorFovUrad: number;
  receiverEfficiency: number;
  referenceWavelengthNm: number;
  filterBandwidthNm: number;
  darkCountRateCps: number;
  deadTimeNs: number;
  timingJitterNs: number;
  tdcBinWidthNs: number;
  irfFwhmPs: number;
  maxCountRateCpsPerPixel: number;
  maxCountPerFrame: number;
  pdeNonuniformSigma: number;
  hotPixelFraction: number;
  hotPixelScale: number;
  darkCountSigma: number;
  assumptions: string[];
}

export interface IResolvedDetectorSettings {
  preset: IDetectorPreset;
  resolution: { width: number; height: number };
  detectorFovDeg: number;
  quantumEfficiency: number;
  solarIrradiance: number;
  systemEfficiency: number;
  filterBandwidthNm: number;
  darkCountRateCps: number;
  deadTimeNs: number;
  timingJitterNs: number;
  irfFwhmPs: number;
  maxCountRateCpsPerPixel: number;
  timeResolutionPs: number;
  tdcMaxCount: number;
}

export const DETECTOR_PRESETS: IDetectorPreset[] = [
  {
    id: 'pf32',
    label: 'PF32',
    roi: { width: 32, height: 32 },
    pixelPitchUm: 50,
    fillFactor: 0.015,
    microlensGain: 13.3,
    detectorFovUrad: 50 * Math.PI / 180 * 1e6,
    receiverEfficiency: 0.48,
    referenceWavelengthNm: 550,
    filterBandwidthNm: 50,
    darkCountRateCps: 100,
    deadTimeNs: 20,
    timingJitterNs: 0.2 / 2.355,
    tdcBinWidthNs: 0.055,
    irfFwhmPs: 200,
    maxCountRateCpsPerPixel: 20e6,
    maxCountPerFrame: 65535,
    pdeNonuniformSigma: 0.05,
    hotPixelFraction: 0.01,
    hotPixelScale: 5,
    darkCountSigma: 0.10,
    assumptions: [
      '32x32 silicon SPAD array based on PF32 public datasheet figures.',
      'Fill factor and microlens gain are engineering approximations for imaging studies.',
      'Detector FOV, receiver efficiency, wavelength, and filter bandwidth are optical-system engineering defaults.',
    ],
  },
];

export function getDetectorPreset(id: string): IDetectorPreset {
  return DETECTOR_PRESETS.find(preset => preset.id === id) ?? DETECTOR_PRESETS[0];
}

export function uradToDeg(urad: number): number {
  return urad * 1e-6 * 180 / Math.PI;
}

export function tdcMaxCountFromBits(bitDepth: number): number {
  return (2 ** Math.max(1, Math.floor(bitDepth))) - 1;
}

export function resolveDetectorSettings(
  presetId: string,
  wavelengthNm: number,
  filterBandwidthNm?: number,
  tdcBitDepth = 16,
): IResolvedDetectorSettings {
  const preset = getDetectorPreset(presetId);
  const bandwidth = filterBandwidthNm ?? preset.filterBandwidthNm;
  return {
    preset,
    resolution: { ...preset.roi },
    detectorFovDeg: uradToDeg(preset.detectorFovUrad),
    quantumEfficiency: pf32PdpFraction(wavelengthNm),
    solarIrradiance: am0SolarIrradianceWM2Nm(wavelengthNm),
    systemEfficiency: preset.receiverEfficiency,
    filterBandwidthNm: bandwidth,
    darkCountRateCps: preset.darkCountRateCps,
    deadTimeNs: preset.deadTimeNs,
    timingJitterNs: preset.timingJitterNs,
    irfFwhmPs: preset.irfFwhmPs,
    maxCountRateCpsPerPixel: preset.maxCountRateCpsPerPixel,
    timeResolutionPs: preset.tdcBinWidthNs * 1000,
    tdcMaxCount: tdcMaxCountFromBits(tdcBitDepth),
  };
}
