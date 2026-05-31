import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';
import { ISimulationParams, ISimulationResult } from '../models/simulation-params.model';
import { am0SolarIrradianceWM2Nm, relativeChannelResponse, spectralBackgroundScale } from './spectral-response.service';

export function encodeCountsCubeFromDataset(
  dataset: Uint16Array,
  nFrames: number,
  height: number,
  width: number,
  emptyPixelValue: number,
): string {
  const counts = new Uint16Array(nFrames * height * width);
  for (let i = 0; i < counts.length && i < dataset.length; i++) {
    counts[i] = dataset[i] === emptyPixelValue ? 0 : 1;
  }
  const bytes = new Uint8Array(counts.buffer);
  let binary = '';
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return `uint16|(${nFrames},${height},${width})|${btoa(binary)}`;
}

export function diagnosticFrequencyBand(params: Pick<ISimulationParams, 'nFrames' | 'frameDurationUs' | 'rotationSpeed' | 'pathRotationSpeeds1' | 'pathRotationSpeeds2' | 'pathRotationSpeeds3' | 'pathRotationSpeeds4'>): { fmin: number; fmax: number } {
  const dtS = Math.max(params.frameDurationUs * 1e-6, 1e-9);
  const sampleRateHz = 1 / dtS;
  const frequencyResolutionHz = sampleRateHz / Math.max(params.nFrames, 1);
  const rpmCandidates = [
    params.rotationSpeed,
    ...(params.pathRotationSpeeds1 ?? []),
    ...(params.pathRotationSpeeds2 ?? []),
    ...(params.pathRotationSpeeds3 ?? []),
    ...(params.pathRotationSpeeds4 ?? []),
  ].filter(value => Number.isFinite(value) && value > 0);
  const maxRotorHz = Math.max(0, ...rpmCandidates.map(value => value / 60));
  const fmin = Math.max(0.1, Math.min(0.5, frequencyResolutionHz * 0.5));
  const fmax = Math.min(
    sampleRateHz * 0.45,
    Math.max(25, maxRotorHz * 3, frequencyResolutionHz * 4),
  );
  return { fmin, fmax: Math.max(fmin * 2, fmax) };
}

export function backendEnvironmentScales(params: Pick<ISimulationParams, 'solarIrradiance' | 'atmosphericVisibilityKm' | 'atmosphericAttenuationEnabled' | 'laserWavelengthNm' | 'filterBandwidth'>): {
  solar_irradiance: number;
  scene_stray_rate: number;
  atmospheric_attenuation_enabled: boolean;
  atmospheric_visibility_km: number;
} {
  const visibilityKm = Math.max(0.5, params.atmosphericVisibilityKm ?? 23);
  const hazeScatterScale = 1 + Math.min(3, Math.max(0, 23 / visibilityKm - 1)) * 0.22;
  const wavelengthNm = params.laserWavelengthNm;
  const filterBandwidthNm = params.filterBandwidth;
  const referenceIrradiance = Math.max(1e-6, am0SolarIrradianceWM2Nm(wavelengthNm));
  const solarScale = Math.max(0, Math.min(8, params.solarIrradiance / referenceIrradiance));
  const channelScale = relativeChannelResponse(wavelengthNm, filterBandwidthNm, 550, 50);
  const referenceSceneStrayCpsPerPixel = 350;
  const sceneStrayRate = params.atmosphericAttenuationEnabled
    ? referenceSceneStrayCpsPerPixel * channelScale * solarScale * hazeScatterScale * spectralBackgroundScale('scene_stray', wavelengthNm, filterBandwidthNm)
    : referenceSceneStrayCpsPerPixel * channelScale * solarScale * spectralBackgroundScale('scene_stray', wavelengthNm, filterBandwidthNm);
  return {
    solar_irradiance: params.solarIrradiance,
    scene_stray_rate: Math.max(0, sceneStrayRate),
    atmospheric_attenuation_enabled: params.atmosphericAttenuationEnabled,
    atmospheric_visibility_km: visibilityKm,
  };
}

const MAX_BACKEND_SHAPE_POINTS = 512;

type BackendTrajectoryPayload = {
  times: number[];
  x: number[];
  y: number[];
  z: number[];
  yaw?: number[];
  pitch?: number[];
  roll?: number[];
  phase?: number[];
  rpm1?: number[];
  rpm2?: number[];
  rpm3?: number[];
  rpm4?: number[];
  phase1?: number[];
  phase2?: number[];
  phase3?: number[];
  phase4?: number[];
};

type PropellerShapePayload = {
  x: number[];
  y: number[];
  intensity: number[];
  aspectRatio: number;
};

export interface IBackendCapabilities {
  python_executable: string;
  cpu_workers_default: number;
  torch_available: boolean;
  torch_version: string | null;
  torch_cuda_version: string | null;
  cuda_available: boolean;
  gpu_name: string | null;
  gpu_total_memory_gb: number;
  gpu_compute_capability: string | null;
  recommended_backend: 'cpu' | 'cuda';
  notes: string[];
}

export interface IBackendSimulationSummary {
  scenario_id: string | null;
  scenario_name: string | null;
  simulation_tier: string;
  output_mode: string;
  lightcurve_mode: string;
  compute_backend: 'cpu' | 'cuda';
  sample_backend: 'cpu' | 'cuda';
  encoded_payload_omitted: boolean;
  n_frames: number;
  roi_h: number;
  roi_w: number;
  sample_rate_hz: number;
  snr_db: number;
  observed_total_counts: number;
  total_signal_photons: number;
  total_background_photons: number;
  total_noise_photons: number;
  mean_signal_per_frame: number;
  mean_background_per_frame: number;
  mean_dark_per_frame: number;
  fov_clipping_ratio: number;
  mean_in_fov_ratio: number;
  atmospheric_transmission_mean: number;
  dead_time_loss_ratio: number;
  saturation_warning: boolean;
  visibility_ratio: number;
  dropout_ratio: number;
  target_detected_rate_cps: number;
  target_laser_detected_rate_cps: number;
  target_solar_detected_rate_cps: number;
  truth_freq_hz: number;
  truth_row: number;
  truth_col: number;
  preview_counts: number[][];
  expected_signal_map: number[][];
  warnings: string[];
  assumptions: string[];
}

export function expectedSignalMapFromSummary(
  summary: Pick<IBackendSimulationSummary, 'roi_h' | 'roi_w' | 'truth_row' | 'truth_col' | 'total_signal_photons' | 'expected_signal_map'>,
): number[][] {
  const height = Math.max(0, Math.floor(summary.roi_h));
  const width = Math.max(0, Math.floor(summary.roi_w));
  const expectedMap = summary.expected_signal_map;
  if (
    Array.isArray(expectedMap)
    && expectedMap.length === height
    && expectedMap.every(row => Array.isArray(row) && row.length === width)
  ) {
    return expectedMap.map(row =>
      row.map(value => Number.isFinite(value) ? Math.max(0, value) : 0),
    );
  }

  const map = Array.from({ length: height }, () => Array(width).fill(0));
  if (summary.truth_row >= 0 && summary.truth_row < height && summary.truth_col >= 0 && summary.truth_col < width) {
    map[summary.truth_row][summary.truth_col] = Math.max(0, summary.total_signal_photons);
  }
  return map;
}

export interface IBackendSimulationResponse {
  n_frames: number;
  roi_h: number;
  roi_w: number;
  sample_rate_hz: number;
  truth_freq_hz: number;
  truth_row: number;
  truth_col: number;
  total_signal_photons: number;
  total_background_photons: number;
  total_noise_photons?: number;
  counts_encoded: string;
  expected_signal_map_encoded: string | null;
  truth_signal_series_encoded: string | null;
  truth_cx_series_encoded: string | null;
  truth_cy_series_encoded: string | null;
  truth_projected_width_px_series_encoded: string | null;
  truth_projected_height_px_series_encoded: string | null;
}

export interface IBackendSimulationJob {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  created_at: number;
  updated_at: number;
  summary: IBackendSimulationSummary | null;
  result: IBackendSimulationResponse | null;
  error: string | null;
  download_url: string | null;
}

@Injectable({ providedIn: 'root' })
export class BackendSimulationService {
  private http = inject(HttpClient);
  private readonly baseUrl = this.resolveBackendBaseUrl();
  private readonly propellerShapeCache = new WeakMap<HTMLImageElement, PropellerShapePayload>();

  getCapabilities(): Promise<IBackendCapabilities> {
    return firstValueFrom(this.http.get<IBackendCapabilities>(`${this.baseUrl}/capabilities`));
  }

  runSummary(params: ISimulationParams): Promise<IBackendSimulationSummary> {
    return firstValueFrom(this.http.post<IBackendSimulationSummary>(`${this.baseUrl}/simulate/summary`, this.toSummaryRequest(params)));
  }

  startJob(params: ISimulationParams): Promise<IBackendSimulationJob> {
    return firstValueFrom(this.http.post<IBackendSimulationJob>(`${this.baseUrl}/simulate/jobs`, this.toSummaryRequest(params)));
  }

  getJob(jobId: string): Promise<IBackendSimulationJob> {
    return firstValueFrom(this.http.get<IBackendSimulationJob>(`${this.baseUrl}/simulate/jobs/${jobId}`));
  }

  downloadUrl(downloadUrl: string): string {
    const origin = this.baseUrl.replace(/\/api\/?$/, '');
    return downloadUrl.startsWith('http') ? downloadUrl : `${origin}${downloadUrl}`;
  }

  summaryToSimulationResult(summary: IBackendSimulationSummary, params: ISimulationParams): ISimulationResult {
    const height = summary.roi_h;
    const width = summary.roi_w;
    const totalPixels = width * height;
    const emptyPixelValue = params.tdcMaxCount + 2;
    const dataset = new Uint16Array(summary.n_frames * totalPixels);
    dataset.fill(emptyPixelValue);

    const photonCountMap = Array.from({ length: height }, (_, row) =>
      Array.from({ length: width }, (_, col) => Math.max(0, Math.round(summary.preview_counts[row]?.[col] ?? 0))),
    );
    const detectedPhotons = photonCountMap.flat().reduce((sum, value) => sum + value, 0);
    const groundTruthMap = expectedSignalMapFromSummary(summary);
    const incidentPhotonMap = groundTruthMap.map(row => [...row]);

    return {
      dataset,
      detectedPhotons,
      noiseEvents: Math.max(0, Math.round(summary.total_noise_photons ?? summary.total_background_photons)),
      signalCoordinates: [],
      incidentPhotons: summary.total_signal_photons,
      incidentPhotonMap,
      maxIncidentPhotonsPerPixel: Math.max(...incidentPhotonMap.flat(), 0),
      photonCountMap,
      groundTruthMap,
      resolution: { width, height },
      groundTruthData: {
        times: Array.from({ length: summary.n_frames }, (_, index) => index / Math.max(summary.sample_rate_hz, 1e-9)),
        frequencies: Array(summary.n_frames).fill(summary.truth_freq_hz),
        phases: Array(summary.n_frames).fill(0),
      },
    };
  }

  backendResponseToSimulationResult(response: IBackendSimulationResponse, params: ISimulationParams): ISimulationResult {
    const counts = this.decodeNumericArray(response.counts_encoded);
    const nFrames = response.n_frames;
    const height = response.roi_h;
    const width = response.roi_w;
    const totalPixels = width * height;
    const emptyPixelValue = params.tdcMaxCount + 2;
    const dataset = new Uint16Array(nFrames * totalPixels);
    dataset.fill(emptyPixelValue);

    const photonCountMap = Array.from({ length: height }, () => Array(width).fill(0));
    for (let frame = 0; frame < nFrames; frame++) {
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
          const flatIndex = frame * totalPixels + row * width + col;
          const count = Math.max(0, Math.round(counts[flatIndex] ?? 0));
          if (count > 0) {
            photonCountMap[row][col] += count;
            dataset[flatIndex] = Math.min(params.tdcMaxCount, count);
          }
        }
      }
    }

    const groundTruthMap = this.backendTruthMap(response, height, width, params);
    return {
      dataset,
      detectedPhotons: Math.max(0, Math.round(photonCountMap.flat().reduce((sum, value) => sum + value, 0))),
      noiseEvents: Math.max(0, Math.round(response.total_noise_photons ?? response.total_background_photons)),
      signalCoordinates: [],
      incidentPhotons: response.total_signal_photons,
      incidentPhotonMap: groundTruthMap,
      maxIncidentPhotonsPerPixel: Math.max(...groundTruthMap.flat(), 0),
      photonCountMap,
      groundTruthMap,
      resolution: { width, height },
      groundTruthData: {
        times: Array.from({ length: nFrames }, (_, index) => index / Math.max(response.sample_rate_hz, 1e-9)),
        frequencies: Array(nFrames).fill(response.truth_freq_hz),
        phases: Array(nFrames).fill(0),
      },
    };
  }

  private backendTruthMap(response: IBackendSimulationResponse, height: number, width: number, params: ISimulationParams): number[][] {
    const map = Array.from({ length: height }, () => Array(width).fill(0));
    if (response.expected_signal_map_encoded) {
      const signalMap = this.decodeNumericArray(response.expected_signal_map_encoded);
      if (signalMap.length === height * width) {
        for (let row = 0; row < height; row++) {
          for (let col = 0; col < width; col++) {
            map[row][col] = Math.max(0, signalMap[row * width + col] ?? 0);
          }
        }
        return map;
      }
    }
    const signal = response.truth_signal_series_encoded ? this.decodeNumericArray(response.truth_signal_series_encoded) : [];
    const cx = response.truth_cx_series_encoded ? this.decodeNumericArray(response.truth_cx_series_encoded) : [];
    const cy = response.truth_cy_series_encoded ? this.decodeNumericArray(response.truth_cy_series_encoded) : [];
    const widthPx = response.truth_projected_width_px_series_encoded ? this.decodeNumericArray(response.truth_projected_width_px_series_encoded) : [];
    const heightPx = response.truth_projected_height_px_series_encoded ? this.decodeNumericArray(response.truth_projected_height_px_series_encoded) : [];
    if (signal.length > 0 && cx.length === signal.length && cy.length === signal.length) {
      for (let i = 0; i < signal.length; i++) {
        this.addProjectedFootprint(
          map,
          cx[i],
          cy[i],
          widthPx.length === signal.length ? widthPx[i] : 1,
          heightPx.length === signal.length ? heightPx[i] : 1,
          Math.max(0, signal[i]),
          params,
          i / Math.max(response.sample_rate_hz, 1e-9),
        );
      }
      return map;
    }

    const row = Math.max(0, Math.min(height - 1, response.truth_row));
    const col = Math.max(0, Math.min(width - 1, response.truth_col));
    map[row][col] = Math.max(0, response.total_signal_photons);
    return map;
  }

  private addProjectedFootprint(
    map: number[][],
    cx: number,
    cy: number,
    widthPx: number,
    heightPx: number,
    value: number,
    params: ISimulationParams,
    timeS: number,
  ): void {
    const height = map.length;
    const width = map[0]?.length ?? 0;
    if (height === 0 || width === 0 || value <= 0) return;

    if (params.targetType === 'Ball') {
      this.addWeightedKernel(map, cx, cy, Math.max(0.45, Math.max(widthPx, heightPx) * 0.45), Math.max(0.45, Math.max(widthPx, heightPx) * 0.45), value);
      return;
    }
    if (params.targetType === 'Blade') {
      const angle = params.rotationSpeed * 2 * Math.PI / 60 * timeS;
      this.addWeightedKernel(map, cx, cy, Math.max(1.6, widthPx * 0.32), Math.max(0.42, heightPx * 0.45), value, angle);
      return;
    }
    if (params.targetType === 'Drone') {
      const safeW = Math.max(widthPx, 5);
      const safeH = Math.max(heightPx, 5);
      const minAxis = Math.max(1, Math.min(safeW, safeH));
      const components: Array<[number, number, number, number, number]> = [
        [0, 0, 0.38 * safeW, 0.20 * safeH, 0.36],
        [-0.34 * safeW, -0.34 * safeH, 0.10 * minAxis, 0.10 * minAxis, 0.16],
        [0.34 * safeW, -0.34 * safeH, 0.10 * minAxis, 0.10 * minAxis, 0.16],
        [-0.34 * safeW, 0.34 * safeH, 0.10 * minAxis, 0.10 * minAxis, 0.16],
        [0.34 * safeW, 0.34 * safeH, 0.10 * minAxis, 0.10 * minAxis, 0.16],
      ];
      for (const [offX, offY, sx, sy, weight] of components) {
        this.addWeightedKernel(map, cx + offX, cy + offY, Math.max(0.45, sx), Math.max(0.45, sy), value * weight);
      }
      return;
    }

    const halfW = Math.max(0.5, widthPx / 2);
    const halfH = Math.max(0.5, heightPx / 2);
    const minCol = Math.max(0, Math.floor(cx - halfW - 1));
    const maxCol = Math.min(width - 1, Math.ceil(cx + halfW + 1));
    const minRow = Math.max(0, Math.floor(cy - halfH - 1));
    const maxRow = Math.min(height - 1, Math.ceil(cy + halfH + 1));
    const weights: { row: number; col: number; weight: number }[] = [];
    let totalWeight = 0;

    for (let row = minRow; row <= maxRow; row++) {
      const yOverlap = Math.max(0, Math.min(cy + halfH, row + 0.5) - Math.max(cy - halfH, row - 0.5));
      for (let col = minCol; col <= maxCol; col++) {
        const xOverlap = Math.max(0, Math.min(cx + halfW, col + 0.5) - Math.max(cx - halfW, col - 0.5));
        const dx = (col - cx) / Math.max(widthPx, 1);
        const dy = (row - cy) / Math.max(heightPx, 1);
        const opticalBlur = Math.exp(-0.5 * (dx * dx + dy * dy));
        const weight = xOverlap * yOverlap * opticalBlur;
        if (weight <= 0) continue;
        weights.push({ row, col, weight });
        totalWeight += weight;
      }
    }

    if (totalWeight <= 0) {
      const row = Math.max(0, Math.min(height - 1, Math.round(cy)));
      const col = Math.max(0, Math.min(width - 1, Math.round(cx)));
      map[row][col] += value;
      return;
    }

    for (const item of weights) {
      map[item.row][item.col] += value * item.weight / totalWeight;
    }
  }

  private addWeightedKernel(
    map: number[][],
    cx: number,
    cy: number,
    sigmaX: number,
    sigmaY: number,
    value: number,
    angleRad = 0,
  ): void {
    const height = map.length;
    const width = map[0]?.length ?? 0;
    if (height === 0 || width === 0 || value <= 0) return;
    const radius = Math.ceil(Math.max(sigmaX, sigmaY) * 3);
    const minCol = Math.max(0, Math.floor(cx - radius));
    const maxCol = Math.min(width - 1, Math.ceil(cx + radius));
    const minRow = Math.max(0, Math.floor(cy - radius));
    const maxRow = Math.min(height - 1, Math.ceil(cy + radius));
    const cosA = Math.cos(angleRad);
    const sinA = Math.sin(angleRad);
    const weights: { row: number; col: number; weight: number }[] = [];
    let totalWeight = 0;
    for (let row = minRow; row <= maxRow; row++) {
      for (let col = minCol; col <= maxCol; col++) {
        const dx = col + 0.5 - cx;
        const dy = row + 0.5 - cy;
        const along = dx * cosA + dy * sinA;
        const across = -dx * sinA + dy * cosA;
        const weight = Math.exp(-0.5 * ((along / sigmaX) ** 2 + (across / sigmaY) ** 2));
        if (weight <= 1e-3) continue;
        weights.push({ row, col, weight });
        totalWeight += weight;
      }
    }
    if (totalWeight <= 0) return;
    for (const item of weights) {
      map[item.row][item.col] += value * item.weight / totalWeight;
    }
  }

  private decodeNumericArray(encoded: string): number[] {
    const [dtype, shapeText, base64] = encoded.split('|', 3);
    const length = shapeText
      .replace(/[()]/g, '')
      .split(',')
      .map(item => item.trim())
      .filter(Boolean)
      .reduce((product, item) => product * Number(item), 1);
    const binary = atob(base64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
      bytes[i] = binary.charCodeAt(i);
    }
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    let view: ArrayLike<number>;
    switch (dtype) {
      case 'uint16':
        view = new Uint16Array(buffer);
        break;
      case 'int32':
        view = new Int32Array(buffer);
        break;
      case 'float32':
        view = new Float32Array(buffer);
        break;
      case 'float64':
        view = new Float64Array(buffer);
        break;
      default:
        throw new Error(`Unsupported backend array dtype: ${dtype}`);
    }
    return Array.from(view).slice(0, length);
  }

  private resolveBackendBaseUrl(): string {
    const desktopUrl = (window as unknown as { spadDesktop?: { backendBaseUrl?: string } }).spadDesktop?.backendBaseUrl;
    return desktopUrl || 'http://127.0.0.1:8000/api';
  }

  private toSummaryRequest(params: ISimulationParams): Record<string, unknown> {
    const dtS = Math.max(params.frameDurationUs * 1e-6, 1e-9);
    const observationTimeS = Math.max(params.nFrames * dtS, dtS);
    const detectorPosition = { x: 0, y: params.cameraHeight, z: 0 };
    const targetRangeM = Math.max(0.1, Math.sqrt(
      (params.initialPos.x - detectorPosition.x) ** 2
      + (params.initialPos.y - detectorPosition.y) ** 2
      + (params.initialPos.z - detectorPosition.z) ** 2,
    ));
    const environmentScales = backendEnvironmentScales(params);
    const droneScale = Math.max(0.05, params.droneScale || 1.0);
    const droneLengthM = params.targetType === 'Drone'
      ? (params.dronePreset?.dimensions.lengthM ?? 1.0) * droneScale
      : 0;
    const droneWidthM = params.targetType === 'Drone'
      ? (params.dronePreset?.dimensions.widthM ?? 1.0) * droneScale
      : 0;
    const droneHeightM = params.targetType === 'Drone'
      ? (params.dronePreset?.dimensions.heightM ?? 0.06) * droneScale
      : 0;
    const dronePropellerDiameterM = params.targetType === 'Drone'
      ? (params.dronePreset?.propellerDiameterM ?? 0.10) * droneScale
      : Math.max(0.01, params.rotationRadius);
    const targetAreaM2 = params.targetType === 'Ball'
      ? Math.PI * 0.0335 ** 2
      : params.targetType === 'Drone'
        ? Math.max(0.01, droneLengthM * droneWidthM)
        : Math.max(0.01, params.rotationRadius * 0.05);
    const recorded = params.targetType === 'Drone' && (params.recordedDroneTrajectory?.length ?? 0) > 1
      ? params.recordedDroneTrajectory!
      : null;
    const ballTrajectory = params.targetType === 'Ball' ? this.generateBallBackendTrajectory(params, dtS) : null;
    const pathTrajectory = (params.targetType === 'Blade' || params.targetType === 'Drone') && params.bladeMotionType === 'Path'
      ? this.generatePathBackendTrajectory(params, dtS)
      : null;
    const fixedRotationTrajectory = (params.targetType === 'Blade' || params.targetType === 'Drone') && params.bladeMotionType === 'Fixed'
      ? this.generateFixedRotationBackendTrajectory(params, dtS, observationTimeS)
      : null;
    const backendTrajectory = recorded
      ? {
          times: recorded.map(sample => sample.time),
          x: recorded.map(sample => sample.pos.x),
          y: recorded.map(sample => sample.pos.y),
          z: recorded.map(sample => sample.pos.z),
          yaw: recorded.map(sample => sample.yawDeg),
          pitch: recorded.map(sample => sample.pitchDeg),
          roll: recorded.map(sample => sample.rollDeg),
          rpm1: recorded.map(sample => sample.propellerRpms[0]),
          rpm2: recorded.map(sample => sample.propellerRpms[1]),
          rpm3: recorded.map(sample => sample.propellerRpms[2]),
          rpm4: recorded.map(sample => sample.propellerRpms[3]),
          phase1: this.integrateRpmPhase(recorded.map(sample => sample.time), recorded.map(sample => sample.propellerRpms[0])),
          phase2: this.integrateRpmPhase(recorded.map(sample => sample.time), recorded.map(sample => sample.propellerRpms[1])),
          phase3: this.integrateRpmPhase(recorded.map(sample => sample.time), recorded.map(sample => sample.propellerRpms[2])),
          phase4: this.integrateRpmPhase(recorded.map(sample => sample.time), recorded.map(sample => sample.propellerRpms[3])),
        }
      : pathTrajectory ?? fixedRotationTrajectory ?? ballTrajectory;
    const trajectoryAverageRpm = backendTrajectory
      ? [...(backendTrajectory.rpm1 ?? []), ...(backendTrajectory.rpm2 ?? []), ...(backendTrajectory.rpm3 ?? []), ...(backendTrajectory.rpm4 ?? [])]
          .filter(value => Number.isFinite(value))
          .reduce((sum, value, _, arr) => sum + value / Math.max(arr.length, 1), 0)
      : params.rotationSpeed;
    const recordedAverageRpm = recorded
      ? recorded.reduce((sum, sample) => sum + sample.propellerRpms.reduce((innerSum, rpm) => innerSum + rpm, 0), 0) / Math.max(1, recorded.length * 4)
      : backendTrajectory ? trajectoryAverageRpm : params.rotationSpeed;
    return {
      scenario: null,
      detector_preset: params.detectorPresetId === 'custom' ? 'custom' : 'pf32',
      observation_time_s: observationTimeS,
      sample_rate_hz: 1 / dtS,
      compute_backend: 'auto',
      simulation_tier: 'physics_informed',
      output_mode: 'frame',
      lightcurve_mode: 'attitude_driven',
      save_truth_series: true,
      target_range_m: targetRangeM,
      target_area_m2: targetAreaM2,
      target_length_m: params.targetType === 'Drone'
        ? droneLengthM
        : params.targetType === 'Blade'
          ? params.rotationRadius
          : 0.067,
      target_width_m: params.targetType === 'Drone'
        ? droneWidthM
        : params.targetType === 'Blade'
          ? 0.05
          : 0.067,
      target_height_m: params.targetType === 'Drone'
        ? droneHeightM
        : params.targetType === 'Blade'
          ? 0.002
          : 0.067,
      propeller_diameter_m: dronePropellerDiameterM,
      target_reflectivity: params.reflectivity,
      propeller_reflectivity: params.propellerReflectivity ?? params.reflectivity,
      solar_irradiance: environmentScales.solar_irradiance,
      illumination_mode: 'laser_plus_solar',
      laser_mode: params.laserMode.toLowerCase(),
      laser_average_power_w: params.laserAveragePower,
      laser_pulse_energy_j: params.laserPulseEnergy,
      laser_repetition_frequency_hz: params.laserRepetitionFrequency,
      laser_pulse_width_ns: params.laserPulseWidthNs,
      transmitter_divergence_mrad: params.transmitterDivergenceMrad,
      specular_fraction: params.targetType === 'Drone' ? 0.08 : 0.04,
      outage_fraction: 0,
      glint_probability: 0,
      target_position_x_m: params.initialPos.x,
      target_position_y_m: params.initialPos.y,
      target_position_z_m: params.initialPos.z,
      target_yaw_deg: params.droneYawDeg ?? 0,
      target_pitch_deg: params.dronePitchDeg ?? 0,
      target_roll_deg: params.droneRollDeg ?? 0,
      target_trajectory_times_s: backendTrajectory?.times,
      target_trajectory_x_m: backendTrajectory?.x,
      target_trajectory_y_m: backendTrajectory?.y,
      target_trajectory_z_m: backendTrajectory?.z,
      target_trajectory_yaw_deg: backendTrajectory?.yaw,
      target_trajectory_pitch_deg: backendTrajectory?.pitch,
      target_trajectory_roll_deg: backendTrajectory?.roll,
      target_trajectory_phase_rad: backendTrajectory?.phase,
      target_trajectory_propeller_rpm1: backendTrajectory?.rpm1,
      target_trajectory_propeller_rpm2: backendTrajectory?.rpm2,
      target_trajectory_propeller_rpm3: backendTrajectory?.rpm3,
      target_trajectory_propeller_rpm4: backendTrajectory?.rpm4,
      target_trajectory_propeller_phase1_rad: backendTrajectory?.phase1,
      target_trajectory_propeller_phase2_rad: backendTrajectory?.phase2,
      target_trajectory_propeller_phase3_rad: backendTrajectory?.phase3,
      target_trajectory_propeller_phase4_rad: backendTrajectory?.phase4,
      detector_position_x_m: detectorPosition.x,
      detector_position_y_m: detectorPosition.y,
      detector_position_z_m: detectorPosition.z,
      detector_yaw_deg: params.detectorYaw,
      detector_pitch_deg: params.detectorPitch,
      spin_hz: Math.max(0, recordedAverageRpm / 60),
      precession_hz: 0.0,
      body_shape: params.targetType === 'Drone' ? 'drone_quad' : params.targetType === 'Blade' ? 'blade_strip' : 'sphere',
      ...this.customShapePayload(params),
      aperture_diameter_m: params.apertureDiameter,
      receiver_efficiency: params.systemEfficiency,
      quantum_efficiency: params.quantumEfficiency,
      wavelength_nm: params.laserWavelengthNm,
      filter_bandwidth_nm: params.filterBandwidth,
      detector_fov_urad: Math.max(1, params.detectorFov * Math.PI / 180 * 1e6),
      atmospheric_attenuation_enabled: environmentScales.atmospheric_attenuation_enabled,
      atmospheric_visibility_km: environmentScales.atmospheric_visibility_km,
      scene_stray_rate: environmentScales.scene_stray_rate,
      dark_count_rate: params.darkCountRate,
      dead_time_ns: params.deadTimeNs ?? 0,
      timing_jitter_ns: params.timingJitterNs ?? 0,
      tdc_bin_width_ns: params.timeResolutionPs * 1e-3,
      irf_fwhm_ps: params.irfFwhmPs ?? 0,
      max_count_rate_cps_per_pixel: params.maxCountRateCpsPerPixel ?? 5e6,
      max_count_per_frame: params.tdcMaxCount,
      roi_w: params.resolution.width,
      roi_h: params.resolution.height,
      pixel_pitch_um: params.pixelPitchUm,
      fill_factor: params.fillFactor,
      microlens_gain: params.microlensGain,
    };
  }

  private customShapePayload(params: ISimulationParams): Record<string, unknown> {
    if (params.targetType !== 'Blade' || !params.uploadedImage) return {};
    const extracted = this.propellerShapeCache.get(params.uploadedImage) ?? this.extractPropellerShapePoints(params.uploadedImage);
    this.propellerShapeCache.set(params.uploadedImage, extracted);
    if (extracted.x.length === 0) return {};
    return {
      custom_shape_x: extracted.x,
      custom_shape_y: extracted.y,
      custom_shape_intensity: extracted.intensity,
      custom_shape_aspect_ratio: extracted.aspectRatio,
    };
  }

  private extractPropellerShapePoints(image: HTMLImageElement): PropellerShapePayload {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx || image.width <= 0 || image.height <= 0) {
      return { x: [], y: [], intensity: [], aspectRatio: 1 };
    }
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    const points: Array<{ x: number; y: number; intensity: number }> = [];
    for (let py = 0; py < imageData.height; py++) {
      for (let px = 0; px < imageData.width; px++) {
        const i = (py * imageData.width + px) * 4;
        const r = imageData.data[i];
        const g = imageData.data[i + 1];
        const b = imageData.data[i + 2];
        const a = imageData.data[i + 3];
        const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
        if (a > 128 && luminance < 128) {
          points.push({ x: px, y: py, intensity: Math.max(0, (128 - luminance) / 128) });
        }
      }
    }
    if (points.length === 0) return { x: [], y: [], intensity: [], aspectRatio: image.width / image.height };
    const stride = Math.max(1, Math.ceil(points.length / MAX_BACKEND_SHAPE_POINTS));
    const sampled = points.filter((_, index) => index % stride === 0).slice(0, MAX_BACKEND_SHAPE_POINTS);
    const centroidX = sampled.reduce((sum, p) => sum + p.x, 0) / sampled.length;
    const centroidY = sampled.reduce((sum, p) => sum + p.y, 0) / sampled.length;
    return {
      x: sampled.map(p => (p.x - centroidX) / image.width),
      y: sampled.map(p => (p.y - centroidY) / image.height),
      intensity: sampled.map(p => p.intensity),
      aspectRatio: image.width / image.height,
    };
  }

  private generatePathBackendTrajectory(params: ISimulationParams, dtS: number): BackendTrajectoryPayload | null {
    if (params.waypoints.length < 2) return null;
    const segments = [];
    let cumulativeTime = 0;
    for (let i = 0; i < params.waypoints.length - 1; i++) {
      const start = params.waypoints[i].pos;
      const end = params.waypoints[i + 1].pos;
      const speed = Math.max(1e-6, params.pathSpeeds[i] ?? 1);
      const distance = Math.hypot(end.x - start.x, end.y - start.y, end.z - start.z);
      const duration = distance / speed;
      segments.push({ start, end, duration, cumulativeTime, index: i });
      cumulativeTime += duration;
    }
    if (segments.length === 0 || cumulativeTime <= 0) return null;

    const times: number[] = [];
    const x: number[] = [];
    const y: number[] = [];
    const z: number[] = [];
    const pitch: number[] = [];
    const yaw: number[] = [];
    const roll: number[] = [];
    const rpm1: number[] = [];
    const rpm2: number[] = [];
    const rpm3: number[] = [];
    const rpm4: number[] = [];
    const phase: number[] = [];
    const phase1: number[] = [];
    const phase2: number[] = [];
    const phase3: number[] = [];
    const phase4: number[] = [];
    const frames = Math.max(1, params.nFrames);
    let currentPhase = 0;
    let currentPhase1 = 0;
    let currentPhase2 = 0;
    let currentPhase3 = 0;
    let currentPhase4 = 0;
    let lastTime = 0;
    let lastRpm = params.pathRotationSpeeds[0] ?? params.rotationSpeed;
    let lastRpm1 = params.pathRotationSpeeds1?.[0] ?? lastRpm;
    let lastRpm2 = params.pathRotationSpeeds2?.[0] ?? lastRpm;
    let lastRpm3 = params.pathRotationSpeeds3?.[0] ?? lastRpm;
    let lastRpm4 = params.pathRotationSpeeds4?.[0] ?? lastRpm;

    for (let frame = 0; frame < frames; frame++) {
      const time = Math.min(frame * dtS, cumulativeTime);
      const seg = segments.find(item => time <= item.cumulativeTime + item.duration) ?? segments[segments.length - 1];
      const localT = seg.duration > 0 ? Math.max(0, Math.min(1, (time - seg.cumulativeTime) / seg.duration)) : 1;
      const segRpm = params.pathRotationSpeeds[seg.index] ?? params.rotationSpeed;
      const segRpm1 = params.pathRotationSpeeds1?.[seg.index] ?? segRpm;
      const segRpm2 = params.pathRotationSpeeds2?.[seg.index] ?? segRpm;
      const segRpm3 = params.pathRotationSpeeds3?.[seg.index] ?? segRpm;
      const segRpm4 = params.pathRotationSpeeds4?.[seg.index] ?? segRpm;
      if (frame > 0) {
        const deltaT = Math.max(0, time - lastTime);
        currentPhase += ((lastRpm + segRpm) * 0.5) * 2 * Math.PI / 60 * deltaT;
        currentPhase1 += ((lastRpm1 + segRpm1) * 0.5) * 2 * Math.PI / 60 * deltaT;
        currentPhase2 += ((lastRpm2 + segRpm2) * 0.5) * 2 * Math.PI / 60 * deltaT;
        currentPhase3 += ((lastRpm3 + segRpm3) * 0.5) * 2 * Math.PI / 60 * deltaT;
        currentPhase4 += ((lastRpm4 + segRpm4) * 0.5) * 2 * Math.PI / 60 * deltaT;
      }
      times.push(time);
      x.push(seg.start.x + (seg.end.x - seg.start.x) * localT);
      y.push(seg.start.y + (seg.end.y - seg.start.y) * localT);
      z.push(seg.start.z + (seg.end.z - seg.start.z) * localT);
      pitch.push(params.targetType === 'Drone' ? params.dronePitchDeg ?? params.bladePitch : params.bladePitch);
      yaw.push(params.droneYawDeg ?? 0);
      roll.push(params.droneRollDeg ?? 0);
      phase.push(currentPhase);
      phase1.push(currentPhase1);
      phase2.push(currentPhase2);
      phase3.push(currentPhase3);
      phase4.push(currentPhase4);
      rpm1.push(segRpm1);
      rpm2.push(segRpm2);
      rpm3.push(segRpm3);
      rpm4.push(segRpm4);
      lastTime = time;
      lastRpm = segRpm;
      lastRpm1 = segRpm1;
      lastRpm2 = segRpm2;
      lastRpm3 = segRpm3;
      lastRpm4 = segRpm4;
    }
    return { times, x, y, z, yaw, pitch, roll, phase, rpm1, rpm2, rpm3, rpm4, phase1, phase2, phase3, phase4 };
  }

  private generateFixedRotationBackendTrajectory(params: ISimulationParams, dtS: number, observationTimeS: number): BackendTrajectoryPayload {
    const frames = Math.max(1, params.nFrames);
    const times = Array.from({ length: frames }, (_, frame) => Math.min(frame * dtS, observationTimeS));
    const x = Array(frames).fill(params.initialPos.x);
    const y = Array(frames).fill(params.initialPos.y);
    const z = Array(frames).fill(params.initialPos.z);
    const yaw = Array(frames).fill(params.droneYawDeg ?? 0);
    const pitch = Array(frames).fill(params.targetType === 'Drone' ? params.dronePitchDeg ?? params.bladePitch : params.bladePitch);
    const roll = Array(frames).fill(params.droneRollDeg ?? 0);

    if (params.targetType === 'Blade') {
      const single = this.sampleKeyframeRpmAndPhase(times, params.rotationKeyframes, params.rotationSpeed);
      return { times, x, y, z, yaw, pitch, roll, phase: single.phase, rpm1: single.rpm, rpm2: single.rpm, rpm3: single.rpm, rpm4: single.rpm };
    }

    const fallback = params.rotationSpeed;
    const prop1 = this.sampleKeyframeRpmAndPhase(times, params.rotationKeyframes1 ?? [], fallback);
    const prop2 = this.sampleKeyframeRpmAndPhase(times, params.rotationKeyframes2 ?? [], fallback);
    const prop3 = this.sampleKeyframeRpmAndPhase(times, params.rotationKeyframes3 ?? [], fallback);
    const prop4 = this.sampleKeyframeRpmAndPhase(times, params.rotationKeyframes4 ?? [], fallback);
    return {
      times,
      x,
      y,
      z,
      yaw,
      pitch,
      roll,
      rpm1: prop1.rpm,
      rpm2: prop2.rpm,
      rpm3: prop3.rpm,
      rpm4: prop4.rpm,
      phase1: prop1.phase,
      phase2: prop2.phase,
      phase3: prop3.phase,
      phase4: prop4.phase,
    };
  }

  private sampleKeyframeRpmAndPhase(times: number[], keyframes: { time: number; rpm: number }[], fallbackRpm: number): { rpm: number[]; phase: number[] } {
    const sorted = [...keyframes].filter(kf => Number.isFinite(kf.time) && Number.isFinite(kf.rpm)).sort((a, b) => a.time - b.time);
    const rpmAt = (time: number): number => {
      if (sorted.length === 0) return Math.max(0, fallbackRpm);
      if (time <= sorted[0].time) return Math.max(0, sorted[0].rpm);
      for (let i = 1; i < sorted.length; i++) {
        const prev = sorted[i - 1];
        const next = sorted[i];
        if (time <= next.time) {
          const alpha = Math.max(0, Math.min(1, (time - prev.time) / Math.max(next.time - prev.time, 1e-9)));
          return Math.max(0, prev.rpm + (next.rpm - prev.rpm) * alpha);
        }
      }
      return Math.max(0, sorted[sorted.length - 1].rpm);
    };
    const rpm = times.map(rpmAt);
    const phase = this.integrateRpmPhase(times, rpm);
    return { rpm, phase };
  }

  private integrateRpmPhase(times: number[], rpms: number[]): number[] {
    const phase: number[] = [];
    let currentPhase = 0;
    for (let i = 0; i < times.length; i++) {
      if (i > 0) {
        const deltaT = Math.max(0, times[i] - times[i - 1]);
        currentPhase += ((rpms[i - 1] + rpms[i]) * 0.5) * 2 * Math.PI / 60 * deltaT;
      }
      phase.push(currentPhase);
    }
    return phase;
  }

  private generateBallBackendTrajectory(params: ISimulationParams, dtS: number): BackendTrajectoryPayload {
    const times: number[] = [];
    const x: number[] = [];
    const y: number[] = [];
    const z: number[] = [];
    let pos = { ...params.initialPos };
    let vel = { ...params.initialVel };
    const g = 9.8;
    const omega = params.rotationSpeed * 2 * Math.PI / 60;
    const isRotating = params.ballMotionType === 'Rotation';

    for (let frame = 0; frame < params.nFrames; frame++) {
      const time = frame * dtS;
      if (isRotating) {
        pos = {
          x: params.rotationCenter.x + params.rotationRadius * Math.cos(omega * time),
          y: params.initialPos.y,
          z: params.rotationCenter.z + params.rotationRadius * Math.sin(omega * time),
        };
      } else {
        pos = {
          x: pos.x + vel.x * dtS,
          y: pos.y + vel.y * dtS,
          z: pos.z + vel.z * dtS,
        };
        vel = { ...vel, y: vel.y - g * dtS };
        if (pos.y <= 0) {
          pos = { ...pos, y: 0 };
          vel = { ...vel, y: -vel.y * params.restitution };
        }
      }
      times.push(time);
      x.push(pos.x);
      y.push(pos.y);
      z.push(pos.z);
    }
    return { times, x, y, z };
  }
}
