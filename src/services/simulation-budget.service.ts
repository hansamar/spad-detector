export type SimulationBudgetLevel = 'safe' | 'caution' | 'blocked';

export interface ISimulationBudget {
  totalSamples: number;
  datasetMb: number;
  estimatedWorkingSetMb: number;
  level: SimulationBudgetLevel;
}

const MAX_BACKEND_FRAMES = 200_000;
const MAX_BACKEND_SAMPLES = 204_800_000;
const CAUTION_SAMPLES = 96_000_000;

export function estimateSimulationBudget(
  resolution: { width: number; height: number },
  nFrames: number,
): ISimulationBudget {
  const width = Math.max(1, Math.floor(resolution.width));
  const height = Math.max(1, Math.floor(resolution.height));
  const frames = Math.max(1, Math.floor(nFrames));
  const totalSamples = width * height * frames;
  const datasetMb = totalSamples * Uint16Array.BYTES_PER_ELEMENT / (1024 * 1024);
  const estimatedWorkingSetMb = totalSamples * 70 / (1024 * 1024) + width * height * 8 / (1024 * 1024);
  const level: SimulationBudgetLevel =
    frames > MAX_BACKEND_FRAMES || totalSamples > MAX_BACKEND_SAMPLES
      ? 'blocked'
      : totalSamples > CAUTION_SAMPLES
        ? 'caution'
        : 'safe';

  return {
    totalSamples,
    datasetMb,
    estimatedWorkingSetMb,
    level,
  };
}

export function recommendedFrameCount(resolution: { width: number; height: number }, targetDatasetMb = 96): number {
  const pixels = Math.max(1, Math.floor(resolution.width) * Math.floor(resolution.height));
  const frames = Math.floor((targetDatasetMb * 1024 * 1024) / (pixels * Uint16Array.BYTES_PER_ELEMENT));
  return Math.max(100, frames);
}
