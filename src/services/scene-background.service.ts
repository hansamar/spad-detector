function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function deterministicPixelNoise(row: number, col: number): number {
  const s = Math.sin((row + 1) * 12.9898 + (col + 1) * 78.233) * 43758.5453;
  return (s - Math.floor(s)) * 2 - 1;
}

export function backgroundTemporalDriftFactor(timeS: number, depth = 0.08, driftHz = 0.7, phaseRad = 0.5): number {
  const safeDepth = clamp(depth, 0, 0.95);
  return Math.max(0, 1 + safeDepth * Math.cos(2 * Math.PI * Math.max(0, driftHz) * timeS + phaseRad));
}

export function backgroundSpatialFactor(
  row: number,
  col: number,
  height: number,
  width: number,
  sigma = 0.05,
  gradientX = 0.08,
  gradientY = -0.04,
): number {
  const x = width > 1 ? (col - (width - 1) / 2) / ((width - 1) / 2) : 0;
  const y = height > 1 ? (row - (height - 1) / 2) / ((height - 1) / 2) : 0;
  const fixedPattern = clamp(sigma, 0, 1) * deterministicPixelNoise(row, col);
  return Math.max(0.05, 1 + gradientX * x + gradientY * y + fixedPattern);
}

export function makeBackgroundSpatialMap(
  height: number,
  width: number,
  sigma = 0.05,
  gradientX = 0.08,
  gradientY = -0.04,
): number[][] {
  const map = Array(height).fill(0).map((_, row) => (
    Array(width).fill(0).map((__, col) => backgroundSpatialFactor(row, col, height, width, sigma, gradientX, gradientY))
  ));
  const values = map.flat();
  const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  return map.map(row => row.map(value => value / Math.max(mean, 1e-12)));
}
