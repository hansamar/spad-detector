export type SpectralBackgroundComponent = 'scene_stray';

export const PF32_PDP_TABLE: Array<[number, number]> = [
  [350, 0.054], [360, 0.053], [370, 0.070], [380, 0.093], [390, 0.105],
  [400, 0.132], [410, 0.158], [420, 0.157], [430, 0.200], [440, 0.230],
  [450, 0.258], [460, 0.259], [470, 0.254], [480, 0.266], [490, 0.273],
  [500, 0.277], [510, 0.255], [520, 0.231], [530, 0.243], [540, 0.265],
  [550, 0.274], [560, 0.260], [570, 0.231], [580, 0.218], [590, 0.210],
  [600, 0.218], [610, 0.230], [620, 0.217], [630, 0.205], [640, 0.192],
  [650, 0.183], [660, 0.165], [670, 0.158], [680, 0.149], [690, 0.146],
  [700, 0.135], [710, 0.131], [720, 0.126], [730, 0.112], [740, 0.107],
  [750, 0.100], [760, 0.089], [770, 0.079], [780, 0.070], [790, 0.066],
  [800, 0.061], [810, 0.059], [820, 0.058], [830, 0.058], [840, 0.055],
  [850, 0.049], [860, 0.044], [870, 0.039], [880, 0.034], [890, 0.031],
  [900, 0.028], [910, 0.026], [920, 0.023], [930, 0.021], [940, 0.019],
  [950, 0.017], [960, 0.015], [970, 0.012], [980, 0.009], [990, 0.008],
  [1000, 0.007], [1010, 0.005], [1020, 0.004], [1030, 0.0035], [1040, 0.0028],
  [1050, 0.002],
];

export const AM0_IRRADIANCE_TABLE: Array<[number, number]> = [
  [350, 1.15], [400, 1.55], [450, 1.80], [500, 1.90], [550, 1.86],
  [600, 1.78], [650, 1.66], [700, 1.55], [750, 1.43], [800, 1.30],
  [850, 1.16], [900, 1.00], [950, 0.86], [1000, 0.72], [1050, 0.58],
];

export function interpolateClamped(x: number, table: Array<[number, number]>): number {
  if (x <= table[0][0]) return table[0][1];
  const last = table[table.length - 1];
  if (x >= last[0]) return last[1];

  for (let i = 0; i < table.length - 1; i++) {
    const [x0, y0] = table[i];
    const [x1, y1] = table[i + 1];
    if (x >= x0 && x <= x1) {
      const t = (x - x0) / (x1 - x0);
      return y0 + t * (y1 - y0);
    }
  }
  return last[1];
}

export function pf32PdpFraction(wavelengthNm: number): number {
  return interpolateClamped(wavelengthNm, PF32_PDP_TABLE);
}

export function am0SolarIrradianceWM2Nm(wavelengthNm: number): number {
  return interpolateClamped(wavelengthNm, AM0_IRRADIANCE_TABLE);
}

function clippedLinear(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function sceneStrayColorFactor(wavelengthNm: number): number {
  return clippedLinear(1.05 - 0.00025 * (wavelengthNm - 550), 0.85, 1.1);
}

export function referenceChannelResponse(referenceWavelengthNm = 550, referenceBandwidthNm = 50): number {
  return am0SolarIrradianceWM2Nm(referenceWavelengthNm)
    * pf32PdpFraction(referenceWavelengthNm)
    * referenceBandwidthNm;
}

export function relativeChannelResponse(
  wavelengthNm: number,
  bandwidthNm: number,
  referenceWavelengthNm = 550,
  referenceBandwidthNm = 50,
): number {
  const numerator = am0SolarIrradianceWM2Nm(wavelengthNm) * pf32PdpFraction(wavelengthNm) * bandwidthNm;
  return numerator / Math.max(referenceChannelResponse(referenceWavelengthNm, referenceBandwidthNm), 1e-12);
}

export function spectralBackgroundScale(
  component: SpectralBackgroundComponent | string,
  wavelengthNm: number,
  bandwidthNm: number,
  referenceWavelengthNm = 550,
  referenceBandwidthNm = 50,
): number {
  void bandwidthNm;
  void referenceBandwidthNm;
  void component;
  const color = sceneStrayColorFactor(wavelengthNm);
  const referenceColor = sceneStrayColorFactor(referenceWavelengthNm);
  return color / Math.max(referenceColor, 1e-12);
}

export function applyDeadTimeRate(rateCps: number, tauSeconds: number, model: 'nonparalyzable' | 'paralyzable' = 'nonparalyzable'): number {
  const rate = Math.max(0, rateCps);
  if (tauSeconds <= 0 || rate === 0) return rate;
  if (model === 'paralyzable') {
    return rate * Math.exp(-rate * tauSeconds);
  }
  return rate / (1 + rate * tauSeconds);
}
