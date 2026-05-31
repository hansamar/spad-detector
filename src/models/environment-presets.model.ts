export type EnvironmentPresetId =
  | 'lab_dim'
  | 'clear_day'
  | 'urban_haze'
  | 'golden_hour'
  | 'night_operation';

export interface IEnvironmentPreset {
  id: EnvironmentPresetId;
  label: string;
  labelZh: string;
  description: string;
  solarScale: number;
  visibilityKm: number;
  skyTurbidity: number;
  horizonHaze: number;
}

export const ENVIRONMENT_PRESETS: IEnvironmentPreset[] = [
  {
    id: 'lab_dim',
    label: 'Indoor / Dim Bench',
    labelZh: '室内弱光台架',
    description: 'Low ambient light for controlled detector checkout.',
    solarScale: 0.00005,
    visibilityKm: 50,
    skyTurbidity: 0.15,
    horizonHaze: 0.1,
  },
  {
    id: 'clear_day',
    label: 'Clear Day',
    labelZh: '晴朗日间',
    description: 'Nominal outdoor daylight with long-range visibility.',
    solarScale: 1.0,
    visibilityKm: 23,
    skyTurbidity: 0.35,
    horizonHaze: 0.22,
  },
  {
    id: 'urban_haze',
    label: 'Urban Haze',
    labelZh: '城市轻霾',
    description: 'High diffuse background and stronger aerosol attenuation.',
    solarScale: 0.78,
    visibilityKm: 6,
    skyTurbidity: 0.78,
    horizonHaze: 0.68,
  },
  {
    id: 'golden_hour',
    label: 'Low Sun',
    labelZh: '低太阳高度',
    description: 'Warm low-angle sunlight with elevated horizon scatter.',
    solarScale: 0.28,
    visibilityKm: 15,
    skyTurbidity: 0.55,
    horizonHaze: 0.5,
  },
  {
    id: 'night_operation',
    label: 'Night Operation',
    labelZh: '夜间探测',
    description: 'Minimal solar background with mostly dark-count-limited operation.',
    solarScale: 0.00002,
    visibilityKm: 30,
    skyTurbidity: 0.05,
    horizonHaze: 0.04,
  },
];

export function findEnvironmentPreset(id: string): IEnvironmentPreset {
  return ENVIRONMENT_PRESETS.find(preset => preset.id === id) ?? ENVIRONMENT_PRESETS[0];
}
