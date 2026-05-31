export interface IDronePreset {
  id: string;
  manufacturer: 'DJI';
  model: string;
  className: string;
  massKg: number;
  dimensions: {
    lengthM: number;
    widthM: number;
    heightM: number;
  };
  propellerDiameterM: number;
  maxHorizontalSpeedMps: number;
  maxAscentSpeedMps: number;
  maxDescentSpeedMps: number;
  maxHoverTimeMin: number;
  nominalHoverRpm: number;
  bodyReflectivity: number;
  propellerReflectivity: number;
  visualScale: number;
  notes: string;
}

export const DJI_DRONE_PRESETS: IDronePreset[] = [
  {
    id: 'dji-mini-4-pro',
    manufacturer: 'DJI',
    model: 'DJI Mini 4 Pro',
    className: 'Sub-250 g mapping drone',
    massKg: 0.249,
    dimensions: { lengthM: 0.298, widthM: 0.373, heightM: 0.101 },
    propellerDiameterM: 0.12,
    maxHorizontalSpeedMps: 16,
    maxAscentSpeedMps: 5,
    maxDescentSpeedMps: 5,
    maxHoverTimeMin: 30,
    nominalHoverRpm: 11500,
    bodyReflectivity: 0.18,
    propellerReflectivity: 0.28,
    visualScale: 1,
    notes: 'Small consumer quadcopter with compact projected area and fast rotor modulation.',
  },
  {
    id: 'dji-mavic-3-pro',
    manufacturer: 'DJI',
    model: 'DJI Mavic 3 Pro',
    className: 'Prosumer imaging drone',
    massKg: 0.958,
    dimensions: { lengthM: 0.3475, widthM: 0.2908, heightM: 0.1077 },
    propellerDiameterM: 0.24,
    maxHorizontalSpeedMps: 21,
    maxAscentSpeedMps: 8,
    maxDescentSpeedMps: 6,
    maxHoverTimeMin: 37,
    nominalHoverRpm: 7200,
    bodyReflectivity: 0.16,
    propellerReflectivity: 0.24,
    visualScale: 1,
    notes: 'Larger folding quadcopter suitable for medium-range active SPAD tests.',
  },
  {
    id: 'dji-inspire-3',
    manufacturer: 'DJI',
    model: 'DJI Inspire 3',
    className: 'Cinema drone',
    massKg: 3.995,
    dimensions: { lengthM: 0.5005, widthM: 0.7098, heightM: 0.176 },
    propellerDiameterM: 0.42,
    maxHorizontalSpeedMps: 26,
    maxAscentSpeedMps: 8,
    maxDescentSpeedMps: 8,
    maxHoverTimeMin: 25,
    nominalHoverRpm: 5200,
    bodyReflectivity: 0.14,
    propellerReflectivity: 0.22,
    visualScale: 1,
    notes: 'Large cinema platform with high cross-section and slower rotor modulation.',
  },
  {
    id: 'dji-matrice-350-rtk',
    manufacturer: 'DJI',
    model: 'DJI Matrice 350 RTK',
    className: 'Enterprise payload drone',
    massKg: 6.47,
    dimensions: { lengthM: 0.81, widthM: 0.67, heightM: 0.43 },
    propellerDiameterM: 0.53,
    maxHorizontalSpeedMps: 23,
    maxAscentSpeedMps: 6,
    maxDescentSpeedMps: 5,
    maxHoverTimeMin: 38,
    nominalHoverRpm: 4300,
    bodyReflectivity: 0.13,
    propellerReflectivity: 0.20,
    visualScale: 1,
    notes: 'Enterprise RTK platform with the largest visual and optical cross-section preset.',
  },
];

export function findDronePreset(id: string): IDronePreset {
  return DJI_DRONE_PRESETS.find(preset => preset.id === id) ?? DJI_DRONE_PRESETS[0];
}
