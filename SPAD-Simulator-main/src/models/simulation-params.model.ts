export interface IWaypoint {
  id: number;
  pos: { x: number; y: number; z: number };
}

export interface ISimulationParams {
  // Target
  targetType: 'Ball' | 'Blade';
  initialPos: { x: number; y: number; z: number };
  initialVel: { x: number; y: number; z: number };
  reflectivity: number;
  restitution: number;

  // Motion Types
  ballMotionType: 'Gravity' | 'Rotation';
  bladeMotionType: 'Fixed' | 'Path';
  
  // Blade Path Planning
  waypoints: IWaypoint[];
  pathSpeeds: number[];

  // Dynamic Target
  rotationRadius: number;
  rotationSpeed: number; // RPM
  bladePitch: number; // degrees
  rotationCenter: { x: number; z: number };
  uploadedImage: HTMLImageElement | null;

  // Detector
  resolution: { width: number; height: number };
  detectorFov: number;
  detectorYaw: number;
  detectorPitch: number;
  frameDurationUs: number;
  quantumEfficiency: number;
  apertureDiameter: number;
  systemEfficiency: number;
  filterBandwidth: number;
  darkCountRate: number;
  timeResolutionPs: number;
  tdcMaxCount: number;

  // Environment & Laser
  solarIrradiance: number;
  atmosphericAttenuationEnabled: boolean;
  laserMode: 'Pulsed' | 'CW';
  laserPulseEnergy: number; // Only for Pulsed
  laserAveragePower: number; // Used for both, but primary for CW
  laserRepetitionFrequency: number; // Only for Pulsed
  laserPulseWidthNs: number; // Only for Pulsed
  laserWavelengthNm: number;

  // Simulation
  nFrames: number;

  // Fixed params
  cameraHeight: number;
}

export interface ISimulationResult {
  dataset: Uint16Array;
  detectedPhotons: number; // Renamed from signalPhotons
  noiseEvents: number;
  signalCoordinates: { row: number; col: number }[];
  incidentPhotons: number; // Total theoretical signal photons hitting the detector
  incidentPhotonMap: number[][]; // For heatmap of incident photons
  maxIncidentPhotonsPerPixel: number; // Peak incident photons on a single pixel from a single pulse (Pulsed) or frame (CW)
}