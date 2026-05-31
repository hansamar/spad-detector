
import { IDronePreset } from './drone-presets.model';
import { IDetectorPreset } from '../services/detector-preset.service';

export interface IWaypoint {
  id: number;
  pos: { x: number; y: number; z: number };
}

export interface IRotationKeyframe {
  id: number;
  time: number;
  rpm: number;
}

export interface IRecordedDroneSample {
  time: number;
  pos: { x: number; y: number; z: number };
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  propellerRpms: [number, number, number, number];
}

// Photon event for event-based processing (particle filter input)
export interface IPhotonEvent {
  timestamp: number;  // Absolute time in nanoseconds
  x: number;          // Pixel column index (0 to width-1)
  y: number;          // Pixel row index (0 to height-1)
  tof: number;        // Time-of-flight value in TDC units
}

// Ground truth data for algorithm validation
export interface IGroundTruthData {
  times: number[];       // Time of each frame in seconds
  frequencies: number[]; // True frequency at each frame in Hz
  phases: number[];      // True phase at each frame in radians
  distances?: number[];  // True distance at each frame in meters (for ToF comparison)
}

export interface ISimulationParams {
  // Target
  targetType: 'Ball' | 'Blade' | 'Drone';
  initialPos: { x: number; y: number; z: number };
  initialVel: { x: number; y: number; z: number };
  reflectivity: number; // Used as Body Reflectivity for Drone
  propellerReflectivity?: number; // Specific for Drone Propellers
  restitution: number;

  // Motion Types
  ballMotionType: 'Gravity' | 'Rotation';
  bladeMotionType: 'Fixed' | 'Path' | 'Manual';

  // Blade Path Planning
  waypoints: IWaypoint[];
  pathSpeeds: number[];
  pathRotationSpeeds: number[]; // RPM for each segment (single blade)
  pathRotationSpeeds1?: number[]; // For Drone Prop 1
  pathRotationSpeeds2?: number[]; // For Drone Prop 2
  pathRotationSpeeds3?: number[]; // For Drone Prop 3
  pathRotationSpeeds4?: number[]; // For Drone Prop 4

  // Dynamic Target
  rotationRadius: number;
  rotationSpeed: number; // RPM
  rotationKeyframes: IRotationKeyframe[]; // For fixed blade with variable speed
  rotationKeyframes1?: IRotationKeyframe[]; // For Drone Prop 1
  rotationKeyframes2?: IRotationKeyframe[]; // For Drone Prop 2
  rotationKeyframes3?: IRotationKeyframe[]; // For Drone Prop 3
  rotationKeyframes4?: IRotationKeyframe[]; // For Drone Prop 4
  bladePitch: number; // degrees
  rotationCenter: { x: number; z: number };
  uploadedImage: HTMLImageElement | null;
  droneScale: number; // New scale parameter for drone
  dronePresetId?: string;
  dronePreset?: IDronePreset;
  droneYawDeg?: number;
  dronePitchDeg?: number;
  droneRollDeg?: number;
  manualControlEnabled?: boolean;
  spadPoseLocked?: boolean;
  recordedDroneTrajectory?: IRecordedDroneSample[];

  // Detector
  resolution: { width: number; height: number };
  detectorPresetId?: string;
  detectorPreset?: IDetectorPreset;
  detectorFov: number;
  detectorYaw: number;
  detectorPitch: number;
  pixelPitchUm: number;
  fillFactor: number;
  microlensGain: number;
  frameDurationUs: number;
  quantumEfficiency: number;
  apertureDiameter: number;
  systemEfficiency: number;
  filterBandwidth: number;
  darkCountRate: number;
  deadTimeNs?: number;
  timingJitterNs?: number;
  irfFwhmPs?: number;
  maxCountRateCpsPerPixel?: number;
  timeResolutionPs: number;
  tdcMaxCount: number;

  // Environment & Laser
  environmentPresetId?: string;
  solarIrradiance: number;
  atmosphericAttenuationEnabled: boolean;
  atmosphericVisibilityKm: number;
  laserMode: 'Pulsed' | 'CW';
  laserPulseEnergy: number; // Only for Pulsed
  laserAveragePower: number; // Used for both, but primary for CW
  laserRepetitionFrequency: number; // Only for Pulsed
  laserPulseWidthNs: number; // Only for Pulsed
  laserWavelengthNm: number;
  transmitterDivergenceMrad: number; // Full-angle transmitter beam divergence

  // Simulation
  nFrames: number;

  // Fixed params
  cameraHeight: number;
}

export interface ISimulationResult {
  dataset: Uint16Array;
  countDataset?: Uint16Array;
  detectedPhotons: number; // Renamed from signalPhotons
  noiseEvents: number;
  signalCoordinates: { row: number; col: number }[];
  incidentPhotons: number; // Total theoretical signal photons hitting the detector
  incidentPhotonMap: number[][]; // For heatmap of incident photons
  maxIncidentPhotonsPerPixel: number; // Peak incident photons on a single pixel from a single pulse (Pulsed) or frame (CW)
  photonCountMap?: number[][]; // Optional pre-aggregated backend frame image
  groundTruthMap?: number[][]; // Optional backend truth image for result display
  resolution?: { width: number; height: number };

  // Ground truth output for algorithm validation
  groundTruthData?: IGroundTruthData;

  // SIM-002: Photon event stream for particle filter
  photonEvents?: IPhotonEvent[];
}
