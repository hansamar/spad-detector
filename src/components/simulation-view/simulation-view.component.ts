
import { ChangeDetectionStrategy, Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, signal, computed, effect, WritableSignal, inject } from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ISimulationParams, ISimulationResult, IWaypoint, IRotationKeyframe, IRecordedDroneSample } from '../../models/simulation-params.model';
import { DJI_DRONE_PRESETS, IDronePreset, findDronePreset } from '../../models/drone-presets.model';
import { ENVIRONMENT_PRESETS, findEnvironmentPreset } from '../../models/environment-presets.model';
import { DETECTOR_PRESETS, getDetectorPreset, resolveDetectorSettings } from '../../services/detector-preset.service';
import { estimateSimulationBudget, recommendedFrameCount } from '../../services/simulation-budget.service';
import { BackendSimulationService, IBackendCapabilities, IBackendSimulationJob, IBackendSimulationSummary } from '../../services/backend-simulation.service';
import { PhysicsService } from '../../services/physics.service';
import { CommonModule } from '@angular/common';
import { LocalizationService } from '../../services/localization.service';

const SCENE_LAYER = 0;
const RIG_LAYER = 1;

@Component({
  selector: 'app-simulation-view',
  standalone: true,
  templateUrl: './simulation-view.component.html',
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SimulationViewComponent implements AfterViewInit, OnDestroy {
  @ViewChild('rendererCanvas', { static: true })
  private rendererCanvas!: ElementRef<HTMLCanvasElement>;

  @ViewChild('canvasContainer', { static: true })
  private canvasContainer!: ElementRef<HTMLElement>;

  @ViewChild('detectorViewport', { static: true })
  private detectorViewport!: ElementRef<HTMLElement>;

  @ViewChild('photonCountingCanvas')
  private photonCountingCanvas?: ElementRef<HTMLCanvasElement>;

  @ViewChild('groundTruthCanvas')
  private groundTruthCanvas?: ElementRef<HTMLCanvasElement>;

  @ViewChild('incidentPhotonsCanvas')
  private incidentPhotonsCanvas?: ElementRef<HTMLCanvasElement>;

  @ViewChild('fileInput')
  private fileInput?: ElementRef<HTMLInputElement>;


  // --- Services ---
  private physicsService = inject(PhysicsService);
  private backendSimulationService = inject(BackendSimulationService);
  l = inject(LocalizationService); // Public for template access
  private backendJobParams = new Map<string, ISimulationParams>();
  private activeBackendJobId: string | null = null;

  // --- UI State ---
  isSimulating = signal(false);
  isPreviewing = signal(false);
  simulationProgress = signal(0);
  simulationResult = signal<ISimulationResult | null>(null);
  backendSummary = signal<IBackendSimulationSummary | null>(null);
  backendJob = signal<IBackendSimulationJob | null>(null);
  backendCapabilities = signal<IBackendCapabilities | null>(null);
  backendError = signal('');
  isBackendSimulating = signal(false);
  showResultsModal = signal(false);
  isPanelOpen = signal(false);
  uploadedImage = signal<HTMLImageElement | null>(null);
  uploadedImageUrl = signal<string | null>(null);
  binarizedImageUrl = signal<string | null>(null);
  activeKeyframeTab = signal(1);

  // Resizable Panel State
  panelWidth = signal(320); // Default width in px
  private isResizing = false;
  private startX = 0;
  private startWidth = 0;

  // Accordion state
  targetSettingsOpen = signal(true);
  detectorSettingsOpen = signal(false);
  envSettingsOpen = signal(false);
  simSettingsOpen = signal(false);

  // --- Simulation Parameters ---
  // Target
  targetType = signal<'Ball' | 'Blade' | 'Drone'>('Blade');
  initialPosX = signal(0);
  initialPosY = signal(1.0);
  initialPosZ = signal(1.5);
  initialVelX = signal(1.0);
  initialVelY = signal(0.0);
  initialVelZ = signal(1.0);
  reflectivity = signal(0.1);
  propellerReflectivity = signal(0.3); // Specific for Drone Propellers
  restitution = signal(0.8);

  // Motion Types
  ballMotionType = signal<'Gravity' | 'Rotation'>('Gravity');
  bladeMotionType = signal<'Fixed' | 'Path' | 'Manual'>('Fixed');

  // Blade Path Planning
  waypoints = signal<IWaypoint[]>([]);
  pathSpeeds = signal<number[]>([]); // Speed in m/s for each segment
  pathDurations = signal<number[]>([]); // Duration in s for each segment
  pathRotationSpeeds = signal<number[]>([]); // RPM for each segment (single blade)

  // Drone Path Planning Speeds
  pathRotationSpeeds1 = signal<number[]>([]);
  pathRotationSpeeds2 = signal<number[]>([]);
  pathRotationSpeeds3 = signal<number[]>([]);
  pathRotationSpeeds4 = signal<number[]>([]);

  // Dynamic Target
  rotationRadius = signal(0.5); // For Blade, this is size/length. For Ball, rotation radius.
  rotationSpeed = signal(12000); // in RPM, for Ball
  bladePitch = signal(90); // in degrees, for Blade target or Drone body
  rotationKeyframes = signal<IRotationKeyframe[]>([]); // Single Blade
  droneScale = signal(1.0);

  // Drone Fixed Position Speeds
  rotationKeyframes1 = signal<IRotationKeyframe[]>([]);
  rotationKeyframes2 = signal<IRotationKeyframe[]>([]);
  rotationKeyframes3 = signal<IRotationKeyframe[]>([]);
  rotationKeyframes4 = signal<IRotationKeyframe[]>([]);

  readonly dronePresets = DJI_DRONE_PRESETS;
  selectedDronePresetId = signal(DJI_DRONE_PRESETS[0].id);
  selectedDronePreset = computed(() => findDronePreset(this.selectedDronePresetId()));
  manualDroneControl = signal(false);
  manualDroneSpeed = signal(4);
  manualDronePosition = signal({ x: 0, y: 1.5, z: 4 });
  manualDroneYaw = signal(0);
  manualDronePitch = signal(0);
  manualDroneRoll = signal(0);
  manualDronePropellerRpms = signal<[number, number, number, number]>([
    DJI_DRONE_PRESETS[0].nominalHoverRpm,
    DJI_DRONE_PRESETS[0].nominalHoverRpm,
    DJI_DRONE_PRESETS[0].nominalHoverRpm,
    DJI_DRONE_PRESETS[0].nominalHoverRpm,
  ]);
  manualDroneRecording = signal(false);
  manualDroneRecordedSamples = signal<IRecordedDroneSample[]>([]);
  private manualDroneRecordingStartedAtMs = 0;
  private manualDroneLastSampleAtS = -Infinity;

  // Detector
  readonly detectorPresets = DETECTOR_PRESETS;
  selectedDetectorPresetId = signal('pf32');
  resolutionW = signal(32);
  resolutionH = signal(32);
  detectorFov = signal(50);
  detectorYaw = signal(0); // degrees
  detectorPitch = signal(0); // degrees
  pixelPitchUm = signal(50);
  fillFactor = signal(0.015);
  microlensGain = signal(13.3);
  frameDurationUs = signal(20);
  quantumEfficiency = signal(0.3);
  apertureDiameter = signal(0.025);
  systemEfficiency = signal(0.05);
  filterBandwidth = signal(10);
  darkCountRate = signal(100);
  deadTimeNs = signal(20);
  timingJitterNs = signal(0.2 / 2.355);
  irfFwhmPs = signal(200);
  maxCountRateCpsPerPixel = signal(20e6);
  timeResolutionPs = signal(256);
  tdcBitDepth = signal(13); // Default to 13 bits (8191)
  tdcMaxCount = signal((2 ** 13) - 1);

  // Environment & Laser
  readonly environmentPresets = ENVIRONMENT_PRESETS;
  selectedEnvironmentPresetId = signal('lab_dim');
  selectedEnvironmentPreset = computed(() => findEnvironmentPreset(this.selectedEnvironmentPresetId()));
  laserMode = signal<'Pulsed' | 'CW'>('Pulsed');
  solarIrradiance = signal(0.000068);
  atmosphericAttenuationEnabled = signal(true);
  atmosphericVisibilityKm = signal(50);
  laserWavelengthNm = signal(780);
  laserAveragePower = signal(1e-6);
  laserPulseWidthNs = signal(1);
  laserRepetitionFrequency = signal(1000000);
  transmitterDivergenceMrad = signal(1);

  // Simulation
  nFrames = signal(20000);

  // --- UI Computed Flags ---
  isPulsedMode = computed(() => this.laserMode() === 'Pulsed');
  laserCssColor = computed(() => `#${this.wavelengthToThreeColor(this.laserWavelengthNm()).getHexString()}`);
  telescopeFocalLength = computed(() => {
    const fovRad = THREE.MathUtils.degToRad(Math.max(0.1, this.detectorFov()));
    return this.apertureDiameter() / (2 * Math.tan(fovRad / 2));
  });
  resolvedDetectorSettings = computed(() => (
    this.selectedDetectorPresetId() === 'custom'
      ? null
      : resolveDetectorSettings(this.selectedDetectorPresetId(), this.laserWavelengthNm(), this.filterBandwidth(), this.tdcBitDepth())
  ));
  activeDronePosition = computed(() => {
    if (this.isManualDroneMode()) {
      return this.manualDronePosition();
    }
    return { x: this.initialPosX(), y: this.initialPosY(), z: this.initialPosZ() };
  });

  // --- Derived Laser Parameters (for UI display) ---
  laserSinglePulseEnergy = computed(() => {
    if (!this.isPulsedMode()) return 0;
    const freq = this.laserRepetitionFrequency();
    return freq > 0 ? this.laserAveragePower() / freq : 0;
  });

  laserSinglePulseEnergyMicroJoules = computed(() => {
    return this.laserSinglePulseEnergy() * 1e6;
  });

  distanceResolution = computed(() => {
    if (!this.isPulsedMode()) return 0;
    const c = 3e8; // speed of light
    const pulseWidthSeconds = this.laserPulseWidthNs() * 1e-9;
    return (c * pulseWidthSeconds) / 2;
  });

  laserPeakPower = computed(() => {
    if (!this.isPulsedMode()) return 0;
    const pulseWidthSeconds = this.laserPulseWidthNs() * 1e-9;
    if (pulseWidthSeconds === 0) {
      return 0;
    }
    return this.laserSinglePulseEnergy() / pulseWidthSeconds;
  });

  maxMeasurableDistance = computed(() => {
    const c = 3e8;
    const maxTofUnits = this.tdcMaxCount();
    const timeResSeconds = this.timeResolutionPs() * 1e-12;
    const maxTofSeconds = maxTofUnits * timeResSeconds;
    return (c * maxTofSeconds) / 2;
  });

  // --- Pile-up Analysis ---
  pileUpFactor = computed(() => {
    const result = this.simulationResult();
    return result ? result.maxIncidentPhotonsPerPixel : 0;
  });

  pileUpStatus = computed(() => {
    const factor = this.pileUpFactor();
    if (factor > 0.95) {
      return { textKey: 'pileupHigh', class: 'text-red-400' };
    }
    if (factor > 0.05) {
      return { textKey: 'pileupModerate', class: 'text-yellow-400' };
    }
    return { textKey: 'pileupLow', class: 'text-green-400' };
  });
  simulationBudget = computed(() => estimateSimulationBudget(
    { width: this.resolutionW(), height: this.resolutionH() },
    this.nFrames(),
  ));
  recommendedFrames = computed(() => recommendedFrameCount({ width: this.resolutionW(), height: this.resolutionH() }));
  simulationBudgetStatus = computed(() => {
    const budget = this.simulationBudget();
    if (budget.level === 'blocked') {
      return { textKey: 'runtimeBlocked', class: 'text-red-300 border-red-500/50 bg-red-950/40' };
    }
    if (budget.level === 'caution') {
      return { textKey: 'runtimeCaution', class: 'text-yellow-200 border-yellow-500/50 bg-yellow-950/40' };
    }
    return { textKey: 'runtimeSafe', class: 'text-emerald-200 border-emerald-500/40 bg-emerald-950/30' };
  });
  simulationBudgetMessage = computed(() => {
    const level = this.simulationBudget().level;
    if (level === 'blocked') return this.l.t('runtimeBlocked')();
    if (level === 'caution') return this.l.t('runtimeCaution')();
    return this.l.t('runtimeSafe')();
  });
  backendDownloadUrl = computed(() => {
    const job = this.backendJob();
    return job?.status === 'completed' && job.download_url ? this.backendSimulationService.downloadUrl(job.download_url) : '';
  });
  manualDroneRecordedDuration = computed(() => {
    const samples = this.manualDroneRecordedSamples();
    return samples.length > 0 ? samples[samples.length - 1].time : 0;
  });

  // Combine all params into a computed signal for the simulation service
  simulationParams = computed<ISimulationParams>(() => {
    const isManualDrone = this.isManualDroneMode();
    const recordedTrajectory = this.manualDroneRecordedSamples();
    const hasRecordedDroneTrajectory = this.targetType() === 'Drone' && recordedTrajectory.length >= 2;
    const activePosition = hasRecordedDroneTrajectory
      ? recordedTrajectory[0].pos
      : isManualDrone ? this.manualDronePosition() : { x: this.initialPosX(), y: this.initialPosY(), z: this.initialPosZ() };
    const activeWaypoints = hasRecordedDroneTrajectory
      ? recordedTrajectory.map((sample, index) => ({ id: index, pos: sample.pos }))
      : this.waypoints();
    const activePathSpeeds = hasRecordedDroneTrajectory
      ? this.pathSpeedsFromRecordedTrajectory(recordedTrajectory)
      : this.pathSpeeds();
    const activePathRpms = hasRecordedDroneTrajectory
      ? this.pathPropellerSpeedsFromRecordedTrajectory(recordedTrajectory, 0)
      : this.pathRotationSpeeds();
    const recordedPathRpms1 = hasRecordedDroneTrajectory ? this.pathPropellerSpeedsFromRecordedTrajectory(recordedTrajectory, 0) : this.pathRotationSpeeds1();
    const recordedPathRpms2 = hasRecordedDroneTrajectory ? this.pathPropellerSpeedsFromRecordedTrajectory(recordedTrajectory, 1) : this.pathRotationSpeeds2();
    const recordedPathRpms3 = hasRecordedDroneTrajectory ? this.pathPropellerSpeedsFromRecordedTrajectory(recordedTrajectory, 2) : this.pathRotationSpeeds3();
    const recordedPathRpms4 = hasRecordedDroneTrajectory ? this.pathPropellerSpeedsFromRecordedTrajectory(recordedTrajectory, 3) : this.pathRotationSpeeds4();
    const recordedAverageRpm = hasRecordedDroneTrajectory
      ? this.averageRecordedPropellerRpm(recordedTrajectory)
      : this.rotationSpeed();
    const effectiveFrameDurationUs = hasRecordedDroneTrajectory
      ? Math.max(1, recordedTrajectory[recordedTrajectory.length - 1].time * 1e6 / Math.max(this.nFrames(), 1))
      : this.frameDurationUs();
    const dronePreset = this.selectedDronePreset();
    const detectorPresetId = this.selectedDetectorPresetId();
    const detectorPreset = detectorPresetId === 'custom' ? undefined : getDetectorPreset(detectorPresetId);

    return {
    targetType: this.targetType(),
    initialPos: activePosition,
    initialVel: { x: this.initialVelX(), y: this.initialVelY(), z: this.initialVelZ() },
    ballMotionType: this.ballMotionType(),
    bladeMotionType: hasRecordedDroneTrajectory ? 'Path' : isManualDrone ? 'Manual' : this.bladeMotionType(),
    waypoints: activeWaypoints,
    pathSpeeds: activePathSpeeds,
    pathRotationSpeeds: activePathRpms,
    pathRotationSpeeds1: recordedPathRpms1,
    pathRotationSpeeds2: recordedPathRpms2,
    pathRotationSpeeds3: recordedPathRpms3,
    pathRotationSpeeds4: recordedPathRpms4,
    reflectivity: this.reflectivity(),
    propellerReflectivity: this.propellerReflectivity(),
    restitution: this.restitution(),
    rotationRadius: this.rotationRadius(),
    rotationSpeed: recordedAverageRpm,
    rotationKeyframes: this.rotationKeyframes(),
    rotationKeyframes1: this.rotationKeyframes1(),
    rotationKeyframes2: this.rotationKeyframes2(),
    rotationKeyframes3: this.rotationKeyframes3(),
    rotationKeyframes4: this.rotationKeyframes4(),
    bladePitch: this.bladePitch(),
    rotationCenter: { x: this.initialPosX(), z: this.initialPosZ() },
    uploadedImage: this.uploadedImage(),
    droneScale: this.droneScale(),
    dronePresetId: dronePreset.id,
    dronePreset,
    droneYawDeg: hasRecordedDroneTrajectory ? recordedTrajectory[0].yawDeg : isManualDrone ? this.manualDroneYaw() : 0,
    dronePitchDeg: hasRecordedDroneTrajectory ? recordedTrajectory[0].pitchDeg : isManualDrone ? this.manualDronePitch() : this.bladePitch(),
    droneRollDeg: hasRecordedDroneTrajectory ? recordedTrajectory[0].rollDeg : isManualDrone ? this.manualDroneRoll() : 0,
    manualControlEnabled: isManualDrone,
    spadPoseLocked: hasRecordedDroneTrajectory,
    recordedDroneTrajectory: hasRecordedDroneTrajectory ? recordedTrajectory : undefined,
    resolution: { width: this.resolutionW(), height: this.resolutionH() },
    detectorPresetId,
    detectorPreset,
    detectorFov: this.detectorFov(),
    detectorYaw: this.detectorYaw(),
    detectorPitch: this.detectorPitch(),
    pixelPitchUm: this.pixelPitchUm(),
    fillFactor: this.fillFactor(),
    microlensGain: this.microlensGain(),
    frameDurationUs: effectiveFrameDurationUs,
    quantumEfficiency: this.quantumEfficiency(),
    apertureDiameter: this.apertureDiameter(),
    systemEfficiency: this.systemEfficiency(),
    filterBandwidth: this.filterBandwidth(),
    darkCountRate: this.darkCountRate(),
    deadTimeNs: this.deadTimeNs(),
    timingJitterNs: this.timingJitterNs(),
    irfFwhmPs: this.irfFwhmPs(),
    maxCountRateCpsPerPixel: this.maxCountRateCpsPerPixel(),
    timeResolutionPs: this.timeResolutionPs(),
    tdcMaxCount: this.tdcMaxCount(),
    environmentPresetId: this.selectedEnvironmentPresetId(),
    solarIrradiance: this.solarIrradiance(),
    atmosphericAttenuationEnabled: this.atmosphericAttenuationEnabled(),
    atmosphericVisibilityKm: this.atmosphericVisibilityKm(),
    laserMode: this.laserMode(),
    laserPulseEnergy: this.laserSinglePulseEnergy(),
    laserAveragePower: this.laserAveragePower(),
    laserRepetitionFrequency: this.laserRepetitionFrequency(),
    laserPulseWidthNs: this.laserPulseWidthNs(),
    laserWavelengthNm: this.laserWavelengthNm(),
    transmitterDivergenceMrad: this.transmitterDivergenceMrad(),
    nFrames: this.nFrames(),
    cameraHeight: 1.0,
    };
  });

  // --- 3D Scene ---
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private detectorCamera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private ball!: THREE.Mesh;
  private blade!: THREE.Mesh;
  private bladePivot!: THREE.Group;
  private drone!: THREE.Group;
  private dronePropellers: THREE.Object3D[] = [];
  private ambientLight!: THREE.AmbientLight;
  private skyLight!: THREE.HemisphereLight;
  private sun!: THREE.DirectionalLight;
  private skyDome!: THREE.Mesh;
  private atmosphereShell!: THREE.Mesh;
  private sunDisc!: THREE.Sprite;
  private sunHalo!: THREE.Sprite;
  private skyTexture!: THREE.CanvasTexture;
  private groundTexture!: THREE.CanvasTexture;
  private atmosphereTexture!: THREE.CanvasTexture;
  private sunDiscTexture!: THREE.CanvasTexture;
  private sunHaloTexture!: THREE.CanvasTexture;
  private laserSpotlight!: THREE.SpotLight;
  private laserBeam!: THREE.Mesh;
  private detectorRig!: THREE.Group;
  private frameId: number | null = null;
  private resizeObserver!: ResizeObserver;
  private pathLine: THREE.Line | null = null;
  private telescopeTube!: THREE.Mesh;
  private apertureGlass!: THREE.Mesh;
  private pressedKeys = new Set<string>();
  private isDroneMouseLook = false;
  private lastRenderTime = performance.now();

  // --- Resize handling state ---
  private viewportSize = { width: 0, height: 0 };
  private resizePending = false;

  // --- Animation State ---
  private simulationTrajectory: { x: number, y: number, z: number }[] = [];
  private trajectoryLine: THREE.Line | null = null;
  private animationStartTime = 0;

  constructor() {
    this.refreshBackendCapabilities();

    effect(() => {
      this.solarIrradiance();
      this.atmosphericVisibilityKm();
      this.selectedEnvironmentPresetId();
      this.updateSolarVisuals();
    });

    effect(() => {
      const avgPower = this.laserAveragePower(); // Use average power for brightness
      if (this.laserSpotlight) {
        this.laserSpotlight.intensity = avgPower * 1500;
      }
      if (this.laserBeam) {
        const maxPower = 3;
        (this.laserBeam.material as THREE.Material).opacity = (avgPower / maxPower) * 0.6;
      }
    });

    effect(() => {
      const color = this.wavelengthToThreeColor(this.laserWavelengthNm());
      if (this.laserSpotlight) {
        this.laserSpotlight.color.copy(color);
      }
      if (this.laserBeam) {
        (this.laserBeam.material as THREE.MeshBasicMaterial).color.copy(color);
      }
    });

    effect(() => {
      const resolved = this.resolvedDetectorSettings();
      if (!resolved) return;
      const environment = this.selectedEnvironmentPreset();
      this.resolutionW.set(resolved.resolution.width);
      this.resolutionH.set(resolved.resolution.height);
      this.pixelPitchUm.set(resolved.preset.pixelPitchUm);
      this.fillFactor.set(resolved.preset.fillFactor);
      this.microlensGain.set(resolved.preset.microlensGain);
      this.quantumEfficiency.set(Number(resolved.quantumEfficiency.toFixed(4)));
      this.solarIrradiance.set(Number((resolved.solarIrradiance * environment.solarScale).toFixed(6)));
      this.systemEfficiency.set(Number(resolved.systemEfficiency.toFixed(4)));
      this.darkCountRate.set(resolved.darkCountRateCps);
      this.deadTimeNs.set(resolved.deadTimeNs);
      this.timingJitterNs.set(resolved.timingJitterNs);
      this.irfFwhmPs.set(resolved.irfFwhmPs);
      this.maxCountRateCpsPerPixel.set(resolved.maxCountRateCpsPerPixel);
      this.timeResolutionPs.set(Number(resolved.timeResolutionPs.toFixed(3)));
      this.detectorFov.set(Number(resolved.detectorFovDeg.toFixed(6)));
      this.tdcMaxCount.set(resolved.tdcMaxCount);
    });

    effect(() => {
      this.apertureDiameter();
      this.detectorFov();
      if (this.telescopeTube && this.apertureGlass) {
        this.updateTelescopeVisual();
      }
    });

    effect(() => {
      this.selectedDronePresetId();
      if (this.drone) {
        this.rebuildDroneModel();
      }
    });

    effect(() => {
      const environment = this.selectedEnvironmentPreset();
      this.atmosphericVisibilityKm.set(environment.visibilityKm);
    });

    effect(() => {
      this.initialPosX(); this.initialPosY(); this.initialPosZ();
      this.initialVelX(); this.initialVelY(); this.initialVelZ();
      this.restitution(); this.frameDurationUs(); this.nFrames();
      this.rotationSpeed(); this.waypoints(); this.pathSpeeds(); this.pathRotationSpeeds();
      this.rotationKeyframes();

      if (!this.isPreviewing()) {
        this.clearTrajectoryLine();
      }
    });

    // Effect for toggling target visibility and updating blade mesh
    effect(() => {
      const type = this.targetType();
      const image = this.uploadedImage();
      const bladeLength = this.rotationRadius(); // Re-trigger on length change

      if (this.ball && this.bladePivot && this.drone) {
        this.ball.visible = type === 'Ball';
        this.bladePivot.visible = type === 'Blade';
        this.drone.visible = type === 'Drone';

        const isRotationalTarget = type === 'Blade' || type === 'Drone';

        if (isRotationalTarget) {
          if (type === 'Blade') this.updateBladeMesh(image, bladeLength);
          // Initialize default path if not present or incomplete for drone
          if (this.waypoints().length < 2 || (type === 'Drone' && this.pathRotationSpeeds1().length === 0)) {
            if (this.waypoints().length < 2) {
              const wps = [
                { id: Date.now(), pos: { x: 1, y: 1, z: 3 } },
                { id: Date.now() + 1, pos: { x: -1, y: 3, z: 6 } }
              ];
              this.waypoints.set(wps);
            }

            // Fill rotation speeds if missing (e.g. switched from Blade to Drone)
            const segmentCount = Math.max(1, this.waypoints().length - 1);
            const defaultRpm = 12000;
            const fillArray = (arr: number[]) => arr.length < segmentCount ? new Array(segmentCount).fill(defaultRpm) : arr;

            if (this.pathSpeeds().length < segmentCount) this.pathSpeeds.set(new Array(segmentCount).fill(2.0));
            if (this.pathDurations().length < segmentCount) {
              // Recalculate based on default speed if needed, for simplicity just fill
              this.pathDurations.set(new Array(segmentCount).fill(1.0));
            }

            this.pathRotationSpeeds.set(fillArray(this.pathRotationSpeeds()));
            this.pathRotationSpeeds1.set(fillArray(this.pathRotationSpeeds1()));
            this.pathRotationSpeeds2.set(fillArray(this.pathRotationSpeeds2()));
            this.pathRotationSpeeds3.set(fillArray(this.pathRotationSpeeds3()));
            this.pathRotationSpeeds4.set(fillArray(this.pathRotationSpeeds4()));
          }

          // Initialize default rotation keyframes if not present
          const defaultKeyframes = [
            { id: Date.now(), time: 0, rpm: 18000 },
            { id: Date.now() + 1, time: 2, rpm: 18000 },
          ];
          if (this.rotationKeyframes().length < 2) this.rotationKeyframes.set(defaultKeyframes);
          if (this.rotationKeyframes1().length < 2) this.rotationKeyframes1.set(defaultKeyframes);
          if (this.rotationKeyframes2().length < 2) this.rotationKeyframes2.set(defaultKeyframes);
          if (this.rotationKeyframes3().length < 2) this.rotationKeyframes3.set(defaultKeyframes);
          if (this.rotationKeyframes4().length < 2) this.rotationKeyframes4.set(defaultKeyframes);
        }
      }
      this.clearTrajectoryLine();
    });

    effect(() => {
      const scale = this.droneScale();
      if (this.drone) {
        this.drone.scale.setScalar(scale);
      }
    });

    effect(() => {
      const fov = Math.max(0.01, Math.min(this.detectorFov(), 179));
      const fovInRad = fov * Math.PI / 180;

      if (this.detectorCamera) {
        this.detectorCamera.fov = fov;
        this.detectorCamera.updateProjectionMatrix();
      }

      if (this.laserSpotlight) {
        this.laserSpotlight.angle = fovInRad / 2;
      }

      if (this.laserBeam) {
        const beamDistance = 20;
        const beamEndRadius = beamDistance * Math.tan(fovInRad / 2);

        const newGeo = new THREE.CylinderGeometry(beamEndRadius, 0.01, beamDistance, 32, 1, true);
        newGeo.rotateX(Math.PI / 2);
        newGeo.translate(0, 0, beamDistance / 2);

        this.laserBeam.geometry.dispose();
        this.laserBeam.geometry = newGeo;
      }

      if (this.telescopeTube && this.apertureGlass) {
        this.updateTelescopeVisual();
      }
    });

    // Effect for rotating the detector rig
    effect(() => {
      const yaw = this.detectorYaw();
      const pitch = this.detectorPitch();

      if (this.detectorRig) {
        const yawRad = THREE.MathUtils.degToRad(yaw);
        const pitchRad = THREE.MathUtils.degToRad(pitch);
        this.detectorRig.rotation.set(pitchRad, yawRad, 0, 'YXZ');
      }
    });

    // Effect to draw/update the blade's path line
    effect(() => {
      const waypoints = this.waypoints();
      const points = waypoints.map(wp => new THREE.Vector3(wp.pos.x, wp.pos.y, wp.pos.z));

      const shouldShowPath = (this.targetType() === 'Blade' || this.targetType() === 'Drone') && this.bladeMotionType() === 'Path' && points.length > 1;

      if (shouldShowPath) {
        if (this.pathLine) {
          this.pathLine.geometry.setFromPoints(points);
          this.pathLine.computeLineDistances(); // Required for dashed lines
          this.pathLine.geometry.attributes.position.needsUpdate = true;
        } else {
          const geometry = new THREE.BufferGeometry().setFromPoints(points);
          const material = new THREE.LineDashedMaterial({ color: 0xffa500, dashSize: 0.2, gapSize: 0.1, linewidth: 2 });
          this.pathLine = new THREE.Line(geometry, material);
          this.pathLine.computeLineDistances();
          this.scene?.add(this.pathLine);
        }
        if (this.pathLine) this.pathLine.visible = true;
      } else {
        if (this.pathLine) this.pathLine.visible = false;
      }
    });
  }

  ngAfterViewInit(): void {
    this.initThreeJs();

    // Set initial size before observing
    const { clientWidth, clientHeight } = this.canvasContainer.nativeElement;
    this.viewportSize.width = clientWidth;
    this.viewportSize.height = clientHeight;

    this.setupResizeObserver();
    this.setupManualDroneControls();
    this.startRenderingLoop();
  }

  ngOnDestroy(): void {
    if (this.frameId != null) {
      cancelAnimationFrame(this.frameId);
    }
    this.resizeObserver.disconnect();
    if (this.renderer) {
      this.renderer.dispose();
    }
    this.removeManualDroneControls();
  }

  // --- Resizable Panel Logic ---
  startResize(event: MouseEvent) {
    event.preventDefault(); // Prevent text selection
    this.isResizing = true;
    this.startX = event.clientX;
    this.startWidth = this.panelWidth();
    document.addEventListener('mousemove', this.onMouseMove);
    document.addEventListener('mouseup', this.stopResize);
    document.body.style.cursor = 'col-resize';
  }

  private onMouseMove = (event: MouseEvent) => {
    if (!this.isResizing) return;
    const dx = event.clientX - this.startX;
    // Limits: Min 250px, Max 600px
    const newWidth = Math.max(250, Math.min(600, this.startWidth + dx));
    this.panelWidth.set(newWidth);
  }

  private stopResize = () => {
    this.isResizing = false;
    document.removeEventListener('mousemove', this.onMouseMove);
    document.removeEventListener('mouseup', this.stopResize);
    document.body.style.cursor = '';
  }

  private setupResizeObserver(): void {
    this.resizeObserver = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        // Instead of resizing here, just record the new size and flag it.
        this.viewportSize.width = entry.contentRect.width;
        this.viewportSize.height = entry.contentRect.height;
        this.resizePending = true;
      }
    });
    this.resizeObserver.observe(this.canvasContainer.nativeElement);
  }

  onFileSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files && input.files[0]) {
      const file = input.files[0];
      const reader = new FileReader();
      reader.onload = (e) => {
        const imageUrl = e.target?.result as string;
        this.uploadedImageUrl.set(imageUrl);
        this.binarizedImageUrl.set(null); // Clear previous

        const image = new Image();
        image.onload = () => {
          this.uploadedImage.set(image);
          this.generateBinarizedPreview(image);
        };
        image.src = imageUrl;
      };
      reader.readAsDataURL(file);
    }
  }

  private generateBinarizedPreview(image: HTMLImageElement): void {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);

    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    const { data } = imageData;
    const outputImageData = ctx.createImageData(image.width, image.height);

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
      const isPropeller = a > 128 && luminance < 128;

      if (isPropeller) {
        outputImageData.data[i] = 0;
        outputImageData.data[i + 1] = 0;
        outputImageData.data[i + 2] = 0;
        outputImageData.data[i + 3] = 255;
      } else {
        outputImageData.data[i] = 255;
        outputImageData.data[i + 1] = 255;
        outputImageData.data[i + 2] = 255;
        outputImageData.data[i + 3] = 255;
      }
    }
    ctx.putImageData(outputImageData, 0, 0);
    this.binarizedImageUrl.set(canvas.toDataURL());
  }

  onNumberInput(signal: WritableSignal<number>, event: Event) {
    const value = (event.target as HTMLInputElement).value;
    const parsedValue = parseFloat(value);
    if (!isNaN(parsedValue)) {
      signal.set(parsedValue);
    }
  }

  onBitDepthInput(event: Event) {
    const value = (event.target as HTMLInputElement).value;
    const newBitDepth = parseInt(value, 10);
    if (!isNaN(newBitDepth)) {
      this.tdcBitDepth.set(newBitDepth);
      this.tdcMaxCount.set((2 ** newBitDepth) - 1);
    }
  }

  onMaxCountInput(event: Event) {
    const value = (event.target as HTMLInputElement).value;
    const newMaxCount = parseInt(value, 10);
    if (!isNaN(newMaxCount) && newMaxCount > 0) {
      this.tdcMaxCount.set(newMaxCount);
      const requiredBitDepth = Math.ceil(Math.log2(newMaxCount + 1));
      this.tdcBitDepth.set(requiredBitDepth);
    }
  }

  formatPeakPower(power: number): string {
    if (power < 1000) {
      return `${power.toFixed(2)} W`;
    } else if (power < 1_000_000) {
      return `${(power / 1000).toFixed(2)} kW`;
    } else {
      return `${(power / 1_000_000).toFixed(2)} MW`;
    }
  }

  formatFrequency(hz: number): string {
    if (hz >= 1_000_000) {
      return `${(hz / 1_000_000).toFixed(2)} MHz`;
    }
    if (hz >= 1000) {
      return `${(hz / 1000).toFixed(1)} kHz`;
    }
    return `${hz} Hz`;
  }

  onDronePresetChange(event: Event): void {
    const preset = findDronePreset((event.target as HTMLSelectElement).value);
    this.selectedDronePresetId.set(preset.id);
    this.droneScale.set(preset.visualScale);
    this.reflectivity.set(preset.bodyReflectivity);
    this.propellerReflectivity.set(preset.propellerReflectivity);
    this.manualDroneSpeed.set(Math.min(8, Math.max(1, preset.maxHorizontalSpeedMps * 0.25)));
    this.rotationSpeed.set(preset.nominalHoverRpm);
    this.manualDronePropellerRpms.set([
      preset.nominalHoverRpm,
      preset.nominalHoverRpm,
      preset.nominalHoverRpm,
      preset.nominalHoverRpm,
    ]);

    const keyframes = [
      { id: Date.now(), time: 0, rpm: preset.nominalHoverRpm },
      { id: Date.now() + 1, time: 2, rpm: preset.nominalHoverRpm },
    ];
    this.rotationKeyframes1.set(keyframes);
    this.rotationKeyframes2.set(keyframes);
    this.rotationKeyframes3.set(keyframes);
    this.rotationKeyframes4.set(keyframes);
    this.pathRotationSpeeds1.update(s => s.map(() => preset.nominalHoverRpm));
    this.pathRotationSpeeds2.update(s => s.map(() => preset.nominalHoverRpm));
    this.pathRotationSpeeds3.update(s => s.map(() => preset.nominalHoverRpm));
    this.pathRotationSpeeds4.update(s => s.map(() => preset.nominalHoverRpm));
  }

  onDetectorPresetChange(event: Event): void {
    this.selectedDetectorPresetId.set((event.target as HTMLSelectElement).value);
  }

  setBladeMotionType(mode: 'Fixed' | 'Path'): void {
    this.bladeMotionType.set(mode);
    if (this.targetType() === 'Drone') {
      this.manualDroneControl.set(false);
    }
  }

  setTargetType(type: 'Ball' | 'Blade' | 'Drone'): void {
    this.targetType.set(type);
    if (type !== 'Drone') {
      this.manualDroneControl.set(false);
      if (this.bladeMotionType() === 'Manual') {
        this.bladeMotionType.set('Fixed');
      }
    }
  }

  setDroneMotionType(mode: 'Fixed' | 'Path' | 'Manual'): void {
    this.bladeMotionType.set(mode);
    this.manualDroneControl.set(mode === 'Manual');
  }

  private isManualDroneMode(): boolean {
    return this.targetType() === 'Drone' && this.bladeMotionType() === 'Manual';
  }

  captureManualDronePose(): void {
    const pos = this.manualDronePosition();
    this.initialPosX.set(Number(pos.x.toFixed(2)));
    this.initialPosY.set(Number(pos.y.toFixed(2)));
    this.initialPosZ.set(Number(pos.z.toFixed(2)));
  }

  startManualDroneRecording(): void {
    this.targetType.set('Drone');
    this.setDroneMotionType('Manual');
    this.manualDroneRecordedSamples.set([]);
    this.manualDroneRecordingStartedAtMs = performance.now();
    this.manualDroneLastSampleAtS = -Infinity;
    this.manualDroneRecording.set(true);
    this.recordManualDroneSample(true);
  }

  stopManualDroneRecording(): void {
    this.recordManualDroneSample(true);
    this.manualDroneRecording.set(false);
  }

  clearManualDroneRecording(): void {
    this.manualDroneRecording.set(false);
    this.manualDroneRecordedSamples.set([]);
  }

  private currentManualDroneSample(timeS: number): IRecordedDroneSample {
    const pos = this.manualDronePosition();
    return {
      time: Number(Math.max(0, timeS).toFixed(3)),
      pos: {
        x: Number(pos.x.toFixed(3)),
        y: Number(pos.y.toFixed(3)),
        z: Number(pos.z.toFixed(3)),
      },
      yawDeg: Number(this.manualDroneYaw().toFixed(3)),
      pitchDeg: Number(this.manualDronePitch().toFixed(3)),
      rollDeg: Number(this.manualDroneRoll().toFixed(3)),
      propellerRpms: this.manualDronePropellerRpms().map(rpm => Number(rpm.toFixed(1))) as [number, number, number, number],
    };
  }

  private recordManualDroneSample(force = false): void {
    if (!this.manualDroneRecording()) return;
    const timeS = (performance.now() - this.manualDroneRecordingStartedAtMs) / 1000;
    if (!force && timeS - this.manualDroneLastSampleAtS < 0.05) return;
    const sample = this.currentManualDroneSample(timeS);
    this.manualDroneLastSampleAtS = timeS;
    this.manualDroneRecordedSamples.update(samples => {
      const previous = samples[samples.length - 1];
      if (
        previous
        && Math.abs(previous.time - sample.time) < 1e-6
        && previous.pos.x === sample.pos.x
        && previous.pos.y === sample.pos.y
        && previous.pos.z === sample.pos.z
      ) {
        return samples;
      }
      return [...samples, sample].slice(-600);
    });
  }

  private pathSpeedsFromRecordedTrajectory(samples: IRecordedDroneSample[]): number[] {
    const speeds: number[] = [];
    for (let i = 0; i < samples.length - 1; i++) {
      const start = samples[i];
      const end = samples[i + 1];
      const dt = Math.max(1e-3, end.time - start.time);
      const dx = end.pos.x - start.pos.x;
      const dy = end.pos.y - start.pos.y;
      const dz = end.pos.z - start.pos.z;
      speeds.push(Math.max(0.01, Math.sqrt(dx * dx + dy * dy + dz * dz) / dt));
    }
    return speeds;
  }

  private pathPropellerSpeedsFromRecordedTrajectory(samples: IRecordedDroneSample[], propellerIndex: number): number[] {
    const speeds: number[] = [];
    for (let i = 0; i < samples.length - 1; i++) {
      const start = samples[i].propellerRpms[propellerIndex] ?? this.rotationSpeed();
      const end = samples[i + 1].propellerRpms[propellerIndex] ?? start;
      speeds.push(Math.max(0, (start + end) / 2));
    }
    return speeds;
  }

  private averageRecordedPropellerRpm(samples: IRecordedDroneSample[]): number {
    let total = 0;
    let count = 0;
    for (const sample of samples) {
      for (const rpm of sample.propellerRpms) {
        total += rpm;
        count++;
      }
    }
    return count > 0 ? Math.max(1, total / count) : this.rotationSpeed();
  }

  private wavelengthToThreeColor(wavelengthNm: number): THREE.Color {
    const wl = Math.max(380, Math.min(780, wavelengthNm));
    let r = 0;
    let g = 0;
    let b = 0;

    if (wl < 440) {
      r = -(wl - 440) / (440 - 380);
      b = 1;
    } else if (wl < 490) {
      g = (wl - 440) / (490 - 440);
      b = 1;
    } else if (wl < 510) {
      g = 1;
      b = -(wl - 510) / (510 - 490);
    } else if (wl < 580) {
      r = (wl - 510) / (580 - 510);
      g = 1;
    } else if (wl < 645) {
      r = 1;
      g = -(wl - 645) / (645 - 580);
    } else {
      r = 1;
    }

    const edgeFactor = wl < 420 ? 0.3 + 0.7 * (wl - 380) / 40 : wl > 700 ? 0.3 + 0.7 * (780 - wl) / 80 : 1;
    const gamma = 0.8;
    return new THREE.Color(
      Math.pow(Math.max(0, r * edgeFactor), gamma),
      Math.pow(Math.max(0, g * edgeFactor), gamma),
      Math.pow(Math.max(0, b * edgeFactor), gamma),
    );
  }

  private setupManualDroneControls(): void {
    window.addEventListener('keydown', this.handleKeyDown);
    window.addEventListener('keyup', this.handleKeyUp);
    const canvas = this.rendererCanvas.nativeElement;
    canvas.addEventListener('mousedown', this.handleDroneMouseDown);
    canvas.addEventListener('mouseup', this.handleDroneMouseUp);
    canvas.addEventListener('mouseleave', this.handleDroneMouseUp);
    canvas.addEventListener('mousemove', this.handleDroneMouseMove);
    canvas.addEventListener('contextmenu', this.preventDroneContextMenu);
  }

  private removeManualDroneControls(): void {
    window.removeEventListener('keydown', this.handleKeyDown);
    window.removeEventListener('keyup', this.handleKeyUp);
    const canvas = this.rendererCanvas?.nativeElement;
    if (!canvas) return;
    canvas.removeEventListener('mousedown', this.handleDroneMouseDown);
    canvas.removeEventListener('mouseup', this.handleDroneMouseUp);
    canvas.removeEventListener('mouseleave', this.handleDroneMouseUp);
    canvas.removeEventListener('mousemove', this.handleDroneMouseMove);
    canvas.removeEventListener('contextmenu', this.preventDroneContextMenu);
  }

  private shouldCaptureDroneInput(): boolean {
    return this.isManualDroneMode() && !this.isSimulating() && !this.isBackendSimulating();
  }

  private handleKeyDown = (event: KeyboardEvent) => {
    if (!this.shouldCaptureDroneInput()) return;
    const controlledKeys = ['KeyW', 'KeyA', 'KeyS', 'KeyD', 'KeyQ', 'KeyE', 'KeyR', 'KeyF', 'ShiftLeft', 'ShiftRight'];
    if (controlledKeys.includes(event.code)) {
      event.preventDefault();
      this.pressedKeys.add(event.code);
    }
  };

  private handleKeyUp = (event: KeyboardEvent) => {
    this.pressedKeys.delete(event.code);
  };

  private handleDroneMouseDown = (event: MouseEvent) => {
    if (!this.shouldCaptureDroneInput() || event.button !== 2) return;
    event.preventDefault();
    this.isDroneMouseLook = true;
    this.controls.enabled = false;
  };

  private handleDroneMouseUp = () => {
    this.isDroneMouseLook = false;
    if (this.controls) {
      this.controls.enabled = true;
    }
  };

  private handleDroneMouseMove = (event: MouseEvent) => {
    if (!this.isDroneMouseLook || !this.shouldCaptureDroneInput()) return;
    this.manualDroneYaw.update(yaw => yaw - event.movementX * 0.12);
  };

  private preventDroneContextMenu = (event: MouseEvent) => {
    if (this.shouldCaptureDroneInput()) {
      event.preventDefault();
    }
  };

  private updateManualDroneFlight(deltaSeconds: number): void {
    if (!this.shouldCaptureDroneInput()) return;

    const boost = this.pressedKeys.has('ShiftLeft') || this.pressedKeys.has('ShiftRight') ? 2 : 1;
    const speed = this.manualDroneSpeed() * boost;
    const yawRate = 75;
    const yawInput = (this.pressedKeys.has('KeyQ') ? 1 : 0) - (this.pressedKeys.has('KeyE') ? 1 : 0);
    if (yawInput !== 0) {
      this.manualDroneYaw.update(yaw => yaw + yawInput * yawRate * deltaSeconds);
    }

    const forwardInput = (this.pressedKeys.has('KeyW') ? 1 : 0) - (this.pressedKeys.has('KeyS') ? 1 : 0);
    const lateralCommand = (this.pressedKeys.has('KeyD') ? 1 : 0) - (this.pressedKeys.has('KeyA') ? 1 : 0);
    const rightMotionInput = -lateralCommand;
    const upInput = (this.pressedKeys.has('KeyR') ? 1 : 0) - (this.pressedKeys.has('KeyF') ? 1 : 0);
    const yawRad = THREE.MathUtils.degToRad(this.manualDroneYaw());
    const forward = new THREE.Vector3(Math.sin(yawRad), 0, Math.cos(yawRad));
    const right = new THREE.Vector3(Math.cos(yawRad), 0, -Math.sin(yawRad));
    const displacement = forward.multiplyScalar(forwardInput).add(right.multiplyScalar(rightMotionInput));
    if (displacement.lengthSq() > 1e-9) {
      displacement.normalize().multiplyScalar(speed * deltaSeconds);
    }

    this.manualDronePosition.update(pos => ({
      x: pos.x + displacement.x,
      y: Math.max(0.08, pos.y + upInput * speed * 0.6 * deltaSeconds),
      z: pos.z + displacement.z,
    }));
    const attitudeAlpha = 1 - Math.exp(-deltaSeconds * 7);
    const targetPitch = THREE.MathUtils.clamp(forwardInput * 13, -18, 18);
    const targetRoll = THREE.MathUtils.clamp(lateralCommand * 14, -20, 20);
    this.manualDronePitch.update(pitch => THREE.MathUtils.lerp(pitch, targetPitch, attitudeAlpha));
    this.manualDroneRoll.update(roll => THREE.MathUtils.lerp(roll, targetRoll, attitudeAlpha));

    const preset = this.selectedDronePreset();
    const hoverRpm = preset.nominalHoverRpm;
    const collectiveMix = upInput * hoverRpm * 0.12;
    const pitchMix = forwardInput * hoverRpm * 0.08;
    const rollMix = lateralCommand * hoverRpm * 0.08;
    const yawMix = yawInput * hoverRpm * 0.055;
    const translationLoad = Math.min(1, Math.abs(forwardInput) + Math.abs(lateralCommand)) * hoverRpm * 0.025;
    const minRpm = hoverRpm * 0.62;
    const maxRpm = hoverRpm * 1.42;
    // 电机顺序：M1 前右、M2 前左、M3 后右、M4 后左。
    // 前进时后桨增速形成机头下压；横移时外侧低、内侧高；偏航按对角桨组差速。
    const targetRpms: [number, number, number, number] = [
      THREE.MathUtils.clamp(hoverRpm + collectiveMix - pitchMix - rollMix + yawMix + translationLoad, minRpm, maxRpm),
      THREE.MathUtils.clamp(hoverRpm + collectiveMix - pitchMix + rollMix - yawMix + translationLoad, minRpm, maxRpm),
      THREE.MathUtils.clamp(hoverRpm + collectiveMix + pitchMix - rollMix - yawMix + translationLoad, minRpm, maxRpm),
      THREE.MathUtils.clamp(hoverRpm + collectiveMix + pitchMix + rollMix + yawMix + translationLoad, minRpm, maxRpm),
    ];
    const rotorAlpha = 1 - Math.exp(-deltaSeconds * 10);
    this.manualDronePropellerRpms.update(current => current.map((rpm, index) => (
      THREE.MathUtils.lerp(rpm, targetRpms[index], rotorAlpha)
    )) as [number, number, number, number]);
    this.recordManualDroneSample(false);
  }

  // --- Path Planning UI Methods ---
  private getSegmentDistance(index: number): number {
    const wps = this.waypoints();
    if (index < 0 || index >= wps.length - 1) {
      return 0;
    }
    const start = wps[index].pos;
    const end = wps[index + 1].pos;
    const dx = end.x - start.x;
    const dy = end.y - start.y;
    const dz = end.z - start.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  addWaypoint(): void {
    this.waypoints.update(wps => {
      const lastPos = wps.length > 0 ? wps[wps.length - 1].pos : { x: 0, y: 2, z: 0 };
      const newWaypoint = { id: Date.now(), pos: { ...lastPos } };
      return [...wps, newWaypoint];
    });
    this.pathSpeeds.update(speeds => {
      const lastSpeed = speeds.length > 0 ? speeds[speeds.length - 1] : 1.0;
      return [...speeds, lastSpeed];
    });
    this.pathDurations.update(durations => [...durations, 0]);

    const lastRpm = 1200;
    this.pathRotationSpeeds.update(s => [...s, lastRpm]);
    this.pathRotationSpeeds1.update(s => [...s, lastRpm]);
    this.pathRotationSpeeds2.update(s => [...s, lastRpm]);
    this.pathRotationSpeeds3.update(s => [...s, lastRpm]);
    this.pathRotationSpeeds4.update(s => [...s, lastRpm]);
  }

  removeWaypoint(index: number): void {
    this.waypoints.update(wps => wps.filter((_, i) => i !== index));
    const speedIndexToRemove = Math.max(0, index - 1);
    this.pathSpeeds.update(s => s.filter((_, i) => i !== speedIndexToRemove));
    this.pathDurations.update(s => s.filter((_, i) => i !== speedIndexToRemove));
    this.pathRotationSpeeds.update(s => s.filter((_, i) => i !== speedIndexToRemove));
    this.pathRotationSpeeds1.update(s => s.filter((_, i) => i !== speedIndexToRemove));
    this.pathRotationSpeeds2.update(s => s.filter((_, i) => i !== speedIndexToRemove));
    this.pathRotationSpeeds3.update(s => s.filter((_, i) => i !== speedIndexToRemove));
    this.pathRotationSpeeds4.update(s => s.filter((_, i) => i !== speedIndexToRemove));

    // Recalculate the duration of the new merged segment
    this.pathDurations.update(durations => {
      const newDurations = [...durations];
      const mergedSegmentIndex = speedIndexToRemove;

      if (mergedSegmentIndex < this.pathSpeeds().length) {
        const distance = this.getSegmentDistance(mergedSegmentIndex);
        const speed = this.pathSpeeds()[mergedSegmentIndex];
        if (speed > 0) {
          newDurations[mergedSegmentIndex] = distance / speed;
        } else {
          newDurations[mergedSegmentIndex] = 0;
        }
      }
      return newDurations;
    });
  }

  updateWaypointPos(index: number, axis: 'x' | 'y' | 'z', event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(value)) {
      this.waypoints.update(wps => {
        const newWps = [...wps];
        newWps[index] = { ...newWps[index], pos: { ...newWps[index].pos, [axis]: value } };
        return newWps;
      });

      this.pathDurations.update(durations => {
        const newDurations = [...durations];
        if (index > 0) {
          const prevSegmentIndex = index - 1;
          const distance = this.getSegmentDistance(prevSegmentIndex);
          const speed = this.pathSpeeds()[prevSegmentIndex];
          newDurations[prevSegmentIndex] = speed > 0 ? distance / speed : 0;
        }
        if (index < this.waypoints().length - 1) {
          const currentSegmentIndex = index;
          const distance = this.getSegmentDistance(currentSegmentIndex);
          const speed = this.pathSpeeds()[currentSegmentIndex];
          newDurations[currentSegmentIndex] = speed > 0 ? distance / speed : 0;
        }
        return newDurations;
      });
    }
  }

  updatePathSpeed(index: number, event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(value) && value > 0) {
      this.pathSpeeds.update(speeds => {
        const newSpeeds = [...speeds];
        newSpeeds[index] = value;
        return newSpeeds;
      });

      const distance = this.getSegmentDistance(index);
      this.pathDurations.update(durations => {
        const newDurations = [...durations];
        newDurations[index] = distance / value;
        return newDurations;
      });
    }
  }

  updatePathDuration(index: number, event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(value) && value > 0) {
      this.pathDurations.update(durations => {
        const newDurations = [...durations];
        newDurations[index] = value;
        return newDurations;
      });

      const distance = this.getSegmentDistance(index);
      this.pathSpeeds.update(speeds => {
        const newSpeeds = [...speeds];
        newSpeeds[index] = distance / value;
        return newSpeeds;
      });
    }
  }

  updatePathRotationSpeed(propIndex: number, segmentIndex: number, event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(value) && value >= 0) {
      const signals = [this.pathRotationSpeeds, this.pathRotationSpeeds1, this.pathRotationSpeeds2, this.pathRotationSpeeds3, this.pathRotationSpeeds4];
      const signalToUpdate = signals[propIndex];
      signalToUpdate.update(speeds => {
        const newSpeeds = [...speeds];
        newSpeeds[segmentIndex] = value;
        return newSpeeds;
      });
    }
  }

  // --- Rotation Keyframe UI Methods ---
  private getRotationKeyframesForProp(propIndex: number): WritableSignal<IRotationKeyframe[]> {
    const signals = [this.rotationKeyframes, this.rotationKeyframes1, this.rotationKeyframes2, this.rotationKeyframes3, this.rotationKeyframes4];
    return signals[propIndex];
  }

  addRotationKeyframe(propIndex: number): void {
    const signal = this.getRotationKeyframesForProp(propIndex);
    signal.update(kfs => {
      const lastKf = kfs.length > 0 ? kfs[kfs.length - 1] : { time: 0, rpm: 1200 };
      const newKf = { id: Date.now(), time: lastKf.time + 1, rpm: lastKf.rpm };
      return [...kfs, newKf];
    });
  }

  removeRotationKeyframe(propIndex: number, kfIndex: number): void {
    const signal = this.getRotationKeyframesForProp(propIndex);
    signal.update(kfs => kfs.filter((_, i) => i !== kfIndex));
  }

  updateRotationKeyframeTime(propIndex: number, kfIndex: number, event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (isNaN(value)) return;

    const signal = this.getRotationKeyframesForProp(propIndex);
    signal.update(kfs => {
      const newKfs = [...kfs];
      newKfs[kfIndex] = { ...newKfs[kfIndex], time: value };
      return newKfs.sort((a, b) => a.time - b.time);
    });
  }

  updateRotationKeyframeRpm(propIndex: number, kfIndex: number, event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (isNaN(value)) return;

    const signal = this.getRotationKeyframesForProp(propIndex);
    signal.update(kfs => {
      const newKfs = [...kfs];
      newKfs[kfIndex] = { ...newKfs[kfIndex], rpm: value };
      return newKfs;
    });
  }

  previewTrajectory() {
    this.clearTrajectoryLine();
    this.isPreviewing.set(true);
    this.animationStartTime = performance.now();

    const params = this.simulationParams();
    this.simulationTrajectory = this.physicsService.calculateSampledTrajectoryForPreview(params);

    if (this.simulationTrajectory.length > 0) {
      const points = this.simulationTrajectory.map(p => new THREE.Vector3(p.x, p.y, p.z));
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x00ffff, linewidth: 2 });
      this.trajectoryLine = new THREE.Line(geometry, material);
      this.trajectoryLine.layers.set(SCENE_LAYER);
      this.scene.add(this.trajectoryLine);
    }
  }

  runSimulation() {
    this.runBackendSimulation();
  }

  refreshBackendCapabilities(): void {
    this.backendError.set('');
    this.backendSimulationService.getCapabilities()
      .then(capabilities => this.backendCapabilities.set(capabilities))
      .catch(error => {
        this.backendCapabilities.set(null);
        this.backendError.set(error?.message || 'Backend unavailable');
      });
  }

  runBackendSimulation(): void {
    const budget = this.simulationBudget();
    if (budget.level === 'blocked') {
      window.alert(`${this.l.t('runtimeBlocked')()} ${this.l.t('recommendedFrames')()}: ${this.recommendedFrames()}`);
      return;
    }

    this.clearTrajectoryLine();
    this.isPreviewing.set(false);
    this.isBackendSimulating.set(true);
    this.simulationResult.set(null);
    this.backendSummary.set(null);
    this.backendJob.set(null);
    this.backendError.set('');

    const paramsSnapshot = this.simulationParams();
    this.backendSimulationService.startJob(paramsSnapshot)
      .then(job => {
        this.activeBackendJobId = job.job_id;
        this.backendJobParams.set(job.job_id, paramsSnapshot);
        this.backendJob.set(job);
        this.pollBackendJob(job.job_id);
      })
      .catch(error => {
        this.backendError.set(error?.message || 'Backend simulation failed');
        this.isBackendSimulating.set(false);
      });
  }

  private pollBackendJob(jobId: string): void {
    this.backendSimulationService.getJob(jobId)
      .then(job => {
        if (this.activeBackendJobId !== jobId) {
          return;
        }
        this.backendJob.set(job);
        if (job.status === 'completed') {
          this.backendSummary.set(job.summary);
          const paramsSnapshot = this.backendJobParams.get(jobId) ?? this.simulationParams();
          if (job.result) {
            const result = this.backendSimulationService.backendResponseToSimulationResult(job.result, paramsSnapshot);
            this.simulationResult.set(result);
          } else if (job.summary) {
            const result = this.backendSimulationService.summaryToSimulationResult(job.summary, paramsSnapshot);
            this.simulationResult.set(result);
          }
          this.isBackendSimulating.set(false);
          this.showResultsModal.set(true);
          setTimeout(() => this.drawResultImages(), 0);
          return;
        }
        if (job.status === 'failed') {
          this.backendError.set(job.error || 'Backend simulation failed');
          this.isBackendSimulating.set(false);
          return;
        }
        window.setTimeout(() => this.pollBackendJob(jobId), 600);
      })
      .catch(error => {
        this.backendError.set(error?.message || 'Backend simulation polling failed');
        this.isBackendSimulating.set(false);
      });
  }

  downloadData() {
    const backendUrl = this.backendDownloadUrl();
    if (backendUrl) {
      const a = document.createElement('a');
      a.href = backendUrl;
      a.download = 'spad_simulation_counts.bin';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      return;
    }

    const result = this.simulationResult();
    if (!result?.dataset) return;

    // Helper function for downloading
    const downloadFile = (blob: Blob, filename: string) => {
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    };

    const dataBytes = new Uint8Array(result.dataset.byteLength);
    dataBytes.set(new Uint8Array(result.dataset.buffer, result.dataset.byteOffset, result.dataset.byteLength));
    const dataBlob = new Blob([dataBytes.buffer as ArrayBuffer], { type: 'application/octet-stream' });
    downloadFile(dataBlob, 'spad_simulation_data.bin');
  }

  private clearTrajectoryLine() {
    if (this.trajectoryLine) {
      this.scene.remove(this.trajectoryLine);
      this.trajectoryLine.geometry.dispose();
      (this.trajectoryLine.material as THREE.Material).dispose();
      this.trajectoryLine = null;
    }
  }

  private buildOperationalScene(): void {
    const runwayMat = new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.92 });
    const runway = new THREE.Mesh(new THREE.PlaneGeometry(4, 18), runwayMat);
    runway.rotation.x = -Math.PI / 2;
    runway.position.set(0, 0.015, 5);
    runway.layers.set(SCENE_LAYER);
    this.scene.add(runway);

    const stripeMat = new THREE.MeshBasicMaterial({ color: 0xe5e7eb });
    for (let i = 0; i < 7; i++) {
      const stripe = new THREE.Mesh(new THREE.PlaneGeometry(0.08, 0.8), stripeMat);
      stripe.rotation.x = -Math.PI / 2;
      stripe.position.set(0, 0.025, -2 + i * 2);
      stripe.layers.set(SCENE_LAYER);
      this.scene.add(stripe);
    }

    const buildingMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.7, metalness: 0.1 });
    [
      { x: -4.5, z: 4, h: 1.4 },
      { x: 4.2, z: 2.5, h: 2.2 },
      { x: -5.2, z: 9, h: 2.8 },
    ].forEach(item => {
      const building = new THREE.Mesh(new THREE.BoxGeometry(1.3, item.h, 1.1), buildingMat);
      building.position.set(item.x, item.h / 2, item.z);
      building.layers.set(SCENE_LAYER);
      this.scene.add(building);
    });

    const mastMat = new THREE.MeshStandardMaterial({ color: 0xd1d5db, metalness: 0.5, roughness: 0.35 });
    [-3, 3].forEach(x => {
      const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, 2.5, 12), mastMat);
      mast.position.set(x, 1.25, 1);
      mast.layers.set(SCENE_LAYER);
      this.scene.add(mast);
    });

    const axes = new THREE.AxesHelper(1.2);
    axes.position.set(-2.7, 0.04, -2.7);
    axes.layers.set(SCENE_LAYER);
    this.scene.add(axes);
  }

  private disposeObject3D(object: THREE.Object3D): void {
    object.traverse(child => {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      const material = mesh.material;
      if (Array.isArray(material)) {
        material.forEach(item => item.dispose());
      } else if (material) {
        material.dispose();
      }
    });
  }

  private rebuildDroneModel(): void {
    if (!this.drone) return;

    for (const child of [...this.drone.children]) {
      this.drone.remove(child);
      this.disposeObject3D(child);
    }

    const preset = this.selectedDronePreset();
    const bodyMat = new THREE.MeshStandardMaterial({ color: 0x2f343b, metalness: 0.18, roughness: 0.48 });
    const armMat = new THREE.MeshStandardMaterial({ color: 0x111827, metalness: 0.25, roughness: 0.5 });
    const propMat = new THREE.MeshStandardMaterial({ color: 0x9ca3af, metalness: 0.1, roughness: 0.35 });
    const diskMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.12, side: THREE.DoubleSide });

    const body = new THREE.Mesh(
      new THREE.BoxGeometry(preset.dimensions.widthM * 0.38, preset.dimensions.heightM, preset.dimensions.lengthM * 0.42),
      bodyMat,
    );
    body.layers.set(SCENE_LAYER);
    this.drone.add(body);

    const armLengthX = preset.dimensions.widthM * 0.72;
    const armLengthZ = preset.dimensions.lengthM * 0.72;
    const armThickness = Math.max(0.012, preset.dimensions.heightM * 0.18);
    const armX = new THREE.Mesh(new THREE.BoxGeometry(armLengthX, armThickness, armThickness), armMat);
    const armZ = new THREE.Mesh(new THREE.BoxGeometry(armThickness, armThickness, armLengthZ), armMat);
    armX.layers.set(SCENE_LAYER);
    armZ.layers.set(SCENE_LAYER);
    this.drone.add(armX, armZ);

    const propCenters = [
      { x: armLengthX / 2, z: armLengthZ / 2 },
      { x: -armLengthX / 2, z: armLengthZ / 2 },
      { x: armLengthX / 2, z: -armLengthZ / 2 },
      { x: -armLengthX / 2, z: -armLengthZ / 2 },
    ];
    this.dronePropellers = [];
    propCenters.forEach(center => {
      const pivot = new THREE.Group();
      pivot.position.set(center.x, preset.dimensions.heightM * 0.12, center.z);
      pivot.layers.set(SCENE_LAYER);

      const disk = new THREE.Mesh(new THREE.CircleGeometry(preset.propellerDiameterM / 2, 48), diskMat);
      disk.rotation.x = -Math.PI / 2;
      disk.layers.set(SCENE_LAYER);
      pivot.add(disk);

      const blade = new THREE.Mesh(new THREE.BoxGeometry(preset.propellerDiameterM, 0.006, preset.propellerDiameterM * 0.08), propMat);
      blade.layers.set(SCENE_LAYER);
      pivot.add(blade);

      this.dronePropellers.push(pivot);
      this.drone.add(pivot);
    });
  }

  private createTelescopeVisual(): void {
    const tubeMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, metalness: 0.45, roughness: 0.34 });
    this.telescopeTube = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.45, 32, 1, true), tubeMat);
    this.telescopeTube.position.z = 0.16;
    this.telescopeTube.layers.set(RIG_LAYER);
    this.detectorRig.add(this.telescopeTube);

    const glassMat = new THREE.MeshStandardMaterial({ color: 0x67e8f9, transparent: true, opacity: 0.42, metalness: 0.1, roughness: 0.08 });
    this.apertureGlass = new THREE.Mesh(new THREE.CircleGeometry(0.07, 48), glassMat);
    this.apertureGlass.position.z = 0.39;
    this.apertureGlass.layers.set(RIG_LAYER);
    this.detectorRig.add(this.apertureGlass);
    this.updateTelescopeVisual();
  }

  private updateTelescopeVisual(): void {
    const apertureRadius = Math.max(0.04, Math.min(0.22, this.apertureDiameter() * 1.8));
    const visualLength = Math.max(0.32, Math.min(1.1, this.telescopeFocalLength() * 0.16));

    this.telescopeTube.geometry.dispose();
    const tubeGeo = new THREE.CylinderGeometry(apertureRadius, apertureRadius * 0.82, visualLength, 48, 1, true);
    tubeGeo.rotateX(Math.PI / 2);
    this.telescopeTube.geometry = tubeGeo;
    this.telescopeTube.position.z = visualLength / 2 + 0.08;

    this.apertureGlass.geometry.dispose();
    this.apertureGlass.geometry = new THREE.CircleGeometry(apertureRadius * 0.92, 64);
    this.apertureGlass.position.z = visualLength + 0.08;
  }

  private createSkyTexture(): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 1024;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;

    const sky = ctx.createLinearGradient(0, 0, 0, canvas.height);
    sky.addColorStop(0, '#1f5f9f');
    sky.addColorStop(0.34, '#66a7dc');
    sky.addColorStop(0.66, '#b9daf0');
    sky.addColorStop(1, '#f0d2a8');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // 程序化薄云和颗粒感，避免天空像纯色背景。
    for (let i = 0; i < 38; i++) {
      const y = 58 + (i * 31) % 250;
      const x = (i * 179) % canvas.width;
      const width = 190 + (i % 7) * 42;
      const height = 10 + (i % 5) * 4;
      const cloud = ctx.createRadialGradient(x, y, 0, x, y, width * 0.55);
      cloud.addColorStop(0, 'rgba(255,255,255,0.18)');
      cloud.addColorStop(0.5, 'rgba(255,255,255,0.08)');
      cloud.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = cloud;
      ctx.save();
      ctx.translate(x, y);
      ctx.scale(1, height / width);
      ctx.beginPath();
      ctx.arc(0, 0, width, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  private createGroundTexture(): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    ctx.fillStyle = '#566858';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let i = 0; i < 2200; i++) {
      const x = (i * 73) % canvas.width;
      const y = (i * 151) % canvas.height;
      const v = 58 + ((i * 29) % 60);
      ctx.fillStyle = `rgba(${v}, ${v + 38}, ${v - 16}, 0.22)`;
      ctx.fillRect(x, y, 1 + (i % 3), 1 + ((i >> 2) % 2));
    }

    ctx.strokeStyle = 'rgba(32, 45, 34, 0.16)';
    ctx.lineWidth = 1;
    for (let x = 0; x < canvas.width; x += 64) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x + 18, canvas.height);
      ctx.stroke();
    }

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.wrapS = THREE.RepeatWrapping;
    texture.wrapT = THREE.RepeatWrapping;
    texture.repeat.set(10, 10);
    texture.needsUpdate = true;
    return texture;
  }

  private createAtmosphereTexture(): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 64;
    canvas.height = 512;
    const ctx = canvas.getContext('2d')!;
    const haze = ctx.createLinearGradient(0, 0, 0, canvas.height);
    haze.addColorStop(0, 'rgba(255,255,255,0)');
    haze.addColorStop(0.48, 'rgba(186,219,239,0.04)');
    haze.addColorStop(0.74, 'rgba(238,221,183,0.22)');
    haze.addColorStop(1, 'rgba(245,202,150,0.42)');
    ctx.fillStyle = haze;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  private createRadialTexture(innerColor: string, outerColor: string, size = 256): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    const ctx = canvas.getContext('2d')!;
    const center = size / 2;
    const gradient = ctx.createRadialGradient(center, center, 0, center, center, center);
    gradient.addColorStop(0, innerColor);
    gradient.addColorStop(0.38, innerColor);
    gradient.addColorStop(1, outerColor);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, size, size);

    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
  }

  private buildAtmosphericScene(): void {
    this.skyTexture = this.createSkyTexture();
    this.atmosphereTexture = this.createAtmosphereTexture();
    this.sunDiscTexture = this.createRadialTexture('rgba(255,244,176,1)', 'rgba(255,174,88,0)');
    this.sunHaloTexture = this.createRadialTexture('rgba(255,230,160,0.42)', 'rgba(255,196,96,0)');

    const skyGeo = new THREE.SphereGeometry(460, 64, 32);
    const skyMat = new THREE.MeshBasicMaterial({
      map: this.skyTexture,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    this.skyDome = new THREE.Mesh(skyGeo, skyMat);
    this.skyDome.renderOrder = -1000;
    this.skyDome.layers.set(SCENE_LAYER);
    this.scene.add(this.skyDome);

    const atmosphereGeo = new THREE.CylinderGeometry(76, 76, 28, 96, 1, true);
    const atmosphereMat = new THREE.MeshBasicMaterial({
      map: this.atmosphereTexture,
      transparent: true,
      side: THREE.BackSide,
      depthWrite: false,
      depthTest: false,
    });
    this.atmosphereShell = new THREE.Mesh(atmosphereGeo, atmosphereMat);
    this.atmosphereShell.position.y = 10;
    this.atmosphereShell.renderOrder = -999;
    this.atmosphereShell.layers.set(SCENE_LAYER);
    this.scene.add(this.atmosphereShell);

    const sunDiscMat = new THREE.SpriteMaterial({
      map: this.sunDiscTexture,
      transparent: true,
      depthWrite: false,
      depthTest: false,
    });
    this.sunDisc = new THREE.Sprite(sunDiscMat);
    this.sunDisc.layers.set(SCENE_LAYER);
    this.scene.add(this.sunDisc);

    const sunHaloMat = new THREE.SpriteMaterial({
      map: this.sunHaloTexture,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      depthTest: false,
    });
    this.sunHalo = new THREE.Sprite(sunHaloMat);
    this.sunHalo.layers.set(SCENE_LAYER);
    this.scene.add(this.sunHalo);
  }

  private updateSolarVisuals(): void {
    if (!this.sun) return;

    const irradiance = Math.max(0, this.solarIrradiance());
    const response = irradiance / (irradiance + 0.18);
    const compressed = Math.pow(response, 0.62);
    const environment = this.selectedEnvironmentPreset();
    const visibilityKm = Math.max(0.5, this.atmosphericVisibilityKm());
    const haze = Math.max(environment.horizonHaze, Math.min(0.9, (23 / visibilityKm - 1) * 0.22));
    const sunPosition = new THREE.Vector3(-6, 3.8, -24);

    this.sun.position.copy(sunPosition);
    this.sun.intensity = 0.72 + compressed * (3.55 - haze * 1.1);

    if (this.ambientLight) {
      this.ambientLight.intensity = 0.42 + compressed * 0.45 + haze * 0.25;
    }
    if (this.skyLight) {
      this.skyLight.intensity = 0.55 + compressed * 0.7 + haze * 0.18;
    }
    if (this.renderer) {
      this.renderer.toneMappingExposure = 0.92 - compressed * 0.14 - haze * 0.04;
    }
    if (this.scene?.fog instanceof THREE.FogExp2) {
      this.scene.fog.density = 0.006 + haze * 0.026;
      this.scene.fog.color.setHSL(0.58, Math.max(0.18, 0.48 - haze * 0.22), 0.82 - haze * 0.18);
    }
    if (this.atmosphereShell) {
      (this.atmosphereShell.material as THREE.MeshBasicMaterial).opacity = 0.18 + haze * 0.52;
    }
    if (this.sunDisc && this.sunHalo) {
      const direction = sunPosition.normalize();
      const visualPosition = direction.multiplyScalar(80);
      this.sunDisc.position.copy(visualPosition);
      this.sunHalo.position.copy(visualPosition);
      this.sunDisc.scale.setScalar(2.0 + compressed * 1.4);
      this.sunHalo.scale.setScalar(15 + compressed * 16 + haze * 12);
      (this.sunDisc.material as THREE.SpriteMaterial).opacity = 0.72 + compressed * 0.24 - haze * 0.2;
      (this.sunHalo.material as THREE.SpriteMaterial).opacity = 0.18 + compressed * 0.18 + haze * 0.14;
    }
  }

  private initThreeJs() {
    const { clientWidth, clientHeight } = this.canvasContainer.nativeElement;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0xb5d5ec);
    this.scene.fog = new THREE.FogExp2(0xb9d6ea, 0.012);
    this.buildAtmosphericScene();

    const gridHelper = new THREE.GridHelper(50, 50, 0x888888, 0x444444);
    gridHelper.position.y = 0.01;
    this.scene.add(gridHelper);

    this.camera = new THREE.PerspectiveCamera(75, clientWidth / clientHeight, 0.1, 1000);
    this.camera.position.set(0, 3, 6);
    this.camera.lookAt(0, 1, 0);
    this.camera.layers.enableAll(); // Main camera sees all layers

    this.renderer = new THREE.WebGLRenderer({ canvas: this.rendererCanvas.nativeElement, antialias: true });
    this.renderer.setSize(clientWidth, clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.86;

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1, 0);
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 20;

    const groundGeo = new THREE.PlaneGeometry(50, 50);
    this.groundTexture = this.createGroundTexture();
    const groundMat = new THREE.MeshStandardMaterial({
      color: 0xb8c7ad,
      map: this.groundTexture,
      roughness: 0.96,
      metalness: 0.02,
      side: THREE.DoubleSide,
    });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.layers.set(SCENE_LAYER);
    this.scene.add(ground);
    this.buildOperationalScene();

    const ballGeo = new THREE.SphereGeometry(0.05, 32, 32);
    const ballMat = new THREE.MeshStandardMaterial({ color: 0xDFFF00 });
    this.ball = new THREE.Mesh(ballGeo, ballMat);
    this.ball.position.set(this.initialPosX(), this.initialPosY(), this.initialPosZ());
    this.ball.layers.set(SCENE_LAYER);
    this.scene.add(this.ball);


    const bladeGeo = new THREE.BoxGeometry(0.05, 0.005, 1);
    const bladeMat = new THREE.MeshStandardMaterial({ color: 0xcccccc });
    this.blade = new THREE.Mesh(bladeGeo, bladeMat);
    this.blade.layers.set(SCENE_LAYER);

    this.bladePivot = new THREE.Group();
    this.bladePivot.add(this.blade);
    this.scene.add(this.bladePivot);

    // Drone
    this.drone = new THREE.Group();
    this.scene.add(this.drone);
    this.rebuildDroneModel();

    this.drone.visible = this.targetType() === 'Drone';
    this.bladePivot.visible = this.targetType() === 'Blade';
    this.ball.visible = this.targetType() === 'Ball';

    // --- Detector Rig, Laser, and Beam ---
    this.detectorRig = new THREE.Group();
    this.detectorRig.position.set(0, 1, 0);
    this.scene.add(this.detectorRig);

    this.detectorCamera = new THREE.PerspectiveCamera(this.detectorFov(), 1.0, 0.1, 10000);
    this.detectorCamera.layers.set(SCENE_LAYER); // Detector camera only sees the scene layer
    this.detectorCamera.layers.enable(RIG_LAYER);
    this.scene.add(this.detectorCamera); // Add to scene, not rig

    const detectorGeo = new THREE.BoxGeometry(0.22, 0.18, 0.08);
    const detectorMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.35, roughness: 0.45 });
    const detector = new THREE.Mesh(detectorGeo, detectorMat);
    detector.layers.set(RIG_LAYER); // Detector visual is on rig layer
    this.detectorRig.add(detector); // Add to rig, position is (0,0,0) relative
    this.createTelescopeVisual();

    this.ambientLight = new THREE.AmbientLight(0x6f7d86, 0.64);
    this.scene.add(this.ambientLight);

    this.skyLight = new THREE.HemisphereLight(0xcfe8ff, 0x46523d, 0.85);
    this.scene.add(this.skyLight);

    this.sun = new THREE.DirectionalLight(0xfff3d0, 1.8);
    this.sun.position.set(-6, 3.8, -24);
    this.scene.add(this.sun);
    this.updateSolarVisuals();

    const laserDivergenceRad = Math.max(1e-6, this.transmitterDivergenceMrad() * 1e-3);

    const laserColor = this.wavelengthToThreeColor(this.laserWavelengthNm());
    this.laserSpotlight = new THREE.SpotLight(laserColor, this.laserAveragePower() * 1500);
    this.laserSpotlight.angle = laserDivergenceRad / 2;
    this.laserSpotlight.penumbra = 0.2;
    this.laserSpotlight.distance = 10000;
    this.laserSpotlight.decay = 1.5;

    this.detectorRig.add(this.laserSpotlight);
    this.detectorRig.add(this.laserSpotlight.target);
    this.laserSpotlight.target.position.set(0, 0, 50);

    const beamDistance = 20;
    const beamEndRadius = beamDistance * Math.tan(this.laserSpotlight.angle);
    const beamGeo = new THREE.CylinderGeometry(beamEndRadius, 0.01, beamDistance, 32, 1, true);
    beamGeo.rotateX(Math.PI / 2);
    beamGeo.translate(0, 0, beamDistance / 2);

    const beamMat = new THREE.MeshBasicMaterial({
      color: laserColor,
      transparent: true,
      opacity: (this.laserAveragePower() / 3) * 0.6,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide
    });
    this.laserBeam = new THREE.Mesh(beamGeo, beamMat);
    this.laserBeam.layers.set(RIG_LAYER); // The visual beam is on rig layer
    this.detectorRig.add(this.laserBeam);
  }

  private updateBladeMesh(image: HTMLImageElement | null, bladeLength: number): void {
    if (!this.blade) return;

    if ((this.blade.material as THREE.MeshStandardMaterial).alphaMap) {
      (this.blade.material as THREE.MeshStandardMaterial).alphaMap!.dispose();
    }

    if (image && bladeLength > 0) {
      const { canvas: alphaCanvas, centroid } = this.createAlphaMapAndCentroid(image);
      const alphaTexture = new THREE.CanvasTexture(alphaCanvas);
      alphaTexture.needsUpdate = true;

      const aspectRatio = image.width / image.height;
      const geomWidth = image.width >= image.height ? bladeLength : bladeLength * aspectRatio;
      const geomHeight = image.width >= image.height ? bladeLength / aspectRatio : bladeLength;

      const propellerGeom = new THREE.PlaneGeometry(geomWidth, geomHeight);

      if (centroid) {
        const offsetX = -(centroid.x / image.width - 0.5) * geomWidth;
        const offsetY = (centroid.y / image.height - 0.5) * geomHeight;
        propellerGeom.translate(offsetX, offsetY, 0);
      }

      propellerGeom.rotateX(Math.PI / 2);

      const propellerMat = this.blade.material as THREE.MeshStandardMaterial;
      propellerMat.alphaMap = alphaTexture;
      propellerMat.transparent = true;
      propellerMat.side = THREE.DoubleSide;
      propellerMat.color.set(0xcccccc);
      propellerMat.needsUpdate = true;

      this.blade.geometry.dispose();
      this.blade.geometry = propellerGeom;

    } else { // Fallback to generic blade
      const bladeWidth = 0.05;
      // Set thickness to 1cm (0.01)
      const newGeo = new THREE.BoxGeometry(bladeWidth, 0.01, bladeLength > 0 ? bladeLength : 0.01);
      newGeo.translate(0, 0, (bladeLength > 0 ? bladeLength : 0.01) / 2);

      const bladeMat = this.blade.material as THREE.MeshStandardMaterial;
      bladeMat.alphaMap = null;
      bladeMat.transparent = false;
      bladeMat.side = THREE.FrontSide;
      bladeMat.needsUpdate = true;

      this.blade.geometry.dispose();
      this.blade.geometry = newGeo;
    }
  }

  private createAlphaMapAndCentroid(image: HTMLImageElement): { canvas: HTMLCanvasElement, centroid: { x: number, y: number } | null } {
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext('2d')!;
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    const { data, width, height } = imageData;

    let sumX = 0;
    let sumY = 0;
    let pointCount = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
      const luminance = 0.299 * r + 0.587 * g + 0.114 * b;

      const isPropeller = a > 128 && luminance < 128;

      if (isPropeller) {
        const x = (i / 4) % width;
        const y = Math.floor((i / 4) / width);
        sumX += x;
        sumY += y;
        pointCount++;
      }

      const alphaColor = isPropeller ? 255 : 0;
      data[i] = alphaColor;
      data[i + 1] = alphaColor;
      data[i + 2] = alphaColor;
      data[i + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    const centroid = pointCount > 0 ? { x: sumX / pointCount, y: sumY / pointCount } : null;
    return { canvas, centroid };
  }

  private startRenderingLoop() {
    const render = () => {
      this.frameId = requestAnimationFrame(render);
      const now = performance.now();
      const deltaSeconds = Math.min(0.05, (now - this.lastRenderTime) / 1000);
      this.lastRenderTime = now;
      this.updateManualDroneFlight(deltaSeconds);

      // --- Handle resize at the start of the frame ---
      if (this.resizePending) {
        const { width, height } = this.viewportSize;
        if (width > 0 && height > 0) {
          this.camera.aspect = width / height;
          this.camera.updateProjectionMatrix();
          this.renderer.setSize(width, height);
        }
        this.resizePending = false;
      }

      this.controls.update();

      const targetType = this.targetType();
      const isBlade = targetType === 'Blade';
      const isDrone = targetType === 'Drone';
      const isRotational = isBlade || isDrone;

      if (this.isPreviewing() && this.simulationTrajectory.length > 0) {
        const params = this.simulationParams();
        const totalDuration = this.physicsService.calculateTotalPathTime(params);
        const elapsedTimeSeconds = (performance.now() - this.animationStartTime) / 1000;

        if (elapsedTimeSeconds >= totalDuration) {
          this.isPreviewing.set(false);
        }
        const progress = totalDuration > 0 ? elapsedTimeSeconds / totalDuration : 1;
        const currentIndex = Math.min(this.simulationTrajectory.length - 1, Math.floor(progress * this.simulationTrajectory.length));
        const pos = this.simulationTrajectory[currentIndex];

        if (isRotational) {
          if (pos) {
            if (isBlade) this.bladePivot.position.set(pos.x, pos.y, pos.z);
            if (isDrone) this.drone.position.set(pos.x, pos.y, pos.z);
          }

          const pitchRad = THREE.MathUtils.degToRad(this.bladePitch());

          if (isBlade) {
            const currentAngle = this.physicsService.calculateBladeRotationAngleAtTime(elapsedTimeSeconds, params);
            this.bladePivot.rotation.order = 'XYZ';
            this.bladePivot.rotation.x = pitchRad;
            this.bladePivot.rotation.y = currentAngle;
            this.bladePivot.rotation.z = 0;
          } else if (isDrone) {
            const recordedPose = params.recordedDroneTrajectory?.[Math.min(currentIndex, (params.recordedDroneTrajectory?.length ?? 1) - 1)];
            this.drone.rotation.order = 'YXZ';
            this.drone.rotation.y = THREE.MathUtils.degToRad(recordedPose?.yawDeg ?? 0);
            this.drone.rotation.x = THREE.MathUtils.degToRad(recordedPose?.pitchDeg ?? this.bladePitch());
            this.drone.rotation.z = THREE.MathUtils.degToRad(recordedPose?.rollDeg ?? 0);

            this.dronePropellers.forEach((prop, i) => {
              const angle = this.physicsService.calculateBladeRotationAngleAtTime(elapsedTimeSeconds, params, (i + 1) as 1 | 2 | 3 | 4);
              // Diagonal pairs rotate same direction for stability visual
              // 1(FR) & 4(RL) (idx 0 & 3) = 1 direction
              // 2(FL) & 3(RR) (idx 1 & 2) = -1 direction
              const dir = (i === 0 || i === 3) ? 1 : -1;
              prop.rotation.y = angle * dir;
            });
          }

        } else { // Ball
          if (pos) this.ball.position.set(pos.x, pos.y, pos.z);
        }

      } else { // Not previewing
        if (!this.trajectoryLine) {
          if (isRotational) {
            let startPos = { x: this.initialPosX(), y: this.initialPosY(), z: this.initialPosZ() };

            if (isDrone && this.isManualDroneMode()) {
              startPos = this.manualDronePosition();
            } else if (this.bladeMotionType() === 'Path') {
              const firstWp = this.waypoints()[0];
              if (firstWp) startPos = firstWp.pos;
            }

            if (isBlade) this.bladePivot.position.set(startPos.x, startPos.y, startPos.z);
            if (isDrone) this.drone.position.set(startPos.x, startPos.y, startPos.z);

          } else {
            this.ball.position.set(this.initialPosX(), this.initialPosY(), this.initialPosZ());
          }
        }

        const pitchRad = THREE.MathUtils.degToRad(isDrone && this.isManualDroneMode() ? this.manualDronePitch() : this.bladePitch());
        if (isBlade) {
          this.bladePivot.rotation.order = 'XYZ';
          this.bladePivot.rotation.x = pitchRad;
          this.bladePivot.rotation.y += 0.01;
          this.bladePivot.rotation.z = 0;
        } else if (isDrone) {
          this.drone.rotation.order = 'YXZ';
          this.drone.rotation.y = THREE.MathUtils.degToRad(this.isManualDroneMode() ? this.manualDroneYaw() : 0);
          this.drone.rotation.x = pitchRad;
          this.drone.rotation.z = THREE.MathUtils.degToRad(this.isManualDroneMode() ? this.manualDroneRoll() : 0);
          const manualRpms = this.manualDronePropellerRpms();
          this.dronePropellers.forEach((prop, i) => {
            const dir = (i === 0 || i === 3) ? 1 : -1;
            const rpm = this.isManualDroneMode() ? manualRpms[i] : this.rotationSpeed();
            prop.rotation.y += rpm * 2 * Math.PI / 60 * deltaSeconds * dir;
          });
        }
      }

      this.detectorRig.getWorldPosition(this.detectorCamera.position);
      this.detectorRig.getWorldQuaternion(this.detectorCamera.quaternion);
      this.detectorCamera.rotateY(Math.PI);

      const { width: clientWidth, height: clientHeight } = this.viewportSize;

      this.renderer.setViewport(0, 0, clientWidth, clientHeight);
      this.renderer.setScissor(0, 0, clientWidth, clientHeight);
      this.renderer.setScissorTest(false);
      this.renderer.render(this.scene, this.camera);

      const rect = this.detectorViewport.nativeElement.getBoundingClientRect();
      const canvasRect = this.rendererCanvas.nativeElement.getBoundingClientRect();

      if (rect.width > 0 && rect.height > 0) {
        const detectorAspect = rect.width / rect.height;
        this.detectorCamera.aspect = detectorAspect;
        this.detectorCamera.updateProjectionMatrix();

        const insetX = rect.left - canvasRect.left;
        const insetY = canvasRect.bottom - rect.bottom;
        const insetWidth = rect.width;
        const insetHeight = rect.height;

        this.renderer.setViewport(insetX, insetY, insetWidth, insetHeight);
        this.renderer.setScissor(insetX, insetY, insetWidth, insetHeight);
        this.renderer.setScissorTest(true);

        this.renderer.render(this.scene, this.detectorCamera);
      }
    };
    render();
  }

  private drawResultImages(): void {
    const result = this.simulationResult();
    const pcCanvas = this.photonCountingCanvas?.nativeElement;
    const gtCanvas = this.groundTruthCanvas?.nativeElement;
    const ipCanvas = this.incidentPhotonsCanvas?.nativeElement;

    if (!result || !pcCanvas || !gtCanvas || !ipCanvas) {
      return;
    }

    const { tdcMaxCount } = this.simulationParams();
    const { width, height } = result.resolution ?? this.simulationParams().resolution;
    const totalPixels = width * height;

    const photonCounts: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    if (result.photonCountMap) {
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
          photonCounts[row][col] = result.photonCountMap[row]?.[col] ?? 0;
        }
      }
    } else {
      const emptyPixelValue = tdcMaxCount + 2;
      for (let i = 0; i < result.dataset.length; i++) {
        if (result.dataset[i] < emptyPixelValue) {
          const pixelIndexInFrame = i % totalPixels;
          const row = Math.floor(pixelIndexInFrame / width);
          const col = pixelIndexInFrame % width;
          photonCounts[row][col]++;
        }
      }
    }

    const groundTruthCounts: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    if (result.groundTruthMap) {
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
          groundTruthCounts[row][col] = result.groundTruthMap[row]?.[col] ?? 0;
        }
      }
    } else {
      for (const coord of result.signalCoordinates) {
        groundTruthCounts[coord.row][coord.col]++;
      }
    }

    this.drawHeatmap(pcCanvas, photonCounts, 'occupancy');
    this.drawHeatmap(gtCanvas, groundTruthCounts, 'intensity');
    this.drawHeatmap(ipCanvas, result.incidentPhotonMap, 'intensity');
  }

  private jet(value: number): [number, number, number] {
    const stops: Array<[number, number, number]> = [
      [8, 14, 70],
      [18, 70, 150],
      [25, 150, 190],
      [92, 205, 150],
      [245, 220, 92],
    ];
    const scaled = Math.min(1, Math.max(0, value)) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    const t = scaled - index;
    const a = stops[index];
    const b = stops[index + 1];
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  private drawHeatmap(canvas: HTMLCanvasElement, data: number[][], mode: 'occupancy' | 'intensity' = 'intensity') {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = data.length;
    if (height === 0) return;
    const width = data[0].length;

    canvas.width = width;
    canvas.height = height;

    const positiveValues: number[] = [];
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        if (data[r][c] > 0) {
          positiveValues.push(data[r][c]);
        }
      }
    }

    if (positiveValues.length === 0) {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, width, height);
      return;
    }
    positiveValues.sort((a, b) => a - b);
    const percentileIndex = Math.min(positiveValues.length - 1, Math.floor(positiveValues.length * 0.995));
    const displayMax = Math.max(positiveValues[percentileIndex], positiveValues[positiveValues.length - 1] * 0.35, 1e-12);

    const imageData = ctx.createImageData(width, height);
    for (let r = 0; r < height; r++) {
      for (let c = 0; c < width; c++) {
        const linear = Math.min(1, Math.max(0, data[r][c] / displayMax));
        const normalizedValue = mode === 'occupancy'
          ? Math.sqrt(linear)
          : Math.log1p(9 * linear) / Math.log(10);
        const [red, green, blue] = this.jet(normalizedValue);
        const index = (r * width + c) * 4;
        imageData.data[index] = red;
        imageData.data[index + 1] = green;
        imageData.data[index + 2] = blue;
        imageData.data[index + 3] = 255;
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }
}
