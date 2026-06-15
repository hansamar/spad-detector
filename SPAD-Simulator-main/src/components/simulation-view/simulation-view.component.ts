import { ChangeDetectionStrategy, Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, signal, computed, effect, WritableSignal, inject } from '@angular/core';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { ISimulationParams, ISimulationResult, IWaypoint } from '../../models/simulation-params.model';
import { SimulationService } from '../../services/simulation.service';
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
  
  @ViewChild('photonCountingCanvas')
  private photonCountingCanvas?: ElementRef<HTMLCanvasElement>;

  @ViewChild('groundTruthCanvas')
  private groundTruthCanvas?: ElementRef<HTMLCanvasElement>;
  
  @ViewChild('incidentPhotonsCanvas')
  private incidentPhotonsCanvas?: ElementRef<HTMLCanvasElement>;

  @ViewChild('fileInput')
  private fileInput?: ElementRef<HTMLInputElement>;


  // --- Services ---
  private simulationService = inject(SimulationService);
  private physicsService = inject(PhysicsService);
  l = inject(LocalizationService); // Public for template access

  // --- UI State ---
  isSimulating = signal(false);
  isPreviewing = signal(false);
  simulationProgress = signal(0);
  simulationResult = signal<ISimulationResult | null>(null);
  showResultsModal = signal(false);
  isPanelOpen = signal(false);
  uploadedImage = signal<HTMLImageElement | null>(null);
  uploadedImageUrl = signal<string | null>(null);
  binarizedImageUrl = signal<string | null>(null);

  // Accordion state
  targetSettingsOpen = signal(true);
  detectorSettingsOpen = signal(false);
  envSettingsOpen = signal(false);
  simSettingsOpen = signal(false);

  // --- Simulation Parameters ---
  // Target
  targetType = signal<'Ball' | 'Blade'>('Ball');
  initialPosX = signal(-1.0); 
  initialPosY = signal(2.0);
  initialPosZ = signal(1.5);
  initialVelX = signal(1.0);
  initialVelY = signal(0.0);
  initialVelZ = signal(1.0);
  reflectivity = signal(0.3);
  restitution = signal(0.8);

  // Motion Types
  ballMotionType = signal<'Gravity' | 'Rotation'>('Gravity');
  bladeMotionType = signal<'Fixed' | 'Path'>('Path');

  // Blade Path Planning
  waypoints = signal<IWaypoint[]>([]);
  pathSpeeds = signal<number[]>([]); // Speed in m/s for each segment
  
  // Dynamic Target
  rotationRadius = signal(0.5); // For Blade, this is size/length. For Ball, rotation radius.
  rotationSpeed = signal(1200); // in RPM
  bladePitch = signal(15); // in degrees, for Blade target

  // Detector
  resolutionW = signal(64);
  resolutionH = signal(64);
  detectorFov = signal(50);
  detectorYaw = signal(0); // degrees
  detectorPitch = signal(0); // degrees
  frameDurationUs = signal(20);
  quantumEfficiency = signal(0.3);
  apertureDiameter = signal(0.025);
  systemEfficiency = signal(0.05);
  filterBandwidth = signal(10);
  darkCountRate = signal(100);
  timeResolutionPs = signal(256);
  tdcBitDepth = signal(13); // Default to 13 bits (8191)
  tdcMaxCount = signal((2 ** 13) - 1);

  // Environment & Laser
  laserMode = signal<'Pulsed' | 'CW'>('Pulsed');
  solarIrradiance = signal(0.001);
  atmosphericAttenuationEnabled = signal(true);
  laserWavelengthNm = signal(780);
  laserAveragePower = signal(0.1);
  laserPulseWidthNs = signal(1);
  laserRepetitionFrequency = signal(1000000);

  // Simulation
  nFrames = signal(100000);
  
  // --- UI Computed Flags ---
  isPulsedMode = computed(() => this.laserMode() === 'Pulsed');
  
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

  // Combine all params into a computed signal for the simulation service
  simulationParams = computed<ISimulationParams>(() => ({
    targetType: this.targetType(),
    initialPos: { x: this.initialPosX(), y: this.initialPosY(), z: this.initialPosZ() },
    initialVel: { x: this.initialVelX(), y: this.initialVelY(), z: this.initialVelZ() },
    ballMotionType: this.ballMotionType(),
    bladeMotionType: this.bladeMotionType(),
    waypoints: this.waypoints(),
    pathSpeeds: this.pathSpeeds(),
    reflectivity: this.reflectivity(),
    restitution: this.restitution(),
    rotationRadius: this.rotationRadius(),
    rotationSpeed: this.rotationSpeed(),
    bladePitch: this.bladePitch(),
    rotationCenter: { x: this.initialPosX(), z: this.initialPosZ() },
    uploadedImage: this.uploadedImage(),
    resolution: { width: this.resolutionW(), height: this.resolutionH() },
    detectorFov: this.detectorFov(),
    detectorYaw: this.detectorYaw(),
    detectorPitch: this.detectorPitch(),
    frameDurationUs: this.frameDurationUs(),
    quantumEfficiency: this.quantumEfficiency(),
    apertureDiameter: this.apertureDiameter(),
    systemEfficiency: this.systemEfficiency(),
    filterBandwidth: this.filterBandwidth(),
    darkCountRate: this.darkCountRate(),
    timeResolutionPs: this.timeResolutionPs(),
    tdcMaxCount: this.tdcMaxCount(),
    solarIrradiance: this.solarIrradiance(),
    atmosphericAttenuationEnabled: this.atmosphericAttenuationEnabled(),
    laserMode: this.laserMode(),
    laserPulseEnergy: this.laserSinglePulseEnergy(),
    laserAveragePower: this.laserAveragePower(),
    laserRepetitionFrequency: this.laserRepetitionFrequency(),
    laserPulseWidthNs: this.laserPulseWidthNs(),
    laserWavelengthNm: this.laserWavelengthNm(),
    nFrames: this.nFrames(),
    cameraHeight: 1.0,
  }));

  // --- 3D Scene ---
  private scene!: THREE.Scene;
  private camera!: THREE.PerspectiveCamera;
  private detectorCamera!: THREE.PerspectiveCamera;
  private renderer!: THREE.WebGLRenderer;
  private controls!: OrbitControls;
  private ball!: THREE.Mesh;
  private blade!: THREE.Mesh;
  private bladePivot!: THREE.Group;
  private sun!: THREE.DirectionalLight;
  private laserSpotlight!: THREE.SpotLight;
  private laserBeam!: THREE.Mesh;
  private detectorRig!: THREE.Group;
  private frameId: number | null = null;
  private resizeObserver!: ResizeObserver;
  private pathLine: THREE.Line | null = null;

  // --- Animation State ---
  private simulationTrajectory: {x: number, y: number, z: number}[] = [];
  private trajectoryLine: THREE.Line | null = null;
  private animationStartTime = 0;

  constructor() {
    effect(() => {
        const irradiance = this.solarIrradiance();
        if(this.sun) {
            this.sun.intensity = irradiance * 100;
        }
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
        this.initialPosX(); this.initialPosY(); this.initialPosZ();
        this.initialVelX(); this.initialVelY(); this.initialVelZ();
        this.restitution(); this.frameDurationUs(); this.nFrames();
        this.rotationSpeed(); this.waypoints(); this.pathSpeeds();
        
        if (!this.isPreviewing()) {
            this.clearTrajectoryLine();
        }
    });
    
    // Effect for toggling target visibility and updating blade mesh
    effect(() => {
        const type = this.targetType();
        const image = this.uploadedImage();
        const bladeLength = this.rotationRadius(); // Re-trigger on length change

        if (this.ball && this.bladePivot) {
            this.ball.visible = type === 'Ball';
            this.bladePivot.visible = type === 'Blade';
            if (type === 'Blade') {
                this.updateBladeMesh(image, bladeLength);
                // Initialize default path if not present
                if (this.waypoints().length < 2) {
                    this.waypoints.set([
                        { id: Date.now(), pos: { x: 1, y: 1, z: 3 } },
                        { id: Date.now() + 1, pos: { x: -1, y: 3, z: 6 } }
                    ]);
                    this.pathSpeeds.set([2.0]);
                }
            }
        }
        this.clearTrajectoryLine();
    });

    effect(() => {
        const fov = Math.max(1, Math.min(this.detectorFov(), 179));
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
      
      const shouldShowPath = this.targetType() === 'Blade' && this.bladeMotionType() === 'Path' && points.length > 1;

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
    this.setupResizeObserver();
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
  }

  private setupResizeObserver(): void {
    this.resizeObserver = new ResizeObserver(() => {
      this.onResize();
    });
    this.resizeObserver.observe(this.canvasContainer.nativeElement);
  }

  private onResize = () => {
    const { clientWidth, clientHeight } = this.canvasContainer.nativeElement;
    this.renderer.setSize(clientWidth, clientHeight);
    this.camera.aspect = clientWidth / clientHeight;
    this.camera.updateProjectionMatrix();
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
  
  // --- Path Planning UI Methods ---
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
  }

  removeWaypoint(index: number): void {
    this.waypoints.update(wps => wps.filter((_, i) => i !== index));
    this.pathSpeeds.update(speeds => speeds.filter((_, i) => i !== index));
  }
  
  updateWaypointPos(index: number, axis: 'x' | 'y' | 'z', event: Event): void {
    const value = parseFloat((event.target as HTMLInputElement).value);
    if (!isNaN(value)) {
      this.waypoints.update(wps => {
        const newWps = [...wps];
        newWps[index] = { ...newWps[index], pos: { ...newWps[index].pos, [axis]: value } };
        return newWps;
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
     }
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
    this.clearTrajectoryLine();
    this.isPreviewing.set(false);
    this.isSimulating.set(true);
    this.simulationProgress.set(0);
    this.simulationResult.set(null);

    const params = this.simulationParams();

    this.simulationService.generateData(params, this.simulationProgress)
      .then(result => {
        this.simulationResult.set(result);
        this.isSimulating.set(false);
        this.showResultsModal.set(true);
        setTimeout(() => this.drawResultImages(), 0);
      });
  }
  
  downloadData() {
    const data = this.simulationResult()?.dataset;
    if (!data) return;

    const blob = new Blob([data.buffer], { type: 'application/octet-stream' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'spad_simulation_data.bin';
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
  }
  
  private clearTrajectoryLine() {
    if (this.trajectoryLine) {
        this.scene.remove(this.trajectoryLine);
        this.trajectoryLine.geometry.dispose();
        (this.trajectoryLine.material as THREE.Material).dispose();
        this.trajectoryLine = null;
    }
  }

  private initThreeJs() {
    const { clientWidth, clientHeight } = this.canvasContainer.nativeElement;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x9AACBE); 

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

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.target.set(0, 1, 0);
    this.controls.maxPolarAngle = Math.PI / 2 - 0.05;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 20;

    const groundGeo = new THREE.PlaneGeometry(50, 50);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x4a5568, side: THREE.DoubleSide });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.layers.set(SCENE_LAYER);
    this.scene.add(ground);

    const ballGeo = new THREE.SphereGeometry(0.05, 32, 32);
    const ballMat = new THREE.MeshStandardMaterial({ color: 0xDFFF00 });
    this.ball = new THREE.Mesh(ballGeo, ballMat);
    this.ball.position.set(this.initialPosX(), this.initialPosY(), this.initialPosZ());
    this.ball.layers.set(SCENE_LAYER);
    this.scene.add(this.ball);

    // Propeller Blade
    const bladeGeo = new THREE.BoxGeometry(0.05, 0.01, 1);
    const bladeMat = new THREE.MeshStandardMaterial({ color: 0xcccccc });
    this.blade = new THREE.Mesh(bladeGeo, bladeMat);
    this.blade.layers.set(SCENE_LAYER);
    
    this.bladePivot = new THREE.Group();
    this.bladePivot.add(this.blade);
    this.scene.add(this.bladePivot);
    this.bladePivot.visible = this.targetType() === 'Blade';
    this.ball.visible = this.targetType() === 'Ball';

    // --- Detector Rig, Laser, and Beam ---
    this.detectorRig = new THREE.Group();
    this.detectorRig.position.set(0, 1, 0);
    this.scene.add(this.detectorRig);

    this.detectorCamera = new THREE.PerspectiveCamera(this.detectorFov(), 1.0, 0.1, 50);
    this.detectorCamera.layers.set(SCENE_LAYER); // Detector camera only sees the scene layer
    this.detectorCamera.layers.enable(RIG_LAYER);
    this.scene.add(this.detectorCamera); // Add to scene, not rig

    const detectorGeo = new THREE.BoxGeometry(0.2, 0.2, 0.1);
    const detectorMat = new THREE.MeshStandardMaterial({ color: 0x9CA3AF });
    const detector = new THREE.Mesh(detectorGeo, detectorMat);
    detector.layers.set(RIG_LAYER); // Detector visual is on rig layer
    this.detectorRig.add(detector); // Add to rig, position is (0,0,0) relative

    const ambientLight = new THREE.AmbientLight(0x404040, 2);
    this.scene.add(ambientLight);
    
    this.sun = new THREE.DirectionalLight(0xffffff, this.solarIrradiance() * 100);
    this.sun.position.set(5, 10, 7.5);
    this.scene.add(this.sun);

    const laserFovRad = this.detectorFov() * Math.PI / 180;

    this.laserSpotlight = new THREE.SpotLight(0x39FF14, this.laserAveragePower() * 1500);
    this.laserSpotlight.angle = laserFovRad / 2;
    this.laserSpotlight.penumbra = 0.2;
    this.laserSpotlight.distance = 30;
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
      color: 0x39FF14,
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
      this.controls.update();

      const isBlade = this.targetType() === 'Blade';

      if (this.isPreviewing() && this.simulationTrajectory.length > 0) {
          const totalDuration = this.physicsService.calculateTotalPathTime(this.simulationParams());
          const elapsedTimeSeconds = (performance.now() - this.animationStartTime) / 1000;
          
          if (elapsedTimeSeconds >= totalDuration) {
              this.isPreviewing.set(false);
          }
          const progress = totalDuration > 0 ? elapsedTimeSeconds / totalDuration : 1;
          const currentIndex = Math.min(this.simulationTrajectory.length - 1, Math.floor(progress * this.simulationTrajectory.length));
          const pos = this.simulationTrajectory[currentIndex];

          if (isBlade) {
            if (pos) this.bladePivot.position.set(pos.x, pos.y, pos.z);
            const omega = this.rotationSpeed() * 2 * Math.PI / 60;
            const pitchRad = THREE.MathUtils.degToRad(this.bladePitch());
            this.bladePivot.rotation.order = 'XYZ';
            this.bladePivot.rotation.x = pitchRad;
            this.bladePivot.rotation.y = omega * elapsedTimeSeconds;
            this.bladePivot.rotation.z = 0;

          } else { // Ball
            if (pos) this.ball.position.set(pos.x, pos.y, pos.z);
          }
      } else { // Not previewing
        if (!this.trajectoryLine) { // Reset positions if no preview line
            if (isBlade) {
                if (this.bladeMotionType() === 'Path') {
                  const firstWp = this.waypoints()[0];
                  if (firstWp) this.bladePivot.position.set(firstWp.pos.x, firstWp.pos.y, firstWp.pos.z);
                } else {
                  this.bladePivot.position.set(this.initialPosX(), this.initialPosY(), this.initialPosZ());
                }
            } else {
                this.ball.position.set(this.initialPosX(), this.initialPosY(), this.initialPosZ());
            }
        }
        if (isBlade) {
             const pitchRad = THREE.MathUtils.degToRad(this.bladePitch());
             this.bladePivot.rotation.order = 'XYZ';
             this.bladePivot.rotation.x = pitchRad;
             this.bladePivot.rotation.y += 0.01; // Gentle idle rotation
             this.bladePivot.rotation.z = 0;
        }
      }
      
      this.detectorRig.getWorldPosition(this.detectorCamera.position);
      this.detectorRig.getWorldQuaternion(this.detectorCamera.quaternion);
      this.detectorCamera.rotateY(Math.PI);

      const { clientWidth, clientHeight } = this.canvasContainer.nativeElement;

      this.renderer.setViewport(0, 0, clientWidth, clientHeight);
      this.renderer.setScissor(0, 0, clientWidth, clientHeight);
      this.renderer.setScissorTest(false);
      this.renderer.render(this.scene, this.camera);

      const insetHeight = clientHeight / 4.5;
      const detectorAspect = this.resolutionW() / this.resolutionH();
      const insetWidth = insetHeight * detectorAspect;
      const insetX = clientWidth - insetWidth - 16;
      const insetY = 16;

      this.detectorCamera.aspect = detectorAspect;
      this.detectorCamera.updateProjectionMatrix();

      this.renderer.setViewport(insetX, insetY, insetWidth, insetHeight);
      this.renderer.setScissor(insetX, insetY, insetWidth, insetHeight);
      this.renderer.setScissorTest(true);
      
      this.renderer.render(this.scene, this.detectorCamera);
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

    const { resolution, tdcMaxCount } = this.simulationParams();
    const { width, height } = resolution;
    const totalPixels = width * height;

    const photonCounts: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    const emptyPixelValue = tdcMaxCount + 2;
    for (let i = 0; i < result.dataset.length; i++) {
        if (result.dataset[i] < emptyPixelValue) {
            const pixelIndexInFrame = i % totalPixels;
            const row = Math.floor(pixelIndexInFrame / width);
            const col = pixelIndexInFrame % width;
            photonCounts[row][col]++;
        }
    }
    
    const groundTruthCounts: number[][] = Array(height).fill(0).map(() => Array(width).fill(0));
    for(const coord of result.signalCoordinates) {
        groundTruthCounts[coord.row][coord.col]++;
    }

    this.drawHeatmap(pcCanvas, photonCounts);
    this.drawHeatmap(gtCanvas, groundTruthCounts);
    this.drawHeatmap(ipCanvas, result.incidentPhotonMap);
  }
  
  private jet(value: number): [number, number, number] {
    const r = Math.min(Math.max(0, 1.5 - Math.abs(1 - 4 * (value - 0.5))), 1);
    const g = Math.min(Math.max(0, 1.5 - Math.abs(1 - 4 * (value - 0.25))), 1);
    const b = Math.min(Math.max(0, 1.5 - Math.abs(1 - 4 * value)), 1);
    return [r * 255, g * 255, b * 255];
  }

  private drawHeatmap(canvas: HTMLCanvasElement, data: number[][]) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const height = data.length;
    if (height === 0) return;
    const width = data[0].length;
    
    canvas.width = width;
    canvas.height = height;

    let maxVal = 0;
    for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
            if (data[r][c] > maxVal) {
                maxVal = data[r][c];
            }
        }
    }
    
    if (maxVal === 0) {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, width, height);
        return;
    }

    const imageData = ctx.createImageData(width, height);
    for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
            const normalizedValue = data[r][c] / maxVal;
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