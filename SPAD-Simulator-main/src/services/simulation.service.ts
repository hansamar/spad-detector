import { Injectable, inject } from '@angular/core';
import { ISimulationParams, ISimulationResult } from '../models/simulation-params.model';
import { PhysicsService } from './physics.service';
import { WritableSignal } from '@angular/core';

/**
 * Checks if a point is inside a polygon using the ray casting algorithm.
 */
function isPointInPolygon(point: {col: number, row: number}, polygon: {col: number, row: number}[]): boolean {
    let isInside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i].col, yi = polygon[i].row;
        const xj = polygon[j].col, yj = polygon[j].row;

        const intersect = ((yi > point.row) !== (yj > point.row))
            && (point.col < (xj - xi) * (point.row - yi) / (yj - yi) + xi);
        if (intersect) isInside = !isInside;
    }
    return isInside;
}

const MAX_PROPELLER_POINTS = 5000; // Cap total points for performance

function getPropellerPointsFromImage(image: HTMLImageElement): { x: number, y: number, intensity: number }[] {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return [];
    canvas.width = image.width;
    canvas.height = image.height;
    ctx.drawImage(image, 0, 0);
    const imageData = ctx.getImageData(0, 0, image.width, image.height);
    const { data, width, height } = imageData;

    const allPoints: {x: number, y: number, intensity: number}[] = [];

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const i = (y * width + x) * 4;
            const r = data[i], g = data[i + 1], b = data[i + 2], a = data[i + 3];
            const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
            if (a > 128 && luminance < 128) {
                // Intensity is 1.0 for pure black (lum=0) and fades to ~0 for dark gray (lum=127)
                const intensity = Math.max(0, (128 - luminance) / 128);
                allPoints.push({ x: x, y: y, intensity });
            }
        }
    }

    if (allPoints.length === 0) return [];

    // --- Optimization: Sample points if there are too many ---
    let points: {x: number, y: number, intensity: number}[];
    if (allPoints.length > MAX_PROPELLER_POINTS) {
        // Efficiently sample points without shuffling the whole (potentially huge) array
        points = new Array(MAX_PROPELLER_POINTS);
        for (let i = 0; i < MAX_PROPELLER_POINTS; i++) {
            points[i] = allPoints[Math.floor(Math.random() * allPoints.length)];
        }
    } else {
        points = allPoints;
    }

    let sumX = 0;
    let sumY = 0;
    for(const p of points) {
        sumX += p.x;
        sumY += p.y;
    }

    const centroidX = sumX / points.length;
    const centroidY = sumY / points.length;

    for (const p of points) {
        p.x = (p.x - centroidX) / width; // Normalize relative to width
        p.y = (p.y - centroidY) / height; // Normalize relative to height
    }
    
    return points;
}

@Injectable({
  providedIn: 'root',
})
export class SimulationService {
  private physicsService = inject(PhysicsService);

  private degreesToRadians(degrees: number): number {
    return degrees * (Math.PI / 180);
  }

  /**
   * Generates a random number from a standard normal distribution using the Box-Muller transform.
   */
  private randn_bm(): number {
      let u = 0, v = 0;
      while(u === 0) u = Math.random();
      while(v === 0) v = Math.random();
      return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
  }

  public generateData(params: ISimulationParams, progress: WritableSignal<number>): Promise<ISimulationResult> {
    return new Promise((resolve) => {
        const { 
          resolution, nFrames, frameDurationUs, cameraHeight, initialPos, initialVel, restitution,
          laserMode, laserRepetitionFrequency, laserPulseWidthNs, laserAveragePower, laserPulseEnergy,
          timeResolutionPs, tdcMaxCount, rotationRadius, rotationSpeed, rotationCenter,
          bladePitch, targetType, ballMotionType, bladeMotionType, detectorYaw, detectorPitch, uploadedImage, waypoints, pathSpeeds
        } = params;
        
        let propellerPoints: {x: number, y: number, intensity: number}[] | null = null;
        if (targetType === 'Blade' && uploadedImage) {
            propellerPoints = getPropellerPointsFromImage(uploadedImage);
        }

        const maxTofUnits = tdcMaxCount;
        const emptyPixelValue = maxTofUnits + 2; // Value indicating no photon was detected
        const timeResolutionNs = timeResolutionPs * 1e-3;

        const totalPixels = resolution.width * resolution.height;
        const dataset = new Uint16Array(nFrames * totalPixels).fill(emptyPixelValue);
        const incidentPhotonMap = Array(resolution.height).fill(0).map(() => Array(resolution.width).fill(0));

        const fovRad = this.degreesToRadians(params.detectorFov);
        const fPixel = (resolution.width / 2) / Math.tan(fovRad / 2);
        const centerRow = resolution.height / 2;
        const centerCol = resolution.width / 2;

        let pos = { ...initialPos };
        let vel = { ...initialVel };
        const g = 9.8;
        const dt = frameDurationUs * 1e-6; // Frame duration in seconds

        let totalDetectedPhotons = 0;
        let totalIncidentPhotons = 0;
        let maxIncidentPhotonsPerPixel = 0;
        const signalCoordinates: { row: number; col: number }[] = [];
        let currentFrame = 0;
        const CHUNK_SIZE = 5000; // Process frames in chunks to avoid blocking the UI thread
        
        const pulsesPerFrameAvg = laserRepetitionFrequency * dt;
        const isRotating = ballMotionType === 'Rotation';
        const omega = rotationSpeed * 2 * Math.PI / 60; // Convert RPM to rad/s
        
        // --- Detector Rotation Setup ---
        const yawRad = this.degreesToRadians(detectorYaw);
        const pitchRad = this.degreesToRadians(detectorPitch);
        const cosYaw = Math.cos(-yawRad); // Use negative angle for inverse rotation
        const sinYaw = Math.sin(-yawRad);
        const cosPitch = Math.cos(-pitchRad);
        const sinPitch = Math.sin(-pitchRad);

        // --- Path Planning Pre-calculation ---
        const pathSegments: { start: {x,y,z}, end: {x,y,z}, distance: number, duration: number, cumulativeTime: number }[] = [];
        if (targetType === 'Blade' && bladeMotionType === 'Path' && waypoints.length > 1) {
            let cumulativeTime = 0;
            for (let i = 0; i < waypoints.length - 1; i++) {
                const start = waypoints[i].pos;
                const end = waypoints[i+1].pos;
                const speed = pathSpeeds[i] || 1;
                const dx = end.x - start.x;
                const dy = end.y - start.y;
                const dz = end.z - start.z;
                const distance = Math.sqrt(dx*dx + dy*dy + dz*dz);
                const duration = speed > 0 ? distance / speed : 0;
                pathSegments.push({ start, end, distance, duration, cumulativeTime });
                cumulativeTime += duration;
            }
        }

        const processSignalChunk = () => {
            const endFrame = Math.min(currentFrame + CHUNK_SIZE, nFrames);

            for (let frameIdx = currentFrame; frameIdx < endFrame; frameIdx++) {
                const timeNow = frameIdx * dt;

                const baseEffectiveReflectivity = params.reflectivity;
                
                const pixelPhysicsInfo: { row: number, col: number, intensityMultiplier: number }[] = [];
                let projectedRadiusInPixels = 0;
                let ballCenterRow = 0;
                let ballCenterCol = 0;
                let localPos: { x: number; y: number; z: number };

                if (targetType === 'Ball') {
                    if (isRotating) {
                        pos.x = rotationCenter.x + rotationRadius * Math.cos(omega * timeNow);
                        pos.y = initialPos.y;
                        pos.z = rotationCenter.z + rotationRadius * Math.sin(omega * timeNow);
                    } else { // Gravity
                        pos.x += vel.x * dt;
                        pos.y += vel.y * dt;
                        pos.z += vel.z * dt;
                        vel.y -= g * dt;
                        if (pos.y <= 0) {
                            pos.y = 0;
                            vel.y = -vel.y * restitution;
                        }
                    }
                } else if (targetType === 'Blade') {
                    if (bladeMotionType === 'Path') {
                        if (pathSegments.length > 0) {
                            let currentSegment = pathSegments[pathSegments.length - 1]; // Default to last segment if time exceeds total
                            for(const seg of pathSegments) {
                                if (timeNow < seg.cumulativeTime + seg.duration) {
                                    currentSegment = seg;
                                    break;
                                }
                            }
                            const timeInSegment = timeNow - currentSegment.cumulativeTime;
                            let t = currentSegment.duration > 0 ? Math.max(0, Math.min(1, timeInSegment / currentSegment.duration)) : 1.0;
                            
                            pos.x = currentSegment.start.x + (currentSegment.end.x - currentSegment.start.x) * t;
                            pos.y = currentSegment.start.y + (currentSegment.end.y - currentSegment.start.y) * t;
                            pos.z = currentSegment.start.z + (currentSegment.end.z - currentSegment.start.z) * t;
                        } else {
                            pos = waypoints[0]?.pos || {x:0, y:2, z:0};
                        }
                    } else { // Fixed motion
                        pos = { ...initialPos };
                    }
                }

                // Transform position to camera-local space
                const relativePos = { x: pos.x, y: pos.y - cameraHeight, z: pos.z };
                const yawedPos = { x: relativePos.x * cosYaw - relativePos.z * sinYaw, y: relativePos.y, z: relativePos.x * sinYaw + relativePos.z * cosYaw };
                localPos = { x: yawedPos.x, y: yawedPos.y * cosPitch - yawedPos.z * sinPitch, z: yawedPos.y * sinPitch + yawedPos.z * cosPitch };
                if (localPos.z <= 0) continue;

                if (targetType === 'Ball') {
                    const TENNIS_BALL_RADIUS = 0.0335;
                    const distance = Math.sqrt(localPos.x**2 + localPos.y**2 + localPos.z**2);
                    projectedRadiusInPixels = fPixel * (TENNIS_BALL_RADIUS / distance);
                    ballCenterRow = centerRow - fPixel * (localPos.y / localPos.z);
                    ballCenterCol = centerCol + fPixel * (localPos.x / localPos.z);

                    const minRow = Math.floor(ballCenterRow - projectedRadiusInPixels);
                    const maxRow = Math.ceil(ballCenterRow + projectedRadiusInPixels);
                    const minCol = Math.floor(ballCenterCol - projectedRadiusInPixels);
                    const maxCol = Math.ceil(ballCenterCol + projectedRadiusInPixels);
                    
                    for (let r = minRow; r <= maxRow; r++) {
                        for (let c = minCol; c <= maxCol; c++) {
                            const distSq = ((r + 0.5) - ballCenterRow)**2 + ((c + 0.5) - ballCenterCol)**2;
                            if (distSq <= projectedRadiusInPixels**2) {
                                if (r >= 0 && r < resolution.height && c >= 0 && c < resolution.width) {
                                    pixelPhysicsInfo.push({ row: r, col: c, intensityMultiplier: 1.0 });
                                }
                            }
                        }
                    }

                } else if (targetType === 'Blade') {
                    const angle = omega * timeNow;
                    const cosA = Math.cos(angle);
                    const sinA = Math.sin(angle);
                    const bladePitchRad = this.degreesToRadians(bladePitch);
                    const cosP = Math.cos(bladePitchRad);
                    const sinP = Math.sin(bladePitchRad);

                    if (propellerPoints && propellerPoints.length > 0 && uploadedImage) {
                        const bladeLength = rotationRadius;
                        const aspectRatio = uploadedImage.width / uploadedImage.height;
                        const geomWidth = uploadedImage.width >= uploadedImage.height ? bladeLength : bladeLength * aspectRatio;
                        const geomHeight = uploadedImage.width >= uploadedImage.height ? bladeLength / aspectRatio : bladeLength;
                        const pixelMap = new Map<string, { count: number, totalIntensity: number }>();
                        const pointsPerFrame = Math.min(propellerPoints.length, 250);

                        for (let i = 0; i < pointsPerFrame; i++) {
                            const p = propellerPoints[Math.floor(Math.random() * propellerPoints.length)];
                            const point_x_flat = p.x * geomWidth;
                            const point_z_flat = p.y * geomHeight;
                            const yawed_x = point_x_flat * cosA - point_z_flat * sinA;
                            const yawed_z = point_x_flat * sinA + point_z_flat * cosA;
                            const final_rotated_x = yawed_x;
                            const final_rotated_y = -yawed_z * sinP;
                            const final_rotated_z = yawed_z * cosP;
                            const worldPos = { x: pos.x + final_rotated_x, y: pos.y + final_rotated_y, z: pos.z + final_rotated_z };
                            
                            const relativePtPos = { x: worldPos.x, y: worldPos.y - cameraHeight, z: worldPos.z };
                            const yawedPtPos = { x: relativePtPos.x * cosYaw - relativePtPos.z * sinYaw, y: relativePtPos.y, z: relativePtPos.x * sinYaw + relativePtPos.z * cosYaw };
                            const pointLocalPos = { x: yawedPtPos.x, y: yawedPtPos.y * cosPitch - yawedPtPos.z * sinPitch, z: yawedPtPos.y * sinPitch + yawedPtPos.z * cosPitch };
                            
                            if (pointLocalPos.z > 0.1) {
                                const c = centerCol + fPixel * (pointLocalPos.x / pointLocalPos.z);
                                const r = centerRow - fPixel * (pointLocalPos.y / pointLocalPos.z);
                                const final_r = Math.round(r); const final_c = Math.round(c);
                                if (final_r >= 0 && final_r < resolution.height && final_c >= 0 && final_c < resolution.width) {
                                    const key = `${final_r},${final_c}`;
                                    const entry = pixelMap.get(key);
                                    if (entry) {
                                        entry.count++;
                                        entry.totalIntensity += p.intensity;
                                    } else {
                                        pixelMap.set(key, { count: 1, totalIntensity: p.intensity });
                                    }
                                }
                            }
                        }
                        for (const [key, data] of pixelMap.entries()) {
                            const [row, col] = key.split(',').map(Number);
                            pixelPhysicsInfo.push({ row, col, intensityMultiplier: data.totalIntensity / data.count });
                        }
                    } else { // Generic Blade
                        if (rotationRadius <= 0) continue;
                        const bladeLength = rotationRadius; const bladeWidth = 0.05;
                        const corners_flat = [ { x: -bladeWidth / 2, z: 0 }, { x: bladeWidth / 2, z: 0 }, { x: bladeWidth / 2, z: bladeLength }, { x: -bladeWidth / 2, z: bladeLength } ];
                        const corners = corners_flat.map(p_flat => {
                            const p_yawed = { x: p_flat.x * cosA - p_flat.z * sinA, y: 0, z: p_flat.x * sinA + p_flat.z * cosA };
                            const p_final_rotated = { x: p_yawed.x, y: -p_yawed.z * sinP, z: p_yawed.z * cosP };
                            return { x: pos.x + p_final_rotated.x, y: pos.y + p_final_rotated.y, z: pos.z + p_final_rotated.z };
                        });
                        const projectedCorners = corners.map(p => {
                            const relativePtPos = { x: p.x, y: p.y - cameraHeight, z: p.z };
                            const yawedPtPos = { x: relativePtPos.x * cosYaw - relativePtPos.z * sinYaw, y: relativePtPos.y, z: relativePtPos.x * sinYaw + relativePtPos.z * cosYaw };
                            const cornerLocalPos = { x: yawedPtPos.x, y: yawedPtPos.y * cosPitch - yawedPtPos.z * sinPitch, z: yawedPtPos.y * sinPitch + yawedPtPos.z * cosPitch };
                            if (cornerLocalPos.z <= 0.1) return null;
                            return { col: centerCol + fPixel * (cornerLocalPos.x / cornerLocalPos.z), row: centerRow - fPixel * (cornerLocalPos.y / cornerLocalPos.z) };
                        }).filter(p => p !== null) as {col: number, row: number}[];

                        if (projectedCorners.length < 4) continue;
                        const minRow = Math.max(0, Math.floor(Math.min(...projectedCorners.map(p => p.row))));
                        const maxRow = Math.min(resolution.height - 1, Math.ceil(Math.max(...projectedCorners.map(p => p.row))));
                        const minCol = Math.max(0, Math.floor(Math.min(...projectedCorners.map(p => p.col))));
                        const maxCol = Math.min(resolution.width - 1, Math.ceil(Math.max(...projectedCorners.map(p => p.col))));
                        for (let r = minRow; r <= maxRow; r++) {
                            for (let c = minCol; c <= maxCol; c++) {
                                if (isPointInPolygon({col: c + 0.5, row: r + 0.5}, projectedCorners)) {
                                     pixelPhysicsInfo.push({ row: r, col: c, intensityMultiplier: 1.0 });
                                }
                            }
                        }
                    }
                }
                
                if (pixelPhysicsInfo.length > 0) {
                    const distance = Math.sqrt(localPos.x**2 + localPos.y**2 + localPos.z**2);

                    for (const info of pixelPhysicsInfo) {
                        const { detectorFov, systemEfficiency, apertureDiameter, quantumEfficiency, laserWavelengthNm } = params;
                        const datasetIndex = frameIdx * totalPixels + info.row * resolution.width + info.col;
                        if (dataset[datasetIndex] !== emptyPixelValue) continue;

                        const beamHalfAngle = this.degreesToRadians(detectorFov) / 2;
                        const beamRadiusAtTarget = distance * Math.tan(beamHalfAngle);
                        const pixelAngularSize = fovRad / resolution.width;
                        const r_dist_from_beam_center = distance * Math.tan(Math.sqrt(((info.col + 0.5) - centerCol)**2 + ((info.row + 0.5) - centerRow)**2) * pixelAngularSize);
                        
                        let brdfWeight = 1.0;
                        if (targetType === 'Ball' && projectedRadiusInPixels > 1e-6) {
                            const distFromBallCenterSq = ((info.row + 0.5) - ballCenterRow)**2 + ((info.col + 0.5) - ballCenterCol)**2;
                            const normalizedDist = Math.sqrt(distFromBallCenterSq) / projectedRadiusInPixels;
                            if (normalizedDist <= 1) brdfWeight = Math.sqrt(1 - normalizedDist**2); else continue;
                        }
                        
                        const effectiveReflectivity = baseEffectiveReflectivity * info.intensityMultiplier;

                        if (laserMode === 'Pulsed') {
                            const numPulsesInFrame = Math.floor(pulsesPerFrameAvg) + (Math.random() < (pulsesPerFrameAvg % 1) ? 1 : 0);
                            if (numPulsesInFrame === 0) continue;

                            const E_ph = this.physicsService.getPhotonEnergy(laserWavelengthNm);
                            const N_tx_total = laserPulseEnergy / E_ph;
                            const irradiance_photons_per_m2 = (2 * N_tx_total / (Math.PI * beamRadiusAtTarget**2)) * Math.exp(-2 * r_dist_from_beam_center**2 / beamRadiusAtTarget**2);
                            const pixelAreaOnTarget = (distance * pixelAngularSize)**2;
                            const photonsHittingPixelArea = irradiance_photons_per_m2 * pixelAreaOnTarget;
                            const photonsReflected = photonsHittingPixelArea * effectiveReflectivity * brdfWeight;
                            const A_rx = Math.PI * (apertureDiameter / 2)**2;
                            const solid_angle_receiver = A_rx / (distance**2);
                            const photonsCollected = photonsReflected * solid_angle_receiver / Math.PI;
                            const transmittance = this.physicsService.calculateAtmosphericTransmittance(params, distance);
                            const perPulseExpectedPhotons = photonsCollected * systemEfficiency * quantumEfficiency * transmittance;

                            const totalIncidentForFrame = perPulseExpectedPhotons * numPulsesInFrame;
                            totalIncidentPhotons += totalIncidentForFrame;
                            incidentPhotonMap[info.row][info.col] += totalIncidentForFrame;
                            maxIncidentPhotonsPerPixel = Math.max(maxIncidentPhotonsPerPixel, perPulseExpectedPhotons);

                            if (1 - Math.pow(1 - (1 - Math.exp(-perPulseExpectedPhotons)), numPulsesInFrame) > Math.random()) {
                                totalDetectedPhotons++;
                                const pulseOffsetNs = (numPulsesInFrame > 1 ? Math.floor(Math.random() * numPulsesInFrame) : 0) * (1e9 / laserRepetitionFrequency);
                                const tofNs = (2 * distance / 3e8) * 1e9 + Math.random() * laserPulseWidthNs + pulseOffsetNs;
                                const tofUnits = Math.floor(tofNs / timeResolutionNs);
                                
                                const finalRow = Math.round(info.row + this.randn_bm() * 0.2);
                                const finalCol = Math.round(info.col + this.randn_bm() * 0.2);

                                if (finalRow >= 0 && finalRow < resolution.height && finalCol >= 0 && finalCol < resolution.width && tofUnits > 0 && tofUnits < maxTofUnits) {
                                    const finalIndex = frameIdx * totalPixels + finalRow * resolution.width + finalCol;
                                    if (dataset[finalIndex] === emptyPixelValue) {
                                        dataset[finalIndex] = tofUnits;
                                        signalCoordinates.push({ row: finalRow, col: finalCol });
                                    }
                                }
                            }
                        } else { // CW Mode
                            const E_ph = this.physicsService.getPhotonEnergy(laserWavelengthNm);
                            const irradiance_photons_per_sec_per_m2 = (2 * (laserAveragePower / E_ph) / (Math.PI * beamRadiusAtTarget**2)) * Math.exp(-2 * r_dist_from_beam_center**2 / beamRadiusAtTarget**2);
                            const photonsHittingPixelArea_per_sec = irradiance_photons_per_sec_per_m2 * (distance * pixelAngularSize)**2;
                            const photonsReflected_per_sec = photonsHittingPixelArea_per_sec * effectiveReflectivity * brdfWeight;
                            const photonsCollected_per_sec = photonsReflected_per_sec * (Math.PI * (apertureDiameter / 2)**2) / (distance**2) / Math.PI;
                            const lambda_frame = photonsCollected_per_sec * systemEfficiency * quantumEfficiency * this.physicsService.calculateAtmosphericTransmittance(params, distance) * dt;

                            totalIncidentPhotons += lambda_frame;
                            incidentPhotonMap[info.row][info.col] += lambda_frame;
                            maxIncidentPhotonsPerPixel = Math.max(maxIncidentPhotonsPerPixel, lambda_frame);

                            if (1 - Math.exp(-lambda_frame) > Math.random()) {
                                totalDetectedPhotons++;
                                const tofUnits = Math.floor(Math.random() * maxTofUnits);
                                const finalRow = Math.round(info.row + this.randn_bm() * 0.2);
                                const finalCol = Math.round(info.col + this.randn_bm() * 0.2);
                                if (finalRow >= 0 && finalRow < resolution.height && finalCol >= 0 && finalCol < resolution.width && tofUnits > 0 && tofUnits < maxTofUnits) {
                                    const finalIndex = frameIdx * totalPixels + finalRow * resolution.width + finalCol;
                                    if (dataset[finalIndex] === emptyPixelValue) {
                                        dataset[finalIndex] = tofUnits;
                                        signalCoordinates.push({ row: finalRow, col: finalCol });
                                    }
                                }
                            }
                        }
                    }
                }
            }

            currentFrame = endFrame;
            progress.set(Math.round((currentFrame / nFrames) * 95));

            if (currentFrame < nFrames) {
                setTimeout(processSignalChunk, 0);
            } else {
                addNoiseAndFinalize();
            }
        };

        const addNoiseAndFinalize = () => {
            const noisePerFrame = this.physicsService.calculateBackgroundNoise(params);
            const noisePerPixelPerFrame = noisePerFrame / totalPixels;
            const noiseDetectionProbability = 1 - Math.exp(-noisePerPixelPerFrame);
            
            let totalNoiseEvents = 0;
            let currentNoiseFrame = 0;
            const NOISE_CHUNK_SIZE = 10000;

            const processNoiseChunk = () => {
                const endFrame = Math.min(currentNoiseFrame + NOISE_CHUNK_SIZE, nFrames);

                for (let frameIdx = currentNoiseFrame; frameIdx < endFrame; frameIdx++) {
                    for (let pixelIdx = 0; pixelIdx < totalPixels; pixelIdx++) {
                        const datasetIndex = frameIdx * totalPixels + pixelIdx;
                        if (dataset[datasetIndex] === emptyPixelValue) {
                            if (Math.random() < noiseDetectionProbability) {
                                const nTof = Math.floor(Math.random() * (maxTofUnits - 1)) + 1;
                                dataset[datasetIndex] = nTof;
                                totalNoiseEvents++;
                            }
                        }
                    }
                }
                
                currentNoiseFrame = endFrame;
                const signalProgress = 95;
                const noiseProgress = (currentNoiseFrame / nFrames) * 5;
                progress.set(signalProgress + Math.round(noiseProgress));

                if (currentNoiseFrame < nFrames) {
                    setTimeout(processNoiseChunk, 0);
                } else {
                    progress.set(100);
                    resolve({ 
                        dataset, 
                        detectedPhotons: totalDetectedPhotons, 
                        noiseEvents: totalNoiseEvents, 
                        signalCoordinates,
                        incidentPhotons: totalIncidentPhotons,
                        incidentPhotonMap: incidentPhotonMap,
                        maxIncidentPhotonsPerPixel: maxIncidentPhotonsPerPixel,
                    });
                }
            };
            
            processNoiseChunk();
        };

        setTimeout(processSignalChunk, 50);
    });
  }
}