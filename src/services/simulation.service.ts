
import { Injectable, inject } from '@angular/core';
import { ISimulationParams, ISimulationResult, IPhotonEvent, IGroundTruthData } from '../models/simulation-params.model';
import { PhysicsService } from './physics.service';
import { WritableSignal } from '@angular/core';
import { applyDeadTimeRate } from './spectral-response.service';
import {
    IVec3,
    monostaticSurfaceReturnFactor,
    normalizeVec3,
    orientNormalToward,
    polygonNormal,
} from './reflectance.service';
import { backgroundTemporalDriftFactor, makeBackgroundSpatialMap } from './scene-background.service';
import { gaussianRandom, samplePoissonCount } from './photon-sampling.service';

/**
 * Checks if a point is inside a polygon using the ray casting algorithm.
 */
function isPointInPolygon(point: { col: number, row: number }, polygon: { col: number, row: number }[]): boolean {
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

/**
 * Calculates the area of a polygon using the Shoelace formula.
 */
function calculatePolygonArea(polygon: { col: number, row: number }[]): number {
    let area = 0;
    for (let i = 0; i < polygon.length; i++) {
        const j = (i + 1) % polygon.length;
        area += polygon[i].col * polygon[j].row;
        area -= polygon[j].col * polygon[i].row;
    }
    return Math.abs(area) / 2.0;
}

function surfaceReturnFactorFromCorners(worldCorners: IVec3[], targetToDetectorUnit: IVec3, specularGain = 0): number {
    const normal = orientNormalToward(polygonNormal(worldCorners), targetToDetectorUnit);
    return monostaticSurfaceReturnFactor(normal, targetToDetectorUnit, {
        specularGain,
        specularWidthDeg: 6,
    });
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

    const allPoints: { x: number, y: number, intensity: number }[] = [];

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
    let points: { x: number, y: number, intensity: number }[];
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
    for (const p of points) {
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
        return gaussianRandom();
    }

    private samplePoisson(lambda: number): number {
        return samplePoissonCount(lambda, Math.random, () => this.randn_bm());
    }

    private getRecordedDroneSampleAtTime(
        timeS: number,
        samples: NonNullable<ISimulationParams['recordedDroneTrajectory']>,
    ): NonNullable<ISimulationParams['recordedDroneTrajectory']>[number] {
        if (samples.length <= 1) return samples[0];
        if (timeS <= samples[0].time) return samples[0];
        const last = samples[samples.length - 1];
        if (timeS >= last.time) return last;

        let lo = 0;
        let hi = samples.length - 1;
        while (hi - lo > 1) {
            const mid = Math.floor((lo + hi) / 2);
            if (samples[mid].time <= timeS) lo = mid;
            else hi = mid;
        }

        const start = samples[lo];
        const end = samples[hi];
        const alpha = (timeS - start.time) / Math.max(1e-6, end.time - start.time);
        return {
            time: timeS,
            pos: {
                x: start.pos.x + (end.pos.x - start.pos.x) * alpha,
                y: start.pos.y + (end.pos.y - start.pos.y) * alpha,
                z: start.pos.z + (end.pos.z - start.pos.z) * alpha,
            },
            yawDeg: start.yawDeg + (end.yawDeg - start.yawDeg) * alpha,
            pitchDeg: start.pitchDeg + (end.pitchDeg - start.pitchDeg) * alpha,
            rollDeg: start.rollDeg + (end.rollDeg - start.rollDeg) * alpha,
            propellerRpms: start.propellerRpms.map((rpm, index) => (
                rpm + ((end.propellerRpms[index] ?? rpm) - rpm) * alpha
            )) as [number, number, number, number],
        };
    }

    private recordPhotonEvent(
        dataset: Uint16Array,
        emptyPixelValue: number,
        frameIdx: number,
        totalPixels: number,
        width: number,
        height: number,
        maxTofUnits: number,
        frameDurationUs: number,
        timeResolutionPs: number,
        row: number,
        col: number,
        tofUnits: number,
        signalCoordinates: { row: number; col: number }[],
        photonEvents: IPhotonEvent[],
    ): void {
        const finalRow = Math.round(row + this.randn_bm() * 0.2);
        const finalCol = Math.round(col + this.randn_bm() * 0.2);
        if (finalRow < 0 || finalRow >= height || finalCol < 0 || finalCol >= width || tofUnits <= 0 || tofUnits >= maxTofUnits) {
            return;
        }

        const finalIndex = frameIdx * totalPixels + finalRow * width + finalCol;
        if (dataset[finalIndex] === emptyPixelValue) {
            dataset[finalIndex] = tofUnits;
            signalCoordinates.push({ row: finalRow, col: finalCol });
        }
        photonEvents.push({
            timestamp: frameIdx * frameDurationUs * 1000 + tofUnits * timeResolutionPs / 1000,
            x: finalCol,
            y: finalRow,
            tof: tofUnits,
        });
    }

    public generateData(params: ISimulationParams, progress: WritableSignal<number>): Promise<ISimulationResult> {
        return new Promise((resolve) => {
            const {
                resolution, nFrames, frameDurationUs, cameraHeight, initialPos, initialVel, restitution,
                laserMode, laserRepetitionFrequency, laserPulseWidthNs, laserAveragePower, laserPulseEnergy,
                timeResolutionPs, tdcMaxCount, rotationRadius, rotationSpeed, rotationCenter,
                bladePitch, targetType, ballMotionType, bladeMotionType, detectorYaw, detectorPitch, uploadedImage, waypoints, pathSpeeds, pathRotationSpeeds, droneScale
            } = params;

            let propellerPoints: { x: number, y: number, intensity: number }[] | null = null;
            if (targetType === 'Blade' && uploadedImage) {
                propellerPoints = getPropellerPointsFromImage(uploadedImage);
            }

            const maxTofUnits = tdcMaxCount;
            const emptyPixelValue = maxTofUnits + 2; // Value indicating no photon was detected
            const timeResolutionNs = timeResolutionPs * 1e-3;
            const timingJitterSigmaNs = Math.max(
                params.timingJitterNs ?? 0,
                (params.irfFwhmPs ?? 0) * 1e-3 / 2.355,
            );

            const totalPixels = resolution.width * resolution.height;
            const dataset = new Uint16Array(nFrames * totalPixels).fill(emptyPixelValue);
            const incidentPhotonMap = Array(resolution.height).fill(0).map(() => Array(resolution.width).fill(0));
            const detectorPreset = params.detectorPreset;
            const pdeMap = Array(resolution.height).fill(0).map(() => Array(resolution.width).fill(1));
            const darkRateMap = Array(resolution.height).fill(0).map(() => Array(resolution.width).fill(params.darkCountRate));
            const backgroundSpatialMap = makeBackgroundSpatialMap(resolution.height, resolution.width, detectorPreset ? 0.06 : 0.03, 0.07, -0.035);
            if (detectorPreset) {
                for (let r = 0; r < resolution.height; r++) {
                    for (let c = 0; c < resolution.width; c++) {
                        pdeMap[r][c] = Math.max(0.5, Math.min(1.5, 1 + this.randn_bm() * detectorPreset.pdeNonuniformSigma));
                        darkRateMap[r][c] = Math.max(1e-10, params.darkCountRate * (1 + this.randn_bm() * detectorPreset.darkCountSigma));
                    }
                }
                const hotPixels = Math.round(totalPixels * detectorPreset.hotPixelFraction);
                for (let i = 0; i < hotPixels; i++) {
                    const idx = Math.floor(Math.random() * totalPixels);
                    const r = Math.floor(idx / resolution.width);
                    const c = idx % resolution.width;
                    pdeMap[r][c] = Math.min(1.5 * detectorPreset.hotPixelScale, pdeMap[r][c] * detectorPreset.hotPixelScale);
                    darkRateMap[r][c] *= detectorPreset.hotPixelScale;
                }
            }

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
            const omega = rotationSpeed * 2 * Math.PI / 60; // Convert RPM to rad/s for Ball/Fallback

            // --- Detector Rotation Setup ---
            const yawRad = this.degreesToRadians(detectorYaw);
            const pitchRad = this.degreesToRadians(detectorPitch);
            const cosYaw = Math.cos(-yawRad); // Use negative angle for inverse rotation
            const sinYaw = Math.sin(-yawRad);
            const cosPitch = Math.cos(-pitchRad);
            const sinPitch = Math.sin(-pitchRad);

            // --- Path Planning Pre-calculation ---
            const recordedDroneTrajectory = targetType === 'Drone' ? (params.recordedDroneTrajectory ?? []) : [];
            const hasRecordedDroneTrajectory = recordedDroneTrajectory.length > 1;
            const pathSegments: { start: { x, y, z }, end: { x, y, z }, distance: number, duration: number, cumulativeTime: number, omega: number, startAngle: number }[] = [];
            if ((targetType === 'Blade' || targetType === 'Drone') && bladeMotionType === 'Path' && waypoints.length > 1) {
                let cumulativeTime = 0;
                let cumulativeAngle = 0;
                for (let i = 0; i < waypoints.length - 1; i++) {
                    const start = waypoints[i].pos;
                    const end = waypoints[i + 1].pos;
                    const speed = pathSpeeds[i] || 1;
                    // For simplified 'Blade' single value usage. Drone has separate calculation per motor.
                    const rpm = pathRotationSpeeds[i] || 0;
                    const segmentOmega = rpm * 2 * Math.PI / 60;
                    const dx = end.x - start.x;
                    const dy = end.y - start.y;
                    const dz = end.z - start.z;
                    const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
                    const duration = speed > 0 ? distance / speed : 0;
                    pathSegments.push({ start, end, distance, duration, cumulativeTime, omega: segmentOmega, startAngle: cumulativeAngle });
                    cumulativeTime += duration;
                    cumulativeAngle += duration * segmentOmega;
                }
            }

            // --- SIM-002: Photon Events Collection ---
            const photonEvents: IPhotonEvent[] = [];

            // --- Ground-truth rotation pre-generation ---
            const groundTruthTimes: number[] = new Array(nFrames);
            const groundTruthFrequencies: number[] = new Array(nFrames);
            const groundTruthPhases: number[] = new Array(nFrames);
            const groundTruthDistances: number[] = new Array(nFrames);
            const accumulatedPhases: number[] = new Array(nFrames);

            for (let i = 0; i < nFrames; i++) {
                const timeNow = i * dt;
                groundTruthTimes[i] = timeNow;

                const angle = this.physicsService.calculateBladeRotationAngleAtTime(timeNow, params);
                groundTruthPhases[i] = angle % (2 * Math.PI);
                accumulatedPhases[i] = angle;

                if (i > 0) {
                    const dPhase = angle - accumulatedPhases[i - 1];
                    groundTruthFrequencies[i] = Math.abs(dPhase) / (2 * Math.PI * dt);
                } else {
                    groundTruthFrequencies[i] = rotationSpeed / 60;
                }
            }



            const processSignalChunk = () => {
                const endFrame = Math.min(currentFrame + CHUNK_SIZE, nFrames);

                for (let frameIdx = currentFrame; frameIdx < endFrame; frameIdx++) {
                    const timeNow = frameIdx * dt;

                    // Note: We now store effective reflectivity (fill factor * material reflectivity) in this array
                    const pixelPhysicsInfo: { row: number, col: number, effectiveReflectivity: number }[] = [];

                    let projectedRadiusInPixels = 0;
                    let ballCenterRow = 0;
                    let ballCenterCol = 0;
                    let localPos: { x: number; y: number; z: number };
                    let angle = 0;
                    let recordedDroneSample: NonNullable<ISimulationParams['recordedDroneTrajectory']>[number] | null = null;

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
                    } else if (targetType === 'Blade' || targetType === 'Drone') {
                        if (hasRecordedDroneTrajectory) {
                            const sample = this.getRecordedDroneSampleAtTime(timeNow, recordedDroneTrajectory);
                            recordedDroneSample = sample;
                            pos = sample.pos;
                            angle = accumulatedPhases[frameIdx];
                        } else if (bladeMotionType === 'Path') {
                            if (pathSegments.length > 0) {
                                let currentSegment = pathSegments[pathSegments.length - 1]; // Default to last segment if time exceeds total
                                for (const seg of pathSegments) {
                                    if (timeNow <= seg.cumulativeTime + seg.duration) {
                                        currentSegment = seg;
                                        break;
                                    }
                                }
                                const timeInSegment = timeNow - currentSegment.cumulativeTime;
                                let t = currentSegment.duration > 0 ? Math.max(0, Math.min(1, timeInSegment / currentSegment.duration)) : 1.0;

                                pos.x = currentSegment.start.x + (currentSegment.end.x - currentSegment.start.x) * t;
                                pos.y = currentSegment.start.y + (currentSegment.end.y - currentSegment.start.y) * t;
                                pos.z = currentSegment.start.z + (currentSegment.end.z - currentSegment.start.z) * t;
                                angle = currentSegment.startAngle + timeInSegment * currentSegment.omega; // For basic blade only
                            } else {
                                pos = waypoints[0]?.pos || { x: 0, y: 2, z: 0 };
                                angle = 0;
                            }
                        } else { // Fixed motion
                            pos = { ...initialPos };
                            angle = accumulatedPhases[frameIdx];
                        }
                    }

                    // Transform position to camera-local space
                    const relativePos = { x: pos.x, y: pos.y - cameraHeight, z: pos.z };
                    const yawedPos = { x: relativePos.x * cosYaw - relativePos.z * sinYaw, y: relativePos.y, z: relativePos.x * sinYaw + relativePos.z * cosYaw };
                    localPos = { x: yawedPos.x, y: yawedPos.y * cosPitch - yawedPos.z * sinPitch, z: yawedPos.y * sinPitch + yawedPos.z * cosPitch };
                    if (localPos.z <= 0) continue;

                    // SIM-003: Record ground truth distance for ToF validation
                    groundTruthDistances[frameIdx] = localPos.z;
                    const targetToDetectorUnit = normalizeVec3({ x: -pos.x, y: cameraHeight - pos.y, z: -pos.z });

                    if (targetType === 'Ball') {
                        const TENNIS_BALL_RADIUS = 0.0335;
                        const distance = Math.sqrt(localPos.x ** 2 + localPos.y ** 2 + localPos.z ** 2);
                        projectedRadiusInPixels = fPixel * (TENNIS_BALL_RADIUS / distance);
                        ballCenterRow = centerRow - fPixel * (localPos.y / localPos.z);
                        ballCenterCol = centerCol + fPixel * (localPos.x / localPos.z);
                        const ballReflectivity = params.reflectivity;

                        // Sub-pixel logic: If the ball is very small (< 0.5 pixel diameter), treat as area-weighted point source
                        if (projectedRadiusInPixels < 0.25) {
                            const area = Math.PI * projectedRadiusInPixels * projectedRadiusInPixels;
                            const r = Math.round(ballCenterRow);
                            const c = Math.round(ballCenterCol);
                            if (r >= 0 && r < resolution.height && c >= 0 && c < resolution.width) {
                                pixelPhysicsInfo.push({ row: r, col: c, effectiveReflectivity: area * ballReflectivity });
                            }
                        } else {
                            // Standard Rasterization
                            const minRow = Math.floor(ballCenterRow - projectedRadiusInPixels);
                            const maxRow = Math.ceil(ballCenterRow + projectedRadiusInPixels);
                            const minCol = Math.floor(ballCenterCol - projectedRadiusInPixels);
                            const maxCol = Math.ceil(ballCenterCol + projectedRadiusInPixels);

                            for (let r = minRow; r <= maxRow; r++) {
                                for (let c = minCol; c <= maxCol; c++) {
                                    const distSq = ((r + 0.5) - ballCenterRow) ** 2 + ((c + 0.5) - ballCenterCol) ** 2;
                                    if (distSq <= projectedRadiusInPixels ** 2) {
                                        if (r >= 0 && r < resolution.height && c >= 0 && c < resolution.width) {
                                            pixelPhysicsInfo.push({ row: r, col: c, effectiveReflectivity: 1.0 * ballReflectivity });
                                        }
                                    }
                                }
                            }
                        }

                    } else if (targetType === 'Blade') {
                        const cosA = Math.cos(angle);
                        const sinA = Math.sin(angle);
                        const bladePitchRad = this.degreesToRadians(bladePitch);
                        const cosP = Math.cos(bladePitchRad);
                        const sinP = Math.sin(bladePitchRad);
                        const bladeReflectivity = params.reflectivity;
                        const bladeNormal = orientNormalToward({ x: 0, y: cosP, z: -sinP }, targetToDetectorUnit);
                        const bladeSurfaceFactor = monostaticSurfaceReturnFactor(bladeNormal, targetToDetectorUnit, {
                            specularGain: 0.2,
                            specularWidthDeg: 8,
                        });

                        if (propellerPoints && propellerPoints.length > 0 && uploadedImage) {
                            // Point cloud logic
                            const bladeLength = rotationRadius;
                            const aspectRatio = uploadedImage.width / uploadedImage.height;
                            const geomWidth = uploadedImage.width >= uploadedImage.height ? bladeLength : bladeLength * aspectRatio;
                            const geomHeight = uploadedImage.width >= uploadedImage.height ? bladeLength / aspectRatio : bladeLength;
                            const pixelMap = new Map<string, { count: number, totalIntensity: number }>();

                            // FIX: Use deterministic stride instead of random sampling to avoid frame-to-frame noise ("shimmer")
                            const pointsToSample = 500; // Increase sample count slightly
                            const stride = Math.max(1, Math.floor(propellerPoints.length / pointsToSample));
                            const startIdx = frameIdx % stride; // Small phase shift

                            for (let i = startIdx; i < propellerPoints.length; i += stride) {
                                const p = propellerPoints[i];
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
                                // Combine average intensity from image with material reflectivity
                                pixelPhysicsInfo.push({ row, col, effectiveReflectivity: (data.totalIntensity / data.count) * bladeReflectivity * bladeSurfaceFactor });
                            }
                        } else { // Generic Blade
                            if (rotationRadius <= 0) continue;
                            const bladeLength = rotationRadius; const bladeWidth = 0.05;
                            const corners_flat = [{ x: -bladeWidth / 2, z: 0 }, { x: bladeWidth / 2, z: 0 }, { x: bladeWidth / 2, z: bladeLength }, { x: -bladeWidth / 2, z: bladeLength }];
                            const corners = corners_flat.map(p_flat => {
                                const p_yawed = { x: p_flat.x * cosA - p_flat.z * sinA, y: 0, z: p_flat.x * sinA + p_flat.z * cosA };
                                const p_final_rotated = { x: p_yawed.x, y: -p_yawed.z * sinP, z: p_yawed.z * cosP };
                                return { x: pos.x + p_final_rotated.x, y: pos.y + p_final_rotated.y, z: pos.z + p_final_rotated.z };
                            });
                            const surfaceFactor = surfaceReturnFactorFromCorners(corners, targetToDetectorUnit, 0.2);

                            // Polygon Projection & Rasterization
                            const projectedCorners = corners.map(p => {
                                const relativePtPos = { x: p.x, y: p.y - cameraHeight, z: p.z };
                                const yawedPtPos = { x: relativePtPos.x * cosYaw - relativePtPos.z * sinYaw, y: relativePtPos.y, z: relativePtPos.x * sinYaw + relativePtPos.z * cosYaw };
                                const cornerLocalPos = { x: yawedPtPos.x, y: yawedPtPos.y * cosPitch - yawedPtPos.z * sinPitch, z: yawedPtPos.y * sinPitch + yawedPtPos.z * cosPitch };
                                if (cornerLocalPos.z <= 0.1) return null;
                                return { col: centerCol + fPixel * (cornerLocalPos.x / cornerLocalPos.z), row: centerRow - fPixel * (cornerLocalPos.y / cornerLocalPos.z) };
                            }).filter(p => p !== null) as { col: number, row: number }[];

                            if (projectedCorners.length >= 3) {
                                const polyArea = calculatePolygonArea(projectedCorners);
                                if (polyArea < 0.5) {
                                    // Sub-pixel Logic
                                    const centerCol = projectedCorners.reduce((sum, p) => sum + p.col, 0) / projectedCorners.length;
                                    const centerRow = projectedCorners.reduce((sum, p) => sum + p.row, 0) / projectedCorners.length;
                                    const r = Math.floor(centerRow);
                                    const c = Math.floor(centerCol);
                                    if (r >= 0 && r < resolution.height && c >= 0 && c < resolution.width) {
                                        const existing = pixelPhysicsInfo.find(p => p.row === r && p.col === c);
                                        const val = polyArea * bladeReflectivity * surfaceFactor;
                                        if (existing) existing.effectiveReflectivity += val;
                                        else pixelPhysicsInfo.push({ row: r, col: c, effectiveReflectivity: val });
                                    }
                                } else {
                                    // Rasterization Logic
                                    const minRow = Math.max(0, Math.floor(Math.min(...projectedCorners.map(p => p.row))));
                                    const maxRow = Math.min(resolution.height - 1, Math.ceil(Math.max(...projectedCorners.map(p => p.row))));
                                    const minCol = Math.max(0, Math.floor(Math.min(...projectedCorners.map(p => p.col))));
                                    const maxCol = Math.min(resolution.width - 1, Math.ceil(Math.max(...projectedCorners.map(p => p.col))));
                                    for (let r = minRow; r <= maxRow; r++) {
                                        for (let c = minCol; c <= maxCol; c++) {
                                            if (isPointInPolygon({ col: c + 0.5, row: r + 0.5 }, projectedCorners)) {
                                                pixelPhysicsInfo.push({ row: r, col: c, effectiveReflectivity: bladeReflectivity * surfaceFactor });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else if (targetType === 'Drone') {
                        // --- Drone Logic ---
                        const pitchRad = this.degreesToRadians(recordedDroneSample?.pitchDeg ?? params.dronePitchDeg ?? bladePitch);
                        const yawBodyRad = this.degreesToRadians(recordedDroneSample?.yawDeg ?? params.droneYawDeg ?? 0);
                        const rollBodyRad = this.degreesToRadians(recordedDroneSample?.rollDeg ?? params.droneRollDeg ?? 0);
                        const cosP = Math.cos(pitchRad);
                        const sinP = Math.sin(pitchRad);
                        const cosYBody = Math.cos(yawBodyRad);
                        const sinYBody = Math.sin(yawBodyRad);
                        const cosRBody = Math.cos(rollBodyRad);
                        const sinRBody = Math.sin(rollBodyRad);

                        const bodyReflectivity = params.reflectivity;
                        const propReflectivity = params.propellerReflectivity || bodyReflectivity;
                        const preset = params.dronePreset;

                        const scale = droneScale || 1.0;
                        const bodyHalfWidth = (preset ? preset.dimensions.widthM * 0.19 : 0.05 * scale);
                        const bodyHalfLength = (preset ? preset.dimensions.lengthM * 0.21 : 0.05 * scale);
                        const propWidth = preset ? Math.max(0.01, preset.propellerDiameterM * 0.08) : 0.02 * scale;
                        const propLen = preset ? preset.propellerDiameterM : 0.10 * scale;
                        const armHalfWidth = preset ? preset.dimensions.widthM * 0.36 : 0.05 * scale;
                        const armHalfLength = preset ? preset.dimensions.lengthM * 0.36 : 0.05 * scale;

                        // Propeller centers relative to drone center (Z+ is Front)
                        const propCenters = [
                            { x: armHalfWidth, z: armHalfLength, id: 1 },  // FR
                            { x: -armHalfWidth, z: armHalfLength, id: 2 }, // FL
                            { x: armHalfWidth, z: -armHalfLength, id: 3 }, // RR
                            { x: -armHalfWidth, z: -armHalfLength, id: 4 } // RL
                        ];

                        const components = [];

                        // 1. Drone Body (Square)
                        const bodyCornersLocal = [
                            { x: -bodyHalfWidth, z: -bodyHalfLength }, { x: bodyHalfWidth, z: -bodyHalfLength },
                            { x: bodyHalfWidth, z: bodyHalfLength }, { x: -bodyHalfWidth, z: bodyHalfLength }
                        ];
                        components.push({ type: 'body', corners: bodyCornersLocal });

                        // 2. Propellers
                        propCenters.forEach(pc => {
                            // Calculate rotation angle for this prop
                            const propAngle = this.physicsService.calculateBladeRotationAngleAtTime(timeNow, params, pc.id as 1 | 2 | 3 | 4);
                            const dir = (pc.id === 1 || pc.id === 4) ? 1 : -1; // Diagonal CW / CCW groups
                            const finalAngle = propAngle * dir;

                            const cA = Math.cos(finalAngle);
                            const sA = Math.sin(finalAngle);

                            // Prop shape: Rectangle centered at (0,0) local to prop shaft
                            const halfL = propLen / 2;
                            const halfW = propWidth / 2;
                            const pCorners = [
                                { x: -halfL, z: -halfW }, { x: halfL, z: -halfW },
                                { x: halfL, z: halfW }, { x: -halfL, z: halfW }
                            ];

                            // Rotate prop corners around shaft, then translate to body corner
                            const transformedCorners = pCorners.map(pcorn => ({
                                x: (pcorn.x * cA - pcorn.z * sA) + pc.x,
                                z: (pcorn.x * sA + pcorn.z * cA) + pc.z
                            }));

                            components.push({ type: 'prop', corners: transformedCorners });
                        });

                        // Project all components
                        for (const comp of components) {
                            const componentReflectivity = comp.type === 'prop' ? propReflectivity : bodyReflectivity;
                            const componentSpecularGain = comp.type === 'prop' ? 0.7 : 0.25;
                            const worldCorners = comp.corners.map(c => {
                                // Apply drone pitch, roll, and yaw before translating into world coordinates.
                                const pitchedY = -c.z * sinP;
                                const pitchedZ = c.z * cosP;
                                const rolledX = c.x * cosRBody - pitchedY * sinRBody;
                                const rolledY = c.x * sinRBody + pitchedY * cosRBody;
                                const yawedX = rolledX * cosYBody + pitchedZ * sinYBody;
                                const yawedZ = -rolledX * sinYBody + pitchedZ * cosYBody;
                                return { x: pos.x + yawedX, y: pos.y + rolledY, z: pos.z + yawedZ };
                            });
                            const surfaceFactor = surfaceReturnFactorFromCorners(worldCorners, targetToDetectorUnit, componentSpecularGain);

                            const projectedCorners = worldCorners.map(p => {
                                const relativePtPos = { x: p.x, y: p.y - cameraHeight, z: p.z };
                                const yawedPtPos = { x: relativePtPos.x * cosYaw - relativePtPos.z * sinYaw, y: relativePtPos.y, z: relativePtPos.x * sinYaw + relativePtPos.z * cosYaw };
                                const localP = { x: yawedPtPos.x, y: yawedPtPos.y * cosPitch - yawedPtPos.z * sinPitch, z: yawedPtPos.y * sinPitch + yawedPtPos.z * cosPitch };
                                if (localP.z <= 0.1) return null;
                                return { col: centerCol + fPixel * (localP.x / localP.z), row: centerRow - fPixel * (localP.y / localP.z) };
                            }).filter(p => p !== null) as { col: number, row: number }[];

                            if (projectedCorners.length < 3) continue;

                            const polyArea = calculatePolygonArea(projectedCorners);

                            // Sub-pixel / Small Area Logic
                            if (polyArea < 0.8) {
                                // Treat as area-weighted point source
                                const centerCol = projectedCorners.reduce((sum, p) => sum + p.col, 0) / projectedCorners.length;
                                const centerRow = projectedCorners.reduce((sum, p) => sum + p.row, 0) / projectedCorners.length;
                                const r = Math.floor(centerRow);
                                const c = Math.floor(centerCol);

                                if (r >= 0 && r < resolution.height && c >= 0 && c < resolution.width) {
                                    // Overlapping components (e.g. arm + prop) overlap in physics. 
                                    // Additive intensity approximation.
                                    const existing = pixelPhysicsInfo.find(p => p.row === r && p.col === c);
                                    const val = polyArea * componentReflectivity * surfaceFactor;
                                    if (existing) {
                                        existing.effectiveReflectivity += val;
                                    } else {
                                        pixelPhysicsInfo.push({ row: r, col: c, effectiveReflectivity: val });
                                    }
                                }
                            } else {
                                // Standard Rasterization
                                const minRow = Math.max(0, Math.floor(Math.min(...projectedCorners.map(p => p.row))));
                                const maxRow = Math.min(resolution.height - 1, Math.ceil(Math.max(...projectedCorners.map(p => p.row))));
                                const minCol = Math.max(0, Math.floor(Math.min(...projectedCorners.map(p => p.col))));
                                const maxCol = Math.min(resolution.width - 1, Math.ceil(Math.max(...projectedCorners.map(p => p.col))));

                                for (let r = minRow; r <= maxRow; r++) {
                                    for (let c = minCol; c <= maxCol; c++) {
                                        if (isPointInPolygon({ col: c + 0.5, row: r + 0.5 }, projectedCorners)) {
                                            // If pixel already added by another component, avoid duplicate for *binary* coverage logic.
                                            // However, for correct multi-material, we should really do Z-buffering or careful ordering.
                                            // Here we check if it exists. If not, add it.
                                            if (!pixelPhysicsInfo.some(info => info.row === r && info.col === c)) {
                                                pixelPhysicsInfo.push({ row: r, col: c, effectiveReflectivity: componentReflectivity * surfaceFactor });
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (pixelPhysicsInfo.length > 0) {
                        const distance = Math.sqrt(localPos.x ** 2 + localPos.y ** 2 + localPos.z ** 2);

                        for (const info of pixelPhysicsInfo) {
                            const { detectorFov, systemEfficiency, apertureDiameter, quantumEfficiency, laserWavelengthNm } = params;
                            const datasetIndex = frameIdx * totalPixels + info.row * resolution.width + info.col;
                            if (dataset[datasetIndex] !== emptyPixelValue) continue;

                            const beamHalfAngle = this.degreesToRadians(detectorFov) / 2;
                            const beamRadiusAtTarget = distance * Math.tan(beamHalfAngle);
                            const pixelAngularSize = fovRad / resolution.width;
                            const r_dist_from_beam_center = distance * Math.tan(Math.sqrt(((info.col + 0.5) - centerCol) ** 2 + ((info.row + 0.5) - centerRow) ** 2) * pixelAngularSize);

                            let brdfWeight = 1.0;
                            if (targetType === 'Ball' && projectedRadiusInPixels > 0.25) { // Only apply BRDF curve for resolved balls
                                const distFromBallCenterSq = ((info.row + 0.5) - ballCenterRow) ** 2 + ((info.col + 0.5) - ballCenterCol) ** 2;
                                const normalizedDist = Math.sqrt(distFromBallCenterSq) / projectedRadiusInPixels;
                                if (normalizedDist <= 1) brdfWeight = Math.sqrt(1 - normalizedDist ** 2); else continue;
                            }

                            // We use the pre-calculated effective reflectivity (MaterialReflectivity * FillFactor)
                            const effectiveReflectivity = info.effectiveReflectivity;
                            const pdeScale = pdeMap[info.row]?.[info.col] ?? 1;

                            if (laserMode === 'Pulsed') {
                                const numPulsesInFrame = Math.floor(pulsesPerFrameAvg) + (Math.random() < (pulsesPerFrameAvg % 1) ? 1 : 0);
                                if (numPulsesInFrame === 0) continue;

                                const E_ph = this.physicsService.getPhotonEnergy(laserWavelengthNm);
                                const N_tx_total = laserPulseEnergy / E_ph;
                                const irradiance_photons_per_m2 = (2 * N_tx_total / (Math.PI * beamRadiusAtTarget ** 2)) * Math.exp(-2 * r_dist_from_beam_center ** 2 / beamRadiusAtTarget ** 2);
                                const pixelAreaOnTarget = (distance * pixelAngularSize) ** 2;
                                const photonsHittingPixelArea = irradiance_photons_per_m2 * pixelAreaOnTarget;
                                const photonsReflected = photonsHittingPixelArea * effectiveReflectivity * brdfWeight;
                                const A_rx = Math.PI * (apertureDiameter / 2) ** 2;
                                const solid_angle_receiver = A_rx / (distance ** 2);
                                const photonsCollected = photonsReflected * solid_angle_receiver / Math.PI;
                                const transmittance = this.physicsService.calculateAtmosphericTransmittance(params, distance);
                                const idealPerPulseExpectedPhotons = photonsCollected * systemEfficiency * quantumEfficiency * pdeScale * transmittance;
                                const perPulseExpectedPhotons = applyDeadTimeRate(
                                    idealPerPulseExpectedPhotons * Math.max(laserRepetitionFrequency, 1),
                                    (params.deadTimeNs || 0) * 1e-9,
                                    'nonparalyzable',
                                ) / Math.max(laserRepetitionFrequency, 1);

                                const totalIncidentForFrame = perPulseExpectedPhotons * numPulsesInFrame;
                                totalIncidentPhotons += totalIncidentForFrame;
                                incidentPhotonMap[info.row][info.col] += totalIncidentForFrame;
                                maxIncidentPhotonsPerPixel = Math.max(maxIncidentPhotonsPerPixel, perPulseExpectedPhotons);

                                const signalCount = this.samplePoisson(totalIncidentForFrame);
                                totalDetectedPhotons += signalCount;
                                for (let eventIdx = 0; eventIdx < signalCount; eventIdx++) {
                                    const pulseIndex = numPulsesInFrame > 1 ? Math.floor(Math.random() * numPulsesInFrame) : 0;
                                    const pulseOffsetNs = pulseIndex * (1e9 / laserRepetitionFrequency);
                                    const jitterNs = timingJitterSigmaNs > 0 ? this.randn_bm() * timingJitterSigmaNs : 0;
                                    const tofNs = (2 * distance / 3e8) * 1e9 + Math.random() * laserPulseWidthNs + pulseOffsetNs + jitterNs;
                                    const tofUnits = Math.floor(tofNs / timeResolutionNs);
                                    this.recordPhotonEvent(
                                        dataset,
                                        emptyPixelValue,
                                        frameIdx,
                                        totalPixels,
                                        resolution.width,
                                        resolution.height,
                                        maxTofUnits,
                                        frameDurationUs,
                                        timeResolutionPs,
                                        info.row,
                                        info.col,
                                        tofUnits,
                                        signalCoordinates,
                                        photonEvents,
                                    );
                                }
                            } else { // CW Mode
                                const E_ph = this.physicsService.getPhotonEnergy(laserWavelengthNm);
                                const irradiance_photons_per_sec_per_m2 = (2 * (laserAveragePower / E_ph) / (Math.PI * beamRadiusAtTarget ** 2)) * Math.exp(-2 * r_dist_from_beam_center ** 2 / beamRadiusAtTarget ** 2);
                                const photonsHittingPixelArea_per_sec = irradiance_photons_per_sec_per_m2 * (distance * pixelAngularSize) ** 2;
                                const photonsReflected_per_sec = photonsHittingPixelArea_per_sec * effectiveReflectivity * brdfWeight;
                                const photonsCollected_per_sec = photonsReflected_per_sec * (Math.PI * (apertureDiameter / 2) ** 2) / (distance ** 2) / Math.PI;
                                const idealRate = photonsCollected_per_sec * systemEfficiency * quantumEfficiency * pdeScale * this.physicsService.calculateAtmosphericTransmittance(params, distance);
                                const lambda_frame = applyDeadTimeRate(idealRate, (params.deadTimeNs || 0) * 1e-9, 'nonparalyzable') * dt;

                                totalIncidentPhotons += lambda_frame;
                                incidentPhotonMap[info.row][info.col] += lambda_frame;
                                maxIncidentPhotonsPerPixel = Math.max(maxIncidentPhotonsPerPixel, lambda_frame);

                                const signalCount = this.samplePoisson(lambda_frame);
                                totalDetectedPhotons += signalCount;
                                for (let eventIdx = 0; eventIdx < signalCount; eventIdx++) {
                                    const tofUnits = Math.floor(Math.random() * maxTofUnits);
                                    this.recordPhotonEvent(
                                        dataset,
                                        emptyPixelValue,
                                        frameIdx,
                                        totalPixels,
                                        resolution.width,
                                        resolution.height,
                                        maxTofUnits,
                                        frameDurationUs,
                                        timeResolutionPs,
                                        info.row,
                                        info.col,
                                        tofUnits,
                                        signalCoordinates,
                                        photonEvents,
                                    );
                                }
                            }
                        }
                    }
                }

                currentFrame = endFrame;
                progress.set(Math.round((currentFrame / nFrames) * 95));

                if (currentFrame < nFrames) {
                    setTimeout(processSignalChunk, 0); // Reduced delay for faster processing
                } else {
                    addNoiseAndFinalize();
                }
            };

            const addNoiseAndFinalize = () => {
                const backgroundRates = this.physicsService.calculateBackgroundRatesPerPixel(params);

                let totalNoiseEvents = 0;
                let currentNoiseFrame = 0;
                const NOISE_CHUNK_SIZE = 10000;

                const processNoiseChunk = () => {
                    const endFrame = Math.min(currentNoiseFrame + NOISE_CHUNK_SIZE, nFrames);

                    for (let frameIdx = currentNoiseFrame; frameIdx < endFrame; frameIdx++) {
                        const timeNow = frameIdx * dt;
                        const temporalDrift = backgroundTemporalDriftFactor(timeNow, detectorPreset ? 0.10 : 0.04, detectorPreset ? 0.7 : 0.35);
                        for (let pixelIdx = 0; pixelIdx < totalPixels; pixelIdx++) {
                            const datasetIndex = frameIdx * totalPixels + pixelIdx;
                            const row = Math.floor(pixelIdx / resolution.width);
                            const col = pixelIdx % resolution.width;
                            const backgroundMu = backgroundRates.backgroundRateCps * temporalDrift * (backgroundSpatialMap[row]?.[col] ?? 1) * dt;
                            const darkMu = (darkRateMap[row]?.[col] ?? params.darkCountRate) * dt;
                            const noiseCount = this.samplePoisson(backgroundMu + darkMu);
                            if (noiseCount > 0) {
                                totalNoiseEvents += noiseCount;
                                if (dataset[datasetIndex] === emptyPixelValue) {
                                    const nTof = Math.floor(Math.random() * (maxTofUnits - 1)) + 1;
                                    dataset[datasetIndex] = nTof;
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

                        // Sort photon events by timestamp for proper event stream ordering
                        photonEvents.sort((a, b) => a.timestamp - b.timestamp);

                        resolve({
                            dataset,
                            detectedPhotons: totalDetectedPhotons,
                            noiseEvents: totalNoiseEvents,
                            signalCoordinates,
                            incidentPhotons: totalIncidentPhotons,
                            incidentPhotonMap: incidentPhotonMap,
                            maxIncidentPhotonsPerPixel: maxIncidentPhotonsPerPixel,
                            // Ground truth data for algorithm validation
                            groundTruthData: {
                                times: groundTruthTimes,
                                frequencies: groundTruthFrequencies,
                                phases: groundTruthPhases,
                                distances: groundTruthDistances  // SIM-003: Distance for ToF validation
                            },
                            // SIM-002: Photon event stream for particle filter
                            photonEvents: photonEvents
                        });
                    }
                };

                processNoiseChunk();
            };

            setTimeout(processSignalChunk, 50);
        });
    }
}
