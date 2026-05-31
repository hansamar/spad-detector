import { Injectable } from '@angular/core';
import { ISimulationParams } from '../models/simulation-params.model';
import { am0SolarIrradianceWM2Nm, relativeChannelResponse, spectralBackgroundScale } from './spectral-response.service';

export function atmosphericAttenuationCoefficientKm(wavelengthNm: number, visibilityKm = 23): number {
  const safeVisibilityKm = Math.max(0.5, visibilityKm);
  const safeWavelengthNm = Math.max(350, Math.min(1800, wavelengthNm));
  const q = safeVisibilityKm > 50
    ? 1.6
    : safeVisibilityKm > 6
      ? 1.3
      : 0.585 * Math.pow(safeVisibilityKm, 1 / 3);

  // Kruse/Koschmieder visibility model, alpha in km^-1.
  const aerosolAlpha = (3.912 / safeVisibilityKm) * Math.pow(safeWavelengthNm / 550, -q);
  const molecularFloor = 0.006 * Math.pow(safeWavelengthNm / 550, -4.08);
  return aerosolAlpha + molecularFloor;
}

export function hazeScatterScaleFromVisibility(visibilityKm = 23): number {
  const safeVisibilityKm = Math.max(0.5, visibilityKm);
  return 1 + Math.min(3, Math.max(0, 23 / safeVisibilityKm - 1)) * 0.22;
}

@Injectable({
  providedIn: 'root',
})
export class PhysicsService {
  // --- Physics Constants ---
  private readonly h = 6.626e-34; // Planck's constant
  private readonly c = 3e8; // Speed of light

  public getPhotonEnergy(wavelengthNm: number): number {
    const wavelengthM = wavelengthNm * 1e-9;
    return (this.h * this.c) / wavelengthM;
  }

  private getAtmosphericAttenuationCoefficient(wavelengthNm: number, visibilityKm = 23): number {
    return atmosphericAttenuationCoefficientKm(wavelengthNm, visibilityKm);
  }

  public calculateAtmosphericTransmittance(params: ISimulationParams, distanceM: number): number {
    if (!params.atmosphericAttenuationEnabled) {
        return 1.0;
    }

    const alpha_per_km = this.getAtmosphericAttenuationCoefficient(params.laserWavelengthNm, params.atmosphericVisibilityKm);
    const alpha_per_m = alpha_per_km / 1000;
    const pathLengthM = 2 * distanceM; // To the target and back

    // Beer-Lambert Law for atmospheric transmittance
    return Math.exp(-alpha_per_m * pathLengthM);
  }

  public calculateBackgroundNoise(params: ISimulationParams): number {
    const perPixel = this.calculateBackgroundRatesPerPixel(params);
    const totalPixels = params.resolution.width * params.resolution.height;
    return (perPixel.backgroundRateCps + perPixel.darkRateCps) * totalPixels * (params.frameDurationUs * 1e-6);
  }

  public calculateBackgroundRatesPerPixel(params: ISimulationParams): { backgroundRateCps: number; darkRateCps: number } {
    const hazeScatterScale = hazeScatterScaleFromVisibility(params.atmosphericVisibilityKm);

    if (params.detectorPreset) {
      const channelScale = relativeChannelResponse(params.laserWavelengthNm, params.filterBandwidth, 550, params.detectorPreset.filterBandwidthNm);
      const referenceSceneStrayCpsPerPixel = 350;
      const backgroundRateCps = referenceSceneStrayCpsPerPixel
        * spectralBackgroundScale('scene_stray', params.laserWavelengthNm, params.filterBandwidth);
      const solarScale = Math.max(0, Math.min(8, params.solarIrradiance / Math.max(1e-6, am0SolarIrradianceWM2Nm(params.laserWavelengthNm))));
      return {
        backgroundRateCps: backgroundRateCps * channelScale * solarScale * hazeScatterScale,
        darkRateCps: params.darkCountRate,
      };
    }

    const { solarIrradiance, filterBandwidth, apertureDiameter, systemEfficiency, quantumEfficiency, laserWavelengthNm, resolution } = params;
    const A_rx = Math.PI * (apertureDiameter / 2) ** 2;
    const P_bg_optical = (solarIrradiance * filterBandwidth) * A_rx * 1e-4 * hazeScatterScale;
    const bgPhotonRate = (P_bg_optical / this.getPhotonEnergy(laserWavelengthNm)) * systemEfficiency * quantumEfficiency;
    const totalPixels = resolution.width * resolution.height;
    return {
      backgroundRateCps: bgPhotonRate / Math.max(totalPixels, 1),
      darkRateCps: params.darkCountRate,
    };
  }

  public calculateSampledTrajectoryForPreview(params: ISimulationParams): {x: number; y: number; z: number}[] {
    if (params.targetType === 'Drone' && (params.recordedDroneTrajectory?.length ?? 0) > 1) {
      return params.recordedDroneTrajectory!.map(sample => sample.pos);
    }
    if (params.targetType === 'Blade' || params.targetType === 'Drone') {
      if (params.bladeMotionType === 'Path' && params.waypoints.length > 1) {
        return this.calculateBladePathTrajectory(params);
      } else { // Fixed position blade
        const trajectory = [];
        // A short array is enough, it doesn't move, just need the position for preview.
        for (let i = 0; i < 100; i++) { 
          trajectory.push(params.initialPos);
        }
        return trajectory;
      }
    }
    return this.calculateBallTrajectory(params);
  }
  
  public calculateTotalPathTime(params: ISimulationParams): number {
    if (params.targetType === 'Drone' && (params.recordedDroneTrajectory?.length ?? 0) > 1) {
      const samples = params.recordedDroneTrajectory!;
      return samples[samples.length - 1].time;
    }
    if ((params.targetType === 'Blade' || params.targetType === 'Drone') && params.bladeMotionType === 'Path' && params.waypoints.length > 1) {
      let totalTime = 0;
      for (let i = 0; i < params.waypoints.length - 1; i++) {
        const start = params.waypoints[i].pos;
        const end = params.waypoints[i + 1].pos;
        const speed = params.pathSpeeds[i] || 1.0;
        
        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const dz = end.z - start.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        
        if (speed > 0) {
          totalTime += distance / speed;
        }
      }
      return totalTime;
    }
    
    if (params.targetType === 'Blade' && params.bladeMotionType === 'Fixed' && params.rotationKeyframes.length > 0) {
      return params.rotationKeyframes[params.rotationKeyframes.length - 1]?.time || params.nFrames * params.frameDurationUs * 1e-6;
    }
    // Simplification for Drone Fixed: just use total simulation time or max of keyframes if we wanted to be precise, 
    // but usually fixed mode runs for nFrames.
    
    // For Ball or other cases, duration is based on nFrames
    return params.nFrames * params.frameDurationUs * 1e-6;
  }

  public calculateBladeRotationAngleAtTime(time: number, params: ISimulationParams, propIndex?: 1 | 2 | 3 | 4): number {
      const { bladeMotionType, waypoints, pathSpeeds, rotationSpeed } = params;
      
      let keyframes = params.rotationKeyframes;
      if (propIndex === 1) keyframes = params.rotationKeyframes1 || [];
      if (propIndex === 2) keyframes = params.rotationKeyframes2 || [];
      if (propIndex === 3) keyframes = params.rotationKeyframes3 || [];
      if (propIndex === 4) keyframes = params.rotationKeyframes4 || [];

      let pathRpms = params.pathRotationSpeeds;
      if (propIndex === 1) pathRpms = params.pathRotationSpeeds1 || [];
      if (propIndex === 2) pathRpms = params.pathRotationSpeeds2 || [];
      if (propIndex === 3) pathRpms = params.pathRotationSpeeds3 || [];
      if (propIndex === 4) pathRpms = params.pathRotationSpeeds4 || [];

      if (bladeMotionType === 'Fixed' || bladeMotionType === 'Manual') {
          if (!keyframes || keyframes.length < 2) {
              const omega = rotationSpeed * 2 * Math.PI / 60;
              return time * omega;
          }

          const sortedKeyframes = keyframes; 
          let totalAngle = 0;
          let lastTime = 0;
          let lastRpm = sortedKeyframes[0].rpm;

          for (let i = 1; i < sortedKeyframes.length; i++) {
              const currentTime = sortedKeyframes[i].time;
              const currentRpm = sortedKeyframes[i].rpm;
              
              const segmentDuration = currentTime - lastTime;
              if (segmentDuration <= 0) { 
                  lastTime = currentTime;
                  lastRpm = currentRpm;
                  continue;
              }

              const omegaStart = lastRpm * 2 * Math.PI / 60;
              const omegaEnd = currentRpm * 2 * Math.PI / 60;

              if (time >= currentTime) {
                  const avgOmega = (omegaStart + omegaEnd) / 2;
                  totalAngle += avgOmega * segmentDuration;
              } else {
                  const timeInSegment = time - lastTime;
                  const t = timeInSegment / segmentDuration;
                  const omegaNow = omegaStart + t * (omegaEnd - omegaStart);
                  const avgOmegaInSegment = (omegaStart + omegaNow) / 2;
                  totalAngle += avgOmegaInSegment * timeInSegment;
                  return totalAngle; 
              }

              lastTime = currentTime;
              lastRpm = currentRpm;
          }

          if (time > lastTime) {
              const timeAfterEnd = time - lastTime;
              const lastOmega = lastRpm * 2 * Math.PI / 60;
              totalAngle += lastOmega * timeAfterEnd;
          }
          
          return totalAngle;
      }
      
      if (waypoints.length < 2) return 0;
      
      let totalAngle = 0;
      let cumulativeTime = 0;

      for (let i = 0; i < waypoints.length - 1; i++) {
          const start = waypoints[i].pos;
          const end = waypoints[i + 1].pos;
          const travelSpeed = pathSpeeds[i] || 1.0;
          const rpm = pathRpms[i] || 0;
          const omega = rpm * 2 * Math.PI / 60;

          const dx = end.x - start.x;
          const dy = end.y - start.y;
          const dz = end.z - start.z;
          const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
          const segmentDuration = travelSpeed > 0 ? distance / travelSpeed : Infinity;
          
          if (time >= cumulativeTime + segmentDuration) {
              totalAngle += segmentDuration * omega;
          } else {
              const timeInSegment = time - cumulativeTime;
              totalAngle += timeInSegment * omega;
              return totalAngle; 
          }
          cumulativeTime += segmentDuration;
      }
      
      return totalAngle;
  }


  private calculateBladePathTrajectory(params: ISimulationParams): {x: number; y: number; z: number}[] {
      const { waypoints, pathSpeeds } = params;
      const totalDuration = this.calculateTotalPathTime(params);
      const totalPoints = 30000; // Fixed number of points for smooth preview
      const trajectory: {x: number, y: number, z: number}[] = [];

      for(let i = 0; i < totalPoints; i++) {
          const progress = i / (totalPoints - 1);
          const time = progress * totalDuration;
          trajectory.push(this.getBladePositionAtTime(time, waypoints, pathSpeeds));
      }

      return trajectory;
  }
  
  private getBladePositionAtTime(time: number, waypoints: ISimulationParams['waypoints'], pathSpeeds: ISimulationParams['pathSpeeds']): { x: number; y: number; z: number } {
    if (waypoints.length < 2) return waypoints[0]?.pos || {x: 0, y: 0, z: 0};

    let cumulativeTime = 0;
    for (let i = 0; i < waypoints.length - 1; i++) {
        const start = waypoints[i].pos;
        const end = waypoints[i + 1].pos;
        const speed = pathSpeeds[i] || 1.0;

        const dx = end.x - start.x;
        const dy = end.y - start.y;
        const dz = end.z - start.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        const segmentDuration = speed > 0 ? distance / speed : 0;
        
        if (time <= cumulativeTime + segmentDuration) {
            const timeInSegment = time - cumulativeTime;
            const t = segmentDuration > 0 ? timeInSegment / segmentDuration : 1.0;
            return {
                x: start.x + dx * t,
                y: start.y + dy * t,
                z: start.z + dz * t,
            };
        }
        cumulativeTime += segmentDuration;
    }
    return waypoints[waypoints.length - 1].pos;
  }

  private calculateBallTrajectory(params: ISimulationParams): {x: number; y: number; z: number}[] {
    const { nFrames, frameDurationUs, initialPos, initialVel, restitution, rotationRadius, rotationSpeed, rotationCenter, ballMotionType } = params;
    const MAX_PREVIEW_POINTS = 30000;
    
    if (nFrames <= 0) return [];

    const sampledTrajectory: {x: number, y: number, z: number}[] = [];
    let pos = { ...initialPos };
    let vel = { ...initialVel };
    const g = 9.8;
    const dt = frameDurationUs * 1e-6;

    const step = nFrames > MAX_PREVIEW_POINTS ? Math.ceil(nFrames / MAX_PREVIEW_POINTS) : 1;
    
    const isRotating = ballMotionType === 'Rotation';
    const omega = rotationSpeed * 2 * Math.PI / 60; // Convert RPM to rad/s

    for (let i = 0; i < nFrames; i++) {
        if (isRotating) {
            const timeNow = i * dt;
            pos.x = rotationCenter.x + rotationRadius * Math.cos(omega * timeNow);
            pos.y = initialPos.y; // Keep altitude constant
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

        if (i % step === 0) {
            sampledTrajectory.push({ ...pos });
        }
    }
    
    if ((nFrames - 1) % step !== 0) {
      sampledTrajectory.push({ ...pos });
    }

    return sampledTrajectory;
  }
}
