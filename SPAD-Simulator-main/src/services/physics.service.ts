import { Injectable } from '@angular/core';
import { ISimulationParams } from '../models/simulation-params.model';

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

  private getAtmosphericAttenuationCoefficient(wavelengthNm: number): number {
    // Coefficients in km^-1 for clear air (visibility ~23km) based on wavelength.
    // Shorter wavelengths scatter more and have higher attenuation.
    const attenuationData: [number, number][] = [
        [400, 0.25], // Blue/Violet
        [550, 0.16], // Green
        [780, 0.12], // Near IR
        [1064, 0.08], // Common laser line
        [1600, 0.04]  // SWIR
    ];

    if (wavelengthNm <= attenuationData[0][0]) {
        return attenuationData[0][1];
    }
    if (wavelengthNm >= attenuationData[attenuationData.length - 1][0]) {
        return attenuationData[attenuationData.length - 1][1];
    }

    for (let i = 0; i < attenuationData.length - 1; i++) {
        const [wl1, coef1] = attenuationData[i];
        const [wl2, coef2] = attenuationData[i + 1];

        if (wavelengthNm >= wl1 && wavelengthNm <= wl2) {
            // Linear interpolation between points
            const factor = (wavelengthNm - wl1) / (wl2 - wl1);
            return coef1 + factor * (coef2 - coef1);
        }
    }

    return attenuationData[attenuationData.length - 1][1]; // Fallback
  }

  public calculateAtmosphericTransmittance(params: ISimulationParams, distanceM: number): number {
    if (!params.atmosphericAttenuationEnabled) {
        return 1.0;
    }

    const alpha_per_km = this.getAtmosphericAttenuationCoefficient(params.laserWavelengthNm);
    const alpha_per_m = alpha_per_km / 1000;
    const pathLengthM = 2 * distanceM; // To the target and back

    // Beer-Lambert Law for atmospheric transmittance
    return Math.exp(-alpha_per_m * pathLengthM);
  }

  public calculateBackgroundNoise(params: ISimulationParams): number {
    const { solarIrradiance, filterBandwidth, apertureDiameter, systemEfficiency, quantumEfficiency, darkCountRate, frameDurationUs, laserWavelengthNm, resolution } = params;
    
    // Solar irradiance noise rate for the whole array
    const A_rx = Math.PI * (apertureDiameter / 2) ** 2;
    const P_bg_optical = (solarIrradiance * filterBandwidth) * A_rx * 1e-4; // Rough estimation coefficient
    const bg_photon_rate = (P_bg_optical / this.getPhotonEnergy(laserWavelengthNm)) * systemEfficiency * quantumEfficiency;

    // Dark count noise rate for the whole array
    const totalPixels = resolution.width * resolution.height;
    const totalDarkCountRate = darkCountRate * totalPixels;

    // Combine both full-array noise rates
    const total_rate = bg_photon_rate + totalDarkCountRate;
    const n_noise = total_rate * (frameDurationUs * 1e-6);

    return n_noise;
  }

  public calculateSampledTrajectoryForPreview(params: ISimulationParams): {x: number; y: number; z: number}[] {
    if (params.targetType === 'Blade') {
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
    if (params.targetType === 'Blade' && params.bladeMotionType === 'Path' && params.waypoints.length > 1) {
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
    
    // For Ball or Fixed Blade, duration is based on nFrames
    return params.nFrames * params.frameDurationUs * 1e-6;
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