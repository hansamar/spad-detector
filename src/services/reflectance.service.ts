export interface IVec3 {
  x: number;
  y: number;
  z: number;
}

export interface ISurfaceReturnOptions {
  specularGain?: number;
  specularWidthDeg?: number;
}

const EPS = 1e-12;

export function dotVec3(a: IVec3, b: IVec3): number {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

export function subtractVec3(a: IVec3, b: IVec3): IVec3 {
  return { x: a.x - b.x, y: a.y - b.y, z: a.z - b.z };
}

export function crossVec3(a: IVec3, b: IVec3): IVec3 {
  return {
    x: a.y * b.z - a.z * b.y,
    y: a.z * b.x - a.x * b.z,
    z: a.x * b.y - a.y * b.x,
  };
}

export function normalizeVec3(v: IVec3): IVec3 {
  const norm = Math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
  if (norm < EPS) return { x: 0, y: 0, z: 0 };
  return { x: v.x / norm, y: v.y / norm, z: v.z / norm };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function phaseAngleRad(a: IVec3, b: IVec3): number {
  const an = normalizeVec3(a);
  const bn = normalizeVec3(b);
  return Math.acos(clamp(dotVec3(an, bn), -1, 1));
}

export function monostaticLambertFactor(faceNormal: IVec3, targetToDetectorUnit: IVec3): number {
  const n = normalizeVec3(faceNormal);
  const los = normalizeVec3(targetToDetectorUnit);
  const cosTheta = Math.max(0, dotVec3(n, los));
  return cosTheta * cosTheta;
}

export function monostaticSurfaceReturnFactor(
  faceNormal: IVec3,
  targetToDetectorUnit: IVec3,
  options: ISurfaceReturnOptions = {},
): number {
  const lambert = monostaticLambertFactor(faceNormal, targetToDetectorUnit);
  if (lambert <= 0) return 0;

  const specularGain = Math.max(0, options.specularGain ?? 0);
  const specularWidthRad = Math.max(1e-6, ((options.specularWidthDeg ?? 5) * Math.PI) / 180);
  if (specularGain <= 0) return lambert;

  const angle = phaseAngleRad(faceNormal, targetToDetectorUnit);
  const specular = specularGain * Math.exp(-(angle * angle) / (2 * specularWidthRad * specularWidthRad));
  return lambert * (1 + specular);
}

export function polygonNormal(corners: IVec3[]): IVec3 {
  if (corners.length < 3) return { x: 0, y: 0, z: 0 };
  const edgeA = subtractVec3(corners[1], corners[0]);
  const edgeB = subtractVec3(corners[2], corners[0]);
  return normalizeVec3(crossVec3(edgeA, edgeB));
}

export function orientNormalToward(normal: IVec3, direction: IVec3): IVec3 {
  return dotVec3(normal, direction) >= 0
    ? normal
    : { x: -normal.x, y: -normal.y, z: -normal.z };
}
