export function gaussianRandom(random: () => number = Math.random): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = random();
  while (v === 0) v = random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

export function samplePoissonCount(
  lambda: number,
  random: () => number = Math.random,
  normal: () => number = () => gaussianRandom(random),
): number {
  if (!Number.isFinite(lambda) || lambda <= 0) return 0;
  if (lambda < 30) {
    const limit = Math.exp(-lambda);
    let product = 1;
    let k = 0;
    do {
      k++;
      product *= random();
    } while (product > limit);
    return k - 1;
  }
  return Math.max(0, Math.round(lambda + Math.sqrt(lambda) * normal()));
}

export function firstPhotonTofUnits(
  count: number,
  baseTofNs: number,
  timeResolutionNs: number,
  random: () => number = Math.random,
): number | null {
  if (count <= 0 || timeResolutionNs <= 0) return null;
  return Math.floor((baseTofNs + random() * timeResolutionNs) / timeResolutionNs);
}
