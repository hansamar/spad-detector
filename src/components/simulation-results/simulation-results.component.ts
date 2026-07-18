import {
  AfterViewInit,
  ChangeDetectionStrategy,
  Component,
  ElementRef,
  OnChanges,
  SimpleChanges,
  ViewChild,
  inject,
  input,
  output,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { ISimulationResult } from '../../models/simulation-params.model';
import { IBackendSimulationSummary } from '../../services/backend-simulation.service';
import { LocalizationService } from '../../services/localization.service';

@Component({
  selector: 'app-simulation-results',
  standalone: true,
  imports: [CommonModule],
  changeDetection: ChangeDetectionStrategy.OnPush,
  templateUrl: './simulation-results.component.html',
})
export class SimulationResultsComponent implements AfterViewInit, OnChanges {
  result = input<ISimulationResult | null>(null);
  summary = input<IBackendSimulationSummary | null>(null);
  error = input('');
  tdcMaxCount = input(65535);
  close = output<void>();

  @ViewChild('photonCountingCanvas') private photonCountingCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('groundTruthCanvas') private groundTruthCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('incidentPhotonsCanvas') private incidentPhotonsCanvas?: ElementRef<HTMLCanvasElement>;
  @ViewChild('mechanicalFrequencyCanvas') private mechanicalFrequencyCanvas?: ElementRef<HTMLCanvasElement>;

  readonly l = inject(LocalizationService);
  private drawTimer: number | null = null;

  ngAfterViewInit(): void {
    this.scheduleDraw();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['result']) {
      this.scheduleDraw();
    }
  }

  get pileUpFactor(): number {
    return this.result()?.maxIncidentPhotonsPerPixel ?? 0;
  }

  get pileUpStatus(): { textKey: 'pileupHigh' | 'pileupModerate' | 'pileupLow'; className: string } {
    if (this.pileUpFactor > 0.95) return { textKey: 'pileupHigh', className: 'text-red-700' };
    if (this.pileUpFactor > 0.05) return { textKey: 'pileupModerate', className: 'text-amber-700' };
    return { textKey: 'pileupLow', className: 'text-emerald-700' };
  }

  get frequencyStats(): { min: number; max: number; mean: number } | null {
    const truth = this.result()?.groundTruthData;
    const propellerFrequencies = truth?.propellerFrequencies;
    const series = propellerFrequencies?.length ? propellerFrequencies : [truth?.frequencies ?? []];
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    let count = 0;
    for (const row of series) {
      for (const value of row) {
        if (!Number.isFinite(value) || value < 0) continue;
        min = Math.min(min, value);
        max = Math.max(max, value);
        sum += value;
        count++;
      }
    }
    return count ? { min, max, mean: sum / count } : null;
  }

  requestClose(): void {
    this.close.emit();
  }

  private scheduleDraw(attempt = 0): void {
    if (this.drawTimer !== null) window.clearTimeout(this.drawTimer);
    this.drawTimer = window.setTimeout(() => {
      const ready = !!this.result()
        && !!this.photonCountingCanvas?.nativeElement.clientWidth
        && !!this.groundTruthCanvas?.nativeElement.clientWidth
        && !!this.incidentPhotonsCanvas?.nativeElement.clientWidth;
      if (ready || attempt >= 30) {
        this.drawResultImages();
      } else {
        this.scheduleDraw(attempt + 1);
      }
    }, attempt === 0 ? 0 : 16);
  }

  private drawResultImages(): void {
    const result = this.result();
    const pcCanvas = this.photonCountingCanvas?.nativeElement;
    const gtCanvas = this.groundTruthCanvas?.nativeElement;
    const ipCanvas = this.incidentPhotonsCanvas?.nativeElement;
    if (!result || !pcCanvas || !gtCanvas || !ipCanvas) return;

    const { width, height } = result.resolution;
    const totalPixels = width * height;
    const photonCounts = Array.from({ length: height }, () => Array(width).fill(0) as number[]);
    if (result.photonCountMap) {
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) photonCounts[row][col] = result.photonCountMap[row]?.[col] ?? 0;
      }
    } else {
      const emptyPixelValue = this.tdcMaxCount() + 2;
      for (let i = 0; i < result.dataset.length; i++) {
        if (result.dataset[i] < emptyPixelValue) {
          const pixel = i % totalPixels;
          photonCounts[Math.floor(pixel / width)][pixel % width]++;
        }
      }
    }

    const groundTruthCounts = Array.from({ length: height }, () => Array(width).fill(0) as number[]);
    if (result.groundTruthMap) {
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) groundTruthCounts[row][col] = result.groundTruthMap[row]?.[col] ?? 0;
      }
    } else {
      for (const coord of result.signalCoordinates) {
        if (coord.row >= 0 && coord.row < height && coord.col >= 0 && coord.col < width) {
          groundTruthCounts[coord.row][coord.col]++;
        }
      }
    }

    this.drawHeatmap(pcCanvas, photonCounts, 'occupancy');
    this.drawHeatmap(gtCanvas, groundTruthCounts, 'intensity');
    this.drawHeatmap(ipCanvas, result.incidentPhotonMap, 'intensity');
    this.drawMechanicalFrequencyChart(this.mechanicalFrequencyCanvas?.nativeElement, result);
  }

  private drawMechanicalFrequencyChart(canvas: HTMLCanvasElement | undefined, result: ISimulationResult): void {
    if (!canvas) return;
    const truth = result.groundTruthData;
    const times = truth?.times ?? [];
    const propellerFrequencies = truth?.propellerFrequencies;
    const rawSeries = propellerFrequencies?.length
      ? [0, 1, 2, 3].map(prop => propellerFrequencies.map(row => row[prop] ?? 0))
      : [truth?.frequencies ?? []];
    if (!times.length || rawSeries.some(values => values.length !== times.length)) return;

    const cssWidth = Math.max(320, Math.floor(canvas.clientWidth || 720));
    const cssHeight = Math.max(180, Math.floor(canvas.clientHeight || 180));
    const dpr = Math.min(2, Math.max(1, window.devicePixelRatio || 1));
    canvas.width = Math.floor(cssWidth * dpr);
    canvas.height = Math.floor(cssHeight * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, cssWidth, cssHeight);

    const left = 58, right = 18, top = 22, bottom = 36;
    const plotW = Math.max(1, cssWidth - left - right);
    const plotH = Math.max(1, cssHeight - top - bottom);
    let minRaw = Number.POSITIVE_INFINITY;
    let maxRaw = Number.NEGATIVE_INFINITY;
    for (const values of rawSeries) {
      for (const value of values) {
        if (!Number.isFinite(value)) continue;
        minRaw = Math.min(minRaw, value);
        maxRaw = Math.max(maxRaw, value);
      }
    }
    if (!Number.isFinite(minRaw) || !Number.isFinite(maxRaw)) return;
    const maxPoints = Math.max(320, Math.floor(plotW * 1.5));
    const stride = Math.max(1, Math.ceil(times.length / maxPoints));
    const sampledTimes = times.filter((_, index) => index % stride === 0 || index === times.length - 1);
    const series = rawSeries.map(values => values.filter((_, index) => index % stride === 0 || index === values.length - 1));
    const tMin = times[0] ?? 0;
    const tMax = times.at(-1) ?? tMin;
    const pad = Math.max((maxRaw - minRaw) * 0.12, maxRaw > 0 ? maxRaw * 0.03 : 1);
    const fMin = Math.max(0, minRaw - pad);
    const fMax = maxRaw + pad;
    const xOf = (time: number) => left + ((time - tMin) / Math.max(tMax - tMin, 1e-9)) * plotW;
    const yOf = (hz: number) => top + (1 - (hz - fMin) / Math.max(fMax - fMin, 1e-9)) * plotH;

    ctx.strokeStyle = 'rgba(100, 116, 139, 0.18)';
    ctx.beginPath();
    for (let i = 0; i <= 4; i++) {
      const y = top + plotH * i / 4;
      ctx.moveTo(left, y);
      ctx.lineTo(left + plotW, y);
    }
    ctx.stroke();

    const colors = ['#0f766e', '#0284c7', '#d97706', '#e11d48'];
    series.forEach((values, index) => {
      ctx.strokeStyle = colors[index % colors.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, point) => point ? ctx.lineTo(xOf(sampledTimes[point]), yOf(value)) : ctx.moveTo(xOf(sampledTimes[point]), yOf(value)));
      ctx.stroke();
    });

    ctx.fillStyle = '#64748b';
    ctx.font = '12px Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const hz = fMax - (fMax - fMin) * i / 4;
      ctx.fillText(`${hz.toFixed(1)} Hz`, left - 8, top + plotH * i / 4);
    }
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText(`${tMin.toFixed(2)} s`, left, top + plotH + 10);
    ctx.textAlign = 'right';
    ctx.fillText(`${tMax.toFixed(2)} s`, left + plotW, top + plotH + 10);
  }

  private drawHeatmap(canvas: HTMLCanvasElement, data: number[][], mode: 'occupancy' | 'intensity'): void {
    const ctx = canvas.getContext('2d');
    if (!ctx || !data.length || !data[0]?.length) return;
    const height = data.length;
    const width = data[0].length;
    canvas.width = width;
    canvas.height = height;
    const positive = data.flat().filter(value => value > 0).sort((a, b) => a - b);
    if (!positive.length) {
      ctx.fillStyle = '#02070c';
      ctx.fillRect(0, 0, width, height);
      return;
    }
    const percentile = positive[Math.min(positive.length - 1, Math.floor(positive.length * 0.995))];
    const displayMax = Math.max(percentile, positive.at(-1)! * 0.35, 1e-12);
    const imageData = ctx.createImageData(width, height);
    for (let row = 0; row < height; row++) {
      for (let col = 0; col < width; col++) {
        const linear = Math.min(1, Math.max(0, data[row][col] / displayMax));
        const normalized = mode === 'occupancy' ? Math.sqrt(linear) : Math.log1p(9 * linear) / Math.log(10);
        const [red, green, blue] = this.palette(normalized);
        const index = (row * width + col) * 4;
        imageData.data.set([red, green, blue, 255], index);
      }
    }
    ctx.putImageData(imageData, 0, 0);
  }

  private palette(value: number): [number, number, number] {
    const stops: Array<[number, number, number]> = [
      [2, 8, 23], [10, 52, 92], [13, 148, 136], [94, 234, 212], [254, 240, 138],
    ];
    const scaled = Math.min(1, Math.max(0, value)) * (stops.length - 1);
    const index = Math.min(stops.length - 2, Math.floor(scaled));
    const t = scaled - index;
    return stops[index].map((channel, channelIndex) => Math.round(channel + (stops[index + 1][channelIndex] - channel) * t)) as [number, number, number];
  }
}
