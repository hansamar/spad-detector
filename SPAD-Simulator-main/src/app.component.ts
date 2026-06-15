import { ChangeDetectionStrategy, Component } from '@angular/core';
import { SimulationViewComponent } from './components/simulation-view/simulation-view.component';

@Component({
  selector: 'app-root',
  template: `
    <main class="h-screen w-screen bg-gray-900">
      <app-simulation-view />
    </main>
  `,
  imports: [SimulationViewComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {}