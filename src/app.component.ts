import { ChangeDetectionStrategy, Component } from '@angular/core';
import { SimulationViewComponent } from './components/simulation-view/simulation-view.component';

@Component({
  selector: 'app-root',
  template: `
    <main class="h-[100dvh] w-screen overflow-hidden bg-[#f5f7f8]">
      <app-simulation-view />
    </main>
  `,
  imports: [SimulationViewComponent],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class AppComponent {}
