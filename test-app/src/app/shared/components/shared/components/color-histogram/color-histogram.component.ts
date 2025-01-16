import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ChartConfiguration, ChartData } from 'chart.js';
import { Chart, registerables } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';

// Register the chart types and elements to use in the app
Chart.register(...registerables);

@Component({
  selector: 'app-color-histogram',
  standalone: true,
  imports: [CommonModule, BaseChartDirective],
  template: `
    <div style="display: block;">
      <canvas baseChart
              [data]="chartData"
              [options]="chartOptions"
              [type]="'bar'">
      </canvas>
    </div>
  `
})
export class ColorHistogramComponent {
  @Input() set histogramData(data: any) {
    if (data) {
      this.chartData = {
        labels: Array(Math.max(data.red.length, data.green.length, data.blue.length)).fill(''),
        datasets: [
          { data: data.red, label: 'Red', backgroundColor: '#ff0000' },
          { data: data.green, label: 'Green', backgroundColor: '#00ff00' },
          { data: data.blue, label: 'Blue', backgroundColor: '#0000ff' },
        ]
      };
    }
  }

  chartData: ChartData<'bar'> = { datasets: [] };
  chartOptions: ChartConfiguration['options'] = {
    responsive: true,
    scales: {
      x: {
        title: {
          display: true,
          text: 'Bins'
        }
      },
      y: {
        title: {
          display: true,
          text: 'Frequency'
        }
      }
    }
  };
}
