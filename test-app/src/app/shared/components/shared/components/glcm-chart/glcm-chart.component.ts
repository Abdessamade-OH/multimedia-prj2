// glcm-chart.component.ts
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { 
  RadarChart, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, Radar 
} from 'recharts';

@Component({
  selector: 'app-glcm-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <RadarChart width={300} height={300} data={chartData}>
      <PolarGrid />
      <PolarAngleAxis dataKey="feature" />
      <PolarRadiusAxis />
      <Radar dataKey="value" fill="#8884d8" fillOpacity={0.6} />
    </RadarChart>
  `
})
export class GlcmChartComponent {
  @Input() set glcmData(data: any) {
    if (data) {
      this.chartData = [
        { feature: 'Contrast', value: data.contrast[0] },
        { feature: 'Dissimilarity', value: data.dissimilarity[0] },
        { feature: 'Homogeneity', value: data.homogeneity[0] },
        { feature: 'Energy', value: data.energy[0] },
        { feature: 'Correlation', value: data.correlation[0] }
      ];
    }
  }
  
  chartData: any[] = [];
}