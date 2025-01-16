// hu-moments-chart.component.ts
import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { 
  LineChart, Line, XAxis, YAxis,
  CartesianGrid, Tooltip 
} from 'recharts';

@Component({
  selector: 'app-hu-moments-chart',
  standalone: true,
  imports: [CommonModule],
  template: `
    <LineChart width={600} height={300} data={chartData}>
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis dataKey="name" />
      <YAxis />
      <Tooltip />
      <Line type="monotone" dataKey="value" stroke="#8884d8" />
    </LineChart>
  `
})
export class HuMomentsChartComponent {
  @Input() set huMoments(data: any) {
    if (data) {
      this.chartData = data.moments.map((value: number, index: number) => ({
        name: data.names[index],
        value: value
      }));
    }
  }
  
  chartData: any[] = [];
}