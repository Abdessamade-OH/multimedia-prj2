import { ComponentFixture, TestBed } from '@angular/core/testing';

import { HuMomentsChartComponent } from './hu-moments-chart.component';

describe('HuMomentsChartComponent', () => {
  let component: HuMomentsChartComponent;
  let fixture: ComponentFixture<HuMomentsChartComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [HuMomentsChartComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(HuMomentsChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
