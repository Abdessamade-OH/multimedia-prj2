import { ComponentFixture, TestBed } from '@angular/core/testing';

import { GlcmChartComponent } from './glcm-chart.component';

describe('GlcmChartComponent', () => {
  let component: GlcmChartComponent;
  let fixture: ComponentFixture<GlcmChartComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [GlcmChartComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(GlcmChartComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
