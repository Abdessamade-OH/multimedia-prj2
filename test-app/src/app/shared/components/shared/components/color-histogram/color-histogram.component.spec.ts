import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ColorHistogramComponent } from './color-histogram.component';

describe('ColorHistogramComponent', () => {
  let component: ColorHistogramComponent;
  let fixture: ComponentFixture<ColorHistogramComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ColorHistogramComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ColorHistogramComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
