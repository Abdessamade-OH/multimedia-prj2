import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageCrudComponent } from './image-crud.component';

describe('ImageCrudComponent', () => {
  let component: ImageCrudComponent;
  let fixture: ComponentFixture<ImageCrudComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImageCrudComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(ImageCrudComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
