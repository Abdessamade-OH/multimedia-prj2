import { TestBed } from '@angular/core/testing';

import { ImageTransformService } from './image-transform.service';

describe('ImageTransformService', () => {
  let service: ImageTransformService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ImageTransformService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
