import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { ReactiveFormsModule, FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { ImageTransformService } from '../../shared/services/image-transform.service';

@Component({
  selector: 'app-image-crud',
  standalone: true,
  imports: [ReactiveFormsModule, FormsModule, CommonModule],
  templateUrl: './image-crud.component.html',
  styleUrls: ['./image-crud.component.css']
})
export class ImageCrudComponent {


  imageCategory: string = 'aGrass'; // Default category
  selectedFiles: File[] = [];
  imagePreview: string | null = null;
  transformedImage: string | null = null;
  multipleFilesSelected: boolean = false;
  transforming: boolean = false; // Track if transformation section is visible
  rotation: number = 0;
  crop: number = 100; // 100% crop (full image)
  resize: number = 100; // 100% size (no resizing)
  grayscale: boolean = false;

   // Properties for resize and crop sliders
   resizeWidth: number = 200;  // Default width value (can be adjusted)
   resizeHeight: number = 200; // Default height value (can be adjusted)
   cropWidth: number = 200;    // Default crop width
   cropHeight: number = 200;   // Default crop height

  constructor(private imageService: ImageServiceService, private imageTransformService: ImageTransformService) {}


  
  async rotateImage(direction: 'left' | 'right') {
    if (this.transformedImage) {
      try {
        const rotatedDataUrl = await this.imageTransformService.rotateImage(this.transformedImage, direction);
        this.transformedImage = rotatedDataUrl;
      } catch (error) {
        console.error('Error rotating image:', error);
      }
    } else {
      console.error('No image to rotate!');
    }
  }

  async resizeImage(width: number, height?: number) {
    if (this.transformedImage) {
      try {
        const resizedDataUrl = await this.imageTransformService.resizeImage(this.transformedImage, width, height);
        this.transformedImage = resizedDataUrl;
      } catch (error) {
        console.error('Error resizing image:', error);
      }
    } else {
      console.error('No image to resize!');
    }
  }
  
  async cropImage(x: number, y: number, width: number, height: number) {
    if (this.transformedImage) {
      try {
        const croppedDataUrl = await this.imageTransformService.cropImage(this.transformedImage, x, y, width, height);
        this.transformedImage = croppedDataUrl;
      } catch (error) {
        console.error('Error cropping image:', error);
      }
    } else {
      console.error('No image to crop!');
    }
  }
  
  
  async grayscaleImage() {
    if (this.transformedImage) {
      try {
        const grayscaleDataUrl = await this.imageTransformService.grayscaleImage(this.transformedImage);
        this.transformedImage = grayscaleDataUrl;
      } catch (error) {
        console.error('Error converting image to grayscale:', error);
      }
    } else {
      console.error('No image to convert to grayscale!');
    }
  }
  
  
  

  onFileChange(event: any) {
    this.selectedFiles = event.target.files;

    const input = event.target as HTMLInputElement;

    if (input.files) {
      this.selectedFiles = Array.from(input.files);

      // Check if there's a single file or multiple files
      if (this.selectedFiles.length === 1) {
        const reader = new FileReader();
        reader.onload = () => {
          this.imagePreview = reader.result as string;
        };
        reader.readAsDataURL(this.selectedFiles[0]);
        this.multipleFilesSelected = false;
      } else if (this.selectedFiles.length > 1) {
        this.imagePreview = null; // Clear single preview if multiple files are selected
        this.multipleFilesSelected = true;
      }
    }
  }

  onUploadImage() {
    // Create FormData
    const formData = new FormData();

    // Append category first
    formData.append('category', this.imageCategory);

    // Append all selected files
    for (let i = 0; i < this.selectedFiles.length; i++) {
      formData.append('images', this.selectedFiles[i]);
    }

    // Call service method to upload
    this.imageService.uploadImage(formData).subscribe({
      next: (response) => {
        console.log('Upload successful', response);
        // Handle success
      },
      error: (error) => {
        console.error('Upload failed', error);
        // Handle error
      }
    });
  }


  
  onTransformImage() {
    // Show the transformation section
    this.transforming = true;
    this.transformedImage = this.imagePreview; // Start with the original image
  }

  applyTransformations() {
    if (this.transformedImage) {
      let image = new Image();
      image.src = this.transformedImage;

      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (ctx) {
        const width = image.width * (this.resize / 100);
        const height = image.height * (this.resize / 100);
        canvas.width = width;
        canvas.height = height;

        ctx.drawImage(image, 0, 0, width, height);

        if (this.rotation) {
          ctx.save();
          ctx.translate(width / 2, height / 2);
          ctx.rotate((this.rotation * Math.PI) / 180);
          ctx.drawImage(image, -width / 2, -height / 2, width, height);
          ctx.restore();
        }

        if (this.grayscale) {
          ctx.filter = 'grayscale(100%)';
        }

        this.transformedImage = canvas.toDataURL();
      }
    }
  }

  finishTransformation() {
    if (this.transformedImage) {
      // Extract the base64 data from the transformed image
      const base64Data = this.transformedImage.split(',')[1]; // Remove "data:image/png;base64,"
      const contentType = this.transformedImage.split(';')[0].split(':')[1]; // Extract MIME type (e.g., "image/png")
  
      // Convert base64 to a Blob
      const binaryData = atob(base64Data);
      const arrayBuffer = new ArrayBuffer(binaryData.length);
      const uint8Array = new Uint8Array(arrayBuffer);
      for (let i = 0; i < binaryData.length; i++) {
        uint8Array[i] = binaryData.charCodeAt(i);
      }
      const blob = new Blob([uint8Array], { type: contentType });
  
      // Convert Blob to File (if required for backend compatibility)
      const file = new File([blob], 'transformed-image.png', { type: contentType });
  
      // Append to FormData
      const formData = new FormData();
      formData.append('category', this.imageCategory);
      formData.append('images', file);
  
      // Upload using the service
      this.imageService.uploadImage(formData).subscribe({
        next: (response) => {
          console.log('Transformed image uploaded successfully', response);
          // Handle success
        },
        error: (error) => {
          console.error('Upload failed', error);
          // Handle error
        }
      });
    } else {
      console.error('No transformed image available');
    }
  }
  

  cancelTransformations() {
  this.transformedImage = this.imagePreview; // Reset to the original image
  this.rotation = 0;
  this.crop = 100;
  this.resize = 100;
  this.grayscale = false;
  //this.transforming = false; // Hide the transformation controls
}

  

  resetForm(): void {
    this.imageCategory = '';
    this.selectedFiles = [];
    this.imagePreview = null;
    this.transformedImage = null;
    this.multipleFilesSelected = false;
    this.transforming = false;
    this.rotation = 0;
    this.crop = 100;
    this.resize = 100;
    this.grayscale = false;
  }

  
}
