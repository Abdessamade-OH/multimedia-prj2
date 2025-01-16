import { Component } from '@angular/core';
import { ImageServiceService } from '../../shared/services/image-service.service';
import { FormsModule } from '@angular/forms';
import { CommonModule } from '@angular/common';
import { catchError, map, Observable, of } from 'rxjs';
import { RemoveFirstLetterPipe } from "../../remove-first-letter.pipe";
import { Chart, ChartConfiguration, ChartData } from 'chart.js';
import { BaseChartDirective } from 'ng2-charts';


@Component({
  selector: 'app-simple-search',
  standalone: true,
  imports: [FormsModule, CommonModule, RemoveFirstLetterPipe],
  templateUrl: './simple-search.component.html',
  styleUrls: ['./simple-search.component.css'],
})
export class SimpleSearchComponent {
  imageCategory!: string;
  numberK!: number;
  imageName: string = '';
  imageUrl: string | null = null;
  similarImages: any[] = [];
  isLoading: boolean = false; // Loading spinner state
  //features: any = {}; // Store extracted features
  alpha!: number; // New alpha input
  beta!: number;   // New beta input
  gamma!: number;  // New gamma input
  imageSelections: any[] = []; // Array to store selected images (irrelevant)
  relevantSearch: boolean = false;

  relevanceSearchResults: any[] = []; // New array for relevance search results
  showRelevanceResults: boolean = false; // New flag to control relevance results visibility


  constructor(private imageService: ImageServiceService) {}

  getImageByName(name: string): void {
    console.log('Starting image search...');
    this.isLoading = true; // Start loading for image retrieval

    this.imageService.getImagesByName(name).subscribe({
      next: (imageInfo) => {
        console.log('Image search completed');
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
          this.imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          const imageNameFromPath = relativePath.split('/').pop();
          this.imageName = imageNameFromPath || 'Unknown';
          this.imageCategory = imageInfo[0].category;
          

          this.isLoading = false; // Stop loading after image retrieval
        } else {
          console.log('No image found');
          this.imageUrl = null;
          this.isLoading = false; // Stop loading on no image found
        }
      },
      error: (err) => {
        console.error('Error fetching image by name:', err);
        this.isLoading = false; // Stop loading on error
      },
    });
  }

  getImageByName2(name: string): Observable<any> {
    console.log('Starting image search...');
    this.isLoading = true; // Start loading for image retrieval

    return this.imageService.getImagesByName(name).pipe(
      map((imageInfo) => {
        console.log('Image search completed');
        if (imageInfo && imageInfo.length > 0) {
          const imagePath = imageInfo[0].path;
          const relativePath = imagePath.split('/src/upload_folder/')[1]; // Extract relative path
          const imageUrl = `http://localhost:3000/uploaded_images/${relativePath}`;
          const imageNameFromPath = relativePath.split('/').pop();
          const imageCategory = imageInfo[0].category;

          return { imageUrl, imageName: imageNameFromPath || 'Unknown', imageCategory };
        } else {
          console.log('No image found');
          return null; // Return null if no image found
        }
      }),
      catchError((err) => {
        console.error('Error fetching image by name:', err);
        return of(null); // Return null if there's an error
      })
    );
}

extractFeatures(): void {
  console.log('Starting feature extraction...');
  this.isLoading = true; // Start loading for feature extraction

  this.imageService.extractFeatures(this.imageName, this.imageCategory, this.numberK || 10).subscribe({
    next: (response) => {
      console.log(response);
      console.log('Feature extraction completed');
      this.similarImages = response.similar_images || []; // Extract similar images
      
      // Fetch the image for each similar image using getImageByName2
      let requestsCompleted = 0;
      const totalRequests = this.similarImages.length;

      this.similarImages.forEach((image: any) => {
        console.log(image);
        const imageName = this.extractImageName(image.image_path);
        this.getImageByName2(imageName).subscribe((imageData) => {
          if (imageData) {
            // Now you can use the imageData (imageUrl, imageName, imageCategory)
            console.log(imageData);
            // For example, you can update the image URL for each similar image
            image.url = imageData.imageUrl;
            image.name = imageData.imageName;
            this.features = response.features || {}; // Store extracted features
            console.log(this.features)
            image.category = imageData.imageCategory;
          }

          // Check if all requests have completed
          requestsCompleted++;
          if (requestsCompleted === totalRequests) {
            this.isLoading = false; // Stop loading after all requests are completed
          }
        });
      });

      this.relevantSearch = true;
      // If no similar images were found, stop loading immediately
      if (totalRequests === 0) {
        this.isLoading = false;
      }
    },
    error: (error) => {
      console.error('Error extracting features:', error);
      this.isLoading = false; // Stop loading on error
    },
  });
}

  extractImageName(imagePath: string): string {
    const pathParts = imagePath.split('\\'); // Split the path by backslashes
    return pathParts[pathParts.length - 1]; // Return the last part (the name)
  }

  toggleSelection(image: any): void {
    image.isSelected = !image.isSelected;
    if (image.isSelected) {
      // If image is selected, it is added to the irrelevant images list
      this.imageSelections.push(image);
    } else {
      // If image is deselected, remove from irrelevant images list
      this.imageSelections = this.imageSelections.filter((img) => img !== image);
    }
  }

  stripPrefix(imageName: string): string {
    const parts = imageName.split('-');
    return parts[parts.length - 1]; // Return the last part (the actual image name)
  }

  performRelevanceSearch(): void {
    console.log('Alpha:', this.alpha, 'Beta:', this.beta, 'Gamma:', this.gamma);
    this.isLoading = true;

    const relevantImages: string[] = [];
    const nonRelevantImages: string[] = [];
    let requestsCompleted = 0;
    const totalRequests = this.similarImages.length + 1;

    const constructImagePath = (imageData: any): string => {
      return `${imageData.imageCategory}/${imageData.imageName}`;
    };

    console.log(this.imageName);
    const mainImageName = this.extractImageName(this.imageName);
    console.log(mainImageName);
    
    this.getImageByName2(mainImageName).subscribe({
      next: (imageData) => {
        if (imageData) {
          relevantImages.push(constructImagePath(imageData));
          console.log('Main image data:', imageData);
          console.log('Main image path:', constructImagePath(imageData));
        }
        checkCompletion();
      },
      error: (err) => {
        console.error('Error fetching main image:', err);
        checkCompletion();
      },
    });

    this.similarImages
      .filter((img) => !this.imageSelections.includes(img))
      .forEach((img) => {
        const imageName = this.extractImageName(img.image_path);
        this.getImageByName2(imageName).subscribe({
          next: (imageData) => {
            if (imageData) {
              relevantImages.push(constructImagePath(imageData));
            }
            checkCompletion();
          },
          error: (err) => {
            console.error('Error fetching relevant image:', err);
            checkCompletion();
          },
        });
      });

    this.imageSelections.forEach((img) => {
      const imageName = this.extractImageName(img.image_path);
      this.getImageByName2(imageName).subscribe({
        next: (imageData) => {
          if (imageData) {
            nonRelevantImages.push(constructImagePath(imageData));
          }
          checkCompletion();
        },
        error: (err) => {
          console.error('Error fetching non-relevant image:', err);
          checkCompletion();
        },
      });
    });

    const checkCompletion = () => {
      requestsCompleted++;
      if (requestsCompleted === totalRequests) {
        const query = {
          name: this.imageName,
          category: this.imageCategory,
          relevant_images: relevantImages,
          non_relevant_images: nonRelevantImages,
          alpha: this.alpha,
          beta: this.beta,
          gamma: this.gamma
        };

        console.log('Query for relevance feedback:', query);

        this.imageService.sendRelevanceFeedback(query).subscribe({
          next: (response) => {
            console.log('Relevance feedback response:', response);
            // Store results in the new array instead of overwriting similarImages
            this.relevanceSearchResults = response.similar_images || [];
            
            let responseRequestsCompleted = 0;
            const totalResponseRequests = this.relevanceSearchResults.length;

            this.relevanceSearchResults.forEach((image: any) => {
              const imageName = this.extractImageName(image.image_path);
              this.getImageByName2(imageName).subscribe({
                next: (imageData) => {
                  if (imageData) {
                    image.url = imageData.imageUrl;
                    image.name = imageData.imageName;
                    image.category = imageData.imageCategory;
                  }
                  
                  responseRequestsCompleted++;
                  if (responseRequestsCompleted === totalResponseRequests) {
                    this.isLoading = false;
                    this.showRelevanceResults = true; // Show relevance results section
                  }
                },
                error: (err) => {
                  console.error('Error processing response image:', err);
                  responseRequestsCompleted++;
                  if (responseRequestsCompleted === totalResponseRequests) {
                    this.isLoading = false;
                    this.showRelevanceResults = true;
                  }
                }
              });
            });

            if (totalResponseRequests === 0) {
              this.isLoading = false;
              this.showRelevanceResults = true;
            }
          },
          error: (error) => {
            console.error('Error sending relevance feedback:', error);
            this.isLoading = false;
          },
        });
      }
    };
  }
  
  features: any = {
    color_histogram: {
      red: [0.2, 0.4, 0.1, 0.3],
      green: [0.1, 0.3, 0.4, 0.2],
      blue: [0.3, 0.2, 0.3, 0.2],
    },
    dominant_colors: {
      colors: [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
      percentages: [0.6, 0.3, 0.1],
    },
    glcm_features: {
      contrast: 0.5,
      dissimilarity: 0.6,
      homogeneity: 0.7,
      energy: 0.8,
      correlation: 0.9,
    },
    lbp_features: {
      histogram: [0.1, 0.2, 0.3, 0.4, 0.5],
      parameters: { radius: 1, n_points: 8 },
    },
    hu_moments: {
      moments: [0.12, 0.45, 0.67, 0.89, 0.23],
      names: ['Hu1', 'Hu2', 'Hu3', 'Hu4', 'Hu5'],
    }
  };

  // Color distribution chart data
  colorChartData: ChartData<'bar'> = {
    labels: ['Red', 'Green', 'Blue'],
    datasets: [
      {
        data: this.features?.color_histogram ? [
          this.features.color_histogram.red.reduce((a: any, b: any) => a + b, 0), 
          this.features.color_histogram.green.reduce((a: any, b: any) => a + b, 0), 
          this.features.color_histogram.blue.reduce((a: any, b: any) => a + b, 0)
        ] : [0, 0, 0],
        backgroundColor: ['red', 'green', 'blue']
      }
    ]
  };

  // Texture GLCM Features chart data
  glcmChartData: ChartData<'bar'> = {
    labels: ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'],
    datasets: [
      {
        data: this.features?.glcm_features ? [
          this.features.glcm_features.contrast,
          this.features.glcm_features.dissimilarity,
          this.features.glcm_features.homogeneity,
          this.features.glcm_features.energy,
          this.features.glcm_features.correlation
        ] : [0, 0, 0, 0, 0],
        backgroundColor: '#007bff'
      }
    ]
  };

  // LBP Histogram chart data
  lbpChartData: ChartData<'bar'> = {
    labels: Array.from({ length: this.features?.lbp_features?.histogram.length }, (_, i) => `Bin ${i + 1}`),
    datasets: [
      {
        data: this.features?.lbp_features?.histogram || [],
        backgroundColor: '#28a745'
      }
    ]
  };

  ngAfterViewInit(): void {
    this.renderColorHistogramChart();
    this.renderDominantColorsChart();
    this.renderGLCMChart();
  }

  renderColorHistogramChart(): void {
    const ctx = document.getElementById('colorHistogramChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: ['Red', 'Green', 'Blue'],
        datasets: [
          {
            label: 'Color Distribution',
            data: [
              this.features.color_histogram.red.reduce((a: any, b: any) => a + b, 0),
              this.features.color_histogram.green.reduce((a: any, b: any) => a + b, 0),
              this.features.color_histogram.blue.reduce((a: any, b: any) => a + b, 0),
            ],
            backgroundColor: ['#ff0000', '#00ff00', '#0000ff'],
          },
        ],
      },
    });
  }

  renderDominantColorsChart(): void {
    const ctx = document.getElementById('dominantColorsChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'pie',
      data: {
        labels: this.features.dominant_colors.colors.map((_: any, index: number) => `Color ${index + 1}`),
        datasets: [
          {
            data: this.features.dominant_colors.percentages.map((p: number) => p * 100),
            backgroundColor: this.features.dominant_colors.colors.map(
              (rgb: any[]) => `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
            ),
          },
        ],
      },
    });
  }

  renderGLCMChart(): void {
    const ctx = document.getElementById('glcmChart') as HTMLCanvasElement;
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: ['Contrast', 'Dissimilarity', 'Homogeneity', 'Energy', 'Correlation'],
        datasets: [
          {
            label: 'GLCM Features',
            data: [
              this.features.glcm_features.contrast.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.dissimilarity.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.homogeneity.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.energy.reduce((a: any, b: any) => a + b, 0),
              this.features.glcm_features.correlation.reduce((a: any, b: any) => a + b, 0),
            ],
            borderColor: '#007bff',
            fill: false,
          },
        ],
      },
    });
  }

  // Method to convert RGB arrays to string
  getRGBString(color: number[]): string {
    if (!color || color.length !== 3) return 'rgb(0, 0, 0)';
    return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
  }
  
}


