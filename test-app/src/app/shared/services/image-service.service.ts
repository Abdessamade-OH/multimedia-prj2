import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ImageServiceService {
  relevanceSearch(relevantImages: any[], irrelevantImages: any[], alpha: number, beta: number, gamma: number) {
    throw new Error('Method not implemented.');
  }

  constructor(private http: HttpClient) { }

  private apiUrl = 'http://localhost:3000/api/images'; // Your backend API URL
  private baseUrl = 'http://localhost:5000'; // Flask API URL


  // Upload images method using FormData
  uploadImage(formData: FormData): Observable<any> {
    return this.http.post(`${this.apiUrl}/upload`, formData);
  }

  // Delete image by MongoDB ID
  deleteImageById(imageId: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/delete/${imageId}`);
  }

  // Get images by name (returns all with the same name)
  getImagesByName(imageName: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/name/${imageName}`);
  }

  // Get all images
  getAllImages(): Observable<any> {
    return this.http.get(`${this.apiUrl}/all`);
  }

  // Simple search by image name (if still applicable in your backend)
  simpleSearch(imageName: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/search/simple`, { params: { name: imageName } });
  }

  // Advanced search with additional criteria (if still applicable in your backend)
  advancedSearch(imageName: string, category: string, otherFilters: any): Observable<any> {
    return this.http.get(`${this.apiUrl}/search/advanced`, {
      params: { 
        name: imageName,
        category: category,
        ...otherFilters
      }
    });
  }

  // Get images by category
  getImagesByCategory(category: string): Observable<any> {
    return this.http.get(`${this.apiUrl}/category/${category}`);
  }

  getImageById(imageId: string): Observable<any> {
    return this.http.get<any>(`http://localhost:3000/get_image/${imageId}`);
  }
  
  


  // POST method for feature extraction
extractFeatures(imageName: string, category: string, k: number): Observable<any> {
  // Create a FormData object to append the category and k parameters
  const formData = new FormData();
  formData.append('category', category);  // Add category to form data
  formData.append('k', k.toString());  // Add k to form data, ensure it's a string

  // Perform the POST request with FormData as the body
  return this.http.post(`${this.baseUrl}/extract_features/${imageName}`, formData);
}

// Function to send relevance feedback query
sendRelevanceFeedback(query: {
  name: string;
  category: string;
  relevant_images: string[];
  non_relevant_images: string[];
}): Observable<any> {
  const headers = new HttpHeaders({ 'Content-Type': 'application/json' });

  return this.http.post(`${this.baseUrl}/relevance-feedback-search`, query, {
    headers: headers,
  });
}

  

}
