// src/app/auth.service.ts
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { map, Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class AuthService {
  private apiUrl = 'http://localhost:3000/api/auth'; // Your backend API URL

  constructor(private http: HttpClient) {}

  login(username: string, password: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/login`, { username, password }, { withCredentials: true });
  }

  register(username: string, password: string): Observable<any> {
    return this.http.post(`${this.apiUrl}/register`, { username, password });
  }

  logout(): Observable<any> {
    return this.http.post(`${this.apiUrl}/logout`, {}, { withCredentials: true });
  }

  isLoggedIn(): Observable<boolean> {
    return this.http.get<{ isLoggedIn: boolean }>('http://localhost:3000/api/auth/check-token', { withCredentials: true }).pipe(
      map((response) => response.isLoggedIn)
    );
  }
  

  // Updated method to get user info
  getUserInfo(): Observable<any> {
    return this.http.get(`${this.apiUrl}/user-info`, { withCredentials: true });
  }

  // src/app/auth.service.ts
  setPreferences(theme: string): Observable<any> {
    return this.http.post('http://localhost:3000/set-preferences', { theme }, { withCredentials: true });
  }

}
