// src/app/login/login.component.ts
import { Component } from '@angular/core';
import { AuthService } from '../../shared/services/auth.service';
import { Router } from '@angular/router';
import { FormsModule } from '@angular/forms'; // Import FormsModule
import { CommonModule } from '@angular/common'; // Import CommonModule

@Component({
  selector: 'app-login',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.css'] // Fixed the typo here from styleUrl to styleUrls
})
export class LoginComponent {
  username: string = '';
  password: string = '';
  errorMessage: string = '';
  
  // New properties for registration
  registerUsername: string = '';
  registerPassword: string = '';
  registerErrorMessage: string = '';

  constructor(private authService: AuthService, private router: Router) {}

  login() {
    this.authService.login(this.username, this.password).subscribe(
      response => {
        console.log('Login successful:', response);
        this.router.navigate(['/protected']); // Navigate to a protected route after login
      },
      error => {
        this.errorMessage = 'Invalid credentials';
        console.error('Login error:', error);
      }
    );
  }

  register() {
    this.authService.register(this.registerUsername, this.registerPassword).subscribe(
      response => {
        console.log('Registration successful:', response);
        // Optionally, you might want to auto-login after registration or navigate somewhere else
        this.router.navigate(['/']); // Redirect to the home or login page after registration
      },
      error => {
        this.registerErrorMessage = 'Registration failed';
        console.error('Registration error:', error);
      }
    );
  }
}
