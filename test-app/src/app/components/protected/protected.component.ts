import { Component, OnInit } from '@angular/core';
import { AuthService } from '../../shared/services/auth.service';
import { Router } from '@angular/router';
import { CommonModule } from '@angular/common'; // Import CommonModule


@Component({
  selector: 'app-protected',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './protected.component.html',
  styleUrl: './protected.component.css'
})
export class ProtectedComponent implements OnInit {
  userInfo: any;
  errorMessage: string = '';

  constructor(private authService: AuthService, private router: Router) {}

  ngOnInit(): void {
    this.fetchUserInfo();
  }

  fetchUserInfo() {
    this.authService.getUserInfo().subscribe(
      response => {
        this.userInfo = response;
        console.log('User info fetched:', this.userInfo);
      },
      error => {
        this.errorMessage = 'Failed to fetch user information';
        console.error('Error fetching user info:', error);
      }
    );
  }
}
