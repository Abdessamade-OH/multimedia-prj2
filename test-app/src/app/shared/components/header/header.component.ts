import { Component, OnInit } from '@angular/core';
import { AuthService } from '../../services/auth.service';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-header',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './header.component.html',
  styleUrl: './header.component.css'
})
export class HeaderComponent implements OnInit{

  constructor(private authService: AuthService){}

  isLoggedIn: boolean = false; // Cached login status


  logOut(){
    this.authService.logout();
  }

  ngOnInit(): void {
  this.authService.isLoggedIn().subscribe({
    next: (status) => {
      this.isLoggedIn = status; // Updates the cached status
      console.log('Login status updated:', this.isLoggedIn); // Log to confirm
    },
    error: (err) => {
      console.error('Error checking login status:', err);
    }
  });
}


  // This method can now safely return the cached value
  getIsLoggedIn(): boolean {
    return this.isLoggedIn;
  }

}
