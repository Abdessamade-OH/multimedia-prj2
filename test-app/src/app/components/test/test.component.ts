import { Component } from '@angular/core';
import { CookieService } from 'ngx-cookie-service';
import { ConsentBannerComponent } from "../../shared/components/consent-banner/consent-banner.component";
import { AuthService } from '../../shared/services/auth.service';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-test',
  standalone: true,
  imports: [ConsentBannerComponent, FormsModule],
  templateUrl: './test.component.html',
  styleUrl: './test.component.css'
})
export class TestComponent {

  constructor(private cookieService: CookieService, private authService: AuthService){}

  // Set a cookie
  setCookie() {
    this.cookieService.set('user', 'John Doe', 1); // Set 'user' cookie with value 'John Doe' for 1 day
  }

  // Get a cookie
  getCookie() {
    const user = this.cookieService.get('user');
    console.log('User:', user);
  }

  // Delete a cookie
  deleteCookie() {
    this.cookieService.delete('user');
  }

  
  theme: string = 'light'; // Default theme

  submitPreferences() {
    this.authService.setPreferences(this.theme).subscribe(
      response => {
        console.log('Preferences set successfully', response);
      },
      error => {
        console.error('Error setting preferences', error);
      }
    );
  }


}
