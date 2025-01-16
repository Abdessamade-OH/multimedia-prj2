import { Component, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms'; // Import FormsModule
import { CommonModule } from '@angular/common'; // Import CommonModule

@Component({
  selector: 'app-consent-banner',
  standalone: true,
  imports: [FormsModule, CommonModule],
  templateUrl: './consent-banner.component.html',
  styleUrl: './consent-banner.component.css'
})
export class ConsentBannerComponent implements OnInit {

  constructor() {}

  ngOnInit(): void {
    this.checkConsent();
  }

  checkConsent() {
    if (typeof window !== 'undefined' && localStorage) {
      const consent = localStorage.getItem('cookieConsent');
      
      if (consent) {
        document.getElementById('cookieConsentBanner')!.style.display = 'none'; // Hide banner if consent exists
      }
    }
  }

  // Accept cookies
  acceptCookies() {
    const firstPartyConsent = (document.getElementById('firstPartyCookies') as HTMLInputElement).checked;
    const thirdPartyConsent = (document.getElementById('thirdPartyCookies') as HTMLInputElement).checked;

    const consent = {
      firstParty: firstPartyConsent,
      thirdParty: thirdPartyConsent,
      timestamp: new Date().getTime()
    };

    localStorage.setItem('cookieConsent', JSON.stringify(consent)); // Store consent in localStorage
    this.setCookiesBasedOnConsent(consent);
    this.hideBanner();
  }

  // Reject all cookies
  rejectCookies() {
    localStorage.setItem('cookieConsent', JSON.stringify({
      firstParty: false,
      thirdParty: false,
      timestamp: new Date().getTime()
    }));

    this.clearNonNecessaryCookies(); // Clear any existing non-essential cookies
    this.hideBanner();
  }

  // Hide the banner after consent/rejection
  hideBanner() {
    document.getElementById('cookieConsentBanner')!.style.display = 'none';
  }

  // Set cookies based on user consent
  setCookiesBasedOnConsent(consent: { firstParty: boolean; thirdParty: boolean; }) {
    if (consent.firstParty) {
      document.cookie = `user_preference=accepted; max-age=86400; path=/;`; // Set first-party cookie (e.g., user preferences)
    }

    if (consent.thirdParty) {
      document.cookie = `_ga=GA_TRACKING_ID; max-age=63072000; path=/;`; // Example: Set third-party tracking cookie
    }
  }

  // Clear all non-necessary cookies
  clearNonNecessaryCookies() {
    // Example: Clear any first-party or third-party cookies
    document.cookie = "user_preference=; Max-Age=0; path=/;";
    document.cookie = "_ga=; Max-Age=0; path=/;";
  }
}
