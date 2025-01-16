import { CanActivateFn, Router } from "@angular/router";
import { AuthService } from "../services/auth.service";
import { map } from "rxjs";
import { inject } from "@angular/core";

export const guestGuard: CanActivateFn = (route, state) => {
  const authService = inject(AuthService);
  const router = inject(Router);

  return authService.isLoggedIn().pipe(
    map((loggedIn) => {
      console.log('guestGuard - Is logged in:', loggedIn);
      if (!loggedIn) {
        return true; // Allow access if not logged in
      } else {
        router.navigate(['/protected']); // Redirect to a protected page if logged in
        return false; // Block access to login page
      }
    })
  );
};
