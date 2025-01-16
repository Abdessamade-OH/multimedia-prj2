import { Routes } from '@angular/router';
import { TestComponent } from './components/test/test.component';
import { LoginComponent } from './components/login/login.component';
import { ProtectedComponent } from './components/protected/protected.component';
import { authGuard } from './shared/guards/auth.guard';
import { guestGuard } from './shared/guards/guest.guard';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ImageCrudComponent } from './components/image-crud/image-crud.component';
import { ImageViewComponent } from './components/image-view/image-view.component';
import { SimpleChange } from '@angular/core';
import { SimpleSearchComponent } from './components/simple-search/simple-search.component';
import { AdvancedSearchComponent } from './components/advanced-search/advanced-search.component';
import { HomeComponent } from './components/home/home.component';

export const routes: Routes = [
    {
        path: 'dashboard',
        component: DashboardComponent
    },
    { path: '', redirectTo: '/login', pathMatch: 'full' },
    { path: 'login', component: LoginComponent},
    { path: 'protected', component: ProtectedComponent},
    { path: 'image-crud', component: ImageCrudComponent},
    { path: 'image-view', component: ImageViewComponent},
    { path: 'simple-search', component: SimpleSearchComponent},
    { path: 'advanced-search', component: AdvancedSearchComponent},
    { path: 'home', component: HomeComponent}

];
