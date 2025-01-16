import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import gabor
from typing import Dict, List, Tuple
import mahotas as mt
import trimesh
from scipy.fft import fftn
from scipy.special import sph_harm
from scipy.spatial.transform import Rotation
import os

class ImageFeatureExtractor:
    """Enhanced feature extractor with multiple descriptors."""
    
    def __init__(self):
        self.gabor_filters = self._create_gabor_filters()
    
    def _create_gabor_filters(self) -> List[Tuple]:
        """Create Gabor filters with different orientations and frequencies."""
        orientations = 8
        frequencies = [0.1, 0.3, 0.5, 0.7]
        return [(frequency, theta) for theta in np.arange(0, np.pi, np.pi/orientations) 
                for frequency in frequencies]

    def extract_color_histogram(self, image: np.ndarray, bins: int = 32) -> Dict[str, np.ndarray]:
        """Extract color histogram features for each channel."""
        histograms = {}
        for i, channel in enumerate(['blue', 'green', 'red']):
            hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            histograms[channel] = hist
        return histograms

    def extract_dominant_colors(self, image: np.ndarray, k: int = 5) -> Dict[str, np.ndarray]:
        """Extract dominant colors using K-means clustering."""
        pixels = image.reshape(-1, 3)
        pixels = np.float32(pixels)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, 
                                      cv2.KMEANS_RANDOM_CENTERS)
        # Calculate percentage of each dominant color
        percentages = np.bincount(labels.flatten()) / len(labels)
        return {
            'colors': centers,
            'percentages': percentages
        }

    def extract_gabor_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Gabor texture features with enhanced parameters."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        feature_names = []
        
        for frequency, theta in self.gabor_filters:
            filt_real, filt_imag = gabor(gray, frequency=frequency, theta=theta)
            features.extend([
                np.mean(filt_real), np.std(filt_real),
                np.mean(filt_imag), np.std(filt_imag)
            ])
            feature_names.extend([
                f'gabor_real_mean_f{frequency}_t{theta}',
                f'gabor_real_std_f{frequency}_t{theta}',
                f'gabor_imag_mean_f{frequency}_t{theta}',
                f'gabor_imag_std_f{frequency}_t{theta}'
            ])
        
        return {
            'features': np.array(features),
            'feature_names': feature_names
        }

    def extract_hu_moments(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Hu moments for shape description."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.HuMoments(cv2.moments(gray)).flatten()
        log_moments = -np.sign(moments) * np.log10(np.abs(moments))
        return {
            'moments': log_moments,
            'names': [f'hu_moment_{i+1}' for i in range(len(log_moments))]
        }

    def extract_lbp_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Local Binary Pattern features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                             range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        return {
            'histogram': hist,
            'parameters': {
                'radius': radius,
                'n_points': n_points
            }
        }

    def extract_glcm_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Gray Level Co-occurrence Matrix (GLCM) features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        
        features = {
            'contrast': graycoprops(glcm, 'contrast')[0],
            'dissimilarity': graycoprops(glcm, 'dissimilarity')[0],
            'homogeneity': graycoprops(glcm, 'homogeneity')[0],
            'energy': graycoprops(glcm, 'energy')[0],
            'correlation': graycoprops(glcm, 'correlation')[0]
        }
        
        return features

    def extract_hog_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract Histogram of Oriented Gradients (HOG) features."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        win_size = (64, 64)
        cell_size = (8, 8)
        block_size = (16, 16)
        block_stride = (8, 8)
        num_bins = 9
        
        # Resize image to match window size
        gray = cv2.resize(gray, win_size)
        
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, 
                               cell_size, num_bins)
        features = hog.compute(gray)
        
        return {
            'features': features.flatten(),
            'parameters': {
                'window_size': win_size,
                'cell_size': cell_size,
                'block_size': block_size,
                'num_bins': num_bins
            }
        }

    def extract_all_features(self, image_path: str) -> Dict[str, Dict]:
        """Extract all features from an image and return structured results."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.resize(image, (224, 224))  # Standardize size
        
        return {
            'color_histogram': self.extract_color_histogram(image),
            'dominant_colors': self.extract_dominant_colors(image),
            'gabor_features': self.extract_gabor_features(image),
            'hu_moments': self.extract_hu_moments(image),
            'lbp_features': self.extract_lbp_features(image),
            'hog_features': self.extract_hog_features(image),
            'glcm_features': self.extract_glcm_features(image)
        }

# class Model3DProcessor:
#     """Handles 3D model preprocessing for invariant feature extraction."""
    
#     def __init__(self, voxel_resolution: int = 64):
#         self.voxel_resolution = voxel_resolution
    
#     def normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
#         """
#         Normalize mesh for scale, rotation and translation invariance.
#         """
#         # Center the mesh
#         mesh = mesh.copy()
#         mesh.vertices -= mesh.centroid
        
#         # Scale to unit sphere
#         scale = np.max(np.linalg.norm(mesh.vertices, axis=1))
#         mesh.vertices /= scale
        
#         # Align principal axes with coordinate system
#         inertia = mesh.moment_inertia
#         if not np.allclose(inertia, 0):
#             _, axes = np.linalg.eigh(inertia)
#             mesh.vertices = mesh.vertices @ axes
            
#         return mesh
    
#     def voxelize(self, mesh: trimesh.Trimesh) -> np.ndarray:
#         """
#         Convert mesh to voxel representation with given resolution.
#         """
#         voxel_grid = mesh.voxelized(pitch=2.0/self.voxel_resolution)
#         voxel_grid = voxel_grid.fill()
#         return voxel_grid.matrix.astype(np.float32)

# # Update the existing Model3DFeatureExtractor class
# class Model3DFeatureExtractor:
#     def __init__(self, voxel_resolution: int = 64, max_zernike_degree: int = 8):
#         self.processor = Model3DProcessor(voxel_resolution)
#         self.voxel_resolution = voxel_resolution
#         self.max_zernike_degree = max_zernike_degree

#     def extract_fourier_features(self, voxel_grid: np.ndarray, num_coefficients: int = 10) -> np.ndarray:
#         """
#         Extract 3D Fourier coefficients.
#         """
#         fourier = fftn(voxel_grid)
#         magnitude = np.abs(fourier)
#         features = magnitude[:num_coefficients, :num_coefficients, :num_coefficients]
#         features = features / (np.max(features) + 1e-10)
#         return features.flatten()

#     def extract_zernike_moments(self, voxel_grid: np.ndarray) -> np.ndarray:
#         """
#         Extract 3D Zernike moments.
#         """
#         moments = []
#         center = np.array([self.voxel_resolution/2]*3)
#         radius = self.voxel_resolution/2
        
#         for n in range(self.max_zernike_degree + 1):
#             for l in range(n + 1):
#                 if (n - l) % 2 == 0:
#                     moment = self._compute_zernike_moment(voxel_grid, n, l, center, radius)
#                     moments.append(moment)
        
#         return np.array(moments)

#     def _compute_zernike_moment(self, voxel_grid: np.ndarray, n: int, l: int, 
#                               center: np.ndarray, radius: float) -> complex:
#         moment = 0j
#         for x in range(voxel_grid.shape[0]):
#             for y in range(voxel_grid.shape[1]):
#                 for z in range(voxel_grid.shape[2]):
#                     if voxel_grid[x,y,z] > 0:
#                         pos = np.array([x,y,z]) - center
#                         r = np.linalg.norm(pos) / radius
#                         if r <= 1:
#                             theta = np.arccos(pos[2]/(r*radius + 1e-10))
#                             phi = np.arctan2(pos[1], pos[0])
#                             moment += voxel_grid[x,y,z] * self._zernike_polynomial(r, theta, phi, n, l)
#         return moment

#     def _zernike_polynomial(self, r: float, theta: float, phi: float, n: int, l: int) -> complex:
#         return r**n * sph_harm(0, l, phi, theta)

#     def extract_all_features(self, model_path: str) -> Dict[str, Dict]:
#         """
#         Extract all features from a 3D model file.
#         """
#         # Load and normalize mesh
#         mesh = trimesh.load(model_path)
#         mesh = self.processor.normalize_mesh(mesh)
        
#         # Convert to voxel representation
#         voxels = self.processor.voxelize(mesh)
        
#         # Extract features
#         fourier_features = self.extract_fourier_features(voxels)
#         zernike_features = self.extract_zernike_moments(voxels)
        
#         return {
#             'fourier': {'features': fourier_features},
#             'zernike': {'features': zernike_features}
#         }