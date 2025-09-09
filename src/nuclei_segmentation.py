"""
Advanced Nuclei Segmentation for Breast Cancer Histopathology
Provides multiple segmentation approaches including Cellpose integration
"""

import cv2
import numpy as np
from skimage import measure, morphology, segmentation
from skimage.feature import peak_local_maxima
from scipy import ndimage
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class NucleiSegmenter:
    def __init__(self, method='watershed'):
        """
        Initialize nuclei segmenter
        
        Args:
            method: 'watershed', 'cellpose', or 'hybrid'
        """
        self.method = method
        
        # Try to import cellpose if available
        self.cellpose_available = False
        try:
            from cellpose import models
            self.cellpose_model = models.Cellpose(gpu=False, model_type='nuclei')
            self.cellpose_available = True
        except ImportError:
            print("⚠️ Cellpose not available, using watershed method")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for nuclei segmentation"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return blurred
    
    def watershed_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Watershed-based nuclei segmentation"""
        # Preprocess
        processed = self.preprocess_image(image)
        
        # Threshold using Otsu's method
        _, binary = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Distance transform
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        
        # Find local maxima (nuclei centers)
        local_maxima = peak_local_maxima(dist_transform, min_distance=10, threshold_abs=0.3*dist_transform.max())
        
        # Create markers for watershed
        markers = np.zeros_like(dist_transform, dtype=np.int32)
        for i, (y, x) in enumerate(local_maxima):
            markers[y, x] = i + 1
        
        # Apply watershed
        labels = segmentation.watershed(-dist_transform, markers, mask=opening)
        
        # Extract properties
        props = measure.regionprops(labels)
        nuclei_data = []
        
        for prop in props:
            if prop.area > 20:  # Filter small regions
                nuclei_data.append({
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'bbox': prop.bbox,
                    'perimeter': prop.perimeter,
                    'major_axis_length': prop.major_axis_length,
                    'minor_axis_length': prop.minor_axis_length
                })
        
        return labels, nuclei_data
    
    def cellpose_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Cellpose-based nuclei segmentation"""
        if not self.cellpose_available:
            print("⚠️ Cellpose not available, falling back to watershed")
            return self.watershed_segmentation(image)
        
        # Cellpose expects RGB image
        if len(image.shape) == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Run cellpose
        masks, flows, styles, diams = self.cellpose_model.eval(rgb_image, diameter=None, channels=[0,0])
        
        # Extract properties
        props = measure.regionprops(masks)
        nuclei_data = []
        
        for prop in props:
            if prop.area > 20:  # Filter small regions
                nuclei_data.append({
                    'centroid': prop.centroid,
                    'area': prop.area,
                    'eccentricity': prop.eccentricity,
                    'solidity': prop.solidity,
                    'bbox': prop.bbox,
                    'perimeter': prop.perimeter,
                    'major_axis_length': prop.major_axis_length,
                    'minor_axis_length': prop.minor_axis_length
                })
        
        return masks, nuclei_data
    
    def hybrid_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Hybrid approach combining multiple methods"""
        # Try cellpose first, fallback to watershed
        if self.cellpose_available:
            try:
                return self.cellpose_segmentation(image)
            except Exception as e:
                print(f"⚠️ Cellpose failed: {e}, using watershed")
        
        return self.watershed_segmentation(image)
    
    def segment(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """Main segmentation method"""
        if self.method == 'cellpose':
            return self.cellpose_segmentation(image)
        elif self.method == 'hybrid':
            return self.hybrid_segmentation(image)
        else:  # watershed
            return self.watershed_segmentation(image)
    
    def analyze_nuclei_morphology(self, nuclei_data: List[Dict]) -> Dict:
        """Analyze morphological characteristics of nuclei"""
        if not nuclei_data:
            return {}
        
        areas = [n['area'] for n in nuclei_data]
        eccentricities = [n['eccentricity'] for n in nuclei_data]
        solidities = [n['solidity'] for n in nuclei_data]
        
        # Compute nuclear-to-cytoplasmic ratio approximation
        perimeters = [n['perimeter'] for n in nuclei_data]
        nc_ratios = [area / (perimeter**2) for area, perimeter in zip(areas, perimeters)]
        
        # Size variation coefficient
        area_cv = np.std(areas) / np.mean(areas) if areas else 0
        
        analysis = {
            'count': len(nuclei_data),
            'area_stats': {
                'mean': np.mean(areas),
                'std': np.std(areas),
                'min': np.min(areas),
                'max': np.max(areas),
                'cv': area_cv
            },
            'shape_stats': {
                'mean_eccentricity': np.mean(eccentricities),
                'mean_solidity': np.mean(solidities),
                'irregular_count': sum(1 for e in eccentricities if e > 0.8)
            },
            'nc_ratio_stats': {
                'mean': np.mean(nc_ratios),
                'std': np.std(nc_ratios)
            }
        }
        
        return analysis
    
    def classify_nuclei_abnormalities(self, nuclei_data: List[Dict]) -> Dict:
        """Classify nuclei based on morphological abnormalities"""
        if not nuclei_data:
            return {'normal': 0, 'enlarged': 0, 'irregular': 0, 'fragmented': 0}
        
        areas = [n['area'] for n in nuclei_data]
        eccentricities = [n['eccentricity'] for n in nuclei_data]
        solidities = [n['solidity'] for n in nuclei_data]
        
        # Define thresholds (these would be calibrated on training data)
        area_threshold_large = np.percentile(areas, 75) * 1.5
        eccentricity_threshold = 0.8
        solidity_threshold = 0.7
        area_threshold_small = np.percentile(areas, 25) * 0.5
        
        classification = {'normal': 0, 'enlarged': 0, 'irregular': 0, 'fragmented': 0}
        
        for nucleus in nuclei_data:
            area = nucleus['area']
            eccentricity = nucleus['eccentricity']
            solidity = nucleus['solidity']
            
            if area > area_threshold_large:
                classification['enlarged'] += 1
            elif eccentricity > eccentricity_threshold or solidity < solidity_threshold:
                classification['irregular'] += 1
            elif area < area_threshold_small:
                classification['fragmented'] += 1
            else:
                classification['normal'] += 1
        
        return classification