"""
PDF Processing Module for Arabic Attendance System

Handles PDF loading, image extraction, and initial preprocessing.
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import logging
from pathlib import Path
import tempfile
import os

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF file processing and image extraction."""
    
    def __init__(self, target_dpi: int = 300):
        """
        Initialize PDF processor.
        
        Args:
            target_dpi: Target DPI for image extraction
        """
        self.target_dpi = target_dpi
        self.min_dpi = 200  # Minimum acceptable DPI
        
    def extract_images_from_pdf(self, pdf_path: str) -> List[np.ndarray]:
        """
        Extract all images from PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of images as numpy arrays
        """
        images = []
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF with {len(doc)} pages: {pdf_path}")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(self.target_dpi / 72, self.target_dpi / 72)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                img_array = np.array(img)
                
                # Convert RGBA to RGB if necessary
                if img_array.shape[2] == 4:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif img_array.shape[2] == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                
                images.append(img_array)
                logger.debug(f"Extracted page {page_num + 1}: {img_array.shape}")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
        
        return images
    
    def validate_image_quality(self, image: np.ndarray) -> Tuple[bool, dict]:
        """
        Validate image quality for OCR processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (is_valid, quality_metrics)
        """
        quality_metrics = {}
        
        # Check image dimensions
        height, width = image.shape[:2]
        quality_metrics['width'] = width
        quality_metrics['height'] = height
        quality_metrics['aspect_ratio'] = width / height
        
        # Estimate DPI (assuming A4 page)
        # A4 at 300 DPI: ~2480x3508 pixels
        estimated_dpi = min(width / 8.27, height / 11.69) * 25.4  # Convert to DPI
        quality_metrics['estimated_dpi'] = estimated_dpi
        
        # Check if DPI is acceptable
        dpi_ok = estimated_dpi >= self.min_dpi
        
        # Check contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        contrast = gray.std()
        quality_metrics['contrast'] = contrast
        contrast_ok = contrast > 30  # Minimum contrast threshold
        
        # Check brightness
        brightness = gray.mean()
        quality_metrics['brightness'] = brightness
        brightness_ok = 50 < brightness < 200  # Reasonable brightness range
        
        # Overall validation
        is_valid = dpi_ok and contrast_ok and brightness_ok
        quality_metrics['is_valid'] = is_valid
        
        if not is_valid:
            logger.warning(f"Image quality issues: DPI={estimated_dpi:.1f}, "
                         f"Contrast={contrast:.1f}, Brightness={brightness:.1f}")
        
        return is_valid, quality_metrics
    
    def process_pdf_file(self, pdf_path: str) -> List[Tuple[np.ndarray, dict]]:
        """
        Complete PDF processing pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of tuples (image, quality_metrics)
        """
        # Extract images
        images = self.extract_images_from_pdf(pdf_path)
        
        # Validate and process each image
        processed_images = []
        for i, image in enumerate(images):
            is_valid, quality_metrics = self.validate_image_quality(image)
            
            if is_valid:
                processed_images.append((image, quality_metrics))
                logger.info(f"Page {i + 1}: Valid image extracted")
            else:
                logger.warning(f"Page {i + 1}: Image quality below threshold")
                # Still include but mark as low quality
                quality_metrics['low_quality'] = True
                processed_images.append((image, quality_metrics))
        
        return processed_images


class ImagePreprocessor:
    """Handles image preprocessing for OCR optimization."""
    
    def __init__(self):
        """Initialize image preprocessor."""
        pass
    
    def deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image skew.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (deskewed_image, skew_angle)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (text should be white on black for HoughLines)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(binary, 1, np.pi/180, threshold=100)
        
        if lines is None:
            logger.warning("No lines detected for skew correction")
            return image, 0.0
        
        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = theta * 180 / np.pi
            # Convert to -90 to 90 range
            if angle > 90:
                angle -= 180
            angles.append(angle)
        
        # Find most common angle (mode)
        angles = np.array(angles)
        # Filter angles close to horizontal (within ±10 degrees)
        horizontal_angles = angles[np.abs(angles) < 10]
        
        if len(horizontal_angles) == 0:
            skew_angle = 0.0
        else:
            skew_angle = np.median(horizontal_angles)
        
        # Apply rotation if significant skew detected
        if abs(skew_angle) > 0.5:  # Only correct if skew > 0.5 degrees
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
            
            # Calculate new dimensions to avoid cropping
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            # Apply rotation
            deskewed = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            logger.info(f"Applied skew correction: {skew_angle:.2f} degrees")
            return deskewed, skew_angle
        
        return image, 0.0
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast for better OCR.
        
        Args:
            image: Input image
            
        Returns:
            Enhanced image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convert back to original format
        if len(image.shape) == 3:
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return enhanced
    
    def remove_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Remove noise while preserving text quality.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Convert back to original format
        if len(image.shape) == 3:
            denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
        
        return denoised
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Complete image preprocessing pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (processed_image, processing_info)
        """
        processing_info = {}
        
        # Step 1: Deskew
        deskewed, skew_angle = self.deskew_image(image)
        processing_info['skew_angle'] = skew_angle
        
        # Step 2: Enhance contrast
        enhanced = self.enhance_contrast(deskewed)
        processing_info['contrast_enhanced'] = True
        
        # Step 3: Remove noise
        denoised = self.remove_noise(enhanced)
        processing_info['noise_removed'] = True
        
        logger.info(f"Image preprocessing completed: skew={skew_angle:.2f}°")
        
        return denoised, processing_info


# Import required modules
import io