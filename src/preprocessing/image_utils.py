"""
Image Utilities for Arabic Attendance System

Common image processing utilities and helper functions.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility functions for image processing."""
    
    @staticmethod
    def resize_image(image: np.ndarray, target_width: Optional[int] = None, 
                    target_height: Optional[int] = None, 
                    maintain_aspect: bool = True) -> np.ndarray:
        """
        Resize image to target dimensions.
        
        Args:
            image: Input image
            target_width: Target width in pixels
            target_height: Target height in pixels
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        if target_width is None and target_height is None:
            return image
        
        if maintain_aspect:
            if target_width is not None and target_height is not None:
                # Calculate scale to fit within both dimensions
                scale_w = target_width / width
                scale_h = target_height / height
                scale = min(scale_w, scale_h)
                new_width = int(width * scale)
                new_height = int(height * scale)
            elif target_width is not None:
                scale = target_width / width
                new_width = target_width
                new_height = int(height * scale)
            else:  # target_height is not None
                scale = target_height / height
                new_width = int(width * scale)
                new_height = target_height
        else:
            new_width = target_width or width
            new_height = target_height or height
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        return resized
    
    @staticmethod
    def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int,
                  padding: int = 0) -> np.ndarray:
        """
        Crop image with optional padding.
        
        Args:
            image: Input image
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Crop width
            height: Crop height
            padding: Additional padding around crop area
            
        Returns:
            Cropped image
        """
        img_height, img_width = image.shape[:2]
        
        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_width, x + width + padding)
        y2 = min(img_height, y + height + padding)
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Convert image to grayscale if needed.
        
        Args:
            image: Input image
            
        Returns:
            Grayscale image
        """
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def apply_threshold(image: np.ndarray, method: str = 'otsu') -> np.ndarray:
        """
        Apply thresholding to create binary image.
        
        Args:
            image: Input grayscale image
            method: Thresholding method ('otsu', 'adaptive', 'fixed')
            
        Returns:
            Binary image
        """
        if method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif method == 'fixed':
            _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        else:
            raise ValueError(f"Unknown thresholding method: {method}")
        
        return binary
    
    @staticmethod
    def detect_text_regions(image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in image using morphological operations.
        
        Args:
            image: Input grayscale image
            
        Returns:
            List of bounding boxes (x, y, width, height)
        """
        # Apply threshold
        binary = ImageUtils.apply_threshold(image, 'otsu')
        
        # Morphological operations to connect text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(binary, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h
            
            # Filter based on size and aspect ratio
            if area > 100 and 0.1 < aspect_ratio < 10:
                text_regions.append((x, y, w, h))
        
        return text_regions
    
    @staticmethod
    def save_image(image: np.ndarray, filepath: str, quality: int = 95) -> bool:
        """
        Save image to file.
        
        Args:
            image: Image to save
            filepath: Output file path
            quality: JPEG quality (0-100)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Determine file extension
            ext = Path(filepath).suffix.lower()
            
            if ext in ['.jpg', '.jpeg']:
                cv2.imwrite(filepath, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            elif ext == '.png':
                cv2.imwrite(filepath, image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                cv2.imwrite(filepath, image)
            
            return True
        except Exception as e:
            logger.error(f"Error saving image to {filepath}: {str(e)}")
            return False
    
    @staticmethod
    def load_image(filepath: str) -> Optional[np.ndarray]:
        """
        Load image from file.
        
        Args:
            filepath: Path to image file
            
        Returns:
            Loaded image or None if failed
        """
        try:
            image = cv2.imread(filepath)
            if image is None:
                logger.error(f"Could not load image: {filepath}")
                return None
            return image
        except Exception as e:
            logger.error(f"Error loading image {filepath}: {str(e)}")
            return None
    
    @staticmethod
    def calculate_image_stats(image: np.ndarray) -> dict:
        """
        Calculate basic image statistics.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with image statistics
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        stats = {
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': len(image.shape),
            'mean_brightness': float(gray.mean()),
            'std_brightness': float(gray.std()),
            'min_brightness': int(gray.min()),
            'max_brightness': int(gray.max()),
            'total_pixels': gray.size
        }
        
        return stats
    
    @staticmethod
    def create_debug_visualization(image: np.ndarray, regions: List[Tuple[int, int, int, int]],
                                 title: str = "Debug Visualization") -> np.ndarray:
        """
        Create debug visualization with bounding boxes.
        
        Args:
            image: Input image
            regions: List of bounding boxes (x, y, width, height)
            title: Title for visualization
            
        Returns:
            Visualization image
        """
        # Create copy for visualization
        vis_image = image.copy()
        
        # Convert to color if grayscale
        if len(vis_image.shape) == 2:
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2BGR)
        
        # Draw bounding boxes
        for i, (x, y, w, h) in enumerate(regions):
            # Different colors for different regions
            color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
                    (255, 0, 255), (0, 255, 255)][i % 6]
            
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(vis_image, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        return vis_image


class GeometryUtils:
    """Geometric utility functions."""
    
    @staticmethod
    def calculate_iou(box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        
        Args:
            box1: First bounding box (x, y, width, height)
            box2: Second bounding box (x, y, width, height)
            
        Returns:
            IoU value between 0 and 1
        """
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]], 
                               iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        Merge overlapping bounding boxes.
        
        Args:
            boxes: List of bounding boxes (x, y, width, height)
            iou_threshold: IoU threshold for merging
            
        Returns:
            List of merged bounding boxes
        """
        if not boxes:
            return []
        
        merged = []
        used = [False] * len(boxes)
        
        for i, box1 in enumerate(boxes):
            if used[i]:
                continue
            
            # Find all boxes that overlap with current box
            group = [box1]
            used[i] = True
            
            for j, box2 in enumerate(boxes[i + 1:], i + 1):
                if used[j]:
                    continue
                
                if GeometryUtils.calculate_iou(box1, box2) > iou_threshold:
                    group.append(box2)
                    used[j] = True
            
            # Merge boxes in group
            if len(group) == 1:
                merged.append(group[0])
            else:
                # Calculate bounding box of all boxes in group
                x_min = min(box[0] for box in group)
                y_min = min(box[1] for box in group)
                x_max = max(box[0] + box[2] for box in group)
                y_max = max(box[1] + box[3] for box in group)
                
                merged.append((x_min, y_min, x_max - x_min, y_max - y_min))
        
        return merged
    
    @staticmethod
    def sort_boxes_by_position(boxes: List[Tuple[int, int, int, int]], 
                              direction: str = 'top_to_bottom') -> List[Tuple[int, int, int, int]]:
        """
        Sort bounding boxes by position.
        
        Args:
            boxes: List of bounding boxes (x, y, width, height)
            direction: Sorting direction ('top_to_bottom', 'left_to_right', 'right_to_left')
            
        Returns:
            Sorted list of bounding boxes
        """
        if direction == 'top_to_bottom':
            return sorted(boxes, key=lambda box: box[1])
        elif direction == 'left_to_right':
            return sorted(boxes, key=lambda box: box[0])
        elif direction == 'right_to_left':
            return sorted(boxes, key=lambda box: -box[0])
        else:
            raise ValueError(f"Unknown sorting direction: {direction}")