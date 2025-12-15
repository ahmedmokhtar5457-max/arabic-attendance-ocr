"""
Table Detection System using PaddleOCR PP-Structure

Detects table regions in Arabic attendance documents.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from paddleocr import PaddleOCR
import json

logger = logging.getLogger(__name__)


class TableDetector:
    """Detects table regions using PaddleOCR PP-Structure."""
    
    def __init__(self, use_gpu: bool = False, lang: str = 'ar'):
        """
        Initialize table detector.
        
        Args:
            use_gpu: Whether to use GPU acceleration
            lang: Language for OCR ('ar' for Arabic)
        """
        self.use_gpu = use_gpu
        self.lang = lang
        
        # Initialize PaddleOCR with structure analysis
        try:
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang=lang,
                use_gpu=use_gpu,
                show_log=False,
                structure_version='PP-StructureV2'
            )
            logger.info(f"PaddleOCR initialized with GPU={use_gpu}, lang={lang}")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            raise
    
    def detect_tables(self, image: np.ndarray) -> List[Dict]:
        """
        Detect table regions in image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of detected table information
        """
        try:
            # Run structure analysis
            result = self.ocr.ocr(image, cls=True)
            
            tables = []
            if result and len(result) > 0:
                for item in result[0]:
                    if len(item) >= 2:
                        bbox, (text, confidence) = item
                        
                        # Convert bbox to standard format
                        if len(bbox) == 4 and len(bbox[0]) == 2:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            table_info = {
                                'bbox': (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                                'confidence': float(confidence),
                                'text': text,
                                'area': (x_max - x_min) * (y_max - y_min)
                            }
                            tables.append(table_info)
            
            # Filter and sort tables by area (largest first)
            tables = [t for t in tables if t['area'] > 10000]  # Minimum area threshold
            tables.sort(key=lambda x: x['area'], reverse=True)
            
            logger.info(f"Detected {len(tables)} table regions")
            return tables
            
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            return []
    
    def detect_table_structure(self, table_image: np.ndarray) -> Dict:
        """
        Analyze table structure to identify rows and columns.
        
        Args:
            table_image: Cropped table image
            
        Returns:
            Dictionary with table structure information
        """
        try:
            # Run OCR on table image
            result = self.ocr.ocr(table_image, cls=True)
            
            cells = []
            if result and len(result) > 0:
                for item in result[0]:
                    if len(item) >= 2:
                        bbox, (text, confidence) = item
                        
                        if len(bbox) == 4 and len(bbox[0]) == 2:
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            
                            x_min, x_max = min(x_coords), max(x_coords)
                            y_min, y_max = min(y_coords), max(y_coords)
                            
                            cell_info = {
                                'bbox': (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)),
                                'text': text.strip(),
                                'confidence': float(confidence),
                                'center_x': int((x_min + x_max) / 2),
                                'center_y': int((y_min + y_max) / 2)
                            }
                            cells.append(cell_info)
            
            # Analyze structure
            structure = self._analyze_table_structure(cells, table_image.shape)
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing table structure: {str(e)}")
            return {'rows': [], 'columns': [], 'cells': []}
    
    def _analyze_table_structure(self, cells: List[Dict], image_shape: Tuple) -> Dict:
        """
        Analyze detected cells to infer table structure.
        
        Args:
            cells: List of detected cells
            image_shape: Shape of table image (height, width, channels)
            
        Returns:
            Dictionary with table structure
        """
        if not cells:
            return {'rows': [], 'columns': [], 'cells': cells}
        
        height, width = image_shape[:2]
        
        # Group cells by approximate Y position (rows)
        y_positions = [cell['center_y'] for cell in cells]
        y_positions.sort()
        
        # Find row boundaries using clustering
        rows = self._cluster_positions(y_positions, threshold=20)
        
        # Group cells by approximate X position (columns)
        x_positions = [cell['center_x'] for cell in cells]
        x_positions.sort()
        
        # Find column boundaries
        columns = self._cluster_positions(x_positions, threshold=30)
        
        # Assign cells to rows and columns
        for cell in cells:
            cell['row'] = self._find_closest_cluster(cell['center_y'], rows)
            cell['column'] = self._find_closest_cluster(cell['center_x'], columns)
        
        structure = {
            'rows': rows,
            'columns': columns,
            'cells': cells,
            'num_rows': len(rows),
            'num_columns': len(columns),
            'table_width': width,
            'table_height': height
        }
        
        logger.info(f"Table structure: {len(rows)} rows, {len(columns)} columns")
        return structure
    
    def _cluster_positions(self, positions: List[int], threshold: int) -> List[int]:
        """
        Cluster positions to find row/column boundaries.
        
        Args:
            positions: List of positions
            threshold: Distance threshold for clustering
            
        Returns:
            List of cluster centers
        """
        if not positions:
            return []
        
        clusters = []
        current_cluster = [positions[0]]
        
        for pos in positions[1:]:
            if pos - current_cluster[-1] <= threshold:
                current_cluster.append(pos)
            else:
                # Finish current cluster and start new one
                clusters.append(int(np.mean(current_cluster)))
                current_cluster = [pos]
        
        # Add last cluster
        clusters.append(int(np.mean(current_cluster)))
        
        return clusters
    
    def _find_closest_cluster(self, position: int, clusters: List[int]) -> int:
        """
        Find the closest cluster to a given position.
        
        Args:
            position: Position to match
            clusters: List of cluster centers
            
        Returns:
            Index of closest cluster
        """
        if not clusters:
            return 0
        
        distances = [abs(position - cluster) for cluster in clusters]
        return distances.index(min(distances))


class TemplateBasedTableDetector:
    """Template-based table detection for known layouts."""
    
    def __init__(self, template_config: Optional[Dict] = None):
        """
        Initialize template-based detector.
        
        Args:
            template_config: Configuration for expected table layout
        """
        self.template_config = template_config or self._get_default_template_config()
    
    def _get_default_template_config(self) -> Dict:
        """Get default template configuration for Arabic attendance forms."""
        return {
            'expected_columns': 7,
            'column_names': ['م', 'الاسم', 'وقت التسليم', 'نقطة الحراسة', 'الوردية', 'التوقيع', 'ملاحظات'],
            'target_columns': [2, 3, 4],  # Indices of columns to extract (0-based)
            'min_rows': 5,
            'max_rows': 30,
            'header_height_ratio': 0.1,  # Header takes ~10% of table height
            'row_height_min': 20,  # Minimum row height in pixels
        }
    
    def detect_table_by_template(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect table using template-based approach.
        
        Args:
            image: Input image
            
        Returns:
            Table detection result or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect vertical lines (column separators)
        vertical_lines = self._detect_vertical_lines(gray)
        
        if len(vertical_lines) < 2:
            logger.warning("Insufficient vertical lines detected for table")
            return None
        
        # Detect horizontal boundaries (top and bottom of table)
        horizontal_bounds = self._detect_table_bounds(gray)
        
        if not horizontal_bounds:
            logger.warning("Could not detect table boundaries")
            return None
        
        # Create table structure
        table_info = {
            'bbox': horizontal_bounds,
            'vertical_lines': vertical_lines,
            'columns': self._define_columns(vertical_lines, horizontal_bounds),
            'template_matched': True,
            'confidence': 0.9  # High confidence for template-based detection
        }
        
        return table_info
    
    def _detect_vertical_lines(self, gray_image: np.ndarray) -> List[int]:
        """
        Detect vertical lines in the image.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            List of x-coordinates of vertical lines
        """
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create vertical kernel
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        
        # Detect vertical lines
        vertical_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        
        # Find contours of vertical lines
        contours, _ = cv2.findContours(vertical_lines_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Extract x-coordinates of vertical lines
        line_x_coords = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h > gray_image.shape[0] * 0.3:  # Line should be at least 30% of image height
                line_x_coords.append(x + w // 2)  # Center of line
        
        # Sort and remove duplicates
        line_x_coords = sorted(list(set(line_x_coords)))
        
        logger.info(f"Detected {len(line_x_coords)} vertical lines")
        return line_x_coords
    
    def _detect_table_bounds(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect top and bottom boundaries of the table.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Bounding box (x, y, width, height) or None
        """
        height, width = gray_image.shape
        
        # Look for horizontal lines or dense text regions
        # Apply threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Create horizontal kernel for detecting horizontal structures
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Find horizontal line positions
        horizontal_positions = []
        for y in range(height):
            if np.sum(horizontal_lines[y, :]) > width * 0.3 * 255:  # At least 30% of row is line
                horizontal_positions.append(y)
        
        if len(horizontal_positions) < 2:
            # Fallback: use text density to find table bounds
            return self._detect_bounds_by_text_density(gray_image)
        
        # Use first and last horizontal lines as bounds
        top = min(horizontal_positions)
        bottom = max(horizontal_positions)
        
        # Add some padding
        top = max(0, top - 10)
        bottom = min(height, bottom + 10)
        
        return (0, top, width, bottom - top)
    
    def _detect_bounds_by_text_density(self, gray_image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect table bounds using text density analysis.
        
        Args:
            gray_image: Grayscale image
            
        Returns:
            Bounding box or None
        """
        height, width = gray_image.shape
        
        # Calculate text density for each row
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = cv2.bitwise_not(binary)  # Invert so text is white
        
        row_densities = []
        for y in range(height):
            density = np.sum(binary[y, :]) / (width * 255)
            row_densities.append(density)
        
        # Smooth densities
        kernel_size = 5
        smoothed = np.convolve(row_densities, np.ones(kernel_size) / kernel_size, mode='same')
        
        # Find regions with high text density
        threshold = np.mean(smoothed) + np.std(smoothed)
        high_density_rows = [i for i, density in enumerate(smoothed) if density > threshold]
        
        if not high_density_rows:
            return None
        
        # Find continuous regions
        regions = []
        start = high_density_rows[0]
        end = start
        
        for i in range(1, len(high_density_rows)):
            if high_density_rows[i] - high_density_rows[i-1] <= 5:  # Gap tolerance
                end = high_density_rows[i]
            else:
                if end - start > 50:  # Minimum region height
                    regions.append((start, end))
                start = high_density_rows[i]
                end = start
        
        # Add last region
        if end - start > 50:
            regions.append((start, end))
        
        if not regions:
            return None
        
        # Use largest region as table
        largest_region = max(regions, key=lambda r: r[1] - r[0])
        top, bottom = largest_region
        
        return (0, top, width, bottom - top)
    
    def _define_columns(self, vertical_lines: List[int], table_bounds: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Define column boundaries based on vertical lines.
        
        Args:
            vertical_lines: List of x-coordinates of vertical lines
            table_bounds: Table bounding box
            
        Returns:
            List of column definitions
        """
        x, y, width, height = table_bounds
        
        columns = []
        
        # Add left boundary if not present
        if not vertical_lines or vertical_lines[0] > 10:
            vertical_lines.insert(0, x)
        
        # Add right boundary if not present
        if not vertical_lines or vertical_lines[-1] < x + width - 10:
            vertical_lines.append(x + width)
        
        # Create column definitions
        for i in range(len(vertical_lines) - 1):
            col_x = vertical_lines[i]
            col_width = vertical_lines[i + 1] - col_x
            
            column = {
                'index': i,
                'x': col_x,
                'y': y,
                'width': col_width,
                'height': height,
                'is_target': i in self.template_config['target_columns']
            }
            columns.append(column)
        
        logger.info(f"Defined {len(columns)} columns")
        return columns