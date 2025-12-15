"""
Column and Row Segmentation System

Segments table regions into individual cells for OCR processing.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.cluster import DBSCAN
from collections import defaultdict

logger = logging.getLogger(__name__)


class TableSegmenter:
    """Segments tables into rows and columns without horizontal lines."""
    
    def __init__(self, min_row_height: int = 20, min_col_width: int = 30):
        """
        Initialize table segmenter.
        
        Args:
            min_row_height: Minimum height for a valid row
            min_col_width: Minimum width for a valid column
        """
        self.min_row_height = min_row_height
        self.min_col_width = min_col_width
    
    def segment_table(self, table_image: np.ndarray, 
                     column_lines: List[int], 
                     ocr_results: Optional[List[Dict]] = None) -> Dict:
        """
        Segment table into cells using column lines and OCR text positions.
        
        Args:
            table_image: Cropped table image
            column_lines: List of x-coordinates of vertical column separators
            ocr_results: Optional OCR results with text positions
            
        Returns:
            Dictionary with segmentation results
        """
        height, width = table_image.shape[:2]
        
        # Define columns based on vertical lines
        columns = self._define_columns(column_lines, width, height)
        
        # Detect rows using text positions or content analysis
        if ocr_results:
            rows = self._detect_rows_from_ocr(ocr_results, height)
        else:
            rows = self._detect_rows_from_content(table_image)
        
        # Create cell grid
        cells = self._create_cell_grid(columns, rows)
        
        # Filter and validate cells
        valid_cells = self._validate_cells(cells, table_image)
        
        segmentation_result = {
            'columns': columns,
            'rows': rows,
            'cells': valid_cells,
            'num_rows': len(rows),
            'num_columns': len(columns),
            'total_cells': len(valid_cells)
        }
        
        logger.info(f"Segmented table: {len(rows)} rows × {len(columns)} columns = {len(valid_cells)} cells")
        return segmentation_result
    
    def _define_columns(self, column_lines: List[int], width: int, height: int) -> List[Dict]:
        """
        Define column boundaries from vertical lines.
        
        Args:
            column_lines: X-coordinates of vertical lines
            width: Image width
            height: Image height
            
        Returns:
            List of column definitions
        """
        columns = []
        
        # Ensure we have left and right boundaries
        if not column_lines or column_lines[0] > 5:
            column_lines.insert(0, 0)
        if not column_lines or column_lines[-1] < width - 5:
            column_lines.append(width)
        
        # Sort column lines
        column_lines = sorted(set(column_lines))
        
        # Create column definitions
        for i in range(len(column_lines) - 1):
            col_x = column_lines[i]
            col_width = column_lines[i + 1] - col_x
            
            if col_width >= self.min_col_width:
                column = {
                    'index': len(columns),
                    'x': col_x,
                    'width': col_width,
                    'center_x': col_x + col_width // 2
                }
                columns.append(column)
        
        return columns
    
    def _detect_rows_from_ocr(self, ocr_results: List[Dict], image_height: int) -> List[Dict]:
        """
        Detect row boundaries using OCR text positions.
        
        Args:
            ocr_results: OCR results with bounding boxes
            image_height: Height of table image
            
        Returns:
            List of row definitions
        """
        if not ocr_results:
            return []
        
        # Extract Y-coordinates of text centers
        y_positions = []
        for result in ocr_results:
            if 'bbox' in result:
                x, y, w, h = result['bbox']
                center_y = y + h // 2
                y_positions.append(center_y)
        
        if not y_positions:
            return []
        
        # Cluster Y-positions to find rows
        y_positions = np.array(y_positions).reshape(-1, 1)
        
        # Use DBSCAN clustering to group nearby Y-positions
        clustering = DBSCAN(eps=15, min_samples=1).fit(y_positions)
        labels = clustering.labels_
        
        # Group positions by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            clusters[label].append(y_positions[i][0])
        
        # Calculate row centers and boundaries
        rows = []
        for cluster_id, positions in clusters.items():
            if cluster_id == -1:  # Noise points
                continue
            
            center_y = int(np.mean(positions))
            min_y = int(np.min(positions))
            max_y = int(np.max(positions))
            
            # Estimate row boundaries
            row_height = max(max_y - min_y, self.min_row_height)
            row_top = max(0, center_y - row_height // 2)
            row_bottom = min(image_height, center_y + row_height // 2)
            
            row = {
                'index': len(rows),
                'y': row_top,
                'height': row_bottom - row_top,
                'center_y': center_y
            }
            rows.append(row)
        
        # Sort rows by Y position
        rows.sort(key=lambda r: r['y'])
        
        # Update row indices
        for i, row in enumerate(rows):
            row['index'] = i
        
        return rows
    
    def _detect_rows_from_content(self, table_image: np.ndarray) -> List[Dict]:
        """
        Detect rows using content analysis when OCR results are not available.
        
        Args:
            table_image: Table image
            
        Returns:
            List of row definitions
        """
        # Convert to grayscale
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY) if len(table_image.shape) == 3 else table_image
        
        height, width = gray.shape
        
        # Calculate horizontal projection (sum of pixels in each row)
        horizontal_projection = np.sum(gray < 128, axis=1)  # Count dark pixels
        
        # Smooth the projection
        kernel_size = 5
        smoothed = np.convolve(horizontal_projection, np.ones(kernel_size) / kernel_size, mode='same')
        
        # Find peaks (rows with content)
        threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
        content_rows = np.where(smoothed > threshold)[0]
        
        if len(content_rows) == 0:
            return []
        
        # Group consecutive rows
        row_groups = []
        current_group = [content_rows[0]]
        
        for i in range(1, len(content_rows)):
            if content_rows[i] - content_rows[i-1] <= 3:  # Gap tolerance
                current_group.append(content_rows[i])
            else:
                if len(current_group) >= 3:  # Minimum group size
                    row_groups.append(current_group)
                current_group = [content_rows[i]]
        
        # Add last group
        if len(current_group) >= 3:
            row_groups.append(current_group)
        
        # Create row definitions
        rows = []
        for group in row_groups:
            row_top = min(group)
            row_bottom = max(group)
            row_height = row_bottom - row_top + 1
            
            if row_height >= self.min_row_height:
                row = {
                    'index': len(rows),
                    'y': row_top,
                    'height': row_height,
                    'center_y': (row_top + row_bottom) // 2
                }
                rows.append(row)
        
        return rows
    
    def _create_cell_grid(self, columns: List[Dict], rows: List[Dict]) -> List[Dict]:
        """
        Create cell grid from columns and rows.
        
        Args:
            columns: List of column definitions
            rows: List of row definitions
            
        Returns:
            List of cell definitions
        """
        cells = []
        
        for row in rows:
            for col in columns:
                cell = {
                    'row_index': row['index'],
                    'col_index': col['index'],
                    'x': col['x'],
                    'y': row['y'],
                    'width': col['width'],
                    'height': row['height'],
                    'center_x': col['center_x'],
                    'center_y': row['center_y'],
                    'area': col['width'] * row['height']
                }
                cells.append(cell)
        
        return cells
    
    def _validate_cells(self, cells: List[Dict], table_image: np.ndarray) -> List[Dict]:
        """
        Validate and filter cells based on size and content.
        
        Args:
            cells: List of cell definitions
            table_image: Table image for validation
            
        Returns:
            List of valid cells
        """
        valid_cells = []
        height, width = table_image.shape[:2]
        
        for cell in cells:
            # Check if cell is within image bounds
            if (cell['x'] >= 0 and cell['y'] >= 0 and 
                cell['x'] + cell['width'] <= width and 
                cell['y'] + cell['height'] <= height):
                
                # Check minimum size
                if cell['width'] >= self.min_col_width and cell['height'] >= self.min_row_height:
                    valid_cells.append(cell)
        
        return valid_cells
    
    def extract_cell_image(self, table_image: np.ndarray, cell: Dict, 
                          padding: int = 2) -> np.ndarray:
        """
        Extract individual cell image with padding.
        
        Args:
            table_image: Full table image
            cell: Cell definition
            padding: Padding around cell
            
        Returns:
            Cropped cell image
        """
        height, width = table_image.shape[:2]
        
        # Calculate padded boundaries
        x1 = max(0, cell['x'] - padding)
        y1 = max(0, cell['y'] - padding)
        x2 = min(width, cell['x'] + cell['width'] + padding)
        y2 = min(height, cell['y'] + cell['height'] + padding)
        
        # Extract cell image
        cell_image = table_image[y1:y2, x1:x2]
        
        return cell_image


class TargetColumnExtractor:
    """Extracts specific target columns from segmented table."""
    
    def __init__(self, target_column_indices: List[int]):
        """
        Initialize target column extractor.
        
        Args:
            target_column_indices: Indices of columns to extract (0-based)
        """
        self.target_column_indices = target_column_indices
        self.column_names = {
            2: 'attendance_time',  # وقت التسليم
            3: 'rank',             # نقطة الحراسة  
            4: 'shift_id'          # الوردية
        }
    
    def extract_target_cells(self, segmentation_result: Dict) -> Dict:
        """
        Extract cells from target columns only.
        
        Args:
            segmentation_result: Result from table segmentation
            
        Returns:
            Dictionary with target cells organized by column type
        """
        all_cells = segmentation_result['cells']
        
        # Group cells by column and row
        target_cells = {
            'attendance_time': [],
            'rank': [],
            'shift_id': []
        }
        
        for cell in all_cells:
            col_index = cell['col_index']
            
            if col_index in self.target_column_indices:
                column_name = self.column_names.get(col_index)
                if column_name:
                    target_cells[column_name].append(cell)
        
        # Sort cells by row index within each column
        for column_name in target_cells:
            target_cells[column_name].sort(key=lambda c: c['row_index'])
        
        # Calculate statistics
        extraction_stats = {
            'total_rows': segmentation_result['num_rows'],
            'extracted_cells': {
                'attendance_time': len(target_cells['attendance_time']),
                'rank': len(target_cells['rank']),
                'shift_id': len(target_cells['shift_id'])
            }
        }
        
        result = {
            'target_cells': target_cells,
            'stats': extraction_stats,
            'segmentation_info': segmentation_result
        }
        
        logger.info(f"Extracted target cells: {extraction_stats['extracted_cells']}")
        return result
    
    def create_cell_crops(self, table_image: np.ndarray, 
                         target_cells: Dict, 
                         padding: int = 3) -> Dict:
        """
        Create cropped images for all target cells.
        
        Args:
            table_image: Full table image
            target_cells: Target cells from extract_target_cells
            padding: Padding around each cell
            
        Returns:
            Dictionary with cropped cell images
        """
        segmenter = TableSegmenter()
        cell_crops = {
            'attendance_time': [],
            'rank': [],
            'shift_id': []
        }
        
        for column_name, cells in target_cells.items():
            for cell in cells:
                # Extract cell image
                cell_image = segmenter.extract_cell_image(table_image, cell, padding)
                
                # Add metadata
                cell_crop = {
                    'image': cell_image,
                    'row_index': cell['row_index'],
                    'col_index': cell['col_index'],
                    'bbox': (cell['x'], cell['y'], cell['width'], cell['height']),
                    'column_type': column_name
                }
                
                cell_crops[column_name].append(cell_crop)
        
        logger.info(f"Created cell crops: {len(cell_crops['attendance_time'])} + "
                   f"{len(cell_crops['rank'])} + {len(cell_crops['shift_id'])} cells")
        
        return cell_crops


class RowDetector:
    """Specialized row detection for tables without horizontal lines."""
    
    def __init__(self):
        """Initialize row detector."""
        pass
    
    def detect_filled_rows(self, table_image: np.ndarray, 
                          columns: List[Dict]) -> Tuple[List[Dict], int]:
        """
        Detect rows that contain handwritten content.
        
        Args:
            table_image: Table image
            columns: Column definitions
            
        Returns:
            Tuple of (detected_rows, filled_row_count)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY) if len(table_image.shape) == 3 else table_image
        
        height, width = gray.shape
        
        # Analyze content in target columns only
        target_col_indices = [2, 3, 4]  # attendance_time, rank, shift_id
        target_columns = [col for col in columns if col['index'] in target_col_indices]
        
        if not target_columns:
            logger.warning("No target columns found for row detection")
            return [], 0
        
        # Create mask for target column areas
        mask = np.zeros_like(gray)
        for col in target_columns:
            x1, x2 = col['x'], col['x'] + col['width']
            mask[:, x1:x2] = 255
        
        # Apply mask to image
        masked_image = cv2.bitwise_and(gray, mask)
        
        # Calculate horizontal projection in target columns
        horizontal_projection = np.sum(masked_image < 200, axis=1)  # Count dark pixels
        
        # Smooth projection
        kernel_size = 3
        smoothed = np.convolve(horizontal_projection, np.ones(kernel_size) / kernel_size, mode='same')
        
        # Find content peaks
        threshold = np.mean(smoothed) + 0.3 * np.std(smoothed)
        content_positions = np.where(smoothed > threshold)[0]
        
        if len(content_positions) == 0:
            return [], 0
        
        # Group consecutive positions into rows
        rows = []
        current_start = content_positions[0]
        current_end = current_start
        
        for i in range(1, len(content_positions)):
            if content_positions[i] - content_positions[i-1] <= 5:  # Gap tolerance
                current_end = content_positions[i]
            else:
                # End current row
                if current_end - current_start >= 10:  # Minimum row height
                    row = {
                        'index': len(rows),
                        'y': current_start,
                        'height': current_end - current_start + 1,
                        'center_y': (current_start + current_end) // 2,
                        'has_content': True
                    }
                    rows.append(row)
                
                current_start = content_positions[i]
                current_end = current_start
        
        # Add last row
        if current_end - current_start >= 10:
            row = {
                'index': len(rows),
                'y': current_start,
                'height': current_end - current_start + 1,
                'center_y': (current_start + current_end) // 2,
                'has_content': True
            }
            rows.append(row)
        
        filled_count = len(rows)
        logger.info(f"Detected {filled_count} filled rows")
        
        return rows, filled_count