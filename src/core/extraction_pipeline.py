"""
Main Data Extraction and Validation Pipeline

Orchestrates the complete extraction process from PDF to structured data.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging
from pathlib import Path
import json
from datetime import datetime
import traceback

from ..preprocessing.pdf_processor import PDFProcessor, ImagePreprocessor
from ..core.table_detector import TableDetector, TemplateBasedTableDetector
from ..core.segmentation import TableSegmenter, TargetColumnExtractor, RowDetector
from ..models.arabic_digit_cnn import ArabicDigitRecognizer, DigitSegmenter
from ..recognition.rank_recognizer import RankRecognizer, RankValidator
from ..utils.arabic_converter import ArabicToEnglishConverter
from ..utils.validation import DataValidator

logger = logging.getLogger(__name__)


class AttendanceExtractionPipeline:
    """Main pipeline for extracting attendance data from Arabic forms."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize extraction pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or self._get_default_config()
        
        # Initialize components
        self._initialize_components()
        
        # Processing statistics
        self.stats = {
            'processed_files': 0,
            'processed_images': 0,
            'detected_tables': 0,
            'extracted_rows': 0,
            'successful_extractions': 0,
            'errors': []
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'use_gpu': False,
            'target_dpi': 300,
            'min_confidence': 0.5,
            'target_columns': [2, 3, 4],  # attendance_time, rank, shift_id
            'max_images_per_pdf': 50,
            'enable_debug_output': False,
            'output_format': 'json',
            'validation_rules': {
                'attendance_time_range': (1, 24),
                'max_shift_id': 999,
                'require_all_fields': False
            }
        }
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # PDF and image processing
            self.pdf_processor = PDFProcessor(target_dpi=self.config['target_dpi'])
            self.image_preprocessor = ImagePreprocessor()
            
            # Table detection
            self.table_detector = TableDetector(
                use_gpu=self.config['use_gpu'],
                lang='ar'
            )
            self.template_detector = TemplateBasedTableDetector()
            
            # Segmentation
            self.table_segmenter = TableSegmenter()
            self.target_extractor = TargetColumnExtractor(self.config['target_columns'])
            self.row_detector = RowDetector()
            
            # Recognition models
            self.digit_recognizer = ArabicDigitRecognizer(
                device='cuda' if self.config['use_gpu'] else 'cpu'
            )
            self.digit_segmenter = DigitSegmenter()
            self.rank_recognizer = RankRecognizer(use_gpu=self.config['use_gpu'])
            self.rank_validator = RankValidator()
            
            # Utilities
            self.arabic_converter = ArabicToEnglishConverter()
            self.data_validator = DataValidator(self.config['validation_rules'])
            
            logger.info("Pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise
    
    def process_pdf_file(self, pdf_path: str) -> Dict:
        """
        Process a single PDF file and extract attendance data.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extraction results
        """
        start_time = datetime.now()
        
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Extract images from PDF
            processed_images = self.pdf_processor.process_pdf_file(pdf_path)
            
            if not processed_images:
                return self._create_error_result("No valid images found in PDF", pdf_path)
            
            # Limit number of images if configured
            max_images = self.config.get('max_images_per_pdf', 50)
            if len(processed_images) > max_images:
                logger.warning(f"PDF contains {len(processed_images)} images, processing first {max_images}")
                processed_images = processed_images[:max_images]
            
            # Process each image
            all_results = []
            for i, (image, quality_metrics) in enumerate(processed_images):
                logger.info(f"Processing image {i + 1}/{len(processed_images)}")
                
                try:
                    image_result = self.process_single_image(image, f"page_{i + 1}")
                    image_result['quality_metrics'] = quality_metrics
                    all_results.append(image_result)
                    
                except Exception as e:
                    logger.error(f"Error processing image {i + 1}: {str(e)}")
                    error_result = self._create_error_result(f"Image processing error: {str(e)}", f"page_{i + 1}")
                    all_results.append(error_result)
            
            # Compile final results
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'pdf_path': pdf_path,
                'processing_time': processing_time,
                'total_images': len(processed_images),
                'successful_extractions': len([r for r in all_results if r.get('success', False)]),
                'image_results': all_results,
                'summary': self._create_summary(all_results),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update statistics
            self.stats['processed_files'] += 1
            self.stats['processed_images'] += len(processed_images)
            self.stats['successful_extractions'] += result['successful_extractions']
            
            logger.info(f"PDF processing completed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return self._create_error_result(f"PDF processing error: {str(e)}", pdf_path)
    
    def process_single_image(self, image: np.ndarray, image_id: str) -> Dict:
        """
        Process a single image and extract attendance data.
        
        Args:
            image: Input image
            image_id: Identifier for the image
            
        Returns:
            Dictionary with extraction results
        """
        try:
            # Step 1: Preprocess image
            preprocessed_image, preprocessing_info = self.image_preprocessor.preprocess_image(image)
            
            # Step 2: Detect table regions
            table_info = self._detect_table_region(preprocessed_image)
            
            if not table_info:
                return self._create_error_result("No table detected", image_id)
            
            # Step 3: Extract table image
            table_image = self._extract_table_image(preprocessed_image, table_info)
            
            # Step 4: Segment table into cells
            segmentation_result = self._segment_table(table_image, table_info)
            
            # Step 5: Extract target column data
            extraction_result = self._extract_target_data(table_image, segmentation_result)
            
            # Step 6: Validate and format results
            validated_result = self._validate_and_format_results(extraction_result)
            
            # Compile final result
            result = {
                'image_id': image_id,
                'success': True,
                'preprocessing_info': preprocessing_info,
                'table_info': table_info,
                'segmentation_stats': {
                    'num_rows': segmentation_result.get('num_rows', 0),
                    'num_columns': segmentation_result.get('num_columns', 0),
                    'total_cells': segmentation_result.get('total_cells', 0)
                },
                'extracted_data': validated_result['data'],
                'validation_results': validated_result['validation'],
                'confidence_scores': validated_result['confidence'],
                'filled_rows_count': validated_result.get('filled_rows_count', 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_id}: {str(e)}")
            logger.debug(traceback.format_exc())
            return self._create_error_result(f"Image processing error: {str(e)}", image_id)
    
    def _detect_table_region(self, image: np.ndarray) -> Optional[Dict]:
        """Detect table region in image."""
        try:
            # Try template-based detection first (more reliable for known layouts)
            template_result = self.template_detector.detect_table_by_template(image)
            
            if template_result and template_result.get('confidence', 0) > 0.8:
                logger.debug("Using template-based table detection")
                return template_result
            
            # Fallback to PaddleOCR detection
            paddle_results = self.table_detector.detect_tables(image)
            
            if paddle_results:
                logger.debug("Using PaddleOCR table detection")
                return paddle_results[0]  # Use the largest/most confident table
            
            return None
            
        except Exception as e:
            logger.error(f"Error in table detection: {str(e)}")
            return None
    
    def _extract_table_image(self, image: np.ndarray, table_info: Dict) -> np.ndarray:
        """Extract table region from image."""
        bbox = table_info.get('bbox')
        if bbox:
            x, y, w, h = bbox
            return image[y:y+h, x:x+w]
        return image
    
    def _segment_table(self, table_image: np.ndarray, table_info: Dict) -> Dict:
        """Segment table into rows and columns."""
        # Get column lines from table info
        column_lines = table_info.get('vertical_lines', [])
        
        # Segment table
        segmentation_result = self.table_segmenter.segment_table(
            table_image, 
            column_lines
        )
        
        return segmentation_result
    
    def _extract_target_data(self, table_image: np.ndarray, segmentation_result: Dict) -> Dict:
        """Extract data from target columns."""
        # Extract target cells
        target_cells_result = self.target_extractor.extract_target_cells(segmentation_result)
        
        # Create cell crops
        cell_crops = self.target_extractor.create_cell_crops(
            table_image, 
            target_cells_result['target_cells']
        )
        
        # Process each column type
        extracted_data = {
            'attendance_time': [],
            'rank': [],
            'shift_id': []
        }
        
        # Process attendance time (Arabic digits)
        for cell_crop in cell_crops['attendance_time']:
            result = self._process_attendance_time_cell(cell_crop)
            extracted_data['attendance_time'].append(result)
        
        # Process rank (English text)
        for cell_crop in cell_crops['rank']:
            result = self._process_rank_cell(cell_crop)
            extracted_data['rank'].append(result)
        
        # Process shift ID (Arabic digits)
        for cell_crop in cell_crops['shift_id']:
            result = self._process_shift_id_cell(cell_crop)
            extracted_data['shift_id'].append(result)
        
        # Detect filled rows
        filled_rows, filled_count = self.row_detector.detect_filled_rows(
            table_image, 
            segmentation_result.get('columns', [])
        )
        
        return {
            'data': extracted_data,
            'filled_rows_count': filled_count,
            'cell_crops_info': {
                'attendance_time': len(cell_crops['attendance_time']),
                'rank': len(cell_crops['rank']),
                'shift_id': len(cell_crops['shift_id'])
            }
        }
    
    def _process_attendance_time_cell(self, cell_crop: Dict) -> Dict:
        """Process attendance time cell (Arabic digits 1-24)."""
        try:
            cell_image = cell_crop['image']
            
            # Segment digits if multiple
            digit_images = self.digit_segmenter.segment_digits(cell_image)
            
            if not digit_images:
                return {
                    'row_index': cell_crop['row_index'],
                    'value': None,
                    'confidence': 0.0,
                    'error': 'No digits detected'
                }
            
            # Recognize digit sequence
            number, confidence = self.digit_recognizer.recognize_number_sequence(digit_images)
            
            # Convert Arabic to English digits
            english_number = self.arabic_converter.convert_digits(number)
            
            # Validate range (1-24)
            try:
                int_value = int(english_number) if english_number else 0
                if not (1 <= int_value <= 24):
                    confidence *= 0.5  # Reduce confidence for out-of-range values
            except ValueError:
                int_value = None
                confidence = 0.0
            
            return {
                'row_index': cell_crop['row_index'],
                'value': english_number,
                'confidence': confidence,
                'original_arabic': number,
                'valid_range': 1 <= (int_value or 0) <= 24
            }
            
        except Exception as e:
            return {
                'row_index': cell_crop['row_index'],
                'value': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _process_rank_cell(self, cell_crop: Dict) -> Dict:
        """Process rank cell (English text)."""
        try:
            cell_image = cell_crop['image']
            
            # Recognize rank
            rank, confidence, metadata = self.rank_recognizer.recognize_rank(cell_image)
            
            # Validate rank
            is_valid, validation_msg = self.rank_validator.validate_rank(rank)
            
            return {
                'row_index': cell_crop['row_index'],
                'value': rank,
                'confidence': confidence,
                'is_valid': is_valid,
                'validation_message': validation_msg,
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'row_index': cell_crop['row_index'],
                'value': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _process_shift_id_cell(self, cell_crop: Dict) -> Dict:
        """Process shift ID cell (Arabic digits)."""
        try:
            cell_image = cell_crop['image']
            
            # Segment digits
            digit_images = self.digit_segmenter.segment_digits(cell_image)
            
            if not digit_images:
                return {
                    'row_index': cell_crop['row_index'],
                    'value': None,
                    'confidence': 0.0,
                    'error': 'No digits detected'
                }
            
            # Recognize digit sequence
            number, confidence = self.digit_recognizer.recognize_number_sequence(digit_images)
            
            # Convert Arabic to English digits
            english_number = self.arabic_converter.convert_digits(number)
            
            return {
                'row_index': cell_crop['row_index'],
                'value': english_number,
                'confidence': confidence,
                'original_arabic': number
            }
            
        except Exception as e:
            return {
                'row_index': cell_crop['row_index'],
                'value': None,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _validate_and_format_results(self, extraction_result: Dict) -> Dict:
        """Validate and format extraction results."""
        data = extraction_result['data']
        
        # Organize data by rows
        max_rows = max(
            len(data['attendance_time']),
            len(data['rank']),
            len(data['shift_id'])
        )
        
        formatted_rows = []
        confidence_scores = []
        validation_results = []
        
        for row_idx in range(max_rows):
            # Get data for this row
            attendance_data = next((item for item in data['attendance_time'] if item['row_index'] == row_idx), None)
            rank_data = next((item for item in data['rank'] if item['row_index'] == row_idx), None)
            shift_data = next((item for item in data['shift_id'] if item['row_index'] == row_idx), None)
            
            # Format row data
            row_data = {
                'row_index': row_idx,
                'attendance_time': attendance_data.get('value') if attendance_data else None,
                'rank': rank_data.get('value') if rank_data else None,
                'shift_id': shift_data.get('value') if shift_data else None
            }
            
            # Calculate row confidence
            confidences = []
            if attendance_data and attendance_data.get('confidence'):
                confidences.append(attendance_data['confidence'])
            if rank_data and rank_data.get('confidence'):
                confidences.append(rank_data['confidence'])
            if shift_data and shift_data.get('confidence'):
                confidences.append(shift_data['confidence'])
            
            row_confidence = np.mean(confidences) if confidences else 0.0
            
            # Validate row
            row_validation = self.data_validator.validate_row(row_data)
            
            # Only include rows with some data
            if any(row_data[field] for field in ['attendance_time', 'rank', 'shift_id']):
                formatted_rows.append(row_data)
                confidence_scores.append(row_confidence)
                validation_results.append(row_validation)
        
        return {
            'data': formatted_rows,
            'confidence': {
                'row_confidences': confidence_scores,
                'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
                'min_confidence': min(confidence_scores) if confidence_scores else 0.0,
                'max_confidence': max(confidence_scores) if confidence_scores else 0.0
            },
            'validation': {
                'row_validations': validation_results,
                'valid_rows': len([v for v in validation_results if v.get('is_valid', False)]),
                'total_rows': len(validation_results)
            },
            'filled_rows_count': extraction_result.get('filled_rows_count', 0)
        }
    
    def _create_summary(self, image_results: List[Dict]) -> Dict:
        """Create summary of processing results."""
        successful_results = [r for r in image_results if r.get('success', False)]
        
        total_rows = sum(len(r.get('extracted_data', [])) for r in successful_results)
        total_filled_rows = sum(r.get('filled_rows_count', 0) for r in successful_results)
        
        # Calculate average confidence
        all_confidences = []
        for result in successful_results:
            confidence_info = result.get('confidence_scores', {})
            row_confidences = confidence_info.get('row_confidences', [])
            all_confidences.extend(row_confidences)
        
        return {
            'total_images': len(image_results),
            'successful_images': len(successful_results),
            'failed_images': len(image_results) - len(successful_results),
            'total_extracted_rows': total_rows,
            'total_filled_rows': total_filled_rows,
            'average_confidence': np.mean(all_confidences) if all_confidences else 0.0,
            'processing_success_rate': len(successful_results) / len(image_results) if image_results else 0.0
        }
    
    def _create_error_result(self, error_message: str, identifier: str) -> Dict:
        """Create error result dictionary."""
        return {
            'identifier': identifier,
            'success': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_processing_statistics(self) -> Dict:
        """Get processing statistics."""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset processing statistics."""
        self.stats = {
            'processed_files': 0,
            'processed_images': 0,
            'detected_tables': 0,
            'extracted_rows': 0,
            'successful_extractions': 0,
            'errors': []
        }