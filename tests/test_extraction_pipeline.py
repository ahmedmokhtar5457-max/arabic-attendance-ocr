"""
Test Suite for Arabic Attendance Extraction Pipeline

Unit and integration tests for the main extraction pipeline.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

from src.core.extraction_pipeline import AttendanceExtractionPipeline
from src.preprocessing.pdf_processor import PDFProcessor, ImagePreprocessor
from src.core.table_detector import TableDetector, TemplateBasedTableDetector
from src.models.arabic_digit_cnn import ArabicDigitRecognizer
from src.recognition.rank_recognizer import RankRecognizer
from src.utils.arabic_converter import ArabicToEnglishConverter
from src.utils.validation import DataValidator


class TestAttendanceExtractionPipeline:
    """Test cases for the main extraction pipeline."""
    
    @pytest.fixture
    def pipeline_config(self):
        """Test configuration for pipeline."""
        return {
            'use_gpu': False,
            'target_dpi': 300,
            'min_confidence': 0.5,
            'target_columns': [2, 3, 4],
            'max_images_per_pdf': 10,
            'enable_debug_output': False,
            'validation_rules': {
                'attendance_time_range': (1, 24),
                'max_shift_id': 999,
                'require_all_fields': False
            }
        }
    
    @pytest.fixture
    def mock_pipeline(self, pipeline_config):
        """Create pipeline with mocked components."""
        with patch.multiple(
            'src.core.extraction_pipeline',
            PDFProcessor=Mock(),
            ImagePreprocessor=Mock(),
            TableDetector=Mock(),
            TemplateBasedTableDetector=Mock(),
            TableSegmenter=Mock(),
            TargetColumnExtractor=Mock(),
            RowDetector=Mock(),
            ArabicDigitRecognizer=Mock(),
            DigitSegmenter=Mock(),
            RankRecognizer=Mock(),
            RankValidator=Mock(),
            ArabicToEnglishConverter=Mock(),
            DataValidator=Mock()
        ):
            pipeline = AttendanceExtractionPipeline(pipeline_config)
            yield pipeline
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add some text-like rectangles
        cv2.rectangle(image, (50, 50), (150, 80), (0, 0, 0), -1)
        cv2.rectangle(image, (200, 50), (300, 80), (0, 0, 0), -1)
        cv2.rectangle(image, (350, 50), (450, 80), (0, 0, 0), -1)
        
        return image
    
    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization."""
        with patch.multiple(
            'src.core.extraction_pipeline',
            PDFProcessor=Mock(),
            ImagePreprocessor=Mock(),
            TableDetector=Mock(),
            TemplateBasedTableDetector=Mock(),
            TableSegmenter=Mock(),
            TargetColumnExtractor=Mock(),
            RowDetector=Mock(),
            ArabicDigitRecognizer=Mock(),
            DigitSegmenter=Mock(),
            RankRecognizer=Mock(),
            RankValidator=Mock(),
            ArabicToEnglishConverter=Mock(),
            DataValidator=Mock()
        ):
            pipeline = AttendanceExtractionPipeline(pipeline_config)
            
            assert pipeline.config == pipeline_config
            assert pipeline.stats['processed_files'] == 0
            assert pipeline.stats['processed_images'] == 0
    
    def test_process_single_image_success(self, mock_pipeline, sample_image):
        """Test successful single image processing."""
        # Mock the processing steps
        mock_pipeline.image_preprocessor.preprocess_image.return_value = (
            sample_image, {'skew_angle': 0.5}
        )
        
        mock_pipeline.template_detector.detect_table_by_template.return_value = {
            'bbox': (10, 10, 580, 380),
            'confidence': 0.9,
            'template_matched': True
        }
        
        mock_pipeline.table_segmenter.segment_table.return_value = {
            'num_rows': 5,
            'num_columns': 7,
            'total_cells': 35,
            'columns': [{'index': i, 'x': i*80, 'width': 80} for i in range(7)]
        }
        
        mock_pipeline.target_extractor.extract_target_cells.return_value = {
            'target_cells': {
                'attendance_time': [{'row_index': 0}, {'row_index': 1}],
                'rank': [{'row_index': 0}, {'row_index': 1}],
                'shift_id': [{'row_index': 0}, {'row_index': 1}]
            }
        }
        
        mock_pipeline.target_extractor.create_cell_crops.return_value = {
            'attendance_time': [
                {'image': sample_image[50:80, 50:150], 'row_index': 0},
                {'image': sample_image[50:80, 50:150], 'row_index': 1}
            ],
            'rank': [
                {'image': sample_image[50:80, 200:300], 'row_index': 0},
                {'image': sample_image[50:80, 200:300], 'row_index': 1}
            ],
            'shift_id': [
                {'image': sample_image[50:80, 350:450], 'row_index': 0},
                {'image': sample_image[50:80, 350:450], 'row_index': 1}
            ]
        }
        
        mock_pipeline.row_detector.detect_filled_rows.return_value = ([], 2)
        
        # Mock recognition results
        mock_pipeline.digit_segmenter.segment_digits.return_value = [sample_image[50:80, 50:80]]
        mock_pipeline.digit_recognizer.recognize_number_sequence.return_value = ('12', 0.9)
        mock_pipeline.arabic_converter.convert_digits.return_value = '12'
        mock_pipeline.rank_recognizer.recognize_rank.return_value = ('GUARD', 0.8, {})
        mock_pipeline.rank_validator.validate_rank.return_value = (True, 'Valid rank')
        mock_pipeline.data_validator.validate_row.return_value = {'is_valid': True, 'errors': []}
        
        # Process image
        result = mock_pipeline.process_single_image(sample_image, 'test_image')
        
        # Assertions
        assert result['success'] is True
        assert result['image_id'] == 'test_image'
        assert 'preprocessing_info' in result
        assert 'table_info' in result
        assert 'extracted_data' in result
        assert result['segmentation_stats']['num_rows'] == 5
        assert result['segmentation_stats']['num_columns'] == 7
    
    def test_process_single_image_no_table(self, mock_pipeline, sample_image):
        """Test image processing when no table is detected."""
        # Mock no table detection
        mock_pipeline.image_preprocessor.preprocess_image.return_value = (
            sample_image, {'skew_angle': 0.0}
        )
        mock_pipeline.template_detector.detect_table_by_template.return_value = None
        mock_pipeline.table_detector.detect_tables.return_value = []
        
        result = mock_pipeline.process_single_image(sample_image, 'test_image')
        
        assert result['success'] is False
        assert 'No table detected' in result['error']
    
    def test_process_pdf_file_success(self, mock_pipeline):
        """Test successful PDF file processing."""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name
        
        try:
            # Mock PDF processing
            sample_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
            mock_pipeline.pdf_processor.process_pdf_file.return_value = [
                (sample_image, {'estimated_dpi': 300, 'is_valid': True}),
                (sample_image, {'estimated_dpi': 300, 'is_valid': True})
            ]
            
            # Mock successful image processing
            with patch.object(mock_pipeline, 'process_single_image') as mock_process:
                mock_process.return_value = {
                    'success': True,
                    'image_id': 'page_1',
                    'extracted_data': [
                        {'row_index': 0, 'attendance_time': '12', 'rank': 'GUARD', 'shift_id': '1'}
                    ],
                    'filled_rows_count': 1,
                    'confidence_scores': {'average_confidence': 0.8}
                }
                
                result = mock_pipeline.process_pdf_file(temp_pdf_path)
            
            # Assertions
            assert 'pdf_path' in result
            assert result['total_images'] == 2
            assert result['successful_extractions'] == 2
            assert len(result['image_results']) == 2
            assert 'summary' in result
            
        finally:
            # Clean up
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
    
    def test_statistics_tracking(self, mock_pipeline):
        """Test statistics tracking functionality."""
        initial_stats = mock_pipeline.get_processing_statistics()
        
        assert initial_stats['processed_files'] == 0
        assert initial_stats['processed_images'] == 0
        
        # Simulate processing
        mock_pipeline.stats['processed_files'] = 5
        mock_pipeline.stats['processed_images'] = 20
        mock_pipeline.stats['successful_extractions'] = 18
        
        updated_stats = mock_pipeline.get_processing_statistics()
        assert updated_stats['processed_files'] == 5
        assert updated_stats['processed_images'] == 20
        assert updated_stats['successful_extractions'] == 18
        
        # Test reset
        mock_pipeline.reset_statistics()
        reset_stats = mock_pipeline.get_processing_statistics()
        assert reset_stats['processed_files'] == 0


class TestPDFProcessor:
    """Test cases for PDF processing."""
    
    def test_validate_image_quality(self):
        """Test image quality validation."""
        processor = PDFProcessor(target_dpi=300)
        
        # Create test image
        good_image = np.random.randint(0, 255, (2480, 3508, 3), dtype=np.uint8)
        is_valid, metrics = processor.validate_image_quality(good_image)
        
        assert is_valid is True
        assert metrics['width'] == 3508
        assert metrics['height'] == 2480
        assert metrics['estimated_dpi'] > 200
        
        # Test low quality image
        bad_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        is_valid, metrics = processor.validate_image_quality(bad_image)
        
        assert is_valid is False
        assert metrics['estimated_dpi'] < 200


class TestArabicConverter:
    """Test cases for Arabic to English conversion."""
    
    def test_convert_digits(self):
        """Test Arabic digit conversion."""
        converter = ArabicToEnglishConverter()
        
        # Test basic conversion
        assert converter.convert_digits('١٢٣') == '123'
        assert converter.convert_digits('٠٩') == '09'
        assert converter.convert_digits('mixed ١٢ text') == 'mixed 12 text'
        
        # Test empty/None input
        assert converter.convert_digits('') == ''
        assert converter.convert_digits(None) is None
    
    def test_validate_attendance_time(self):
        """Test attendance time validation."""
        converter = ArabicToEnglishConverter()
        
        # Valid times
        is_valid, value, error = converter.validate_attendance_time('١٢')
        assert is_valid is True
        assert value == '12'
        assert error is None
        
        is_valid, value, error = converter.validate_attendance_time('24')
        assert is_valid is True
        assert value == '24'
        
        # Invalid times
        is_valid, value, error = converter.validate_attendance_time('٢٥')
        assert is_valid is False
        assert 'out of range' in error
        
        is_valid, value, error = converter.validate_attendance_time('abc')
        assert is_valid is False
        assert 'No numeric value found' in error


class TestDataValidator:
    """Test cases for data validation."""
    
    def test_validate_row(self):
        """Test row validation."""
        validator = DataValidator()
        
        # Valid row
        valid_row = {
            'attendance_time': '12',
            'rank': 'GUARD',
            'shift_id': '1'
        }
        result = validator.validate_row(valid_row)
        assert result['is_valid'] is True
        assert len(result['errors']) == 0
        
        # Invalid attendance time
        invalid_row = {
            'attendance_time': '25',  # Out of range
            'rank': 'GUARD',
            'shift_id': '1'
        }
        result = validator.validate_row(invalid_row)
        assert result['is_valid'] is False
        assert any('out of range' in error for error in result['errors'])
    
    def test_validate_batch(self):
        """Test batch validation."""
        validator = DataValidator()
        
        data_rows = [
            {'attendance_time': '12', 'rank': 'GUARD', 'shift_id': '1'},
            {'attendance_time': '25', 'rank': 'INVALID_RANK_VERY_LONG_NAME', 'shift_id': '1'},
            {'attendance_time': '8', 'rank': 'OFFICER', 'shift_id': '2'}
        ]
        
        result = validator.validate_batch(data_rows)
        
        assert result['total_rows'] == 3
        assert result['valid_rows'] == 2
        assert result['invalid_rows'] == 1
        assert len(result['row_validations']) == 3


class TestRankRecognizer:
    """Test cases for rank recognition."""
    
    @pytest.fixture
    def mock_rank_recognizer(self):
        """Create rank recognizer with mocked OCR."""
        with patch('src.recognition.rank_recognizer.PaddleOCR'):
            recognizer = RankRecognizer(use_gpu=False)
            yield recognizer
    
    def test_vocabulary_validation(self, mock_rank_recognizer):
        """Test vocabulary validation."""
        # Valid ranks
        match, similarity = mock_rank_recognizer.validate_against_vocabulary('GUARD')
        assert match == 'GUARD'
        assert similarity == 1.0
        
        # Fuzzy matching
        match, similarity = mock_rank_recognizer.validate_against_vocabulary('GVARD')
        assert match == 'GUARD'
        assert similarity > 0.7
        
        # Invalid rank
        match, similarity = mock_rank_recognizer.validate_against_vocabulary('INVALID_RANK_XYZ')
        assert match is None
        assert similarity < 0.7
    
    def test_clean_text(self, mock_rank_recognizer):
        """Test text cleaning."""
        # Test cleaning
        cleaned = mock_rank_recognizer._clean_text('  guard  ')
        assert cleaned == 'GUARD'
        
        cleaned = mock_rank_recognizer._clean_text('G@U#A$R%D')
        assert cleaned == 'GUARD'
        
        # Test OCR corrections
        cleaned = mock_rank_recognizer._clean_text('QUARD')
        assert cleaned == 'GUARD'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])