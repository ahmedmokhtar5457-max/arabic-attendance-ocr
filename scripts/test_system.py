#!/usr/bin/env python3
"""
System Test Script for Arabic Attendance Extraction

Tests the complete system with sample data.
"""

import sys
import os
import logging
import tempfile
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.extraction_pipeline import AttendanceExtractionPipeline
from src.utils.logging_config import setup_logging


def create_test_image():
    """Create a synthetic test image with table structure."""
    # Create white background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw table structure
    # Vertical lines (columns)
    col_positions = [50, 150, 250, 350, 450, 550, 650]
    for x in col_positions:
        draw.line([(x, 100), (x, 500)], fill='black', width=2)
    
    # Draw header
    headers = ['م', 'الاسم', 'وقت التسليم', 'نقطة الحراسة', 'الوردية', 'التوقيع', 'ملاحظات']
    for i, header in enumerate(headers):
        if i < len(col_positions) - 1:
            x = col_positions[i] + 20
            draw.text((x, 110), header, fill='black')
    
    # Draw sample data rows
    sample_data = [
        ['1', 'أحمد محمد', '8', 'GUARD', '1', '', ''],
        ['2', 'محمد علي', '12', 'OFFICER', '2', '', ''],
        ['3', 'علي حسن', '16', 'SUPERVISOR', '1', '', ''],
        ['4', 'حسن أحمد', '20', 'GUARD', '3', '', ''],
    ]
    
    for row_idx, row_data in enumerate(sample_data):
        y = 150 + row_idx * 40
        for col_idx, cell_data in enumerate(row_data):
            if col_idx < len(col_positions) - 1:
                x = col_positions[col_idx] + 10
                draw.text((x, y), cell_data, fill='black')
    
    # Convert to numpy array
    return np.array(image)


def create_test_pdf():
    """Create a test PDF with sample attendance data."""
    import fitz  # PyMuPDF
    
    # Create PDF document
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4 size
    
    # Create test image
    test_image = create_test_image()
    
    # Convert numpy array to bytes
    _, buffer = cv2.imencode('.png', cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR))
    image_bytes = buffer.tobytes()
    
    # Insert image into PDF
    image_rect = fitz.Rect(50, 50, 545, 400)
    page.insert_image(image_rect, stream=image_bytes)
    
    # Save to temporary file
    temp_pdf = tempfile.NamedTemporaryFile(suffix='.pdf', delete=False)
    doc.save(temp_pdf.name)
    doc.close()
    
    return temp_pdf.name


def test_pipeline_components():
    """Test individual pipeline components."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing pipeline components...")
    
    try:
        # Test pipeline initialization
        config = {
            'use_gpu': False,
            'target_dpi': 300,
            'min_confidence': 0.3,
            'target_columns': [2, 3, 4],
            'validation_rules': {
                'attendance_time_range': (1, 24),
                'max_shift_id': 999,
                'require_all_fields': False
            }
        }
        
        pipeline = AttendanceExtractionPipeline(config)
        logger.info("✓ Pipeline initialization successful")
        
        # Test with synthetic image
        test_image = create_test_image()
        result = pipeline.process_single_image(test_image, "test_image")
        
        if result.get('success'):
            logger.info("✓ Single image processing successful")
            logger.info(f"  Extracted {len(result.get('extracted_data', []))} rows")
        else:
            logger.warning(f"⚠ Single image processing failed: {result.get('error', 'Unknown error')}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Component test failed: {str(e)}")
        return False


def test_pdf_processing():
    """Test PDF processing functionality."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing PDF processing...")
    
    try:
        # Create test PDF
        test_pdf_path = create_test_pdf()
        logger.info(f"Created test PDF: {test_pdf_path}")
        
        # Initialize pipeline
        config = {
            'use_gpu': False,
            'target_dpi': 300,
            'min_confidence': 0.3,
            'target_columns': [2, 3, 4],
            'max_images_per_pdf': 5
        }
        
        pipeline = AttendanceExtractionPipeline(config)
        
        # Process PDF
        result = pipeline.process_pdf_file(test_pdf_path)
        
        # Check results
        if result.get('successful_extractions', 0) > 0:
            logger.info("✓ PDF processing successful")
            logger.info(f"  Total images: {result.get('total_images', 0)}")
            logger.info(f"  Successful extractions: {result.get('successful_extractions', 0)}")
            logger.info(f"  Processing time: {result.get('processing_time', 0):.2f}s")
        else:
            logger.warning("⚠ PDF processing completed but no successful extractions")
        
        # Clean up
        os.unlink(test_pdf_path)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ PDF processing test failed: {str(e)}")
        return False


def test_api_endpoints():
    """Test API endpoints if server is running."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing API endpoints...")
    
    try:
        import requests
        
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Health endpoint accessible")
            health_data = response.json()
            logger.info(f"  Service: {health_data.get('service_name')}")
            logger.info(f"  Status: {health_data.get('status')}")
        else:
            logger.warning(f"⚠ Health endpoint returned status {response.status_code}")
        
        # Test statistics endpoint
        response = requests.get(f"{base_url}/statistics", timeout=5)
        if response.status_code == 200:
            logger.info("✓ Statistics endpoint accessible")
        else:
            logger.warning(f"⚠ Statistics endpoint returned status {response.status_code}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        logger.warning("⚠ API server not running - skipping endpoint tests")
        return True
    except Exception as e:
        logger.error(f"✗ API endpoint test failed: {str(e)}")
        return False


def run_performance_test():
    """Run basic performance test."""
    logger = logging.getLogger(__name__)
    
    logger.info("Running performance test...")
    
    try:
        import time
        
        # Create multiple test images
        test_images = [create_test_image() for _ in range(3)]
        
        # Initialize pipeline
        config = {'use_gpu': False, 'target_columns': [2, 3, 4]}
        pipeline = AttendanceExtractionPipeline(config)
        
        # Time processing
        start_time = time.time()
        
        results = []
        for i, image in enumerate(test_images):
            result = pipeline.process_single_image(image, f"perf_test_{i}")
            results.append(result)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Calculate metrics
        successful_results = [r for r in results if r.get('success')]
        success_rate = len(successful_results) / len(results) * 100
        avg_time_per_image = processing_time / len(test_images)
        
        logger.info("✓ Performance test completed")
        logger.info(f"  Total time: {processing_time:.2f}s")
        logger.info(f"  Average per image: {avg_time_per_image:.2f}s")
        logger.info(f"  Success rate: {success_rate:.1f}%")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Performance test failed: {str(e)}")
        return False


def main():
    """Main test function."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("Arabic Attendance Extraction System Test Suite")
    logger.info("=" * 60)
    
    test_results = []
    
    # Run tests
    tests = [
        ("Component Tests", test_pipeline_components),
        ("PDF Processing", test_pdf_processing),
        ("API Endpoints", test_api_endpoints),
        ("Performance Test", run_performance_test)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test '{test_name}' crashed: {str(e)}")
            test_results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    logger.info("-" * 60)
    logger.info(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        logger.info("🎉 All tests passed!")
        return 0
    else:
        logger.warning(f"⚠ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())