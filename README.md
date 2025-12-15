# Offline Arabic Handwritten Attendance Data Extraction System

## Overview

This system is designed to extract specific handwritten data from Arabic attendance templates using offline OCR technology. The system processes approximately 100 images per day from PDF files and outputs structured data for review and validation.

## Key Features

- **Fully Offline**: No internet dependency
- **Arabic Optimized**: Specialized for handwritten Arabic numbers and text
- **Template-Driven**: Maximizes OCR accuracy through optimized template design
- **FastAPI Service**: Local API for processing jobs
- **Review-Friendly**: Structured output with confidence scoring

## Target Data Fields

The system extracts only the following columns from attendance tables:

1. **Attendance Time (وقت التسليم)** - Arabic numbers (1-24)
2. **Shift ID (الوردية)** - Arabic numbers
3. **Rank/Guard Point (نقطة الحراسة)** - English words (fixed vocabulary)

## System Requirements

- Python 3.8+
- CPU: Multi-core recommended
- RAM: 8GB minimum, 16GB recommended
- GPU: Optional (CUDA-compatible for faster processing)
- Disk: 10GB+ for models and processing

## Project Structure

```
arabic_attendance_ocr/
├── src/
│   ├── api/                 # FastAPI service
│   ├── core/               # Core processing pipeline
│   ├── models/             # ML models (CNN, OCR)
│   ├── preprocessing/      # Image and PDF processing
│   ├── recognition/        # Digit and text recognition
│   └── utils/              # Utilities and helpers
├── templates/              # Template designs and specifications
├── data/                   # Training data and samples
├── logs/                   # Processing logs
├── tests/                  # Unit and integration tests
└── docs/                   # Documentation
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python scripts/download_models.py
```

## Usage

```bash
# Start the FastAPI service
python -m src.api.main

# Process a PDF file
curl -X POST "http://localhost:8000/process" \
     -F "file=@attendance.pdf"
```

## License

Private/Internal Use Only