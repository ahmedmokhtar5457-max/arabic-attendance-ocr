# Arabic Attendance Data Extraction API Documentation

## Overview

The Arabic Attendance Data Extraction API is a RESTful service that processes PDF files containing handwritten Arabic attendance forms and extracts structured data. The system operates completely offline and provides comprehensive OCR capabilities optimized for Arabic handwritten text.

## Base URL

```
http://localhost:8000
```

## Authentication

This is an offline system with no authentication required. In production environments, implement appropriate security measures.

## API Endpoints

### 1. Health Check

#### GET /health

Check the health status of the API service.

**Response:**
```json
{
  "service_name": "Arabic Attendance Extraction",
  "version": "1.0.0",
  "status": "healthy",
  "uptime": "2h 30m",
  "pipeline_initialized": true,
  "processing_statistics": {
    "processed_files": 15,
    "processed_images": 45,
    "successful_extractions": 42,
    "errors": []
  }
}
```

### 2. Process PDF

#### POST /process

Submit a PDF file for attendance data extraction.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: PDF file

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/process" \
     -F "file=@attendance_form.pdf"
```

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Processing started",
  "status_url": "/status/550e8400-e29b-41d4-a716-446655440000"
}
```

**Error Responses:**
- 400: Invalid file type (only PDF supported)
- 500: Processing initialization failed

### 3. Job Status

#### GET /status/{job_id}

Get the current status of a processing job.

**Parameters:**
- job_id (path): UUID of the processing job

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "processing",
  "progress": 65.0,
  "message": "Processing image 3 of 5",
  "started_at": "2024-01-15T10:30:00Z",
  "completed_at": null
}
```

**Status Values:**
- `pending`: Job queued for processing
- `processing`: Currently being processed
- `completed`: Processing finished successfully
- `failed`: Processing failed with errors

### 4. Get Results

#### GET /result/{job_id}

Retrieve the extraction results for a completed job.

**Parameters:**
- job_id (path): UUID of the processing job

**Response:**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "pdf_filename": "attendance_form.pdf",
  "processing_time": 45.2,
  "total_images": 3,
  "successful_extractions": 3,
  "extracted_data": [
    {
      "image_id": "page_1",
      "success": true,
      "extracted_data": [
        {
          "row_index": 0,
          "attendance_time": "8",
          "rank": "GUARD",
          "shift_id": "1"
        },
        {
          "row_index": 1,
          "attendance_time": "12",
          "rank": "OFFICER",
          "shift_id": "2"
        }
      ],
      "filled_rows_count": 2,
      "confidence_scores": {
        "average_confidence": 0.85,
        "min_confidence": 0.72,
        "max_confidence": 0.94
      },
      "validation_results": {
        "valid_rows": 2,
        "total_rows": 2
      }
    }
  ],
  "summary": {
    "total_images": 3,
    "successful_images": 3,
    "failed_images": 0,
    "total_extracted_rows": 15,
    "total_filled_rows": 12,
    "average_confidence": 0.82,
    "processing_success_rate": 1.0
  },
  "timestamp": "2024-01-15T10:35:45Z"
}
```

**Error Responses:**
- 404: Job not found
- 400: Job not completed yet

### 5. Download Results

#### GET /result/{job_id}/download

Download the extraction results as a JSON file.

**Parameters:**
- job_id (path): UUID of the processing job

**Response:**
- Content-Type: application/json
- Content-Disposition: attachment; filename="attendance_data_{job_id}.json"

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/result/550e8400-e29b-41d4-a716-446655440000/download" \
     -o attendance_results.json
```

### 6. List Jobs

#### GET /jobs

List all processing jobs and their current status.

**Response:**
```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress": 100.0,
    "message": "Processing completed successfully",
    "started_at": "2024-01-15T10:30:00Z",
    "completed_at": "2024-01-15T10:35:45Z"
  },
  {
    "job_id": "660f9511-f39c-52e5-b827-557766551111",
    "status": "processing",
    "progress": 45.0,
    "message": "Processing image 2 of 4",
    "started_at": "2024-01-15T10:40:00Z",
    "completed_at": null
  }
]
```

### 7. Delete Job

#### DELETE /jobs/{job_id}

Delete a processing job and its results.

**Parameters:**
- job_id (path): UUID of the processing job

**Response:**
```json
{
  "message": "Job 550e8400-e29b-41d4-a716-446655440000 deleted successfully"
}
```

### 8. Cleanup Jobs

#### POST /jobs/cleanup

Clean up completed and failed jobs older than 24 hours.

**Response:**
```json
{
  "message": "Cleaned up 5 old jobs",
  "deleted_jobs": [
    "550e8400-e29b-41d4-a716-446655440000",
    "660f9511-f39c-52e5-b827-557766551111"
  ]
}
```

### 9. Statistics

#### GET /statistics

Get comprehensive processing statistics.

**Response:**
```json
{
  "pipeline_statistics": {
    "processed_files": 25,
    "processed_images": 75,
    "detected_tables": 73,
    "extracted_rows": 450,
    "successful_extractions": 68,
    "errors": []
  },
  "job_statistics": {
    "total_jobs": 25,
    "pending_jobs": 2,
    "processing_jobs": 1,
    "completed_jobs": 20,
    "failed_jobs": 2
  },
  "timestamp": "2024-01-15T11:00:00Z"
}
```

### 10. Reset Statistics

#### POST /reset-statistics

Reset all processing statistics to zero.

**Response:**
```json
{
  "message": "Statistics reset successfully"
}
```

## Data Models

### Extracted Row Data

Each extracted row contains the following fields:

```json
{
  "row_index": 0,
  "attendance_time": "12",
  "rank": "GUARD",
  "shift_id": "1"
}
```

**Field Descriptions:**
- `row_index`: Zero-based index of the row in the table
- `attendance_time`: Extracted attendance time (1-24 range)
- `rank`: Extracted rank/guard point (English text)
- `shift_id`: Extracted shift identifier (numeric)

### Confidence Scores

```json
{
  "row_confidences": [0.85, 0.92, 0.78],
  "average_confidence": 0.85,
  "min_confidence": 0.78,
  "max_confidence": 0.92
}
```

### Validation Results

```json
{
  "row_validations": [
    {
      "is_valid": true,
      "errors": [],
      "warnings": []
    }
  ],
  "valid_rows": 15,
  "total_rows": 16
}
```

## Error Handling

### HTTP Status Codes

- **200 OK**: Request successful
- **400 Bad Request**: Invalid request parameters or file format
- **404 Not Found**: Resource not found (job, endpoint)
- **500 Internal Server Error**: Server processing error
- **503 Service Unavailable**: Service not initialized

### Error Response Format

```json
{
  "detail": "Error description",
  "error_code": "PROCESSING_ERROR",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Rate Limiting

The API does not implement rate limiting by default. In production environments, consider implementing:

- Request rate limiting per IP
- Concurrent job limits per client
- File size limits
- Processing queue limits

## Usage Examples

### Python Client Example

```python
import requests
import time
import json

# Upload PDF for processing
with open('attendance_form.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/process',
        files={'file': f}
    )
    
job_data = response.json()
job_id = job_data['job_id']

# Poll for completion
while True:
    status_response = requests.get(f'http://localhost:8000/status/{job_id}')
    status_data = status_response.json()
    
    print(f"Status: {status_data['status']} - Progress: {status_data['progress']}%")
    
    if status_data['status'] in ['completed', 'failed']:
        break
    
    time.sleep(5)

# Get results
if status_data['status'] == 'completed':
    results_response = requests.get(f'http://localhost:8000/result/{job_id}')
    results = results_response.json()
    
    print(f"Extracted {len(results['extracted_data'])} pages")
    print(f"Total rows: {results['summary']['total_extracted_rows']}")
    
    # Save results
    with open('extraction_results.json', 'w') as f:
        json.dump(results, f, indent=2)
```

### JavaScript Client Example

```javascript
async function processAttendanceForm(file) {
    // Upload file
    const formData = new FormData();
    formData.append('file', file);
    
    const uploadResponse = await fetch('http://localhost:8000/process', {
        method: 'POST',
        body: formData
    });
    
    const jobData = await uploadResponse.json();
    const jobId = jobData.job_id;
    
    // Poll for completion
    while (true) {
        const statusResponse = await fetch(`http://localhost:8000/status/${jobId}`);
        const statusData = await statusResponse.json();
        
        console.log(`Status: ${statusData.status} - Progress: ${statusData.progress}%`);
        
        if (statusData.status === 'completed') {
            // Get results
            const resultsResponse = await fetch(`http://localhost:8000/result/${jobId}`);
            const results = await resultsResponse.json();
            
            console.log('Processing completed:', results);
            return results;
        } else if (statusData.status === 'failed') {
            throw new Error('Processing failed');
        }
        
        await new Promise(resolve => setTimeout(resolve, 5000));
    }
}
```

### cURL Workflow Example

```bash
#!/bin/bash

# Upload PDF
RESPONSE=$(curl -s -X POST "http://localhost:8000/process" -F "file=@attendance.pdf")
JOB_ID=$(echo $RESPONSE | jq -r '.job_id')

echo "Job ID: $JOB_ID"

# Poll for completion
while true; do
    STATUS_RESPONSE=$(curl -s "http://localhost:8000/status/$JOB_ID")
    STATUS=$(echo $STATUS_RESPONSE | jq -r '.status')
    PROGRESS=$(echo $STATUS_RESPONSE | jq -r '.progress')
    
    echo "Status: $STATUS - Progress: $PROGRESS%"
    
    if [ "$STATUS" = "completed" ]; then
        break
    elif [ "$STATUS" = "failed" ]; then
        echo "Processing failed"
        exit 1
    fi
    
    sleep 5
done

# Download results
curl -X GET "http://localhost:8000/result/$JOB_ID/download" -o "results_$JOB_ID.json"
echo "Results saved to results_$JOB_ID.json"
```

## Best Practices

### File Preparation
- Ensure PDFs are high quality (300 DPI minimum)
- Use clean, well-lit scans
- Avoid skewed or rotated images
- Ensure handwriting is clear and legible

### API Usage
- Always check job status before requesting results
- Implement proper error handling
- Clean up completed jobs regularly
- Monitor system resources during processing

### Performance Optimization
- Process files during off-peak hours for better performance
- Consider batch processing for multiple files
- Monitor memory usage for large PDF files
- Use appropriate timeout values for long-running jobs

## Troubleshooting

### Common Issues

1. **File Upload Fails**
   - Check file format (only PDF supported)
   - Verify file size limits
   - Ensure proper multipart/form-data encoding

2. **Processing Takes Too Long**
   - Check system resources (CPU, memory)
   - Verify image quality and complexity
   - Consider reducing PDF resolution

3. **Low Extraction Accuracy**
   - Improve source document quality
   - Ensure proper lighting and contrast
   - Use the optimized template design
   - Check handwriting clarity

4. **Service Unavailable**
   - Verify service is running
   - Check system resources
   - Review error logs
   - Restart service if necessary