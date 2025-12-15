"""
FastAPI Service for Arabic Attendance Data Extraction

Offline API service for processing attendance PDFs and returning structured data.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import logging
import tempfile
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import asyncio
from contextlib import asynccontextmanager

from ..core.extraction_pipeline import AttendanceExtractionPipeline
from ..utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global pipeline instance
pipeline = None
processing_jobs = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global pipeline
    
    # Startup
    logger.info("Initializing Arabic Attendance Extraction Service...")
    try:
        pipeline = AttendanceExtractionPipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")


# Create FastAPI app
app = FastAPI(
    title="Arabic Attendance Data Extraction API",
    description="Offline OCR service for extracting handwritten Arabic attendance data",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ProcessingStatus(BaseModel):
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float
    message: str
    started_at: datetime
    completed_at: Optional[datetime] = None


class ExtractionResult(BaseModel):
    job_id: str
    pdf_filename: str
    processing_time: float
    total_images: int
    successful_extractions: int
    extracted_data: List[Dict[str, Any]]
    summary: Dict[str, Any]
    validation_report: Optional[str] = None
    timestamp: datetime


class SystemStatus(BaseModel):
    service_name: str
    version: str
    status: str
    uptime: str
    pipeline_initialized: bool
    processing_statistics: Dict[str, Any]


# Dependency to get pipeline
def get_pipeline():
    global pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Arabic Attendance Data Extraction API",
        "version": "1.0.0",
        "status": "running",
        "description": "Offline OCR service for extracting handwritten Arabic attendance data"
    }


@app.get("/health", response_model=SystemStatus)
async def health_check(pipeline: AttendanceExtractionPipeline = Depends(get_pipeline)):
    """Health check endpoint."""
    return SystemStatus(
        service_name="Arabic Attendance Extraction",
        version="1.0.0",
        status="healthy",
        uptime="N/A",  # Could implement actual uptime tracking
        pipeline_initialized=pipeline is not None,
        processing_statistics=pipeline.get_processing_statistics()
    )


@app.post("/process", response_model=Dict[str, str])
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    pipeline: AttendanceExtractionPipeline = Depends(get_pipeline)
):
    """
    Process a PDF file for attendance data extraction.
    
    Args:
        file: PDF file to process
        
    Returns:
        Job ID for tracking processing status
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    processing_jobs[job_id] = ProcessingStatus(
        job_id=job_id,
        status="pending",
        progress=0.0,
        message="Job queued for processing",
        started_at=datetime.now()
    )
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        # Add background task
        background_tasks.add_task(
            process_pdf_background,
            job_id,
            temp_file_path,
            file.filename,
            pipeline
        )
        
        logger.info(f"Started processing job {job_id} for file {file.filename}")
        
        return {
            "job_id": job_id,
            "message": "Processing started",
            "status_url": f"/status/{job_id}"
        }
        
    except Exception as e:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
        
        logger.error(f"Error starting processing job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")


async def process_pdf_background(
    job_id: str,
    file_path: str,
    filename: str,
    pipeline: AttendanceExtractionPipeline
):
    """Background task for processing PDF."""
    try:
        # Update status
        processing_jobs[job_id].status = "processing"
        processing_jobs[job_id].message = "Processing PDF file..."
        processing_jobs[job_id].progress = 10.0
        
        logger.info(f"Starting background processing for job {job_id}")
        
        # Process the PDF
        result = pipeline.process_pdf_file(file_path)
        
        # Update progress
        processing_jobs[job_id].progress = 90.0
        processing_jobs[job_id].message = "Finalizing results..."
        
        # Store result
        processing_jobs[job_id].result = ExtractionResult(
            job_id=job_id,
            pdf_filename=filename,
            processing_time=result.get('processing_time', 0.0),
            total_images=result.get('total_images', 0),
            successful_extractions=result.get('successful_extractions', 0),
            extracted_data=result.get('image_results', []),
            summary=result.get('summary', {}),
            timestamp=datetime.now()
        )
        
        # Mark as completed
        processing_jobs[job_id].status = "completed"
        processing_jobs[job_id].progress = 100.0
        processing_jobs[job_id].message = "Processing completed successfully"
        processing_jobs[job_id].completed_at = datetime.now()
        
        logger.info(f"Completed processing job {job_id}")
        
    except Exception as e:
        logger.error(f"Error in background processing job {job_id}: {str(e)}")
        
        # Mark as failed
        processing_jobs[job_id].status = "failed"
        processing_jobs[job_id].message = f"Processing failed: {str(e)}"
        processing_jobs[job_id].completed_at = datetime.now()
        
    finally:
        # Clean up temp file
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
            temp_dir = os.path.dirname(file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp files: {str(cleanup_error)}")


@app.get("/status/{job_id}", response_model=ProcessingStatus)
async def get_job_status(job_id: str):
    """Get processing status for a job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_jobs[job_id]


@app.get("/result/{job_id}", response_model=ExtractionResult)
async def get_job_result(job_id: str):
    """Get processing result for a completed job."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if not hasattr(job, 'result'):
        raise HTTPException(status_code=500, detail="Result not available")
    
    return job.result


@app.get("/result/{job_id}/download")
async def download_job_result(job_id: str):
    """Download processing result as JSON file."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job.status != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed. Current status: {job.status}"
        )
    
    if not hasattr(job, 'result'):
        raise HTTPException(status_code=500, detail="Result not available")
    
    # Create temporary JSON file
    temp_dir = tempfile.mkdtemp()
    json_file_path = os.path.join(temp_dir, f"attendance_data_{job_id}.json")
    
    try:
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(job.result.dict(), f, indent=2, ensure_ascii=False, default=str)
        
        return FileResponse(
            json_file_path,
            media_type='application/json',
            filename=f"attendance_data_{job_id}.json"
        )
        
    except Exception as e:
        logger.error(f"Error creating download file: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to create download file")


@app.get("/jobs", response_model=List[ProcessingStatus])
async def list_jobs():
    """List all processing jobs."""
    return list(processing_jobs.values())


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a processing job and its results."""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del processing_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.post("/jobs/cleanup")
async def cleanup_completed_jobs():
    """Clean up completed and failed jobs older than 24 hours."""
    current_time = datetime.now()
    jobs_to_delete = []
    
    for job_id, job in processing_jobs.items():
        if job.status in ["completed", "failed"] and job.completed_at:
            hours_since_completion = (current_time - job.completed_at).total_seconds() / 3600
            if hours_since_completion > 24:
                jobs_to_delete.append(job_id)
    
    for job_id in jobs_to_delete:
        del processing_jobs[job_id]
    
    return {
        "message": f"Cleaned up {len(jobs_to_delete)} old jobs",
        "deleted_jobs": jobs_to_delete
    }


@app.get("/statistics", response_model=Dict[str, Any])
async def get_statistics(pipeline: AttendanceExtractionPipeline = Depends(get_pipeline)):
    """Get processing statistics."""
    stats = pipeline.get_processing_statistics()
    
    # Add job statistics
    job_stats = {
        "total_jobs": len(processing_jobs),
        "pending_jobs": len([j for j in processing_jobs.values() if j.status == "pending"]),
        "processing_jobs": len([j for j in processing_jobs.values() if j.status == "processing"]),
        "completed_jobs": len([j for j in processing_jobs.values() if j.status == "completed"]),
        "failed_jobs": len([j for j in processing_jobs.values() if j.status == "failed"])
    }
    
    return {
        "pipeline_statistics": stats,
        "job_statistics": job_stats,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/reset-statistics")
async def reset_statistics(pipeline: AttendanceExtractionPipeline = Depends(get_pipeline)):
    """Reset processing statistics."""
    pipeline.reset_statistics()
    return {"message": "Statistics reset successfully"}


# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run the server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disable reload for production
        log_level="info"
    )