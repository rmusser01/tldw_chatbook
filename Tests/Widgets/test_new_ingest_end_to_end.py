# test_new_ingest_end_to_end.py
"""
Simple end-to-end test for the new ingest workflow.
Tests core functionality without complex UI interactions.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock
import tempfile

from tldw_chatbook.Widgets.NewIngest import (
    SmartFileDropZone, ProcessingDashboard, get_processing_service,
    VideoConfig, ProcessingState, ProcessingJob
)


@pytest.mark.asyncio
async def test_file_selection_workflow():
    """Test basic file selection and metadata workflow."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        test_video = Path(temp_dir) / "test_video.mp4"
        test_audio = Path(temp_dir) / "test_audio.mp3"
        
        test_video.write_bytes(b"fake video content" * 100)
        test_audio.write_bytes(b"fake audio content" * 50)
        
        # Test file selection
        drop_zone = SmartFileDropZone()
        
        # Add files
        drop_zone.files = [test_video, test_audio]
        
        # Verify file tracking
        assert len(drop_zone.files) == 2
        assert drop_zone.file_count == 2
        assert drop_zone.total_size_mb > 0
        
        # Test file type detection
        video_files = [f for f in drop_zone.files if f.suffix == '.mp4']
        audio_files = [f for f in drop_zone.files if f.suffix == '.mp3']
        
        assert len(video_files) == 1
        assert len(audio_files) == 1
        
        print("✅ File selection workflow test passed")


@pytest.mark.asyncio
async def test_configuration_validation():
    """Test media configuration validation."""
    
    # Test valid video config
    video_config = VideoConfig(
        files=[Path("test.mp4")],
        extract_audio_only=True,
        transcription_provider="whisper",
        title="Test Video",
        author="Test Author",
        keywords="test,video"
    )
    
    assert video_config.files == [Path("test.mp4")]
    assert video_config.extract_audio_only == True
    assert video_config.transcription_provider == "whisper"
    assert video_config.title == "Test Video"
    
    # Test config serialization/deserialization
    config_dict = video_config.model_dump()
    new_config = VideoConfig(**config_dict)
    
    assert new_config.files == video_config.files
    assert new_config.extract_audio_only == video_config.extract_audio_only
    
    print("✅ Configuration validation test passed")


@pytest.mark.asyncio
async def test_processing_job_lifecycle():
    """Test processing job state management."""
    
    # Create test job
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    job = ProcessingJob("job-123", "Test Processing", test_files)
    
    # Test initial state
    assert job.job_id == "job-123"
    assert job.title == "Test Processing"
    assert job.files == test_files
    assert job.state == ProcessingState.QUEUED
    assert job.progress == 0.0
    
    # Test state transitions
    job.start()
    assert job.state == ProcessingState.PROCESSING
    assert job.start_time is not None
    
    # Test file progress tracking
    job.update_file_progress("test1.mp4", 0.5, "processing")
    assert job.file_progress["test1.mp4"] == 0.5
    assert job.file_statuses["test1.mp4"] == "processing"
    assert job.progress == 0.25  # 0.5 / 2 files
    
    # Complete first file
    job.update_file_progress("test1.mp4", 1.0, "completed")
    job.update_file_progress("test2.mp4", 1.0, "completed")
    assert job.progress == 1.0
    
    # Complete job
    job.complete()
    assert job.state == ProcessingState.COMPLETED
    assert job.end_time is not None
    
    print("✅ Processing job lifecycle test passed")


@pytest.mark.asyncio
async def test_processing_dashboard_management():
    """Test processing dashboard job management."""
    
    # Create dashboard
    dashboard = ProcessingDashboard()
    
    # Add test jobs
    job1 = dashboard.add_job("job-1", "Video Job", [Path("test1.mp4")])
    job2 = dashboard.add_job("job-2", "Audio Job", [Path("test2.mp3")])
    
    assert len(dashboard.active_jobs) == 2
    assert "job-1" in dashboard.active_jobs
    assert "job-2" in dashboard.active_jobs
    
    # Test job status updates
    dashboard.update_job_status("job-1", ProcessingState.PROCESSING, 0.5, "Processing...")
    assert job1.state == ProcessingState.PROCESSING
    assert job1.progress == 0.5
    
    # Test job statistics
    dashboard.update_job_status("job-2", ProcessingState.COMPLETED, 1.0, "Done")
    assert dashboard.get_completed_job_count() == 1
    assert dashboard.get_active_job_count() == 1  # job-1 still processing
    
    # Test job removal
    dashboard.remove_job("job-2")
    assert len(dashboard.active_jobs) == 1
    assert "job-2" not in dashboard.active_jobs
    
    print("✅ Processing dashboard management test passed")


@pytest.mark.asyncio
async def test_backend_service_integration():
    """Test backend service integration without UI."""
    
    # Mock app instance
    mock_app = Mock()
    
    # Test service creation
    service = get_processing_service(mock_app)
    assert service is not None
    assert service.app_instance == mock_app
    
    # Test job submission
    config = VideoConfig(
        files=[Path("test.mp4")],
        title="Test Video Processing"
    )
    
    job_id = service.submit_job(config, "Custom Job Title")
    assert job_id.startswith("job-")
    
    # Test job status
    job_status = service.get_job_status(job_id)
    assert job_status is not None
    assert job_status.title == "Custom Job Title"
    
    # Test active jobs tracking
    active_jobs = service.get_active_jobs()
    assert len(active_jobs) == 1
    assert job_id in active_jobs
    
    # Test job cancellation
    success = service.cancel_job(job_id)
    assert success == True
    
    updated_status = service.get_job_status(job_id)
    assert updated_status.state == ProcessingState.CANCELLED
    
    print("✅ Backend service integration test passed")


@pytest.mark.asyncio  
async def test_error_handling():
    """Test error handling scenarios."""
    
    # Test processing job error handling
    job = ProcessingJob("error-job", "Error Test", [Path("test.mp4")])
    
    # Test failure
    job.fail("Processing failed: File not found")
    assert job.state == ProcessingState.FAILED
    assert job.error == "Processing failed: File not found"
    assert "failed" in job.message.lower()
    
    # Test cancellation
    job2 = ProcessingJob("cancel-job", "Cancel Test", [Path("test.mp4")])
    job2.start()
    job2.cancel()
    assert job2.state == ProcessingState.CANCELLED
    
    # Test service error handling
    mock_app = Mock()
    service = get_processing_service(mock_app)
    
    # Test nonexistent job operations
    assert service.get_job_status("nonexistent") is None
    assert service.cancel_job("nonexistent") == False
    
    print("✅ Error handling test passed")


@pytest.mark.asyncio
async def test_complete_workflow_simulation():
    """Test complete workflow from file selection to completion."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Step 1: Create test files
        test_files = []
        for i in range(3):
            test_file = Path(temp_dir) / f"test_video_{i}.mp4"
            test_file.write_bytes(b"fake content" * (i + 1) * 100)
            test_files.append(test_file)
        
        # Step 2: File selection
        drop_zone = SmartFileDropZone()
        drop_zone.files = test_files
        assert len(drop_zone.files) == 3
        
        # Step 3: Configuration
        config = VideoConfig(
            files=test_files,
            extract_audio_only=False,
            transcription_provider="whisper",
            title="Batch Video Processing",
            author="Test User",
            keywords="test,batch,video"
        )
        
        # Step 4: Processing service setup
        mock_app = Mock()
        service = get_processing_service(mock_app)
        
        # Step 5: Job submission
        job_id = service.submit_job(config)
        job_status = service.get_job_status(job_id)
        
        assert "Video Processing" in job_status.title
        assert "(3 items)" in job_status.title
        
        # Step 6: Dashboard tracking
        dashboard = ProcessingDashboard()
        dashboard_job = dashboard.add_job(job_id, job_status.title, test_files)
        
        # Step 7: Simulate processing progress
        dashboard.update_job_status(job_id, ProcessingState.PROCESSING, 0.0, "Starting...")
        
        # Process each file
        for i, test_file in enumerate(test_files):
            progress = (i + 1) / len(test_files)
            dashboard.update_job_file_progress(job_id, str(test_file), 1.0, "completed")
            dashboard.update_job_status(
                job_id, 
                ProcessingState.PROCESSING, 
                progress, 
                f"Processed {i + 1}/{len(test_files)} files"
            )
        
        # Step 8: Complete processing
        dashboard.update_job_status(job_id, ProcessingState.COMPLETED, 1.0, "All files processed")
        
        # Step 9: Verify final state
        assert dashboard_job.state == ProcessingState.COMPLETED
        assert dashboard_job.progress == 1.0
        assert dashboard_job.completed_files == 3
        assert dashboard_job.failed_files == 0
        
        print("✅ Complete workflow simulation test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])