# test_backend_integration.py
"""
Unit tests for BackendIntegration component.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from textual.app import App
from textual.widgets import Static

from tldw_chatbook.Widgets.NewIngest.BackendIntegration import (
    MediaProcessingService,
    ProcessingJobResult,
    get_processing_service
)
from tldw_chatbook.Widgets.NewIngest.UnifiedProcessor import (
    VideoConfig, AudioConfig, DocumentConfig, MediaConfig, WebConfig
)
from tldw_chatbook.Widgets.NewIngest.ProcessingDashboard import ProcessingState


class TestApp(App):
    """Test app for component testing."""
    
    def compose(self):
        yield Static("Test")


@pytest.mark.asyncio
async def test_media_processing_service_initialization():
    """Test MediaProcessingService initializes correctly."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    assert service.app_instance == mock_app
    assert service._active_jobs == {}
    assert service._job_workers == {}


@pytest.mark.asyncio
async def test_media_processing_service_submit_job():
    """Test MediaProcessingService can submit jobs."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    # Create test config
    config = VideoConfig(files=[Path("test.mp4")])
    
    # Mock the worker start method
    service._start_processing_worker = Mock(return_value=Mock())
    
    # Submit job
    job_id = service.submit_job(config, "Test Video Job")
    
    # Check job was created
    assert job_id.startswith("job-")
    assert job_id in service._active_jobs
    
    job = service._active_jobs[job_id]
    assert job.title == "Test Video Job"
    assert job.files == [Path("test.mp4")]
    assert job.state == ProcessingState.QUEUED


@pytest.mark.asyncio
async def test_media_processing_service_auto_title():
    """Test MediaProcessingService auto-generates job titles."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    service._start_processing_worker = Mock(return_value=Mock())
    
    # Submit job without title
    config = VideoConfig(files=[Path("test1.mp4"), Path("test2.mp4")])
    job_id = service.submit_job(config)
    
    job = service._active_jobs[job_id]
    assert "Video Processing" in job.title
    assert "(2 items)" in job.title


@pytest.mark.asyncio
async def test_media_processing_service_cancel_job():
    """Test MediaProcessingService can cancel jobs."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    # Mock worker
    mock_worker = Mock()
    service._start_processing_worker = Mock(return_value=mock_worker)
    
    # Submit and cancel job
    config = VideoConfig(files=[Path("test.mp4")])
    job_id = service.submit_job(config)
    
    # Store mock worker
    service._job_workers[job_id] = mock_worker
    
    # Cancel job
    result = service.cancel_job(job_id)
    
    assert result == True
    mock_worker.cancel.assert_called_once()
    
    job = service._active_jobs[job_id]
    assert job.state == ProcessingState.CANCELLED


@pytest.mark.asyncio
async def test_media_processing_service_cancel_nonexistent_job():
    """Test MediaProcessingService handles cancelling nonexistent jobs."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    result = service.cancel_job("nonexistent-job")
    assert result == False


@pytest.mark.asyncio
async def test_media_processing_service_get_job_status():
    """Test MediaProcessingService can get job status."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    service._start_processing_worker = Mock(return_value=Mock())
    
    # Submit job
    config = VideoConfig(files=[Path("test.mp4")])
    job_id = service.submit_job(config)
    
    # Get status
    status = service.get_job_status(job_id)
    assert status is not None
    assert status.job_id == job_id
    
    # Get nonexistent job
    nonexistent_status = service.get_job_status("nonexistent")
    assert nonexistent_status is None


@pytest.mark.asyncio
async def test_media_processing_service_get_active_jobs():
    """Test MediaProcessingService can get all active jobs."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    service._start_processing_worker = Mock(return_value=Mock())
    
    # Submit multiple jobs
    config1 = VideoConfig(files=[Path("test1.mp4")])
    config2 = AudioConfig(files=[Path("test2.mp3")])
    
    job_id1 = service.submit_job(config1)
    job_id2 = service.submit_job(config2)
    
    # Get active jobs
    active_jobs = service.get_active_jobs()
    assert len(active_jobs) == 2
    assert job_id1 in active_jobs
    assert job_id2 in active_jobs


@pytest.mark.asyncio
async def test_media_processing_service_cleanup_completed():
    """Test MediaProcessingService can cleanup completed jobs."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    service._start_processing_worker = Mock(return_value=Mock())
    
    # Submit jobs and set different states
    config = VideoConfig(files=[Path("test.mp4")])
    
    job_id1 = service.submit_job(config)
    job_id2 = service.submit_job(config) 
    job_id3 = service.submit_job(config)
    
    # Set states
    service._active_jobs[job_id1].state = ProcessingState.COMPLETED
    service._active_jobs[job_id2].state = ProcessingState.PROCESSING
    service._active_jobs[job_id3].state = ProcessingState.FAILED
    
    # Cleanup
    service.cleanup_completed_jobs()
    
    # Check results
    active_jobs = service.get_active_jobs()
    assert len(active_jobs) == 1  # Only processing job should remain
    assert job_id2 in active_jobs
    assert job_id1 not in active_jobs
    assert job_id3 not in active_jobs


@pytest.mark.asyncio
async def test_media_processing_service_detect_config_type():
    """Test MediaProcessingService can detect config types."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    # Test different config types
    assert service._detect_config_type(VideoConfig()) == "video"
    assert service._detect_config_type(AudioConfig()) == "audio"
    assert service._detect_config_type(DocumentConfig()) == "document"
    assert service._detect_config_type(MediaConfig()) == "media"


@pytest.mark.asyncio
async def test_video_processor_simulation():
    """Test video processor with simulation fallback."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = VideoConfig(
        files=[Path("test.mp4")],
        extract_audio_only=True,
        transcription_provider="whisper"
    )
    
    # Test with import error (simulation mode)
    with patch('builtins.__import__', side_effect=ImportError):
        result = await service._call_video_processor(Path("test.mp4"), config)
        
        assert result["status"] == "simulated"
        assert result["file_path"] == "test.mp4"
        assert "processed_at" in result


@pytest.mark.asyncio
async def test_audio_processor_simulation():
    """Test audio processor with simulation fallback."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = AudioConfig(
        files=[Path("test.mp3")],
        transcription_provider="whisper",
        speaker_diarization=True
    )
    
    # Test simulation mode
    result = await service._call_audio_processor(Path("test.mp3"), config)
    
    assert result["status"] == "simulated"
    assert result["file_path"] == "test.mp3"


@pytest.mark.asyncio
async def test_document_processor_simulation():
    """Test document processor with simulation fallback."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = DocumentConfig(
        files=[Path("test.docx")],
        ocr_enabled=True,
        preserve_formatting=True
    )
    
    # Test simulation mode
    result = await service._call_document_processor(Path("test.docx"), config)
    
    assert result["status"] == "simulated"
    assert result["file_path"] == "test.docx"


@pytest.mark.asyncio
async def test_generic_processor():
    """Test generic media processor."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = MediaConfig(files=[Path("test.file")])
    
    # Create fake job
    from tldw_chatbook.Widgets.NewIngest.ProcessingDashboard import ProcessingJob
    job = ProcessingJob("test-job", "Test", [Path("test.file")])
    service._active_jobs["test-job"] = job
    
    # Process generic file
    await service._process_generic("test-job", config)
    
    # Check results
    assert job.results["test.file"]["status"] == "processed"
    assert job.file_statuses["test.file"] == "completed"


@pytest.mark.asyncio
async def test_processing_job_result_message():
    """Test ProcessingJobResult message creation."""
    # Test success result
    success_msg = ProcessingJobResult("job-1", True, {"processed": 1})
    assert success_msg.job_id == "job-1"
    assert success_msg.success == True
    assert success_msg.results == {"processed": 1}
    assert success_msg.error is None
    
    # Test failure result
    failure_msg = ProcessingJobResult("job-2", False, {}, "Test error")
    assert failure_msg.job_id == "job-2"
    assert failure_msg.success == False
    assert failure_msg.results == {}
    assert failure_msg.error == "Test error"


@pytest.mark.asyncio
async def test_get_processing_service_singleton():
    """Test get_processing_service singleton pattern."""
    # Reset singleton
    import tldw_chatbook.Widgets.NewIngest.BackendIntegration as backend_module
    backend_module._processing_service = None
    
    mock_app = Mock()
    
    # First call creates service
    service1 = get_processing_service(mock_app)
    assert service1 is not None
    assert service1.app_instance == mock_app
    
    # Second call returns same instance
    service2 = get_processing_service()
    assert service2 is service1
    
    # Test error without app instance on first call
    backend_module._processing_service = None
    with pytest.raises(ValueError):
        get_processing_service()


@pytest.mark.asyncio
async def test_video_processing_with_urls():
    """Test video processing with URLs.""" 
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = VideoConfig(
        urls=["https://youtube.com/watch?v=test"],
        files=[]
    )
    
    # Create fake job
    from tldw_chatbook.Widgets.NewIngest.ProcessingDashboard import ProcessingJob
    job = ProcessingJob("test-job", "Test", [Path("https://youtube.com/watch?v=test")])
    service._active_jobs["test-job"] = job
    
    # Process video
    await service._process_video("test-job", config)
    
    # Check URL was processed
    url = "https://youtube.com/watch?v=test"
    assert url in job.results
    assert job.file_statuses[url] == "completed"


@pytest.mark.asyncio  
async def test_processing_job_cancellation_during_processing():
    """Test job cancellation stops processing."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = VideoConfig(files=[Path("test1.mp4"), Path("test2.mp4")])
    
    # Create job
    from tldw_chatbook.Widgets.NewIngest.ProcessingDashboard import ProcessingJob
    job = ProcessingJob("test-job", "Test", config.files)
    service._active_jobs["test-job"] = job
    
    # Start processing then cancel
    job.start()
    job.cancel()  # Cancel before processing second file
    
    # Process should respect cancellation
    await service._process_video("test-job", config)
    
    # Should not have processed all files due to cancellation
    assert job.state == ProcessingState.CANCELLED


@pytest.mark.asyncio
async def test_web_processing():
    """Test web content processing."""
    mock_app = Mock()
    service = MediaProcessingService(mock_app)
    
    config = WebConfig(
        urls=["https://example.com/article"],
        extract_links=True,
        clean_html=True
    )
    
    # Create fake job
    from tldw_chatbook.Widgets.NewIngest.ProcessingDashboard import ProcessingJob  
    job = ProcessingJob("test-job", "Test", [Path("https://example.com/article")])
    service._active_jobs["test-job"] = job
    
    # Process web content
    await service._process_web("test-job", config)
    
    # Check results
    url = "https://example.com/article"
    assert url in job.results
    assert job.results[url]["status"] == "simulated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])