# test_processing_dashboard.py
"""
Unit tests for ProcessingDashboard component.
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from textual.app import App

from tldw_chatbook.Widgets.NewIngest.ProcessingDashboard import (
    ProcessingDashboard,
    ProcessingJob,
    JobStatusWidget,
    ProcessingState,
    ProcessingJobStatus,
    ProcessingCancelled,
    ProcessingPaused,
    ProcessingResumed
)


class TestApp(App):
    """Test app for component testing."""
    
    def compose(self):
        yield ProcessingDashboard()


@pytest.mark.asyncio
async def test_processing_job_initialization():
    """Test ProcessingJob initializes correctly."""
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    job = ProcessingJob("job-1", "Test Video Processing", test_files)
    
    assert job.job_id == "job-1"
    assert job.title == "Test Video Processing"
    assert job.files == test_files
    assert job.state == ProcessingState.QUEUED
    assert job.progress == 0.0
    assert job.current_file_index == 0
    assert job.message == "Queued"
    assert job.error is None
    assert job.start_time is None
    assert job.end_time is None


@pytest.mark.asyncio
async def test_processing_job_properties():
    """Test ProcessingJob computed properties."""
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    job = ProcessingJob("job-1", "Test Processing", test_files)
    
    # Test current_file
    assert job.current_file == test_files[0]
    
    job.current_file_index = 1
    assert job.current_file == test_files[1]
    
    job.current_file_index = 10  # Out of range
    assert job.current_file is None
    
    # Test file counts
    assert job.completed_files == 0
    assert job.failed_files == 0
    
    # Mark some files as completed/failed
    job.file_statuses["test1.mp4"] = "completed"
    job.file_statuses["test2.mp4"] = "failed"
    
    assert job.completed_files == 1
    assert job.failed_files == 1


@pytest.mark.asyncio
async def test_processing_job_time_calculations():
    """Test ProcessingJob time calculation properties."""
    job = ProcessingJob("job-1", "Test", [Path("test.mp4")])
    
    # No times set initially
    assert job.elapsed_time is None
    assert job.estimated_remaining is None
    
    # Set start time
    start_time = datetime.now()
    job.start_time = start_time
    
    # Should have elapsed time now
    assert job.elapsed_time is not None
    assert job.elapsed_time.total_seconds() >= 0
    
    # Set progress and check estimation
    job.progress = 0.5  # 50% done
    estimated = job.estimated_remaining
    assert estimated is not None
    # Should estimate roughly the same time remaining as elapsed
    assert abs(estimated.total_seconds() - job.elapsed_time.total_seconds()) < 5.0


@pytest.mark.asyncio
async def test_processing_job_state_transitions():
    """Test ProcessingJob state transition methods."""
    job = ProcessingJob("job-1", "Test", [Path("test.mp4")])
    
    # Test start
    job.start()
    assert job.state == ProcessingState.PROCESSING
    assert job.start_time is not None
    assert job.message == "Processing started"
    
    # Test pause/resume
    job.pause()
    assert job.state == ProcessingState.PAUSED
    assert job.message == "Paused"
    
    job.resume()
    assert job.state == ProcessingState.PROCESSING
    assert job.message == "Resumed processing"
    
    # Test completion
    job.complete()
    assert job.state == ProcessingState.COMPLETED
    assert job.end_time is not None
    assert job.progress == 1.0
    assert job.message == "Completed successfully"


@pytest.mark.asyncio
async def test_processing_job_failure():
    """Test ProcessingJob failure handling."""
    job = ProcessingJob("job-1", "Test", [Path("test.mp4")])
    
    job.fail("Test error")
    assert job.state == ProcessingState.FAILED
    assert job.end_time is not None
    assert job.error == "Test error"
    assert job.message == "Failed: Test error"


@pytest.mark.asyncio
async def test_processing_job_cancellation():
    """Test ProcessingJob cancellation."""
    job = ProcessingJob("job-1", "Test", [Path("test.mp4")])
    
    job.cancel()
    assert job.state == ProcessingState.CANCELLED
    assert job.end_time is not None
    assert job.message == "Cancelled by user"


@pytest.mark.asyncio
async def test_processing_job_file_progress():
    """Test ProcessingJob file-level progress tracking."""
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    job = ProcessingJob("job-1", "Test", test_files)
    
    # Update progress for first file
    job.update_file_progress("test1.mp4", 0.5, "processing")
    
    assert job.file_progress["test1.mp4"] == 0.5
    assert job.file_statuses["test1.mp4"] == "processing"
    assert job.progress == 0.25  # 0.5 / 2 files
    assert "test1.mp4" in job.message
    
    # Complete first file, start second
    job.update_file_progress("test1.mp4", 1.0, "completed")
    job.update_file_progress("test2.mp4", 0.3, "processing")
    
    assert job.progress == 0.65  # (1.0 + 0.3) / 2


@pytest.mark.asyncio
async def test_processing_job_current_file_update():
    """Test ProcessingJob current file tracking."""
    test_files = [Path("test1.mp4"), Path("test2.mp4")]
    job = ProcessingJob("job-1", "Test", test_files)
    
    job.update_current_file(1)
    assert job.current_file_index == 1
    assert job.current_file == test_files[1]
    assert "test2.mp4" in job.message


@pytest.mark.asyncio
async def test_job_status_widget_initialization():
    """Test JobStatusWidget initializes correctly."""
    test_files = [Path("test.mp4")]
    job = ProcessingJob("job-1", "Test Job", test_files)
    widget = JobStatusWidget(job)
    
    assert widget.job == job


@pytest.mark.asyncio
async def test_job_status_widget_compose():
    """Test JobStatusWidget composes correctly."""
    app = TestApp()
    test_files = [Path("test.mp4")]
    job = ProcessingJob("job-1", "Test Job", test_files)
    
    async with app.run_test() as pilot:
        widget = JobStatusWidget(job)
        await app.mount(widget)
        await pilot.pause()
        
        # Check components exist
        assert widget.query(".job-status-widget")
        assert widget.query(".job-header")
        assert widget.query(".job-title")
        assert widget.query(f"#status-{job.job_id}")
        assert widget.query(".job-controls")
        assert widget.query(".job-progress")
        assert widget.query(f"#progress-{job.job_id}")


@pytest.mark.asyncio
async def test_job_status_widget_status_display():
    """Test JobStatusWidget status display formatting."""
    test_files = [Path("test.mp4")]
    job = ProcessingJob("job-1", "Test", test_files)
    widget = JobStatusWidget(job)
    
    # Test different states
    job.state = ProcessingState.QUEUED
    status = widget._get_status_display()
    assert "⏳" in status and "Queued" in status
    
    job.state = ProcessingState.PROCESSING
    job.progress = 0.75
    status = widget._get_status_display()
    assert "⚙️" in status and "75%" in status
    
    job.state = ProcessingState.COMPLETED
    job.file_statuses = {"test.mp4": "completed"}
    status = widget._get_status_display()
    assert "✅" in status and "Done" in status and "1/1" in status
    
    job.state = ProcessingState.FAILED
    job.file_statuses = {"test.mp4": "failed"}
    status = widget._get_status_display()
    assert "❌" in status and "Failed" in status


@pytest.mark.asyncio
async def test_job_status_widget_time_display():
    """Test JobStatusWidget time display formatting."""
    test_files = [Path("test.mp4")]
    job = ProcessingJob("job-1", "Test", test_files)
    widget = JobStatusWidget(job)
    
    # No start time
    assert widget._get_time_display() == ""
    
    # Set start time
    job.start_time = datetime.now() - timedelta(seconds=65)  # 1m 5s ago
    time_display = widget._get_time_display()
    assert "1m" in time_display and "5s" in time_display
    
    # With progress and estimation
    job.state = ProcessingState.PROCESSING
    job.progress = 0.5  # 50% done
    time_display = widget._get_time_display()
    # Should include estimated remaining time
    assert "left" in time_display or "≈" in time_display


@pytest.mark.asyncio
async def test_job_status_widget_control_buttons():
    """Test JobStatusWidget control button messages."""
    app = TestApp()
    test_files = [Path("test.mp4")]
    job = ProcessingJob("job-1", "Test", test_files)
    
    async with app.run_test() as pilot:
        widget = JobStatusWidget(job)
        await app.mount(widget)
        await pilot.pause()
        
        # Track messages
        messages = []
        original_post = widget.post_message
        widget.post_message = lambda msg: messages.append(msg)
        
        # Click pause button
        pause_btn = widget.query_one(f"#pause-{job.job_id}")
        pause_btn.press()
        await pilot.pause()
        
        # Should post ProcessingPaused message
        assert len(messages) == 1
        assert isinstance(messages[0], ProcessingPaused)
        assert messages[0].job_id == job.job_id


@pytest.mark.asyncio
async def test_processing_dashboard_initialization():
    """Test ProcessingDashboard initializes correctly."""
    dashboard = ProcessingDashboard()
    
    assert dashboard.active_jobs == {}
    assert dashboard.total_progress == 0.0
    assert dashboard.is_processing == False


@pytest.mark.asyncio
async def test_processing_dashboard_compose():
    """Test ProcessingDashboard composes correctly."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Check main components exist
        assert dashboard.query(".dashboard-header")
        assert dashboard.query(".dashboard-title")
        assert dashboard.query(".overall-status")
        assert dashboard.query("#overall-message")
        assert dashboard.query("#overall-progress")
        assert dashboard.query(".jobs-title")
        assert dashboard.query("#jobs-container")
        assert dashboard.query("#empty-state")
        assert dashboard.query(".dashboard-controls")


@pytest.mark.asyncio
async def test_processing_dashboard_add_job():
    """Test ProcessingDashboard can add jobs."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add a job
        test_files = [Path("test.mp4")]
        job = dashboard.add_job("job-1", "Test Job", test_files)
        await pilot.pause()
        
        # Check job was added
        assert "job-1" in dashboard.active_jobs
        assert dashboard.active_jobs["job-1"] == job
        assert job.job_id == "job-1"
        assert job.title == "Test Job"
        assert job.files == test_files
        
        # Check widget was created
        assert "job-1" in dashboard._job_widgets
        
        # Check empty state is hidden
        empty_state = dashboard.query_one("#empty-state")
        assert "hidden" in empty_state.classes


@pytest.mark.asyncio
async def test_processing_dashboard_update_job_status():
    """Test ProcessingDashboard can update job status."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add and update job
        job = dashboard.add_job("job-1", "Test", [Path("test.mp4")])
        await pilot.pause()
        
        # Track messages
        messages = []
        original_post = dashboard.post_message
        dashboard.post_message = lambda msg: messages.append(msg)
        
        # Update status
        dashboard.update_job_status("job-1", ProcessingState.PROCESSING, 0.5, "Processing file")
        await pilot.pause()
        
        # Check job was updated
        assert job.state == ProcessingState.PROCESSING
        assert job.progress == 0.5
        assert job.message == "Processing file"
        assert job.start_time is not None
        
        # Check status message posted
        assert len(messages) == 1
        assert isinstance(messages[0], ProcessingJobStatus)
        assert messages[0].job_id == "job-1"
        assert messages[0].status == ProcessingState.PROCESSING


@pytest.mark.asyncio
async def test_processing_dashboard_job_statistics():
    """Test ProcessingDashboard job statistics methods."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add jobs with different states
        job1 = dashboard.add_job("job-1", "Test 1", [Path("test1.mp4")])
        job2 = dashboard.add_job("job-2", "Test 2", [Path("test2.mp4")])
        job3 = dashboard.add_job("job-3", "Test 3", [Path("test3.mp4")])
        await pilot.pause()
        
        # Set different states
        job1.state = ProcessingState.PROCESSING
        job2.state = ProcessingState.COMPLETED
        job3.state = ProcessingState.FAILED
        
        # Check statistics
        assert dashboard.get_active_job_count() == 1
        assert dashboard.get_completed_job_count() == 1
        assert dashboard.get_failed_job_count() == 1


@pytest.mark.asyncio
async def test_processing_dashboard_remove_job():
    """Test ProcessingDashboard can remove jobs."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add job
        job = dashboard.add_job("job-1", "Test", [Path("test.mp4")])
        await pilot.pause()
        
        # Remove job
        dashboard.remove_job("job-1")
        await pilot.pause()
        
        # Check job was removed
        assert "job-1" not in dashboard.active_jobs
        assert "job-1" not in dashboard._job_widgets
        
        # Check empty state is shown
        empty_state = dashboard.query_one("#empty-state")
        assert "hidden" not in empty_state.classes


@pytest.mark.asyncio
async def test_processing_dashboard_file_progress():
    """Test ProcessingDashboard file-level progress updates."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add job with multiple files
        test_files = [Path("test1.mp4"), Path("test2.mp4")]
        job = dashboard.add_job("job-1", "Test", test_files)
        await pilot.pause()
        
        # Update file progress
        dashboard.update_job_file_progress("job-1", "test1.mp4", 0.8, "processing")
        await pilot.pause()
        
        # Check file progress was updated
        assert job.file_progress["test1.mp4"] == 0.8
        assert job.file_statuses["test1.mp4"] == "processing"
        assert job.progress == 0.4  # 0.8 / 2 files


@pytest.mark.asyncio
async def test_processing_dashboard_control_buttons():
    """Test ProcessingDashboard control buttons."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add jobs
        job1 = dashboard.add_job("job-1", "Test 1", [Path("test1.mp4")])
        job2 = dashboard.add_job("job-2", "Test 2", [Path("test2.mp4")])
        await pilot.pause()
        
        # Set states
        job1.state = ProcessingState.PROCESSING
        job2.state = ProcessingState.PROCESSING
        
        # Click pause all
        await pilot.click("#pause-all")
        await pilot.pause()
        
        # Should pause both jobs
        assert job1.state == ProcessingState.PAUSED
        assert job2.state == ProcessingState.PAUSED
        
        # Click resume all
        await pilot.click("#resume-all")
        await pilot.pause()
        
        # Should resume both jobs
        assert job1.state == ProcessingState.PROCESSING
        assert job2.state == ProcessingState.PROCESSING


@pytest.mark.asyncio
async def test_processing_dashboard_clear_completed():
    """Test ProcessingDashboard clear completed functionality."""
    app = TestApp()
    async with app.run_test() as pilot:
        dashboard = app.query_one(ProcessingDashboard)
        
        # Add jobs with different states
        job1 = dashboard.add_job("job-1", "Test 1", [Path("test1.mp4")])
        job2 = dashboard.add_job("job-2", "Test 2", [Path("test2.mp4")])
        job3 = dashboard.add_job("job-3", "Test 3", [Path("test3.mp4")])
        await pilot.pause()
        
        # Set states
        job1.state = ProcessingState.COMPLETED
        job2.state = ProcessingState.PROCESSING
        job3.state = ProcessingState.FAILED
        
        # Clear completed
        await pilot.click("#clear-completed")
        await pilot.pause()
        
        # Should remove completed and failed, keep processing
        assert "job-1" not in dashboard.active_jobs  # Completed - removed
        assert "job-2" in dashboard.active_jobs      # Processing - kept
        assert "job-3" not in dashboard.active_jobs  # Failed - removed


@pytest.mark.asyncio
async def test_processing_messages():
    """Test processing message creation."""
    # Test ProcessingJobStatus
    status_msg = ProcessingJobStatus("job-1", ProcessingState.PROCESSING, 0.5, "Processing...")
    assert status_msg.job_id == "job-1"
    assert status_msg.status == ProcessingState.PROCESSING
    assert status_msg.progress == 0.5
    assert status_msg.message == "Processing..."
    
    # Test ProcessingCancelled
    cancel_msg = ProcessingCancelled("job-1")
    assert cancel_msg.job_id == "job-1"
    
    # Test ProcessingPaused
    pause_msg = ProcessingPaused("job-1")
    assert pause_msg.job_id == "job-1"
    
    # Test ProcessingResumed
    resume_msg = ProcessingResumed("job-1")
    assert resume_msg.job_id == "job-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])