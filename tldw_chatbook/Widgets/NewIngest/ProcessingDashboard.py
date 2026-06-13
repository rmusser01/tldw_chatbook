"""Compatibility processing dashboard for the legacy NewIngest test surface."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, ProgressBar, Static


class ImmediateButton(Button):
    """Legacy test-compatible button without Textual's active animation wait."""

    def press(self) -> "ImmediateButton":
        if self.disabled or not self.display:
            return self
        if self.action is None:
            self.post_message(Button.Pressed(self))
        else:
            self.call_later(self.app.run_action, self.action, default_namespace=self._parent)
        return self


class CaptureSafePostMixin:
    """Allow legacy tests to capture messages without replacing Textual internals."""

    _post_message_capture: Any | None = None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "post_message" and callable(value):
            object.__setattr__(self, "_post_message_capture", value)
            return
        super().__setattr__(name, value)

    def _emit_message(self, message: Message) -> None:
        capture = getattr(self, "_post_message_capture", None)
        if capture is not None:
            capture(message)
        super().post_message(message)  # type: ignore[misc]


class ProcessingState(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingJob:
    job_id: str
    title: str
    files: list[Path]
    state: ProcessingState = ProcessingState.QUEUED
    progress: float = 0.0
    current_file_index: int = 0
    message: str = "Queued"
    error: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    file_progress: dict[str, float] = field(default_factory=dict)
    file_statuses: dict[str, str] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)

    @property
    def current_file(self) -> Path | None:
        if 0 <= self.current_file_index < len(self.files):
            return self.files[self.current_file_index]
        return None

    @property
    def completed_files(self) -> int:
        return sum(1 for status in self.file_statuses.values() if status == "completed")

    @property
    def failed_files(self) -> int:
        return sum(1 for status in self.file_statuses.values() if status == "failed")

    @property
    def elapsed_time(self) -> timedelta | None:
        if self.start_time is None:
            return None
        return (self.end_time or datetime.now()) - self.start_time

    @property
    def estimated_remaining(self) -> timedelta | None:
        elapsed = self.elapsed_time
        if elapsed is None or self.progress <= 0 or self.progress >= 1:
            return None
        return timedelta(seconds=elapsed.total_seconds() * ((1 - self.progress) / self.progress))

    def start(self) -> None:
        self.state = ProcessingState.PROCESSING
        self.start_time = self.start_time or datetime.now()
        self.message = "Processing started"

    def pause(self) -> None:
        self.state = ProcessingState.PAUSED
        self.message = "Paused"

    def resume(self) -> None:
        self.state = ProcessingState.PROCESSING
        self.message = "Resumed processing"

    def complete(self) -> None:
        self.state = ProcessingState.COMPLETED
        self.end_time = datetime.now()
        self.progress = 1.0
        self.message = "Completed successfully"

    def fail(self, error: str) -> None:
        self.state = ProcessingState.FAILED
        self.end_time = datetime.now()
        self.error = error
        self.message = f"Failed: {error}"

    def cancel(self) -> None:
        self.state = ProcessingState.CANCELLED
        self.end_time = datetime.now()
        self.message = "Cancelled by user"

    def update_file_progress(self, file_key: str, progress: float, status: str) -> None:
        self.file_progress[file_key] = max(0.0, min(1.0, progress))
        self.file_statuses[file_key] = status
        total = len(self.files) or 1
        self.progress = sum(self.file_progress.values()) / total
        self.message = f"{status}: {file_key}"

    def update_current_file(self, index: int) -> None:
        self.current_file_index = index
        if self.current_file is not None:
            self.message = f"Processing {self.current_file}"


class ProcessingJobStatus(Message):
    def __init__(self, job_id: str, status: ProcessingState, progress: float, message: str) -> None:
        super().__init__()
        self.job_id = job_id
        self.status = status
        self.progress = progress
        self.message = message


class ProcessingCancelled(Message):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id


class ProcessingPaused(Message):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id


class ProcessingResumed(Message):
    def __init__(self, job_id: str) -> None:
        super().__init__()
        self.job_id = job_id


class JobStatusWidget(CaptureSafePostMixin, Widget):
    def __init__(self, job: ProcessingJob, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.job = job

    def compose(self) -> ComposeResult:
        with Container(classes="job-status-widget"):
            with Horizontal(classes="job-header"):
                yield Static(self.job.title, classes="job-title")
                yield Static(self._get_status_display(), id=f"status-{self.job.job_id}")
            with Horizontal(classes="job-controls"):
                yield ImmediateButton("Pause", id=f"pause-{self.job.job_id}")
                yield ImmediateButton("Resume", id=f"resume-{self.job.job_id}")
                yield ImmediateButton("Cancel", id=f"cancel-{self.job.job_id}")
            yield ProgressBar(total=100, id=f"progress-{self.job.job_id}", classes="job-progress")

    def _get_status_display(self) -> str:
        if self.job.state == ProcessingState.QUEUED:
            return "⏳ Queued"
        if self.job.state == ProcessingState.PROCESSING:
            return f"⚙️ {int(self.job.progress * 100)}%"
        if self.job.state == ProcessingState.COMPLETED:
            return f"✅ Done {self.job.completed_files}/{len(self.job.files)}"
        if self.job.state == ProcessingState.FAILED:
            return f"❌ Failed {self.job.failed_files}/{len(self.job.files)}"
        if self.job.state == ProcessingState.CANCELLED:
            return "🚫 Cancelled"
        return "⏸ Paused"

    def _get_time_display(self) -> str:
        elapsed = self.job.elapsed_time
        if elapsed is None:
            return ""
        seconds = int(elapsed.total_seconds())
        text = f"{seconds // 60}m {seconds % 60}s"
        remaining = self.job.estimated_remaining
        if remaining is not None:
            text = f"{text} left ≈ {int(remaining.total_seconds())}s"
        return text

    @on(Button.Pressed)
    def _handle_button(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.startswith("pause-"):
            self._emit_message(ProcessingPaused(self.job.job_id))
        elif button_id.startswith("resume-"):
            self._emit_message(ProcessingResumed(self.job.job_id))
        elif button_id.startswith("cancel-"):
            self._emit_message(ProcessingCancelled(self.job.job_id))


class ProcessingDashboard(CaptureSafePostMixin, Widget):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.active_jobs: dict[str, ProcessingJob] = {}
        self._job_widgets: dict[str, JobStatusWidget] = {}
        self.total_progress = 0.0
        self.is_processing = False

    def compose(self) -> ComposeResult:
        with Vertical():
            with Horizontal(classes="dashboard-header"):
                yield Static("Processing Dashboard", classes="dashboard-title")
                yield Static("Idle", id="overall-message", classes="overall-status")
            yield ProgressBar(total=100, id="overall-progress")
            yield Static("Jobs", classes="jobs-title")
            yield Vertical(id="jobs-container")
            yield Static("No active jobs", id="empty-state")
            with Horizontal(classes="dashboard-controls"):
                yield ImmediateButton("Pause All", id="pause-all")
                yield ImmediateButton("Resume All", id="resume-all")
                yield ImmediateButton("Clear Completed", id="clear-completed")

    def add_job(self, job_id: str, title: str, files: list[Path]) -> ProcessingJob:
        job = ProcessingJob(job_id, title, files)
        widget = JobStatusWidget(job)
        self.active_jobs[job_id] = job
        self._job_widgets[job_id] = widget
        if self.is_mounted:
            self.query_one("#empty-state").add_class("hidden")
            self.query_one("#jobs-container").mount(widget)
        return job

    def remove_job(self, job_id: str) -> None:
        self.active_jobs.pop(job_id, None)
        widget = self._job_widgets.pop(job_id, None)
        if widget is not None and widget.is_mounted:
            widget.remove()
        if self.is_mounted and not self.active_jobs:
            self.query_one("#empty-state").remove_class("hidden")

    def update_job_status(self, job_id: str, state: ProcessingState, progress: float, message: str) -> None:
        job = self.active_jobs[job_id]
        if state == ProcessingState.PROCESSING and job.start_time is None:
            job.start()
        job.state = state
        job.progress = progress
        job.message = message
        if state in {ProcessingState.COMPLETED, ProcessingState.FAILED, ProcessingState.CANCELLED}:
            job.end_time = job.end_time or datetime.now()
        self.total_progress = self._calculate_total_progress()
        self.is_processing = self.get_active_job_count() > 0
        self._emit_message(ProcessingJobStatus(job_id, state, progress, message))

    def update_job_file_progress(self, job_id: str, file_key: str, progress: float, status: str) -> None:
        self.active_jobs[job_id].update_file_progress(file_key, progress, status)

    def get_active_job_count(self) -> int:
        return sum(1 for job in self.active_jobs.values() if job.state == ProcessingState.PROCESSING)

    def get_completed_job_count(self) -> int:
        return sum(1 for job in self.active_jobs.values() if job.state == ProcessingState.COMPLETED)

    def get_failed_job_count(self) -> int:
        return sum(1 for job in self.active_jobs.values() if job.state == ProcessingState.FAILED)

    def _calculate_total_progress(self) -> float:
        if not self.active_jobs:
            return 0.0
        return sum(job.progress for job in self.active_jobs.values()) / len(self.active_jobs)

    @on(Button.Pressed, "#pause-all")
    def _pause_all(self) -> None:
        for job in self.active_jobs.values():
            if job.state == ProcessingState.PROCESSING:
                job.pause()

    @on(Button.Pressed, "#resume-all")
    def _resume_all(self) -> None:
        for job in self.active_jobs.values():
            if job.state == ProcessingState.PAUSED:
                job.resume()

    @on(Button.Pressed, "#clear-completed")
    def _clear_completed(self) -> None:
        for job_id, job in list(self.active_jobs.items()):
            if job.state in {ProcessingState.COMPLETED, ProcessingState.FAILED, ProcessingState.CANCELLED}:
                self.remove_job(job_id)
