"""Compatibility backend integration for legacy NewIngest tests."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from textual.message import Message

from .ProcessingDashboard import ProcessingJob, ProcessingState
from .UnifiedProcessor import AudioConfig, DocumentConfig, MediaConfig, VideoConfig, WebConfig


class ProcessingJobResult(Message):
    def __init__(self, job_id: str, success: bool, results: dict[str, Any], error: str | None = None) -> None:
        super().__init__()
        self.job_id = job_id
        self.success = success
        self.results = results
        self.error = error


class MediaProcessingService:
    def __init__(self, app_instance: Any) -> None:
        self.app_instance = app_instance
        self._active_jobs: dict[str, ProcessingJob] = {}
        self._job_workers: dict[str, Any] = {}

    def submit_job(self, config: MediaConfig, title: str | None = None) -> str:
        job_id = f"job-{len(self._active_jobs) + 1}"
        files = list(getattr(config, "files", []) or [])
        urls = [Path(url) for url in getattr(config, "urls", []) or []]
        job = ProcessingJob(job_id, title or self._auto_title(config), files + urls)
        self._active_jobs[job_id] = job
        worker = self._start_processing_worker(job_id, config)
        self._job_workers[job_id] = worker
        return job_id

    def _auto_title(self, config: MediaConfig) -> str:
        config_type = self._detect_config_type(config).title()
        count = len(getattr(config, "files", []) or []) + len(getattr(config, "urls", []) or [])
        suffix = f" ({count} items)" if count else ""
        return f"{config_type} Processing{suffix}"

    def _start_processing_worker(self, _job_id: str, _config: MediaConfig) -> Any:
        return Mock()

    def cancel_job(self, job_id: str) -> bool:
        job = self._active_jobs.get(job_id)
        if job is None:
            return False
        worker = self._job_workers.get(job_id)
        if worker is not None and hasattr(worker, "cancel"):
            worker.cancel()
        job.cancel()
        return True

    def get_job_status(self, job_id: str) -> ProcessingJob | None:
        return self._active_jobs.get(job_id)

    def get_active_jobs(self) -> dict[str, ProcessingJob]:
        return dict(self._active_jobs)

    def cleanup_completed_jobs(self) -> None:
        self._active_jobs = {
            job_id: job
            for job_id, job in self._active_jobs.items()
            if job.state not in {ProcessingState.COMPLETED, ProcessingState.FAILED, ProcessingState.CANCELLED}
        }

    def _detect_config_type(self, config: MediaConfig) -> str:
        if isinstance(config, VideoConfig):
            return "video"
        if isinstance(config, AudioConfig):
            return "audio"
        if isinstance(config, DocumentConfig):
            return "document"
        if isinstance(config, WebConfig):
            return "web"
        return "media"

    async def _call_video_processor(self, file_path: Path, _config: VideoConfig) -> dict[str, Any]:
        return self._simulated_result(file_path)

    async def _call_audio_processor(self, file_path: Path, _config: AudioConfig) -> dict[str, Any]:
        return self._simulated_result(file_path)

    async def _call_document_processor(self, file_path: Path, _config: DocumentConfig) -> dict[str, Any]:
        return self._simulated_result(file_path)

    @staticmethod
    def _simulated_result(file_path: Path | str) -> dict[str, Any]:
        return {
            "status": "simulated",
            "file_path": str(file_path),
            "processed_at": datetime.now().isoformat(),
        }

    async def _process_generic(self, job_id: str, config: MediaConfig) -> None:
        job = self._active_jobs[job_id]
        for file_path in config.files:
            key = str(file_path)
            job.results[key] = {"status": "processed", "file_path": key}
            job.file_statuses[key] = "completed"

    async def _process_video(self, job_id: str, config: VideoConfig) -> None:
        job = self._active_jobs[job_id]
        for file_path in config.files:
            if job.state == ProcessingState.CANCELLED:
                return
            key = str(file_path)
            job.results[key] = await self._call_video_processor(file_path, config)
            job.file_statuses[key] = "completed"
        for url in config.urls:
            if job.state == ProcessingState.CANCELLED:
                return
            job.results[url] = self._simulated_result(url)
            job.file_statuses[url] = "completed"

    async def _process_audio(self, job_id: str, config: AudioConfig) -> None:
        job = self._active_jobs[job_id]
        for file_path in config.files:
            key = str(file_path)
            job.results[key] = await self._call_audio_processor(file_path, config)
            job.file_statuses[key] = "completed"

    async def _process_document(self, job_id: str, config: DocumentConfig) -> None:
        job = self._active_jobs[job_id]
        for file_path in config.files:
            key = str(file_path)
            job.results[key] = await self._call_document_processor(file_path, config)
            job.file_statuses[key] = "completed"

    async def _process_web(self, job_id: str, config: WebConfig) -> None:
        job = self._active_jobs[job_id]
        for url in config.urls:
            job.results[url] = self._simulated_result(url)
            job.file_statuses[url] = "completed"


_processing_service: MediaProcessingService | None = None


def get_processing_service(app_instance: Any | None = None) -> MediaProcessingService:
    global _processing_service
    if _processing_service is None:
        if app_instance is None:
            raise ValueError("app_instance is required to initialize processing service")
        _processing_service = MediaProcessingService(app_instance)
    return _processing_service
