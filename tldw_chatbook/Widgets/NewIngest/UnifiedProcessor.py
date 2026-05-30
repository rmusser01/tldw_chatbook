"""Compatibility unified processor for legacy NewIngest tests."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Checkbox, Input, Select, Static

from .SmartFileDropZone import SmartFileDropZone


class ProcessingMode(str, Enum):
    SIMPLE = "simple"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ProcessingStatus(BaseModel):
    state: str = "idle"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_file: str | None = None
    files_completed: int = Field(0, ge=0)
    total_files: int = Field(0, ge=0)
    message: str = ""
    elapsed_time: float = 0.0


class BaseMediaConfig(BaseModel):
    files: list[Path] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    title: str | None = None
    author: str | None = None
    keywords: str | None = None
    chunk_size: int = Field(1000, ge=100)
    chunk_overlap: int = Field(200, ge=0, le=200)

    model_config = ConfigDict(arbitrary_types_allowed=True)


class VideoConfig(BaseMediaConfig):
    extract_audio_only: bool = False
    start_time: str | None = None
    end_time: str | None = None
    transcription_provider: str | None = None


class AudioConfig(BaseMediaConfig):
    speaker_diarization: bool = False
    noise_reduction: bool = False
    transcription_model: str | None = None
    transcription_provider: str | None = None


class DocumentConfig(BaseMediaConfig):
    ocr_enabled: bool = False
    preserve_formatting: bool = True
    chunk_method: str = "words"


class PDFConfig(DocumentConfig):
    extract_images: bool = False
    preserve_layout: bool = False


class EbookConfig(BaseMediaConfig):
    extract_metadata: bool = True
    preserve_chapters: bool = True
    include_toc: bool = True
    chunk_method: str = "chapter"


class WebConfig(BaseMediaConfig):
    extract_links: bool = False
    include_images: bool = False
    clean_html: bool = True


class MediaConfig(BaseMediaConfig):
    pass


class ProcessingStarted(Message):
    def __init__(self, job_id_or_config: Any) -> None:
        super().__init__()
        self.job_id = job_id_or_config if isinstance(job_id_or_config, str) else None
        self.config = None if isinstance(job_id_or_config, str) else job_id_or_config


class ProcessingComplete(Message):
    def __init__(self, job_id_or_results: Any, results: Any | None = None) -> None:
        super().__init__()
        self.job_id = job_id_or_results if isinstance(job_id_or_results, str) and results is not None else None
        self.results = results if results is not None else job_id_or_results


class ProcessingError(Message):
    def __init__(self, job_id_or_error: str | None, error: str | None = None) -> None:
        super().__init__()
        self.job_id = job_id_or_error if error is not None else None
        self.error = error if error is not None else str(job_id_or_error)


class ModeToggle(Widget):
    current_mode = reactive(ProcessingMode.SIMPLE)

    def __init__(self, *, button_id_prefix: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.button_id_prefix = button_id_prefix

    def compose(self) -> ComposeResult:
        with Container(classes="mode-toggle"):
            yield Static("Mode", classes="mode-label")
            with Horizontal(id="mode-selector"):
                yield Button("Simple", id=f"{self.button_id_prefix}simple-mode")
                yield Button("Advanced", id=f"{self.button_id_prefix}advanced-mode")
                yield Button("Expert", id=f"{self.button_id_prefix}expert-mode")
            yield Static("Simple processing mode", id="mode-description")

    def on_mount(self) -> None:
        """Synchronize selected mode state after the widget is mounted."""
        self._sync_mode_buttons()

    def watch_current_mode(self, _mode: ProcessingMode) -> None:
        """Refresh selected mode presentation when the reactive mode changes.

        Args:
            _mode: Updated processing mode value from the reactive watcher.
        """
        if self.is_mounted:
            self._sync_mode_buttons()

    @on(Button.Pressed)
    def _set_mode(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id.endswith("simple-mode"):
            self.current_mode = ProcessingMode.SIMPLE
        elif button_id.endswith("advanced-mode"):
            self.current_mode = ProcessingMode.ADVANCED
        elif button_id.endswith("expert-mode"):
            self.current_mode = ProcessingMode.EXPERT

    def _sync_mode_buttons(self) -> None:
        mode_str = (
            self.current_mode.value
            if isinstance(self.current_mode, ProcessingMode)
            else str(self.current_mode)
        )
        active_button_id = f"{self.button_id_prefix}{mode_str}-mode"
        for button in self.query("#mode-selector Button"):
            button.set_class(button.id == active_button_id, "active")

        description = self.query_one("#mode-description", Static)
        description.update(f"{mode_str.title()} processing mode")


class MediaSpecificOptions(Widget):
    media_type = reactive("auto")
    processing_mode = reactive(ProcessingMode.SIMPLE)

    def compose(self) -> ComposeResult:
        with Container(classes="media-options"):
            yield Static("Media Options", classes="options-title")
            yield Vertical(id="options-content")

    def on_mount(self) -> None:
        self._rebuild_options()

    def watch_media_type(self, _media_type: str) -> None:
        if self.is_mounted:
            self._rebuild_options()

    def _rebuild_options(self) -> None:
        content = self.query_one("#options-content")
        content.remove_children()
        if self.media_type == "video":
            content.mount(Checkbox("Extract audio only", id="extract-audio-only"))
            content.mount(Select([("Whisper", "whisper")], id="transcription-provider"))
        elif self.media_type == "audio":
            content.mount(Checkbox("Speaker diarization", id="speaker-diarization"))
            content.mount(Checkbox("Noise reduction", id="noise-reduction"))
        else:
            content.mount(Static("No specific options"))

    def get_config_data(self) -> dict[str, Any]:
        return {"media_type": self.media_type, "processing_mode": self.processing_mode.value}


class UnifiedProcessor(Widget):
    def __init__(
        self,
        app_instance: Any | None = None,
        *,
        initial_files: list[Path] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self._selected_files = list(initial_files or [])
        self._media_type = self._detect_media_type(self._selected_files)
        self._processing_mode = ProcessingMode.SIMPLE
        self._processing_status = ProcessingStatus()
        self.styles.height = 8

    def __iter__(self):
        yield self

    @property
    def selected_files(self) -> list[Path]:
        return self._selected_files

    @selected_files.setter
    def selected_files(self, value: list[Path]) -> None:
        self._selected_files = list(value)
        self.media_type = self._detect_media_type(self._selected_files)
        self._update_process_button()

    @property
    def media_type(self) -> str:
        return self._media_type

    @media_type.setter
    def media_type(self, value: str) -> None:
        self._media_type = value
        if self.is_mounted:
            try:
                self.query_one("#media-options", MediaSpecificOptions).media_type = value
            except Exception:
                pass

    @property
    def processing_mode(self) -> ProcessingMode:
        return self._processing_mode

    @processing_mode.setter
    def processing_mode(self, value: ProcessingMode) -> None:
        self._processing_mode = value
        if self.is_mounted:
            try:
                self.query_one("#media-options", MediaSpecificOptions).processing_mode = value
            except Exception:
                pass

    @property
    def processing_status(self) -> ProcessingStatus:
        return self._processing_status

    @processing_status.setter
    def processing_status(self, value: ProcessingStatus) -> None:
        self._processing_status = value
        self._update_status_display()

    @property
    def current_media_type(self) -> str:
        return self.media_type

    @current_media_type.setter
    def current_media_type(self, value: str) -> None:
        self.media_type = value

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Unified Processor", classes="processor-title")
            yield Static("Configure and process selected media", classes="processor-subtitle")
            yield Static("", id="status-container")
            yield Button("Process", id="process-button", disabled=not self.selected_files)
            with VerticalScroll(classes="processor-content"):
                with Horizontal():
                    with Vertical(classes="file-panel"):
                        yield SmartFileDropZone(id="file-selector")
                    with Vertical(classes="options-panel"):
                        yield Input(id="title-input", placeholder="Title")
                        yield Input(id="author-input", placeholder="Author")
                        yield Input(id="keywords-input", placeholder="Keywords")
                        yield ModeToggle(id="mode-toggle", button_id_prefix="processor-")
                        yield MediaSpecificOptions(id="media-options")

    def _detect_media_type(self, files: list[Path]) -> str:
        if not files:
            return "auto"
        groups = {self._type_for_suffix(path.suffix.lower()) for path in files}
        return groups.pop() if len(groups) == 1 else "mixed"

    @staticmethod
    def _type_for_suffix(suffix: str) -> str:
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            return "video"
        if suffix in {".mp3", ".wav", ".flac", ".m4a"}:
            return "audio"
        if suffix == ".pdf":
            return "pdf"
        if suffix in {".doc", ".docx", ".txt"}:
            return "document"
        if suffix in {".epub", ".mobi"}:
            return "ebook"
        if suffix in {".html", ".xml"}:
            return "web"
        return "media"

    def _get_config_model(self) -> type[BaseMediaConfig]:
        return {
            "video": VideoConfig,
            "audio": AudioConfig,
            "document": DocumentConfig,
            "pdf": PDFConfig,
            "ebook": EbookConfig,
            "web": WebConfig,
        }.get(self.media_type, MediaConfig)

    def _get_configuration(self) -> dict[str, Any]:
        data = {"files": self.selected_files}
        if self.is_mounted:
            for widget_id, key in (
                ("#title-input", "title"),
                ("#author-input", "author"),
                ("#keywords-input", "keywords"),
            ):
                try:
                    data[key] = self.query_one(widget_id, Input).value
                except Exception:
                    pass
        return data

    def _update_process_button(self) -> None:
        if not self.is_mounted:
            return
        self.query_one("#process-button", Button).disabled = not self.selected_files

    def _update_status_display(self) -> None:
        if not self.is_mounted:
            return
        try:
            status = self.processing_status
            self.query_one("#status-container", Static).update(
                f"{status.state}: {status.message} ({int(status.progress * 100)}%)"
            )
        except Exception:
            pass

    async def _process_media(self, config: BaseMediaConfig) -> dict[str, Any]:
        processed_files: list[dict[str, Any]] = []
        errors: list[str] = []
        processor = getattr(self, "_call_backend_processor", None)
        for file_path in config.files:
            try:
                if processor is None:
                    result = {"file": str(file_path), "status": "success"}
                else:
                    result = processor(file_path, config)
                    if hasattr(result, "__await__"):
                        result = await result
                processed_files.append(result)
            except Exception as exc:
                errors.append(str(exc))
        return {"processed_files": processed_files, "errors": errors}

    @on(Button.Pressed, "#process-button")
    def _process(self) -> None:
        if not self.selected_files:
            self._update_process_button()
            return
        from .BackendIntegration import get_processing_service

        config_model = self._get_config_model()
        config = config_model(**self._get_configuration())
        service = get_processing_service(self.app_instance or self.app)
        job_id = service.submit_job(config)
        self.post_message(ProcessingStarted(job_id))
