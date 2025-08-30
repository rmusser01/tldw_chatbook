# NewIngestWindow Extension Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Media Type Parameters Reference](#media-type-parameters-reference)
4. [Extending the Interface](#extending-the-interface)
5. [Form Customization Patterns](#form-customization-patterns)
6. [Backend Integration](#backend-integration)
7. [Advanced Features](#advanced-features)
8. [Testing Extensions](#testing-extensions)
9. [Best Practices](#best-practices)

## Overview

The `NewIngestWindow` is a modern, extensible interface for media ingestion in tldw_chatbook. It provides a unified interface for processing various media types (video, audio, documents, PDFs, ebooks, web content) with support for batch processing, metadata management, and advanced analysis options.

### Key Features
- **Multi-file/Multi-line Input**: Process multiple files with corresponding metadata
- **Batch Processing**: Queue management for efficient processing
- **Flexible Analysis**: Configurable LLM analysis with provider/model selection
- **Progressive Disclosure**: Simple mode for basic users, advanced options for power users
- **Real-time Validation**: Input validation with immediate feedback
- **Extensible Architecture**: Easy to add new media types and processing options

## Architecture

### Component Structure
```
NewIngestWindow
├── Media Selection Panel (Left)
│   └── Media Type Cards (clickable selection)
├── Ingestion Panel (Right)
│   ├── Form Container (dynamic based on media type)
│   │   ├── Source Input Section
│   │   ├── Metadata Section
│   │   ├── Processing Options
│   │   └── Analysis Options
│   └── Action Buttons (Process/Queue)
└── Queue Management (background)
```

### Form Factory Pattern
Each media type has a dedicated form creation method:
- `_create_video_form()` → Video ingestion form
- `_create_audio_form()` → Audio ingestion form
- `_create_pdf_form()` → PDF processing form
- `_create_document_form()` → Document processing form
- `_create_ebook_form()` → Ebook ingestion form
- `_create_web_form()` → Web scraping form
- `_create_notes_form()` → Notes import form
- `_create_character_form()` → Character card import
- `_create_conversation_form()` → Conversation import

## Media Type Parameters Reference

### Video Processing Parameters

**Function**: `LocalVideoProcessor.process_videos()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | List[str] | Required | List of video URLs or local file paths |
| `download_video_flag` | bool | False | If True, keep video file; if False, extract audio only |
| `start_time` | Optional[str] | None | Start time for extraction (HH:MM:SS or seconds) |
| `end_time` | Optional[str] | None | End time for extraction (HH:MM:SS or seconds) |
| `use_cookies` | bool | False | Use cookies for authenticated downloads |
| `cookies` | Optional[Dict] | None | Cookie dictionary for authentication |
| `vad_use` | bool | False | Enable Voice Activity Detection |
| `transcription_provider` | str | "faster-whisper" | Transcription backend |
| `transcription_model` | str | "base" | Model size for transcription |
| `transcription_language` | str | "en" | Language code for transcription |
| `timestamp_option` | bool | True | Include timestamps in transcription |
| `perform_analysis` | bool | False | Run LLM analysis on content |
| `api_name` | Optional[str] | None | LLM API for analysis |
| `api_key` | Optional[str] | None | API key for LLM service |
| `custom_prompt` | Optional[str] | None | Custom analysis prompt |

### Audio Processing Parameters

**Function**: `LocalAudioProcessor.process_audio_files()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inputs` | List[str] | Required | List of audio URLs or file paths |
| `transcription_provider` | str | "faster-whisper" | Options: "faster-whisper", "parakeet-mlx", "lightning-whisper-mlx", "qwen2audio", "nemo" |
| `transcription_model` | str | "base" | Model size: "tiny", "base", "small", "medium", "large", "large-v2", "large-v3" |
| `transcription_language` | Optional[str] | "en" | Language code (ISO 639-1) |
| `translation_target_language` | Optional[str] | None | Target language for translation |
| `perform_chunking` | bool | True | Enable text chunking |
| `chunk_method` | Optional[str] | "sentences" | Options: "words", "sentences", "paragraphs", "tokens", "semantic" |
| `max_chunk_size` | int | 500 | Maximum chunk size |
| `chunk_overlap` | int | 200 | Overlap between chunks |
| `use_adaptive_chunking` | bool | False | Enable adaptive chunk sizing |
| `use_multi_level_chunking` | bool | False | Enable hierarchical chunking |
| `chunk_language` | Optional[str] | None | Language for semantic chunking |
| `diarize` | bool | False | Enable speaker diarization |
| `vad_use` | bool | False | Enable Voice Activity Detection |
| `timestamp_option` | bool | True | Include timestamps |
| `start_time` | Optional[str] | None | Start time (HH:MM:SS) |
| `end_time` | Optional[str] | None | End time (HH:MM:SS) |
| `perform_analysis` | bool | True | Run analysis/summarization |
| `api_name` | Optional[str] | None | LLM API provider |
| `api_key` | Optional[str] | None | API key |
| `custom_prompt` | Optional[str] | None | Custom prompt |
| `system_prompt` | Optional[str] | None | System prompt |
| `summarize_recursively` | bool | False | Recursive summarization |
| `save_original_file` | bool | False | Save downloaded file |

### PDF Processing Parameters

**Function**: `process_pdf()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_input` | Union[str, bytes, Path] | Required | PDF file path, bytes, or Path object |
| `filename` | str | Required | Original filename for metadata |
| `parser` | str | "pymupdf4llm" | Options: "pymupdf4llm", "pymupdf", "docling", "marker" |
| `title_override` | Optional[str] | None | Custom title |
| `author_override` | Optional[str] | None | Custom author |
| `keywords` | Optional[List[str]] | None | Document keywords |
| `perform_chunking` | bool | True | Enable chunking |
| `chunk_options` | Optional[Dict] | None | Chunking configuration |
| `perform_analysis` | bool | False | Run LLM analysis |
| `api_name` | Optional[str] | None | LLM provider |
| `api_key` | Optional[str] | None | API key |
| `custom_prompt` | Optional[str] | None | Analysis prompt |
| `system_prompt` | Optional[str] | None | System prompt |
| `summarize_recursively` | bool | False | Recursive summarization |
| `enable_ocr` | bool | False | Enable OCR for scanned documents |
| `ocr_language` | str | "en" | OCR language code |
| `ocr_backend` | str | "auto" | Options: "auto", "tesseract", "easyocr", "doctr", "paddle" |

### Document Processing Parameters

**Function**: `process_document()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | Required | Path to document file |
| `title_override` | Optional[str] | None | Custom title |
| `author_override` | Optional[str] | None | Custom author |
| `keywords` | Optional[List[str]] | None | Keywords list |
| `custom_prompt` | Optional[str] | None | Analysis prompt |
| `system_prompt` | Optional[str] | None | System prompt |
| `summary` | Optional[str] | None | Pre-provided summary |
| `auto_summarize` | bool | False | Auto-generate summary |
| `api_name` | Optional[str] | None | LLM provider |
| `api_key` | Optional[str] | None | API key |
| `chunk_options` | Optional[Dict] | None | Chunking config |
| `processing_method` | str | "auto" | Options: "auto", "docling", "native" |
| `enable_ocr` | bool | False | Enable OCR (docling only) |
| `ocr_language` | str | "en" | OCR language |

**Supported Formats**: .docx, .doc, .odt, .rtf, .pptx, .ppt, .xlsx, .xls, .ods, .odp

### Ebook Processing Parameters

**Function**: `process_ebook()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | str | Required | Path to ebook file |
| `title_override` | Optional[str] | None | Custom title |
| `author_override` | Optional[str] | None | Custom author |
| `keywords` | Optional[List[str]] | None | Keywords |
| `custom_prompt` | Optional[str] | None | Analysis prompt |
| `system_prompt` | Optional[str] | None | System prompt |
| `perform_chunking` | bool | True | Enable chunking |
| `chunk_options` | Optional[Dict] | None | Chunking configuration |
| `perform_analysis` | bool | False | Run analysis |
| `api_name` | Optional[str] | None | LLM provider |
| `api_key` | Optional[str] | None | API key |
| `summarize_recursively` | bool | False | Recursive summarization |
| `extraction_method` | str | "filtered" | Options: "filtered", "raw" |

**Supported Formats**: .epub, .mobi, .azw, .azw3, .fb2

### Web Scraping Parameters

**Function**: `scrape_article()`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | str | Required | URL to scrape |
| `custom_cookies` | Optional[List[Dict]] | None | Browser cookies for auth |
| `use_playwright` | bool | True | Use browser automation |
| `stealth_mode` | bool | False | Enable stealth mode |
| `wait_for_selector` | Optional[str] | None | CSS selector to wait for |
| `timeout` | int | 30000 | Page load timeout (ms) |
| `javascript_enabled` | bool | True | Enable JavaScript |
| `extract_images` | bool | False | Extract image URLs |
| `extract_links` | bool | False | Extract hyperlinks |
| `content_selector` | Optional[str] | None | CSS selector for content |

## Extending the Interface

### Adding a New Media Type

1. **Add the media type constant** in `Constants.py`:
```python
MEDIA_TYPE_PODCAST = "podcast"
```

2. **Create the form method** in `NewIngestWindow.py`:
```python
def _create_podcast_form(self) -> List[Widget]:
    """Create podcast-specific ingestion form."""
    widgets = []
    
    # RSS Feed URL input
    widgets.append(Label("RSS Feed URL:", classes="form-label"))
    widgets.append(TextArea(
        "",
        id="podcast-rss",
        classes="form-textarea"
    ))
    
    # Episode selection
    widgets.append(Label("Episode Selection:", classes="form-label"))
    widgets.append(Select(
        [("latest", "Latest Episode"), 
         ("all", "All Episodes"),
         ("range", "Date Range")],
        id="podcast-episodes"
    ))
    
    # Processing options
    widgets.append(Checkbox("Download audio files", False, id="podcast-download"))
    widgets.append(Checkbox("Generate transcripts", True, id="podcast-transcribe"))
    
    # Add action buttons
    widgets.extend(self._create_action_buttons("podcast"))
    
    return widgets
```

3. **Add to media card selection** in `compose()`:
```python
with Container(
    Label("🎙️", classes="media-icon"),
    Label("Podcast", classes="media-label"),
    id="media-card-podcast",
    classes="media-card"
):
    pass
```

4. **Handle in update method**:
```python
def _update_ingestion_form(self, media_type: str):
    # ... existing code ...
    elif media_type == "podcast":
        widgets = self._create_podcast_form()
```

### Adding Custom Fields to Existing Forms

To add new fields to an existing media type form:

```python
def _create_video_form(self) -> List[Widget]:
    widgets = []
    
    # ... existing fields ...
    
    # Add custom field - Video Quality Selection
    widgets.append(Label("Video Quality:", classes="form-label"))
    widgets.append(Select(
        [("auto", "Auto"),
         ("1080p", "1080p HD"),
         ("720p", "720p HD"),
         ("480p", "480p SD"),
         ("360p", "360p")],
        id="video-quality",
        value="auto"
    ))
    
    # Add custom field - Subtitle Options
    subtitle_container = Container(
        Label("Subtitle Options:", classes="form-label"),
        Checkbox("Download subtitles", False, id="video-subtitles"),
        Select(
            [("en", "English"), ("es", "Spanish"), ("fr", "French")],
            id="video-subtitle-lang",
            disabled=True  # Enable when checkbox is checked
        ),
        classes="subtitle-section"
    )
    widgets.append(subtitle_container)
    
    # ... rest of form ...
    
    return widgets
```

## Form Customization Patterns

### Progressive Disclosure Pattern

Show advanced options only when needed:

```python
def _create_audio_form(self) -> List[Widget]:
    widgets = []
    
    # Basic fields always visible
    basic_container = Container(
        Label("Audio Source:", classes="form-label"),
        TextArea("", id="audio-source"),
        classes="basic-section"
    )
    widgets.append(basic_container)
    
    # Advanced options in collapsible
    with Collapsible("Advanced Options", collapsed=True, id="audio-advanced"):
        advanced_widgets = []
        
        # Noise reduction
        advanced_widgets.append(Container(
            Label("Noise Reduction:", classes="form-label"),
            Select([
                ("none", "None"),
                ("light", "Light"),
                ("moderate", "Moderate"),
                ("aggressive", "Aggressive")
            ], id="audio-noise-reduction"),
            classes="noise-section"
        ))
        
        # Audio enhancement
        advanced_widgets.append(Container(
            Checkbox("Normalize audio levels", False, id="audio-normalize"),
            Checkbox("Remove silence", False, id="audio-remove-silence"),
            classes="enhancement-section"
        ))
    
    widgets.extend(advanced_widgets)
    return widgets
```

### Dynamic Field Dependencies

Enable/disable fields based on other field values:

```python
@on(Checkbox.Changed, "#video-subtitles")
def handle_subtitle_toggle(self, event):
    """Enable/disable subtitle language when checkbox changes."""
    subtitle_lang = self.query_one("#video-subtitle-lang", Select)
    subtitle_lang.disabled = not event.value
    
    if event.value:
        subtitle_lang.add_class("enabled")
    else:
        subtitle_lang.remove_class("enabled")

@on(Select.Changed, "#pdf-parser")
def handle_parser_change(self, event):
    """Show/hide OCR options based on parser selection."""
    ocr_container = self.query_one("#pdf-ocr-options", Container)
    
    if event.value == "docling":
        ocr_container.remove_class("hidden")
    else:
        ocr_container.add_class("hidden")
```

### Custom Validation

Add field-specific validation:

```python
def _validate_video_source(self, source: str) -> Tuple[bool, Optional[str]]:
    """Validate video source input."""
    lines = self._parse_multiline_input(source)
    
    for line in lines:
        # Check if it's a URL
        if line.startswith(('http://', 'https://')):
            # Validate URL format
            if not self._is_valid_url(line):
                return False, f"Invalid URL: {line}"
            
            # Check supported video platforms
            if "youtube.com" in line or "youtu.be" in line:
                if not self._has_youtube_dl():
                    return False, "yt-dlp not installed for YouTube videos"
        else:
            # Check if it's a valid file path
            path = Path(line)
            if not path.exists():
                return False, f"File not found: {line}"
            
            # Check file extension
            valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            if path.suffix.lower() not in valid_extensions:
                return False, f"Unsupported format: {path.suffix}"
    
    return True, None

@on(TextArea.Changed, "#video-source")
def validate_video_input(self, event):
    """Real-time validation of video source."""
    is_valid, error = self._validate_video_source(event.value)
    
    if not is_valid and event.value:
        self.notify(error, severity="warning")
        event.control.add_class("error")
    else:
        event.control.remove_class("error")
```

## Backend Integration

### Connecting UI to Processing Functions

1. **Gather form data**:
```python
def _gather_form_data(self, media_type: str) -> Dict[str, Any]:
    """Gather all form data for processing."""
    data = {"media_type": media_type}
    
    if media_type == "video":
        # Required fields
        source_widget = self.query_one("#video-source", TextArea)
        data["sources"] = self._parse_multiline_input(source_widget.text)
        
        # Optional metadata
        title_widget = self.query_one("#video-title", TextArea)
        data["titles"] = self._parse_multiline_input(title_widget.text)
        
        # Processing options
        data["vad_use"] = self.query_one("#video-vad", Checkbox).value
        data["transcribe"] = self.query_one("#video-transcribe", Checkbox).value
        
        # Time range
        start_time = self.query_one("#video-start-time", Input).value
        if start_time:
            data["start_time"] = start_time
            
        # Analysis options
        if self.query_one("#video-enable-analysis", Checkbox).value:
            data["perform_analysis"] = True
            data["api_name"] = self.query_one("#video-analysis-provider", Select).value
            data["api_model"] = self.query_one("#video-analysis-model", Select).value
            
            # Custom prompt if provided
            prompt_widget = self.query_one("#video-custom-prompt", TextArea)
            if prompt_widget.text:
                data["custom_prompt"] = prompt_widget.text
    
    return data
```

2. **Create processing task**:
```python
@work(exclusive=True)
async def _process_media(self, form_data: Dict[str, Any]):
    """Process media based on form data."""
    media_type = form_data["media_type"]
    
    try:
        if media_type == "video":
            from ...Local_Ingestion.video_processing import LocalVideoProcessor
            
            processor = LocalVideoProcessor(self.media_db)
            
            # Map form data to processor parameters
            result = await processor.process_videos(
                inputs=form_data["sources"],
                download_video_flag=form_data.get("save_original", False),
                start_time=form_data.get("start_time"),
                end_time=form_data.get("end_time"),
                vad_use=form_data.get("vad_use", False),
                transcription_provider=form_data.get("transcription_provider", "faster-whisper"),
                transcription_model=form_data.get("transcription_model", "base"),
                perform_analysis=form_data.get("perform_analysis", False),
                api_name=form_data.get("api_name"),
                custom_prompt=form_data.get("custom_prompt")
            )
            
            self.notify(f"Successfully processed {len(form_data['sources'])} video(s)")
            
        elif media_type == "pdf":
            from ...Local_Ingestion.PDF_Processing_Lib import process_pdf
            
            for source in form_data["sources"]:
                result = await process_pdf(
                    file_input=source,
                    filename=Path(source).name,
                    parser=form_data.get("parser", "pymupdf4llm"),
                    enable_ocr=form_data.get("enable_ocr", False),
                    ocr_language=form_data.get("ocr_language", "en"),
                    ocr_backend=form_data.get("ocr_backend", "auto"),
                    perform_analysis=form_data.get("perform_analysis", False),
                    api_name=form_data.get("api_name"),
                    custom_prompt=form_data.get("custom_prompt")
                )
                
        # ... handle other media types ...
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        self.notify(f"Processing failed: {str(e)}", severity="error")
```

### Adding Progress Callbacks

Integrate progress reporting:

```python
def _create_progress_callback(self, total_items: int):
    """Create a progress callback for processing operations."""
    self.processing_progress = 0
    self.processing_total = total_items
    
    def update_progress(current: int, message: str = ""):
        """Update progress indicator."""
        self.processing_progress = current
        
        # Update UI from worker thread
        self.call_from_thread(
            self._update_progress_display,
            current,
            total_items,
            message
        )
        
        # Check for cancellation
        if self._processing_cancelled:
            raise ProcessingCancelled("User cancelled operation")
    
    return update_progress

def _update_progress_display(self, current: int, total: int, message: str):
    """Update progress display in UI."""
    progress_bar = self.query_one("#processing-progress", ProgressBar)
    progress_bar.update(progress=current, total=total)
    
    status_label = self.query_one("#processing-status", Label)
    status_label.update(f"{message} ({current}/{total})")
```

## Advanced Features

### OCR Configuration

Add comprehensive OCR settings:

```python
def _create_ocr_settings(self) -> Container:
    """Create OCR configuration section."""
    return Container(
        Label("OCR Settings:", classes="section-title"),
        
        # OCR Backend Selection
        Container(
            Label("OCR Backend:", classes="form-label"),
            Select([
                ("auto", "Auto-detect"),
                ("tesseract", "Tesseract"),
                ("easyocr", "EasyOCR"),
                ("doctr", "DocTR"),
                ("paddle", "PaddleOCR")
            ], id="ocr-backend", value="auto"),
            classes="ocr-backend-section"
        ),
        
        # Language Selection
        Container(
            Label("OCR Language:", classes="form-label"),
            Select([
                ("en", "English"),
                ("de", "German"),
                ("fr", "French"),
                ("es", "Spanish"),
                ("zh", "Chinese"),
                ("ja", "Japanese"),
                ("ko", "Korean"),
                ("ar", "Arabic"),
                ("multi", "Multi-language")
            ], id="ocr-language", value="en"),
            classes="ocr-language-section"
        ),
        
        # OCR Options
        Container(
            Checkbox("Preserve layout", True, id="ocr-preserve-layout"),
            Checkbox("Detect tables", True, id="ocr-detect-tables"),
            Checkbox("Extract images", False, id="ocr-extract-images"),
            classes="ocr-options-section"
        ),
        
        # Preprocessing
        Container(
            Label("Image Preprocessing:", classes="form-label"),
            Checkbox("Auto-rotate", True, id="ocr-auto-rotate"),
            Checkbox("Deskew", True, id="ocr-deskew"),
            Checkbox("Remove noise", False, id="ocr-denoise"),
            Checkbox("Enhance contrast", False, id="ocr-enhance"),
            classes="ocr-preprocessing-section"
        ),
        
        id="ocr-settings",
        classes="ocr-settings-container"
    )
```

### Transcription Provider Configuration

Dynamic transcription settings based on provider:

```python
def _create_transcription_settings(self) -> Container:
    """Create transcription configuration section."""
    from ...Local_Ingestion.transcription_service import TranscriptionService
    
    service = TranscriptionService()
    available_providers = service.get_available_providers()
    
    provider_options = [(p, p.replace("-", " ").title()) for p in available_providers]
    
    return Container(
        Label("Transcription Settings:", classes="section-title"),
        
        # Provider Selection
        Container(
            Label("Provider:", classes="form-label"),
            Select(provider_options, id="transcription-provider"),
            classes="provider-section"
        ),
        
        # Model Selection (dynamic based on provider)
        Container(
            Label("Model:", classes="form-label"),
            Select([], id="transcription-model"),  # Populated dynamically
            classes="model-section"
        ),
        
        # Language
        Container(
            Label("Language:", classes="form-label"),
            Input(value="en", id="transcription-language", placeholder="Language code (e.g., en, es, fr)"),
            classes="language-section"
        ),
        
        # Advanced Options
        Container(
            Checkbox("Enable VAD", False, id="transcription-vad"),
            Checkbox("Word timestamps", True, id="transcription-timestamps"),
            Checkbox("Speaker diarization", False, id="transcription-diarize"),
            classes="transcription-options"
        ),
        
        id="transcription-settings",
        classes="transcription-container"
    )

@on(Select.Changed, "#transcription-provider")
def update_model_options(self, event):
    """Update model options based on selected provider."""
    provider = event.value
    model_select = self.query_one("#transcription-model", Select)
    
    # Provider-specific models
    models = {
        "faster-whisper": [
            ("tiny", "Tiny (39M)"),
            ("base", "Base (74M)"),
            ("small", "Small (244M)"),
            ("medium", "Medium (769M)"),
            ("large-v2", "Large-v2 (1.5G)"),
            ("large-v3", "Large-v3 (1.5G)")
        ],
        "parakeet-mlx": [
            ("tiny", "Tiny"),
            ("base", "Base"),
            ("small", "Small"),
            ("large", "Large")
        ],
        "qwen2audio": [
            ("qwen2-audio-7b", "Qwen2-Audio 7B"),
            ("qwen2-audio-7b-instruct", "Qwen2-Audio 7B Instruct")
        ]
    }
    
    model_select.set_options(models.get(provider, [("default", "Default")]))
```

### Chunking Configuration

Advanced chunking options:

```python
def _create_chunking_settings(self) -> Container:
    """Create chunking configuration section."""
    return Container(
        Label("Chunking Settings:", classes="section-title"),
        
        # Method Selection
        Container(
            Label("Method:", classes="form-label"),
            Select([
                ("words", "By Words"),
                ("sentences", "By Sentences"),
                ("paragraphs", "By Paragraphs"),
                ("tokens", "By Tokens"),
                ("semantic", "Semantic"),
                ("sliding_window", "Sliding Window"),
                ("recursive", "Recursive Split")
            ], id="chunk-method", value="sentences"),
            classes="method-section"
        ),
        
        # Size Configuration
        Container(
            Label("Chunk Size:", classes="form-label"),
            Input(value="500", id="chunk-size", placeholder="Max chunk size"),
            Label("Overlap:", classes="form-label"),
            Input(value="100", id="chunk-overlap", placeholder="Overlap size"),
            classes="size-section"
        ),
        
        # Advanced Options
        Container(
            Checkbox("Adaptive chunking", False, id="chunk-adaptive"),
            Checkbox("Multi-level", False, id="chunk-multilevel"),
            Checkbox("Preserve sentences", True, id="chunk-preserve-sentences"),
            classes="chunk-options"
        ),
        
        # Semantic Chunking Options (shown when semantic selected)
        Container(
            Label("Embedding Model:", classes="form-label"),
            Select([
                ("sentence-transformers/all-MiniLM-L6-v2", "MiniLM-L6"),
                ("sentence-transformers/all-mpnet-base-v2", "MPNet Base"),
                ("BAAI/bge-small-en", "BGE Small"),
                ("BAAI/bge-base-en", "BGE Base")
            ], id="chunk-embedding-model"),
            Label("Similarity Threshold:", classes="form-label"),
            Input(value="0.7", id="chunk-similarity", placeholder="0.0 - 1.0"),
            id="semantic-chunk-options",
            classes="semantic-options hidden"  # Show when semantic selected
        ),
        
        id="chunking-settings",
        classes="chunking-container"
    )
```

## Testing Extensions

### Unit Testing Form Creation

```python
import pytest
from textual.app import App
from tldw_chatbook.UI.NewIngestWindow import NewIngestWindow

@pytest.mark.asyncio
async def test_custom_form_fields():
    """Test that custom form fields are created correctly."""
    class TestApp(App):
        def __init__(self):
            super().__init__()
            self.app_config = {"api_settings": {}}
        
        def compose(self):
            yield NewIngestWindow(self)
    
    app = TestApp()
    async with app.run_test() as pilot:
        window = app.query_one(NewIngestWindow)
        
        # Select video card to load video form
        await pilot.click("#media-card-video")
        await pilot.pause()
        
        # Check custom fields exist
        quality_select = window.query("#video-quality")
        assert len(quality_select) > 0
        assert quality_select.first().value == "auto"
        
        subtitle_checkbox = window.query("#video-subtitles")
        assert len(subtitle_checkbox) > 0
        assert not subtitle_checkbox.first().value
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_form_to_backend_integration():
    """Test that form data correctly maps to backend parameters."""
    from unittest.mock import Mock, patch
    
    app = TestApp()
    async with app.run_test(size=(120, 50)) as pilot:
        window = app.query_one(NewIngestWindow)
        
        # Fill out form
        await pilot.click("#media-card-pdf")
        await pilot.pause()
        
        pdf_source = window.query_one("#pdf-source", TextArea)
        pdf_source.load_text("test.pdf")
        
        # Enable OCR
        ocr_checkbox = window.query_one("#pdf-enable-ocr", Checkbox)
        ocr_checkbox.toggle()
        
        # Mock the backend processing
        with patch('tldw_chatbook.Local_Ingestion.PDF_Processing_Lib.process_pdf') as mock_process:
            mock_process.return_value = {"success": True}
            
            # Trigger processing
            await pilot.click("#submit-pdf")
            await pilot.pause(0.5)
            
            # Verify backend was called with correct parameters
            mock_process.assert_called_once()
            call_args = mock_process.call_args[1]
            
            assert call_args["enable_ocr"] == True
            assert call_args["ocr_language"] == "en"
            assert "test.pdf" in call_args["file_input"]
```

### Performance Testing

```python
@pytest.mark.asyncio
async def test_large_batch_processing():
    """Test performance with large number of files."""
    import time
    
    app = TestApp()
    async with app.run_test() as pilot:
        window = app.query_one(NewIngestWindow)
        
        # Create large file list
        files = [f"file_{i}.mp4" for i in range(100)]
        
        await pilot.click("#media-card-video")
        await pilot.pause()
        
        # Load files
        start_time = time.time()
        
        source_widget = window.query_one("#video-source", TextArea)
        source_widget.load_text("\n".join(files))
        await pilot.pause()
        
        # Add to queue
        await pilot.click("#video-add-queue")
        await pilot.pause()
        
        load_time = time.time() - start_time
        
        # Should handle 100 files in reasonable time
        assert load_time < 2.0
        assert len(window.ingestion_queue) == 1
        assert len(window.ingestion_queue[0].sources) == 100
```

## Best Practices

### 1. Form Organization

- **Group related fields** in Container widgets with descriptive classes
- **Use consistent labeling** for all form fields
- **Provide helpful placeholders** and tooltips
- **Implement progressive disclosure** for advanced options

### 2. Validation

- **Validate on input change** for immediate feedback
- **Provide clear error messages** that explain how to fix issues
- **Disable submit button** until required fields are valid
- **Use visual indicators** (red borders, warning icons) for invalid fields

### 3. Performance

- **Lazy load heavy components** (e.g., model lists)
- **Use workers for async operations** to keep UI responsive
- **Batch process multiple files** efficiently
- **Implement cancellation** for long-running operations

### 4. User Experience

- **Remember user preferences** between sessions
- **Provide sensible defaults** for all options
- **Show processing status** with progress bars
- **Allow queue management** (reorder, remove items)
- **Support drag-and-drop** for file selection

### 5. Code Structure

- **Keep form methods focused** - one method per media type
- **Extract common patterns** into helper methods
- **Use consistent naming** for IDs and classes
- **Document complex logic** with inline comments
- **Write tests** for new features

### 6. Error Handling

```python
def _safe_process(self, form_data: Dict[str, Any]):
    """Process with comprehensive error handling."""
    try:
        # Validate inputs first
        validation_errors = self._validate_form_data(form_data)
        if validation_errors:
            for error in validation_errors:
                self.notify(error, severity="warning")
            return
        
        # Process with timeout
        with timeout(300):  # 5 minute timeout
            result = self._process_media(form_data)
            
        # Handle success
        self.notify("Processing complete!", severity="success")
        self._clear_form()
        
    except TimeoutError:
        self.notify("Processing timed out", severity="error")
        logger.error(f"Timeout processing {form_data['media_type']}")
        
    except MemoryError:
        self.notify("Out of memory - try smaller files", severity="error")
        logger.error("Memory error during processing")
        
    except Exception as e:
        self.notify(f"Processing failed: {str(e)}", severity="error")
        logger.exception("Unexpected error during processing")
        
    finally:
        # Always cleanup
        self._cleanup_temp_files()
        self._reset_progress()
```

## Conclusion

The NewIngestWindow provides a flexible, extensible framework for media ingestion. By following the patterns and practices outlined in this guide, you can:

- Add support for new media types
- Expose advanced processing options
- Integrate with backend processing libraries
- Create intuitive, responsive user interfaces
- Maintain code quality through testing

Remember to consider the user experience, performance implications, and maintainability when extending the interface. The modular architecture makes it easy to add features while keeping the codebase organized and testable.

For questions or contributions, please refer to the project's contribution guidelines and open an issue or pull request on the repository.