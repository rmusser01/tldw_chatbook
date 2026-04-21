# Embeddings UI Rebuild Guide

## Current UI Flow

### 1. Entry Point

- The user clicks the "Ingest Content" tab in the main app.
- `app.py` line 1575 loads `("ingest", IngestWindow, "ingest-window")`.

### 2. `IngestWindow` Layout

- `Ingest_Window.py` line 348: `compose()` creates the left sidebar with navigation buttons.
- Line 362: Creates the "Video (Local)" button with ID `ingest-nav-local-video`.

### 3. Video Button Click

- When the user clicks "Video (Local)", the event handler calls `compose_local_video_tab()` around lines 792-798.
- That handler calls `create_ingest_ui(self.app_instance, media_type="video")`.

### 4. Factory Creates the UI

`IngestUIFactory.py` lines 42-43:

```python
if media_type == "video":
    return VideoIngestWindowRedesigned(app_instance)
```

### 5. `VideoIngestWindowRedesigned`

- Lives in `Ingest_Local_Video_Window.py`.
- Inherits from `BaseMediaIngestWindow`.
- Owns the advanced video-ingestion features currently wired into the ingest flow.

### 6. `BaseMediaIngestWindow` Structure

`base_media_ingest_window.py` lines 105-125 build the main layout:

```python
with VerticalScroll(classes="ingest-main-scroll"):
    yield from self.create_status_dashboard()
    yield from self.create_mode_toggle()
    with Container(classes="essential-section"):
        yield from self.create_file_selector()
        yield from self.create_basic_metadata()
    with Container(classes="media-options-section"):
        yield from self.create_media_specific_options()
    yield from self.create_process_button()
```

### 7. Video-Specific Options

`Ingest_Local_Video_Window.py` lines 94-147 create:

- Video processing options such as extract audio, download video, and time range.
- Transcription options such as provider, model, language, diarization, VAD, and timestamps.
- Analysis options such as enable analysis, prompts, and user/system prompts.
- Chunking options such as method, size, and overlap.

## Problem Summary

The broken "Video (Local)" UI is most likely caused by one or more of these issues:

1. CSS is not loading correctly, so the status dashboard and mode toggle render as empty boxes.
2. The mode toggle is not driving visibility correctly, so advanced sections stay hidden when they should be shown.
3. Containers mount but size incorrectly, leaving large empty regions in the layout.

## Manual Validation Steps

### Step 1: Verify the Factory Wiring

```bash
grep -n "VideoIngestWindowRedesigned" tldw_chatbook/Widgets/Media_Ingest/IngestUIFactory.py
```

### Step 2: Rebuild CSS

```bash
./build_css.sh
```

### Step 3: Smoke-Test UI Creation

```python
from tldw_chatbook.Widgets.Media_Ingest.IngestUIFactory import IngestUIFactory


class MockApp:
    def __init__(self):
        self.app_config = {"api_settings": {}}


app = MockApp()
widget = IngestUIFactory.create_ui(app, "video")
print(f"Created widget: {widget}")
print(f"Widget class: {widget.__class__.__name__}")
```

### Step 4: Check Visibility and Layout

Inspect the status dashboard and mode-toggle sections for these failure modes:

1. They render but remain invisible because of CSS visibility or display rules.
2. They occupy space but appear empty because the container height is wrong.
3. They fail to mount because composition or child-widget creation throws an error.

## Rebuilding the UI

### File Structure

```text
tldw_chatbook/
├── UI/
│   └── Ingest_Window.py             # Main ingest window with sidebar navigation
├── Widgets/Media_Ingest/
│   ├── IngestUIFactory.py           # Factory that chooses which ingest UI to use
│   ├── base_media_ingest_window.py  # Base class with shared functionality
│   ├── Ingest_Local_Video_Window.py # Existing video ingest implementation
│   └── [other media types...]
└── css/
    ├── components/_forms.tcss       # Form styling
    ├── features/_ingest.tcss        # Ingest-specific styling
    └── tldw_cli_modular.tcss        # Built stylesheet
```

### Step 1: Decide Where the New Widget Lives

- Option A: Replace the existing implementation in `tldw_chatbook/Widgets/Media_Ingest/Ingest_Local_Video_Window.py`.
- Option B: Create a new widget such as `tldw_chatbook/Widgets/Media_Ingest/MyVideoIngestWindow.py`.

### Step 2: Build the New Widget Class

```python
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll
from textual.widgets import Button, Checkbox, Label, Static, TextArea

if TYPE_CHECKING:
    from tldw_chatbook.app import TldwCli


class MyVideoIngestWindow(Container):
    """Custom video-ingestion window."""

    def __init__(self, app_instance: "TldwCli", **kwargs):
        super().__init__(**kwargs)
        self.app_instance = app_instance

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="my-video-ingest-scroll"):
            yield Static("Video Ingestion", classes="title")

            with Container(classes="file-section"):
                yield Label("Select Video Files:")
                yield Button("Browse Files", id="browse-files")
                yield Container(id="file-list")

            with Container(classes="url-section"):
                yield Label("Or Enter URLs:")
                yield TextArea(
                    text="# Enter video URLs here...",
                    id="url-input",
                    classes="url-textarea",
                )

            with Container(classes="options-section"):
                yield Label("Transcription Options:")
                yield Checkbox("Enable Speaker Diarization", id="diarization")
                yield Checkbox("Enable VAD", id="vad")
                yield Checkbox("Include Timestamps", value=True, id="timestamps")

                yield Label("Analysis Options:")
                yield Checkbox("Enable Analysis", id="enable-analysis")
                yield TextArea(
                    text="# User prompt for analysis...",
                    id="user-prompt",
                    classes="prompt-textarea",
                )
                yield TextArea(
                    text="# System prompt for analysis...",
                    id="system-prompt",
                    classes="prompt-textarea",
                )

            yield Button("Process Video", id="process-video", variant="primary")

    @on(Button.Pressed, "#browse-files")
    async def handle_browse_files(self) -> None:
        """Handle file selection."""
        pass

    @on(Button.Pressed, "#process-video")
    async def handle_process(self) -> None:
        """Handle video processing."""
        pass
```

### Step 3: Register the Widget in the Factory

```python
from .MyVideoIngestWindow import MyVideoIngestWindow


class IngestUIFactory:
    @staticmethod
    def create_ui(app_instance: "TldwCli", media_type: str = "video") -> Container:
        if media_type == "video":
            return MyVideoIngestWindow(app_instance)
        elif media_type == "audio":
            ...
```

### Step 4: Add Styling

Create styles in `tldw_chatbook/css/components/_forms.tcss`:

```css
.my-video-ingest-scroll {
    height: 100%;
    width: 100%;
    padding: 2;
}

.title {
    text-style: bold;
    color: $primary;
    margin-bottom: 2;
    text-align: center;
}

.file-section,
.url-section,
.options-section {
    margin-bottom: 2;
    padding: 1;
    border: round $primary;
    background: $surface;
}

.url-textarea,
.prompt-textarea {
    min-height: 5;
    max-height: 10;
    margin-bottom: 1;
    border: solid $primary;
    padding: 1;
}

.url-textarea:focus,
.prompt-textarea:focus {
    border: solid $accent;
    background: $accent 10%;
}

Checkbox {
    margin: 1 0;
}

Button#process-video {
    width: 100%;
    height: 3;
    margin-top: 2;
    text-style: bold;
}
```

Rebuild the stylesheet after editing:

```bash
./build_css.sh
```

### Step 5: Test the Widget in Isolation

```python
from textual.app import App

from tldw_chatbook.Widgets.Media_Ingest.IngestUIFactory import IngestUIFactory


class TestApp(App):
    def __init__(self):
        super().__init__()
        self.app_config = {"api_settings": {}}

    def compose(self):
        yield IngestUIFactory.create_ui(self, "video")


if __name__ == "__main__":
    app = TestApp()
    app.run()
```

Run it with:

```bash
python test_my_ui.py
```

### Step 6: Bypass the Factory Entirely

If you want to wire the widget directly instead of going through the factory, update `tldw_chatbook/UI/Ingest_Window.py`:

```python
def compose_local_video_tab(self) -> ComposeResult:
    """Compose the local video tab."""
    from ..Widgets.Media_Ingest.MyVideoIngestWindow import MyVideoIngestWindow

    window = MyVideoIngestWindow(self.app_instance)
    self._local_video_window = window
    yield window
```

## Key Points

1. Build every UI element in `compose()` so layout and state stay predictable.
2. Use `@on(...)` handlers plus CSS selectors for event wiring.
3. Apply reusable styling with `classes="..."` and rebuild CSS after style changes.
4. Keep the factory path when you want swap-in flexibility without changing the rest of the ingest flow.
5. Start with a minimal widget, verify layout, and then add advanced sections incrementally.
