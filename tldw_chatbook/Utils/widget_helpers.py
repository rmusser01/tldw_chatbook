# widget_helpers.py
# Common widget utilities for the application
#
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.widgets import Static, Button, Label
from textual.message import Message
from typing import Optional, List, Dict


class FeatureNotAvailableDialog(Container):
    """A dialog widget to show when optional features are not available."""
    
    DEFAULT_CSS = """
    FeatureNotAvailableDialog {
        width: 70;
        height: auto;
        background: $panel;
        border: thick $primary;
        align: center middle;
        layer: dialog;
        padding: 2 4;
    }
    
    FeatureNotAvailableDialog > Vertical {
        width: 100%;
        height: auto;
    }
    
    FeatureNotAvailableDialog .dialog-title {
        text-align: center;
        color: $warning;
        text-style: bold;
        margin-bottom: 1;
    }
    
    FeatureNotAvailableDialog .dialog-message {
        margin-bottom: 1;
    }
    
    FeatureNotAvailableDialog .install-command {
        background: $boost;
        padding: 1 2;
        margin: 1 0;
        border: solid $primary;
    }
    
    FeatureNotAvailableDialog .button-container {
        dock: bottom;
        align: center middle;
        height: 3;
        margin-top: 1;
    }
    
    FeatureNotAvailableDialog Button {
        margin: 0 1;
    }
    """
    
    class Dismissed(Message):
        """Message sent when dialog is dismissed."""
        pass
    
    def __init__(
        self,
        feature_name: str,
        missing_deps: List[str],
        install_command: str,
        additional_info: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_name = feature_name
        self.missing_deps = missing_deps
        self.install_command = install_command
        self.additional_info = additional_info
    
    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label(f"Feature Not Available: {self.feature_name}", classes="dialog-title")
            
            if len(self.missing_deps) == 1:
                yield Static(f"The required dependency '{self.missing_deps[0]}' is not installed.", classes="dialog-message")
            else:
                deps_text = ", ".join(f"'{dep}'" for dep in self.missing_deps[:-1])
                deps_text += f" and '{self.missing_deps[-1]}'"
                yield Static(f"The required dependencies {deps_text} are not installed.", classes="dialog-message")
            
            yield Static("To install the missing dependencies, run:", classes="dialog-message")
            yield Static(self.install_command, classes="install-command")
            
            if self.additional_info:
                yield Static(self.additional_info, classes="dialog-message")
            
            with Horizontal(classes="button-container"):
                yield Button("Copy Command", variant="primary", id="copy-command")
                yield Button("OK", variant="default", id="dismiss")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "copy-command":
            try:
                import pyperclip
                pyperclip.copy(self.install_command)
                self.notify("Command copied to clipboard!", severity="information")
            except Exception:
                self.notify("Could not copy to clipboard", severity="warning")
        elif event.button.id == "dismiss":
            self.post_message(self.Dismissed())
            self.remove()


def show_feature_alert(
    parent,
    feature_name: str,
    feature_key: str,
    extra_name: str,
    additional_info: Optional[str] = None,
    missing_deps: Optional[List[str]] = None
) -> None:
    """
    Show a standardized alert dialog for missing features.
    
    Args:
        parent: The parent widget (usually self)
        feature_name: Human-readable feature name
        feature_key: Key in DEPENDENCIES_AVAILABLE dict
        extra_name: Name of the pip extra to install
        additional_info: Optional additional information to display
        missing_deps: Optional list of specific missing dependencies
    """
    from .optional_deps import DEPENDENCIES_AVAILABLE
    
    # Check if already available (in case it was installed since startup)
    if DEPENDENCIES_AVAILABLE.get(feature_key, False):
        return
    
    # Default missing deps if not provided
    if missing_deps is None:
        missing_deps = [extra_name]
    
    # Standard install command
    install_command = f"pip install tldw_chatbook[{extra_name}]"
    
    # For development installs
    if additional_info is None and extra_name in ["embeddings_rag", "websearch", "pdf", "ebook"]:
        additional_info = "For development installations, use: pip install -e \".[{extra_name}]\""
    
    dialog = FeatureNotAvailableDialog(
        feature_name=feature_name,
        missing_deps=missing_deps,
        install_command=install_command,
        additional_info=additional_info
    )
    
    parent.mount(dialog)


# Feature-specific alert functions for common cases

def alert_embeddings_not_available(parent) -> None:
    """Show alert for missing embeddings/RAG dependencies."""
    from .optional_deps import DEPENDENCIES_AVAILABLE
    
    missing = []
    for dep in ['torch', 'transformers', 'numpy', 'chromadb', 'sentence_transformers']:
        if not DEPENDENCIES_AVAILABLE.get(dep, False):
            missing.append(dep)
    
    show_feature_alert(
        parent,
        feature_name="Embeddings & RAG",
        feature_key="embeddings_rag",
        extra_name="embeddings_rag",
        missing_deps=missing if missing else ['embeddings_rag'],
        additional_info="This feature requires PyTorch and other ML libraries. Installation may take several minutes."
    )


def alert_web_server_not_available(parent) -> None:
    """Show alert for missing web server dependencies."""
    show_feature_alert(
        parent,
        feature_name="Web Server",
        feature_key="web",
        extra_name="web",
        missing_deps=['textual-serve'],
        additional_info="After installation, restart the application and use --serve to run in web mode."
    )


def alert_pdf_not_available(parent) -> None:
    """Show alert for missing PDF processing dependencies."""
    show_feature_alert(
        parent,
        feature_name="PDF Processing",
        feature_key="pdf_processing",
        extra_name="pdf",
        missing_deps=['pymupdf']
    )


def alert_audio_not_available(parent) -> None:
    """Show alert for missing audio processing dependencies."""
    from .optional_deps import DEPENDENCIES_AVAILABLE
    
    missing = []
    if not DEPENDENCIES_AVAILABLE.get('soundfile', False):
        missing.append('soundfile')
    if not DEPENDENCIES_AVAILABLE.get('faster_whisper', False):
        missing.append('faster-whisper')
    
    show_feature_alert(
        parent,
        feature_name="Audio Processing",
        feature_key="audio_processing",
        extra_name="audio",
        missing_deps=missing if missing else ['audio'],
        additional_info="For YouTube downloads, yt-dlp is also required. FFmpeg must be installed separately."
    )


def alert_local_llm_not_available(parent) -> None:
    """Show alert for missing local LLM dependencies."""
    import sys
    
    if sys.platform == "darwin":
        extra = "local_mlx"
        info = "MLX-LM is optimized for Apple Silicon Macs."
    else:
        extra = "local_vllm"
        info = "vLLM requires a CUDA-capable GPU."
    
    show_feature_alert(
        parent,
        feature_name="Local LLM",
        feature_key="local_llm",
        extra_name=extra,
        additional_info=info
    )


def alert_ocr_not_available(parent) -> None:
    """Show alert for missing OCR dependencies."""
    show_feature_alert(
        parent,
        feature_name="OCR Processing",
        feature_key="ocr_processing",
        extra_name="ocr_docext",
        additional_info="Multiple OCR backends are available. At least one is required."
    )


def alert_tts_not_available(parent) -> None:
    """Show alert for missing TTS dependencies."""
    show_feature_alert(
        parent,
        feature_name="Text-to-Speech",
        feature_key="tts_processing",
        extra_name="local_tts",
        missing_deps=['kokoro-onnx', 'pyaudio'],
        additional_info="Audio playback requires pyaudio. On macOS, install with: brew install portaudio && pip install pyaudio"
    )


def alert_mindmap_not_available(parent) -> None:
    """Show alert for missing mindmap dependencies."""
    show_feature_alert(
        parent,
        feature_name="Mindmap",
        feature_key="mindmap",
        extra_name="mindmap",
        missing_deps=['anytree'],
        additional_info="Note: anytree is included in base dependencies, so this should not normally appear."
    )


def alert_subscriptions_not_available(parent) -> None:
    """Show alert for missing subscriptions dependencies."""
    from .optional_deps import DEPENDENCIES_AVAILABLE
    
    missing = []
    for dep in ['markdown', 'schedule', 'feedparser', 'defusedxml']:
        if not DEPENDENCIES_AVAILABLE.get(dep, False):
            missing.append(dep)
    
    show_feature_alert(
        parent,
        feature_name="Subscriptions & RSS",
        feature_key="subscriptions",
        extra_name="subscriptions",
        missing_deps=missing if missing else ['subscriptions']
    )


def check_feature_availability(feature_key: str) -> bool:
    """
    Check if a feature is available and return bool.
    This is a convenience wrapper around DEPENDENCIES_AVAILABLE.
    """
    from .optional_deps import DEPENDENCIES_AVAILABLE
    return DEPENDENCIES_AVAILABLE.get(feature_key, False)