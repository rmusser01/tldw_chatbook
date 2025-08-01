# Dictation_Window.py
"""
Full dictation interface for the STTS tab.
Provides comprehensive live dictation functionality with settings and export options.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Label, Button, TextArea, Select, Input, Static, 
    RichLog, Switch, Collapsible, Rule, ListView, ListItem
)
from textual.widget import Widget
from textual.reactive import reactive
from textual.message import Message
from textual import work
from textual.binding import Binding
from loguru import logger
import json

# Local imports
from ..config import get_cli_setting, save_setting_to_cli_config
from ..Widgets.voice_input_widget import VoiceInputWidget, VoiceInputMessage
from ..Event_Handlers.Audio_Events import (
    DictationStartedEvent, DictationStoppedEvent,
    PartialTranscriptEvent, FinalTranscriptEvent,
    VoiceCommandEvent
)
from ..Utils.input_validation import validate_text_input


class DictationWindow(Widget):
    """
    Full dictation interface for the STTS tab.
    
    Features:
    - Live transcription display
    - Recording controls
    - Language and model selection
    - Export options
    - Command detection
    - Transcription history
    """
    
    BINDINGS = [
        Binding("ctrl+d", "toggle_dictation", "Start/Stop Dictation"),
        Binding("ctrl+p", "pause_dictation", "Pause/Resume"),
        Binding("ctrl+e", "export_transcript", "Export Transcript"),
        Binding("ctrl+c", "copy_transcript", "Copy to Clipboard"),
        Binding("ctrl+shift+c", "clear_transcript", "Clear Transcript"),
    ]
    
    DEFAULT_CSS = """
    DictationWindow {
        height: 100%;
        width: 100%;
    }
    
    .dictation-container {
        height: 100%;
        layout: vertical;
    }
    
    .dictation-header {
        height: auto;
        padding: 1;
        border-bottom: solid $surface;
    }
    
    .dictation-content {
        height: 1fr;
        layout: horizontal;
    }
    
    .transcript-area {
        width: 2fr;
        padding: 1;
        border-right: solid $surface;
    }
    
    .transcript-display {
        height: 1fr;
        padding: 1;
        border: round $surface;
        background: $boost;
    }
    
    .dictation-sidebar {
        width: 1fr;
        padding: 1;
        overflow-y: auto;
    }
    
    .control-section {
        margin-bottom: 2;
    }
    
    .section-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .stats-display {
        padding: 1;
        border: round $surface;
        background: $panel;
        margin-top: 1;
    }
    
    .export-buttons {
        layout: vertical;
        height: auto;
        margin-top: 1;
    }
    
    .export-buttons Button {
        width: 100%;
        margin-bottom: 1;
    }
    
    .history-list {
        height: 15;
        border: round $surface;
        margin-top: 1;
    }
    
    .command-indicator {
        color: $warning;
        text-style: italic;
    }
    
    .segment-timestamp {
        color: $text-muted;
        text-style: italic;
    }
    """
    
    # Reactive attributes
    is_dictating = reactive(False)
    transcript_text = reactive("")
    word_count = reactive(0)
    duration = reactive(0.0)
    
    def __init__(self):
        """Initialize dictation window."""
        super().__init__()
        
        # Transcript management
        self.transcript_segments = []
        self.transcript_history = []
        
        # Settings
        self.settings = self._load_settings()
        
        # Voice input widget reference
        self.voice_input = None
    
    def compose(self) -> ComposeResult:
        """Compose the dictation UI."""
        with Vertical(classes="dictation-container"):
            # Header
            with Horizontal(classes="dictation-header"):
                yield Label("Live Dictation", classes="section-title")
                yield Label("Press Ctrl+D to start/stop dictation", classes="help-text")
            
            # Main content area
            with Horizontal(classes="dictation-content"):
                # Transcript area
                with Vertical(classes="transcript-area"):
                    yield Label("Transcript", classes="section-title")
                    yield TextArea(
                        id="transcript-display",
                        classes="transcript-display",
                        read_only=True
                    )
                    
                    # Voice input widget
                    self.voice_input = VoiceInputWidget(
                        show_device_selector=True,
                        show_transcript_preview=False,
                        transcription_provider=self.settings.get('provider', 'auto'),
                        transcription_model=self.settings.get('model'),
                        language=self.settings.get('language', 'en')
                    )
                    yield self.voice_input
                
                # Sidebar
                with ScrollableContainer(classes="dictation-sidebar"):
                    # Settings section
                    with Vertical(classes="control-section"):
                        yield Label("Settings", classes="section-title")
                        
                        # Language selection
                        yield Label("Language:")
                        yield Select(
                            options=[
                                ("English", "en"),
                                ("Spanish", "es"),
                                ("French", "fr"),
                                ("German", "de"),
                                ("Italian", "it"),
                                ("Portuguese", "pt"),
                                ("Russian", "ru"),
                                ("Chinese", "zh"),
                                ("Japanese", "ja"),
                                ("Korean", "ko"),
                            ],
                            value=self.settings.get('language', 'en'),
                            id="language-select"
                        )
                        
                        # Provider selection
                        yield Label("Provider:")
                        yield Select(
                            options=[
                                ("Auto", "auto"),
                                ("Parakeet MLX", "parakeet-mlx"),
                                ("Faster Whisper", "faster-whisper"),
                                ("Lightning Whisper", "lightning-whisper"),
                            ],
                            value=self.settings.get('provider', 'auto'),
                            id="provider-select"
                        )
                        
                        # Options
                        yield Switch(
                            value=self.settings.get('punctuation', True),
                            id="punctuation-switch"
                        )
                        yield Label("Auto punctuation", classes="switch-label")
                        
                        yield Switch(
                            value=self.settings.get('commands', True),
                            id="commands-switch"
                        )
                        yield Label("Voice commands", classes="switch-label")
                    
                    # Statistics section
                    with Vertical(classes="control-section"):
                        yield Label("Statistics", classes="section-title")
                        with Vertical(id="stats-display", classes="stats-display"):
                            yield Static("Words: 0", id="word-count")
                            yield Static("Duration: 0:00", id="duration-display")
                            yield Static("Speed: 0 WPM", id="speed-display")
                    
                    # Export section
                    with Vertical(classes="control-section"):
                        yield Label("Export", classes="section-title")
                        with Vertical(classes="export-buttons"):
                            yield Button("Copy to Clipboard", id="copy-button")
                            yield Button("Save as Text", id="save-text-button")
                            yield Button("Save as Markdown", id="save-md-button")
                            yield Button("Save with Timestamps", id="save-timestamps-button")
                    
                    # History section
                    with Vertical(classes="control-section"):
                        yield Label("History", classes="section-title")
                        yield ListView(
                            id="history-list",
                            classes="history-list"
                        )
                        yield Button("Clear History", id="clear-history-button")
    
    def on_mount(self):
        """Initialize on mount."""
        # Load history
        self._load_history()
        
        # Set initial focus
        self.voice_input.focus()
    
    def on_voice_input_message(self, event: VoiceInputMessage) -> None:
        """Handle voice input messages."""
        if event.is_final:
            # Add to transcript
            self._add_transcript_segment(event.text)
        else:
            # Update current segment
            self._update_current_segment(event.text)
    
    def on_partial_transcript_event(self, event: PartialTranscriptEvent) -> None:
        """Handle partial transcript updates."""
        # The VoiceInputWidget already handles this, but we can add custom logic
        pass
    
    def on_final_transcript_event(self, event: FinalTranscriptEvent) -> None:
        """Handle final transcript segments."""
        self._add_transcript_segment(event.text, event.timestamp)
    
    def on_voice_command_event(self, event: VoiceCommandEvent) -> None:
        """Handle voice commands."""
        self._process_voice_command(event.command)
    
    def on_dictation_started_event(self, event: DictationStartedEvent) -> None:
        """Handle dictation start."""
        self.is_dictating = True
        self._reset_stats()
        logger.info(f"Dictation started with {event.provider}")
    
    def on_dictation_stopped_event(self, event: DictationStoppedEvent) -> None:
        """Handle dictation stop."""
        self.is_dictating = False
        self.duration = event.duration
        self.word_count = event.word_count
        self._update_stats()
        
        # Save to history
        if event.transcript:
            self._add_to_history(event.transcript)
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle select widget changes."""
        if event.select.id == "language-select":
            self.settings['language'] = event.value
            self._save_settings()
        elif event.select.id == "provider-select":
            self.settings['provider'] = event.value
            self._save_settings()
    
    def on_switch_changed(self, event: Switch.Changed) -> None:
        """Handle switch changes."""
        if event.switch.id == "punctuation-switch":
            self.settings['punctuation'] = event.value
            self._save_settings()
        elif event.switch.id == "commands-switch":
            self.settings['commands'] = event.value
            self._save_settings()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "copy-button":
            self.action_copy_transcript()
        elif button_id == "save-text-button":
            self._export_as_text()
        elif button_id == "save-md-button":
            self._export_as_markdown()
        elif button_id == "save-timestamps-button":
            self._export_with_timestamps()
        elif button_id == "clear-history-button":
            self._clear_history()
    
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle history item selection."""
        if event.list_view.id == "history-list":
            # Load selected transcript
            index = event.list_view.index
            if 0 <= index < len(self.transcript_history):
                item = self.transcript_history[index]
                self._load_transcript(item['transcript'])
    
    def action_toggle_dictation(self) -> None:
        """Toggle dictation on/off."""
        if self.is_dictating:
            self.voice_input.stop_recording()
        else:
            self.voice_input.start_recording()
    
    def action_pause_dictation(self) -> None:
        """Pause/resume dictation."""
        if self.is_dictating:
            self.voice_input.pause_recording()
    
    def action_export_transcript(self) -> None:
        """Export transcript with default format."""
        self._export_as_text()
    
    def action_copy_transcript(self) -> None:
        """Copy transcript to clipboard."""
        try:
            import pyperclip
            pyperclip.copy(self.transcript_text)
            self.notify("Transcript copied to clipboard")
        except Exception as e:
            logger.error(f"Failed to copy transcript: {e}")
            self.notify("Failed to copy transcript", severity="error")
    
    def action_clear_transcript(self) -> None:
        """Clear current transcript."""
        self.transcript_text = ""
        self.transcript_segments = []
        self.word_count = 0
        self.duration = 0.0
        self._update_display()
        self._update_stats()
    
    def _add_transcript_segment(self, text: str, timestamp: Optional[float] = None):
        """Add a new transcript segment."""
        if not text.strip():
            return
        
        segment = {
            'text': text,
            'timestamp': timestamp or (datetime.now() - self._start_time).total_seconds(),
            'is_command': False
        }
        
        self.transcript_segments.append(segment)
        self.transcript_text = self._format_transcript()
        self.word_count = len(self.transcript_text.split())
        self._update_display()
        self._update_stats()
    
    def _update_current_segment(self, text: str):
        """Update the current (partial) segment."""
        # Update display with partial text
        display = self.query_one("#transcript-display", TextArea)
        full_text = self._format_transcript()
        if text:
            full_text += f" {text}"
        display.load_text(full_text)
    
    def _format_transcript(self) -> str:
        """Format transcript segments into text."""
        lines = []
        for segment in self.transcript_segments:
            if segment.get('is_command'):
                lines.append(f"[Command: {segment['text']}]")
            else:
                lines.append(segment['text'])
        
        return ' '.join(lines)
    
    def _format_transcript_with_timestamps(self) -> str:
        """Format transcript with timestamps."""
        lines = []
        for segment in self.transcript_segments:
            timestamp = self._format_time(segment.get('timestamp', 0))
            if segment.get('is_command'):
                lines.append(f"[{timestamp}] [Command: {segment['text']}]")
            else:
                lines.append(f"[{timestamp}] {segment['text']}")
        
        return '\n'.join(lines)
    
    def _process_voice_command(self, command: str):
        """Process detected voice command."""
        logger.info(f"Voice command: {command}")
        
        # Add command to transcript
        segment = {
            'text': command,
            'timestamp': (datetime.now() - self._start_time).total_seconds(),
            'is_command': True
        }
        self.transcript_segments.append(segment)
        
        # Execute command
        if command == 'new_paragraph':
            self._add_transcript_segment("\n\n")
        elif command == 'new_line':
            self._add_transcript_segment("\n")
        elif command == 'clear_all':
            self.action_clear_transcript()
        elif command == 'stop_dictation':
            self.action_toggle_dictation()
    
    def _update_display(self):
        """Update transcript display."""
        display = self.query_one("#transcript-display", TextArea)
        display.load_text(self.transcript_text)
        
        # Auto-scroll to bottom
        display.scroll_end()
    
    def _update_stats(self):
        """Update statistics display."""
        self.query_one("#word-count", Static).update(f"Words: {self.word_count}")
        self.query_one("#duration-display", Static).update(f"Duration: {self._format_time(self.duration)}")
        
        # Calculate WPM
        wpm = 0
        if self.duration > 0:
            wpm = int((self.word_count / self.duration) * 60)
        self.query_one("#speed-display", Static).update(f"Speed: {wpm} WPM")
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds to MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"
    
    def _reset_stats(self):
        """Reset statistics for new session."""
        self._start_time = datetime.now()
        self.word_count = 0
        self.duration = 0.0
        self._update_stats()
    
    def _add_to_history(self, transcript: str):
        """Add transcript to history."""
        item = {
            'timestamp': datetime.now().isoformat(),
            'transcript': transcript,
            'word_count': len(transcript.split()),
            'duration': self.duration
        }
        
        self.transcript_history.insert(0, item)
        
        # Limit history size
        if len(self.transcript_history) > 50:
            self.transcript_history = self.transcript_history[:50]
        
        self._update_history_display()
        self._save_history()
    
    def _update_history_display(self):
        """Update history list display."""
        history_list = self.query_one("#history-list", ListView)
        history_list.clear()
        
        for item in self.transcript_history:
            timestamp = datetime.fromisoformat(item['timestamp'])
            label = f"{timestamp.strftime('%Y-%m-%d %H:%M')} - {item['word_count']} words"
            history_list.append(ListItem(Static(label)))
    
    def _load_transcript(self, transcript: str):
        """Load a transcript from history."""
        self.transcript_text = transcript
        self.transcript_segments = [{'text': transcript, 'timestamp': 0}]
        self.word_count = len(transcript.split())
        self._update_display()
        self._update_stats()
    
    def _export_as_text(self):
        """Export transcript as plain text."""
        self._save_file(self.transcript_text, "transcript.txt")
    
    def _export_as_markdown(self):
        """Export transcript as markdown."""
        content = f"# Transcript\n\n"
        content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        content += f"**Duration:** {self._format_time(self.duration)}\n"
        content += f"**Words:** {self.word_count}\n\n"
        content += "---\n\n"
        content += self.transcript_text
        
        self._save_file(content, "transcript.md")
    
    def _export_with_timestamps(self):
        """Export transcript with timestamps."""
        content = self._format_transcript_with_timestamps()
        self._save_file(content, "transcript_timestamps.txt")
    
    def _save_file(self, content: str, default_name: str):
        """Save content to file."""
        # In a real implementation, this would open a file dialog
        # For now, we'll save to a default location
        try:
            from pathlib import Path
            
            # Create exports directory
            export_dir = Path.home() / ".config" / "tldw_cli" / "dictation_exports"
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = export_dir / f"{timestamp}_{default_name}"
            
            # Save file
            filename.write_text(content)
            
            self.notify(f"Exported to: {filename.name}")
            logger.info(f"Exported transcript to: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export transcript: {e}")
            self.notify("Failed to export transcript", severity="error")
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load dictation settings."""
        settings = {
            'provider': get_cli_setting('dictation.provider', 'auto'),
            'model': get_cli_setting('dictation.model', None),
            'language': get_cli_setting('dictation.language', 'en'),
            'punctuation': get_cli_setting('dictation.punctuation', True),
            'commands': get_cli_setting('dictation.commands', True),
        }
        return settings
    
    def _save_settings(self):
        """Save dictation settings."""
        save_setting_to_cli_config('dictation', 'provider', self.settings['provider'])
        save_setting_to_cli_config('dictation', 'model', self.settings.get('model'))
        save_setting_to_cli_config('dictation', 'language', self.settings['language'])
        save_setting_to_cli_config('dictation', 'punctuation', self.settings['punctuation'])
        save_setting_to_cli_config('dictation', 'commands', self.settings['commands'])
    
    def _load_history(self):
        """Load transcript history."""
        try:
            history_file = Path.home() / ".config" / "tldw_cli" / "dictation_history.json"
            if history_file.exists():
                self.transcript_history = json.loads(history_file.read_text())
            else:
                self.transcript_history = []
            
            self._update_history_display()
            
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            self.transcript_history = []
    
    def _save_history(self):
        """Save transcript history."""
        try:
            history_file = Path.home() / ".config" / "tldw_cli" / "dictation_history.json"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            history_file.write_text(json.dumps(self.transcript_history, indent=2))
            
        except Exception as e:
            logger.error(f"Failed to save history: {e}")
    
    def _clear_history(self):
        """Clear transcript history."""
        self.transcript_history = []
        self._update_history_display()
        self._save_history()
        self.notify("History cleared")