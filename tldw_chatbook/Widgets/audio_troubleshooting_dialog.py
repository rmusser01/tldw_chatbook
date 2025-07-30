# audio_troubleshooting_dialog.py
"""
Audio device troubleshooting dialog with diagnostics and testing capabilities.
Helps users resolve common microphone and audio issues.
"""

from typing import Optional, List, Dict, Any
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Button, Label, Select, Static, ProgressBar, 
    LoadingIndicator, Collapsible, Rule, RichLog
)
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual import work
from loguru import logger
import asyncio
import time

from ..Audio.dictation_service_lazy import LazyLiveDictationService, AudioInitializationError


class AudioTroubleshootingDialog(ModalScreen[bool]):
    """
    Modal dialog for audio troubleshooting and device testing.
    
    Features:
    - Device enumeration and selection
    - Microphone level visualization
    - Test recording
    - Permission checks
    - Common issue diagnostics
    """
    
    DEFAULT_CSS = """
    AudioTroubleshootingDialog {
        align: center middle;
    }
    
    .dialog-container {
        width: 80;
        height: 50;
        background: $panel;
        border: thick $primary;
        padding: 2;
    }
    
    .dialog-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 2;
    }
    
    .status-section {
        margin: 1 0;
        padding: 1;
        border: round $surface;
    }
    
    .status-ok {
        background: $success-darken-3;
        border-color: $success;
    }
    
    .status-warning {
        background: $warning-darken-3;
        border-color: $warning;
    }
    
    .status-error {
        background: $error-darken-3;
        border-color: $error;
    }
    
    .device-info {
        margin: 1 0;
    }
    
    .level-meter {
        width: 100%;
        height: 3;
        margin: 1 0;
    }
    
    .test-controls {
        layout: horizontal;
        height: auto;
        margin: 1 0;
    }
    
    .test-controls Button {
        margin: 0 1;
    }
    
    .diagnostic-log {
        height: 15;
        border: round $surface;
        margin: 1 0;
    }
    
    .help-section {
        margin-top: 2;
        padding: 1;
        background: $boost;
        border: round $primary;
    }
    
    .help-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .help-item {
        margin: 0.5 0;
    }
    
    .close-buttons {
        layout: horizontal;
        height: auto;
        margin-top: 2;
        align: center middle;
    }
    
    .close-buttons Button {
        margin: 0 1;
    }
    """
    
    # Reactive state
    is_testing = reactive(False)
    current_level = reactive(0.0)
    test_status = reactive("")
    
    def __init__(self):
        """Initialize troubleshooting dialog."""
        super().__init__()
        self.dictation_service = None
        self.audio_devices = []
        self.selected_device = None
        self.test_recording = False
        self._level_update_task = None
    
    def compose(self) -> ComposeResult:
        """Compose the troubleshooting UI."""
        with Container(classes="dialog-container"):
            yield Label("🔧 Audio Troubleshooting", classes="dialog-title")
            
            with ScrollableContainer():
                # Status Overview
                with Container(id="status-container", classes="status-section"):
                    yield Label("Checking audio system...", id="status-text")
                    yield LoadingIndicator(id="status-loading")
                
                # Device Selection
                with Collapsible(title="🎤 Audio Devices", collapsed=False):
                    yield Label("Select Input Device:")
                    yield Select(
                        options=[("Detecting devices...", None)],
                        id="device-select",
                        disabled=True
                    )
                    yield Container(id="device-info", classes="device-info")
                
                # Level Meter
                with Collapsible(title="📊 Input Level", collapsed=False):
                    yield Label("Microphone Level:")
                    yield ProgressBar(
                        total=100,
                        show_eta=False,
                        id="level-meter",
                        classes="level-meter"
                    )
                    yield Static("Level: 0%", id="level-text")
                    
                    with Container(classes="test-controls"):
                        yield Button(
                            "Start Test",
                            id="test-start-btn",
                            variant="primary"
                        )
                        yield Button(
                            "Stop Test",
                            id="test-stop-btn",
                            disabled=True
                        )
                
                # Diagnostic Log
                with Collapsible(title="📋 Diagnostics", collapsed=True):
                    yield RichLog(
                        id="diagnostic-log",
                        classes="diagnostic-log",
                        highlight=True,
                        markup=True
                    )
                
                # Common Issues Help
                with Container(classes="help-section"):
                    yield Label("Common Solutions:", classes="help-title")
                    yield Static(
                        "• Check microphone is plugged in\n"
                        "• Grant microphone permissions to the app\n"
                        "• Close other apps using the microphone\n"
                        "• Try a different USB port (for USB mics)\n"
                        "• Check system sound settings\n"
                        "• Restart the application",
                        classes="help-item"
                    )
            
            # Close buttons
            with Container(classes="close-buttons"):
                yield Button("Apply & Close", id="apply-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")
    
    def on_mount(self):
        """Initialize when dialog is shown."""
        self.run_worker(self._initialize_audio())
    
    @work(exclusive=True)
    async def _initialize_audio(self):
        """Initialize audio system and enumerate devices."""
        log = self.query_one("#diagnostic-log", RichLog)
        log.write("[bold]Starting audio system check...[/bold]")
        
        try:
            # Show loading
            status_container = self.query_one("#status-container", Container)
            status_container.add_class("status-warning")
            
            # Initialize service
            self.dictation_service = LazyLiveDictationService()
            
            # Try to get devices
            await asyncio.sleep(0.5)  # Brief delay for UI
            
            try:
                self.audio_devices = await self.run_worker(
                    self._get_devices_safe
                ).wait()
                
                if self.audio_devices:
                    self._update_device_list()
                    self._set_status("✅ Audio system ready", "ok")
                    log.write(f"[green]Found {len(self.audio_devices)} audio devices[/green]")
                else:
                    self._set_status("⚠️ No audio devices found", "warning")
                    log.write("[yellow]No audio input devices detected[/yellow]")
                    
            except AudioInitializationError as e:
                self._set_status("❌ Audio initialization failed", "error")
                log.write(f"[red]Audio error: {e}[/red]")
                self._show_error_help(str(e))
                
        except Exception as e:
            self._set_status("❌ Unexpected error", "error")
            log.write(f"[red]Unexpected error: {e}[/red]")
            logger.error(f"Audio initialization error: {e}")
        
        finally:
            # Hide loading indicator
            loading = self.query_one("#status-loading", LoadingIndicator)
            loading.display = False
    
    def _get_devices_safe(self) -> List[Dict[str, Any]]:
        """Safely get audio devices."""
        try:
            return self.dictation_service.get_audio_devices()
        except Exception as e:
            logger.error(f"Failed to get audio devices: {e}")
            return []
    
    def _update_device_list(self):
        """Update device selection list."""
        if not self.audio_devices:
            return
        
        # Create options for select
        options = []
        for device in self.audio_devices:
            name = device.get('name', 'Unknown Device')
            device_id = device.get('id', None)
            options.append((name, device_id))
        
        # Update select widget
        device_select = self.query_one("#device-select", Select)
        device_select.set_options(options)
        device_select.disabled = False
        
        # Select first device by default
        if options:
            device_select.value = options[0][1]
            self._show_device_info(self.audio_devices[0])
    
    def _show_device_info(self, device: Dict[str, Any]):
        """Show information about selected device."""
        info_container = self.query_one("#device-info", Container)
        info_container.remove_children()
        
        info_text = f"Channels: {device.get('channels', 'Unknown')}\n"
        info_text += f"Sample Rate: {device.get('sample_rate', 'Unknown')} Hz\n"
        
        if 'is_default' in device and device['is_default']:
            info_text += "✓ System default device"
        
        info_container.mount(Static(info_text))
    
    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle device selection change."""
        if event.select.id == "device-select":
            self.selected_device = event.value
            
            # Find and show device info
            for device in self.audio_devices:
                if device.get('id') == event.value:
                    self._show_device_info(device)
                    break
            
            # Log change
            log = self.query_one("#diagnostic-log", RichLog)
            log.write(f"Selected device: {event.select.value}")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "test-start-btn":
            self._start_level_test()
        elif event.button.id == "test-stop-btn":
            self._stop_level_test()
        elif event.button.id == "apply-btn":
            self._apply_and_close()
        elif event.button.id == "cancel-btn":
            self.dismiss(False)
    
    def _start_level_test(self):
        """Start microphone level testing."""
        if not self.dictation_service:
            return
        
        self.is_testing = True
        
        # Update UI
        start_btn = self.query_one("#test-start-btn", Button)
        stop_btn = self.query_one("#test-stop-btn", Button)
        start_btn.disabled = True
        stop_btn.disabled = False
        
        # Log
        log = self.query_one("#diagnostic-log", RichLog)
        log.write("[blue]Starting microphone level test...[/blue]")
        
        # Set selected device if any
        if self.selected_device is not None:
            self.dictation_service.set_audio_device(self.selected_device)
        
        # Start level monitoring
        self._level_update_task = self.set_interval(
            0.1, self._update_level_meter
        )
        
        self._set_status("🎤 Testing microphone levels...", "ok")
    
    def _stop_level_test(self):
        """Stop microphone level testing."""
        self.is_testing = False
        
        # Cancel level updates
        if self._level_update_task:
            self._level_update_task.stop()
        
        # Update UI
        start_btn = self.query_one("#test-start-btn", Button)
        stop_btn = self.query_one("#test-stop-btn", Button)
        start_btn.disabled = False
        stop_btn.disabled = True
        
        # Reset meter
        meter = self.query_one("#level-meter", ProgressBar)
        meter.update(progress=0)
        
        level_text = self.query_one("#level-text", Static)
        level_text.update("Level: 0%")
        
        # Log
        log = self.query_one("#diagnostic-log", RichLog)
        log.write("[blue]Stopped microphone test[/blue]")
        
        self._set_status("✅ Test completed", "ok")
    
    def _update_level_meter(self):
        """Update the level meter with current audio level."""
        if not self.is_testing or not self.dictation_service:
            return
        
        try:
            # Get current audio level (0.0 to 1.0)
            level = self.dictation_service.get_audio_level()
            
            # Convert to percentage
            level_percent = int(level * 100)
            
            # Update UI
            meter = self.query_one("#level-meter", ProgressBar)
            meter.update(progress=level_percent)
            
            level_text = self.query_one("#level-text", Static)
            level_text.update(f"Level: {level_percent}%")
            
            # Add visual feedback for level ranges
            if level_percent > 80:
                level_text.styles.color = "red"
            elif level_percent > 50:
                level_text.styles.color = "green" 
            elif level_percent > 20:
                level_text.styles.color = "yellow"
            else:
                level_text.styles.color = "white"
                
        except Exception as e:
            logger.error(f"Failed to update level meter: {e}")
    
    def _set_status(self, text: str, status_type: str = "ok"):
        """Update status display."""
        status_container = self.query_one("#status-container", Container)
        status_text = self.query_one("#status-text", Label)
        
        status_text.update(text)
        
        # Update container class
        status_container.remove_class("status-ok", "status-warning", "status-error")
        status_container.add_class(f"status-{status_type}")
    
    def _show_error_help(self, error: str):
        """Show specific help for the error."""
        help_section = self.query_one(".help-section", Container)
        
        # Add error-specific help
        if "permission" in error.lower():
            extra_help = Static(
                "\n🔒 Permission Issue Detected:\n"
                "• On macOS: System Preferences → Security & Privacy → Microphone\n"
                "• On Windows: Settings → Privacy → Microphone\n"
                "• On Linux: Check PulseAudio/ALSA permissions",
                classes="help-item"
            )
            help_section.mount(extra_help)
        elif "no audio" in error.lower() or "no device" in error.lower():
            extra_help = Static(
                "\n🎤 No Microphone Detected:\n"
                "• Check if microphone is properly connected\n"
                "• Try a different USB port\n"
                "• Check if device appears in system sound settings",
                classes="help-item"
            )
            help_section.mount(extra_help)
    
    def _apply_and_close(self):
        """Apply settings and close dialog."""
        # Save selected device preference
        if self.selected_device is not None:
            from ..config import save_setting_to_cli_config
            save_setting_to_cli_config(
                'dictation', 
                'preferred_device_id', 
                self.selected_device
            )
            
            log = self.query_one("#diagnostic-log", RichLog)
            log.write("[green]Saved device preference[/green]")
        
        # Clean up
        if self._level_update_task:
            self._level_update_task.stop()
        
        self.dismiss(True)
    
    def on_dismiss(self, result: bool) -> None:
        """Clean up when dialog is dismissed."""
        if self._level_update_task:
            self._level_update_task.stop()
        
        if self.dictation_service:
            # Ensure any test recording is stopped
            if self.is_testing:
                self._stop_level_test()