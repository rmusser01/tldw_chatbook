# Voice_Cloning_Window.py
# Description: Voice Cloning management window for multiple TTS backends
#
# Imports
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
from datetime import datetime
import asyncio
from loguru import logger

# Textual imports
from textual.app import ComposeResult
from textual.screen import Screen
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Static, Button, Label, Input, TextArea, Select, DataTable, 
    TabbedContent, TabPane, LoadingIndicator, Rule, Collapsible,
    Switch, ListView, ListItem, RichLog
)
from textual.reactive import reactive
from textual.binding import Binding
from textual import work
from textual.worker import Worker
import asyncio

# Local imports
from ..TTS.backends.higgs_voice_manager import HiggsVoiceProfileManager
from ..TTS.backends.chatterbox_voice_manager import ChatterboxVoiceManager
from ..TTS.backends.voice_manager_base import VoiceManagerBase
from ..config import get_cli_setting
from ..Widgets.enhanced_file_picker import EnhancedFileOpen as FileOpen, EnhancedFileSave as FileSave
from ..Third_Party.textual_fspicker import Filters
from ..Event_Handlers.STTS_Events.stts_events import STTSPlaygroundGenerateEvent

#######################################################################################################################
#
# Voice Cloning Window
#

class VoiceCloningWindow(Screen):
    """
    Voice Cloning management window supporting multiple TTS backends.
    
    Features:
    - Multi-backend support (Higgs, Chatterbox, GPT-SoVITS)
    - Voice profile management (create, edit, delete, export)
    - Voice testing playground
    - Profile search and filtering
    """
    
    BINDINGS = [
        Binding("escape", "close", "Close"),
        Binding("ctrl+n", "new_profile", "New Profile"),
        Binding("ctrl+d", "delete_profile", "Delete Profile"),
        Binding("ctrl+e", "export_profile", "Export Profile"),
        Binding("ctrl+i", "import_profile", "Import Profile"),
        Binding("ctrl+t", "test_voice", "Test Voice"),
        Binding("ctrl+r", "refresh", "Refresh"),
    ]
    
    DEFAULT_CSS = """
    VoiceCloningWindow {
        align: center middle;
    }
    
    #voice-cloning-container {
        width: 90%;
        height: 90%;
        max-width: 120;
        max-height: 40;
        border: thick $accent;
        background: $surface;
    }
    
    .window-title {
        text-align: center;
        text-style: bold;
        color: $text;
        margin: 1;
    }
    
    #backend-selector-container {
        height: 3;
        margin: 1;
    }
    
    #profile-list-container {
        height: 15;
        border: solid $primary;
        margin: 1;
    }
    
    #profile-details-container {
        height: 10;
        border: solid $secondary;
        margin: 1;
        padding: 1;
    }
    
    #action-buttons {
        height: 3;
        margin: 1;
    }
    
    .form-group {
        margin-bottom: 1;
    }
    
    .form-label {
        margin-bottom: 0;
        color: $text-muted;
    }
    
    #test-playground {
        border: solid $primary;
        padding: 1;
        margin: 1;
    }
    
    #test-text-input {
        height: 5;
        margin-bottom: 1;
    }
    
    #test-log {
        height: 8;
        border: solid $secondary;
        margin-top: 1;
    }
    
    .status-indicator {
        margin: 0 1;
    }
    
    .profile-count {
        color: $text-muted;
        margin-left: 1;
    }
    
    .no-profiles-message {
        text-align: center;
        color: $text-muted;
        margin: 2;
    }
    
    .loading-container {
        align: center middle;
        height: 100%;
    }
    """
    
    def __init__(self):
        super().__init__()
        self.backend_managers: Dict[str, VoiceManagerBase] = {}
        self.current_backend = reactive("higgs")
        self.selected_profile: Optional[str] = None
        self.profiles_data: List[Dict[str, Any]] = []
        self._loading = False
        
    def compose(self) -> ComposeResult:
        """Compose the Voice Cloning UI"""
        with Container(id="voice-cloning-container"):
            yield Label("🎙️ Voice Cloning Manager", classes="window-title")
            
            # Backend selector
            with Horizontal(id="backend-selector-container"):
                yield Label("Backend:", classes="form-label")
                yield Select(
                    options=[
                        ("Higgs Audio", "higgs"),
                        ("Chatterbox", "chatterbox"),
                        ("GPT-SoVITS (Coming Soon)", "gpt-sovits"),
                    ],
                    value="higgs",
                    id="backend-select"
                )
                yield Static("", id="backend-status", classes="status-indicator")
            
            # Tabbed interface
            with TabbedContent():
                # Profile Management Tab
                with TabPane("Profile Management", id="profiles-tab"):
                    # Profile list
                    with Container(id="profile-list-container"):
                        yield DataTable(id="profile-table")
                    
                    # Profile details
                    with Collapsible(title="Profile Details", collapsed=True, id="profile-details-container"):
                        yield Static("Select a profile to view details", id="profile-details")
                    
                    # Action buttons
                    with Horizontal(id="action-buttons"):
                        yield Button("➕ New Profile", id="new-profile-btn", variant="primary")
                        yield Button("✏️ Edit", id="edit-profile-btn", disabled=True)
                        yield Button("🗑️ Delete", id="delete-profile-btn", disabled=True, variant="error")
                        yield Button("📤 Export", id="export-profile-btn", disabled=True)
                        yield Button("📥 Import", id="import-profile-btn")
                        yield Button("🔄 Refresh", id="refresh-btn")
                
                # Voice Testing Tab
                with TabPane("Voice Testing", id="testing-tab"):
                    with Container(id="test-playground"):
                        yield Label("Test Text:")
                        yield TextArea(
                            "Hello! This is a test of the voice cloning system. "
                            "The quick brown fox jumps over the lazy dog.",
                            id="test-text-input"
                        )
                        
                        with Horizontal():
                            yield Label("Profile:", classes="form-label")
                            yield Select(
                                options=[],
                                id="test-profile-select"
                            )
                        
                        with Horizontal(classes="form-group"):
                            yield Button("🔊 Generate", id="test-generate-btn", variant="primary")
                            yield Button("▶️ Play", id="test-play-btn", disabled=True)
                            yield Button("💾 Save", id="test-save-btn", disabled=True)
                        
                        yield Label("Test Log:")
                        yield RichLog(id="test-log", highlight=True, markup=True)
            
            # Close button
            yield Rule()
            with Horizontal():
                yield Button("Close", id="close-btn", variant="default")
    
    async def on_mount(self) -> None:
        """Initialize the window on mount"""
        # Check if TTS dependencies are available
        from ..Utils.optional_deps import DEPENDENCIES_AVAILABLE
        if not DEPENDENCIES_AVAILABLE.get('tts_processing', False):
            from ..Utils.widget_helpers import alert_tts_not_available
            # Show alert after a short delay to ensure UI is ready
            self.set_timer(0.1, lambda: alert_tts_not_available(self))
        
        # Initialize backend managers
        await self._initialize_backends()
        
        # Setup profile table
        table = self.query_one("#profile-table", DataTable)
        table.add_columns("Name", "Display Name", "Language", "Created", "Tags")
        table.cursor_type = "row"
        
        # Load profiles for default backend
        self.set_timer(0.1, self._load_profiles)
    
    async def _initialize_backends(self) -> None:
        """Initialize available backend managers"""
        try:
            # Initialize Higgs
            higgs_dir = Path(get_cli_setting("HiggsSettings", "voice_samples_dir", 
                                           "~/.config/tldw_cli/higgs_voices")).expanduser()
            self.backend_managers["higgs"] = HiggsVoiceProfileManager(higgs_dir)
            
            # Initialize Chatterbox
            chatterbox_dir = Path(get_cli_setting("app_tts", "CHATTERBOX_VOICE_DIR",
                                                "~/.config/tldw_cli/chatterbox_voices")).expanduser()
            self.backend_managers["chatterbox"] = ChatterboxVoiceManager(chatterbox_dir)
            
            # GPT-SoVITS placeholder
            # self.backend_managers["gpt-sovits"] = GPTSoVITSVoiceManager(...)
            
            logger.info(f"Initialized {len(self.backend_managers)} voice backend managers")
        except Exception as e:
            logger.error(f"Error initializing backends: {e}")
            self.notify(f"Failed to initialize backends: {e}", severity="error")
    
    @work(exclusive=True)
    async def _load_profiles(self) -> None:
        """Load profiles for the current backend"""
        if self._loading:
            return
            
        self._loading = True
        try:
            manager = self.backend_managers.get(self.current_backend)
            if not manager:
                profiles = []
            else:
                # Run the blocking operation in a thread
                profiles = await asyncio.to_thread(manager.list_profiles)
            
            # Update UI from main thread
            self._update_profile_display(profiles)
        except Exception as e:
            logger.error(f"Error loading profiles: {e}")
            self._update_profile_display([])
        finally:
            self._loading = False
    
    # Remove the on_worker_state_changed method as it's no longer needed
    
    def _update_profile_display(self, profiles: List[Dict[str, Any]]) -> None:
        """Update the profile table and selectors"""
        self.profiles_data = profiles
        
        # Update table
        table = self.query_one("#profile-table", DataTable)
        table.clear()
        
        # Update test profile selector
        test_select = self.query_one("#test-profile-select", Select)
        test_options = []
        
        for profile in profiles:
            # Add to table
            table.add_row(
                profile["name"],
                profile["display_name"],
                profile["language"],
                profile["created_at"][:10] if len(profile["created_at"]) > 10 else profile["created_at"],
                ", ".join(profile["tags"][:2]) if profile["tags"] else ""
            )
            
            # Add to test selector
            test_options.append((profile["name"], profile["display_name"]))
        
        # Update test selector
        test_select.set_options(test_options)
        
        # Update status
        backend_status = self.query_one("#backend-status", Static)
        backend_status.update(f"[green]{len(profiles)} profiles[/green]")
        
        # Reset selection
        self.selected_profile = None
        self._update_button_states()
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle profile selection"""
        if event.row_key is not None and event.row_key.value is not None:
            row_index = int(event.row_key.value)
            if 0 <= row_index < len(self.profiles_data):
                profile = self.profiles_data[row_index]
                self.selected_profile = profile["name"]
                self._show_profile_details(profile)
                self._update_button_states()
    
    def _show_profile_details(self, profile: Dict[str, Any]) -> None:
        """Display detailed profile information"""
        details = self.query_one("#profile-details", Static)
        
        details_text = f"""[bold]{profile['display_name']}[/bold]
Name: {profile['name']}
Language: {profile['language']}
Created: {profile['created_at']}
Backend: {profile.get('backend', self.current_backend)}
Tags: {', '.join(profile['tags']) if profile['tags'] else 'None'}

[dim]Description:[/dim]
{profile['description'] if profile['description'] else 'No description'}
"""
        
        if profile.get("audio_duration"):
            details_text += f"\nAudio Duration: {profile['audio_duration']:.1f}s"
        
        if profile.get("watermarked"):
            details_text += "\n[yellow]⚠️ Audio includes watermark[/yellow]"
        
        details.update(details_text)
        
        # Expand details if collapsed
        details_container = self.query_one("#profile-details-container", Collapsible)
        if details_container.collapsed:
            details_container.collapsed = False
    
    def _update_button_states(self) -> None:
        """Update button states based on selection"""
        has_selection = self.selected_profile is not None
        
        self.query_one("#edit-profile-btn", Button).disabled = not has_selection
        self.query_one("#delete-profile-btn", Button).disabled = not has_selection
        self.query_one("#export-profile-btn", Button).disabled = not has_selection
    
    def _reset_backend_to_higgs(self) -> None:
        """Reset backend select to higgs"""
        try:
            backend_select = self.query_one("#backend-select", Select)
            backend_select.value = "higgs"
            self.current_backend = "higgs"
        except Exception as e:
            logger.error(f"Failed to reset backend: {e}")
    
    async def on_select_changed(self, event: Select.Changed) -> None:
        """Handle backend selection change"""
        if event.select.id == "backend-select":
            self.current_backend = event.value
            if event.value == "gpt-sovits":
                self.notify("GPT-SoVITS support coming soon!", severity="information")
                # Reset to previous backend
                await self.call_after_refresh(self._reset_backend_to_higgs)
            else:
                self._load_profiles()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "close-btn":
            self.app.pop_screen()
        
        elif button_id == "new-profile-btn":
            await self._create_new_profile()
        
        elif button_id == "edit-profile-btn":
            await self._edit_profile()
        
        elif button_id == "delete-profile-btn":
            await self._delete_profile()
        
        elif button_id == "export-profile-btn":
            await self._export_profile()
        
        elif button_id == "import-profile-btn":
            await self._import_profile()
        
        elif button_id == "refresh-btn":
            self._load_profiles()
        
        elif button_id == "test-generate-btn":
            await self._test_generate_voice()
        
        elif button_id == "test-play-btn":
            await self._test_play_audio()
        
        elif button_id == "test-save-btn":
            await self._test_save_audio()
    
    async def _create_new_profile(self) -> None:
        """Create a new voice profile"""
        # Show file picker for reference audio
        def handle_audio_selection(path: Optional[Path]) -> None:
            if path:
                # Create the profile creation dialog
                from ..Widgets.voice_profile_dialog import VoiceProfileDialog
                
                def handle_profile_data(profile_data: Dict[str, Any]) -> None:
                    # Create profile using the current backend manager
                    manager = self.backend_managers.get(self.current_backend)
                    if manager:
                        success, message = manager.create_profile(
                            profile_name=profile_data["name"],
                            reference_audio_path=str(path),
                            display_name=profile_data["display_name"],
                            language=profile_data["language"],
                            description=profile_data["description"],
                            tags=profile_data["tags"]
                        )
                        
                        if success:
                            self.notify(message, severity="information")
                            self._load_profiles()
                        else:
                            self.notify(message, severity="error")
                
                # Push the profile dialog
                dialog = VoiceProfileDialog(
                    str(path),
                    on_submit=handle_profile_data
                )
                self.app.push_screen(dialog)
        
        # Show file picker
        file_open = FileOpen(
            title="Select Reference Audio",
            filters=Filters(
                (
                    "Audio Files",
                    lambda p: p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
                ),
                ("All Files", lambda p: True)
            )
        )
        file_open.on_file_selected = handle_audio_selection
        self.app.push_screen(file_open)
    
    async def _edit_profile(self) -> None:
        """Edit selected profile"""
        if not self.selected_profile:
            return
        
        # TODO: Implement profile editing dialog
        self.notify("Profile editing coming soon!", severity="information")
    
    async def _delete_profile(self) -> None:
        """Delete selected profile"""
        if not self.selected_profile:
            return
        
        # Confirm deletion
        from textual.widgets import Button as ConfirmButton
        from textual.containers import Container as ConfirmContainer
        from textual.screen import ModalScreen
        
        class ConfirmDialog(ModalScreen):
            def compose(self) -> ComposeResult:
                with ConfirmContainer(id="confirm-dialog"):
                    yield Label(f"Delete profile '{self.selected_profile}'?")
                    with Horizontal():
                        yield ConfirmButton("Delete", id="confirm-yes", variant="error")
                        yield ConfirmButton("Cancel", id="confirm-no")
            
            def on_button_pressed(self, event: ConfirmButton.Pressed) -> None:
                self.dismiss(event.button.id == "confirm-yes")
        
        if await self.app.push_screen_wait(ConfirmDialog()):
            # Delete the profile
            manager = self.backend_managers.get(self.current_backend)
            if manager:
                success, message = manager.delete_profile(self.selected_profile)
                if success:
                    self.notify(message, severity="information")
                    self._load_profiles()
                else:
                    self.notify(message, severity="error")
    
    async def _export_profile(self) -> None:
        """Export selected profile"""
        if not self.selected_profile:
            return
        
        # Show directory picker
        def handle_export_path(path: Optional[Path]) -> None:
            if path:
                manager = self.backend_managers.get(self.current_backend)
                if manager:
                    success, message = manager.export_profile(
                        self.selected_profile,
                        str(path)
                    )
                    if success:
                        self.notify(message, severity="information")
                    else:
                        self.notify(message, severity="error")
        
        # Use FileSave in directory mode
        file_save = FileSave(
            title="Select Export Directory",
            show_dirs_only=True
        )
        file_save.on_file_save = handle_export_path
        self.app.push_screen(file_save)
    
    async def _import_profile(self) -> None:
        """Import a voice profile"""
        # Show file picker for profile package
        def handle_import_selection(path: Optional[Path]) -> None:
            if path:
                manager = self.backend_managers.get(self.current_backend)
                if manager:
                    success, message = manager.import_profile(
                        str(path),
                        overwrite=False
                    )
                    if success:
                        self.notify(message, severity="information")
                        self._load_profiles()
                    else:
                        self.notify(message, severity="error")
        
        file_open = FileOpen(
            title="Select Profile Package",
            filters=Filters(
                ("Profile Files", lambda p: p.name == "profile.json" or p.is_dir()),
                ("All Files", lambda p: True)
            )
        )
        file_open.on_file_selected = handle_import_selection
        self.app.push_screen(file_open)
    
    async def _test_generate_voice(self) -> None:
        """Generate test audio with selected profile"""
        test_profile = self.query_one("#test-profile-select", Select).value
        test_text = self.query_one("#test-text-input", TextArea).text
        
        if not test_profile:
            self.notify("Please select a profile", severity="warning")
            return
        
        if not test_text.strip():
            self.notify("Please enter test text", severity="warning")
            return
        
        # Log to test log
        test_log = self.query_one("#test-log", RichLog)
        test_log.write(f"[yellow]Generating with profile '{test_profile}'...[/yellow]")
        
        # Post TTS generation event
        # The actual generation will be handled by the STTS event handler
        provider = "higgs" if self.current_backend == "higgs" else "chatterbox"
        
        # For profile-based voices, format as "profile:name"
        voice = f"profile:{test_profile}"
        
        event = STTSPlaygroundGenerateEvent(
            text=test_text,
            provider=provider,
            voice=voice,
            model="default",
            speed=1.0,
            format="mp3",
            extra_params={"source": "voice_cloning_test"}
        )
        
        # Post the event to be handled by the main app
        self.app.post_message(event)
        
        test_log.write("[green]Generation request sent![/green]")
        
        # Enable play button (actual audio will be handled by STTS handler)
        self.query_one("#test-play-btn", Button).disabled = False
    
    async def _test_play_audio(self) -> None:
        """Play the generated test audio"""
        # This would be handled by the main STTS audio player
        self.notify("Use the main TTS Playground to play generated audio", severity="information")
    
    async def _test_save_audio(self) -> None:
        """Save the generated test audio"""
        # This would be handled by the main STTS export functionality
        self.notify("Use the main TTS Playground to save generated audio", severity="information")
    
    def action_close(self) -> None:
        """Close the window"""
        self.app.pop_screen()
    
    def action_new_profile(self) -> None:
        """Create new profile action"""
        asyncio.create_task(self._create_new_profile())
    
    def action_delete_profile(self) -> None:
        """Delete profile action"""
        if self.selected_profile:
            asyncio.create_task(self._delete_profile())
    
    def action_export_profile(self) -> None:
        """Export profile action"""
        if self.selected_profile:
            asyncio.create_task(self._export_profile())
    
    def action_import_profile(self) -> None:
        """Import profile action"""
        asyncio.create_task(self._import_profile())
    
    def action_test_voice(self) -> None:
        """Test voice action"""
        asyncio.create_task(self._test_generate_voice())
    
    def action_refresh(self) -> None:
        """Refresh profiles"""
        self._load_profiles()

#
# End of Voice_Cloning_Window.py
#######################################################################################################################