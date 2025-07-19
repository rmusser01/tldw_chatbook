"""Voice blend management dialog for Kokoro TTS"""
from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Button, Input, Label, Select, Static
from textual.screen import ModalScreen
from textual.reactive import reactive
from textual.message import Message
from typing import List, Tuple, Dict, Optional, Any
import json
from loguru import logger


class VoiceBlendEntry(Container):
    """A single voice entry in the blend"""
    
    class Removed(Message):
        """Voice entry removal requested"""
        def __init__(self, entry: "VoiceBlendEntry") -> None:
            self.entry = entry
            super().__init__()
    
    def __init__(self, index: int = 0, voice: str = "", weight: float = 1.0) -> None:
        super().__init__()
        self.index = index
        self.initial_voice = voice
        self.initial_weight = weight
    
    def compose(self) -> ComposeResult:
        """Build the voice entry UI"""
        with Horizontal(classes="voice-blend-entry"):
            yield Select(
                options=[
                    ("af_bella", "Bella (American Female)"),
                    ("af_nicole", "Nicole (American Female)"),
                    ("af_sarah", "Sarah (American Female)"),
                    ("af_sky", "Sky (American Female)"),
                    ("am_adam", "Adam (American Male)"),
                    ("am_michael", "Michael (American Male)"),
                    ("bf_emma", "Emma (British Female)"),
                    ("bf_isabella", "Isabella (British Female)"),
                    ("bm_george", "George (British Male)"),
                    ("bm_lewis", "Lewis (British Male)"),
                ],
                id=f"voice-select-{self.index}",
                allow_blank=False
            )
            yield Input(
                value=str(self.initial_weight),
                placeholder="Weight (0.0-1.0)",
                type="number",
                id=f"weight-input-{self.index}",
                classes="weight-input"
            )
            yield Button("❌", id=f"remove-voice-{self.index}", classes="remove-btn")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press"""
        if event.button.id.startswith("remove-voice-"):
            self.post_message(self.Removed(self))


class VoiceBlendDialog(ModalScreen[Optional[Dict[str, Any]]]):
    """Dialog for creating/editing voice blends"""
    
    CSS = """
    VoiceBlendDialog {
        align: center middle;
    }
    
    #voice-blend-container {
        width: 60;
        height: auto;
        max-height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
    }
    
    .form-label {
        width: 15;
        height: 1;
        margin-top: 1;
    }
    
    #blend-name-input, #blend-description-input {
        width: 100%;
    }
    
    .voice-blend-entry {
        height: 3;
        margin-bottom: 1;
    }
    
    .weight-input {
        width: 15;
    }
    
    .remove-btn {
        width: 3;
        min-width: 3;
    }
    
    #add-voice-btn {
        margin: 1 0;
    }
    
    .button-row {
        dock: bottom;
        height: 3;
        align: center middle;
        margin-top: 1;
    }
    
    .button-row Button {
        margin: 0 1;
    }
    """
    
    def __init__(
        self,
        blend_data: Optional[Dict[str, Any]] = None,
        name: str = "",
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.blend_data = blend_data
        self.edit_mode = blend_data is not None
        self.voice_entries: List[VoiceBlendEntry] = []
        self.next_index = 0
    
    def compose(self) -> ComposeResult:
        """Build the dialog UI"""
        with Container(id="voice-blend-container"):
            yield Label(
                "Edit Voice Blend" if self.edit_mode else "Create Voice Blend",
                classes="dialog-title"
            )
            
            # Basic info
            with Vertical():
                with Horizontal(classes="form-row"):
                    yield Label("Name:", classes="form-label")
                    yield Input(
                        value=self.blend_data.get("name", "") if self.blend_data else "",
                        placeholder="My Custom Voice",
                        id="blend-name-input"
                    )
                
                with Horizontal(classes="form-row"):
                    yield Label("Description:", classes="form-label")
                    yield Input(
                        value=self.blend_data.get("description", "") if self.blend_data else "",
                        placeholder="Optional description",
                        id="blend-description-input"
                    )
            
            # Voice entries
            yield Label("Voices:", classes="form-label")
            with ScrollableContainer(id="voices-container"):
                yield Vertical(id="voice-entries-list")
            
            yield Button("➕ Add Voice", id="add-voice-btn", variant="default")
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")
    
    def on_mount(self) -> None:
        """Initialize with existing data if editing"""
        if self.blend_data and "voices" in self.blend_data:
            # Add existing voice entries
            for voice, weight in self.blend_data["voices"]:
                self.add_voice_entry(voice, weight)
        else:
            # Start with one empty entry
            self.add_voice_entry()
    
    def add_voice_entry(self, voice: str = "", weight: float = 1.0) -> None:
        """Add a new voice entry"""
        # Validate that the voice is in our known list
        known_voices = [
            "af_bella", "af_nicole", "af_sarah", "af_sky",
            "am_adam", "am_michael", 
            "bf_emma", "bf_isabella",
            "bm_george", "bm_lewis"
        ]
        
        # If voice is provided but not in list, use empty string to get default
        if voice and voice not in known_voices:
            logger.warning(f"Unknown voice '{voice}', will use default")
            voice = ""
        
        entry = VoiceBlendEntry(self.next_index, voice, weight)
        self.voice_entries.append(entry)
        self.next_index += 1
        
        # Mount the entry
        voice_list = self.query_one("#voice-entries-list", Vertical)
        voice_list.mount(entry)
    
    def on_voice_blend_entry_removed(self, message: VoiceBlendEntry.Removed) -> None:
        """Handle voice entry removal"""
        if len(self.voice_entries) > 1:  # Keep at least one entry
            self.voice_entries.remove(message.entry)
            message.entry.remove()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "add-voice-btn":
            self.add_voice_entry()
        elif event.button.id == "save-btn":
            self.save_blend()
        elif event.button.id == "cancel-btn":
            self.dismiss(None)
    
    def save_blend(self) -> None:
        """Validate and save the blend"""
        try:
            # Get basic info
            name = self.query_one("#blend-name-input", Input).value.strip()
            if not name:
                self.app.notify("Blend name is required", severity="error")
                return
            
            description = self.query_one("#blend-description-input", Input).value.strip()
            
            # Collect voices
            voices = []
            total_weight = 0.0
            
            for entry in self.voice_entries:
                voice_select = entry.query_one(f"#voice-select-{entry.index}", Select)
                weight_input = entry.query_one(f"#weight-input-{entry.index}", Input)
                
                voice = voice_select.value
                try:
                    weight = float(weight_input.value)
                    if weight <= 0:
                        self.app.notify("All weights must be positive", severity="error")
                        return
                    voices.append((voice, weight))
                    total_weight += weight
                except ValueError:
                    self.app.notify("Invalid weight value", severity="error")
                    return
            
            if not voices:
                self.app.notify("At least one voice is required", severity="error")
                return
            
            # Normalize weights to sum to 1.0
            voices = [(v, w/total_weight) for v, w in voices]
            
            # Build result
            result = {
                "name": name,
                "description": description,
                "voices": voices,
                "metadata": {
                    "created_by": "TUI",
                    "normalized": True
                }
            }
            
            self.dismiss(result)
            
        except Exception as e:
            logger.error(f"Failed to save blend: {e}")
            self.app.notify(f"Error saving blend: {e}", severity="error")