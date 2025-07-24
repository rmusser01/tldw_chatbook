# character_voice_widget.py
# Description: Character voice assignment widget for audiobook generation
#
# Imports
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from loguru import logger
import re
#
# Third-party imports
from textual.app import ComposeResult
from textual.containers import ScrollableContainer, Horizontal, Vertical, Container, Grid
from textual.widget import Widget
from textual.widgets import (
    DataTable, Button, Label, Select, 
    Input, Switch, Static, Rule, Collapsible, TextArea
)
from textual.reactive import reactive
from textual.message import Message
from rich.text import Text
#
# Local imports
from tldw_chatbook.TTS.audiobook_generator import Character
#
#######################################################################################################################
#
# Events

@dataclass
class CharacterVoiceAssignEvent(Message):
    """Event emitted when character voice is assigned"""
    character_name: str
    voice_id: str
    voice_settings: Optional[Dict[str, Any]] = None

@dataclass
class CharacterDetectionEvent(Message):
    """Event emitted to request character detection"""
    content: str
    auto_assign: bool = False

#######################################################################################################################
#
# Character Voice Widget

class CharacterVoiceWidget(Widget):
    """Widget for managing character voice assignments"""
    
    DEFAULT_CSS = """
    CharacterVoiceWidget {
        height: 100%;
        width: 100%;
    }
    
    .voice-widget-container {
        height: 100%;
        layout: horizontal;
    }
    
    .character-list-section {
        width: 40%;
        padding: 1;
        border-right: solid $primary;
    }
    
    .voice-assignment-section {
        width: 60%;
        padding: 1;
    }
    
    .character-table {
        height: 70%;
        margin: 1 0;
    }
    
    .voice-controls {
        height: auto;
        margin: 1 0;
    }
    
    .voice-assignment-grid {
        grid-size: 2;
        grid-gutter: 1;
        height: auto;
        margin: 1 0;
    }
    
    .character-name-label {
        text-style: bold;
        width: 100%;
    }
    
    .voice-select {
        width: 100%;
    }
    
    .detection-controls {
        margin: 1 0;
    }
    
    .sample-text-area {
        height: 10;
        margin: 1 0;
    }
    
    .voice-preview-button {
        margin: 0 1;
    }
    """
    
    # Reactive properties
    characters = reactive([], recompose=True)
    selected_character_index = reactive(-1)
    voice_assignments = reactive({})
    provider = reactive("openai")
    
    def __init__(self, provider: str = "openai", **kwargs):
        super().__init__(**kwargs)
        self.provider = provider
        self.detected_characters = set()
        self.voice_options = []
    
    def compose(self) -> ComposeResult:
        """Compose the character voice UI"""
        with Container(classes="voice-widget-container"):
            # Left side - Character list
            with Vertical(classes="character-list-section"):
                yield Label("🎭 Detected Characters", classes="section-title")
                
                # Detection controls
                with Vertical(classes="detection-controls"):
                    with Horizontal(classes="form-row"):
                        yield Switch(
                            id="auto-detect-switch",
                            value=True
                        )
                        yield Label("Auto-detect characters")
                    
                    yield Button(
                        "🔍 Detect Characters", 
                        id="detect-characters-btn",
                        variant="primary"
                    )
                    
                    yield Button(
                        "➕ Add Character", 
                        id="add-character-btn",
                        variant="default"
                    )
                
                # Character table
                character_table = DataTable(
                    id="character-table",
                    classes="character-table",
                    show_cursor=True,
                    cursor_type="row",
                    zebra_stripes=True
                )
                character_table.add_columns(
                    "Character", "Lines", "Voice"
                )
                yield character_table
                
                # Quick actions
                with Horizontal(classes="form-row"):
                    yield Button("🗑️ Remove", id="remove-character-btn", variant="warning")
                    yield Button("🔄 Reset All", id="reset-voices-btn", variant="warning")
            
            # Right side - Voice assignment
            with Vertical(classes="voice-assignment-section"):
                yield Label("🎤 Voice Assignment", classes="section-title")
                
                # Selected character info
                with Collapsible(title="Character Details", collapsed=False):
                    yield Label("", id="selected-character-name", classes="character-name-label")
                    
                    # Voice selection
                    with Horizontal(classes="form-row"):
                        yield Label("Voice:", classes="form-label")
                        yield Select(
                            options=[],
                            id="character-voice-select",
                            classes="voice-select"
                        )
                    
                    # Voice settings (provider-specific)
                    with Collapsible(title="Voice Settings", collapsed=True):
                        # Speed
                        with Horizontal(classes="form-row"):
                            yield Label("Speed:", classes="form-label")
                            yield Input(
                                id="voice-speed-input",
                                value="1.0",
                                placeholder="0.25-4.0"
                            )
                        
                        # Pitch (if supported)
                        with Horizontal(classes="form-row"):
                            yield Label("Pitch:", classes="form-label")
                            yield Input(
                                id="voice-pitch-input",
                                value="0",
                                placeholder="-20 to 20"
                            )
                        
                        # Emotion/Style (if supported)
                        with Horizontal(classes="form-row"):
                            yield Label("Style:", classes="form-label")
                            yield Select(
                                options=[
                                    ("neutral", "Neutral"),
                                    ("happy", "Happy"),
                                    ("sad", "Sad"),
                                    ("angry", "Angry"),
                                    ("excited", "Excited"),
                                    ("calm", "Calm"),
                                ],
                                id="voice-style-select"
                            )
                
                # Sample text for preview
                yield Label("Sample Text:")
                yield TextArea(
                    id="sample-text-area",
                    classes="sample-text-area",
                    text="Hello, this is a sample of my voice for this character."
                )
                
                # Preview controls
                with Horizontal(classes="form-row"):
                    yield Button(
                        "🔊 Preview Voice", 
                        id="preview-voice-btn",
                        variant="success",
                        classes="voice-preview-button"
                    )
                    yield Button(
                        "💾 Save Assignment", 
                        id="save-assignment-btn",
                        variant="primary"
                    )
                
                # Bulk assignment
                with Collapsible(title="Bulk Assignment", collapsed=True):
                    yield Label("Assign voice to multiple characters:")
                    yield Select(
                        options=[],
                        id="bulk-voice-select"
                    )
                    with Horizontal(classes="form-row"):
                        yield Button("Apply to All", id="apply-all-btn", variant="default")
                        yield Button("Apply to Selected", id="apply-selected-btn", variant="default")
                
                # Voice assignment summary
                yield Rule()
                yield Label("Voice Assignments:")
                yield Static("No assignments yet", id="assignment-summary")
    
    def on_mount(self) -> None:
        """Initialize voice options when mounted"""
        self._update_voice_options()
    
    def watch_provider(self) -> None:
        """Update voice options when provider changes"""
        self._update_voice_options()
    
    def watch_characters(self) -> None:
        """Refresh the character table when characters change"""
        if self.is_mounted:
            self._refresh_character_table()
    
    def watch_selected_character_index(self) -> None:
        """Update the assignment UI when selection changes"""
        if self.is_mounted:
            self._update_assignment_ui()
    
    def _update_voice_options(self) -> None:
        """Update voice options based on current provider"""
        try:
            voice_select = self.query_one("#character-voice-select", Select)
            bulk_select = self.query_one("#bulk-voice-select", Select)
        except Exception as e:
            logger.debug(f"Voice select widgets not ready: {e}")
            return
        
        if self.provider == "openai":
            self.voice_options = [
                ("alloy", "Alloy"),
                ("ash", "Ash"),
                ("ballad", "Ballad"),
                ("coral", "Coral"),
                ("echo", "Echo"),
                ("fable", "Fable"),
                ("onyx", "Onyx"),
                ("nova", "Nova"),
                ("sage", "Sage"),
                ("shimmer", "Shimmer"),
                ("verse", "Verse"),
            ]
        elif self.provider == "elevenlabs":
            self.voice_options = [
                ("21m00Tcm4TlvDq8ikWAM", "Rachel"),
                ("AZnzlk1XvdvUeBnXmlld", "Domi"),
                ("EXAVITQu4vr4xnSDxMaL", "Bella"),
                ("ErXwobaYiN019PkySvjV", "Antoni"),
                ("MF3mGyEYCl7XYWbV9V6O", "Elli"),
                ("TxGEqnHWrfWFTfGW9XjX", "Josh"),
                ("VR6AewLTigWG4xSOukaG", "Arnold"),
                ("pNInz6obpgDQGcFmaJgB", "Adam"),
                ("yoZ06aMxZJJ28mfd3POQ", "Sam"),
            ]
        elif self.provider == "kokoro":
            self.voice_options = [
                ("af_bella", "Bella (US Female)"),
                ("af_nicole", "Nicole (US Female)"),
                ("am_adam", "Adam (US Male)"),
                ("am_michael", "Michael (US Male)"),
                ("bf_emma", "Emma (UK Female)"),
                ("bm_george", "George (UK Male)"),
            ]
        else:
            self.voice_options = [("default", "Default Voice")]
        
        # Add narrator option
        self.voice_options.insert(0, ("narrator", "Use Narrator Voice"))
        
        voice_select.set_options(self.voice_options)
        bulk_select.set_options(self.voice_options)
    
    def _refresh_character_table(self) -> None:
        """Refresh the character table"""
        try:
            table = self.query_one("#character-table", DataTable)
            table.clear()
        except Exception as e:
            logger.debug(f"Character table not ready: {e}")
            return
        
        for i, character in enumerate(self.characters):
            # Get assigned voice
            assigned_voice = self.voice_assignments.get(character.name, "narrator")
            voice_label = self._get_voice_label(assigned_voice)
            
            # Count dialogue lines (simplified)
            line_count = "N/A"  # Would need actual dialogue detection
            
            table.add_row(
                character.name,
                line_count,
                voice_label,
                key=str(i)
            )
    
    def _get_voice_label(self, voice_id: str) -> str:
        """Get display label for voice ID"""
        if voice_id == "narrator":
            return "📖 Narrator"
        
        for vid, label in self.voice_options:
            if vid == voice_id:
                return label
        
        return voice_id
    
    def _update_assignment_ui(self) -> None:
        """Update the assignment UI for selected character"""
        if 0 <= self.selected_character_index < len(self.characters):
            character = self.characters[self.selected_character_index]
            
            try:
                # Update character name
                name_label = self.query_one("#selected-character-name", Label)
                name_label.update(f"Character: {character.name}")
            except Exception as e:
                logger.debug(f"UI elements not ready: {e}")
                return
            
            try:
                # Update voice selection
                voice_select = self.query_one("#character-voice-select", Select)
                assigned_voice = self.voice_assignments.get(character.name, "narrator")
                voice_select.value = assigned_voice
                
                # Update sample text
                sample_text = self.query_one("#sample-text-area", TextArea)
                sample_text.text = f'"{character.name} speaking: {sample_text.text.strip()}"'
            except Exception as e:
                logger.debug(f"Some UI elements not ready: {e}")
        else:
            try:
                # Clear UI
                name_label = self.query_one("#selected-character-name", Label)
                name_label.update("No character selected")
            except Exception as e:
                logger.debug(f"UI elements not ready: {e}")
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle character selection"""
        if event.row_key is not None:
            self.selected_character_index = int(event.row_key.value)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        button_id = event.button.id
        
        if button_id == "detect-characters-btn":
            self._request_character_detection()
        elif button_id == "add-character-btn":
            self._add_character_manually()
        elif button_id == "remove-character-btn":
            self._remove_selected_character()
        elif button_id == "reset-voices-btn":
            self._reset_all_voices()
        elif button_id == "preview-voice-btn":
            self._preview_character_voice()
        elif button_id == "save-assignment-btn":
            self._save_voice_assignment()
        elif button_id == "apply-all-btn":
            self._apply_voice_to_all()
        elif button_id == "apply-selected-btn":
            self._apply_voice_to_selected()
    
    def _request_character_detection(self) -> None:
        """Request character detection from parent"""
        auto_assign = self.query_one("#auto-detect-switch", Switch).value
        # This would typically get content from the parent widget
        self.post_message(CharacterDetectionEvent("", auto_assign))
        self.app.notify("Character detection requested", severity="information")
    
    def _add_character_manually(self) -> None:
        """Add a character manually"""
        # In a real implementation, this would show a dialog
        # For now, add a placeholder
        new_character = Character(
            name=f"Character {len(self.characters) + 1}",
            voice="narrator",
            description="Manually added character"
        )
        self.characters.append(new_character)
        self.characters = self.characters  # Trigger reactive update
    
    def _remove_selected_character(self) -> None:
        """Remove the selected character"""
        if 0 <= self.selected_character_index < len(self.characters):
            character = self.characters.pop(self.selected_character_index)
            # Remove voice assignment
            if character.name in self.voice_assignments:
                del self.voice_assignments[character.name]
            
            self.characters = self.characters  # Trigger reactive update
            self.selected_character_index = min(
                self.selected_character_index, 
                len(self.characters) - 1
            )
            self._update_assignment_summary()
    
    def _reset_all_voices(self) -> None:
        """Reset all voice assignments to narrator"""
        self.voice_assignments.clear()
        self.voice_assignments = self.voice_assignments  # Trigger reactive
        self._refresh_character_table()
        self._update_assignment_summary()
        self.app.notify("All voices reset to narrator", severity="information")
    
    def _preview_character_voice(self) -> None:
        """Preview the selected character's voice"""
        if not (0 <= self.selected_character_index < len(self.characters)):
            self.app.notify("Please select a character", severity="warning")
            return
        
        character = self.characters[self.selected_character_index]
        voice_select = self.query_one("#character-voice-select", Select)
        voice_id = voice_select.value
        
        if not voice_id or voice_id == Select.BLANK:
            self.app.notify("Please select a voice", severity="warning")
            return
        
        # Get sample text
        sample_text = self.query_one("#sample-text-area", TextArea).text.strip()
        if not sample_text:
            sample_text = f"Hello, I am {character.name}."
        
        # Get voice settings
        voice_settings = self._get_current_voice_settings()
        
        # This would typically trigger a TTS preview event
        self.app.notify(f"Preview: {character.name} with voice {self._get_voice_label(voice_id)}", severity="information")
    
    def _save_voice_assignment(self) -> None:
        """Save the current voice assignment"""
        if not (0 <= self.selected_character_index < len(self.characters)):
            self.app.notify("Please select a character", severity="warning")
            return
        
        character = self.characters[self.selected_character_index]
        voice_select = self.query_one("#character-voice-select", Select)
        voice_id = voice_select.value
        
        if not voice_id or voice_id == Select.BLANK:
            self.app.notify("Please select a voice", severity="warning")
            return
        
        # Save assignment
        self.voice_assignments[character.name] = voice_id
        
        # Update character object
        character.voice = voice_id
        character.voice_settings = self._get_current_voice_settings()
        
        # Emit event
        self.post_message(CharacterVoiceAssignEvent(
            character_name=character.name,
            voice_id=voice_id,
            voice_settings=character.voice_settings
        ))
        
        # Update UI
        self._refresh_character_table()
        self._update_assignment_summary()
        
        self.app.notify(f"Voice assigned: {character.name} → {self._get_voice_label(voice_id)}", severity="success")
    
    def _get_current_voice_settings(self) -> Dict[str, Any]:
        """Get current voice settings from UI"""
        settings = {}
        
        # Speed
        speed_input = self.query_one("#voice-speed-input", Input)
        try:
            settings["speed"] = float(speed_input.value)
        except ValueError:
            settings["speed"] = 1.0
        
        # Pitch
        pitch_input = self.query_one("#voice-pitch-input", Input)
        try:
            settings["pitch"] = int(pitch_input.value)
        except ValueError:
            settings["pitch"] = 0
        
        # Style
        style_select = self.query_one("#voice-style-select", Select)
        if style_select.value and style_select.value != Select.BLANK:
            settings["style"] = style_select.value
        
        return settings
    
    def _apply_voice_to_all(self) -> None:
        """Apply selected voice to all characters"""
        bulk_select = self.query_one("#bulk-voice-select", Select)
        voice_id = bulk_select.value
        
        if not voice_id or voice_id == Select.BLANK:
            self.app.notify("Please select a voice", severity="warning")
            return
        
        for character in self.characters:
            self.voice_assignments[character.name] = voice_id
            character.voice = voice_id
        
        self._refresh_character_table()
        self._update_assignment_summary()
        self.app.notify(f"Applied {self._get_voice_label(voice_id)} to all characters", severity="success")
    
    def _apply_voice_to_selected(self) -> None:
        """Apply voice to selected characters (would need multi-select)"""
        self.app.notify("Multi-select not yet implemented", severity="information")
    
    def _update_assignment_summary(self) -> None:
        """Update the assignment summary display"""
        summary = self.query_one("#assignment-summary", Static)
        
        if not self.voice_assignments:
            summary.update("No assignments yet")
            return
        
        lines = []
        for char_name, voice_id in self.voice_assignments.items():
            voice_label = self._get_voice_label(voice_id)
            lines.append(f"• {char_name}: {voice_label}")
        
        summary.update("\n".join(lines))
    
    def detect_characters_from_text(self, text: str, auto_assign: bool = False) -> List[Character]:
        """Detect characters from text content"""
        # Simple implementation - would be more sophisticated in practice
        characters = []
        detected_names = set()
        
        # Look for dialogue patterns
        dialogue_patterns = [
            r'"[^"]+"\s+said\s+(\w+)',  # "Hello," said John
            r'(\w+)\s+said,?\s+"[^"]+"',  # John said, "Hello"
            r'"[^"]+"\s+(\w+)\s+replied',  # "Hello," John replied
            r'(\w+):\s+"[^"]+"',  # John: "Hello"
        ]
        
        for pattern in dialogue_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.strip()
                if name and name not in detected_names:
                    detected_names.add(name)
                    characters.append(Character(
                        name=name,
                        voice="narrator",
                        description=f"Character detected from dialogue"
                    ))
        
        # Auto-assign voices if requested
        if auto_assign and characters:
            # Distribute available voices among characters
            voice_count = len(self.voice_options) - 1  # Exclude narrator
            for i, character in enumerate(characters):
                if i < voice_count:
                    voice_id, _ = self.voice_options[i + 1]  # Skip narrator
                    character.voice = voice_id
                    self.voice_assignments[character.name] = voice_id
        
        self.characters = characters
        return characters
    
    def get_voice_assignments(self) -> Dict[str, str]:
        """Get all voice assignments"""
        return self.voice_assignments.copy()
    
    def get_character_voices(self) -> List[Character]:
        """Get all characters with their voice assignments"""
        return self.characters.copy()