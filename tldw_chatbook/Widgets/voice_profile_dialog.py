# voice_profile_dialog.py
# Description: Dialog for creating/editing voice profiles
#
# Imports
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, Input, TextArea, Select, Button, Static
from textual.validation import Length, Regex
from loguru import logger

#######################################################################################################################
#
# Voice Profile Dialog
#

class VoiceProfileDialog(ModalScreen):
    """
    Modal dialog for creating or editing voice profiles.
    """
    
    DEFAULT_CSS = """
    VoiceProfileDialog {
        align: center middle;
    }
    
    #profile-dialog-container {
        width: 60;
        height: auto;
        max-height: 30;
        border: thick $accent;
        background: $surface;
        padding: 1;
    }
    
    .dialog-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .form-group {
        margin-bottom: 1;
    }
    
    .form-label {
        margin-bottom: 0;
        color: $text-muted;
    }
    
    .form-input {
        width: 100%;
    }
    
    .audio-info {
        color: $text-muted;
        margin-bottom: 1;
    }
    
    .button-row {
        margin-top: 2;
    }
    
    #profile-name-input {
        width: 100%;
    }
    
    #display-name-input {
        width: 100%;
    }
    
    #description-input {
        height: 5;
        width: 100%;
    }
    
    #tags-input {
        width: 100%;
    }
    
    .help-text {
        color: $text-muted;
        font-size: 12;
    }
    """
    
    def __init__(
        self,
        reference_audio_path: str,
        profile_data: Optional[Dict[str, Any]] = None,
        on_submit: Optional[Callable[[Dict[str, Any]], None]] = None
    ):
        """
        Initialize the voice profile dialog.
        
        Args:
            reference_audio_path: Path to the reference audio file
            profile_data: Existing profile data for editing (optional)
            on_submit: Callback function when profile is submitted
        """
        super().__init__()
        self.reference_audio_path = Path(reference_audio_path)
        self.profile_data = profile_data or {}
        self.on_submit = on_submit
        self.is_edit_mode = bool(profile_data)
    
    def compose(self) -> ComposeResult:
        """Compose the dialog UI"""
        with Container(id="profile-dialog-container"):
            # Title
            title = "Edit Voice Profile" if self.is_edit_mode else "New Voice Profile"
            yield Label(f"🎙️ {title}", classes="dialog-title")
            
            # Audio file info
            yield Static(
                f"Reference Audio: {self.reference_audio_path.name}",
                classes="audio-info"
            )
            
            # Profile name
            with Vertical(classes="form-group"):
                yield Label("Profile Name *", classes="form-label")
                yield Input(
                    value=self.profile_data.get("name", ""),
                    placeholder="unique_identifier",
                    id="profile-name-input",
                    validators=[
                        Length(minimum=1, maximum=50),
                        Regex(r"^[a-zA-Z0-9_-]+$", failure_description="Use only letters, numbers, underscore, and hyphen")
                    ]
                )
                yield Static("Unique identifier for this profile", classes="help-text")
            
            # Display name
            with Vertical(classes="form-group"):
                yield Label("Display Name *", classes="form-label")
                yield Input(
                    value=self.profile_data.get("display_name", ""),
                    placeholder="My Voice",
                    id="display-name-input",
                    validators=[Length(minimum=1, maximum=100)]
                )
            
            # Language
            with Vertical(classes="form-group"):
                yield Label("Language", classes="form-label")
                yield Select(
                    options=[
                        ("en", "English"),
                        ("es", "Spanish"),
                        ("fr", "French"),
                        ("de", "German"),
                        ("it", "Italian"),
                        ("pt", "Portuguese"),
                        ("ru", "Russian"),
                        ("zh", "Chinese"),
                        ("ja", "Japanese"),
                        ("ko", "Korean"),
                    ],
                    value=self.profile_data.get("language") or "en",
                    id="language-select"
                )
            
            # Description
            with Vertical(classes="form-group"):
                yield Label("Description", classes="form-label")
                yield TextArea(
                    self.profile_data.get("description", ""),
                    id="description-input"
                )
            
            # Tags
            with Vertical(classes="form-group"):
                yield Label("Tags", classes="form-label")
                yield Input(
                    value=", ".join(self.profile_data.get("tags", [])),
                    placeholder="professional, male, narrator",
                    id="tags-input"
                )
                yield Static("Comma-separated tags for categorization", classes="help-text")
            
            # Buttons
            with Horizontal(classes="button-row"):
                yield Button("Save", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")
    
    def on_mount(self) -> None:
        """Focus on the first input when mounted"""
        if not self.is_edit_mode:
            self.query_one("#profile-name-input", Input).focus()
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses"""
        if event.button.id == "cancel-btn":
            self.dismiss(None)
        
        elif event.button.id == "save-btn":
            # Validate inputs
            profile_name = self.query_one("#profile-name-input", Input).value.strip()
            display_name = self.query_one("#display-name-input", Input).value.strip()
            
            if not profile_name:
                self.notify("Profile name is required", severity="error")
                return
            
            if not display_name:
                self.notify("Display name is required", severity="error")
                return
            
            # Validate profile name input
            name_input = self.query_one("#profile-name-input", Input)
            if not name_input.is_valid:
                self.notify("Invalid profile name format", severity="error")
                return
            
            # Collect profile data
            language = self.query_one("#language-select", Select).value
            description = self.query_one("#description-input", TextArea).text.strip()
            tags_text = self.query_one("#tags-input", Input).value.strip()
            
            # Parse tags
            tags = []
            if tags_text:
                tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
            
            # Create profile data dictionary
            profile_data = {
                "name": profile_name,
                "display_name": display_name,
                "language": language,
                "description": description,
                "tags": tags
            }
            
            # Call the callback if provided
            if self.on_submit:
                self.on_submit(profile_data)
            
            # Dismiss the dialog
            self.dismiss(profile_data)

#
# End of voice_profile_dialog.py
#######################################################################################################################