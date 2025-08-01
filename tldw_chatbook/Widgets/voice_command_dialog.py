# voice_command_dialog.py
"""
Dialog for configuring voice commands, including custom command creation.
"""

from typing import Optional, List
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, ScrollableContainer, Container
from textual.widgets import (
    Button, Label, Input, Select, Static, Switch,
    DataTable, TextArea, Rule, Collapsible
)
from textual.screen import ModalScreen
from textual.reactive import reactive
from loguru import logger

from ..Audio.voice_commands import (
    VoiceCommand, CommandType, get_command_processor
)


class VoiceCommandDialog(ModalScreen[bool]):
    """
    Dialog for managing voice commands.
    
    Features:
    - View all available commands
    - Enable/disable commands
    - Create custom commands
    - Test command recognition
    """
    
    DEFAULT_CSS = """
    VoiceCommandDialog {
        align: center middle;
    }
    
    .dialog-container {
        width: 90;
        height: 90%;
        max-height: 50;
        background: $panel;
        border: thick $primary;
        padding: 2;
    }
    
    .dialog-title {
        text-style: bold;
        text-align: center;
        margin-bottom: 2;
    }
    
    .command-table {
        height: 20;
        margin: 1 0;
    }
    
    .custom-form {
        padding: 1;
        border: round $surface;
        margin: 1 0;
    }
    
    .form-row {
        height: 3;
        margin-bottom: 1;
        layout: horizontal;
    }
    
    .form-label {
        width: 20;
        height: 3;
        content-align: center left;
    }
    
    .form-input {
        width: 1fr;
    }
    
    .test-area {
        height: 10;
        border: round $surface;
        padding: 1;
        margin: 1 0;
    }
    
    .test-input {
        width: 100%;
        margin-bottom: 1;
    }
    
    .test-result {
        padding: 1;
        background: $boost;
    }
    
    .dialog-buttons {
        layout: horizontal;
        height: auto;
        margin-top: 2;
        align: center middle;
    }
    
    .dialog-buttons Button {
        margin: 0 1;
    }
    
    .command-enabled {
        color: $success;
    }
    
    .command-disabled {
        color: $text-muted;
    }
    
    .command-custom {
        text-style: italic;
    }
    """
    
    def __init__(self):
        """Initialize voice command dialog."""
        super().__init__()
        self.command_processor = get_command_processor()
        self.selected_command = None
    
    def compose(self) -> ComposeResult:
        """Compose the dialog UI."""
        with Container(classes="dialog-container"):
            yield Label("🎤 Voice Command Configuration", classes="dialog-title")
            
            with ScrollableContainer():
                # Command List
                with Collapsible(title="📋 Available Commands", collapsed=False):
                    yield DataTable(
                        id="command-table",
                        classes="command-table",
                        cursor_type="row"
                    )
                
                # Custom Command Creation
                with Collapsible(title="➕ Create Custom Command"):
                    with Container(classes="custom-form"):
                        with Container(classes="form-row"):
                            yield Label("Phrase:", classes="form-label")
                            yield Input(
                                placeholder="e.g., 'add bullet point'",
                                id="phrase-input",
                                classes="form-input"
                            )
                        
                        with Container(classes="form-row"):
                            yield Label("Action:", classes="form-label")
                            yield Input(
                                placeholder="e.g., 'insert_bullet'",
                                id="action-input",
                                classes="form-input"
                            )
                        
                        with Container(classes="form-row"):
                            yield Label("Insert Text:", classes="form-label")
                            yield Input(
                                placeholder="e.g., '• '",
                                id="text-input",
                                classes="form-input"
                            )
                        
                        with Container(classes="form-row"):
                            yield Label("Description:", classes="form-label")
                            yield Input(
                                placeholder="What this command does",
                                id="description-input",
                                classes="form-input"
                            )
                        
                        with Container():
                            yield Button(
                                "Add Custom Command",
                                id="add-command-btn",
                                variant="primary"
                            )
                
                # Command Testing
                with Collapsible(title="🧪 Test Commands"):
                    with Container(classes="test-area"):
                        yield Label("Enter text to test command detection:")
                        yield Input(
                            placeholder="Type or speak a command...",
                            id="test-input",
                            classes="test-input"
                        )
                        yield Button("Test", id="test-btn")
                        yield Static(
                            "Test results will appear here",
                            id="test-result",
                            classes="test-result"
                        )
                
                # Help Section
                with Collapsible(title="❓ Command Help", collapsed=True):
                    yield Container(id="help-container")
            
            # Dialog buttons
            with Container(classes="dialog-buttons"):
                yield Button("Save & Close", id="save-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn")
    
    def on_mount(self):
        """Initialize dialog on mount."""
        self._populate_command_table()
        self._populate_help()
    
    def _populate_command_table(self):
        """Populate the command table with available commands."""
        table = self.query_one("#command-table", DataTable)
        
        # Add columns
        table.add_column("Enabled", width=8)
        table.add_column("Phrase", width=30)
        table.add_column("Type", width=15)
        table.add_column("Description", width=40)
        
        # Add rows
        for command in self.command_processor.get_command_list():
            enabled_text = "✓" if command.enabled else "✗"
            enabled_style = "command-enabled" if command.enabled else "command-disabled"
            
            type_text = command.command_type.value.replace("_", " ").title()
            if command.command_type == CommandType.CUSTOM:
                type_text = f"[italic]{type_text}[/italic]"
            
            table.add_row(
                enabled_text,
                command.phrase,
                type_text,
                command.description or "",
                key=command.phrase
            )
            
            # Style the enabled column
            table.get_cell_at((table.row_count - 1, 0)).add_class(enabled_style)
    
    def _populate_help(self):
        """Populate the help section with command categories."""
        help_container = self.query_one("#help-container", Container)
        help_container.remove_children()
        
        help_dict = self.command_processor.get_command_help()
        
        for category, commands in help_dict.items():
            # Category title
            title = category.replace("_", " ").title()
            help_container.mount(Label(f"{title}:", classes="help-title"))
            
            # Commands in category
            for cmd in commands:
                help_container.mount(Static(f"  • {cmd}"))
            
            help_container.mount(Rule())
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle command selection."""
        if event.data_table.id == "command-table":
            self.selected_command = event.row_key
            
            # Toggle enabled state
            command = self.command_processor.commands.get(event.row_key.value.lower())
            if command:
                command.enabled = not command.enabled
                self.command_processor.toggle_command(command.phrase, command.enabled)
                
                # Update table
                self._populate_command_table()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "add-command-btn":
            self._add_custom_command()
        elif event.button.id == "test-btn":
            self._test_command()
        elif event.button.id == "save-btn":
            self.dismiss(True)
        elif event.button.id == "cancel-btn":
            self.dismiss(False)
    
    def _add_custom_command(self):
        """Add a new custom command."""
        # Get form values
        phrase = self.query_one("#phrase-input", Input).value.strip()
        action = self.query_one("#action-input", Input).value.strip()
        text = self.query_one("#text-input", Input).value
        description = self.query_one("#description-input", Input).value.strip()
        
        if not phrase or not action:
            self.app.notify("Phrase and action are required", severity="error")
            return
        
        # Create command
        command = VoiceCommand(
            phrase=phrase,
            action=action,
            command_type=CommandType.CUSTOM,
            parameters={"text": text} if text else None,
            description=description,
            enabled=True
        )
        
        # Save command
        self.command_processor.save_custom_command(command)
        
        # Update UI
        self._populate_command_table()
        
        # Clear form
        self.query_one("#phrase-input", Input).value = ""
        self.query_one("#action-input", Input).value = ""
        self.query_one("#text-input", Input).value = ""
        self.query_one("#description-input", Input).value = ""
        
        self.app.notify(f"Added custom command: {phrase}", severity="success")
    
    def _test_command(self):
        """Test command recognition."""
        test_input = self.query_one("#test-input", Input).value
        if not test_input:
            return
        
        result = self.command_processor.process_text(test_input)
        
        result_widget = self.query_one("#test-result", Static)
        
        if result:
            action, params = result
            result_text = f"✅ Detected command:\n"
            result_text += f"Action: {action}\n"
            if params:
                result_text += f"Parameters: {params}"
            result_widget.update(result_text)
            result_widget.styles.background = "$success-darken-3"
        else:
            result_widget.update("❌ No command detected")
            result_widget.styles.background = "$error-darken-3"
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submission."""
        if event.input.id == "test-input":
            self._test_command()


class QuickCommandPalette(ModalScreen[Optional[str]]):
    """
    Quick command palette for selecting voice commands during dictation.
    """
    
    DEFAULT_CSS = """
    QuickCommandPalette {
        align: center middle;
    }
    
    .palette-container {
        width: 60;
        max-height: 30;
        background: $panel;
        border: thick $primary;
        padding: 1;
    }
    
    .palette-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    .command-list {
        height: 20;
        overflow-y: auto;
    }
    
    .command-item {
        padding: 0.5 1;
        margin-bottom: 0.5;
    }
    
    .command-item:hover {
        background: $boost;
    }
    
    .search-input {
        width: 100%;
        margin-bottom: 1;
    }
    """
    
    filter_text = reactive("")
    
    def compose(self) -> ComposeResult:
        """Compose quick palette UI."""
        with Container(classes="palette-container"):
            yield Label("Voice Commands", classes="palette-title")
            yield Input(
                placeholder="Search commands...",
                id="search-input",
                classes="search-input"
            )
            with ScrollableContainer(classes="command-list", id="command-list"):
                # Commands will be added dynamically
                pass
    
    def on_mount(self):
        """Initialize palette."""
        self._populate_commands()
        self.query_one("#search-input", Input).focus()
    
    def _populate_commands(self):
        """Populate command list."""
        container = self.query_one("#command-list", ScrollableContainer)
        container.remove_children()
        
        processor = get_command_processor()
        filter_lower = self.filter_text.lower()
        
        for command in processor.get_command_list():
            if command.enabled and (
                not filter_lower or 
                filter_lower in command.phrase.lower() or
                (command.description and filter_lower in command.description.lower())
            ):
                item = Button(
                    f"{command.phrase}\n{command.description or ''}",
                    id=f"cmd-{command.phrase}",
                    classes="command-item"
                )
                container.mount(item)
    
    def watch_filter_text(self, old_value: str, new_value: str):
        """Update command list when filter changes."""
        self._populate_commands()
    
    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle search input."""
        if event.input.id == "search-input":
            self.filter_text = event.value
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle command selection."""
        if event.button.id.startswith("cmd-"):
            command_phrase = event.button.id[4:]  # Remove "cmd-" prefix
            self.dismiss(command_phrase)