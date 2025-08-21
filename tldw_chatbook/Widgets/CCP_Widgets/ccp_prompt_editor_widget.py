"""Prompt editor widget for the CCP screen.

This widget provides a comprehensive form for editing prompts,
following Textual best practices with focused components.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List
from loguru import logger
from textual.app import ComposeResult
from textual.containers import Container, VerticalScroll, Horizontal
from textual.widgets import Static, Label, Input, TextArea, Button, Select, Switch
from textual.reactive import reactive
from textual import on
from textual.message import Message
from textual.validation import Length

if TYPE_CHECKING:
    from ...UI.Screens.ccp_screen import CCPScreen, CCPScreenState

logger = logger.bind(module="CCPPromptEditorWidget")


# ========== Messages ==========

class PromptEditorMessage(Message):
    """Base message for prompt editor events."""
    pass


class PromptSaveRequested(PromptEditorMessage):
    """User requested to save the prompt."""
    def __init__(self, prompt_data: Dict[str, Any]) -> None:
        super().__init__()
        self.prompt_data = prompt_data


class PromptDeleteRequested(PromptEditorMessage):
    """User requested to delete the prompt."""
    def __init__(self, prompt_id: int) -> None:
        super().__init__()
        self.prompt_id = prompt_id


class PromptTestRequested(PromptEditorMessage):
    """User requested to test the prompt."""
    def __init__(self, prompt_data: Dict[str, Any]) -> None:
        super().__init__()
        self.prompt_data = prompt_data


class PromptEditorCancelled(PromptEditorMessage):
    """User cancelled prompt editing."""
    pass


class PromptVariableAdded(PromptEditorMessage):
    """User added a variable to the prompt."""
    def __init__(self, variable_name: str, variable_type: str = "text") -> None:
        super().__init__()
        self.variable_name = variable_name
        self.variable_type = variable_type


class PromptVariableRemoved(PromptEditorMessage):
    """User removed a variable from the prompt."""
    def __init__(self, variable_name: str) -> None:
        super().__init__()
        self.variable_name = variable_name


# ========== Prompt Editor Widget ==========

class CCPPromptEditorWidget(Container):
    """
    Prompt editor widget for the CCP screen.
    
    This widget provides a comprehensive editing form for prompts,
    including variables, categories, and testing capabilities.
    """
    
    DEFAULT_CSS = """
    CCPPromptEditorWidget {
        width: 100%;
        height: 100%;
        overflow-y: auto;
        overflow-x: hidden;
    }
    
    CCPPromptEditorWidget.hidden {
        display: none !important;
    }
    
    .prompt-editor-header {
        width: 100%;
        height: 3;
        background: $primary-background-darken-1;
        padding: 0 1;
        margin-bottom: 1;
        text-align: center;
        text-style: bold;
    }
    
    .prompt-editor-content {
        width: 100%;
        height: 1fr;
        overflow-y: auto;
        padding: 1;
    }
    
    .prompt-section {
        width: 100%;
        margin-bottom: 2;
        padding: 1;
        border: round $surface;
        background: $surface-darken-1;
    }
    
    .section-title {
        margin-bottom: 1;
        text-style: bold;
        color: $primary;
    }
    
    .field-container {
        width: 100%;
        margin-bottom: 1;
    }
    
    .field-label {
        margin-bottom: 0;
        color: $text-muted;
    }
    
    .field-input {
        width: 100%;
        margin-top: 0;
    }
    
    .prompt-textarea {
        width: 100%;
        height: 15;
        margin-top: 0;
        border: round $surface;
        background: $surface;
    }
    
    .prompt-textarea.large {
        height: 20;
    }
    
    .variables-container {
        width: 100%;
        padding: 1;
        background: $surface;
        border: round $surface-darken-1;
    }
    
    .variable-item {
        layout: horizontal;
        width: 100%;
        height: 3;
        margin-bottom: 1;
        padding: 0 1;
        background: $surface-lighten-1;
        border: round $surface;
        align: left middle;
    }
    
    .variable-name {
        width: 1fr;
        text-style: bold;
    }
    
    .variable-type {
        width: auto;
        margin-right: 2;
        color: $text-muted;
    }
    
    .variable-remove-btn {
        width: auto;
        height: 3;
        background: $error-darken-1;
    }
    
    .variable-remove-btn:hover {
        background: $error;
    }
    
    .add-variable-container {
        layout: horizontal;
        width: 100%;
        height: 3;
        margin-top: 1;
    }
    
    .add-variable-input {
        width: 1fr;
        margin-right: 1;
    }
    
    .add-variable-type {
        width: 10;
        margin-right: 1;
    }
    
    .add-variable-btn {
        width: auto;
        padding: 0 1;
    }
    
    .category-select {
        width: 100%;
        height: 3;
    }
    
    .test-section {
        width: 100%;
        padding: 1;
        border: round $secondary;
        background: $secondary-darken-2;
    }
    
    .test-input-container {
        width: 100%;
        margin-bottom: 1;
    }
    
    .test-result {
        width: 100%;
        height: 10;
        padding: 1;
        background: $surface;
        border: round $surface-lighten-1;
        overflow-y: auto;
    }
    
    .test-button {
        width: 100%;
        height: 3;
        margin-bottom: 1;
        background: $secondary;
    }
    
    .test-button:hover {
        background: $secondary-lighten-1;
    }
    
    .prompt-preview {
        width: 100%;
        height: 10;
        padding: 1;
        background: $surface-darken-2;
        border: round $surface-darken-1;
        overflow-y: auto;
        margin-top: 1;
    }
    
    .system-prompt-toggle {
        layout: horizontal;
        height: 3;
        width: 100%;
        align: left middle;
        margin-bottom: 1;
    }
    
    .toggle-label {
        width: auto;
        margin-right: 2;
    }
    
    .prompt-actions {
        layout: horizontal;
        height: 3;
        width: 100%;
        padding: 1;
        background: $surface;
        border-top: thick $background-darken-1;
    }
    
    .prompt-action-button {
        width: 1fr;
        height: 3;
        margin-right: 1;
    }
    
    .prompt-action-button:last-child {
        margin-right: 0;
    }
    
    .prompt-action-button.primary {
        background: $success;
    }
    
    .prompt-action-button.primary:hover {
        background: $success-lighten-1;
    }
    
    .prompt-action-button.danger {
        background: $error-darken-1;
    }
    
    .prompt-action-button.danger:hover {
        background: $error;
    }
    
    .prompt-action-button.cancel {
        background: $warning-darken-1;
    }
    
    .prompt-action-button.cancel:hover {
        background: $warning;
    }
    
    .no-prompt-message {
        width: 100%;
        height: 100%;
        align: center middle;
        text-align: center;
        color: $text-muted;
        padding: 2;
    }
    """
    
    # Reactive state reference (will be linked to parent screen's state)
    state: reactive[Optional['CCPScreenState']] = reactive(None)
    
    # Current prompt data being edited
    prompt_data: reactive[Dict[str, Any]] = reactive({})
    
    # Variables list
    variables: reactive[List[Dict[str, str]]] = reactive([])
    
    # Is system prompt
    is_system_prompt: reactive[bool] = reactive(False)
    
    # Test results
    test_result: reactive[str] = reactive("")
    
    # Available categories
    CATEGORIES = [
        ("general", "General"),
        ("creative", "Creative Writing"),
        ("technical", "Technical"),
        ("analysis", "Analysis"),
        ("translation", "Translation"),
        ("summarization", "Summarization"),
        ("conversation", "Conversation"),
        ("roleplay", "Roleplay"),
        ("custom", "Custom"),
    ]
    
    def __init__(self, parent_screen: Optional['CCPScreen'] = None, **kwargs):
        """Initialize the prompt editor widget.
        
        Args:
            parent_screen: Reference to the parent CCP screen
            **kwargs: Additional arguments for Container
        """
        super().__init__(id="ccp-prompt-editor-view", classes="ccp-view-area hidden", **kwargs)
        self.parent_screen = parent_screen
        
        # Field references for quick access
        self._name_input: Optional[Input] = None
        self._prompt_area: Optional[TextArea] = None
        self._description_area: Optional[TextArea] = None
        self._category_select: Optional[Select] = None
        self._system_toggle: Optional[Switch] = None
        self._preview_area: Optional[Static] = None
        self._test_result_area: Optional[Static] = None
        
        logger.debug("CCPPromptEditorWidget initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the prompt editor UI."""
        # Header
        yield Static("Prompt Editor", classes="prompt-editor-header pane-title")
        
        # Content container with scroll
        with VerticalScroll(classes="prompt-editor-content"):
            # No prompt placeholder (shown when no prompt is loaded)
            yield Static(
                "No prompt loaded.\nSelect a prompt from the sidebar or create a new one.",
                classes="no-prompt-message",
                id="no-prompt-placeholder"
            )
            
            # Editor container (hidden by default)
            with Container(id="prompt-editor-container", classes="hidden"):
                # Basic Information Section
                with Container(classes="prompt-section"):
                    yield Static("Basic Information", classes="section-title")
                    
                    # Name field
                    with Container(classes="field-container"):
                        yield Label("Prompt Name:", classes="field-label")
                        yield Input(
                            placeholder="Enter prompt name",
                            id="ccp-prompt-name",
                            classes="field-input",
                            validators=[Length(1, 100)]
                        )
                    
                    # Category selection
                    with Container(classes="field-container"):
                        yield Label("Category:", classes="field-label")
                        yield Select(
                            options=[(value, label) for value, label in self.CATEGORIES],
                            id="ccp-prompt-category",
                            classes="category-select",
                            value="general"
                        )
                    
                    # System prompt toggle
                    with Container(classes="system-prompt-toggle"):
                        yield Label("System Prompt:", classes="toggle-label")
                        yield Switch(id="ccp-prompt-system-toggle", value=False)
                    
                    # Description field
                    with Container(classes="field-container"):
                        yield Label("Description:", classes="field-label")
                        yield TextArea(
                            "",
                            id="ccp-prompt-description",
                            classes="field-textarea small"
                        )
                
                # Prompt Content Section
                with Container(classes="prompt-section"):
                    yield Static("Prompt Content", classes="section-title")
                    
                    # Main prompt text
                    with Container(classes="field-container"):
                        yield Label("Prompt Text (use {{variable}} for variables):", classes="field-label")
                        yield TextArea(
                            "",
                            id="ccp-prompt-content",
                            classes="prompt-textarea large"
                        )
                    
                    # Preview
                    yield Label("Preview:", classes="field-label")
                    with Container(classes="prompt-preview", id="ccp-prompt-preview"):
                        yield Static("Enter prompt text to see preview")
                
                # Variables Section
                with Container(classes="prompt-section"):
                    yield Static("Variables", classes="section-title")
                    
                    # Variables list
                    with Container(classes="variables-container", id="ccp-variables-list"):
                        yield Static("No variables defined", classes="no-variables-placeholder")
                    
                    # Add variable controls
                    with Container(classes="add-variable-container"):
                        yield Input(
                            placeholder="Variable name",
                            id="ccp-variable-name-input",
                            classes="add-variable-input"
                        )
                        yield Select(
                            options=[
                                ("text", "Text"),
                                ("number", "Number"),
                                ("boolean", "Boolean"),
                                ("list", "List"),
                            ],
                            id="ccp-variable-type-select",
                            classes="add-variable-type",
                            value="text"
                        )
                        yield Button("Add Variable", id="add-variable-btn", classes="add-variable-btn")
                
                # Test Section
                with Container(classes="prompt-section test-section"):
                    yield Static("Test Prompt", classes="section-title")
                    
                    # Test inputs container (will be populated based on variables)
                    with Container(id="ccp-test-inputs-container", classes="test-input-container"):
                        yield Static("Define variables first to test the prompt")
                    
                    # Test button
                    yield Button("Test Prompt", id="test-prompt-btn", classes="test-button")
                    
                    # Test result
                    yield Label("Result:", classes="field-label")
                    with Container(classes="test-result", id="ccp-test-result"):
                        yield Static("Test result will appear here")
        
        # Action buttons
        with Container(classes="prompt-actions"):
            yield Button("Save Prompt", classes="prompt-action-button primary", id="save-prompt-btn")
            yield Button("Delete", classes="prompt-action-button danger", id="delete-prompt-btn")
            yield Button("Reset", classes="prompt-action-button", id="reset-prompt-btn")
            yield Button("Cancel", classes="prompt-action-button cancel", id="cancel-prompt-btn")
    
    async def on_mount(self) -> None:
        """Handle widget mount."""
        # Cache field references
        self._cache_field_references()
        
        # Link to parent screen's state if available
        if self.parent_screen and hasattr(self.parent_screen, 'state'):
            self.state = self.parent_screen.state
        
        logger.debug("CCPPromptEditorWidget mounted")
    
    def _cache_field_references(self) -> None:
        """Cache references to frequently used fields."""
        try:
            self._name_input = self.query_one("#ccp-prompt-name", Input)
            self._prompt_area = self.query_one("#ccp-prompt-content", TextArea)
            self._description_area = self.query_one("#ccp-prompt-description", TextArea)
            self._category_select = self.query_one("#ccp-prompt-category", Select)
            self._system_toggle = self.query_one("#ccp-prompt-system-toggle", Switch)
            self._preview_area = self.query_one("#ccp-prompt-preview", Container)
            self._test_result_area = self.query_one("#ccp-test-result", Container)
        except Exception as e:
            logger.warning(f"Could not cache all field references: {e}")
    
    # ===== Public Methods =====
    
    def load_prompt(self, prompt_data: Dict[str, Any]) -> None:
        """Load prompt data into the editor.
        
        Args:
            prompt_data: Dictionary containing prompt information
        """
        self.prompt_data = prompt_data.copy()
        
        # Hide placeholder, show editor
        try:
            placeholder = self.query_one("#no-prompt-placeholder")
            placeholder.add_class("hidden")
            
            editor = self.query_one("#prompt-editor-container")
            editor.remove_class("hidden")
        except:
            pass
        
        # Load basic fields
        if self._name_input:
            self._name_input.value = prompt_data.get('name', '')
        if self._prompt_area:
            self._prompt_area.text = prompt_data.get('content', prompt_data.get('prompt', ''))
        if self._description_area:
            self._description_area.text = prompt_data.get('description', '')
        if self._category_select:
            self._category_select.value = prompt_data.get('category', 'general')
        if self._system_toggle:
            self._system_toggle.value = prompt_data.get('is_system', False)
        
        # Load variables
        self.variables = prompt_data.get('variables', []).copy()
        self._update_variables_display()
        
        # Update preview
        self._update_preview()
        
        # Setup test inputs
        self._setup_test_inputs()
        
        logger.info(f"Loaded prompt for editing: {prompt_data.get('name', 'Unknown')}")
    
    def new_prompt(self) -> None:
        """Initialize the editor for a new prompt."""
        self.prompt_data = {}
        self.variables = []
        self.is_system_prompt = False
        self.test_result = ""
        
        # Hide placeholder, show editor
        try:
            placeholder = self.query_one("#no-prompt-placeholder")
            placeholder.add_class("hidden")
            
            editor = self.query_one("#prompt-editor-container")
            editor.remove_class("hidden")
        except:
            pass
        
        # Clear all fields
        if self._name_input:
            self._name_input.value = ""
        if self._prompt_area:
            self._prompt_area.text = ""
        if self._description_area:
            self._description_area.text = ""
        if self._category_select:
            self._category_select.value = "general"
        if self._system_toggle:
            self._system_toggle.value = False
        
        self._update_variables_display()
        self._update_preview()
        self._setup_test_inputs()
        
        logger.info("Initialized editor for new prompt")
    
    def get_prompt_data(self) -> Dict[str, Any]:
        """Get the current prompt data from the editor.
        
        Returns:
            Dictionary containing all prompt data
        """
        data = self.prompt_data.copy()
        
        # Update with current field values
        if self._name_input:
            data['name'] = self._name_input.value
        if self._prompt_area:
            data['content'] = self._prompt_area.text
        if self._description_area:
            data['description'] = self._description_area.text
        if self._category_select:
            data['category'] = self._category_select.value
        if self._system_toggle:
            data['is_system'] = self._system_toggle.value
        
        # Add variables
        data['variables'] = self.variables.copy()
        
        return data
    
    # ===== Private Helper Methods =====
    
    def _update_variables_display(self) -> None:
        """Update the variables display."""
        try:
            container = self.query_one("#ccp-variables-list")
            container.remove_children()
            
            if self.variables:
                for i, var in enumerate(self.variables):
                    var_container = Container(classes="variable-item")
                    
                    name_widget = Static(f"{{{{ {var['name']} }}}}", classes="variable-name")
                    var_container.mount(name_widget)
                    
                    type_widget = Static(f"({var.get('type', 'text')})", classes="variable-type")
                    var_container.mount(type_widget)
                    
                    remove_btn = Button("Remove", classes="variable-remove-btn", 
                                      id=f"remove-var-{i}")
                    var_container.mount(remove_btn)
                    
                    container.mount(var_container)
            else:
                placeholder = Static("No variables defined", classes="no-variables-placeholder")
                container.mount(placeholder)
        except Exception as e:
            logger.warning(f"Could not update variables display: {e}")
    
    def _update_preview(self) -> None:
        """Update the prompt preview."""
        if not self._prompt_area or not self._preview_area:
            return
        
        try:
            preview_container = self._preview_area
            preview_container.remove_children()
            
            prompt_text = self._prompt_area.text
            if prompt_text:
                # Highlight variables in preview
                for var in self.variables:
                    var_placeholder = f"{{{{{var['name']}}}}}"
                    prompt_text = prompt_text.replace(
                        var_placeholder,
                        f"[bold cyan]{var_placeholder}[/bold cyan]"
                    )
                
                preview_widget = Static(prompt_text)
                preview_container.mount(preview_widget)
            else:
                placeholder = Static("Enter prompt text to see preview")
                preview_container.mount(placeholder)
        except Exception as e:
            logger.warning(f"Could not update preview: {e}")
    
    def _setup_test_inputs(self) -> None:
        """Setup test input fields based on variables."""
        try:
            container = self.query_one("#ccp-test-inputs-container")
            container.remove_children()
            
            if self.variables:
                for var in self.variables:
                    # Create input for each variable
                    label = Label(f"{var['name']}:", classes="field-label")
                    container.mount(label)
                    
                    input_widget = Input(
                        placeholder=f"Enter {var.get('type', 'text')} value",
                        id=f"test-var-{var['name']}",
                        classes="field-input"
                    )
                    container.mount(input_widget)
            else:
                placeholder = Static("Define variables first to test the prompt")
                container.mount(placeholder)
        except Exception as e:
            logger.warning(f"Could not setup test inputs: {e}")
    
    # ===== Event Handlers =====
    
    @on(Button.Pressed, "#save-prompt-btn")
    async def handle_save_prompt(self, event: Button.Pressed) -> None:
        """Handle save prompt button press."""
        event.stop()
        prompt_data = self.get_prompt_data()
        
        # Validate required fields
        if not prompt_data.get('name'):
            logger.warning("Cannot save prompt without name")
            return
        
        self.post_message(PromptSaveRequested(prompt_data))
    
    @on(Button.Pressed, "#delete-prompt-btn")
    async def handle_delete_prompt(self, event: Button.Pressed) -> None:
        """Handle delete prompt button press."""
        event.stop()
        if self.prompt_data and 'id' in self.prompt_data:
            self.post_message(PromptDeleteRequested(self.prompt_data['id']))
    
    @on(Button.Pressed, "#reset-prompt-btn")
    async def handle_reset_prompt(self, event: Button.Pressed) -> None:
        """Handle reset prompt button press."""
        event.stop()
        if self.prompt_data:
            self.load_prompt(self.prompt_data)
        else:
            self.new_prompt()
    
    @on(Button.Pressed, "#cancel-prompt-btn")
    async def handle_cancel_edit(self, event: Button.Pressed) -> None:
        """Handle cancel edit button press."""
        event.stop()
        self.post_message(PromptEditorCancelled())
    
    @on(Button.Pressed, "#add-variable-btn")
    async def handle_add_variable(self, event: Button.Pressed) -> None:
        """Handle add variable button press."""
        event.stop()
        
        try:
            name_input = self.query_one("#ccp-variable-name-input", Input)
            type_select = self.query_one("#ccp-variable-type-select", Select)
            
            if name_input.value.strip():
                var_name = name_input.value.strip()
                var_type = type_select.value
                
                # Check for duplicates
                if not any(v['name'] == var_name for v in self.variables):
                    self.variables.append({'name': var_name, 'type': var_type})
                    name_input.value = ""
                    self._update_variables_display()
                    self._update_preview()
                    self._setup_test_inputs()
                    self.post_message(PromptVariableAdded(var_name, var_type))
        except Exception as e:
            logger.warning(f"Could not add variable: {e}")
    
    @on(Button.Pressed, "[id^='remove-var-']")
    async def handle_remove_variable(self, event: Button.Pressed) -> None:
        """Handle remove variable button press."""
        event.stop()
        if event.button.id:
            index = int(event.button.id.replace("remove-var-", ""))
            if 0 <= index < len(self.variables):
                var_name = self.variables[index]['name']
                del self.variables[index]
                self._update_variables_display()
                self._update_preview()
                self._setup_test_inputs()
                self.post_message(PromptVariableRemoved(var_name))
    
    @on(Button.Pressed, "#test-prompt-btn")
    async def handle_test_prompt(self, event: Button.Pressed) -> None:
        """Handle test prompt button press."""
        event.stop()
        
        # Gather test values
        test_values = {}
        for var in self.variables:
            try:
                input_widget = self.query_one(f"#test-var-{var['name']}", Input)
                test_values[var['name']] = input_widget.value
            except:
                test_values[var['name']] = ""
        
        # Create test data
        prompt_data = self.get_prompt_data()
        prompt_data['test_values'] = test_values
        
        self.post_message(PromptTestRequested(prompt_data))
    
    @on(TextArea.Changed, "#ccp-prompt-content")
    async def handle_prompt_content_changed(self, event: TextArea.Changed) -> None:
        """Handle prompt content changes."""
        self._update_preview()
    
    @on(Switch.Changed, "#ccp-prompt-system-toggle")
    async def handle_system_toggle(self, event: Switch.Changed) -> None:
        """Handle system prompt toggle."""
        self.is_system_prompt = event.value