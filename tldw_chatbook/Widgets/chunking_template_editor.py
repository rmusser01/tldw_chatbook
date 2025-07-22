"""
Modal screen for creating and editing chunking templates.
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, Callable, List, Tuple
import json
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, VerticalScroll, Grid
from textual.widgets import (
    Static, Button, Label, Input, TextArea, Select, 
    Checkbox, Collapsible, TabbedContent, TabPane, DataTable
)
from textual.screen import ModalScreen
from textual.validation import ValidationResult, Validator
from loguru import logger

from ..Widgets.form_components import create_form_field, create_button_group
from ..Chunking.chunking_interop_library import (
    ChunkingInteropService, 
    get_chunking_service,
    ChunkingTemplateError,
    TemplateNotFoundError,
    SystemTemplateError
)
from ..DB.Client_Media_DB_v2 import InputError

if TYPE_CHECKING:
    from ..app import TldwCli


class JSONValidator(Validator):
    """Validator for JSON strings."""
    
    def validate(self, value: str) -> ValidationResult:
        """Validate that the input is valid JSON."""
        if not value.strip():
            return ValidationResult.failure("JSON cannot be empty")
        
        try:
            json.loads(value)
            return ValidationResult.success()
        except json.JSONDecodeError as e:
            return ValidationResult.failure(f"Invalid JSON: {str(e)}")


class ChunkingTemplateEditor(ModalScreen):
    """Modal for creating and editing chunking templates."""
    
    DEFAULT_CSS = """
    ChunkingTemplateEditor {
        align: center middle;
    }
    
    #template-editor-container {
        width: 90%;
        height: 90%;
        max-width: 120;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    #editor-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    
    .form-section {
        margin: 1 0;
    }
    
    .pipeline-stage {
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
    }
    
    .stage-actions {
        align: right middle;
        height: 3;
    }
    
    .editor-actions {
        align: center middle;
        height: 3;
        margin-top: 1;
    }
    
    #json-editor {
        height: 20;
        margin: 1 0;
    }
    
    .preview-container {
        height: 15;
        border: solid $secondary;
        padding: 1;
        margin: 1 0;
    }
    """
    
    def __init__(
        self,
        app_instance: 'TldwCli',
        mode: str = "create",
        template_data: Optional[Dict[str, Any]] = None,
        on_save: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize the template editor.
        
        Args:
            app_instance: Reference to the main app instance
            mode: "create" or "edit"
            template_data: Existing template data for editing
            on_save: Callback to call after successful save
        """
        super().__init__(**kwargs)
        self.app_instance = app_instance
        self.mode = mode
        self.template_data = template_data or {}
        self.on_save_callback = on_save
        self.pipeline_stages = []
        self.chunking_service = None
        
        # Parse existing pipeline if editing
        if template_data and 'template_json' in template_data:
            try:
                template_obj = json.loads(template_data['template_json'])
                self.pipeline_stages = template_obj.get('pipeline', [])
            except:
                self.pipeline_stages = []
    
    def compose(self) -> ComposeResult:
        """Compose the editor UI."""
        with Container(id="template-editor-container"):
            # Title
            title = "Create New Template" if self.mode == "create" else "Edit Template"
            yield Label(title, id="editor-title")
            
            with TabbedContent():
                # Basic Info Tab
                with TabPane("Basic Info", id="basic-info-tab"):
                    with VerticalScroll():
                        yield from self._compose_basic_info()
                
                # Pipeline Builder Tab
                with TabPane("Pipeline Builder", id="pipeline-tab"):
                    with VerticalScroll():
                        yield from self._compose_pipeline_builder()
                
                # JSON Editor Tab
                with TabPane("JSON Editor", id="json-tab"):
                    with VerticalScroll():
                        yield from self._compose_json_editor()
                
                # Preview Tab
                with TabPane("Preview", id="preview-tab"):
                    with VerticalScroll():
                        yield from self._compose_preview()
            
            # Action buttons
            with Horizontal(classes="editor-actions"):
                yield Button("Save", id="save-template-btn", variant="primary")
                yield Button("Cancel", id="cancel-btn", variant="default")
                yield Button("Validate", id="validate-btn", variant="default")
    
    def _compose_basic_info(self) -> ComposeResult:
        """Compose the basic info section."""
        # Template name
        yield from create_form_field(
            "Template Name",
            "template-name",
            "input",
            placeholder="e.g., Academic Paper Chunking",
            default_value=self.template_data.get('name', ''),
            required=True
        )
        
        # Description
        yield from create_form_field(
            "Description",
            "template-description",
            "textarea",
            placeholder="Describe what this template is designed for...",
            default_value=self.template_data.get('description', ''),
            required=True
        )
        
        # Base method
        template_obj = {}
        if self.template_data and 'template_json' in self.template_data:
            try:
                template_obj = json.loads(self.template_data['template_json'])
            except:
                pass
        
        yield from create_form_field(
            "Base Chunking Method",
            "base-method",
            "select",
            options=[
                ("Words", "words"),
                ("Sentences", "sentences"),
                ("Paragraphs", "paragraphs"),
                ("Hierarchical", "hierarchical"),
                ("Structural", "structural"),
                ("Contextual", "contextual")
            ],
            default_value=template_obj.get('base_method', 'words')
        )
        
        # Version
        yield from create_form_field(
            "Version",
            "template-version",
            "input",
            placeholder="1.0",
            default_value=template_obj.get('metadata', {}).get('version', '1.0')
        )
    
    def _compose_pipeline_builder(self) -> ComposeResult:
        """Compose the pipeline builder section."""
        yield Label("Pipeline Stages", classes="section-title")
        yield Static(
            "Define the processing pipeline for this template. Stages are executed in order.",
            classes="help-text"
        )
        
        # Stage list container
        yield Container(id="pipeline-stages-container")
        
        # Add stage button
        with Horizontal(classes="stage-actions"):
            yield Button("➕ Add Stage", id="add-stage-btn", variant="primary")
    
    def _compose_json_editor(self) -> ComposeResult:
        """Compose the JSON editor section."""
        yield Label("Template JSON Configuration", classes="section-title")
        yield Static(
            "Advanced users can edit the template JSON directly.",
            classes="help-text"
        )
        
        # Generate current JSON
        current_json = self._generate_template_json()
        
        yield TextArea(
            json.dumps(current_json, indent=2),
            id="json-editor",
            language="json",
            theme="monokai"
        )
        
        # Validation result
        yield Static("", id="json-validation-result", classes="validation-result")
    
    def _compose_preview(self) -> ComposeResult:
        """Compose the preview section."""
        yield Label("Template Preview", classes="section-title")
        
        # Sample text input
        yield from create_form_field(
            "Sample Text",
            "preview-sample-text",
            "textarea",
            placeholder="Enter sample text to preview chunking...",
            default_value="This is a sample paragraph. It contains multiple sentences. " * 5
        )
        
        # Preview button
        yield Button("Generate Preview", id="generate-preview-btn", variant="primary")
        
        # Preview results
        with Container(classes="preview-container"):
            yield Label("Chunk Preview Results", classes="preview-title")
            yield DataTable(id="preview-results-table")
    
    def on_mount(self) -> None:
        """Initialize the editor when mounted."""
        # Initialize chunking service
        if hasattr(self.app_instance, 'media_db') and self.app_instance.media_db:
            self.chunking_service = get_chunking_service(self.app_instance.media_db)
        else:
            logger.warning("Media database not available for chunking service")
        
        # Set up preview table
        table = self.query_one("#preview-results-table", DataTable)
        table.add_columns("Chunk #", "Text Preview", "Words", "Type")
        
        # Populate pipeline stages if editing
        if self.pipeline_stages:
            self._populate_pipeline_stages()
    
    def _populate_pipeline_stages(self) -> None:
        """Populate the pipeline stages from existing data."""
        container = self.query_one("#pipeline-stages-container", Container)
        
        for i, stage in enumerate(self.pipeline_stages):
            stage_widget = self._create_stage_widget(i, stage)
            container.mount(stage_widget)
    
    def _create_stage_widget(self, index: int, stage_data: Dict[str, Any]) -> Container:
        """Create a widget for a pipeline stage."""
        stage_container = Container(classes="pipeline-stage", id=f"stage-{index}")
        
        with stage_container:
            # Stage header
            with Horizontal():
                Label(f"Stage {index + 1}: {stage_data.get('stage', 'Unknown').title()}", classes="stage-title")
                Button("❌", id=f"remove-stage-{index}", classes="remove-stage-btn")
            
            # Stage type
            with Horizontal():
                Label("Stage Type:", classes="form-label")
                stage_select = Select(
                    [
                        ("Preprocess", "preprocess"),
                        ("Chunk", "chunk"),
                        ("Postprocess", "postprocess")
                    ],
                    id=f"stage-type-{index}",
                    value=stage_data.get('stage', 'chunk')
                )
                yield stage_select
            
            # Method (for chunk stage)
            if stage_data.get('stage') == 'chunk':
                with Horizontal():
                    Label("Method:", classes="form-label")
                    method_select = Select(
                        [
                            ("Words", "words"),
                            ("Sentences", "sentences"),
                            ("Paragraphs", "paragraphs"),
                            ("Hierarchical", "hierarchical"),
                            ("Structural", "structural"),
                            ("Contextual", "contextual")
                        ],
                        id=f"stage-method-{index}",
                        value=stage_data.get('method', 'words')
                    )
                    yield method_select
            
            # Options
            options = stage_data.get('options', {})
            if options:
                Label("Options:", classes="form-label")
                options_json = json.dumps(options, indent=2)
                TextArea(
                    options_json,
                    id=f"stage-options-{index}",
                    classes="stage-options"
                )
        
        return stage_container
    
    def _generate_template_json(self) -> Dict[str, Any]:
        """Generate the current template JSON from form fields."""
        try:
            # Get basic info
            name = self.query_one("#template-name", Input).value
            description = self.query_one("#template-description", TextArea).text
            base_method = self.query_one("#base-method", Select).value
            version = self.query_one("#template-version", Input).value
            
            # Build pipeline from stages
            pipeline = []
            stages_container = self.query_one("#pipeline-stages-container", Container)
            
            for i, stage_widget in enumerate(stages_container.children):
                if not isinstance(stage_widget, Container):
                    continue
                
                stage_type = stage_widget.query_one(f"#stage-type-{i}", Select).value
                stage_data = {"stage": stage_type}
                
                # Add method for chunk stages
                if stage_type == "chunk":
                    try:
                        method_select = stage_widget.query_one(f"#stage-method-{i}", Select)
                        stage_data["method"] = method_select.value
                    except:
                        stage_data["method"] = base_method
                
                # Add options if present
                try:
                    options_area = stage_widget.query_one(f"#stage-options-{i}", TextArea)
                    if options_area.text.strip():
                        stage_data["options"] = json.loads(options_area.text)
                except:
                    pass
                
                pipeline.append(stage_data)
            
            # If no pipeline stages, add default chunk stage
            if not pipeline:
                pipeline = [{
                    "stage": "chunk",
                    "method": base_method,
                    "options": {
                        "max_size": 400,
                        "overlap": 100
                    }
                }]
            
            return {
                "name": name,
                "description": description,
                "base_method": base_method,
                "pipeline": pipeline,
                "metadata": {
                    "version": version or "1.0"
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating template JSON: {e}")
            return {}
    
    @on(Button.Pressed, "#add-stage-btn")
    def add_pipeline_stage(self) -> None:
        """Add a new pipeline stage."""
        container = self.query_one("#pipeline-stages-container", Container)
        index = len(container.children)
        
        # Create default stage
        stage_data = {
            "stage": "chunk",
            "method": "words",
            "options": {
                "max_size": 400,
                "overlap": 100
            }
        }
        
        stage_widget = self._create_stage_widget(index, stage_data)
        container.mount(stage_widget)
    
    @on(Button.Pressed)
    def handle_remove_stage(self, event: Button.Pressed) -> None:
        """Handle removing a pipeline stage."""
        if event.button.id and event.button.id.startswith("remove-stage-"):
            stage_container = event.button.parent.parent
            stage_container.remove()
            
            # Renumber remaining stages
            self._renumber_stages()
    
    def _renumber_stages(self) -> None:
        """Renumber pipeline stages after removal."""
        container = self.query_one("#pipeline-stages-container", Container)
        
        for i, stage_widget in enumerate(container.children):
            if isinstance(stage_widget, Container):
                # Update stage ID
                stage_widget.id = f"stage-{i}"
                
                # Update stage title
                try:
                    title_label = stage_widget.query(".stage-title")[0]
                    stage_type = stage_widget.query_one(f"#stage-type-{i}", Select).value
                    title_label.update(f"Stage {i + 1}: {stage_type.title()}")
                except:
                    pass
    
    @on(Button.Pressed, "#save-template-btn")
    def save_template(self) -> None:
        """Save the template."""
        if not self.chunking_service:
            self.app_instance.notify("Chunking service not available", severity="error")
            return
        
        # Validate required fields
        name = self.query_one("#template-name", Input).value.strip()
        description = self.query_one("#template-description", TextArea).text.strip()
        
        if not name:
            self.app_instance.notify("Template name is required", severity="warning")
            return
        
        if not description:
            self.app_instance.notify("Template description is required", severity="warning")
            return
        
        # Generate template JSON
        template_json = self._generate_template_json()
        
        try:
            if self.mode == "create":
                # Create new template using the service
                template_id = self.chunking_service.create_template(
                    name=name,
                    description=description,
                    template_json=template_json,
                    is_system=False
                )
                logger.info(f"Created template '{name}' with ID {template_id}")
                
            else:  # edit mode
                # Update existing template
                template_id = self.template_data.get('id')
                if not template_id:
                    raise ValueError("No template ID for update")
                
                self.chunking_service.update_template(
                    template_id=template_id,
                    name=name,
                    description=description,
                    template_json=template_json
                )
                logger.info(f"Updated template '{name}' (ID: {template_id})")
            
            self.app_instance.notify(
                f"Template '{name}' saved successfully",
                severity="information"
            )
            
            # Call the callback if provided
            if self.on_save_callback:
                self.on_save_callback()
            
            # Close the modal
            self.dismiss()
            
        except InputError as e:
            self.app_instance.notify(str(e), severity="warning")
        except SystemTemplateError as e:
            self.app_instance.notify(str(e), severity="warning")
        except ChunkingTemplateError as e:
            logger.error(f"Error saving template: {e}")
            self.app_instance.notify(
                f"Error saving template: {str(e)}",
                severity="error"
            )
        except Exception as e:
            logger.error(f"Unexpected error saving template: {e}")
            self.app_instance.notify(
                f"Unexpected error: {str(e)}",
                severity="error"
            )
    
    @on(Button.Pressed, "#cancel-btn")
    def cancel_edit(self) -> None:
        """Cancel editing and close the modal."""
        self.dismiss()
    
    @on(Button.Pressed, "#validate-btn")
    def validate_template(self) -> None:
        """Validate the current template configuration."""
        try:
            # Try to generate JSON
            template_json = self._generate_template_json()
            
            # Validate structure
            if not template_json.get('name'):
                raise ValueError("Template name is required")
            
            if not template_json.get('pipeline'):
                raise ValueError("At least one pipeline stage is required")
            
            # Check JSON editor if on that tab
            try:
                json_editor = self.query_one("#json-editor", TextArea)
                json.loads(json_editor.text)
            except:
                pass
            
            self.app_instance.notify(
                "Template configuration is valid",
                severity="information"
            )
            
        except Exception as e:
            self.app_instance.notify(
                f"Validation error: {str(e)}",
                severity="error"
            )
    
    @on(Button.Pressed, "#generate-preview-btn")
    def generate_preview(self) -> None:
        """Generate a preview of the chunking results."""
        sample_text = self.query_one("#preview-sample-text", TextArea).text
        
        if not sample_text.strip():
            self.app_instance.notify("Please enter sample text for preview", severity="warning")
            return
        
        # Generate template JSON
        template_json = self._generate_template_json()
        
        # Use the chunking service to generate preview
        self._generate_preview_chunks(sample_text, template_json)
    
    @work(thread=True)
    def _generate_preview_chunks(self, text: str, template_config: Dict[str, Any]) -> None:
        """Generate preview chunks in a worker thread."""
        try:
            from ..Chunking.Chunk_Lib import Chunker
            
            # Extract base method and options
            base_method = template_config.get('base_method', 'words')
            pipeline = template_config.get('pipeline', [])
            
            # Find chunk stage options
            chunk_options = {
                'max_size': 400,
                'overlap': 100
            }
            
            for stage in pipeline:
                if stage.get('stage') == 'chunk':
                    chunk_options.update(stage.get('options', {}))
                    base_method = stage.get('method', base_method)
                    break
            
            # Create chunker and generate chunks
            chunker = Chunker()
            chunker.options.update(chunk_options)
            
            chunks = chunker.chunk_text(text, method=base_method)
            
            # Update preview table
            self.call_from_thread(self._update_preview_table, chunks)
            
        except Exception as e:
            logger.error(f"Error generating preview: {e}")
            self.call_from_thread(
                self.app_instance.notify,
                f"Error generating preview: {str(e)}",
                severity="error"
            )
    
    def _update_preview_table(self, chunks: List[Any]) -> None:
        """Update the preview table with generated chunks."""
        table = self.query_one("#preview-results-table", DataTable)
        table.clear()
        
        for i, chunk in enumerate(chunks):
            if isinstance(chunk, dict):
                text = chunk.get('text', str(chunk))
                chunk_type = chunk.get('type', 'text')
            else:
                text = str(chunk)
                chunk_type = 'text'
            
            # Truncate text for display
            preview = text[:80] + "..." if len(text) > 80 else text
            word_count = len(text.split())
            
            table.add_row(
                str(i + 1),
                preview,
                str(word_count),
                chunk_type
            )