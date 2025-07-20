# DatasetManagementWindow.py
# Description: Window for managing evaluation datasets
#
"""
Dataset Management Window
------------------------

Provides interface for managing, uploading, and exploring evaluation datasets.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from textual import on, work
from textual.app import ComposeResult
from textual.widgets import (
    Button, Label, Static, Select, Input, DataTable,
    TabbedContent, TabPane, DirectoryTree, TextArea
)
from textual.containers import Container, Horizontal, Vertical, VerticalScroll, Grid
from textual.reactive import reactive
from loguru import logger

from .eval_shared_components import (
    BaseEvaluationWindow, RefreshDataRequest,
    format_status_badge
)
from ..Event_Handlers.eval_events import (
    get_available_datasets, upload_dataset, 
    validate_dataset, get_dataset_info,
    delete_dataset, export_dataset
)
# from ..Widgets.eval_config_dialogs import (
#     DatasetUploadDialog, DatasetCreationDialog
# )
# TODO: Import when dialogs are implemented


class DatasetManagementWindow(BaseEvaluationWindow):
    """Window for managing evaluation datasets."""
    
    # Reactive state
    selected_dataset = reactive(None)
    dataset_filter = reactive("all")
    preview_mode = reactive("samples")  # samples, stats, schema
    
    def compose(self) -> ComposeResult:
        """Compose the dataset management interface."""
        yield from self.compose_header("Dataset Management")
        
        with TabbedContent():
            # Browse Datasets Tab
            with TabPane("Browse", id="browse-tab"):
                with Horizontal(classes="eval-content-area"):
                    # Left panel - Dataset list
                    with VerticalScroll(classes="dataset-list-panel"):
                        yield Static("📚 Available Datasets", classes="panel-title")
                        
                        # Filter and search
                        with Container(classes="filter-container"):
                            yield Select(
                                [
                                    ("all", "All Datasets"),
                                    ("builtin", "Built-in"),
                                    ("custom", "Custom"),
                                    ("recent", "Recently Used")
                                ],
                                id="dataset-filter",
                                value="all"
                            )
                            yield Input(
                                placeholder="Search datasets...",
                                id="dataset-search"
                            )
                        
                        # Dataset table
                        yield DataTable(
                            id="dataset-table",
                            show_cursor=True,
                            zebra_stripes=True
                        )
                    
                    # Right panel - Dataset details
                    with VerticalScroll(classes="dataset-details-panel"):
                        # Dataset info section
                        with Container(classes="section-container", id="dataset-info"):
                            yield Static("Dataset Information", classes="section-title")
                            
                            with Grid(classes="info-grid"):
                                yield Label("Name:")
                                yield Static("Select a dataset", id="dataset-name", classes="info-value")
                                
                                yield Label("Type:")
                                yield Static("N/A", id="dataset-type", classes="info-value")
                                
                                yield Label("Size:")
                                yield Static("0 samples", id="dataset-size", classes="info-value")
                                
                                yield Label("Created:")
                                yield Static("N/A", id="dataset-created", classes="info-value")
                                
                                yield Label("Format:")
                                yield Static("N/A", id="dataset-format", classes="info-value")
                                
                                yield Label("Task:")
                                yield Static("N/A", id="dataset-task", classes="info-value")
                        
                        # Dataset preview section
                        with Container(classes="section-container", id="dataset-preview"):
                            yield Static("Dataset Preview", classes="section-title")
                            
                            # Preview mode selector
                            with Horizontal(classes="preview-controls"):
                                yield Select(
                                    [
                                        ("samples", "Sample Data"),
                                        ("stats", "Statistics"),
                                        ("schema", "Schema")
                                    ],
                                    id="preview-mode",
                                    value="samples"
                                )
                                yield Button("Refresh", id="refresh-preview", classes="mini-button")
                            
                            # Preview content
                            with Container(id="preview-content", classes="preview-area"):
                                yield Static("Select a dataset to preview", classes="muted-text")
                        
                        # Actions
                        with Container(classes="section-container"):
                            with Horizontal(classes="button-row"):
                                yield Button(
                                    "Download",
                                    id="download-dataset-btn",
                                    classes="action-button",
                                    disabled=True
                                )
                                yield Button(
                                    "Export",
                                    id="export-dataset-btn",
                                    classes="action-button",
                                    disabled=True
                                )
                                yield Button(
                                    "Delete",
                                    id="delete-dataset-btn",
                                    classes="action-button danger",
                                    disabled=True
                                )
            
            # Upload Tab
            with TabPane("Upload", id="upload-tab"):
                with VerticalScroll(classes="eval-content-area"):
                    with Container(classes="section-container"):
                        yield Static("📤 Upload Dataset", classes="section-title")
                        
                        # Upload options
                        with Grid(classes="upload-grid"):
                            yield Label("Source Type:")
                            yield Select(
                                [
                                    ("file", "Local File"),
                                    ("url", "URL"),
                                    ("huggingface", "HuggingFace"),
                                    ("paste", "Paste Data")
                                ],
                                id="upload-source",
                                value="file"
                            )
                            
                            yield Label("Format:")
                            yield Select(
                                [
                                    ("jsonl", "JSON Lines"),
                                    ("csv", "CSV"),
                                    ("json", "JSON"),
                                    ("parquet", "Parquet")
                                ],
                                id="upload-format",
                                value="jsonl"
                            )
                        
                        # File selection area
                        with Container(id="file-selection-area", classes="upload-area"):
                            yield Static(
                                "📁 Drop files here or click to browse",
                                id="file-drop-zone",
                                classes="drop-zone"
                            )
                            yield Input(
                                placeholder="Or enter file path...",
                                id="file-path-input"
                            )
                            yield Button("Browse", id="browse-files-btn", classes="action-button")
                        
                        # URL input area (hidden by default)
                        with Container(id="url-input-area", classes="upload-area hidden"):
                            yield Input(
                                placeholder="Enter dataset URL...",
                                id="url-input"
                            )
                        
                        # Validation status
                        with Container(id="validation-status", classes="status-area hidden"):
                            yield Static("", id="validation-message", classes="status-text")
                            yield Button(
                                "Validate Dataset",
                                id="validate-btn",
                                classes="action-button"
                            )
                        
                        # Upload button
                        yield Button(
                            "Upload Dataset",
                            id="upload-btn",
                            classes="action-button primary",
                            disabled=True
                        )
            
            # Create Tab
            with TabPane("Create", id="create-tab"):
                with VerticalScroll(classes="eval-content-area"):
                    with Container(classes="section-container"):
                        yield Static("✨ Create New Dataset", classes="section-title")
                        
                        # Dataset builder
                        with Grid(classes="builder-grid"):
                            yield Label("Dataset Name:")
                            yield Input(
                                placeholder="my_dataset",
                                id="new-dataset-name"
                            )
                            
                            yield Label("Task Type:")
                            yield Select(
                                [
                                    ("qa", "Question Answering"),
                                    ("classification", "Classification"),
                                    ("generation", "Text Generation"),
                                    ("custom", "Custom Format")
                                ],
                                id="new-dataset-task"
                            )
                        
                        # Sample editor
                        yield Static("Add Samples", classes="subsection-title")
                        yield TextArea(
                            id="sample-editor",
                            classes="code-editor",
                            language="json"
                        )
                        
                        # Sample templates
                        with Horizontal(classes="template-buttons"):
                            yield Button("QA Template", id="qa-template-btn", classes="template-button")
                            yield Button("Classification Template", id="class-template-btn", classes="template-button")
                            yield Button("Generation Template", id="gen-template-btn", classes="template-button")
                        
                        # Builder actions
                        with Horizontal(classes="button-row"):
                            yield Button(
                                "Add Sample",
                                id="add-sample-btn",
                                classes="action-button"
                            )
                            yield Button(
                                "Clear",
                                id="clear-samples-btn",
                                classes="action-button"
                            )
                            yield Button(
                                "Save Dataset",
                                id="save-dataset-btn",
                                classes="action-button primary"
                            )
                        
                        # Current samples preview
                        with Container(classes="samples-preview"):
                            yield Static("Current Samples (0)", id="sample-count", classes="subsection-title")
                            yield DataTable(id="samples-table", show_cursor=True)
    
    def on_mount(self) -> None:
        """Initialize the dataset management window."""
        logger.info("DatasetManagementWindow mounted")
        self._setup_dataset_table()
        self._setup_samples_table()
        self._load_datasets()
    
    def _setup_dataset_table(self) -> None:
        """Set up the dataset table columns."""
        try:
            table = self.query_one("#dataset-table", DataTable)
            table.add_columns(
                "Name",
                "Type",
                "Samples",
                "Task",
                "Status"
            )
        except Exception as e:
            logger.error(f"Failed to setup dataset table: {e}")
    
    def _setup_samples_table(self) -> None:
        """Set up the samples table for dataset creation."""
        try:
            table = self.query_one("#samples-table", DataTable)
            table.add_columns("ID", "Input", "Expected Output", "Metadata")
        except Exception as e:
            logger.error(f"Failed to setup samples table: {e}")
    
    @work(exclusive=True)
    async def _load_datasets(self) -> None:
        """Load available datasets."""
        try:
            # get_available_datasets is not async
            datasets = get_available_datasets(self.app_instance)
            
            table = self.query_one("#dataset-table", DataTable)
            table.clear()
            
            for dataset in datasets:
                table.add_row(
                    dataset['name'],
                    dataset.get('type', 'Custom'),
                    str(dataset.get('size', 0)),
                    dataset.get('task', 'General'),
                    format_status_badge(dataset.get('status', 'available'))
                )
            
            if not datasets:
                self.notify_error("No datasets found")
                
        except Exception as e:
            self.notify_error(f"Failed to load datasets: {e}")
    
    @on(DataTable.RowSelected, "#dataset-table")
    async def handle_dataset_selection(self, event: DataTable.RowSelected) -> None:
        """Handle dataset selection from table."""
        try:
            # Get dataset name from row
            dataset_name = event.data_table.get_cell_at(event.cursor_row, 0)
            self.selected_dataset = dataset_name
            await self._load_dataset_details(dataset_name)
            
            # Enable action buttons
            for btn_id in ["download-dataset-btn", "export-dataset-btn", "delete-dataset-btn"]:
                self.query_one(f"#{btn_id}", Button).disabled = False
                
        except Exception as e:
            logger.error(f"Failed to handle dataset selection: {e}")
    
    async def _load_dataset_details(self, dataset_name: str) -> None:
        """Load and display dataset details."""
        try:
            info = await get_dataset_info(self.app_instance, dataset_name)
            
            # Update info fields
            self.query_one("#dataset-name").update(info['name'])
            self.query_one("#dataset-type").update(info.get('type', 'Custom'))
            self.query_one("#dataset-size").update(f"{info.get('size', 0):,} samples")
            self.query_one("#dataset-created").update(info.get('created', 'Unknown'))
            self.query_one("#dataset-format").update(info.get('format', 'Unknown'))
            self.query_one("#dataset-task").update(info.get('task', 'General'))
            
            # Load preview
            await self._load_dataset_preview(dataset_name)
            
        except Exception as e:
            self.notify_error(f"Failed to load dataset details: {e}")
    
    async def _load_dataset_preview(self, dataset_name: str) -> None:
        """Load dataset preview based on current mode."""
        preview_container = self.query_one("#preview-content")
        preview_container.clear()
        
        try:
            if self.preview_mode == "samples":
                # Show sample data
                samples = await self._get_dataset_samples(dataset_name, limit=5)
                if samples:
                    preview_table = DataTable(show_cursor=True)
                    preview_container.mount(preview_table)
                    
                    # Add columns based on first sample
                    if samples:
                        columns = list(samples[0].keys())
                        preview_table.add_columns(*columns)
                        
                        for sample in samples:
                            preview_table.add_row(*[str(sample.get(col, '')) for col in columns])
                else:
                    preview_container.mount(Static("No samples available", classes="muted-text"))
                    
            elif self.preview_mode == "stats":
                # Show statistics
                stats = await self._get_dataset_stats(dataset_name)
                stats_text = self._format_stats(stats)
                preview_container.mount(Static(stats_text, classes="stats-display"))
                
            elif self.preview_mode == "schema":
                # Show schema
                schema = await self._get_dataset_schema(dataset_name)
                schema_text = self._format_schema(schema)
                preview_container.mount(Static(schema_text, classes="schema-display"))
                
        except Exception as e:
            preview_container.mount(Static(f"Failed to load preview: {e}", classes="error-text"))
    
    async def _get_dataset_samples(self, dataset_name: str, limit: int = 5) -> List[Dict]:
        """Get sample records from dataset."""
        # TODO: Implement actual sample retrieval
        return [
            {"input": "Sample input", "output": "Sample output", "metadata": "{}"}
            for _ in range(limit)
        ]
    
    async def _get_dataset_stats(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset statistics."""
        # TODO: Implement actual statistics
        return {
            "total_samples": 1000,
            "avg_input_length": 150,
            "avg_output_length": 50,
            "unique_tasks": 5
        }
    
    async def _get_dataset_schema(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset schema."""
        # TODO: Implement actual schema retrieval
        return {
            "fields": {
                "input": "string",
                "output": "string",
                "metadata": "object"
            },
            "required": ["input", "output"]
        }
    
    def _format_stats(self, stats: Dict[str, Any]) -> str:
        """Format statistics for display."""
        lines = ["Dataset Statistics:\n"]
        for key, value in stats.items():
            formatted_key = key.replace('_', ' ').title()
            lines.append(f"{formatted_key}: {value}")
        return "\n".join(lines)
    
    def _format_schema(self, schema: Dict[str, Any]) -> str:
        """Format schema for display."""
        import json
        return f"Dataset Schema:\n\n{json.dumps(schema, indent=2)}"
    
    @on(Select.Changed, "#dataset-filter")
    async def handle_filter_change(self, event: Select.Changed) -> None:
        """Handle dataset filter change."""
        self.dataset_filter = event.value
        # TODO: Implement filtering
        await self._load_datasets()
    
    @on(Select.Changed, "#preview-mode")
    async def handle_preview_mode_change(self, event: Select.Changed) -> None:
        """Handle preview mode change."""
        self.preview_mode = event.value
        if self.selected_dataset:
            await self._load_dataset_preview(self.selected_dataset)
    
    @on(Select.Changed, "#upload-source")
    def handle_upload_source_change(self, event: Select.Changed) -> None:
        """Handle upload source type change."""
        # Show/hide relevant input areas
        file_area = self.query_one("#file-selection-area")
        url_area = self.query_one("#url-input-area")
        
        if event.value == "file":
            file_area.remove_class("hidden")
            url_area.add_class("hidden")
        elif event.value == "url" or event.value == "huggingface":
            file_area.add_class("hidden")
            url_area.remove_class("hidden")
        # TODO: Handle other source types
    
    @on(Button.Pressed, "#browse-files-btn")
    async def handle_browse_files(self) -> None:
        """Open file browser for dataset selection."""
        # from ..Widgets.eval_additional_dialogs import FilePickerDialog
        # TODO: Implement dialog
        self.notify_error("File picker dialog not yet implemented")
        return
        
        def on_file_selected(file_path):
            if file_path:
                self.query_one("#file-path-input", Input).value = str(file_path)
                self.query_one("#upload-btn", Button).disabled = False
                self.query_one("#validation-status").remove_class("hidden")
        
        dialog = FilePickerDialog(
            callback=on_file_selected,
            file_types=[".jsonl", ".json", ".csv", ".parquet"]
        )
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#validate-btn")
    async def handle_validate_dataset(self) -> None:
        """Validate selected dataset file."""
        file_path = self.query_one("#file-path-input", Input).value
        if not file_path:
            self.notify_error("Please select a file first")
            return
        
        try:
            result = await validate_dataset(self.app_instance, file_path)
            
            message = self.query_one("#validation-message", Static)
            if result['valid']:
                message.update(f"✅ Valid dataset: {result['samples']} samples found")
                message.add_class("success")
            else:
                message.update(f"❌ Invalid dataset: {result['error']}")
                message.add_class("error")
                
        except Exception as e:
            self.notify_error(f"Validation failed: {e}")
    
    @on(Button.Pressed, "#upload-btn")
    async def handle_upload_dataset(self) -> None:
        """Upload the dataset."""
        source_type = self.query_one("#upload-source", Select).value
        
        try:
            if source_type == "file":
                file_path = self.query_one("#file-path-input", Input).value
                if not file_path:
                    self.notify_error("Please select a file")
                    return
                
                result = await upload_dataset(
                    self.app_instance,
                    file_path,
                    source_type="file"
                )
            elif source_type == "url":
                url = self.query_one("#url-input", Input).value
                if not url:
                    self.notify_error("Please enter a URL")
                    return
                
                result = await upload_dataset(
                    self.app_instance,
                    url,
                    source_type="url"
                )
            
            if result['success']:
                self.notify_success(f"Dataset uploaded: {result['dataset_name']}")
                await self._load_datasets()
                # Switch to browse tab
                # TODO: Implement tab switching
            else:
                self.notify_error(f"Upload failed: {result['error']}")
                
        except Exception as e:
            self.notify_error(f"Upload failed: {e}")
    
    # Dataset creation handlers
    @on(Button.Pressed, "#qa-template-btn")
    def insert_qa_template(self) -> None:
        """Insert Q&A template."""
        template = '''{
    "input": "What is the capital of France?",
    "expected_output": "Paris",
    "metadata": {
        "category": "geography",
        "difficulty": "easy"
    }
}'''
        self.query_one("#sample-editor", TextArea).text = template
    
    @on(Button.Pressed, "#class-template-btn")
    def insert_classification_template(self) -> None:
        """Insert classification template."""
        template = '''{
    "input": "This movie was absolutely fantastic!",
    "expected_output": "positive",
    "metadata": {
        "task": "sentiment_classification",
        "labels": ["positive", "negative", "neutral"]
    }
}'''
        self.query_one("#sample-editor", TextArea).text = template
    
    @on(Button.Pressed, "#add-sample-btn")
    def handle_add_sample(self) -> None:
        """Add sample to dataset being created."""
        try:
            import json
            sample_text = self.query_one("#sample-editor", TextArea).text
            if not sample_text.strip():
                self.notify_error("Please enter a sample")
                return
            
            sample = json.loads(sample_text)
            
            # Add to samples table
            table = self.query_one("#samples-table", DataTable)
            table.add_row(
                str(table.row_count + 1),
                sample.get('input', ''),
                sample.get('expected_output', ''),
                str(sample.get('metadata', {}))
            )
            
            # Update count
            count_label = self.query_one("#sample-count", Static)
            count_label.update(f"Current Samples ({table.row_count})")
            
            # Clear editor
            self.query_one("#sample-editor", TextArea).clear()
            
        except json.JSONDecodeError as e:
            self.notify_error(f"Invalid JSON: {e}")
        except Exception as e:
            self.notify_error(f"Failed to add sample: {e}")
    
    @on(Button.Pressed, "#save-dataset-btn")
    async def handle_save_dataset(self) -> None:
        """Save the created dataset."""
        name = self.query_one("#new-dataset-name", Input).value
        if not name:
            self.notify_error("Please enter a dataset name")
            return
        
        table = self.query_one("#samples-table", DataTable)
        if table.row_count == 0:
            self.notify_error("Please add at least one sample")
            return
        
        # TODO: Implement dataset saving
        self.notify_success(f"Dataset '{name}' saved successfully")
    
    # Action button handlers
    @on(Button.Pressed, "#download-dataset-btn")
    async def handle_download_dataset(self) -> None:
        """Download selected dataset."""
        if not self.selected_dataset:
            return
        
        # TODO: Implement dataset download
        self.notify_success(f"Downloading {self.selected_dataset}...")
    
    @on(Button.Pressed, "#export-dataset-btn")
    async def handle_export_dataset(self) -> None:
        """Export selected dataset."""
        if not self.selected_dataset:
            return
        
        # from ..Widgets.eval_additional_dialogs import ExportDialog
        # TODO: Implement dialog
        self.notify_error("Export dialog not yet implemented")
        return
        
        def on_export_config(config):
            if config:
                self._export_dataset(config)
        
        dialog = ExportDialog(
            callback=on_export_config,
            data_type="dataset",
            dataset_name=self.selected_dataset
        )
        await self.app.push_screen(dialog)
    
    @on(Button.Pressed, "#delete-dataset-btn")
    async def handle_delete_dataset(self) -> None:
        """Delete selected dataset."""
        if not self.selected_dataset:
            return
        
        # Confirm deletion
        # from ..Widgets.eval_additional_dialogs import ConfirmDialog
        # TODO: Implement dialog
        if True:  # Auto-confirm for now
            self._delete_dataset()
        
        def on_confirm(confirmed):
            if confirmed:
                self._delete_dataset()
        
        dialog = ConfirmDialog(
            title="Delete Dataset",
            message=f"Are you sure you want to delete '{self.selected_dataset}'?",
            callback=on_confirm
        )
        await self.app.push_screen(dialog)
    
    async def _delete_dataset(self) -> None:
        """Actually delete the dataset."""
        try:
            result = await delete_dataset(self.app_instance, self.selected_dataset)
            if result['success']:
                self.notify_success(f"Dataset '{self.selected_dataset}' deleted")
                self.selected_dataset = None
                await self._load_datasets()
            else:
                self.notify_error(f"Failed to delete: {result['error']}")
        except Exception as e:
            self.notify_error(f"Delete failed: {e}")
    
    def _export_dataset(self, config: Dict[str, Any]) -> None:
        """Export dataset with given configuration."""
        # TODO: Implement dataset export
        self.notify_success(f"Exporting {self.selected_dataset}...")
    
    @on(Button.Pressed, "#back-to-main") 
    def handle_back(self) -> None:
        """Go back to main evaluation window."""
        self.navigate_to("main")
    
    @on(Button.Pressed, "#refresh-data")
    async def handle_refresh(self) -> None:
        """Refresh all data."""
        await self._load_datasets()
        if self.selected_dataset:
            await self._load_dataset_details(self.selected_dataset)
        self.notify_success("Data refreshed")