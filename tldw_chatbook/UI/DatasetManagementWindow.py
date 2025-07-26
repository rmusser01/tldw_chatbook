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
    Button, Label, Static, Input, ListView, ListItem
)
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from loguru import logger
import json
from datetime import datetime

from .eval_shared_components import (
    BaseEvaluationWindow, RefreshDataRequest,
    format_status_badge
)
from ..Event_Handlers.eval_events import (
    get_available_datasets, get_dataset_info, validate_dataset,
    upload_dataset, delete_dataset, export_dataset
)


class DatasetListItem(ListItem):
    """Custom list item for dataset display."""
    
    def __init__(self, dataset_info: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self.dataset_info = dataset_info
        
    def compose(self) -> ComposeResult:
        """Compose the dataset list item."""
        with Horizontal(classes="dataset-list-item"):
            yield Static(self.dataset_info['name'], classes="dataset-name")
            yield Static(f"({self.dataset_info.get('size', 0)} samples)", classes="dataset-size")


class DatasetManagementWindow(BaseEvaluationWindow):
    """Window for managing evaluation datasets."""
    
    # Reactive state
    selected_dataset = reactive(None)
    datasets = reactive([])
    search_query = reactive("")
    
    def compose(self) -> ComposeResult:
        """Compose the dataset management interface."""
        yield from self.compose_header("📚 Dataset Management")
        
        with Horizontal(classes="dataset-management-layout"):
            # Left panel - Dataset list
            with Vertical(classes="dataset-list-panel"):
                yield Static("Datasets", classes="panel-title")
                
                # Search input
                yield Input(
                    placeholder="Search datasets...",
                    id="dataset-search",
                    classes="dataset-search-input"
                )
                
                # Dataset list
                with VerticalScroll(classes="dataset-list-container"):
                    yield ListView(
                        id="dataset-list",
                        classes="dataset-list"
                    )
                
                # Action buttons
                with Horizontal(classes="dataset-actions"):
                    yield Button("Upload", id="upload-dataset", classes="action-button")
                    yield Button("Import", id="import-dataset", classes="action-button")
            
            # Right panel - Dataset preview
            with Vertical(classes="dataset-preview-panel"):
                yield Static("Dataset Preview", classes="panel-title")
                
                # Dataset info section
                with Container(classes="dataset-info-section"):
                    yield Label("Name:", classes="info-label")
                    yield Static("Select a dataset", id="dataset-name", classes="info-value")
                    
                    yield Label("Type:", classes="info-label")
                    yield Static("-", id="dataset-type", classes="info-value")
                    
                    yield Label("Size:", classes="info-label") 
                    yield Static("-", id="dataset-size", classes="info-value")
                
                # Sample preview section
                yield Static("Sample Preview:", classes="section-title")
                with VerticalScroll(classes="sample-preview-container"):
                    yield Container(id="sample-preview", classes="sample-preview")
                
                # Action buttons
                with Horizontal(classes="preview-actions"):
                    yield Button("Validate", id="validate-dataset", classes="action-button", disabled=True)
                    yield Button("Edit", id="edit-dataset", classes="action-button", disabled=True)
    
    def on_mount(self) -> None:
        """Initialize the dataset management window."""
        logger.info("DatasetManagementWindow mounted")
        self._load_datasets()
    
    @work(exclusive=True)
    async def _load_datasets(self) -> None:
        """Load available datasets."""
        try:
            datasets = get_available_datasets(self.app_instance)
            self.datasets = datasets
            self._update_dataset_list()
            
            if not datasets:
                self.notify_error("No datasets found")
                
        except Exception as e:
            self.notify_error(f"Failed to load datasets: {e}")
    
    def _update_dataset_list(self) -> None:
        """Update the dataset list view."""
        list_view = self.query_one("#dataset-list", ListView)
        list_view.clear()
        
        # Filter datasets based on search query
        filtered_datasets = self.datasets
        if self.search_query:
            query = self.search_query.lower()
            filtered_datasets = [
                d for d in self.datasets 
                if query in d['name'].lower()
            ]
        
        # Add dataset items
        for dataset in filtered_datasets:
            list_view.append(DatasetListItem(dataset))
    
    @on(Input.Changed, "#dataset-search")
    def handle_search_change(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        self.search_query = event.value
        self._update_dataset_list()
    
    @on(ListView.Selected, "#dataset-list")
    async def handle_dataset_selection(self, event: ListView.Selected) -> None:
        """Handle dataset selection from list."""
        if event.item and isinstance(event.item, DatasetListItem):
            self.selected_dataset = event.item.dataset_info
            await self._update_dataset_preview()
            
            # Enable action buttons
            self.query_one("#validate-dataset", Button).disabled = False
            self.query_one("#edit-dataset", Button).disabled = False
    
    async def _update_dataset_preview(self) -> None:
        """Update the dataset preview panel."""
        if not self.selected_dataset:
            return
        
        # Update dataset info
        self.query_one("#dataset-name").update(self.selected_dataset['name'])
        self.query_one("#dataset-type").update(
            self.selected_dataset.get('type', 'Multiple Choice')
        )
        self.query_one("#dataset-size").update(
            f"{self.selected_dataset.get('size', 0):,} samples"
        )
        
        # Update sample preview
        await self._load_dataset_samples()
    
    async def _load_dataset_samples(self) -> None:
        """Load and display dataset samples."""
        preview_container = self.query_one("#sample-preview")
        preview_container.remove_children()
        
        try:
            # Get detailed dataset info
            dataset_info = await get_dataset_info(self.app_instance, self.selected_dataset['name'])
            
            # For now, show mock samples
            # TODO: Load actual samples from dataset file
            samples = self._get_mock_samples()
            
            for i, sample in enumerate(samples[:3]):  # Show first 3 samples
                with preview_container:
                    sample_widget = Container(classes="dataset-sample")
                    
                    # Add sample number
                    sample_widget.mount(
                        Static(f"Sample {i+1}:", classes="sample-header")
                    )
                    
                    # Add question/input
                    sample_widget.mount(
                        Static(f"Q: {sample['question']}", classes="sample-question")
                    )
                    
                    # Add options if multiple choice
                    if 'options' in sample:
                        for option in sample['options']:
                            sample_widget.mount(
                                Static(f"   {option}", classes="sample-option")
                            )
                    
                    # Add answer
                    if 'answer' in sample:
                        sample_widget.mount(
                            Static(f"A: {sample['answer']}", classes="sample-answer")
                        )
                    
                    preview_container.mount(sample_widget)
                    
        except Exception as e:
            preview_container.mount(
                Static(f"Failed to load samples: {e}", classes="error-text")
            )
    
    def _get_mock_samples(self) -> List[Dict[str, Any]]:
        """Get mock samples for preview."""
        # TODO: Replace with actual dataset loading
        return [
            {
                "question": "What is the capital of France?",
                "options": ["A) London", "B) Paris", "C) Berlin", "D) Madrid"],
                "answer": "B) Paris"
            },
            {
                "question": "Which planet is known as the Red Planet?",
                "options": ["A) Venus", "B) Mars", "C) Jupiter", "D) Saturn"],
                "answer": "B) Mars"
            },
            {
                "question": "What is 2 + 2?",
                "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
                "answer": "B) 4"
            }
        ]
    
    @on(Button.Pressed, "#upload-dataset")
    async def handle_upload_dataset(self) -> None:
        """Handle dataset upload."""
        # TODO: Implement file picker dialog
        self.notify("Upload functionality coming soon", severity="information")
    
    @on(Button.Pressed, "#import-dataset")
    async def handle_import_dataset(self) -> None:
        """Handle dataset import from standard sources."""
        # TODO: Implement import dialog
        self.notify("Import functionality coming soon", severity="information")
    
    @on(Button.Pressed, "#validate-dataset")
    async def handle_validate_dataset(self) -> None:
        """Validate selected dataset."""
        if not self.selected_dataset:
            return
        
        self.notify(f"Validating {self.selected_dataset['name']}...", severity="information")
        
        # TODO: Implement actual validation
        await self.app.sleep(1)  # Simulate validation
        self.notify("Dataset validation complete", severity="success")
    
    @on(Button.Pressed, "#edit-dataset")
    async def handle_edit_dataset(self) -> None:
        """Edit selected dataset."""
        if not self.selected_dataset:
            return
        
        # TODO: Implement dataset editor
        self.notify("Edit functionality coming soon", severity="information")
    
    @on(Button.Pressed, "#back-to-main")
    def handle_back(self) -> None:
        """Go back to main evaluation window."""
        self.navigate_to("main")
    
    @on(Button.Pressed, "#refresh-data")
    async def handle_refresh(self) -> None:
        """Refresh all data."""
        await self._load_datasets()
        self.notify_success("Datasets refreshed")