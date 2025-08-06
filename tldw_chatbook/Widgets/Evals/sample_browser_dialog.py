# sample_browser_dialog.py
# Description: Dialog for browsing and viewing dataset samples
#
"""
Sample Browser Dialog
--------------------

Interactive dialog for:
- Browsing dataset samples with pagination
- Searching and filtering samples
- Viewing sample details with syntax highlighting
- Exporting selected samples
- Basic sample editing
"""

import json
from typing import Dict, List, Any, Optional, Callable, Union
from pathlib import Path
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer, Grid
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Label, Static, Input, Select, DataTable, 
    Tabs, Tab, TabPane, Footer, Markdown, TextArea
)
from textual.reactive import reactive
from textual.binding import Binding
from textual.message import Message
from loguru import logger

from tldw_chatbook.Evals.eval_runner import DatasetLoader

class SampleBrowserDialog(ModalScreen):
    """Dialog for browsing dataset samples."""
    
    CSS = """
    SampleBrowserDialog {
        align: center middle;
    }
    
    .browser-dialog {
        width: 95%;
        height: 95%;
        max-width: 150;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    .dialog-header {
        height: 4;
        margin-bottom: 1;
    }
    
    .dialog-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        color: $primary;
    }
    
    .controls-row {
        layout: horizontal;
        height: 3;
        margin-bottom: 1;
    }
    
    .search-input {
        width: 50%;
        margin-right: 1;
    }
    
    .filter-select {
        width: 20%;
        margin-right: 1;
    }
    
    .pagination-info {
        width: 1fr;
        text-align: right;
        padding: 0 1;
        color: $text-muted;
    }
    
    .main-content {
        height: 1fr;
        margin-bottom: 1;
    }
    
    /* Sample list styles */
    .samples-list {
        height: 100%;
        overflow-y: auto;
        background: $panel;
        border: solid $primary-background;
    }
    
    .sample-row {
        padding: 1;
        border-bottom: solid $primary-background;
    }
    
    .sample-row:hover {
        background: $accent 20%;
    }
    
    .sample-row.selected {
        background: $accent 40%;
        border-left: thick $primary;
    }
    
    .sample-header {
        layout: horizontal;
        width: 100%;
        margin-bottom: 0;
    }
    
    .sample-id {
        width: 15%;
        text-style: bold;
        color: $primary;
    }
    
    .sample-type {
        width: 20%;
        color: $text-muted;
    }
    
    .sample-preview {
        width: 65%;
        color: $text;
    }
    
    /* Sample detail styles */
    .sample-detail {
        height: 100%;
        padding: 1;
        background: $panel;
        border: solid $primary-background;
        overflow-y: auto;
    }
    
    .detail-section {
        margin-bottom: 1;
        padding: 1;
        background: $surface;
        border: round $primary-background;
    }
    
    .section-title {
        text-style: bold underline;
        color: $primary;
        margin-bottom: 1;
    }
    
    .json-display {
        font-family: monospace;
        background: $background;
        padding: 1;
        border: solid $primary-background;
        overflow-x: auto;
    }
    
    /* Navigation styles */
    .nav-controls {
        layout: horizontal;
        height: 3;
        align: center middle;
        padding: 0 1;
    }
    
    .nav-button {
        margin: 0 1;
        min-width: 12;
    }
    
    .page-info {
        width: 20;
        text-align: center;
        color: $text-muted;
    }
    
    /* Action buttons */
    .action-row {
        layout: horizontal;
        height: 3;
        align: right middle;
        padding: 0 1;
    }
    
    .action-button {
        margin-left: 1;
        min-width: 16;
    }
    
    /* Statistics panel */
    .stats-panel {
        height: 8;
        padding: 1;
        background: $panel;
        border: solid $primary-background;
        margin-bottom: 1;
    }
    
    .stats-grid {
        grid-size: 4 2;
        grid-gutter: 1;
        height: 100%;
    }
    
    .stat-item {
        layout: vertical;
        align: center middle;
    }
    
    .stat-value {
        text-style: bold;
        color: $primary;
    }
    
    .stat-label {
        color: $text-muted;
        text-align: center;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("left", "prev_page", "Previous Page"),
        Binding("right", "next_page", "Next Page"),
        Binding("ctrl+f", "focus_search", "Search"),
        Binding("ctrl+e", "export", "Export Selected"),
        Binding("f5", "refresh", "Refresh"),
    ]
    
    # Reactive attributes
    current_page = reactive(0)
    samples_per_page = reactive(20)
    search_query = reactive("")
    filter_field = reactive("")
    selected_indices = reactive(set())
    
    def __init__(self,
                 dataset_path: Union[str, Path],
                 task_type: str = 'auto',
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None,
                 **kwargs):
        """
        Initialize the sample browser.
        
        Args:
            dataset_path: Path to the dataset file
            task_type: Type of task for proper display
            callback: Callback for actions (export, edit, etc.)
        """
        super().__init__(**kwargs)
        self.dataset_path = Path(dataset_path)
        self.task_type = task_type
        self.callback = callback
        self.samples = []
        self.filtered_samples = []
        self.current_sample_index = 0
    
    def compose(self) -> ComposeResult:
        with Container(classes="browser-dialog"):
            # Header
            with Container(classes="dialog-header"):
                yield Label(
                    f"Sample Browser: {self.dataset_path.name}",
                    classes="dialog-title"
                )
            
            # Statistics panel
            with Container(classes="stats-panel"):
                with Grid(classes="stats-grid"):
                    # Total samples
                    with Container(classes="stat-item"):
                        yield Static("0", id="total-samples", classes="stat-value")
                        yield Static("Total Samples", classes="stat-label")
                    
                    # Filtered samples
                    with Container(classes="stat-item"):
                        yield Static("0", id="filtered-samples", classes="stat-value")
                        yield Static("Filtered", classes="stat-label")
                    
                    # Selected samples
                    with Container(classes="stat-item"):
                        yield Static("0", id="selected-samples", classes="stat-value")
                        yield Static("Selected", classes="stat-label")
                    
                    # Current page
                    with Container(classes="stat-item"):
                        yield Static("1 / 1", id="page-info", classes="stat-value")
                        yield Static("Page", classes="stat-label")
                    
                    # Avg length
                    with Container(classes="stat-item"):
                        yield Static("0", id="avg-length", classes="stat-value")
                        yield Static("Avg Length", classes="stat-label")
                    
                    # Task type
                    with Container(classes="stat-item"):
                        yield Static(self.task_type, id="task-type", classes="stat-value")
                        yield Static("Task Type", classes="stat-label")
                    
                    # Unique values (for classification)
                    with Container(classes="stat-item"):
                        yield Static("0", id="unique-values", classes="stat-value")
                        yield Static("Unique Answers", classes="stat-label")
                    
                    # File format
                    with Container(classes="stat-item"):
                        yield Static(self.dataset_path.suffix[1:].upper(), classes="stat-value")
                        yield Static("Format", classes="stat-label")
            
            # Search and filter controls
            with Horizontal(classes="controls-row"):
                yield Input(
                    placeholder="Search samples...",
                    id="search-input",
                    classes="search-input"
                )
                yield Select(
                    [("All Fields", ""), ("Question", "question"), ("Answer", "answer"), 
                     ("Topic", "topic"), ("ID", "id")],
                    id="filter-field",
                    classes="filter-select",
                    value=""
                )
                yield Static("", id="filter-status", classes="pagination-info")
            
            # Main content area with tabs
            with Tabs(classes="main-content"):
                with TabPane("List View", id="list-tab"):
                    with Horizontal():
                        # Sample list (left)
                        yield ScrollableContainer(
                            id="samples-list",
                            classes="samples-list"
                        )
                        
                        # Sample detail (right)
                        yield ScrollableContainer(
                            Static("Select a sample to view details"),
                            id="sample-detail",
                            classes="sample-detail"
                        )
                
                with TabPane("Table View", id="table-tab"):
                    yield DataTable(id="samples-table", show_cursor=True)
                
                with TabPane("JSON View", id="json-tab"):
                    yield TextArea(
                        "",
                        id="json-editor",
                        language="json",
                        theme="monokai"
                    )
            
            # Navigation controls
            with Horizontal(classes="nav-controls"):
                yield Button("◀ Previous", id="prev-btn", classes="nav-button", disabled=True)
                yield Static("Page 1 / 1", id="page-display", classes="page-info")
                yield Button("Next ▶", id="next-btn", classes="nav-button", disabled=True)
            
            # Action buttons
            with Horizontal(classes="action-row"):
                yield Button("Select All", id="select-all-btn", classes="action-button")
                yield Button("Clear Selection", id="clear-btn", classes="action-button")
                yield Button("Export Selected", id="export-btn", classes="action-button", variant="primary")
                yield Button("Close", id="close-btn", classes="action-button")
    
    def on_mount(self) -> None:
        """Initialize when mounted."""
        self.load_samples()
    
    @work(exclusive=True)
    async def load_samples(self) -> None:
        """Load samples from the dataset."""
        try:
            # Load samples using DatasetLoader
            samples = await self.app.run_in_executor(
                None,
                DatasetLoader.load_dataset_samples,
                {'dataset_name': str(self.dataset_path), 'task_type': self.task_type},
                None,
                None  # Load all samples
            )
            
            # Convert to list of dicts
            self.samples = []
            for i, sample in enumerate(samples):
                sample_dict = {
                    'index': i,
                    'id': getattr(sample, 'id', f'sample_{i}'),
                    'input_text': getattr(sample, 'input_text', ''),
                    'expected_output': getattr(sample, 'expected_output', ''),
                    'metadata': getattr(sample, 'metadata', {})
                }
                
                # Add choices for classification
                if hasattr(sample, 'choices'):
                    sample_dict['choices'] = sample.choices
                
                self.samples.append(sample_dict)
            
            self.filtered_samples = self.samples.copy()
            self.update_display()
            self.update_statistics()
            
        except Exception as e:
            logger.error(f"Error loading samples: {e}")
            self.app.notify(f"Failed to load samples: {str(e)}", severity="error")
    
    def update_display(self) -> None:
        """Update the samples display."""
        # Update list view
        self.update_list_view()
        
        # Update table view
        self.update_table_view()
        
        # Update JSON view
        self.update_json_view()
        
        # Update pagination
        self.update_pagination()
    
    def update_list_view(self) -> None:
        """Update the list view display."""
        samples_list = self.query_one("#samples-list", ScrollableContainer)
        samples_list.remove_children()
        
        # Calculate page range
        start = self.current_page * self.samples_per_page
        end = min(start + self.samples_per_page, len(self.filtered_samples))
        
        # Add samples to list
        for i in range(start, end):
            sample = self.filtered_samples[i]
            
            with samples_list.mount(Container(classes="sample-row", id=f"sample-{i}")):
                with samples_list.mount(Horizontal(classes="sample-header")):
                    samples_list.mount(Static(sample['id'], classes="sample-id"))
                    samples_list.mount(Static(self.task_type, classes="sample-type"))
                    
                    # Create preview
                    preview = sample.get('input_text', '')[:100]
                    if len(sample.get('input_text', '')) > 100:
                        preview += "..."
                    samples_list.mount(Static(preview, classes="sample-preview"))
    
    def update_table_view(self) -> None:
        """Update the table view display."""
        table = self.query_one("#samples-table", DataTable)
        table.clear()
        
        if not self.filtered_samples:
            return
        
        # Add columns based on first sample
        sample = self.filtered_samples[0]
        columns = ['Index', 'ID']
        
        if 'input_text' in sample:
            columns.append('Input')
        if 'expected_output' in sample:
            columns.append('Expected')
        if 'choices' in sample:
            columns.extend([f'Choice {i+1}' for i in range(len(sample['choices']))])
        
        for col in columns:
            table.add_column(col)
        
        # Add rows
        start = self.current_page * self.samples_per_page
        end = min(start + self.samples_per_page, len(self.filtered_samples))
        
        for i in range(start, end):
            sample = self.filtered_samples[i]
            row = [str(i), sample['id']]
            
            if 'input_text' in sample:
                row.append(sample['input_text'][:50] + '...' if len(sample['input_text']) > 50 else sample['input_text'])
            if 'expected_output' in sample:
                row.append(str(sample['expected_output']))
            if 'choices' in sample:
                row.extend(sample['choices'])
            
            table.add_row(*row)
    
    def update_json_view(self) -> None:
        """Update the JSON view."""
        editor = self.query_one("#json-editor", TextArea)
        
        # Show current page samples in JSON
        start = self.current_page * self.samples_per_page
        end = min(start + self.samples_per_page, len(self.filtered_samples))
        
        page_samples = self.filtered_samples[start:end]
        editor.text = json.dumps(page_samples, indent=2)
    
    def update_statistics(self) -> None:
        """Update statistics display."""
        # Total samples
        self.query_one("#total-samples").update(str(len(self.samples)))
        
        # Filtered samples
        self.query_one("#filtered-samples").update(str(len(self.filtered_samples)))
        
        # Selected samples
        self.query_one("#selected-samples").update(str(len(self.selected_indices)))
        
        # Average length
        if self.filtered_samples:
            avg_len = sum(len(s.get('input_text', '')) for s in self.filtered_samples) / len(self.filtered_samples)
            self.query_one("#avg-length").update(f"{avg_len:.0f}")
        
        # Unique answers (for classification)
        if self.task_type == 'classification' and self.filtered_samples:
            unique_answers = len(set(s.get('expected_output', '') for s in self.filtered_samples))
            self.query_one("#unique-values").update(str(unique_answers))
    
    def update_pagination(self) -> None:
        """Update pagination controls."""
        total_pages = (len(self.filtered_samples) + self.samples_per_page - 1) // self.samples_per_page
        current = self.current_page + 1
        
        # Update page info
        self.query_one("#page-info").update(f"{current} / {total_pages}")
        self.query_one("#page-display").update(f"Page {current} / {total_pages}")
        
        # Update button states
        self.query_one("#prev-btn").disabled = self.current_page == 0
        self.query_one("#next-btn").disabled = self.current_page >= total_pages - 1
    
    @on(Input.Changed, "#search-input")
    def handle_search(self, event: Input.Changed) -> None:
        """Handle search input changes."""
        self.search_query = event.value
        self.filter_samples()
    
    @on(Select.Changed, "#filter-field")
    def handle_filter_change(self, event: Select.Changed) -> None:
        """Handle filter field selection."""
        self.filter_field = event.value
        self.filter_samples()
    
    def filter_samples(self) -> None:
        """Filter samples based on search and filter criteria."""
        if not self.search_query:
            self.filtered_samples = self.samples.copy()
        else:
            self.filtered_samples = []
            query = self.search_query.lower()
            
            for sample in self.samples:
                # Search in specific field or all fields
                if self.filter_field:
                    value = str(sample.get(self.filter_field, '')).lower()
                    if query in value:
                        self.filtered_samples.append(sample)
                else:
                    # Search all fields
                    for value in sample.values():
                        if query in str(value).lower():
                            self.filtered_samples.append(sample)
                            break
        
        # Reset to first page
        self.current_page = 0
        self.update_display()
        self.update_statistics()
        
        # Update filter status
        status = f"Showing {len(self.filtered_samples)} of {len(self.samples)} samples"
        if self.search_query:
            status += f" matching '{self.search_query}'"
        self.query_one("#filter-status").update(status)
    
    @on(Button.Pressed, "#prev-btn")
    def handle_prev_page(self) -> None:
        """Go to previous page."""
        if self.current_page > 0:
            self.current_page -= 1
            self.update_display()
    
    @on(Button.Pressed, "#next-btn")
    def handle_next_page(self) -> None:
        """Go to next page."""
        total_pages = (len(self.filtered_samples) + self.samples_per_page - 1) // self.samples_per_page
        if self.current_page < total_pages - 1:
            self.current_page += 1
            self.update_display()
    
    @on(Button.Pressed, "#export-btn")
    def handle_export(self) -> None:
        """Export selected samples."""
        if not self.selected_indices:
            self.app.notify("No samples selected for export", severity="warning")
            return
        
        selected = [self.samples[i] for i in self.selected_indices]
        
        if self.callback:
            self.callback({"action": "export", "samples": selected})
        
        self.dismiss({"action": "export", "samples": selected})
    
    @on(Button.Pressed, "#close-btn")
    def handle_close(self) -> None:
        """Close the dialog."""
        self.dismiss(None)
    
    def action_prev_page(self) -> None:
        """Navigate to previous page."""
        self.handle_prev_page()
    
    def action_next_page(self) -> None:
        """Navigate to next page."""
        self.handle_next_page()
    
    def action_focus_search(self) -> None:
        """Focus the search input."""
        self.query_one("#search-input").focus()
    
    def action_export(self) -> None:
        """Export selected samples."""
        self.handle_export()