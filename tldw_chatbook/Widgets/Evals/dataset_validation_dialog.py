# dataset_validation_dialog.py
# Description: Dialog for dataset validation results display
#
"""
Dataset Validation Dialog
-------------------------

Provides an interactive dialog for:
- Displaying validation results
- Showing issues by severity
- Providing fix suggestions
- Exporting validation reports
"""

from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
from textual import on, work
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer, Grid
from textual.screen import ModalScreen
from textual.widgets import (
    Button, Label, Static, DataTable, Tabs, Tab, TabPane,
    ListView, ListItem, Collapsible, ProgressBar, Markdown
)
from textual.reactive import reactive
from textual.binding import Binding
from loguru import logger

from tldw_chatbook.Evals.dataset_validator import DatasetValidator, ValidationReport, ValidationIssue

class IssueListItem(ListItem):
    """Custom list item for displaying validation issues."""
    
    def __init__(self, issue: ValidationIssue, index: int):
        super().__init__()
        self.issue = issue
        self.index = index
        
        # Set classes based on severity
        self.add_class(f"issue-{issue.severity}")
    
    def compose(self) -> ComposeResult:
        with Horizontal(classes="issue-item"):
            # Severity icon
            icon = {
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️'
            }.get(self.issue.severity, '•')
            
            yield Static(icon, classes="issue-icon")
            
            # Issue details
            with Vertical(classes="issue-details"):
                yield Static(self.issue.message, classes="issue-message")
                
                details = []
                if self.issue.sample_index is not None:
                    details.append(f"Sample #{self.issue.sample_index}")
                if self.issue.field_name:
                    details.append(f"Field: {self.issue.field_name}")
                
                if details:
                    yield Static(" | ".join(details), classes="issue-metadata")

class DatasetValidationDialog(ModalScreen):
    """Dialog for displaying dataset validation results."""
    
    CSS = """
    DatasetValidationDialog {
        align: center middle;
    }
    
    .validation-dialog {
        width: 90%;
        height: 90%;
        max-width: 120;
        max-height: 40;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }
    
    .dialog-title {
        text-style: bold;
        text-align: center;
        width: 100%;
        margin-bottom: 1;
        color: $primary;
    }
    
    .summary-section {
        height: 8;
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border: solid $primary-background;
    }
    
    .summary-grid {
        grid-size: 3 2;
        grid-gutter: 1;
        height: 100%;
    }
    
    .summary-stat {
        layout: vertical;
        align: center middle;
    }
    
    .stat-value {
        text-style: bold;
        text-align: center;
        width: 100%;
    }
    
    .stat-label {
        text-align: center;
        width: 100%;
        color: $text-muted;
    }
    
    .stat-value.error {
        color: $error;
    }
    
    .stat-value.warning {
        color: $warning;
    }
    
    .stat-value.success {
        color: $success;
    }
    
    .tabs-container {
        height: 1fr;
        margin-bottom: 1;
    }
    
    #issues-list {
        height: 100%;
        overflow-y: auto;
    }
    
    .issue-item {
        padding: 0 1;
        width: 100%;
    }
    
    .issue-error {
        background: $error 20%;
    }
    
    .issue-warning {
        background: $warning 20%;
    }
    
    .issue-info {
        background: $primary 10%;
    }
    
    .issue-icon {
        width: 3;
        text-align: center;
    }
    
    .issue-details {
        width: 1fr;
    }
    
    .issue-message {
        text-style: bold;
    }
    
    .issue-metadata {
        color: $text-muted;
        text-style: italic;
    }
    
    .statistics-content {
        padding: 1;
        overflow-y: auto;
    }
    
    .suggestions-content {
        padding: 1;
        overflow-y: auto;
    }
    
    .suggestion-item {
        margin-bottom: 1;
        padding: 1;
        background: $panel;
        border-left: thick $primary;
    }
    
    .button-row {
        layout: horizontal;
        height: 3;
        align: right middle;
        padding: 0 1;
    }
    
    .action-button {
        margin-left: 1;
        min-width: 16;
    }
    """
    
    BINDINGS = [
        Binding("escape", "dismiss", "Close"),
        Binding("e", "export", "Export Report"),
        Binding("f", "fix", "Apply Fixes"),
    ]
    
    def __init__(self,
                 validation_report: ValidationReport,
                 dataset_path: Optional[Path] = None,
                 callback: Optional[Callable[[str], None]] = None,
                 **kwargs):
        """
        Initialize the validation dialog.
        
        Args:
            validation_report: The validation report to display
            dataset_path: Path to the dataset file
            callback: Callback for actions (export, fix, etc.)
        """
        super().__init__(**kwargs)
        self.validation_report = validation_report
        self.dataset_path = dataset_path
        self.callback = callback
        self.validator = DatasetValidator()
    
    def compose(self) -> ComposeResult:
        with Container(classes="validation-dialog"):
            yield Label(
                f"Validation Report: {self.validation_report.dataset_name}",
                classes="dialog-title"
            )
            
            # Summary section
            with Container(classes="summary-section"):
                with Grid(classes="summary-grid"):
                    # Total samples
                    with Container(classes="summary-stat"):
                        yield Static(
                            str(self.validation_report.total_samples),
                            classes="stat-value"
                        )
                        yield Static("Total Samples", classes="stat-label")
                    
                    # Valid samples
                    with Container(classes="summary-stat"):
                        yield Static(
                            str(self.validation_report.valid_samples),
                            classes="stat-value success"
                        )
                        yield Static("Valid Samples", classes="stat-label")
                    
                    # Validation status
                    with Container(classes="summary-stat"):
                        status = "✅ VALID" if self.validation_report.is_valid else "❌ INVALID"
                        yield Static(
                            status,
                            classes=f"stat-value {'success' if self.validation_report.is_valid else 'error'}"
                        )
                        yield Static("Status", classes="stat-label")
                    
                    # Error count
                    with Container(classes="summary-stat"):
                        yield Static(
                            str(self.validation_report.error_count),
                            classes="stat-value error"
                        )
                        yield Static("Errors", classes="stat-label")
                    
                    # Warning count
                    with Container(classes="summary-stat"):
                        yield Static(
                            str(self.validation_report.warning_count),
                            classes="stat-value warning"
                        )
                        yield Static("Warnings", classes="stat-label")
                    
                    # Info count
                    info_count = len([i for i in self.validation_report.issues if i.severity == 'info'])
                    with Container(classes="summary-stat"):
                        yield Static(
                            str(info_count),
                            classes="stat-value"
                        )
                        yield Static("Info", classes="stat-label")
            
            # Tabs for different views
            with Tabs(classes="tabs-container"):
                with TabPane("Issues", id="issues-tab"):
                    yield ListView(id="issues-list")
                
                with TabPane("Statistics", id="stats-tab"):
                    yield ScrollableContainer(
                        Static("Loading statistics..."),
                        id="statistics-content",
                        classes="statistics-content"
                    )
                
                with TabPane("Suggestions", id="suggestions-tab"):
                    yield ScrollableContainer(
                        Static("Generating suggestions..."),
                        id="suggestions-content",
                        classes="suggestions-content"
                    )
            
            # Action buttons
            with Horizontal(classes="button-row"):
                yield Button("Export Report", id="export-btn", classes="action-button", variant="primary")
                yield Button("Apply Fixes", id="fix-btn", classes="action-button", variant="warning")
                yield Button("Close", id="close-btn", classes="action-button")
    
    def on_mount(self) -> None:
        """Initialize the dialog when mounted."""
        self.populate_issues()
        self.populate_statistics()
        self.populate_suggestions()
    
    def populate_issues(self) -> None:
        """Populate the issues list."""
        issues_list = self.query_one("#issues-list", ListView)
        
        # Sort issues by severity (errors first, then warnings, then info)
        severity_order = {'error': 0, 'warning': 1, 'info': 2}
        sorted_issues = sorted(
            self.validation_report.issues,
            key=lambda x: (severity_order.get(x.severity, 3), x.sample_index or 0)
        )
        
        # Add issues to list
        for i, issue in enumerate(sorted_issues):
            issues_list.append(IssueListItem(issue, i))
        
        if not sorted_issues:
            issues_list.append(ListItem(Static("✅ No issues found!", classes="success-message")))
    
    def populate_statistics(self) -> None:
        """Populate the statistics view."""
        stats_content = self.query_one("#statistics-content", ScrollableContainer)
        stats_content.remove_children()
        
        # Build statistics markdown
        stats_md = "## Dataset Statistics\n\n"
        
        # Basic stats
        stats_md += "### Overview\n"
        stats_md += f"- **Total Samples**: {self.validation_report.total_samples}\n"
        stats_md += f"- **Valid Samples**: {self.validation_report.valid_samples}\n"
        stats_md += f"- **Invalid Samples**: {self.validation_report.total_samples - self.validation_report.valid_samples}\n"
        stats_md += f"- **Validation Rate**: {(self.validation_report.valid_samples / self.validation_report.total_samples * 100) if self.validation_report.total_samples > 0 else 0:.1f}%\n\n"
        
        # Quality metrics
        if 'quality' in self.validation_report.statistics:
            quality = self.validation_report.statistics['quality']
            stats_md += "### Quality Metrics\n"
            
            if 'avg_question_length' in quality:
                stats_md += f"- **Average Question Length**: {quality['avg_question_length']:.1f} chars\n"
                stats_md += f"- **Min Question Length**: {quality['min_question_length']} chars\n"
                stats_md += f"- **Max Question Length**: {quality['max_question_length']} chars\n"
            
            if 'unique_questions' in quality:
                stats_md += f"- **Unique Questions**: {quality['unique_questions']}\n"
                stats_md += f"- **Duplicate Questions**: {quality['duplicate_questions']}\n"
            
            if 'num_topics' in quality:
                stats_md += f"- **Number of Topics**: {quality['num_topics']}\n"
            
            stats_md += "\n"
        
        # Balance statistics
        if 'balance' in self.validation_report.statistics:
            balance = self.validation_report.statistics['balance']
            stats_md += "### Class Balance\n"
            
            if 'answer_distribution' in balance:
                stats_md += "**Answer Distribution:**\n"
                for answer, count in balance['answer_distribution'].items():
                    percentage = (count / self.validation_report.total_samples * 100) if self.validation_report.total_samples > 0 else 0
                    stats_md += f"- {answer}: {count} ({percentage:.1f}%)\n"
            
            if 'imbalance_ratio' in balance:
                stats_md += f"\n**Imbalance Ratio**: {balance['imbalance_ratio']:.2f}\n"
            
            stats_md += "\n"
        
        # Topic distribution
        if 'quality' in self.validation_report.statistics:
            quality = self.validation_report.statistics['quality']
            if 'topic_distribution' in quality:
                stats_md += "### Topic Distribution\n"
                for topic, count in sorted(quality['topic_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
                    stats_md += f"- {topic}: {count}\n"
                
                if len(quality['topic_distribution']) > 10:
                    stats_md += f"\n*...and {len(quality['topic_distribution']) - 10} more topics*\n"
        
        stats_content.mount(Markdown(stats_md))
    
    def populate_suggestions(self) -> None:
        """Populate the suggestions view."""
        suggestions_content = self.query_one("#suggestions-content", ScrollableContainer)
        suggestions_content.remove_children()
        
        # Get fix suggestions
        suggestions = self.validator.suggest_fixes(self.validation_report)
        
        if suggestions:
            for i, suggestion in enumerate(suggestions, 1):
                with suggestions_content.mount(Container(classes="suggestion-item")):
                    suggestions_content.mount(Static(f"{i}. {suggestion}"))
        else:
            suggestions_content.mount(Static(
                "✅ No fixes needed - dataset is valid!",
                classes="success-message"
            ))
    
    @on(Button.Pressed, "#export-btn")
    def handle_export(self, event: Button.Pressed) -> None:
        """Handle export button press."""
        logger.info("Export validation report requested")
        if self.callback:
            self.callback("export")
        self.dismiss({"action": "export", "report": self.validation_report})
    
    @on(Button.Pressed, "#fix-btn")
    def handle_fix(self, event: Button.Pressed) -> None:
        """Handle fix button press."""
        logger.info("Apply fixes requested")
        if self.callback:
            self.callback("fix")
        self.dismiss({"action": "fix", "report": self.validation_report})
    
    @on(Button.Pressed, "#close-btn")
    def handle_close(self, event: Button.Pressed) -> None:
        """Handle close button press."""
        self.dismiss(None)
    
    def action_export(self) -> None:
        """Export the validation report."""
        self.handle_export(None)
    
    def action_fix(self) -> None:
        """Apply suggested fixes."""
        self.handle_fix(None)