# template_selector.py
# Description: Template selection widget for evaluation tasks
#
"""
Template Selector Widget
------------------------

Provides an organized interface for browsing and selecting evaluation templates:
- Categorized template display
- Template preview and description
- Quick template creation
- Search and filtering capabilities
"""

from typing import Dict, List, Any, Optional, Callable
from textual import on
from textual.app import ComposeResult
from textual.containers import Container, Vertical, Horizontal
from textual.screen import ModalScreen
from textual.timer import Timer
from textual.widgets import (
    Button,
    Label,
    Input,
    ListView,
    ListItem,
    Static,
    Collapsible,
    Tabs,
    TabPane,
)
from loguru import logger


#: Debounce for the template search `Input` -- mirrors the picker/filter
#: family's 0.2 s shape (`console_prompt_picker_modal.py`). A settled
#: search clears and rebuilds every category's `ListView`, which should
#: not happen on every keystroke (task-15476).
TEMPLATE_SEARCH_DEBOUNCE_SECONDS = 0.2

TEMPLATE_CREATE_DISABLED_TOOLTIP = (
    "Select an evaluation template before creating a task."
)
TEMPLATE_EXPORT_DISABLED_TOOLTIP = "Select an evaluation template before exporting it."
TEMPLATE_CREATE_ENABLED_TOOLTIP = "Create an evaluation task from this template."
TEMPLATE_EXPORT_ENABLED_TOOLTIP = "Export this evaluation template."
TEMPLATE_SELECT_DISABLED_TOOLTIP = "Select an evaluation template before continuing."
TEMPLATE_SELECT_ENABLED_TOOLTIP = "Use the selected evaluation template."

#: Categories the built-in EvalTemplateManager lists (mirrors
#: EvalTemplateManager.list_templates; the research category is not listed
#: by the manager itself).
_EVAL_TEMPLATE_CATEGORIES = (
    "reasoning",
    "language",
    "coding",
    "safety",
    "creative",
    "multimodal",
)


def _builtin_eval_template_records() -> list[dict[str, Any]]:
    """Build the dict records this widget family renders for the built-ins.

    ``get_eval_templates()`` lives in ``tldw_chatbook.Evals.eval_templates``
    (NOT the dependency-light ``tldw_chatbook.Evals`` package root, which
    stopped exporting it -- TASK-32831). Its manager maps snake_case
    template ids to spec dicts.

    Each record carries the FULL source specification (metric, dataset,
    generation arguments, prompts, filters, ...) with the display fields
    overlaid on top: ``name`` stays the snake_case id (widget ids and
    selection matching depend on it) while ``display_name`` holds the
    spec's human-readable name. Preview/create/export/select paths
    therefore receive the complete evaluation definition, not just the
    display metadata (Qodo PR-2751 finding 3).
    """
    from tldw_chatbook.Evals.eval_templates import get_eval_templates

    manager = get_eval_templates()
    records: list[dict[str, Any]] = []
    for category in _EVAL_TEMPLATE_CATEGORIES:
        for template_id, spec in manager.get_templates_by_category(
            category
        ).items():
            record = dict(spec)
            metadata = record.get("metadata") or {}
            display_name = record.get("name", template_id)
            record.update(
                {
                    "name": template_id,
                    "display_name": display_name,
                    "description": record.get("description", ""),
                    "category": metadata.get("category", category),
                    "task_type": record.get("task_type", ""),
                    "difficulty": record.get("difficulty", "Unknown"),
                }
            )
            records.append(record)
    return records


class TemplatePreviewWidget(Container):
    """Widget for displaying template preview information."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_template = None

    def compose(self) -> ComposeResult:
        yield Label("Template Preview", classes="preview-title")
        yield Static(
            "Select a template to see details",
            id="template-description",
            classes="template-description",
        )

        with Collapsible(
            title="Configuration Details", collapsed=True, id="config-details"
        ):
            yield Static("", id="config-display", classes="config-display")

        with Horizontal(classes="preview-actions"):
            yield Button(
                "Create Task",
                id="create-task-btn",
                variant="primary",
                disabled=True,
                tooltip=TEMPLATE_CREATE_DISABLED_TOOLTIP,
            )
            yield Button(
                "Export Template",
                id="export-template-btn",
                disabled=True,
                tooltip=TEMPLATE_EXPORT_DISABLED_TOOLTIP,
            )

    def update_preview(self, template: Dict[str, Any]):
        """Update the preview with template information."""
        self.current_template = template

        try:
            # Update description
            description = f"**{template.get('name', 'Unknown')}**\n\n"
            description += template.get("description", "No description available.")
            description += f"\n\n**Category:** {template.get('category', 'General')}"
            description += f"\n**Difficulty:** {template.get('difficulty', 'Unknown')}"
            description += f"\n**Task Type:** {template.get('task_type', 'Unknown')}"

            desc_widget = self.query_one("#template-description")
            desc_widget.update(description)

            # Update configuration details
            config_text = "**Configuration:**\n"
            for key, value in template.items():
                if key not in ["name", "description", "category", "difficulty"]:
                    config_text += f"- {key}: {value}\n"

            config_widget = self.query_one("#config-display")
            config_widget.update(config_text)

            # Enable buttons
            self._set_action_state(has_template=True)

        except Exception as e:
            logger.error(f"Error updating template preview: {e}")

    def clear_preview(self):
        """Clear the preview display."""
        self.current_template = None

        try:
            self.query_one("#template-description").update(
                "Select a template to see details"
            )
            self.query_one("#config-display").update("")
            self._set_action_state(has_template=False)
        except Exception:
            pass

    def _set_action_state(self, *, has_template: bool) -> None:
        """Keep disabled preview actions paired with the required next step."""
        create_button = self.query_one("#create-task-btn", Button)
        create_button.disabled = not has_template
        create_button.tooltip = (
            TEMPLATE_CREATE_ENABLED_TOOLTIP
            if has_template
            else TEMPLATE_CREATE_DISABLED_TOOLTIP
        )

        export_button = self.query_one("#export-template-btn", Button)
        export_button.disabled = not has_template
        export_button.tooltip = (
            TEMPLATE_EXPORT_ENABLED_TOOLTIP
            if has_template
            else TEMPLATE_EXPORT_DISABLED_TOOLTIP
        )


class TemplateListWidget(Container):
    """Widget for displaying template lists organized by category."""

    def __init__(
        self,
        templates: List[Dict[str, Any]],
        on_template_selected: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.templates = templates
        self.on_template_selected = on_template_selected
        self.templates_by_category = self._organize_by_category()
        self._search_debounce_timer: Timer | None = None

    def _organize_by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        """Organize templates by category."""
        categories = {}
        for template in self.templates:
            category = template.get("category", "General")
            if category not in categories:
                categories[category] = []
            categories[category].append(template)
        return categories

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search templates...", id="template-search")

        with Tabs(id="category-tabs"):
            # Create tab for each category
            for category, templates in self.templates_by_category.items():
                with TabPane(category.title(), id=f"tab-{category}"):
                    with ListView(id=f"list-{category}"):
                        for template in templates:
                            yield ListItem(
                                Label(
                                    template.get(
                                        "display_name", template.get("name", "Unknown")
                                    )
                                ),
                                name=template.get("name", ""),
                                id=f"template-{template.get('name', '')}",
                            )

    @on(Input.Changed, "#template-search")
    def handle_search(self, event: Input.Changed):
        """Handle template search (debounced -- task-15476)."""
        raw_value = event.value
        if self._search_debounce_timer is not None:
            self._search_debounce_timer.stop()
        self._search_debounce_timer = self.set_timer(
            TEMPLATE_SEARCH_DEBOUNCE_SECONDS,
            lambda: self._apply_search_debounced(raw_value),
        )

    def _apply_search_debounced(self, raw_value: str) -> None:
        self._search_debounce_timer = None
        search_term = raw_value.lower()

        # Filter and update template lists
        for category, templates in self.templates_by_category.items():
            try:
                list_widget = self.query_one(f"#list-{category}")
                list_widget.clear()

                filtered_templates = [
                    t
                    for t in templates
                    if search_term in t.get("name", "").lower()
                    or search_term in t.get("description", "").lower()
                ]

                for template in filtered_templates:
                    list_widget.append(
                        ListItem(
                            Label(
                                template.get(
                                    "display_name", template.get("name", "Unknown")
                                )
                            ),
                            name=template.get("name", ""),
                            id=f"template-{template.get('name', '')}",
                        )
                    )
            except Exception as e:
                logger.error(f"Error filtering templates for category {category}: {e}")

    @on(ListView.Selected)
    def handle_template_selected(self, event: ListView.Selected):
        """Handle template selection."""
        if event.item and event.item.name:
            template_name = event.item.name

            # Find the template
            template = None
            for templates in self.templates_by_category.values():
                for t in templates:
                    if t.get("name") == template_name:
                        template = t
                        break
                if template:
                    break

            if template and self.on_template_selected:
                self.on_template_selected(template)


class TemplateSelectorDialog(ModalScreen):
    """Modal dialog for selecting evaluation templates."""

    # Base geometry picked up by the modal wide tier (task: wave 2 of the
    # repo-wide tier, 2026-09-19). Numeric literals (not $ds-* tokens) on
    # purpose: BUNDLED_CSS must also resolve in bare-App test harnesses that
    # never load the token file (same constraint documented in
    # detail_value_row.py / personas_character_editor_widget.py).
    BUNDLED_CSS = """
    TemplateSelectorDialog {
        align: center middle;
    }
    TemplateSelectorDialog .template-selector-dialog {
        width: 76;
        max-width: 95%;
        height: 80%;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    """

    def __init__(
        self,
        callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.callback = callback
        self.selected_template = None
        self.templates = []
        # Load before the first compose so the categorized ListViews render
        # the templates (the old on_mount load mutated data the already-
        # composed list could never display -- TASK-32831).
        self._load_templates()

    def _load_templates(self):
        """Load available templates."""
        try:
            self.templates = _builtin_eval_template_records()
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates = []

    def compose(self) -> ComposeResult:
        with Container(classes="template-selector-dialog"):
            yield Label("Select Evaluation Template", classes="dialog-title")

            with Horizontal(classes="template-content"):
                # Left side - template list
                with Vertical(classes="template-list-container"):
                    yield TemplateListWidget(
                        templates=self.templates,
                        on_template_selected=self._on_template_selected,
                        id="template-list",
                    )

                # Right side - preview
                with Vertical(classes="template-preview-container"):
                    yield TemplatePreviewWidget(id="template-preview")

            with Horizontal(classes="dialog-buttons"):
                yield Button("Cancel", id="cancel-button", variant="error")
                yield Button(
                    "Select Template",
                    id="select-button",
                    variant="primary",
                    disabled=True,
                    tooltip=TEMPLATE_SELECT_DISABLED_TOOLTIP,
                )

    def _on_template_selected(self, template: Dict[str, Any]):
        """Handle template selection."""
        self.selected_template = template

        # Update preview
        try:
            preview = self.query_one("#template-preview")
            preview.update_preview(template)
        except Exception:
            pass

        # Enable select button
        try:
            select_button = self.query_one("#select-button", Button)
            select_button.disabled = False
            select_button.tooltip = TEMPLATE_SELECT_ENABLED_TOOLTIP
        except Exception:
            pass

    @on(Button.Pressed, "#create-task-btn")
    def handle_create_task(self):
        """Handle create task from template."""
        if self.selected_template:
            # Close dialog and return template for task creation
            if self.callback:
                self.callback(self.selected_template)
            self.dismiss(self.selected_template)

    @on(Button.Pressed, "#export-template-btn")
    def handle_export_template(self):
        """Handle template export."""
        if self.selected_template:
            # This would open an export dialog
            self.app.notify(
                "Template export not yet implemented", severity="information"
            )

    @on(Button.Pressed, "#select-button")
    def handle_select(self):
        """Handle select button press."""
        if self.selected_template:
            if self.callback:
                self.callback(self.selected_template)
            self.dismiss(self.selected_template)

    @on(Button.Pressed, "#cancel-button")
    def handle_cancel(self):
        """Handle cancel button press."""
        if self.callback:
            self.callback(None)
        self.dismiss(None)


class QuickTemplateSelector(Container):
    """Quick template selector for inline use."""

    def __init__(
        self,
        category_filter: Optional[str] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.category_filter = category_filter
        self.callback = callback
        self.templates = []
        # Load before the first compose so the quick buttons render (same
        # reason as TemplateSelectorDialog -- TASK-32831).
        self._load_templates()

    def _load_templates(self):
        """Load and filter templates."""
        try:
            all_templates = _builtin_eval_template_records()

            if self.category_filter:
                self.templates = [
                    t
                    for t in all_templates
                    if t.get("category", "").lower() == self.category_filter.lower()
                ]
            else:
                self.templates = all_templates

        except Exception as e:
            logger.error(f"Error loading templates: {e}")
            self.templates = []

    def compose(self) -> ComposeResult:
        yield Label("Quick Templates", classes="section-title")

        with Horizontal(classes="quick-template-buttons"):
            # Show first few templates as quick buttons
            for template in self.templates[:6]:  # Limit to 6 quick buttons
                yield Button(
                    template.get("display_name", template.get("name", "Unknown")),
                    name=template.get("name", ""),
                    classes="template-quick-button",
                )

            yield Button(
                "More Templates...",
                id="more-templates-btn",
                classes="more-templates-button",
            )

    @on(Button.Pressed, ".template-quick-button")
    def handle_quick_template(self, event: Button.Pressed):
        """Handle quick template button press."""
        template_name = event.button.name

        # Find the template
        template = None
        for t in self.templates:
            if t.get("name") == template_name:
                template = t
                break

        if template and self.callback:
            self.callback(template)

    @on(Button.Pressed, "#more-templates-btn")
    def handle_more_templates(self):
        """Open the full template selector dialog."""
        dialog = TemplateSelectorDialog(callback=self.callback)
        self.app.push_screen(dialog)
