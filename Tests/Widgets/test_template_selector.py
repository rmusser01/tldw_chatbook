# TASK-32831: ``TemplateSelectorDialog._load_templates`` (and
# ``QuickTemplateSelector._load_templates``) imported ``get_eval_templates``
# from ``tldw_chatbook.Evals``, whose ``__init__`` no longer exports it -- the
# ImportError was swallowed and the template list was always empty (documented
# in .superpowers/wave2-report.md). The symbol lives in
# ``tldw_chatbook.Evals.eval_templates``. These tests pin that the selector
# actually lists the built-in evaluation templates.

from textual.app import App
from textual.widgets import Button, Label, ListItem

from tldw_chatbook.Widgets.template_selector import (
    QuickTemplateSelector,
    TemplateSelectorDialog,
    _builtin_eval_template_records,
)


class _ScreenHost(App):
    def __init__(self, screen):
        super().__init__()
        self.screen_under_test = screen

    async def on_mount(self) -> None:
        await self.push_screen(self.screen_under_test)


class _WidgetHost(App):
    def __init__(self, widget):
        super().__init__()
        self.widget_under_test = widget

    def compose(self):
        yield self.widget_under_test


async def test_template_selector_dialog_lists_builtin_eval_templates():
    dialog = TemplateSelectorDialog()
    app = _ScreenHost(dialog)

    async with app.run_test() as pilot:
        await pilot.pause()

        # The built-in evaluation templates (reasoning, language, coding,
        # safety, creative, multimodal categories) are loaded as dict
        # records the list/preview widgets consume.
        assert len(dialog.templates) >= 1
        first = dialog.templates[0]
        assert isinstance(first, dict)
        assert first.get("name")
        assert first.get("display_name")
        assert first.get("category")

        # ...and they are actually rendered: the categorized ListViews carry
        # at least one ListItem and a known built-in template is displayed.
        items = dialog.query(ListItem)
        assert len(items) >= 1
        labels = [str(item.query_one(Label).renderable) for item in items]
        assert "GSM8K" in labels

        # Selection wiring still works off the rendered records.
        dialog._on_template_selected(first)
        assert dialog.selected_template is first
        assert dialog.query_one("#select-button", Button).disabled is False


async def test_quick_template_selector_lists_builtin_eval_templates():
    widget = QuickTemplateSelector()
    app = _WidgetHost(widget)

    async with app.run_test() as pilot:
        await pilot.pause()

        assert len(widget.templates) >= 1
        quick_buttons = widget.query(".template-quick-button")
        assert len(quick_buttons) >= 1


# --- Isolated unit tests for _builtin_eval_template_records (Qodo PR-2751
# --- findings 2 and 3): controlled manager data, no real built-ins.


class _FakeTemplateManager:
    """Stand-in for EvalTemplateManager with controlled category data."""

    def __init__(self, categories):
        self._categories = categories

    def get_templates_by_category(self, category):
        return self._categories.get(category, {})


def _install_fake_manager(monkeypatch, **categories):
    """Point the helper's call-time import at a fake manager.

    ``_builtin_eval_template_records`` imports ``get_eval_templates`` from
    ``tldw_chatbook.Evals.eval_templates`` inside the function body, so
    patching the module attribute intercepts it without touching the real
    singleton.
    """
    import tldw_chatbook.Evals.eval_templates as eval_templates_module

    fake = _FakeTemplateManager(categories)
    monkeypatch.setattr(eval_templates_module, "get_eval_templates", lambda: fake)
    return fake


def test_records_carry_the_full_source_specification(monkeypatch):
    """Qodo PR-2751 finding 3: display metadata alone is not enough.

    The record must carry the complete evaluation definition (metric,
    dataset, generation arguments, prompts, filters, ...) so the
    preview/create/export/select paths receive what the source template
    actually specifies.
    """
    gsm8k_spec = {
        "name": "GSM8K",
        "description": "Grade school math word problems from GSM8K dataset",
        "task_type": "question_answer",
        "metric": "exact_match",
        "dataset_name": "gsm8k",
        "dataset_config": "main",
        "split": "test",
        "generation_kwargs": {"temperature": 0.0, "max_tokens": 512},
        "doc_to_text": "Question: {question}\nAnswer:",
        "doc_to_target": "{answer}",
        "filter_list": [
            {
                "filter": "regex",
                "regex_pattern": r"####\s*([+-]?\d+(?:\.\d+)?)",
                "group": 1,
            }
        ],
        "requires_reasoning": True,
        "difficulty": "elementary",
        "metadata": {"category": "reasoning", "subcategory": "mathematical"},
    }
    _install_fake_manager(monkeypatch, reasoning={"gsm8k": gsm8k_spec})

    records = _builtin_eval_template_records()

    assert len(records) == 1
    record = records[0]
    # Display contract is unchanged (id stays the record name).
    assert record["name"] == "gsm8k"
    assert record["display_name"] == "GSM8K"
    assert record["category"] == "reasoning"
    # The full source specification is carried through.
    for key in (
        "description",
        "task_type",
        "difficulty",
        "metric",
        "dataset_name",
        "dataset_config",
        "split",
        "generation_kwargs",
        "doc_to_text",
        "doc_to_target",
        "filter_list",
        "requires_reasoning",
        "metadata",
    ):
        assert record[key] == gsm8k_spec[key], key


def test_record_mapping_follows_manager_category_order(monkeypatch):
    """Qodo PR-2751 finding 2: field names and categories map correctly."""
    _install_fake_manager(
        monkeypatch,
        reasoning={
            "logical_reasoning": {
                "name": "Logical Reasoning",
                "description": "Tests logical deduction.",
                "task_type": "question_answer",
                "difficulty": "advanced",
                "metadata": {"category": "reasoning", "subcategory": "deduction"},
            }
        },
        # coding sits between language and safety in the manager's order,
        # but is empty: it must contribute nothing.
        coding={},
        language={
            "translation": {
                "name": "Translation",
                "description": "Translates text between languages.",
                "task_type": "generation",
                "difficulty": "intermediate",
                "metadata": {"category": "language"},
            }
        },
    )

    records = _builtin_eval_template_records()

    assert [record["name"] for record in records] == [
        "logical_reasoning",
        "translation",
    ]
    first, second = records
    assert first["display_name"] == "Logical Reasoning"
    assert first["description"] == "Tests logical deduction."
    assert first["category"] == "reasoning"
    assert first["task_type"] == "question_answer"
    assert first["difficulty"] == "advanced"
    assert second["display_name"] == "Translation"
    assert second["category"] == "language"
    assert second["task_type"] == "generation"
    assert second["difficulty"] == "intermediate"


def test_record_fallbacks_for_missing_display_fields(monkeypatch):
    """Qodo PR-2751 finding 2: fallbacks for empty/missing spec fields."""
    _install_fake_manager(monkeypatch, safety={"bare": {}})

    records = _builtin_eval_template_records()

    assert len(records) == 1
    record = records[0]
    assert record["name"] == "bare"
    # Spec carried no display name: fall back to the template id.
    assert record["display_name"] == "bare"
    assert record["description"] == ""
    # Spec carried no metadata: fall back to the manager's category.
    assert record["category"] == "safety"
    assert record["task_type"] == ""
    assert record["difficulty"] == "Unknown"


def test_manager_with_no_templates_yields_no_records(monkeypatch):
    """Qodo PR-2751 finding 2: all-empty categories yield an empty list."""
    _install_fake_manager(monkeypatch)

    assert _builtin_eval_template_records() == []
