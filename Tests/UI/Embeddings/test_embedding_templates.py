"""
Tests for the current embeddings template utilities and widgets.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Utils.embedding_templates import (
    EmbeddingTemplate,
    EmbeddingTemplateManager,
    TemplateCategory,
)
from tldw_chatbook.Widgets.embedding_template_selector import (
    EmbeddingTemplateCard,
    EmbeddingTemplateQuickSelect,
)

from .test_base import EmbeddingsTestBase, WidgetTestApp


def make_template(
    *,
    template_id: str = "test_template",
    name: str = "Test Template",
    category: TemplateCategory = TemplateCategory.QUICK_START,
    config: dict | None = None,
) -> EmbeddingTemplate:
    """Create a minimal valid template for tests."""
    return EmbeddingTemplate(
        id=template_id,
        name=name,
        description="A test template",
        category=category,
        config=config or {"model_id": "e5-small-v2", "chunk_size": 1000},
        recommended_for=["Smoke tests"],
        pros=["Simple"],
        cons=["Limited"],
    )


def _static_text(widget: Static) -> str:
    return str(widget.render())


class TestEmbeddingTemplate(EmbeddingsTestBase):
    """Test current EmbeddingTemplate dataclass behavior."""

    def test_template_creation(self):
        template = make_template()

        assert template.id == "test_template"
        assert template.name == "Test Template"
        assert template.category == TemplateCategory.QUICK_START
        assert template.config["model_id"] == "e5-small-v2"

    def test_template_fields_are_export_ready(self):
        template = make_template(
            template_id="custom_template",
            name="Custom",
            config={"model_id": "custom-model", "batch_size": 16},
        )

        assert template.id == "custom_template"
        assert template.description == "A test template"
        assert template.recommended_for == ["Smoke tests"]
        assert template.pros == ["Simple"]
        assert template.cons == ["Limited"]


class TestEmbeddingTemplateManager(EmbeddingsTestBase):
    """Test template manager operations."""

    def test_manager_initialization_loads_builtin_templates(self):
        manager = EmbeddingTemplateManager()

        templates = manager.get_all_templates()
        assert len(templates) >= 5
        assert manager.get_template("quick_local") is not None
        assert manager.get_template("premium_openai") is not None

    def test_get_templates_by_category(self):
        manager = EmbeddingTemplateManager()

        quick_start = manager.get_templates_by_category(TemplateCategory.QUICK_START)
        assert quick_start
        assert all(template.category == TemplateCategory.QUICK_START for template in quick_start)

    def test_apply_template_merges_with_base_config(self):
        manager = EmbeddingTemplateManager()

        merged = manager.apply_template(
            "quick_local",
            {"chunk_size": 250, "custom_flag": True},
        )

        assert merged["chunk_size"] == 1000
        assert merged["custom_flag"] is True
        assert merged["model_id"] == "e5-small-v2"

    def test_apply_template_raises_for_unknown_id(self):
        manager = EmbeddingTemplateManager()

        with pytest.raises(ValueError):
            manager.apply_template("missing-template", {})

    def test_export_and_import_template_round_trip(self):
        manager = EmbeddingTemplateManager()
        custom = make_template(
            template_id="custom_round_trip",
            name="Round Trip",
            category=TemplateCategory.SPECIALIZED,
            config={"model_id": "custom-model", "batch_size": 8},
        )

        manager.add_template(custom)
        exported = manager.export_template("custom_round_trip")

        new_manager = EmbeddingTemplateManager()
        new_manager.import_template(exported)
        restored = new_manager.get_template("custom_round_trip")

        assert restored is not None
        assert restored.name == "Round Trip"
        assert restored.category == TemplateCategory.SPECIALIZED
        assert restored.config["batch_size"] == 8

    def test_template_summary_includes_key_sections(self):
        manager = EmbeddingTemplateManager()

        summary = manager.get_template_summary("quick_local")
        assert "Quick Local Setup" in summary
        assert "Recommended for" in summary
        assert "Configuration" in summary


class TestEmbeddingTemplateWidgets(EmbeddingsTestBase):
    """Test the current selector widgets."""

    @pytest.mark.asyncio
    async def test_template_card_renders_core_content(self):
        card = EmbeddingTemplateCard(make_template(name="Display Test"))

        app = WidgetTestApp(card)
        async with app.run_test() as pilot:
            await pilot.pause()

            title = pilot.app.query_one(".embedding-template-card-title", Static)
            description = pilot.app.query_one(".embedding-template-card-description", Static)
            category = pilot.app.query_one(".embedding-template-card-category", Static)
            action = pilot.app.query_one("#select-test_template", Button)

            assert "Display Test" in _static_text(title)
            assert "A test template" in _static_text(description)
            assert "Quick Start" in _static_text(category)
            assert action.label == "Use This Template"

    @pytest.mark.asyncio
    async def test_quick_select_renders_common_actions(self):
        quick_select = EmbeddingTemplateQuickSelect()

        app = WidgetTestApp(quick_select)
        async with app.run_test() as pilot:
            await pilot.pause()

            assert pilot.app.query_one("#template-quick_local", Button).label == "Quick Start"
            assert pilot.app.query_one("#template-high_quality_local", Button).label == "High Quality"
            assert pilot.app.query_one("#template-balanced_performance", Button).label == "Performance"
            assert pilot.app.query_one("#template-browse", Button).label == "Browse All..."

    @pytest.mark.asyncio
    async def test_quick_select_apply_template_updates_state(self):
        quick_select = EmbeddingTemplateQuickSelect()

        app = WidgetTestApp(quick_select)
        async with app.run_test() as pilot:
            await pilot.pause()

            template = quick_select.template_manager.get_template("quick_local")
            assert template is not None

            quick_select._apply_template(template)
            await pilot.pause()

            current_name = pilot.app.query_one("#current-template-name", Static)
            assert quick_select.current_template == "quick_local"
            assert "Using: Quick Local Setup" in _static_text(current_name)

            config = quick_select.get_current_config()
            assert config is not None
            assert config["model_id"] == "e5-small-v2"
            assert config is not template.config

    @pytest.mark.asyncio
    async def test_quick_select_handle_none_selection_is_noop(self):
        quick_select = EmbeddingTemplateQuickSelect()

        app = WidgetTestApp(quick_select)
        async with app.run_test() as pilot:
            await pilot.pause()

            quick_select._handle_template_selection(None)
            await pilot.pause()

            assert quick_select.current_template is None
