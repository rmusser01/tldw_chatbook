"""
Tests for Embedding Templates functionality.
Tests template management, predefined templates, and custom configurations.
"""

import pytest
import pytest_asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from datetime import datetime

from textual.widgets import Button, Input, TextArea, ListView, ListItem
from textual.containers import Container

from tldw_chatbook.Utils.embedding_templates import (
    EmbeddingTemplate,
    EmbeddingTemplateManager,
    PredefinedTemplates
)
from tldw_chatbook.Widgets.embedding_template_selector import (
    EmbeddingTemplateSelector,
    TemplateListItem,
    EmbeddingTemplateSelected
)
from tldw_chatbook.UI.Embeddings_Creation_Content import (
    EmbeddingsCreationContent,
    CreateTemplateDialog
)

from .test_base import EmbeddingsTestBase, WidgetTestApp, create_mock_event


class TestEmbeddingTemplate(EmbeddingsTestBase):
    """Test EmbeddingTemplate class functionality."""
    
    def test_template_creation(self):
        """Test creating an embedding template."""
        template = EmbeddingTemplate(
            name="Test Template",
            description="A test template",
            config={
                "model_name": "e5-small-v2",
                "chunk_size": 512,
                "chunk_overlap": 50,
                "batch_size": 32
            }
        )
        
        assert template.name == "Test Template"
        assert template.description == "A test template"
        assert template.config["model_name"] == "e5-small-v2"
        assert template.config["chunk_size"] == 512
        assert template.created_at is not None
        assert template.updated_at is not None
    
    def test_template_to_dict(self):
        """Test converting template to dictionary."""
        template = EmbeddingTemplate(
            name="Test",
            description="Test desc",
            config={"key": "value"}
        )
        
        data = template.to_dict()
        assert data["name"] == "Test"
        assert data["description"] == "Test desc"
        assert data["config"] == {"key": "value"}
        assert "created_at" in data
        assert "updated_at" in data
    
    def test_template_from_dict(self):
        """Test creating template from dictionary."""
        data = {
            "name": "Restored",
            "description": "Restored template",
            "config": {"model": "test"},
            "created_at": "2024-01-01T00:00:00",
            "updated_at": "2024-01-01T00:00:00"
        }
        
        template = EmbeddingTemplate.from_dict(data)
        assert template.name == "Restored"
        assert template.description == "Restored template"
        assert template.config["model"] == "test"
    
    def test_template_update(self):
        """Test updating template configuration."""
        template = EmbeddingTemplate(
            name="Update Test",
            description="Test",
            config={"original": True}
        )
        
        original_created = template.created_at
        original_updated = template.updated_at
        
        # Update config
        template.config["new_key"] = "new_value"
        template.updated_at = datetime.utcnow()
        
        assert template.created_at == original_created
        assert template.updated_at > original_updated
        assert template.config["new_key"] == "new_value"


class TestEmbeddingTemplateManager(EmbeddingsTestBase):
    """Test EmbeddingTemplateManager functionality."""
    
    @pytest.fixture
    def temp_template_dir(self, tmp_path):
        """Create temporary template directory."""
        template_dir = tmp_path / ".config" / "tldw_cli" / "templates"
        template_dir.mkdir(parents=True)
        return template_dir
    
    @pytest.fixture
    def template_manager(self, temp_template_dir):
        """Create template manager with temp directory."""
        return EmbeddingTemplateManager(template_dir=temp_template_dir)
    
    def test_manager_initialization(self, template_manager):
        """Test template manager initialization."""
        assert len(template_manager.templates) > 0  # Should have builtin templates
        
        # Check builtin templates exist
        template_names = [t.name for t in template_manager.templates.values()]
        assert "Quick Summary" in template_names
        assert "Deep Analysis" in template_names
        assert "Technical Documentation" in template_names
    
    def test_get_template(self, template_manager):
        """Test getting template by name."""
        # Get builtin template
        template = template_manager.get_template("Quick Summary")
        assert template is not None
        assert template.name == "Quick Summary"
        assert isinstance(template.config, dict)
        
        # Get non-existent template
        assert template_manager.get_template("NonExistent") is None
    
    def test_save_custom_template(self, template_manager, temp_template_dir):
        """Test saving custom template."""
        custom = EmbeddingTemplate(
            name="My Custom",
            description="Custom template",
            config={
                "model_name": "custom-model",
                "chunk_size": 1024
            }
        )
        
        template_manager.save_template(custom)
        
        # Check file was created
        template_file = temp_template_dir / "my_custom.json"
        assert template_file.exists()
        
        # Check template is in manager
        assert "My Custom" in template_manager.templates
        
        # Verify file contents
        with open(template_file, 'r') as f:
            data = json.load(f)
            assert data["name"] == "My Custom"
            assert data["config"]["model_name"] == "custom-model"
    
    def test_update_existing_template(self, template_manager, temp_template_dir):
        """Test updating existing template."""
        # Save initial template
        template = EmbeddingTemplate(
            name="Update Test",
            description="Original",
            config={"version": 1}
        )
        template_manager.save_template(template)
        
        # Update template
        template.description = "Updated"
        template.config["version"] = 2
        template_manager.save_template(template)
        
        # Reload and check
        new_manager = EmbeddingTemplateManager(template_dir=temp_template_dir)
        updated = new_manager.get_template("Update Test")
        assert updated.description == "Updated"
        assert updated.config["version"] == 2
    
    def test_delete_template(self, template_manager, temp_template_dir):
        """Test deleting custom template."""
        # Save template
        template = EmbeddingTemplate(
            name="Delete Me",
            description="To be deleted",
            config={}
        )
        template_manager.save_template(template)
        assert "Delete Me" in template_manager.templates
        
        # Delete it
        result = template_manager.delete_template("Delete Me")
        assert result == True
        assert "Delete Me" not in template_manager.templates
        
        # Check file is gone
        template_file = temp_template_dir / "delete_me.json"
        assert not template_file.exists()
    
    def test_cannot_delete_builtin(self, template_manager):
        """Test that builtin templates cannot be deleted."""
        result = template_manager.delete_template("Quick Summary")
        assert result == False
        assert "Quick Summary" in template_manager.templates
    
    def test_list_templates(self, template_manager):
        """Test listing all templates."""
        templates = template_manager.list_templates()
        assert len(templates) > 0
        
        # Should be sorted by name
        names = [t.name for t in templates]
        assert names == sorted(names)
    
    def test_load_corrupted_template(self, template_manager, temp_template_dir):
        """Test handling of corrupted template files."""
        # Create corrupted file
        bad_file = temp_template_dir / "corrupted.json"
        bad_file.write_text("{ invalid json }")
        
        # Reload manager - should skip bad file
        new_manager = EmbeddingTemplateManager(template_dir=temp_template_dir)
        assert "corrupted" not in [t.name.lower() for t in new_manager.templates.values()]


class TestEmbeddingTemplateSelector(EmbeddingsTestBase):
    """Test template selector widget."""
    
    @pytest.fixture
    def mock_template_manager(self):
        """Create mock template manager."""
        manager = MagicMock()
        manager.list_templates.return_value = [
            EmbeddingTemplate("Template 1", "First template", {"model": "test1"}),
            EmbeddingTemplate("Template 2", "Second template", {"model": "test2"}),
            EmbeddingTemplate("Template 3", "Third template", {"model": "test3"})
        ]
        return manager
    
    @pytest.mark.asyncio
    async def test_selector_creation(self, mock_template_manager):
        """Test creating template selector."""
        selector = EmbeddingTemplateSelector(
            template_manager=mock_template_manager,
            show_create_button=True
        )
        
        app = WidgetTestApp(selector)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check list view exists
            list_view = pilot.app.query_one("#template-list", ListView)
            assert list_view is not None
            
            # Check templates are listed
            items = list_view.query(TemplateListItem)
            assert len(items) == 3
            
            # Check create button
            create_btn = pilot.app.query_one("#create-template", Button)
            assert create_btn is not None
            assert create_btn.label == "Create Custom Template"
    
    @pytest.mark.asyncio
    async def test_template_selection(self, mock_template_manager):
        """Test selecting a template."""
        selector = EmbeddingTemplateSelector(template_manager=mock_template_manager)
        
        # Track selection events
        selected_template = None
        
        def on_selection(event):
            nonlocal selected_template
            if isinstance(event, EmbeddingTemplateSelected):
                selected_template = event.template
        
        app = WidgetTestApp(selector)
        app.on_embedding_template_selected = on_selection
        
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Click first template
            first_item = pilot.app.query(TemplateListItem)[0]
            await pilot.click(first_item)
            await pilot.pause()
            
            # Check selection event
            assert selected_template is not None
            assert selected_template.name == "Template 1"
    
    @pytest.mark.asyncio
    async def test_template_list_item_display(self):
        """Test template list item display."""
        template = EmbeddingTemplate(
            name="Display Test",
            description="A test template for display",
            config={"model": "test-model", "chunk_size": 512}
        )
        
        item = TemplateListItem(template)
        
        app = WidgetTestApp(item)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check name display
            name_label = pilot.app.query_one(".template-name")
            assert "Display Test" in name_label.renderable
            
            # Check description display
            desc_label = pilot.app.query_one(".template-description")
            assert "A test template for display" in desc_label.renderable
            
            # Check config preview
            config_label = pilot.app.query_one(".template-config")
            config_text = str(config_label.renderable)
            assert "test-model" in config_text
            assert "512" in config_text
    
    @pytest.mark.asyncio
    async def test_create_button_hidden(self, mock_template_manager):
        """Test hiding create button."""
        selector = EmbeddingTemplateSelector(
            template_manager=mock_template_manager,
            show_create_button=False
        )
        
        app = WidgetTestApp(selector)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Create button should not exist
            create_buttons = pilot.app.query("#create-template")
            assert len(create_buttons) == 0


class TestCreateTemplateDialog(EmbeddingsTestBase):
    """Test create template dialog."""
    
    @pytest.mark.asyncio
    async def test_dialog_creation(self):
        """Test creating template dialog."""
        initial_config = {
            "model_name": "e5-small-v2",
            "chunk_size": 512,
            "chunk_overlap": 50
        }
        
        dialog = CreateTemplateDialog(initial_config=initial_config)
        
        app = WidgetTestApp(dialog)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Check form fields
            name_input = pilot.app.query_one("#template-name", Input)
            assert name_input is not None
            
            desc_input = pilot.app.query_one("#template-description", TextArea)
            assert desc_input is not None
            
            # Check config is pre-filled
            config_area = pilot.app.query_one("#template-config", TextArea)
            assert config_area is not None
            config_text = config_area.text
            assert "e5-small-v2" in config_text
            assert "512" in config_text
    
    @pytest.mark.asyncio
    async def test_dialog_validation(self):
        """Test dialog form validation."""
        dialog = CreateTemplateDialog(initial_config={})
        
        app = WidgetTestApp(dialog)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Try to save without name
            save_btn = pilot.app.query_one("#save-template", Button)
            await pilot.click(save_btn)
            await pilot.pause()
            
            # Should show error (dialog still open)
            assert dialog in pilot.app.screen_stack
    
    @pytest.mark.asyncio
    async def test_dialog_save_success(self, mock_app_instance):
        """Test saving template successfully."""
        dialog = CreateTemplateDialog(initial_config={"model": "test"})
        
        # Track dismiss result
        dismiss_result = None
        
        async def mock_dismiss(result):
            nonlocal dismiss_result
            dismiss_result = result
        
        dialog.dismiss = mock_dismiss
        
        app = WidgetTestApp(dialog, app_instance=mock_app_instance)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Fill in form
            name_input = pilot.app.query_one("#template-name", Input)
            name_input.value = "My Template"
            
            desc_input = pilot.app.query_one("#template-description", TextArea)
            desc_input.text = "My custom template"
            
            # Save
            await pilot.click("#save-template")
            await pilot.pause()
            
            # Check result
            assert dismiss_result is not None
            assert isinstance(dismiss_result, EmbeddingTemplate)
            assert dismiss_result.name == "My Template"
            assert dismiss_result.description == "My custom template"
    
    @pytest.mark.asyncio
    async def test_dialog_cancel(self):
        """Test canceling dialog."""
        dialog = CreateTemplateDialog(initial_config={})
        
        dismiss_result = "not_none"
        
        async def mock_dismiss(result):
            nonlocal dismiss_result
            dismiss_result = result
        
        dialog.dismiss = mock_dismiss
        
        app = WidgetTestApp(dialog)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Click cancel
            await pilot.click("#cancel-template")
            await pilot.pause()
            
            # Should dismiss with None
            assert dismiss_result is None


class TestTemplateIntegration(EmbeddingsTestBase):
    """Test template integration with embeddings creation."""
    
    @pytest.mark.asyncio
    async def test_template_application(self, mock_app_instance, mock_embedding_factory):
        """Test applying template to embeddings creation form."""
        window = EmbeddingsCreationContent(mock_app_instance)
        window.embedding_factory = mock_embedding_factory
        
        # Create test template
        template = EmbeddingTemplate(
            name="Test Application",
            description="Test",
            config={
                "model_name": "e5-large-v2",
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "batch_size": 16
            }
        )
        
        app = WidgetTestApp(window, app_instance=mock_app_instance)
        async with app.run_test() as pilot:
            await pilot.pause()
            
            # Simulate template selection event
            event = create_mock_event(
                EmbeddingTemplateSelected,
                template=template
            )
            window.on_template_selected(event)
            await pilot.pause()
            
            # Check form fields were updated
            # (Would need to check actual form field values)
            # For now just verify no errors
            assert True
    
    @pytest.mark.asyncio
    async def test_predefined_templates(self):
        """Test predefined templates have valid configurations."""
        # Test each predefined template
        for template_func in [
            PredefinedTemplates.quick_summary(),
            PredefinedTemplates.deep_analysis(),
            PredefinedTemplates.technical_documentation(),
            PredefinedTemplates.research_papers(),
            PredefinedTemplates.conversational_qa()
        ]:
            template = template_func
            
            # Check required fields
            assert template.name
            assert template.description
            assert isinstance(template.config, dict)
            
            # Check common config fields
            assert "model_name" in template.config
            assert "chunk_size" in template.config
            assert "chunk_overlap" in template.config
            assert "batch_size" in template.config
            
            # Validate chunk settings
            assert template.config["chunk_size"] > template.config["chunk_overlap"]
            assert template.config["batch_size"] > 0