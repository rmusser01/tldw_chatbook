# test_prompt_template_manager.py

import pytest
import json
from pathlib import Path

# Local Imports from this project
from tldw_chatbook.Chat.prompt_template_manager import (
    PromptTemplate,
    load_template,
    apply_template_to_string,
    get_available_templates,
    _loaded_templates,
    DEFAULT_RAW_PASSTHROUGH_TEMPLATE
)


# --- Test Setup ---

@pytest.fixture(autouse=True)
def clear_template_cache():
    """Fixture to clear the template cache before each test."""
    original_templates = _loaded_templates.copy()
    _loaded_templates.clear()
    # Ensure the default is always there for tests that might rely on it
    _loaded_templates["raw_passthrough"] = DEFAULT_RAW_PASSTHROUGH_TEMPLATE
    yield
    # Restore original cache state if needed, though clearing is usually sufficient
    _loaded_templates.clear()
    _loaded_templates.update(original_templates)


@pytest.fixture
def mock_templates_dir(tmp_path, monkeypatch):
    """Creates a temporary directory for prompt templates and patches the module-level constant."""
    templates_dir = tmp_path / "prompt_templates"
    templates_dir.mkdir()

    # Patch the PROMPT_TEMPLATES_DIR constant in the target module
    monkeypatch.setattr('tldw_chatbook.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR', templates_dir)

    # Create some dummy template files
    template1_data = {
        "name": "test_template_1",
        "description": "A simple test template.",
        "user_message_content_template": "User said: {{message_content}}"
    }
    (templates_dir / "test_template_1.json").write_text(json.dumps(template1_data))

    template2_data = {
        "name": "test_template_2",
        "system_message_template": "System context: {{system_context}}",
        "user_message_content_template": "{{message_content}}"
    }
    (templates_dir / "test_template_2.json").write_text(json.dumps(template2_data))

    # Create a malformed JSON file
    (templates_dir / "malformed.json").write_text("{'invalid': 'json'")

    return templates_dir


# --- Test Cases ---

class TestPromptTemplateManager:

    def test_load_template_success(self, mock_templates_dir):
        """Test successfully loading a valid template file."""
        template = load_template("test_template_1")
        assert template is not None
        assert isinstance(template, PromptTemplate)
        assert template.name == "test_template_1"
        assert template.user_message_content_template == "User said: {{message_content}}"

    def test_load_template_not_found(self, mock_templates_dir):
        """Test loading a template that does not exist."""
        template = load_template("non_existent_template")
        assert template is None

    def test_load_template_malformed_json(self, mock_templates_dir):
        """Test loading a template from a file with invalid JSON."""
        template = load_template("malformed")
        assert template is None

    def test_load_template_caching(self, mock_templates_dir):
        """Test that a loaded template is cached and not re-read from disk."""
        template1 = load_template("test_template_1")
        assert "test_template_1" in _loaded_templates

        # Modify the file on disk
        (mock_templates_dir / "test_template_1.json").write_text(json.dumps({"name": "modified"}))

        # Load again - should return the cached version
        template2 = load_template("test_template_1")
        assert template2 is not None
        assert template2.name == "test_template_1"  # Original name from cache
        assert template2 == template1

    def test_get_available_templates(self, mock_templates_dir):
        """Test discovering available templates from the directory."""
        available = get_available_templates()
        assert isinstance(available, list)
        assert set(available) == {"test_template_1", "test_template_2", "malformed"}

    def test_get_available_templates_no_dir(self, tmp_path, monkeypatch):
        """Test getting templates when the directory doesn't exist."""
        non_existent_dir = tmp_path / "non_existent_dir"
        monkeypatch.setattr('tldw_chatbook.Chat.prompt_template_manager.PROMPT_TEMPLATES_DIR', non_existent_dir)
        assert get_available_templates() == []

    def test_default_passthrough_template_is_available(self):
        """Test that the default 'raw_passthrough' template is loaded."""
        template = load_template("raw_passthrough")
        assert template is not None
        assert template.name == "raw_passthrough"
        assert template.user_message_content_template == "{{message_content}}"


class TestTemplateRendering:

    def test_apply_template_to_string_success(self):
        """Test basic successful rendering."""
        template_str = "Hello, {{ name }}!"
        data = {"name": "World"}
        result = apply_template_to_string(template_str, data)
        assert result == "Hello, World!"

    def test_apply_template_to_string_missing_placeholder(self):
        """Test rendering when a placeholder in the template is not in the data."""
        template_str = "Hello, {{ name }}! Your age is {{ age }}."
        data = {"name": "World"}  # 'age' is missing
        result = apply_template_to_string(template_str, data)
        assert result == "Hello, World! Your age is ."  # Jinja renders missing variables as empty strings

    def test_apply_template_with_none_input_string(self):
        """Test that a None template string returns an empty string."""
        data = {"name": "World"}
        result = apply_template_to_string(None, data)
        assert result == ""

    def test_apply_template_with_complex_data(self):
        """Test rendering with more complex data structures like lists and dicts."""
        template_str = "User: {{ user.name }}. Items: {% for item in items %}{{ item }}{% if not loop.last %}, {% endif %}{% endfor %}."
        data = {
            "user": {"name": "Alice"},
            "items": ["apple", "banana", "cherry"]
        }
        result = apply_template_to_string(template_str, data)
        assert result == "User: Alice. Items: apple, banana, cherry."

    def test_safe_render_prevents_unsafe_operations(self):
        """Test that the sandboxed environment prevents access to unsafe attributes."""
        # Attempt to access a private attribute or a method that could be unsafe
        template_str = "Unsafe access: {{ my_obj.__class__ }}"

        class MyObj: pass

        data = {"my_obj": MyObj()}

        # In a sandboxed environment, unsafe attributes are silently filtered out
        result = apply_template_to_string(template_str, data)
        assert result == "Unsafe access: "