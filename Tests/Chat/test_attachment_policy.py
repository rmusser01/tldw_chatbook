"""Policy-function and drift-by-construction tests (TASK-222)."""

import tomllib

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.Chat import attachment_core
from tldw_chatbook.Chat.attachment_core import (
    DEFAULT_RESIZE_MAX_DIMENSION,
    DEFAULT_SUPPORTED_IMAGE_FORMATS,
    MAX_IMAGE_BYTES,
    attachment_filter_specs,
    image_resize_max_dimension,
    max_image_bytes,
    supported_image_formats,
)


@pytest.fixture
def defaults_config(monkeypatch):
    """Simulate a config with no [chat.images] overrides (never read live config)."""
    monkeypatch.setattr(
        config_mod, "get_cli_setting",
        lambda section, key=None, default=None: default,
    )


@pytest.fixture
def svg_on(monkeypatch):
    monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: True)


@pytest.fixture
def config_override(monkeypatch):
    """Return a setter that overrides one [chat.images] key, defaults elsewhere."""

    def _set(key, value):
        monkeypatch.setattr(
            config_mod, "get_cli_setting",
            lambda section, k=None, default=None: (
                value if section == "chat.images" and k == key else default
            ),
        )

    return _set


class TestSupportedImageFormats:
    def test_defaults(self, defaults_config, svg_on):
        assert supported_image_formats() == DEFAULT_SUPPORTED_IMAGE_FORMATS

    def test_svg_dropped_when_unavailable(self, defaults_config, monkeypatch):
        monkeypatch.setattr(attachment_core, "svg_rendering_available", lambda: False)
        formats = supported_image_formats()
        assert ".svg" not in formats
        assert formats == tuple(
            f for f in DEFAULT_SUPPORTED_IMAGE_FORMATS if f != ".svg"
        )

    def test_normalization(self, config_override, svg_on):
        config_override(
            "supported_formats", ["PNG", "jpg", ".JPEG", "png", 42, "  .webp "]
        )
        assert supported_image_formats() == (".png", ".jpg", ".jpeg", ".webp")

    def test_invalid_value_falls_back_to_defaults(self, config_override, svg_on):
        config_override("supported_formats", "not-a-list")
        assert supported_image_formats() == DEFAULT_SUPPORTED_IMAGE_FORMATS

    def test_empty_list_falls_back_to_defaults(self, config_override, svg_on):
        config_override("supported_formats", [])
        assert supported_image_formats() == DEFAULT_SUPPORTED_IMAGE_FORMATS


class TestCaps:
    def test_max_image_bytes_default(self, defaults_config):
        assert max_image_bytes() == MAX_IMAGE_BYTES

    def test_max_image_bytes_override(self, config_override):
        config_override("max_size_mb", 2.5)
        assert max_image_bytes() == int(2.5 * 1024 * 1024)

    def test_max_image_bytes_invalid(self, config_override):
        config_override("max_size_mb", -3)
        assert max_image_bytes() == MAX_IMAGE_BYTES

    def test_max_image_bytes_non_numeric(self, config_override):
        config_override("max_size_mb", "lots")
        assert max_image_bytes() == MAX_IMAGE_BYTES

    def test_resize_dimension_default(self, defaults_config):
        assert image_resize_max_dimension() == DEFAULT_RESIZE_MAX_DIMENSION

    def test_resize_dimension_override(self, config_override):
        config_override("resize_max_dimension", 512)
        assert image_resize_max_dimension() == 512

    def test_resize_dimension_invalid(self, config_override):
        config_override("resize_max_dimension", 0)
        assert image_resize_max_dimension() == DEFAULT_RESIZE_MAX_DIMENSION


class TestFilterSpecsDrift:
    def test_image_row_derives_from_formats(self, defaults_config, svg_on):
        specs = attachment_filter_specs()
        expected = ";".join(f"*{ext}" for ext in supported_image_formats())
        assert specs[1] == ("Image Files", expected)

    def test_all_files_row_leads_with_image_patterns(self, defaults_config, svg_on):
        specs = attachment_filter_specs()
        image_patterns = ";".join(f"*{ext}" for ext in supported_image_formats())
        assert specs[0][0] == "All Supported Files"
        assert specs[0][1].startswith(image_patterns + ";")
        # non-image tail preserved verbatim from the legacy literal
        assert specs[0][1].endswith("*.epub;*.mobi;*.azw;*.azw3;*.fb2")

    def test_specs_follow_config_narrowing(self, config_override, svg_on):
        config_override("supported_formats", [".png"])
        specs = attachment_filter_specs()
        assert specs[1] == ("Image Files", "*.png")

    def test_non_image_rows_unchanged(self, defaults_config, svg_on):
        labels = [label for label, _ in attachment_filter_specs()]
        assert labels == [
            "All Supported Files", "Image Files", "Document Files",
            "E-book Files", "Text Files", "Code Files", "Data Files",
        ]


def test_config_template_matches_policy_default():
    """CONFIG_TOML_CONTENT's [chat.images].supported_formats == the policy default."""
    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    parsed = tomllib.loads(CONFIG_TOML_CONTENT)
    assert parsed["chat"]["images"]["supported_formats"] == list(
        DEFAULT_SUPPORTED_IMAGE_FORMATS
    )
