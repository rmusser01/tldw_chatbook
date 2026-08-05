"""Tests for user-defined `/generate-image` style templates (Task-559 AC4).

Covers the two load sources (`[image_generation.styles]` config section,
`<user_data_dir>/image_generation_styles/*.toml` templates dir), their merge
order over `BUILTIN_TEMPLATES`, malformed-input skip-with-warning behavior,
and that `get_template`/`apply_template_to_prompt` (the seams
`console_generate_image.resolve_style_token`/`compose_styled_request` build
on) resolve user templates exactly like builtins.

Mirrors `Tests/Image_Generation/test_config_loader.py`'s patch-the-reader +
`reload=True`/reset-cache pattern, since `get_all_templates` is cached the
same way `get_image_generation_config` is.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Media_Creation import generation_templates as gt


@pytest.fixture(autouse=True)
def _reset_cache():
    gt.reset_templates_cache()
    yield
    gt.reset_templates_cache()


@pytest.fixture(autouse=True)
def _no_directory_templates(monkeypatch, tmp_path):
    """Default every test to an empty (non-existent) templates dir.

    Individual tests override `gt._user_templates_dir` again when they need
    real files -- this just keeps tests that only care about the config
    section from being polluted by files a differently-ordered test left in
    a shared user-data dir.
    """
    monkeypatch.setattr(
        gt, "_user_templates_dir", lambda: tmp_path / "unused_templates_dir", raising=False
    )
    monkeypatch.setattr(gt, "_read_style_config_section", lambda: {}, raising=False)


# ---------------------------------------------------------------------------
# Baseline: no user templates configured.
# ---------------------------------------------------------------------------


def test_get_all_templates_returns_only_builtins_when_unconfigured():
    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13
    assert set(merged) == set(gt.BUILTIN_TEMPLATES)


# ---------------------------------------------------------------------------
# Config-section templates.
# ---------------------------------------------------------------------------


def test_config_section_template_extends_the_set(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "base_prompt": "{{subject}}, soft glow lighting",
                "negative_prompt": "harsh lighting",
            }
        },
        raising=False,
    )
    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 14
    assert "my_glow" in merged
    template = merged["my_glow"]
    assert template.id == "my_glow"
    assert template.name == "My Glow"
    assert template.category == "Custom"
    assert template.base_prompt == "{{subject}}, soft glow lighting"
    assert template.negative_prompt == "harsh lighting"


def test_config_section_template_full_field_shape(monkeypatch):
    """Mirrors a builtin's shape exactly: description, default_params, context_mappings, tags."""
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "description": "Soft dreamy glow",
                "base_prompt": "{{subject}}, soft glow lighting",
                "negative_prompt": "harsh lighting",
                "default_params": {"width": 768, "height": 768, "steps": 28, "cfg_scale": 7.5},
                "context_mappings": {"subject": "last_message"},
                "tags": ["custom", "glow"],
            }
        },
        raising=False,
    )
    template = gt.get_all_templates(reload=True)["my_glow"]
    assert template.description == "Soft dreamy glow"
    assert template.default_params == {"width": 768, "height": 768, "steps": 28, "cfg_scale": 7.5}
    assert template.context_mappings == {"subject": "last_message"}
    assert template.tags == ["custom", "glow"]


def test_config_section_template_negative_prompt_defaults_when_absent(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "no_negative": {
                "name": "No Negative",
                "category": "Custom",
                "base_prompt": "{{subject}}",
            }
        },
        raising=False,
    )
    template = gt.get_all_templates(reload=True)["no_negative"]
    assert template.negative_prompt  # falls back to the dataclass default, non-empty


# ---------------------------------------------------------------------------
# Directory templates.
# ---------------------------------------------------------------------------


def _write_template_file(directory, stem: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.toml").write_text(content, encoding="utf-8")


def test_directory_template_extends_the_set(monkeypatch, tmp_path):
    templates_dir = tmp_path / "image_generation_styles"
    _write_template_file(
        templates_dir,
        "my_glow",
        """
        name = "My Glow"
        category = "Custom"
        base_prompt = "{{subject}}, soft glow lighting"
        negative_prompt = "harsh lighting"
        """,
    )
    monkeypatch.setattr(gt, "_user_templates_dir", lambda: templates_dir, raising=False)

    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 14
    template = merged["my_glow"]
    assert template.id == "my_glow"  # id comes from the FILENAME stem
    assert template.name == "My Glow"


def test_directory_template_id_is_filename_stem_not_internal_field(monkeypatch, tmp_path):
    """An `id` field inside the file (if present) is ignored -- the filename
    stem is authoritative, so a file can never spoof a different template's
    id."""
    templates_dir = tmp_path / "image_generation_styles"
    _write_template_file(
        templates_dir,
        "my_glow",
        """
        id = "style_anime"
        name = "My Glow"
        category = "Custom"
        base_prompt = "{{subject}}, soft glow lighting"
        """,
    )
    monkeypatch.setattr(gt, "_user_templates_dir", lambda: templates_dir, raising=False)

    merged = gt.get_all_templates(reload=True)
    assert merged["my_glow"].name == "My Glow"
    assert merged["style_anime"].name == "Anime Style"  # builtin, untouched


def test_directory_template_missing_dir_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(
        gt, "_user_templates_dir", lambda: tmp_path / "does-not-exist", raising=False
    )
    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13


# ---------------------------------------------------------------------------
# Merge/override ordering.
# ---------------------------------------------------------------------------


def test_user_template_overrides_builtin_by_id(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "style_anime": {
                "name": "Custom Anime",
                "category": "Style",
                "base_prompt": "{{subject}}, my custom anime look",
            }
        },
        raising=False,
    )
    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13  # still 13 ids -- overridden, not extended
    assert merged["style_anime"].name == "Custom Anime"


def test_directory_template_overrides_config_section_on_id_collision(monkeypatch, tmp_path):
    """Both sources define 'dup_style' -- the directory template wins."""
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "dup_style": {
                "name": "From Config",
                "category": "Custom",
                "base_prompt": "{{subject}}, from config",
            }
        },
        raising=False,
    )
    templates_dir = tmp_path / "image_generation_styles"
    _write_template_file(
        templates_dir,
        "dup_style",
        """
        name = "From Directory"
        category = "Custom"
        base_prompt = "{{subject}}, from directory"
        """,
    )
    monkeypatch.setattr(gt, "_user_templates_dir", lambda: templates_dir, raising=False)

    merged = gt.get_all_templates(reload=True)
    assert merged["dup_style"].name == "From Directory"


# ---------------------------------------------------------------------------
# Malformed input: skip with warning, never crash.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "record",
    [
        {},  # missing everything
        {"name": "Only Name"},  # missing category/base_prompt
        {"name": "", "category": "Custom", "base_prompt": "x"},  # blank name
        {"name": "N", "category": "C", "base_prompt": ""},  # blank base_prompt
        {"name": "N", "category": "C", "base_prompt": 123},  # wrong type
        "not-a-table",
        None,
        123,
    ],
)
def test_malformed_config_section_template_is_skipped(monkeypatch, record):
    monkeypatch.setattr(
        gt, "_read_style_config_section", lambda: {"broken": record}, raising=False
    )
    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13
    assert "broken" not in merged


def test_malformed_config_section_template_logs_warning(monkeypatch):
    monkeypatch.setattr(
        gt, "_read_style_config_section", lambda: {"broken": {}}, raising=False
    )
    warnings: list[str] = []
    monkeypatch.setattr(gt.logger, "warning", lambda msg: warnings.append(str(msg)), raising=False)
    gt.get_all_templates(reload=True)
    assert any("broken" in w for w in warnings)


def test_malformed_directory_template_file_is_skipped(monkeypatch, tmp_path):
    templates_dir = tmp_path / "image_generation_styles"
    _write_template_file(templates_dir, "broken", "name = \"Only Name\"\n")  # missing required fields
    monkeypatch.setattr(gt, "_user_templates_dir", lambda: templates_dir, raising=False)

    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13
    assert "broken" not in merged


def test_unparsable_toml_file_is_skipped(monkeypatch, tmp_path):
    templates_dir = tmp_path / "image_generation_styles"
    _write_template_file(templates_dir, "not_toml", "this is [ not valid toml =")
    monkeypatch.setattr(gt, "_user_templates_dir", lambda: templates_dir, raising=False)

    merged = gt.get_all_templates(reload=True)  # must not raise
    assert len(merged) == 13
    assert "not_toml" not in merged


def test_invalid_filename_stem_is_skipped(monkeypatch, tmp_path):
    templates_dir = tmp_path / "image_generation_styles"
    _write_template_file(
        templates_dir,
        "has spaces and !bang",
        """
        name = "Bad Filename"
        category = "Custom"
        base_prompt = "{{subject}}"
        """,
    )
    monkeypatch.setattr(gt, "_user_templates_dir", lambda: templates_dir, raising=False)

    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13


def test_invalid_config_section_id_is_skipped(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "bad id with spaces": {
                "name": "N",
                "category": "C",
                "base_prompt": "{{subject}}",
            }
        },
        raising=False,
    )
    merged = gt.get_all_templates(reload=True)
    assert len(merged) == 13


# ---------------------------------------------------------------------------
# Resolver + apply_template_to_prompt integration (the seams
# console_generate_image.py builds on).
# ---------------------------------------------------------------------------


def test_get_template_resolves_user_template(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "base_prompt": "{{subject}}, soft glow lighting",
                "negative_prompt": "harsh lighting",
                "context_mappings": {"subject": "last_message"},
            }
        },
        raising=False,
    )
    gt.get_all_templates(reload=True)
    template = gt.get_template("my_glow")
    assert template is not None
    assert template.name == "My Glow"


def test_apply_template_to_prompt_works_for_user_template(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "base_prompt": "{{subject}}, soft glow lighting",
                "negative_prompt": "harsh lighting",
                "context_mappings": {"subject": "last_message"},
                "default_params": {"width": 512},
            }
        },
        raising=False,
    )
    gt.get_all_templates(reload=True)
    prompt, negative, params = gt.apply_template_to_prompt(
        "my_glow", {"last_message": "a red dragon"}
    )
    assert prompt == "a red dragon, soft glow lighting"
    assert negative == "harsh lighting"
    assert params == {"width": 512}


def test_get_templates_by_category_includes_user_templates(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "base_prompt": "{{subject}}",
            }
        },
        raising=False,
    )
    gt.get_all_templates(reload=True)
    results = gt.get_templates_by_category("Custom")
    assert [t.id for t in results] == ["my_glow"]


def test_get_all_categories_includes_user_category(monkeypatch):
    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "base_prompt": "{{subject}}",
            }
        },
        raising=False,
    )
    gt.get_all_templates(reload=True)
    assert "Custom" in gt.get_all_categories()


def test_get_all_templates_cache_reload(monkeypatch):
    monkeypatch.setattr(gt, "_read_style_config_section", lambda: {}, raising=False)
    first = gt.get_all_templates(reload=True)
    assert len(first) == 13

    monkeypatch.setattr(
        gt,
        "_read_style_config_section",
        lambda: {
            "my_glow": {
                "name": "My Glow",
                "category": "Custom",
                "base_prompt": "{{subject}}",
            }
        },
        raising=False,
    )
    # Without reload the cache is stale.
    assert len(gt.get_all_templates()) == 13
    # With reload=True (or reset_templates_cache) it picks up the change.
    assert len(gt.get_all_templates(reload=True)) == 14
