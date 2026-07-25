"""Tests for the Settings > Image Gen defaults data layer (task-2 of the
Settings > Image Gen plan): FIELD_SCHEMA, build_backend_rows,
effective_placeholder, ImageGenDraftValues, diff_to_sections, validate_draft,
and the adapter's delete_values wrapper.
"""

from __future__ import annotations

import tomllib

import pytest
import toml

from tldw_chatbook.Image_Generation.config import _NON_SECRET, _SECRETS
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_image_gen_defaults import (
    BACKEND_IDS,
    BACKEND_LABELS,
    FIELD_SCHEMA,
    ImageGenDraftValues,
    build_backend_rows,
    diff_to_sections,
    effective_placeholder,
    validate_draft,
)


@pytest.fixture(autouse=True)
def _reset_image_gen_cache():
    from tldw_chatbook.Image_Generation import config as img_cfg

    img_cfg.reset_image_generation_config_cache()
    yield
    img_cfg.reset_image_generation_config_cache()


def _draft(
    *,
    default_backend=None,
    enabled_backends=None,
    default_batch=None,
    max_variants_per_message=None,
    context_llm_enabled=None,
    context_llm_turns=None,
    context_llm_timeout_seconds=None,
    backend_fields=None,
    cleared_fields=None,
) -> ImageGenDraftValues:
    """Build a draft. Unspecified scalar/global fields stay at the
    "untouched this session" sentinel (None), which diff_to_sections treats
    as "nothing to write" -- only `enabled_backends` defaults to [] (its
    declared type is `list[str]`, never Optional)."""
    return ImageGenDraftValues(
        default_backend=default_backend,
        enabled_backends=list(enabled_backends) if enabled_backends is not None else [],
        default_batch=default_batch,
        max_variants_per_message=max_variants_per_message,
        context_llm_enabled=context_llm_enabled,
        context_llm_turns=context_llm_turns,
        context_llm_timeout_seconds=context_llm_timeout_seconds,
        backend_fields=backend_fields or {},
        cleared_fields=cleared_fields or {},
    )


_ALL_SECRET_ENV_VARS = (
    "OPENROUTER_API_KEY",
    "NOVITA_API_KEY",
    "TOGETHER_API_KEY",
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "SWARMUI_TOKEN",
)


def _fake_cfg(monkeypatch, *, section=None, env=None, keyring=None):
    """A real ImageGenerationConfig built through the loader with a crafted
    raw section + env + keyring -- sturdier than a Mock, since it exercises
    the real key_sources precedence (matches Task 1's test helper pattern:
    Tests/Image_Generation/test_config_loader.py::_load_config_with_section).
    """
    from tldw_chatbook.Image_Generation import config as img_cfg

    for var in _ALL_SECRET_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(img_cfg, "_read_image_generation_toml", lambda: section or {}, raising=False)
    kr = keyring or {}
    monkeypatch.setattr(img_cfg, "_keyring_get", lambda backend: kr.get(backend), raising=False)
    return img_cfg.get_image_generation_config(reload=True)


# --- FIELD_SCHEMA / BACKEND_IDS / BACKEND_LABELS -----------------------------


def test_field_schema_maps_to_real_loader_keys():
    """Drift guard: every schema field must be a key the loader actually reads."""
    for backend, specs in FIELD_SCHEMA.items():
        for spec in specs:
            if spec.kind == "secret":
                assert _SECRETS[backend], backend
            else:
                assert (backend, spec.toml_key) in _NON_SECRET


def test_field_schema_covers_every_backend_id():
    assert set(FIELD_SCHEMA) == set(BACKEND_IDS)


def test_backend_labels_cover_every_backend_id():
    assert set(BACKEND_LABELS) == set(BACKEND_IDS)


# --- diff_to_sections ---------------------------------------------------------


def test_diff_emits_only_changed_keys_to_nested_sections():
    draft = _draft(backend_fields={"openrouter": {"default_model": "openai/gpt-5-image-mini"}})
    sections, deletions = diff_to_sections(draft, raw_config={"image_generation": {}})
    assert sections == {"image_generation.openrouter": {"default_model": "openai/gpt-5-image-mini"}}
    assert deletions == {}


def test_diff_never_copies_env_resolved_secret(monkeypatch):
    """THE no-secret-copy pin: effective cfg holds env-resolved keys; the diff
    must not see them because it only reads the draft + raw config."""
    draft = _draft()  # user typed nothing
    sections, _ = diff_to_sections(draft, raw_config={"image_generation": {}})
    flat = {k: v for sec in sections.values() for k, v in sec.items()}
    assert "api_key" not in flat and "swarm_token" not in flat


def test_cleared_field_becomes_deletion_not_empty_write():
    draft = _draft(cleared_fields={"openrouter": ["default_model"]})
    sections, deletions = diff_to_sections(
        draft, raw_config={"image_generation": {"openrouter": {"default_model": "x"}}}
    )
    assert "default_model" not in sections.get("image_generation.openrouter", {})
    assert deletions == {"image_generation.openrouter": ["default_model"]}


def test_diff_coerces_int_fields_and_skips_unchanged_int():
    """An int field that round-trips to the same value as raw must not be
    re-emitted -- proves coercion happens before the equality check."""
    draft = _draft(backend_fields={"openrouter": {"timeout_seconds": "120"}})
    sections, _ = diff_to_sections(
        draft, raw_config={"image_generation": {"openrouter": {"timeout_seconds": 120}}}
    )
    assert sections == {}


def test_diff_emits_coerced_int_when_it_actually_changed():
    draft = _draft(backend_fields={"openrouter": {"timeout_seconds": "45"}})
    sections, _ = diff_to_sections(
        draft, raw_config={"image_generation": {"openrouter": {"timeout_seconds": 120}}}
    )
    assert sections == {"image_generation.openrouter": {"timeout_seconds": 45}}


def test_diff_edit_and_clear_same_key_is_a_deletion_not_a_write():
    """A key present in both backend_fields and cleared_fields must resolve
    to deletion, never a write of the stale edit."""
    draft = _draft(
        backend_fields={"openrouter": {"default_model": "typed-then-cleared"}},
        cleared_fields={"openrouter": ["default_model"]},
    )
    sections, deletions = diff_to_sections(draft, raw_config={"image_generation": {}})
    assert "default_model" not in sections.get("image_generation.openrouter", {})
    assert deletions == {"image_generation.openrouter": ["default_model"]}


def test_diff_emits_changed_global_keys_under_top_level_section():
    draft = _draft(default_batch=3, context_llm_enabled=False)
    sections, _ = diff_to_sections(draft, raw_config={"image_generation": {"default_batch": 1}})
    assert sections == {"image_generation": {"default_batch": 3, "context_llm_enabled": False}}


def test_diff_untouched_enabled_backends_matches_absent_raw_list():
    """enabled_backends defaults to [] (never None); an absent raw key must
    normalize to [] too so an unedited draft doesn't spuriously diff."""
    draft = _draft()
    sections, _ = diff_to_sections(draft, raw_config={"image_generation": {}})
    assert "enabled_backends" not in sections.get("image_generation", {})


# --- validate_draft -------------------------------------------------------------


def test_validate_blocks_disabled_default():
    errors, _ = validate_draft(_draft(default_backend="openrouter", enabled_backends=["swarmui"]))
    assert any("Default backend must be enabled" in e for e in errors)


def test_validate_allows_enabled_default():
    errors, _ = validate_draft(_draft(default_backend="openrouter", enabled_backends=["openrouter"]))
    assert errors == []


def test_validate_warns_all_disabled_and_batch_over_cap():
    _, warnings = validate_draft(_draft(enabled_backends=[], default_batch=9, max_variants_per_message=4))
    assert len(warnings) == 2


def test_validate_rejects_non_numeric_timeout():
    errors, _ = validate_draft(_draft(backend_fields={"openrouter": {"timeout_seconds": "soon"}}))
    assert any("whole number" in e for e in errors)


def test_validate_rejects_timeout_below_minimum():
    errors, _ = validate_draft(_draft(backend_fields={"openrouter": {"timeout_seconds": "0"}}))
    assert any("at least 1" in e for e in errors)


def test_validate_rejects_malformed_base_url():
    errors, _ = validate_draft(_draft(backend_fields={"swarmui": {"base_url": "not-a-url"}}))
    assert any("valid http" in e for e in errors)


def test_validate_accepts_well_formed_base_url():
    errors, _ = validate_draft(_draft(backend_fields={"swarmui": {"base_url": "http://127.0.0.1:7801"}}))
    assert errors == []


# --- build_backend_rows ----------------------------------------------------------


def test_build_backend_rows_status_and_sources(monkeypatch):
    cfg = _fake_cfg(
        monkeypatch,
        env={"OPENROUTER_API_KEY": "fake-env-key"},
        section={
            "novita": {"api_key": "novita-config-key"},
            "enabled_backends": ["openrouter", "novita"],
            "default_backend": "openrouter",
        },
        keyring={"together": "kr-secret"},
    )
    rows = {r.backend_id: r for r in build_backend_rows(cfg, raw_section={})}

    assert set(rows) == set(BACKEND_IDS)
    assert rows["openrouter"].key_source == "env:OPENROUTER_API_KEY"
    assert rows["novita"].key_source == "config"
    assert rows["together"].key_source == "keyring"
    assert rows["modelstudio"].key_source == "missing"
    assert rows["swarmui"].key_source == "missing"
    assert rows["swarmui"].secret_optional is True
    assert rows["openrouter"].secret_optional is False
    assert rows["openrouter"].enabled is True
    assert rows["swarmui"].enabled is False
    assert rows["openrouter"].is_default is True
    assert rows["novita"].is_default is False
    assert rows["openrouter"].configured is True  # enabled + has an api_key
    assert rows["swarmui"].configured is False  # not enabled


# --- effective_placeholder -------------------------------------------------------


def test_effective_placeholder_shows_baked_default(monkeypatch):
    cfg = _fake_cfg(monkeypatch)  # nothing set
    assert effective_placeholder(cfg, "openrouter", "default_model") == "google/gemini-2.5-flash-image"


def test_effective_placeholder_shows_configured_value(monkeypatch):
    cfg = _fake_cfg(monkeypatch, section={"openrouter": {"default_model": "custom/model"}})
    assert effective_placeholder(cfg, "openrouter", "default_model") == "custom/model"


def test_effective_placeholder_empty_when_unset_and_no_baked_default(monkeypatch):
    cfg = _fake_cfg(monkeypatch)
    assert effective_placeholder(cfg, "stable_diffusion_cpp", "model_path") == ""


# --- SettingsConfigAdapter.delete_values ------------------------------------------


def test_adapter_delete_values(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        toml.dumps(
            {
                "image_generation": {
                    "openrouter": {"default_model": "old-model", "base_url": "http://example"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert SettingsConfigAdapter().delete_values("image_generation.openrouter", ["default_model"])

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    section = saved["image_generation"]["openrouter"]
    assert "default_model" not in section
    assert section["base_url"] == "http://example"
