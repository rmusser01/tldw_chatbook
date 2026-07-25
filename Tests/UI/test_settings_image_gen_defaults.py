"""Tests for the Settings > Image Gen defaults data layer: FIELD_SCHEMA,
build_backend_rows, effective_placeholder, ImageGenDraftValues,
diff_to_sections, validate_draft, and the adapter's delete_values wrapper
(task-2), plus the backend probes / "Test" action (task-3).
"""

from __future__ import annotations

import os
import tomllib

import httpx
import pytest
import toml

from tldw_chatbook.Image_Generation.config import _NON_SECRET, _SECRETS
from tldw_chatbook.UI.Screens import settings_image_gen_defaults as sigd
from tldw_chatbook.UI.Screens.settings_config_adapter import SettingsConfigAdapter
from tldw_chatbook.UI.Screens.settings_image_gen_defaults import (
    BACKEND_IDS,
    BACKEND_LABELS,
    FIELD_SCHEMA,
    PROBE_TIMEOUT_SECONDS,
    ImageGenDraftValues,
    ImageGenProbeResult,
    build_backend_rows,
    diff_to_sections,
    effective_placeholder,
    effective_secret_value,
    load_user_image_generation_table,
    probe_backend,
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


@pytest.mark.parametrize(
    "backend_id",
    [
        backend
        for backend, specs in FIELD_SCHEMA.items()
        if any(spec.kind == "secret" for spec in specs)
    ],
)
def test_secret_field_round_trips_through_the_real_loader(monkeypatch, backend_id):
    """Closes the drift-test hole the swarmui `swarm_token`/`api_key`
    mismatch slipped through (final review CRITICAL fix):
    `test_field_schema_maps_to_real_loader_keys` above only checks that a
    `_SECRETS` entry EXISTS for the backend, never that the schema's own
    `toml_key` is what the loader actually reads back. Writing the secret
    via FIELD_SCHEMA's `toml_key` (never a hardcoded "api_key") into a
    crafted section must resolve into the backend's flat field with
    `key_sources == "config"`, for EVERY backend with a secret field --
    this was RED for swarmui (whose real `toml_key` is `swarm_token`, not
    `api_key`) before the loader fix.
    """
    spec = next(s for s in FIELD_SCHEMA[backend_id] if s.kind == "secret")
    cfg = _fake_cfg(
        monkeypatch, section={backend_id: {spec.toml_key: "fake-secret-value"}}
    )
    assert cfg.key_sources[backend_id] == "config"
    flat_field = _SECRETS[backend_id][0]
    assert getattr(cfg, flat_field) == "fake-secret-value"


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


def test_diff_never_copies_preexisting_secret_when_editing_other_field():
    """THE no-secret-copy pin, realistic leak shape: raw_config ALREADY has a
    config-saved secret for a backend; editing a DIFFERENT (non-secret)
    field on that same backend must emit only the edited key -- never the
    pre-existing secret riding along via a wholesale section rewrite. (A
    draft signature that never accepts an ImageGenerationConfig makes
    copying an *env-resolved* secret a code-level impossibility; this test
    guards the separate, equally real risk of leaking an already-saved
    *config* secret when the diff logic touches that backend's section at
    all.)"""
    raw_config = {
        "image_generation": {
            "openrouter": {"api_key": "already-saved-secret", "default_model": "old-model"},
            "swarmui": {"swarm_token": "already-saved-token", "base_url": "http://old"},
        }
    }
    draft = _draft(
        backend_fields={
            "openrouter": {"default_model": "new-model"},
            "swarmui": {"base_url": "http://new"},
        }
    )

    sections, _ = diff_to_sections(draft, raw_config)

    assert sections == {
        "image_generation.openrouter": {"default_model": "new-model"},
        "image_generation.swarmui": {"base_url": "http://new"},
    }
    assert "api_key" not in sections["image_generation.openrouter"]
    assert "swarm_token" not in sections["image_generation.swarmui"]


def test_diff_emits_typed_secret_exactly_and_nothing_else():
    """A secret the user actually typed this session is emitted verbatim
    and alone -- the diff neither drops a deliberately-typed secret nor
    smuggles in any other field (secret or not) from raw_config alongside it."""
    raw_config = {
        "image_generation": {
            "openrouter": {"api_key": "already-saved-secret", "default_model": "old-model"},
        }
    }
    draft = _draft(backend_fields={"openrouter": {"api_key": "typed-this-session"}})

    sections, _ = diff_to_sections(draft, raw_config)

    assert sections == {"image_generation.openrouter": {"api_key": "typed-this-session"}}


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


# --- Final-review Minor 1: enabled_backends order-insensitivity -------------


def test_diff_enabled_backends_same_set_different_order_is_not_a_diff():
    """A config file's enabled_backends list order is meaningless (it's a
    set) -- comparing it against the draft's list (always built in
    canonical BACKEND_IDS order) without normalizing BOTH sides first
    would spuriously diff whenever the file happens to list the same
    backends in a different order, rewriting the file on every save and
    leaving the rail dirty marker stuck."""
    draft = _draft(enabled_backends=["swarmui", "openrouter"])  # canonical order
    sections, _ = diff_to_sections(
        draft,
        raw_config={"image_generation": {"enabled_backends": ["openrouter", "swarmui"]}},
    )
    assert "enabled_backends" not in sections.get("image_generation", {})


def test_diff_enabled_backends_genuine_change_emits_canonical_order():
    draft = _draft(enabled_backends=["openrouter"])
    sections, _ = diff_to_sections(
        draft,
        raw_config={"image_generation": {"enabled_backends": ["swarmui", "openrouter"]}},
    )
    assert sections == {"image_generation": {"enabled_backends": ["openrouter"]}}


def test_diff_enabled_backends_drops_unrecognized_entries_from_both_sides():
    draft = _draft(enabled_backends=["swarmui", "openrouter", "not-a-real-backend"])
    sections, _ = diff_to_sections(
        draft,
        raw_config={
            "image_generation": {
                "enabled_backends": ["openrouter", "also-not-real", "swarmui"]
            }
        },
    )
    assert "enabled_backends" not in sections.get("image_generation", {})


# --- Final-review Important 1: emptying a field deletes, never blanks ------


def test_diff_emptying_saved_secret_deletes_not_blanks():
    draft = _draft(backend_fields={"openrouter": {"api_key": ""}})
    sections, deletions = diff_to_sections(
        draft,
        raw_config={"image_generation": {"openrouter": {"api_key": "sk-saved-key"}}},
    )
    assert "api_key" not in sections.get("image_generation.openrouter", {})
    assert deletions == {"image_generation.openrouter": ["api_key"]}


def test_diff_emptying_saved_model_deletes_not_blanks():
    draft = _draft(backend_fields={"openrouter": {"default_model": "   "}})
    sections, deletions = diff_to_sections(
        draft,
        raw_config={"image_generation": {"openrouter": {"default_model": "old-model"}}},
    )
    assert "default_model" not in sections.get("image_generation.openrouter", {})
    assert deletions == {"image_generation.openrouter": ["default_model"]}


def test_diff_emptying_set_global_deletes_not_blanks():
    draft = _draft(default_batch="")
    sections, deletions = diff_to_sections(
        draft, raw_config={"image_generation": {"default_batch": 4}}
    )
    assert "default_batch" not in sections.get("image_generation", {})
    assert deletions == {"image_generation": ["default_batch"]}


def test_diff_emptying_unset_backend_field_is_a_no_op_not_a_diff():
    draft = _draft(backend_fields={"openrouter": {"default_model": ""}})
    sections, deletions = diff_to_sections(
        draft, raw_config={"image_generation": {"openrouter": {}}}
    )
    assert sections == {}
    assert deletions == {}


def test_diff_emptying_unset_global_is_a_no_op_not_a_diff():
    draft = _draft(default_batch="")
    sections, deletions = diff_to_sections(draft, raw_config={"image_generation": {}})
    assert sections == {}
    assert deletions == {}


def test_diff_emptying_and_clearing_same_key_merges_deletions_not_overwrites():
    """An emptied field on one backend and an explicit Clear on another
    must both survive in `deletions` -- neither source overwrites the
    other's section entry."""
    draft = _draft(
        backend_fields={
            "openrouter": {"default_model": ""},
            "swarmui": {"swarm_token": "typed-then-cleared"},
        },
        cleared_fields={"swarmui": ["swarm_token"]},
    )
    sections, deletions = diff_to_sections(
        draft,
        raw_config={
            "image_generation": {
                "openrouter": {"default_model": "old-model"},
                "swarmui": {"swarm_token": "old-token"},
            }
        },
    )
    assert sections == {}
    assert deletions == {
        "image_generation.openrouter": ["default_model"],
        "image_generation.swarmui": ["swarm_token"],
    }


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


# --- Final review Minor 2: global scalar fields' spec min-clamps -----------


def test_validate_rejects_default_batch_below_minimum():
    errors, _ = validate_draft(_draft(default_batch=0))
    assert any("Default batch must be at least 1" in e for e in errors)


def test_validate_accepts_default_batch_at_minimum():
    errors, _ = validate_draft(_draft(default_batch=1, enabled_backends=["openrouter"]))
    assert errors == []


def test_validate_rejects_max_variants_below_minimum():
    errors, _ = validate_draft(_draft(max_variants_per_message=0))
    assert any("Max variants / message must be at least 1" in e for e in errors)


def test_validate_rejects_context_llm_turns_below_minimum():
    errors, _ = validate_draft(_draft(context_llm_turns=0))
    assert any("Context LLM turns must be at least 1" in e for e in errors)


def test_validate_rejects_context_llm_timeout_below_minimum():
    errors, _ = validate_draft(_draft(context_llm_timeout_seconds=0.05))
    assert any("Context LLM timeout (s) must be at least 0.1" in e for e in errors)


def test_validate_accepts_context_llm_timeout_at_minimum():
    errors, _ = validate_draft(_draft(context_llm_timeout_seconds=0.1))
    assert errors == []


def test_validate_emptied_global_field_is_not_validated_as_invalid():
    """An emptied global field becomes a deletion (diff_to_sections), not
    a value validate_draft should ever reject."""
    errors, _ = validate_draft(_draft(default_batch=""))
    assert errors == []


def test_validate_emptied_backend_field_is_not_validated_as_invalid():
    errors, _ = validate_draft(
        _draft(backend_fields={"openrouter": {"timeout_seconds": ""}})
    )
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
    rows = {r.backend_id: r for r in build_backend_rows(cfg)}

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


# --- effective_secret_value (task-6: Test probe secret fallback) -----------------


def test_effective_secret_value_returns_resolved_config_secret(monkeypatch):
    cfg = _fake_cfg(monkeypatch, section={"openrouter": {"api_key": "sk-config-key"}})
    assert effective_secret_value(cfg, "openrouter") == "sk-config-key"


def test_effective_secret_value_returns_resolved_env_secret(monkeypatch):
    cfg = _fake_cfg(monkeypatch, env={"OPENROUTER_API_KEY": "sk-env-key"})
    assert effective_secret_value(cfg, "openrouter") == "sk-env-key"


def test_effective_secret_value_none_when_unresolved(monkeypatch):
    cfg = _fake_cfg(monkeypatch)
    assert effective_secret_value(cfg, "openrouter") is None


def test_effective_secret_value_none_for_backend_without_secret(monkeypatch):
    cfg = _fake_cfg(monkeypatch, section={"stable_diffusion_cpp": {}})
    assert effective_secret_value(cfg, "stable_diffusion_cpp") is None


# --- load_user_image_generation_table (Fix Round 1: set-vs-default blur) ----------


def test_load_user_table_missing_config_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "does-not-exist.toml"))
    assert load_user_image_generation_table() == {}


def test_load_user_table_no_image_generation_section_returns_empty(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text('[general]\ndefault_theme = "textual-dark"\n', encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    assert load_user_image_generation_table() == {}


def test_load_user_table_returns_raw_unmerged_content(tmp_path, monkeypatch):
    """The core set-vs-default-blur fix: this must return EXACTLY what the
    user wrote -- no other backend keys, no baked-in template values --
    unlike SettingsConfigAdapter.load(), which deep-merges config.py's
    bundled default template into every [image_generation.*] field."""
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[image_generation]\ndefault_backend = "openrouter"\n\n'
        '[image_generation.openrouter]\ndefault_model = "m-x"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    table = load_user_image_generation_table()

    assert table == {
        "default_backend": "openrouter",
        "openrouter": {"default_model": "m-x"},
    }
    # The blur this fixes: a merged read would also carry every OTHER
    # backend's baked default section (swarmui, novita, together, ...).
    assert "swarmui" not in table
    assert "novita" not in table


def test_load_user_table_malformed_toml_returns_empty_without_raising(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text("this is not [ valid toml", encoding="utf-8")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    assert load_user_image_generation_table() == {}


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


# --- probe_backend (task-3) -------------------------------------------------------


def _fake_client_cls(*, response=None, raise_exc=None, calls=None):
    """A fake httpx.Client following Tests/Image_Generation/test_http_client.py's
    style: context-manager stub whose `.get()` either raises or returns a
    canned response, recording every call for assertions."""

    class _FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url, headers=None):
            if calls is not None:
                calls.append((url, dict(headers or {})))
            if raise_exc is not None:
                raise raise_exc
            return _FakeResponse(response)

    return FakeClient


@pytest.fixture(autouse=True)
def _policy_env(monkeypatch):
    """Deterministic egress policy for every probe test in this section,
    mirroring Tests/Image_Generation/test_http_client.py's `_policy_env`:
    resolve any non-IP-literal hostname to a fixed public IP, and force
    [web_security] to its enabled/no-extra-allowlist defaults so these
    tests are not at the mercy of a developer's local config.toml."""
    from tldw_chatbook.Utils import egress

    monkeypatch.setattr(egress, "_resolve", lambda host: ["93.184.216.34"])
    monkeypatch.setattr(egress, "get_cli_setting", lambda s, k=None, d=None: d)


def test_probe_timeout_constant():
    assert PROBE_TIMEOUT_SECONDS == 5.0


def test_probe_openrouter_reachable_2xx(monkeypatch):
    calls = []
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200, calls=calls))
    result = probe_backend("openrouter", {"base_url": "http://127.0.0.1:9900"}, "sk-real-key")
    assert result == ImageGenProbeResult(ok=True, badge="Reachable")
    url, headers = calls[0]
    assert url == "http://127.0.0.1:9900/models"
    assert headers == {"Authorization": "Bearer sk-real-key"}


def test_probe_connect_error_is_connection_refused(monkeypatch):
    monkeypatch.setattr(
        sigd.httpx, "Client", _fake_client_cls(raise_exc=httpx.ConnectError("connect failed"))
    )
    result = probe_backend("openrouter", {"base_url": "http://127.0.0.1:9900"}, "sk-real-key")
    assert result == ImageGenProbeResult(ok=False, badge="Unreachable: connection refused")


def test_probe_read_timeout_is_timeout(monkeypatch):
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(raise_exc=httpx.ReadTimeout("slow")))
    result = probe_backend("together", {"base_url": "http://127.0.0.1:9900"}, "sk-real-key")
    assert result == ImageGenProbeResult(ok=False, badge="Unreachable: timeout")


def test_probe_auth_failed_401_with_key(monkeypatch):
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=401))
    result = probe_backend("openrouter", {"base_url": "http://127.0.0.1:9900"}, "sk-bad-key")
    assert result == ImageGenProbeResult(ok=False, badge="Auth failed")


def test_probe_auth_failed_403_with_key(monkeypatch):
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=403))
    result = probe_backend("together", {"base_url": "http://127.0.0.1:9900"}, "sk-bad-key")
    assert result == ImageGenProbeResult(ok=False, badge="Auth failed")


def test_probe_other_http_status_with_key(monkeypatch):
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=500))
    result = probe_backend("openrouter", {"base_url": "http://127.0.0.1:9900"}, "sk-real-key")
    assert result == ImageGenProbeResult(ok=False, badge="Unreachable: HTTP 500")


def test_probe_no_key_openrouter_reachable_auth_unverified(monkeypatch):
    calls = []
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200, calls=calls))
    result = probe_backend("openrouter", {"base_url": "http://127.0.0.1:9900"}, None)
    assert result == ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")
    assert calls[0][1] == {}  # no Authorization header sent


def test_probe_no_key_any_answer_counts_even_non_2xx(monkeypatch):
    """Without a secret, status code is not interpreted at all -- any answer
    means "server responded", per the spec's "answered" language."""
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=503))
    result = probe_backend("together", {"base_url": "http://127.0.0.1:9900"}, None)
    assert result == ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")


def test_probe_swarmui_any_http_answer_is_reachable(monkeypatch):
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=404))
    result = probe_backend("swarmui", {"base_url": "http://127.0.0.1:7801"}, None)
    assert result == ImageGenProbeResult(ok=True, badge="Reachable")


def test_probe_novita_unauthenticated_reachability_only(monkeypatch):
    """No cheap authenticated GET was confirmed in novita_image_adapter.py
    (only async submit/poll routes exist) -- novita probes the same way as
    modelstudio: unauthenticated reachability only."""
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200))
    result = probe_backend("novita", {"base_url": "http://127.0.0.1:9900"}, "some-key")
    assert result == ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")


def test_probe_modelstudio_unauthenticated_reachability_only(monkeypatch):
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200))
    result = probe_backend("modelstudio", {"base_url": "http://127.0.0.1:9900"}, "some-key")
    assert result == ImageGenProbeResult(ok=True, badge="Reachable (auth unverified)")


def test_probe_unknown_backend_id_raises():
    with pytest.raises(ValueError):
        probe_backend("not-a-real-backend", {}, None)


# --- probe_backend: sanitization + egress pins --------------------------------


def test_probe_sanitization_never_leaks_exception_text(monkeypatch):
    """THE sanitization pin: an exception message carrying a fake secret and
    URL must never reach the badge -- only the closed-set category string."""
    monkeypatch.setattr(
        sigd.httpx,
        "Client",
        _fake_client_cls(
            raise_exc=httpx.ConnectError(
                "secret sk-abcdef123456 in text http://10.0.0.1/leak"
            )
        ),
    )
    result = probe_backend("openrouter", {"base_url": "http://127.0.0.1:9900"}, "sk-real-key")
    assert result.badge == "Unreachable: connection refused"
    assert "sk-abcdef123456" not in result.badge
    assert "10.0.0.1" not in result.badge


def test_probe_egress_allows_private_base_url_via_self_trust(monkeypatch):
    """A private base_url (e.g. a local SwarmUI instance) is trusted because
    its own host is threaded in as trusted_origins(url) -- the probe still
    reaches the transport layer instead of being blocked outright."""
    calls = []
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200, calls=calls))
    result = probe_backend("swarmui", {"base_url": "http://127.0.0.1:7801"}, None)
    assert result == ImageGenProbeResult(ok=True, badge="Reachable")
    assert calls  # the fake transport was actually reached


def test_probe_egress_allows_public_api_shaped_url(monkeypatch):
    """A normal public API base_url also passes check_url_or_raise (public
    IPs are allowed regardless of trusted_origins)."""
    calls = []
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200, calls=calls))
    result = probe_backend(
        "openrouter", {"base_url": "https://openrouter.ai/api/v1"}, "sk-real-key"
    )
    assert result == ImageGenProbeResult(ok=True, badge="Reachable")
    assert calls[0][0] == "https://openrouter.ai/api/v1/models"


def test_probe_egress_blocks_metadata_ip_even_self_trusted(monkeypatch):
    """The one case check_url_or_raise(url, trusted_origins=origin_set(url))
    still blocks: cloud metadata endpoints are blocked regardless of trust
    (Utils/egress.py's hard rule)."""
    monkeypatch.setattr(sigd.httpx, "Client", _fake_client_cls(response=200))
    result = probe_backend(
        "swarmui", {"base_url": "http://169.254.169.254/latest/meta-data/"}, None
    )
    assert result == ImageGenProbeResult(ok=False, badge="Unreachable: blocked by egress policy")


# --- probe_backend: sd.cpp (filesystem-only, no network) -----------------------


def test_probe_sd_cpp_binary_and_model_present(tmp_path):
    binary = tmp_path / "sd-cpp-bin"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_text("fake model bytes", encoding="utf-8")

    result = probe_backend(
        "stable_diffusion_cpp",
        {"binary_path": str(binary), "model_path": str(model)},
        None,
    )
    assert result == ImageGenProbeResult(ok=True, badge="Binary found")


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root satisfies os.access(X_OK) regardless of the mode bits",
)
def test_probe_sd_cpp_binary_not_executable(tmp_path):
    binary = tmp_path / "sd-cpp-bin"
    binary.write_text("not actually executable", encoding="utf-8")
    binary.chmod(0o644)
    model = tmp_path / "model.gguf"
    model.write_text("fake model bytes", encoding="utf-8")

    result = probe_backend(
        "stable_diffusion_cpp",
        {"binary_path": str(binary), "model_path": str(model)},
        None,
    )
    assert result == ImageGenProbeResult(ok=False, badge="Binary missing or not executable")


def test_probe_sd_cpp_binary_missing_entirely(tmp_path):
    result = probe_backend(
        "stable_diffusion_cpp",
        {"binary_path": str(tmp_path / "nope"), "model_path": str(tmp_path / "also-nope")},
        None,
    )
    assert result == ImageGenProbeResult(ok=False, badge="Binary missing or not executable")


def test_probe_sd_cpp_model_missing(tmp_path):
    binary = tmp_path / "sd-cpp-bin"
    binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary.chmod(0o755)

    result = probe_backend(
        "stable_diffusion_cpp",
        {"binary_path": str(binary), "model_path": str(tmp_path / "missing-model.gguf")},
        None,
    )
    assert result == ImageGenProbeResult(ok=False, badge="Model file missing")


def test_probe_sd_cpp_empty_form_values(tmp_path):
    """Neither field set at all -- must not raise, must report the binary
    gap first (matches the spec's check order)."""
    result = probe_backend("stable_diffusion_cpp", {}, None)
    assert result == ImageGenProbeResult(ok=False, badge="Binary missing or not executable")
