import pytest

@pytest.fixture(autouse=True)
def _reset_cache():
    from tldw_chatbook.Image_Generation import config as c
    c.reset_image_generation_config_cache()
    yield
    c.reset_image_generation_config_cache()

def test_defaults_when_unconfigured(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    # No TOML section, no env, no keyring: fall back to documented defaults.
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: None, raising=False)  # avoid real keyring
    for var in ("OPENROUTER_API_KEY", "NOVITA_API_KEY", "TOGETHER_API_KEY", "DASHSCOPE_API_KEY", "QWEN_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.swarmui_base_url == c.DEFAULT_SWARMUI_BASE_URL
    assert cfg.max_width == c.DEFAULT_MAX_WIDTH
    assert cfg.openrouter_image_api_key in (None, "")  # unconfigured

def test_nested_toml_flattens_to_flat_fields(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {
        "default_backend": "swarmui",
        "enabled_backends": ["swarmui", "openrouter"],
        "swarmui": {"base_url": "http://example:9999"},
        "openrouter": {"default_model": "openai/gpt-image-1", "timeout_seconds": 42},
    }
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.swarmui_base_url == "http://example:9999"
    assert cfg.openrouter_image_default_model == "openai/gpt-image-1"
    assert cfg.openrouter_image_timeout_seconds == 42
    assert cfg.enabled_backends == ["swarmui", "openrouter"]

def test_secret_precedence_env_over_config(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {"openrouter": {"api_key": "from-config"}}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.openrouter_image_api_key == "from-env"

def test_secret_from_keyring_populates_field(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    # keyring-only secret must land on the config field so listing.is_configured sees it (spec §4.2 step 5)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: "kr-secret" if backend == "novita" else None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.novita_image_api_key == "kr-secret"

def test_sd_cpp_llm_path_flattens(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {"stable_diffusion_cpp": {"llm_path": "/models/qwen.gguf"}}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda backend: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.sd_cpp_llm_path == "/models/qwen.gguf"

def test_batch_and_variant_cap_defaults(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.default_batch == 1 and cfg.max_variants_per_message == 8

def test_batch_and_variant_cap_from_toml_clamped(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"default_batch": 3, "max_variants_per_message": 0}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.default_batch == 3 and cfg.max_variants_per_message == 1  # clamped >=1

def test_context_llm_defaults_when_unconfigured(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: {}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.context_llm_enabled is c.DEFAULT_CONTEXT_LLM_ENABLED
    assert cfg.context_llm_turns == c.DEFAULT_CONTEXT_LLM_TURNS
    assert cfg.context_llm_timeout_seconds == c.DEFAULT_CONTEXT_LLM_TIMEOUT_SECONDS

def test_context_llm_custom_values_from_toml(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {
        "context_llm_enabled": False,
        "context_llm_turns": 4,
        "context_llm_timeout_seconds": 7.5,
    }
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.context_llm_enabled is False
    assert cfg.context_llm_turns == 4
    assert cfg.context_llm_timeout_seconds == 7.5

def test_context_llm_enabled_accepts_string_bool_forms(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml",
                        lambda: {"context_llm_enabled": "false"}, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.context_llm_enabled is False

def test_context_llm_turns_and_timeout_clamped_to_minimums(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    fake = {"context_llm_turns": 0, "context_llm_timeout_seconds": 0}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)
    cfg = c.get_image_generation_config(reload=True)
    assert cfg.context_llm_turns == 1
    assert cfg.context_llm_timeout_seconds == 0.1


def _capture_warnings():
    """loguru is this project's logger; caplog does not intercept it -- attach
    a temporary sink and return (messages, sink_id)."""
    from loguru import logger as loguru_logger
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def test_flat_backend_key_under_image_generation_warns_with_nested_replacement(monkeypatch):
    # task-621: writing the FLAT dataclass field name directly under
    # [image_generation] (instead of nested under [image_generation.openrouter])
    # is silently ignored by the flattener -- it must log a warning naming the
    # key and the exact nested replacement.
    from loguru import logger as loguru_logger
    from tldw_chatbook.Image_Generation import config as c
    fake = {"openrouter_image_default_model": "google/gemini-2.5-flash-image"}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)

    messages, sink_id = _capture_warnings()
    try:
        cfg = c.get_image_generation_config(reload=True)
    finally:
        loguru_logger.remove(sink_id)

    # Silently ignored: the flat key never reaches the dataclass field.
    assert cfg.openrouter_image_default_model == c.DEFAULT_OPENROUTER_IMAGE_MODEL

    matches = [m for m in messages if "openrouter_image_default_model" in m]
    assert len(matches) == 1
    assert "[image_generation.openrouter] default_model" in matches[0]


def _load_config_with_section(monkeypatch, section: dict, *, keyring: dict | None = None):
    """Shared helper for the key_sources tests below: monkeypatch the raw
    [image_generation] TOML section (+ optional keyring hits) the same way
    every other test in this file does inline, then load. `keyring` maps
    backend id -> fake keyring secret (default: keyring never hits)."""
    from tldw_chatbook.Image_Generation import config as c
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: section, raising=False)
    kr = keyring or {}
    monkeypatch.setattr(c, "_keyring_get", lambda backend: kr.get(backend), raising=False)
    return c.get_image_generation_config(reload=True)


def test_key_sources_env_wins(monkeypatch, tmp_path):
    """key_sources records env origin with the winning variable name."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-env-key")
    cfg = _load_config_with_section(monkeypatch, {"openrouter": {}})
    assert cfg.key_sources["openrouter"] == "env:OPENROUTER_API_KEY"


def test_key_sources_config(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {"openrouter": {"api_key": "fake-config-key"}})
    assert cfg.key_sources["openrouter"] == "config"


def test_key_sources_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.key_sources["openrouter"] == "missing"
    assert set(cfg.key_sources) == {
        "stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio",
        "fal", "gemini",
    }


def test_key_sources_modelstudio_names_winning_env(monkeypatch):
    monkeypatch.setenv("QWEN_API_KEY", "fake-2")
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.key_sources["modelstudio"] == "env:QWEN_API_KEY"


def test_key_sources_keyring(monkeypatch):
    """keyring-origin secret is recorded as "keyring" (not the raw value)."""
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {}, keyring={"novita": "kr-secret"})
    assert cfg.key_sources["novita"] == "keyring"
    assert cfg.novita_image_api_key == "kr-secret"  # existing secret-field behavior unchanged


def test_unrecognized_key_under_image_generation_warns_generically(monkeypatch):
    from loguru import logger as loguru_logger
    from tldw_chatbook.Image_Generation import config as c
    fake = {"totally_made_up_key": "x"}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)

    messages, sink_id = _capture_warnings()
    try:
        c.get_image_generation_config(reload=True)
    finally:
        loguru_logger.remove(sink_id)

    matches = [m for m in messages if "totally_made_up_key" in m]
    assert len(matches) == 1
    assert "unknown key" in matches[0]


def test_nested_config_produces_no_unknown_key_warnings(monkeypatch):
    from loguru import logger as loguru_logger
    from tldw_chatbook.Image_Generation import config as c
    fake = {
        "default_backend": "swarmui",
        "enabled_backends": ["swarmui", "openrouter"],
        "swarmui": {"base_url": "http://example:9999"},
        "openrouter": {"default_model": "google/gemini-2.5-flash-image", "timeout_seconds": 42},
        "styles": {"my_glow": {"name": "My Glow"}},
    }
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)

    messages, sink_id = _capture_warnings()
    try:
        c.get_image_generation_config(reload=True)
    finally:
        loguru_logger.remove(sink_id)

    assert messages == []


def test_unknown_key_warning_fires_once_per_load_not_per_field_access(monkeypatch):
    from loguru import logger as loguru_logger
    from tldw_chatbook.Image_Generation import config as c
    fake = {"openrouter_image_default_model": "x"}
    monkeypatch.setattr(c, "_read_image_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)

    messages, sink_id = _capture_warnings()
    try:
        cfg = c.get_image_generation_config(reload=True)
        # Repeated field access on the already-built, cached dataclass must
        # not re-trigger the loader (and thus must not add more warnings).
        for _ in range(5):
            _ = cfg.openrouter_image_default_model
        _ = c.get_image_generation_config()  # cache hit, no reload
    finally:
        loguru_logger.remove(sink_id)

    matches = [m for m in messages if "openrouter_image_default_model" in m]
    assert len(matches) == 1


# --- Final-review CRITICAL fix: swarmui's real config key is swarm_token ----
#
# FIELD_SCHEMA (Settings > Image Gen, matching the design spec) writes/
# clears swarmui's secret under its OWN nested key, `swarm_token` -- but
# _resolve_secret's config branch used to read a hardcoded "api_key" for
# EVERY backend, so a pasted-and-saved swarm token landed in config.toml
# yet was never read back (key_sources stayed "missing", the value stayed
# None). Fixed by making the nested config key per-backend DATA in
# _SECRETS, with "api_key" kept as a back-compat fallback ONLY when a
# backend's own key is unset (so it never masks a real swarm_token, and
# every OTHER backend -- whose nested key was already "api_key" -- is
# unaffected).


def test_key_sources_swarmui_config_via_swarm_token(monkeypatch):
    """The round-trip the bug broke: write the secret via FIELD_SCHEMA's
    actual toml_key (swarm_token) and the loader must resolve it."""
    cfg = _load_config_with_section(
        monkeypatch, {"swarmui": {"swarm_token": "fake-swarm-token"}}
    )
    assert cfg.key_sources["swarmui"] == "config"
    assert cfg.swarmui_swarm_token == "fake-swarm-token"


def test_key_sources_swarmui_legacy_api_key_fallback(monkeypatch):
    """Back-compat: a config hand-written (or saved before this fix) with
    the wrong/legacy `api_key` key for swarmui must still resolve."""
    cfg = _load_config_with_section(
        monkeypatch, {"swarmui": {"api_key": "fake-legacy-key"}}
    )
    assert cfg.key_sources["swarmui"] == "config"
    assert cfg.swarmui_swarm_token == "fake-legacy-key"


def test_key_sources_swarmui_swarm_token_wins_over_legacy_api_key(monkeypatch):
    """When both are somehow set, the real key wins -- the fallback never
    overrides an explicit swarm_token value."""
    cfg = _load_config_with_section(
        monkeypatch,
        {"swarmui": {"swarm_token": "real-token", "api_key": "stale-legacy-key"}},
    )
    assert cfg.key_sources["swarmui"] == "config"
    assert cfg.swarmui_swarm_token == "real-token"


def test_key_sources_swarmui_missing_when_neither_key_set(monkeypatch):
    monkeypatch.delenv("SWARMUI_TOKEN", raising=False)
    cfg = _load_config_with_section(monkeypatch, {"swarmui": {}})
    assert cfg.key_sources["swarmui"] == "missing"
    assert cfg.swarmui_swarm_token in (None, "")


def test_key_sources_swarmui_env_wins_over_config_key(monkeypatch):
    """Precedence (env > config > keyring) is unaffected by the config-key
    fix -- swarm_token in config is still beaten by SWARMUI_TOKEN in env."""
    monkeypatch.setenv("SWARMUI_TOKEN", "fake-env-token")
    cfg = _load_config_with_section(
        monkeypatch, {"swarmui": {"swarm_token": "fake-swarm-token"}}
    )
    assert cfg.key_sources["swarmui"] == "env:SWARMUI_TOKEN"
    assert cfg.swarmui_swarm_token == "fake-env-token"


def test_other_backends_config_key_unaffected_by_swarmui_fix(monkeypatch):
    """Every non-swarmui backend's config_key was already "api_key" --
    the fix must be a no-op for them (no fallback ever engages, since
    config_key == "api_key" already)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = _load_config_with_section(
        monkeypatch, {"openrouter": {"api_key": "fake-openrouter-key"}}
    )
    assert cfg.key_sources["openrouter"] == "config"
    assert cfg.openrouter_image_api_key == "fake-openrouter-key"


# --- task-2 (fal/Gemini image backends) --------------------------------
# Fireworks was dropped 2026-07-26 -- vendor deprecated image generation
# (see Docs/superpowers/specs/2026-07-26-imagegen-fal-gemini-fireworks-design.md
# and the sibling plan doc for the decision note).


def _delenv_new_backend_vars(monkeypatch):
    for var in ("FAL_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def test_fal_defaults_when_unset(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    _delenv_new_backend_vars(monkeypatch)
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.fal_image_base_url == c.DEFAULT_FAL_IMAGE_BASE_URL == "https://queue.fal.run"
    assert cfg.fal_image_default_model == c.DEFAULT_FAL_IMAGE_MODEL == "fal-ai/flux/schnell"
    assert cfg.fal_image_poll_interval_seconds == c.DEFAULT_FAL_IMAGE_POLL_INTERVAL_SECONDS == 2
    assert cfg.fal_image_timeout_seconds == c.DEFAULT_FAL_IMAGE_TIMEOUT_SECONDS == 120
    assert cfg.fal_image_api_key in (None, "")
    assert cfg.key_sources["fal"] == "missing"


def test_fal_nested_toml_round_trip(monkeypatch):
    _delenv_new_backend_vars(monkeypatch)
    fake = {
        "fal": {
            "base_url": "https://queue.example.fal.run",
            "api_key": "fake-fal-key",
            "default_model": "fal-ai/other-model",
            "poll_interval_seconds": 5,
            "timeout_seconds": 30,
        }
    }
    cfg = _load_config_with_section(monkeypatch, fake)
    assert cfg.fal_image_base_url == "https://queue.example.fal.run"
    assert cfg.fal_image_api_key == "fake-fal-key"
    assert cfg.fal_image_default_model == "fal-ai/other-model"
    assert cfg.fal_image_poll_interval_seconds == 5
    assert cfg.fal_image_timeout_seconds == 30
    assert cfg.key_sources["fal"] == "config"


def test_fal_env_key_precedence(monkeypatch):
    _delenv_new_backend_vars(monkeypatch)
    monkeypatch.setenv("FAL_KEY", "fake-fal-env-key")
    cfg = _load_config_with_section(monkeypatch, {"fal": {"api_key": "fake-fal-config-key"}})
    assert cfg.fal_image_api_key == "fake-fal-env-key"
    assert cfg.key_sources["fal"] == "env:FAL_KEY"


def test_gemini_defaults_when_unset(monkeypatch):
    from tldw_chatbook.Image_Generation import config as c
    _delenv_new_backend_vars(monkeypatch)
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.gemini_image_base_url == c.DEFAULT_GEMINI_IMAGE_BASE_URL == "https://generativelanguage.googleapis.com/v1beta"
    assert cfg.gemini_image_default_model == c.DEFAULT_GEMINI_IMAGE_MODEL == "gemini-2.5-flash-image"
    assert cfg.gemini_image_timeout_seconds == c.DEFAULT_GEMINI_IMAGE_TIMEOUT_SECONDS == 120
    assert cfg.gemini_image_api_key in (None, "")
    assert cfg.key_sources["gemini"] == "missing"


def test_gemini_nested_toml_round_trip(monkeypatch):
    _delenv_new_backend_vars(monkeypatch)
    fake = {
        "gemini": {
            "base_url": "https://example.googleapis.com/v1beta",
            "api_key": "fake-gemini-key",
            "default_model": "gemini-other-model",
            "timeout_seconds": 45,
        }
    }
    cfg = _load_config_with_section(monkeypatch, fake)
    assert cfg.gemini_image_base_url == "https://example.googleapis.com/v1beta"
    assert cfg.gemini_image_api_key == "fake-gemini-key"
    assert cfg.gemini_image_default_model == "gemini-other-model"
    assert cfg.gemini_image_timeout_seconds == 45
    assert cfg.key_sources["gemini"] == "config"


def test_gemini_env_precedence_gemini_key_wins_over_google_key(monkeypatch):
    """GEMINI_API_KEY is listed first in the precedence order, so when both
    env vars are set it must win over GOOGLE_API_KEY."""
    _delenv_new_backend_vars(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-gemini-key")
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.gemini_image_api_key == "fake-gemini-key"
    assert cfg.key_sources["gemini"] == "env:GEMINI_API_KEY"


def test_gemini_env_fallback_to_google_api_key(monkeypatch):
    """When only GOOGLE_API_KEY is set (no GEMINI_API_KEY), it must be used
    and named explicitly as the source."""
    _delenv_new_backend_vars(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.gemini_image_api_key == "fake-google-key"
    assert cfg.key_sources["gemini"] == "env:GOOGLE_API_KEY"


# --- task-686: per-backend keyring-source tests (fal/gemini) ---------------
# The generic keyring path is already covered for novita
# (test_key_sources_keyring above); fal and gemini share the same
# _resolve_secret/_keyring_get machinery but had no dedicated coverage.


def test_fal_key_sources_keyring(monkeypatch):
    """keyring-origin secret is recorded as "keyring" (not the raw value)
    and lands on the flat field, same as the generic-backend case."""
    _delenv_new_backend_vars(monkeypatch)
    cfg = _load_config_with_section(monkeypatch, {}, keyring={"fal": "kr-fal-secret"})
    assert cfg.key_sources["fal"] == "keyring"
    assert cfg.fal_image_api_key == "kr-fal-secret"


def test_gemini_key_sources_keyring(monkeypatch):
    """keyring-origin secret is recorded as "keyring" (not the raw value)
    and lands on the flat field, same as the generic-backend case."""
    _delenv_new_backend_vars(monkeypatch)
    cfg = _load_config_with_section(monkeypatch, {}, keyring={"gemini": "kr-gemini-secret"})
    assert cfg.key_sources["gemini"] == "keyring"
    assert cfg.gemini_image_api_key == "kr-gemini-secret"
