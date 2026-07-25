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
    assert set(cfg.key_sources) == {"stable_diffusion_cpp", "swarmui", "openrouter", "novita", "together", "modelstudio"}


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
