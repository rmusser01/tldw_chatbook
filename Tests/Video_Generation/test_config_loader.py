import pytest


@pytest.fixture(autouse=True)
def _reset_cache():
    from tldw_chatbook.Video_Generation import config as c
    c.reset_video_generation_config_cache()
    yield
    c.reset_video_generation_config_cache()


def _load_config_with_section(monkeypatch, section: dict, *, keyring: dict | None = None):
    """Shared helper: monkeypatch the raw [video_generation] TOML section (+
    optional keyring hits), then load. `keyring` maps backend id -> fake
    keyring secret (default: keyring never hits)."""
    from tldw_chatbook.Video_Generation import config as c
    monkeypatch.setattr(c, "_read_video_generation_toml", lambda: section, raising=False)
    kr = keyring or {}
    monkeypatch.setattr(c, "_keyring_get", lambda backend: kr.get(backend), raising=False)
    return c.get_video_generation_config(reload=True)


def test_defaults_when_unconfigured(monkeypatch):
    from tldw_chatbook.Video_Generation import config as c
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.default_backend == c.DEFAULT_BACKEND
    assert cfg.enabled_backends == []
    assert cfg.minimax_video_base_url == c.DEFAULT_MINIMAX_VIDEO_BASE_URL
    assert cfg.minimax_video_default_model == c.DEFAULT_MINIMAX_VIDEO_MODEL
    assert cfg.minimax_video_api_key in (None, "")
    assert cfg.comfyui_base_url == c.DEFAULT_COMFYUI_BASE_URL
    assert cfg.retention == c.DEFAULT_RETENTION
    assert cfg.max_store_mb == c.DEFAULT_MAX_STORE_MB
    assert cfg.confirm_cost_estimate is c.DEFAULT_CONFIRM_COST_ESTIMATE


def test_malformed_scalar_video_generation_section_uses_defaults(monkeypatch):
    """A malformed top-level section must not crash config construction."""
    from tldw_chatbook.Video_Generation import config as c

    cfg = _load_config_with_section(monkeypatch, "not-a-table")

    assert cfg.default_backend == c.DEFAULT_BACKEND
    assert cfg.enabled_backends == []
    assert cfg.key_sources == {
        "minimax": "missing",
        "comfyui": "missing",
        "stable_diffusion_cpp": "missing",
    }


@pytest.mark.parametrize("backend", ("minimax", "comfyui", "stable_diffusion_cpp"))
def test_malformed_scalar_backend_section_uses_defaults(monkeypatch, backend):
    """A malformed backend subsection must not crash config construction."""
    from tldw_chatbook.Video_Generation import config as c

    cfg = _load_config_with_section(monkeypatch, {backend: 1})

    assert cfg.default_backend == c.DEFAULT_BACKEND
    assert cfg.enabled_backends == []
    assert cfg.comfyui_default_workflow == c.DEFAULT_COMFYUI_WORKFLOW
    assert cfg.key_sources["minimax"] == "missing"


def test_comfyui_default_workflow_is_base_h3(monkeypatch):
    cfg = _load_config_with_section(monkeypatch, {})

    assert cfg.comfyui_default_workflow == "minimax_h3_t2v.json"


def test_comfyui_explicit_spectrum_workflow_wins(monkeypatch):
    cfg = _load_config_with_section(
        monkeypatch,
        {"comfyui": {"default_workflow": "minimax_h3_t2v_spectrum.json"}},
    )

    assert cfg.comfyui_default_workflow == "minimax_h3_t2v_spectrum.json"


def test_nested_toml_flattens_to_flat_fields(monkeypatch):
    fake = {
        "default_backend": "minimax",
        "enabled_backends": ["minimax", "comfyui"],
        "minimax": {"base_url": "https://example.invalid", "poll_interval_seconds": 3},
        "comfyui": {"default_workflow": "wan22_t2v", "timeout_seconds": 900},
        "stable_diffusion_cpp": {"binary_path": "/opt/sd/bin/sd-cli", "default_fps": 12},
    }
    cfg = _load_config_with_section(monkeypatch, fake)
    assert cfg.default_backend == "minimax"
    assert cfg.enabled_backends == ["minimax", "comfyui"]
    assert cfg.minimax_video_base_url == "https://example.invalid"
    assert cfg.minimax_video_poll_interval_seconds == 3
    assert cfg.comfyui_default_workflow == "wan22_t2v"
    assert cfg.comfyui_timeout_seconds == 900
    assert cfg.sd_cpp_binary_path == "/opt/sd/bin/sd-cli"
    assert cfg.sd_cpp_default_fps == 12


def test_secret_precedence_env_over_config(monkeypatch):
    fake = {"minimax": {"api_key": "from-config"}}
    monkeypatch.setenv("MINIMAX_API_KEY", "from-env")
    cfg = _load_config_with_section(monkeypatch, fake)
    assert cfg.minimax_video_api_key == "from-env"
    assert cfg.key_sources["minimax"] == "env:MINIMAX_API_KEY"


def test_secret_from_config_and_keyring_fallback(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {"minimax": {"api_key": "from-config"}})
    assert cfg.minimax_video_api_key == "from-config"
    assert cfg.key_sources["minimax"] == "config"

    cfg2 = _load_config_with_section(monkeypatch, {}, keyring={"minimax": "kr-secret"})
    assert cfg2.minimax_video_api_key == "kr-secret"
    assert cfg2.key_sources["minimax"] == "keyring"


def test_key_sources_missing_covers_all_backends(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.key_sources["minimax"] == "missing"
    # Backends with no _SECRETS entry are always "missing".
    assert set(cfg.key_sources) == {"minimax", "comfyui", "stable_diffusion_cpp"}


def test_allow_uploads_defaults_off_and_parses(monkeypatch):
    cfg = _load_config_with_section(monkeypatch, {})
    assert cfg.minimax_video_allow_uploads is False
    cfg2 = _load_config_with_section(monkeypatch, {"minimax": {"allow_uploads": True}})
    assert cfg2.minimax_video_allow_uploads is True


def test_retention_choice_and_clamps(monkeypatch):
    from tldw_chatbook.Video_Generation import config as c
    cfg = _load_config_with_section(monkeypatch, {"retention": "bogus", "max_store_mb": 0})
    assert cfg.retention == c.DEFAULT_RETENTION  # invalid choice falls back
    assert cfg.max_store_mb == 1  # clamped >= 1
    cfg2 = _load_config_with_section(monkeypatch, {"retention": "TTL", "retention_ttl_hours": 48})
    assert cfg2.retention == "ttl"  # normalized lowercase
    assert cfg2.retention_ttl_hours == 48


def _capture_warnings():
    """loguru is this project's logger; caplog does not intercept it -- attach
    a temporary sink and return (messages, sink_id)."""
    from loguru import logger as loguru_logger
    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="WARNING", format="{message}")
    return messages, sink_id


def test_flat_backend_key_under_video_generation_warns_with_nested_replacement(monkeypatch):
    # Same trap as the image package's task-621: writing the FLAT dataclass
    # field name directly under [video_generation] is silently ignored -- it
    # must log a warning naming the key and the exact nested replacement.
    from loguru import logger as loguru_logger
    from tldw_chatbook.Video_Generation import config as c
    fake = {"minimax_video_default_model": "MiniMax-Hailuo-2.3"}
    monkeypatch.setattr(c, "_read_video_generation_toml", lambda: fake, raising=False)
    monkeypatch.setattr(c, "_keyring_get", lambda b: None, raising=False)

    messages, sink_id = _capture_warnings()
    try:
        cfg = c.get_video_generation_config(reload=True)
    finally:
        loguru_logger.remove(sink_id)

    # Silently ignored: the flat key never reaches the dataclass field.
    assert cfg.minimax_video_default_model == c.DEFAULT_MINIMAX_VIDEO_MODEL

    matches = [m for m in messages if "minimax_video_default_model" in m]
    assert len(matches) == 1
    assert "[video_generation.minimax] default_model" in matches[0]
