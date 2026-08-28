"""Atomic exact-model Console default persistence tests."""

from __future__ import annotations

from pathlib import Path
import tomllib

import pytest
import toml

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleSettingsAction,
)
from tldw_chatbook.Chat import console_settings_defaults as defaults_module
from tldw_chatbook.Chat.console_settings_defaults import (
    ConsoleDefaultMutationIntent,
    ConsoleDefaultRecoveryAction,
    ConsoleDefaultRecoveryRequest,
    ConsoleDefaultSavePhase,
    ConsoleEndpointPatch,
    apply_console_default_intent,
    format_console_endpoint_preview,
    parse_console_endpoint_preview,
    refresh_console_runtime_after_saved_default,
)


LITERAL_MODEL = "org/model.v2:fast[beta]"


def _write_config(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(toml.dumps(data), encoding="utf-8")


def _ready_openai_config(*, section: str = "OpenAI") -> dict[str, object]:
    return {
        "api_settings": {
            section: {
                "credential_source": "stored",
                "api_key": "test-key",
                "api_base_url": "https://old.example.test/v1",
                "model_defaults": {
                    LITERAL_MODEL: {
                        "temperature": 0.8,
                        "top_p": 0.9,
                        "max_tokens": 1234,
                        "unexposed": "preserved",
                    },
                    "sibling/model": {"temperature": 0.4},
                },
            }
        },
        "chat_defaults": {"provider": "anthropic", "model": "old-model"},
        "unrelated": {"concurrent": "preserved"},
    }


@pytest.fixture(autouse=True)
def _reset_default_generation(monkeypatch):
    monkeypatch.setattr(defaults_module, "_LATEST_INTENT_GENERATION", None)
    monkeypatch.setattr(defaults_module, "_LATEST_INTENT_FINGERPRINT", None)


def _intent(
    *,
    generation: int = 1,
    action: ConsoleSettingsAction = ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
    field_mask: frozenset[str] = QUICK_MODEL_DEFAULT_FIELDS,
    values: dict[str, object | None] | None = None,
    endpoint_patch: ConsoleEndpointPatch | None = None,
) -> ConsoleDefaultMutationIntent:
    return ConsoleDefaultMutationIntent(
        generation=generation,
        action=action,
        provider_config_key="openai",
        literal_model_id=LITERAL_MODEL,
        field_mask=field_mask,
        values=(
            {"temperature": 0.25, "streaming": False}
            if values is None
            else values
        ),
        endpoint_patch=endpoint_patch,
    )


def test_quick_save_patches_only_temperature_and_streaming(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    outcome = apply_console_default_intent(
        _intent(values={"temperature": 0.25, "streaming": False, "top_p": 0.1})
    )

    assert outcome.file_replaced is True
    assert outcome.runtime_published is True
    assert outcome.settings_view is not None
    assert outcome.failure_phase is None
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    profile = saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL]
    assert profile == {
        "temperature": 0.25,
        "streaming": False,
        "top_p": 0.9,
        "max_tokens": 1234,
        "unexposed": "preserved",
    }
    assert saved["chat_defaults"] == {
        "provider": "anthropic",
        "model": "old-model",
    }


def test_full_save_deletes_exact_inherited_fields_and_preserves_siblings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    outcome = apply_console_default_intent(
        _intent(
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
            values={
                "temperature": None,
                "streaming": None,
                "top_p": 0.72,
                "thinking_budget_tokens": 222,
                "not_exposed": "ignored",
            },
        )
    )

    assert outcome.runtime_published is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    profiles = saved["api_settings"]["OpenAI"]["model_defaults"]
    assert profiles[LITERAL_MODEL] == {
        "top_p": 0.72,
        "max_tokens": 1234,
        "unexposed": "preserved",
    }
    assert profiles["sibling/model"] == {"temperature": 0.4}
    assert saved["unrelated"] == {"concurrent": "preserved"}


@pytest.mark.parametrize(
    "intent",
    [
        pytest.param(
            _intent(
                field_mask=frozenset({"temperature"}),
                values={"temperature": 0.2},
            ),
            id="partial-non-surface-mask",
        ),
        pytest.param(
            _intent(values={"temperature": None, "streaming": True}),
            id="quick-inherit-temperature",
        ),
        pytest.param(
            _intent(values={"temperature": 0.2}),
            id="quick-missing-streaming",
        ),
        pytest.param(
            _intent(
                field_mask=FULL_MODEL_DEFAULT_FIELDS,
                values={"streaming": "false"},
            ),
            id="non-boolean-streaming",
        ),
    ],
)
def test_default_intent_rejects_non_surface_masks_and_unmaterialized_quick_values(
    tmp_path: Path,
    monkeypatch,
    intent: ConsoleDefaultMutationIntent,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()

    outcome = apply_console_default_intent(intent)

    assert outcome.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    assert outcome.settings_view is None
    assert config_path.read_bytes() == original


def test_make_default_atomically_patches_profile_globals_and_checked_endpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config(section="openai"))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    write_calls = 0
    real_write = config_module.atomic_private_write_text

    def counted_write(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        return real_write(*args, **kwargs)

    monkeypatch.setattr(config_module, "atomic_private_write_text", counted_write)
    endpoint = ConsoleEndpointPatch(
        value="https://new.example.test:8443/v1?ignored=yes",
        bound_provider_config_key="openai",
        dirty=True,
        checked=True,
    )
    locked = config_module.get_atomic_config_snapshot()
    target = defaults_module.build_target_default_console_session_settings(
        locked.values,
        "openai",
        LITERAL_MODEL,
    )
    assert defaults_module.validate_console_session_settings(
        target,
        app_config=locked.values,
    ) == []
    assert defaults_module.build_console_settings_readiness(
        target,
        app_config=locked.values,
    ).native_send_supported is True

    outcome = apply_console_default_intent(
        _intent(
            action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
            endpoint_patch=endpoint,
        )
    )

    assert outcome.runtime_published is True
    assert write_calls == 1
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["chat_defaults"] == {
        "provider": "openai",
        "model": LITERAL_MODEL,
    }
    assert saved["api_settings"]["openai"]["api_base_url"] == endpoint.value
    assert saved["api_settings"]["openai"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.25


@pytest.mark.parametrize(
    ("action", "mask", "patch"),
    [
        (
            ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
            FULL_MODEL_DEFAULT_FIELDS,
            ConsoleEndpointPatch(
                "https://new.example.test/v1", "openai", True, True
            ),
        ),
        (
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            QUICK_MODEL_DEFAULT_FIELDS,
            ConsoleEndpointPatch(
                "https://new.example.test/v1", "openai", True, True
            ),
        ),
        (
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            FULL_MODEL_DEFAULT_FIELDS,
            ConsoleEndpointPatch(
                "https://new.example.test/v1", "anthropic", True, True
            ),
        ),
        (
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            FULL_MODEL_DEFAULT_FIELDS,
            ConsoleEndpointPatch(
                "https://new.example.test/v1", "openai", False, True
            ),
        ),
        (
            ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            FULL_MODEL_DEFAULT_FIELDS,
            ConsoleEndpointPatch(
                "https://new.example.test/v1", "openai", True, False
            ),
        ),
    ],
)
def test_unauthorized_endpoint_patch_fails_closed_without_writing(
    tmp_path: Path,
    monkeypatch,
    action: ConsoleSettingsAction,
    mask: frozenset[str],
    patch: ConsoleEndpointPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()

    outcome = apply_console_default_intent(
        _intent(action=action, field_mask=mask, endpoint_patch=patch)
    )

    assert outcome.file_replaced is False
    assert outcome.runtime_published is False
    assert outcome.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    assert config_path.read_bytes() == original


def test_before_replace_failure_retains_immutable_retry_intent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    values: dict[str, object | None] = {"temperature": 0.31, "streaming": True}
    intent = _intent(values=values)
    real_write = config_module.atomic_private_write_text
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private detail")),
    )

    failed = apply_console_default_intent(intent)

    assert failed.file_replaced is False
    assert failed.runtime_published is False
    assert failed.settings_view is None
    assert failed.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    values["temperature"] = 1.99
    monkeypatch.setattr(config_module, "atomic_private_write_text", real_write)

    retried = apply_console_default_intent(intent)

    assert retried.runtime_published is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.31


def test_cache_failure_is_saved_and_refresh_continuation_never_rewrites(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    real_load = config_module.load_settings
    monkeypatch.setattr(
        config_module,
        "load_settings",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache detail")),
    )

    outcome = apply_console_default_intent(_intent())

    assert outcome.file_replaced is True
    assert outcome.runtime_published is False
    assert outcome.settings_view is None
    assert outcome.failure_phase is ConsoleDefaultSavePhase.CACHE_PUBLICATION
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.25
    monkeypatch.setattr(config_module, "load_settings", real_load)
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("refresh must not write")
        ),
    )

    refreshed = refresh_console_runtime_after_saved_default()

    assert refreshed.published is True
    assert refreshed.settings_view is not None
    assert refreshed.failure_phase is None


def test_newer_intent_supersedes_older_retry_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    old = _intent(generation=10, values={"temperature": 0.1, "streaming": True})
    real_write = config_module.atomic_private_write_text
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("first failure")),
    )
    assert apply_console_default_intent(old).failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    monkeypatch.setattr(config_module, "atomic_private_write_text", real_write)
    newer = _intent(
        generation=11,
        values={"temperature": 0.6, "streaming": False},
    )
    assert apply_console_default_intent(newer).runtime_published is True

    stale = apply_console_default_intent(old)

    assert stale.file_replaced is False
    assert stale.runtime_published is False
    assert stale.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.6


def test_make_default_rechecks_locked_readiness_and_rejects_removed_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config(section="openai"))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    intent = _intent(action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT)
    externally_changed = _ready_openai_config(section="OpenAI")
    externally_changed["api_settings"]["OpenAI"].pop("api_key")
    externally_changed["api_settings"]["OpenAI"]["credential_source"] = "stored"
    _write_config(config_path, externally_changed)
    original = config_path.read_bytes()

    outcome = apply_console_default_intent(intent)

    assert outcome.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    assert outcome.settings_view is None
    assert config_path.read_bytes() == original
    assert "openai" not in tomllib.loads(config_path.read_text(encoding="utf-8"))[
        "api_settings"
    ]


def test_locked_builder_uses_authoritative_raw_alias_and_preserves_newer_edits(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    current = _ready_openai_config(section="OpenAI")
    current["unrelated"] = {"concurrent": "newer"}
    current["api_settings"]["OpenAI"]["newer_provider_field"] = 42
    _write_config(config_path, current)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    outcome = apply_console_default_intent(_intent())

    assert outcome.runtime_published is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "openai" not in saved["api_settings"]
    assert saved["api_settings"]["OpenAI"]["newer_provider_field"] == 42
    assert saved["unrelated"] == {"concurrent": "newer"}


def test_make_default_uses_raw_alias_while_locked_readiness_uses_effective_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    current = _ready_openai_config(section="OpenAI")
    current["api_settings"]["OpenAI"].pop("api_key")
    _write_config(config_path, current)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENAI_API_KEY", "effective-test-key")

    outcome = apply_console_default_intent(
        _intent(
            action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
        )
    )

    assert outcome.runtime_published is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "openai" not in saved["api_settings"]
    assert saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.25
    assert saved["chat_defaults"] == {
        "provider": "openai",
        "model": LITERAL_MODEL,
    }


@pytest.mark.parametrize(
    ("value", "authority", "classification"),
    [
        ("http://localhost:8080/v1?x=1#frag", "localhost:8080", "Local"),
        ("https://127.0.0.1/path", "127.0.0.1", "Local"),
        ("192.168.1.20:8443/v1", "192.168.1.20:8443", "LAN"),
        ("http://[::1]:9000/v1", "[::1]:9000", "Local"),
        ("https://8.8.8.8/v1", "8.8.8.8", "Remote"),
        ("host.example:443/path", "host.example:443", "Remote/unknown"),
        ("printer.local/api", "printer.local", "LAN"),
    ],
)
def test_endpoint_preview_exposes_only_authority_and_network_class(
    value: str,
    authority: str,
    classification: str,
) -> None:
    preview = parse_console_endpoint_preview(value)

    assert preview is not None
    assert preview.authority == authority
    assert preview.network_classification == classification
    assert format_console_endpoint_preview(value) == f"{authority} · {classification}"
    assert not any(token in preview.authority for token in ("http", "/", "?", "#", "@"))


@pytest.mark.parametrize(
    "value",
    [
        "https://user:password@example.test/v1",
        "user@example.test:8080",
        "https://example.test:99999/v1",
        "https:///missing-host",
        "",
    ],
)
def test_endpoint_preview_rejects_credentials_and_invalid_authorities(value: str) -> None:
    assert parse_console_endpoint_preview(value) is None
    assert format_console_endpoint_preview(value) is None


def test_recovery_request_is_bounded_to_action_and_intent_generation() -> None:
    request = ConsoleDefaultRecoveryRequest(
        action=ConsoleDefaultRecoveryAction.RETRY_SAVE,
        intent_generation=19,
    )

    assert request.action is ConsoleDefaultRecoveryAction.RETRY_SAVE
    assert request.intent_generation == 19
