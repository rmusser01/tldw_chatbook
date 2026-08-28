"""Atomic exact-model Console default persistence tests."""

from __future__ import annotations

from pathlib import Path
import threading
import tomllib

import pytest
import toml

from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_settings_apply import (
    FULL_MODEL_DEFAULT_FIELDS,
    QUICK_MODEL_DEFAULT_FIELDS,
    ConsoleEndpointDraft,
    ConsoleSettingsAction,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
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
    monkeypatch.setattr(defaults_module, "_PENDING_RETRY_STATE", None, raising=False)


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


@pytest.mark.parametrize("changed_target", ["profile", "global", "endpoint"])
def test_retry_rejects_external_change_to_any_owned_field(
    tmp_path: Path,
    monkeypatch,
    changed_target: str,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config(section="openai"))
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    endpoint = ConsoleEndpointPatch(
        value="https://new.example.test/v1",
        bound_provider_config_key="openai",
        dirty=True,
        checked=True,
    )
    intent = _intent(
        action=ConsoleSettingsAction.MAKE_NEW_CHAT_DEFAULT,
        field_mask=FULL_MODEL_DEFAULT_FIELDS,
        endpoint_patch=endpoint,
    )
    real_write = config_module.atomic_private_write_text
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("first failure")),
    )
    failed = apply_console_default_intent(intent)
    assert failed.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE

    externally_changed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    if changed_target == "profile":
        externally_changed["api_settings"]["openai"]["model_defaults"][
            LITERAL_MODEL
        ]["temperature"] = 0.67
    elif changed_target == "global":
        externally_changed["chat_defaults"]["model"] = "external-model"
    else:
        externally_changed["api_settings"]["openai"][
            "api_base_url"
        ] = "https://external.example.test/v1"
    externally_changed["unrelated"] = {"concurrent": "newer"}
    _write_config(config_path, externally_changed)
    before_retry = config_path.read_bytes()
    monkeypatch.setattr(config_module, "atomic_private_write_text", real_write)

    retried = apply_console_default_intent(intent)

    assert retried.file_replaced is False
    assert retried.runtime_published is False
    assert retried.settings_view is None
    assert retried.failure_phase is ConsoleDefaultSavePhase.BEFORE_REPLACE
    assert config_path.read_bytes() == before_retry


def test_retry_rebases_when_only_sibling_and_unrelated_fields_changed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    intent = _intent(values={"temperature": 0.31, "streaming": True})
    real_write = config_module.atomic_private_write_text
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("first failure")),
    )
    assert (
        apply_console_default_intent(intent).failure_phase
        is ConsoleDefaultSavePhase.BEFORE_REPLACE
    )
    externally_changed = tomllib.loads(config_path.read_text(encoding="utf-8"))
    externally_changed["api_settings"]["OpenAI"]["model_defaults"][
        "sibling/model"
    ]["temperature"] = 0.58
    externally_changed["unrelated"] = {"concurrent": "newer"}
    _write_config(config_path, externally_changed)
    monkeypatch.setattr(config_module, "atomic_private_write_text", real_write)

    retried = apply_console_default_intent(intent)

    assert retried.runtime_published is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.31
    assert saved["api_settings"]["OpenAI"]["model_defaults"]["sibling/model"] == {
        "temperature": 0.58
    }
    assert saved["unrelated"] == {"concurrent": "newer"}


def test_newer_generation_cannot_become_current_mid_transaction(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    generation_a_paused = threading.Event()
    release_generation_a = threading.Event()
    generation_b_invoked = threading.Event()
    generation_b_reserved = threading.Event()
    violations: list[str] = []
    real_apply = config_module._apply_literal_mutation_unlocked
    real_reserve = defaults_module._reserve_intent_generation
    real_write = config_module.atomic_private_write_text

    def paused_apply(config_data, mutation):
        if threading.current_thread().name == "generation-a":
            generation_a_paused.set()
            if not release_generation_a.wait(timeout=5):
                raise AssertionError("generation A was not released")
        return real_apply(config_data, mutation)

    def observed_reserve(intent):
        reserved = real_reserve(intent)
        if intent.generation == 2 and reserved is not None:
            generation_b_reserved.set()
        return reserved

    def observed_write(*args, **kwargs):
        if (
            threading.current_thread().name == "generation-a"
            and defaults_module._LATEST_INTENT_GENERATION == 2
        ):
            violations.append("generation A replaced after B became current")
        return real_write(*args, **kwargs)

    monkeypatch.setattr(config_module, "_apply_literal_mutation_unlocked", paused_apply)
    monkeypatch.setattr(defaults_module, "_reserve_intent_generation", observed_reserve)
    monkeypatch.setattr(config_module, "atomic_private_write_text", observed_write)
    outcomes = {}
    intent_a = _intent(
        generation=1,
        values={"temperature": 0.1, "streaming": True},
    )
    intent_b = _intent(
        generation=2,
        values={"temperature": 0.6, "streaming": False},
    )

    worker_a = threading.Thread(
        name="generation-a",
        target=lambda: outcomes.setdefault("a", apply_console_default_intent(intent_a)),
    )
    def invoke_generation_b() -> None:
        generation_b_invoked.set()
        outcomes.setdefault("b", apply_console_default_intent(intent_b))

    worker_b = threading.Thread(name="generation-b", target=invoke_generation_b)
    worker_a.start()
    assert generation_a_paused.wait(timeout=5)
    worker_b.start()
    assert generation_b_invoked.wait(timeout=5)
    reserved_while_a_paused = generation_b_reserved.wait(timeout=0.25)
    release_generation_a.set()
    worker_a.join(timeout=5)
    worker_b.join(timeout=5)

    assert not worker_a.is_alive()
    assert not worker_b.is_alive()
    assert reserved_while_a_paused is False
    assert violations == []
    assert outcomes["a"].runtime_published is True
    assert outcomes["b"].runtime_published is True
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["OpenAI"]["model_defaults"][LITERAL_MODEL][
        "temperature"
    ] == 0.6


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


def test_noop_disk_state_cache_failure_still_requires_cache_only_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()
    writes = []
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: writes.append((args, kwargs)),
    )
    monkeypatch.setattr(
        config_module,
        "load_settings",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("cache detail")),
    )

    outcome = apply_console_default_intent(
        _intent(
            field_mask=FULL_MODEL_DEFAULT_FIELDS,
            values={"streaming": None},
        )
    )

    assert outcome.file_replaced is False
    assert outcome.runtime_published is False
    assert outcome.settings_view is None
    assert outcome.failure_phase is ConsoleDefaultSavePhase.CACHE_PUBLICATION
    assert writes == []
    assert config_path.read_bytes() == original


def test_runtime_refresh_contains_lock_failure_and_never_writes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, _ready_openai_config())
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    class FailingLock:
        def __enter__(self):
            raise OSError("lock detail")

        def __exit__(self, exc_type, exc_value, traceback):
            return False

    monkeypatch.setattr(config_module, "_config_write_lock", lambda _path: FailingLock())
    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("refresh must not write")
        ),
    )

    result = refresh_console_runtime_after_saved_default()

    assert result.published is False
    assert result.settings_view is None
    assert result.failure_phase == "cache_reload"


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
    before_lock = threading.Event()
    external_edit_complete = threading.Event()
    real_config_write_lock = config_module._config_write_lock

    class DelayedConfigWriteLock:
        def __init__(self, path: Path) -> None:
            self._path = path
            self._inner = None

        def __enter__(self):
            before_lock.set()
            if not external_edit_complete.wait(timeout=5):
                raise AssertionError("external edit did not complete")
            self._inner = real_config_write_lock(self._path)
            return self._inner.__enter__()

        def __exit__(self, exc_type, exc_value, traceback):
            assert self._inner is not None
            return self._inner.__exit__(exc_type, exc_value, traceback)

    monkeypatch.setattr(
        config_module,
        "_config_write_lock",
        lambda path: DelayedConfigWriteLock(path),
    )
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(apply_console_default_intent(intent)),
    )
    worker.start()
    assert before_lock.wait(timeout=5)
    externally_changed = _ready_openai_config(section="OpenAI")
    externally_changed["api_settings"]["OpenAI"].pop("api_key")
    externally_changed["api_settings"]["OpenAI"]["credential_source"] = "stored"
    _write_config(config_path, externally_changed)
    original = config_path.read_bytes()
    external_edit_complete.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    outcome = outcomes[0]

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


def test_build_default_intent_full_uses_only_exposed_profile_overrides() -> None:
    drafts = (
        ConsoleSettingsFieldDraft(
            name="temperature",
            effective_value=0.73,
            profile_override=None,
            provenance=ConsoleSettingsFieldProvenance.INHERITED,
            dirty=True,
        ),
        ConsoleSettingsFieldDraft(
            name="top_p",
            effective_value=0.91,
            profile_override=0.42,
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
        ConsoleSettingsFieldDraft(
            name="not_exposed_by_mask",
            effective_value="effective",
            profile_override="override",
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
    )
    endpoint = ConsoleEndpointDraft(
        value="https://localhost:8080/v1",
        bound_provider_config_key="openai",
        dirty=True,
        checked=True,
    )

    intent = defaults_module.build_console_default_intent(
        generation=7,
        action=ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        provider_config_key="openai",
        literal_model_id=LITERAL_MODEL,
        field_drafts=drafts,
        field_mask=FULL_MODEL_DEFAULT_FIELDS,
        endpoint=endpoint,
    )

    assert dict(intent.values) == {"temperature": None, "top_p": 0.42}
    assert intent.endpoint_patch == ConsoleEndpointPatch(
        value=endpoint.value,
        bound_provider_config_key="openai",
        dirty=True,
        checked=True,
    )


def test_build_default_intent_quick_materializes_displayed_effective_values() -> None:
    drafts = (
        ConsoleSettingsFieldDraft(
            name="temperature",
            effective_value=0.73,
            profile_override=None,
            provenance=ConsoleSettingsFieldProvenance.INHERITED,
            dirty=False,
        ),
        ConsoleSettingsFieldDraft(
            name="streaming",
            effective_value=False,
            profile_override=None,
            provenance=ConsoleSettingsFieldProvenance.INHERITED,
            dirty=False,
        ),
        ConsoleSettingsFieldDraft(
            name="top_p",
            effective_value=0.91,
            profile_override=0.42,
            provenance=ConsoleSettingsFieldProvenance.EXPLICIT,
            dirty=True,
        ),
    )

    intent = defaults_module.build_console_default_intent(
        generation=8,
        action=ConsoleSettingsAction.SAVE_MODEL_DEFAULT,
        provider_config_key="openai",
        literal_model_id=LITERAL_MODEL,
        field_drafts=drafts,
        field_mask=QUICK_MODEL_DEFAULT_FIELDS,
        endpoint=None,
    )

    assert dict(intent.values) == {"temperature": 0.73, "streaming": False}


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
        "example.com\\secret",
        "https://example..com/v1",
        "https://-bad.example/v1",
        "https://bad-.example/v1",
        "https://bad_host.example/v1",
        "https://example.com|evil/v1",
        "https://example.com;evil/v1",
        "https://ex\x00ample.com/v1",
        "https://café.example/v1",
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
