"""Tests for config_module.delete_settings_from_cli_config."""

import os
from pathlib import Path
import threading
import tomllib

import pytest
import toml

from tldw_chatbook import config as config_module


def _write_config(config_path, data: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(toml.dumps(data), encoding="utf-8")


def test_deletes_existing_keys_from_nested_section(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {
                    "console_rail_state:ws:orphan-1": {
                        "left_open": True,
                        "right_open": False,
                    },
                    "console_rail_state:ws:live-conv": {
                        "left_open": False,
                        "right_open": True,
                    },
                }
            }
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:orphan-1"],
    )

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    rail_state = saved["console"]["rail_state"]
    assert "console_rail_state:ws:orphan-1" not in rail_state
    assert rail_state["console_rail_state:ws:live-conv"] == {
        "left_open": False,
        "right_open": True,
    }


def test_missing_section_is_a_noop_returning_true(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"chat_defaults": {"streaming": True}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_content = config_path.read_text(encoding="utf-8")
    original_mtime_ns = config_path.stat().st_mtime_ns

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["whatever-key"],
    )

    assert config_path.read_text(encoding="utf-8") == original_content
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_missing_file_returns_true(tmp_path, monkeypatch):
    config_path = tmp_path / "does-not-exist" / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["some-key"],
    )

    assert not config_path.exists()


def test_non_matching_delete_leaves_file_byte_identical(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {
                    "console_rail_state:ws:live-conv": {
                        "left_open": False,
                        "right_open": True,
                    },
                }
            }
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()
    original_mtime_ns = config_path.stat().st_mtime_ns

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:key-that-does-not-exist"],
    )

    # No key was actually removed, so the file must not be rewritten at all.
    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_other_sections_and_keys_are_untouched(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {
                    "console_rail_state:ws:orphan-1": {"left_open": True},
                    "console_rail_state:ws:live-conv": {"left_open": False},
                },
                "collapse_large_pastes": True,
            },
            "chat_defaults": {"streaming": True, "temperature": 0.33},
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:orphan-1"],
    )

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert "console_rail_state:ws:orphan-1" not in saved["console"]["rail_state"]
    assert saved["console"]["rail_state"]["console_rail_state:ws:live-conv"] == {
        "left_open": False,
    }
    assert saved["console"]["collapse_large_pastes"] is True
    assert saved["chat_defaults"] == {"streaming": True, "temperature": 0.33}


def test_delete_preserves_existing_file_permissions(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "console": {
                "rail_state": {"console_rail_state:ws:orphan-1": {"left_open": True}}
            }
        },
    )
    os.chmod(config_path, 0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.delete_settings_from_cli_config(
        "console.rail_state",
        ["console_rail_state:ws:orphan-1"],
    )

    # A hardened 0600 config must not be silently widened by the atomic write.
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_save_preserves_existing_file_permissions(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"chat_defaults": {"streaming": True}})
    os.chmod(config_path, 0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    assert config_module.save_settings_to_cli_config(
        {"chat_defaults": {"temperature": 0.5}}
    )

    assert config_path.stat().st_mode & 0o777 == 0o600


def test_structured_mutation_sets_and_deletes_with_one_atomic_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "app_tts": {
                "default_provider": "openai",
                "default_model": "stale-model",
                "default_voice": "stale-voice",
                "unrelated": "preserved",
            },
            "tts_settings": {
                "default_openai_tts_model": "stale-model",
                "default_tts_voice": "stale-voice",
                "provider_specific_default": "preserved",
            },
            "unrelated": {"enabled": True},
        },
    )
    os.chmod(config_path, 0o600)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    atomic_write = config_module.atomic_private_write_text
    load_settings = config_module.load_settings
    write_calls = 0
    reload_calls = 0

    def counted_atomic_write(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        return atomic_write(*args, **kwargs)

    def counted_load_settings(*args, **kwargs):
        nonlocal reload_calls
        reload_calls += 1
        return load_settings(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        counted_atomic_write,
    )
    monkeypatch.setattr(config_module, "load_settings", counted_load_settings)

    result = config_module.apply_settings_mutation_to_cli_config(
        {
            "app_tts": {
                "default_provider": "audio_cpp",
                "default_model_mode": "first_available",
                "default_voice_mode": "server_default",
                "default_format": "wav",
                "default_speed": 1.0,
            }
        },
        delete_keys={
            "app_tts": ("default_model", "default_voice"),
            "tts_settings": (
                "default_openai_tts_model",
                "default_tts_voice",
            ),
        },
    )

    assert result == config_module.ConfigMutationResult(
        file_replaced=True,
        caches_reloaded=True,
        failure_phase=None,
    )
    assert result.fully_applied is True
    assert write_calls == 1
    assert reload_calls == 1
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["app_tts"] == {
        "default_provider": "audio_cpp",
        "default_model_mode": "first_available",
        "default_voice_mode": "server_default",
        "default_format": "wav",
        "default_speed": 1.0,
        "unrelated": "preserved",
    }
    assert saved["tts_settings"] == {
        "provider_specific_default": "preserved",
    }
    assert saved["unrelated"] == {"enabled": True}
    assert config_path.stat().st_mode & 0o777 == 0o600


def test_structured_mutation_rejects_set_delete_overlap_before_write(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_model": "old"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()
    original_mtime_ns = config_path.stat().st_mtime_ns
    write_calls = 0

    def unexpected_write(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        raise AssertionError("overlap must fail before writing")

    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        unexpected_write,
    )

    result = config_module.apply_settings_mutation_to_cli_config(
        {"app_tts": {"default_model": "new"}},
        delete_keys={"app_tts": ("default_model",)},
    )

    assert result == config_module.ConfigMutationResult(
        file_replaced=False,
        caches_reloaded=False,
        failure_phase="before_replace",
    )
    assert result.fully_applied is False
    assert write_calls == 0
    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_structured_mutation_contains_malformed_input_as_validation_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_model": "old"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()

    result = config_module.apply_settings_mutation_to_cli_config(
        {"app_tts": ["not", "a", "mapping"]},  # type: ignore[dict-item]
    )

    assert result == config_module.ConfigMutationResult(
        file_replaced=False,
        caches_reloaded=False,
        failure_phase="before_replace",
    )
    assert config_path.read_bytes() == original_bytes


def test_structured_mutation_reports_cache_reload_failure_after_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_provider": "openai"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    def fail_cache_reload(*args, **kwargs):
        raise RuntimeError("injected cache reload failure")

    monkeypatch.setattr(config_module, "load_settings", fail_cache_reload)
    messages: list[str] = []
    sink_id = config_module.logger.add(
        messages.append,
        level="DEBUG",
        format="{message}",
    )

    try:
        result = config_module.apply_settings_mutation_to_cli_config(
            {"app_tts": {"default_provider": "audio_cpp"}},
        )
    finally:
        config_module.logger.remove(sink_id)

    assert result == config_module.ConfigMutationResult(
        file_replaced=True,
        caches_reloaded=False,
        failure_phase="cache_reload",
    )
    assert result.fully_applied is False
    assert not hasattr(result, "error")
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["app_tts"]["default_provider"] == "audio_cpp"
    rendered = "\n".join(messages)
    assert "phase=cache_reload" in rendered
    assert f"config_path={config_path}" in rendered
    assert "error_type=RuntimeError" in rendered
    assert "injected cache reload failure" not in rendered


def test_structured_mutation_reports_write_failure_before_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_provider": "openai"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()
    original_mtime_ns = config_path.stat().st_mtime_ns

    def fail_atomic_write(*args, **kwargs):
        raise OSError("injected pre-replacement failure")

    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        fail_atomic_write,
    )
    messages: list[str] = []
    sink_id = config_module.logger.add(
        messages.append,
        level="DEBUG",
        format="{message}",
    )

    try:
        result = config_module.apply_settings_mutation_to_cli_config(
            {"app_tts": {"default_provider": "audio_cpp"}},
        )
    finally:
        config_module.logger.remove(sink_id)

    assert result == config_module.ConfigMutationResult(
        file_replaced=False,
        caches_reloaded=False,
        failure_phase="before_replace",
    )
    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime_ns
    rendered = "\n".join(messages)
    assert "phase=before_replace" in rendered
    assert f"config_path={config_path}" in rendered
    assert "error_type=OSError" in rendered
    assert "injected pre-replacement failure" not in rendered


def test_batch_save_delete_keys_delegates_to_one_structured_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "app_tts": {
                "default_provider": "openai",
                "default_model": "stale",
            }
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    atomic_write = config_module.atomic_private_write_text
    write_calls = 0

    def counted_atomic_write(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        return atomic_write(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        counted_atomic_write,
    )

    assert config_module.save_settings_to_cli_config(
        {
            "app_tts": {
                "default_provider": "audio_cpp",
                "default_model_mode": "first_available",
            }
        },
        delete_keys={"app_tts": ("default_model",)},
    )

    assert write_calls == 1
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["app_tts"] == {
        "default_provider": "audio_cpp",
        "default_model_mode": "first_available",
    }


def test_empty_batch_save_remains_a_successful_noop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_provider": "audio_cpp"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()
    original_mtime_ns = config_path.stat().st_mtime_ns

    assert config_module.save_settings_to_cli_config({})

    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_batch_save_with_only_empty_sections_remains_a_successful_noop(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_provider": "audio_cpp"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original_bytes = config_path.read_bytes()
    original_mtime_ns = config_path.stat().st_mtime_ns

    assert config_module.save_settings_to_cli_config(
        {"app_tts": {}, "tts_settings": {}}
    )

    assert config_path.read_bytes() == original_bytes
    assert config_path.stat().st_mtime_ns == original_mtime_ns


def test_shared_lock_prevents_lost_concurrent_set_and_delete_updates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "shared": {
                "obsolete": "remove-me",
                "preserved": "keep-me",
            }
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    real_atomic_write = config_module.atomic_private_write_text
    set_at_write = threading.Event()
    release_set_write = threading.Event()
    delete_attempting_lock = threading.Event()

    class InstrumentedLock:
        def __init__(self, lock) -> None:
            self._lock = lock
            self._state_lock = threading.Lock()
            self._owner: str | None = None
            self.blocked_threads: list[str] = []

        def __enter__(self):
            thread_name = threading.current_thread().name
            with self._state_lock:
                if self._owner is not None:
                    self.blocked_threads.append(thread_name)
            if thread_name == "delete-worker":
                delete_attempting_lock.set()
            self._lock.acquire()
            with self._state_lock:
                self._owner = thread_name
            return self

        def __exit__(self, exc_type, exc_value, traceback) -> None:
            del exc_type, exc_value, traceback
            with self._state_lock:
                self._owner = None
            self._lock.release()

    instrumented_lock = InstrumentedLock(config_module._CONFIG_FILE_LOCK)
    monkeypatch.setattr(config_module, "_CONFIG_FILE_LOCK", instrumented_lock)
    monkeypatch.setattr(
        config_module,
        "load_settings",
        lambda *, force_reload=False: {},
    )

    def controlled_atomic_write(*args, **kwargs):
        if threading.current_thread().name == "set-worker":
            set_at_write.set()
            if not release_set_write.wait(timeout=5):
                raise AssertionError("set worker was not released")
        return real_atomic_write(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        controlled_atomic_write,
    )
    results: dict[str, config_module.ConfigMutationResult] = {}

    def set_value() -> None:
        results["set"] = config_module.apply_settings_mutation_to_cli_config(
            {"new_section": {"value": "set"}},
        )

    def delete_value() -> None:
        results["delete"] = config_module.apply_settings_mutation_to_cli_config(
            {},
            delete_keys={"shared": ("obsolete",)},
        )

    set_thread = threading.Thread(target=set_value, name="set-worker")
    delete_thread = threading.Thread(target=delete_value, name="delete-worker")
    set_thread.start()
    if not set_at_write.wait(timeout=5):
        release_set_write.set()
        set_thread.join(timeout=5)
        pytest.fail("set worker did not reach the atomic write")

    delete_thread.start()
    if not delete_attempting_lock.wait(timeout=5):
        release_set_write.set()
        set_thread.join(timeout=5)
        delete_thread.join(timeout=5)
        pytest.fail("delete worker did not attempt the shared lock")

    release_set_write.set()
    set_thread.join(timeout=5)
    delete_thread.join(timeout=5)

    assert not set_thread.is_alive()
    assert not delete_thread.is_alive()
    assert instrumented_lock.blocked_threads == ["delete-worker"]
    assert results == {
        "set": config_module.ConfigMutationResult(True, True, None),
        "delete": config_module.ConfigMutationResult(True, True, None),
    }
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved == {
        "shared": {"preserved": "keep-me"},
        "new_section": {"value": "set"},
    }


def test_delete_wrapper_performs_one_atomic_write_for_actual_mutation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"app_tts": {"default_model": "stale"}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    atomic_write = config_module.atomic_private_write_text
    write_calls = 0

    def counted_atomic_write(*args, **kwargs):
        nonlocal write_calls
        write_calls += 1
        return atomic_write(*args, **kwargs)

    monkeypatch.setattr(
        config_module,
        "atomic_private_write_text",
        counted_atomic_write,
    )

    assert config_module.delete_settings_from_cli_config(
        "app_tts",
        ["default_model"],
    )

    assert write_calls == 1
