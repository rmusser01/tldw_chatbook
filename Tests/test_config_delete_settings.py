"""Tests for config_module.delete_settings_from_cli_config."""

import json
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

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

    instrumented_lock = InstrumentedLock(config_module._settings_rebuild_lock())
    monkeypatch.setattr(config_module, "_SETTINGS_REBUILD_LOCK", instrumented_lock)
    monkeypatch.setattr(
        config_module,
        "load_settings",
        lambda *, force_reload=False, reload_bootstrap=None: {},
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


def test_revisioned_section_replace_is_atomic_and_preserves_other_sections(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "speech_studio": {
                "schema_version": 1,
                "revision": 2,
                "selection": {"provider_id": "openai"},
            },
            "global": {"credential": "preserved", "enabled": True},
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    result = config_module.replace_revisioned_settings_section_to_cli_config(
        "speech_studio",
        {
            "schema_version": 1,
            "revision": 3,
            "selection": {"provider_id": "audio_cpp"},
        },
        expected_revision=2,
    )

    assert result == config_module.ConfigMutationResult(True, True, None)
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved == {
        "speech_studio": {
            "schema_version": 1,
            "revision": 3,
            "selection": {"provider_id": "audio_cpp"},
        },
        "global": {"credential": "preserved", "enabled": True},
    }


def test_revisioned_section_replace_reports_conflict_without_writing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {"speech_studio": {"schema_version": 1, "revision": 4}},
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()

    result = config_module.replace_revisioned_settings_section_to_cli_config(
        "speech_studio",
        {"schema_version": 1, "revision": 4},
        expected_revision=3,
    )

    assert result == config_module.ConfigMutationResult(
        False,
        False,
        None,
        conflict=True,
    )
    assert result.fully_applied is False
    assert config_path.read_bytes() == original


@pytest.mark.parametrize("second_writer", ["revisioned", "generic"])
def test_config_writes_serialize_across_processes(
    tmp_path: Path,
    second_writer: str,
) -> None:
    config_path = tmp_path / "config.toml"
    data_path = tmp_path / "data"
    user_data_path = data_path / "cross_process_writer"
    user_data_path.mkdir(parents=True)
    data_path.chmod(0o700)
    user_data_path.chmod(0o700)
    _write_config(
        config_path,
        {
            "general": {"users_name": "cross_process_writer"},
            "paths": {"data_dir": str(data_path)},
            "speech_studio": {"schema_version": 1, "revision": 0},
        },
    )
    go_path = tmp_path / "go"
    ready_paths = (tmp_path / "ready-1", tmp_path / "ready-2")
    write_paths = (tmp_path / "write-1", tmp_path / "write-2")
    script = r"""
import json
from pathlib import Path
import sys
import time

from tldw_chatbook import config as config_module

ready_path = Path(sys.argv[1])
go_path = Path(sys.argv[2])
write_path = Path(sys.argv[3])
other_write_path = Path(sys.argv[4])
provider_id = sys.argv[5]
writer_kind = sys.argv[6]
real_write = config_module._write_raw_cli_config_unlocked

def coordinated_write(config_path, config_data):
    write_path.write_text("ready", encoding="utf-8")
    deadline = time.monotonic() + 0.75
    while not other_write_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    real_write(config_path, config_data)

config_module._write_raw_cli_config_unlocked = coordinated_write
ready_path.write_text("ready", encoding="utf-8")
while not go_path.exists():
    time.sleep(0.01)

if writer_kind == "revisioned":
    result = config_module.replace_revisioned_settings_section_to_cli_config(
        "speech_studio",
        {
            "schema_version": 1,
            "revision": 1,
            "selection": {"provider_id": provider_id},
        },
        expected_revision=0,
    )
else:
    result = config_module.apply_settings_mutation_to_cli_config(
        {"global": {"ordinary_writer": True}},
    )
print(json.dumps({
    "replaced": result.file_replaced,
    "conflict": result.conflict,
    "failure_phase": result.failure_phase,
}))
"""
    environment = os.environ.copy()
    environment["TLDW_CONFIG_PATH"] = str(config_path)
    repository_root = Path(__file__).resolve().parents[1]
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(repository_root), environment.get("PYTHONPATH", "")))
    )
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                script,
                str(ready_paths[index]),
                str(go_path),
                str(write_paths[index]),
                str(write_paths[1 - index]),
                provider_id,
                "revisioned" if index == 0 else second_writer,
            ],
            cwd=repository_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for index, provider_id in enumerate(("openai", "alltalk"))
    ]
    try:
        deadline = time.monotonic() + 20.0
        while not all(path.exists() for path in ready_paths):
            exited = next(
                (process for process in processes if process.poll() is not None),
                None,
            )
            if exited is not None:
                stdout, stderr = exited.communicate()
                pytest.fail(
                    "cross-process writer exited before readiness: "
                    f"stdout={stdout!r}, stderr={stderr!r}"
                )
            if time.monotonic() >= deadline:
                pytest.fail("cross-process writers did not become ready")
            time.sleep(0.01)
        go_path.write_text("go", encoding="utf-8")
        completed = [process.communicate(timeout=30.0) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()

    outcomes = []
    for process, (stdout, stderr) in zip(processes, completed, strict=True):
        assert process.returncode == 0, stderr
        outcome = json.loads(stdout.strip().splitlines()[-1])
        outcomes.append(outcome)
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["speech_studio"]["revision"] == 1
    if second_writer == "revisioned":
        assert sorted(
            (outcome["replaced"], outcome["conflict"]) for outcome in outcomes
        ) == [(False, True), (True, False)], outcomes
        assert saved["speech_studio"]["selection"]["provider_id"] in {
            "openai",
            "alltalk",
        }
    else:
        assert all(outcome["replaced"] for outcome in outcomes), outcomes
        assert not any(outcome["conflict"] for outcome in outcomes), outcomes
        assert saved["speech_studio"]["selection"] == {"provider_id": "openai"}
        assert saved["global"] == {"ordinary_writer": True}


@pytest.mark.parametrize("current", [None, "corrupt", {"revision": "bad"}])
def test_revisioned_section_replace_recovers_missing_or_corrupt_revision_zero(
    tmp_path: Path,
    monkeypatch,
    current: object,
) -> None:
    config_path = tmp_path / "config.toml"
    raw = {"global": {"keep": True}}
    if current is not None:
        raw["speech_studio"] = current
    _write_config(config_path, raw)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    result = config_module.replace_revisioned_settings_section_to_cli_config(
        "speech_studio",
        {"schema_version": 1, "revision": 1},
        expected_revision=0,
    )

    assert result == config_module.ConfigMutationResult(True, True, None)
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["speech_studio"] == {"schema_version": 1, "revision": 1}
    assert saved["global"] == {"keep": True}


@pytest.mark.parametrize("operation", ["set", "delete"])
def test_generic_mutation_cannot_bypass_revisioned_section_owner(
    tmp_path: Path,
    monkeypatch,
    operation: str,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {"speech_studio": {"schema_version": 1, "revision": 4}},
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()

    result = config_module.apply_settings_mutation_to_cli_config(
        ({"speech_studio": {"revision": 1}} if operation == "set" else {}),
        delete_keys=(
            {"speech_studio": ["revision"]} if operation == "delete" else None
        ),
    )

    assert result == config_module.ConfigMutationResult(
        False,
        False,
        "before_replace",
    )
    assert config_path.read_bytes() == original


def test_settings_mutation_precondition_rejects_inside_atomic_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"global": {"keep": True}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()
    checks = []

    result = config_module.apply_settings_mutation_to_cli_config(
        {"global": {"stale": True}},
        mutation_precondition=lambda: checks.append("checked") or False,
    )

    assert result == config_module.ConfigMutationResult(
        False,
        False,
        None,
        conflict=True,
        conflict_reason="identity_changed",
    )
    assert checks == ["checked"]
    assert config_path.read_bytes() == original


def test_settings_locked_snapshot_precondition_observes_authoritative_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {"api_settings": {"moonshot": {"api_region": "china"}}},
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    snapshot = config_module.get_atomic_config_snapshot()
    assert config_module.apply_settings_mutation_to_cli_config(
        {"api_settings.moonshot": {"api_region": "global"}}
    ).fully_applied
    observed = []

    result = config_module.apply_settings_mutation_to_cli_config(
        {"chat_defaults": {"model": "stale-model"}},
        locked_snapshot_precondition=lambda current: observed.append(current) or False,
    )

    assert result == config_module.ConfigMutationResult(
        False,
        False,
        None,
        conflict=True,
        conflict_reason="identity_changed",
    )
    assert len(observed) == 1
    assert observed[0].generation > snapshot.generation
    assert observed[0].values["api_settings"]["moonshot"]["api_region"] == "global"
    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["api_settings"]["moonshot"]["api_region"] == "global"
    assert "stale-model" not in config_path.read_text(encoding="utf-8")


def test_runtime_generation_guard_linearizes_nonblocking_handoff_ack(
    monkeypatch,
) -> None:
    from tldw_chatbook.UI.Navigation.pending_handoff_store import (
        ConsoleFirstChatIntent,
        HandoffChannel,
        PendingHandoffStore,
    )

    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 71)
    pending = PendingHandoffStore()
    pending.stage_reserved_console_first_chat(
        ConsoleFirstChatIntent("future-session", "openai", "model-a", 71)
    )
    claim = pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT)
    assert claim is not None
    writer_attempting = threading.Event()
    writer_published = threading.Event()

    def publish_generation() -> None:
        writer_attempting.set()
        with config_module._config_file_lock():
            config_module._CONFIG_GENERATION += 1
            writer_published.set()

    writer = threading.Thread(target=publish_generation)

    def acknowledge() -> bool:
        writer.start()
        assert writer_attempting.wait(timeout=1)
        assert writer_published.is_set() is False
        return pending.acknowledge_current(claim)

    assert config_module.run_if_runtime_config_generation_current(
        71,
        acknowledge,
    ) is True
    writer.join(timeout=1)

    assert writer.is_alive() is False
    assert writer_published.is_set() is True
    assert config_module._CONFIG_GENERATION == 72
    assert pending.claim(HandoffChannel.CONSOLE_FIRST_CHAT) is None


def test_runtime_generation_guard_skips_ack_after_publication(monkeypatch) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 82)
    acknowledged = []

    assert config_module.run_if_runtime_config_generation_current(
        81,
        lambda: acknowledged.append(True) or True,
    ) is False
    assert acknowledged == []


def test_runtime_generation_guard_callback_exception_does_not_poison_lock(
    monkeypatch,
) -> None:
    monkeypatch.setattr(config_module, "_CONFIG_GENERATION", 83)

    with pytest.raises(RuntimeError, match="callback failure"):
        config_module.run_if_runtime_config_generation_current(
            83,
            lambda: (_ for _ in ()).throw(RuntimeError("callback failure")),
        )

    assert config_module.run_if_runtime_config_generation_current(
        83,
        lambda: True,
    ) is True
    with config_module._config_file_lock():
        config_module._CONFIG_GENERATION += 1
    assert config_module.run_if_runtime_config_generation_current(
        83,
        lambda: True,
    ) is False


@pytest.mark.parametrize("serialized", [False, True])
def test_whole_config_replacement_preserves_revisioned_owned_section(
    tmp_path: Path,
    monkeypatch,
    serialized: bool,
) -> None:
    config_path = tmp_path / "config.toml"
    current_studio = {
        "schema_version": 1,
        "revision": 4,
        "selection": {"provider_id": "audio_cpp"},
    }
    _write_config(
        config_path,
        {
            "speech_studio": current_studio,
            "global": {"old": True},
        },
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    stale_replacement = {
        "speech_studio": {"schema_version": 1, "revision": 1},
        "global": {"new": True},
    }

    if serialized:
        config_module.replace_cli_config_serialized(
            toml.dumps(stale_replacement),
            create_backup=False,
        )
    else:
        config_module.replace_cli_config(stale_replacement)

    saved = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert saved["speech_studio"] == current_studio
    assert saved["global"] == {"new": True}


@pytest.mark.parametrize(
    ("expected_revision", "replacement_revision"),
    [(0, 0), (0, 2), (-1, 0), (True, 2)],
)
def test_revisioned_section_replace_rejects_invalid_revision_transition(
    tmp_path: Path,
    monkeypatch,
    expected_revision: object,
    replacement_revision: int,
) -> None:
    config_path = tmp_path / "config.toml"
    _write_config(config_path, {"global": {"keep": True}})
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    original = config_path.read_bytes()

    result = config_module.replace_revisioned_settings_section_to_cli_config(
        "speech_studio",
        {"schema_version": 1, "revision": replacement_revision},
        expected_revision=expected_revision,  # type: ignore[arg-type]
    )

    assert result == config_module.ConfigMutationResult(
        False,
        False,
        "before_replace",
    )
    assert config_path.read_bytes() == original


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
