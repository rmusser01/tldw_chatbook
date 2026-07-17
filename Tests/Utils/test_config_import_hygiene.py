# Tests/Utils/test_config_import_hygiene.py
"""
Regression tests for TASK-258: config.py import hygiene (single embedded-TOML
parse, consolidated user-config load, lazy chardet).

Before this task:
  1. `config.py` parsed the 1,285-line embedded default TOML twice at import
     -- once into `DEFAULT_CONFIG_FROM_TOML`, and a second time (discarded
     after reading `providers`) into `_temp_loaded_config_for_models`.
  2. `load_cli_config_and_ensure_existence()` and `load_settings()` each
     independently opened+parsed the *same* user config.toml file at import
     (and on every `force_reload=True`), then independently deep-merged it
     onto a fresh copy of `DEFAULT_CONFIG_FROM_TOML`. `load_settings()` also
     probed a "primary/server-component config" path
     (`APP_COMPONENT_ROOT/Config_Files/config.toml`) that never exists in the
     packaged app, so that half of its merge was always a no-op.
  3. `Utils/Utils.py` imported `chardet` (~21ms, ~40 submodules) at module
     scope for two rarely-used functions (`safe_read_file`,
     `FileProcessor.detect_encoding`) that are the module's only callers.

This file has three kinds of coverage, matching the task's ACs:
  (a) An UNMOCKED integration test (the T229 `_real_config` pattern -- see
      Tests/Utils/test_config_nested_settings.py) proving a scratch TOML's
      values are visible through both `get_cli_setting()` (backed by
      `_CONFIG_CACHE`) and `load_settings()`-derived accessors (backed by
      `_SETTINGS_CACHE`) after `force_reload=True`, and that
      `save_setting_to_cli_config()` invalidates both caches together.
  (b) A parse-count test: monkeypatching `tomllib.loads` to count calls
      proves the embedded default TOML is parsed at most once per process,
      even across a forced full reload of both loaders.
  (c) Subprocess-based `sys.modules` assertions proving `chardet` is not
      resident after `import tldw_chatbook.config` (nor after importing
      `tldw_chatbook.Utils.Utils` in isolation), plus in-process functional
      tests proving `safe_read_file`/`FileProcessor.detect_encoding` still
      work, and that the current (no try/except) absence behavior --
      propagating the ImportError from the two call sites -- is unchanged.

      Note: `import tldw_chatbook.app` does end up loading chardet
      transitively, via `requests/__init__.py`'s own top-of-module
      `from chardet import __version__ as chardet_version` dependency-version
      check (unrelated to anything in this codebase) -- so that assertion is
      scoped to `tldw_chatbook.config` and `tldw_chatbook.Utils.Utils`
      directly, which is the actual surface TASK-258 touches.
"""
from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest

import tldw_chatbook.config as config_mod
from tldw_chatbook.config import get_cli_setting

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- (a) Unmocked integration: one load, both caches serving from it -------

SCRATCH_TOML = """
[chat_defaults]
temperature = 0.42
streaming = true

[console]
collapse_large_pastes = true
paste_collapse_threshold = 77
"""


@contextmanager
def _real_config(tmp_path, monkeypatch, toml_text: str):
    """Point the real loader at a scratch TOML; restore + reload afterwards.

    Mirrors Tests/Utils/test_config_nested_settings.py's `_real_config` --
    drives the real TLDW_CONFIG_PATH + force_reload machinery, zero
    monkeypatching of the loaders themselves.
    """
    config_path = tmp_path / "scratch-import-hygiene-config.toml"
    config_path.write_text(toml_text, encoding="utf-8")
    original_env = os.environ.get("TLDW_CONFIG_PATH")
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config_mod.load_cli_config_and_ensure_existence(force_reload=True)
    try:
        yield config_path
    finally:
        if original_env is not None:
            monkeypatch.setenv("TLDW_CONFIG_PATH", original_env)
        else:
            monkeypatch.delenv("TLDW_CONFIG_PATH", raising=False)
        config_mod.load_cli_config_and_ensure_existence(force_reload=True)


class TestBothCachesServeFromOneLoad:
    """AC #2: user config read+parsed once at startup, both caches serving
    from it. Every value below is driven through the real loaders."""

    def test_get_cli_setting_sees_scratch_values(self, tmp_path, monkeypatch):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            assert get_cli_setting("chat_defaults", "temperature", 0.0) == 0.42
            assert get_cli_setting("console", "paste_collapse_threshold", 50) == 77

    def test_load_settings_derived_accessor_sees_same_scratch_values(
        self, tmp_path, monkeypatch
    ):
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            settings = config_mod.load_settings(force_reload=True)
            assert settings["chat_defaults"]["temperature"] == 0.42
            assert settings["console"]["paste_collapse_threshold"] == 77
            # COMPREHENSIVE_CONFIG_RAW is the raw merged tree load_settings()
            # builds internally -- must carry the same values as the
            # extracted top-level keys above (single source, not two
            # independently-reparsed trees that could drift).
            raw = settings["COMPREHENSIVE_CONFIG_RAW"]
            assert raw["chat_defaults"]["temperature"] == 0.42
            assert raw["console"]["paste_collapse_threshold"] == 77

    def test_get_cli_setting_and_load_settings_agree(self, tmp_path, monkeypatch):
        """Both accessors must resolve the identical value for the identical
        key -- the whole point of serving both caches from one load."""
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            via_get_cli_setting = get_cli_setting("chat_defaults", "temperature", None)
            via_load_settings = config_mod.load_settings(force_reload=True)[
                "chat_defaults"
            ]["temperature"]
            assert via_get_cli_setting == via_load_settings == 0.42

    def test_save_setting_invalidates_both_caches(self, tmp_path, monkeypatch):
        """save_setting_to_cli_config() must make the new value visible
        through BOTH accessors without any further force_reload call."""
        with _real_config(tmp_path, monkeypatch, SCRATCH_TOML):
            # Prime both caches with the original scratch value first.
            assert get_cli_setting("chat_defaults", "temperature", None) == 0.42
            assert (
                config_mod.load_settings()["chat_defaults"]["temperature"] == 0.42
            )

            assert config_mod.save_setting_to_cli_config(
                "chat_defaults", "temperature", 0.99
            )

            # No force_reload here on purpose -- save_setting_to_cli_config
            # must have already invalidated + reloaded both caches.
            assert get_cli_setting("chat_defaults", "temperature", None) == 0.99
            assert (
                config_mod.load_settings()["chat_defaults"]["temperature"] == 0.99
            )


# --- (b) Embedded default TOML parsed at most once per process ------------


_EMBEDDED_TOML_PARSE_COUNT_SNIPPET = """
import json
import sys

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

calls = []
_real_loads = tomllib.loads

def _counting_loads(s, *a, **kw):
    calls.append(None)
    return _real_loads(s, *a, **kw)

tomllib.loads = _counting_loads

import tldw_chatbook.config

print(json.dumps({"loads_calls": len(calls)}))
"""


class TestEmbeddedDefaultParsedOnce:
    """AC #1: the 1,285-line embedded default TOML is parsed exactly once
    per process; DEFAULT_CONFIG_FROM_TOML is a module-level constant and
    must never be re-parsed -- not at import, and not by a forced reload of
    either loader afterwards."""

    def test_import_parses_embedded_default_exactly_once(self, tmp_path):
        """Counts `tomllib.loads()` calls during a fresh
        `import tldw_chatbook.config`, in a subprocess (a monkeypatch
        installed *after* config.py is already imported in-process cannot
        observe the import-time parse -- it has already happened by the
        time any test function runs)."""
        result = _run_isolated_python(tmp_path, _EMBEDDED_TOML_PARSE_COUNT_SNIPPET)
        assert result.returncode == 0, (
            f"import tldw_chatbook.config failed in isolated subprocess:\n"
            f"stdout={result.stdout}\nstderr={result.stderr}"
        )
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["loads_calls"] == 1, (
            f"tomllib.loads() (embedded-TOML string parser) called "
            f"{payload['loads_calls']} times during `import tldw_chatbook.config`; "
            f"expected exactly 1 (DEFAULT_CONFIG_FROM_TOML)."
        )

    def test_forced_full_reload_does_not_reparse_embedded_default(
        self, tmp_path, monkeypatch
    ):
        calls: list[str] = []
        real_loads = config_mod.tomllib.loads

        def _counting_loads(s, *a, **kw):
            calls.append(s)
            return real_loads(s, *a, **kw)

        monkeypatch.setattr(config_mod.tomllib, "loads", _counting_loads)

        config_path = tmp_path / "parse-count-config.toml"
        config_path.write_text("[general]\ndefault_tab = \"chat\"\n", encoding="utf-8")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

        config_mod.load_cli_config_and_ensure_existence(force_reload=True)
        config_mod.load_settings(force_reload=True)

        # DEFAULT_CONFIG_FROM_TOML was already computed once at import time,
        # long before this test's monkeypatch was installed -- a forced
        # reload of both loaders must not call tomllib.loads() again at all.
        assert len(calls) <= 1, (
            f"tomllib.loads() (embedded-TOML parser) called {len(calls)} times "
            f"during a forced full reload; expected the module-level "
            f"DEFAULT_CONFIG_FROM_TOML constant to be reused, not re-parsed."
        )

    def test_user_config_file_parsed_once_across_startup_sequence(
        self, tmp_path, monkeypatch
    ):
        """The *user* config file itself must be opened+parsed once across
        config.py's own module-scope startup sequence: `load_cli_config_
        and_ensure_existence()` followed by `load_settings()`, both with
        their default `force_reload=False` (exactly how the bottom of
        config.py calls them) -- not twice, once independently by each
        loader.

        Note: this specifically exercises the *default* (force_reload=False)
        pairing. When a caller explicitly forces both loaders to bypass
        their cache (e.g. save_settings_to_cli_config's
        `load_cli_config_and_ensure_existence(force_reload=True)` followed
        by `load_settings(force_reload=True)`), each forced call
        legitimately re-hits disk by design -- that pattern is unchanged by
        this consolidation and intentionally not asserted here.
        """
        config_path = tmp_path / "parse-count-user-config.toml"
        config_path.write_text("[general]\ndefault_tab = \"chat\"\n", encoding="utf-8")
        monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
        # Start both caches cold for this path, matching a fresh process.
        monkeypatch.setattr(config_mod, "_CONFIG_CACHE", None)
        monkeypatch.setattr(config_mod, "_CONFIG_CACHE_SOURCE", None)
        monkeypatch.setattr(config_mod, "_SETTINGS_CACHE", None)
        monkeypatch.setattr(config_mod, "_SETTINGS_CACHE_SOURCE", None)

        calls: list[str] = []
        real_load = config_mod.tomllib.load

        def _counting_load(fp, *a, **kw):
            calls.append(getattr(fp, "name", None))
            return real_load(fp, *a, **kw)

        monkeypatch.setattr(config_mod.tomllib, "load", _counting_load)

        # Mirrors config.py's own startup sequence (module-scope lines near
        # the bottom of the file) exactly: no force_reload argument on
        # either call.
        config_mod.load_cli_config_and_ensure_existence()
        config_mod.load_settings()

        assert len(calls) == 1, (
            f"tomllib.load() (user config file parser) called {len(calls)} "
            f"times across the startup-style load_cli_config_and_ensure_existence() "
            f"+ load_settings() pairing; expected exactly 1 -- load_settings() "
            f"must reuse load_cli_config_and_ensure_existence()'s now-warm cache "
            f"instead of re-opening the same file."
        )


# --- (c) chardet stays out of sys.modules for config.py's own import chain -


def _run_isolated_python(tmp_path: Path, code: str) -> "subprocess.CompletedProcess[str]":
    """Run a Python snippet in a fresh interpreter with isolated config/data
    dirs -- NEVER the real `~/.config/tldw_cli`.

    A fresh interpreter is required because `sys.modules` is process-global:
    once chardet is imported by anything else in this pytest session, it
    stays cached in-process and an in-process check would false-pass.
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )


_CONFIG_IMPORT_SNIPPET = """
import json
import sys

import tldw_chatbook.config

print(json.dumps({"chardet_loaded": "chardet" in sys.modules}))
"""

_UTILS_IMPORT_SNIPPET = """
import json
import sys

import tldw_chatbook.Utils.Utils

print(json.dumps({"chardet_loaded": "chardet" in sys.modules}))
"""


def test_config_import_does_not_load_chardet(tmp_path: Path) -> None:
    """Plain `import tldw_chatbook.config` must not load chardet -- neither
    directly (config.py never imported it) nor transitively."""
    result = _run_isolated_python(tmp_path, _CONFIG_IMPORT_SNIPPET)
    assert result.returncode == 0, (
        f"import tldw_chatbook.config failed in isolated subprocess:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["chardet_loaded"] is False


def test_utils_module_import_does_not_load_chardet(tmp_path: Path) -> None:
    """Plain `import tldw_chatbook.Utils.Utils` must not load chardet --
    it is now deferred into its two callers (safe_read_file,
    FileProcessor.detect_encoding)."""
    result = _run_isolated_python(tmp_path, _UTILS_IMPORT_SNIPPET)
    assert result.returncode == 0, (
        f"import tldw_chatbook.Utils.Utils failed in isolated subprocess:\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["chardet_loaded"] is False


class TestChardetConsumersStillWork:
    """The two functions that actually use chardet must still work
    correctly now that the import moved inside them."""

    def test_safe_read_file_detects_and_decodes(self, tmp_path):
        from tldw_chatbook.Utils.Utils import safe_read_file

        sample = tmp_path / "sample.txt"
        sample.write_bytes("hello from tldw_chatbook – café".encode("utf-8"))

        content = safe_read_file(str(sample))

        assert "hello from tldw_chatbook" in content
        assert "caf" in content

    def test_file_processor_detect_encoding_returns_a_usable_codec(self, tmp_path):
        from tldw_chatbook.Utils.Utils import FileProcessor

        sample = tmp_path / "sample.txt"
        sample.write_bytes(b"plain ascii content")

        encoding = FileProcessor.detect_encoding(str(sample))

        assert isinstance(encoding, str)
        assert encoding  # non-empty
        # Must actually be usable to decode the file it was detected from.
        sample.read_bytes().decode(encoding)

    def test_file_processor_read_file_content_round_trips(self, tmp_path):
        from tldw_chatbook.Utils.Utils import FileProcessor

        sample = tmp_path / "sample.md"
        sample.write_text("# Title\n\nSome body text.\n", encoding="utf-8")

        content = FileProcessor.read_file_content(str(sample))

        assert "# Title" in content
        assert "Some body text." in content

    def test_safe_read_file_propagates_missing_chardet_unchanged(self, monkeypatch):
        """chardet is a hard (non-optional) dependency (pyproject.toml core
        deps), so no try/except was added around the deferred import --
        preserve the exact prior behavior: if chardet were unimportable, the
        ImportError propagates out of the caller uncaught."""
        from tldw_chatbook.Utils.Utils import safe_read_file

        real_import = builtins.__import__

        def _fail_chardet(name, *args, **kwargs):
            if name == "chardet":
                raise ImportError("simulated missing chardet")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail_chardet)

        with pytest.raises(ImportError):
            safe_read_file(__file__)
