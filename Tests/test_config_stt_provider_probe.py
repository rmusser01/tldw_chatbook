from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (("parakeet_mlx", "lightning_whisper_mlx"), "parakeet-mlx"),
        (("lightning_whisper_mlx",), "lightning-whisper-mlx"),
        ((), "faster-whisper"),
    ],
)
def test_macos_stt_default_probes_packages_without_importing_them(
    tmp_path: Path,
    installed: tuple[str, ...],
    expected: str,
) -> None:
    """Choosing an STT default must not initialize optional native runtimes."""

    script = textwrap.dedent(
        f"""
        import builtins
        import importlib.machinery
        import importlib.util
        import json
        import sys

        installed = {installed!r}
        guarded = {{"parakeet_mlx", "lightning_whisper_mlx"}}
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec(name, package=None):
            if name in guarded:
                if name in installed:
                    return importlib.machinery.ModuleSpec(name, loader=None)
                return None
            return original_find_spec(name, package)

        def reject_optional_runtime_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in guarded:
                raise AssertionError(f"optional runtime imported: {{name}}")
            return original_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        builtins.__import__ = reject_optional_runtime_import
        sys.platform = "darwin"

        from tldw_chatbook.config import settings

        print(json.dumps(settings["STT_settings"]["default_stt_provider"]))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=_hermetic_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == f'"{expected}"'


def _hermetic_env(tmp_path: Path) -> dict:
    """Env vars that keep a `config.py` import from touching the real home dir."""
    env = os.environ.copy()
    private_home = tmp_path / "home"
    private_data = tmp_path / "data"
    private_config = tmp_path / "config"
    private_temp = tmp_path / "tmp"
    for directory in (private_home, private_data, private_config, private_temp):
        directory.mkdir(parents=True, mode=0o700)
    env.update(
        {
            "HOME": str(private_home),
            "USERPROFILE": str(private_home),
            "XDG_DATA_HOME": str(private_data),
            "XDG_CONFIG_HOME": str(private_config),
            "TLDW_CONFIG_PATH": str(private_config / "config.toml"),
            "TMPDIR": str(private_temp),
            "PYTHONPATH": str(PROJECT_ROOT),
        }
    )
    return env


def test_non_macos_stt_default_fallback_is_hyphenated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every provider id actually dispatched on is hyphenated --
    `console_voice_input.py`'s `LOCAL_PROVIDER_MODULES` (`"faster-whisper"`)
    and `transcription_service.py`'s provider-branch matching. On a non-macOS
    platform neither `if sys.platform == "darwin"` branch below the initial
    assignment in `config.py` ever runs, so that initial value is exactly
    what `STT_settings.default_stt_provider` resolves to.

    This machine is darwin, so `sys.platform` is patched to prove the
    fallback path (an unpatched test would prove nothing). It is patched
    *in-process*, around a `load_settings(force_reload=True)` call on the
    already-imported `config` module, not before a fresh interpreter imports
    it: spawning a subprocess with `sys.platform` forced to a fake value
    *before* `import config` (and everything it transitively imports --
    `loguru`, `psutil`, ...) reliably crashes the interpreter itself, because
    several of those packages pick a platform-specific C extension or probe
    `sysconfig` at import time using the real build's platform tag. Patching
    only around the call keeps every other import on the real platform and
    exercises just the one `sys.platform` read this fix touches.

    Args:
        tmp_path: pytest's per-test temporary directory; used as the isolated
            `TLDW_CONFIG_PATH` so this test never touches a real config file.
        monkeypatch: pytest's monkeypatch fixture; sets `TLDW_CONFIG_PATH`
            and forces `config_module.sys.platform` to `"linux"`.
    """
    from tldw_chatbook import config as config_module

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(tmp_path / "config.toml"))
    config_module._CONFIG_CACHE = None
    config_module._CONFIG_CACHE_SOURCE = None
    config_module._SETTINGS_CACHE = None
    config_module._SETTINGS_CACHE_SOURCE = None
    monkeypatch.setattr(config_module.sys, "platform", "linux")

    settings = config_module.load_settings(force_reload=True)

    assert settings["STT_settings"]["default_stt_provider"] == "faster-whisper"


@pytest.mark.parametrize(
    ("installed", "expected"),
    [
        (("parakeet_mlx", "lightning_whisper_mlx"), "parakeet-mlx"),
        (("lightning_whisper_mlx",), "lightning-whisper-mlx"),
        ((), "faster-whisper"),
    ],
)
def test_macos_transcription_template_default_matches_platform_preference(
    tmp_path: Path,
    installed: tuple[str, ...],
    expected: str,
) -> None:
    """The freshly generated `[transcription] default_provider` must agree
    with the platform preference (task-867).

    Before this fix `CONFIG_TOML_CONTENT` hardcoded `default_provider =
    "faster-whisper"` unconditionally, so a macOS install with parakeet-mlx
    (or lightning-whisper-mlx) available still generated a config that
    permanently pinned faster-whisper for `[transcription]` -- the darwin
    preference computed for `STT_settings.default_stt_provider` could never
    engage for the setting transcription/ingest actually reads. Reuses the
    subprocess harness above: `CONFIG_TOML_CONTENT` is resolved once at
    `config.py` import time, so `sys.platform` and the availability probe
    must be patched *before* import, not around an in-process reload.
    """

    script = textwrap.dedent(
        f"""
        import builtins
        import importlib.machinery
        import importlib.util
        import json
        import sys

        installed = {installed!r}
        guarded = {{"parakeet_mlx", "lightning_whisper_mlx"}}
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec(name, package=None):
            if name in guarded:
                if name in installed:
                    return importlib.machinery.ModuleSpec(name, loader=None)
                return None
            return original_find_spec(name, package)

        def reject_optional_runtime_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in guarded:
                raise AssertionError(f"optional runtime imported: {{name}}")
            return original_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        builtins.__import__ = reject_optional_runtime_import
        sys.platform = "darwin"

        from tldw_chatbook.config import get_cli_setting

        print(json.dumps(get_cli_setting("transcription", "default_provider", None)))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=_hermetic_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip().splitlines()[-1] == f'"{expected}"'


def test_non_macos_transcription_template_default_is_faster_whisper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-darwin platforms must keep resolving `default_provider` to
    "faster-whisper" (task-867 AC#3): the template interpolation must not
    change behaviour off macOS.

    Unlike the darwin cases above, this cannot use the before-import
    subprocess harness: spoofing `sys.platform` to something that does not
    match the interpreter's real build reliably crashes `import loguru`
    itself (`sysconfig.get_config_vars()` looks for a
    `_sysconfigdata__<fake platform>` module that does not exist -- this
    was reproduced while writing this test). `_default_stt_provider_for_platform()`
    is the one call `CONFIG_TOML_CONTENT`'s interpolation makes (task-867),
    so patching `sys.platform` in-process, after every import has already
    happened on the real platform, and calling the helper directly proves
    the same thing `test_non_macos_stt_default_fallback_is_hyphenated`
    proves for `STT_settings.default_stt_provider` -- both read this one
    function.
    """
    from tldw_chatbook import config as config_module

    monkeypatch.setattr(config_module.sys, "platform", "linux")

    assert config_module._default_stt_provider_for_platform() == "faster-whisper"


def test_transcription_template_default_matches_platform_helper_at_import() -> None:
    """`CONFIG_TOML_CONTENT`'s baked-in `[transcription] default_provider`
    must equal what `_default_stt_provider_for_platform()` computes now
    (task-867). Nothing here spoofs the platform -- neither `sys.platform`
    nor the installed-package set changes between process start and this
    assertion, so a fresh call must reproduce the exact value the module
    interpolated into the template at import time. This is what pins the
    darwin-vs-non-darwin split for the *template* specifically, complementing
    the subprocess-based darwin cases (which prove the darwin branch) and
    `test_non_macos_transcription_template_default_is_faster_whisper`
    (which proves the non-darwin branch of the shared helper) without
    re-crossing the platform-spoofing crash either of those work around.
    """
    import tomllib

    from tldw_chatbook import config as config_module

    template = tomllib.loads(config_module.CONFIG_TOML_CONTENT)

    assert (
        template["transcription"]["default_provider"]
        == config_module._default_stt_provider_for_platform()
    )


def test_existing_transcription_default_provider_is_respected_and_file_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An explicit `[transcription] default_provider` in an existing config
    must win over the platform preference, and loading it must not rewrite
    the file (task-867 requirement #2).

    "parakeet-onnx" is deliberately a value `_default_stt_provider_for_platform()`
    never returns, so this cannot pass by coincidentally matching whatever
    this host's own platform default happens to be.
    """
    from tldw_chatbook import config as config_module

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[transcription]\ndefault_provider = "parakeet-onnx"\n',
        encoding="utf-8",
    )
    before_bytes = config_path.read_bytes()
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))

    resolved = config_module.get_cli_setting("transcription", "default_provider", None)

    assert resolved == "parakeet-onnx"
    assert config_path.read_bytes() == before_bytes


def test_console_dictation_resolver_prefers_parakeet_mlx_on_fresh_darwin_install(
    tmp_path: Path,
) -> None:
    """End-to-end: `console_voice_input.resolve()` must land on parakeet-mlx
    for a fresh config on a darwin install with parakeet-mlx installed
    (task-867). Guards every provider-detection module explicitly so the
    result cannot depend on what happens to be importable on the machine
    actually running this test suite; faster-whisper is deliberately also
    reported installed so the test can tell "picked the configured
    parakeet-mlx default" apart from "picked the only installed provider
    regardless of what the template said".
    """

    script = textwrap.dedent(
        """
        import builtins
        import importlib.machinery
        import importlib.util
        import json
        import sys

        installed = {"parakeet_mlx", "faster_whisper"}
        guarded = {
            "onnx_asr",
            "parakeet_mlx",
            "lightning_whisper_mlx",
            "faster_whisper",
            "torch",
            "transformers",
            "nemo",
        }
        original_find_spec = importlib.util.find_spec
        original_import = builtins.__import__

        def find_spec(name, package=None):
            if name in guarded:
                if name in installed:
                    return importlib.machinery.ModuleSpec(name, loader=None)
                return None
            return original_find_spec(name, package)

        def reject_optional_runtime_import(name, *args, **kwargs):
            if name.split(".", 1)[0] in guarded:
                raise AssertionError(f"optional runtime imported: {name}")
            return original_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec
        builtins.__import__ = reject_optional_runtime_import
        sys.platform = "darwin"

        from tldw_chatbook.Chat import console_voice_input as cvi

        effective = cvi.resolve()
        print(
            json.dumps(
                {
                    "provider": effective.provider if effective else None,
                    "configured_provider": (
                        effective.configured_provider if effective else None
                    ),
                    "was_overridden": (
                        effective.was_overridden if effective else None
                    ),
                }
            )
        )
        """
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=_hermetic_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip().splitlines()[-1])
    assert result == {
        "provider": "parakeet-mlx",
        "configured_provider": "parakeet-mlx",
        "was_overridden": False,
    }


def test_undotted_two_arg_call_returns_the_caller_default_without_crashing() -> None:
    """A 2-arg call on an undotted section returns the default, never TypeError.

    `get_cli_setting("database", {})` is a long-lived misuse shape: keys are
    always strings, so the second positional can only be a default. It never
    resolved config, but it also never crashed. The TASK-1754 sentinel change
    briefly let it reach `dict.get()` with an unhashable key, which raised
    `TypeError: unhashable type: 'dict'` and broke
    `Helper_Scripts/Mass-Ingestion/mass_ingest.py` (found in PR review).

    Returns:
        None.
    """
    from tldw_chatbook.config import get_cli_setting

    sentinel: dict = {}
    assert get_cli_setting("database", sentinel) is sentinel
    assert get_cli_setting("no_such_section_at_all", sentinel) is sentinel
    # A non-dict default of non-string type takes the same path.
    assert get_cli_setting("database", 17) == 17


def test_supported_call_shapes_still_resolve_configured_values() -> None:
    """The shapes that must keep working after the misuse guard was added.

    Guards against "fix the crash, break resolution": the dotted 2-arg form
    and the canonical 3-arg form must still read real config values, and both
    must still fall back to the caller's default for an absent key.

    Returns:
        None.
    """
    from tldw_chatbook.config import get_cli_setting

    assert get_cli_setting("transcription", "default_provider", "FB") != "FB"
    assert get_cli_setting("transcription.default_provider", "FB") != "FB"
    assert get_cli_setting("transcription", "definitely_absent_key", "FB") == "FB"
    assert get_cli_setting("transcription.definitely_absent_key", "FB") == "FB"
