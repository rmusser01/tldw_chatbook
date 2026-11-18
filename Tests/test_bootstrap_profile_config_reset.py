"""A bootstrap-profile test never sees another test's config (TASK-36478).

Every ``bootstrap_profile`` test in one pytest process shares the same
collection-time profile, so they all read and write ONE ``config.toml``
(``Tests/conftest.py``'s ``isolate_test_environment``). Before TASK-36478
nothing reset that file between tests, so whatever one test wrote stayed for
every later test in the process:

* ``Tests/UI/test_console_native_chat_flow.py``'s
  ``_configure_native_ready_console`` persists a ready llama.cpp provider.
  ``test_console_character_avatar.py`` (module-marked ``bootstrap_profile``)
  calls it, and a later Console test that expects the shipped "not set up"
  config booted ready instead: its composer took focus and drew a ``thick``
  left edge (``test_console_empty_regions_do_not_stack_nested_terminal_frames``).
* The Console persists its rail layout (``console.rail_state``) on every
  toggle, so one test opening the Inspector left it open for the next.
* ``load_settings()`` hands out its cached dict by reference and test apps
  hold it as ``app.app_config``, so an in-place edit
  (``app.app_config["chat_defaults"] = {}``) outlived its test until the next
  config write happened to rebuild the cache.

The ``probe_*`` functions below reproduce those shapes in a child pytest
process, in file order, as two writer/reader pairs: ``a`` writes the config
file (and edits the cache in place) for ``b`` to miss, and ``c`` edits the
cache in place with no write at all for ``d`` to miss. They are collected
only by the child
(``-o python_functions=probe_*``); the regular run collects just the ``test_``
function that launches it.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_STASH_ENV = "TLDW_TEST_BOOTSTRAP_CONFIG_RESET_STASH"
_LEAKED_MODEL = "task-36478-leaked-model"
_IN_MEMORY_KEY = "task_36478_in_memory_only"
_IN_MEMORY_MODEL = "task-36478-in-memory-model"
_RAIL_KEY = "task-36478-probe"


def _bootstrap_config_path() -> Path:
    return Path(os.environ["TLDW_CONFIG_PATH"])


@pytest.mark.bootstrap_profile
def probe_a_writes_config_and_mutates_the_cached_settings() -> None:
    """Leave behind the three shapes of change the Console tests leaked."""
    from tldw_chatbook import config

    stash = Path(os.environ[_STASH_ENV])
    config_path = _bootstrap_config_path()
    stash.write_bytes(config_path.read_bytes())

    # What `_configure_native_ready_console` persists, and what the Console's
    # rail toggles persist.
    assert config.save_settings_to_cli_config(
        {"chat_defaults": {"provider": "llama_cpp", "model": _LEAKED_MODEL}}
    )
    assert config.save_setting_to_cli_config(
        "console.rail_state", _RAIL_KEY, {"right_open": True}
    )
    # An in-place edit of the shared cached dict, written nowhere.
    config.load_settings().setdefault("console", {})[_IN_MEMORY_KEY] = True

    # The changes really happened, so the reader's checks are not vacuous.
    assert config_path.read_bytes() != stash.read_bytes()
    settings = config.load_settings()
    assert settings["chat_defaults"]["model"] == _LEAKED_MODEL
    assert settings["console"]["rail_state"][_RAIL_KEY] == {"right_open": True}
    assert settings["console"][_IN_MEMORY_KEY] is True


@pytest.mark.bootstrap_profile
def probe_b_sees_the_profile_config_as_it_was() -> None:
    """The next bootstrap-profile test starts from the untouched profile."""
    from tldw_chatbook import config

    stash = Path(os.environ[_STASH_ENV])
    assert stash.is_file(), "the writer probe must run first in this process"

    assert _bootstrap_config_path().read_bytes() == stash.read_bytes()
    settings = config.load_settings()
    assert settings["chat_defaults"].get("model") != _LEAKED_MODEL
    console = settings.get("console", {})
    assert _RAIL_KEY not in console.get("rail_state", {})
    assert _IN_MEMORY_KEY not in console


@pytest.mark.bootstrap_profile
def probe_c_mutates_the_cached_settings_without_writing() -> None:
    """An in-place edit with no config write after it, as
    ``app.app_config["chat_defaults"] = {}`` does in
    test_console_run_inspector_shows_blocked_provider_and_missing_rag_source.
    Nothing changes on disk, so only dropping the cache undoes it."""
    from tldw_chatbook import config

    marker = Path(os.environ[_STASH_ENV]).with_suffix(".in-memory-ran")
    marker.write_text("ran")
    before = _bootstrap_config_path().read_bytes()

    config.load_settings()["chat_defaults"]["model"] = _IN_MEMORY_MODEL

    assert config.load_settings()["chat_defaults"]["model"] == _IN_MEMORY_MODEL
    assert _bootstrap_config_path().read_bytes() == before


@pytest.mark.bootstrap_profile
def probe_d_does_not_see_the_in_place_edit() -> None:
    """The next bootstrap-profile test reads settings rebuilt from the file."""
    from tldw_chatbook import config

    marker = Path(os.environ[_STASH_ENV]).with_suffix(".in-memory-ran")
    assert marker.is_file(), "the in-place probe must run first in this process"

    assert config.load_settings()["chat_defaults"].get("model") != _IN_MEMORY_MODEL


@pytest.mark.bootstrap_profile
def probe_e_writes_config_then_its_teardown_raises(request) -> None:
    """A writer whose fixture teardown raises before the restore would run.

    ``isolate_test_environment`` raises at teardown when a test's native app
    workers are uncertain (``retire_test_app_owners`` sets the flag when the
    default executor does not stop within its grace period, which happens
    under load). The restore must still run (PR #3254's review).
    """
    from tldw_chatbook import config

    stash = Path(os.environ[_STASH_ENV]).with_suffix(".teardown-raise")
    stash.write_bytes(_bootstrap_config_path().read_bytes())
    assert config.save_setting_to_cli_config(
        "console.rail_state", _RAIL_KEY, {"right_open": True}
    )
    assert _bootstrap_config_path().read_bytes() != stash.read_bytes()
    request.node._test_app_native_workers_uncertain = True


@pytest.mark.bootstrap_profile
def probe_f_sees_the_config_after_a_teardown_that_raised() -> None:
    """The next bootstrap-profile test still starts from the untouched profile."""
    from tldw_chatbook import config

    stash = Path(os.environ[_STASH_ENV]).with_suffix(".teardown-raise")
    assert stash.is_file(), "the raising writer probe must run first in this process"

    assert _bootstrap_config_path().read_bytes() == stash.read_bytes()
    rail_state = config.load_settings().get("console", {}).get("rail_state", {})
    assert _RAIL_KEY not in rail_state


def _run_probes(tmp_path: Path, selection: str) -> subprocess.CompletedProcess:
    """Run the ``probe_*`` functions matching ``selection`` in one fresh pytest process."""
    profile = tmp_path / "profile"
    profile.mkdir(mode=0o700)
    environment = dict(os.environ)
    environment.update(
        # A bootstrap root of the child's own, so it never shares this
        # process's profile; Tests/conftest.py points HOME/XDG/
        # TLDW_CONFIG_PATH into it at import.
        TLDW_TEST_CONFIG_ROOT=str(profile),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    environment[_STASH_ENV] = str(tmp_path / "config-before-writer.toml")
    for name in (
        "TLDW_TEST_CONFIG_ROOT_OWNER",
        "PYTEST_XDIST_WORKER",
        "PYTEST_XDIST_WORKER_COUNT",
        "PYTEST_ADDOPTS",
        "PYTEST_CURRENT_TEST",
    ):
        environment.pop(name, None)

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(Path(__file__).relative_to(REPO_ROOT)),
            "-o",
            "python_functions=probe_*",
            "-k",
            selection,
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "pytest_timeout",
            "-p",
            "no:cacheprovider",
            "-q",
            "-rA",
            f"--basetemp={tmp_path / 'child'}",
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )


def test_a_bootstrap_profile_tests_config_changes_never_reach_the_next_one(
    tmp_path: Path,
) -> None:
    """Run the four writer/reader probes in one fresh pytest process; all pass."""
    result = _run_probes(tmp_path, "probe_a or probe_b or probe_c or probe_d")

    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-6000:]
    assert "4 passed" in output, output[-6000:]


def test_the_config_is_restored_even_when_an_earlier_teardown_step_raises(
    tmp_path: Path,
) -> None:
    """The writer's teardown errors on purpose; the reader after it still passes."""
    result = _run_probes(tmp_path, "probe_e or probe_f")

    output = result.stdout + result.stderr
    # The writer itself passes and then errors at teardown, as designed.
    assert "Native app workers remain uncertain" in output, output[-6000:]
    summary = output.strip().splitlines()[-1]
    assert re.search(r"\b2 passed\b", summary), output[-6000:]
    assert re.search(r"\b1 error\b", summary), output[-6000:]
    assert "failed" not in summary, output[-6000:]
    passed = [
        line
        for line in output.splitlines()
        if line.startswith("PASSED ") and "probe_f_" in line
    ]
    assert passed, output[-6000:]
