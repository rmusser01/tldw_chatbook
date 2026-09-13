"""Import-closure guard: `console_raw_cli` reaches its executor lazily.

TASK-23112 / ADR-097. ``app.py`` imports ``Chat/console_raw_cli.py`` at module
scope (for ``RawCliRuntime``, constructed in ``TldwCli.__init__``), and that
module used to import ``Tools.raw_cli_executor`` at module scope. Measured with
an import-parent tracer, the edge put ``Tools.raw_cli_executor`` and
``Agents.run_log`` on the boot import path (648 -> 646 own modules when
deferred; the sibling ``Tools``/``Tools.tool_executor``/``Agents``/
``Agents.run_log_format`` stay resident because ``UI.Tools_Settings_Window ->
Agents.local_tool_provider -> Agents.tool_catalog`` imports them anyway --
a pre-existing boot edge this task did not touch).

Two things had to move together, because ``RawCliRuntime`` is constructed
during ``TldwCli.__init__``: the module-scope import (now the lazy
``_raw_cli_executor()`` accessor) AND the default ``RawShellExecutor()``
construction (now ``RawCliRuntime._executor_or_default()``, resolved on the
first ``execute()``). Deferring only the import would have satisfied the
import-weight ratchet while a real boot still paid the modules at construction
-- the exact half-fix recorded in TASK-21200's lesson.

Subprocess-isolated for the same reason as ``test_app_import_diet_closure.py``
(TASK-21108), whose pattern this file follows.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
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
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


_APP_CLOSURE_SNIPPET = """
import sys

import tldw_chatbook.app  # noqa: F401

forbidden = (
    "tldw_chatbook.Tools.raw_cli_executor",
    "tldw_chatbook.Agents.run_log",
)
resident = sorted(m for m in forbidden if sys.modules.get(m) is not None)
assert not resident, (
    "raw CLI executor resident after `import tldw_chatbook.app`: "
    + repr(resident)
    + " -- defer it (Chat/console_raw_cli.py:_raw_cli_executor) rather than "
    "raising MAX_TLDW_MODULE_COUNT (ADR-097)."
)

# Anti-vacuity: console_raw_cli itself must still be in the closure, or the
# assertion above would pass without exercising the deferral.
assert "tldw_chatbook.Chat.console_raw_cli" in sys.modules, (
    "console_raw_cli left the app import closure; this guard no longer tests "
    "the deferral"
)

print("RAW_CLI_CLOSURE_OK")
"""


_CONSTRUCTION_SNIPPET = """
import sys

from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime

assert "tldw_chatbook.Tools.raw_cli_executor" not in sys.modules, (
    "importing console_raw_cli still executes Tools.raw_cli_executor"
)

# TldwCli.__init__ builds exactly this object. Constructing it must not pull
# the executor either -- that is the half of the fix an import-only deferral
# would have missed (TASK-21200 lesson).
runtime = RawCliRuntime(lambda: False)
assert "tldw_chatbook.Tools.raw_cli_executor" not in sys.modules, (
    "constructing RawCliRuntime executed Tools.raw_cli_executor"
)
assert runtime._executor is None

# First real use resolves the default executor, once, and caches it.
executor = runtime._executor_or_default()
assert executor is not None
from tldw_chatbook.Tools.raw_cli_executor import RawShellExecutor

assert isinstance(executor, RawShellExecutor)
assert runtime._executor_or_default() is executor, "default executor rebuilt per call"

# An injected executor still wins and is never replaced.
sentinel = object()
injected = RawCliRuntime(lambda: False, executor=sentinel)
assert injected._executor is sentinel
assert injected._executor_or_default() is sentinel

# Concurrent first use builds exactly ONE executor, not one per racing thread
# (Qodo review, PR #2180): the post-import check and the construction must
# happen together under the lock. A build-then-discard shape returns the same
# cached object to every caller and so passes the identity checks above while
# still constructing N times -- only counting constructions catches it.
import threading
import time

import tldw_chatbook.Tools.raw_cli_executor as _executor_module

builds = []
_real_init = _executor_module.RawShellExecutor.__init__


def _counting_init(self):
    builds.append(1)
    # Hold the constructor open long enough that every racing thread is
    # certainly inside the window. Without this the real __init__ (which only
    # grabs a spawn context) finishes inside one GIL switch interval, so the
    # racy shape serialises by luck and the assertion below cannot fail --
    # a test that passes on the bug it exists to catch.
    time.sleep(0.05)
    _real_init(self)


_executor_module.RawShellExecutor.__init__ = _counting_init
try:
    racing = RawCliRuntime(lambda: False)
    start = threading.Barrier(8)
    seen = []

    def _race():
        start.wait()
        seen.append(racing._executor_or_default())

    threads = [threading.Thread(target=_race) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
finally:
    _executor_module.RawShellExecutor.__init__ = _real_init

assert len(builds) == 1, f"default executor built {len(builds)} times under concurrency"
assert len({id(obj) for obj in seen}) == 1, "racing callers got different executors"

print("RAW_CLI_CONSTRUCTION_OK")
"""


_REAL_USE_PATH_SNIPPET = """
import sys
from pathlib import Path

import pydantic

from tldw_chatbook.Chat.console_raw_cli import (
    LocalCommandCallArgs,
    LocalCommandResultArgs,
    RawCliRuntime,
)

# 1. The deferred byte limits still gate the pydantic validators. Both
# constants live in Tools.raw_cli_executor; if the deferred lookup broke, the
# over-limit payloads below would be accepted.
from tldw_chatbook.Tools.raw_cli_executor import (
    MAX_RAW_COMMAND_BYTES,
    MAX_RAW_PREVIEW_BYTES,
    RawCliRequest,
    RawCliResult,
)

ok = LocalCommandCallArgs(
    invocation_id="inv-1",
    command="echo hello",
    shell="bash",
    cwd="/tmp",
)
assert ok.command == "echo hello"

try:
    LocalCommandCallArgs(
        invocation_id="inv-2",
        command="x" * (MAX_RAW_COMMAND_BYTES + 1),
        shell="bash",
        cwd="/tmp",
    )
except pydantic.ValidationError:
    pass
else:  # pragma: no cover - only on a broken deferred lookup
    raise AssertionError("MAX_RAW_COMMAND_BYTES no longer gates the command field")


def _result_args(**overrides):
    fields = dict(
        invocation_id="inv-r",
        shell="bash",
        cwd="/tmp",
        stdout_preview="out",
        stderr_preview="err",
        elapsed_seconds=1.0,
        exit_code=0,
        terminal_state="exited",
        truncated=False,
        cleanup_proven=True,
    )
    fields.update(overrides)
    return LocalCommandResultArgs(**fields)


assert _result_args().stdout_preview == "out"

# Per-field limit (the `_validate_preview` field validator).
try:
    _result_args(stdout_preview="x" * (MAX_RAW_PREVIEW_BYTES + 1))
except pydantic.ValidationError:
    pass
else:  # pragma: no cover - only on a broken deferred lookup
    raise AssertionError("MAX_RAW_PREVIEW_BYTES no longer gates a preview field")

# Combined limit (the `_validate_combined_preview` model validator).
half = "x" * ((MAX_RAW_PREVIEW_BYTES // 2) + 1)
try:
    _result_args(stdout_preview=half, stderr_preview=half)
except pydantic.ValidationError:
    pass
else:  # pragma: no cover - only on a broken deferred lookup
    raise AssertionError("MAX_RAW_PREVIEW_BYTES no longer gates the combined preview")

# 2. The refusal path: execute() on a runtime that was never armed runs
# `validate_raw_cli_request` and `_refused_result` -- both deferred lookups --
# without spawning a process.
runtime = RawCliRuntime(lambda: False)
request = RawCliRequest(
    invocation_id="inv-3",
    caller="user",
    console_session_id="session-1",
    command="echo hello",
    initial_directory=Path("/tmp"),
    shell="bash",
    timeout_seconds=1.0,
)
events = []
result = runtime.execute(request, events.append)
assert isinstance(result, RawCliResult), type(result)
assert result.invocation_id == "inv-3"
assert result.exit_code is None
assert events == []

print("RAW_CLI_USE_PATH_OK")
"""


def test_app_import_does_not_execute_the_raw_cli_executor(tmp_path: Path) -> None:
    """`import tldw_chatbook.app` must not pull `Tools.raw_cli_executor`.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _APP_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        "Tools.raw_cli_executor must stay off the app import closure:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "RAW_CLI_CLOSURE_OK" in result.stdout


def test_raw_cli_runtime_construction_stays_executor_free(tmp_path: Path) -> None:
    """Constructing `RawCliRuntime` must not import or build the executor.

    ``TldwCli.__init__`` constructs this object, so an import-only deferral
    would leave a real boot paying the same modules a moment later.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _CONSTRUCTION_SNIPPET)
    assert result.returncode == 0, (
        "RawCliRuntime construction pulled the deferred executor:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "RAW_CLI_CONSTRUCTION_OK" in result.stdout


def test_deferred_raw_cli_executor_resolves_on_its_real_use_paths(
    tmp_path: Path,
) -> None:
    """Every deferred name still resolves where the module actually uses it.

    Covers the two live call shapes that no longer have a module-scope binding:
    the pydantic field validators reading ``MAX_RAW_COMMAND_BYTES``, and
    ``RawCliRuntime.execute``'s ``validate_raw_cli_request`` +
    ``_refused_result`` pair (exercised through the unarmed-refusal path, which
    spawns no process).

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _REAL_USE_PATH_SNIPPET)
    assert result.returncode == 0, (
        "a deferred raw CLI name failed on its real use path:\n"
        f"stdout={result.stdout}\nstderr={result.stderr[-4000:]}"
    )
    assert "RAW_CLI_USE_PATH_OK" in result.stdout
