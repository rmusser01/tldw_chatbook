"""Import closure of the pinned workspace worker must stay stdlib-only.

Task 8 concatenates the worker's module closure into a remote bundle that
runs against a bare interpreter, so importing
``tldw_chatbook.Tools.workspace_tool_worker`` must never pull ANY module
outside the standard library and the ``tldw_chatbook`` package itself into
``sys.modules``. This test is that gate: it imports the worker in a fresh
isolated interpreter and enforces that allowlist invariant
(``sys.stdlib_module_names`` + ``tldw_chatbook.*``) over the delta the
import produced.

The frozen denylist below is only a fast diagnostic that names the
historically offending roots in failure output — the allowlist, not the
denylist, is the enforced rule, so a future transitive third-party import
(``tiktoken`` was the first one caught) fails the gate even when it is not
on the denylist.

The probe environment deliberately drops ``TIKTOKEN_CACHE_DIR`` /
``DATA_GYM_CACHE_DIR``: the outer pytest process imports
``tldw_chatbook`` itself, whose tiktoken arming exports
``TIKTOKEN_CACHE_DIR`` into this process's environment, and an inherited
override silently disarms the very import this gate must catch. The real
executor launches the worker with its own small env allowlist that does
not carry these variables either
(``workspace_tool_executor.workspace_worker_environment``), so stripping
them matches production launch semantics.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Diagnostic-only fast path (Phase 0c history): ``pydantic`` arrived via
# ``workspace_tool_protocol`` and the root package init's tiktoken arming;
# ``loguru`` via ``Utils/path_validation``, ``Utils/sensitive_paths``,
# ``Tools/__init__`` and the root package init. Listed here so a failure
# message names the usual suspects; NOT the enforced invariant.
_BLOCKED_TOP_LEVEL = frozenset(
    {"pydantic", "loguru", "httpx", "rich", "textual", "tiktoken"}
)
_BLOCKED_MODULES = frozenset({"tldw_chatbook.config"})


def _probe_environment() -> dict[str, str]:
    """One launch-faithful environment for the import probe."""
    environment = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith(("PYTHON", "TIKTOKEN_CACHE_DIR", "DATA_GYM_CACHE_DIR"))
    }
    environment.setdefault("PATH", os.defpath)
    return environment


def test_worker_import_closure_is_stdlib_only() -> None:
    probe = textwrap.dedent(
        f"""
        import sys
        baseline = frozenset(sys.modules)
        sys.path.insert(0, {str(_REPOSITORY_ROOT)!r})
        import tldw_chatbook.Tools.workspace_tool_worker  # noqa: F401
        stdlib = sys.stdlib_module_names
        pulled = sorted(set(sys.modules) - baseline)
        offenders = sorted(
            name
            for name in pulled
            if name.split(".")[0] != "tldw_chatbook"
            and name.split(".")[0] not in stdlib
        )
        closure = sorted(name for name in pulled if name.startswith("tldw_chatbook"))
        blocked_top_level = {set(_BLOCKED_TOP_LEVEL)!r}
        blocked_modules = {set(_BLOCKED_MODULES)!r}
        named = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in blocked_top_level or name in blocked_modules
        )
        if offenders:
            print("OFFENDERS: " + ", ".join(offenders), file=sys.stderr)
            print("CLOSURE: " + ", ".join(closure), file=sys.stderr)
            sys.exit(1)
        if named:
            print(
                "DENYLIST-HIT (non-fatal above): " + ", ".join(named),
                file=sys.stderr,
            )
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        timeout=60,
        env=_probe_environment(),
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")


def test_a_validated_path_in_the_running_worker_keeps_the_closure_stdlib_only(tmp_path) -> None:
    """TASK-40608: serving a request must not load what importing the worker avoids.

    The gate above covers importing the worker. Each worker then serves one
    request, and its first path validation used to fetch loguru's logger and
    record metrics: 106 modules (loguru, psutil, asyncio, multiprocessing, ssl)
    loaded on every local tool call, most of the call's time. The probe runs
    the real choke point, ``resolve_workspace_path``, after importing the
    worker as a fresh worker process has, and applies the same allowlist to
    what the validation loaded, with the metrics module added to the refused
    names.
    """
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_REPOSITORY_ROOT)!r})
        import tldw_chatbook.Tools.workspace_tool_worker  # noqa: F401
        from tldw_chatbook.Tools.local_tool_impls import resolve_workspace_path
        from tldw_chatbook.Utils.sensitive_paths import SensitivePathContext
        baseline = frozenset(sys.modules)
        from pathlib import Path
        root = Path(sys.argv[1])
        # The parent sends the worker its deny set; an empty one keeps the probe
        # off this machine's real configuration.
        context = SensitivePathContext((), (), (), None, ())
        resolved = resolve_workspace_path("notes.txt", root, intent="read", context=context)
        assert resolved.name == "notes.txt", resolved
        stdlib = sys.stdlib_module_names
        pulled = sorted(set(sys.modules) - baseline)
        offenders = sorted(
            name
            for name in pulled
            if name.split(".")[0] not in stdlib
            and not (name.split(".")[0] == "tldw_chatbook" and ".Metrics" not in name)
        )
        print("PULLED: " + ", ".join(offenders))
        """
    )

    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(tmp_path)],
        capture_output=True,
        timeout=120,
        env=_probe_environment(),
    )
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    pulled = completed.stdout.decode(errors="replace").strip().splitlines()[-1]
    assert pulled == "PULLED:", "one validation in the worker loaded third-party or metrics modules"
