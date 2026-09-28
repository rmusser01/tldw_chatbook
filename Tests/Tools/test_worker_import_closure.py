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
