"""Import closure of the pinned workspace worker must stay stdlib-only.

Task 8 concatenates the worker's module closure into a remote bundle that
runs against a bare interpreter, so importing
``tldw_chatbook.Tools.workspace_tool_worker`` must never pull a third-party
package or the app's config bootstrap into ``sys.modules``. This test is
that gate: it imports the worker in a fresh isolated interpreter and
refuses an allowlist of known-heavy roots.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

# Frozen offender policy (Phase 0c). ``pydantic`` arrives via
# ``workspace_tool_protocol`` and ``loguru`` via ``Utils/path_validation``
# and ``Utils/sensitive_paths`` — the two offenders the RED run of this
# test surfaced. The remaining roots are the app's other heavy packages
# (sketched in the task brief) plus the config bootstrap module, none of
# which a pinned worker may import transitively. Grow this set only when a
# new offender is demonstrated and judged acceptable — never silently.
_BLOCKED_TOP_LEVEL = frozenset({"pydantic", "loguru", "httpx", "rich", "textual"})
_BLOCKED_MODULES = frozenset({"tldw_chatbook.config"})


def test_worker_import_closure_is_stdlib_only() -> None:
    probe = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(_REPOSITORY_ROOT)!r})
        import tldw_chatbook.Tools.workspace_tool_worker  # noqa: F401
        blocked_top_level = {set(_BLOCKED_TOP_LEVEL)!r}
        blocked_modules = {set(_BLOCKED_MODULES)!r}
        offenders = sorted(
            name
            for name in sys.modules
            if name.split(".")[0] in blocked_top_level or name in blocked_modules
        )
        closure = sorted(
            name for name in sys.modules if name.startswith("tldw_chatbook")
        )
        if offenders:
            print(
                "OFFENDERS: " + ", ".join(offenders),
                file=sys.stderr,
            )
            print("CLOSURE: " + ", ".join(closure), file=sys.stderr)
            sys.exit(1)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
