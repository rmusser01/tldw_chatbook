"""`collect_w002` must only treat a protected `Try.body` as guarded.

Tier-2 review S19 [D3]. The W002 collector's guard walk stopped at the first
ancestor `ast.Try` regardless of which section the lookup sat in, so a
`query_one` in an `except:`/`else:`/`finally:` body counted as guarded. None
of those is protected by that statement's own handlers -- the exception
propagates straight out of the function -- and the `finally` variant is the
worst of the three, because that is the section that runs during
CANCELLATION, when the tree is already being torn down. A bare `try/finally`
with no handlers protects nothing either.

That blind spot hid 34 functions repo-wide from both the census and the gate
that reads it, including `scheduling/schedules_workbench._run_sync`, which
the same review filed separately as a P1 app-exit. The guard exists because
this exact defect class took the app down three times (TASK-32800.4).

Gate-free: the collector is a pure AST function.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "check_textual_worker_contract.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("_w002_checker", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sites(source: str) -> list[str]:
    checker = _load_checker()
    # The collector only looks at files under a UI package.
    fake_path = _REPO_ROOT / "tldw_chatbook" / "UI" / "probe.py"
    return checker.collect_w002(ast.parse(source), fake_path)


_PROTECTED = """
class W:
    async def run(self):
        await self.work()
        try:
            self.query_one("#a")
        except Exception:
            pass
"""

_FINALLY = """
class W:
    async def run(self):
        try:
            await self.work()
        finally:
            self.query_one("#a").disabled = False
"""

_EXCEPT = """
class W:
    async def run(self):
        try:
            await self.work()
        except Exception:
            self.query_one("#a").disabled = False
"""

_ELSE = """
class W:
    async def run(self):
        try:
            await self.work()
        except Exception:
            pass
        else:
            self.query_one("#a")
"""

_BARE_TRY_FINALLY = """
class W:
    async def run(self):
        await self.work()
        try:
            self.query_one("#a")
        finally:
            self.release()
"""

_NESTED_IN_PROTECTED_BODY = """
class W:
    async def run(self):
        await self.work()
        try:
            if self.ready:
                for _ in range(2):
                    self.query_one("#a")
        except Exception:
            pass
"""


@pytest.mark.unit
def test_a_protected_try_body_still_counts_as_guarded() -> None:
    """The case the guard was always right about must not regress."""
    assert _sites(_PROTECTED) == []


@pytest.mark.unit
def test_a_lookup_nested_deep_inside_a_protected_body_is_still_guarded() -> None:
    """The section test must look at the SECTION, not the direct parent."""
    assert _sites(_NESTED_IN_PROTECTED_BODY) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("label", "source"),
    (
        ("finally", _FINALLY),
        ("except", _EXCEPT),
        ("else", _ELSE),
        ("bare try/finally body", _BARE_TRY_FINALLY),
    ),
)
def test_an_unprotected_try_section_is_not_guarded(label: str, source: str) -> None:
    """Each of these propagates straight out of the function."""
    assert _sites(source) == ["tldw_chatbook/UI/probe.py::run"], (
        f"a post-await DOM lookup in a {label} is not protected by that "
        f"statement's own handlers and must be censused"
    )


@pytest.mark.unit
def test_the_censused_p1_finally_site_is_actually_seen() -> None:
    """The real site the blind spot hid, at its real source.

    `SchedulesWorkbench._run_sync` re-enables two buttons from a `finally`
    after an awaited network sync, in a worker dispatched without
    `exit_on_error=False`. It had no census row at all before the fix.
    """
    checker = _load_checker()
    path = (
        _REPO_ROOT
        / "tldw_chatbook"
        / "UI"
        / "Screens"
        / "scheduling"
        / "schedules_workbench.py"
    )
    sites = checker.collect_w002(
        ast.parse(path.read_text(encoding="utf-8")), path
    )
    assert any(site.endswith("::_run_sync") for site in sites), (
        "_run_sync's finally-body query_one calls must be visible to the "
        "census; they were the review's P1"
    )
