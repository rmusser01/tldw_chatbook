"""Negative controls for the Textual worker-contract guard (TASK-32897).

This guard was green while hiding 50 W002 sites: its ancestor walk set
``guarded = True`` on *any* enclosing ``ast.Try``, but a ``query_one`` in an
``except``/``else``/``finally`` clause is a child of the ``Try`` node while
sitting outside the region that ``Try``'s own handlers cover -- an exception
there propagates straight out of the worker and, with ``exit_on_error``
defaulting to True, exits the application.

A guard that has never been observed to fail is not known to work, so every
test here feeds it a deliberately bad sample and asserts it FAILS.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SPEC = importlib.util.spec_from_file_location(
    "_ctwc",
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "check_textual_worker_contract.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]


_SAMPLE = _mod.REPO_ROOT / "tldw_chatbook" / "UI" / "sample.py"


def _w002(source: str) -> list[str]:
    """Run the W002 collector over `source` as if it were a UI module."""
    return _mod.collect_w002(ast.parse(source), _SAMPLE)


_IN_TRY_BODY = """
class S:
    async def run(self):
        try:
            await self.work()
            self.query_one("#target")
        except Exception:
            pass
"""

_IN_EXCEPT = """
class S:
    async def run(self):
        try:
            await self.work()
        except Exception:
            self.query_one("#target")
"""

_IN_FINALLY = """
class S:
    async def run(self):
        try:
            await self.work()
        finally:
            self.query_one("#target")
"""

_IN_ELSE = """
class S:
    async def run(self):
        try:
            await self.work()
        except Exception:
            pass
        else:
            self.query_one("#target")
"""

_OUTER_TRY_COVERS_INNER_FINALLY = """
class S:
    async def run(self):
        try:
            try:
                await self.work()
            finally:
                self.query_one("#target")
        except Exception:
            pass
"""


@pytest.mark.parametrize(
    "clause, source",
    [
        ("except", _IN_EXCEPT),
        ("finally", _IN_FINALLY),
        ("else", _IN_ELSE),
    ],
)
def test_a_lookup_in_a_non_body_try_clause_is_reported(clause, source):
    """The defect. Each of these propagates out; none is guarded by its own
    `try`, and the shipped guard called all three protected."""
    assert _w002(source) == ["tldw_chatbook/UI/sample.py::run"], clause


def test_a_lookup_in_the_try_body_is_still_treated_as_guarded():
    """The fix must not turn the check into "every try is worthless"."""
    assert _w002(_IN_TRY_BODY) == []


def test_an_outer_try_still_protects_an_inner_finally():
    """The ascent must CONTINUE past a non-guarding `Try`, not break out of
    the loop: an enclosing `try` legitimately covers the inner `finally`."""
    assert _w002(_OUTER_TRY_COVERS_INNER_FINALLY) == []


def test_a_lookup_before_the_first_await_is_out_of_scope():
    source = """
class S:
    async def run(self):
        self.query_one("#target")
        await self.work()
"""
    assert _w002(source) == []


def _run_main(monkeypatch, tmp_path, source: str, census: str) -> int:
    package = tmp_path / "tldw_chatbook"
    module = package / "UI" / "sample.py"
    module.parent.mkdir(parents=True)
    module.write_text(source, encoding="utf-8")
    census_path = tmp_path / "census.tsv"
    census_path.write_text(census, encoding="utf-8")
    monkeypatch.setattr(_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_mod, "PACKAGE", package)
    monkeypatch.setattr(_mod, "CENSUS", census_path)
    monkeypatch.setattr("sys.argv", ["check_textual_worker_contract.py"])
    return _mod.main()


def test_main_exits_nonzero_on_an_uncensused_finally_lookup(
    monkeypatch, tmp_path, capsys
):
    """End-to-end negative control: the whole script, bad input, exit 1."""
    assert _run_main(monkeypatch, tmp_path, _IN_FINALLY, "# empty\n") == 1
    out = capsys.readouterr().out
    assert "new post-await DOM lookup" in out
    assert "tldw_chatbook/UI/sample.py::run" in out


def test_main_exits_zero_when_that_same_site_is_pinned(monkeypatch, tmp_path):
    """And the ratchet still works: a pinned site is a baseline, not a failure."""
    census = "# header\ntldw_chatbook/UI/sample.py::run\t1\n"
    assert _run_main(monkeypatch, tmp_path, _IN_FINALLY, census) == 0


def test_main_exits_nonzero_on_a_synchronous_run_worker_target(
    monkeypatch, tmp_path, capsys
):
    """W001 is a hard gate with no allowlist; prove it can still fail."""
    source = """
class S:
    def sync_target(self):
        pass

    def start(self):
        self.run_worker(self.sync_target)
"""
    assert _run_main(monkeypatch, tmp_path, source, "# empty\n") == 1
    assert "without thread=True" in capsys.readouterr().out


# --------------------------------------------------------------------------
# A `try` body only protects when a handler can actually catch the failure
# (TASK-32897 follow-up). `query_one` raises `NoMatches(QueryError(Exception))`.
# --------------------------------------------------------------------------

_TRY_FINALLY_NO_HANDLER = """
class S:
    async def run(self):
        try:
            await self.work()
            self.query_one("#target")
        finally:
            self.cleanup()
"""

_TRY_UNRELATED_HANDLER = """
class S:
    async def run(self):
        try:
            await self.work()
            self.query_one("#target")
        except ValueError:
            pass
"""


@pytest.mark.parametrize(
    "shape, source",
    [
        ("try/finally", _TRY_FINALLY_NO_HANDLER),
        ("except ValueError", _TRY_UNRELATED_HANDLER),
    ],
)
def test_a_try_body_whose_handlers_cannot_catch_is_not_protection(shape, source):
    """`finally:` runs but re-raises, and `except ValueError` never sees a
    `NoMatches` -- both propagate out of the worker exactly like no `try` at
    all. Treating placement in `Try.body` as protection hid them."""
    assert _w002(source) == ["tldw_chatbook/UI/sample.py::run"], shape


@pytest.mark.parametrize(
    "handler",
    [
        "except Exception:",
        "except BaseException:",
        "except:",
        "except NoMatches:",
        "except QueryError:",
        "except query.NoMatches:",
        "except (ValueError, NoMatches):",
    ],
)
def test_a_handler_that_can_catch_the_lookup_still_protects_its_body(handler):
    """The fix must not turn the check into "every try is worthless"."""
    source = f"""
class S:
    async def run(self):
        try:
            await self.work()
            self.query_one("#target")
        {handler}
            pass
"""
    assert _w002(source) == [], handler


def test_an_outer_catching_try_protects_a_non_catching_inner_one():
    """The ascent must continue past a `try` that cannot catch, not stop at
    it: the outer `except Exception` legitimately covers the inner body."""
    source = """
class S:
    async def run(self):
        try:
            try:
                await self.work()
                self.query_one("#target")
            except ValueError:
                pass
        except Exception:
            pass
"""
    assert _w002(source) == []


# --------------------------------------------------------------------------
# Nested `async def`s are scanned once, under their own name
# --------------------------------------------------------------------------


def test_a_nested_async_function_is_censused_once_under_its_own_name():
    """`ast.walk` reaches an inner `async def` from the outer function too, so
    an unguarded lookup in it was emitted twice -- once misattributed to the
    enclosing function. The inner function gets its own scan; the outer must
    not claim its sites."""
    source = """
class S:
    async def run(self):
        await self.work()

        async def inner():
            await self.work()
            try:
                pass
            finally:
                self.query_one("#target")

        self.run_worker(inner)
"""
    assert _w002(source) == ["tldw_chatbook/UI/sample.py::inner"]


def test_a_lookup_in_a_nested_sync_callback_still_belongs_to_its_async_owner():
    """Only *async* nested defs get their own scan. A lambda or plain `def`
    callback is never scanned separately, so dropping it from the enclosing
    function's census would silently delete real coverage -- a dialog
    callback dereferencing a screen that resolved after the await is exactly
    the W002 defect class."""
    source = """
class S:
    async def run(self):
        await self.work()
        self.push_screen(Modal(), lambda _: self.query_one("#target"))
"""
    assert _w002(source) == ["tldw_chatbook/UI/sample.py::run"]
