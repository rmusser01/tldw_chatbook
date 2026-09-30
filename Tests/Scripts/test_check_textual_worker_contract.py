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


# --------------------------------------------------------------------------
# W003: a wait-for-dismiss screen push reachable from a non-worker coroutine
# (TASK-33621.13). Textual appends the screen and only then raises
# NoActiveWorker, killing the dispatching pump under the painted screen --
# the Console Inspector 'Choose folder' freeze (GAP4-01).
# --------------------------------------------------------------------------


def _w003(*sources: str) -> list[str]:
    """Run the W003 collector over one or more modules (cross-module on purpose)."""
    modules = [
        (ast.parse(source), _mod.REPO_ROOT / "tldw_chatbook" / "UI" / f"m{index}.py")
        for index, source in enumerate(sources)
    ]
    return sorted(_mod.collect_w003(modules))


def test_w003_flags_a_handler_that_awaits_push_screen_wait_directly():
    source = """
class S:
    @on(Button.Pressed, "#go")
    async def _go(self, event):
        await self.app.push_screen_wait(Picker())
"""
    assert _w003(source) == ["tldw_chatbook/UI/m0.py::S._go"]


@pytest.mark.parametrize(
    "push",
    [
        "await self.app.push_screen(Picker(), wait_for_dismiss=True)",
        "await self.app.push_screen(Picker(), None, True)",
    ],
)
def test_w003_flags_push_screen_with_wait_for_dismiss_from_an_action(push):
    source = f"""
class S:
    async def action_pick(self):
        {push}
"""
    assert _w003(source) == ["tldw_chatbook/UI/m0.py::S.action_pick"]


def test_w003_follows_awaited_helpers_to_the_handler():
    source = """
class S:
    async def on_button_pressed(self, event):
        await self._choose()

    async def _choose(self):
        return await self._ask()

    async def _ask(self):
        return await self.app.push_screen_wait(Picker())
"""
    assert _w003(source) == ["tldw_chatbook/UI/m0.py::S.on_button_pressed"]


def test_w003_follows_the_gap4_callable_chain_across_modules():
    """The real defect: a push reached through a `partial` keyword argument, a
    dict key, a constructor parameter and a `self.` attribute, in three files."""
    session = """
class SessionController:
    async def _select_binding(self, session_id):
        return await self._screen.app.push_screen_wait(SetupModal())
"""
    panel = """
async def recover_session(session_id, *, select_binding):
    return await select_binding(session_id)


def context_kwargs(screen):
    controller = screen._session
    return {
        "project_recovery": partial(
            recover_session, select_binding=controller._select_binding
        ),
    }
"""
    inspector = """
class Inspector(ModalScreen):
    def __init__(self, project_recovery=None):
        super().__init__()
        self._project_recovery = project_recovery

    @on(Panel.RecoveryRequested)
    async def _recover(self, event):
        state = await self._project_recovery(event.session_id)
"""
    assert _w003(session, panel, inspector) == [
        "tldw_chatbook/UI/m2.py::Inspector._recover"
    ]


def test_w003_follows_a_local_rename_of_the_callable():
    """`recovery = self._recovery; await recovery()` is the same await. A
    local name is resolved in its own function's scope: through the
    package-wide alias table, one unrelated `recovery = ...` elsewhere hid it."""
    source = """
class Inspector(ModalScreen):
    def __init__(self, recovery=None):
        super().__init__()
        self._recovery = recovery

    @on(Panel.RecoveryRequested)
    async def _recover(self, event):
        recovery = self._recovery
        await recovery(event.session_id)


async def _pick(session_id):
    await app.push_screen_wait(SetupModal())


def build():
    return Inspector(recovery=_pick)


def unrelated(items):
    recovery = len
    return recovery(items)
"""
    assert _w003(source) == ["tldw_chatbook/UI/m0.py::Inspector._recover"]


def test_w003_accepts_the_fixed_shape_a_handler_that_starts_a_worker():
    """The fix TASK-33621.13 shipped: the handler returns, a worker awaits."""
    source = """
class Inspector(ModalScreen):
    @on(Panel.RecoveryRequested)
    def _recover(self, event):
        self.run_worker(self._apply(event.session_id), group="g")

    async def _apply(self, session_id):
        await self.app.push_screen_wait(SetupModal())
"""
    assert _w003(source) == []


@pytest.mark.parametrize(
    "body",
    [
        # A `@work` coroutine is a worker.
        "@work(exclusive=True, group='g')\n    async def action_pick(self):\n"
        "        await self.app.push_screen_wait(Picker())",
        # Pushing with a callback never waits.
        "def action_pick(self):\n"
        "        self.app.push_screen(Picker(), callback=self._picked)",
        # `await run_worker(coro)` runs the coroutine in the worker.
        "async def action_pick(self):\n"
        "        await self.run_worker(self._pick()).wait()\n\n"
        "    async def _pick(self):\n"
        "        await self.app.push_screen_wait(Picker())",
    ],
    ids=["work-decorator", "callback", "run-worker-argument"],
)
def test_w003_does_not_flag_worker_or_callback_shapes(body):
    source = f"""
class S:
    {body}
"""
    assert _w003(source) == []


def test_w003_flags_a_waiting_callable_handed_to_a_pump_scheduler():
    source = """
class S:
    def on_mount(self):
        self.call_after_refresh(self._ask)

    async def _ask(self):
        await self.app.push_screen_wait(Picker())
"""
    assert _w003(source) == ["tldw_chatbook/UI/m0.py::S.on_mount->_ask"]


def test_w003_a_generic_alias_bound_to_one_waiting_callable_does_not_cascade():
    """An alias waits only when EVERY binding of it waits. With "any", one
    `callback=<waiting>` made every `await callback()` in the package wait."""
    source = """
class S:
    async def _ask(self):
        await self.app.push_screen_wait(Picker())

    def wire(self):
        Thing(callback=self._ask)
        Other(callback=self._plain)

    def _plain(self):
        pass

    async def on_key(self, event):
        await self.callback()
"""
    assert _w003(source) == []


def test_w003_resolves_self_calls_to_the_enclosing_class_first():
    """A same-named waiting method elsewhere does not make `self._ask()` wait
    when this class defines its own, non-waiting `_ask`."""
    other = """
class Elsewhere:
    async def _ask(self):
        await self.app.push_screen_wait(Picker())
"""
    source = """
class S:
    async def on_button_pressed(self, event):
        await self._ask()

    async def _ask(self):
        return 1
"""
    assert _w003(other, source) == []


def test_w003_real_inspector_recovery_still_resolves_as_waiting():
    """Regression pin on the REAL tree: the chain from the Inspector's worker
    coroutine through `project_instruction_context_kwargs`'s `partial` to the
    session controller's `push_screen_wait` still resolves -- so if anyone
    awaits that coroutine from the handler again, W003 fails."""
    import warnings

    collected = []
    for path in _mod._source_files():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        collected.append(_mod._collect_module(tree, _mod._rel(path)))
    graph = _mod._WaitGraph(collected)
    by_key = {fn.key: fn for fn in graph.functions}
    inspector = "tldw_chatbook/Widgets/Console/console_conversation_inspector.py"
    worker = by_key[
        f"{inspector}::ConsoleConversationInspector._apply_project_instruction_recovery"
    ]
    handler = by_key[
        f"{inspector}::ConsoleConversationInspector._recover_project_instructions"
    ]
    assert worker.waiting and not worker.is_root
    assert handler.is_root and not handler.waiting
    assert (
        f"{inspector}::ConsoleConversationInspector._recover_project_instructions"
        not in (graph.roots())
    )


def _run_main_w003(monkeypatch, tmp_path, source: str, census: str) -> int:
    package = tmp_path / "tldw_chatbook"
    module = package / "UI" / "sample.py"
    module.parent.mkdir(parents=True)
    module.write_text(source, encoding="utf-8")
    empty = tmp_path / "census.tsv"
    empty.write_text("# empty\n", encoding="utf-8")
    wait_census = tmp_path / "wait_census.tsv"
    wait_census.write_text(census, encoding="utf-8")
    monkeypatch.setattr(_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(_mod, "PACKAGE", package)
    monkeypatch.setattr(_mod, "CENSUS", empty)
    monkeypatch.setattr(_mod, "WAIT_PUSH_CENSUS", wait_census)
    monkeypatch.setattr("sys.argv", ["check_textual_worker_contract.py"])
    return _mod.main()


_W003_HANDLER = """
class S:
    async def action_pick(self):
        await self.app.push_screen_wait(Picker())
"""


def test_main_exits_nonzero_on_an_uncensused_w003_root(monkeypatch, tmp_path, capsys):
    """End-to-end negative control: the whole script, bad input, exit 1."""
    assert _run_main_w003(monkeypatch, tmp_path, _W003_HANDLER, "# empty\n") == 1
    out = capsys.readouterr().out
    assert "wait-for-dismiss screen push" in out
    assert "tldw_chatbook/UI/sample.py::S.action_pick" in out


def test_main_exits_zero_when_that_w003_root_is_pinned(monkeypatch, tmp_path):
    census = "# header\ntldw_chatbook/UI/sample.py::S.action_pick\t1\n"
    assert _run_main_w003(monkeypatch, tmp_path, _W003_HANDLER, census) == 0
