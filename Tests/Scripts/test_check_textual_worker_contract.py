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
import io
import shutil
import subprocess
import tarfile
import warnings
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


def test_a_lookup_inside_the_first_awaits_own_arguments_is_out_of_scope():
    """It runs before the suspension however a formatter wraps the call.

    The comparison used to be by line number, so wrapping the call turned a
    pre-await lookup into a reported one (and unwrapping hid five others).
    """
    one_line = """
class S:
    async def run(self):
        await self.save(self.query_one("#target").value)
"""
    wrapped = """
class S:
    async def run(self):
        await self.save(
            self.query_one("#target").value
        )
"""
    assert _w002(one_line) == []
    assert _w002(wrapped) == []


def test_a_wrapped_lookup_after_the_first_await_is_still_reported():
    source = """
class S:
    async def run(self):
        await self.work()
        await self.save(
            self.query_one("#target").value
        )
"""
    assert len(_w002(source)) == 1


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
# the Console Inspector 'Choose folder' freeze (GAP4-01). A row is
# "<entry point> => <function holding the push>".
# --------------------------------------------------------------------------

_M0 = "tldw_chatbook/UI/m0.py"


def _w003(*sources: str) -> list[str]:
    """Run the W003 collector over one or more modules (cross-module on purpose)."""
    modules = [
        (ast.parse(source), _mod.REPO_ROOT / "tldw_chatbook" / "UI" / f"m{index}.py")
        for index, source in enumerate(sources)
    ]
    return sorted(_mod.collect_w003(modules))


def _unresolved(*sources: str) -> list[str]:
    """The positional handoffs of a waiting callable W003 could not bind."""
    modules = [
        (ast.parse(source), _mod.REPO_ROOT / "tldw_chatbook" / "UI" / f"m{index}.py")
        for index, source in enumerate(sources)
    ]
    return sorted(_mod.collect_w003_unresolved(modules))


def _row(root: str, site: str | None = None, module: str = _M0) -> str:
    """A census key in ``module``; a root that pushes itself is its own site."""
    return f"{module}::{root} => {module}::{site or root}"


def _parse_file(path: Path) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return None


@pytest.fixture(scope="module")
def real_tree() -> dict[str, tuple]:
    """The real package, collected once: ``{relative path: (module, functions)}``.

    ``_WaitGraph`` resets the solved state it keeps on the functions, so
    tests may build several graphs from this one collection.
    """
    collected = {}
    for path in _mod._source_files():
        tree = _parse_file(path)
        if tree is not None:
            rel = _mod._rel(path)
            collected[rel] = _mod._collect_module(tree, rel)
    return collected


def test_w003_flags_a_handler_that_awaits_push_screen_wait_directly():
    """The base case: Textual runs ``push_screen_wait`` only in a worker. From
    an ``@on`` handler it pushes the screen, then raises ``NoActiveWorker``
    on the handler's own pump (GAP4-01)."""
    source = """
class S:
    @on(Button.Pressed, "#go")
    async def _go(self, event):
        await self.app.push_screen_wait(Picker())
"""
    assert _w003(source) == [_row("S._go")]


@pytest.mark.parametrize(
    "push",
    [
        "await self.app.push_screen(Picker(), wait_for_dismiss=True)",
        "await self.app.push_screen(Picker(), None, True)",
    ],
)
def test_w003_flags_push_screen_with_wait_for_dismiss_from_an_action(push):
    """``push_screen(..., wait_for_dismiss=True)`` is the same wait, as a
    keyword or as the third positional argument, and an ``action_*``
    method is a root like a handler."""
    source = f"""
class S:
    async def action_pick(self):
        {push}
"""
    assert _w003(source) == [_row("S.action_pick")]


def test_w003_follows_awaited_helpers_to_the_handler():
    """A push two awaits below the handler is reported at the handler and
    keyed to the helper that holds it: the handler never names the screen."""
    source = """
class S:
    async def on_button_pressed(self, event):
        await self._choose()

    async def _choose(self):
        return await self._ask()

    async def _ask(self):
        return await self.app.push_screen_wait(Picker())
"""
    assert _w003(source) == [_row("S.on_button_pressed", "S._ask")]


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
        "tldw_chatbook/UI/m2.py::Inspector._recover => "
        "tldw_chatbook/UI/m0.py::SessionController._select_binding"
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
    assert _w003(source) == [_row("Inspector._recover", "_pick")]


def test_w003_follows_a_local_rename_chain_of_any_length():
    """A chain of local renames is followed to its end. Bounded by a fixed
    depth of 16, the 17th rename returned no targets and the handler dropped
    out of the census with no error (PR #2944 review)."""
    renames = "\n".join(f"        r{i + 1} = r{i}" for i in range(20))
    source = f"""
class Inspector(ModalScreen):
    def __init__(self, recovery=None):
        super().__init__()
        self._recovery = recovery

    @on(Panel.RecoveryRequested)
    async def _recover(self, event):
        r0 = self._recovery
{renames}
        await r20(event.session_id)


async def _pick(session_id):
    await app.push_screen_wait(SetupModal())


def build():
    return Inspector(recovery=_pick)
"""
    assert _w003(source) == [_row("Inspector._recover", "_pick")]


def test_w003_a_local_rename_cycle_ends_without_a_row():
    """`a = b; b = a` names nothing: resolution stops at the repeat, and no
    row is invented from it."""
    source = """
class S:
    async def on_button_pressed(self, event):
        a = b
        b = a
        await a()
"""
    assert _w003(source) == []


def test_w003_a_local_bound_on_either_branch_waits_if_either_binding_does():
    """A local bound on two paths may hold either callable at the await, so
    every binding counts. Keeping only the last one by position would read
    `_plain` here and miss the push on the `if` path (PR #2944 review)."""
    source = """
class S:
    async def on_button_pressed(self, event):
        if event.button.id == "pick":
            choose = self._pick
        else:
            choose = self._plain
        await choose()

    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    async def _plain(self):
        return None
"""
    assert _w003(source) == [_row("S.on_button_pressed", "S._pick")]


# An unrelated class owns a waiting method that shares the bare name below.
_PICKER_CLASS = """
class Picker:
    async def pick(self):
        await self.app.push_screen_wait(PickerModal())
"""


_BARE_PICK = {
    "imported": """
from helpers import pick


class S:
    async def on_button_pressed(self, event):
        await pick()
""",
    "parameter": """
async def run(pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        await run(self._plain)

    async def _plain(self):
        return None
""",
}


@pytest.mark.parametrize("shape", sorted(_BARE_PICK))
def test_w003_a_bare_call_never_resolves_to_a_class_method(shape):
    """A bare `pick()` is a local, a module global, an import or a
    parameter -- never a method, which only an attribute (`obj.pick`)
    reaches. Falling back to every def named `pick`, methods included, made
    both of these wait through the unrelated `Picker.pick` (PR #2944
    review)."""
    assert _w003(_PICKER_CLASS, _BARE_PICK[shape]) == []


def test_w003_a_bare_call_still_reaches_a_module_function_in_another_module():
    """...while an imported module-level function that waits still does."""
    helpers = """
async def pick():
    await app.push_screen_wait(PickerModal())
"""
    source = """
from helpers import pick


class S:
    async def on_button_pressed(self, event):
        await pick()
"""
    assert _w003(_PICKER_CLASS, helpers, source) == [
        "tldw_chatbook/UI/m2.py::S.on_button_pressed => tldw_chatbook/UI/m1.py::pick"
    ]


def test_w003_attribute_calls_still_resolve_to_methods_by_name():
    """`obj.pick()` keeps the by-name over-approximation: `obj`'s type is
    not known statically, so every def named `pick` -- methods included --
    still counts (the GAP4-01 chain ran through one)."""
    source = """
class S:
    async def on_button_pressed(self, event):
        await self._picker.pick()
"""
    assert _w003(_PICKER_CLASS, source) == [
        "tldw_chatbook/UI/m1.py::S.on_button_pressed => "
        "tldw_chatbook/UI/m0.py::Picker.pick"
    ]


#: Ways a class body binds one of its own methods under a second name.
_CLASS_BODY_BINDINGS = {
    "alias": "choose = _pick",
    "partial": "choose = partial(_pick, which=1)",
    "alias-of-an-alias": "first = _pick\n    choose = first",
}


@pytest.mark.parametrize("binding", sorted(_CLASS_BODY_BINDINGS))
def test_w003_a_class_body_alias_of_a_method_still_waits(binding):
    """A bare name in a CLASS BODY reads that class's own namespace, so
    `choose = _pick` there names `S._pick`. Resolving it like a bare name in
    a function body -- module-level functions only -- lost this push (PR
    #2944 review of the bare-name fix)."""
    source = f"""
class S:
    async def _pick(self, which=0):
        await self.app.push_screen_wait(Picker())

    {_CLASS_BODY_BINDINGS[binding]}

    async def on_button_pressed(self, event):
        await self.choose()
"""
    assert _w003(source) == [_row("S.on_button_pressed", "S._pick")]


def test_w003_a_class_body_name_reads_its_own_class_before_the_module():
    """Python looks a class-body name up in the class first, then in the
    module: the class's own `_pick` shadows the module-level one, so the
    module's waiting `_pick` is not what `choose` holds."""
    source = """
async def _pick():
    await app.push_screen_wait(Picker())


class S:
    async def _pick(self):
        return None

    choose = _pick

    async def on_button_pressed(self, event):
        await self.choose()
"""
    assert _w003(source) == []


def test_w003_a_class_body_name_the_class_lacks_reads_the_module():
    """...and a name the class does not define is the module's."""
    source = """
async def _pick():
    await app.push_screen_wait(Picker())


class S:
    choose = _pick

    async def on_button_pressed(self, event):
        await self.choose()
"""
    assert _w003(source) == [_row("S.on_button_pressed", "_pick")]


#: A bare `_pick` read anywhere but the class body itself: each is the
#: imported (non-waiting) `_pick`, never the class's waiting method.
_OUTSIDE_THE_CLASS_BODY = {
    # A function body skips the class scope.
    "method-body": """
from helpers import _pick


class S:
    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    async def on_button_pressed(self, event):
        await _pick(self)
""",
    # So does a lambda's body, even a lambda written in the class body: it
    # runs later, in its own scope, and reads globals.
    "lambda-in-class-body": """
from helpers import _pick


class S:
    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    choose = lambda self: _pick(self)

    async def on_button_pressed(self, event):
        await self.choose()
""",
    "module-level": """
from helpers import _pick


class S:
    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    async def on_button_pressed(self, event):
        await choose()


choose = _pick
""",
}


@pytest.mark.parametrize("shape", sorted(_OUTSIDE_THE_CLASS_BODY))
def test_w003_a_bare_name_outside_a_class_body_never_reads_the_class(shape):
    """Only a class body reads the class namespace: a method body, a lambda
    body and module level all skip it, as Python's scoping does."""
    assert _w003(_OUTSIDE_THE_CLASS_BODY[shape]) == []


#: A waiting callable bound to an ATTRIBUTE named ``choose`` in one module:
#: a class-body assignment, or one through ``self``.
_ATTRIBUTE_CHOOSE = {
    "class-body": """
class S:
    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    choose = _pick
""",
    "self-attribute": """
class S:
    def __init__(self):
        self.choose = self._pick

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
}

#: How an unrelated module reaches a name ``choose``: a bare (imported) name
#: can never be another class's attribute; ``obj.choose()`` may be.
_READ_CHOOSE = {
    "bare-import": ("from helpers import choose", "await choose()", False),
    "attribute": ("", "await self._s.choose()", True),
}


@pytest.mark.parametrize("read", sorted(_READ_CHOOSE))
@pytest.mark.parametrize("binding", sorted(_ATTRIBUTE_CHOOSE))
def test_w003_an_attribute_binding_never_reaches_another_modules_bare_name(
    binding, read
):
    """A class-body ``choose = _pick`` (or ``self.choose = ...``) entered
    the package-wide alias table that a bare name's fallback reads, so an
    unrelated module's imported ``choose()`` waited through it (PR #2944
    round-6 review). ``obj.choose()`` still reaches it: by name, like any
    attribute call."""
    preamble, call, flagged = _READ_CHOOSE[read]
    other = f"""
{preamble}


class T:
    async def on_button_pressed(self, event):
        {call}
"""
    expected = (
        [
            "tldw_chatbook/UI/m1.py::T.on_button_pressed => "
            "tldw_chatbook/UI/m0.py::S._pick"
        ]
        if flagged
        else []
    )
    assert _w003(_ATTRIBUTE_CHOOSE[binding], other) == expected


#: What a BASE class binds under ``_pick``: a waiting method, or a waiting
#: class-body alias. Neither is visible to a subclass's class BODY.
_BASE_PICK = {
    "method": (
        "async def _pick(self):\n        await self.app.push_screen_wait(Picker())"
    ),
    "class-body-alias": "_pick = _waiting",
}


@pytest.mark.parametrize("base", sorted(_BASE_PICK))
def test_w003_a_class_body_never_reads_its_base_classes(base):
    """``choose = _pick`` in ``S``'s body reads ``S``'s own namespace and then
    the module -- never ``Base``'s, which a class body cannot see (Python
    raises NameError there unless the module binds ``_pick``). Here the
    module's ``_pick`` is a non-waiting import, so nothing waits. The
    ``class-body-alias`` base used to leak through the package-wide alias
    table and make ``S.choose`` wait."""
    source = f"""
from helpers import _pick


async def _waiting(self):
    await self.app.push_screen_wait(Picker())


class Base:
    {_BASE_PICK[base]}


class S(Base):
    choose = _pick

    async def on_button_pressed(self, event):
        await self.choose()
"""
    assert _w003(source) == []


#: A class defined INSIDE a function, reading a waiting callable that only
#: the enclosing function binds -- as a nested def or a local alias.
_CLASS_IN_A_FUNCTION = {
    "class-body-alias": """
def build_screen(app):
    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    class S(Screen):
        choose = _pick

        async def on_button_pressed(self, event):
            await self.choose()

    return S
""",
    "method-body": """
def build_screen(app):
    async def _pick():
        await app.push_screen_wait(Picker())

    class S(Screen):
        async def on_button_pressed(self, event):
            await _pick()

    return S
""",
    "method-body-local-alias": """
def build_screen(app):
    async def _pick():
        await app.push_screen_wait(Picker())

    ask = _pick

    class S(Screen):
        async def on_button_pressed(self, event):
            await ask()

    return S
""",
    "lambda-in-class-body": """
def build_screen(app):
    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    class S(Screen):
        choose = lambda self: _pick(self)

        async def on_button_pressed(self, event):
            await self.choose()

    return S
""",
}


@pytest.mark.parametrize("shape", sorted(_CLASS_IN_A_FUNCTION))
def test_w003_a_class_inside_a_function_reads_the_enclosing_functions_locals(
    shape,
):
    """Python resolves a free name in a nested class's body (and in its
    methods) through the enclosing function's scope before the module. The
    collector reset the enclosing function at the ``class`` statement, so
    these resolved as module names and lost the push (PR #2944 round-6
    review)."""
    assert _w003(_CLASS_IN_A_FUNCTION[shape]) == [_row("S.on_button_pressed", "_pick")]


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
        # Pushing with a callback, and returning, never waits.
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
    """Shapes that never block a pump: a ``@work`` method, a callback push
    that returns at once, and a coroutine handed to ``run_worker``."""
    source = f"""
class S:
    {body}
"""
    assert _w003(source) == []


def test_w003_a_push_built_inline_as_the_worker_argument_is_not_the_handlers():
    """`run_worker(app.push_screen_wait(m))` -- the fix this check's own error
    text recommends -- runs the push in the worker. Review of TASK-33621.13:
    the first cut reported it as a direct push by the handler."""
    source = """
class S:
    def action_pick(self):
        self.run_worker(self.app.push_screen_wait(Picker()), exit_on_error=False)

    @on(Button.Pressed)
    def _pressed(self, event):
        self.run_worker(coro=self.app.push_screen(Picker(), wait_for_dismiss=True))
"""
    assert _w003(source) == []


@pytest.mark.parametrize(
    "push",
    [
        "self.app.push_screen(Picker(), callback=self._after)",
        "self.app.push_screen(Picker(), self._after)",
        "self.app.push_screen(Picker(), lambda result: self._after(result))",
    ],
    ids=["keyword", "positional", "lambda"],
)
def test_w003_flags_a_waiting_push_screen_result_callback(push):
    """Textual runs a ``push_screen`` result callback through the requester's
    ``call_next`` -- on a pump, never in a worker -- and ``invoke`` awaits the
    coroutine a one-call ``lambda`` returns."""
    source = f"""
class S:
    def action_pick(self):
        {push}

    async def _after(self, result):
        await self.app.push_screen_wait(Confirm())
"""
    assert _w003(source) == [_row("S.action_pick->_after", "S._after")]


@pytest.mark.parametrize(
    "schedule, flagged",
    [
        ("self.call_after_refresh(self._ask)", True),
        ("self.set_timer(0.5, self._ask)", True),
        ("self.set_timer(0.5, callback=self._ask)", True),
        # `_ask` is run_worker's argument here: it runs in a worker. Scanning
        # every argument flagged ChunkingLabScreen.on_mount->_load this way.
        ("self.call_after_refresh(self.run_worker, self._ask)", False),
    ],
)
def test_w003_flags_a_waiting_callable_handed_to_a_pump_scheduler(schedule, flagged):
    """``call_after_refresh`` and ``set_timer`` run their callable on the pump,
    so a waiting one is a root; handed on as ``run_worker``'s argument it
    runs in a worker and is not."""
    source = f"""
class S:
    def on_mount(self):
        {schedule}

    async def _ask(self):
        await self.app.push_screen_wait(Picker())
"""
    expected = [_row("S.on_mount->_ask", "S._ask")] if flagged else []
    assert _w003(source) == expected


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


# The shape of PR #2922's ConsoleHooksController: a constructor-injected
# `request_review` callable stored as `self._review`, awaited from a Send
# dispatcher. BuddyManagementModal happens to own an unrelated, waiting
# `_review` method.
_BUDDY = """
class BuddyManagementModal(ModalScreen):
    async def _review(self):
        await self.app.push_screen_wait(Confirm())
"""

_HOOKS = """
class HooksController:
    def __init__(self, *, request_review):
        self._review = request_review

    async def dispatch(self, draft):
        return await self._review(draft)


class Console(Screen):
    async def on_button_pressed(self, event):
        await self._hooks.dispatch("draft")
"""


def test_w003_a_self_attribute_never_resolves_to_an_unrelated_class_method():
    """TASK-33621.13 review: `self._review` resolved by NAME to
    BuddyManagementModal._review, so the Console's Send dispatchers were
    censused through a collision -- and, keyed by entry point alone, that one
    row exempted every later push reachable from them."""
    wiring = """
def wire(screen):
    screen._hooks = HooksController(request_review=lambda draft: len(draft))
"""
    assert _w003(_BUDDY, _HOOKS, wiring) == []


def test_w003_follows_the_injected_callable_to_its_real_push_site():
    """...while the REAL chain -- a one-call `lambda` bound to the keyword the
    constructor stores on `self` -- still resolves, to the real push site."""
    wiring = """
def wire(screen):
    screen._hooks = HooksController(
        request_review=lambda draft: screen._request_review(draft)
    )


class Screen2:
    async def _request_review(self, draft):
        return await self.app.push_screen_wait(Review(draft))
"""
    assert _w003(_BUDDY, _HOOKS, wiring) == [
        "tldw_chatbook/UI/m1.py::Console.on_button_pressed => "
        "tldw_chatbook/UI/m2.py::Screen2._request_review"
    ]


# A positional-or-keyword constructor parameter, so the same controller can
# be wired either way (PR #2945 review: `HooksController(_pick)`).
_POSITIONAL_HOOKS = """
class HooksController:
    def __init__(self, request_review, *, start_worker=None):
        self._review = request_review

    async def dispatch(self, draft):
        return await self._review(draft)


class SubController(HooksController):
    pass


class Console(Screen):
    async def on_button_pressed(self, event):
        await self._hooks.dispatch("draft")
"""

_WAITING_PICK = """
async def _pick(draft):
    return await app.push_screen_wait(Review(draft))


def wire(screen):
    screen._hooks = {wiring}
"""


@pytest.mark.parametrize(
    "wiring",
    [
        "HooksController(request_review=_pick)",
        "HooksController(_pick)",
        "HooksController(_pick, start_worker=None)",
        # The constructor is inherited: found through the class's MRO.
        "SubController(_pick)",
        "SubController(lambda draft: _pick(draft))",
        "SubController(partial(_pick))",
    ],
    ids=[
        "keyword",
        "positional",
        "positional-and-keyword",
        "inherited-init",
        "lambda",
        "partial",
    ],
)
def test_w003_a_positional_handoff_binds_the_constructor_parameter(wiring):
    """``HooksController(_pick)`` binds ``_pick`` to ``request_review``
    exactly as the keyword form does. Only keywords made an alias, so the
    positional wiring hid the push (PR #2945 thread PRRT_kwDOOcyyl86nx4XM)."""
    assert _w003(_POSITIONAL_HOOKS, _WAITING_PICK.format(wiring=wiring)) == [
        "tldw_chatbook/UI/m0.py::Console.on_button_pressed => "
        "tldw_chatbook/UI/m1.py::_pick"
    ]


#: A waiting callable handed positionally to a function or a method, which
#: awaits its parameter.
_POSITIONAL_CALLEES = {
    "module-function": """
async def run(pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        await run(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
    "self-method": """
class S:
    async def on_button_pressed(self, event):
        await self._run(self._pick)

    async def _run(self, pick):
        await pick()

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
    "static-method": """
class S:
    async def on_button_pressed(self, event):
        await self._run(self._pick)

    @staticmethod
    async def _run(pick):
        await pick()

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
    "second-position": """
async def run(label, pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        await run("go", self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
}


@pytest.mark.parametrize("callee", sorted(_POSITIONAL_CALLEES))
def test_w003_a_positional_handoff_binds_a_function_or_method_parameter(callee):
    """The same for a plain call: the argument at position N binds the
    callee's Nth parameter (after ``self`` for a bound method)."""
    assert _w003(_POSITIONAL_CALLEES[callee]) == [
        _row("S.on_button_pressed", "S._pick")
    ]


# Not ids "live"/"dead": Tests/conftest.py skips any test keyworded "live"
# unless --run-live is given.
@pytest.mark.parametrize(
    "live_takes_pick", [True, False], ids=["live-def-takes-pick", "dead-def-takes-pick"]
)
def test_w003_a_positional_handoff_binds_only_the_live_definitions_parameter(
    live_takes_pick,
):
    """``obj.run(self._pick)`` binds the parameter of the ``run`` Python
    actually binds -- the last definition -- not a dead earlier one whose
    parameter happens to be named ``pick``."""
    pick, other = "def run(self, pick):", "def run(self, other):"
    first, last = (other, pick) if live_takes_pick else (pick, other)
    source = f"""
class Runner:
    {first}
        return None

    {last}
        return None


async def use(pick=None):
    await pick()


class S:
    def wire(self):
        self._runner.run(self._pick)

    async def on_button_pressed(self, event):
        await use()

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    expected = [_row("S.on_button_pressed", "S._pick")] if live_takes_pick else []
    assert _w003(source) == expected


def test_w003_a_positional_binding_counts_exactly_like_a_keyword_binding():
    """Positional bindings join the keyword ones under the same alias rule (a
    parameter waits only when EVERY binding of it does), so wiring the
    non-waiting call positionally or by keyword yields the same rows. Only
    the keyword binding was visible, so the positional wiring made ``pick``
    look bound to ``_pick`` alone."""
    template = """
async def run(pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        await {plain}

    async def action_go(self):
        await run(pick=self._pick)

    async def _plain(self):
        return None

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    by_keyword = _w003(template.format(plain="run(pick=self._plain)"))
    assert _w003(template.format(plain="run(self._plain)")) == by_keyword


#: A waiting callable handed positionally where W003 cannot name the
#: parameter it binds: no definition of the callee in the package, or a
#: ``*args`` that swallows it.
_UNRESOLVED_HANDOFFS = {
    "external-callee": (
        """
from somewhere import external


class S:
    async def on_button_pressed(self, event):
        await external(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
        "external#0",
    ),
    "star-args": (
        """
async def run(*callbacks):
    for callback in callbacks:
        await callback()


class S:
    async def on_button_pressed(self, event):
        await run(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
        "run#0",
    ),
}


@pytest.mark.parametrize("shape", sorted(_UNRESOLVED_HANDOFFS))
def test_w003_reports_a_positional_handoff_it_cannot_resolve(shape):
    """A positional handoff of a WAITING callable that W003 cannot bind to a
    parameter is reported as unresolved -- naming the caller, the callee and
    the position -- rather than silently dropped."""
    source, handoff = _UNRESOLVED_HANDOFFS[shape]
    assert _unresolved(source) == [
        f"{_M0}::S.on_button_pressed -> {handoff} => {_M0}::S._pick"
    ]


def test_w003_a_non_waiting_or_known_positional_handoff_is_not_reported():
    """Only a WAITING callable is worth naming, and only where no rule
    already covers the callee: a worker, a pump scheduler and a resolvable
    parameter are all accounted for. An ``obj.x`` value matches every ``x``
    by name, so it is not named either: on the real tree each such report
    was ``getattr(Stylesheet.apply, ...)`` colliding with an unrelated
    dialog's waiting ``apply``."""
    source = """
from somewhere import external


class Dialog:
    async def apply(self):
        await self.app.push_screen_wait(Confirm())


class S:
    def on_mount(self):
        external(self._plain)
        getattr(Stylesheet.apply, "_marker", False)
        self.run_worker(self._pick)
        self.call_after_refresh(self._pick)
        self.call_from_thread(self._pick)
        self._keep(self._pick)

    def _keep(self, callback):
        self._callback = callback

    async def _plain(self):
        return None

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    assert _unresolved(source) == []


_PARTIAL_WIRING = """
from somewhere import external


async def run(label, pick):
    await pick()


class S:
    def on_mount(self):
        self.call_later({wiring})

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""


@pytest.mark.parametrize(
    "wiring",
    ["partial(run, 'go', pick=self._pick)", "partial(run, 'go', self._pick)"],
    ids=["keyword", "positional"],
)
def test_w003_a_positional_argument_after_partials_target_binds_its_parameter(
    wiring,
):
    """``partial(run, 'go', self._pick)`` hands ``_pick`` to ``run``'s second
    parameter, as ``partial(run, 'go', pick=self._pick)`` does by keyword.
    ``partial`` was skipped as a modelled callee, so only its target was
    followed and the positional argument was neither bound nor reported."""
    assert _w003(_PARTIAL_WIRING.format(wiring=wiring)) == [
        _row("S.on_mount->run", "S._pick")
    ]


def test_w003_reports_a_positional_argument_after_an_unresolvable_partial_target():
    """The unresolved report covers ``partial``'s arguments too, at the
    position the target sees."""
    source = _PARTIAL_WIRING.format(wiring="partial(external, self._pick)")
    assert _unresolved(source) == [f"{_M0}::S.on_mount -> external#0 => {_M0}::S._pick"]


#: A waiting ``self.pick`` handed to ``run``'s parameter of the SAME name.
_SAME_NAME = """
async def run(pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        {call}

    async def pick(self):
        await self.app.push_screen_wait(Picker())
"""


@pytest.mark.parametrize(
    "call",
    [
        "await run(self.pick)",
        "await run(pick=self.pick)",
        "pick = self.pick\n        await pick()",
    ],
    ids=["positional", "keyword", "local-alias"],
)
def test_w003_a_same_named_value_still_binds_the_parameter(call):
    """``run(self.pick)`` binds ``run``'s ``pick`` to the method: the two
    share a name, not a value. Each form was skipped on the name alone
    (``param != ref[1]``), so ``await pick()`` in ``run`` -- and in the
    caller, for the local alias -- reached nothing (PR #2987 review)."""
    assert _w003(_SAME_NAME.format(call=call)) == [
        _row("S.on_button_pressed", "S.pick")
    ]


@pytest.mark.parametrize(
    "forward",
    ["await run(pick)", "await run(pick=pick)"],
    ids=["positional", "keyword"],
)
def test_w003_forwarding_a_same_named_parameter_neither_binds_nor_blocks(forward):
    """``relay(pick)`` handing its own ``pick`` on to ``run(pick)`` binds the
    parameter to itself. That binding says nothing, so it is dropped -- not
    recorded as one more binding that must wait, which would stop ``pick``
    ever waiting through ``run(self._pick)``."""
    source = f"""
async def run(pick):
    await pick()


async def relay(pick):
    {forward}


class S:
    async def on_button_pressed(self, event):
        await run(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    assert _w003(source) == [_row("S.on_button_pressed", "S._pick")]


#: A positional handoff through a local alias of the callee.
_ALIASED_CALLEE = {
    "module-function": """
async def run(pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        go = run
        await go(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
    "bound-method": """
class S:
    async def on_button_pressed(self, event):
        go = self._run
        await go(self._pick)

    async def _run(self, pick):
        await pick()

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
    "alias-chain": """
async def run(pick):
    await pick()


class S:
    async def on_button_pressed(self, event):
        first = run
        go = first
        await go(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
""",
}


@pytest.mark.parametrize("shape", sorted(_ALIASED_CALLEE))
def test_w003_a_positional_handoff_through_a_local_alias_binds_the_parameter(shape):
    """``go = run; go(self._pick)`` binds ``run``'s ``pick`` as ``run(self._pick)``
    does: the local alias is followed to the callee (a bound method keeps
    its ``self`` offset). It was reported as an unresolved handoff to
    ``go`` instead (PR #2987 review)."""
    source = _ALIASED_CALLEE[shape]
    assert _unresolved(source) == []
    assert _w003(source) == [_row("S.on_button_pressed", "S._pick")]


def test_w003_a_local_alias_cycle_as_a_callee_is_reported_not_followed_forever():
    """``a = b; b = a; a(self._pick)`` names no callee: the alias walk stops
    at the repeat and the handoff is reported as unresolved."""
    source = """
class S:
    async def on_button_pressed(self, event):
        first = second
        second = first
        await first(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    assert _unresolved(source) == [
        f"{_M0}::S.on_button_pressed -> first#0 => {_M0}::S._pick"
    ]


def test_w003_a_class_body_call_binds_the_parameter_of_the_classs_own_function():
    """A class body reads its own namespace first, so ``_keep(_pick)`` there
    calls the class's ``_keep`` -- a plain function at that point, with no
    ``self`` to skip -- and binds its ``pick``. It was searched for in the
    module only and reported as unresolved (PR #2987 review)."""
    source = """
async def use(pick=None):
    await pick()


class S:
    def _keep(pick):
        return pick

    async def _pick(self):
        await self.app.push_screen_wait(Picker())

    _kept = _keep(_pick)

    async def on_button_pressed(self, event):
        await use()
"""
    assert _unresolved(source) == []
    assert _w003(source) == [_row("S.on_button_pressed", "S._pick")]


#: Calls elsewhere in the package that name an ``__init__``/``run`` by
#: attribute and pass a NON-waiting value positionally.
_UNRELATED_POSITIONAL = {
    # `Base.__init__(self, ...)`: the receiver is passed explicitly, so
    # position 0 is `self`, not the first parameter after it. As a bound
    # call by name it bound `self` to EVERY package `__init__`'s first
    # parameter -- `request_review` included.
    "unbound-base-init": """
class Failure(RuntimeError):
    def __init__(self, reason):
        super().__init__(reason)


class Timeout(Failure):
    def __init__(self):
        Failure.__init__(self, "timeout")
""",
    # `runner.run(len)`: `runner`'s type is unknown, so `run` matched every
    # package def of that name by name -- one that is no relation of the
    # call's real callee binds `len` to its `request_review`.
    "by-name-method": """
class Runner:
    def run(self, request_review):
        return request_review


def elsewhere(runner):
    runner.run(len)
""",
}


@pytest.mark.parametrize("shape", sorted(_UNRELATED_POSITIONAL))
def test_w003_an_unrelated_positional_call_never_cancels_a_keyword_binding(shape):
    """``HooksController(request_review=_pick)`` stays a row whatever an
    unrelated module passes positionally. A parameter waits only when EVERY
    binding of it does, so a guessed, non-waiting binding to
    ``request_review`` silently removed the row dev reported (PR #2987
    review)."""
    wiring = _WAITING_PICK.format(wiring="HooksController(request_review=_pick)")
    expected = [
        "tldw_chatbook/UI/m0.py::Console.on_button_pressed => "
        "tldw_chatbook/UI/m1.py::_pick"
    ]
    # Precondition: the row, without the unrelated module.
    assert _w003(_POSITIONAL_HOOKS, wiring) == expected
    assert _w003(_POSITIONAL_HOOKS, wiring, _UNRELATED_POSITIONAL[shape]) == expected


def test_w003_a_guessed_waiting_binding_is_not_cancelled_by_another_guess():
    """With only by-name bindings of ``pick`` -- ``self._runner.run(self._pick)``
    and an unrelated ``o.run(len)`` -- the waiting one is enough: a guess
    that does not wait is more likely another ``run``'s, so it never cancels
    one that does. As ordinary bindings, "every binding must wait" let the
    unrelated call hide the row."""
    source = """
class Runner:
    async def run(self, pick):
        await pick()


class Other:
    def run(self, pick):
        return pick


def elsewhere(o):
    o.run(len)


class S:
    async def on_button_pressed(self, event):
        await self._runner.run(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    # Precondition: the waiting guess alone.
    alone = source.replace("    o.run(len)", "    pass")
    assert _w003(alone) == [_row("S.on_button_pressed", "S._pick")]
    assert _w003(source) == [_row("S.on_button_pressed", "S._pick")]


def test_w003_a_classmethod_called_on_its_class_binds_after_cls():
    """``Base.make(self._pick)`` on a ``@classmethod`` binds ``cls`` itself,
    so the argument is ``make``'s parameter AFTER it."""
    source = """
class Base:
    @classmethod
    def make(cls, pick):
        cls._pick_cb = pick

    async def on_button_pressed(self, event):
        await self._pick_cb()


class Child(Base):
    def __init__(self):
        Base.make(self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    assert _unresolved(source) == []
    assert _w003(source) == [_row("Base.on_button_pressed", "Child._pick")]


def test_w003_an_unbound_base_call_binds_the_bases_own_parameter():
    """``Base.__init__(self, _pick)`` binds ``Base.__init__``'s parameter
    AFTER ``self`` -- position 1, offset 0 -- and only that class's (through
    its MRO), not every ``__init__`` sharing the name."""
    source = """
class Base:
    def __init__(self, pick):
        self._pick_cb = pick

    async def on_button_pressed(self, event):
        await self._pick_cb()


class Child(Base):
    def __init__(self):
        Base.__init__(self, self._pick)

    async def _pick(self):
        await self.app.push_screen_wait(Picker())
"""
    assert _unresolved(source) == []
    assert _w003(source) == [_row("Base.on_button_pressed", "Child._pick")]


def test_w003_self_call_resolves_through_a_package_base_class():
    """An inherited `self._pick()` is the base class's `_pick`, found by the
    class's own bases -- not by every `_pick` in the package."""
    base = """
class PickerBase(Screen):
    async def _pick(self):
        return await self.app.push_screen_wait(Picker())
"""
    source = """
class Child(PickerBase):
    async def action_pick(self):
        await self._pick()


class Unrelated(Screen):
    async def action_pick(self):
        await self._pick()
"""
    assert _w003(base, source) == [
        "tldw_chatbook/UI/m1.py::Child.action_pick => "
        "tldw_chatbook/UI/m0.py::PickerBase._pick"
    ]


# `self.x()` is dynamic dispatch: it runs whatever the INSTANCE's class
# resolves `x` to, and that class may be a subclass of the one holding the
# call. 2ebe5b1a7e resolved only upward (the class and its bases), so both
# shapes below -- a mixin calling what its host class defines, and a base
# class template method calling what a subclass overrides -- went invisible
# to W003 (TASK-33621.13, round 3 review).

_SETTINGS_MIXIN = """
class SettingsMixin:
    async def action_leave(self):
        await self._ask_leave_choice()
"""

#: An unrelated class with a waiting method of the same name, which must not
#: be reached: subclass-aware is not "by bare name".
_UNRELATED_LEAVE = """
class UnrelatedPanel(Vertical):
    async def _ask_leave_choice(self):
        return await self.app.push_screen_wait(OtherLeaveModal())
"""


@pytest.mark.parametrize(
    "host, site",
    [
        (
            # The class that mixes the mixin in defines the method.
            """
class SettingsPane(SettingsMixin, Vertical):
    async def _ask_leave_choice(self):
        return await self.app.push_screen_wait(LeaveModal())
""",
            "SettingsPane._ask_leave_choice",
        ),
        (
            # A SIBLING mixin of the same host defines it.
            """
class LeaveMixin:
    async def _ask_leave_choice(self):
        return await self.app.push_screen_wait(LeaveModal())


class SettingsPane(SettingsMixin, LeaveMixin, Vertical):
    pass
""",
            "LeaveMixin._ask_leave_choice",
        ),
        (
            # The host binds it as an attribute. The package-wide alias
            # `_ask_leave_choice` does NOT wait (`wire` binds it to `len`),
            # so only the host's own binding can make this row.
            """
class SettingsPane(SettingsMixin, Vertical):
    def __init__(self):
        super().__init__()
        self._ask_leave_choice = self._confirm

    async def _confirm(self):
        return await self.app.push_screen_wait(LeaveModal())


def wire(other):
    other._ask_leave_choice = len
""",
            "SettingsPane._confirm",
        ),
    ],
    ids=["host-class", "sibling-mixin", "host-attribute"],
)
def test_w003_a_mixin_self_call_resolves_through_the_class_mixing_it_in(host, site):
    """The real-tree shape: ``SpeechSettingsMixin`` code calling
    ``SpeechSettingsPane._ask_leave_choice``. Flagged at 0bc03b4715 (by-name
    fallback), silently passed at 2ebe5b1a7e (upward-only resolution)."""
    assert _w003(_SETTINGS_MIXIN, host, _UNRELATED_LEAVE) == [
        f"{_M0}::SettingsMixin.action_leave => tldw_chatbook/UI/m1.py::{site}"
    ]


_IMPORTER_BASE = """
class ImporterBase(Screen):
    async def action_import(self):
        path = await self._choose_path()
        self.load(path)

    async def _choose_path(self):
        {default}
"""

_FILE_IMPORTER = """
class FileImporter(ImporterBase):
    async def _choose_path(self):
        return await self.app.push_screen_wait(FilePicker())
"""


@pytest.mark.parametrize(
    "default", ["return None", "raise NotImplementedError"], ids=["default", "abstract"]
)
@pytest.mark.parametrize("depth", ["child", "grandchild"])
def test_w003_a_template_method_reaches_a_subclass_override(default, depth):
    """A base-class action awaiting ``self._choose_path()`` runs the
    subclass's override on a subclass instance -- and that override waits."""
    subclass = (
        _FILE_IMPORTER
        if depth == "child"
        else _FILE_IMPORTER.replace("(ImporterBase)", "(MidImporter)")
        + "\n\nclass MidImporter(ImporterBase):\n    pass\n"
    )
    assert _w003(_IMPORTER_BASE.format(default=default), subclass) == [
        f"{_M0}::ImporterBase.action_import => "
        "tldw_chatbook/UI/m1.py::FileImporter._choose_path"
    ]


def test_w003_a_sibling_subclass_override_does_not_reach_another_subclass():
    """Subclass-aware resolution follows the calling class's OWN descendants.
    ``Quiet`` instances run ``ImporterBase._choose_path``; ``FileImporter``'s
    override is a sibling's and can never run for them."""
    quiet = """
class Quiet(ImporterBase):
    async def action_quiet(self):
        await self._choose_path()
"""
    base = _IMPORTER_BASE.format(default="return None")
    assert _w003(base, quiet, _FILE_IMPORTER) == [
        f"{_M0}::ImporterBase.action_import => "
        "tldw_chatbook/UI/m2.py::FileImporter._choose_path"
    ]


# Python binds a name defined twice to its LAST definition. The collector
# walks a LIFO stack, so siblings arrive last-first, and 2ebe5b1a7e let the
# later-arriving FIRST definition overwrite it.
_TWICE = {
    "class": (
        "class S:\n"
        "    async def _ask(self):\n        {first}\n"
        "    async def _ask(self):\n        {last}\n"
        "    async def on_button_pressed(self, event):\n        await self._ask()\n",
        "S._ask",
    ),
    "module": (
        "async def _ask():\n    {first}\n"
        "async def _ask():\n    {last}\n"
        "class S:\n"
        "    async def on_button_pressed(self, event):\n        await _ask()\n",
        "_ask",
    ),
    "nested": (
        "class S:\n"
        "    async def on_button_pressed(self, event):\n"
        "        async def _ask():\n            {first}\n"
        "        async def _ask():\n            {last}\n"
        "        await _ask()\n",
        "_ask",
    ),
}

_WAITS = "return await app.push_screen_wait(Picker())"
_RETURNS = "return 1"


@pytest.mark.parametrize("scope", sorted(_TWICE))
@pytest.mark.parametrize("live_waits", [True, False], ids=["live-waits", "dead-waits"])
def test_w003_a_name_defined_twice_resolves_to_its_last_definition(scope, live_waits):
    """Only the live (last) definition decides whether the call waits, in
    either order and at class, module and nested scope."""
    template, site = _TWICE[scope]
    first, last = (_RETURNS, _WAITS) if live_waits else (_WAITS, _RETURNS)
    source = template.format(first=first, last=last)
    expected = [_row("S.on_button_pressed", site)] if live_waits else []
    assert _w003(source) == expected


@pytest.mark.parametrize("live_waits", [True, False], ids=["live-waits", "dead-waits"])
def test_w003_a_dead_definition_is_not_reached_by_name_either(live_waits):
    """``obj._ask()`` matches every ``_ask`` by name -- but only the bound,
    live definitions: a dead first ``S._ask`` that waits put ``_ask`` in the
    by-name table and made an unrelated ``T`` handler wait through it."""
    first, last = (_RETURNS, _WAITS) if live_waits else (_WAITS, _RETURNS)
    source = (
        "class S:\n"
        f"    async def _ask(self):\n        {first}\n"
        f"    async def _ask(self):\n        {last}\n"
        "class T:\n"
        "    async def on_button_pressed(self, event):\n"
        "        await self._s._ask()\n"
    )
    expected = [_row("T.on_button_pressed", "S._ask")] if live_waits else []
    assert _w003(source) == expected


#: A ROOT defined twice in one scope, as ``(template, root key)``. Python
#: never runs the first definition, so only the last one can be an entry
#: point -- at class, module and nested scope, as a handler and as a
#: function that schedules a waiting callable.
_ROOT_TWICE = {
    "handler": (
        "class S:\n"
        "    async def on_button_pressed(self, event):\n        {first}\n"
        "    async def on_button_pressed(self, event):\n        {last}\n",
        "S.on_button_pressed",
    ),
    "module-handler": (
        "async def on_ready(app):\n    {first}\nasync def on_ready(app):\n    {last}\n",
        "on_ready",
    ),
    "nested-handler": (
        "def build():\n"
        "    async def on_ready():\n        {first}\n"
        "    async def on_ready():\n        {last}\n"
        "    return on_ready\n",
        "on_ready",
    ),
    "scheduler": (
        "class S:\n"
        "    def _start(self):\n        {first}\n"
        "    def _start(self):\n        {last}\n"
        "    async def _ask(self):\n"
        "        await self.app.push_screen_wait(Picker())\n",
        "S._start->_ask",
    ),
}

_ROOT_BODIES = {
    True: "await app.push_screen_wait(Picker())",
    False: "return None",
}


@pytest.mark.parametrize("scope", sorted(_ROOT_TWICE))
@pytest.mark.parametrize("live_waits", [True, False], ids=["live-waits", "dead-waits"])
def test_w003_only_a_live_definition_of_a_root_is_a_root(scope, live_waits):
    """A handler whose dead first definition waits made a census row that no
    fix to the live handler could clear (PR #2945 thread
    PRRT_kwDOOcyyl86nx4Xw); a live definition that waits is still one."""
    template, root = _ROOT_TWICE[scope]
    if scope == "scheduler":
        bodies = {True: "self.call_later(self._ask)", False: "return None"}
        site = "S._ask"
    else:
        bodies, site = _ROOT_BODIES, root
    first, last = (
        (bodies[False], bodies[True])
        if live_waits
        else (
            bodies[True],
            bodies[False],
        )
    )
    source = template.format(first=first, last=last)
    expected = [_row(root, site)] if live_waits else []
    assert _w003(source) == expected


_SCHEDULES_ASK = "self.call_later(self._ask)"
_ASK = "    async def _ask(self):\n        await self.app.push_screen_wait(Picker())\n"

#: A def that a later def of the SAME name does not make dead, because its
#: value outlives the rebinding, as ``(source, expected rows)``. The first
#: cut called every earlier same-named def dead, and lost rows dev reported
#: (PR checkpoint review: ``ServiceWiringMixin.llamacpp_snapshot_service``
#: is a getter with a setter, and schedules).
_STILL_BOUND = {
    # `@service.setter` reads `service`: the property keeps the getter.
    "property-getter": (
        "class S:\n"
        "    @property\n"
        f"    def service(self):\n        {_SCHEDULES_ASK}\n"
        "        return self._service\n\n"
        "    @service.setter\n"
        "    def service(self, value):\n        self._service = value\n\n" + _ASK,
        [_row("S.service->_ask", "S._ask")],
    ),
    "property-getter-setter-deleter": (
        "class S:\n"
        "    @property\n"
        f"    def service(self):\n        {_SCHEDULES_ASK}\n"
        "        return self._service\n\n"
        "    @service.setter\n"
        "    def service(self, value):\n        self._service = value\n\n"
        "    @service.deleter\n"
        "    def service(self):\n        self._service = None\n\n" + _ASK,
        [_row("S.service->_ask", "S._ask")],
    ),
    # A statement between the two put the first one somewhere first.
    "read-before-rebinding": (
        "async def on_ready(app):\n"
        "    await app.push_screen_wait(Picker())\n\n"
        "READY = {'ready': on_ready}\n\n"
        "async def on_ready(app):\n"
        "    return None\n",
        [_row("on_ready")],
    ),
    # Its OWN decorator may have handed it somewhere first: the package's
    # `@self.mcp.tool()` registers the function with a server, and the
    # registration outlives the name.
    "registered-by-its-own-decorator": (
        "@register\n"
        "async def on_ready(app):\n"
        "    await app.push_screen_wait(Picker())\n\n"
        "async def on_ready(app):\n"
        "    return None\n",
        [_row("on_ready")],
    ),
    # Negative controls: nothing outlives these rebindings.
    # `@overload` only describes a signature; the stub is never kept.
    "overload-stub": (
        "@overload\n"
        "async def on_ready(app):\n"
        "    await app.push_screen_wait(Picker())\n\n"
        "async def on_ready(app):\n"
        "    return None\n",
        [],
    ),
    # Textual collects `@on` handlers from the finished class namespace,
    # where only the last binding is left.
    "on-handler-rebound": (
        "class S:\n"
        "    @on(Button.Pressed)\n"
        "    async def on_pick(self, event):\n"
        "        await self.app.push_screen_wait(Picker())\n\n"
        "    @on(Button.Pressed)\n"
        "    async def on_pick(self, event):\n"
        "        return None\n",
        [],
    ),
    "getter-rebound-by-a-plain-def": (
        "class S:\n"
        "    @property\n"
        f"    def service(self):\n        {_SCHEDULES_ASK}\n"
        "        return self._service\n\n"
        "    def service(self):\n        return None\n\n" + _ASK,
        [],
    ),
    # The setter keeps the getter -- inside a property that is itself
    # rebound, so neither ever runs.
    "whole-property-rebound": (
        "class S:\n"
        "    @property\n"
        f"    def service(self):\n        {_SCHEDULES_ASK}\n"
        "        return self._service\n\n"
        "    @service.setter\n"
        "    def service(self, value):\n        self._service = value\n\n"
        "    def service(self):\n        return None\n\n" + _ASK,
        [],
    ),
    # The rebinding def's BODY reads the name only when called, and by then
    # the name is the rebinding def itself.
    "read-only-in-the-rebinding-body": (
        "async def on_ready(app):\n"
        "    await app.push_screen_wait(Picker())\n\n"
        "async def on_ready(app):\n"
        "    return on_ready\n",
        [],
    ),
}


@pytest.mark.parametrize("shape", sorted(_STILL_BOUND))
def test_w003_a_definition_whose_value_outlives_its_rebinding_stays_a_root(shape):
    """Dead means rebound before anything read it. A property getter is read
    by its own ``@x.setter``, and a def handed somewhere before the
    rebinding is still reachable there; both are roots."""
    source, expected = _STILL_BOUND[shape]
    assert _w003(source) == expected


#: The same name defined in ALTERNATIVE branches, as ``(template, root,
#: site)``: whichever branch runs binds it, so neither is dead.
_ALTERNATIVES = {
    "class-if-else": (
        "class S:\n"
        "    if WIDE:\n"
        "        async def action_pick(self):\n            {first}\n"
        "    else:\n"
        "        async def action_pick(self):\n            {second}\n",
        "S.action_pick",
        "S.action_pick",
    ),
    "module-try-except": (
        "try:\n"
        "    import fast\n\n"
        "    async def on_ready(app):\n        {first}\n"
        "except ImportError:\n"
        "    async def on_ready(app):\n        {second}\n",
        "on_ready",
        "on_ready",
    ),
    # SchedulesWorkbench._reminder_owner_action's shape: an early-return
    # branch and the fall-through each define `_do`.
    "nested-early-return": (
        "class S:\n"
        "    def _go(self, action):\n"
        "        if action == 'a':\n"
        "            def _do():\n                {first}\n"
        "            self.run_worker(_do, thread=True)\n"
        "            return\n"
        "        def _do():\n            {second}\n"
        "        self.run_worker(_do, thread=True)\n\n" + _ASK,
        "_do->_ask",
        "S._ask",
    ),
}


@pytest.mark.parametrize("scope", sorted(_ALTERNATIVES))
@pytest.mark.parametrize(
    "first_waits", [True, False], ids=["first-branch-waits", "second-branch-waits"]
)
def test_w003_definitions_in_alternative_branches_are_all_roots(scope, first_waits):
    """``if``/``else`` and ``try``/``except`` alternatives are not a
    redefinition: the branch that runs binds the name. Calling the earlier
    one dead dropped a waiting handler that dev reported."""
    template, root, site = _ALTERNATIVES[scope]
    waits = _SCHEDULES_ASK if scope == "nested-early-return" else _ROOT_BODIES[True]
    first, second = (waits, "return None") if first_waits else ("return None", waits)
    assert _w003(template.format(first=first, second=second)) == [_row(root, site)]


#: A def nested in another, whose ENCLOSING def is defined twice, as
#: ``(template, root, site)``. ``{first}``/``{last}`` are the two bodies of
#: the enclosing def; whatever the dead one defines never exists.
_NESTED_IN_TWICE = {
    "nested-def": (
        "class S:\n"
        "    def build(self):\n        {first}\n"
        "    def build(self):\n        {last}\n\n" + _ASK,
        "on_click->_ask",
        "S._ask",
    ),
    "class-in-a-function": (
        "def build():\n    {first}\ndef build():\n    {last}\n",
        "Pane.on_click",
        "Pane.on_click",
    ),
}

_NESTED_BODIES = {
    "nested-def": (
        "def on_click():\n            self.call_later(self._ask)\n"
        "        return on_click\n"
    ),
    "class-in-a-function": (
        "class Pane:\n"
        "        async def on_click(self):\n"
        "            await self.app.push_screen_wait(Picker())\n"
        "    return Pane\n"
    ),
}


@pytest.mark.parametrize("scope", sorted(_NESTED_IN_TWICE))
@pytest.mark.parametrize(
    "in_last", [True, False], ids=["in-the-bound-def", "in-the-rebound-def"]
)
def test_w003_a_definition_nested_in_a_dead_one_is_dead_too(scope, in_last):
    """A dead def's body never runs, so nothing it defines -- a nested def,
    a class and its methods -- is ever a root."""
    template, root, site = _NESTED_IN_TWICE[scope]
    body = _NESTED_BODIES[scope]
    first, last = ("return None", body) if in_last else (body, "return None")
    expected = [_row(root, site)] if in_last else []
    assert _w003(template.format(first=first, last=last)) == expected


#: What a def defined twice DOES in its body, as ``(template, body that
#: hands the waiting ``_pick`` on, body that does not, root, site)``. Only
#: the live def's body ever runs.
_DEAD_BODY_EFFECTS = {
    # `self.choose = self._pick`: a class-bound attribute waits when ANY of
    # its bindings does, so a dead binding alone would make it wait.
    "self-attribute-binding": (
        "class S:\n"
        "    def _wire(self):\n        {first}\n"
        "    def _wire(self):\n        {last}\n\n"
        "    async def on_button_pressed(self, event):\n"
        "        await self.choose()\n\n"
        "    async def _pick(self):\n"
        "        await self.app.push_screen_wait(Picker())\n",
        "self.choose = self._pick",
        "self.choose = self._noop",
        "S.on_button_pressed",
        "S._pick",
    ),
    # A future stored on `self` by a callback push, awaited by the handler.
    "self-future-publisher": (
        "class S:\n"
        "    def _open(self):\n        {first}\n"
        "    def _open(self):\n        {last}\n\n"
        "    async def on_button_pressed(self, event):\n"
        "        self._open()\n"
        "        await self._answer\n",
        "self._answer = asyncio.get_running_loop().create_future()\n"
        "        self.app.push_screen(Review(), callback=self._answer.set_result)",
        "self._answer = None",
        "S.on_button_pressed",
        "S._open",
    ),
    # A positional handoff binds `run`'s parameter, which `run` schedules.
    "positional-handoff": (
        "def run(pick):\n"
        "    app.call_later(pick)\n\n"
        "class S:\n"
        "    def _wire(self):\n        {first}\n"
        "    def _wire(self):\n        {last}\n\n"
        "    async def _pick(self):\n"
        "        await self.app.push_screen_wait(Picker())\n",
        "run(self._pick)",
        "return None",
        "run->pick",
        "S._pick",
    ),
    # A class-body binding in a class defined inside the def: both `build`s
    # define a `Pane`, and one class name is one class to W003.
    "class-body-binding": (
        "async def _pick():\n"
        "    await app.push_screen_wait(Picker())\n\n"
        "def build():\n"
        "    class Pane:\n        {first}\n\n"
        "        async def on_click(self):\n"
        "            await self.choose()\n"
        "    return Pane\n\n"
        "def build():\n"
        "    class Pane:\n        {last}\n\n"
        "        async def on_click(self):\n"
        "            await self.choose()\n"
        "    return Pane\n",
        "choose = _pick",
        "choose = None",
        "Pane.on_click",
        "_pick",
    ),
    # Rebound inside an `else:` -- a statement list of its own.
    "in-an-else-branch": (
        "class S:\n"
        "    if NARROW:\n"
        "        pass\n"
        "    else:\n"
        "        def _start(self):\n            {first}\n"
        "        def _start(self):\n            {last}\n\n"
        "    async def _pick(self):\n"
        "        await self.app.push_screen_wait(Picker())\n",
        "self.call_later(self._pick)",
        "return None",
        "S._start->_pick",
        "S._pick",
    ),
}


@pytest.mark.parametrize("effect", sorted(_DEAD_BODY_EFFECTS))
@pytest.mark.parametrize(
    "in_last", [True, False], ids=["in-the-bound-def", "in-the-rebound-def"]
)
def test_w003_what_a_dead_definition_does_never_happens(effect, in_last):
    """A dead def's body never runs: an attribute it binds, a future it
    stores on ``self`` and a callable it schedules are all absent -- and
    present when the live def does the same."""
    template, hands_on, inert, root, site = _DEAD_BODY_EFFECTS[effect]
    first, last = (inert, hands_on) if in_last else (hands_on, inert)
    expected = [_row(root, site)] if in_last else []
    assert _w003(template.format(first=first, last=last)) == expected


# `push_screen_wait` by hand -- PR #2922's request_hook_review until
# TASK-33621.28 made it await the modal's own answer.
_HAND_ROLLED = """
async def request_review(screen):
    answer = asyncio.get_running_loop().create_future()

    def done(result):
        if not answer.done():
            answer.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer


async def confirm_by_event(screen):
    decided = asyncio.Event()
    screen.app.push_screen(Review(), lambda _: decided.set())
    await decided.wait()
"""


@pytest.mark.parametrize("helper", ["request_review", "confirm_by_event"])
def test_w003_flags_a_hand_rolled_wait_from_a_handler(helper):
    """No NoActiveWorker here -- but Textual queues `done` on the requester
    pump via `call_next`, and from a handler that pump is the one blocked on
    `await answer`. The Console's Send froze exactly this way."""
    source = f"""
class S:
    async def on_button_pressed(self, event):
        await {helper}(self)
"""
    assert _w003(_HAND_ROLLED + source) == [_row("S.on_button_pressed", helper)]


@pytest.mark.parametrize(
    "body",
    [
        # A separate task: the scheduling pump is free to run the callback.
        "def on_mount(self):\n        asyncio.create_task(request_review(self))",
        # A worker: likewise.
        "@work\n    async def action_review(self):\n        await request_review(self)",
    ],
    ids=["create-task", "worker"],
)
def test_w003_a_hand_rolled_wait_off_the_pump_is_not_flagged(body):
    """In a separate task or a worker the requester pump is free to run the
    callback that completes the future, so the wait cannot deadlock."""
    source = f"""
class S:
    {body}
"""
    assert _w003(_HAND_ROLLED + source) == []


def test_w003_a_callback_push_that_does_not_await_its_future_is_not_a_wait():
    """A future handed to a callback push and stored, not awaited, leaves the
    handler free to return, so its pump can run the callback."""
    source = """
class S:
    async def on_button_pressed(self, event):
        answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=answer.set_result)
        self._pending = answer
"""
    assert _w003(source) == []


def test_w003_a_push_with_no_callback_is_not_a_hand_rolled_wait():
    """``callback=None`` is no callback: Textual queues nothing for the pump,
    so the push is not the hand-rolled wait, whatever the function awaits.
    Only a real callback beside it counts."""
    source = """
class S:
    async def on_button_pressed(self, event):
        answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Notice(), callback=None)
        self.app.push_screen(Review(), None)
        await answer
"""
    assert _w003(source) == []
    with_callback = source.replace(
        "        await answer",
        "        self.app.push_screen(Review(), callback=answer.set_result)\n"
        "        await answer",
    )
    assert _w003(with_callback) == [_row("S.on_button_pressed")]


def test_w003_a_wait_push_and_a_hand_rolled_push_in_one_function_are_two_rows():
    """A ``push_screen_wait`` and a hand-rolled wait in one function are two
    pushes, so two rows: a new hand-rolled push beside a censused wait push
    is a new row, not hidden behind the one already pinned."""
    source = """
class S:
    async def on_button_pressed(self, event):
        answer = asyncio.get_running_loop().create_future()
        await self.app.push_screen_wait(Picker())
        self.app.push_screen(Review(), callback=answer.set_result)
        await answer
"""
    assert _w003(source) == [_row("S.on_button_pressed")] * 2


def test_w003_a_wait_for_dismiss_push_with_a_callback_is_one_row():
    """``push_screen(..., callback=..., wait_for_dismiss=True)`` is ONE push:
    a wait push, never also a hand-rolled one, even in a function that
    awaits the future its callback settles."""
    source = """
class S:
    async def on_button_pressed(self, event):
        answer = asyncio.get_running_loop().create_future()
        await self.app.push_screen(
            Review(), callback=answer.set_result, wait_for_dismiss=True
        )
        await answer
"""
    assert _w003(source) == [_row("S.on_button_pressed")]


# The hand-rolled wait split across two functions, as ``(split, joined,
# push site)``: ``split`` creates the future and pushes in a helper and
# awaits it in the handler; ``joined`` is the same wait in ONE function.
_SPLIT_WAITS = {
    "returned-future": (
        """
def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return answer


class S:
    async def on_button_pressed(self, event):
        await open_review(self)
""",
        """
async def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return await answer


class S:
    async def on_button_pressed(self, event):
        await open_review(self)
""",
        "open_review",
    ),
    "future-on-self": (
        """
class S:
    def _open_review(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)

    async def on_button_pressed(self, event):
        self._open_review()
        await self._answer
""",
        """
class S:
    async def _open_review(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)
        await self._answer

    async def on_button_pressed(self, event):
        await self._open_review()
""",
        "S._open_review",
    ),
    "returned-future-bound-to-a-local": (
        """
def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return answer


class S:
    async def on_button_pressed(self, event):
        pending = open_review(self)
        await pending
""",
        """
async def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return await answer


class S:
    async def on_button_pressed(self, event):
        await open_review(self)
""",
        "open_review",
    ),
    "returned-future-under-wait-for": (
        """
def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return answer


class S:
    async def on_button_pressed(self, event):
        await asyncio.wait_for(open_review(self), 30)
""",
        """
async def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return await asyncio.wait_for(answer, 30)


class S:
    async def on_button_pressed(self, event):
        await open_review(self)
""",
        "open_review",
    ),
    "returned-event": (
        """
def open_review(screen):
    decided = asyncio.Event()
    screen.app.push_screen(Review(), lambda _: decided.set())
    return decided


class S:
    async def on_button_pressed(self, event):
        await open_review(self).wait()
""",
        """
async def open_review(screen):
    decided = asyncio.Event()
    screen.app.push_screen(Review(), lambda _: decided.set())
    await decided.wait()


class S:
    async def on_button_pressed(self, event):
        await open_review(self)
""",
        "open_review",
    ),
    "event-on-self": (
        """
class S:
    def _open_review(self):
        self._decided = asyncio.Event()
        self.app.push_screen(Review(), lambda _: self._decided.set())

    async def on_button_pressed(self, event):
        self._open_review()
        await self._decided.wait()
""",
        """
class S:
    async def _open_review(self):
        self._decided = asyncio.Event()
        self.app.push_screen(Review(), lambda _: self._decided.set())
        await self._decided.wait()

    async def on_button_pressed(self, event):
        await self._open_review()
""",
        "S._open_review",
    ),
    # The helper lives on a base class (or a mixin): `self` in the handler
    # is the same object, so the future it stored is the one awaited.
    "future-on-self-from-a-base-class": (
        """
class Base:
    def _open_review(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)


class S(Base):
    async def on_button_pressed(self, event):
        self._open_review()
        await self._answer
""",
        """
class Base:
    async def _open_review(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)
        await self._answer


class S(Base):
    async def on_button_pressed(self, event):
        await self._open_review()
""",
        "Base._open_review",
    ),
}


@pytest.mark.parametrize("shape", sorted(_SPLIT_WAITS))
def test_w003_a_hand_rolled_wait_split_across_functions_is_flagged(shape):
    """A helper creates the future, pushes with ``callback=`` and hands the
    future back -- returned, or stored on ``self`` -- and the handler awaits
    it. That is the hand-rolled deadlock across two functions, and it is
    reported at the same key as the one-function wait. Until TASK-33621.33
    this was a strict xfail: W003 recognized the shape only inside ONE
    function."""
    split, joined, site = _SPLIT_WAITS[shape]
    expected = [_row("S.on_button_pressed", site)]
    # Precondition: the same wait in ONE function is flagged, at the same
    # key, so `split` differs only in the cross-function link.
    assert _w003(joined) == expected, "precondition: the one-function wait"
    assert _w003(split) == expected, shape


#: Who receives ``open_review``'s pending future, and whether that blocks a
#: pump on it. Only the handler awaiting it does.
_SPLIT_RECEIVERS = {
    "awaited-by-a-handler": (
        "async def on_button_pressed(self, event):\n        await open_review(self)",
        True,
    ),
    "stored-not-awaited": (
        "async def on_button_pressed(self, event):\n"
        "        self._pending = open_review(self)",
        False,
    ),
    "awaited-by-a-worker": (
        "@work\n    async def action_review(self):\n        await open_review(self)",
        False,
    ),
    # A separate task: the scheduling pump is free to run the callback.
    "awaited-in-a-created-task": (
        "def on_mount(self):\n        asyncio.create_task(self._wait())\n\n"
        "    async def _wait(self):\n        await open_review(self)",
        False,
    ),
}


@pytest.mark.parametrize("receiver", sorted(_SPLIT_RECEIVERS))
def test_w003_a_handed_back_future_waits_only_where_a_pump_awaits_it(receiver):
    """Returning the pending future is not itself the freeze: storing it, or
    awaiting it in a worker or a separate task, leaves the requester pump
    free to run the callback. Only a pump-run await of it deadlocks."""
    body, flagged = _SPLIT_RECEIVERS[receiver]
    source = f"""
def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return answer


class S:
    {body}
"""
    expected = [_row("S.on_button_pressed", "open_review")] if flagged else []
    assert _w003(source) == expected


def test_w003_a_future_on_self_is_followed_only_within_selfs_classes():
    """``self._answer`` in ``S`` is ``S``'s (or a base's, or a subclass's)
    attribute: an unrelated class that pushes and stores its own
    ``self._answer`` is a different object, and must not make ``S`` wait."""
    source = """
class Elsewhere:
    def _open_review(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)


class S:
    def _open_review(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)

    async def on_button_pressed(self, event):
        self._open_review()
        await self._answer


class Unrelated:
    async def on_button_pressed(self, event):
        await self._answer
"""
    assert _w003(source) == [_row("S.on_button_pressed", "S._open_review")]


#: An ASYNC helper that pushes and returns its pending future or event, and
#: how a handler receives it. ``await open_review(self)`` runs the coroutine
#: and gets the future back -- it does not wait for the modal. Only awaiting
#: that RESULT does.
_ASYNC_HANDBACK = {
    "future": """
async def open_review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    return answer
""",
    "event": """
async def open_review(screen):
    decided = asyncio.Event()
    screen.app.push_screen(Review(), lambda _: decided.set())
    return decided
""",
}

_ASYNC_HANDBACK_RECEIVERS = {
    "awaits-the-coroutine": ("await open_review(self)", False),
    "keeps-the-result": ("self._pending = await open_review(self)", False),
    "awaits-the-result": ("await (await open_review(self))", True),
    "awaits-the-result-via-a-local": (
        "pending = await open_review(self)\n        await pending",
        True,
    ),
    "awaits-the-result-under-wait-for": (
        "await asyncio.wait_for(await open_review(self), 30)",
        True,
    ),
    "waits-on-the-result": ("await (await open_review(self)).wait()", True),
}


#: A future is awaited itself; an event only through ``.wait()``.
_ASYNC_HANDBACK_CASES = [
    (handed, receiver)
    for handed in sorted(_ASYNC_HANDBACK)
    for receiver in sorted(_ASYNC_HANDBACK_RECEIVERS)
    if receiver in ("awaits-the-coroutine", "keeps-the-result")
    or (handed == "event") == (receiver == "waits-on-the-result")
]


@pytest.mark.parametrize("handed, receiver", _ASYNC_HANDBACK_CASES)
def test_w003_awaiting_an_async_helper_obtains_its_pending_future_only(
    handed, receiver
):
    """``await open_review(self)`` on an ``async def`` that RETURNS its pending
    future runs the coroutine to its ``return``: the handler holds the
    future and returns, so its pump is free to run the callback. Counted as
    a wait, that was a false row; awaiting the result -- ``await (await
    open_review(self))``, a local, ``wait_for``, ``.wait()`` on an event --
    is the wait (PR #2987 review). Textual's ``invoke`` awaits a handler's
    return value once, which for an ``async def`` is the coroutine."""
    body, flagged = _ASYNC_HANDBACK_RECEIVERS[receiver]
    source = _ASYNC_HANDBACK[handed] + f"""

class S:
    async def on_button_pressed(self, event):
        {body}
"""
    expected = [_row("S.on_button_pressed", "open_review")] if flagged else []
    assert _w003(source) == expected


#: An ``async def`` that hands on ANOTHER async helper's pending future.
_ASYNC_RELAYS = {
    "returns-the-awaited-call": "return await open_review(screen)",
    "returns-a-local-of-it": "pending = await open_review(screen)\n    return pending",
    "relays-a-relay": "return await relay_once(screen)",
}


@pytest.mark.parametrize("relay", sorted(_ASYNC_RELAYS))
@pytest.mark.parametrize(
    "receiver, flagged",
    [
        ("await relay(self)", False),
        ("await (await relay(self))", True),
    ],
    ids=["awaits-the-relay", "awaits-the-result"],
)
def test_w003_an_async_relay_hands_back_the_future_it_obtained(
    relay, receiver, flagged
):
    """``return await open_review(screen)`` in an ``async def relay`` returns
    the PENDING future ``open_review`` handed back, unawaited: ``await
    relay(self)`` only obtains it, and awaiting that result is the wait --
    reported at ``open_review``, which holds the push. Before the async
    hand-back was modelled both forms were rows (one by accident); without
    following the relay, neither was."""
    source = (
        _ASYNC_HANDBACK["future"]
        + """

async def relay_once(screen):
    return await open_review(screen)


async def relay(screen):
    """
        + _ASYNC_RELAYS[relay]
        + f"""


class S:
    async def on_button_pressed(self, event):
        {receiver}
"""
    )
    expected = [_row("S.on_button_pressed", "open_review")] if flagged else []
    assert _w003(source) == expected


#: A getter that returns a future ANOTHER method created on ``self`` and
#: pushed with ``callback=``: (getter, how the handler receives it, waits).
_FUTURE_GETTERS = {
    "sync-getter-awaited": (
        "def _get_answer(self):\n        return self._answer",
        "await self._get_answer()",
        True,
    ),
    "sync-getter-kept": (
        "def _get_answer(self):\n        return self._answer",
        "self._kept = self._get_answer()",
        False,
    ),
    "async-getter-awaited-once": (
        "async def _get_answer(self):\n        return self._answer",
        "await self._get_answer()",
        False,
    ),
    "async-getter-result-awaited": (
        "async def _get_answer(self):\n        return self._answer",
        "await (await self._get_answer())",
        True,
    ),
}


@pytest.mark.parametrize("shape", sorted(_FUTURE_GETTERS))
def test_w003_a_getter_hands_back_the_future_its_publisher_stored(shape):
    """``self._open()`` stores the pending future on ``self`` and pushes;
    ``await self._get_answer()`` awaits it through a getter. The getter
    neither created it nor pushed, and the handler awaited no ``self``
    attribute, so the wait was invisible. A getter that returns ``self.X``
    hands back X's publishers' wait, exactly as awaiting ``self.X`` does
    (PR #2987 review)."""
    getter, receive, flagged = _FUTURE_GETTERS[shape]
    source = f"""
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)

    {getter}

    async def on_button_pressed(self, event):
        self._open()
        {receive}
"""
    expected = [_row("S.on_button_pressed", "S._open")] if flagged else []
    assert _w003(source) == expected


def test_w003_a_getter_of_an_unrelated_classs_future_does_not_wait():
    """The getter follows ``self.X`` only to the classes ``self`` can be."""
    source = """
class Elsewhere:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)


class S:
    def _get_answer(self):
        return self._answer

    async def on_button_pressed(self, event):
        await self._get_answer()
"""
    assert _w003(source) == []


#: The hand-rolled wait in one function, and split across two by returning
#: the future: (template, push site).
_SETTLING_FORMS = {
    "awaited-in-one-function": ("async def", "return await answer"),
    "returned-to-the-caller": ("def", "return answer"),
}

#: The nested defs a ``_settling_source`` callback may name.
_SETTLING_DEFS = {
    "done_answer": "\n    def done_answer(result):\n        answer.set_result(result)\n",
    "done_other": "\n    def done_other(result):\n        other.set_result(result)\n",
}


def _settling_source(callback: str, form: str) -> str:
    """``review`` creates ``answer`` and ``other``, pushes with ``callback``,
    settles ``answer`` itself and awaits or returns it (``form``); a handler
    awaits ``review``. ``other`` is completed by nothing the caller waits
    on."""
    keyword, finish = _SETTLING_FORMS[form]
    return f"""
def record(result):
    print(result)


{keyword} review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
{_SETTLING_DEFS.get(callback, "")}
    screen.app.push_screen(Review(), callback={callback})
    answer.set_result(None)
    {finish}


class S:
    async def on_button_pressed(self, event):
        await review(self)
"""


#: Callbacks that may settle the awaited or returned ``answer``: they settle
#: it, or nothing shows what they do.
_SETTLING_CALLBACKS = {
    "completer-of-the-awaited-future": "answer.set_result",
    "lambda-completing-the-awaited-future": "lambda result: answer.set_result(result)",
    "nested-def-completing-the-awaited-future": "done_answer",
    "callback-of-unknown-effect": "record",
    "lambda-of-unknown-effect": "lambda result: record(result)",
}


@pytest.mark.parametrize("form", sorted(_SETTLING_FORMS))
@pytest.mark.parametrize("callback", sorted(_SETTLING_CALLBACKS))
def test_w003_a_callback_push_that_may_settle_the_awaited_future_waits(form, callback):
    """A ``push_screen(..., callback=...)`` in a function that awaits or
    returns a future it created is a hand-rolled wait, whatever the
    callback is: the future's completer, a lambda, a nested def, or a
    callback whose effect nothing shows. W003 matches it by SHAPE, so a
    callback that settles only ANOTHER future counts too -- those cases are
    listed in ``_ACCEPTED_FALSE_POSITIVES``."""
    source = _settling_source(_SETTLING_CALLBACKS[callback], form)
    assert _w003(source) == [_row("S.on_button_pressed", "review")]


@pytest.mark.parametrize(
    "value",
    ["self._answer.set_result", "self._settle_answer"],
    ids=["completes-the-stored-future", "method-completing-the-stored-future"],
)
def test_w003_a_future_on_self_waits_on_a_push_that_may_complete_it(value):
    """The same for a future stored on ``self``: the handler awaiting it waits
    on ``_open``'s push. A callback settling ANOTHER stored future counts
    too (``_ACCEPTED_FALSE_POSITIVES``)."""
    source = f"""
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback={value})

    def _settle_answer(self, result):
        self._answer.set_result(result)

    def _settle_other(self, result):
        self._other.set_result(result)

    async def on_button_pressed(self, event):
        self._open()
        await self._answer
"""
    assert _w003(source) == [_row("S.on_button_pressed", "S._open")]


# Every shape from here to the end of this section is a hand-rolled wait
# shape that 5918cfd1df reports. Each one but the last accepted false
# positive -- a mutation pin that no version dropped -- lost a push to some
# version of a "leave this push out" rule during the PR #2987 review (rounds
# 2-6, 2026-10-03/04): a recall pin as a missed wait, an accepted false
# positive as one of the precision controls the rule was for. The rule
# left a callback push out when its callback seemed to settle a future other
# than the awaited one: first by comparing names, then by proving what the
# callback's body does, then by also requiring that nothing else could read
# the settled future, then only for a direct settle (`callback=
# other.set_result`, a one-call lambda, a `partial` of it), and last only
# for such a settle of a local future nothing else read. Each version
# missed a way the awaited future can depend on the callback running. The
# real-tree census stayed byte-identical every time, because the rule never
# dropped a real push. The rule is gone: every callback push counts by shape,
# and `_ACCEPTED_FALSE_POSITIVES` lists what that costs. The recall pins
# stay, so any future precision rule has to keep every one of them.

_AWAIT_REVIEW = """

class S:
    async def on_button_pressed(self, event):
        await review(self)
"""

_AWAIT_STORED = """

    async def on_button_pressed(self, event):
        self._open()
        await self._answer
"""

#: (source, row): a callback that settles the awaited future under a name
#: the pushing function never bound to a fresh future of its own.
_SETTLED_UNDER_ANOTHER_NAME = {
    "partial-of-a-module-helper": (
        """
def settle(fut, result):
    fut.set_result(result)


async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=partial(settle, answer))
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "own-method-through-a-local": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        future = self._answer
        future.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "nested-def-through-a-local": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()

    def done(result):
        fut = answer
        fut.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "pusher-local-of-the-stored-future": (
        """
class S:
    async def on_button_pressed(self, event):
        self._answer = asyncio.get_running_loop().create_future()
        answer = self._answer
        self.app.push_screen(Review(), callback=answer.set_result)
        await self._answer
""",
        ("S.on_button_pressed", None),
    ),
    "nested-def-settling-a-pusher-alias": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    pending = answer

    def done(result):
        pending.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "own-method-settling-a-future-made-elsewhere": (
        """
class S:
    def __init__(self):
        self._closed = asyncio.Event()

    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._closed.set()
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    # Same-named, different binding: the callback's `other` is not the
    # pushing function's `other`.
    "module-helper-parameter-named-like-another-future": (
        """
def settle(other, result):
    other.set_result(result)


async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=partial(settle, answer))
    other.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "nested-def-local-named-like-another-future": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done(result):
        other = answer
        other.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "nested-def-parameter-named-like-another-future": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done(result, other=answer):
        other.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "lambda-parameter-named-like-another-future": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(
        Review(), callback=lambda result, other=answer: other.set_result(result)
    )
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "own-method-local-named-like-another-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        other = self._answer
        other.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    # The pushing function's own name, but not only ever a fresh future.
    "rebound-to-the-awaited-future": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    other = answer
    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "one-future-under-two-names": (
        """
async def review(screen):
    answer = other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    # The next four were pinned in round 4: a mutation could drop any of
    # their premises with every other case still green.
    "rebound-by-a-walrus-in-a-nested-defs-default": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def link(_=(other := answer)):
        return _

    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "rebound-by-a-nested-class-of-the-same-name": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    class other:
        set_result = answer.set_result

    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "one-future-under-two-names-the-other-way-round": (
        """
async def review(screen):
    other = answer = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-parameter-only-maybe-rebound-to-a-fresh-future": (
        """
async def review(screen, flag, closed):
    answer = asyncio.get_running_loop().create_future()
    screen.pending = answer
    if flag:
        closed = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=closed.set_result)
    return await answer


class S:
    async def on_button_pressed(self, event):
        closed = asyncio.get_running_loop().create_future()
        closed.add_done_callback(lambda f: self.pending.set_result(f.result()))
        await review(self, self.flag, closed)
""",
        ("S.on_button_pressed", "review"),
    ),
    "rebound-by-a-nested-nonlocal": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def swap():
        nonlocal other
        other = answer

    swap()
    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "rebound-by-a-nonlocal-in-a-nested-class": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    class _Link:
        nonlocal other
        other = answer

    screen.app.push_screen(Review(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "pusher-rebinding-self": (
        """
class S:
    async def on_button_pressed(self, event):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self = self._twin
        self.app.push_screen(Review(), callback=self._other.set_result)
        await self._answer
""",
        ("S.on_button_pressed", None),
    ),
    # The callback is not what it looks like.
    "callback-name-rebound-to-another-def": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done(result):
        other.set_result(result)

    def finish(result):
        answer.set_result(result)

    done = finish
    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "own-method-rebinding-self": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        self = self._twin
        self._other.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
}


@pytest.mark.parametrize("shape", sorted(_SETTLED_UNDER_ANOTHER_NAME))
def test_w003_a_callback_settling_the_future_under_another_name_still_waits(shape):
    """A name the callback settles is the pushing function's future only in
    the pushing function's own scope, and only when that name is only ever
    bound to one fresh future: a helper's parameter, a callback's local or
    default, a pusher-side alias, a rebinding (a nested def's ``nonlocal``
    or a rebound ``self`` included), a chained assignment, or a parameter
    only conditionally rebound can all be the awaited future (or chained to
    it). Each is a row on 5918cfd1df. The round-2 cases were silent on
    c2e4b15fe7; a ``nonlocal`` in a nested class was silent at 4c87eb4d3e;
    the four round-4 pins were rows there with no case covering them."""
    source, (root, site) = _SETTLED_UNDER_ANOTHER_NAME[shape]
    assert _w003(source) == [_row(root, site)]


#: (source, row): a callback that visibly settles a future the pushing
#: function did not create, or another one it did, but ALSO does something
#: that may settle the awaited one -- a call W003 does not follow, or a
#: subclass override.
_SETTLES_MORE_THAN_IT_SHOWS = {
    "closes-an-event-then-calls-a-resolver": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._closed.set()
        self._resolve(result)

    def _resolve(self, result):
        self._answer.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "cancels-a-timer-then-calls-a-finisher": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._timer.cancel()
        self._finish(result)

    def _finish(self, result):
        self._answer.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "settles-its-own-event-then-calls-a-resolver": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._closed.set()
        self._resolve(result)

    def _resolve(self, result):
        self._answer.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "settles-another-future-then-calls-a-nested-finisher": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def finish(result):
        answer.set_result(result)

    def done(result):
        other.set_result(result)
        finish(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-lambda-whose-argument-calls-a-finisher": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def finish(result):
        answer.set_result(result)
        return result

    screen.app.push_screen(
        Review(), callback=lambda result: other.set_result(finish(result))
    )
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-subclass-override-settles-the-awaited-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED
        + """

class Sub(S):
    def _settle(self, result):
        self._answer.set_result(result)
""",
        ("S.on_button_pressed", "S._open"),
    ),
}


@pytest.mark.parametrize("shape", sorted(_SETTLES_MORE_THAN_IT_SHOWS))
def test_w003_a_callback_that_may_also_settle_the_awaited_future_still_waits(shape):
    """Settling one future visibly proves nothing about the rest of the
    callback: a call it makes, or the override ``self`` may dispatch to,
    can settle the awaited future too. Each of these was a row on
    5918cfd1df and silent on c2e4b15fe7."""
    source, (root, site) = _SETTLES_MORE_THAN_IT_SHOWS[shape]
    assert _w003(source) == [_row(root, site)]


#: (source, rows): ACCEPTED FALSE POSITIVES -- the precision gap PR #2987's
#: "unrelated callback" thread reported, and why W003 keeps it. Each callback
#: settles only a future the awaited or returned one cannot depend on, so the
#: push cannot complete it and the row is false. Five versions of a rule that
#: left such pushes out each dropped waits 5918cfd1df reports (the shapes
#: pinned above and below: a chain through a done-callback or a relay, a
#: store, a property or a watcher, a settle's answer, a frame, a closure
#: cell), and none of them ever dropped a push on the real tree. So every
#: callback push counts by shape and these rows are expected: a false row
#: costs one census line that a reviewer can read and annotate, while a
#: missed wait is a frozen UI. Each case was a precision control at some
#: head of that review or in its reviewers' shape runs, except the last,
#: which pins that a publisher's site counts every callback push it makes.
_ACCEPTED_FALSE_POSITIVES = {
    # The thread's own shape (returned to the caller) and its one-function
    # form, for each kind of callback that settles only `other`.
    **{
        f"{callback}-{form}": (
            _settling_source(value, form),
            [_row("S.on_button_pressed", "review")],
        )
        for callback, value in {
            "completer-of-another-future": "other.set_result",
            "lambda-completing-another-future": "lambda result: other.set_result(result)",
            "nested-def-completing-another-future": "done_other",
        }.items()
        for form in _SETTLING_FORMS
    },
    # Two pushes in one site, only one of which can settle the awaited
    # future: the row is counted twice.
    "two-pushes-one-settling-another-future": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=answer.set_result)
    screen.app.push_screen(Notice(), callback=other.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")] * 2,
    ),
    "another-future-beside-a-lambda-settling-the-awaited-one": (
        """
class S:
    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        answer = loop.create_future()
        other = loop.create_future()
        self.app.push_screen(Review(), callback=other.set_result)
        self.app.push_screen(Notice(), callback=lambda r, other=answer: other.set_result(r))
        await answer
""",
        [_row("S.on_button_pressed")] * 2,
    ),
    "the-completer-of-a-local-future-in-a-handler": (
        """
class S:
    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        answer = loop.create_future()
        other = loop.create_future()
        self.app.push_screen(Review(), callback=other.set_result)
        await answer
""",
        [_row("S.on_button_pressed")],
    ),
    "an-event-a-lambda-sets": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.Event()
    screen.app.push_screen(Review(), callback=lambda _: closed.set())
    answer.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "an-event-a-lambda-sets-beside-an-awaited-event": (
        """
async def review(screen):
    answer = asyncio.Event()
    other = asyncio.Event()
    screen.app.push_screen(Review(), callback=lambda _: other.set())
    answer.set()
    await answer.wait()
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "a-lambda-settling-it-with-a-constant": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=lambda _: other.set_result(None))
    answer.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "another-future-made-fresh-on-either-branch": (
        """
async def review(screen, flag):
    answer = asyncio.get_running_loop().create_future()
    if flag:
        other = asyncio.get_running_loop().create_future()
    else:
        other = asyncio.Future()
    screen.app.push_screen(Review(), callback=other.set_result)
    answer.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "settled-again-on-the-way-out": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=other.set_result)
    answer.set_result(None)
    try:
        return await answer
    finally:
        other.cancel()
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "an-event-set-again-on-the-way-out": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.Event()
    screen.app.push_screen(Review(), callback=lambda _: closed.set())
    answer.set_result(None)
    try:
        return await answer
    finally:
        closed.set()
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "cancelled-again-in-a-nested-def": (
        """
class S:
    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        answer = loop.create_future()
        other = loop.create_future()
        self.app.push_screen(Review(), callback=other.set_result)

        def stop():
            other.cancel()

        answer.add_done_callback(lambda _: stop())
        await answer
""",
        [_row("S.on_button_pressed")],
    ),
    "a-partial-of-its-completer": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=partial(other.set_result))
    answer.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "a-nested-def-inspecting-the-future-it-settles": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done(result):
        if not other.done():
            other.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    answer.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    "a-lambda-forwarding-to-a-nested-def": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done_other(result):
        other.set_result(result)

    screen.app.push_screen(Review(), callback=lambda result: done_other(result))
    answer.set_result(None)
    return await answer
"""
        + _AWAIT_REVIEW,
        [_row("S.on_button_pressed", "review")],
    ),
    # Futures stored on `self`.
    "a-method-settling-another-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle_other)

    def _settle_other(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")],
    ),
    "an-inherited-method": (
        """
class Base:
    def _settle(self, result):
        self._other.set_result(result)


class S(Base):
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")],
    ),
    "a-method-inspecting-the-future-it-settles": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self._answer.set_result(None)
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        if not self._other.done():
            self._other.set_result(result)
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")],
    ),
    "the-completer-of-another-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._other.set_result)
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")],
    ),
    "a-lambda-settling-another-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=lambda _: self._other.set_result(None))
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")],
    ),
    "a-stored-event-a-lambda-sets": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        self._answer.set_result(None)
        self.app.push_screen(Review(), callback=lambda _: self._closed.set())
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")],
    ),
    # A publisher's site counts every callback push it makes, as an own
    # site does.
    "two-pushes-in-a-publisher-one-settling-another-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._answer.set_result)
        self.app.push_screen(Notice(), callback=self._other.set_result)
"""
        + _AWAIT_STORED,
        [_row("S.on_button_pressed", "S._open")] * 2,
    ),
}


@pytest.mark.parametrize("shape", sorted(_ACCEPTED_FALSE_POSITIVES))
def test_w003_a_callback_push_counts_even_when_it_settles_only_another_future(shape):
    """Accepted false positives: a callback that settles only a future the
    awaited one cannot depend on -- directly (``other.set_result``, a
    lambda, ``partial``), through a nested def, an own or inherited method,
    or on ``self`` -- still makes its push a wait row. The finding is
    accurate as a precision gap; it is kept on purpose. Five versions of a
    rule leaving these out each lost waits 5918cfd1df reports, and none
    dropped a push on the real tree. Not reporting one of these again needs
    a reviewed design that keeps every recall case in this section, not a
    quieter rule."""
    source, rows = _ACCEPTED_FALSE_POSITIVES[shape]
    assert _w003(source) == rows


# Settling a DIFFERENT future does not make the push unrelated to the
# awaited one. Textual runs the callback through the requester's
# `call_next`, so while the handler awaits, nothing the callback would
# settle is ever settled -- and any awaited future that is chained to one
# of those futures hangs too. 511b3ddd49 asked only whether the settled
# name was bound fresh, never whether that future ESCAPES: is read anywhere
# other than as the receiver of a settle call. Each shape in the next table
# is a row on 5918cfd1df and was silent on 511b3ddd49 (PR #2987 review,
# round 3).

#: (source, row): the settled future feeds the awaited one.
_SETTLED_FUTURE_FEEDS_THE_AWAITED_ONE = {
    "a-done-callback-chains-it": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.get_running_loop().create_future()
    closed.add_done_callback(lambda f: answer.set_result(f.result()))
    screen.app.push_screen(Review(), callback=closed.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-relay-task-awaits-it": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.Event()

    async def relay():
        await closed.wait()
        answer.set_result(True)

    asyncio.create_task(relay())
    screen.app.push_screen(Review(), callback=lambda _: closed.set())
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-relay-task-polls-it": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.get_running_loop().create_future()

    async def relay():
        while not closed.done():
            await asyncio.sleep(0.05)
        answer.set_result(closed.result())

    asyncio.create_task(relay())
    screen.app.push_screen(Review(), callback=closed.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "it-is-handed-on": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.get_running_loop().create_future()
    screen.watch_closed(closed, answer)
    screen.app.push_screen(Review(), callback=closed.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "every-local-is-handed-on": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.get_running_loop().create_future()
    screen.chain(**locals())
    screen.app.push_screen(Review(), callback=closed.set_result)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-done-callback-chains-the-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.get_running_loop().create_future()
        self._closed.add_done_callback(self._relay)
        self.app.push_screen(Review(), callback=self._closed.set_result)

    def _relay(self, closed):
        self._answer.set_result(closed.result())
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-relay-method-awaits-the-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        asyncio.create_task(self._relay())
        self.app.push_screen(Review(), callback=lambda _: self._closed.set())

    async def _relay(self):
        await self._closed.wait()
        self._answer.set_result(True)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-timer-polls-the-stored-future": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.get_running_loop().create_future()
        self.set_interval(0.05, self._poll)
        self.app.push_screen(Review(), callback=self._closed.set_result)

    def _poll(self):
        if self._closed.done() and not self._answer.done():
            self._answer.set_result(self._closed.result())
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
}


@pytest.mark.parametrize("shape", sorted(_SETTLED_FUTURE_FEEDS_THE_AWAITED_ONE))
def test_w003_a_callback_settling_a_future_the_awaited_one_hangs_on_still_waits(
    shape,
):
    """A done-callback, a relay that awaits or polls it, a handoff, or
    ``locals()`` can chain the settled future to the awaited one, and then
    the await hangs exactly as if the callback settled it directly."""
    source, (root, site) = _SETTLED_FUTURE_FEEDS_THE_AWAITED_ONE[shape]
    assert _w003(source) == [_row(root, site)]


#: (source, row): the shapes the round-3 proof needed premises for -- "this
#: name is only ever that fresh future", "this callback is that def or
#: method". Counting every callback push by shape needs neither. These stay
#: pinned.
_SETTLED_ATTRIBUTE_REBOUND_OR_CALLBACK_INDIRECT = {
    "a-nested-def-rebinds-the-settled-attribute": (
        """
class S:
    async def on_button_pressed(self, event):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()

        def link():
            self._other = self._answer

        link()
        self.app.push_screen(Review(), callback=self._other.set_result)
        await self._answer
""",
        ("S.on_button_pressed", None),
    ),
    "another-method-rebinds-the-settled-attribute": (
        """
class S:
    async def on_button_pressed(self, event):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self._link()
        self.app.push_screen(Review(), callback=self._other.set_result)
        await self._answer

    def _link(self):
        self._other = self._answer
""",
        ("S.on_button_pressed", None),
    ),
    "another-method-rebinds-what-the-callback-method-settles": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self._link()
        self.app.push_screen(Review(), callback=self._settle)

    def _link(self):
        self._other = self._answer

    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-decorated-nested-callback": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    @also_settle(answer)
    def done(result):
        other.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-decorated-callback-method": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    @also_settle("_answer")
    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "an-instance-attribute-shadows-the-callback-method": (
        """
class S:
    def __init__(self):
        self._settle = self._finish

    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        self._other.set_result(result)

    def _finish(self, result):
        self._answer.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-built-instance-attribute-shadows-the-callback-method": (
        """
class S:
    def __init__(self):
        self._settle = self._settler()

    def _settler(self):
        return lambda result: self._answer.set_result(result)

    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-subclass-body-rebinds-the-callback-method": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED
        + """

class Sub(S):
    _settle = make_settler("_answer")
""",
        ("S.on_button_pressed", "S._open"),
    ),
    "a-lambda-shadows-self": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(
            Review(), callback=lambda result, self=self._peer: self._settle(result)
        )

    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-nested-def-may-not-rebind-the-callback-parameter": (
        """
async def review(screen, flag, done=None):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    if flag:
        def done(result):
            other.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer


class S:
    async def on_button_pressed(self, event):
        await review(self, self.flag)
""",
        ("S.on_button_pressed", "review"),
    ),
    # Pinned in round 4 (each was a row at 4c87eb4d3e too, but no case
    # covered it).
    "a-lambda-shadows-self-and-settles-directly": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(
            Review(),
            callback=lambda result, self=self._peer: self._other.set_result(result),
        )
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-subclass-body-imports-over-the-callback-method": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result):
        self._other.set_result(result)
"""
        + _AWAIT_STORED
        + """

class Sub(S):
    from helpers import settle_answer as _settle
""",
        ("S.on_button_pressed", "S._open"),
    ),
    "a-class-body-inside-the-callback": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done(result):
        other.set_result(result)

        class _Settle:
            answer.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
}


@pytest.mark.parametrize(
    "shape", sorted(_SETTLED_ATTRIBUTE_REBOUND_OR_CALLBACK_INDIRECT)
)
def test_w003_a_rebound_settled_attribute_or_an_indirect_callback_still_waits(shape):
    """A nested def or another method can rebind a settled ``self``
    attribute, and a lambda's default can rebind ``self``. Every other
    shape here is a def or method callback that is not what it looks like:
    decorated, shadowed by an instance attribute, a subclass body's
    assignment or import, only maybe bound, or with a class body inside.
    Each is a row on 5918cfd1df."""
    source, (root, site) = _SETTLED_ATTRIBUTE_REBOUND_OR_CALLBACK_INDIRECT[shape]
    assert _w003(source) == [_row(root, site)]


# The round-3 proof read only a callback's CALLS ("every call settles or
# inspects what it settles") and took that for "settling is all it does".
# A callback whose settled future nothing reads can matter ONLY through its
# other effects -- and a nonlocal or item store a relay polls, an attribute
# store a watcher or a property setter acts on, an await, a `with` block or
# a returned inspection are effects with no call of their own. Each shape
# in the next table is a row on 5918cfd1df and was silent at 4c87eb4d3e
# (PR #2987 review, round 4); its six nested-def shapes are rows on
# origin/dev too.

#: (source, row): a callback that settles another future AND does more.
_CALLBACK_EFFECTS_BEYOND_SETTLING = {
    "a-nonlocal-store-a-relay-polls": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.Event()
    choice = None

    def done(result):
        nonlocal choice
        choice = result
        closed.set()

    async def relay():
        while choice is None:
            await asyncio.sleep(0.05)
        answer.set_result(choice)

    asyncio.create_task(relay())
    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "an-item-store-a-relay-polls": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.Event()
    state = {}

    def done(result):
        state["choice"] = result
        closed.set()

    async def relay():
        while "choice" not in state:
            await asyncio.sleep(0.05)
        answer.set_result(state["choice"])

    asyncio.create_task(relay())
    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "an-attribute-store-a-relay-polls": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        self._choice = None
        asyncio.create_task(self._relay())
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._choice = result
        self._closed.set()

    async def _relay(self):
        while self._choice is None:
            await asyncio.sleep(0.05)
        self._answer.set_result(self._choice)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-reactive-store-whose-watcher-settles-the-awaited-future": (
        """
class S:
    choice = reactive(None)

    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._closed.set()
        self.choice = result

    def watch_choice(self, choice):
        self._answer.set_result(choice)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-property-store-whose-setter-settles-the-awaited-future": (
        """
class S:
    @property
    def choice(self):
        return self._choice

    @choice.setter
    def choice(self, value):
        self._choice = value
        self._answer.set_result(value)

    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.Event()
        self.app.push_screen(Review(), callback=self._on_review)

    def _on_review(self, result):
        self._closed.set()
        self.choice = result
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    "a-store-on-a-handed-on-object": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()

    def done(result):
        other.set_result(result)
        screen.choice = result

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "an-await-of-what-settles-the-awaited-future": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    finish = screen.finish(answer)

    async def done(result):
        other.set_result(result)
        await finish

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-with-block-around-the-settle": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    settling = screen.settling(answer)

    def done(result):
        with settling:
            other.set_result(result)

    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-nested-def-returning-an-inspection-a-relay-polls": (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.get_running_loop().create_future()

    def done(result=None):
        if result is not None:
            closed.set_result(result)
        return closed.done()

    async def relay():
        while not done():
            await asyncio.sleep(0.05)
        answer.set_result(True)

    asyncio.create_task(relay())
    screen.app.push_screen(Review(), callback=done)
    return await answer
"""
        + _AWAIT_REVIEW,
        ("S.on_button_pressed", "review"),
    ),
    "a-method-returning-an-inspection-a-relay-polls": (
        """
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.get_running_loop().create_future()
        asyncio.create_task(self._relay())
        self.app.push_screen(Review(), callback=self._settle)

    def _settle(self, result=None):
        if result is not None:
            self._closed.set_result(result)
        return self._closed.done()

    async def _relay(self):
        while not self._settle():
            await asyncio.sleep(0.05)
        self._answer.set_result(True)
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
    # A lambda's argument is part of its body: a property getter runs when
    # the callback does.
    "a-lambda-whose-argument-reads-a-property": (
        """
class S:
    @property
    def choice(self):
        self._answer.set_result(True)
        return self._choice

    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._other = asyncio.get_running_loop().create_future()
        self.app.push_screen(
            Review(), callback=lambda _: self._other.set_result(self.choice)
        )
"""
        + _AWAIT_STORED,
        ("S.on_button_pressed", "S._open"),
    ),
}


@pytest.mark.parametrize("shape", sorted(_CALLBACK_EFFECTS_BEYOND_SETTLING))
def test_w003_a_callback_that_settles_another_future_and_does_more_still_waits(
    shape,
):
    """A callback can settle a future nothing reads and still complete the
    awaited one through any other effect -- a store, an await, a ``with``
    block, a returned value, a property read -- none of which is a call, so
    a proof that read only the callback's calls missed every one."""
    source, (root, site) = _CALLBACK_EFFECTS_BEYOND_SETTLING[shape]
    assert _w003(source) == [_row(root, site)]


#: (source, row): the settled future handed on by name at run time, which no
#: load of the name shows. Each is a row on 5918cfd1df; ``eval``, ``exec``,
#: ``__self__``, a frame's ``f_locals`` and ``self.__dict__`` were silent at
#: 4c87eb4d3e (``vars()`` was a row there, but no case pinned it). Round 5
#: (review of a106783a85): a scope reader called through a module
#: (``builtins.eval``) or fetched by a constant name, and a frame of the
#: pushing function reached without ``f_locals`` -- ``getargvalues`` of
#: ``currentframe()``, ``sys._getframe()``, ``inspect.stack()``/``trace()``,
#: ``traceback.walk_stack``, ``sys.exc_info()``, an exception's
#: ``__traceback__``, a frame's ``f_back`` or a coroutine's ``cr_frame`` --
#: were silent at a106783a85.
_DYNAMIC_HAND_ONS = {
    "vars": "screen.chain(**vars())",
    "locals": "screen.chain(**locals())",
    "eval": 'screen.chain(eval("closed"), answer)',
    "exec": 'exec("screen.chain(closed, answer)")',
    "a-bound-settle-methods-self": "screen.chain(closed.set_result.__self__, answer)",
    "a-frames-locals": "screen.chain(**sys._getframe().f_locals)",
    "a-callers-frame-handed-back": (
        'screen.chain(screen.caller_frame().f_locals["closed"], answer)'
    ),
    "builtins-eval": 'screen.chain(builtins.eval("closed"), answer)',
    "eval-by-a-constant-name": (
        'screen.chain(getattr(builtins, "eval")("closed"), answer)'
    ),
    "currentframe": (
        "screen.chain(inspect.getargvalues(inspect.currentframe()), answer)"
    ),
    "getframe": "screen.chain(inspect.getargvalues(sys._getframe()), answer)",
    "inspect-stack": "screen.chain(inspect.stack(), answer)",
    "inspect-trace": "screen.chain(inspect.trace(), answer)",
    "walk-stack": "screen.chain(traceback.walk_stack(None), answer)",
    "exc-info": "screen.chain(sys.exc_info(), answer)",
    "an-exceptions-traceback": (
        "try:\n        raise RuntimeError\n    except RuntimeError as error:\n"
        "        screen.chain(error.__traceback__, answer)"
    ),
    "a-callee-frames-f-back": "screen.chain(screen.frame().f_back, answer)",
    "a-coroutines-frame": (
        "screen.chain(asyncio.current_task().get_coro().cr_frame, answer)"
    ),
    "a-frame-attribute-by-a-constant-name": (
        'screen.chain(getattr(screen.frame(), "f_back"), answer)'
    ),
}


@pytest.mark.parametrize("shape", sorted(_DYNAMIC_HAND_ONS))
def test_w003_a_settled_future_handed_on_dynamically_still_waits(shape):
    """``vars()``/``locals()``, ``eval``/``exec`` (however reached), a frame
    of the pushing function (however reached) and a bound settle method's
    ``__self__`` hand the settled future on without a plain load of its
    name: the push still counts."""
    source = (
        f"""
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    closed = asyncio.get_running_loop().create_future()
    {_DYNAMIC_HAND_ONS[shape]}
    screen.app.push_screen(Review(), callback=closed.set_result)
    return await answer
"""
        + _AWAIT_REVIEW
    )
    assert _w003(source) == [_row("S.on_button_pressed", "review")]


# Round 6 (PR #2987 review of f606e0c3ef). The last version of the rule left
# out only a direct settle of a LOCAL future that nothing else read, counting
# a `cancel()`/`set()` statement as no read, and naming the introspection
# that hands on every local. It still missed these. A nested def or
# generator holds the future in a closure cell, which `__closure__`,
# `inspect.getclosurevars` or a generator's locals read without loading the
# name; and `asyncio.current_task().get_stack()` or `sys._current_frames()`
# reach the pusher's frame through APIs the list did not name. Each shape in
# the next table but one, and the Event shape after it, is a row on
# origin/dev and 5918cfd1df and was silent at a106783a85 and f606e0c3ef.
# The one, `a-closure-cell-of-a-lambda-cancel`, is a control: silent at
# a106783a85 too, it was already a row at f606e0c3ef, which exempted a
# `cancel()` STATEMENT only, and a lambda's body is an expression.

_PUSH_SETTLING_OTHER = """
class S:
    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        answer = loop.create_future()
        other = loop.create_future()
        self.app.push_screen(Review(), callback=other.set_result)
"""

#: The handler's lines after it pushes with ``callback=other.set_result``,
#: before it awaits ``answer``.
_CLOSURE_AND_FRAME_HAND_ONS = {
    "a-closure-cell-of-a-nested-cancel": """
        def stop():
            other.cancel()

        chain(stop.__closure__[0].cell_contents, answer)
""",
    "inspect-getclosurevars": """
        def stop():
            other.cancel()

        chain(inspect.getclosurevars(stop).nonlocals["other"], answer)
""",
    "a-closure-read-by-a-constant-name": """
        def stop():
            other.cancel()

        chain(getattr(stop, "__closure__")[0].cell_contents, answer)
""",
    "a-nested-generators-locals": """
        def stop():
            other.cancel()
            yield

        gen = stop()
        next(gen)
        chain(inspect.getgeneratorlocals(gen)["other"], answer)
""",
    "a-closure-cell-of-a-lambda-cancel": """
        stop = lambda: other.cancel()
        chain(stop.__closure__[0].cell_contents, answer)
""",
    "a-tasks-stack-frame": """
        frame = asyncio.current_task().get_stack()[0]
        chain(inspect.getargvalues(frame).locals["other"], answer)
""",
    "the-threads-current-frame": """
        frame = sys._current_frames()[threading.get_ident()]
        chain(inspect.getargvalues(frame).locals["other"], answer)
""",
    "a-tasks-stack-handed-to-a-relay": """
        chain(asyncio.current_task().get_stack(), answer)
""",
}


@pytest.mark.parametrize("shape", sorted(_CLOSURE_AND_FRAME_HAND_ONS))
def test_w003_a_settled_future_reached_through_a_closure_or_a_frame_still_waits(
    shape,
):
    """A closure cell or a frame hands the settled future on with no load of
    its name, so the awaited future can be chained to it: the push counts,
    as every callback push does."""
    source = (
        _PUSH_SETTLING_OTHER
        + _CLOSURE_AND_FRAME_HAND_ONS[shape]
        + "        await answer\n"
    )
    assert _w003(source) == [_row("S.on_button_pressed")]


def test_w003_a_settled_event_reached_through_a_closure_still_waits():
    """The same through an Event: a lambda pushed as the callback sets
    ``closed``, a nested def sets it again, and the nested def's closure
    cell hands ``closed`` to a relay."""
    source = """
class S:
    async def on_button_pressed(self, event):
        answer = asyncio.get_running_loop().create_future()
        closed = asyncio.Event()
        self.app.push_screen(Review(), callback=lambda _: closed.set())

        def finish():
            closed.set()

        relay(finish.__closure__[0].cell_contents, answer)
        await answer
"""
    assert _w003(source) == [_row("S.on_button_pressed")]


#: (hand-on in the pushing function, hand-on in another method).
_SELF_DYNAMIC_HAND_ONS = {
    "getattr-by-a-constant-name-in-another-method": (
        "",
        'screen.chain(getattr(self, "_closed"), self._answer)',
    ),
    "the-instance-dict-in-the-pushing-function": (
        "self.screen.chain(self.__dict__)",
        "",
    ),
    "vars-of-self-in-the-pushing-function": ("self.screen.chain(vars(self))", ""),
    "a-bound-settle-methods-self-in-another-method": (
        "",
        "screen.chain(self._closed.set_result.__self__, self._answer)",
    ),
}


@pytest.mark.parametrize("shape", sorted(_SELF_DYNAMIC_HAND_ONS))
def test_w003_a_settled_self_future_handed_on_by_name_still_waits(shape):
    """The same for a ``self`` future, which hands on through ``getattr``
    with a constant name, a bound settle method's ``__self__``,
    ``self.__dict__`` or ``vars(self)``. All are rows on 5918cfd1df; all
    but ``vars(self)`` were silent at 4c87eb4d3e."""
    in_pusher, elsewhere = _SELF_DYNAMIC_HAND_ONS[shape]
    source = (
        f"""
class S:
    def _open(self):
        self._answer = asyncio.get_running_loop().create_future()
        self._closed = asyncio.get_running_loop().create_future()
        {in_pusher or "pass"}
        self.app.push_screen(Review(), callback=self._closed.set_result)

    def _link(self, screen):
        {elsewhere or "pass"}
"""
        + _AWAIT_STORED
    )
    assert _w003(source) == [_row("S.on_button_pressed", "S._open")]


# Round 5 (PR #2987 review of a106783a85). A direct settle of a `self.X`
# future was left out whenever every `.X` attribute node in the package was
# one of the pusher's own fresh stores. But evaluating `self.X` -- the
# callback's receiver -- and storing to it run code that is no `.X` node
# anywhere: a property's getter or setter (`@property`, `property(...)`, one
# in a base class in another module), a reactive's watcher or validator, a
# `__setattr__` override; and a class-level `X` is what the callback settles
# whenever the instance store is skipped. Each shape below is a row on
# 5918cfd1df and on origin/dev and was silent at a106783a85.

_M1 = "tldw_chatbook/UI/m1.py"

#: (sources, row): a direct settle of a ``self`` future whose read or store
#: runs code.
_SELF_FUTURE_READ_OR_STORE_RUNS_CODE = {
    "a-property-getter-returns-the-awaited-future": (
        (
            """
class S:
    @property
    def other(self):
        return self._answer

    @other.setter
    def other(self, value):
        self._shadow = value

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.other = loop.create_future()
        self.app.push_screen(Review(), callback=self.other.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-property-setter-chains-the-stored-future": (
        (
            """
class S:
    @property
    def other(self):
        return self._backing

    @other.setter
    def other(self, fut):
        self._backing = fut
        fut.add_done_callback(lambda f: self._answer.set_result(f.result()))

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.other = loop.create_future()
        self.app.push_screen(Review(), callback=self.other.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-reactive-watcher-chains-the-stored-future": (
        (
            """
from textual.reactive import reactive


class S(Screen):
    pending = reactive(None)

    def watch_pending(self, fut):
        if fut is not None:
            fut.add_done_callback(lambda f: self._answer.set_result(f.result()))

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.pending = loop.create_future()
        self.app.push_screen(Review(), callback=self.pending.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-reactive-validator-swaps-in-the-awaited-future": (
        (
            """
from textual.reactive import reactive


class S(Screen):
    pending = reactive(None)

    def validate_pending(self, fut):
        return self._answer if fut is not None else None

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.pending = loop.create_future()
        self.app.push_screen(Review(), callback=self.pending.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-lambda-settling-through-a-property-getter": (
        (
            """
class S:
    @property
    def other(self):
        return self._answer

    @other.setter
    def other(self, value):
        self._shadow = value

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.other = loop.create_future()
        self.app.push_screen(Review(), callback=lambda r: self.other.set_result(r))
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-property-in-a-base-class-in-another-module": (
        (
            """
class Base:
    @property
    def other(self):
        return self._answer

    @other.setter
    def other(self, value):
        self._shadow = value
""",
            """
from m0 import Base


class S(Base):
    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.other = loop.create_future()
        self.app.push_screen(Review(), callback=self.other.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed", module=_M1),
    ),
    "a-property-built-by-a-call": (
        (
            """
class S:
    def _get(self):
        return self._answer

    def _set(self, value):
        self._shadow = value

    other = property(_get, _set)

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self.other = loop.create_future()
        self.app.push_screen(Review(), callback=self.other.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-setattr-override-chains-the-stored-future": (
        (
            """
class S:
    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if name == "_other":
            value.add_done_callback(lambda f: self._answer.set_result(f.result()))

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        self._other = loop.create_future()
        self.app.push_screen(Review(), callback=self._other.set_result)
        await self._answer
""",
        ),
        _row("S.on_button_pressed"),
    ),
    "a-class-level-future-behind-a-conditional-store": (
        (
            """
class S:
    _other = SHARED

    async def on_button_pressed(self, event):
        loop = asyncio.get_running_loop()
        self._answer = loop.create_future()
        if event.fresh:
            self._other = loop.create_future()
        self.app.push_screen(Review(), callback=self._other.set_result)
        await self._answer


def wire():
    SHARED.add_done_callback(lambda f: ANSWERS[0].set_result(f.result()))
""",
        ),
        _row("S.on_button_pressed"),
    ),
}


@pytest.mark.parametrize("shape", sorted(_SELF_FUTURE_READ_OR_STORE_RUNS_CODE))
def test_w003_a_direct_settle_of_a_self_future_still_waits(shape):
    """Reading ``self.X`` -- the callback's receiver -- or storing a fresh
    future there can run code that no ``.X`` attribute node shows: a
    property's getter or setter, a reactive's watcher or validator, a
    ``__setattr__`` override. And a class-level ``X`` is what the callback
    settles whenever the instance store is skipped."""
    sources, row = _SELF_FUTURE_READ_OR_STORE_RUNS_CODE[shape]
    assert _w003(*sources) == [row]


# Round 5, too: `fut.cancel()` returns whether the future was still pending,
# and `set_result`/`set_exception` raise InvalidStateError on a done one. A
# settle call elsewhere in the body whose outcome is used therefore reports
# whether the callback ran -- the same hand-on as a `done()` probe -- and the
# awaited future can depend on it. a106783a85 took every settle call's
# receiver for a use that hands nothing on. Each shape below is a row on
# 5918cfd1df and on origin/dev.

#: Timeout relays whose answer depends on the outcome of settling the future
#: the push's callback settles.
_SETTLE_OUTCOME_OBSERVED = {
    "the-result-of-cancel-picks-the-answer": """
        if other.cancel():
            answer.set_result(None)
        else:
            answer.set_result("chosen")
""",
    "the-result-of-cancel-handed-to-a-call": """
        answer.set_result(other.cancel())
""",
    "the-result-of-cancel-returned-by-a-lambda": """
        probe = lambda: other.cancel()
        answer.set_result(None if probe() else "chosen")
""",
    "the-error-set-exception-raises-picks-the-answer": """
        try:
            other.set_exception(TimeoutError())
        except asyncio.InvalidStateError:
            answer.set_result("chosen")
""",
    "the-error-set-result-raises-picks-the-answer": """
        try:
            other.set_result(None)
        except asyncio.InvalidStateError:
            answer.set_result("chosen")
""",
    # The probe's `try` need not be where the settle is: the error reaches
    # whoever calls it.
    "the-error-a-nested-settle-raises-picks-the-answer": """
        def probe():
            other.set_result(None)

        try:
            probe()
        except asyncio.InvalidStateError:
            answer.set_result("chosen")
""",
}


@pytest.mark.parametrize("shape", sorted(_SETTLE_OUTCOME_OBSERVED))
def test_w003_a_settle_whose_outcome_is_used_still_waits(shape):
    """A settle call outside the push whose result or error the body uses
    tells whether the callback ran, so the awaited future can depend on the
    callback running: the push counts."""
    source = (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=other.set_result)

    async def timeout():
        await asyncio.sleep(30)
"""
        + _SETTLE_OUTCOME_OBSERVED[shape]
        + """
    asyncio.create_task(timeout())
    return await answer
"""
        + _AWAIT_REVIEW
    )
    assert _w003(source) == [_row("S.on_button_pressed", "review")]


#: Callbacks built by a call that is NAMED ``partial`` -- the name is all
#: a106783a85 checked. Each is a row on 5918cfd1df and on origin/dev.
_A_CALL_NAMED_PARTIAL = {
    "a-module-function": """
def partial(settle):
    settle.__self__.add_done_callback(lambda f: ANSWERS[0].set_result(f.result()))
    return settle


async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    ANSWERS.append(answer)
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=partial(other.set_result))
    return await answer
""",
    "a-method": """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=screen.partial(other.set_result))
    return await answer
""",
}


@pytest.mark.parametrize("shape", sorted(_A_CALL_NAMED_PARTIAL))
def test_w003_a_partial_of_a_settle_counts_by_shape(shape):
    """A call that is only NAMED ``partial`` can hand ``fut`` on
    (``settle.__self__``) to whatever settles the awaited future."""
    assert _w003(_A_CALL_NAMED_PARTIAL[shape] + _AWAIT_REVIEW) == [
        _row("S.on_button_pressed", "review")
    ]


def test_w003_a_settle_through_a_computed_receiver_hands_the_future_on():
    """``screen.chain(other, answer).cancel()`` hands ``other`` to ``chain``
    before anything is cancelled. A row on 5918cfd1df and origin/dev, and
    at every head of the PR #2987 review from 7e1a1eccfc on; silent at
    c2e4b15fe7 and 511b3ddd49."""
    source = (
        """
async def review(screen):
    answer = asyncio.get_running_loop().create_future()
    other = asyncio.get_running_loop().create_future()
    screen.app.push_screen(Review(), callback=other.set_result)
    screen.chain(other, answer).cancel()
    return await answer
"""
        + _AWAIT_REVIEW
    )
    assert _w003(source) == [_row("S.on_button_pressed", "review")]


def test_w003_a_settle_handed_to_a_partial_hands_its_future_on():
    """Handing ``other.set_result`` to a call -- one named ``partial``
    included -- hands ``other`` on, and a second push that settles
    ``other`` directly is a second row. Both are rows on 5918cfd1df;
    neither was at a106783a85."""
    source = (
        _A_CALL_NAMED_PARTIAL["a-module-function"].replace(
            "    return await answer\n",
            "    screen.app.push_screen(Notice(), callback=other.set_result)\n"
            "    return await answer\n",
        )
        + _AWAIT_REVIEW
    )
    assert "Notice()" in source
    assert _w003(source) == [_row("S.on_button_pressed", "review")] * 2


#: (source, row): a package definition named like a future factory or like
#: ``push_screen`` that hands the "fresh" future, or the callback's, on to
#: the awaited one. Silent at every head of the PR #2987 review from
#: c2e4b15fe7 through f606e0c3ef, and pinned as strict-xfail known misses
#: only at f606e0c3ef (added in 59b64872be), whose rule left a direct
#: settle of a fresh local out and so had to trust those names. Counting
#: every callback push needs no such premise, so each is a row again, as on
#: 5918cfd1df and origin/dev.
_DEFINED_UNDER_A_FACTORY_OR_PUSH_NAME = {
    "a-package-create-future-returning-the-awaited-future": (
        """
class S:
    def create_future(self):
        return self._answer

    async def on_button_pressed(self, event):
        self._answer = asyncio.get_running_loop().create_future()
        other = self.create_future()
        self.app.push_screen(Review(), callback=other.set_result)
        await self._answer
""",
        ("S.on_button_pressed", None),
    ),
    "a-package-create-future-keeping-what-it-makes": (
        """
class S:
    def create_future(self):
        fut = asyncio.get_running_loop().create_future()
        self._tracked.append(fut)
        return fut

    async def _relay(self):
        for fut in self._tracked:
            self._answer.set_result(await fut)

    async def on_button_pressed(self, event):
        self._tracked = []
        self._answer = asyncio.get_running_loop().create_future()
        other = self.create_future()
        self.app.push_screen(Review(), callback=other.set_result)
        asyncio.create_task(self._relay())
        await self._answer
""",
        ("S.on_button_pressed", None),
    ),
    "a-package-push-screen-handing-its-callback-on": (
        """
class S:
    def push_screen(self, screen, callback=None):
        callback.__self__.add_done_callback(
            lambda f: self._answer.set_result(f.result())
        )
        self.app.push_screen(screen, callback=callback)

    async def on_button_pressed(self, event):
        self._answer = asyncio.get_running_loop().create_future()
        other = asyncio.get_running_loop().create_future()
        self.push_screen(Review(), callback=other.set_result)
        await self._answer
""",
        ("S.on_button_pressed", None),
    ),
}


@pytest.mark.parametrize("shape", sorted(_DEFINED_UNDER_A_FACTORY_OR_PUSH_NAME))
def test_w003_a_package_definition_under_a_factory_or_push_name_still_waits(shape):
    """A package ``create_future`` that hands back an existing or a kept
    future, and a package ``push_screen`` that hands its callback's future
    on, chain the settled future to the awaited one: the push counts."""
    source, (root, site) = _DEFINED_UNDER_A_FACTORY_OR_PUSH_NAME[shape]
    assert _w003(source) == [_row(root, site)]


# --------------------------------------------------------------------------
# Rows are keyed by entry point AND push site (TASK-33621.13 review): keyed
# by entry point alone, a censused ChatScreen.on_button_pressed let a brand
# new inline push_screen_wait in that handler pass W003 silently.
# --------------------------------------------------------------------------

_SEND = """
class S:
    async def on_button_pressed(self, event):
        await self._send()

    async def _send(self):
        await self._review()

    async def _review(self):
        await self.app.push_screen_wait(Review())
"""

_NEW_INLINE_PUSH = _SEND.replace(
    "        await self._send()\n",
    "        await self._send()\n        await self.app.push_screen_wait(Picker())\n",
)

_SECOND_PUSH_IN_THE_SITE = _SEND.replace(
    "        await self.app.push_screen_wait(Review())\n",
    "        await self.app.push_screen_wait(Review())\n"
    "        await self.app.push_screen_wait(Again())\n",
)


def test_w003_a_new_push_in_a_reaching_entry_point_is_a_new_key():
    """One row per push, keyed by entry point AND push site: a new push in a
    censused handler, or a second one in its site, adds a row."""
    assert _w003(_SEND) == [_row("S.on_button_pressed", "S._review")]
    assert _w003(_NEW_INLINE_PUSH) == sorted(
        [_row("S.on_button_pressed"), _row("S.on_button_pressed", "S._review")]
    )
    assert (
        _w003(_SECOND_PUSH_IN_THE_SITE)
        == [_row("S.on_button_pressed", "S._review")] * 2
    )


_SAMPLE_M = "tldw_chatbook/UI/sample.py"
_PINNED_SEND = (
    "# header\n"
    f"{_SAMPLE_M}::S.on_button_pressed => {_SAMPLE_M}::S._review\t1\t"
    "REAL freeze, reviewed: TASK-1; proof: Tests/UI/x.py\n"
)


@pytest.mark.parametrize(
    "source, new_key",
    [
        (
            _NEW_INLINE_PUSH,
            f"{_SAMPLE_M}::S.on_button_pressed => {_SAMPLE_M}::S.on_button_pressed",
        ),
        (
            _SECOND_PUSH_IN_THE_SITE,
            f"{_SAMPLE_M}::S.on_button_pressed => {_SAMPLE_M}::S._review",
        ),
    ],
    ids=["inline-in-the-entry-point", "second-push-in-the-site"],
)
def test_main_flags_a_new_push_reachable_from_an_already_censused_entry_point(
    monkeypatch, tmp_path, capsys, source, new_key
):
    """End to end: with the entry point already pinned, ``main`` still
    exits 1 on a new push it reaches, and names the new key."""
    assert _run_main_w003(monkeypatch, tmp_path, _SEND, _PINNED_SEND) == 0
    capsys.readouterr()
    (tmp_path / "tldw_chatbook" / "UI" / "sample.py").write_text(source)
    assert _mod.main() == 1
    out = capsys.readouterr().out
    assert "new wait-for-dismiss screen push" in out
    assert new_key in out


_CHAT = "tldw_chatbook/UI/Screens/chat_screen.py"


def _with_inline_push(tree: ast.Module, cls: str, method: str) -> ast.Module:
    """``tree`` with ``await self.app.push_screen_wait(Picker())`` prepended to
    the LIVE (last) definition of ``cls.method``."""
    push = (
        ast.parse("async def _():\n    await self.app.push_screen_wait(Picker())\n")
        .body[0]
        .body[0]
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == cls
    )
    live = [
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method
    ][-1]
    live.body.insert(0, push)
    return tree


@pytest.mark.parametrize(
    "method, new_key",
    [
        (
            "on_button_pressed",
            f"{_CHAT}::ChatScreen.on_button_pressed => "
            f"{_CHAT}::ChatScreen.on_button_pressed",
        ),
        (
            "on_console_workbench_action_requested",
            f"{_CHAT}::ChatScreen.on_console_workbench_action_requested => "
            f"{_CHAT}::ChatScreen.on_console_workbench_action_requested",
        ),
        (
            "_send_console_message_from_visible_action",
            f"{_CHAT}::ChatScreen.on_key->_send_console_message_from_visible_action"
            f" => {_CHAT}::ChatScreen._send_console_message_from_visible_action",
        ),
    ],
)
def test_a_new_push_in_a_console_send_dispatcher_is_flagged_on_the_real_tree(
    real_tree, method, new_key
):
    """The reviewer's reproduction, on the real tree against the real census:
    a new push in one of the Console's Send entry points must fail W003.
    Until TASK-33621.28 each of them also had a census row of its own (the
    hook-review Send freeze), which must not have hidden the new one."""
    known = _mod._read_census(_mod.WAIT_PUSH_CENSUS)
    clean = _mod._WaitGraph(list(real_tree.values())).roots()
    assert _mod._added(known, _mod._tally(clean)) == [], "real tree drifted"

    mutated = dict(real_tree)
    tree = _with_inline_push(_parse_file(_mod.REPO_ROOT / _CHAT), "ChatScreen", method)
    mutated[_CHAT] = _mod._collect_module(tree, _CHAT)
    rows = _mod._WaitGraph(list(mutated.values())).roots()
    assert new_key in _mod._added(known, _mod._tally(rows))


_SPEECH_MIXIN = "tldw_chatbook/UI/Speech/speech_settings_mixin.py"
_SPEECH_PANE = "tldw_chatbook/UI/Speech/speech_settings_pane.py"


def test_a_new_mixin_action_reaching_its_host_pane_push_is_flagged_on_the_real_tree(
    real_tree,
):
    """The round-3 reviewer's reproduction: a new ``SpeechSettingsMixin``
    action awaiting ``self._ask_leave_choice()`` -- defined only by
    ``SpeechSettingsPane``, the class that mixes it in -- was flagged at
    0bc03b4715 and passed at 2ebe5b1a7e. Exactly one new row: the unrelated
    ``SpeechTTSSettingsPanel._ask_leave_choice`` is not reached by name."""
    known = _mod._read_census(_mod.WAIT_PUSH_CENSUS)
    clean = _mod._WaitGraph(list(real_tree.values())).roots()
    assert _mod._added(known, _mod._tally(clean)) == [], "real tree drifted"

    tree = _parse_file(_mod.REPO_ROOT / _SPEECH_MIXIN)
    mixin = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SpeechSettingsMixin"
    )
    assert not _mod._class_methods(mixin).get("_ask_leave_choice"), (
        "precondition: the mixin does not define the method itself"
    )
    mixin.body.append(
        ast.parse(
            "async def action_probe_leave(self):\n    await self._ask_leave_choice()\n"
        ).body[0]
    )
    mutated = dict(real_tree)
    mutated[_SPEECH_MIXIN] = _mod._collect_module(tree, _SPEECH_MIXIN)
    rows = _mod._WaitGraph(list(mutated.values())).roots()
    assert _mod._added(known, _mod._tally(rows)) == [
        f"{_SPEECH_MIXIN}::SpeechSettingsMixin.action_probe_leave => "
        f"{_SPEECH_PANE}::SpeechSettingsPane._ask_leave_choice"
    ]


def test_w003_real_inspector_recovery_still_resolves_as_waiting(real_tree):
    """Regression pin on the REAL tree: the chain from the Inspector's worker
    coroutine through `project_instruction_context_kwargs`'s `partial` to the
    session controller's `push_screen_wait` still resolves -- so if anyone
    awaits that coroutine from the handler again, W003 fails."""
    graph = _mod._WaitGraph(list(real_tree.values()))
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
    assert not [
        row
        for row in graph.roots()
        if row.startswith(
            f"{inspector}::ConsoleConversationInspector._recover_project_instructions"
        )
    ]


#: dev immediately before TASK-33621.13: the Inspector's RecoveryRequested
#: handler still awaited the folder picker (GAP4-01).
_GAP4_BASE = "dfee4bf4c66ec2656a6c4ea2667edb63cc38a445"


def test_w003_flags_the_original_inspector_freeze_on_the_merge_base_tree():
    """Negative control on the tree that actually froze: whatever W003's
    resolution rules become, the GAP4-01 chain -- handler, `partial`, dict
    key, constructor parameter, `self.` attribute, session controller -- must
    still be reported, at its real push site."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not available")
    present = subprocess.run(
        [git, "-C", str(_mod.REPO_ROOT), "cat-file", "-e", f"{_GAP4_BASE}^{{commit}}"],
        capture_output=True,
    )
    if present.returncode != 0:
        pytest.skip(f"{_GAP4_BASE[:10]} is not in this clone (shallow checkout?)")
    archive = subprocess.run(
        [
            git,
            "-C",
            str(_mod.REPO_ROOT),
            "archive",
            "--format=tar",
            _GAP4_BASE,
            "--",
            ":(glob)tldw_chatbook/**/*.py",
        ],
        capture_output=True,
        check=True,
    ).stdout
    collected = []
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar:
            if not member.isfile() or _mod.SKIP_PARTS & set(member.name.split("/")):
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SyntaxWarning)
                    tree = ast.parse(tar.extractfile(member).read().decode("utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            collected.append(_mod._collect_module(tree, member.name))
    rows = _mod._WaitGraph(collected).roots()
    assert (
        "tldw_chatbook/Widgets/Console/console_conversation_inspector.py::"
        "ConsoleConversationInspector._recover_project_instructions => "
        "tldw_chatbook/UI/Console_Modules/session.py::"
        "ConsoleSessionController._select_project_instruction_binding"
    ) in rows


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

_S_PICK = f"{_SAMPLE_M}::S.action_pick => {_SAMPLE_M}::S.action_pick"
_T_PICK = f"{_SAMPLE_M}::T.action_pick => {_SAMPLE_M}::T.action_pick"


def test_main_exits_nonzero_on_an_uncensused_w003_root(monkeypatch, tmp_path, capsys):
    """End-to-end negative control: the whole script, bad input, exit 1."""
    assert _run_main_w003(monkeypatch, tmp_path, _W003_HANDLER, "# empty\n") == 1
    out = capsys.readouterr().out
    assert "wait-for-dismiss screen push" in out
    assert _S_PICK in out


def test_main_exits_zero_when_that_w003_root_is_pinned(monkeypatch, tmp_path):
    """Positive control for the test above: the same root, pinned with its
    count, passes."""
    census = f"# header\n{_S_PICK}\t1\n"
    assert _run_main_w003(monkeypatch, tmp_path, _W003_HANDLER, census) == 0


_UNRESOLVED_KEY = (
    f"{_SAMPLE_M}::S.on_button_pressed -> external#0 => {_SAMPLE_M}::S._pick"
)


def test_main_fails_on_a_new_unresolved_positional_handoff_and_accepts_a_pinned_one(
    monkeypatch, tmp_path, capsys
):
    """End to end: an unresolved positional handoff of a waiting callable is
    a W003 census row -- new, it fails ``main``; pinned, it passes."""
    source = _UNRESOLVED_HANDOFFS["external-callee"][0]
    assert _run_main_w003(monkeypatch, tmp_path, source, "# empty\n") == 1
    out = capsys.readouterr().out
    assert _UNRESOLVED_KEY in out
    assert "positional argument" in out
    (tmp_path / "wait_census.tsv").write_text(f"# header\n{_UNRESOLVED_KEY}\t1\n")
    assert _mod.main() == 0


# T's row counts 2: two wait pushes in one live handler. (It once counted a
# handler defined twice, but the first definition is dead code and no
# longer a root -- TASK-33621.33.)
_W003_TWO_ROOTS = """
class S:
    async def action_pick(self):
        await self.app.push_screen_wait(Picker())

class T:
    async def action_pick(self):
        await self.app.push_screen_wait(Picker())
        await self.app.push_screen_wait(Picker())
"""

_NOTE = "REAL freeze, reviewed: proof in Tests/UI/x.py; follow-up: fix the pump"


def test_main_reads_the_count_of_a_row_that_carries_a_review_note(
    monkeypatch, tmp_path
):
    """A reviewed row's third column is its note. Reading ``2\\t<note>`` as a
    malformed count collapsed it to 1, so pinning a noted row with two
    occurrences still failed as "1 in census, 2 now"."""
    census = f"# header\n{_S_PICK}\t1\n{_T_PICK}\t2\t{_NOTE}\n"
    assert _run_main_w003(monkeypatch, tmp_path, _W003_TWO_ROOTS, census) == 0


def test_write_carries_a_review_note_forward_and_drops_a_resolved_rows(
    monkeypatch, tmp_path
):
    """``--write`` regenerates the census from the tree; a review note must
    survive that for as long as its row does, or every re-pin silently erases
    the verdict and the follow-up it names."""
    resolved = (
        "tldw_chatbook/UI/gone.py::G.action_gone => tldw_chatbook/UI/gone.py::G.x"
    )
    census = (
        "# stale header\n"
        f"{_T_PICK}\t2\t{_NOTE}\n"
        f"{resolved}\t1\tREAL freeze, since fixed\n"
    )
    _run_main_w003(monkeypatch, tmp_path, _W003_TWO_ROOTS, census)
    monkeypatch.setattr("sys.argv", ["check_textual_worker_contract.py", "--write"])
    assert _mod.main() == 0
    rows = [
        line
        for line in (tmp_path / "wait_census.tsv").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert rows == [f"{_S_PICK}\t1", f"{_T_PICK}\t2\t{_NOTE}"]
    monkeypatch.setattr("sys.argv", ["check_textual_worker_contract.py"])
    assert _mod.main() == 0


def test_write_carries_a_w002_review_note_forward_too(monkeypatch, tmp_path):
    """Both censuses share one reader, which accepts a third (note) column --
    so W002's ``--write`` must carry it as W003's does, not drop it."""
    note = "reviewed: the await cannot remove #target; TASK-2"
    census = f"# header\n{_SAMPLE_M}::run\t1\t{note}\n"
    assert _run_main(monkeypatch, tmp_path, _IN_FINALLY, census) == 0
    monkeypatch.setattr(_mod, "WAIT_PUSH_CENSUS", tmp_path / "wait_census.tsv")
    monkeypatch.setattr("sys.argv", ["check_textual_worker_contract.py", "--write"])
    assert _mod.main() == 0
    rows = [
        line
        for line in (tmp_path / "census.tsv").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert rows == [f"{_SAMPLE_M}::run\t1\t{note}"]
    monkeypatch.setattr("sys.argv", ["check_textual_worker_contract.py"])
    assert _mod.main() == 0
