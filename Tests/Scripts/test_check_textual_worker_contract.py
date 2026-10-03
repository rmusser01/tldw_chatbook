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
