"""A bypassed-``__init__`` ChatScreen shell that sets its store must wire EVERY
controller the store setter reaches.

`ChatScreen.__init__` installs its sub-controllers via
`Console_Modules/wiring.build_console_controllers`, so a
`ChatScreen.__new__(ChatScreen)` shell has none of them. That stays invisible
until the shell touches a seam that reaches one — and one of the most ordinary
lines a Console test writes does. `screen._console_chat_store = store` is a
property whose setter calls `self._console_runtime()`, which
`ensure_console_runtime` answers by calling `ConsoleRuntime.attach_view`, which
calls `ChatScreen.console_view_hooks()` — and that method reads its controllers
by attribute, unguarded.

The result is a test that dies while being *set up*, with an `AttributeError`
naming an attribute the test file never mentions. TASK-21381 found 115 of them
across 8 files; TASK-23144 found 46 more.

**Why this file was rewritten (TASK-23144).** Until then this guard was named
`test_bare_chat_screen_shells_wire_the_fleet.py` and asked one question: "does
this shell call `stub_fleet_controller`?" `_fleet` was the only controller that
hook build read *at the time it was written*. When PR #2154 added a second one
(`self._library_activity.build_provider`) the guard stayed green while 46 tests
died in setup, because a hard-coded controller name cannot see a controller
being added next to it. So the question is now derived, not spelled:

    Which controllers does `screen._console_chat_store = ...` ACTUALLY read?

`controllers_the_store_setter_reads()` answers that by *performing the
assignment* on a bare shell and collecting the `AttributeError`s it raises, one
per round, until it succeeds. No production function is named here, no AST
pattern has to keep up with how the read is spelled, and a controller added to
that path in future appears in the derived set the moment it lands. The only
hand-written thing left is `CONTROLLER_STUBS`, the slot -> stub-helper mapping
(a person has to say which helper builds which controller), and
`test_every_controller_the_store_setter_reads_has_a_stub` holds it to
set-equality against the derived set in both directions — so a new controller
fails HERE, naming itself, instead of in dozens of unrelated tests.

There are two ways for a shell to satisfy the ratchet. Wire every controller in
`CONTROLLER_STUBS` from `Tests/UI/console_controller_stubs`, whose raiser
defaults keep the shell fail-loud at any seam it has not wired; or hand the
shell its own `_console_runtime_ref`, which `_console_runtime()` returns
verbatim so the attach — and therefore the hook build — never happens. The
guard accepts either, because the invariant is "do not let the store setter
reach a runtime this shell cannot satisfy", not "call these helpers".

**Either way it must happen first.** The assignment is not a note the setter
reads later; it *is* the attach, reading the controllers as it runs. So a
fixture that assigns the store and stubs afterwards dies in setup exactly like
one that never stubbed at all, and a scan that merely asks "does this function
mention the helper?" passes it. `_runs_before` is what closes that: wiring
counts only where it provably precedes every assignment in the function.
"""

from __future__ import annotations

import ast
import re
import textwrap
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

TESTS_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TESTS_ROOT.parent

#: Controller slot -> the helper in `Tests/UI/console_controller_stubs` that
#: builds one. HAND-WRITTEN, because only a person knows which helper builds
#: which controller -- but never hand-maintained as a *set*:
#: `test_every_controller_the_store_setter_reads_has_a_stub` holds these keys
#: to set-equality with `controllers_the_store_setter_reads()`, so production
#: growing (or shedding) a controller on that path fails here.
CONTROLLER_STUBS: dict[str, str] = {
    "_fleet": "stub_fleet_controller",
    "_library_activity": "stub_library_activity_controller",
}

#: Functions allowed to build a store-setting shell without wiring the
#: controllers. Shrink-only: an entry may be removed when its function is
#: fixed, never added. Empty on purpose -- every known case was repaired in
#: TASK-21381 and TASK-23144.
ALLOWLIST: frozenset[str] = frozenset()

#: This module's own two store-setting shells, which are the guard rather
#: than fixtures under it. `controllers_the_store_setter_reads` must fail the
#: assignment (that is how it derives the set), and the positive control
#: resolves its helpers through `CONTROLLER_STUBS` instead of naming them, so
#: the AST cannot see the calls. Named individually and not by file, so a
#: THIRD shell added here is still caught.
_SELF_REL = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
_SELF_EXEMPT: frozenset[str] = frozenset(
    {
        f"{_SELF_REL}::controllers_the_store_setter_reads",
        f"{_SELF_REL}::test_the_mapped_stubs_are_together_sufficient_for_a_bare_shell",
    }
)

_MISSING_SCREEN_ATTR = re.compile(
    r"'ChatScreen' object has no attribute '(?P<name>[^']+)'"
)

#: One round per missing controller. A bound only so a production change that
#: makes the assignment raise forever fails with a readable message instead of
#: hanging.
_MAX_PROBE_ROUNDS = 25


class _AnyAttribute:
    """Stand-in that answers any attribute read with another of itself.

    Stands where a controller would during the probe. It must satisfy
    *reads* only: the hook build stores bound methods for later, it never
    calls them, which is exactly why an all-raisers stub is enough for a
    real shell too.
    """

    def __getattr__(self, name: str) -> "_AnyAttribute":
        return _AnyAttribute()


@lru_cache(maxsize=1)
def controllers_the_store_setter_reads() -> tuple[str, ...]:
    """Every screen attribute `screen._console_chat_store = ...` requires.

    Derived by doing it: assign the store on a bare shell, record the
    attribute the `AttributeError` names, install a stand-in for it, and
    repeat until the assignment succeeds. What comes back is the exact set a
    `ChatScreen.__new__` fixture must provide, in the order production
    demands them.

    Returns:
        tuple[str, ...]: Attribute names, in discovery order.

    Raises:
        AssertionError: If the assignment still fails after
            `_MAX_PROBE_ROUNDS`, or fails with an `AttributeError` that is
            not about a missing `ChatScreen` attribute.
    """
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    found: list[str] = []
    for _ in range(_MAX_PROBE_ROUNDS):
        # A FRESH shell every round, deliberately. `_console_runtime()`
        # caches the runtime on the shell as `_console_runtime_ref`, and
        # `ensure_console_runtime` re-attaches (the step that reads the
        # controllers) only when the view CHANGED -- so a second assignment
        # on the same shell silently succeeds without reading anything, and
        # the probe would report the first missing controller as the only
        # one. That false answer is precisely the blind spot this file
        # exists to remove.
        screen = ChatScreen.__new__(ChatScreen)
        screen.app_instance = None
        for name in found:
            setattr(screen, name, _AnyAttribute())
        try:
            screen._console_chat_store = None
        except AttributeError as exc:
            match = _MISSING_SCREEN_ATTR.search(str(exc))
            assert match is not None, (
                "The store setter raised an AttributeError this probe cannot "
                f"classify, so the derived controller set is not trustworthy: {exc}"
            )
            found.append(match.group("name"))
            continue
        return tuple(found)
    raise AssertionError(
        "The store setter still fails on a bare shell after "
        f"{_MAX_PROBE_ROUNDS} rounds; last derived set: {found}"
    )


@lru_cache(maxsize=1)
def _controllers_wiring_installs() -> frozenset[str]:
    """Slots `build_console_controllers` assigns onto the screen.

    Used only to make a failure message actionable -- it lets the guard say
    "this is a controller, built at `wiring.py`" rather than leaving the
    reader to work out what the missing attribute is.
    """
    source = (REPO_ROOT / "tldw_chatbook/UI/Console_Modules/wiring.py").read_text(
        encoding="utf-8"
    )
    slots: set[str] = set()
    for fn in ast.walk(ast.parse(source)):
        if not isinstance(fn, ast.FunctionDef):
            continue
        if fn.name != "build_console_controllers":
            continue
        for node in ast.walk(fn):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "screen"
                ):
                    slots.add(target.attr)
    return frozenset(slots)


def _is_chat_screen_new(node: ast.AST) -> bool:
    """``ChatScreen.__new__(...)``, however the module spells the attribute."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "__new__"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "ChatScreen"
    )


def _assigned_attributes(node: ast.AST) -> list[str]:
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    return [t.attr for t in targets if isinstance(t, ast.Attribute)]


def _sets_console_chat_store(node: ast.AST) -> bool:
    return "_console_chat_store" in _assigned_attributes(node)


def _supplies_own_runtime(node: ast.AST) -> bool:
    """``screen._console_runtime_ref = ...`` — the other way to be safe.

    `_console_runtime()` returns a pre-set `_console_runtime_ref` verbatim and
    only calls `ensure_console_runtime` when it finds none. A shell that
    supplies its own runtime therefore never reaches `attach_view`, so it
    needs no controllers at all. `Tests/UI/test_console_native_chat_flow.py`
    does exactly this, and it is arguably the cleaner of the two fixes.
    """
    return "_console_runtime_ref" in _assigned_attributes(node)


def _called_names(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


# ---------------------------------------------------------------------------
# ordering: wiring only counts if it happens BEFORE the assignment
# ---------------------------------------------------------------------------
#
# The store assignment is not a fact recorded for later -- it is the setter
# running `attach_view` and reading the controllers *there and then*. So a
# fixture that assigns first and stubs afterwards still dies in setup, and a
# scan that only asks "is this helper called anywhere in this function?" waves
# it through. That is the same shape of hole this file was rewritten to close
# (a check that looked stronger than it was), one level down, so the scan asks
# for order too.

#: Node types whose *body* runs later than the statement that creates them --
#: the point at which "further down the file" stops implying "afterwards".
#: `ClassDef` is in here although a class body executes immediately; treating
#: it as deferred can only make the guard stricter, never blinder.
_DEFERRED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _scope_chains(fn: ast.AST) -> dict[ast.AST, tuple[ast.AST, ...]]:
    """Every node under `fn` -> the deferred scopes standing between them.

    A node with an empty chain runs in `fn`'s own body. A node with a chain
    runs only when each scope in it is *called*. Note a nested `def` node
    itself carries its **enclosing** chain: the `def` statement executes where
    it is written; only its children are deferred.
    """
    chains: dict[ast.AST, tuple[ast.AST, ...]] = {fn: ()}

    def descend(parent: ast.AST, chain: tuple[ast.AST, ...]) -> None:
        for child in ast.iter_child_nodes(parent):
            chains[child] = chain
            deeper = chain + (child,) if isinstance(child, _DEFERRED_SCOPES) else chain
            descend(child, deeper)

    descend(fn, ())
    return chains


def _pos(node: ast.AST) -> tuple[int, int]:
    return (node.lineno, node.col_offset)


def _runs_before(
    earlier: ast.AST, later: ast.AST, chains: dict[ast.AST, tuple[ast.AST, ...]]
) -> bool:
    """Does `earlier` provably execute before `later` within one function?

    Two shapes are provable, and everything else is deliberately refused:

    1. **Same scope** -- source order decides. Straight-line, `with`, `if` and
       `for` bodies all read top-to-bottom, and the loop case stays correct for
       the reason that matters: wiring written after the assignment is too late
       on the first iteration, which is the iteration that dies.
    2. **`earlier` in a scope that encloses `later`'s** -- `later` cannot run
       until the nested `def`/`lambda` around it is called, and nothing can call
       it before its own `def` statement has executed. So `earlier` is first
       exactly when it precedes that statement (not merely when it precedes
       `later`, which a call site sitting in between would falsify).

    **The deliberate refusal:** a helper called inside a nested function that is
    *defined* above the assignment is NOT credited, because whether it runs at
    all -- let alone first -- depends on a call site this scan does not follow.
    Such a fixture is reported, and the fix is to hoist the stub call or to
    let the guard see the nested function on its own (it scans every `def`,
    nested ones included, so a self-contained closure is already checked
    directly). Flagging a fixture that may be fine costs one hoisted line;
    missing one costs a setup-time `AttributeError` in a test that never
    mentions the attribute -- which is the whole reason this file exists.

    This proves **order, not reachability**: a stub call written above the
    assignment but inside an `if` still counts. Ordering was the hole; making
    the scan a reachability analysis is a different, much larger claim.
    """
    outer, inner = chains[earlier], chains[later]
    if outer == inner:
        return _pos(earlier) < _pos(later)
    if len(outer) < len(inner) and inner[: len(outer)] == outer:
        return _pos(earlier) < _pos(inner[len(outer)])
    return False


def _wiring_gaps(fn: ast.AST) -> list[str]:
    """What a store-setting shell is missing, in mapping order.

    Returns an empty list for a function that never sets the store, that hands
    itself a runtime before every assignment, or that wires every mapped stub
    before every assignment. `every` is not belt-and-braces: each assignment is
    its own attach, so wiring that only precedes the second one is still late
    for the first.

    Returns:
        list[str]: One row per unsatisfied stub -- bare name when the helper is
        never called, or a line-numbered row when it is called too late, since
        "you forgot this" and "you did this afterwards" are different repairs.
    """
    nodes = list(ast.walk(fn))
    stores = [n for n in nodes if _sets_console_chat_store(n)]
    if not stores:
        return []
    chains = _scope_chains(fn)

    def guards_every_assignment(node: ast.AST) -> bool:
        return all(_runs_before(node, store, chains) for store in stores)

    if any(guards_every_assignment(n) for n in nodes if _supplies_own_runtime(n)):
        return []

    first_store = min(stores, key=_pos)
    gaps: list[str] = []
    for stub in CONTROLLER_STUBS.values():
        calls = sorted((n for n in nodes if _called_names(n) == stub), key=_pos)
        if any(guards_every_assignment(call) for call in calls):
            continue
        if not calls:
            gaps.append(stub)
        else:
            where = ", ".join(f"line {call.lineno}" for call in calls)
            gaps.append(
                f"{stub} -- called at {where}, which is not provably before the "
                f"_console_chat_store assignment at line {first_store.lineno}"
            )
    return gaps


def _offending_functions(tree: ast.AST, rel: str) -> dict[str, list[str]]:
    """Store-setting shells in one module -> the wiring they lack."""
    out: dict[str, list[str]] = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_is_chat_screen_new(n) for n in ast.walk(fn)):
            continue
        gaps = _wiring_gaps(fn)
        if gaps:
            out[f"{rel}::{fn.name}"] = gaps
    return out


def _scan_test_tree() -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        # Cheap reject before paying for a parse: both spellings must appear.
        if "ChatScreen.__new__" not in text or "_console_chat_store" not in text:
            continue
        rel = path.relative_to(REPO_ROOT).as_posix()
        offenders.update(_offending_functions(ast.parse(text), rel))
    return offenders


# ---------------------------------------------------------------------------
# the derivation, and the mapping it holds honest
# ---------------------------------------------------------------------------


def test_every_controller_the_store_setter_reads_has_a_stub() -> None:
    """`CONTROLLER_STUBS` must name exactly what the store setter reads.

    This is the half TASK-23144 added, and the half that removes the blind
    spot: the old guard hard-coded `stub_fleet_controller`, so a second
    controller entering the same path was invisible to it. Here the set comes
    from production and the mapping has to match it in both directions -- a
    new controller fails with its own name, and a controller that leaves the
    path fails as a stale entry rather than quietly making every fixture do
    unnecessary work.

    Raises:
        AssertionError: If production reads a controller with no stub mapped,
            or `CONTROLLER_STUBS` names one production no longer reads.
    """
    derived = set(controllers_the_store_setter_reads())
    mapped = set(CONTROLLER_STUBS)
    unmapped = sorted(derived - mapped)
    stale = sorted(mapped - derived)
    built_by_wiring = _controllers_wiring_installs()
    detail = ", ".join(
        "{} ({})".format(
            name,
            "built by build_console_controllers"
            if name in built_by_wiring
            else "NOT a wiring-installed controller",
        )
        for name in unmapped
    )
    assert not unmapped and not stale, (
        "The set of controllers `screen._console_chat_store = ...` reads no "
        "longer matches CONTROLLER_STUBS.\n"
        f"  read by production but unmapped here: {unmapped}{f' -- {detail}' if detail else ''}\n"
        f"  mapped here but no longer read: {stale}\n"
        "For an unmapped controller: add a `stub_<name>_controller` helper to "
        "Tests/UI/console_controller_stubs.py (every constructor callable "
        "defaulting to a raiser), map it here, and wire it in the fixtures the "
        "ratchet then names. Do NOT satisfy this by deleting the derivation -- "
        "that is exactly how 46 tests came to die in setup at TASK-23144."
    )


def test_the_mapped_stubs_are_together_sufficient_for_a_bare_shell() -> None:
    """The positive control: these helpers, and only these, unblock the setter.

    Proves the mapping is real rather than plausible -- that each named
    helper installs the slot it is mapped to, and that together they are
    enough for the assignment the ratchet is about. Without this, a mapping
    could name a helper that builds a *different* controller and the
    set-equality test above would still pass.

    Raises:
        AssertionError: If the store assignment still fails once every mapped
            helper has run against a bare shell.
    """
    import Tests.UI.console_controller_stubs as stubs
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    screen = ChatScreen.__new__(ChatScreen)
    # A plain object, not `None`: the library-activity stub refuses to
    # snapshot a missing app unless the fixture says `NO_APP`, and this
    # control is not making that statement on anyone's behalf.
    screen.app_instance = SimpleNamespace()
    for slot, helper_name in CONTROLLER_STUBS.items():
        helper = getattr(stubs, helper_name, None)
        assert callable(helper), (
            f"CONTROLLER_STUBS maps {slot!r} to {helper_name!r}, which "
            "Tests/UI/console_controller_stubs.py does not define."
        )
        helper(screen, context="architecture positive control")
        assert getattr(screen, slot, None) is not None, (
            f"{helper_name} did not install {slot!r}; the mapping is wrong."
        )

    screen._console_chat_store = None  # must not raise


# ---------------------------------------------------------------------------
# the ratchet
# ---------------------------------------------------------------------------


def test_no_bare_chat_screen_shell_sets_its_store_without_its_controllers() -> None:
    """Fail on any shell that assigns the store without satisfying the runtime.

    Scans every test module that mentions both spellings, and reports the
    offending functions -- with the stub each one is missing -- rather than
    the first one found: a sweep that stops at one violation makes a
    multi-file regression take as many rounds to clear as it has files.

    Raises:
        AssertionError: If any function outside `ALLOWLIST` builds a shell with
            `ChatScreen.__new__` and assigns `_console_chat_store` without
            first wiring every `CONTROLLER_STUBS` helper or first supplying
            `_console_runtime_ref`.
    """
    offenders = {
        name: missing
        for name, missing in _scan_test_tree().items()
        if name not in ALLOWLIST and name not in _SELF_EXEMPT
    }
    rows = "\n  ".join(
        f"{name}\n      - " + "\n      - ".join(missing)
        for name, missing in sorted(offenders.items())
    )
    assert not offenders, (
        "These functions build a ChatScreen shell with __new__ and assign "
        "_console_chat_store without first wiring every controller that "
        "assignment reaches. The assignment is a property whose setter attaches "
        "the view to the Console runtime, which reads "
        f"{list(controllers_the_store_setter_reads())} off the screen as it "
        "runs, so the shell will die during setup with an AttributeError that "
        "names neither this function nor the behaviour under test. Wiring "
        "written after the assignment is as absent as no wiring at all. Either "
        "call the Tests.UI.console_controller_stubs helper(s) BEFORE the "
        "assignment, or give the shell its own _console_runtime_ref first.\n  " + rows
    )


def test_the_allowlist_does_not_name_a_function_that_is_already_clean() -> None:
    """Refuse allowlist entries that no longer describe a violation.

    A ratchet that keeps stale entries stops ratcheting: the list grows a
    reputation for being noise, and the next real entry is waved through with
    it. Anything listed must still be a genuine violation, or the list should
    shrink.

    Raises:
        AssertionError: If an `ALLOWLIST` entry names a function that no longer
            violates the rule.
    """
    if not ALLOWLIST:
        return
    stale = sorted(ALLOWLIST - set(_scan_test_tree()))
    assert not stale, f"allowlist entries no longer violate; remove them: {stale}"


# ---------------------------------------------------------------------------
# the ordering rule, proved both ways
# ---------------------------------------------------------------------------
#
# The scan over the real tree can only ever prove the guard is quiet. These run
# it over hand-written shells so the other half is proved too: that wiring
# written after the assignment is actually reported. Without them the ordering
# rule could rot into a no-op and the tree scan would stay green -- which is
# precisely how the pre-TASK-23144 guard passed while 46 tests died in setup.


def _fixture_source(body: str) -> str:
    """A synthetic module from `body`, with `<STUBS>` expanded where written.

    `<STUBS>` becomes one call per helper in `CONTROLLER_STUBS`, at the
    indentation the placeholder was written at, so adding a controller does not
    send anyone editing these controls.

    Args:
        body: Module source, dedented and stripped for you.

    Returns:
        str: Source ready for `ast.parse`.
    """
    lines: list[str] = []
    for line in textwrap.dedent(body).strip("\n").splitlines():
        if line.strip() == "<STUBS>":
            indent = line[: len(line) - len(line.lstrip())]
            lines += [f"{indent}{name}(screen)" for name in CONTROLLER_STUBS.values()]
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def _scan_fixture(body: str) -> dict[str, list[str]]:
    """The real scan, run over one synthetic module."""
    return _offending_functions(ast.parse(_fixture_source(body)), "synthetic.py")


def test_the_scan_accepts_wiring_that_precedes_the_assignment() -> None:
    """The shape every real fixture uses must stay clean.

    Raises:
        AssertionError: If the ordering rule flags correctly ordered wiring.
    """
    assert (
        _scan_fixture(
            """
            def fixture(store):
                screen = ChatScreen.__new__(ChatScreen)
                <STUBS>
                screen._console_chat_store = store
                return screen
            """
        )
        == {}
    )


def test_the_scan_reports_wiring_that_follows_the_assignment() -> None:
    """The negative control this whole section exists for.

    A fixture that calls every mapped helper -- but after the assignment that
    reads the controllers -- passed the name-only scan and still died in setup.

    Raises:
        AssertionError: If a late-stubbing fixture is not reported, or is
            reported without saying that ordering is the problem.
    """
    offenders = _scan_fixture(
        """
        def fixture(store):
            screen = ChatScreen.__new__(ChatScreen)
            screen._console_chat_store = store
            <STUBS>
            return screen
        """
    )
    assert list(offenders) == ["synthetic.py::fixture"]
    rows = offenders["synthetic.py::fixture"]
    assert len(rows) == len(CONTROLLER_STUBS)
    for stub, row in zip(CONTROLLER_STUBS.values(), rows):
        assert row.startswith(stub)
        assert "not provably before" in row


def test_the_scan_reports_a_runtime_supplied_after_the_assignment() -> None:
    """The escape hatch is order-sensitive too, and for the same reason.

    `_console_runtime()` only honours a `_console_runtime_ref` that is already
    there; one set afterwards leaves the assignment to attach a real runtime.

    Raises:
        AssertionError: If a late `_console_runtime_ref` is accepted, or an
            early one is not.
    """
    early = """
        def fixture(store, runtime):
            screen = ChatScreen.__new__(ChatScreen)
            screen._console_runtime_ref = runtime
            screen._console_chat_store = store
    """
    late = """
        def fixture(store, runtime):
            screen = ChatScreen.__new__(ChatScreen)
            screen._console_chat_store = store
            screen._console_runtime_ref = runtime
    """
    assert _scan_fixture(early) == {}
    assert list(_scan_fixture(late)) == ["synthetic.py::fixture"]


def test_the_scan_reads_every_assignment_not_just_the_first() -> None:
    """Wiring that only precedes the second assignment is late for the first.

    Raises:
        AssertionError: If a fixture that stubs between two assignments passes.
    """
    offenders = _scan_fixture(
        """
        def fixture(first, second):
            screen = ChatScreen.__new__(ChatScreen)
            screen._console_chat_store = first
            <STUBS>
            screen._console_chat_store = second
        """
    )
    assert list(offenders) == ["synthetic.py::fixture"]


def test_the_scan_follows_wiring_across_nesting_it_can_prove() -> None:
    """`with`/`if`/`for` bodies read top-to-bottom, and a closure is its own scope.

    Both shapes below occur in the tree today -- `test_console_rag_settings_modal`
    builds its shell entirely inside a closure -- so neither may be flagged.

    Raises:
        AssertionError: If provably ordered wiring is reported.
    """
    nested_blocks = """
        def fixture(store, lock):
            screen = ChatScreen.__new__(ChatScreen)
            with lock:
                if store:
                    <STUBS>
                screen._console_chat_store = store
    """
    self_contained_closure = """
        def fixture(store):
            def build():
                screen = ChatScreen.__new__(ChatScreen)
                <STUBS>
                screen._console_chat_store = store
                return screen

            return build()
    """
    wired_before_the_closure_exists = """
        def fixture(store):
            screen = ChatScreen.__new__(ChatScreen)
            <STUBS>

            def attach():
                screen._console_chat_store = store

            return attach
    """
    assert _scan_fixture(nested_blocks) == {}
    assert _scan_fixture(self_contained_closure) == {}
    assert _scan_fixture(wired_before_the_closure_exists) == {}


def test_the_scan_reports_wiring_it_cannot_prove_runs_first() -> None:
    """Deferred wiring is refused, deliberately -- see `_runs_before`.

    Neither shape is necessarily broken: the first really does wire before it
    assigns, and the second may never call `attach` at all. But both hide the
    order behind a call site the scan does not follow, and this guard exists
    because a check that assumed the best about a fixture let 46 tests die in
    setup. The cost of being wrong here is one hoisted line; the cost of being
    wrong the other way is an `AttributeError` in a test that never mentions
    the attribute.

    Raises:
        AssertionError: If either deferred shape is silently accepted.
    """
    wiring_hidden_in_a_helper = """
        def fixture(store):
            screen = ChatScreen.__new__(ChatScreen)

            def wire():
                <STUBS>

            wire()
            screen._console_chat_store = store
    """
    wiring_written_after_the_closure = """
        def fixture(store):
            screen = ChatScreen.__new__(ChatScreen)

            def attach():
                screen._console_chat_store = store

            attach()
            <STUBS>
    """
    assert list(_scan_fixture(wiring_hidden_in_a_helper)) == ["synthetic.py::fixture"]
    assert list(_scan_fixture(wiring_written_after_the_closure)) == [
        "synthetic.py::fixture"
    ]
