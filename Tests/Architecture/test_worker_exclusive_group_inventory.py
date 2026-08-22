# test_worker_exclusive_group_inventory.py
# Description: Guard against exclusive workers scheduled into the shared
#              "default" group (task-19559)
"""task-19559: In Textual (verified against the installed 8.2.8), a worker
scheduled with ``exclusive=True`` and no ``group=`` lands in the group
literally named ``"default"`` -- ``Worker.__init__``'s signature is
``group: str = "default"``, and so is ``work()``'s. ``WorkerManager.add_worker``
then calls ``cancel_group(worker.node, worker.group)``, and ``cancel_group``
filters on ``(worker.group, worker.node)`` alone::

    workers = [
        worker
        for worker in self._workers
        if (worker.group == group and worker.node == node)
    ]

It **never consults ``name=``**. So every ungrouped exclusive worker on a node
mutually cancels every other one, no matter how distinct their ``name=`` values
are, and ``name=`` scopes nothing at all.

Two facts make this quietly destructive rather than merely surprising:

* ``asyncio.CancelledError`` derives from ``BaseException``, so the
  ``except Exception:`` blocks these handlers use cannot observe it. A
  cancelled save does not log, does not toast, does not raise -- it vanishes.
* ``Worker.cancel()`` does **not** stop a thread worker. Its body runs to
  completion in the executor and its ``call_from_thread`` callbacks still land,
  so for thread workers grouping alone is not even sufficient -- the result
  must additionally be refused at arrival (see task-19563).

**Counting.** A naive ``grep exclusive=True | grep -v group=`` over the package
over-counts by roughly 4x (523 vs 133 at this task's branch base), because a
multi-line call carries ``group=`` on a following line. This guard walks the
AST instead, which is also what produced the census in the task notes.

**The contract.** Every ``@work(...)`` decorator and every ``run_worker(...)``
call inside ``tldw_chatbook/`` that requests exclusivity must name its group,
so the blast radius of the cancellation is written down at the call site. The
group name should describe the *work*, not the caller: two call sites that
start the same load belong in the same group (a refresh supersedes a refresh),
two different operations do not (a section load must not kill a save).

An exclusive worker that genuinely wants the shared ``"default"`` group is
allowed, but only through ``DEFAULT_GROUP_ALLOWLIST`` -- an entry there is a
recorded decision with a reason, not a silent one.

**Relationship to the existing guard.**
``Tests/UI/test_chat_screen_worker_groups.py`` (TASK-228) enforces the same
rule over ``chat_screen.py`` plus ``UI/Console_Modules/``, and additionally
pins that the Console run and sync workers use *disjoint* groups. This sweep
generalises the first half to the whole package; the Console-specific
disjointness check stays where it is.

Born red: at the branch base this sweep reported **133 sites across 32 files**
(the four user-visible instances task-19559 names among them: Study rating
submission, Media analysis generation, four of the six Watchlists section
loaders, and the Settings advanced-config backup load).
``test_multiline_ungrouped_exclusive_is_flagged`` pins the multi-line form the
naive grep is blind to, and ``test_ungrouped_exclusive_decorator_is_flagged``
pins the decorator form.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: Callables that schedule a Textual worker. ``work`` is the decorator factory
#: (``@work(...)``); ``run_worker`` is ``DOMNode.run_worker``.
SCHEDULER_NAMES = {"work", "run_worker"}

#: ``DOMNode.run_worker``'s positional parameter order, so a ``group`` or
#: ``exclusive`` passed positionally is still seen. ``work()`` takes everything
#: after ``method`` as keyword-only, so it needs no equivalent.
RUN_WORKER_POSITIONAL = (
    "work",
    "name",
    "group",
    "description",
    "exit_on_error",
    "start",
    "exclusive",
    "thread",
)

#: task-19559: sites that deliberately want default-group mutual exclusion, or
#: that this repo does not own. Keyed by ``"<path>::<owning function>"`` -- a
#: qualified name rather than a line number, so the entry survives edits above
#: it. The value is the reason the exemption is correct; adding a row here is a
#: decision to accept that *every* other ungrouped exclusive worker on the same
#: node cancels this one, and vice versa.
DEFAULT_GROUP_ALLOWLIST: dict[str, str] = {
    "tldw_chatbook/Third_Party/textual_fspicker/parts/directory_navigation.py"
    "::DirectoryNavigation._load": (
        "Vendored upstream code (textual-fspicker), kept byte-compatible with "
        "its source tree so it can be re-synced. It is also benign: `_load` is "
        "the only exclusive worker on that widget, so the default group "
        "degenerates to self-exclusion, which is exactly what it wants."
    ),
}


def _call_name(node: ast.Call) -> str | None:
    """Return the bare callee name for ``f()``, ``a.f()`` and ``f[T]()``."""
    func = node.func
    if isinstance(func, ast.Subscript):
        func = func.value
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _keyword(node: ast.Call, name: str) -> ast.expr | None:
    for keyword in node.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _owner_index(tree: ast.Module) -> dict[int, str]:
    """Map every line to the qualified name of the function that owns it.

    A ``@work(...)`` decorator is attributed to the function it decorates (the
    decorator's own line falls outside that function's ``lineno`` range), which
    is the name a reader would use to talk about the worker.
    """
    owners: dict[int, str] = {}
    stack: list[str] = []

    class _Visitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            stack.append(node.name)
            self.generic_visit(node)
            stack.pop()

        def _visit_function(self, node: ast.AST) -> None:
            stack.append(node.name)  # type: ignore[attr-defined]
            qualname = ".".join(stack)
            start = min(
                [node.lineno]  # type: ignore[attr-defined]
                + [d.lineno for d in node.decorator_list]  # type: ignore[attr-defined]
            )
            end = node.end_lineno or node.lineno  # type: ignore[attr-defined]
            for line in range(start, end + 1):
                owners[line] = qualname
            self.generic_visit(node)
            stack.pop()

        visit_FunctionDef = _visit_function  # type: ignore[assignment]
        visit_AsyncFunctionDef = _visit_function  # type: ignore[assignment]

    _Visitor().visit(tree)
    return owners


def _violations_in_tree(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Return ``(lineno, owner_qualname, detail)`` for every ungrouped
    exclusive worker schedule in an already-parsed module.

    Path-free, so tests can drive it with synthetic source rather than only
    with real package files.
    """
    owners = _owner_index(tree)
    violations: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scheduler = _call_name(node)
        if scheduler not in SCHEDULER_NAMES:
            continue

        exclusive = _keyword(node, "exclusive")
        group = _keyword(node, "group")
        # A `**kwargs` spread can carry `exclusive=` and can equally fail to
        # carry `group=`, so neither can be proved from the AST. Fail CLOSED,
        # as `Tests/UI/test_chat_screen_worker_groups.py` (TASK-228) does: a
        # guard that skips what it cannot prove certifies nothing.
        has_spread = any(keyword.arg is None for keyword in node.keywords)
        if scheduler == "run_worker":
            for index, arg in enumerate(node.args):
                if index >= len(RUN_WORKER_POSITIONAL):
                    break
                if RUN_WORKER_POSITIONAL[index] == "group":
                    group = arg
                elif RUN_WORKER_POSITIONAL[index] == "exclusive":
                    exclusive = arg

        if exclusive is None and not has_spread:
            continue
        # `exclusive=False` is not exclusive; anything else (including a
        # variable or an expression) may be, so it must still name a group.
        if isinstance(exclusive, ast.Constant) and exclusive.value is False:
            continue
        if group is not None:
            continue

        shown = "**kwargs" if exclusive is None else ast.unparse(exclusive)
        violations.append(
            (
                node.lineno,
                owners.get(node.lineno, "<module>"),
                f"{scheduler}(exclusive={shown}) with no group=",
            )
        )
    return violations


def _violations_in_file(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    relative = path.relative_to(PACKAGE_ROOT.parent)
    reported: list[str] = []
    for lineno, owner, detail in _violations_in_tree(tree):
        if f"{relative}::{owner}" in DEFAULT_GROUP_ALLOWLIST:
            continue
        reported.append(f"{relative}:{lineno} ({owner}): {detail}")
    return reported


def _all_site_keys() -> set[str]:
    keys: set[str] = set()
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = path.relative_to(PACKAGE_ROOT.parent)
        for _lineno, owner, _detail in _violations_in_tree(tree):
            keys.add(f"{relative}::{owner}")
    return keys


def test_no_ungrouped_exclusive_workers() -> None:
    """Every exclusive worker in the package names the group it may cancel."""
    assert PACKAGE_ROOT.is_dir(), f"package root not found: {PACKAGE_ROOT}"
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        violations.extend(_violations_in_file(path))
    assert not violations, (
        "exclusive=True scheduled without an explicit group= lands in the "
        'shared "default" group, where it cancels every other ungrouped '
        "exclusive worker on the same node (cancel_group filters on "
        "(node, group) and never on name=). Name the group after the WORK "
        "being done, or -- if default-group mutual exclusion really is what "
        "the site wants -- add it to DEFAULT_GROUP_ALLOWLIST with the reason:\n"
        + "\n".join(violations)
    )


def test_ungrouped_exclusive_decorator_is_flagged() -> None:
    """The ``@work(exclusive=True)`` decorator form must be caught.

    Born red against the branch base, where `settings_screen.py`'s twelve
    thread workers were written exactly like this.
    """
    tree = ast.parse(
        "from textual import work\n"
        "from textual.widget import Widget\n\n"
        "class ScratchWidget(Widget):\n"
        "    @work(exclusive=True, thread=True)\n"
        "    def _save(self) -> None:\n"
        "        pass\n"
    )
    violations = _violations_in_tree(tree)
    assert len(violations) == 1
    lineno, owner, detail = violations[0]
    assert lineno == 5
    assert owner == "ScratchWidget._save"
    assert "no group=" in detail


def test_multiline_ungrouped_exclusive_is_flagged() -> None:
    """The multi-line call form -- the one a line-oriented grep gets wrong.

    A naive ``grep exclusive=True | grep -v group=`` reports this site twice
    over (once for the ``exclusive=True`` line of the flagged call, once for
    the grouped call's ``exclusive=True`` line whose ``group=`` sits on the
    NEXT line). The AST sees one violation and one clean site.
    """
    tree = ast.parse(
        "from textual.widget import Widget\n\n"
        "class ScratchWidget(Widget):\n"
        "    def _start(self) -> None:\n"
        "        self.run_worker(\n"
        "            self._flagged(),\n"
        "            exclusive=True,\n"
        "        )\n"
        "        self.run_worker(\n"
        "            self._clean(),\n"
        "            exclusive=True,\n"
        "            group='scratch-clean',\n"
        "        )\n"
    )
    violations = _violations_in_tree(tree)
    assert len(violations) == 1
    lineno, owner, _detail = violations[0]
    assert lineno == 5
    assert owner == "ScratchWidget._start"


def test_positional_group_is_not_a_violation() -> None:
    """``run_worker(work, name, group, ...)`` takes group positionally too.

    A site that passes it that way has named its group, even though no
    ``group=`` string appears on the line.
    """
    tree = ast.parse(
        "from textual.widget import Widget\n\n"
        "class ScratchWidget(Widget):\n"
        "    def _start(self) -> None:\n"
        "        self.run_worker(\n"
        "            self._work, 'worker-name', 'scratch-group', exclusive=True\n"
        "        )\n"
    )
    assert _violations_in_tree(tree) == []


def test_name_keyword_does_not_satisfy_the_guard() -> None:
    """``name=`` is the exact confusion this guard exists to catch.

    ``cancel_group`` never reads ``name``; a site that passes one and believes
    it is scoped is still in the shared ``"default"`` group.
    """
    tree = ast.parse(
        "from textual.widget import Widget\n\n"
        "class ScratchWidget(Widget):\n"
        "    def _start(self) -> None:\n"
        "        self.run_worker(\n"
        "            self._work, thread=True, exclusive=True, name='my_search'\n"
        "        )\n"
    )
    assert len(_violations_in_tree(tree)) == 1


def test_non_exclusive_worker_needs_no_group() -> None:
    """A worker that is not exclusive cancels nothing, so it is unconstrained."""
    tree = ast.parse(
        "from textual.widget import Widget\n\n"
        "class ScratchWidget(Widget):\n"
        "    def _start(self) -> None:\n"
        "        self.run_worker(self._work())\n"
        "        self.run_worker(self._other(), exclusive=False)\n"
    )
    assert _violations_in_tree(tree) == []


def test_allowlisted_site_is_not_flagged() -> None:
    """An allowlisted owner is exempt, and the exemption is keyed by name."""
    tree = ast.parse(
        "from textual import work\n"
        "from textual.widget import Widget\n\n"
        "class DirectoryNavigation(Widget):\n"
        "    @work(exclusive=True)\n"
        "    def _load(self) -> None:\n"
        "        pass\n"
    )
    violations = _violations_in_tree(tree)
    assert len(violations) == 1
    _lineno, owner, _detail = violations[0]
    assert (
        "tldw_chatbook/Third_Party/textual_fspicker/parts/directory_navigation.py"
        f"::{owner}"
    ) in DEFAULT_GROUP_ALLOWLIST


def test_allowlist_has_no_stale_entries() -> None:
    """Every allowlist row must still describe a real site.

    Without this, an entry outlives the code it excused and quietly widens the
    exemption for whatever function later takes that name.
    """
    live = _all_site_keys()
    stale = sorted(key for key in DEFAULT_GROUP_ALLOWLIST if key not in live)
    assert not stale, (
        "DEFAULT_GROUP_ALLOWLIST entries that no longer match an ungrouped "
        "exclusive worker -- delete them:\n" + "\n".join(stale)
    )


def test_allowlist_entries_state_a_reason() -> None:
    """An allowlist row without reasoning is folklore; require prose."""
    for key, reason in DEFAULT_GROUP_ALLOWLIST.items():
        assert len(reason.split()) >= 10, f"{key}: reason is too thin to review"
