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

TASK-19870 extends the same ownership rule to mutation-triggered refreshes in
Schedules and Watchlists. Awaiting a raw loader inside a mutation worker
bypasses the loader's exclusive group, so the two affected modules are scanned
for ``await self.<raw loader>(...)`` and the eleven known mutation owners are
pinned to exactly one call to their group-dispatch helper.
"""

from __future__ import annotations

import ast
import functools
import re
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"

#: Callables that schedule a Textual worker. ``work`` is the decorator factory
#: (``@work(...)``); ``run_worker`` is ``DOMNode.run_worker``.
SCHEDULER_NAMES = {"work", "run_worker"}

#: Qodo review of PR #1951: the sweep parsed all ~1,780 package modules twice.
#: It is now a single cached pass over the files that could possibly contain a
#: scheduler call, which is a cost fix and NOT a coverage fix -- the set of
#: sites reported is byte-identical (``test_prefilter_admits_every_flagged_form``
#: pins that, and ``test_scan_is_a_single_cached_pass`` pins that the cache does
#: not quietly re-read).
#:
#: The prefilter is sound because ``_call_name`` only ever returns an
#: ``ast.Name.id`` or an ``ast.Attribute.attr``: both are identifiers that must
#: appear verbatim as a token in the source. A file with neither token cannot
#: contain a call this guard would flag. (``\b`` boundaries mean ``workspace``
#: and ``workflow`` do not match, while ``x.work()`` and ``run_worker`` do.)
_SCHEDULER_TOKEN = re.compile(r"\b(?:work|run_worker)\b")

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

SCHEDULES_WORKBENCH_PATH = "tldw_chatbook/UI/Screens/scheduling/schedules_workbench.py"
WATCHLISTS_COLLECTIONS_PATH = (
    "tldw_chatbook/UI/Screens/watchlists_collections_screen.py"
)

#: TASK-19870: raw loaders whose direct await bypasses the exclusive loader
#: group in each affected production module. This is deliberately path scoped:
#: identically named methods elsewhere have not been audited into this contract.
FORBIDDEN_MUTATION_LOADERS_BY_PATH: dict[str, frozenset[str]] = {
    SCHEDULES_WORKBENCH_PATH: frozenset({"load_tasks"}),
    WATCHLISTS_COLLECTIONS_PATH: frozenset({"_load_notifications", "_load_briefings"}),
}

#: Every audited mutation owner must dispatch exactly once through the named
#: helper. Qualified owners survive edits above the site; nested worker bodies
#: remain distinct from their enclosing handlers without relying on line pins.
MUTATION_REFRESH_HELPER_INVENTORY: dict[str, str] = {
    f"{SCHEDULES_WORKBENCH_PATH}::SchedulesWorkbench."
    "_on_delete_task_requested._delete_and_refresh": "_request_tasks_refresh",
    f"{SCHEDULES_WORKBENCH_PATH}::SchedulesWorkbench."
    "_on_reminder_form_result._save_and_refresh": "_request_tasks_refresh",
    f"{SCHEDULES_WORKBENCH_PATH}::SchedulesWorkbench."
    "_run_reminder_now._run_and_refresh": "_request_tasks_refresh",
    f"{SCHEDULES_WORKBENCH_PATH}::SchedulesWorkbench."
    "_set_reminder_enabled._update_and_refresh": "_request_tasks_refresh",
    f"{SCHEDULES_WORKBENCH_PATH}::SchedulesWorkbench."
    "_on_bulk_delete_confirmed._bulk_delete": "_request_tasks_refresh",
    f"{SCHEDULES_WORKBENCH_PATH}::SchedulesWorkbench."
    "_bulk_toggle_marked._bulk_toggle": "_request_tasks_refresh",
    f"{WATCHLISTS_COLLECTIONS_PATH}::WatchlistsCollectionsScreen."
    "_mark_notification_read": "_request_notifications_refresh",
    f"{WATCHLISTS_COLLECTIONS_PATH}::WatchlistsCollectionsScreen."
    "_dismiss_notification": "_request_notifications_refresh",
    f"{WATCHLISTS_COLLECTIONS_PATH}::WatchlistsCollectionsScreen."
    "_generate_briefing": "_request_briefings_refresh",
    f"{WATCHLISTS_COLLECTIONS_PATH}::WatchlistsCollectionsScreen."
    "_cast_script": "_request_briefings_refresh",
    f"{WATCHLISTS_COLLECTIONS_PATH}::WatchlistsCollectionsScreen."
    "_synthesize_audio": "_request_briefings_refresh",
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


def _self_call_name(node: ast.Call) -> str | None:
    """Return the method name only for a direct ``self.method(...)`` call."""
    func = node.func
    if not isinstance(func, ast.Attribute):
        return None
    if not isinstance(func.value, ast.Name) or func.value.id != "self":
        return None
    return func.attr


def _mutation_refresh_violations_in_tree(
    tree: ast.Module,
    *,
    forbidden_loaders: set[str] | frozenset[str],
    owner_helpers: dict[str, str],
) -> dict[str, str]:
    """Return refresh-contract failures keyed by qualified function owner.

    A row aggregates both halves of the contract so a current inline refresh
    reports one mutation owner, not one raw-await row plus a second missing-
    helper row. The caller supplies the path-specific raw-loader set and the
    owner inventory, keeping this helper path-free for synthetic AST tests.
    """
    owners = _owner_index(tree)
    issues: dict[str, list[str]] = {}
    helper_counts = dict.fromkeys(owner_helpers, 0)

    for node in ast.walk(tree):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            loader = _self_call_name(node.value)
            if loader in forbidden_loaders:
                owner = owners.get(node.lineno, "<module>")
                issues.setdefault(owner, []).append(
                    f"await self.{loader}(...) is forbidden; dispatch through "
                    "the loader-group refresh helper"
                )

        if not isinstance(node, ast.Call):
            continue
        owner = owners.get(node.lineno, "<module>")
        expected_helper = owner_helpers.get(owner)
        if expected_helper is not None and _self_call_name(node) == expected_helper:
            helper_counts[owner] += 1

    for owner, expected_helper in owner_helpers.items():
        count = helper_counts[owner]
        if count != 1:
            issues.setdefault(owner, []).append(
                f"expected exactly one self.{expected_helper}(...) call; found {count}"
            )

    return {owner: "; ".join(issues[owner]) for owner in sorted(issues)}


def _mutation_refresh_contract_violations() -> list[str]:
    """Scan only the two TASK-19870 production modules."""
    by_path: dict[str, dict[str, str]] = {
        relative: {} for relative in FORBIDDEN_MUTATION_LOADERS_BY_PATH
    }
    for site, helper in MUTATION_REFRESH_HELPER_INVENTORY.items():
        relative, owner = site.split("::", maxsplit=1)
        by_path[relative][owner] = helper

    violations: list[str] = []
    for relative, forbidden_loaders in FORBIDDEN_MUTATION_LOADERS_BY_PATH.items():
        path = PACKAGE_ROOT.parent / relative
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for owner, detail in _mutation_refresh_violations_in_tree(
            tree,
            forbidden_loaders=forbidden_loaders,
            owner_helpers=by_path[relative],
        ).items():
            violations.append(f"{relative}::{owner}: {detail}")
    return violations


def _violations_in_tree(tree: ast.Module) -> list[tuple[int, str, str]]:
    """Return ``(lineno, owner_qualname, detail)`` for every ungrouped
    exclusive worker schedule in an already-parsed module.

    Path-free, so tests can drive it with synthetic source rather than only
    with real package files.
    """
    owners: dict[int, str] | None = None
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
        # A `group=` whose value cannot be read from the AST (a constant, an
        # f-string, an attribute) is accepted: the site has named *something*,
        # and the name is the reviewable artefact. But three literal values are
        # NOT names, and each defeats the guard while looking like it satisfies
        # it, so they are reported as if no group had been given at all:
        #
        # * `group="default"` IS the shared bucket this whole guard exists to
        #   keep sites out of -- it is byte-for-byte what omitting `group=`
        #   produces (`Worker.__init__`/`work()`/`run_worker` all default to
        #   the string "default"). It is the likeliest wrong "fix" for a guard
        #   failure, and this repo already spells that shape by hand at
        #   `UI/Console_Modules/agent.py` (non-exclusive there, so benign).
        # * `group=""` / `group=None` are falsy, and `WorkerManager.add_worker`
        #   reads `if exclusive and worker.group:` -- a falsy group skips
        #   `cancel_group` entirely, so the site silently gets NO exclusivity
        #   despite asking for it. That is a different bug, not a fix.
        group_is_a_name = group is not None and not (
            isinstance(group, ast.Constant) and group.value in ("default", "", None)
        )
        if group_is_a_name:
            continue

        shown = "**kwargs" if exclusive is None else ast.unparse(exclusive)
        if group is None:
            detail = f"{scheduler}(exclusive={shown}) with no group="
        else:
            detail = (
                f"{scheduler}(exclusive={shown}, group={ast.unparse(group)}) "
                "-- that is not a group name"
            )
        # Built only once a violation exists. Attributing every line in the
        # package to its owning function is the single most expensive step in
        # the sweep, and the overwhelming majority of files have nothing to
        # attribute.
        if owners is None:
            owners = _owner_index(tree)
        violations.append(
            (
                node.lineno,
                owners.get(node.lineno, "<module>"),
                detail,
            )
        )
    return violations


@functools.lru_cache(maxsize=1)
def _scan_package() -> tuple[tuple[Path, int, str, str], ...]:
    """Every flagged site in the package, found in one cached pass.

    Returns ``(relative_path, lineno, owner_qualname, detail)`` rows *before*
    the allowlist is applied, so both the guard and the allowlist-staleness
    check can be answered from the same scan.
    """
    rows: list[tuple[Path, int, str, str]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if not _SCHEDULER_TOKEN.search(source):
            continue
        relative = path.relative_to(PACKAGE_ROOT.parent)
        for lineno, owner, detail in _violations_in_tree(ast.parse(source)):
            rows.append((relative, lineno, owner, detail))
    return tuple(rows)


def _all_site_keys() -> set[str]:
    return {f"{relative}::{owner}" for relative, _l, owner, _d in _scan_package()}


def test_no_ungrouped_exclusive_workers() -> None:
    """Every exclusive worker in the package names the group it may cancel."""
    assert PACKAGE_ROOT.is_dir(), f"package root not found: {PACKAGE_ROOT}"
    violations = [
        f"{relative}:{lineno} ({owner}): {detail}"
        for relative, lineno, owner, detail in _scan_package()
        if f"{relative}::{owner}" not in DEFAULT_GROUP_ALLOWLIST
    ]
    assert not violations, (
        "exclusive=True scheduled without an explicit group= lands in the "
        'shared "default" group, where it cancels every other ungrouped '
        "exclusive worker on the same node (cancel_group filters on "
        "(node, group) and never on name=). Name the group after the WORK "
        "being done, or -- if default-group mutual exclusion really is what "
        "the site wants -- add it to DEFAULT_GROUP_ALLOWLIST with the reason:\n"
        + "\n".join(violations)
    )


def test_mutation_refreshes_dispatch_through_loader_group() -> None:
    """Mutation workers dispatch refreshes through their loader-group helper."""
    violations = _mutation_refresh_contract_violations()
    assert not violations, (
        "mutation-triggered refreshes must not await raw loaders outside their "
        "exclusive loader group. Replace each raw await with exactly one call "
        "to the inventoried dispatch helper:\n" + "\n".join(violations)
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


def test_group_default_does_not_satisfy_the_guard() -> None:
    """`group="default"` is the shared bucket, not an escape from it.

    Found in review: the guard originally accepted any `group=` node without
    reading its value, so the likeliest wrong "fix" for a guard failure --
    pasting in `group="default"` -- silenced the guard while changing the
    runtime behaviour not at all. `Worker`, `work()` and `run_worker` all
    default `group` to the literal string `"default"`, so this call is
    byte-for-byte the ungrouped one.
    """
    tree = ast.parse(
        "from textual.widget import Widget\n\n"
        "class ScratchWidget(Widget):\n"
        "    def _start(self) -> None:\n"
        "        self.run_worker(self._work, exclusive=True, group='default')\n"
    )
    violations = _violations_in_tree(tree)
    assert len(violations) == 1
    assert "not a group name" in violations[0][2]


def test_falsy_group_does_not_satisfy_the_guard() -> None:
    """`group=""`/`group=None` silently disable exclusivity altogether.

    `WorkerManager.add_worker` reads ``if exclusive and worker.group:`` before
    calling ``cancel_group``, so a falsy group means the site asked for
    exclusivity and received none -- a different bug wearing the fix's clothes.
    """
    for literal in ("''", "None"):
        tree = ast.parse(
            "from textual.widget import Widget\n\n"
            "class ScratchWidget(Widget):\n"
            "    def _start(self) -> None:\n"
            f"        self.run_worker(self._work, exclusive=True, group={literal})\n"
        )
        violations = _violations_in_tree(tree)
        assert len(violations) == 1, literal
        assert "not a group name" in violations[0][2], literal


def test_non_literal_group_is_accepted() -> None:
    """A constant or f-string group is a name; the guard must not reject it.

    33 real sites name their group through a module constant or an f-string
    (`f"console-run-{session_id}"`). Those are named groups and stay clean --
    only the three literals above are rejected.
    """
    tree = ast.parse(
        "from textual.widget import Widget\n"
        "GROUP = 'scratch-load'\n\n"
        "class ScratchWidget(Widget):\n"
        "    def _start(self, key) -> None:\n"
        "        self.run_worker(self._a, exclusive=True, group=GROUP)\n"
        "        self.run_worker(self._b, exclusive=True, group=f'scratch-{key}')\n"
    )
    assert _violations_in_tree(tree) == []


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


def test_prefilter_admits_every_flagged_form() -> None:
    """The cheap text prefilter must never hide a form the AST would flag.

    Qodo's third finding on PR #1951 asked for a speed-up. Making a guard fast
    by making it blind is the failure mode, so the prefilter is pinned here
    against every shape the other tests in this file assert on -- decorator,
    multi-line, positional, ``name=``-only, ``group="default"``, falsy group,
    and the ``**kwargs`` spread the guard fails closed on. Each source is
    checked twice: the AST must flag it, and the prefilter must admit it for
    parsing in the first place.
    """
    flagged_sources = [
        # decorator form
        "from textual import work\n"
        "class W:\n"
        "    @work(exclusive=True, thread=True)\n"
        "    def _save(self): ...\n",
        # multi-line call form
        "class W:\n"
        "    def _s(self):\n"
        "        self.run_worker(\n"
        "            self._f(),\n"
        "            exclusive=True,\n"
        "        )\n",
        # name= mistaken for scoping
        "class W:\n"
        "    def _s(self):\n"
        "        self.run_worker(self._f, exclusive=True, name='search')\n",
        # group='default' is the shared bucket
        "class W:\n"
        "    def _s(self):\n"
        "        self.run_worker(self._f, exclusive=True, group='default')\n",
        # falsy group silently disables exclusivity
        "class W:\n"
        "    def _s(self):\n"
        "        self.run_worker(self._f, exclusive=True, group=None)\n",
        # **kwargs spread: unprovable, so failed closed
        "class W:\n    def _s(self, **kw):\n        self.run_worker(self._f, **kw)\n",
        # positional exclusive, no group
        "class W:\n"
        "    def _s(self):\n"
        "        self.run_worker(self._f, 'nm', None, 'desc', True, True, True)\n",
    ]
    for source in flagged_sources:
        assert _violations_in_tree(ast.parse(source)), f"AST missed:\n{source}"
        assert _SCHEDULER_TOKEN.search(source), (
            "the prefilter would have skipped a file the AST flags:\n" + source
        )


def test_prefilter_only_skips_files_with_no_scheduler_token() -> None:
    """Every package file the prefilter skips really has no scheduler call.

    Re-parses the *skipped* files -- the ones the fast path never looks at --
    and asserts the AST finds nothing there either. This is the check that
    would catch a prefilter regex that got too clever.
    """
    skipped_with_violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if _SCHEDULER_TOKEN.search(source):
            continue
        if _violations_in_tree(ast.parse(source)):
            skipped_with_violations.append(str(path))
    assert not skipped_with_violations, (
        "the prefilter skipped files that do contain flagged worker sites:\n"
        + "\n".join(skipped_with_violations)
    )


def test_scan_is_a_single_cached_pass() -> None:
    """The package is read once per session, not once per assertion.

    The sweep used to walk and parse all of ``tldw_chatbook/`` twice -- once in
    ``test_no_ungrouped_exclusive_workers`` and again in ``_all_site_keys()``
    for the allowlist-staleness check. Both now answer from one cached scan.
    """
    first = _scan_package()
    assert _scan_package() is first, "the scan is being recomputed per call"
    # Deliberately no `cache_clear()`: forcing a rebuild here would spend the
    # very seconds this change exists to save.
    assert _scan_package.cache_info().misses <= 1, _scan_package.cache_info()


def test_awaited_mutation_loader_is_flagged() -> None:
    """An awaited raw loader bypasses the helper's exclusive worker group."""
    tree = ast.parse(
        "class ScratchScreen:\n"
        "    async def _mutate(self):\n"
        "        await self.load_tasks()\n"
    )

    violations = _mutation_refresh_violations_in_tree(
        tree,
        forbidden_loaders={"load_tasks"},
        owner_helpers={},
    )

    assert set(violations) == {"ScratchScreen._mutate"}
    assert (
        "await self.load_tasks(...) is forbidden" in violations["ScratchScreen._mutate"]
    )


def test_helper_dispatch_and_unrelated_loader_are_clean() -> None:
    """The required helper is enough; unrelated awaited loaders stay out of scope."""
    tree = ast.parse(
        "class ScratchScreen:\n"
        "    async def _mutate(self):\n"
        "        self._request_tasks_refresh()\n"
        "        await self.load_people()\n"
    )

    assert not _mutation_refresh_violations_in_tree(
        tree,
        forbidden_loaders={"load_tasks"},
        owner_helpers={"ScratchScreen._mutate": "_request_tasks_refresh"},
    )


def test_missing_or_wrong_refresh_helper_is_flagged() -> None:
    """Every inventoried mutation owner calls its one named helper exactly once."""
    for body in ("pass", "self._request_notifications_refresh()"):
        tree = ast.parse(
            f"class ScratchScreen:\n    def _mutate(self):\n        {body}\n"
        )

        violations = _mutation_refresh_violations_in_tree(
            tree,
            forbidden_loaders={"load_tasks"},
            owner_helpers={"ScratchScreen._mutate": "_request_tasks_refresh"},
        )

        assert (
            "expected exactly one self._request_tasks_refresh(...) call; found 0"
            in (violations["ScratchScreen._mutate"])
        )
