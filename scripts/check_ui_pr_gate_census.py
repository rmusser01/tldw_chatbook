#!/usr/bin/env python3
"""Guard: the Tests/UI slice that gates pull requests can only grow.

TASK-32908. Before this, ``Tests/UI`` -- 25,917 collected tests, the largest
single directory in the tree -- gated **nothing** on a pull request:

* ``.github/workflows/test.yml`` does have a 12-shard ``ui-tests`` job, but
  that workflow is ``on: push: branches: ["main"]`` plus
  ``workflow_dispatch``. It does not accept ``pull_request`` events at all,
  so neither its core shards nor its UI shards are a PR gate.
* ``.github/workflows/nightly-deep.yml`` runs ``pytest ./Tests/`` on a
  schedule, which is where the UI suite was assumed to be covered. It is
  not: with no ``--continue-on-collection-errors`` a single bad import
  aborts the whole session, and it has been doing exactly that -- measured
  on run 35706024071 (2026-09-22), ``collected 101851 items / 1 error``
  then ``Interrupted: 1 error during collection``, **zero tests executed**,
  six consecutive nights.

So the only PR gate is ``derived-artifacts.yml``, and the cost of that gap
is not theoretical. ``Tests/UI/test_console_library_tool_setting.py``
asserted ``service._collections is app.local_library_collections_service``
against a ``LocalLibraryToolService`` that has had no ``_collections``
attribute since 5dd1077df6 retired it. The assertion could not pass. It sat
red and blocked nothing -- and the sibling copy of that same factory in
``Chat/console_runtime.py`` went on passing a now-rejected
``collections_service=`` keyword, i.e. a live ``TypeError`` on the
Console-direct Library tool path, for three weeks.

The full directory cannot go in the PR gate: measured at 25,917 tests and
~5.5 CPU-hours, against a fast lane budgeted in minutes. What goes in is a
**verified-green subset**, listed one path per line in
``scripts/ui_pr_gate_census.txt`` and run by the ``ui-fast-lane`` job.

This checker is what stops that subset from quietly evaporating. The tests
themselves are the ratchet on *behaviour* -- a censused file that goes red
turns the job red, which is the whole point. What tests cannot catch is the
census being edited instead of the bug:

* a listed path that no longer exists (renamed or deleted) makes pytest
  collect fewer files while still exiting 0 -- a gate that silently tests
  less than it claims;
* deleting lines is the path of least resistance when a censused file goes
  red, and nothing else would notice.

Hence: every listed path must exist, no duplicates, and the census may
never fall below ``MINIMUM_FILES``. Growing it is free; shrinking it
requires editing this file, which is the review checkpoint.

An entry is a whole file or, since TASK-33621.27, one test's pytest node id
(``Tests/UI/test_x.py::test_name`` or ``...::test_name[param]``). Node ids let
a P0 regression test whose file is too slow for the lane be gated on its own,
instead of the whole file staying ungated. A node id entry must name a test
its file still defines (a renamed test would otherwise make pytest refuse the
whole shard), may not contain whitespace (the lane reads the census one entry
per line), and may not sit beside its own file as a whole entry (pytest
collapses overlapping arguments, ADR-103).

Exits 0 when the census is intact, 1 otherwise.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CENSUS_PATH = REPO_ROOT / "scripts" / "ui_pr_gate_census.txt"

#: Ratchet floor. Raise it when the census grows; never lower it without
#: saying why in the commit message. This is deliberately a literal rather
#: than "len(census) at HEAD" -- a floor derived from the file it guards
#: guards nothing.
# TASK-32908 set this to 120 (120 files / 851 tests, verified green).
# TASK-32899 lowered it to 118: Tests/UI/test_media_handoffs.py and
# Tests/UI/test_media_v88_simple.py were removed because their SUBJECT was
# removed -- both import tldw_chatbook.UI.MediaWindow_v2 (and V88), which
# that task deleted as dead code. This is the one shrink the floor is not
# meant to stop: no behaviour went uncovered, the covered thing is gone.
# TASK-33621.15 raised it to 119: Tests/UI/test_console_session_tab_close.py
# (8 private-profile tests, ~90 s serial) guards the Console tab close that
# stayed broken for three weeks because no PR gate ran its routing tests.
# TASK-33622.10 raised it to 120: Tests/UI/test_quit_prompt_vanish.py gates
# the Ctrl+Q orphaned-prompt hang, which no other PR-gated test would catch.
# TASK-33661 raised it to 121: Tests/UI/test_console_turn_resend_ui.py pins
# the Resend action row, its `r` key, the in-flight guard, and a real-Console
# click and keypress through both Resend paths.
# Roleplay frame B0 raised it to 125: test_adaptive_pane_shell.py,
# test_destination_rail_row.py and test_base_app_screen_tab_region.py gate the
# shared pane shell, rail-row fitting and the behaviour-neutral Tab region.
# The mounted every-route Tab test (test_base_app_screen_tab_region_routes.py,
# ~65 s) stays out of the fast lane; the B0 gate run executes it on both arms.
# TASK-33006 raised it to 130: the five Chat settings files (core-first,
# disclosures, hidden fields, model change, saved defaults; ~5.3 min serial)
# gate the redesigned modal, now inside TASK-34353 sharded lane.
# Retain the existing Console pending-kind/compact approval floor increment
# (+3 over dev), alongside every incoming settings and existing UI census entry.
# TASK-33622.15 raised it to 135: test_close_under_quit_question.py and
# test_modal_quit_in_flight_hooks.py pin Ctrl+Q's still-working answer and a
# dialog that finishes under "Quit while still working?" closing after Wait
# (~35 s serial together). The PR's two real-app quit files
# (test_app_quit_in_flight_modals.py ~2.7 min, test_console_video_picker_cancel.py
# ~1 min locally) stay out until the lane has a shard with room for them.
# TASK-33003.20 adds the production approval batch geometry guard.
# TASK-34000.1 raised it to 139: the census held 138 files on dev (two past
# the 136 floor), and Tests/UI/test_library_quit_guard.py joins it. It is a
# deliberately lean core: 3 real-app boots, about 35-45 s locally; the typed
# tail and a new note reach the DB before exit; a refused save keeps the
# typist in place and Ctrl+Q asks. The other quit variants live in
# test_library_quit_guard_extended.py, outside this lane: it was near its
# 20-minute cap when they were written (TASK-34353 has since sharded it).
# TASK-34000.2 raised it to 140: Tests/UI/test_library_notes_sync_attention.py
# is one real-app boot over a real lasting-sync root wedged the way review
# finding N-02 left it: the tree row, the Notes list and the editor say "needs
# attention", and the real Recovery button heals the folder with no
# "RuntimeError". Measured 24 s wall (14 s call) alone and 56 s wall under
# local load, against the lane's 60 s per-file rule -- keep it a SINGLE test;
# further attention variants go in non-gated files.
# Re-measured at 142 after rebasing onto dev (138 files there + this branch's
# four): TASK-34000.3's test_library_export_replace_confirm.py (two real-app
# boots: a note and a prompt export ask before replacing) and the TASK-32633
# slice's test_library_notes_sync_delete_restore.py (one boot: Delete holds the
# synced folder, Undo returns it) joined the census without a floor bump of
# their own, which left them free to be deleted unnoticed.
# The wave's final review raised it to 143 (C1):
# Tests/UI/test_library_note_autosave_recovers.py is two real-app boots, about
# 20 s locally. A burst whose max wait had run out armed a 0 s timer, which
# Textual never fires, so autosave stayed dead after Keep editing on a refused
# quit and after a rail switch away and back. The unit test that should have
# caught it asserted the 0.0 against a fake ``set_timer``; these read the row.
# TASK-34100.5 raised it to 146 (dev's 143 plus its three files): the
# first-run handoff's mounted guards -- Console's real first mount warns
# nothing (test_console_first_chat_first_mount.py), toasts clear the nav and
# chips at 120x40 (test_console_toast_clears_header.py), and a turn finishing
# in the visible tab raises no hidden notice (test_console_visible_turn_
# attention.py). About 3.5 min serial under load average 30 locally.
# TASK-33620.9 adds the private-profile mounted rename publication regressions.
# TASK-31245 adds the private-profile hydration and handle-ownership regressions.
# TASK-31966 adds the mounted recovery-bar idempotence/geometry regressions.
# TASK-31966 adds the shared Send-reason size idempotence/resize regressions.
# TASK-34100.16 raised it to 151 (dev's floor plus its one file):
# Tests/UI/test_backup_restore_setup_entry.py -- setup's Restore entry opens
# on Inspect, names the format, and explains a settings file, a folder and a
# disabled Create.
# TASK-33620.5 raised it to 155: dev's census held 154 files (three above the
# 151 floor) and Tests/UI/test_console_send_acknowledgement.py joins it. It is
# the lean core -- Enter's "Sending..." frame ordering at 80x24 plus the
# acknowledgement's own rules, 9 tests, about 10 s locally. The mounted
# variants stay in test_console_send_acknowledgement_extended.py, outside the
# lane: the whole set ran 58-110 s serially, over the 60 s per-file rule. An
# earlier cut of that PR kept the file out, saying it pushed shard 3 past the
# 20-minute cap. It did not: shard 3 timed out again with the file removed,
# and none of that shard's 581 tests ran the acknowledgement. The shard was
# at capacity, which the fourth shard (TASK-34353, a920bfe149) fixed.
# TASK-34000.7 (2026-10-08) raised it to 156:
# Tests/UI/test_library_notes_tree_scrolls_wide.py -- the wide Library Notes
# tree is a scroll owner and a Down walk keeps the focused row in view
# (120x36/160x45 plus the 80x24/100x30 compact pins), 6 tests, about
# 26 s locally. The slower variants (200x50/235x52, wheel events, the
# breakpoint round trip, the Trash opener) stay in
# test_library_notes_tree_scrolls_wide_extended.py, outside the lane.
# TASK-34000.13 (2026-10-08) raised it to 157:
# Tests/UI/test_library_note_delete_prompt_visible.py -- Info > Delete shows
# the whole prompt (copy, Cancel, Delete) inside the Info box at 120x36 and
# 80x24, focus lands on a visible Cancel, Tab/Shift+Tab stay inside, and
# Cancel restores Info's scroll; 4 tests, about 14 s locally. The slower
# variants (160x30/160x45/235x52, the resize while open, the real-DB
# "Linked from" case and the DB-level Cancel/Delete checks) stay in
# test_library_note_delete_prompt_visible_extended.py, outside the lane.
# TASK-34000.8 (2026-10-09) raised it to 158:
# Tests/UI/test_library_note_header_fits_pane.py -- the note editor's Save
# and the whole "Use in Console" have a region inside the editor pane at
# 120x36 and 160x45, Discard new note is whole and its hiding never moves
# the mode row, widening 119 -> 235 never hides a header control, the save
# state is never a one-column strip, and F6/Tab/the footer only name a Save
# the user can see; 6 tests, about 22 s locally. The slower variants (140x40,
# 200x50 and 235x52 with the rail open, the long title, 200x24, the
# 235 -> 120 -> 235 round trip, the delete prompt while stacked) stay in
# test_library_note_header_fits_pane_extended.py, outside the lane.
# TASK-34000.25 (2026-10-09) raised it to 159:
# Tests/UI/test_library_rail_switch_keeps_reader_and_note.py -- a rail round
# trip keeps the open Media item (tab, scroll, the row's loaded marker) and a
# New-note note with autosaved text; a dirty note switched inside the
# debounce comes back with its text, the next autosave saves it and the DB
# version equals the snapshot's at every step; `n` in the Reader creates a
# note titled after the document whose first line is the media:// source
# link; 4 tests, about 29 s locally. The slower arms (the Info tab, the real
# external-edit conflict, `n` inside Find, the server-item refusal, the
# untouched-blank GC, the vetoed title, Back at 100x30, 120x36) stay in
# test_library_rail_switch_keeps_reader_and_note_extended.py, outside the lane.
# Roleplay frame B1 raised it to 163: test_roleplay_frame_state.py,
# test_roleplay_stylesheet.py and test_workbench_fitted_text.py gate the pure
# header state, the lazy Roleplay sheet's ownership and FittedText; TASK-34400's
# test_roleplay_hostile_text_surfaces.py (no lane ran it; B1 re-pins one of
# its tests) gates the widget-level hostile-text sinks. B1's mounted Roleplay
# files are bootstrap-profile and run in the PR Fast Lane's
# admission-sensitive step instead (TASK-32873).
# TASK-33621.27 (2026-10-10) raised it by 12, to 175: the Console review's P0
# regression tests (Save .md, Choose folder, the keep-alive's Ctrl+Q, Stop and
# the composer buttons) had run in no PR lane since they merged. Their files
# measured 148-942 s each under load, so they come in as 12 node-id entries --
# the first entries that are not whole files; see the module docstring. A
# node id counts as one entry toward this floor, like a file.
MINIMUM_FILES = 175


def read_census(path: Path) -> list[str]:
    """Read the census, dropping comments and blank lines.

    Args:
        path: The census file, one `Tests/UI/...` path per line.

    Returns:
        The listed paths in file order, comments and blanks removed.
    """
    """Return the census entries, in file order, ignoring blanks/comments."""
    entries: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


def shard(entries: list[str], index: int, total: int) -> list[str]:
    """Pick one UI Fast Lane shard: every `total`-th entry from `index`.

    Round-robin, not contiguous halves (TASK-34353): the census's slow files
    sit together (the Console cluster), so a contiguous split measured 14.2
    vs 2.3 min where round-robin gives 6.8 vs 9.7. Each shard is still a
    subsequence of the census, so census order holds inside it.

    Args:
        entries: The census, in file order.
        index: This shard, 0-based (`strategy.job-index`).
        total: The shard count (`strategy.job-total`).

    Returns:
        This shard's paths, in census order.

    Raises:
        ValueError: When `index` is not in `range(total)`.
    """
    if not 0 <= index < total:
        raise ValueError(f"shard index {index} is not in range({total})")
    return entries[index::total]


class _Unresolvable(Exception):
    """A parametrize id that cannot be worked out without importing the test."""


_PRIMITIVE = (str, int, float, bool, type(None))


def _module_values(tree: ast.Module) -> dict[str, ast.AST]:
    """Module-level ``NAME = <expr>`` assignments and def/class names."""
    values: dict[str, ast.AST] = {}
    for item in tree.body:
        if isinstance(item, ast.Assign) and len(item.targets) == 1:
            target = item.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = item.value
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            if item.value is not None:
                values[item.target.id] = item.value
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            values[item.name] = item
    return values


def _elements(node: ast.AST, scope: dict[str, ast.AST]) -> list[ast.AST]:
    """The argvalues sequence, as AST nodes, in iteration order."""
    if isinstance(node, ast.Name) and node.id in scope:
        return _elements(scope[node.id], scope)
    if isinstance(node, (ast.List, ast.Tuple)):
        return list(node.elts)
    if isinstance(node, ast.Dict):  # iterating a dict yields its keys
        if any(key is None for key in node.keys):
            raise _Unresolvable("a ** splat in a dict literal")
        return list(node.keys)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"list", "tuple", "sorted"}
        and len(node.args) == 1
        and not node.keywords
    ):
        items = _elements(node.args[0], scope)
        if node.func.id == "sorted":
            keys = [_literal(item, scope) for item in items]
            if not all(isinstance(key, str) for key in keys):
                raise _Unresolvable("sorted() over non-strings")
            return [ast.Constant(key) for key in sorted(keys)]
        return items
    raise _Unresolvable(f"argvalues built by {ast.unparse(node)!r}")


def _literal(node: ast.AST, scope: dict[str, ast.AST]):
    if isinstance(node, ast.Name) and node.id in scope:
        return _literal(scope[node.id], scope)
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError, TypeError):
        raise _Unresolvable(f"value {ast.unparse(node)!r}") from None


def _value_id(node: ast.AST, argname: str, index: int, scope: dict[str, ast.AST]) -> str:
    """The id pytest gives one parameter value (``_idval`` in pytest 8)."""
    if isinstance(node, ast.Name) and isinstance(
        scope.get(node.id), (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    ):
        return node.id  # pytest uses a function's or class's __name__
    try:
        value = _literal(node, scope)
    except _Unresolvable:
        if isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set)):
            return f"{argname}{index}"
        raise
    if isinstance(value, str):
        return value.encode("unicode_escape").decode("ascii")
    if isinstance(value, _PRIMITIVE):
        return str(value)
    return f"{argname}{index}"


def _is_param_call(node: ast.AST) -> bool:
    func = node.func if isinstance(node, ast.Call) else None
    return isinstance(func, ast.Attribute) and func.attr == "param"


def _parametrize_ids(decorator: ast.Call, scope: dict[str, ast.AST]) -> list[str]:
    """The ids one ``@pytest.mark.parametrize`` call generates, in order."""
    args = list(decorator.args)
    keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
    argnames_node = args[0] if args else keywords.get("argnames")
    argvalues_node = args[1] if len(args) > 1 else keywords.get("argvalues")
    if argnames_node is None or argvalues_node is None:
        raise _Unresolvable("parametrize without literal argnames/argvalues")
    argnames = _literal(argnames_node, scope)
    if isinstance(argnames, str):
        argnames = [name.strip() for name in argnames.split(",") if name.strip()]
    argnames = list(argnames)
    explicit = None
    ids_node = args[3] if len(args) > 3 else keywords.get("ids")
    if ids_node is not None:
        if not isinstance(ids_node, (ast.List, ast.Tuple, ast.Name)):
            raise _Unresolvable(f"ids={ast.unparse(ids_node)}")
        explicit = [
            None if _literal(item, scope) is None else str(_literal(item, scope))
            for item in _elements(ids_node, scope)
        ]
    ids: list[str] = []
    for index, item in enumerate(_elements(argvalues_node, scope)):
        given = None
        values = item
        if _is_param_call(item):
            id_keyword = next((k.value for k in item.keywords if k.arg == "id"), None)
            if id_keyword is not None:
                given = _literal(id_keyword, scope)
            values = ast.Tuple(elts=list(item.args)) if len(argnames) > 1 else item.args[0]
        if explicit is not None and index < len(explicit) and explicit[index] is not None:
            given = explicit[index]
        if given is not None:
            ids.append(str(given))
            continue
        if len(argnames) == 1:
            ids.append(_value_id(values, argnames[0], index, scope))
            continue
        if isinstance(values, ast.Name) and values.id in scope:
            values = scope[values.id]
        if not isinstance(values, (ast.Tuple, ast.List)) or len(values.elts) != len(argnames):
            raise _Unresolvable(f"multi-argument value {ast.unparse(values)!r}")
        ids.append(
            "-".join(
                _value_id(element, name, index, scope)
                for element, name in zip(values.elts, argnames)
            )
        )
    if len(set(ids)) != len(ids):
        raise _Unresolvable("duplicate ids (pytest would suffix them)")
    return ids


def _is_parametrize(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "parametrize"
    )


def resolve_node(path: Path, node: str) -> str | None:
    """Check that `path` still defines the test a node id names.

    Static on purpose: the census check runs install-free, so it cannot import
    the test module. A bracketed id (``test_x[enter-80x24]``) is resolved from
    the function's literal ``@pytest.mark.parametrize`` decorators -- closest
    decorator first, as pytest joins them -- because a renamed id makes pytest
    exit 4 and the whole shard runs nothing (TASK-33621.27 review). An id built
    by code (a computed list, ``ids=lambda ...``) cannot be checked here, so it
    is refused: gate the whole test, or give it literal ids.

    Args:
        path: The test file.
        node: The node id after the file, ``test_x``, ``test_x[id]`` or
            ``TestClass::test_x[id]``.

    Returns:
        None when the node resolves; otherwise why it does not.
    """
    name_part, bracket, param = node.partition("[")
    names = name_part.split("::")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError) as error:
        return f"cannot read {path.name}: {type(error).__name__}"
    body: list[ast.stmt] = tree.body
    owners: list[ast.AST] = []
    match: ast.AST | None = None
    for depth, name in enumerate(names):
        last = depth == len(names) - 1
        kinds = (ast.FunctionDef, ast.AsyncFunctionDef) if last else (ast.ClassDef,)
        match = next(
            (item for item in body if isinstance(item, kinds) and item.name == name),
            None,
        )
        if match is None:
            return f"{path.name} does not define {name_part}"
        owners.append(match)
        body = getattr(match, "body", [])
    if not bracket:
        return None
    if not param.endswith("]"):
        return f"malformed node id {node!r}"
    wanted = param[:-1]
    module_marks = any(
        isinstance(item, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in item.targets)
        and any(_is_parametrize(sub) for sub in ast.walk(item.value))
        for item in tree.body
    )
    class_marks = any(
        _is_parametrize(decorator)
        for owner in owners[:-1]
        for decorator in getattr(owner, "decorator_list", [])
    )
    if module_marks or class_marks:
        return (
            f"parametrize id [{wanted}] cannot be resolved statically: module- or "
            "class-level parametrize. Gate the whole test instead."
        )
    scope = _module_values(tree)
    decorators = [d for d in reversed(match.decorator_list) if _is_parametrize(d)]
    if not decorators:
        return f"{name_part} is not parametrized, so it has no id [{wanted}]"
    try:
        combined = [""]
        for decorator in decorators:  # closest to the function first
            ids = _parametrize_ids(decorator, scope)
            combined = [f"{left}-{right}" if left else right for left in combined for right in ids]
    except _Unresolvable as reason:
        return (
            f"parametrize id [{wanted}] cannot be resolved statically ({reason}). "
            "Gate the whole test, or give its parametrize literal ids."
        )
    if wanted not in combined:
        shown = ", ".join(combined[:8]) + (" ..." if len(combined) > 8 else "")
        return f"{name_part} has no parametrize id [{wanted}] (it has: {shown})"
    return None


def defines_test(path: Path, node: str) -> bool:
    """Whether `path` defines the test (and parametrize id) a node id names."""
    return resolve_node(path, node) is None


def overlapping_targets(targets: list[str] | tuple[str, ...]) -> list[str]:
    """Pytest arguments that overlap another in the same invocation.

    Pytest collapses overlapping arguments (ADR-103): a file listed whole
    beside one of its node ids ran 1 of its 21 tests with everything green
    (TASK-33621.27 review). Shared by the census check and the Tests/CI pins
    on every PR-lane pytest step.

    Args:
        targets: One pytest invocation's targets (directories, files or node
            ids), in order.

    Returns:
        One problem line per overlap; empty when the targets are disjoint.
    """
    problems: list[str] = []
    seen: set[str] = set()
    wholes = {target.rstrip("/") for target in targets if "::" not in target}
    directories = {target for target in wholes if not target.endswith(".py")}
    for target in targets:
        if target in seen:
            problems.append(f"duplicate entry: {target}")
        seen.add(target)
        file_part = target.split("::", 1)[0].rstrip("/")
        if "::" in target and file_part in wholes:
            problems.append(
                f"node id overlaps a whole-file entry: {target}\n"
                f"    {file_part} is already listed whole; pytest collapses "
                "overlapping arguments (ADR-103). Drop one of the two."
            )
        for directory in sorted(directories):
            if file_part != directory and file_part.startswith(directory + "/"):
                problems.append(
                    f"{target} sits under the listed directory {directory}; "
                    "pytest collapses overlapping arguments (ADR-103)."
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    """Verify the PR-gate census is intact, or print one shard of it.

    Args:
        argv: Command-line arguments; ``--shard INDEX TOTAL`` prints that
            shard's paths, one per line, instead of checking the census.

    Returns:
        0 when every entry is unique, sits under `Tests/UI/`, names a file
        that exists (and, for a node id, a test that file defines), and the
        census has not shrunk below its floor; 1 otherwise.
    """
    if not CENSUS_PATH.exists():
        print(f"FAIL: census file is missing: {CENSUS_PATH}", file=sys.stderr)
        return 1

    entries = read_census(CENSUS_PATH)
    args = sys.argv[1:] if argv is None else argv
    if args[:1] == ["--shard"]:
        print("\n".join(shard(entries, int(args[1]), int(args[2]))))
        return 0
    problems: list[str] = []

    # Duplicates, a node id beside its own whole file, and anything under a
    # listed directory: the same helper pins every PR-lane pytest step.
    problems.extend(overlapping_targets(entries))
    for entry in entries:
        if not entry.startswith("Tests/UI/"):
            problems.append(f"not a Tests/UI path: {entry}")
            continue
        if any(character.isspace() for character in entry):
            problems.append(
                f"entry contains whitespace: {entry!r}\n"
                "    The lane reads one entry per line; gate a node id without "
                "spaces (pick another parametrization, or the whole test)."
            )
            continue
        file_part, _, node = entry.partition("::")
        if not (REPO_ROOT / file_part).is_file():
            problems.append(
                f"listed file does not exist: {entry}\n"
                "    A renamed or deleted censused file makes the gate collect "
                "fewer tests while still exiting 0.\n"
                "    Update the census to the new path, or remove the line and "
                "lower MINIMUM_FILES with a reason."
            )
            continue
        if not node:
            continue
        reason = resolve_node(REPO_ROOT / file_part, node)
        if reason is not None:
            problems.append(
                f"listed test does not resolve: {entry}\n"
                f"    {reason}\n"
                "    A renamed or deleted gated test makes pytest refuse the "
                "whole shard with 'not found' (exit 4, nothing runs).\n"
                "    Update the census to the test's current id, or remove the "
                "line and lower MINIMUM_FILES with a reason."
            )

    if len(entries) < MINIMUM_FILES:
        problems.append(
            f"census has shrunk: {len(entries)} entries, floor is {MINIMUM_FILES}.\n"
            "    If a censused file genuinely had to leave the gate, lower\n"
            f"    MINIMUM_FILES in {Path(__file__).name} in the SAME commit and say why.\n"
            "    Deleting the line on its own is how a gate rots to nothing."
        )

    if problems:
        print(
            f"FAIL: {CENSUS_PATH.relative_to(REPO_ROOT)} is not intact "
            f"({len(problems)} problem(s)):",
            file=sys.stderr,
        )
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    print(
        f"OK: {len(entries)} Tests/UI entries in the PR gate "
        f"(floor {MINIMUM_FILES}); every listed path exists."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
