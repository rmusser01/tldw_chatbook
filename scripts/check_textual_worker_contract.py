#!/usr/bin/env python3
"""Guard: Textual's worker contract, and DOM lookups that resume after an await.

TASK-32800.4. Three of the four P0s in the 2026-09-17 core-runtime review
(`qa/core-code-review-2026-09-17/report.md`) were the same defect class, and
nothing in the repo could catch it:

* `Widgets/audio_troubleshooting_dialog.py` ran a plain ``def`` as an async
  worker. ``Worker._run_async`` raises ``WorkerError("Request to run a
  non-async function as an async worker")`` for a non-coroutine target,
  ``Worker._run`` catches that generically, and ``exit_on_error`` defaults to
  ``True`` -- so opening the dialog exited the whole application.
* `UI/MCP_Modules/mcp_inspector.py` dereferenced ``#mcp-adv-content`` after
  awaiting a section load. The sibling handler removes that subtree, and
  ``exclusive=True`` does not help (it cancels another worker in the same
  *group*, never the running one), so the await resumed into a removed
  subtree, raised ``NoMatches``, and again exited the application.
* `Widgets/Library/library_prompts_canvas.py` kept a flag describing a region
  that a recompose had removed, and the next key press dereferenced it.

Two rules, deliberately of different strengths:

W001 (hard gate, zero tolerance)
    ``run_worker(<target>)`` whose target statically resolves to a
    non-coroutine method, without ``thread=True``. This is precise: the whole
    package currently has zero violations, so any new one is a real defect
    and there is no allowlist to hide behind.

W002 (census ratchet, not a gate)
    A ``query_one``/``query_exactly_one`` that is reached after an ``await``
    inside an ``async def``, with no enclosing ``try`` **body**. "Body" is
    load-bearing: this check shipped treating any ``ast.Try`` ancestor as
    protection, but a lookup in an ``except``/``else``/``finally`` clause is a
    child of the same ``Try`` node while sitting outside the region its own
    handlers cover -- an exception there propagates straight out. That hid 50
    sites across 21 functions (269 reported vs 319 real) behind a green check.
    Whether this crashes depends on whether the awaited work can remove the
    subtree, which is not statically decidable -- there are 319 such sites
    today and the vast majority are fine. Failing on all of them would be
    exactly the guard that cries wolf and gets muted, which
    ``scripts/preflight.sh`` warns about in its own header. So the existing
    sites are recorded in a census and the check fails only when a site
    appears that is not in it. Shrinking the
    census is always allowed; growing it is a deliberate act.

    The census rows are a *baseline*, not an endorsement: they were captured
    mechanically and have not been individually reviewed.

Stdlib-only, like the other derived-artifact checkers, so it runs with no
dependency install.

Usage:
    python scripts/check_textual_worker_contract.py
    python scripts/check_textual_worker_contract.py --write   # re-pin W002
"""

from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = REPO_ROOT / "tldw_chatbook"
CENSUS = REPO_ROOT / "scripts" / "textual_await_dom_census.tsv"

#: Directories that never ship a Textual widget.
SKIP_PARTS = {".venv", "Third_Party", "__pycache__", "node_modules"}

#: W002 only looks at the packages that define screens and widgets. Service
#: and database modules have no DOM to lose.
UI_PACKAGES = {"UI", "Widgets"}

DOM_LOOKUPS = {"query_one", "query_exactly_one"}


def _source_files() -> list[Path]:
    return [
        path
        for path in sorted(PACKAGE.rglob("*.py"))
        if not SKIP_PARTS & set(path.parts)
    ]


def _rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _is_coroutine_def(node: ast.AST) -> bool:
    """True for ``async def``, and for ``@work``-decorated methods.

    A ``@work`` decorator returns a function that schedules a Worker; Textual
    accepts it either way, and ``inspect.iscoroutinefunction`` reports False on
    the decorated attribute, which is precisely the false positive that fooled
    the review's own first enumeration of this defect.
    """
    if isinstance(node, ast.AsyncFunctionDef):
        return True
    if not isinstance(node, ast.FunctionDef):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "attr", None) or getattr(target, "id", None)
        if name == "work":
            return True
    return False


def _is_property(node: ast.AST) -> bool:
    """A ``@property`` hands back a value we cannot resolve statically.

    ``UI/Console_Modules/agent.py`` has a property whose value is an ``async
    def`` on the screen; treating the property itself as the target reported a
    false positive, so these are skipped rather than guessed at.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    return any(
        getattr(decorator, "id", None) == "property"
        for decorator in node.decorator_list
    )


def _class_methods(cls: ast.ClassDef) -> dict[str, ast.AST]:
    return {
        node.name: node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _worker_target(call: ast.Call) -> ast.AST | None:
    """The callable ``run_worker`` will invoke, or None when unresolvable.

    ``run_worker(self.method())`` passes an already-created coroutine or Worker
    and is always valid; only a bare reference names something to be called.
    ``partial(self.method, ...)`` is unwrapped because it is the shape the MCP
    inspector uses.
    """
    if not call.args:
        return None
    target = call.args[0]
    if isinstance(target, ast.Call):
        func_name = getattr(target.func, "id", None) or getattr(
            target.func, "attr", None
        )
        if func_name == "partial" and target.args:
            target = target.args[0]
        else:
            return None
    if not isinstance(target, ast.Attribute):
        return None
    if not (isinstance(target.value, ast.Name) and target.value.id == "self"):
        return None
    return target


def _declares_thread(call: ast.Call) -> bool:
    return any(
        keyword.arg == "thread"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in call.keywords
    )


def collect_w001(tree: ast.Module, path: Path) -> list[str]:
    """Synchronous ``run_worker`` targets that do not declare ``thread=True``."""
    violations: list[str] = []
    for cls in [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]:
        methods = _class_methods(cls)
        for node in ast.walk(cls):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "run_worker"
            ):
                continue
            if _declares_thread(node):
                continue
            target = _worker_target(node)
            if target is None:
                continue
            method = methods.get(target.attr)
            if method is None:
                continue  # inherited or assigned elsewhere; not resolvable here
            if _is_property(method) or _is_coroutine_def(method):
                continue
            violations.append(
                f"{_rel(path)}:{node.lineno}\t{cls.name}.{target.attr}"
            )
    return violations


def collect_w002(tree: ast.Module, path: Path) -> list[str]:
    """DOM lookups reached after an await, with no enclosing ``try``."""
    if not UI_PACKAGES & set(path.parts):
        return []
    sites: list[str] = []
    for func in [
        node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
    ]:
        parents: dict[ast.AST, ast.AST] = {}
        for node in ast.walk(func):
            for child in ast.iter_child_nodes(node):
                parents[child] = node
        awaits = [
            node.lineno for node in ast.walk(func) if isinstance(node, ast.Await)
        ]
        if not awaits:
            continue
        first_await = min(awaits)
        for node in ast.walk(func):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in DOM_LOOKUPS
            ):
                continue
            if node.lineno <= first_await:
                continue
            guarded = False
            child: ast.AST = node
            cursor = parents.get(child)
            while cursor is not None and cursor is not func:
                # Only `Try.body` is covered by that `Try`'s own handlers. A
                # lookup in its `except`/`else`/`finally` is a *child* of the
                # same node but sits OUTSIDE the protected region -- an
                # exception there propagates straight out. Treating any `Try`
                # ancestor as protection hid 50 sites across 21 functions.
                # Keep ascending when it is not `body`: an OUTER try may still
                # legitimately cover this lookup.
                if isinstance(cursor, (ast.Try, ast.TryStar)) and any(
                    child is stmt for stmt in cursor.body
                ):
                    guarded = True
                    break
                child = cursor
                cursor = parents.get(cursor)
            if guarded:
                continue
            # Keyed by enclosing function, NOT by line: an edit anywhere above
            # a site shifts its line number, and a census that churns on every
            # unrelated edit is one people re-pin without reading. The count
            # still catches a NEW lookup added to an already-censused function.
            sites.append(f"{_rel(path)}::{func.name}")
    return sites


def _read_census() -> dict[str, int]:
    if not CENSUS.exists():
        return {}
    rows: dict[str, int] = {}
    for line in CENSUS.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, count = line.partition("\t")
        rows[key] = int(count) if count.strip().isdigit() else 1
    return rows


def _tally(sites: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in sites:
        counts[site] = counts.get(site, 0) + 1
    return counts


def _write_census(sites: list[str]) -> None:
    header = (
        "# Baseline census for W002: `query_one` reached after an `await` with no\n"
        "# enclosing `try`, in UI/ and Widgets/. Generated by\n"
        "# scripts/check_textual_worker_contract.py --write\n"
        "#\n"
        "# These rows are a BASELINE, not an endorsement: they were captured\n"
        "# mechanically and have not been individually reviewed. Whether one of\n"
        "# them can actually crash depends on whether the awaited work can remove\n"
        "# the subtree, which is not statically decidable -- see TASK-32800.1 for\n"
        "# one that could.\n"
        "#\n"
        "# Removing a row is always fine. Adding one is a deliberate act: prefer\n"
        "# guarding the lookup (`query()` + `first()`, or an enclosing `try`) or\n"
        "# re-checking that the widget is still mounted after the await.\n"
        "#\n"
        "# Rows are keyed by ENCLOSING FUNCTION, not by line number, so an edit\n"
        "# elsewhere in the file does not churn the census.\n"
        "#\n"
        "# path::async def\tunguarded lookups in it\n"
    )
    counts = _tally(sites)
    body = "\n".join(f"{key}\t{counts[key]}" for key in sorted(counts))
    CENSUS.write_text(header + body + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--write",
        action="store_true",
        help="re-pin the W002 census to what the tree currently contains",
    )
    args = parser.parse_args()

    w001: list[str] = []
    w002: list[str] = []
    for path in _source_files():
        try:
            with warnings.catch_warnings():
                # A few modules carry invalid escape sequences; they are a
                # separate finding, and their SyntaxWarnings are not this
                # check's output.
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue
        w001.extend(collect_w001(tree, path))
        w002.extend(collect_w002(tree, path))

    if args.write:
        _write_census(w002)
        print(f"W002 census re-pinned: {len(w002)} site(s) -> {_rel(CENSUS)}")
        if w001:
            print("W001 violations are NOT allowlistable; fix them:")
            for row in w001:
                print(f"  {row}")
            return 1
        return 0

    failed = False

    if w001:
        failed = True
        print(
            f"::error::{len(w001)} synchronous run_worker target(s) without thread=True."
        )
        print(
            "Textual's Worker._run_async raises WorkerError for a non-coroutine "
            "target, and exit_on_error defaults to True, so each of these exits "
            "the application when it runs:"
        )
        for row in w001:
            site, name = row.split("\t")
            print(f"  {site}  ({name})")
        print("Pass thread=True, or make the target a coroutine.")
        print()

    known = _read_census()
    current = _tally(w002)
    added = sorted(
        key for key, count in current.items() if count > known.get(key, 0)
    )
    if added:
        failed = True
        print(f"::error::{len(added)} new post-await DOM lookup(s) with no guard.")
        print(
            "A `query_one` that resumes after an `await` can find its subtree "
            "removed -- NoMatches propagates out of the worker and, with "
            "exit_on_error defaulting to True, exits the application "
            "(TASK-32800.1 was exactly this). Guard the lookup with `query()` + "
            "`first()`, or an enclosing `try`, or re-check the widget is still "
            "mounted after the await:"
        )
        for key in added:
            print(f"  {key}  ({known.get(key, 0)} in census, {current[key]} now)")
        print()
        print(
            "If the await genuinely cannot remove the subtree, re-pin the census "
            "with:  python scripts/check_textual_worker_contract.py --write"
        )
        print()

    if failed:
        return 1

    resolved = sum(
        max(0, count - current.get(key, 0)) for key, count in known.items()
    )
    note = f"; {resolved} baseline lookup(s) resolved" if resolved else ""
    print(
        f"textual worker contract: no synchronous run_worker targets; "
        f"{sum(current.values())} post-await DOM lookup(s) in "
        f"{len(current)} function(s), none new{note}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
