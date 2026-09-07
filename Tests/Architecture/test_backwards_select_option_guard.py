# test_backwards_select_option_guard.py
# Description: Guard against (value, label)-reversed Select option tuples (task-16841)
"""
TASK-16841: Textual's ``Select(options=...)`` (and ``Select.set_options()``)
takes an iterable of ``(label, value)`` tuples -- element 0 is what's
*rendered*, element 1 is what ``.value`` returns. Getting that order backwards
has shipped **six** times in this repo, always found by manual review, never
by tooling:

* TASK-15772 (PR #1691) -- six sites across ``UI/STTS_Window.py`` +
  ``Widgets/TTS/``.
* TASK-15991 (PR #1701) -- two sites in ``UI/ScraperBuilderWindow.py``.
* TASK-16841's own repo-wide sweep -- four more: ``UI/SiteConfigSettings.py``
  (``#auth-type-select``, the task's own headline bug -- ``auth_select.value
  = config.auth_type or "none"`` raised ``InvalidSelectValueError`` because
  the Select's real values were the display labels), ``Widgets/
  voice_profile_dialog.py`` (``#language-select`` -- same crash, reachable
  from Lab > Speech > Voice Cloning > New Profile), ``UI/
  Voice_Cloning_Window.py`` (``#test-profile-select`` -- no crash, but the
  dropdown showed the internal profile id instead of its display name and
  sent the wrong profile reference to the TTS backend), and
  ``UI/Study_Window.py`` (``#guide-topic-select`` -- an unconsumed but still
  visibly wrong dropdown, showing "new" instead of "New Topic").
* A post-commit review of this very task found a **fifth** site this
  guard's first cut still missed: ``UI/Study_Window.py``'s
  ``CourseCreationWidget.compose`` composed ``#course-level`` as
  ``[("beginner", "Beginner"), ("intermediate", "Intermediate"),
  ("advanced", "Advanced")]`` -- backwards, reachable (``StudyScreen`` is a
  live nav route), unconsumed (same cosmetic-only profile as
  ``#guide-topic-select``). It slipped through both the manual sweep AND
  the guard for the same reason -- see the new gap below. It was NOT fixed
  in this repo: an unrelated, already-merged TASK-16845 deleted the whole
  "Course Details" form (title/description/level/prerequisites feeding a
  dead ``#create-course-btn``) on ``origin/dev`` before this task's branch
  was rebased onto it, so ``#course-level`` no longer exists there --
  fixing it on this task's stale base would only be reverted by the
  rebase. Confirmed via ``git show origin/dev:tldw_chatbook/UI/
  Study_Window.py`` (``CourseCreationWidget.compose`` has no ``#course-
  level`` and no "Course Details" form at all, replaced by a comment
  explaining the removal).

Every one of those four new sites was *reachable* -- a nav-unreachable
Select is nice-to-have severity mitigation, not a reason a bug is allowed to
ship. This guard is the tooling TASK-16841 asked for so a seventh instance
cannot land undetected.

## What this catches

A ``Select(...)``/``.set_options(...)`` call whose option list is (or
resolves, through **one** same-scope ``name = [...]`` assignment, to) a
**literal list of 2-tuples of string constants**, where at least one pair
has:

* element 0 shaped like a bare machine token -- lowercase, snake_case or
  kebab-case (``^[a-z][a-z0-9_-]*$``), e.g. ``"basic"``, ``"en"``, ``"new"``;
* element 1 shaped like human display text -- contains a space, or is
  Title-Cased (``"Basic Auth"``, ``"English"``, ``"New Topic"``);
* the two are not the same word after normalizing case/separators (so
  ``("none", "None")`` is not flagged on its own -- it is exactly the
  label/value CONVENTION, just capitalized -- while ``("basic", "Basic
  Auth")`` alongside it in the same options list still is).

This is a heuristic over TEXT SHAPE, run against every current literal
Select/``set_options`` option list in this repo as part of TASK-16841's own
sweep: of the five new-and-newly-found backwards sites (four from the
original sweep, one -- ``#course-level`` -- from the follow-up review), it
correctly flagged three that had literal string options with at least one
non-uniformly-cased pair (``#auth-type-select``, ``#language-select``,
``#guide-topic-select``), and raised **zero** false positives across the
roughly 230 other literal option-tuple sites checked by hand in that sweep.
It does NOT catch ``#course-level`` (a uniformly same-word-cased list) or
``#test-profile-select`` (dynamic, non-literal options) -- see both gaps
below.

## What this does NOT catch (stated honestly, not aspirationally)

* **Options built from per-item data**, e.g.
  ``test_options.append((profile["name"], profile["display_name"]))`` --
  exactly the ``UI/Voice_Cloning_Window.py`` bug this same task's sweep
  found *by hand*. Neither element is a string literal, so there is no text
  shape to pattern-match; catching this needs semantic knowledge of what
  ``"name"``/``"display_name"`` mean, which is not statically decidable in
  general. Tuple order isn't decidable in general -- this guard trades
  completeness for zero false positives on the common, literal-tuple shape.
* Options where the label doesn't happen to contain a space or Title-Case
  (a single lowercase label word paired with a single lowercase value word
  reversed would look identical to a correct pair under this heuristic).
* Options resolved through more than one hop of indirection (a dict lookup,
  a helper function call, a loop-accumulator built across multiple
  statements) -- ``_resolve_pairs`` follows exactly one ``Name ->
  assignment`` hop and gives up rather than guess further.
* Non-string values (``Select.BLANK``, an enum's ``.value`` member, etc.) --
  these fail the "both elements are string constants" check and are
  silently skipped, never flagged, which is correct (nothing to compare).
* **An options list whose EVERY pair is the same-word-different-case
  shape.** The equality check that keeps ``("none", "None")`` from being
  flagged on its own (see "What this catches" above) is a per-PAIR
  exemption; it only stays safe in aggregate because a real backwards list
  usually has at least one pair -- like ``("basic", "Basic Auth")`` next to
  ``("none", "None")`` -- where label and value are different WORDS, not
  just different case, which still trips the shape check. When an entire
  list is uniformly same-word-cased (``[("beginner", "Beginner"),
  ("intermediate", "Intermediate"), ("advanced", "Advanced")]``), every
  pair individually satisfies the exemption and the whole site goes
  unflagged, backwards or not -- exactly what happened at ``UI/
  Study_Window.py``'s ``#course-level`` (see above). This is a MEASURED
  risk, not a guessed one: an independent same-word-different-case census
  across the whole package at the time of that review found **141** such
  pairs; **140 of 141 were Title-first/lowercase-second (the correct
  ``(label, value)`` convention)**, and ``#course-level`` was the lone
  outlier. A 140:1 signal means this gap is narrow in practice, but it is
  real and this guard cannot see inside a uniformly-cased list -- a case
  this stark (and this rare) is exactly why the ``ALLOWLIST`` mechanism
  and human code review still matter alongside this guard, not instead of
  it.

A guard that only catches the common, literal shape -- and never cries wolf
on the ~230 correct sites already in this repo -- is worth more than one
that tries to be exhaustive and breaks trust with a false positive. Coverage
gaps are closed by code review and the ``ALLOWLIST`` below, not by this
test trying to do static analysis it structurally cannot do.

## Allowlist

``ALLOWLIST`` holds ``(relative_path, lineno)`` pairs for sites reviewed and
confirmed to be intentional -- e.g. a Select whose "value" is genuinely
meant to be human text. It is empty at landing: TASK-16841's sweep found no
such site in this repo. Add an entry only for a real, reviewed exception,
with a comment explaining why.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "tldw_chatbook"

#: (relative_path, lineno) pairs reviewed and confirmed intentional. Empty
#: at landing -- see the module docstring's "Allowlist" section.
ALLOWLIST: frozenset[tuple[str, int]] = frozenset()

_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_-]*$")


def _looks_like_token(value: str) -> bool:
    """A bare machine token: lowercase, snake_case or kebab-case."""
    return bool(_TOKEN_RE.match(value))


def _looks_human(value: str) -> bool:
    """Human display text: contains a space, or is Title-Cased."""
    if not value:
        return False
    if " " in value:
        return True
    return value[0].isupper() and any(char.islower() for char in value[1:])


def _normalized(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _pair_is_backwards(elem0: str, elem1: str) -> bool:
    """True if (elem0, elem1) looks like (value, label) rather than (label, value)."""
    if not (_looks_like_token(elem0) and _looks_human(elem1)):
        return False
    return _normalized(elem0) != _normalized(elem1)


def _literal_pairs(node: ast.expr) -> list[tuple[str, str]] | None:
    """Resolve a literal list of 2-tuples of string constants, or None.

    Returns None (not "not backwards") for anything that isn't cleanly a
    literal list of ``(str_constant, str_constant)`` -- a dynamic option
    list is not this detector's job; see the module docstring.
    """
    if not isinstance(node, (ast.List, ast.Tuple)):
        return None
    pairs: list[tuple[str, str]] = []
    for elt in node.elts:
        if not (isinstance(elt, ast.Tuple) and len(elt.elts) == 2):
            return None
        first, second = elt.elts
        if not (
            isinstance(first, ast.Constant)
            and isinstance(first.value, str)
            and isinstance(second, ast.Constant)
            and isinstance(second.value, str)
        ):
            return None
        pairs.append((first.value, second.value))
    return pairs


def _collect_simple_assigns(body: list[ast.stmt]) -> dict[str, ast.expr]:
    """``name = <expr>`` / ``name: T = <expr>`` assignments directly in ``body``."""
    assigns: dict[str, ast.expr] = {}
    for stmt in body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            target = stmt.targets[0]
            if isinstance(target, ast.Name):
                assigns[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            if isinstance(stmt.target, ast.Name):
                assigns[stmt.target.id] = stmt.value
    return assigns


def _resolve_pairs(
    node: ast.expr, scope_assigns: dict[str, ast.expr]
) -> list[tuple[str, str]] | None:
    """Literal pairs for ``node``, following at most one ``Name`` indirection."""
    direct = _literal_pairs(node)
    if direct is not None:
        return direct
    if isinstance(node, ast.Name) and node.id in scope_assigns:
        return _literal_pairs(scope_assigns[node.id])
    return None


def _options_arg(call: ast.Call) -> ast.expr | None:
    """The options expression for a ``Select(...)``/``.set_options(...)`` call."""
    func = call.func
    is_select_ctor = (isinstance(func, ast.Name) and func.id == "Select") or (
        isinstance(func, ast.Attribute) and func.attr == "Select"
    )
    is_set_options = isinstance(func, ast.Attribute) and func.attr == "set_options"
    if is_select_ctor:
        for keyword in call.keywords:
            if keyword.arg == "options":
                return keyword.value
        if call.args:
            return call.args[0]
        return None
    if is_set_options and call.args:
        return call.args[0]
    return None


def _select_option_sites(tree: ast.Module) -> list[tuple[int, list[tuple[str, str]]]]:
    """``(lineno, pairs)`` for every resolvable Select/set_options options site."""
    sites: list[tuple[int, list[tuple[str, str]]]] = []

    def visit_scope(body: list[ast.stmt], inherited: dict[str, ast.expr]) -> None:
        scope_assigns = {**inherited, **_collect_simple_assigns(body)}
        synthetic = ast.Module(body=body, type_ignores=[])
        for node in ast.walk(synthetic):
            if isinstance(node, ast.Call):
                options_arg = _options_arg(node)
                if options_arg is not None:
                    pairs = _resolve_pairs(options_arg, scope_assigns)
                    if pairs:
                        sites.append((node.lineno, pairs))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_scope(node.body, scope_assigns)

    visit_scope(tree.body, {})
    return sites


def _violations_in_file(
    path: Path,
    *,
    package_root: Path = PACKAGE_ROOT,
    allowlist: frozenset[tuple[str, int]] = ALLOWLIST,
) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    relative = path.relative_to(package_root.parent).as_posix()
    seen: set[tuple[int, tuple[tuple[str, str], ...]]] = set()
    violations: list[str] = []
    for lineno, pairs in _select_option_sites(tree):
        key = (lineno, tuple(pairs))
        if key in seen:
            continue
        seen.add(key)
        if (relative, lineno) in allowlist:
            continue
        for elem0, elem1 in pairs:
            if _pair_is_backwards(elem0, elem1):
                violations.append(
                    f"{relative}:{lineno}: option pair {(elem0, elem1)!r} looks "
                    "(value, label) -- Textual Select options are (label, value)"
                )
                break
    return violations


def test_no_backwards_select_option_literals() -> None:
    """Every literal Select/set_options option list must be (label, value)."""
    assert PACKAGE_ROOT.is_dir(), f"package root not found: {PACKAGE_ROOT}"
    violations: list[str] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        violations.extend(_violations_in_file(path))
    assert not violations, (
        "Textual Select options must be (label, value), not (value, label) "
        "(TASK-16841 -- six prior instances of this bug class) -- found:\n"
        + "\n".join(violations)
    )


def test_guard_detects_a_synthetic_backwards_select(tmp_path: Path) -> None:
    """Proves the detector actually fires, per this repo's mutation-test convention."""
    package_root = tmp_path / "tldw_chatbook"
    ui_dir = package_root / "UI"
    ui_dir.mkdir(parents=True)

    (ui_dir / "offending_widget.py").write_text(
        "\n".join(
            (
                "from textual.widgets import Select",
                "",
                "class OffendingWidget:",
                "    def compose(self):",
                "        yield Select(",
                "            options=[",
                '                ("none", "None"),',
                '                ("basic", "Basic Auth"),',
                "            ],",
                '            id="auth-type-select",',
                "        )",
                "",
            )
        ),
        encoding="utf-8",
    )
    (ui_dir / "offending_via_variable.py").write_text(
        "\n".join(
            (
                "from textual.widgets import Select",
                "",
                "class OffendingViaVariable:",
                "    def compose(self):",
                "        language_options = [",
                '            ("en", "English"),',
                '            ("es", "Spanish"),',
                "        ]",
                "        yield Select(options=language_options, id=\"language-select\")",
                "",
            )
        ),
        encoding="utf-8",
    )
    (ui_dir / "clean_widget.py").write_text(
        "\n".join(
            (
                "from textual.widgets import Select",
                "",
                "class CleanWidget:",
                "    def compose(self):",
                "        yield Select(",
                "            options=[",
                '                ("None", "none"),',
                '                ("Basic Auth", "basic"),',
                "            ],",
                '            id="auth-type-select",',
                "        )",
                "",
            )
        ),
        encoding="utf-8",
    )
    (ui_dir / "dynamic_widget.py").write_text(
        "\n".join(
            (
                "from textual.widgets import Select",
                "",
                "class DynamicWidget:",
                "    def _update(self, profiles):",
                "        test_options = []",
                "        for profile in profiles:",
                '            test_options.append((profile["name"], profile["display_name"]))',
                "        select = self.query_one(\"#test-profile-select\", Select)",
                "        select.set_options(test_options)",
                "",
            )
        ),
        encoding="utf-8",
    )

    (ui_dir / "uniformly_cased_widget.py").write_text(
        "\n".join(
            (
                "from textual.widgets import Select",
                "",
                "class UniformlyCasedWidget:",
                "    def compose(self):",
                "        yield Select(",
                "            options=[",
                '                ("beginner", "Beginner"),',
                '                ("intermediate", "Intermediate"),',
                '                ("advanced", "Advanced"),',
                "            ],",
                '            id="course-level",',
                "        )",
                "",
            )
        ),
        encoding="utf-8",
    )

    violations: list[str] = []
    for path in sorted(package_root.rglob("*.py")):
        violations.extend(_violations_in_file(path, package_root=package_root))

    flagged_files = {violation.split(":", 1)[0] for violation in violations}
    assert "tldw_chatbook/UI/offending_widget.py" in flagged_files
    assert "tldw_chatbook/UI/offending_via_variable.py" in flagged_files
    assert "tldw_chatbook/UI/clean_widget.py" not in flagged_files
    # The dynamic (data-driven) site is the documented gap: no literal text
    # to pattern-match, so it must NOT be flagged (and must not crash).
    assert "tldw_chatbook/UI/dynamic_widget.py" not in flagged_files
    # The documented "uniformly same-word-cased list" gap (module docstring,
    # "What this does NOT catch") -- this is the real UI/Study_Window.py
    # #course-level shape, reproduced byte-for-byte. It is backwards
    # ((value, label) order throughout) but every pair individually passes
    # the same-word exemption, so the guard cannot see it. This assertion
    # pins that gap as MEASURED, not merely claimed in prose: if a future
    # change to `_pair_is_backwards`/`_normalized` accidentally started
    # catching this shape, this test would need updating -- a deliberate
    # tripwire, not an oversight.
    assert "tldw_chatbook/UI/uniformly_cased_widget.py" not in flagged_files


def test_allowlist_suppresses_a_reviewed_site(tmp_path: Path) -> None:
    """A ``(path, lineno)`` entry in the allowlist silences that one site."""
    package_root = tmp_path / "tldw_chatbook"
    ui_dir = package_root / "UI"
    ui_dir.mkdir(parents=True)

    (ui_dir / "allowlisted_widget.py").write_text(
        "\n".join(
            (
                "from textual.widgets import Select",
                "",
                "class AllowlistedWidget:",
                "    def compose(self):",
                "        yield Select(",
                '            options=[("basic", "Basic Auth")],',
                '            id="auth-type-select",',
                "        )",
                "",
            )
        ),
        encoding="utf-8",
    )
    target = ui_dir / "allowlisted_widget.py"
    lineno = next(
        i + 1
        for i, line in enumerate(target.read_text().splitlines())
        if "Select(" in line
    )

    without_allowlist = _violations_in_file(target, package_root=package_root)
    assert without_allowlist, "fixture must be a real violation before allowlisting"

    with_allowlist = _violations_in_file(
        target,
        package_root=package_root,
        allowlist=frozenset({("tldw_chatbook/UI/allowlisted_widget.py", lineno)}),
    )
    assert with_allowlist == []


def test_current_repo_has_no_allowlist_entries() -> None:
    """The allowlist is a documented escape hatch, not somewhere bugs hide.

    TASK-16841's sweep found zero genuinely intentional (value, label)
    Selects. If this ever needs an entry, it must come with a reviewed
    reason in the module docstring, not accumulate silently.
    """
    assert ALLOWLIST == frozenset()


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
