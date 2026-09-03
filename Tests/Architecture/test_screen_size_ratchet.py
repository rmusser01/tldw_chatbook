"""A one-way ratchet on the screens being decomposed.

**Why this test exists.** Between 2026-08-02 and 2026-08-06 the Console
decomposition extracted ~4,900 lines out of `chat_screen.py` across two
reviewed waves — and the file ended up *larger* than when the work started,
because ~5,500 lines of concurrent feature work landed in it over the same
window. Every one of those lines went into the screen because the screen was
the path of least resistance: `UI/Console_Modules/` did not exist yet, or was
not yet the obvious place to put things.

Extraction alone therefore cannot win. This test makes the screen's current
size a *ceiling* rather than a waypoint, so a wave's gain cannot be silently
re-consumed by the next feature.

**This is a ratchet, not a limit.** The budgets below may only ever go DOWN.
When a wave lands, lower them to the new measurement in the same PR. If you
are here because CI failed, do NOT raise a number to make it pass — that
defeats the entire mechanism and re-opens the hole this test was written to
close.

**What to do when this test fails.** Your new Console code belongs in
`tldw_chatbook/UI/Console_Modules/`, next to `workspace.py`, `session.py`,
`hands_free.py` and `dictation.py`. `DESIGN.md` §7 states the rule and the
binding contract those controllers follow. A region that owns pixels becomes
a widget; behaviour and state with no region become a controller. Both take
their dependencies as named constructor callables rather than reaching back
through the screen.

Method count is tracked alongside line count deliberately: a screen can be
made shorter without being made simpler (by compressing bodies), and it is
the number of responsibilities the class holds — not its character count —
that made it hard to change.
"""

from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]

#: path -> (class name, max lines, max methods in that class).
#: LOWER these when a decomposition wave lands. Never raise them.
#: Lowered 2026-08-07 again by task-3023, which repointed the tests that
#: patched controllers through this module's namespace and so freed the
#: wave-4 re-export imports for deletion: 17,749 -> 17,727 lines (methods
#: unchanged at 593 — only imports went).
#: NOT lowered by wave 5's composer-keymap task, deliberately. That task
#: earned 42 lines, but between its start and its merge ~540 lines of feature
#: work landed in `chat_screen.py` on dev and this budget was ALREADY exceeded
#: (see task-3751). Lowering to the earned number would have been meaningless
#: and raising it to the measured number would have defeated the mechanism, so
#: the number is left exactly as dev has it. Lowered 2026-08-07 at the wave-4 close (controller wiring out of
#: `__init__`, button-dispatch routing, the agent cluster): 18,930/600 ->
#: 17,749/593. Wave 3 recorded 18,909/598; dev grew the screen by 21 lines
#: and 2 methods while wave 4 was in flight, which is the whole reason this
#: file exists. First recorded after wave 2 (PR #1381) at 20,964/612.
#:
#: Always MEASURE after the final rebase. Wave 3 set its budget twice from a
#: pre-rebase measurement and both landed red, because dev moved underneath
#: it -- a budget derived from a stale base fails the moment it merges.
#: Lowered 2026-08-27 by TASK-3070.14 after the amended Wave 6 realtime and
#: review/selection extractions merged and the final tree measured exactly.
#: The first closeout measurement was 16,968/562; dev then advanced through
#: PR #2125 and subsequent Console work through PR #2147 before this ratchet
#: landed. The final live measurement is: 17,727/593 -> 17,037/565.
#: Lowered again by TASK-19900.5 after moving Library provider construction and
#: selected-turn activity projection into ``UI/Console_Modules/library_activity``.
#: Concurrent dev work moved the final base to 17,059/566; the post-rebase
#: feature tree measured 17,000/565. PR #2050 removes two remaining
#: compatibility methods and keeps fork state behind the message controller;
#: the final merged tree measures 16,966/563.
_BUDGETS: dict[str, tuple[str, int, int]] = {
    "tldw_chatbook/UI/Screens/chat_screen.py": ("ChatScreen", 16966, 563),
    #: Added 2026-09 by the Library decomposition plan (PR 0b): this row was
    #: missing for the entire month in which library_screen.py tripled from
    #: 15,819 to 46,109 lines while chat_screen.py shrank under its budget.
    #: New Library code belongs in tldw_chatbook/UI/Library_Modules/ — a
    #: subsystem's controller file may be created BEFORE its extraction
    #: series to receive new methods. See
    #: Docs/superpowers/specs/2026-09-01-library-screen-decomposition-design.md.
    #: Lowered at the reader-controller move (exemplar 2/4): 45134/1300 ->
    #: 44715/1300 -- the recipe's lower-in-the-same-PR contract (§6), not
    #: deferred to the cleanup task as originally (incorrectly) instructed.
    # Lowered at the browse-controller move (exemplar 3/4, task 8):
    # 44715/1300 -> 44084/1300 -- per the recipe's lower-in-the-same-PR
    # contract (§6). Framed at the task level, not the intermediate one:
    # this single PR's net movement is 44715 -> 44084, a lowering, same as
    # every other entry in this ledger -- there is no raise-precedent here.
    # Task 8 moved 40 method bodies to `LibraryConversationsController`,
    # replaced by one-line delegators (a pure move nets zero methods -- the
    # count is unchanged). 6 more names were moved and then reverted in
    # this same task after test-suite evidence caught two distinct
    # test-bypass shapes (5 via `-k "conversation and library"`'s fake-self
    # `SimpleNamespace` calls; 1 via the paired-baseline xdist sweep's
    # instance-attribute monkeypatch -- see that controller module's
    # docstring) -- they stay real, full-body methods on `LibraryScreen`,
    # which is why this measurement is a smaller shrink than a 46-method
    # move would have produced. Mid-review, before this PR ever landed,
    # that method-move alone measured 44060 -- an intra-task fix-round
    # intermediate that was never intended to land on its own and never
    # did: the same review round that produced it also caught the
    # class-level `_safe_text` rebinding silently destroying the
    # controller's own `_safe_text` property (a plain class-attribute
    # assignment always overwrites a same-named class member, including a
    # property descriptor); the fix removed the dead
    # property/constructor-param/backing-attribute from the controller
    # module and added an explanatory comment at the rebinding site plus
    # one net wiring-call line removed here -- net +24 lines of
    # documentation, no logic change, method count unchanged -- landing
    # this PR's only committed measurement, 44084.
    # Lowered at the cleanup PR (exemplar 4/4, task 9): 44084/1300 ->
    # 43974/1282 -- the conversations state shim block (28 generated
    # properties) is deleted wholesale, every remaining screen-side
    # `_library_conversation*` reference is retargeted to
    # `self._conversations_state.<field>`, and 18 of the 61 screen
    # delegators (9 reader-cluster, 9 browse-cluster) were pruned after a
    # repo-wide census proved zero references anywhere outside their own
    # one-line body -- exactly 18 fewer `FunctionDef`s, matching the
    # method-count drop 1-for-1 (a pure deletion, not a move: nothing
    # replaces them). 11 of the ledger's stated 12 dead imports were
    # genuinely removable; the 12th, `LIBRARY_CONVERSATION_READER_MAX_CHARS`,
    # had to be added back -- it is pinned by PR 0a's OWN re-export
    # contract (`test_screen_still_re_exports_every_moved_name` in
    # `Tests/Architecture/test_library_support_layer_surface.py`), which
    # requires `library_screen.py` to keep re-exporting every name Task 1
    # moved to `Library_Modules/`, whether or not the screen's own logic
    # still reads it. This is the conversations exemplar series' FINAL
    # measurement (state PR, 2 controller PRs, cleanup PR complete) -- see
    # backlog/docs/library-decomposition-recipe.md's updated §11 for the
    # full pin trajectory, framed at the task level (each entry below is
    # one PR's single landed measurement; 44060 was an intra-task-8
    # fix-round intermediate, never itself a landed value, and is omitted
    # here for that reason):
    # 45134 -> 44715 -> 44084 -> 43974/1300 -> 1282.
    # Lowered again at the final-review fix wave (outside the exemplar
    # series proper): 43974 -> 43965 -- dead-import prune. The final
    # reviewer AST-verified 15 more dead imports (`Enum`, `TypeAlias`,
    # `AdaptiveReaderLayoutProfile`, `ConversationReaderRequest`,
    # `NormalizedDatabaseNote`, `DatabaseNotePortLoadReply`,
    # `DatabaseNotePortSaveReply`, the four `parse_*_prompts_from_content`
    # names, `event_principal_id_from_active_context`, `is_gguf_file`,
    # `Filters`, `PostRecomposeCallback`) that the cleanup PR's own
    # 12-import ledger missed -- each confirmed single-occurrence (its own
    # import line only), not pinned by
    # `test_screen_still_re_exports_every_moved_name`, and not imported by
    # anything outside this module. Method count unchanged (imports only).
    # NOT lowered by the 2026-09-03 dev merge (~380 dev commits absorbed),
    # same "concurrent growth outran the earned shrink" posture as the
    # chat_screen wave-5 note above. The merge itself removed zero moved
    # lines/methods from LibraryScreen (see the transplant list in
    # merge-update-report.md) -- every net line/method added here is dev's
    # review-sets phases 2-5, the media sort chooser, and the Reader
    # accelerator keys, all genuinely new screen-owned code, landed on dev
    # while this PR was in flight. Measured on the merged tree: 43965/1282
    # -> 45439/1330.
    "tldw_chatbook/UI/Screens/library_screen.py": ("LibraryScreen", 45439, 1330),
}

# Task 22507.4 started from this reviewed measurement. The repository-wide
# ratchet predates concurrent Console growth, so this explicit comparison
# proves this task does not add to that debt while the stale ceiling is fixed
# independently.
_TASK_22507_4_CHAT_SCREEN_BASE = (20099, 633)


@lru_cache(maxsize=None)
def _measure(rel_path: str, class_name: str) -> tuple[int, int]:
    """Line count of a module and method count of one class inside it.

    Cached: both tests below run over the same parametrization, so an
    uncached version reads and `ast.parse`s every budgeted file twice
    per session for no extra coverage. The cache is per-process, so
    mutation-testing this ratchet (edit the file, re-run) still works --
    just not by editing a file from inside a running test.

    Args:
        rel_path: Repo-relative path to the module.
        class_name: The dominant class whose methods are counted.

    Returns:
        tuple[int, int]: ``(module line count, class method count)``.

    Raises:
        AssertionError: If the module or the named class is missing — either
            means the budget entry is stale and must be updated deliberately
            rather than silently skipped.
    """
    path = _REPO_ROOT / rel_path
    assert path.exists(), f"{rel_path} not found; the budget entry is stale."
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert classes, f"class {class_name} not found in {rel_path}; budget stale."
    methods = [
        node
        for node in classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    return len(source.splitlines()), len(methods)


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_screen_does_not_grow_past_its_budget(rel_path: str) -> None:
    """The ceiling itself: a budgeted screen may not exceed its numbers.

    Args:
        rel_path: Repo-relative path of the budgeted module, supplied by
            the parametrization over `_BUDGETS`.

    Raises:
        AssertionError: If the module grew past its line budget or the
            class grew past its method budget. The message names
            `UI/Console_Modules/` and `DESIGN.md` section 7, because the
            fix is to put the new code somewhere else -- never to raise
            the number.
    """
    class_name, max_lines, max_methods = _BUDGETS[rel_path]
    lines, methods = _measure(rel_path, class_name)

    guidance = (
        f"\n\n{rel_path} is under a one-way size ratchet "
        f"(Tests/Architecture/test_screen_size_ratchet.py).\n"
        f"New Console code belongs in tldw_chatbook/UI/Console_Modules/ — see "
        f"DESIGN.md section 7 for the region-vs-controller rule and the "
        f"dependency-binding contract.\n"
        f"Do NOT raise the budget to make this pass. Lower it when a "
        f"decomposition wave lands."
    )

    assert lines <= max_lines, (
        f"{rel_path} grew to {lines} lines (budget {max_lines}, "
        f"+{lines - max_lines}).{guidance}"
    )
    assert methods <= max_methods, (
        f"{class_name} grew to {methods} methods (budget {max_methods}, "
        f"+{methods - max_methods}).{guidance}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("rel_path", sorted(_BUDGETS))
def test_budget_is_not_left_slack_after_a_wave(rel_path: str) -> None:
    """The recorded budget should track reality, not drift above it.

    A ratchet with slack silently permits regrowth up to the stale number, so
    a wave that forgets to lower its budget quietly buys the next feature
    headroom it was never meant to have. The tolerance is deliberately loose
    (200 lines / 10 methods) so ordinary in-file edits do not fail CI — it
    only fires when a decomposition landed and its budget was not updated.
    """
    class_name, max_lines, max_methods = _BUDGETS[rel_path]
    lines, methods = _measure(rel_path, class_name)

    assert max_lines - lines <= 200, (
        f"{rel_path} is {max_lines - lines} lines under its budget "
        f"({lines} vs {max_lines}). A wave landed without lowering the "
        f"ratchet — set it to {lines} so the gain is locked in."
    )
    assert max_methods - methods <= 10, (
        f"{class_name} is {max_methods - methods} methods under its budget "
        f"({methods} vs {max_methods}). Set it to {methods}."
    )


@pytest.mark.unit
def test_task_22507_4_does_not_worsen_chat_screen_base() -> None:
    """Task 4 must not exceed its reviewed screen line or method counts."""

    lines, methods = _measure(
        "tldw_chatbook/UI/Screens/chat_screen.py",
        "ChatScreen",
    )

    assert lines <= _TASK_22507_4_CHAT_SCREEN_BASE[0]
    assert methods <= _TASK_22507_4_CHAT_SCREEN_BASE[1]
