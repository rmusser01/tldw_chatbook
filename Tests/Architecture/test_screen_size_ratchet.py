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
    # Wave-2 task 3 (export series 2/3, controller PR): 43930 -> 43432 --
    # 22 method bodies moved verbatim to `LibraryExportController`
    # (`UI/Library_Modules/library_export_controller.py`), replaced by
    # 22 one-line screen delegators. Method count unchanged (1282): a pure
    # move, 22 `FunctionDef`s out, 22 back in as delegators. Of the 51
    # "export"-named candidates the task's own census started from, 29 stay
    # screen-resident, unmoved and byte-for-byte untouched (18 belong to
    # other subsystems; 2 carry `@work` and would fail Textual's
    # `isinstance(self, DOMNode)` check on a plain controller; 9 more are
    # reached by unbound-fake-self test calls this task's own verification
    # battery found -- see `library_export_controller.py`'s module
    # docstring for the full per-name accounting and its own exclusion
    # reasoning, which is where that reasoning lives, NOT as inline
    # comments on the 29 untouched screen methods themselves).
    # Wave-2 task 4 (export series 3/3, cleanup PR): 43432/1282 -> 43413/1281.
    # The Task-2-generated export-state shim block (13 properties) is
    # deleted wholesale; every remaining screen-side `_library_export_*`
    # field reference (42 literal `self._library_export_<field>` sites, an
    # AST-driven mechanical pass) is retargeted to
    # `self._export_state.<field>`, including a dynamic-dispatch site
    # (`_library_open_choice_strip`/`_close_open_library_choice_strip`'s
    # visibility-attr string, one of the four converged choice-strip
    # destinations) via the recipe's own dotted-vs-flat `operator.attrgetter`
    # passthrough helper (already installed by the conversations exemplar's
    # Task 9). Exactly ONE of the 22 screen delegators
    # (`_library_export_is_server_mode`) had zero references anywhere
    # outside its own one-line body (a repo-wide census; the other 21 all
    # have a genuine production caller -- mostly the round-2/round-3
    # screen-resident siblings task 3 excluded -- or a direct test call) and
    # was pruned: exactly 1 fewer `FunctionDef`, matching the method-count
    # drop 1-for-1. 5 named dead imports pruned
    # (`LIBRARY_EXPORT_SERVER_DISABLED_TOOLTIP`, `MEDIA_QUALITY_OPTIONS`,
    # `count_export_scope`, `default_export_name`,
    # `normalize_export_destination`), each confirmed single-occurrence and
    # not a member of any `_SURFACE`-shaped re-export contract in
    # `Tests/Architecture/test_library_support_layer_surface.py` first (the
    # conversations exemplar's own "dead within this file is not the same
    # question as dead" lesson).
    # Wave-2 task 6 (collections controller PR, collections series 2/3):
    # 64 method bodies moved into `LibraryCollectionsController`
    # (`UI/Library_Modules/library_collections_controller.py`), each
    # replaced by a one-line screen delegator (63 `self._collections_
    # controller.<name>(...)` forwards + 1 `LibraryCollectionsController.
    # <name>(...)` class-forward for the cluster's single staticmethod,
    # `_restore_library_collections_page`) -- a pure move, so the method
    # count is unchanged (64 `FunctionDef`s out, 64 one-line delegators
    # in). Measured 42486 lines, 1281 methods at that point.
    # Wave-2 task 7 (collections cleanup, collections series 3/3): the
    # generated collections-state shim block deleted wholesale, every
    # remaining screen-side `_library_collections_<field>` reference
    # retargeted to `self._collections_state.<field>` (14 literal sites +
    # 2 dynamic-dispatch dict-of-name-strings entries shared with
    # Conversations/Export's own precedent), and 14 of the 64 screen
    # delegators pruned (repo-wide census across `tldw_chatbook/` and
    # `Tests/`, including `Tests/Live/`: zero references anywhere outside
    # their own one-line body) -- a much larger prune fraction than the
    # export series' 1-of-22 because task 6 found zero method-level
    # test-bypass exclusions, so Collections' whole 64-method cluster
    # moved onto ONE controller with no screen-resident sibling left to
    # call any of them back. Fresh post-cleanup measurement: 42411 lines,
    # 1267 methods (1281 - 14 pruned `FunctionDef`s, exactly) -- lowered
    # in this same commit per recipe §6 (never deferred to a later task).
    # Raised 2026-09-03 by the wave-2 final review's fix wave: +9
    # documentation-only lines (a construction-order sentinel comment on
    # the `LibraryExportController` construction site, finding 3 of the
    # review) pushed the file to 42420/1267 (methods unchanged -- comment
    # lines only, no code). Re-measured and raised in this same commit per
    # the foundation run's own task-8 precedent (a strengthened comment
    # there pushed the file 24 lines over its just-set ceiling; the fix
    # was to re-measure and raise with a dated justification comment, not
    # to leave the ceiling red or strip the comment to fit). Net wave-2
    # trajectory is still down: 43965 (wave-2 start) -> 42420, a shrink of
    # 1545 lines despite this fix wave's own small increase.
    #
    # NOT lowered by the 2026-09-03 merge of origin/dev (base retargeted
    # to `dev`, absorbing foundation PR #2315 -- identical commits on both
    # sides, so trivial -- plus ~435 further dev feature commits), same
    # "concurrent growth outran the earned shrink" posture as the
    # chat_screen wave-5 note and this same file's own two prior
    # dev-catch-up entries above (which this wave-2 branch never saw
    # directly since the foundation commits are identical on both sides;
    # this is wave-2's own first catch-up). The merge itself transplanted
    # zero lines into a wave-2-moved method body: a repo-wide diff of
    # every export-cluster (22) and collections-cluster (64) method name
    # between the wave-2 fork point and dev found exactly one body
    # difference, `_close_open_library_choice_strip` (dev added a
    # `_library_media_sort_choices_visible` dict entry for the new media
    # sort chooser) -- and that method is screen-owned, never moved (the
    # export controller's own same-named method is a callback property
    # forwarding to it, not the moved logic), so git's ordinary 3-way
    # merge combined dev's new dict entry with wave-2's unrelated
    # `_library_export_quality_choices_visible` -> `_export_state.
    # quality_choices_visible` retarget on separate lines of the same
    # dict literal with no conflict and no manual transplant needed.
    # Every other net line/method here is dev's review-sets work, the
    # media sort chooser, and further Reader/library feature work that
    # landed on dev while this PR was in flight (dev's own pin going into
    # this merge was 45522/1331). Measured on the merged tree: 42420/1267
    # -> 43977/1316.
    #
    # 2026-09-03, wave-3 task 2 (combined search+RAG state PR, series 1/3):
    # the 20-field `__init__` block (19 `_library_rag_*` + 1
    # `_library_search_history`) collapsed into one `LibraryRagSearchState`
    # constructor call plus a generated shim loop; methods unchanged (pure
    # field move, zero FunctionDefs touched). 43977/1316 -> 43923/1316.
    # 2026-09-03, wave-3 task 3 (combined search+RAG controller PR, series
    # 2/3): 42 of the 50 combined search+RAG candidates (60 raw "search"/
    # "rag" matches minus 3 Prompts-owned + 7 Media-owned) moved verbatim
    # into `LibraryRagSearchController`
    # (`UI/Library_Modules/library_rag_search_controller.py`), each
    # replaced by a one-line screen delegator (41 `self._rag_search_
    # controller.<name>(...)` instance-forwards + 1 `LibraryRagSearchController.
    # <name>(...)` class-forward for the cluster's single staticmethod,
    # `_library_rag_scope_summary`) -- a pure move, so the method count is
    # unchanged (42 `FunctionDef`s out, 42 one-line delegators in). 8 of
    # the 50 candidates stay screen-resident, unmoved and byte-for-byte
    # untouched: 3 carry `@work` and would fail Textual's `isinstance(self,
    # DOMNode)` check on a plain controller (`_execute_library_rag_
    # answer`/`_execute_library_rag_search`/`_save_library_search_
    # history`); 4 more are the "instance-attribute monkeypatch"
    # test-bypass shape (recipe §11 lesson 2) -- `_library_rag_panel_
    # state`, `_refresh_search_rag_panel_state_widgets`, `_patch_sibling_
    # library_search_input`, `_mirror_library_rag_scope_recovery` -- found
    # by a repo-wide monkeypatch census across all three test roots (2 of
    # the 4 already flagged by task 2's own forward note; the other 2 are
    # new findings from this task's own wider census) -- see
    # `library_rag_search_controller.py`'s module docstring for the full
    # per-name reasoning; and 1 more, `_load_library_search_history`, was
    # excluded mid-task after the verification battery found a real
    # regression -- its bare `get_cli_setting` reference resolves against
    # the DEFINING module's globals, and moving it silently broke every
    # `monkeypatch.setattr(library_screen_module, "get_cli_setting", ...)`
    # test (a `test_library_shell.py` fixture several tests depend on).
    # (3 + 4 + 1 = 8 excluded; 50 - 8 = 42 moved -- every count in this
    # entry agrees.) It stays a real, full-bodied screen method
    # (byte-for-byte identical to before this task), constructed at its
    # original `__init__` position with no controller involvement. Net
    # diff: +96 insertions (the import line, the new
    # `LibraryRagSearchController` constructor call in `__init__` right
    # after `self._collections_controller`, plus the 42 one-line delegator
    # bodies) / -1010 deletions (the 42 moved bodies' original lines, net
    # of restoring `_load_library_search_history`'s full body once it was
    # excluded). 43923/1316 -> 43009/1316.
    # 2026-09-03, wave-3 task 4 (combined search+RAG cleanup, series 3/3):
    # the generated search+rag-state shim block (task 2) deleted wholesale;
    # every remaining screen-side `_library_rag_<field>`/`_library_search_
    # history` literal retargeted to `self._rag_search_state.<field>` (66
    # occurrences across 11 screen methods via one mechanical regex pass,
    # AST-reverified to zero remaining live consumers -- corrected here
    # from an initial undercount of "35 occurrences across 9 methods";
    # see the fix-round comment below). A wider census also
    # flagged `canvas_sync.py`'s `_sync_library_canvas` (its `"search"`
    # branch writes `screen._library_rag_answer_render_key` directly) as a
    # candidate -- an initial retarget to `screen._rag_search_state.
    # answer_render_key` broke `test_library_canvas_scoped_sync.py::
    # test_media_choice_and_rag_toggles_are_canvas_scoped` (caught by this
    # task's own sweep, not left latent): that branch's ONLY two callers
    # (`cycle_library_rag_mode`/`toggle_library_rag_scope_source`) forward
    # `self` = the CONTROLLER as the `screen` parameter, which has no
    # `_rag_search_state` attribute at all (by design -- see the
    # controller's own permanent shim's docstring). Reverted: the flat
    # name is correct AS-IS, resolving through the controller's own
    # mirrored shim exactly the way the conversations controller's
    # identical `self`-forwarding shape already relies on;
    # `canvas_sync.py` needed NO change. Of the 42 delegators
    # task 3 left in place, 12 had ZERO references anywhere outside their
    # own one-line body (14 `@on` handlers + 3 `action_*` handlers always
    # kept per the recipe's transform whitelist; 13 more non-`@on` names
    # kept for a genuine screen-resident or test caller) -- pruned, along
    # with the one import (`build_library_rag_console_live_work_payload`)
    # that prune made newly dead, plus 3 more already-dead-since-task-3
    # imports (`LIBRARY_RAG_QUERY_MAX_LENGTH`, `LIBRARY_RAG_USE_IN_CONSOLE_
    # LOCKED_NOTICE`, `library_rag_scope_summary`) and the `SEARCH_PREFIXED_
    # STATE_FIELDS` import the deleted shim was the screen's only consumer
    # of. 12 fewer `FunctionDef`s -- exactly the 12 pruned delegators; no
    # method body touched. 43009/1316 -> 42949/1304.
    # 2026-09-03, wave-3 task 4 fix round 1: 9 more cluster-caused dead
    # imports pruned from the same `Widgets.Library` import block --
    # `library_rag_answer_children`, `library_rag_history_children`,
    # `library_rag_query_quiet_text`, `library_rag_query_shows_full_
    # recovery`, `library_rag_query_status_children`, `library_rag_
    # results_body_children`, `library_rag_scope_recovery_children`,
    # `results_heading_text`, `scope_toggle_label` -- each verified
    # single-occurrence (import line only) before deletion; the neighbour
    # `library_rag_scope_shows_recovery` stayed (still live, ~line 42446).
    # Comment-only otherwise; no method body touched. 42949/1304 ->
    # 42940/1304.
    #: Re-measured 2026-09-03 at the wave-3 dev catch-up merge (42 dev commits
    #: incl. Console-interaction PRD work landed inside the budgeted file):
    #: 42940/1304 -> 43225/1311. Post-merge re-measure per the standing
    #: dev-race protocol; the decomposition's own trajectory remains down.
    "tldw_chatbook/UI/Screens/library_screen.py": ("LibraryScreen", 43225, 1311),
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
