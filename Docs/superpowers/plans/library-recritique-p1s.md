# Plan: Library re-critique P1s (backlog tasks 4020, 4021, 4022)

Branch `fix/library-recritique-p1s` (worktree `.worktrees/library-rc-p1s`), cut from dev `e13608106`.
Source: re-critique snapshot `.impeccable/critique/2026-08-09T20-15-07Z__tldw-chatbook-ui-screens-library-screen-py.md` (22/40). Three tasks, one plan task each; each owns its backlog file end-to-end (In Progress + plan before code; ACs + notes + Done after).

Two of these are **our own shipped fixes that are present in code but not producing their effect** — the arc's governing lesson. Do not re-patch the old assumptions; re-root-cause against what the code actually does today.

## Global Constraints

- **Python/tests**: `.venv/bin/python` from the repo root; pytest is the ONLY entry point that may import app modules (a bare `python3 -c` importing the app writes to the LIVE config). Targeted tests + `--collect-only -q` sanity over touched trees. "no tests ran" = failed gate; read the passed count. **Foreground Bash only — NEVER end your turn waiting on background work** (three agents have died parked on background suites this programme).
- **TDD**: failing test first with RED evidence, then green. For "present but not working" defects the RED must reproduce the *observed effect*, not merely call the function — a test that passes against the broken code is the failure mode being fixed here.
- **Rendered-geometry / real-DB assertions**: layout claims need region assertions; deletion/import claims need a real DB, not mocks. This programme has caught vacuous guards three times.
- **CSS**: never edit the generated bundle; source tcss → `build_css.py` → `check_bundle_sync.py`. Widget `DEFAULT_CSS` needs no bundle step.
- **Live verification (required per task)**: unique socket `rcT<N>lib$RANDOM`, scratch `/tmp/rcT<N>`, `users_name = "sdd_rct<N>"`, `[first_run] setup_started/completed = true`. Cold start ~12s; palette nav; clicks by CHARACTER index (python `str.find`), never byte offsets; `capture-pane -p -e` for styling/ANSI proof. Cleanup: `C-q`, `kill-server`, `rm -rf ~/.local/share/tldw_cli/sdd_rct<N>`.
- **Git**: commit per task, message ends `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`; `git add` SPECIFIC paths; never stash; never push. **Always operate with the worktree path explicit** — a controller push from the wrong checkout已 sent foreign commits this session.
- **Docs**: UI-visible changes update the matching `Docs/User_Guide/` page or its stamp (`Verified against dev @ e13608106`).
- **Known ambient**: dev carries ~52 failures in `Tests/UI/test_library_shell.py` (note/ingest real-DB churn) + 1 video-gen shadow-name failure + 4 Settings failures from task-4010's missing `_appearance_bool_label`. A/B against clean origin/dev before blaming your change; do NOT fix those here.

## Task 1: Nav ghosting under the overflow-menu model (backlog task-4020)

Read `backlog/tasks/task-4020 - *.md`; its 4 ACs bind. Measured at dev: tab labels cut **mid-word at 80 AND 120** (`⌃6 Watc`, `⌃9 M`), scroll fragments (`‹ oleplay…`, `‹ edules…`), and **no ghosted tabs at all** — the bar scrolls.

The ghosting machinery IS present (10 refs in `UI/Navigation/main_navigation.py`, 2 in `css/components/_navigation.tcss`). Task-3200's four rounds built it against an in-strip pager; dev replaced that with `NavOverflowMenu` while the arc was in flight, and PR #1459's rebase kept ghosting while the scroll/paging model beneath it changed.

**Root-cause first, then fix.** Read `_ghost_clipped_buttons`, `_recenter_strip`, and how the overflow menu now decides what renders; instrument with a headless probe (the established pattern) to find why straddle detection no longer fires or no longer matters. Then decide the honest shape: ghosting may no longer be the right mechanism under a menu-based overflow model — if the correct fix is "the strip only ever renders whole tabs and the rest go to the menu", say so and implement that instead of resurrecting ghosting. **Correct the now-obsolete task-3200 tests rather than leaving them passing vacuously** (AC#4) — a test that still passes while the guarantee is broken is itself a finding.

Live-verify at 80/100/120 with both an early and a late active tab.

## Task 2: Blank-note GC (backlog task-4021)

Read `backlog/tasks/task-4021 - *.md`; its 4 ACs bind. **Root cause already confirmed** (by two agents independently): `_flush_library_note_save`'s emptiness test reads the coordinator snapshot's `title`, which `handle_library_notes_create_blank` seeds with the **literal string `"Untitled"`**, so `any(value.strip() for value in (title, content, keywords))` is always truthy and the GC branch is unreachable. Wired to ~7 exit paths, none of which can fire it.

**Read the prior art before implementing**: commit `f8bd6e8ac` on the unmerged branch `feat/media-ingest-followups` (task-3315) fixes this in tandem with a coupled save-seam change — that coupling is why it was not a one-liner. `git show f8bd6e8ac` and reconcile: adopt, adapt, or supersede it, and say which. Do NOT duplicate it into a conflicting second fix; if adopting wholesale is cleanest, cherry-pick with attribution in the notes.

Guard rails: a pre-existing note emptied out must still SAVE (only session-created blanks GC); the P1-arc dirty-exit veto and the Escape path (just repaired in PR #1464) must both keep working; empty **prompt** and **skill** drafts already discard correctly — match that behaviour, and check whether their seam can be shared rather than parallelled.

Tests against a real DB: open-and-leave persists nothing; type-then-delete-all persists nothing; pre-existing-emptied still saves; the three currently-failing GC tests pass.

## Task 3: Soft-deleted media re-import + delete receipt (backlog task-4022)

Read `backlog/tasks/task-4022 - *.md`; its 4 ACs bind. Two coupled defects:
1. **Import dedup matches soft-deleted rows**, so a deleted file can never be re-added (`≡ matched · Already in Library` while the item is absent and the count stays down). Find the dedup lookup (the sha256/url match in the ingest writer — the ingest arc's notes call it `get_media_by_hash`/`get_media_by_url`) and decide the honest behaviour: exclude soft-deleted rows from the match, or match-and-restore. **Prefer restore if the row still carries its content** (the user's intent is "I want this file in my library"), and say why in the notes; a silent refusal is the one option that must not survive.
2. **Bulk delete emits no receipt and no undo**, while the confirm promises `This moves them to trash.` and no trash exists anywhere (not in the rail, not in the `type:` filter, not on any canvas). Ship the receipt with an undo affordance at the point of action. For the trash itself: a full Trash view is likely its own task — if you conclude that, make the COPY honest in this task (state what actually happens) and file the Trash view separately with the ID scanned fresh; do not leave a promise the product cannot keep.

Mind: deletion goes through the existing soft-delete seam (`mark_as_trash`) — do not add raw SQL or a second deletion path. Test against a real file-backed DB, and live-verify the full cycle: import → delete → re-import → present exactly once.
