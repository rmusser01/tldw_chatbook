# Media riders PR L — focus and press behaviour (tasks 31950, 31954, 31946, 31945)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. The four backlog task files are the briefs; this plan carries the batching, constraints and verification.

**Goal:** close the four focus/press riders from wave 5: the remaining `call_after_refresh` focus sites after a viewer sync (31950), the doubled progress restore (31954), a bare whole-screen `refresh(recompose=True)` dropping focus to None (31946), and sibling canvases' row buttons dropping the second click inside the 0.2 s active flash (31945).

**Spec:** the task files under `backlog/tasks/` (31950, 31954, 31946, 31945) — their Descriptions carry the incident and file:line anchors; their ACs govern.

## Global Constraints
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-riders-l`, branch `fix/media-riders-l` off dev. Every command `cd <worktree> && git branch --show-current`; python `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=$PWD`; `-p no:cacheprovider`; UI test files one per process, SERIAL.
- Focus seams: the viewer-scoped one is `_after_library_media_viewer_sync` (library_screen.py; queues on the viewer's post-recompose hook chained with the pending restore, `finally`-protected); the screen-level one is `LibraryScreen.refresh(..., recompose=True)` (PR F's capture/restore for the Media canvas). 31946's restore must live at ONE shared seam (the task says BaseAppScreen) and must not double-fire with PR F's override.
- Pins observe real focus on a MOUNTED widget (`screen.focused is <widget>` and `is_attached`, after a bounded wait), never a call-recording fake; the sibling-canvas click pin drives two presses inside 0.2 s.
- No new `logger.*`; CSS only if needed (source `css/components/_agentic_terminal.tcss`, regenerated sheet + bundle committed, `check_bundle_sync` exit 0); no new toolbar buttons; Find focus token untouched; do NOT edit `backlog/` (controller flips tasks).
- Evidence: compare failing NAME sets against a detached copy of the merge-base (never an earlier head of this branch); known dev reds are in task-31249's census (whole-file `test_library_shell.py` ~226–230 red here; recompose-ratchet count; ingest Select mount race; trash escape-return under load).
- Live: ONE app instance, tmux socket `wrl` via `t() { tmux -L wrl "$@"; }`, sleeps inside, `t kill-server` at the end; three orphaned instances on `/private/tmp/tldw-uat-wizard` are another session's — ignore them.
- Commit per batch with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Batches
### Batch 1: viewer-seam siblings (31950 + 31954)
Route the four remaining `_sync_library_media_viewer_or_recompose()` + `call_after_refresh(<focus/sync>)` sites (the brief's `awk`: a sync followed within ~14 lines by `call_after_refresh(`) through `_after_library_media_viewer_sync`; make ONE owner schedule the reader progress restore on a mode change (31954) and pin the invocation count. Verification: `Tests/UI/test_library_media_render_fixes.py`, `test_library_media_reader_flow.py`, `test_library_media_reader_shell.py` whole-file; the `awk` census returns zero sites.
### Batch 2: shared seams (31946 + 31945)
31946: a bare `refresh(recompose=True)` on a Library screen leaves focus on the equivalent widget or a defined fallback — at one shared seam (BaseAppScreen or the LibraryScreen.refresh override PR F added; pick the one seam and say why), pinned by a background recompose. 31945: `active_effect_duration = 0` (or the shared helper) on the conversations/notes/prompts row buttons with one fast-second-click pin on a sibling canvas. Verification: `test_library_shell.py -k "focus or recompose or row"` vs base; the sibling canvas test files touched.

## Land
Final whole-branch review → fix round → PR L.
