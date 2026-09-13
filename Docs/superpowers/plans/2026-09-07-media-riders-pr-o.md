# Media riders PR O — row markers and the sibling canvases (tasks 31955, 31961, 31962, 31956, 31957, 31958, 31959)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. The seven backlog task files are the briefs; this plan carries the batching, constraints and verification.

**Goal:** close the wave-5 riders left on the row-marker and sibling-canvas surfaces: cell-width-aware keyword truncation plus in-place toggle pins (31955); no re-projection fetch for off-page rows (31961); one backing-id coercion helper (31962); the reviewed decoration costs O(visible rows) with a cache invalidated at the mark seam (31956); the preview pane shows the analysis marker the row shows (31957); the Rendered-view note covers any non-Markdown item with content and never paints over an empty one (31958); the sibling canvases' select-mode labels hold their column (31959).

**Spec:** the task files under `backlog/tasks/` (31955, 31961, 31962, 31956, 31957, 31958, 31959).

## Global Constraints
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-riders-o`, branch `fix/media-riders-o` off dev. Every command `cd <worktree> && git branch --show-current`; python `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=$PWD`; `-p no:cacheprovider`; UI test files one per process, SERIAL.
- The seven-key browse-row contract stays exactly seven; `has_analysis` stays a SQL projection (31961's fix is a membership test BEFORE the id-scoped fetch, not a new query); `reviewed` comes from the active review set (31956 caches the done-map on the screen and invalidates at the ONE mark seam PR I consolidated — `review_set_state.py` stays read-only); PR J's `library_disabled_action_label(align=True)` is the padding idiom (31959 applies it to the Conversations/Notes/Prompts row toggles with painted column pins); PR J's `RENDERED_VIEW_NOTE_TYPES` gate becomes "any non-Markdown item with content" (31958).
- Text carries meaning; painted pins for anything a user sees; no new `logger.*`; CSS only if needed (regenerated files committed); do NOT edit `backlog/`.
- Evidence: failing NAME sets vs a detached copy of the merge-base; known dev reds per task-31249's census; measurements (31956 AC#3) recorded in the report at REVIEW_SET_CAP.
- Live: ONE app instance, tmux socket `wro`.
- Commit per batch with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Batches
### Batch 1: 31955 + 31961 + 31962 (state-layer: truncation + toggle pins; off-page guard; one coercion helper)
### Batch 2: 31956 + 31957 (decoration cache with measurement; preview pane marker)
### Batch 3: 31958 + 31959 (note coverage; sibling padding)

## Land
Final whole-branch review → fix round → PR O. User Guide: preview-pane marker, note coverage.
