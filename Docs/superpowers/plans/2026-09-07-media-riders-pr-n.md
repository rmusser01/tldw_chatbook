# Media riders PR N — grips and the reader resolver (tasks 31951, 31952, 31953)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. The three backlog task files are the briefs; this plan carries the batching, constraints and verification.

**Goal:** finish what PR H started on the adaptive reader shell: the three sibling readers (Conversations, Skills, Collections) opt into the one-cell grip with the same glyphs as Media (31951); the dead five-column grip CSS and the unused `PANE_GRIP_WIDTH` readers go, so the reserved and painted grip widths come from one value (31952); the custom-items-width pin stops overclaiming the library-closed comfort branch, or that branch obeys a typed width (31953 — decide and record).

**Spec:** the task files under `backlog/tasks/` (31951, 31952, 31953).

## Global Constraints
- Worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/media-riders-n`, branch `fix/media-riders-n` off dev. Every command `cd <worktree> && git branch --show-current`; python `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python` with `PYTHONPATH=$PWD`; `-p no:cacheprovider`; UI test files one per process, SERIAL.
- `tldw_chatbook/Utils/adaptive_reader_state.py` is shared by all four readers; `AdaptiveReaderLayoutProfile.grip_width` (default 5) is the per-profile knob PR H added (Media = 1); the grip widget paints `‹`/`›` under four cells. Opting the siblings in CHANGES their widths: re-pin their resolver tables honestly (old value in the comment, no ranges) and their painted shell pins; the Items floors and PR D's row positions on the Media surface do not move.
- CSS: `css/screen_agentic_library.tcss` and the bundle are GENERATED from `css/components/_agentic_terminal.tcss` — edit the source, `python -m tldw_chatbook.css.build_css`, `python tldw_chatbook/css/check_bundle_sync.py` exit 0, commit the regenerated files.
- No new `logger.*`; do NOT edit `backlog/`; painted pins for anything a user sees.
- Evidence: compare failing NAME sets against a detached copy of the merge-base; known dev reds per task-31249's census.
- Live: ONE app instance, tmux socket `wrn` via `t() { tmux -L wrn "$@"; }`; three orphaned instances on `/private/tmp/tldw-uat-wizard` are another session's. Capture each sibling reader's grips at 235×52.
- Commit per batch with the trailer `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## Batches
### Batch 1: 31951 + 31952 (one commit each is fine)
Opt the three sibling profiles into `grip_width=1`; re-pin; remove the dead `.library-adaptive-reader-pane-grip { width/min-width/max-width: 5 }` rule; remove the unused `PANE_GRIP_WIDTH` import in `library_screen.py` and make `library_skills_controller.py`'s `2 * PANE_GRIP_WIDTH` read the profile; consider carrying `grip_width` on the effective layout so the resolver's reservation and the shell's paint are one number (say whether you did, and why).
### Batch 2: 31953
Decide: obey a typed custom width in the library-closed comfort branch (`adaptive_reader_state.py` ~295-303), or rename/re-document the pin to state the branch it does not cover. Record the decision and reason beside the pin.

## Land
Final whole-branch review → fix round → PR N. Update `Docs/User_Guide/library/*.md` where the grip is described for the sibling surfaces.
