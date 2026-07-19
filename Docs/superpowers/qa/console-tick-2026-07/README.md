# Console 0.2s sync-tick gating — TASK-251 live QA (2026-07-16)

## TASK-251 live QA

Live interactive verification of AC #4: *"streaming remains smooth and all
synced surfaces still update on real changes (send/finish/rename/switch)"* —
i.e. that the tick-gating fix (TTL-cached browser rows, equality-guarded
sub-syncs, static streaming excerpt) does **not** make any synced surface
stale or stuck.

### Verdict — GATING TRANSPARENT (surfaces fresh)

| AC / surface | What it gates | Verdict | Evidence |
|---|---|---|---|
| **#1** TTL-cache persisted browser rows (no per-tick DB query) + explicit invalidation on finish/rename/etc. | left-rail conversation list | **PASS** | new conversation appeared in Conversations + Chats sections within the TTL/finish-invalidation window; rename reflected in **0.5s** |
| **#2** equality-guarded settings-summary / agent / system-line / rail sub-syncs | provider/model/status/rail lines | **PASS** | all rail surfaces stayed correct and updated on every real change; nothing stale, nothing stuck |
| **#3** static "Streaming…" excerpt (stops 5×/s inspector recompose) | run-inspector Excerpt row | **PASS** | Excerpt read `Streaming…` mid-stream, real excerpt after settle |
| **#4** smoothness + freshness on send/finish/rename/switch | whole Console | **PASS** | 4/4 sends finalized clean; poll cadence steady; 0 stuck frames, 0 crashes |

**No DEFECTS. No app death.** 0 Python tracebacks, 0 `[ERROR]`/`[CRITICAL]`
log lines, 0 worker-cancel lines (no TASK-228-style regression) across 6 app
processes.

### Rig provenance

Reuses the proven TASK-215/216/217/222/228 geometry+transport rig verbatim;
only `PYTHONPATH`/worktree and port were re-pointed at the fix.

- **Worktree**: `.claude/worktrees/console-tick-251`, branch
  `worktree-console-tick-251`, **HEAD `fde45d75`**. Fix under test =
  `4763b969` (TTL-cache persisted browser rows behind the 0.2s tick) +
  `46d4d575` (equality-guard tick sub-syncs + dedupe rail-state build) +
  `93aacde5` (static streaming excerpt stops per-tick inspector recompose).
- **Serve**: raw `textual_serve.Server` running `python -m tldw_chatbook.app`
  from this worktree (real app TCSS), patched `textual.js`
  (`/private/tmp/tldw-qa-inline-20260713/static_patched`, exposes
  `window.__drv` for real xterm-buffer reads). Env
  `PYTHONPATH=<worktree>:<QA-HOME>`, `cwd=<QA-HOME>` (branch code wins),
  `TERM=xterm` with iTerm/VTE markers unset, `ESCDELAY=1500`. **Port 9251**,
  bundled Playwright chromium, headless, viewport **2050×1240** dsf=1
  (terminal **227 cols × 59 rows**, cell 9×21), external `https://**`
  aborted, one fresh app process per driver run.
- **Isolated HOME** `/private/tmp/tldw-qa-config-caps-20260714`
  (`HOME`+`XDG_CONFIG_HOME`+`XDG_DATA_HOME`), seeded config → llama.cpp @
  `127.0.0.1:9099`. The real `~/.config/tldw_cli` was never touched.
- **Server pre-flight**: `GET /health` = `{"status":"ok"}`; `GET /v1/models`
  reported the loaded model `gemma-4-26B-A4B-it-ultra-uncensored-heretic-
  Q4_K_M.gguf` with `capabilities: ["completion"]`. **Model-name note**: the
  seeded config still names `Qwen3.6-27B-…gguf`; a direct
  `POST /v1/chat/completions` with that exact name returned a valid
  completion served by the loaded gemma model (llama.cpp ignores the `model`
  field). TASK-251's scenario is **text streaming**, so `completion` is the
  capability that matters — not `multimodal`. **Not** BLOCKED-SERVER. The
  model *thinks* (`reasoning_content`); prompt-eval ~700 ms/token, so
  finalize latencies below (2–24 s) are model-bound, not UI-bound.
- **DB**: `…/qa_user/tldw_chatbook_ChaChaNotes.db`. Assistant replies verified
  persisted full-length (e.g. the "hash table" send → 184-char assistant row).

**Rig hardening (this run only, not a product issue):** the heavier Console
screen occasionally returned a racy empty xterm buffer / one black canvas
frame on the *first* read after load. Fixed rig-side by (a) a
`terminal.refresh(0,rows-1)` repaint nudge before every screenshot and (b) a
content-gated read (`wait_content("Composer")`) after page-ready. All
evidence frames below are painted (120–224 KB; a black frame is ~10 KB).

### Per-step results

**Step 1 — send + finish → rail refresh (AC #1).**
`walk_a2` sent *"List five common data structures and give a one-line
description of each."* → streamed → **FINALIZED at t=24 s** (`[streaming]`
cleared, Stop dropped, Send re-enabled). Within the post-finish settle
(2.8 s wait ≈ TTL 2.0 s + finish-invalidation) the left rail refreshed:
**Conversations** → `List five common data struc… - Chats`; **Chats**
active-session row → `▸ List five common…`. The earlier `walk_a` send
(`In two short sentences…`, assistant row 184 chars in DB) was already listed
as `saved chat - 6m`, confirming persistence + listing. Frames:
`v251-after-finish.png`, `v251-final-excerpt.png`.

**Step 2 — streaming excerpt (AC #3).**
Right rail opened, streaming assistant message selected mid-stream. Inspector
"Selected Message" → **`Excerpt: Streaming…`** while the transcript showed the
live text (`4. Queue: A [streaming]`) — the placeholder, not the live body
(`v251-during-stream.png` / `v251-streaming-excerpt.png`). After the message
settled the same row showed the **real** excerpt:
`Excerpt: 1. Array: A collection of elements identified by index. 2. Linked
List: A sequence of nod…` (`v251-final-excerpt.png`). Confirms the static
placeholder stops the per-tick recompose *and* the real excerpt returns on
finalize.

**Step 3 — rename (AC #1 rename-invalidation).**
`walk_b` sent a short prompt (finalized ~5 s), then clicked the active header
tab → `ConsoleRenameSessionModal` → renamed to **"QA Renamed 251"**. Reflected
in **0.5 s** across all three surfaces: header tab (`QA Renamed 251✕ New tab`),
left-rail **Conversations** (`QA Renamed 251 - Chats`), left-rail **Chats**
active session (`▸ QA Renamed 251`). Well within the ~3 s bar. Frames:
`v251-before-rename.png`, `v251-rename-reflects.png`.

**Step 4 — two tabs + switch (AC #4 switch).**
`walk_c`: tab 1 send *"Say only the word RED…"* (finalized ~4 s), header
**New tab**, tab 2 send *"Say only the word BLUE…"* (finalized ~2 s). Both
tabs shown in the header; the left rail distinguished them
(`Chats - active session` vs `Chats - open session`). Switch → tab 1:
transcript `Assistant RED` (RED present, BLUE absent). Switch → tab 2:
transcript `Assistant BLUE` (BLUE present, RED absent). Switch → tab 1 again:
still clean RED-only. No split-brain, no stale transcript. Frames:
`v251-two-tabs.png` (tab 1), `v251-two-tabs-b.png` (tab 2).

**Step 5 — smoothness (228-era checks).**
Driver-side buffer-poll cadence during the stream: **min 1.00 s, mean 1.08 s,
max 2.68 s** over 23 polls (1.00 s nominal). The single 2.68 s gap coincided
with the mid-stream click-to-select + a screenshot repaint nudge (script-side
awaits), not a UI-loop stall — the UI stayed responsive: the mid-stream
message selection registered, the inspector updated, and the transcript
streamed incrementally (items 1→4). All 4 sends cleared `[streaming]`, dropped
the Stop button, and re-enabled Send on finalize (24 s / 5 s / 4 s / 2 s);
**0** stuck-`[streaming]` frames. Server log: **0** tracebacks, **0**
`[ERROR]`/`[CRITICAL]`, **0** cancellation/worker-cancel lines across 6 app
processes.

### Evidence files (this directory)

- `v251-during-stream.png` / `v251-streaming-excerpt.png` — `Excerpt: Streaming…`
  while `4. Queue: A [streaming]` streams (left rail + transcript + inspector).
- `v251-after-finish.png` / `v251-final-excerpt.png` — post-finalize: real
  excerpt + refreshed left-rail conversation list.
- `v251-before-rename.png` / `v251-rename-reflects.png` — rename reflected in
  header tab + both left-rail sections (0.5 s).
- `v251-two-tabs.png` / `v251-two-tabs-b.png` — clean transcript switch between
  two sessions; both listed.
- `v251-rail-open-baseline.png`, `diag-after-load.png`, `probe-initial.png` —
  rig baselines.

### Reproduce

Session scratchpad (not committed) `…/scratchpad/tick251/`: `serve251.py`
(port 9251, worktree PYTHONPATH), `drv251.py` (geometry + ws-stdin driver with
repaint-nudge + content-gate, OUT → this dir), `walk_a2.py` (send + streaming
excerpt + finish rail), `walk_b.py` (rename reflection), `walk_c.py` (two-tab
switch). QA HOME `/private/tmp/tldw-qa-config-caps-20260714`.
