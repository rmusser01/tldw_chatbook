# Console mid-run retry/regenerate/continue GATE — TASK-232 live QA (2026-07-14)

## Verdict

**GATE WORKS + STREAM SURVIVES.** Clicking *regenerate* on an older completed
assistant message while a vision run was streaming showed the warning toast
**"A Console run is already running."** and did **not** spawn a console-run
worker — the in-flight vision run finished normally and persisted its full
reply. The idle control (same click with no run active) started a real new run
and produced a new variant.

| Step | Expectation | Result |
|------|-------------|--------|
| 1. text send finalizes | older completed assistant exists | ✅ `hello -- answer in one word` → `Hi`, finalized t=9s |
| 2. vision run starts | multi-minute in-flight window | ✅ Stop + server busy at t=2s |
| 3. **mid-run regenerate click** | warning toast, stream NOT cancelled | ✅ toast **"A Console run is already running."**; server still busy after click |
| 4. vision run finalizes | `[streaming]` cleared, Stop gone, Send re-enabled, full reply | ✅ finalized t=159s; DB row 150 chars |
| 5. DB row intact | full vision reply persisted | ✅ see below |
| 6. idle regenerate control | new run actually starts | ✅ new run t=4s, finalized t=9s, new variant (`< >` nav) mounted |

Unit tests (`Tests/UI/test_console_run_gate.py`) — **6 passed** (mid-run
notify-instead-of-spawn + idle-still-spawns, parametrized over
retry/regenerate/continue).

## Verbatim toast (from the xterm buffer)

The gate warning rendered as a bottom-right toast overlaying the composer band:

```
A Console run is already running.
```

Captured verbatim from the live terminal buffer at click time
(`v232-midrun-gate-toast.png`), matching the fix's `notify(...)` copy exactly.

## DB proof — the in-flight stream was NOT cancelled

`…/qa_user/tldw_chatbook_ChaChaNotes.db`, table `messages`, the gate-run
conversation `9ebd19df` (in `rowid` order):

| timestamp (Z) | sender | len | content |
|---|---|---:|---|
| 02:46:12 | user | 27 | `hello -- answer in one word` |
| 02:46:27 | assistant | 2 | `Hi` — the older completed msg the mid-run regenerate targeted |
| 02:46:40 | user | 42 | `Describe this image in one short sentence.` |
| 02:50:02 | assistant | 150 | `The image shows a flag design with a central white vertical stripe flanked by two identical panels that are horizontally divided into orange and teal.` |

The vision assistant row is the **complete** reply (150 chars, matches the UI
verbatim) — the in-flight run finalized cleanly despite the mid-run regenerate
click. There is exactly **one** assistant reply for the vision turn (no
truncation, no stuck-`[streaming]`, no orphaned user row).

## Rig provenance

Reuses the proven T215/216/217/222/228 geometry+transport rig verbatim; only
`PYTHONPATH`/worktree + port were re-pointed at the fix.

- **Worktree**: `.claude/worktrees/console-retry-gate-232`, branch
  `worktree-console-retry-gate-232`, **HEAD `713963dc`**
  (`fix(console): gate retry/regenerate/continue on run state before spawning
  console-run workers` — TASK-232). The screen mirrors the submit path's
  `run_state.is_send_allowed` gate at all three dispatch sites
  (`retry`/`regenerate`/`continue`) in
  `tldw_chatbook/UI/Screens/chat_screen.py`, notifying instead of spawning an
  exclusive `console-run` worker that would cancel the in-flight run at
  worker-creation time.
- **Serve**: raw `textual_serve.Server` running `python -m tldw_chatbook.app`
  from this worktree (real app TCSS), patched `textual.js`
  (`/private/tmp/tldw-qa-inline-20260713/static_patched`, exposes `window.__drv`
  for real xterm-buffer reads). Env `PYTHONPATH=<worktree>:<QA-HOME>`,
  `cwd=<QA-HOME>`, `TERM=xterm` with iTerm/VTE markers unset (`auto` → pixels →
  inline image render), `ESCDELAY=1500`. **Port 9232**, bundled Playwright
  chromium, headless, viewport **2050×1240** dsf=1, external `https://**`
  aborted, one fresh app process (fresh empty session) per page load.
- **Isolated HOME** `/private/tmp/tldw-qa-config-caps-20260714`
  (`HOME`+`XDG_CONFIG_HOME`+`XDG_DATA_HOME`), seeded config → llama.cpp @
  `127.0.0.1:9099` (`Qwen3.6-27B-…gguf`, vision override, `[chat.images]`
  `default_render_mode = "pixels"`, `strip_thinking_tags = true`). The real
  `~/.config/tldw_cli` was never touched.
- **Server pre-flight**: `GET /v1/models` reported
  `capabilities: ["completion","multimodal"]` (mmproj vision projector loaded)
  before any capture.

## How the mid-run click was driven (layout-independent)

The vision model *thinks* for tens of seconds before emitting the first token;
during that phase no chunks stream, so the transcript view is stable. Within
that window the driver: (a) mouse-wheeled the transcript to the top, (b) clicked
the first assistant message row (`on_click` → `select_message`, mounting its
action row), (c) clicked the **♻** regenerate action button, (d) read the xterm
buffer for the toast. Selection uses the dim `Assistant` role label
(excluding the status-bar `Assistant:`) rather than a hardcoded column, because
the transcript's left offset shifts with sidebar width (observed col 46 vs 59).
Completion vs. cancellation was told apart by llama.cpp `/slots` `is_processing`
(server-side ground truth) plus the xterm buffer (`[streaming]` / Stop / Send).

## Per-step detail

- **Step 3 (headline)**: pre-click flags `{stop:True, busy:True}` (run in
  flight, 2 assistant rows: the completed `Hi` + the pending vision reply).
  First select attempt raced the deferred action-row mount (♻ not yet present);
  a second select+click landed the ♻ at buffer cell (78,16). Immediately after:
  `toast:True`, `busy:True` — the toast appeared **and** the server was still
  processing the vision run (stream alive). (The Stop label read False for that
  one frame only because the toast overlay covered the composer band; Stop was
  observed True continuously through Step 4.)
- **Step 4**: the vision run stayed `busy:True` / Stop visible from t=21s→147s,
  then **finalized at t=159s** — `[streaming]` cleared, Stop gone, Send
  re-enabled, full 150-char reply shown (`v232-stream-survives.png`).
- **Step 6 (idle control)**: with no run active, selecting the first assistant
  and clicking ♻ started a new run at t=4s that finalized at t=9s; the action
  row gained `< >` variant-navigation buttons, proving a new variant was
  generated (`v232-idle-regen-works.png`).

## Evidence (this directory)

- `v232-midrun-gate-toast.png` — mid-run: first assistant `Hello` selected with
  action row, vision run pending at bottom, toast **"A Console run is already
  running."** bottom-right.
- `v232-stream-survives.png` — vision reply finalized in full, Send re-enabled,
  no Stop, no `[streaming]`.
- `v232-idle-regen-works.png` — idle regenerate produced a new run + variant
  (`< >` nav present).

## Reproduce

Session scratchpad (not committed)
`…/scratchpad/caps232/`: `serve232.py` (port 9232, this-worktree PYTHONPATH),
`drv232.py` (geometry + ws-stdin driver, OUT → this dir), `gate232.py`
(the full mid-run gate walk: text finalize → vision send → mid-run regenerate →
toast → vision finalize → idle regenerate). QA HOME
`/private/tmp/tldw-qa-config-caps-20260714`; llama.cpp @ `127.0.0.1:9099`.
