# Console vision-streaming finalize — TASK-228 fix verification (2026-07-14)

## TASK-228 verification

Defect-targeted re-run of the three live defects found in the TASK-222 vision
walk (`Docs/superpowers/qa/console-config-caps-2026-07/README.md`, "Vision
re-run" section), now against the fix.

### Verdict per defect

| Defect | Prior symptom | Verdict | Evidence |
|--------|---------------|---------|----------|
| **V1** — successful image-payload streams never finalize (`[streaming]` + Stop + disabled Send persist forever) | seen >5 min past completion | **FIXED** | all 8 sends cleared `[streaming]`, dropped Stop, re-enabled Send; 60s post-completion stability watch held; within-session send #2 proves the session is not blocked |
| **V2** — assistant DB row truncated to a mid-stream fragment (e.g. `The ima`, `A w`) | 2 truncated rows prior | **FIXED** | every one of my assistant rows persisted the **full** reply matching the UI verbatim |
| **V3** — ~half of vision sends stall (no token in 480s, no DB row) | 2 total stalls prior (~50%) | **FIXED** | **0** no-token stalls across 8 sends; 0 user rows without a matching assistant reply |

**No new defects** observed in the run-finalization / streaming path.

### Rig provenance

Reuses the proven TASK-215/216/217/222 geometry+transport rig verbatim; only
`PYTHONPATH`/worktree and port were re-pointed at the fix.

- **Worktree**: `.claude/worktrees/console-run-groups-228`, branch
  `worktree-console-run-groups-228`, **HEAD `53323078`**
  (`fix(console): dedicated worker groups so UI-sync kicks can't cancel in-flight
  runs` — the TASK-228 fix; dedicated groups `console-run` / `console-sync` /
  `console-save-as`, replacing the ungrouped default-group `exclusive=True`
  workers that let a UI-sync re-kick silently cancel the in-flight send).
- **Serve**: raw `textual_serve.Server` running `python -m tldw_chatbook.app`
  from this worktree (real app TCSS), patched `textual.js`
  (`/private/tmp/tldw-qa-inline-20260713/static_patched`, exposes `window.__drv`
  for real xterm-buffer reads). Env `PYTHONPATH=<worktree>:<QA-HOME>`,
  `cwd=<QA-HOME>` (branch code wins), `TERM=xterm` with iTerm/VTE markers unset
  (`auto` → pixels → inline image render), `ESCDELAY=1500`. **Port 9151**,
  bundled Playwright chromium, headless, viewport **2050×1240** dsf=1, external
  `https://**` aborted, one fresh app process per send.
- **Isolated HOME** `/private/tmp/tldw-qa-config-caps-20260714`
  (`HOME`+`XDG_CONFIG_HOME`+`XDG_DATA_HOME`), seeded config → llama.cpp @
  `127.0.0.1:9099` (`Qwen3.6-27B-…gguf`) with the vision override and extended
  `[chat.images]` formats. The real `~/.config/tldw_cli` was never touched.
- **Server pre-flight**: `GET /v1/models` reported
  `capabilities: ["completion","multimodal"]` (mmproj vision projector loaded)
  before any capture; `/slots` idle at start.
- **DB queried**:
  `…/.local/share/tldw_cli/qa_user/tldw_chatbook_ChaChaNotes.db`, table
  `messages`. Raw `timestamp` is ISO-Z (`2026-07-14T20:28:34.241Z`); my rows are
  isolated with `timestamp > '2026-07-14T20:00:00'` (all prior-round rows are
  ≤17:29Z). Query:
  `SELECT timestamp, sender, length(content), content FROM messages
  WHERE timestamp > '2026-07-14T20:00:00' ORDER BY rowid;`

### Completion gate (how finalize/stall were told apart)

Poll the xterm buffer + llama.cpp `/slots` `is_processing` (server-side ground
truth) every 3s:
- **FINALIZED** = assistant text present AND `[streaming]` gone AND Stop button
  gone (Send re-enabled). On the two headline sends + consec #3 the gate then
  watched **60 more seconds** and confirmed the finalized state **held**.
- **V1 detector** (STREAM_STUCK) = text present, server idle >60s, still
  `[streaming]` — would have caught the prior finalize hang. **Never tripped.**
- **V3 detector** (NO_TOKEN) = no token ever while server idle >180s (the app's
  request never reached the server). **Never tripped for a vision send.**

### Per-check results

**Check 1 — tiff (`v228-tiff-finalized.png`)**: `scan.tiff` staged via the real
picker + sent with *"Describe this image in one short sentence."* Reply streamed,
then **FINALIZED at t=129s** (`[streaming]` cleared, Stop gone, Send re-enabled);
finalized state **held the full 60s** post-completion watch. Frame shows the
inline transcoded-PNG render (orange/teal quadrants + white centre stripe) and
the complete reply with **no** `[streaming]` suffix and **no** Stop button.
Assistant DB row (conv `4bfa5d76`, 20:28:34Z, 124 chars) verbatim:

> This is the flag of Côte d'Ivoire, featuring horizontal orange and green bands separated by a central white vertical stripe.

**Check 2 — svg (`v228-svg-finalized.png`)**: same flow with `logo.svg`.
**FINALIZED at t=42s**, held 60s. Frame shows the cairosvg-rasterized inline
render (blue field, white circle, blue inner square) + finalized reply. Assistant
DB row (20:31:40Z, 81 chars) verbatim:

> A white square is centered inside a white circle against a solid blue background.

**Check 3 — V3 consecutive sends** (3 more, alternating images, fresh app process
each — the exact config that stalled ~50% before). All finalized; `_consec1.png`,
`_consec2.png`, and `v228-consecutive-sends.png` (after the last).

| # | image | gate | reply @ t | DB row (len) — verbatim |
|---|-------|------|-----------|--------------------------|
| c1 | pic.png | FINALIZED | 18s | (19) `A solid red square.` |
| c2 | scan.tiff | FINALIZED | 99s | (84) `A flag design featuring a central white stripe flanked by orange and teal quadrants.` |
| c3 | logo.svg | FINALIZED (held 60s) | 42s | (96) `The image features a white circle containing a blue square, set against a solid blue background.` |

**Check 4 — text-only control**: *"Reply with exactly one word: hello"*.
Finalized cleanly (`[streaming]` gone, Stop gone, Send re-enabled), DB row
(20:41:34Z, 5 chars) = `hello`. `v228-control-text.png` shows the clean state.
(Rig note: the automated gate reported `NO_TOKEN` here only because its
`has_text` guard requires >10 chars and the reply is 5 — a measurement artifact;
the buffer + DB confirm a clean finalize. This is a rig quirk, not a product
defect.)

**Bonus — within-session multi-send** (`v228-session-multi.png`): directly
falsifies V1's "blocks further sends in that session". Send #1 = `scan.tiff`
vision (finalized t=69s), then **without restarting**, send #2 = a text prompt in
the same session → **posted and finalized (t=36s)**. One conversation
(`4223bd4c`), 4 messages, both assistant rows full (`A flag with a central white
stripe…` 104 chars, then `continued` 9 chars).

### Aggregate

8 sends total (5 fresh-process vision + 1 text control + 2 within-session).
Every send produced a full assistant reply and a clean UI finalize; **0**
truncated DB rows, **0** no-token stalls, **0** stuck-`[streaming]` frames.
`SELECT COUNT(*)` of user rows lacking any assistant reply in the conversation
(the V3 signature) since 20:00Z = **0** (prior round had 2).

### Observation (not a defect)

The model semantically misreads the two-tone test TIFF as a national flag
("Côte d'Ivoire") — a model-side content quirk, irrelevant to the transport /
finalization surface under test (the payload demonstrably reached the model and
the run finalized correctly every time).

### Reproduce

Session scratchpad (not committed) `…/scratchpad/caps228/`: `serve228.py`
(port 9151, worktree PYTHONPATH), `drv228.py` (geometry + ws-stdin driver, OUT →
this dir), `verify228.py` (per-send finalize gate + captures, steps
tiff/svg/c1/c2/c3/control), `session_multi.py` (within-session two-send V1
proof). QA HOME `/private/tmp/tldw-qa-config-caps-20260714`.
