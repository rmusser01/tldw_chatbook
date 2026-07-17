# TASK-252 live QA — Library selection targeted (non-recompose) updates (2026-07-16)

Live QA for **AC #4** of task-252: the converted Library selection interactions
must "look and behave exactly as before, just snappier." The fix replaces
whole-screen `self.refresh(recompose=True)` calls with (Tier 1) in-place
row-marker/count/export-button patches and (Tier 2) canvas-scoped
`canvas.sync_state()` for the conversations/media browse+select flows.

## Verdict

**TARGETED UPDATES CLEAN** — all six converted flows render and behave
correctly and identically to the pre-fix build in steady state (markers,
"N selected" count, rail counts, preview, and mode enter/exit/switch), just
without the whole-screen rebuild.

Two rendering nuances were investigated and **both proven PRE-EXISTING via an
A/B against the parent commit** (neither is a task-252 regression) — see
§Investigations. One carries a minor, self-healing task-252 timing delta,
documented there for the author's awareness.

## Rig provenance

Reuses the proven TASK-222/228 textual-serve + headless-chromium geometry rig
verbatim; only `PYTHONPATH`/worktree and port were re-pointed at the fix.

- **Worktree under test**: `.claude/worktrees/library-sync-252`, branch
  `worktree-library-sync-252`, **HEAD `7dc925ae`** (fix commit `480affb8`
  "perf(library): targeted updates for selection interactions (task-252)").
- **Serve**: raw `textual_serve.Server` running `python -m tldw_chatbook.app`
  from this worktree, patched `textual.js`
  (`/private/tmp/tldw-qa-inline-20260713/static_patched`, exposes
  `window.__drv` for real xterm-buffer reads). `PYTHONPATH=<worktree>:<QA-HOME>`,
  `cwd=<QA-HOME>`, `TERM=xterm` (iTerm/VTE markers unset), `ESCDELAY=1500`.
  **Port 9161**, bundled Playwright chromium, headless, viewport **2050×1240**
  dsf=1, external `https://**` aborted. Captures are xterm-buffer-gated
  (poll `translateToString` until the target text is present before each shot).
- **A/B baseline serve**: same rig from a read-only `git worktree` at the
  **parent commit `df5fba20`** (pre-fix; `self.refresh(recompose=True)` on
  every selection handler) at `/private/tmp/tldw-prefix-252`, **port 9162**.
- **Isolated QA HOME** `/private/tmp/tldw-qa-config-caps-20260714`
  (`HOME`+`XDG_CONFIG_HOME`+`XDG_DATA_HOME`); the real `~/.config/tldw_cli`
  was never touched. No llama server needed (pure UI). Splash disabled,
  `console.onboarding.first_send_completed = true`.

### DB seeding (and a real gotcha worth recording)

Seeded the QA HOME ChaChaNotes DB
(`…/.local/share/tldw_cli/qa_user/tldw_chatbook_ChaChaNotes.db`) with **6
conversations "QA Conv 1..6"** (each 2 messages) + **4 notes "QA Note …"**, then
soft-deleted the 28 pre-existing vision-test conversations so the browse list is
short enough to show the preview pane in one frame. Final visible state:
**Conversations (6)**, **Notes (4)**.

Three things had to be right for seeded conversations to render in the Library
Browse ▸ Conversations canvas (notes have none of these constraints, which is
why they showed immediately):

1. **`client_id`** — `search_conversations_page` filters
   `client_id = <db instance client_id>`. Bare `add_conversation` stamped my
   seed rows with the seeder's `client_id`; the app filters by
   **`tldw_cli_local_instance_v1`** (the id the real 28 rows carried). Rows with
   any other `client_id` are silently invisible/uncounted.
2. **Durable commit** — `CharactersRAGDB.execute_query(..., commit=True)` only
   commits when `not conn.in_transaction`, but DML auto-opens a transaction, so
   `commit=True` **silently no-ops** and rolls back on connection teardown. The
   fix reported success in-process (its own connection saw the change) yet
   nothing hit the file. Must use `with db.transaction() as cur: …` (what
   `add_conversation`/`add_note` use internally).
3. **Scope** — the Library snapshot lists `scope_type="all"` (global +
   workspace), so scope alone doesn't gate visibility; set to
   `workspace/workspace-default` for parity with real Console chats anyway.

Seed/verify scripts (session scratchpad, not committed): `…/scratchpad/caps252/`
`seed252.py`, `serve252.py`, `drv252.py`, `cap_all.py` (the 6-capture walk),
`serve_prefix.py` + `cap_prefix*.py` + `cap_persist*.py` + `cap_hover.py`
(A/B + artifact characterization).

## Per-step results (all PASS — verified from the xterm buffer before each shot)

| # | Capture | Verified | Result |
|---|---------|----------|--------|
| 1 | `v252-browse-baseline.png` | `▸ QA Conv 6` selected; preview pane shows `QA Conv 6 / Messages: 2 / Updated / Open in Console`; rail `Conversations (6)` + all counts intact | **PASS** |
| 2 | `v252-select-mode.png` | `Done` toggle + `0 selected`; all six rows show `☐` | **PASS** |
| 3 | `v252-toggled-two.png` | `☑ QA Conv 5` + `☑ QA Conv 3`, others `☐`; `2 selected`; **rail region byte-identical to capture 2** (AC #2 — asserted by string-equality of the whole rail pane) | **PASS** |
| 4 | `v252-untoggle-one.png` | re-click QA Conv 5 → `☐ QA Conv 5`, `☑ QA Conv 3` stays; `1 selected` | **PASS** |
| 5 | `v252-browse-switch.png` | `Done` exits select mode (checkboxes gone, `Select` restored); click QA Conv 3 → `▸` moves off QA Conv 6 onto QA Conv 3; preview pane retitles to `QA Conv 3`; rail intact | **PASS** |
| 6 | `v252-rail-roundtrip.png` | rail Notes → Media → Conversations round-trip; no dead clicks, no corruption; QA Conv rows re-render; rail `Conversations (6)` intact | **PASS** |

Selection state that is not observable from the buffer (export-selected
`disabled` flip; no screen-level recompose; Tier-2 mouse-capture release;
tier-1 fallback-to-recompose) is authoritatively covered by the fix's own suite
**`Tests/UI/test_library_selection_updates.py` — 5/5 pass** on this worktree.

## Investigations (rendering nuances — both PRE-EXISTING, proven by A/B)

### 1. Select-mode action buttons not visibly rendered (`Select all N shown` / `Clear` / `Export selected`)

In the wide 2050-px canvas only the `Done` toggle and the `N selected` Static
render on the action strip; the three action buttons are not painted. **A/B
against the pre-fix `df5fba20` build (`prefix-select-mode.png`) is
pixel-for-pixel identical** — the canvas `compose()` that yields the strip is
byte-identical between the two commits (`diff` = IDENTICAL) and task-252 only
changed the *trigger*, not the layout. Their existence + `disabled` state is
validated by the passing unit suite. **Not a task-252 regression** (pre-existing
canvas-layout quirk).

### 2. Doubled/ghosted render of a hovered 2-line row

While the pointer **rests on** a conversation row, that row renders doubled
(the `2 messages - Nm` second line overlaps a duplicated title). Characterized
on the fix (`cap_hover.py`): it reproduces on **(A) a browse-mode row,
(B) a select-mode unchecked row, and (C) a select-mode checked row** — i.e. it
is a general hover-render artifact of the fixed-`height:2` two-line `Button`
rows in the textual-serve/xterm.js pipeline, **not** checkbox- or
select-mode-specific. **It reproduces identically in the pre-fix build on
hover** (`cap_persist_prefix.py`, final block = doubled). The row is clean
whenever the pointer is elsewhere; all six deliverable captures above park the
pointer off-row and show the clean settled state. **Not a task-252-introduced
defect.**

- **One honest task-252 timing delta**: in the *immediate post-click window*
  (pointer still resting on the just-toggled row), the fix's in-place
  `button.label` patch keeps the hovered button, so the pre-existing artifact is
  visible right away (`diag-doubled-row-fix.png`); the pre-fix full recompose
  rebuilt the button and showed a clean frame for ~3.5 s until the next
  hover-move (`cap_persist.py` vs `cap_persist_prefix.py`, t=0.4/1.5/3.5 s). It
  **self-heals the instant the pointer moves**. Minor, cosmetic, self-healing,
  no app impact — flagged for author awareness; root cause is the pre-existing
  multi-line-row hover quirk, out of task-252's logic scope.

No stale markers/counts, no dead clicks, no crashes, no app death observed.

## Evidence files

- `v252-browse-baseline.png`, `v252-select-mode.png`, `v252-toggled-two.png`,
  `v252-untoggle-one.png`, `v252-browse-switch.png`, `v252-rail-roundtrip.png`
  — the six required captures (clean settled state).
- `prefix-select-mode.png`, `prefix-toggled-two.png` — pre-fix A/B baselines.
- `diag-doubled-row-fix.png`, `diag-doubled-row-prefix.png` — hover-artifact
  characterization (both builds).
