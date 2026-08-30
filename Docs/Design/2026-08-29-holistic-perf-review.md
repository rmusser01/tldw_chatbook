# Holistic performance review — 2026-08-29

**Pin:** dev `bc1e26ce60` (524 commits since the 2026-08-27 review pin `c6218918d1`).
**Method:** pinned worktree `.worktrees/perf-review-0829` with its own uv venv and a scratch
profile; headless Textual Pilot at 170×48; **configured** state (valid-shaped
`api_settings.openai.api_key`, so Console is the real app, not the setup modal); all numbers
from the **warm** profile (second and later runs) unless stated. Package identity asserted on
every probe. A/B runs interleaved base/patched/base/patched under identical load.

**Verdict: idle is clean; interaction is not.** The prior cycles' idle/timer fixes held
(12 s idle = 0.09 s CPU, zero stalls, zero SQL). Today's lag is concentrated in **CSS style
application**, which is O(total rules) per styled node and runs hundreds of times per
interaction. Screen switches stall the event loop for up to **399 ms**.

---

## 0. All four boot-budget ratchets are RED on pristine dev

ADR-097 made these ratchets on 2026-08-27; TASK-23112 paid the import debt down to 646/660
(headroom 14) on 2026-08-28. **One day later, all four are breached.**

| Ratchet | Limit | Now | Detail |
|---|---|---|---|
| `MAX_TLDW_MODULE_COUNT` (import) | 660 | **662** | was 646 with 14 headroom on 08-28 |
| `MAX_TLDW_MODULES_AT_UI_READY` | 970 | **981** | 19 new modules, 1 shed |
| `MAX_BOOT_PARSED_CSS_BYTES` | 860,000 | **862,184** | +3,424 B in the monolith alone |
| boot worker allowlist | — | **1 unlisted** | `console-persisted-browser-cache` (`UI/Console_Modules/workspace.py:2553`) |

The new modules cluster in one family — the workspace/tool-execution work:
`Tools.{git,local,patch,virtual_cli}_tool_impls`, `Tools.workspace_tool_{executor,protocol}`,
`Tools.workspace_root_pin`, `Agents.{raw_shell,virtual_cli}_tool_provider`,
`Workspaces.change_review_{consent,finalization}`, `Chat.console_settings_*`,
`TTS.{legacy_request_builder,text_processing}`.

**The process finding outranks the numbers.** Three cycles running, the budgets are re-breached
within ~24 h of every paydown. The guards only run in the slow suite; last cycle's open owner
call — move them into the fast `perf-guard.yml` so a breach surfaces per-PR — is the fix, and
it is now unblocked (the debt it was waiting on was paid, then immediately re-incurred).

---

## 1. HEADLINE — every style application scans all 4,324 CSS rules

`Stylesheet.apply()` pre-filters candidate rules through `rules_map`, but then does:

```python
rules = list(filter(limit_rules.__contains__, reversed(self.rules)))
```

which **walks the entire rule list on every call** to recover source order. The pre-filter
reduces *matching* work but not the *scan*. Measured consequences:

| Measurement | Value |
|---|---|
| Global rules / selector sets | **4,324 / 5,188** |
| `stylesheet.apply()` on one node | **0.52 ms** |
| `app.update_styles(screen)` (500 widgets) | **240 ms** |
| `RuleSet.__hash__` calls in one Console switch | **7,335,029** |

7.33M ≈ 1,667 applies × 4,324 rules — the scan, exactly.

Stack sampling during observed stalls ranks `textual/css/model.py:__hash__` the #1 frame, then
`_check_selectors`, `check`, `_check_id`. **Screen switching produced 17 event-loop stalls,
worst 399 ms.**

### 1a. Validated fix — replace the scan with an index sort

Sorting the already-filtered candidate set by a cached position index, instead of scanning all
rules, is ~15 lines and changes no CSS and no application code. Interleaved A/B, 2 rounds:

| Metric | base | patched | Δ |
|---|---|---|---|
| `apply()` per node | 0.521 / 0.515 ms | 0.391 / 0.373 ms | **−27%** |
| `update_styles(screen)` | 239.7 / 242.5 ms | 160.4 / 150.7 ms | **−35%** |
| Console switch (6 samples each) | 1.927 s mean | 1.668 s mean | **−13% (−260 ms)** |
| Library switch | 1.34 s mean | 1.14 s mean | −15% |
| Typing CPU/key | 17.6–18.0 ms | 17.4–20.6 ms | no effect (few applies) |

Belongs upstream in Textual, or as a vendored `Stylesheet` subclass. **Caveat: correctness was
not verified beyond "the app boots, switches screens and accepts input" — rendering fidelity
needs a proper check before adopting.**

### 1b. The other half of the lever — cut the rule count

Because cost is linear in total rules, deleting rules is a direct app-wide win, and it
compounds with 1a.

**`css/components/_agentic_terminal.tcss` is 272,312 B and holds 1,214 of the app's 4,198
source rule blocks — 29% of all CSS, and 41% of the global monolith.** Despite the name it is
not agentic-terminal CSS; it is an unstructured dumping ground:

| Selector family in that file | Rule blocks |
|---|---|
| `#library-shell-grid.library-notes-compact` | 72 |
| `LibraryIngestCanvas` | 30 |
| `ConsoleSettingsModal` | 16 |
| `#settings-shell` | 10 |
| `#mcp-mode-strip` / `#lab-mode-strip` | 7 / 6 |
| `#workflows-*`, `#personas-inspector-pane`, `#watchlists-inspector-pane` | 13+ |

All of it is global, so every Library rule is scanned when styling a Console button. It grew
262,634 → 272,312 B (+3.7%) in the two days since the last pin.

Note `SCOPED_CSS` does **not** help here — scoped rules still live in `self.rules` and are
still scanned. Only deleting/consolidating rules, or swapping stylesheet sources per screen,
reduces N. The byte budget should be joined by a **rule-count** ratchet; bytes ≠ rules.

---

## 2. The Console rebuilds itself completely on every visit

Identical cold and warm — there is no warm benefit at all:

| Per Console visit | Count |
|---|---|
| Widgets constructed | **559** (212 `Static`, 108 `Button`, 44 `Horizontal`, 24 `Vertical`) |
| `stylesheet.apply()` | **1,668** |
| `update_nodes` passes | **402** (971 nodes) |
| `set_class` | **907** |
| Wall | **1.89–1.98 s** |

The dominant chain is `Button.watch_flat` → `set_class` → `app.update_styles` →
`stylesheet.update_nodes` → 974 applies ≈ **1.5 s** of the switch. 108 Buttons per visit each
fire that watcher on construction.

Two independent fixes: (a) keep the screen instance alive / lazily build off-screen sections so
559 widgets are not re-minted per visit; (b) wrap screen construction in `app.batch_update()`
so 402 separate `update_nodes` passes collapse.

---

## 3. Typing costs 20.6 ms CPU per keystroke on an *empty* Console

Isolated probe (screen switch excluded from the measured window — an earlier reading that
blamed typing for a 457 ms stall was **wrong**; that stall belonged to the switch):

| Per keystroke | Count |
|---|---|
| CPU | **20.6 ms** |
| `query_one` | **35.3** |
| `Static.update()` | **8.4** |
| `set_class` | **31.8** |

`Widgets/Console/console_composer_bar.py::_sync_collapsed_presentation` runs **3× per
keystroke**, and each run does 4 `query_one` + a `Static.update()` + 4 `set_class` — on the
**collapsed row, which is `display:none` while you type**. It is unconditional: no early-return
on unchanged state, no cached widget handles.

Other unguarded per-key repeats: `_apply_draft_height` (4 `query_one`), `_refresh_visible_draft`
(2 + 2 `Static.update`), `_console_command_popup_or_none` (2), `_sync_raw_cli_state` (2).

Separately, provider readiness is recomputed on the keystroke path:
`normalize_provider_config_key` runs **11,003 times across 43 keys (256/key)** via
`_ensure_active_console_session_settings` → `build_console_settings_readiness` →
`get_provider_readiness`. Cheap per call (~30 ms total) but it does not belong there at all.

---

## 4. Smaller, still real

- **598 `query_one` calls per screen switch.** Hot sites: `console_bounded_section._reconcile`
  (61), `left_rail._mounted_descriptors` (56), `left_rail._run_allocation_reconcile`
  (4 sites × 28), `destination_rail.sync_open` (36).
- **Library re-reads 43 config settings on every visit.**
  `library_screen._load_library_ingest_options_from_config` runs from `on_mount`, and the app
  builds a NEW `LibraryScreen` per visit (verified: three visits, three distinct instance ids).
  **CORRECTION:** this was first written up as "config re-read on unrelated screen switches",
  from a 33.5-`get_cli_setting`-per-switch average taken over a Library+Console *pair*. Measured
  per destination, the reads are Library's alone — Console switches read 18–21, none of them from
  `library_screen`. The averaging smeared one screen's mount cost across both. The waste is real
  (43 reads per visit, forever) but it is a per-visit remount cost, not cross-screen leakage.
- **Library opens 8 fresh SQLite connections per visit** via
  `DB/private_sqlite.py:1156 _connect_registered_sqlite`, each paying `journal_mode=WAL` +
  `synchronous=NORMAL` + `foreign_keys=ON`. No pooling/reuse.
- **Extraction is still losing to growth.** `UI/Screens/library_screen.py` is **42,556 lines**;
  `settings_screen.py` 24,621; `console_chat_controller.py` 20,234; `chat_screen.py` 18,238.

## 5. Verified healthy

- **Idle: 0.09 s CPU over 12 s (0.75% of a core), zero stalls, zero SQL.** The 08-22/08-24
  timer and idle-SQL fixes held.
- The Console 5 Hz transcript-sync timer is correctly self-stopping and gated.
- `main_navigation._update_overflow_hints` is signature-guarded and skips cleanly.
- No `refresh_css()` call sites anywhere in the app.

---

## Recommended order

1. **Adopt 1a** (Textual `apply()` index sort) — ~15 lines, −27% per apply, −260 ms per Console
   switch, app-wide, no CSS or app changes. Verify rendering fidelity first.
2. **Split `_agentic_terminal.tcss` by owning screen and delete duplicates** — targets 29% of
   all rules; compounds multiplicatively with 1.
3. **Stop re-minting 559 widgets per Console visit** + `batch_update()` around construction.
4. **Guard the composer sync functions** on unchanged state; cache widget handles; never touch
   the hidden collapsed row.
5. **Pay the four ratchet breaches, and move the budget guards into `perf-guard.yml`** so the
   next breach costs one PR instead of one review cycle.
6. Cache per-screen config reads; pool the Library's 8 connections/visit.

## Root cause shared by findings 2, 4 and the Library reads

`switch_screen` is handed a freshly constructed screen every time, with state restored from
`screen_state_store` — verified live: three visits to Library produced three distinct instance
ids. That single decision is why the Console re-mints 559 widgets per visit, why Library re-reads
43 config settings per visit, and why Library opens 8 SQLite connections per visit. Screen
instance reuse is therefore the highest-leverage structural change available, and also the most
invasive: the snapshot/restore design assumes fresh construction.

## Probe traps hit this cycle

- **A phase that contains a screen switch cannot measure typing.** My first typing reading
  showed a 457 ms stall; isolating the switch out of the measured window reduced it to one
  113 ms stall. Attribute before filing.
- **`__init__` was 0.949 s on the first run and 0.088 s on the second** — the 0.949 s was
  one-time schema creation on a fresh DB, not a boot cost real users pay. Always re-run warm.
- `sqlite3.Cursor.execute` cannot be monkeypatched (immutable C type); wrap `sqlite3.connect`
  and install `set_trace_callback` instead.
