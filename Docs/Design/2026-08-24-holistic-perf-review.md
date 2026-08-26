# 2026-08-24 Holistic Performance Review — dev `a71e62e4b`

Fourth full performance review (after 2026-07, the 2026-08-11 input-latency audit, and the
2026-08-22 review whose burn-down closed 2026-08-24 with 30/35 findings shipped).
Commissioned because users report the app "recently slowed down a bit … again".

**Method.** Seven parallel read-only lanes against a pinned worktree
`.worktrees/perf-review-2026-08-24` (dev `a71e62e4b`): Console chrome (#2034/#2053/#2029),
streaming/run path (#2020/#2028/#2021/#2013), Library/Watchlists readers (#2064/#2063),
data layer, boot-path accretion, timers/pollers census, and a fix-regression audit of all
20 burn-down headline fixes. Plus live A/B probes against the PREVIOUS review's pin
`35d4bf3a1` — two additional worktrees (`perf-probe-2026-08-24`, `perf-base-2026-08-24`),
each with its own uv venv (`.[dev]`), isolated scratch profiles (own HOME/XDG/
TLDW_CONFIG_PATH; `[first_run] setup_completed=true`, splash disabled, valid-shaped openai
key so the Console is in the configured state, not the setup modal). 87 first-parent merges
landed between the pins. Top claims were re-verified first-hand before filing (the standing
lesson: a finding tells you where to look, not what to do — last review 5 of 7 filings
needed correction before dispatch).

Filed as TASK-22200 … TASK-22228 (29 tasks). Finding numbers below are task ids.

---

## The verdict in one paragraph

The burn-down's fixes **held** — every one of the 20 headline mechanisms is present and
load-bearing at tip (one exception: the CSS allowlist ratchet is RED, see 22212), and the
live A/B proves the two biggest wins in the starkest way (19 keystrokes: **48 SQL
statements at the pin → 0 at tip**; idle 4 s: **8 layout passes → 0**; fresh-profile first
boot construct **4.70 s → 2.35 s**). The new slowdown is **new code**: (1) every user who
upgraded across schema v46 gets a first session where an **unpaced whole-history FTS
rebuild** contends with every UI write — the single most plausible mechanism behind the
complaints; (2) the Console edge-rails redesign (#2034) and streaming emotes (#2020) made
the **5 Hz run tick** expensive — registry SQL on the loop, tripled O(all-conversations)
row pipelines, doubled-to-sextupled whole-transcript copies — the same defect *classes* the
burn-down removed, reintroduced on a different trigger; (3) the Library media reader
(#2064) rebuilds and re-parses the whole document **per traversal keystroke**; and (4) warm
boot-to-ready **regressed ~11%** while every import guard stayed green, because the growth
is on the legs the guards cannot see (Chat first-paint import leg, pre-importer payload,
boot worker fleet, eagerly-parsed CSS bytes).

---

## Measured A/B (tip `a71e62e4b` vs pin `35d4bf3a1`, same machine, interleaved runs)

| Metric | Pin 35d4bf3a1 | Tip a71e62e4b | Verdict |
|---|---|---|---|
| Warm `import tldw_chatbook.app` | 0.73 s | 0.64–0.88 s | equal (interleaved ×3) |
| tldw modules in import closure | 748 (incl. PIL, bs4, keyring, 39 Chunking) | 638 (none of those) | burn-down wins HELD |
| Warm `TldwCli()` construct | 58.5 ms | 63.5–105.7 ms | ~equal |
| Fresh-profile first-boot construct | 4.70 s | 2.35 s | IMPROVED 2× |
| Warm boot → `_ui_ready` (headless Pilot) | 1323 / 1331 / 1360 / 1368 ms | 1413 / 1435 / 1467 / 1473 / 1509 ms | **REGRESSED ~140 ms (~11%)** |
| tldw modules at `_ui_ready` | 984 | 931 (−93 removed, +40 new) | net smaller, yet slower |
| Widgets on Chat screen at ready | 460 | 469 | equal |
| `chat_screen` module import (after app import) | 186–199 ms | 182–198 ms | equal |
| 19 keystrokes into the focused Console composer: SQL statements | **48** | **0** | TASK-21118 verified live |
| 19 keystrokes: screen layout passes | 24 | 21 | equal (~1/key) |
| Idle 4 s: screen layout passes | 8 | 0 | TASK-21692 verified live |
| First-visit switch Library / Artifacts / Personas / Watchlists | 665 / 387 / 1135 / 606 ms | 581 / 368 / 1159 / 589 ms | equal (Personas slowest, pre-existing) |

TTI-regression attribution (honest state): NOT import time, NOT `chat_screen` import time,
NOT widget count, NOT the parallel init tasks (~62 ms total on the pool, of which
notes_service 53 ms). The identified contributors, each traced: the Chat first-paint import
leg grew +11,638 LOC (22213); the model-catalog refresh moved onto the initial-screen push
line (filed inside 22222's census scope; mechanism `app.py:12080`); boot-parsed CSS grew
+43 KB and `TieAwareStylesheet` arms full reparses during first mount (22222); boot workers
went 4 → 7 (22215); PIL loads on every boot via the Samira seeding chain (22217 — traced
live: `app.py:8784 _init_notes_service → get_chachanotes_db_lazy → seed_builtin_content →
Character_Chat/visual_identity.py:24`). A residual remains unattributed; 22213's AC
requires re-measuring with this review's interleaved method.

**Correction recorded during the review** (the filing-points-at-the-wrong-thing class): the
new `config.py:1802` import of `Library.library_media_reader_state` inside
`_load_settings_uncached` initially looked like ~100 ms moved onto config load — but the
Library package was already in the pin's import closure via `app.py:238`, so the
incremental cost today is small. Filed honestly as a **layering** finding (22223), not a
milliseconds finding.

Guard status at tip, run first-hand in the probe venv:
`Tests/Performance/test_app_import_weight.py` 5 passed (budget 660, measured 637–638);
`Tests/App/test_boot_no_feature_db_files.py` passed;
`Tests/UI/test_library_recompose_ratchet.py` 4 passed (census 97 == pin 97 — ZERO
headroom, watch item);
`Tests/UI/test_widget_css_consolidation.py::test_class_level_css_stays_within_the_allowlist`
**FAILED on pristine tip** (22212).

---

## Findings by theme (task id = finding id)

### A. The most plausible complaint mechanism: the post-upgrade window

- **22200 (high)** — v46 truncates `messages_fts`; `DB/chachanotes_fts_backfill.py:82-97`
  then re-tokenizes the ENTIRE history in back-to-back 500-row `BEGIN IMMEDIATE`
  transactions with **no pacing between chunks** (verified — no sleep in the loop), while
  every UI write is also `BEGIN IMMEDIATE` (15 s busy timeout) and chunk commits kill
  concurrent DEFERRED read-then-write writers instantly (task-21100's own wave-1 lesson).
  Every upgrading user's first session contends with a whole-history rebuild — concurrently
  with the pre-importer, the subscriptions backfill, and actor-pack recovery.
- **22215 (medium)** — boot worker fleet 4 → 7 since the pin; the aggregate GIL contention
  at first interaction is what the user feels, worst on that same first-post-upgrade boot.

### B. The Console run tick got heavy (PR #2034 + #2020) — the burn-down's defect classes on a new trigger

- **22201 (high)** — 3 × `_build_console_workspace_context_state()` per 0.2 s tick
  (`chat_screen.py:15206-15218`), each now reaching `ensure_default_workspace()` +
  `list_workspaces()` twice (browser labels + the NEW `workspace_tree_projection`,
  `workspace.py:2708→1525→3865-3887`; `registry_service.py:572-609`, `:1173-1203`): ~45
  extra synchronous queries/second on the loop while streaming, plus the O(all-
  conversations) canonical-owner/merge/overlay pipeline run twice per build × 3 builds.
  The keystroke path is still clean (memo intact; A/B-proven above) — this is the same
  shape on the tick trigger.
- **22202 (high)** — `ConsoleWorkspaceTree.sync_projection` (new file) is an unconditional
  O(rows) reconcile per push; one changed node blows Textual's whole tree line cache.
- **22203 (high)** — every tree cursor move posts a context update whose unguarded
  `Static.update()` arms a layout pass; boundary crossings trigger the full 7-section,
  ~45-`query_one` rail allocation pipeline. One arrow key = up to 2 extra frames + a full
  rail measure.
- **22204 (high)** — streaming emotes doubled (to as much as 6×) the per-tick
  whole-transcript `dataclasses.replace` copies via double expression resolution +
  re-entrant `_request_is_current` (`character.py:284-295`, `:353-354`;
  `console_expression_state.py:71`; `console_chat_store.py:5227-5234`). Default-ON
  (`resolve_show_character_avatar` defaults True). The tick already pays ~3 other full
  copies (native transcript, cost chip, guidance) — a shared per-tick snapshot is the
  stretch direction.
- **22221 (medium)** — avatar geometry: PIL LANCZOS resample on the UI thread per viewport
  size; allocation reconcile gained three per-pass legs and clears tree hover every pass
  (~5 Hz during runs).

### C. Send/restore path (new dispatch-checkpoint machinery)

- **22205 (high)** — per send, a ~10-statement `BEGIN IMMEDIATE` transaction (incl. FTS
  triggers + full-content `sync_log` JSON = ~3× write amplification) runs synchronously on
  the event loop before dispatch (`console_chat_controller.py:5635`;
  `chat_persistence_service.py:384`; `console_dispatch_repository.py:103-262`), plus a
  second pre-dispatch and a third at settle; every restore takes an IMMEDIATE (write)
  transaction just to read recovery state. Unbounded (up to 15 s) under 22200's window.
- **22206 (high)** — conversation resume is O(N²): per-node full-conversation scans with
  BLOB hydration (plan verified with `sqlite_stat1` ABSENT — `idx_msgs_parent` exists but
  is never chosen). **Measured: 89.8 ms @600 msgs vs 2.0 ms with the parent index forced
  (45×, O(N))**. Also RecursionError at ~980-message linear conversations.
- **22226 (low)** — create-path readbacks hydrate `image_data` up to 3× per persist.

### D. Library media reader (#2064) + Watchlists reader (#2063)

- **22207 (high)** — the reader now opens on FOCUS; each traversal keystroke synchronously
  recomposes the full document body (fresh `Markdown(content)` parse on the loop), then the
  settle recomposes a second time; the always-falls-through `unchanged` test at
  `library_screen.py:33096-33100`; no windowing. The merge is otherwise net-NEGATIVE on
  recompose count (245 → 237 sites; media-scoped whole-screen 27 → 21) — the regression is
  the new trigger, not site count.
- **22208 (medium)** — PIL mosaic preview built on the loop BEFORE the unchanged test and
  discarded on the no-change path; every interaction copies the whole content string.
- **22209 (medium)** — match navigation: 3–4 O(document) passes per Prev/Next click.
- **22210 (medium)** — one un-deduplicated SQLite progress write per traversal step, no
  exclusive worker.
- **22211 (medium)** — Watchlists collapse boundaries have NO hysteresis (bare threshold at
  `region_layout.py:132-175`; ±1 cell mounts/unmounts a whole pane; scrollbar-sensitive
  width source). The Library reader carries the fix (`LAYOUT_HYSTERESIS_WIDTH = 4`);
  Watchlists does not — the documented width-flap trap.
- **22228 items 6–7 (low)** — six reader presses still whole-screen recompose; two layout
  resolves per Resize event.

### E. Boot: the import guards are green and the boot still got slower

- **22213 (high)** — Chat first-paint import leg +11,638 LOC / +10 modules since the pin
  (AST closure, not diff-grep): `chat_screen.py:51` module-level-imports the entire
  TrajectoryScreen; `console_voice_input` (2,260 LOC) new on the leg; `Widgets/Console/
  __init__.py` eagerly re-exports the new tree/speech/authority widgets; `Internal_Prompts`
  (10 modules) still on the mount leg despite TASK-21731's title — its guard imports one
  module, never `chat_screen`. PIL + keyring load pre-first-paint via chat_screen chains
  (pre-existing).
- **22214 (medium)** — pre-importer payload +99 modules / +74,524 LOC; the 0.10 s max-gap
  cap makes the yield term a near-no-op (~92% GIL duty for a 1.2 s route); macOS falls back
  to `os.cpu_count()` → unthrottled tier. (TASK-21113 history honored: a sleep cannot
  subdivide `import_module` — the levers are payload, order, cap.)
- **22216 (medium)** — PR #1998 put a synchronous staging sweep back into
  `TldwCli.__init__` (`Actor_Packs/importer.py:216` from `app.py:7322`): a per-component
  privacy walk from `/` + scandir + per-candidate O_NOFOLLOW opens, every boot — the
  task-21106 class, invisible to the six-filename boot guard.
- **22217 (medium)** — PIL re-enters EVERY boot via `seed_builtin_content` →
  `visual_identity.py:24` module-level import; the seeding preflight exits early but the
  import is paid first. PIL confirmed present at `_ui_ready` on tip.
- **22222 (medium)** — the guard blind spots as a class: no census at `_ui_ready`, no CSS
  byte budget (770,285 → 813,605 B; the 21115 ratchet FORCES new widget CSS into the
  eagerly-parsed bundle with no size budget), no boot-thread census, construct-time runtime
  imports invisible (`app.py:7273-7274` re-imports `Persona_Visual.*` — harmless today,
  boundary crossed silently), no TTI tripwire; `TieAwareStylesheet` full-reparse count
  uninstrumented.
- **22223 (low)** — config load imports a feature package (layering; correction note above).

### F. Idle/recurring (timer census: 42 sites, ZERO new timers since the pin)

- **22218 (medium)** — composer caret blink at 1.89 Hz does 2 `query_one` + (with a draft)
  a ≤1000-entry history scan + a full grapheme wrap of the ENTIRE draft per tick, and keeps
  ticking under every modal (`has_focus_within` survives `push_screen`). The 21692 layout
  half holds; the compute half was never addressed.
- **22219 (medium)** — file-notes workspace: 1.5 s filesystem reconcile, 40×/min, no
  `screen.is_active` gate — keeps scanning under modals and other screens.
- **22220 (low)** — db-size stat burst (~15 syscalls) ON the loop every 120 s; Ollama
  probe constructs a worker every 3 s just for the in-coroutine gate to drop it;
  `InlineLoadingIndicator` tick never stopped after terminal states.
- Dead-timer inventory: `Tamagotchi` and `DetailedProgress` (1 Hz psutil import per fire)
  have NO production mount path — zero cost today, flagged so a future mount is a decision.

### G. Data-layer hardening

- **22224 (medium)** — `isolation_level=None` missing on held connections across ~8 stores
  INCLUDING the template (`Library_Ingest_Jobs_DB.py`) and ChaChaNotes; mechanism verified
  empirically: bare DML leaves an implicit DEFERRED txn that makes
  `transaction(immediate=True)` silently BORROW — defeating the 21100 hardening the moment
  one bare DML lands. Zero firing sites today (loaded gun, not a firing one).
  `Notifications/event_state_repository.py` is the one store that gets it right.
- **22225 (low)** — v48 seeds policy rows for deleted conversations.
- **22227 (low)** — character emote pipeline constants (per-character UTF-16 encode loop;
  16 regex passes per turn; O(assets²) snapshot projection).

### H. Dev health

- **22212 (high)** — the CSS allowlist ratchet is RED on pristine tip: PR #2053 (the tip
  merge) shipped `ConsolePromptComparisonModal.DEFAULT_CSS` neither allowlisted nor
  bundle-ridden. Every PR inherits the red. Otherwise the ratchet worked exactly as
  designed across the window: 26 class-CSS blocks removed, 31 new blocks correctly rode the
  bundle, ONE defector.

---

## Verified clean / regression audit (do NOT re-fix)

All 20 burn-down headline fixes re-audited at tip; every mechanism present and
load-bearing, with the hot path routed through it (workspace keystroke memo 21118;
selection-dismissal registries 21119; transcript-copy guard 21121 — the one new bypass is
22204's, which was already a bypass at the pin, just doubled; Inspector pure-scroll split
21117; blink layout gate 21692; drag-selection 21114; Library canvas projections 21116
(ratchet at exactly 97/97); lock-free config reads 21124; PIL/Chunking closure guards
21103/21200/21102 (import-time); six lazy feature DBs 21105; notes-sync gate + backoff
21112; splash overlap 21110 (delay 0.2 > 0; the `set_timer(0.0)` trap correctly branched);
dead token timer 21133; device store 21101; bindings projections 21129; research store
21127; event-state repo 21131 — the one store with `isolation_level=None` correct; import
weight guard 21108 at 637/660). Streaming is O(1) per chunk with 5 Hz coalescing intact;
markdown streaming append-only; reconciler move-guard intact; cost-chip gate intact;
Persona Buddy paint gate holds and post-dates the chrome rewrite (merge order verified);
auto-speak (#2028) is a pure ownership refactor; prompt workbench (#2053) is off the
run/keystroke path (except 22228 item 5); per-chat sandbox (#2021) is well-bounded
(3 syscalls once per session); subscriptions scheduler materially fixed (in-memory pop_due;
DB via to_thread every ~30 min); no new store shipped DELETE/FULL journaling (32/32 sites
NORMAL); connection creation centralized (one raw `sqlite3.connect` outside tests); both
new indexes verified chosen with `sqlite_stat1` absent; v47/v49 migrations are O(1) DDL
(v49 genuinely REDUCES per-update trigger work); no new `exclusive=True` with a falsy
group; no `set_timer(0.0)`; no `run_worker` positional-arg misuse; the reactive-mutable
inventory holds.

---

## Traps recorded (for the next review)

- **Python `sorted()` order ≠ `comm`'s C-locale order.** The first module-set diff produced
  false "tip-only" rows (`Text2SQL_Interop` etc. appeared new while present in both).
  `LC_ALL=C sort` both files before `comm`. The corrected diff changed the boot story.
- **Interleave A/B runs under identical load.** Tip-vs-base import first measured 1.4 s vs
  0.82 s — pure contention noise from seven concurrently-running review agents; interleaved
  re-runs showed tip equal-or-faster. Five interleaved TTI pairs were needed before the
  ~140 ms delta was trustworthy.
- **`python -c` from the wrong cwd resolves the MAIN checkout's package** even under the
  probe venv (the documented editable-install trap, fired again). Assert
  `tldw_chatbook.__file__` in every probe.
- **The default focus target matters:** the first keystroke probe focused
  `Input#compact-temperature` (first Input in walk order), not the composer — the composer
  IS the default focus, so the right probe presses keys without touching focus.
- **A "lazy import" comment is not a lazy import** if the enclosing function runs at module
  import (`config.py:1802` — `load_settings()` executes at config import).
- **Attribute before filing:** the Library-package-in-config-load finding shrank from
  "~100 ms on config load" to a layering note once the base closure was checked — the
  package was already there. Same class as wave-7's five corrected briefs.

## Probe recipes (reusable)

- SQL-per-interaction: wrap `sqlite3.connect` to `set_trace_callback` a global list BEFORE
  importing the app; count statements inside the interaction window.
- Layout passes: count `Screen._refresh_layout` calls via a subclass-safe wrapper.
- Module censuses: `sys.modules` snapshots at import / construct / `_ui_ready`, diffed
  between pins with `LC_ALL=C sort` + `comm`.
- Import chains: a `builtins.__import__` wrapper capturing the first-import stack of target
  modules (note: misses `importlib.import_module` callers — Actor_Packs needed the static
  AST closure instead).
- Worktrees left in place for the burn-down: `perf-review-2026-08-24` (pinned review tree),
  `perf-probe-2026-08-24` + `perf-base-2026-08-24` (venvs installed, scratch profiles under
  the session scratchpad are ephemeral — recreate per the Method section).

---

# Close-out amendment (2026-08-26) — what the burn-down proved, and where this document was wrong

All 29 filed tasks (TASK-22200…22228) were implemented, adversarially reviewed or
controller-verified, and merged to dev on 2026-08-25/26 across 6 single PRs and 6 batch PRs
(#2077, #2081, #2083, #2084, #2087, #2090/#2091, and batches A/B/C/D/E/F + #2110/#2111).

## Headline measured outcomes

| Surface | Before | After |
|---|---|---|
| Library reader, per traversal keystroke (1 MB doc) | 16.9 s | **0.40 s** (22207) |
| Conversation resume @600 / @2000 msgs | 110 ms / RecursionError | **5.2 ms / 21 ms** (22206) |
| Console run tick, registry SQL per 5 ticks | 400 | **0** (22201) |
| Send-path loop stall under a 2 s write-lock holder | 2049 ms | **11 ms** (22205) |
| Post-upgrade backfill, foreground write mean/max | 153-197 / 462-470 ms | **3.5 / 24 ms** (22200) |
| Worst keypress, first post-upgrade boot | 625-876 ms | **395-467 ms** (22215) |
| Avatar on-loop work per resize drag | 152.7 ms | **0.9 ms** (22221) |
| Composer blink tick (20 KB draft) | 1.58 ms | **0.11 ms** (22218) |
| Emote encodes per 16k-char reply | 16,000 | **329** (22227) |
| Match-nav handler (2.5 MB doc) | 52.4 ms | **7.5 ms** (22209) |
| `import tldw_chatbook.config` closure | 106 modules | **40** + a live circular import killed (22223) |

## Corrections to THIS document (the findings were wrong; the code was measured)

1. **Finding 22202's mechanism is false as written.** "One changed node invalidates every
   cached tree line" does not hold for what the run tick produces: Textual 8.2.8 keys
   `Tree._line_cache` per node (`_tree.py:1325-1333`) and `set_label` bumps only that node's
   `_updates`. Measured over 200 marker toggles: `Tree._invalidate` **0**. Only STRUCTURAL
   edits invalidate tree-wide, and that cost is **viewport-bounded, not row-bounded**
   (cold repaint 0.348 ms @50 rows vs 0.351 ms @200 at a 34-row viewport). The shipped fix is
   the equality fast-path (0.517 → 0.020 ms per unchanged push @200 rows).
2. **Finding 22228 items 1-2 were noise, and its item 1 fix would have been a pessimization.**
   `query_one("#id")` takes an id fast path with a per-node cache — **0.3 µs warm** on the
   475-widget Console — while `query("*")` is a real walk. The prescribed
   `_console_composer_or_none()` memo measured **slower** (0.7 µs). Four of that task's seven
   items were declined with measurements; the item that looked smallest (`query("*")` in
   `_focusable_body_controls`, 87.8 µs) was the only expensive one.
3. **Finding 22221 named 28 ms of a 215 ms cost.** The discarded-pixels resample was real but
   minor; the visible mosaic render (187 ms) was the bulk and had to move off the loop too.
4. **The +140 ms warm-TTI regression figure carries positional-bias doubt.** TASK-22213's A/A
   control found a ±400 ms noise floor with a systematic second-position advantage in
   interleaved boot pairs; this review's pairs ran tip-first throughout, so the true regression
   is **at most** the reported number. The mechanism findings (import-leg growth, pre-importer
   payload, worker count, CSS bytes) are unaffected — each was confirmed on deterministic axes.
5. **Finding 22220 named the wrong class**: the cited lines are `InlineLoader`, not
   `InlineLoadingIndicator` (which was already clean).
6. **Finding 22211's scrollbar note**: `workbench.size.width` is `content_region` and is
   scrollbar-sensitive only via an ANCESTOR, not its own scrollbars; a genuine scrollbar
   appearance never triggers re-resolution at all (benign, pre-existing).
7. **Finding 22214 confirmed, with an honest limit**: the 0.10 s cap really had turned the
   proportional yield back into the flat sleep it replaced (requested gaps
   `[0.0, 0.1, 0.1, …]` → `[0.0, 0.529, 0.245, …]`), but the low-core tier is a **wash** in
   both cache states — shipped as hardening, not a measured gain.
8. **Finding 22225 is not a boot-time win**: the added cleanup costs slightly more than the
   inserts it saves, once. Its value is 2,000 fewer permanent rows and a 114,688 B smaller
   file on the fresh-upgrade path.

## Follow-ups filed at close-out (TASK-22500…22506)

**22500 is the important one**: the reader body is not virtualized, so a 2.5 MB document
repaints all 45,000 lines and costs **~1.4-1.5 s per click both before and after** everything
this burn-down fixed around it — the single largest remaining Library cost. Also filed: 22501
(`add_conversation` DEFERRED writer dies un-retried, 3/3 repro — this refutes 22200's own
description), 22502 (cancel-during-CAS wedge), 22503 (`Library/__init__` lazy facade), 22504
(`console_voice_input` off the first-paint leg), 22505 (8 TieAware full reparses in the
first-paint window), 22506 (dead `loading_states` module — owner call).

## Process notes worth keeping

- **The dominant pattern held for a third programme running**: a finding tells you where to
  look, not what to do. Two findings were refuted by measurement, one was 13% of its own cost,
  four were declined as not worth fixing, and two shipped as honest washes.
- **Adversarial review earned its cost every time it ran**: 22201's fingerprint could lose its
  store half with 161 tests green; 22203's geometry gate was vacuous (`layout=False` freezes
  the stale region, so a CSS unpin could never red it); 22204's avatar-hidden fence clear was
  load-bearing and uncovered; 22200's abort-poll inside the backoff sleep was unpinned; 22224's
  reviewer re-censused 319 execute sites from scratch and found zero piecewise-commit windows.
- **Implementers deleted their own work when it did not earn its place**: 22221 removed a fence
  that survived mutation rather than ship an unkillable line; 22225's mutation test exposed a
  hole in its own suite (the v50 cleanup masked a broken v48 seed, leaving all 33 tests green).
