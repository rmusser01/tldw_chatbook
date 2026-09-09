# Library Screen Decomposition — Design

**Status:** draft for review, 2026-09-01
**Parent doctrine:** `Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`
(approved rev 3) and `DESIGN.md` §7. That doctrine's "Order and delivery" section calls
for one implementation plan per screen; Console's ran (waves 1–6, closed). This is the
Library plan. Everything the doctrine settles is adopted here **by reference**, not
restated: the One Rule (a region widget owns pixels; a controller does not), the One
Home Rule (`UI/Library_Modules/`), the Six Migration Rules, the dependency-naming canon
(`ConsoleDictationController.__init__`), and the testing rules. This document covers
only what is Library-specific.

## Why now — the measured problem, one month on

| | doctrine baseline (2026-08-02) | today (2026-09-01) |
|---|---|---|
| File lines | 15,819 | **46,109** |
| `LibraryScreen` methods (ast) | 508 | **1,270** |
| Class lines | 14,973 | 43,814 |
| `__init__` lines | — | 1,305 |
| `compose_content` lines | 381 | 796 |
| `@on` handlers | — | 379 |

The file nearly **tripled in the month** since the doctrine was approved, absorbing 428
commits in the last 30 days. The cause is exactly the one the doctrine's own
retrospective predicted ("every one of those lines went into the screen because the
screen was the path of least resistance"): `Tests/Architecture/test_screen_size_ratchet.py`
— the mechanism built to stop this — **has a budget row only for `chat_screen.py`.**
Library was never added. Console shrank under its ratchet while Library, unguarded,
grew without limit. Closing that gap is this plan's first, cheapest, and most urgent
step.

Secondary motivation, measured 2026-09-01 with a headless click probe
(`Helper_Scripts/library_click_probe.py`, checked in by this plan): every rail-mode
switch blocks the main thread 139–380 ms (mount-storm of a 113–165-widget canvas
subtree, per-node CSS apply), independent of data volume. The freeze fix itself is
**out of scope here** (Six Migration Rules, rule 5: extractions change no behaviour) —
but it is the first named phase-C candidate below, and this decomposition is what makes
it safely buildable.

Facts that shape the plan, verified against source:

- **Zero `reactive` attributes and zero `watch_` methods** on `LibraryScreen`. All 112
  `__init__`-assigned fields are plain attributes — state can move into plain objects
  with no reactive entanglement.
- State and methods cluster cleanly by subsystem prefix: media (~94 methods), notes
  (~138), prompts (~84), skills (~70), conversations (68), ingest (~37), plus
  export/collections/search/RAG and focus/sync plumbing.
- The canvases are **already region widgets** (`Widgets/Library/`:
  `LibraryMediaCanvas`, `LibraryNotesCanvas`, the prompts/skills canvases, reader
  shells). Library's remaining monolith is therefore overwhelmingly **controller-shaped**
  work under the One Rule — the region-widget half largely exists.
- 90 `test_library*` UI test files, ~2,470 test functions, 144 test files touching the
  screen overall: a dense regression net.
- 100 test files poke 402 distinct private screen attributes; 37 sites monkeypatch
  `LibraryScreen` members (mostly framework members that stay; ~8 hit subsystem
  methods).
- ~90 module-level support names sit above the class in the same file; controllers
  need them, so they move out first or every extraction hits an import cycle.

## Goal and non-goals

**Goal:** full decomposition. `LibraryScreen` ends as layout, routing, cross-subsystem
coordination, and collaborator wiring — the Evals-screen shape, realistically ~4–6k
lines (379 `@on` delegator one-liners are the honest floor cost of Textual's
message-routing model; phase C lowers it per subsystem later).

**Non-goals:** no behaviour changes of any kind in extraction PRs (the freeze fix, the
`browse-collections` 219 ms service call, and every other wart land as separate,
attributable changes); no renames-for-taste; no merging of the existing browse
controllers into the new controllers during migration; no settings-screen work (its
missing ratchet row is noted for a follow-up task, not taken on here).

## Constraints the plan is built around

1. **Pure moves only, at the canon's strictness.** Moved method bodies are
   **byte-for-byte unedited** — the `ConsoleDictationController` mechanism: the
   constructor binds, under the same names the bodies already use, every referenced
   name that is not the controller's own (named callable dependencies; framework
   services live-read from the screen via `@property`; generated controller-local
   properties for the subsystem's own state fields). The only other transforms an
   extraction PR may contain are import-path changes and delegator insertion on the
   screen for externally-referenced names. Receiver normalisation (direct
   `self._state.` access, dropping screen-routed same-subsystem hops) happens in that
   subsystem's cleanup PR, never in a move PR. Nothing else.
2. **Interleaved with feature work.** Small PRs landing on dev continuously; no
   feature freeze. Never two subsystems' extraction PRs in flight at once.
3. **One region or one controller per change** (Six Migration Rules, rule 1) — this
   supersedes any batching instinct; a "facet" below is exactly one controller.
4. **Doctrine testing rules apply verbatim**, including: DOM-driving tests pass
   unchanged; private-method tests are retargeted with assertions byte-for-byte;
   characterisation tests precede any extraction of uncovered behaviour;
   painted-geometry assertions (hit-testable controls at 160x45 and 235x52) accompany
   any change that moves DOM — most Library extractions move none, because the region
   widgets already exist.

## Library-specific design

### Collaborator inventory under the One Rule

Each subsystem gets, in `UI/Library_Modules/`:

- **A controller** (`library_media_controller.py`, …) owning the subsystem's moved
  methods. Constructor is the dependency list, per the
  `ConsoleDictationController.__init__` canon: the screen handle is taken **only** for
  the live-read framework-service properties (`run_worker`, `post_message`,
  `set_timer`, `is_mounted`, …); everything else is a named constructor dependency —
  the wave-1 "reach through `screen`" third kind is retired, per that canon's own
  docstring. Controllers never import each other; cross-subsystem effects go through a
  named screen-provided callable, so the screen stays the one visible mediator.
- **A state object** (`library_media_state.py`: `LibraryMediaState`, plain mutable
  dataclass) holding the fields that subsystem exclusively owns, moved verbatim with
  identical defaults; computed defaults become constructor arguments so `__init__`
  evaluation order is preserved.

Rules that resolve the known hard cases:

- **Shared fields stay on the screen.** A field referenced by ≥2 subsystems
  (`_library_selected_row_id` 226 refs, `_library_lifecycle` 83,
  `_library_snapshot_state_generation` 35, `_pending_library_source_open` 29, …) is
  shared shell state, accessed via named dependencies — never forced into a subsystem
  state object. Ownership is determined mechanically (ref-count by method cluster) and
  recorded per subsystem in the recipe.
- **The `*_local_source_snapshot` trio is shared shell infrastructure** (it feeds
  notes+media+conversations counts, `_refresh_local_source_snapshot` has 29 internal
  call sites, and tests patch it on `LibraryScreen`). It stays screen-routed.
- **Any name tests patch on `LibraryScreen` keeps its whole call graph routed through
  the screen** until that subsystem's cleanup PR retargets the tests to the
  controller. This prevents the monkeypatch-bypass failure (a moved internal call
  skipping a screen-level patch) from breaking "tests pass unchanged" mid-migration.
- **`@on`/`action_` members stay on the screen as one-line delegators** — Textual
  resolves bubbled messages and bindings along the DOM/focus path, and controllers are
  not on it. The delegation table is the screen's routing role, kept deliberately.
- **Migration shims:** while a subsystem's references migrate, the screen carries
  generated getter/setter `@property` shims for its moved fields, between sentinel
  comments, deleted wholesale by that subsystem's cleanup PR (the one PR type allowed
  to edit tests — attribute-path retargets only, assertions byte-for-byte). Verified
  safe: no Library code or test reaches these fields via `vars()`/`__dict__`.

### Order of work

**PR 0a — support layer.** The ~90 module-level names above the class (support
classes, `_sync_library_canvas`, `_is_ingestible` and the ingest-shortcut tables,
constants) move to `UI/Library_Modules/`, with re-export aliases left in
`library_screen.py` (75 test files import from the module; besides `LibraryScreen`
itself only 5 imports depend on these names). This unblocks every later PR from import
cycles and is required before any controller can exist.

**PR 0b — guards and recipe.**
- Add the missing `library_screen.py` row to `_BUDGETS` in
  `Tests/Architecture/test_screen_size_ratchet.py` at the exact post-0a measurement.
  Budgets only go down, per that file's own contract — no grace band; the ratchet's
  failure message is the enforcement of "new Library code lands in
  `UI/Library_Modules/` from day one," including for not-yet-extracted subsystems
  (a subsystem's controller file may be created early to receive new methods).
- Widen `Tests/UI/test_library_recompose_ratchet.py` to count whole-screen-recompose
  statements across the screen **plus** `UI/Library_Modules/` as one surface, so moves
  cannot silently drain the TASK-21116 pin.
- Check in `Helper_Scripts/library_click_probe.py` (the headless before/after
  instrument) and the recipe doc (`backlog/docs/`) that every subsequent PR follows.
- Start the `.git-blame-ignore-revs` list; every pure-move commit is appended, so
  blame keeps resolving to the scar tissue's real authors.

**Per-subsystem series** — state PR (fields + shims), one-controller-per-PR moves
(each move PR lowers the `library_screen.py` row in `_BUDGETS` to its own
post-move measurement, per the ratchet file's lower-in-the-same-PR contract —
not deferred to cleanup), cleanup PR (shims and dead delegators deleted, tests
retargeted, budgets lowered — the cleanup PR's lowering is the series' final
one, not its only one). Sequenced cold-to-hot so the exemplar never fights
rebases and hot subsystems migrate in short, fast series once the recipe is
rehearsed (churn = commits touching the file in the last 30 days whose
subjects name the subsystem):

(Corrected during execution of the conversations exemplar: the size-ratchet
slack guard went red between the reader controller's move commit and its
pin-lowering commit, because the pin lowering had been deferred to the
cleanup PR instead of landing with the move — the per-move-PR lowering above
is the fix.)

1. **conversations** (exemplar: 68 methods, 19 fields, churn 10; lowest
   cross-coupling — 3 notes refs plus shared fields already handled above)
2. **export** (churn 3), **collections** (6), **search** (6) — recipe rehearsal
3. **skills** (15), **RAG/onboarding plumbing** (16), **ingest** (23)
4. **prompts** (41), **media** (55), **notes** (72; most scarred; its sync controller
   already lives in `Library_Modules/`)
5. Final shell pass: residual focus/lifecycle plumbing, delegator table tidy,
   `compose_content` reduced to the region-yielding skeleton.

Roughly 35–50 small PRs. Every intermediate state ships.

**Rollback policy:** a landed extraction implicated in a regression is **reverted**,
not fixed forward — pure moves revert cleanly, and single-candidate attribution is the
property the pure-move policy exists to buy.

## Phase C — region ownership (after a subsystem's series completes)

Phase C is the doctrine's region-widget endgame applied per subsystem: `@on` handlers
for **canvas-origin** messages migrate from the screen's routing table into the
already-existing canvas widget, bindings move with behaviour ownership, and the state
object moves from screen-held to widget-held. Scope honesty: messages originating in
the rail, footer, or header can only be caught at the screen — those delegator rows
are permanent; phase C shrinks the table, it cannot empty it.

Graduation criteria, all three required: the subsystem's phase-A series is fully
landed including cleanup; its mounted coverage is dense (characterisation tests added
where the pre-series spot-check found gaps); and a concrete motivating change exists —
phase C is never done for its own sake. **First motivated candidates: media and
notes**, whose motivating change is the resident-canvas fix for the measured 139–380 ms
mode-switch freeze (probe numbers above are its before/after acceptance evidence).
Each graduation is its own designed behaviour-change series, explicitly outside the
pure-move policy.

## Risks

| Risk | Mitigation |
|---|---|
| Concurrent feature work re-inflates the screen mid-migration (Console lost ~5,500 lines of gains this way) | The 0b ratchet row lands before any extraction; failure message names the controller destination, including early controller files for unextracted subsystems |
| Monkeypatch bypass breaks tests inside a "pure move" | Screen-routed call graphs for test-patched names until cleanup; snapshot trio stays shared infrastructure |
| Import cycles between screen and controllers | PR 0a support-layer move first; re-export aliases preserve the module's import surface (and the task-15472 preimport behaviour) |
| Recompose ratchet silently drained by moves | Widened to screen+modules surface in 0b, before the first move |
| Subtle regression surfaces days after a move lands | Revert-don't-fix-forward; `.git-blame-ignore-revs` keeps archaeology usable |
| Stale-base budget numbers (Console wave 3 landed red twice this way) | Doctrine rule adopted: measure after final rebase, lower budgets in the landing PR itself |

## Relationship to the parent doctrine — deltas, declared

Three ideas from this plan's drafting were **revised in favour of doctrine precedent**
once Console's execution record was read: a bespoke typed `LibraryScreenHost` facade
(superseded by the named-constructor-dependency canon — visible coupling over a
concealed facade); a grace-band size ratchet (superseded by the existing
budgets-only-go-down contract, which Console proved livable at comparable churn); and
a receiver-rewrite transform whitelist (superseded by the stricter byte-for-byte body
discipline the dictation extraction demonstrated — bodies unedited, names rebound in
the constructor). Everything else here extends the doctrine without contradicting it.

---

## Design record — phase C, media: the resident-canvas mechanism

**Status:** decided, 2026-09-08. **Plan:**
`Docs/superpowers/plans/2026-09-08-library-phase-c-media-graduation.md`, Task 1.
**Instrument:** `Helper_Scripts/library_switch_teardown_probe.py` (added by this
task) beside the existing `Helper_Scripts/library_click_probe.py`.
**Acceptance pin:** `Tests/UI/test_library_phase_c_switch_residency.py`
(red by construction until Task 2 lands).
All numbers below were measured on `b81cb98b0` from a **scratch worktree**
(`git worktree add --detach`, own `uv venv`, `diff -r` against the working tree
clean), per recipe §9's same-checkout-location rule.

### Context — what a rail-mode switch actually does today

Recipe §25 handed phase C a band (settle 243–494 ms, max gap 37–179 ms, mounts
177/89/85/114/38/175/114/175, **recompose 0**) and a target: "the re-click rows
go to ~0 mounts with `recompose` still 0". Instrumenting the switch changed the
shape of the problem in two ways before any mechanism was evaluated.

**1. That `recompose 0` is an instrument artifact.** The click probe counts
`BaseAppScreen.refresh(recompose=True)`.
`_select_library_rail_row_after_source_admission` instead **awaits
`Widget.recompose()` directly** (`library_screen.py`, the `if not replaced:`
arm after `_replace_library_browse_canvas`), which that column cannot see.
Every media/notes rail switch measured runs exactly **one whole-screen
recompose**. The targeted path is entered and bails in 0.6 ms: it requires the
destination's contextual chrome to already be mounted, and media↔notes always
crosses that boundary (`#library-media-reader-shell` for media,
`#library-notes-source-strip` for notes), so `_replace_library_browse_canvas`
returns `False` on every real mode switch and succeeds only on a re-click of
the already-selected row.

**2. The click probe's `removes` column is dead.** Textual 8.2.8 prunes through
`Widget._message_loop_exit` (which performs the `App._registry.discard`), not
through `App._unregister`, so a counter on `_unregister` reports zero
removals forever — which is what "N mounts / 0 removes" has printed for four
waves. Hooked correctly, a media switch unmounts 162–176 widgets.

#### The freeze-band composition table

Per click, from the teardown probe. `block` = longest single main-thread block
(the freeze; the click probe's `max_gap`). `cpu` = sum of the additive
synchronous buckets = measured main-thread work.

| interaction | block | cpu | mounts | unmounts | targeted swap | whole-screen recompose |
|---|---|---|---|---|---|---|
| media (switch-in) | 132 ms | 285 ms | 177 | 162 | tried, **False** | **1** |
| media (re-click same) | 59 ms | 117 ms | 85–89 | 85–89 | **True** | 0 |
| notes (switch) | 154 ms | 180 ms | 114 | 121 | tried, **False** | **1** |
| notes (re-click same) | 53 ms | 66 ms | 38 | 38 | not called | 0 |
| media (switch-back) | 90 ms | 205 ms | 175–179 | 172–176 | tried, **False** | **1** |

Where the work goes, media switch-in (additive, self time):

| bucket | ms | share |
|---|---|---|
| CSS restyle (`Stylesheet.apply`, 534 calls) | 111.0 | 39% |
| render / paint prep (`Compositor.render_*`) | 76.4 | 27% |
| widget construction (`textual.compose`, 182 calls) | 47.0 | 17% |
| DOM registration (`App._register`, 59 calls) | 38.2 | 13% |
| compositor reflow | 11.5 | 4% |

Where the mounts go, media switch-in, by region (the full breakdown; it sums
to the total, which is the point of the table):

| region | mounts |
|---|---|
| canvas (media) | **87** |
| rail | **52** |
| nav bar (the bar, its 16 destination buttons, its 2 overflow hints) | 19 |
| footer | 6 |
| screen chrome (header line, chunking tools, lifecycle status) | 6 |
| reader shell + its 2 pane grips | 3 |
| media viewer | 2 |
| shell grid | 1 |
| canvas host | 1 |
| **total** | **177** |

Notes (switch) decomposes the same way and also sums exactly: rail 52 +
canvas (notes) 19 + nav bar 19 + screen chrome 11 + footer 6 + shell grid 6 +
canvas host 1 = **114**.

#### Where §25's mount numbers come from — the arithmetic closes

Measured DOM sizes: the media route is **119** nodes and the notes route
**114**; the media canvas subtree is **29** widgets (`LibraryMediaCanvas` + 28
children) and the notes canvas subtree **19**.

* **notes (switch) = 114 mounts = the entire notes route tree, built once.**
* **media (switch-in) = 177 = 119 (the whole route tree) + 2 × 29 (the media
  canvas built twice MORE).** The extra builds are visible in the probe's
  causal trace (`LIBRARY_TEARDOWN_TRACE=1`): after the recompose returns at
  114 ms, **four** `LibraryMediaCanvas.sync_state` calls land at 118.5, 118.6,
  123.9 and 124.0 ms. Each is `self.refresh(recompose=True)` — a canvas-scoped
  rebuild of the whole child list. Consecutive ones coalesce, so four calls
  produce two rebuilds.
* The same redundancy is worse elsewhere: **notes (re-click same) fires seven
  `LibraryNotesCanvas.sync_state` and seven `LibraryNoteWorkPane.sync_state`
  calls inside 6 ms.**

So the freeze has two independent causes, and the plan's framing named only the
second: a **whole-screen rebuild** on every mode switch, and a **redundant sync
storm** that rebuilds the canvas two to seven times per click.

### Options, with measured deltas

Each mechanism was spiked on the real mounted canvases and measured with the
same buckets. n = 8 switches per process run; `ref` and `A` were run
**interleaved and order-swapped** (three rounds each way, six runs per arm) from
the same scratch worktree, per recipe §9. The order swap moved nothing:
`ref` block medians were 124.2 / 121.3 / 123.8 running first and 125.1 / 125.8 /
123.9 running second.

| mechanism | block (median) | cpu (median) | mounts | unmounts | nodes at rest |
|---|---|---|---|---|---|
| **ref** — today's rail switch | **124 ms** | **187 ms** | 114–179 | 121–176 | 119 |
| **A** — both canvases resident, `display` toggled | **30 ms** | **15 ms** | **0** | **0** | 136 (+17) |
| **B** — `remove()` then re-mount the same instance | 60 ms | 64 ms | 29 | 29 | 119 |
| **C** — mount-once-then-toggle (lazy residency) | **29 ms** | **14 ms** | **0** | **0** | 136 after first visit |

C's first visit pays one mount: block 40–47 ms, cpu 38–49 ms, 17 mounts —
already ~3× cheaper than today's switch, and paid once per route per app run.

Under A/C the per-switch buckets collapse to render 12 ms, reflow 3 ms, **css
1.2 ms** — the CSS restyle that owns 39% of today's work is a function of the
tree being rebuilt, not of the stylesheet.

**Threshold, stated before choosing:** a mechanism must (i) take steady-state
mounts to **0** and (ii) at least **halve** the longest main-thread block.
A and C clear it (0 mounts; 124 → 30 ms, −76%; cpu −92%). B fails (i) outright.

### Textual 8.2.8 behaviour, verified rather than assumed

1. **`display = False` keeps the widget registered.** `DOMNode.display`'s setter
   writes `styles.display = "none"`; the node stays in `App._registry` and in
   every query. Its docstring records that the setter *forgets* the original
   value (False→True resets to `block`, not to the CSS-specified display), so
   phase C must toggle a class or save/restore the rule, never bare booleans, on
   any widget whose stylesheet sets a non-`block` display.
2. **Hiding the focused subtree drops focus to `None`.** Measured: focus went
   `Button#library-media-review-sets` → `None` on the toggle. Phase C must move
   focus explicitly *before* hiding, or a switch lands the user nowhere.
3. **A hidden widget still receives and dispatches events.** `press()` on a
   `Button` inside a `display:none` subtree produced `Prompt`, `Pressed`,
   `Callback`. A resident-but-unselected canvas WILL process row events unless
   explicitly gated — the plan's Task 2 ruling is required, not precautionary.
4. **`remove()`-then-re-mount of the same instance is accepted** in 8.2.8 (the
   canvas came back in the DOM with 28 children, `is_running` True,
   `_closing`/`_closed` False) — but it re-composes, so it is a remount wearing
   the old instance's identity, not a reattachment. See "rejected" below.
5. **`Widget.recompose()` destroys residency.** A resident canvas mounted beside
   the active one is gone (`0` hits) after one `screen.recompose()`, because
   `recompose` removes every non-system child. **No residency mechanism can work
   while the rail switch calls it.**
6. **The toggle repaints by itself; no explicit refresh is needed.** Writing
   `styles.display` produced a reflow (3 ms) and a render pass (12 ms) per
   switch with no `refresh()` call in the spike, and **no CSS re-apply beyond
   1.2 ms** — the stylesheet is not re-matched against an already-registered
   subtree. That 15 ms is the entire mechanism cost. A/C's 30 ms `block`
   is larger because a block is measured between pilot polls and also contains
   loop time this instrument does not attribute (timers, message-pump turns,
   poll granularity) -- so 15 ms is the number to compare against `ref`'s
   187 ms, and 30 vs 124 ms is the conservative reading.

### Composition rulings

**TASK-31521 screen reuse — compatible, no conflict.** `on_screen_suspend` /
`on_screen_resume` stop and restart timers and re-kick visit surfaces; neither
touches the DOM. Measured with a resident hidden canvas across a
suspend/resume cycle: node count 136 → 136, the hidden canvas's
`styles.display` stayed `"none"`, `_library_screen_suspended` toggled correctly.
The "is this surface live" predicate the resume path already uses
(`_library_media_list_surface_active`) is gated on
`_library_selected_row_id != LIBRARY_ROW_BROWSE_MEDIA` — a **state** check, not
a DOM-presence check — so it stays correct under residency for free.

**But DOM presence is used as a route proxy elsewhere, and residency breaks
it.** `#library-media-reader-shell` appears at **17** non-comment sites and
`#library-media-canvas` at **7** — 5 as a query selector, plus **2 written as
an id EQUALITY comparison** (`any(widget.id == "library-media-canvas" for
widget in focused.ancestors)`, `library_screen.py:14306` and `:14327`). Several
read presence as "the media route is active" (`_library_media_escape_label`
returns `"back"` when the shell is absent; `library_notes_controller`'s
`adaptive_media = bool(self.query("#library-media-reader-shell"))`). Every one
of those becomes permanently true under residency. Task 2 must convert the
route-meaning ones to the `_library_selected_row_id` state check and leave the
genuinely structural ones alone; this is the largest single piece of Task 2's
work and it is not optional.

**Its census must not be a selector-string grep.** The two equality sites are
*ancestor-walk* checks — "is the focused widget inside the media canvas" — which
is precisely the route-proxy shape residency breaks (the focused widget can now
sit inside a canvas that is resident but hidden), and a
`grep '"#library-media-canvas"'` scores zero on both. This is recipe §3's
fifth-spelling lesson arriving in a new costume: the same name, spelled as a
comparison rather than a selector.

**`canvas_sync` and the TASK-31880 blanket-`except` finding — a NEW failure mode,
and the swallow must be narrowed.** Today `_sync_library_canvas(screen, "media")`
called while media is not the route raises `NoMatches` from its `query_one`,
is caught by the function's blanket `except Exception`, and falls back to
`screen.refresh(recompose=True)`. Measured under residency: the same call
**returns `True` and rebuilds the hidden canvas's 28 children**, because the
query now succeeds. There is no exception for the blanket `except` to swallow —
the coverage-solvent property TASK-31880 recorded is not merely preserved here,
it is upgraded to a silent success. Ruling for Task 2, to land as its own
TASK-32089-labelled commit: `_sync_library_canvas` gains an explicit
**route-ownership guard** (sync only the canvas whose kind owns
`_library_selected_row_id`) ahead of the `try`, and the blanket `except` is
narrowed so a genuinely unexpected failure raises instead of being converted
into a whole-screen recompose. Without that guard, residency silently redirects
the sync storm onto invisible canvases.

*The dispatcher's OTHER known hazard is untouched by residency.* Whether
`_sync_library_canvas` is handed a screen or a controller as its `screen`
argument is decided by the caller; residency changes what the DOM query finds,
not who calls it, so the two-receiver dimension recipe §3 earned is neither
helped nor worsened here. Re-derived at this record rather than quoted from
`canvas_sync.py`'s own comments (which name only the notes movers and the RAG
kind): **seven** of the program's controllers forward a bare `self` to this
dispatcher — conversations, ingest, media, notes, prompts, rag_search, skills —
and **all seven reference `_library_selected_row_id`**, which is exactly why it,
and not a DOM query, is the right predicate for the route-ownership guard. That
guard must be written to read through whichever receiver it is given.

### Decision

**Mechanism C — mount-once-then-toggle lazy residency for the Library canvas
host — with the whole-screen recompose on the rail-switch path replaced first.**

The second half is not a caveat, it is the larger half. Finding 5 makes it a
precondition: while `_select_library_rail_row_after_source_admission` awaits
`self.recompose()`, no canvas can be resident. So Task 2's real scope is:

1. Extend the targeted route update so a media↔notes switch no longer needs the
   whole-screen seam — the two routes' structural delta is small and known
   (media wraps its panes in `LibraryMediaReaderShell`, a **subclass** of the
   `LibraryAdaptiveReaderShell` notes uses; notes adds
   `#library-notes-source-strip` above the grid), and the nav bar and footer are
   identical across them. **Screen chrome is not** — 6 mounts on the media route
   against 11 on notes — and the 5-widget difference is exactly the Notes source
   strip named in the previous clause, i.e. the one structural delta, not a
   second one.
   **The rail is the constraint here, and it is not obvious from the outside.**
   `LibraryRail#library-rail` is a CHILD of the route's reader shell, not a
   sibling of it (`#library-shell-grid > LibraryMediaReaderShell >
   LibraryRail`), so the rail's 52 mounts are unavoidable for as long as the
   shell is swapped per route. Keeping the rail resident therefore requires the
   shell itself to be resident or shared, and the two shells' ids differ
   (`#library-media-reader-shell` vs `#library-notes-reader-shell`) precisely so
   they can be told apart. That id split is what step 1 has to resolve — it is
   the reason **55** of the 177 mounts exist at all (rail 52 + the shell and its
   two pane grips 3, straight off the region table above).
2. Keep each visited canvas mounted in the canvas host and toggle visibility on
   switch, mounting a canvas the first time its route is entered.
3. Gate a resident-but-unselected canvas out of event handling (finding 3) and
   out of `_sync_library_canvas` (the route-ownership guard above).
4. Move focus before hiding (finding 2).

**Why C over A:** their steady states are indistinguishable (block 29 vs 30 ms,
cpu 14 vs 15 ms, both 0 mounts), but `_sync_library_canvas` dispatches
eleven canvas kinds and eager residency would pay resident nodes for routes the
user never opens.
Lazy residency bounds the cost to visited routes and its one-time entry cost
(40–47 ms) is already 3× better than the switch it replaces.

**Consequences.**
* Steady-state DOM grows by one canvas subtree per visited route: measured
  **+17 nodes** for media + notes resident (119 → 136, +14%). Media's own
  subtree is 29 widgets and notes' is 19; `_sync_library_canvas` dispatches
  **eleven** canvas kinds, so a user who visits them all pays a resident
  subtree for each (only the two measured here are known sizes). Node count is
  the memory proxy this instrument can measure; if that becomes a concern the
  eviction policy is "keep the last N visited", which C's lazy structure
  already supports and A's does not.
* The redundant sync storm (2–7 canvas rebuilds per click) is **not** fixed by
  residency — it becomes cheaper per rebuild but stays a rebuild. It is the
  obvious next motivated change and is deliberately out of Task 2's scope; the
  acceptance pin below does not depend on it.
* `display`-toggle semantics forget the CSS-declared value (finding 1), so the
  implementation toggles a class, not the boolean property.
* Seventeen `#library-media-reader-shell` presence sites change meaning.

### Rejected, and why

* **B, widget-instance cache with detach/reattach.** It *works* in Textual 8.2.8
  — that was the surprise — but the re-mount re-composes, so it costs 29 mounts
  + 29 unmounts and 17 ms of widget construction per switch, landing at block
  60 ms / cpu 64 ms: **2× worse than A/C on every column** while also depending
  on re-mounting a widget after `remove()`, whose docstring reads "Remove the
  Widget from the DOM (effectively deleting it)" — no contract that a deleted
  widget may return, so nothing stops any 8.x release from closing it. Rejected
  on measured cost first, unsupportedness second.
* **A, eager residency of every canvas.** Identical on the switch, strictly
  worse at rest, and unbounded across eleven canvas kinds.
* **Optimising the CSS restyle instead.** It is the largest bucket (39% of
  285 ms) and looks like the obvious target, but under residency it falls to
  1.2 ms per switch: the restyle cost exists *because* the tree is rebuilt.
  Attacking it directly would be optimising the symptom, and the same reasoning
  retires "make `Stylesheet.apply` cheaper" as a phase-C candidate.
* **Keeping the whole-screen recompose and making it faster.** Ruled out by
  finding 5: it is not compatible with any residency at all, so no amount of
  making it faster reaches the 0-mount target.

### Acceptance evidence, and what it replaces

`Tests/UI/test_library_phase_c_switch_residency.py` pins the switch
**structurally**, not on a settle band: zero whole-screen `LibraryScreen.
recompose()` calls, and ≤ 25 mounts / ≤ 25 unmounts per media↔notes rail switch.
Both quantities are load-independent — recipe §9's own rule is to read those
columns as the verdict and wall-clock as context — so the pin is deterministic
on a loaded machine, which a settle-time pin is not. Measured red today:
114 mounts / 120 unmounts / 1 recompose (notes switch) and 179 / 175 / 1
(media switch-back). The 25 ceiling is more than three times the measured
structural delta between the two routes -- the 5-widget Notes source strip plus
the 2-widget media viewer, seven widgets in total -- and still a 7× cut on
today's 177.

Ceiling provenance: the 25 derives from a mechanism floor measured **without**
the reader-shell id split resolved, whose own mount cost was therefore
unmeasured (the test's docstring carries the same note). If Task 2 finds 25
tight, re-derive the ceiling from a fresh measurement with the split resolved
and record the derivation here -- never quietly raise the number.

Wall-clock stays in this record as context, not as a gate: **block 124 → 30 ms
(−76%), cpu 187 → 15 ms (−92%)** at the mechanism's measured floor. The landed
change will sit above that floor because a real switch also re-applies the rail
selection, header, footer context and focus; the floor is what the mechanism
costs, not a prediction of the landed number.

---

## Implementation addendum — phase C task 2, as landed (2026-09-08)

**Status:** landed. Commits `7b4a7e289` (the mechanism) and `51533602c`
(the TASK-32089 hazards) on `feat/library-phase-c-resident-canvas`.
Acceptance pin `Tests/UI/test_library_phase_c_switch_residency.py` is green
and its strict-xfail marker is gone. This section records what the decision
above could not: **how the reader-shell id split was resolved**, what the
landed switch actually costs, and the two places the record's own predictions
turned out wrong.

### 1. The id strategy — a route-neutral id plus marker classes

The decision above named the shell split as "the largest single piece of Task
2's work". Its resolution:

* **One `LibraryBrowseReaderShell`**, `Widgets/Library/library_browse_reader_
  shell.py`, evolved from `library_media_reader_shell.py` (which is gone; the
  Media shell was a subclass of the shared one, and the shared one is now what
  both browse routes mount). Its id is **`#library-browse-reader-shell`** —
  route-neutral, because it is the same widget on both routes and naming it
  after either would be a lie a future reader pays for.
* **Route meaning moves to a marker class.** The shell wears
  `.library-media-route` or `.library-notes-route`, and `apply_route` is their
  only writer, called from BOTH the compose path and the resident switch path.
  This is what preserved the ~35 production sites the decision above worried
  about: a site that read "is `#library-media-reader-shell` mounted?" as "is
  the media route active?" becomes `.library-media-route` and keeps its exact
  meaning. Shell PRESENCE is now permanently true; the marker is not.
  * *Why a class and not 35 `_library_selected_row_id` checks.* A marker
    projected from state at one seam IS the state check, written once. The
    hazard the record names is a proxy that goes permanently true, not a proxy
    as such; a marker with a single writer cannot. It also kept the census
    mechanical (`#` -> `.`) across 100 test-side occurrences of the retired
    id tokens (measured with `git grep -h -o` at the base commit — occurrences,
    not lines), which is the
    difference between a reviewable diff and an unreviewable one.
  * The two **ancestor-walk equality sites** the record warned would score
    zero on a selector grep (`widget.id == "library-media-canvas"`,
    `library_screen.py`) needed no change at all: the *canvas* id is
    unchanged. It is the SHELL id that moved. The record's census warning was
    right about the shape and wrong about which name it applied to.
* **Grips are shared, so their ids are neutral too**:
  `#library-browse-library-grip` / `#library-browse-items-grip` (was
  `#library-media-*` / `#library-notes-*`). The per-route parts of a grip
  travel with `apply_route`: the `library-media-pane-grip` visual class, and
  the items pane's user-visible label ("Notes" on one route, "Items" on the
  other — it is the grip's tooltip and accessible name).
* **CSS implications: none.** Neither retired id appeared in any stylesheet —
  every one of the 35 sites was a Python query selector, and
  `test_library_adaptive_reader_shell.py` already asserted the one CSS rule
  that used to exist had been removed. Widths and pane geometry are set
  imperatively by `sync_layout`, not by id selectors.
* **Who owns the rail: the shell, unchanged.** The rail stays a CHILD of the
  shell (`#library-shell-grid > LibraryBrowseReaderShell > LibraryRail`) —
  hoisting it to be a sibling was considered and rejected, because the shell's
  three-pane resolver treats the rail as its `library` pane and four other
  routes share that widget. Residency of the SHELL is what makes the rail's 52
  mounts disappear, which is exactly the leverage the record identified.

### 2. What is resident, and what is not

* **Resident (mechanism C, mount-once-then-toggle):** the Media and Notes list
  canvases, both children of `#library-canvas`, toggled by `display`.
  Residency is **stateless** — "resident" means "already a child of the canvas
  host", read off the DOM at each switch. That was not tidiness: the
  ordinary-route projection (`_project_library_canvas_child`) removes every
  child of that host, and a stored widget reference would have been a stale
  pointer plus a `DuplicateIds` mount. Read from the DOM, that same code
  evicts residency correctly for free and the next switch pays one lazy
  re-mount (17 mounts, the record's measured first-visit cost).
* **Not resident: the work pane.** It is 2–3 widgets, every route entry
  already rebuilt it, and swapping it keeps the shell's child list identical
  in shape to the pre-phase-C DOM. Residency is spent where it was measured to
  pay: the 29-widget Media canvas subtree.
* **The one structural delta is the Notes source strip**, and
  `compose_content` and the swap now build it through one shared builder, so
  the strip a targeted switch mounts is the strip a recompose would have
  composed.

### 3. Measured, landed — and the two predictions that were wrong

Acceptance pin, before -> after (same test, same fixture):

| switch | recomposes | mounts | unmounts |
|---|---|---|---|
| notes (switch) | 1 -> **0** | 114 -> **26** | 120 -> **2** |
| media (switch-back) | 1 -> **0** | 179 -> **81** | 175 -> **86** |

**Prediction 1, wrong: "the acceptance pin does not depend on the sync
storm".** It does, and the ceiling had to be re-derived from 25 to 90/95 under
the ceiling-provenance clause above. A real switch cannot reach the spike's
zero-mount floor, because the canvas it switches TO has been off-route since
the last visit — and, under the route-ownership guard this task added,
deliberately un-synced while it was there — so switching back MUST repaint it.
That repaint is `sync_state` -> `refresh(recompose=True)`, a rebuild of the
canvas's own children.

**Attribution, measured (three runs; mount and apply counts identical on all
three).** Every `sync_state` call was instrumented with the canvas's `display`
at call time, and mounts attributed to the interval after it:

| switch | mounts | composition |
|---|---|---|
| media (switch-back) | 77–81 | **21 = the OUTGOING Notes canvas, rebuilt while still displayed**, + ~60 across **two** rebuilds of the destination Media canvas |
| notes (switch), 1st | 26 | **zero canvas rebuilds** — the lazy first mount of the Notes route |
| notes (switch), later | 62 | **24 = a rebuild that runs while the canvas is still HIDDEN** and is superseded before it is ever shown, + 37 across the rebuilds after it |

So a media switch-back carries **three** canvas rebuilds, not two, and one of
them belongs to the route being LEFT. The earlier text in this record said two
and named only the destination; that was wrong in both respects.

**Prediction 2, half wrong: the wall clock.** The record said the landed
number would sit above the mechanism's 30 ms floor. It sits *at the old
number*. Paired probe (`Helper_Scripts/library_switch_teardown_probe.py`),
ONE scratch worktree at a fixed path with one venv, arms selected by
`git switch --detach` between runs so both are measured from the same
checkout location; order-swapped (base-first, mine-first, base-first), n = 3
runs per arm, medians:

| interaction | block before | block after | cpu before | cpu after | mounts | recompose |
|---|---|---|---|---|---|---|
| media (switch-back) | 98 ms | **98 ms** | 216 ms | **150 ms** | 175–179 -> **77–81** | 1 -> **0** |
| notes (switch) | 142 ms | **147 ms** | 165 ms | **133 ms** | 114 -> **26** | 1 -> **0** |

CPU falls 19–31% and mounts fall 55–77%; the longest main-thread BLOCK does
not move (the 142 -> 147 ms on notes is inside this instrument's run-to-run
spread, which was 128–149 ms on the base arm alone).

**Where the remaining cost is NOT yet attributed, stated carefully.** After
residency, CSS restyle is the largest measured CPU bucket on the media arm:
88 ms of 154 ms, 58%. An earlier version of this paragraph went on to claim
that every `Stylesheet.apply` call is triggered by mounting one of the 81
widgets the repaints produce, and that the storm therefore owns 100% of the
remaining freeze. **Both claims are false, and the other arm falsifies them.**
Measured on the same runs, `Stylesheet.apply` calls per switch:

| switch | mounts | canvas rebuilds | apply calls | restyle |
|---|---|---|---|---|
| notes (switch), 1st | 26 | **0** | **459** | 95–99 ms |
| media (switch-back) | 77–81 | 3 | 405–421 | 85–93 ms |
| notes (switch), later | 62 | 3 | 331 | 65–71 ms |

The arm with the FEWEST mounts and NO canvas rebuild runs the MOST apply calls
and the most restyle. Apply count is therefore not mount-proportional, a large
share of the restyle is full-tree work that residency does not remove, and 58%
of measured CPU is not 100% of the block in any case (`block` and `cpu` are
different quantities — see the probe's own docstring).

**So the honest summary of this task is: the structural cause is gone (no
whole-screen recompose, no rail/nav/footer/chrome rebuild, mounts more than
halved), and the freeze is not yet fixed — with the remaining cost only
partly attributed.**

**Leads for the storm task, in the order the measurement ranks them.**

1. **The outgoing-canvas rebuild.** Every media switch-back rebuilds the
   *Notes* canvas (21 mounts) while it is still displayed, immediately before
   it is hidden. Nobody ever sees the result.
2. **The pre-display hidden rebuild.** A resident Notes switch spends 24 of
   its 62 mounts rebuilding the canvas while `display` is still `False` — the
   route STATE flips before the swap shows the canvas, so syncs land in that
   window and are superseded by the rebuilds that follow. This is the cheapest
   lead: the work is provably discarded.
3. **The restyle split, which is unknown.** How much of the 405–459 apply
   calls per switch is mount-proportional and how much is full-tree is not
   measured, and the two leads above only address the first part.

**The storm task must OPEN with an attribution measurement — apply calls by
trigger — not with a fix.** Ranking leads 1 and 2 by mounts and then assuming
the restyle follows is exactly the inference this section had to retract.

### 4. Hazard rulings, as landed

* **Route-ownership guard: landed as specified**, scoped to the two resident
  kinds so every other canvas kind keeps its pre-phase-C behaviour exactly. It
  reads `_library_selected_row_id` through whichever receiver the dispatcher
  was given, per this record's seven-controller derivation.
* **Event gating: landed at the canvas.** Textual's `Button.press()` consults
  the button's own `disabled`/`display` and nothing above it, so the gate has
  to be a handler, not a style. It sits on the canvas that owns the residency
  state rather than on each of the screen's row handlers, and it covers every
  button in the subtree rather than only rows. Its scope stops there:
  `Input.Changed`/`Submitted` and `LibraryMediaRowGeometryChanged` are
  ungated (not reachable today — a hidden widget is out of the focus chain,
  and stale geometry is rejected screen-side by the return-settlement fence).
* **Narrowing the blanket `except`: NOT landed, deliberately.** It was
  implemented and reverted. `test_library_canvas_sync_defects.py::
  test_strict_failure_retry_retains_original_semantic_focus` forces a
  `RuntimeError` out of `canvas.sync_state` and REQUIRES that swallow: the
  dispatcher must recompose and let the reconcile RETRY the sync (`assert
  attempts == 2`) with the original focus capture intact. Task 1 cleared a
  different neighbour of this conflict (`test_tier1_toggle_falls_back_to_
  recompose_on_query_one_failure`, which pins `_apply_library_row_toggle`'s
  except). Narrowing needs its own task that decides what that retry path
  should do with a non-DOM failure; the residency mechanism does not require
  it. What DID land from that round is the traceback — the handler used to
  discard the exception entirely.
* **TASK-31521 composition: confirmed by test**, not only by the spike.
  Suspend and resume leave the resident set and which member of it is showing
  untouched.
