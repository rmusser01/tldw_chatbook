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

**Per-subsystem series** — state PR (fields + shims), one-controller-per-PR moves,
cleanup PR (shims and dead delegators deleted, tests retargeted, budgets lowered).
Sequenced cold-to-hot so the exemplar never fights rebases and hot subsystems migrate
in short, fast series once the recipe is rehearsed (churn = commits touching the file
in the last 30 days whose subjects name the subsystem):

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
