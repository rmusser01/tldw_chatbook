---
id: TASK-22222
title: Boot guards for what the import guards cannot see
status: Done
assignee: []
created_date: '2026-08-24'
updated_date: '2026-08-26 04:40'
labels:
  - testing
  - startup
  - guard-efficacy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22222).

This review measured a real user-visible boot regression (+~11% to `_ui_ready`) with every
import guard green, because all of them assert on `import tldw_chatbook.app`:
- no census at `_ui_ready` (the mount leg grew invisibly; PIL present at ready);
- no budget on boot-parsed CSS bytes (770,285 -> 813,605 B since the pin; the TASK-21115
  ratchet design FORCES new widget CSS into the eagerly-parsed bundle with no size budget);
- no census of boot-time worker threads (4 -> 7 unnoticed);
- construct-time runtime imports are invisible (`app.py:7273-7274` re-imports
  `Persona_Visual.*` at construct — harmless today, boundary crossed silently);
- `Tests/App/test_boot_no_feature_db_files.py` is a fixed six-filename list (a seventh
  store, or non-DB side effects like 22216's staging sweep, pass silently);
- no wall-clock or structural TTI regression tripwire at all.
Also: TieAwareStylesheet (`app.py:5893`, new since pin) arms full ~814 KB reparses during
first mount when tie-breakers lower — instrument the count so the cost is visible.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A `sys.modules` census at `_ui_ready` is pinned (allowlist + budget), so mount-leg growth and construct-time imports land in review
- [x] #2 Boot-parsed CSS bytes carry a budget with a stated raise procedure
- [x] #3 Boot worker census pinned (see TASK-22215)
- [x] #4 The boot-files guard's fixed-list blind spot is documented in the test and extended where cheap
- [x] #5 TieAwareStylesheet reparse count during a cold boot is measured once and recorded (instrumentation may be temporary); each new guard documents its own blind spots
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Verify AC1 already shipped (TASK-22213's Tests/Performance/test_ui_ready_module_census.py) and reference it
2. CSS byte budget guard: pin total boot-parsed CSS bytes (3 CSS_PATH files + 2 widget_defaults sheets) with raise procedure + the 21115 ratchet design-tension note; mutation-test both directions
3. Boot worker/thread census: subprocess run_test probe recording every WorkerManager worker + Thread start up to _ui_ready+eps, pinned as an allowlist (the list 22215's stagger will reorder); mutation-test
4. Construct-time runtime-import allowlist: sys.modules diff around TldwCli() construction in a subprocess, pinned allowlist + budget; mutation-test
5. Extend test_boot_no_feature_db_files.py: honest blind-spot docstring + glob *.db census allowlist at construct
6. TieAwareStylesheet: permanent cheap reparse-arm counter + measure once during a real boot, record the number
7. Run all new guards twice for stability, targeted tests + --collect-only sweep, preflight, commit, push
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Four guards and one permanent instrument, closing the boot blind spots the
import guards structurally cannot see. **No production behavior changed** —
the only non-test edit is a counter increment in `TieAwareStylesheet`.

**AC 1 was already shipped by TASK-22213**, not by this task:
`Tests/Performance/test_ui_ready_module_census.py` boots headless to
`_ui_ready` and pins this repo's resident modules (budget 970, measured
938-939 warm) with its blind spots and raise procedure documented. Verified
green here as part of the targeted run; ticked by reference, no duplicate
census written.

**AC 2 — `Tests/Performance/test_boot_css_byte_budget.py` (new).** Censuses
every CSS source parsed on the boot path — the three `TldwCli.CSS_PATH`
members read from the class itself (so a new entry counts automatically)
plus the two `widget_defaults` sheets `_get_default_css` registers. Measured
**833,841 B** (modular bundle 640,599 + widget_defaults self 89,127 +
scoped 89,082 + screen_css scoped 12,615 + self 2,418); budget **860,000**.
Note the review's 813,605 figure has already grown another ~20 KB. The
docstring states the TASK-15450/21115 design tension explicitly: the ratchet
*forces* new widget CSS into these eagerly-parsed sheets to stay under
Textual's 64-source parse-cache cliff, so the trade is bytes-for-cliff-
safety — the ratchet counts SOURCES, this counts BYTES, and between them the
trade is priced in both currencies. Growth is normal; silent growth is not.
An anti-vacuity floor (700,000 B) fails a hollow census rather than passing.

**AC 3 — `Tests/Performance/test_boot_worker_census.py` (new).** Instruments
`WorkerManager._new_worker` and `threading.Thread.start` *before* importing
the app, boots headless to `_ui_ready` + 1.0 s, and pins **16 (name, group)
worker pairs** (+2 fresh-boot-only rail-preference writers) and **4 thread
families** as allowlists. Records STARTS, not a snapshot, so a worker that
finishes inside the window is still censused. Thread names are normalized
(`asyncio_3` -> `asyncio_#`) and thread COUNTS are deliberately not budgeted,
because pool sizes follow `cpu_count()`. This is the list TASK-22215's
stagger work should reorder; staggering is explicitly out of scope here.

**AC 4 — `Tests/App/test_construct_runtime_imports.py` (new).** Diffs this
repo's `sys.modules` around `TldwCli()` construction in a subprocess and
pins the **13 modules** a warm construct newly imports (the Persona_Visual
wiring the review named, Persona_Buddy preferences, two Scheduling migration
modules, Subscriptions startup reconcile, the Video_Generation store
family). Warm, not fresh: a fresh-profile construct additionally pulls the
whole Chunking engine (60 modules total) through the media-DB v6->v7
migration, which is legitimate one-time work — so the probe constructs once
to build the profile and measures the second construct, with a separate
assert that Chunking stays OFF the warm path.

**AC 4 (second half) — `Tests/App/test_boot_no_feature_db_files.py`
(extended).** The fixed six-name list's blind spot is now stated in the
docstring, and the cheap generalization landed: a second test globs every
`*.db` in the scratch profile after construction (sidecars stripped) against
an 8-row allowlist of the stores construction is known to open. Mutation
proof of the blind spot being real: a synthetic seventh store made the new
census RED while the original six-name test stayed GREEN in the same run.

**AC 5 — TieAwareStylesheet reparse count, measured.** Added
`tie_breaker_lowering_rearm_count`, a permanent cheap counter (class-level
default, incremented only inside the lowering branch), plus
`Tests/UI/test_tie_aware_stylesheet_counter.py` keeping it honest in both
directions. Measured on a real headless boot to `_ui_ready`: **14 arms,
which coalesced into 8 actual extra full reparses** — 20 total
`Stylesheet.parse()` calls with arming vs **12 with the arming neutered**,
A/B on the same profile. Identical on fresh and warm boots; all 14 land
during first mount, none in the following second. So the docstring's
125-380 ms price is per-reparse and cold; these are warm (parse-cached), but
8 full passes over ~834 KB inside the first-paint window is a real cost that
was previously unmeasured. Recorded in the counter's docstring so the
number travels with the mechanism.

**Mutation results (every guard proven RED both ways):** CSS budget —
+4,000 lines of padding -> RED, hollow bundle -> RED on the anti-vacuity
floor. Worker census — a synthetic eighth `run_worker` -> RED naming it, a
raw `threading.Thread` -> RED naming it, dropped records -> RED on the
degeneracy assert. Construct imports — an unlisted function-level import in
`__init__` -> RED naming the module, and the warm-construct leak assert
proven live by pointing it at an allowlisted-but-present family. DB census —
seventh store -> RED (old test green). TieAware counter — increment removed
-> RED, increment moved outside the lowering branch -> both tests RED.

Trade-off accepted: the worker and DB allowlists are membership tests
(`observed <= allowed`), not equality. Removing a boot worker or deferring a
store never fails — a deliberate choice, since every removal here is a win
and the guard exists to catch unreviewed ARRIVALS.

Files: added `Tests/Performance/test_boot_css_byte_budget.py`,
`Tests/Performance/test_boot_worker_census.py`,
`Tests/App/test_construct_runtime_imports.py`,
`Tests/UI/test_tie_aware_stylesheet_counter.py`; modified
`Tests/App/test_boot_no_feature_db_files.py`,
`tldw_chatbook/css/tie_aware_stylesheet.py`.

Verification: new guards 7 passed, twice (12.06 s / 11.70 s). Targeted
branch-relevant run (CSS consolidation, consolidated-CSS harness, 22213's
`_ui_ready` census, 22216's construct side-effects, import weight): 44
passed, 3 skipped (torch/numpy absent). `--collect-only`: 59,602 collected;
28 pre-existing collection errors, all in optional-dep trees
(numpy/playwright, Audio/TTS cross-module imports) and none in the trees
this branch touches — `Tests/App Tests/Performance Tests/UI` collect clean
at 16,399. `./scripts/preflight.sh`: all derived-artifact checks passed.

Note for TASK-22212's owner: the review recorded
`test_class_level_css_stays_within_the_allowlist` failing on pristine tip;
it passes on this base (dev `f0e896122`), so that red appears already fixed.
<!-- SECTION:NOTES:END -->
