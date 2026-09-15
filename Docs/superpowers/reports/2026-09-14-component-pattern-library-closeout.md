# Component-pattern library completion — 2026-09-14

Implementation commit: `ae9093a714`.

The design-system workstream is implemented on `feat/component-pattern-library`
in `.worktrees/component-pattern-library`. TASK-32596 and its twelve subtasks
track the work; TASK-24451 was superseded and closed during the carve-up.

ADR required: no new ADR. Existing decisions:
`backlog/decisions/161-component-pattern-library.md` and
`backlog/decisions/097-boot-budget-ratchets.md`.
The original approved plan is
`Docs/superpowers/plans/2026-09-13-component-pattern-library.md`.

## Result

- A machine-readable registry, documented component catalog, and command-palette
  Pattern Gallery establish canonical component vocabulary, including shared
  sizing/state utilities. The original nine families remain covered.
- The agentic stylesheet was carved into owning component/layout/feature sheets.
  Governance enforces the 2,000 active-line ceiling and unique canonical homes.
- Source-sheet numeric dimensions and non-token hex colors have zero allowances.
  The dimension checker examines every shorthand slot, including signed/decimal
  values. Fixed Python visual values and finite states use token-backed classes.
- The Python guard inventories direct/tuple/annotated/augmented assignments,
  `set_styles` and `setattr` forms, including opaque CSS input. Measured geometry,
  caller-supplied profile dimensions and user-entered theme preview colors remain
  visible, specifically annotated runtime data; `None` resets are explicit.
- Console frame states remove obsolete edges; retained widgets clear old inline
  sizes or mutually exclusive classes when sizing ownership changes. Library
  emergency widths, navigation handles, compact layouts and theme/status changes
  have mounted regression coverage.

The migration's property scope is the approved plan's `background`, `color`,
`border*`, `width`, `height`, `padding*`, `margin*`, and `opacity*` set. It does not
claim to migrate min/max dimensions or display/layout properties. The AST guard
is a conservative syntax inventory, not whole-program dataflow analysis.

## Recovery and deviations from the original recipes

The recovered ZCode work had completed/reviewed plan steps 1–11, but step 12's
empty regex baseline hid 567 new `set_styles` calls containing 579 visual writes.
The continuation replaced fixed values with real classes and strengthened the
inventory before claiming zero. No allowance was raised to hide remaining work.
Review also caught five mixed-token spacing declarations missed by the original
first-value regex; these are now tokenized and covered by negative tests.
A final manifest audit found 43 more declarations in active
`components/stats_screen.css`, skipped by the old extension filter. Governance
now includes every active CSS_MODULES entry alongside the authoring TCSS files.
Its duplicated section-header defaults moved to the canonical owner, preserving
computed padding, margins and border behavior with a mounted regression.

Utility overrides use `!important` to preserve previous inline precedence against
ID rules. Explicit inline resets remain necessary: Textual inline styles beat
important classes. Lightweight test hosts now load the production app bundle
where they exercise these utility classes. No global test-host shortcut was added.

Tokenization had widened an existing startup CSS budget breach. Instead of
raising the ceiling, generated module payloads omit authoring comments while the
editable source retains them and generated MODULE markers retain provenance.
An initial space-replacement implementation changed compound selectors; review
caught it and regressions now preserve original whitespace boundaries, quoted
strings and parsed selector structure across all source modules.

## Verification

All runs were targeted; no full repository sweep was requested or run.

- Final combined suite after the manifest correction: **176 passed**. Covers component/token governance,
  AST inventory regressions, comment/selector integrity, generated CSS integrity,
  utility precedence, dark/light gallery snapshots, Console frames, Library
  retained-state transitions, and theme/ingest/audio visual states plus bundle sync and boot bytes.
- Final bundle-sync and tightened boot-budget checks: **6 passed**.
- Independent shared/Library reviewer: **80 passed**, both review findings fixed,
  no remaining actionable issue. The late Statistics correction also passed
  independent inventory/ownership checks and paired legacy/current computed
  geometry probes. Independent Console reviewer: **5 mounted
  retained-state probes passed**, including blocked/ready 160→80→160→80 Settings.
- Library worker: **74 primary** and **10 secondary** checks passed, followed by
  **6** transition/rail checks after fixing inherited navigation-handle classes.
- Console worker: **41** focused checks passed plus **6** Settings matrix cases.
- Additional Home/Personas/theme/rail/trace/snapshot checks are recorded in the
  local work ledger; the two pre-existing failures below were isolated explicitly.
- Fatal Python lint (`E9,F63,F7,F82`) passed across **118 changed/new files**.
  All seven new Python files pass full Ruff and formatting checks. Final
  The uncommitted working diff passed `git diff --check` at closeout; that
  command did not compare the complete branch against its base. The later
  component audit found eight historical trailing-whitespace lines in the
  full branch range, tracked separately by TASK-32595. Existing-file lint debt
  was not broadly reformatted.
- Actual terminal app launched at 160×50 with a scratch config/data profile.
  Command-palette navigation opened the gallery; top forms and bottom utility
  samples rendered in both textual-dark and textual-light. ANSI captures confirm
  dark/light backgrounds. This verifies startup/navigation/rendering, not a
  provider conversation or a claim of pixel parity across every destination.

Dark/light snapshots are checked in at
`Tests/UI/snapshots/pattern_gallery/{dark,light}.svg`.
Local raw logs, review reports, scratch profile and ANSI captures are retained in
`.superpowers/sdd/2026-09-13-component-pattern-library/` (ignored scratch evidence).

## Startup byte accounting

| Census | Bytes |
|---|---:|
| Resumed tree before paydown | 841,903 |
| Final generated boot sources | 609,446 |
| Reduction | 232,457 |
| Previous limit | 768,000 |
| Tightened limit | 634,050 |

The final snapshot was written by
`scripts/update_boot_budget_snapshots.py --only css`. The limit was banked at the initial 609,050-byte census plus ADR-097's
25,000-byte slack; the final manifest correction uses 396 bytes of that headroom.
The limit remains unchanged, leaving 24,604 bytes. The 600,000-byte anti-vacuity
floor remains unchanged. This is byte accounting, not a measured speedup.
TASK-31500's separate modal-default deferrals remain outside this workstream.

## Pre-existing failures and limits

Two failures encountered in adjacent targeted files predate workstream base
`06cc148a91`; neither was silently counted as passing:

- Trace live insertion: the test requests `event:3`, while table keys already use
  `record:event:3`. Six relevant methods and the test are AST-identical to base.
  Projecting the exact base methods reproduces the missing-key failure before
  refresh; changing only the requested test key in memory passes all retained
  cursor/filter/viewport assertions.
- Personas unchanged-size optimization: the exact base method queries
  `#personas-conversation-actions` before its unchanged-state guard, contrary to
  the unchanged test's no-query expectation. Projecting that method reproduces
  the failure before any migrated visual write.

These were bounded base-method projections with current styles, not a claimed
full base-checkout test run. Their string-key/control-flow causes are independent
of CSS. Detailed evidence is in the local
`task-12-preexisting-failures-check.md`. Earlier-stage reports also recorded
pre-existing broad-test debt; this closeout does not claim the entire suite green.

No schema, dependency, provider protocol or data-ownership changes were introduced.
The branch and worktree are preserved; integration is separate from completion.
