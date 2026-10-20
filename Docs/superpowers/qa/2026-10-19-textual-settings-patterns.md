# Settings Textual patterns qualification

Scope: TASK-35676, TASK-35677 and TASK-35678. Initial base:
`98c7eb308d46d25f8297fb40b1d23dc377e2c8e0`. Python 3.12 and Textual 8.2.8.
Only targeted tests ran, using the repository's Windows private-profile runner.
No real user configuration was modified.

## Reproduced event-origin defect

A value-changing programmatic paste-control refresh initially passed on a
clean category. With an unrelated `max_parallel_runs` draft edit, each control
failed the same mounted regression: the queued Changed event added an
untouched `collapse_large_pastes` or `paste_collapse_threshold` value to the
draft. The existing dirty key remained, masking the extra staged value.
Both cases failed on the original source, then passed with Textual's
`prevent(...)` around the two assignments, and passed after widget extraction.
This qualifies the concern from the book; it does not establish a defect in
the other temporary synchronization flags elsewhere in Settings.

## Ownership and compatibility census

- The moved region contains only the paste section heading, explanation,
  collapse checkbox, threshold row and help. Existing field IDs, labels and
  token-backed classes remain. The direct-child Static inset now also targets
  the panel's Statics, preserving its original geometry.
- `ComposerPasteSettingsPanel` owns composition, displayed values and local
  Changed handling. Its `Edited` message reaches the screen's existing staging
  helpers. Validation, category drafts and the configuration writer remain
  screen-owned (ADR-033); the panel has no config/app proxy.
- Field search still targets the same Input ID. The screen's sync method calls
  the panel's targeted sync; equal values leave the Input selection untouched.
  The threshold handler remains as a thin direct-call compatibility entry for
  existing tests, without a second decorated event subscription.
- The widget import occurs only when rendering Console Behavior, matching the
  approved screen-decomposition design. TASK-35017's page-order work is not
  implemented or displaced. No other section moved.

## Action and focus matrix

The before/after category rules are unchanged:

| Action | Supported categories | Other categories |
| --- | --- | --- |
| Save | Existing guided categories, plus Network's own writer | Existing guidance |
| Revert | Existing guided categories, plus Advanced Config's raw draft | No changes/guidance path |
| Test | Existing `TESTABLE_SETTINGS_CATEGORIES` | Existing no-test warning |
| Set active / clone / backfill | Library & RAG | Inert |
| Add / edit / delete / export record | Personal Context | Inert |

The pure support helper feeds dispatch, guided buttons, footer and F1 help.
It performs no config or database reads. Category-owned Theme and Personal
Context widgets retain their context-sensitive presentation. Dirty, busy,
validation and confirmation checks remain at execution time. Printable keys
continue typing in text fields; button/direct calls can explicitly bypass that
focus guard. ThemeOptionList retains its own `t` Try action.

One existing footer test expected the screen-level Theme action while leaving
focus in ThemeOptionList. The failure reproduced on the original source.
The test now explicitly releases widget focus before testing the screen binding;
its assertions are unchanged. Affected real-app tests also gained the required
private-profile marker instead of reading the developer profile.

## Behavioral evidence

- Original affected Settings run: 11 passed; the one Theme focus setup failure
  described above was then reproduced on the original source and corrected.
- Final contract run: 8 passed. This includes both programmatic-origin cases,
  compact/wide widget identity/selection tests, compact/wide field-search,
  actual typing and category-return journeys, the category capability matrix,
  and the corrected screen-binding test.
- Existing affected tests cover clean initial state, toggle focus, threshold
  save, staged Save/Revert, active-category help, printable keys, confirmation,
  clean Revert and narrow-footer/F1 discoverability.
- A full private-profile TldwCli served locally was inspected in the browser:
  field search found the paste controls; Enter changed the checkbox; `s` saved
  it. Typing threshold 1500 staged one edit. Revert opened its confirmation;
  Tab then Enter confirmed, restoring 1000 and leaving no unsaved changes.
  The private TOML contained the saved checkbox value and threshold 1000.
  The application log contained zero `unhandled_exception` records.
- Independent review inspected final event ownership, action support, CSS
  geometry and compatibility paths and reported no remaining defects.

## Validation boundaries

New files pass scoped Ruff lint and formatting. SettingsScreen has the same
13 pre-existing Ruff findings as its original source; none were introduced.
Generated stylesheet sync, worker-contract and Backlog guards pass. The new
regressions are appended to the existing PR UI census without moving earlier
entries. Full-suite verification was not requested and was not run.

The broader governance selection finished with 81 passes and 16 failures:
seven size violations in unchanged modules; existing dimension/Python-style/
Library-sheet violations; Windows path-separator and text-decoding assumptions
in guard self-tests; and the inherited boot-CSS byte ceiling breach
(624,809 / 608,090 bytes in this Windows checkout). Eighteen relevant guard
sources/subjects, including all five boot-parsed CSS sources, were compared
with dev `8136af6f6f` and are unchanged by this PR. The added selector is in the
lazy Settings sheet. No existing ceiling or allowance was raised. The screen
pre-import payload guard passed; Settings' own size ratchet passed both checks
and was tightened to the measured 35,805 lines.

Eleven derived-artifact preflight guards passed: generated stylesheets,
profile paths, persistent diagnostics, Backlog IDs/files, schema allowlist,
index pins, worker contract, timestamp writers, markup interpolation and UI
census. Canvas Mermaid reproduction requires the exact pinned Python 3.12.11;
the installed interpreters do not match that version, so this unrelated check
could not qualify locally. It was not bypassed or regenerated.

## Updated-dev integration

Rebased onto `8136af6f6f`, preserving its new Tools & MCP category. Its Test,
Revert and Add action branches remain before generic Settings guards, and its
live mode-specific footer/F1 projection stays instance-owned (ADR-240).
Independent review confirmed these branches are AST-identical to dev and
all 327 pre-existing UI census entries remain in order, followed by six new
entries (floor 333).

The first post-rebase rerun passed six cases before later child processes
refused the test root's newly broadened ACL. Writing a diagnostic helper with
the sandbox file tool had made that disposable parent group-writable. The
failure occurred during config import, before the test body. Restored the
exact owned `.test-tmp` directory to private mode using the application's
native Windows filesystem API, leaving all product guards enabled. Subsequent
diagnostic writes use the native command runner and do not widen that ACL.
The rejected runs are retained; they are not counted as behavioral passes.

Final updated-dev receipts: eight contract cases passed in 238.08 s (new
component/origin/category-return/action cases plus Tools outside-pane dispatch).
The four affected Console Behavior configuration cases passed in their private
children. The subsequent all-category footer child also passed, but its
startup plus 96.75 s child session exceeded the outer 110 s timeout. Reran the
six affected footer cases with a startup-inclusive 180 s timeout: six passed,
eight deselected, 303.56 s. This is 18 distinct selected behavior cases passed
across batches, plus the two Settings size-ratchet checks. No test assertion,
product timing ceiling or security guard was relaxed to obtain these results.

Delivered in [PR #3176](https://github.com/rmusser01/tldw_chatbook/pull/3176)
against dev. TASK-35676/35677/35678 are complete in this branch. The
[follow-up analysis](./2026-10-19-textual-book-followup.md) records the separate
parallel-run diagnostic failure as new work, not as a failure of the fixed
paste-region contract or a reason to widen the completed tasks.


## Current-dev qualification and UI budget follow-up (2026-10-20)

TASK-35677 and TASK-35678 were already implemented, Done and included in
PR #3176. The requested fresh review audited them against their acceptance
criteria on `1bbc145ffc..3c2b0e5601`; no Critical, Important or Minor issue
was found, and Declined to judge was empty. Rebase conflicts affected only
the census and size ratchet. Product patches survived the replay, no new
duplicate Settings methods appeared, and the measured size is 35,757 lines.

On that head, 17 selected Settings behavior cases and both size/no-slack
checks passed. All ten new panel/action/paste/parallel cases passed. Seven
additional footer cases stopped before behavior at
`RecoveryRequired: raw_source_selection_changed`; exactly the same seven
nodes and exception signatures reproduced on unchanged `1bbc145ffc` in a
separate clean worktree. The config, raw-participant and app-factory paths
are unchanged by the PR. Scoped lint/format, CSS reproduction, census and
Backlog checks passed. No full suite ran.

[Run 39447290920, UI shard 9](https://github.com/rmusser01/tldw_chatbook/actions/runs/39447290920/job/118712469761)
then reached its fixed 20-minute job limit while the final added wide paste
journey was running. The last preceding file finished at 15:46:56 UTC and
GitHub canceled the step at 15:47:11; the log contains no assertion failure.
The other nine UI shards, all three fast lanes, Console P0, Dreams and Perf
Guard passed. This is a CI scheduling qualification issue, not a reproduced
failure of the journey (its local wide case passed in 35.47 s).

Following the existing workflow's reorder-first policy under
[ADR-103](../../../backlog/decisions/103-fast-pr-lane-and-required-gate-aggregation.md),
three pairs of census slots were exchanged across shards 5, 6 and 9:

- Model-default rows and the default-model picker: 380.89/314.36 s in this
  run; 360.72/331.14 s in the latest passing upstream run 39420530080.
- Console model-switcher and rename-consistency files: 153.53/123.15 s in
  this run, shifting about 30 s away from shard 9.
- The new wide paste journey and action-availability file: the latter took
  8.49 s on hosted CI; shard 5 completed in 17m45s before the exchange.

Four upstream entries therefore change placement, an explicit exception to
the initial append-only placement. All 372 entries remain exactly once;
only six slots change, and their subject comments move with them. No
assertion, product source, workflow, shard count, timeout or census floor
changes. Seventy-eight census rejection-path and shard-count checks pass,
and the committed census passes its real checker. New hosted qualification
must still confirm the adjusted placement; these timings are evidence for
choosing the swaps, not a claim that the next run has passed.

The stock replay helper assigned a negative model-row cost because the Theme
file appears on both sides of it. These measurements instead use consecutive
per-occurrence progress timestamps. The helper itself is outside this change.
No new ADR is required: this applies the existing CI placement policy.


## Rebase after S1 merged (2026-10-20)

While run 39447290920 was finishing, PR #3179 merged S1 into dev at
`ccf99a59c41419922e8f02db7e3bc93021da632f`. GitHub reported a fresh merge
conflict and no Actions run for `c8d9dd1d9e`. The second rebase preserves
S1's 399 census entries, including its complete 36-node block in the same
positions, then retains this PR's nine regressions: 408 unique entries and
floor 408. Settings measures 35,733 lines; an AST comparison finds no new
duplicate methods against this base.

The 36 added base entries shift the new nine-entry tail to census index
399 (jobs 10, then 1 through 8). Its placement was checked again against
[the latest passing S1 run](https://github.com/rmusser01/tldw_chatbook/actions/runs/39446897270).
That run's jobs 1 and 3 took 19m09s and 19m37s, while jobs 10 and 4 took
15m05s and 12m47s. Two further measured pairs exchange slots:

- Connect key rows (331.30 s, job 1) and Connect paint (215.19 s, job 10).
- Advanced disclosures (444.15 s, job 3) and Connect Tab budget (170.45 s,
  job 4).

Together with the previous adjustment, eight older entries change position;
every other base entry stays put, including S1. All assertions, test
functions, workflow limits and product patches are preserved. The real
census checker passes. The 80 selected census/count and size/no-slack checks
pass; the 17 qualified Settings cases are being rerun against the new base.
Hosted CI must qualify the final ordering before a green-head claim.

The existing [TASK-35834](../../../backlog/tasks/task-35834%20-%20UI-Fast-Lane-job-9-runs-within-a-minute-of-its-20-minute-cap-on-dev.md)
owns broader shard headroom, function-level distribution and three-run
confirmation. This placement-only qualification does not close that task or
claim its three-minute margin. It applies existing ADR-103; no new ADR.
