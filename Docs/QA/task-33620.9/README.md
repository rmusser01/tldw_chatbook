# Explicit Console rename publication — 2026-10-04

Frozen base: `7d155170dc95557736a239a1ce7427981f4d50ec` (merged PR #3009).
Task-owned branch: `codex/switcher-workstream-burndown`. These are isolated
real-SQLite and Textual compositor tests, not native-terminal evidence.

## Contract and scope

Saved tab/F2/palette and rail/tree actions share the incumbent workspace rename
worker. It reserves every exact open alias before its optimistic durable write,
then publishes to those retained instances. New hydration/rebinding into the
transition is refused; rebinding away and same-binding preparation cleanup stay
safe. Cancellation drains the already-running SQLite write before publishing
its committed title into the captured store. Profile/lifecycle changes prohibit
painting or confirmation in the new or retired view.

Success awaits tab/header publication and both mounted sidebar projections,
including filtered named-workspace rows. The receipt checks completed compose
state and actual DOM signatures, and retires when its tray detaches. An inactive
transcript gets the renamed header when selected; renaming never activates it.
Unbound/automatic titles retain their incumbent behavior. Library currently has
no title-edit control; this patch does not create one. ADR-085 owns the narrow
amendment; no schema, dependency, event bus or GC policy changed.

## Evidence

- Actual mounted rail-menu active/inactive and bound tab-modal tests on the
  frozen pre-fix source: **4 failed**, all at the saved/live title divergence.
  The saved write succeeded; the runtime stayed old. One old-path unawaited
  refresh warning also occurred. No setup failure is counted as this RED.
- Targeted store, rename, persistence and shared recomposition guard batch:
  **46 passed in 235.70s**, **one aggregate descriptor warning** (+297).
  Fourteen rename cases cover the actual menu, selected/unselected tab, profile
  departure during a held real write, cancellation, changed-profile modal,
  durable refusal/exception, filtered tree titles and retired publication owner.
- Final added inactive-header checks plus existing Ctrl+K rename dispatch:
  **3 passed in 35.57s**, no warnings.
- Earlier bounded core: **31 passed in 92.03s**, no warnings; shared guards
  separately: **14 passed in 44.53s**, no warnings. The combined warning is not
  hidden by those smaller runs. Eight legacy store rename cases and two named
  preparation cases also passed without warnings.
- Wider controller comparison: feature batch **121 passed, 2 failed,
  33 warnings**. Frozen-base controller batch **90 passed, the same 2 failed,
  28 warnings**. Both isolated base cases passed individually: the failures
  depend on batch lifetime/configuration. They and the timer/descriptor warnings
  remain for qualification/resource remediation, not a clean-suite claim.
- Changed tests pass Ruff. Six baseline-clean Python paths format clean;
  [immutable formatter baseline](formatter-baseline.json) verifies that the
  other three source paths add no formatting debt. Normalized source Ruff
  comparison: **431 base / 431 feature, zero additions or removals**.
- All eleven derived-artifact checks pass. Diagnostic-inventory change is one
  owner digest for reindented existing logging: **53 calls before and after**,
  no added/removed messages or sinks. Whitespace clean.
- The rename UI module is added to the PR fast-lane census. No full sweep ran.

Reproduce the affected batch with the repository Python >=3.12 environment:

```sh
python -m pytest Tests/Chat/test_console_title_publication.py \
  Tests/UI/test_console_rename_consistency.py \
  Tests/UI/test_console_conversation_persistence.py \
  Tests/UI/test_console_tray_read_aware_recompose.py \
  Tests/UI/test_console_workspace_tray_recompose_guard.py \
  -p no:cacheprovider --tb=short
python scripts/terminal_qualification/format_ratchet.py verify \
  --baseline Docs/QA/task-33620.9/formatter-baseline.json
PYTHON=/path/to/python3.12 bash scripts/preflight.sh
```

## Limits and unsuccessful attempts

No native control, Windows host, first-time participant or full latency matrix
was exercised here. Existing character-navigation acceptance remains open.
Independent source review's admission, cancellation, rebinding and receipt
findings were corrected and re-reviewed with no remaining production blocker.

An early success assertion included the glyph-only menu button; it was not a
title assertion and was corrected. A later inactive-menu assertion assumed
runtime metadata survives a saved-row rebuild; it now admits only the exact
runtime or exact persisted conversation, never a matching title. Genuine stale
and empty-DOM failures remain in the evidence history. A rail profile test
initially pushed its modal through the unmounted owner rather than the mounted
harness, and an early tree test queried flat buttons instead of tree nodes.
Those fixture errors are not product RED evidence. A too-broad legacy cancel
selection hung; only its verified task-owned pytest PID was terminated, and no
result count is claimed. No thresholds or warning filters were raised.

Raw local logs remain in `/tmp/rename-real-menu-red.log`,
`/tmp/rename-core-verified.log`, `/tmp/rename-inactive-final.log`,
`/tmp/rename-base-controller-batch.log`, `/tmp/rename-integration.log` and
`/tmp/rename-preflight-verified.log`. These temporary paths are not portable
evidence; the source-bound commands and bounded outcomes above are the durable
receipt. Frozen baseline diagnostic copy is `/tmp/rename-frozen-base-PbBiHI`.
