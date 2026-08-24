# Task 4 report — Research Sources workbench

## Status

Complete. Research Workspace now has a compose-once, authority-explicit Sources
workbench with durable intake receipts, paged sources, exact desired selection,
readiness/status inspection, device-only folders/annotations, and honest owner
capability gates. Quick Notes bodies and Task 5 ownership were not added.

ADR required: no new ADR

ADR path: `backlog/decisions/078-research-workspace-authority-and-screen-boundaries.md`

Reason: this task implements ADR-078's accepted authority, canonical catalog,
association-only removal, durable qualified intake, and device-overlay rules.

## Implementation

- Replaced the Sources placeholder with 25 stable source slots, 20 stable
  receipt slots, page/search/filter/sort/selection controls, explicit folder
  actions, row actions, and recovery states. Async owner updates patch existing
  widgets so focus and identity survive refresh.
- Added the five-path intake modal with exact Local/Server vocabulary:
  `Import Files`/`Upload`, `Local Library`/`My Media`, URL, Paste, and
  `Search Local`/`Search Server`. URL batch intake creates one durable operation
  per item; directory expansion with one operation is rejected by the app seam.
- Captured qualified workspace refs are persisted before ingest submission.
  `required_origin` makes app ingest fail before queue mutation when the active
  backend differs. Existing-catalog attachment routes through the explicit
  captured ref even after visible navigation and repaints only a still-current
  workspace.
- Added bounded qualified recent-operation lookup and receipts with independent
  Library/Media, workspace-association, and index/readiness stages. Retry emits
  the exact failed stage; readiness uses Refresh/Recheck and never claims a
  server indexing retry.
- Migrated the private overlay from schema v1 to v2 for bounded device-only
  source folders and annotations. V1 migration invents no folders; records are
  isolated across data source, server profile, principal, and workspace axes.
- Source selected intent and readiness are rendered separately. Select all
  reads the exact paged owner, remove unlinks only the workspace association,
  and row/batch controls fail closed from typed capabilities. Move/Copy remains
  visible and disabled because neither current owner exposes that canonical
  action.
- Added a status/preview/annotation inspector with lifecycle, reason, source of
  truth, progress disclosure, retry eligibility, stale state, readiness,
  identifiers, next action, honest missing-preview copy, and Device-only label.

## Parity action matrix

| Surface | Implemented owner action | Honest gate |
| --- | --- | --- |
| Quick URL, Upload/Import, URL batch, Paste | durable operation before app ingest; exact required origin | disabled/recovery when no workspace; submit failure remains a receipt |
| Local Library/My Media and Search Local/Server | bounded explicit-authority catalog search and receipt-first attach | unavailable callback reports the selected authority |
| Attached search, filters, date, sort, pagination | local stable view over a bounded owner page | unavailable owner fields are named and never guessed |
| Select all/visible/clear and row Select | exact owner desired IDs through typed selection capability | unknown/unavailable selection disables controls with reason |
| Preview/details/annotation | owner preview plus normalized status; annotation stays device-only | missing preview and progress are explicit |
| Remove | workspace association only | canonical Library/Media item is never deleted |
| Reorder | exact owner mutation when capability and owner bound allow | Local unsupported, temporary sort disabled, Server preflight remains typed |
| Move/Copy | none | visible disabled control and canonical-owner reason |
| Retry | exact failed catalog/association stage; readiness recheck | no fake server indexing retry |

## Verification evidence

- Fresh full focused Research Workspace/UI/app-origin gate: `359 passed,
  1 warning in 56.15s`.
- Explicit inverse matrix: `30 passed, 1 warning in 25.63s`, covering no Local
  fallback, operation-before-submit, captured-ref attach, late fencing,
  association-only removal, exact Server reorder preflight, stage retry, v1
  migration/qualified isolation, owner clearing, enabled event behavior, ASCII
  handles, and all production geometries.
- Production geometry: all 15 cases pass, including active Sources at 160x40,
  120x30, 100x30, 84x24, 80x24, and 60x20.
- Scoped Ruff lint passes. Eighteen Task 4-owned/heavily changed files are Ruff
  format-clean; the large legacy app/tiny seam files were kept free of unrelated
  whole-file formatting churn.
- Changed production `compileall`, `git diff --check`, and explicit Unicode
  arrow scan pass. Requested collapse handles remain exact ASCII `<---`/`--->`.
- Two consecutive timestamp-normalized CSS bundle hashes match
  (`145da747…`); the bundle-sync checker passes all five outputs.
- Impeccable detector result: `[]`.
- The only warning is the accepted environment `RequestsDependencyWarning`.
  Full pytest was not run, per repository policy.

## Inverse mutation evidence

- Removing the app `required_origin` precondition admitted a Local queue write
  for captured Server intake and made the no-mutation trace red.
- Submitting before operation creation inverted the URL trace and made the
  per-item durable ordering test red.
- Deriving attach from the currently visible workspace dropped an old modal's
  captured intent; the explicit-ref controller test was red until the owner ref
  was routed directly and validated on every identity axis.
- Restoring current-page-only Select all lost owner row 101. Restoring stale
  controller acceptance allowed late Local results to repaint another context.
- Enabling preview/remove/reorder without typed capabilities and retaining old
  folders during an authority switch made the mounted fail-closed guards red.
- Replacing unlink with canonical deletion, advertising indexing retry, or
  moving Server reorder before its exact-owner preflight makes the retained
  Task 2/3 inverse guards red.
- Decoding v1 as v2 without deterministic empty source fields and collapsing
  qualified keys made the overlay migration/isolation guards red.
- Replacing 25 stable source slots with 100 materially increased the mounted
  tree; the final bounded page uses 25 slots. The mounted guard measures 577
  descendants, requires fewer than 650 and mount under 1.0 second, and
  preserves focus without normalizing a slower wait.

All inverse mutations were restored before the final gates.

## Files and boundaries

Changes are limited to Research source controller/overlay/receipt seams, the
Research Workspace Sources UI and CSS, exact app ingest origin precondition,
and focused tests. No canonical content database, server folder/annotation API,
Quick Notes body, cross-authority transfer, source deletion, or keybinding was
introduced.
