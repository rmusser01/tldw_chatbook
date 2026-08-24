# Task 4 review package — Research Sources workbench

Review base:

`416905730`

Expected commit subject:

`feat: add Research Sources workbench`

The controller-owned dirty TASK-21508 file is excluded from this implementation
and commit.

## Review order

1. `overlay_store.py` and tests: schema-v2 migration, bounds, privacy, qualified
   isolation, and optimistic save preservation.
2. `controller.py`, operation-store seam, app origin precondition, and screen
   intake: explicit captured owner, operation-before-submit, no fallback, late
   repaint fencing, and association-only removal.
3. `sources_region.py` and `source_list.py`: complete parity inventory, typed
   capability enablement, stable 25-slot page, selection/readiness separation,
   folders, reorder, and honest Move/Copy gate.
4. Add modal, receipts, and inspector: exact five-tab vocabulary, catalog
   search/page/select/add, stage receipts/retry, status detail, preview, and
   Device-only annotation.
5. Production CSS and mounted geometry: all six required terminal sizes,
   footer/mode containment, focus-preserving collapse, and ASCII handles.

## High-risk invariants to challenge

- A captured Server intake or existing-catalog attach cannot route to Local or
  the newly visible workspace.
- Every ingest operation exists durably before the app queue is mutated; one
  operation cannot ambiguously own multiple expanded files.
- Local desired IDs remain catalog Media IDs while Server desired IDs remain
  workspace-source association IDs; selected intent never impersonates
  readiness.
- Remove never deletes canonical Library/Media. Readiness retry only refreshes
  or rechecks; Server reorder never mutates before exact-owner preflight.
- Authority/workspace switches immediately clear old source/folder/receipt
  content. Late source/capability/attach results cannot repaint current state.
- Overlay v1 migration invents no organization, stores no paths/URLs/secrets or
  source/Quick-Note bodies, remains bounded, and isolates every qualified key
  axis.
- Every visible enabled control has a mounted event/action test. Unsupported
  controls remain visible, disabled, and carry a reason.
- Source/receipt widgets are composed once, bounded at 25/20, and remain
  reachable at 60x20 without covering the footer or pane mode strip.

## Verification snapshot

- Full focused gate: `359 passed`.
- Explicit inverse matrix: `30 passed`.
- Geometry: `15 passed`, including all six requested active Sources sizes.
- Ruff lint, scoped format, compileall, diff whitespace, ASCII-arrow scan,
  deterministic CSS parity/checker, and Impeccable detector: pass.
- One accepted Requests dependency warning; full pytest intentionally not run.
