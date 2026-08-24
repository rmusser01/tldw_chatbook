# Task 4 review package — Research Sources workbench

Fix-round review base:

`399f91b365`

Expected commit subject:

`fix: harden Research Sources workbench`

The controller-owned dirty TASK-21508 file is excluded from this implementation
and commit.

## Review order

1. `app.py`, screen intake, and app tests: complete durable authority validation
   at admission and delayed dispatch, plus terminal paste cleanup.
2. `paste_staging.py`, `private_paths.py`, and tests: app-owned private files,
   operation binding, terminal cleanup, retry retention, and bounded orphan
   sweep.
3. `controller.py`, `sources_region.py`, and tests: exact-owner reorder bounds,
   explicit page-local semantics, typed Add gate, nested folders, and mounted
   worker containment.
4. Inspector/overlay conflict/modal tests: annotation CRUD/restart, CAS draft
   retention and explicit recovery, Escape/focus restore, and unlink
   confirmation.
5. CSS, all six geometries, and detector output: full-opacity disabled state,
   non-color recovery, stable tree/perf guard, and exact ASCII handles.

## High-risk invariants to challenge

- A captured Server intake cannot route to Local, another profile/principal, or
  the newly visible workspace at either admission or delayed/restarted dispatch.
- Every ingest operation exists durably before the app queue is mutated; one
  operation cannot ambiguously own multiple expanded files.
- Local desired IDs remain catalog Media IDs while Server desired IDs remain
  workspace-source association IDs; selected intent never impersonates
  readiness.
- Remove never deletes canonical Library/Media. Readiness retry only refreshes
  or rechecks; Server reorder never mutates before exact-owner preflight.
- Authority/workspace switches clear old source/folder projections but never
  erase qualified recent receipts. Late source/capability/attach results cannot
  repaint current state.
- Overlay v1 migration invents no organization, stores no paths/URLs/secrets or
  source/Quick-Note bodies, remains bounded, and isolates every qualified key
  axis.
- Every visible enabled control has a mounted event/action test. Unsupported
  controls remain visible, disabled, and carry a reason.
- Source/receipt widgets are composed once, bounded at 25/20, and remain
  reachable at 60x20 without covering the footer or pane mode strip.

## Verification snapshot

- Split focused gates: `638 passed, 1 Windows-only skipped` before the final
  single-selection copy rerun; final affected modal tests add `2 passed`.
- Review-family inverse matrix: `27 passed`.
- Geometry: `15 passed`, including all six requested active Sources sizes.
- Ruff lint, compileall, diff whitespace, ASCII/keybinding/privacy scans,
  deterministic CSS parity/checker, and Impeccable detector: pass.
- One accepted Requests dependency warning in changed gates; full pytest
  intentionally not run.
