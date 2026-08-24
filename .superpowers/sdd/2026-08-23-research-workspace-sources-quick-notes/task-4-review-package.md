# Task 4 review package — Research Sources workbench

Fix-round-2 review base:

`2bcf9c46c70bbd137543bd8f29ec06e8cb4313df`

Expected commit subject:

`fix: close Research source intake ownership gaps`

The controller-owned dirty TASK-21508 file is excluded from this implementation
and commit.

## Fix-round-2 review order

1. `source_urls.py`, add-source modal, and screen boundary: one exact URL gate
   before any Local/Server write, including direct non-modal calls.
2. `library_ingest_jobs.py` and `app.py`: strict durable Research prepare and
   settlement, existing runner dispatch, ordinary caller compatibility, and
   terminal listener idempotence.
3. Screen intake and tests: prepare, persist catalog/job link, dispatch; cancel
   and clean on link failure; fail and retain retry staging on dispatch failure.
4. Sources region and mounted test: displayed stable slots own batch IDs while
   owner desired count stays separate across search/type/status/date/folder
   views.
5. Split regression, inverse, geometry, CSS, privacy/static, and detector
   evidence in the Task 4 report.

### New high-risk invariants

- A hidden selected association cannot enter the exact preview/remove list or
  mutation merely because the owner still desires it.
- URL admission cannot be bypassed through Quick URL, modal batch intake, or a
  direct screen call, and credential-bearing/local paths never reach an
  operation, staging file, registry row, or Server request.
- No Research job dispatch can occur before its qualified operation link is
  durable. A link-write failure leaves a durable cancelled job, no dispatch,
  cleaned managed paste staging, and no canonical catalog mutation.
- Immediate terminal completion observes the already-linked operation once;
  dispatch exceptions cannot overwrite a terminal result or schedule a second
  listener transition.

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
