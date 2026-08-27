# Library adaptive-reader programme closeout design

**Status:** Approved for implementation planning
**Date:** 2026-08-27
**Task:** TASK-23019
**Decision:** [ADR-086](../../../backlog/decisions/086-library-adaptive-reader-shell.md)

## Purpose

Close the Library adaptive-reader programme with one reproducible, production-shaped body of
evidence that Media, Conversations, Notes, Prompts, and Skills still satisfy their shared shell and
destination-owned contracts after all migrations have landed together. The closeout may repair a
small regression found by that evidence, but it does not add capabilities or reopen the approved
architecture.

The programme already has strong destination-specific tests and evidence. This task does not copy
those suites into a new exhaustive meta-suite. It adds a thin cross-reader orchestrator, fills only
the durable cross-destination gaps, runs a final live matrix through the production hierarchy, and
maps the bounded TASK-23019 contract catalogue to evidence from an exact recorded subject revision.

## Scope

### In scope

- Media, Conversations, Notes, Prompts, and Skills as consumers of the Library-local adaptive
  reader shell.
- Retained Library, Items, and Work identity; independent collapse; requested versus effective
  geometry; list comfort expansion; focus; selection/loading truth; stale settlement; and resize
  purity.
- Shared Library preferences and destination-specific Items preferences across sequential route
  changes.
- Destination-specific modes, drafts, read-only boundaries, recovery, and authority as already
  approved by the programme design and ADR-086.
- Automated production-shaped verification and live verification at 160x50, 120x35, 100x30, and
  80x24.
- Localized fixes to an existing contract when the repair meets the admission policy below.

### Out of scope

- New destination capabilities, interaction models, or visual redesigns.
- Watchlists integration or an application-wide reader primitive.
- New storage, schema, service, draft, trust, conflict, or persistence authority.
- A full repository test sweep.
- Unrelated baseline repairs or refactors.
- A second generic reader model or data-driven destination framework.

## Architecture and ownership

The closeout harness is verification infrastructure, not product architecture. It mounts the real
Library screen and shared shell, loads the exact consolidated application stylesheet stack, and
drives existing destination widgets and handlers. It does not reproduce destination business
logic, call private storage paths directly to simulate success, or create a parallel mutable
reader state.

Durable reusable helpers belong with production-shaped tests only when they express a stable
contract shared by more than one task. The executable live runner and its scenario declarations
remain task-local under `Docs/superpowers/reviews/evidence/task-23019/`, following the existing
TASK-22031 and TASK-22033 evidence pattern. This prevents a one-time evidence driver from becoming
an accidental production or CI API.

ADR required: **no**. TASK-23019 verifies ADR-086 without changing its boundary. A finding that
requires a schema, service-authority, persistence, security, or application-wide structural change
is not a small repair and must move to a separate task with its own ADR assessment.

## Hermetic execution boundary

Containment is established before any application or third-party module that can resolve runtime
paths is imported. Each run receives new scratch-owned configuration, profile, XDG
config/data/cache/state, database, temporary, and raw-evidence roots. The implementation plan
enumerates every supported writable path owner and environment override used by the five readers.
Preflight resolves each one and aborts unless it is inside the run's scratch root. It does not open,
inventory, or hash content in the corresponding real user-owned defaults. Deterministic fixtures
provide records for all five destinations and exercise only existing service seams.

Filesystem authority is phase-scoped and declared before the tripwires activate. During execution,
the harness may read only the clean subject checkout resources required to import the application
and load its consolidated CSS, templates, static fixtures, and declared test data, plus the resolved
Python interpreter, standard-library, and installed-dependency roots required to run that code; it
may write only inside the run's scratch root. It may not mutate the checkout or runtime roots. After
the application host and all tracked owners have settled, the promotion phase may additionally
write only the validated, allowlisted retained artifacts to
`Docs/superpowers/reviews/evidence/task-23019/`. No other checkout or host path is readable or
writable through the harness. The manifest records the read-only checkout and runtime allowlists
plus the single promotion destination so the tripwire does not acquire ad hoc exceptions.

External network, provider, and unauthorized filesystem access fail closed. Pre-import tripwires
record attempts as well as successful effects; a scenario that attempts HTTP, sockets,
model-provider traffic, or a path outside the active phase's declared authority fails rather than
silently using a real account or host resource. Optional capabilities use deterministic contained
substitutes or are reported as explicitly not applicable with a catalogue-level reason; they are
never silently skipped.

Postflight is deliberately scoped to observable harness owners. It closes every connection in the
fixture database registry, settles or invalidates workers owned by the mounted Textual host,
checks tasks/threads registered after the harness baseline, verifies that no prohibited-access
attempt was recorded, and inventories the declared ephemeral roots. The runner does not claim that
no unrelated process-wide database, thread, or task exists. An access outside the active phase's
allowlist, external-call attempt, tracked owner leak, checkout mutation, or undeclared residue
fails the closeout. A containment escape stops the run immediately; ordinary product assertions
continue in burn-down mode so one regression does not hide the remaining matrix.

## Verification model

### Automated cross-reader matrix

The automated layer uses the production Library hierarchy and consolidated CSS. It reuses existing
destination assertions and adds only missing cross-reader contracts:

- exact Library, Items, and Work widget identity across mode changes, scoped refreshes, imports,
  trust/setup flows, recovery, and destructive settlements;
- independent Library and Items collapse, restoration, comfort expansion, requested/effective
  truth, explicit priority, and narrow-width escape behavior;
- shared Library preferences plus destination-specific Items preferences, including a sequential
  route cycle and restart/reload truth;
- F6 region participation, focus evacuation from collapsed/replaced controls, and refusal of stale
  asynchronous focus intent;
- selected, pending, and loaded identity; destination/item/revision/generation fencing; and
  rejection of late success and late failure;
- resize purity, including no database read, list/detail load, worker start, config read, preference
  write, or polling on unchanged/effective-only geometry updates; and
- destination modes and authority boundaries without a parallel reader model.

The manifest owns the complete closeout catalogue below. These stable IDs bound “every contract”
for TASK-23019; the thousands of historical destination assertions remain regression support but
are not individually replayed as live journeys.

| ID | Closeout contract group |
| --- | --- |
| SH-01 | Retained Library, Items, and Work topology and exact widget identity |
| SH-02 | Collapse, restoration, comfort expansion, requested/effective geometry, and containment |
| SH-03 | Shared Library and destination-specific Items preference truth and restoration |
| SH-04 | F6 regions, focus evacuation, footer/action truth, and stale focus-intent refusal |
| SH-05 | Selected/pending/loaded identity and stale success/failure fencing |
| SH-06 | Equality-guarded resize purity with no data, worker, config, persistence, or polling work |
| SH-07 | Sequential route-cycle isolation for preferences, drafts, selection, modes, focus, and workers |
| ME-01 | Media Read/mode/Find/progress continuity and work identity |
| ME-02 | Media bulk preview and confirmed destructive boundary |
| CO-01 | Conversations complete transcript Find and Read/Info continuity |
| CO-02 | Conversations rapid traversal, retry/deletion truth, and Console handoff boundary |
| NO-01 | Notes clean Edit mount and one current draft across Edit/Preview/Info |
| NO-02 | Notes dirty admission, conflict/recovery, and labelled bulk preview |
| PR-01 | Prompts one lossless Basic/Advanced draft and validation-focus ownership |
| PR-02 | Prompts Info/history, bulk/import, and browse/detail recovery |
| SK-01 | Skills Overview/Edit continuity and exact trust revision/fingerprint truth |
| SK-02 | Skills read-only Files, stale review/grant truth, and delete recovery |

Earlier per-destination evidence is linked as lineage and helps choose representative scenarios,
but it cannot by itself mark a TASK-23019 catalogue ID `PASS`. Each ID receives fresh automated
and live evidence against the recorded subject revision.

Two execution shapes are required. Isolated destination cases make ownership failures easy to
attribute. A sequential single-app cycle visits all five readers, mutates only permitted session
state, returns to each destination, and proves that preferences, drafts, selections, modes, focus,
and asynchronous workers neither leak nor reset incorrectly.

Fixed sleeps are prohibited as settlement evidence. Scenarios wait on the production state,
widget, worker, modal, notification, or receipt that defines completion. Every matrix result is
`PASS`, `FAIL`, or justified `NOT_APPLICABLE`; there is no implicit skip state.

### Live terminal matrix

Every destination runs at every approved size:

| Size | Common obligations |
| --- | --- |
| 160x50 | All three regions, bounded Library width, normal Items target, Library collapse and Items comfort expansion, modes and primary actions |
| 120x35 | Responsive Library behavior, readable list titles/details, usable Work mode, both collapse controls |
| 100x30 | Deterministic effective state, compact mode controls, truthful focus/footer, selection and restoration |
| 80x24 | Permanent Work, reachable restore controls, no compositor overflow/intersection, compact navigation |

Each of the 20 cells records containment, pane regions, widget identities, focus owner, selected and
loaded identity, requested and effective preferences, active workers, visible controls, and
compositor text. Text and SVG captures support review, but structured facts are the acceptance
oracle.

Capability-heavy journeys run at representative sizes instead of being repeated mechanically in
all 20 cells:

| Destination | Catalogue IDs | Required live journey |
| --- | --- |
| Media | ME-01, ME-02 | Read/mode continuity, Find or progress, bulk read-only/work truth, and confirmed destructive boundary |
| Conversations | CO-01, CO-02 | Complete transcript Find, rapid A-to-B loading/retry truth, Read/Info, and Open in Console handoff boundary |
| Notes | NO-01, NO-02 | Clean Edit mount, unsaved-draft Preview, dirty navigation admission, conflict/recovery, and labelled bulk preview |
| Prompts | PR-01, PR-02 | One Basic/Advanced draft, hidden-field preservation, validation focus ownership, Info/history, and browse/detail retry |
| Skills | SK-01, SK-02 | Overview/Edit draft continuity, exact trust generation or fingerprint, stale review rejection, read-only Files, and delete recovery |

The contract manifest assigns every catalogue ID to at least one automated result and one live
journey. A catalogue ID cannot be marked verified by a widget merely existing or by an attractive
capture.

## Failure classification and repair admission

Every failure is classified before code changes:

1. **Contract regression** — current behavior violates ADR-086 or an approved destination contract.
2. **Harness defect** — the test does not reproduce the production hierarchy, CSS, data flow,
   authority, or settlement behavior.
3. **Environmental issue** — the host or dependency prevents valid evidence.
4. **Out of scope** — unrelated baseline behavior or a requested new capability.

A contract regression may be repaired in TASK-23019 only when all of the following hold:

- the expected behavior is already represented by a TASK-23019 catalogue ID and is explicit in
  ADR-086, the programme design, a destination task, or its signed-off capability inventory;
- the change is localized and introduces no schema, ADR, service authority, capability, or redesign;
- a focused test fails against the pre-fix final programme state and passes after the repair; and
- the task acceptance criteria are updated before implementation if the repair adds an outcome not
  already represented.

Larger findings receive separate atomic Backlog tasks and do not expand this PR. Harness defects
are fixed as evidence infrastructure, not reported as product regressions. Environmental issues
must be reproduced or explained; they cannot be converted into a passing product result.

After the last admitted repair, all product code, harness code, tests, and scenario declarations are
committed as one clean **subject revision**. The complete automated and live matrices rerun from a
clean checkout of that revision. Any later product, harness, test, or scenario change invalidates
the evidence and requires a new subject revision plus a complete rerun.

## Evidence and reporting

Raw output is written only under the run's ephemeral root. A promotion step validates the manifest,
normalizes host paths and nondeterministic fields, rejects secrets or undeclared artifacts, and
copies only the allowlisted retained files into
`Docs/superpowers/reviews/evidence/task-23019/`. It records hashes for the promoted artifacts, then
deletes the complete raw root. Scratch cleanup therefore does not conflict with retaining the
review bundle.

The bounded retained evidence bundle contains:

- `README.md` with exact commands, environment, subject revision, results, repair history, promotion,
  and cleanup proof;
- a machine-readable summary and contract-to-evidence manifest;
- structured facts for every matrix cell and capability journey; and
- representative text and SVG captures selected for geometry, recovery, focus, and authority
  review.

Temporary databases, profiles, configs, caches, raw logs, secrets, host paths, and redundant frame
captures are not committed. Evidence is normalized so it contains no user-owned absolute path or
credential material.

The retained evidence, task notes, and capability ledger land in a later evidence-only commit. The
manifest records the exact subject commit and tree tested; it does not try to embed the hash of the
commit that contains itself. Required derived-artifact and targeted static checks run again on the
final branch head, whose product, harness, test, and scenario sources must remain identical to the
recorded subject revision.

The existing [Library adaptive-reader capability inventory](../reviews/2026-08-24-library-adaptive-reader-capability-inventory.md)
gains a final programme-closeout entry that links TASK-23019 evidence and distinguishes new
closeout proof from earlier per-destination proof.

## Completion criteria

TASK-23019 is complete only when:

- all five destinations pass the production-shaped automated cross-reader matrix;
- all 20 destination/size live cells pass their common obligations;
- every TASK-23019 catalogue ID maps to fresh automated and live evidence from the recorded subject
  revision;
- the sequential single-app route cycle proves preference, draft, selection, mode, focus, and
  worker isolation;
- every declared writable runtime path resolves inside scratch before import; read-only subject
  checkout and resolved Python-runtime access plus the evidence destination obey their phase
  allowlists; no prohibited access attempt or checkout/runtime mutation is recorded; every
  harness-created database, host-owned worker/task, and ephemeral root closes or is removed; and
  normalized evidence is promoted through the allowlist;
- every admitted repair has a focused red/green regression and the full final matrix passes after
  it;
- required derived-artifact checks, targeted Ruff, compilation, and diff checks pass;
- the evidence README, manifest, and capability ledger identify the exact tested subject revision,
  and the final branch head differs only by retained evidence, task, and ledger documentation; and
- task acceptance criteria, implementation notes, ADR check, and Backlog status satisfy the
  repository Definition of Done.

The full repository suite is intentionally not part of this closeout. The approved evidence is the
production-shaped cross-reader suites plus the four-size live matrix.
