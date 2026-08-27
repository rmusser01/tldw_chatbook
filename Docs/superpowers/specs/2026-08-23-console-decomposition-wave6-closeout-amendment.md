# Console Decomposition Wave 6 Closeout Amendment

**Status:** approved by the owner 2026-08-23 after the final Wave 6 RED check

**Amends:** `2026-08-13-console-decomposition-wave6-design.md`, specifically its
final closeout projection and atomic-delivery sequence. Completed TASK-3070.1 through
TASK-3070.10 ownership decisions remain unchanged.

## Problem

The original design required TASK-3070.11 to rebase the completed extraction series,
measure `ChatScreen`, and lower the ratchets. That measurement correctly stopped the
closeout instead:

| Evidence | Lines | Direct methods |
|---|---:|---:|
| Immutable post-image Wave 6 base (`8d806b71d`) | 22,172 | 712 |
| Completed TASK-3070.3-.10 task deltas | -4,958 | -130 |
| Final Wave 6 delivery base (`87791f855`) | 19,863 | 630 |
| Current amendment base (`d20dd733b`) | 19,884 | 632 |
| Concurrent Console growth at the amendment base | +2,670 | +50 |
| Immutable ratchet ceilings | 17,727 | 593 |
| Remaining deficit | 2,157 | 39 |

Wave 6 delivered more task-local line reduction than its 4,521-line projection and
missed its 131-method projection by one method, but unrelated work landed faster than
the serial extraction series removed it. Raising or resetting the ratchet remains
forbidden. Treating 19,863 / 630 as the new ceiling would erase the exact signal the
ratchet preserved.

## Decision

Replace the original docs-only TASK-3070.11 closeout with four atomic children:

1. **TASK-3070.11 — characterize and amend.** Freeze the delivery evidence, exact
   source inventory, conservative residue, and binding task sequence. It changes no
   production behavior.
2. **TASK-3070.12 — realtime orchestration.** Classify the coherent 57-method,
   1,997-line realtime lifecycle/presentation family and move its 56 policy methods
   behind one explicit owner while retaining only the reviewed screen-owned repaint
   presentation stay. No realtime screen delegate is justified by a framework binding.
3. **TASK-3070.13 — review and selection workflows.** Classify the coherent
   26-method, 1,114-line changed-files, change-review, annotation, feedback, note,
   quote, and trajectory family and move its 15 policy methods behind one explicit
   owner while retaining four framework-bound delegates and seven reviewed
   screen-owned stays.
   Provider-selection policy is not part of this family.
4. **TASK-3070.14 — final closeout.** Rebase, measure, lower both ceilings to the
   exact earned counts, run the approved focused and required CI gates, update the
   records, and close TASK-3070.

Each extraction starts from the latest `dev` only after its predecessor merges. A
later rebase that invalidates either exact family or the conservative projection stops
that child for amendment; it never rewrites the evidence or raises a budget.

### TASK-3070.13 current-base amendment (2026-08-27)

TASK-3070.12 and an independent per-turn changed-files simplification merged before
TASK-3070.13 began. On the resulting `dev` base `ee8dc24115`, ten methods from the
historical 26-method review/selection inventory no longer exist. Reintroducing those
deleted paths would be a regression, while pretending the frozen inventory still
matched the implementation would violate the stop-and-amend rule above.

The approved task-specific design at
`2026-08-27-task-3070-13-console-review-selection-controller-design.md` therefore
supersedes only TASK-3070.13's current implementation boundary. The immutable
amendment evidence remains historical source-of-truth. The surviving family is 16
methods and 840 physical lines: seven moves, three framework delegates, and six
screen stays. With 426 stay lines and a maximum 15 delegate lines, TASK-3070.13 must
remove at least 399 lines and seven direct methods. This revised extraction still
reduces both current screen counts and never raises either Wave 6 ratchet.

## Ownership Boundary

Both new owners follow `DESIGN.md` section 7:

- The screen and region widgets own Textual composition, DOM queries, focus, modals,
  and framework decorators.
- Controllers own state transitions, sequencing, cancellation, persistence policy,
  and orchestration with no DOM or sibling-controller reference.
- Late-bound dependencies are named callables installed through
  `UI/Console_Modules/wiring.py`; sibling traffic uses named callables rather than
  controller objects.
- A framework-required screen method may remain only as a real, complete delegate of
  at most five physical source lines. It contains no duplicated policy.
- Existing private compatibility attributes may use fail-loud descriptors during the
  move; no new permanent mirrored state is introduced.

Realtime transport/session/FSM/audio owners remain where they are. TASK-3070.12 owns
only the Console coordination currently embedded in `ChatScreen`; it must preserve
session/tap/sink identity, first-words buffering, transcript publication, usage,
fallback, reconnect, barge-in, remount, teardown, and privacy behavior.

TASK-3070.13 does not acquire Git or database authority. Existing services retain
changed-file reads, review application, note persistence, annotations, and run data.
The new controller owns only the methods classified as moves in the source-inspected
inventory. Modal/DOM presentation stays on the screen. In particular,
`on_console_review_notes_requested` and `_console_review_notes_flow` remain
screen-owned under ADR-068: the off-thread fetch, never-raises wrappers, and forced
preview reload are not transferred.

## Conservative Projection

The characterization test preserves final-delivery evidence at `87791f855` and locks
the current candidate source spans at amendment base `d20dd733b`. The implementation
may earn more reduction, but it may not assume more than this residue budget:

| Family | Candidate lines | Move / delegate / stay methods | Full stay lines | Delegate residue (5 lines each) | Maximum screen residue | Minimum net lines | Removed methods |
|---|---:|---:|---:|---:|---:|---:|---:|
| Realtime orchestration | 1,997 | 56 / 0 / 1 | 19 | 0 | 19 | 1,978 | 56 |
| Review/selection | 1,114 | 15 / 4 / 7 | 438 | 20 | 458 | 656 | 15 |
| **Total** | **3,111** | **71 / 4 / 8** | **457** | **20** | **477** | **2,634** | **71** |

Every stay is an exact named screen/region-owned method, not an unbudgeted exception;
every delegate has a five-line complete-definition ceiling. The minimum projection
clears the 2,157-line deficit by 477 lines and the 39-method deficit by 32 methods.
TASK-3070.14 still measures the actual final tree and lowers the
ratchet to that exact result; projections never become budgets.

## Verification

TASK-3070.11 adds source-inspected architecture evidence for the immutable revisions,
arithmetic, exact move/delegate/stay membership, source spans, and conservative
projection.
TASK-3070.12 and TASK-3070.13 each require RED-first ownership/dependency/delegate
contracts, isolated controller tests with plain fakes, focused mounted product suites,
mutation checks, targeted Ruff/format, isolated compile, diagnostic inventory,
privacy, and diff gates.

The owner explicitly prohibits a local full-suite run. Required GitHub Actions remain
the broad integration gate. TASK-3070.14 runs only related local tests plus the
architecture/static/privacy/diagnostic gates and waits for required CI before merge.

## ADR Check

ADR required: no.

ADR path: `backlog/decisions/068-console-text-selection-and-annotations.md`.

Reason: this amendment directly applies the already accepted controller/region rules
in `DESIGN.md` section 7 and preserves ADR-068's explicit screen ownership for review
note fetching, never-raises wrappers, and forced preview reload. It changes neither
product behavior nor storage, security, provider, process, or application-session
ownership, so a new ADR would duplicate existing decisions.
