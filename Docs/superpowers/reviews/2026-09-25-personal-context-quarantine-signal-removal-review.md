# Personal Context quarantine signal removal execution review

Date: 2026-09-25
Task: TASK-25907.11
Status: Complete — approved native fix, targeted verification and bounded review.

Plan: [Approved native plan](../plans/2026-09-25-personal-context-quarantine-signal-removal.md)
Design: [Accepted disclosure contract](../specs/2026-09-25-personal-context-provider-disclosure-controls-design.md#noninterference-and-user-facing-explanation)
ADR: [ADR-187](../../../backlog/decisions/187-personal-context-provider-disclosure-authority.md), [ADR-182](../../../backlog/decisions/182-personal-context-memory-evolution.md)
Tracker: [Memory roadmap](../../../backlog/docs/personal-context-memory-roadmap.md)

## Change and scope

The user approved this bounded runtime fix and retained native execution.
The model block no longer carries the unscoped unsupported-record existence
hint. Whole-record packing depends only on permitted records and the existing
limits; zero selected records yield an empty block. Quarantine maintenance,
the authorized-view compatibility field and internal authority fingerprint stay
with their existing owner. Filters, priorities, expiry, root snapshot ownership,
V1 shared-core schemas/bytes and Sync semantics are unchanged.

The source change is confined to the context serializer, plus Ruff's
behavior-preserving wrapping of its existing priority predicate. Three
regression modules cover actual snapshot/explanation construction, Console
first-request/system-message assembly, and the encrypted h11 baseline.

## Native RED and GREEN receipts

The pre-change context baseline passed 25 tests in 11.78 seconds. Pytest then
reported cleanup warnings about unrelated old shared temporary directories;
subsequent runs use separate fresh --basetemp roots and avoid those warnings.

Before the production change, the new controls produced 4 expected failures,
4 passes and 109 deselections in 240.15 seconds. The failures demonstrated an
unsupported-only block, different payload/token estimates with allowed input,
a Console block, and the actual h11 unscoped policy observation. There was no
fixture, schema, import or harness failure masquerading as the RED result.

After the serializer change, the entire context module passed 30 tests in
11.65 seconds. The affected five-module run passed 172 tests in 702.10 seconds,
including the full encrypted baseline and its independent fresh-root
reproduction. The three existing root/child prompt controls passed in
0.90 seconds: 175 distinct targeted cases in total. The 30 context cases are
included in the 172 and are not added again.

After the reviewer-requested assertion was strengthened, the two empty-profile
cases plus Console consumer passed again (3 passes, 48 deselections,
0.52 seconds). This repeat changed a test assertion only; production code
remained the exact version exercised by the affected run. No full sweep ran.

The tests use the native worktree venv with explicit
PYTHONPATH=.:packages/tldw_profile_core/src. Imported context code resolves
inside the isolated worktree on codex/personal-context-memory-baseline;
the pre-fix starting commit is 9adeba518e5b39ecaeabc4f42b61e167b10b56ad.

## Static checks

Ruff and formatter checks pass for context_service.py, test_context_service.py
and test_memory_baseline.py. All four changed files pass formatting and
changed-line lint; whitespace passes. The older Chat snapshot file retains
exactly two baseline diagnostics outside this diff: C408 at line 211 and I001
at line 703. Diagnostics were compared with the exact starting Git blob,
including code, message and location; no new finding was introduced. This
does not claim whole-file Chat lint is clean.

## Independent read-only review

One reviewer inspected the four-file diff against the approved plan and task.
It found no Critical or Important issue and one Minor test-strength issue:
unsupported-only assertions should explicitly require explanation state empty,
so an unrelated fail-closed empty result cannot satisfy that control. This
assertion was added after the stable running tests finished and verified in the
3-case focused repeat: explanation.state must equal empty.

At review time, the reviewer did not run tests or certify the then-pending
reports/receipts, provider
egress, real-profile behavior, or broader V2 safeguards. No further reviewer
cycle is needed for this test-only strengthening.

## Limits and remaining work

This closes one model-context metadata leak. It does not implement device-only
egress enforcement, destination/purpose enrollment, V2 evidence, suppression,
cross-owner forgetting, source-derived Notes custody, background recovery,
consolidation or repair. No provider call, real profile, UI/app launch, schema
migration, dependency install, background job, push, PR or merge is included.

## Reproduced report observations

The separate [quarantine report](evidence/personal-context-memory/quarantine-signal-v1.json)
has SHA-256 1ef924748fc91a34a3db778ac55adc73d6f3d1ad594af8d93a950fd66be9b363.
Its fixture hash and retrieval summaries equal lexical-v1.json. Returned and
selected labels are unchanged across all 24 cases; only h11's policy gap and
context byte/token observations change (467 → 432 bytes, 138 → 128 tokens).
It retains a successful allowed control. Harness, authority and context errors
are empty. The device-only d10 gap remains and disclosure_checks_passed stays
false. The independent fresh-root test passed, and an additional stdlib-only check
confirmed its second report is byte-identical to the exported artifact.

The three existing native/fenced prompt and spawned-child propagation checks
passed in 0.90 seconds through offline capture doubles. They were added to the
verification scope because the five-module list alone did not explicitly cover
the task's root/child snapshot criterion; no production/source test was added
for that extension.

## Acceptance and closeout

| Criterion | Verified evidence |
| --- | --- |
| 1: No unscoped model signal | Renderer accepts records only; paired real snapshot builds and h11 detector |
| 2: Empty/non-fitting input yields no block | Successful empty state, constrained packing and real Console/message append |
| 3: Hint cannot change permitted outputs | Matched builds compare whole snapshot and explanation, with allowed/private controls |
| 4: Existing filters, budgets and root ownership | Complete context, Console, tools and three root/child prompt tests |
| 5: Native offline evidence with frozen history | Byte-identical 24-case reproduction, preserved fixtures/reports and honest d10 gap |
| 6: Bounded unchanged owners/schema and review | Four-file source/test diff; no shared-core/service/repository change; static checks and independent review |

The report was exported through the existing TLDW_MEMORY_BASELINE_REPORT pytest
hook, preserving repository config/keyring/network isolation; the plan's bare
import/report example was corrected to use that existing path. Separate pytest
temporary roots avoided the old shared cleanup warnings. Final Ruff checks were
rerun without a cache because the default sandbox cannot write the worktree's
cache; all scoped checks passed with the original two Chat diagnostics excluded
and documented. The reviewer finding is resolved and there is no blocking issue.

Existing ADR-187/182 govern the correction; no new ADR is required. Task criteria,
implementation notes, plan and evaluation/tracker documentation are updated via
the Backlog workflow. Original 59 child criteria and all six .11 criterion texts
remain unchanged; independent .10 and roadmap suffix bytes remain preserved.
This branch/worktree is kept locally, with no push, PR or merge.
