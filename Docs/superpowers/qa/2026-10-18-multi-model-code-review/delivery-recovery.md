# Bounded delivery recovery — 2026-10-28

PR: https://github.com/rmusser01/tldw_chatbook/pull/3286 (draft, targets dev).
This preparation checkpoint preceded the later approved live run. The
[final outcome](bounded-final/report.md) supersedes its pending-approval status:
AC24 remains unchecked, with59 new requests/3 children consumed and no further
live invocation authorized. Earlier receipts accounted for255 requests/18 children.

## Demonstrated integration corrections

The [CI guard report](ci-guard-fix-report.md) records the exact startup-import and
selector failures, their causal fixes and green controls. All three existing
ratchets pass without raising limits: UI-ready is back to dev's 1031 modules.
Ten directly affected owner, Settings, modal and continuation controls passed.
The scoped review then identified a first-use race in the lazy initialization:
two concurrent sends could create different cached and registered review owners.
A controlled constructor-overlap test reproduced that exact mismatch. One
reentrant lock now covers creation and ledger attachment; the race, sequential
first-use and native [2,1] controls passed (three tests). The
[single-finding review closeout](ci-guard-fix-review.md) records the correction
and finds no remaining issue in this scoped change.

A separate CI job passed 1072 tests and failed only two modal inventory tests:
the existing inventory omitted ConsoleCodeReviewModal and its save-dialog edge.
Both failures reproduced locally, then both passed after adding those explicit
declarations and the corresponding 76-to-77 inventory count. This changes the
test inventory, not a performance ceiling or product behavior.

The six changed Python files pass the fatal/undefined-name/duplicate-definition
Ruff guard with the project's Python 3.12 target; whitespace checks pass. The
initial isolated Ruff invocation assumed an older Python and flagged the existing
BaseExceptionGroup builtin; correcting the target resolved that tooling mismatch.
The earlier 99-test integration cohort was not repeated. Old pytest temporary
cleanup warnings and the unchanged zero-headroom census warning remain recorded.

## Qualification preparation (offline evidence only)

The next private runner adopts the supplied result gate: incomplete R1/R2/V1,
missing or duplicated original-finding expansion, or an unavailable/incomplete
unified report raises the existing nonzero qualification failure. A mechanical
pass still requires semantic assessment before AC24 can close.

Five result-gate controls and eight receipt/accounting controls passed. A
socket/DNS-denied normal Console composition passed. A separate three-turn fake
Anthropic control exercised the corrected production request builder, actual
verifier schemas, real tool execution, accepted disposition checkpoint, tool
results and final response. Its simulated usage reached the native accounting
owner (three calls, 360 tokens, zero reserved tokens). This is explicitly fake
transport: zero physical provider requests and zero product children. It does
not prove real Haiku behavior or a complete live review.

Private preparation and failed fixture controls are preserved at
/private/tmp/task35582-bounded-recovery/qualification; the full preparation report
is /private/tmp/task35582-bounded-recovery/preparation-report.md. The original
preparation source was cee6fdcf2695b6bf37c4a8f147a47f66f9ff35fd; the runner must
be rebound to the corrected delivery commit before live execution. The live
preflight must pass on the same candidate and exact human authorization receipt
before one final Console invocation is admitted.

The proposed single allowance is 66 additional requests: at most six for the
preflight (one three-call attempt, at most one manual retry after a narrow causal
fix), plus a separate fixed maximum of 60 for one final Console journey, with
three fresh product children under native [2,1]. Unused preflight allowance cannot
enlarge the final cap. No automatic full-run retry or historical allowance reuse.
That exact scope was later approved, raising the aggregate ceiling to321. The
final report records314 consumed and why unused requests do not permit a replay.

## Remaining acceptance boundary

Real Haiku must complete native tool-call → tool-result → final with usable
usage settling at the owner. The final Console journey must present both
independent reports before a fresh verifier and produce a complete disposition
and mapping for every original finding plus a readable unified report. Existing
scripted fault controls remain separately attributed. A new failure is diagnosed
at its first causal boundary; it does not authorize a broad repair or full rerun.
