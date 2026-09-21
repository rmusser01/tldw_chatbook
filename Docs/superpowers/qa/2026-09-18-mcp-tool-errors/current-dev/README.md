# PR2716 current-dev tool-error qualification

Owner approved the gallery at `19ceab4dc3`. A subsequent
[test/CI-only follow-up](ci-followup/README.md) repairs a splash-dependent watcher
test and accommodates both required test steps within a bounded 30-minute job.
The approved production and native source hashes are unchanged. The original
75-case MCP inventory below remains valid; the follow-up separately verifies
123 admission-sensitive cases (plus one existing expected failure), 26 CI
contracts and all eight artifact checks. Current-head remote CI still gates merge.

TASK-32831 resumes from merged PR2714 dev `e4096e2059`. The production repair
changes `MCP/client.py` and the existing shared `Utils/input_validation.py`:
stdio validates and retains `isError`, and the client returns the
existing error shape with nonblank text details or a generic fallback. Successful
result shapes and connection lifetime are preserved. No UI, CSS, schema,
permission, remote-transport or dependency change. Existing ADR-111/161 apply;
no new ADR.

## Verification

[75 distinct targeted cases](test-inventory.json) pass: sixteen real-stdio execution
cases, three isolated malformed-content cases, 26 fixture boundary cases,
27 native-runner/ownership cases and three transport neighbors. Six execution
regressions [fail on merged dev](red-merged.txt), with success compatibility
already passing. Eleven new fixture boundary cases [fail before repair](fixture-red.txt).
The resulting [execution/fixture](green.txt), [isolated](unit.txt),
[runner](runner.txt) and [neighbor](neighbors.txt) runs pass. No full suite.

Qodo identified non-boolean `isError` values falling through as success. All eight
[wire regressions fail before repair](flags-red.txt) and the
[22 affected cases pass afterward](flags-green.txt). A strict aliased boolean in
the existing shared validation module rejects malformed flags with a fixed error;
validator details and response bodies never reach logs or Audit. Absent/false/true
flags, opaque content and same-session retry retain their existing behavior.
Function-local imports preserve the boot dependency graph.

Both stdio fixtures share the executable path/malformed-request test contract.
The tool fixture validates state and trace below an explicit canonical root before
I/O; strict test-local models and the shared size validator reject malformed wire
requests with bounded errors, then serve the next valid request. This remains
fixture scaffolding. Production validation covers only the existing result flag.

[Eight artifact checks](preflight.txt) pass. [Static comparison](static.json)
retains the client's 123 existing Ruff diagnostics with zero introduced; new and
modified fixture/test/runner files and changed production ranges pass formatting.
[Repeated artifact checks](qodo-preflight.txt) pass after the Qodo repair;
[both production files](qodo-static.json) retain their 123/8 baseline diagnostics
with none introduced.
[Independent reviews](independent-review.json) clear the production/fixture code
and runner. The runner's discovered tmux-path issue was fixed before native launch.

## Real native evidence

The [supported runner](native_check.py) pins this checkout before shared argument
validation/imports; uses a fresh private HOME/config/data profile, real TldwCli,
LinuxDriver and attached TTY; warms the terminal through the app helper; and
records actual loaded module origins plus launch revision, dirty state, PID and
time. Network is blocked and no attempts occur. Fixture state/trace live in the
exclusive evidence directory. An occupied profile ID is rejected before save;
cleanup is limited to the newly owned profile.

Current exports come from fresh `wide-002`/`compact-002` runs after the flag repair.
The launch receipt records parent `8b7be87978` plus dirty state; source hashes bind
the final code actually loaded. Previous `001` exports remain immutable at that
commit. All eight wide PNG renders match the previously inspected frames exactly;
the fresh compact render was separately inspected.

[Wide results](native-result.json) pass in dark/light at 170×48: the server error
shows **Failed** with complete text, the real execution log records error, and
**Approve & run once** successfully retries on the same live subprocess. Four
wire tool calls and alternating failed/successful audit outcomes are recorded.
Each invocation uses the ordinary one-shot Ask approval path.

Eight SVGs preserve feedback and settled states. Their four feedback/settled
pairs render identically; all four distinct frames were inspected. Failure text,
retry action and recovered **OK** are visible. [Inspection](visual-inspection.json),
[render hashes](render-hashes.json). Raw-response expansion, complete permission
flows and full Audit navigation remain outside this bounded slice.

| Theme | Failure | Retry |
| --- | --- | --- |
| Dark | [Failed](textual-dark-170x48-failed.svg) | [Recovered](textual-dark-170x48-recovered.svg) |
| Light | [Failed](textual-light-170x48-failed.svg) | [Recovered](textual-light-170x48-recovered.svg) |

[Wide lifecycle](native-lifecycle.json) verifies exit 0, app/fixture PIDs absent,
released instance lock, ten healthy private databases, zero conversations/messages,
unchanged user defaults and fixture-file sentinels, no logged errors/faulthandler
output, zero network attempts, and exact source/runner hashes.

## Compact limitation retained

A separate fresh [80×24 attempt](compact-attempt.json) fails before execution:
**Test Tool is offscreen even after focus and scroll reveal**. Its
[capture](compact-unreachable.svg) is inspected and explicitly unqualified.
The earlier historical attempt reached the argument-field boundary; this current
attempt fails sooner, at Test Tool. Neither establishes usable compact execution.
[Compact lifecycle](compact-lifecycle.json) confirms the expected exit 1, clean
app/fixture shutdown, unchanged defaults/sentinels, healthy databases, released
lock and matching source hashes. This remains a separate layout review.

[Export manifest](export-manifest.json) binds originals to whitespace-normalized
copies. Historical evidence and the immutable retired runner remain one level up.
TASK-32831 stays In Progress until current-head CI,
accumulated review and merge. Includes the verified PR2714 closeout ledger.
