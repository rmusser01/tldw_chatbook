# PR2716 current-dev tool-error qualification

TASK-32831 resumes from merged PR2714 dev `e4096e2059`. The production repair
changes only `MCP/client.py`: stdio retains `isError`, and the client returns the
existing error shape with nonblank text details or a generic fallback. Successful
result shapes and connection lifetime are preserved. No UI, CSS, schema,
permission, remote-transport or dependency change. Existing ADR-111/161 apply;
no new ADR.

## Verification

[67 distinct targeted cases](test-inventory.json) pass: eight real-stdio execution
cases, three isolated malformed-content cases, 26 fixture boundary cases,
27 native-runner/ownership cases and three transport neighbors. Six execution
regressions [fail on merged dev](red-merged.txt), with success compatibility
already passing. Eleven new fixture boundary cases [fail before repair](fixture-red.txt).
The resulting [execution/fixture](green.txt), [isolated](unit.txt),
[runner](runner.txt) and [neighbor](neighbors.txt) runs pass. No full suite.

Both stdio fixtures share the executable path/malformed-request test contract.
The tool fixture validates state and trace below an explicit canonical root before
I/O; strict test-local models and the shared size validator reject malformed wire
requests with bounded errors, then serve the next valid request. This remains
fixture scaffolding, not a new shared production protocol model.

[Eight artifact checks](preflight.txt) pass. [Static comparison](static.json)
retains the client's 123 existing Ruff diagnostics with zero introduced; new and
modified fixture/test/runner files and changed production ranges pass formatting.
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
TASK-32831 stays In Progress until fresh owner visual approval, current-head CI,
accumulated review and merge. Includes the verified PR2714 closeout ledger.
