# Textual reminder bulk and snippet import fixes — 2026-10-21

Implementation qualification for [TASK-36036](../../../backlog/tasks/task-36036%20-%20Keep-admitted-Schedules-reminder-bulk-actions-and-newer-marks-intact.md) and [TASK-36037](../../../backlog/tasks/task-36037%20-%20Preserve-unfinished-Evals-dataset-names-when-snippet-imports-complete.md), on dev `85d140df8777df7eaeb5d0c6d326051cbe965351`. Product commits: `867494bc97` and `6e01bf68de`. Python 3.12.10 / Textual 8.2.8, native Windows.

## Result and boundaries

Schedules admits one reminder bulk mutation synchronously, freezes IDs/owners/intended toggle values and visibly refuses further bulk/mark changes while pending. Settlement releases admission, retains truthful row counts and removes only captured marks. Worker submission failure retains marks and permits retry. Existing SchedulingService local/server, offline and transfer policies remain authoritative; this creates no server atomicity or cross-screen lifetime contract.

Evals imports recompose only the read-only snippet results, using the existing persisted combined-list return value. Heading, Rename row, live Input and Import button survive. Raw invalid/whitespace drafts, exact caret/selection and focus remain uncommitted until explicit Rename. No stylesheet, schema, dependency, root state store or import policy changed.

Existing ADRs [018](../../../backlog/decisions/018-local-server-hybrid-scheduled-tasks.md), [033](../../../backlog/decisions/033-application-session-state-ownership.md), [150](../../../backlog/decisions/150-design-token-system-and-design-language.md), and [161](../../../backlog/decisions/161-component-pattern-library.md) apply. No new ADR is required. ADR-116 remains Proposed context.

## Targeted evidence

| Evidence | Result |
| --- | --- |
| Schedules fresh RED | Six original mounted cases failed; the capture oracle was refined to an explicit missing server-call assertion; both submission-error controls also failed before product edits. |
| Schedules candidate | 14 distinct new nodes have passing receipts: 12 unchanged passing nodes from the initial batch plus 2 corrected notification-drain controls. The initial complete run was 12 passed / 2 failed, not all green. |
| SchedulingService controls | 4 pass: offline update/delete bookkeeping and transfer refusal. |
| Evals fresh RED | TXT/CSV/JSON each lose the exact unfinished draft after successful real import-worker completion: 3 intended failures. |
| Evals final candidate | 6 passed in 108.70s: three formats, failed/cancelled imports, explicit Rename validation/DB/heading/catalog updates, and overflowing table/pinned Import. Initial batch was 5 passed / 1 test timing failure; keyboard edits replaced immediate repeated clicks before the final run. |
| UI token/component governance | 2 passed, including Python style ratchet. |
| CI census/workflow contracts | 121 passed; new required selectors collect 6 cases. |
| Worker/markup inventories | Both pass, no new exceptions. |
| Static analysis | Ruff lint passes for both product files, both new test modules and census checker. Evals and new tests pass formatting; UTF8/compile and whitespace checks pass. |

Native mounted tests use existing private-profile admission and real SQLite. Schedules controls only server await/fault boundaries; Evals drives the selected-file callback and awaits the actual thread worker. Intermediate failures, source provenance and per-node timings are retained in the [structured receipt](2026-10-21-textual-bulk-and-import-fixes-evidence.json). Ignored raw logs/reviews are archived in `.test-tmp/task-36036-36037-evidence/` before execution-workspace cleanup; only named reports/test receipts are archived; profile directories are excluded.

Independent Superpowers task reviews both approve spec compliance and code quality, with no Critical/Important findings. Inherited warning noise remains disclosed. Whole-branch review and current-head hosted CI are tracked with the PR; this local record does not claim a hosted result.

## Required CI admission

Four selectors append to the census: admitted toggle, admitted confirmed deletion, all three Evals format/draft cases, and pending mark/clear refusal. All 430 prior entries and positions remain identical, with 434 total. New indices 430–433 occupy UI shards 2–5; thirteen shards, 20-minute job caps, 180-second per-test limits, dependencies and aggregate protection are unchanged.

For placement only, the earlier current-census hosted run 39565331697 measured those jobs at 799/756/624/813 seconds. Native representative costs are 27.621/30.206/53.658/28.091 seconds. A conservative 2.6x addition projects 871/835/764/887 seconds, below the 1200-second cap. This estimate is not fresh hosted performance qualification; the PR run is authoritative.

## CI follow-up

The initial required run 39660610647 reported the existing Console mapped-learning Stop-visibility timing failure and a stale production-diagnostic inventory. The Console test/fixture/ChatScreen are unchanged from current dev; [TASK-35645](../../../backlog/tasks/task-35645%20-%20Console-screen-refreshes-on-the-UI-loop-wait-out-a-config-rebuild.md) already records the exact Stop-shown flake. The exact failed parameter passed natively (1 pass / 44.85s), and the failed-job rerun passed UI shard 7. These receipts do not claim the underlying existing timing flake is repaired.

The inventory drift is one Schedules owner: 32 calls/digest `db4904d58b553c5b3111` to 33 calls/digest `b1e6c5cad5edc5c931f6`. Statement review confirms two existing ID diagnostics now use the captured `task_id` instead of `task.id`, plus `logger.exception("Could not start reminder bulk action {}", group)` with the two fixed bulk worker-group labels. No user content, secret, path, new destination or sink topology is added by those arguments. The inventory is a recording, not blanket privacy approval for existing exception capture.

Regenerated the pin with `python scripts/check_persistent_diagnostic_inventory.py --write`. The exact production-inventory/sink-topology test then passes (1 pass / 118.99s). No application source, test, UI census, limit or workflow is changed by this follow-up; fresh hosted checks are tracked on the PR. Original logs, statement review and native controls remain archived with the other receipts.

## Verification limits

Eight Schedules and six Evals older UI controls failed in fixture setup with `RecoveryRequired("raw_source_selection_changed")` before their bodies. They are not behavioral passes, and no baseline product swap or admission bypass was used. New private-profile tests cover the changed paths; untouched-method AST receipts and independent contract inspection support the unchanged boundaries. Full-suite, real HTTP/server effects, shutdown/cross-screen durability, native terminal and OS picker journeys remain unqualified.

Existing pytest-asyncio loop-scope and Pydantic deprecation warnings are visible. The original Schedules file and census checker have full-file formatter debt; unrelated formatting and warning suppression are excluded. No new general lesson was identified beyond the existing event-drain, fixture admission and recomposition guidance.
