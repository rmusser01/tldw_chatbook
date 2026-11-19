# TASK-36728 / TASK-36729 verification

Targeted local verification only (AGENTS.md); Python 3.12.10 / Textual 8.2.8, actual production owners mounted through isolated private profiles.

- Reminder RED: 4 expected failures / 8 cases, then GREEN 8/8. Two existing inline-edit controls also pass. Actual service and private SQLite pin captured IDs/values; late success/refusal/exception feedback cannot alter another selected reminder's row.
- Quiz RED: 4 expected failures / 10 cases, then GREEN 10/10 plus five existing answer/grading/history cases and one policy preflight control (16/16). Held fake server API beneath real QuizScopeService/ServerQuizService, real shell/inner Button events and independent App Ping. Dashboard opens during the hold; both repeat directions preserve one uncancelled request; failure and policy refusal remain retryable.
- CI contract tests: 167/167, no skips/errors. Twenty-two new selectors append after the unchanged 845-entry census prefix; 17 shards and all budgets remain unchanged (final floor867).
- Workflow source guards: 11/12 pass locally. Canvas Mermaid requires the pinned Python build; hosted natural PR CI must qualify it. No asset source changed.
- Ruff lint passes for changed source/tests/checker. New tests, Study files and census contracts format clean. SchedulesWorkbench has identical pre-existing unrelated formatter differences on HEAD and the changed tree; the changed function formats clean. Whitespace check passes.

The first reminder collection import error, first incomplete quiz run (held-response cleanup), and first CI contract invocation (wrong test filename) are excluded from counts. Corrected runs are retained. Private full logs remain in `.test-tmp/task-36728-36729/`; adjacent JSON/compact JUnit records retain exact commands, results, failure excerpts and source hashes. Command paths identify the local runtime; they are evidence, not portable instructions.

ADR required: no. Reminder repair implements ADR-018; ADR-116 remains Proposed context. Quiz repair stays within ADR-011/033 owners. The existing metadata finding remains with TASK-32802.1 and is not implemented here.

Fresh Superpowers review on4fd871e769 found one Important panel-retirement crash, no Critical/Minor findings. The fix captures the original answer widget before the await; a retired/replaced panel receives no state or widget publication, and a successful remote result is reported with its created attempt ID. Four mounted retirement/replacement cases now pass. Returning to Quizzes and a new explicit Start remain usable. Retired panels do not imply remote rollback. Final quiz group:20/20, including19/19 private children, zero failures/errors/skips; matched initialized-panel RED2/2 and retired-panel NoMatches RED retained. The first replacement seed raced initialization; that unqualified draft-loss evidence is superseded by the corrected matched run. See adjacent superpowers-review.md for every scope ruling and review-fix-evidence.json for current source hashes. No second reviewer was dispatched, per executing-plans; the verified correction closes the finding.

Final targeted total:30 UI/controller cases plus167 CI contracts, all passing. Natural hosted PR CI is required after publication; its current-head status and hosted Canvas qualification are recorded on the PR, not claimed from these local receipts.
