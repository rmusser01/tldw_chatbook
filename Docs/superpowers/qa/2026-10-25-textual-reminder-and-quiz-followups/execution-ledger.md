# SDD ledger — plan: Docs/superpowers/plans/2026-10-25-textual-reminder-and-quiz-followups.md

Pre-flight: no shared interfaces between reminder edits and quiz attempts.
Execution: native inline; one fresh whole-branch Superpowers review. AGENTS.md targeted-only verification overrides skill requests for an unapproved full sweep.

- [x] Task 1: TASK-36728 red/green, scoped checks and commit.
- [x] Task 2: TASK-36729 red/green, scoped checks and commit.
- [x] Census/source integration and fresh review.
- [ ] PR against dev, natural CI/review feedback and task closeout.

Task 1: baseline collection attempt task1-red excluded: test import resolved a four-level relative module incorrectly; corrected to production tldw_chatbook.Scheduling.events. No production edits.

Task 1: RED verified on unchanged source: 4 expected failures (independent target + saved/refused/exception feedback), 4 ordinary controls pass, zero errors/skips; task1-red-correct-import result.xml,150.47s. Implemented captured reminder+field group and target-scoped feedback.

Task 1: GREEN 8 new mounted cases + 2 existing inline controls, zero skips/errors. Ruff lint and new-test/changed-function format pass. Whole schedules_workbench formatter differences match HEAD baseline exactly (unrelated _mark/_mark_all/delete_reminder formatting); no unrelated reformat. git diff --check passes. Fresh whole-branch review remains pending.
Task 1: complete (commits 6b75fc9..ad2483e, tests: C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/task-36728-36729/verify-recorded.py task1-evidence.json → Verified same-source SHA256 and recorded complete green JUnit receipts; no redundant test rerun.)

Task 2: initial RED proves shell queue timeout and inner-inner duplicate; run stopped after fifth case cleanup stalled because held response was released outside run_test exit. Added inner async context cleanup to release before App teardown; incomplete run excluded from aggregate counts. Production untouched.

Task 2: complete unchanged-source RED 10 cases: 4 expected failures (shell navigation and inner-inner/both cross-entry duplicate starts), 6 controls pass, zero errors/skips, 96.67s. Implemented shell worker and shared pending admission; removed inner cancellation of admitted Start.

Task 2: GREEN16/16 (10new+5attempt+1policy), 209.87s, zero skips/errors; Ruff lint/format clean. Integration contracts167/167, source guards11/12(local pinned Canvas runtime unavailable, hosted CI required).
Task 2: complete (commits ad2483e..0443c60, tests: C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe -X utf8 .test-tmp/task-36728-36729/verify-recorded.py task2-evidence.json → Verified same-source SHA256 and recorded complete green JUnit receipts; no redundant test rerun.)

Task 2 final commit amended to 4fd871e769 solely to compact generated JUnit traces; product/test source hashes unchanged. Fresh reviewer dispatched on1850d2599c..4fd871e769 using gpt-6-astra/high, clean context.

Final review: Important P1 new shell Start completion after panel replacement can crash. No Critical/Minor. AC5 added before implementation; four mounted shell/inner retirement/replacement tests starting RED.
Final: Ruling: Physical server rollback/order after same-reminder cancellation: preserve existing mutation semantics; mounted fake-client tests do not claim rollback. If wrong, physical remote ordering remains unqualified.
Final: Ruling: Concurrent question-list rebuilding/teardown: remains TASK-34752; this fix must independently cover newly enabled Start completion. If wrong, existing list interleavings/teardown can still fail outside Start.
Final: Ruling: Session metadata after refused/failed Start or then-selected label: preserve existing behavior; no new durable success claim. If wrong, resume metadata can still describe a failed or later-selected quiz.
Final: Ruling: Bracket preview rendering: remains existing TASK-32802.1, not duplicate implementation. If wrong, the documented label escaping defect remains visible.
Final: Ruling: Full-app pointer/live-server behavior: not established by mounted fake-client receipts; retain that evidence limit. If wrong, untested integration differences could remain.
Final: Ruling: Hosted Canvas qualification: require natural hosted CI; do not count local pinned-runtime refusal as passing. If ignored, asset reproduction drift could go undetected.

Review fix RED: 4/4 fail; both retired paths raise WorkerFailed/NoMatches. Replacement-only RED2/2 directly confirms lost newer draft (empty versus replacement draft). Guard captured answer widget before state/UI publication, report actual created attempt ID on retirement; retain existing return/new explicit Start behavior, no remote rollback claim.

Review fix first full group:18pass/2replacement failures, zero errors/skips. Replacement seed raced independent panel initialization; worker readiness is not proven by default-enabled button. Added explicit join of non-held workers and queue drain before seeding, plus pre-release retained-value assertion. Original retirement crash RED remains valid; replacement-only earlier RED is superseded by matched corrected-fixture RED. No broader question-list fix.

Matched replacement RED with initialized replacement panel:2/2 fail on old4fd source with empty versus replacement draft. Guarded source restored by finally. Complete20-case corrected-fixture group running.

Final: fixed P1 shell Start completion on retired/replaced quiz panel — test_start_completion_respects_replaced_quiz_panel, retired NoMatches RED2/2 and matched replacement overwrite RED2/2→GREEN4/4; affected group20/20, private children19/19, census contracts167/167. Captured-widget guard; successful remote outcome names attempt ID, no new/replaced panel write.
Both tasks marked Done via CLI with all ACs checked (36729 AC5 added before repair), plans/notes/ADRs/docs/lesson/evidence present. Task1 source hashes unchanged; scoped lint/format and affected guards pass; inherited formatter/pinned Canvas limits retained. Review fixes ready for commit. PR against dev and natural hosted qualification follow through finishing workflow; keep attached worktree.
