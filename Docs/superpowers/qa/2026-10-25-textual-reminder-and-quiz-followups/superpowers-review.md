# Fresh Superpowers review

Reviewer: gpt-6-astra/high, clean context, read-only range 1850d2599c..4fd871e769. No Qodo. Full package, plan, tasks and Review Focus supplied. No tests run by reviewer; source-traced finding reproduced by executor before repair.

Strengths: captured reminder identity/field, target-scoped feedback, pre-await shared quiz admission with finally release, real owner/service/persistence tests and preserved census placement.

Critical: none. Important P1: shell worker survives Flashcards/Paths panel replacement, then Start completion queries removed #quiz-answer-input and can terminate the app. Dashboard-only coverage hides instead of removes the panel. Add actual section swap/return tests and guard publication without pretending the remote attempt rolled back. Minor: none. Initial verdict: With fixes.

Declined-to-judge items and executor rulings:

1. Physical server rollback/order after same-reminder cancellation: preserve existing mutation semantics; mounted fake-client tests do not claim rollback. If wrong, physical remote ordering remains unqualified.
2. Concurrent question-list rebuilding/teardown: remains TASK-34752; this fix must independently cover newly enabled Start completion. If wrong, existing list interleavings/teardown can still fail outside Start.
3. Session metadata after refused/failed Start or then-selected label: preserve existing behavior; no new durable success claim. If wrong, resume metadata can still describe a failed or later-selected quiz.
4. Bracket preview rendering: remains existing TASK-32802.1, not duplicate implementation. If wrong, the documented label escaping defect remains visible.
5. Full-app pointer/live-server behavior: not established by mounted fake-client receipts; retain that evidence limit. If wrong, untested integration differences could remain.
6. Hosted Canvas qualification: require natural hosted CI; do not count local pinned-runtime refusal as passing. If ignored, asset reproduction drift could go undetected.

The single fix pass uses RED→GREEN regression evidence rather than a second reviewer, per executing-plans. Resolution and hosted CI will be recorded in the main QA report.
