# Textual follow-up fixes: TASK-36930 through TASK-36932

Implementation base: dev `7a5e2513c5d7e00ef57f3d19cd8e81a01e006d05` (merged PR #3248). Implementation commits: `6100adf984`, `6df4ea6dc2`, `f905a3e154`. The earlier audit packet in this branch is historical, source-pinned reproduction evidence; it is not the repair acceptance suite.

[Plan](../plans/2026-10-25-textual-followup-fixes.md). Existing canonical ADRs [007](../../../backlog/decisions/007-personas-workbench-route-consolidation.md), [011](../../../backlog/decisions/011-chatbook-workbench-ui-system.md), [033](../../../backlog/decisions/033-application-session-state-ownership.md), and [042](../../../backlog/decisions/042-watchlists-reader-first-ia.md) apply. No schema, dependency or service-boundary decision was added.

## Behavior and targeted verification

| Task | Result | Evidence on the final implementation |
|---|---|---|
| TASK-36930 | Final quiz Submit checks runtime/scope generation, attempt and original answer-panel identity before publishing success, failure or awaited history. Obsolete completion leaves the current attempt and exact draft intact. | 15 final mounted cases passed, with 15 clean private-child reports (including obsolete history success and failure); 11 adjacent Start/retry children and 12 Study contract/keyword guards passed. |
| TASK-36931 | Canonical file-backed inbox list/read/dismiss calls use the existing finite-worker ownership helper. Noncancelling finite mutation workers retain each successful write’s completion-to-refresh path; the shared loader remains latest-wins. In-memory and alternate async/custom stores retain their caller-thread contracts; policy and errors remain nonfatal. | 12 final cases passed (6 mounted private children and 6 controller contracts), including both late native mutation refreshes; 11 existing controller/store/nonfatal cases and 7 finite-worker lifecycle outcomes passed. |
| TASK-36932 | Lore Save promptly admits one worker with captured identity/version/payload, releases the screen queue, preserves newer raw edits, and rejects publication to retired selections/panels. Cancellation before entry permits explicit retry; running native work retains admission through physical completion. Worker metadata is fixed, without captured Lore text. | Final 9-case file passed, with 9 clean private-child reports; a final focused 4-case control and the existing Lore optimistic-version regression passed. |

The new files are `Tests/UI/test_study_quiz_submit_lifetime.py`, `Tests/UI/test_notifications_inbox_responsiveness.py`, and `Tests/UI/test_personas_lore_save_responsiveness.py`. Intended behavioral REDs were observed before the fixes. Lore metadata also had an intended failing privacy assertion before fixed worker name/description were supplied. Root independently parsed the final private-child XML and checked the frozen Lore source hashes.

The post-change message-pump blocking-I/O guard passed all 6 cases. Combined census, queue-pressure and exact Personas module-size/slack contracts passed 166 cases. The Personas ceiling decreased from 16,883 to the measured 16,818 lines. Scoped lint/format checks introduced no new diagnostics: pre-existing Personas screen/helper E721 debt is unchanged, while changed blocks and new test files are formatted. No full local suite was run.

Local receipts are retained privately under `.test-tmp/task36930/` and `.test-tmp/textual-followup-fixes/`, with exact commands, parent XML, actual private-child XML, RED attribution and source hashes. Implementation/review reports are `.test-tmp/task36930-implementation.md`, `.test-tmp/task36931-implementation.md`, `.test-tmp/task36932-implementation.md`, and the corresponding `review36930.md`, `review36931.md`, `review36932.md`. These are local evidence locations, not portable committed test artifacts.

## CI coverage

All 36 new exact cases are registered in the existing UI gate: 15 Study, 12 inbox and 9 Lore. The census floor is 929. The three review-follow-up cases add an obsolete history exception and both late native mutation refreshes. The original dev 893 selectors retain their exact order; only the appended new tail was balanced across the existing 17 shards. No test, shard or timeout guard was removed or increased.

Placement used actual PR #3248 job durations (run 40241711952, UI shards 776–1,081 seconds) and local case timings. These estimates are not hosted CI results. Hosted outcomes and execution counts will be recorded after this PR runs.

## Review and delivery

Superpowers requesting-code-review replaces Qodo as requested. Study and inbox task-scoped reviews approved the implementation with no Critical/Important findings. Lore task review also approved without findings. The fresh whole-branch review found one Important inbox mutation-cancellation/refresh race; both real-SQLite ordering variants demonstrated the defect before the two-line noncancelling mutation-worker correction. The final inbox12/Study15 files, adjacent11 and lifecycle7 selections all pass (45 parent cases, 21 corresponding actual private children), with matching frozen source hashes. One scoped re-review of the complete fix wave is pending. Draft [PR #3251](https://github.com/rmusser01/tldw_chatbook/pull/3251) targets dev. Initial Perf Guard passed; current-head hosted delivery qualification remains pending.

The optional direct obsolete-history-exception witness now passes with existing production guards, alongside the successful late-history witness. The task’s incorrect affected-source path was corrected through Backlog CLI. Existing dependency deprecation warnings are retained rather than hidden.

## Limits and existing failures

- Two legacy unmarked Study tests and one legacy Watchlists refresh test fail before the controller/UI under the current profile bootstrap (`raw_source_selection_changed`). A narrowly supported Study bootstrap diagnostic passed both test bodies but retained one native-app worker/config-retirement teardown error. These are not counted as green runs; no fixture, lifecycle or permission guard was weakened.
- A baseline screen-ratchet selection found unchanged `chat_screen.py` at 24,345 lines against a 24,344 ceiling on both dev and the branch. This PR changes no Chat code or ceiling. The modified Personas module/slack guards pass.
- An inbox parent/child run collected mismatched parameter IDs while test source was changing; it is invalid evidence, retained and excluded. The source-freeze lesson records the actual recurrence. All quoted final children ran on frozen files.
- The first Lore launch used system Python without pytest-timeout and collected no cases. An intermediate Lore run had one forced-panel setup failure; the final replacement fixture evicts only its owned readiness-cache entry and all nine final children pass. Production center-view caching is unchanged.
- Lore responsiveness uses a controlled hold around the real manager/private SQLite boundary. It establishes screen-queue progress, not natural storage latency or a whole-app freeze. The inbox mutation tests use real competing SQLite writes. No native-terminal, full TldwCli policy-commit or live-server qualification is claimed.
- TASK-33283 (Notifications pane rebuilding/focus), TASK-33279 (Research follow-up admission), TASK-34752 (general question/history list ownership), and existing Roleplay draft/transition owners remain separate.


## Review scope decisions

The whole-branch reviewer explicitly set aside the following behaviors. The coordinator accepted each boundary because the corresponding production contract is unchanged or has a separate existing owner; this does not certify or fix it here. The practical cost if that judgment is wrong is an undiscovered existing defect in the named area.

1. Existing exclusive Study Submit cancellation and uncertain-commit/server idempotency; definite pre-write retry remains covered.
2. Same-source active-server binding replacement and full TldwCli runtime-policy handoff; controlled mounted-host tests do not establish them.
3. Independent Study question-list overlap and non-Submit history publication; existing owners retain these paths.
4. Notifications table recomposition and focus loss; TASK-33283 retains the demonstrated defect.
5. Lore entry/Options aggregate dirty/transition domains and general center-view cache repair; TASK-33910.8/.14 retain those boundaries. Settings-save raw edits are covered here.
6. Research follow-up queue retention/default HTTP client work; TASK-33279 retains the archived finding.
7. Replaying historical observation probes on repaired sources; the source-pinned archive must refuse changed sources, so current regressions are the repair gate.
8. Baseline profile-bootstrap/native teardown failures, unrelated Chat size and dependency warnings; they remain qualified existing debt without weakened guards.
9. Native terminal frames, natural latency/contention frequency, live-server behavior and full-suite regression freedom; none is claimed from controlled targeted evidence.


Initial hosted run [40254312374](https://github.com/rmusser01/tldw_chatbook/actions/runs/40254312374) on `f905a3e154` reported an existing Chunking Lab failure in UI shard 11: `test_sample_read_rechecks_intent_after_queued_transition_computes` retained `x` at its final Undo editor assertion. The test and all Chunking production files match dev; the exact unchanged-source local rerun passed both parent and actual private child without errors or skips. This is retained as an unresolved initial-run timing observation, not a claimed repair or a clean initial CI run. TASK-36695 is its existing owner. Required CI on the final review-fix head will run the case again unchanged.
