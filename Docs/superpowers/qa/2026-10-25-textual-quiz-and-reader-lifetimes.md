# Quiz Start and Reader lifetime repairs

Implements [TASK-36899](../../../backlog/tasks/task-36899%20-%20Keep-pending-quiz-Start-within-its-runtime-and-scope-lifetime.md) and [TASK-36900](../../../backlog/tasks/task-36900%20-%20Keep-Watchlists-navigation-responsive-during-article-body-reads.md) on dev 79da83a8dca98eab6ca854a7505b8abd264eb959. The [plan](../plans/2026-10-25-quiz-start-and-reader-lifetimes.md), existing ADR-011/033/042 and [original audit](2026-10-25-textual-book-after-pr3247.md) bound the work. No new service, storage, dependency or token contract was introduced.

## Behavior

Study captures the controller reset generation, runtime mode and scope arguments before Start. A reset, including away/back, retires that publication authority even when the answer widget remains attached. Obsolete success does not adopt an attempt or clear a newer draft; feedback identifies an already-created remote attempt in the previous context. Obsolete failure is similarly distinguished. Shared single-pending admission, removed-panel protection and ordinary retry remain intact.

Watchlists ItemSelected returns after admitting an explicitly named, exclusive wc_item_detail worker. Body reads remain off the event loop. Selection generation, snapshot generation/identity, page, runtime, active section and attachment checks guard hydration, feedback, reader publication and automatic mark-read together. Checks after arrival protect against a DB thread that finishes despite cancellation. List bodies remain narrow and successful publication retains the shared row identity. j/k/space navigate from the latest list selection while hydration is pending; their displayed-page bounds remain intact.

## Local evidence

Python 3.12.10, Textual 8.2.8; targeted tests only. [Machine-readable receipts](2026-10-25-textual-quiz-and-reader-lifetimes/verification.json) retain exact selectors, XML digests, independently inspected child counts, source pins and census proof. Private raw profiles/logs/XML remain under .test-tmp/task-36899-36900, outside the published artifact.

| Gate | Result and scope |
| --- | --- |
| Study | 61 passed: complete Start responsiveness and attempt-retry files plus scope, local and server quiz service contracts. All 27 private child reports independently clean. |
| Reader | 18 passed in the complete final file, including all three review keyboard cases; all 18 private child reports independently clean. |
| CI contracts | 152 passed across census, Derived Artifacts, shard replay and shard-count contracts after the review repair. |
| Census | All 26 new exact selectors actually collect. Original 867-entry prefix unchanged; floor 893; 17 shards and 20-minute cap unchanged. |
| Static | Changed Study source/tests and Reader tests pass Ruff and format checks; Reader source has exactly the same three E721 diagnostics as baseline, no new issues. All other Reader lint and changed-range format/whitespace checks pass. |

The regression development retained valid pre-fix assertion failures for context adoption, previous-context failure notices, same-screen navigation, obsolete results/errors and keyboard selection. Two process-local arrival-guard removal controls fail at stale hydration/mark-read assertions after joining the actual old Worker, proving protection independently of cancellation. Controlled holds are released inside the mounted context; final stale-effect assertions join the held worker, not merely its thread-boundary event.

## Invocation qualifications and excluded runs

The unchanged server from_config test initially failed through source-bound TLS config admission (raw_source_selection_changed). The qualified 61-case run applies the repository-supported bootstrap_profile marker only to that case through an invocation plugin. No profile/network guard or product code was relaxed.

Legacy Watchlists neighbors initially hit that profile admission and then incomplete cold-start setup before fake-controller injection. With the supported marker and actual initial wc_snapshot/wc_items/wc_tree/wc_overview plus queued surface work settled, 51 of 53 cases passed; the remaining two atomic-scope/header cases subsequently passed 2/2 on both the repaired screen and the exact dev79da screen. Those are separate receipts, not a claim of one 53/53 run. The original assertions remain intact.

Four additional legacy j/k/space controls were compared without changing assertions. Both current and exact dev79da source produce the same 1 pass/3 failures: missing content-body before the first j, unset initial selection before typing, and an empty startup list before space. Their fixed-delay/cold snapshot failures are excluded from clean counts and do not demonstrate a repair-caused regression. The new isolated production-owner keyboard tests supply the review gate.

Incomplete early fixtures, interrupted runs, pre-settlement header observations and historical audit probes are excluded from repair success counts. The historical probes intentionally assert symptoms pinned to dev79da.

## Fresh review and resolution

A fresh gpt-6-astra/high Superpowers requesting-code-review reviewed the complete dev79da..eaa2728a5cc6bede6af5a4e62974b6fd174ad7b7 branch, inspected the tests/adapters/receipts and found one Important issue, no Critical or Minor issues: j/k and next-unread still anchored to the last published article while a later selection was pending.

The parent reproduced all three paths with actual Pilot keys (3 expected assertion failures, 0 errors), changed the two navigation anchor expressions to prefer the list selection and retained the hydrated article as publication state. All three keyboard cases then passed within the complete 18/18 Reader run. One review and one TDD repair pass were used; no second-review approval is claimed. Required hosted CI and current-head review qualification are recorded on the PR after publication.

## Decisions and limits

- Sources form rebuilding remains with existing TASK-33283. The separate form-selection disturbance remains possible until that work is implemented.
- Study question-list overlap/teardown remains with TASK-34752; this Start-context correction does not certify that independent owner.
- Same-source active-server binding replacement is uncertified. Tests establish local/server and Study-scope reset behavior, using a controlled API client and actual scope/local SQLite services; no live server or full TldwCli policy/binding journey is claimed.
- Historical audit probes remain baseline observations; treating their old-symptom assertions as current acceptance tests can produce false regression reports.
- Full-suite health is not asserted. Targeted local verification follows the user's AGENTS instructions; behavior outside targeted checks and required hosted CI remains uncertified.

Reader tests mount the production destination/CSS and private file-backed SubscriptionsDB with a controlled hold before its actual body read. They do not establish ordinary SQLite latency or a physical terminal journey. Existing task/ADR links were retained rather than creating duplicate owners.

## CI-driven corrections after publication

First head 0d91e6c10774dadc55cde3156a116e681aaa7630 ran all 26 new cases successfully across 17 UI shards (maximum 988 seconds). The [retained first-attempt receipt](2026-10-25-textual-quiz-and-reader-lifetimes/first-ci-attempt.json) records two failing gates: the Study keyword architecture checker could not resolve a tuple-index spread, and UI11's unrelated Library receipt test missed its click. The required aggregate correctly failed on those results; the original logs/results are retained, not counted as green CI.

The Start call now passes named captured mode/scope locals, composing the same generation/context key. The fail-closed checker remains unchanged: its full file went from 7 passed/1 failed to 8 passed. The full qualified Study selection was rerun successfully (61 passed, 27 clean private children).

A mouse-only receipt trace captured events at (48,30) on the queue panel while the enabled button was at row34; no Button.press fired. The test now waits across layout turns for a stable hit-tested receipt region and reacquires the button before Pilot freezes coordinates. The actual click and Reader/SQLite assertions remain unchanged. This is a narrow CI test-readiness correction under existing TASK-34000.18, not a Library behavior change or duplicate backlog task. All three receipt variants pass. The complete ten-case Library file was 9 passed/1 failed at its unrelated, unchanged Edit/Save final-highlight assertion. A longer-wait diagnostic did not settle that sequence and was reverted; no fix or clean full-file claim is made for it. The original sequence also fails against pinned dev79da with both changed production modules replaced by their original Git blobs: its final Reader is Team handbook. The test function AST matches dev exactly. This fake-service sequence is a separate baseline limitation under TASK-34000.18; this patch makes no broader Library correctness claim. Early diagnostic import/delegation-property failures were probe errors and are excluded.

The next natural PR CI cycle qualifies the corrected published head. Its outcome and review-thread read are recorded on the PR, without another evidence-only commit.
