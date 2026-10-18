# PR3162 baseline and descriptor repair qualification — 2026-10-18

TASK35584 continues [the recorded rebase qualification](2026-10-18-archive-recovery-pr3162-rebase.md) from published head `ccab42365855c5a37c17a2c6bed5c4bb6f2ee96f`, based on dev `9af0c59263dce00fa44be4a74d5d7e78712346bf`. The original verified feature branch remains unchanged. [Plan](../../plans/2026-10-18-archive-recovery-baseline-repair.md); [independent review](2026-10-18-archive-recovery-baseline-repair/review.md). ADR166, SQLite kernel ADR228 and design-token/component ADR150/161 apply; no new architectural boundary or ADR.

## Ten original failures

All ten were reproduced before repair. These changes retain the checks they were intended to exercise:

- Rollback observer2: enter a real managed SQLite transaction before injecting commit/rollback faults; restore native rollback before cleanup. Ambiguous completion and observer release remain required.
- Oversized response2: multibyte UTF-8 exceeds the byte cap while staying within the credential sanitizer's codepoint cap. Bounded replacement, one artifact and absence of private payloads remain required.
- Cold restart1: derive recovery time from the persisted reservation rather than a fixed historical date. Grace, monotonic recovery and idempotency remain required.
- Migration inventory2: align the exact Evals allowlist with the existing deliberately pinned schema7/results-index migration; future schema changes still fail the pin.
- Component governance3: replace dimensions with equal-valued existing tokens, add four necessary semantic feature tokens, use existing width classes and an explicit dimension reset. Merge three adjacent identical Library receipt rules, reducing active lines2004→1998. Guard budgets and all 262 existing token values stay unchanged. Resolved per-selector declarations in all five sheets are identical; the normal builder regenerated bundles.

Trace verification: **8 passed** (the five failures plus three connected controls). Governance/migration/token/bundle verification: **50 passed**. These cohorts overlap other checks and are not additive totals.

## Descriptor owners and repairs

The original nine-file storage cohort reproduced **+295 descriptors**, retained after diagnostic GC. Passing SQL and transaction rollback did not establish native retirement.

- Archive search opened worker-local Workspace and chat handles without a finite owner. Both read scopes now use the existing operation-owned connection guard. Newly opened handles retire on their owning thread; live borrowed handles and transactions remain untouched. Actual native-closure RED2/borrower-pass2 precedes GREEN across 81 search cases.
- Workspace schema initialization could throw after acquiring a held connection and leave it reachable through custody leases. Its constructor now closes its own current-thread handle before rethrowing the same exception. Real native-handle tests cover RuntimeError and KeyboardInterrupt; five actual migration-failure controls still pass.
- Registry tests created unclosed Workspace and Media owners; migration inspection tests used transaction contexts without closing SQLite. Local test-owner cleanup now closes those resources, including the one worker's own finally. Raw inspection contexts preserve commit/rollback semantics while closing.
- Mounted tests disposed a runtime directly but left it attached to a factory-registered app. Later factory cleanup disposed it again and reopened chat storage after quiescence. The fixture now uses the existing dispose-and-detach helper. A real factory cleanup regression proves no reopened connection.
- Four thinking controls closed only their main-thread replay connection, leaving capture-privacy worker handles. Their fixture now drains the controller, closes its gateway and quiesces the exact test database. A retained worker-handle regression proves physical closure.

Independent review found a related deadline defect: with a0.5s search budget, SQL finished in0.052s but owned WAL TRUNCATE cleanup waited1.175s for another live reader. An optional monotonic deadline now caps only a newly departing handle's busy timeout before close. Ordinary close behavior and borrowed timeouts/transactions remain unchanged. A native descriptor recognizes a handle already retired by maintenance, preserving the operation's original exception. Final ownership cohort: **18 passed**, including held-reader, borrowed-timeout and quiescent-retirement cases. The earlier deadline/search cohort passed 97 cases.

## Integrated qualification

All runs use Python3.12, repository disposable profile/keyring/offline protections, fake providers, fresh explicit basetemp/cache directories and ordinary GC settings. Diagnostic plugins import application code only after normal collection. No per-test forced collection, threshold changes, global bootstrap-marker injection or warning suppression was introduced.

- Storage plus ownership: **282 passed in95.49s**, no warnings; final post-teardown/session-GC descriptors17→17, zero owned SQLite files. The two later quiescent-retirement controls are additionally covered by the18-case final ownership run.
- Captured follow-up: the first post-search-fix50-case run passed but exposed20 remaining descriptors from four thinking fixtures. All 46 archive follow-up cases plateaued at17 with no SQLite files. The repaired thinking fixture's five-case run then plateaued at17/zero files/zero observed live native handles. Final combined rerun: **50 passed in114.70s**, same-process final teardown/session GC17→17 with zero SQLite files.
- Mounted archive flow: **39 passed/3 failed/1 deselected** across 42 cases, with all complete teardowns and session GC at17 descriptors and zero SQLite files. The three failures overcounted generic hidden-turn notifications because setup turn custody completion was mistaken for rendered receipt acknowledgement. A controlled delay of only the real final acknowledgement reproduced all three; test preconditions now await the real persisted unseen-mark retirement before installing spies. A passive wait alone exposed the harness backing app's unpumped callback queue. The fixture now routes only `call_later` through its mounted host, allowing the unchanged real receipt acknowledgement to finish. A temporary equivalent adapter passed all six ordinary parameters at17 descriptors/zero SQLite files; the final source passes **9 ordinary cases in51.97s**, including the six notification parameters plus core button/agent paths and native-cleanup regression. Every teardown and session GC stays17 descriptors/zero SQLite files. The same controlled delayed-ack plugin also passes **3 cases in26.85s**. The42-case flow has passing coverage across the39 earlier passes and these corrected controls; the whole flow was not redundantly repeated. Assertions, navigation spies, view-generation checks and production notification policy are unchanged.
- Separate mounted native-owner qualification: **3 passed**, repeated fresh app rounds stay17 descriptors/zero SQLite files/zero observed live native handles after complete teardown and GC. This is distinct from the excluded long five-cycle test.

The CSS control cohort has 21 passes and four additional failures. Two voice tests fail identically with the exact pre-edit helper; they remain unrelated existing failures. Two Media controls fail before rendering due to profile-selection context; their unchanged real bodies pass2/2 through the established private-profile runner. No voice behavior or profile guard was changed for these unrelated failures.

## Evidence and limits

[Validation manifest](2026-10-18-archive-recovery-baseline-repair/validation.json) records source/evidence hashes and commands. Raw logs, native creation/close stacks, per-test FD paths and diagnostic scripts remain at `/private/tmp/archive-recovery-pr3162-baseline-20261018/`; helper reports remain in the task's `.superpowers/sdd/2026-10-18-archive-recovery-baseline-repair/` directory. Intermediate diagnostic setup mistakes and actual RED outcomes are preserved, not counted as successful qualification.

All 16 modified Python files parse without duplicate definitions; critical Ruff passes, with zero introduced per-file diagnostics against the exact published parent. Existing lint/format debt is retained. Generated CSS equivalence, bundle sync and whitespace checks pass. The frozen code diff is bound by the validation manifest; independent review records the resolved checkpoint finding and final limits. Publication is the remaining step. Historic QA/captures are unchanged.

No full suite, installed-distribution build, full sidecar replay module, long resource-plateau repeat or new live app/model requests. The authorized model budget remains30/30 consumed. Existing unrelated voice failures and inherited static debt remain explicit. PR remains draft; no merge or auto-merge is requested.
