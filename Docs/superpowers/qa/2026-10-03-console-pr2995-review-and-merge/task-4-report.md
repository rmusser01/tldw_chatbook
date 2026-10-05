# Task 4 report: uncertain Console starts and commit ownership

Status: **DONE_WITH_CONCERNS** — source implementation, affected qualification and self-review complete; independent review and combined startup/import qualification remain controller-owned.

- Commit: `d3b2ade44b49afc96f78c592ac271e3231472b27`
- Immutable parent/BASE: `fc1627e99b8722a0a69812530d950410e8356ba0`
- Aggregate source SHA256: `c0c6ecb2e37deb5858e455f032790eb1315d4508af42696ad1e8cc054e3271e4`
- Exact paths and individual source hashes: [source fingerprint](task-4-source-fingerprint.json).
- Frozen patch: [owned diff](task-4-owned.diff). [Commit verification](task-4-commit-verification.json) proves the exact ten-file manifest, parent, identical committed bytes, clean owned paths and empty index.
- Root plan and Backlog task remained dirty and unstaged. No report/evidence/root documentation was committed.

## Result and architecture

ADR required: no. Existing [ADR-211](../../../backlog/decisions/211-console-chat-destinations-and-bounded-starts.md) governs accepted charging, dual receipts and review. This implements its existing state/ownership contracts without a schema change, new eager module or retry policy.

1. **Durable accepted uncertainty is reviewed, charged and paused.** `AutomaticWorkLedger.mark_chat_start_review_required` checks the current runtime owner and exact attempt owner inside the FULL-durability transaction. An accepted attempt becomes `review_required`, and its canonical root pauses with the fixed `interrupted_work` reason. Counters, generation and deadline are preserved. Prepared/aborted/completed attempts cannot be reclassified by this API. Repeating verified review is idempotent; accepted work never refunds or replays. A failed preaccept abort with an actually accepted row routes conservatively to review.
2. **Failed settlement restricts shared allowance admission.** A private process-local registry is checked by the existing ledger admission gate. Its key identifies the canonical native file store and exact owner/attempt; unrelated in-memory databases receive distinct private store identities. The restriction is installed before uncertain settlement I/O or physical drain, so another ledger handle cannot admit sibling work while settlement is blocked. Established chains restrict their canonical root, allowing an independent manual allowance. When no chain can be established, denial is conservatively owner-wide. A verified committed review clears that attempt's restriction. Same-owner `recover` does not clear unresolved current-owner uncertainty. Committed replacement-owner recovery retires foreign-owner restrictions; a late old-owner failure cannot pause or deny healthy replacement work. No public registry/service contract was added.
3. **The exact native conversation commit remains owned until physical completion.** Authorization retains a dedicated commit task before its first await. The native branch captures the exact store/database, shields the task and uses the existing owned DB call for file-backed work. Repeated cancellation drains this task before settlement, outcome publication or automatic-primary capacity release. Provider work has separate custody. Postcommit publication also checks store/incarnation and runtime owner/bridge identity. Ordinary manual/queue DB helper bodies are unchanged; memory DB behavior stays inline.
4. **Writer contention is refused at the existing source cutoff.** Final source/target/owner checks and native ledger acceptance still have no intervening await. The same held operation-owned connection temporarily uses `busy_timeout=0`, restoring its original timeout in `finally`. Accepted custody is assigned before fallible restoration. BUSY/LOCKED classification uses SQLite codes and a fixed refusal reason. `synchronous=FULL` and existing durability policy are retained.
5. **Status and failure vocabulary is shared without startup growth.** Named handoff launch values and their Literal type live in the already-loaded `message_metadata` module. Unexpected native-start failure logging contains only fixed phase and exception type, with no raw exception message/traceback/body. Expected refusal, cancellation and proven preaccept unexpected failure are distinct; uncertain accepted work remains review-required. Public new ledger API docstrings include Google Args/Returns/Raises.

## Owned files

Production: `Chat/console_chat_start.py`, `Chat/console_chat_controller.py`, `Chat/message_metadata.py`, `DB/automatic_work.py` under `tldw_chatbook`.

Tests: `Tests/Chat/test_console_chat_start.py`, `Tests/DB/test_automatic_chat_starts.py`, the three directly affected durable commit lifecycle owners, and `Tests/Agents/test_automatic_child_scope.py`.

The four inherited lifecycle/context fixtures were repaired only after exact immutable BASE reproduction and explicit controller scope authorization. They use the repository's existing `private_profile_test` helper and request isolation. Original synchronous test bodies/assertions are AST-identical to BASE: 7 offload bodies, 2 diagnostics bodies, 16 postcommit bodies and 17 child-scope bodies. Parameterization yields 30 manual lifecycle cases and 21 child-scope cases. No new marker registration, guards disabled or assertion weakening. [Self-review proof](task-4-self-review.json) also records unchanged ordinary manual durable DB helper bodies.

## Verification and diagnostic history

Every entry below has a corresponding `task-4-<name>-receipt.json`, `.log` and, for pytest, `.xml` in this directory. Receipts retain exact argv/cwd/revision/exit/profile root/source hashes. Tests used the existing Python 3.12 venv and disposable private profiles. No dependency install, full sweep, warning suppression, new marker, fetch/rebase/push or Git housekeeping occurred.

| Evidence name | Exit | Result / interpretation |
| --- | ---: | --- |
| `red` | 1 | Meaningful preimplementation RED: 12 failed, 1 passed, 140 deselected; 9.11s. Real review, physical cancellation, unexpected failure/privacy and file-lock controls reproduced missing behavior. |
| `focused-green` | 1 | 12 passed, 1 failed; 7.87s. A new test incorrectly forbade an empty assistant placeholder after physical commit. Corrected its oracle to forbid provider content while retaining durable placeholder visibility. Original failure retained. |
| `affected-green` | 1 | 221 passed, 18 failed; 172.26s. Exposed inherited profile seams plus new healthy-completion restriction timing, alias-policy and owned-handle expectations. Fixed those concrete issues, retained this failed run. |
| `focused-final-green` | 0 | Full start/native-ledger/budget/wake owners: 209 passed; 100.57s; no warnings. |
| `base-commit` | 1 | Exact BASE in disposable archive: 13 failed, 17 passed; 53.92s. Inherited lifecycle failures occurred before the protected commit path because profile configuration made Hooks unavailable. |
| `commit-owner-repair-green` | 1 | 13 failed, 17 passed; 54.18s. A disposable fixture-edit script raised ValueError before touching files, but the shell continued with unchanged fixtures. This is a setup diagnostic, not a successful repair qualification. |
| `final-owners-green` | 0 | Complete seven affected start/ledger/manual commit owners: 239 passed; 177.93s; no warnings. |
| `final-reconciliation-owners-green` | 1 | 65 passed, 15 failed, 1 teardown error; 88.48s. All DB29 and manual30 cases passed after final reconciliation assertions; the added unisolated child owner reproduced its inherited profile seam. Its passing subset is used only for final-byte DB/manual evidence. |
| `base-child` | 1 | Exact BASE: 15 failed, 6 passed, 1 teardown error; 26.88s. Child/provider work refused with `raw_source_selection_changed` before entry. |
| `final-context-green` | 0 | Final DB29 + repaired child21 owner: 50 passed; 61.27s; no warnings. |
| `latency-controls-green` | 0 | Two final real-file acceptance controls: 2 passed; 3.25s; no warnings. |
| `fatal-ruff-final` | 0 | All ten owned files: Ruff `E9,F63,F7,F82` passed. |
| `format-verify-final` | 0 | Recorded immutable BASE formatter debt and changed-line ratchet passed on final worktree. |
| `format-verify-commit` | 0 | Same formatter baseline verified against exact committed SHA. |
| `source-whitespace-final`, `staged-whitespace` | 0 | Source and exact staged whitespace checks passed. |
| `self-review`, `source-stage`, `source-commit` | 0 | AST/coverage proof, exact staging and normal source-only commit succeeded. |

**Final successful coverage is 260 distinct affected cases**, using the complete 239-case owner run plus 21-case child owner and subsequent final-byte DB/manual/latency qualification. It is a union of these passing cases, not a claim that the diagnostic reconciliation batch exited successfully. Exact successful node census is in the self-review proof. Production source was unchanged between the complete owner run and commit; later edits added DB reconciliation/manual allowance assertions, child profile fixtures and changed-range formatting. Committed bytes match the frozen source fingerprint. Formatter repair was limited to changed ranges against recorded BASE; unrelated formatting debt remains ratcheted.

### Warnings and oracle corrections

- Initial RED used `record_property`, producing one xunit2 incompatibility warning. Replaced new metrics capture with printed values; no warning filter/config change.
- Initial affected batch reported open FDs grew by 223 over the existing 200 limit during its failure-heavy session. The warning remains in the log. No limit, GC policy, cleanup guard or suppression was changed. Final complete affected batches reported no warnings.
- Native repeated-cancellation tests always release physical-thread controls in `finally` and drain them; failing controls did not leave intentionally blocked workers running.
- Canonical alias control uses an allowed lexical file alias (`alias-dir/../runs.db`). An initial symlink alias conflicted with the repository's private SQLite file policy; that test setup was corrected without weakening policy.
- Healthy receipted completion does not install a transient restriction. The initial broad control exposed that doing so could deny legitimate immediate child/wake admission; the final complete shared-allowance child/wake controls pass.

## Latency measurements and limits

From the final real SQLite controls ([log](task-4-latency-controls-green.log)):

| Control | Observed seconds |
| --- | ---: |
| Warm held-connection writer contention cutoff | 0.001272 |
| Maximum event-loop ticker gap during lock control | 0.009862 |
| Ordinary uncontended acceptance cutoff | 0.045913 |

The lock control verifies borrowed timeout 1739 and original synchronous policy restoration, a refused/aborted prepared attempt with used count 0 and unchanged draft, and continuing loop progress. The uncontended control verifies original FULL policy/timeout and retirement of a newly operation-owned main-thread connection.

These measurements bound writer acquisition in the exercised warmed-connection scenario only. Cold connection creation/setup/WAL work and FULL-durability fsync still run synchronously and can stall on slow storage. Bare offloaded acceptance would move the durable transaction past mutable loop-owned source/Stop authority and violate the required no-await cutoff. A future offthread admission protocol would require a separate architecture decision; this patch does not claim to eliminate all I/O latency.

## Review handoff and concerns

1. Review the private transient registry identity, fencing and lifecycle, especially current-owner reconciliation versus foreign-owner recovery. Unknown-root uncertainty conservatively denies owner-wide automatic allowance admission until verified reconciliation/replacement recovery; established roots preserve independent allowance admission. The fallback is process-local and does not substitute for durable owner recovery after restart.
2. Review the exact physical native commit custody/drain sequence and no-await acceptance cutoff. Real cancellation, accepted cleanup failure, stale owner, current-owner recovery, peer-handle denial before blocked settlement, memory-store isolation and nonlogging controls are included.
3. Cold connection/fsync latency remains as described above.
4. Controller owns independent spec/quality review, sequential Task5 CI ownership repairs, final combined startup/import ratchets, external Qodo thread replies/publication and merge. These are not claimed complete here.


## Fix round 1: I1 recovery cleanup ordering

Status: **DONE_WITH_CONCERNS**. I1 is repaired and qualified; independent follow-up review remains controller-owned. This section supersedes the original delivered source SHA above and preserves its historical evidence.

- FIX_BASE: `d3b2ade44b49afc96f78c592ac271e3231472b27`
- Fix commit: `c0b251a71422536a3da2be421dd60f79e2cb230a`
- Exactly two committed paths: `tldw_chatbook/DB/automatic_work.py`, `Tests/DB/test_automatic_chat_starts.py`.
- Full final Task4 source aggregate SHA256: `7b099980ab9b73d5d7383f3e2dc2a4d47514e8ceac4a73d5e41350df07ac89cc`.
- Final individual modified-file SHA256: ledger `53e3cc87df9d6029aa34076b2456a9333a041caac8ed558342c88bb797ba6f1c`; test owner `7055d0addda58c916d3a3f68f96e6b57ebb5c2de03f343c7dc842012c1ff2b6a`.
- [Exact patch](task-4-fix1-owned.diff), [self-review and two-file fingerprints](task-4-fix1-self-review.json), [committed manifest and all ten final hashes](task-4-fix1-commit-verification.json).

### Correction

The original recovery sweep ran after SQLite commit and removed every foreign-owner restriction then present. A superseded recovery could consequently erase a newer owner's uncertainty installed after a later recovery. The deterministic real-file regression pauses only the old transaction context's return after the actual commit, lets the replacement recover/accept a charged attempt/restrict its allowance, and then resumes the older recovery. It retains actual SQL/admission behavior.

Recovery now captures specific foreign registry records while it still owns the durable transaction. After successful commit it removes only those captured records whose entry identity is unchanged. Each setter creates a fresh private per-write entry token, so a later refresh of even a captured key survives. The registry lock covers short snapshot/dictionary operations; SQLite queries, transaction commit and I/O run outside that lock. There is no unfenced owner read followed by a global sweep. Failed transactions cannot reach cleanup. Same-owner restrictions remain excluded from the captured set.

The three overlap cases cover established-root and owner-wide unknown-root restrictions plus a captured key refreshed after commit when an owner identity becomes current again. They verify retained accepted state/generation charge, sibling refusal after same-owner recovery, no abort/refund/replay, verified review/root pause, and healthy sequential replacement recovery. The refreshed-key case also verifies that removing only its reconciled unknown-root restriction permits an independent manual allowance while the unresolved accepted root remains denied. The conservative unknown-root fallback and all other Task4 contracts are preserved. ADR required: no; existing ADR211 applies.

### Round receipts

All `task-4-fix1-<name>` logs, receipts and pytest JUnit artifacts remain in this directory. Receipts preserve exact command, revision, profile and source hash. All pytest runs used disposable private profiles. No full/adjacent owner sweeps, dependencies, markers, guard bypass, suppression, nested agents or network/Git remote operations.

| Evidence suffix | Exit | Output / interpretation |
| --- | ---: | --- |
| `red` | 1 | 2 failed/29 deselected; 1.63s. Initial new cleanup incorrectly called nonexistent `AgentRunsDB.close_connection`; corrected only the test to use `operation_owned_connection`. Retained as setup diagnostic, not meaningful RED. |
| `red-confirmed` | 1 | Meaningful RED: 2 failed/29 deselected; 1.58s. Both known/unknown-root cases failed at expected `DID NOT RAISE settlement_unconfirmed` after late old cleanup, with durable accepted/charged work unchanged. |
| `focused-green` | 0 | 4 passed/27 deselected; 2.34s; existing current-owner/replacement/store-isolation controls and original two race cases pass. |
| `covering-green` | 0 | 40 passed; 34.45s; no warnings. Complete affected DB owner32 + peer settlement4 + real child/wake2 + manual child allowance1 + physical commit cancellation1. |
| `final-regression-green` | 0 | 3 passed/29 deselected; 2.76s; no warnings, after changed-range formatting. |
| `final-cleanup-green` | 0 | 3 passed/29 deselected; 2.79s; no warnings, final bytes after ensuring peer/restriction cleanup runs even if the recovery worker raises. |
| `ruff-final` | 0 | Fatal Ruff `E9,F63,F7,F82`: all checks passed on both final owned files. |
| `format-final`, `format-commit` | 0 | Formatter changed-line/debt ratchet against recorded immutable FIX_BASE passed on final worktree and committed SHA. |
| `original-base-format-commit` | 0 | Original ten-path Task4 formatter baseline against immutable BASE passed on committed SHA. |
| `whitespace-final`, `staged-whitespace` | 0 | Final owned source and exact two-file staged whitespace: empty output/pass. |
| `self-review`, `stage`, `source-commit` | 0 | Original20 DB test function ASTs unchanged; other8 Task4 files unchanged; exact two-file staging and normal source-only commit succeeded. |

The complete40-case run preceded formatting of only the new test range and a failure-path cleanup refinement in that new test. Production bytes are identical across covering/final/commit receipts; existing20 test functions are AST-identical to FIX_BASE. Final3-case run qualifies the changed test bytes. No broader passing suites were repeated. The worker always receives release and is joined; its operation-owned DB connection and peer are retired in cleanup paths. Root plan/Backlog edits remain dirty and unstaged; index is empty and both owned files are clean.

### Exact covering commands

The following are the actual executed commands, including their disposable profile and JUnit arguments; full outputs are in the linked logs. Exit/output summaries are above.

[fix1-red-confirmed log](task-4-fix1-red-confirmed.log), [receipt](task-4-fix1-red-confirmed-receipt.json), exit `1`:

```sh
TLDW_TEST_CONFIG_ROOT=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-red-confirmed-profile-c3wlmpxx /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/DB/test_automatic_chat_starts.py -k late_recovery_cleanup --basetemp=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-red-confirmed-profile-c3wlmpxx/pytest --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-red-confirmed.xml
```

[fix1-covering-green log](task-4-fix1-covering-green.log), [receipt](task-4-fix1-covering-green-receipt.json), exit `0`:

```sh
TLDW_TEST_CONFIG_ROOT=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-covering-green-profile-2gv4vnj_ /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/DB/test_automatic_chat_starts.py Tests/Chat/test_console_chat_start.py::test_failed_review_settlement_blocks_siblings_across_ledger_handles Tests/Chat/test_console_chat_start.py::test_native_start_child_and_wake_share_original_allowance Tests/Agents/test_automatic_child_scope.py::test_three_generations_share_six_actual_launches_and_manual_is_uncharged Tests/Chat/test_console_durable_commit_offload.py::test_cancel_during_offloaded_commit_leaves_consistent_state --basetemp=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-covering-green-profile-2gv4vnj_/pytest --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-covering-green.xml
```

[fix1-final-cleanup-green log](task-4-fix1-final-cleanup-green.log), [receipt](task-4-fix1-final-cleanup-green-receipt.json), exit `0`:

```sh
TLDW_TEST_CONFIG_ROOT=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-final-cleanup-green-profile-xjxwfj_q /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/DB/test_automatic_chat_starts.py -k late_recovery_cleanup --basetemp=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-final-cleanup-green-profile-xjxwfj_q/pytest --junitxml=/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook/.superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-final-cleanup-green.xml
```

[fix1-ruff-final log](task-4-fix1-ruff-final.log), [receipt](task-4-fix1-ruff-final-receipt.json), exit `0`:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check --select E9,F63,F7,F82 tldw_chatbook/DB/automatic_work.py Tests/DB/test_automatic_chat_starts.py
```

[fix1-format-commit log](task-4-fix1-format-commit.log), [receipt](task-4-fix1-format-commit-receipt.json), exit `0`:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-fix1-format-baseline.json --head c0b251a71422536a3da2be421dd60f79e2cb230a
```

[fix1-original-base-format-commit log](task-4-fix1-original-base-format-commit.log), [receipt](task-4-fix1-original-base-format-commit-receipt.json), exit `0`:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python scripts/terminal_qualification/format_ratchet.py verify --baseline .superpowers/sdd/2026-10-03-console-pr2995-review-and-merge/task-4-format-baseline-final.json --head c0b251a71422536a3da2be421dd60f79e2cb230a
```

[fix1-whitespace-final log](task-4-fix1-whitespace-final.log), [receipt](task-4-fix1-whitespace-final-receipt.json), exit `0`:

```sh
git -c gc.auto=0 diff --check -- tldw_chatbook/DB/automatic_work.py Tests/DB/test_automatic_chat_starts.py
```

### Self-review and remaining limits

Reviewed transaction/registry ordering, exact entry identity comparisons, current-owner retention, rollback control flow, lock ordering and all changed source hunks. No registry lock is held across SQLite I/O, and cleanup cannot retire records created/refreshed after its capture. Existing native commit custody, no-await final cutoff, accepted charging, privacy logging/status vocabulary and file/memory store identity mechanisms are untouched. Independent review should judge I1 on this exact commit.

Existing concerns remain unchanged: cold connection/WAL setup and FULL fsync are synchronous; unknown-root uncertainty is conservatively owner-wide until verified reconciliation or replacement-owner recovery; the runtime restriction is process-local. Final combined startup/import ratchets, Task5 ownership repair and external publication/merge remain controller-owned. No new unresolved implementation finding is known after this fix-round qualification.
