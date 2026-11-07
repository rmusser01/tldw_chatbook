Task 1 foundation is complete and committed as `b45ce02976fe90a62d211e6a9387ee2ccc4bba24` (`feat: preserve automatic allowance across agent-created chats`). TASK-33804 remains In Progress for independent review; its five implementation acceptance criteria are checked with recorded evidence. The managed worktree is clean. No original-checkout application files were edited.

All commands ran from `/Users/macbook-dev/.codex/worktrees/console-chat-starts/tldw_chatbook` with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. Mutations, tests, Backlog edits and commits used approved escalated commands for this managed worktree. Every pytest run used a unique owned `--basetemp` under this SDD directory.

The inspected schema heads were AgentRunsDB 18 and ChaChaNotesDB 73. This slice raises only AgentRunsDB to 19. The accepted ADR-211 body and related ADR bodies were preserved. The applicable task, Task 1 brief, approved shared-budget/acceptance spec sections, linked ADRs and relevant testing/Console wiring lessons were read. TDD skill instructions were applied; no subagents or reviewers were spawned.

**Implemented contracts**

- Direct immutable `allowance_root_chain_id` membership, rejecting reparenting, cycles and descendant-to-descendant references on persisted writes. Old roots keep NULL membership, original limits and local conversation ownership.
- `_allowance_chain` resolves the canonical root once per lookup while `_chain` remains local. Snapshot chain/conversation identity stays local; resource totals aggregate root and registered descendants using the specified parameterized membership/reservation join.
- Admission, start/deadline anchors, refusal rollback observations, pause state, unknown/excess token settlement and explicit startup recovery update the root. Runtime-owner replacement remains exempt from stale-callback pauses. Recovery deduplicates roots across wake/start attempts and unresolved reservations, keeps conservative charges and never restores old execution authority after late settlement.
- `AutomaticChatStartAttempt` has the exact approved body-free fields. Native prepare/read/accept/abort/complete APIs use existing FULL-synchronous ledger transactions. Preparation derives persisted source-chain lineage, validates runtime owner/source status and exact duplicate identity, and atomically creates the target member and generation reservation. Acceptance grants only the first caller the cutoff. Abort releases only proven prepared/uncommitted work; complete creates no wake claim/delivery marks.
- Both wake and native preparation query both active-attempt tables, preserving one outstanding target owner across the two kinds.
- `AutomaticWorkContext.attempt_kind` defaults to wake. Both `mark_accepted` and `check` read the corresponding exact-owner attempt. The process-local latch remains unset by ledger acceptance; Task 2's trusted coordinator must call `mark_accepted` only after the conversation receipt also commits.
- Existing `create_run` scope and same-conversation parent guards remain unchanged and were verified through API and direct-SQL refusal tests. No per-run runtime-owner column or cross-conversation parent links were added.

**Files committed**

- `tldw_chatbook/DB/AgentRuns_DB.py`
- `tldw_chatbook/DB/automatic_work.py`
- `tldw_chatbook/Agents/automatic_work_budget.py`
- `tldw_chatbook/Agents/automatic_work_runtime.py`
- `tldw_chatbook/DB/migrations/agent_runs_v18_to_v19_chat_starts.sql`
- `Tests/DB/test_automatic_chat_starts.py`
- `Tests/DB/test_automatic_work_budget.py`
- `Tests/DB/test_automatic_work_deadlines.py`
- `Tests/DB/test_automatic_work_migration.py`
- `Tests/DB/test_automatic_wake_attempts.py`
- `Tests/Chat/test_automatic_work_lineage.py`
- `backlog/tasks/task-33804 - Console-shared-automatic-allowances-and-chat-start-attempts.md`

**RED/GREEN and closure evidence**

The common pytest prefix below is the absolute interpreter above followed by `-m pytest`; all runs include `-q` and their individually owned `--basetemp`.

| Run | Selection | Result | Log in this SDD directory |
| --- | --- | --- | --- |
| Baseline | Four existing DB budget/deadline/migration/wake files plus `Tests/Chat/test_automatic_work_lineage.py` | 66 passed, 18.64s | `task1-baseline.log` |
| Initial RED | `Tests/DB/test_automatic_chat_starts.py` | 9 failed, 2.56s; missing `prepare_chat_start` | `task1-red.log` |
| Initial native GREEN iteration | New native file | 7 passed, 2 failed; recovery no-op checks surfaced runtime-owner check ordering | `task1-green.log` |
| Root/context RED | New-file `test_descendant_unknown_usage_blocks_siblings`, `test_descendant_overage_pauses_siblings_and_late_usage_keeps_review`, plus lineage `test_native_automatic_context_requires_exact_accepted_kind_and_owner` | 4 failed, 2.88s; descendant root state missing and attempt_kind absent | `task1-root-red.log` |
| Root/context GREEN | New native file, deadline file, and the parameterized native-context test | 31 passed, 9.40s | `task1-root-green.log` |
| Migration/index refinement | Migration file, budget index-plan test and parameterized cross-kind target test | 4 passed, 3 failed; predecessor fixture retained the new table and trace matcher named the old reservation query | `task1-migration.log` |
| First closure | All six changed test files | 99 passed, 28.16s | `task1-closure.log` |
| Self-review refinement | Migration file, budget index-plan test and new refused-root-clock test | 5 passed, 1 failed; clock probe used check_active, whose sticky pause guard precedes fresh admission | `task1-final-refinement.log` |
| Corrected clock probe | Deadline `test_refused_descendant_prepare_retains_observed_root_time`, using reserve to observe admission | 1 passed, 2.33s | `task1-clock-refinement.log` |
| Final closure | All six changed test files | **100 passed, 35.07s** | `task1-final-closure.log` |

Final closure command:

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/DB/test_automatic_chat_starts.py Tests/DB/test_automatic_work_budget.py Tests/DB/test_automatic_work_deadlines.py Tests/DB/test_automatic_work_migration.py Tests/DB/test_automatic_wake_attempts.py Tests/Chat/test_automatic_work_lineage.py -q --basetemp=.superpowers/sdd/2026-10-02-console-chat-destinations-and-starts/task1-final-closure-temp
```

The native attempt interfaces and root settlement/context behavior had observed missing-interface/root/context RED evidence before their production implementation. Additional self-review tests pressure-tested implemented transaction/migration contracts; their fixture and probe corrections required no new behavior. Native initial recovery failures were resolved by returning False for non-prepared/non-accepted attempts before current-owner checks; any actual refund/completion mutation still validates current runtime ownership.

**Migration, accounting and authority verification**

Runtime and standalone v18→v19 upgrades preserve roots, limits, committed/settled reservations, run lineage and prepared wake records. Standalone root-membership guards are exercised before AgentRunsDB runtime opening can repair schema omissions. `PRAGMA foreign_key_check` passes on both paths, fresh databases and predecessor fixtures. The v16 fixture now removes the new native table before reconstructing its historical schema.

Real SQLite tests cover shared generation/child/model/token balances; two-target last-generation races; accept-versus-abort races; recursive direct-root starts; failed preparation/acceptance rollback; no refunds after separately committed generation; source absence/terminal status; malformed or changed exact identity; direct SQL immutability and conversation guards; prepared/accepted foreign-owner recovery; late settlement with review retained; current kill switch; wrong attempt kind/owner/chain; and the required explicit acceptance latch.

Injected wall/monotonic clocks verify the original deadline across members and reopen, backwards clock refusal and refused-admission root observations. Query-plan tests capture actual production SQL without sqlite_stat1 and prove searches through `idx_automatic_chains_allowance_root`, `idx_automatic_reservations_chain`, both active target indexes and the existing claim-release index.

**Static checks and self review**

- `python -m ruff check --select E9,F63,F7,F82` on all ten modified/new Python files: exit 0, All checks passed.
- `python -m ruff format --check Tests/DB/test_automatic_chat_starts.py`: exit 0, already formatted.
- Scoped formatter baseline captured with `scripts/terminal_qualification/format_ratchet.py snapshot --base 9ba96ebb626dd010f14d093b0e8d40f37c715d32 --output .../task1-format-baseline.json --path` for each of the nine modified existing Python files. Initial verification found added migration-block formatter debt; only those added layouts were corrected. Uncommitted verify passed, and `verify --baseline .../task1-format-baseline.json --head HEAD` passed against `b45ce02976fe90a62d211e6a9387ee2ccc4bba24` with exit 0. Inherited existing-file debt was preserved, without a whole-file AgentRunsDB sweep.
- `git diff --check`: exit 0. `git log -1 --oneline` confirmed the required commit subject; `git status --short` was empty after commit.
- Backlog CLI updated only TASK-33804's implementation plan/notes/verified ACs. Status remains In Progress for independent review. No new ADR was needed beyond direct implementation of accepted ADR-211.

Self review checked every root mutation path, transaction rollback/cancellation policy, duplicate identities, late owner fencing, index plans, standalone/runtime migration parity and preservation of existing run scope. No unrelated refactor or application/UI route was added. No generalizable new lesson was identified.

**Concerns and next boundary**

No unresolved foundation concern. Live source-session/run ownership remains the controller's responsibility before this trusted ledger API. The native coordinator must use the existing runtime owner, share wake capacity, and commit the conversation receipt before latching automatic context. These are approved Task 2 integration contracts, not dispatch routes in this slice.

Only targeted tests ran. No full suite, real provider execution, killed-process/power-loss test, or exactly-once external-effect guarantee is claimed.
