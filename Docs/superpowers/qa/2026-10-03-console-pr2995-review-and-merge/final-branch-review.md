# Final whole-branch review — PR 2995

## Scope

- Base: `0001eba40419859ce39ed4952f0f8df7b40639bd`
- Head: `bc7c2dde9febfd93cbc3bc115c8cbc61c3ee752b`
- Requirements: ADR-211; `Docs/superpowers/specs/2026-10-02-console-chat-destinations-and-starts-design.md`; current review-and-merge plan; TASK-34215.2.
- Read-only review in manageable passes. Reviewed production changes across the coordinator/controller, approval bridge, shared allowance ledger, persistence and metadata, both schema migrations and recovery catalogs, runtime ownership and UI projection. Reviewed changed test owners, with detailed tracing of native acceptance/cancellation, child confirmation, shared descendant/wake budget, mounted draft consumption, and migration/recovery controls. Read the user-guide delta, current plan/task, Task 1 report/review, and historical QA README.
- `whole-branch-manifest.json` identifies 95 non-QA paths and 1,524 historical QA paths, with zero historical QA changes since the original PR. Historical logs were not crawled or treated as fresh-head executions.
- No tests, app launches, dependency installs, branch/index mutations or source edits were performed. This report is the sole write.

## Strengths

- Approval authority is captured before mutation and bound to source incarnation, resolved destination and mode. Model-supplied lineage is discarded. Both the bridge memo and controller grants retain genuine-child confirmation; a child cannot use a primary grant to acquire start or casual-destination authority. Revoked rounds remove prepared tokens and keep exact request IDs.
- Fresh creation resolves destination defaults, persists the generation snapshot and identity, and restores without activating the target. Explicit instructions retain plain/custom identity. The source's draft, settings, folder bindings and staged inputs are not copied into the new chat.
- Acceptance has two explicit durable boundaries. The ledger transition retains the automatic charge; the conversation transaction binds the exact attempt to the request/checkpoint and consumes only the pending draft revision. Dispatch authority is latched only after the matching receipt. Duplicate receipt handling compares content, provenance, message identities and frozen context after common validation.
- Direct immutable allowance-root membership preserves one-conversation run ownership while sharing counters, uncertainty, limits and elapsed deadline. Descendant token settlement and recovery update the canonical root. New starts and wakes share physical automatic-primary slots and the manual capacity reserve.
- The new origin uses literal slash/@ input, durable machine provenance and configured capture/retrieval without trusted human-profile mutation authority. Explicit Retry retains provenance but begins manual work. Native starts remain outside queue Stop-parent eligibility.
- AgentRuns v21→v22 and ChaChaNotes v75→v76 retain predecessor ownership/data, existing routing/worktree schema, hook continuation receipts and checkpoint indexes. Recovery catalogs cover predecessor and current schema variants without accepting mismatched version stamps. The fixed migration owner grants are limited to the named schema objects.
- The mounted navigation repair compares widget generation, authored edit serial and draft segments while preserving session/incarnation/revision fencing. Cursor/selection movement no longer retains the consumed original, while later same-text authored input remains protected.

## Issues

### Critical (Must Fix)

None found.

### Important (Should Fix)

None found.

### Minor (Nice to Have)

1. **[P3] Remove grant mutation before close-ticket validation.**
   - File: `tldw_chatbook/Chat/console_chat_controller.py:15071-15072`.
   - The added `pop(ticket.session_id, None)` runs before checking the ticket identity/generation. If a stale callback is rejected at lines 15075-15078, it has already removed a still-live session's remembered creation approval. The existing identical removal at line 15082 already performs the cleanup after validation.
   - Impact: unnecessary new approval prompts after a rejected stale close; no permission escalation.
   - Fix: remove the new prevalidation pop/comment and retain the existing postvalidation cleanup. Preserve the stale-ticket refusal behavior.
   - The controller identified existing focused controls in `Tests/Chat/test_console_runtime_shutdown.py`: `test_rejected_session_close_finalization_preserves_chat_create_grants` and `test_session_close_clears_grants_only_after_valid_ticket`. The controller owns reproduction/repair; this reviewer did not run them.

2. **[P3] Correct the guide's sub-agent creation restriction.**
   - File: `Docs/User_Guide/console/agent-runs-and-tools.md:1993-1994`.
   - The edited introduction says sub-agents cannot use either creation tool. Current behavior explicitly supports child `fork_chat` and same-workspace draft `new_chat`; only new destination selection and bounded starts are primary-only. `prepare_agent_chat_create` and the genuine-child integration controls enforce that distinction.
   - Impact: the guide contradicts the supported workflow and its approval behavior.
   - Fix: state that child creation retains the existing fork/same-workspace-draft contract and each child request needs fresh confirmation; reserve casual selection and start mode for primary agents.

## Evidence and qualification

- Current follow-up evidence: `task-1-report.md`, `task-1-review.md`, `docs-dev-rebase-receipt.json`, `baseline21.json/log`, `derived-checks.json`, `latest-dev-backlog-checks.json`, and `shared-child-closure-green.json/log` in this SDD directory.
- Baseline log records 21 passes. Derived checks record 11 exit-zero commands; latest-dev Backlog checks record two exit-zero commands. The separate genuine-child shared-closure receipt records the exact integration node at this reviewed head and exit zero. These scopes overlap and are not summed.
- Task 1 records 78 runtime-owner passes, one unchanged strict XFAIL and five warnings; 181 affected start/compaction/RAG/trace passes; focused successor controls qualify repaired fixture failures separately. Its RED/GREEN census is 1034→1033 against the unchanged 1033 limit. Fatal Ruff, formatter ratchets and whitespace receipts are scoped checks, not a repository-wide clean bill.
- The docs-only rebase receipt records unchanged tested source/test blob identity. Historical QA README identifies prior source revisions, real-provider/PTY coverage and its limits. Prior live results are not relabeled as current-head live qualification.
- Resource cleanup remains partly unqualified: retained test-process descriptor-growth and timer/escape warnings do not establish a localized production leak, and this review did not infer that they were fixed. The existing timer-overlap XFAIL remains unchanged. The isolated skill snapshot test does not establish the combined skill-await plus actual hook-refusal mounted flow; the report discloses that limit and separate real hook unchanged/stale controls exist.
- Named outside-hunk checks: default assistant resolution and app-owned fresh-default provider in `console_runtime.py`; origin-dependent submission branches and prepared RAG lease path in `console_chat_controller.py`; canonical root clock/admission/recovery methods in `DB/automatic_work.py`; session close's pre-existing ticket validation and postvalidation grant removal; checkpoint-reference migration search. These were inspected only to resolve concrete boundary risks.

## Recommendations

Make the two small corrections above, retain the recorded evidence limits, and finish current-head GitHub/Qodo checks before merging. Do not treat the historical QA archive or this code-review verdict as an external merge-gate result.

## Assessment

**Ready to merge? Yes on code-review grounds, with the two minor corrections recommended. External merge eligibility is still pending.**

No Critical or Important defect was established in the reviewed branch. The approval, custody, machine provenance, original allowance and migration boundaries are consistent with ADR-211, and the navigation-only draft resurrection fix is supported by focused mounted evidence. Current GitHub checks, Qodo completion and exact-head publication remain controller-owned gates.
