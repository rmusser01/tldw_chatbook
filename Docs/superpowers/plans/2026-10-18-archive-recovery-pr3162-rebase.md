# PR3162 archive recovery rebase implementation plan

> Execute the approved archive recovery integration in the existing isolated workspace. Use subagent-driven development with disjoint storage/runtime/UI helpers and one root-owned Git index; all helpers leave edits unstaged. One atomic rebase receives one combined integration review.

**Goal:** Rebase PR3162 onto fetched dev9af0c59263dce00fa44be4a74d5d7e78712346bf, resolve its33conflicts without dropping current behavior, pass affected targeted tests and publish with an exact lease.
**Spec:** Docs/superpowers/specs/2026-09-17-agent-archive-recovery-design.md; Docs/superpowers/specs/2026-10-18-versioned-thinking-response-profile-design.md. Read the approved feature version from fa86ea508a where its file is conflicted; current dev architecture and ownership contracts remain binding.
**Architecture:** Carry existing archive providers/coordinator/cards through current runtime admission, native-I/O custody, database migrations and source CSS. No new feature, dependency, permission, schema policy or generic guard relaxation.
**Tech Stack:** Existing Python3.12/Textual8/SQLite and repository pytest/Ruff/CSS builder.
**Task:** TASK-35583.

ADR required: no
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md and backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; preserve current dev's lifecycle/ownership ADRs.
Reason: integration of approved contracts. Renumber only PR-added migrations after current versions and amend exact recovery declarations; do not change an existing dev migration.

## Global constraints

- Work only in /Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook. The original checkout stays read-only; original verified branch ea5c335462 remains untouched.
- Current dev is9af0c59263dce00fa44be4a74d5d7e78712346bf; PR lease is fa86ea508a4c7e607dc954c0d76077bd24d50437. Rebase ours/stage2 is dev; theirs/stage3 is the archive snapshot. Preserve both sets of intended behavior, including auto-merged files.
- Root alone stages, commits, continues rebase and pushes. No worker Git/index/ref mutation, stash, cleanup, whole-file theirs replacement, network/model/app call or full suite. All model requests30of30 consumed; no new inference authorized.
- Archive remains local saved ordinary interactive primary only, Direct/RAG under frozen LibraryAllowed/Persona/destination/kill/lifetime gates. Metadata remains closed/bounded/previously model-observed. Exact one-shot target/scope consent precedes restore; explicit Open is separate and preserves original IDs/history/source draft.
- Preserve current dev callback/connection ownership, received-input admission, Canvas/response-rules/plugin/runtime behavior and deletion of legacy CSS. Keep strict trace validators and older capture meanings.
- No pytest until all unmerged production Python is syntactically resolved. Full scratch/config/keyring/network isolation must precede production imports. Use existing test seams and real SQLite; do not turn a fixture concession into source authority.

## Resolution and validation tasks

- [x] Storage helper: DB/ChaChaNotes_DB.py, DB/Workspace_DB.py, DB/base_db.py, Workspaces/models.py and registry_service.py; related PR-added migration/query/search/operation and DB/Workspaces tests. Establish current schema versions, move only new chronology migrations after them, update exact backup/recovery declarations and focused tests if required. Preserve existing columns/triggers/sync/local-only archive semantics and operation-owned physical close. Read Architecture/database-layer.md and backup-recovery.md; migration-rebase lessons. Report migration/interface numbers immediately. Target archive chronology/query, conversation/workspace archive, registry and relevant exact recovery checks; no unrelated sweep.
- [x] Runtime helper: Agents/agent_runtime.py, agent_service.py, tool_catalog.py, archive_recovery_tool_provider.py; Chat/console_runtime.py, console_agent_bridge.py; archive_recovery runtime/contracts/coordinator/result_store and related Agents/Chat tests except root-owned followup/trace tests. Preserve current Canvas/response-rules/new runtime branches, admission and settlement; add only archive provider and existing dispatch correlation at exact seams. Read Architecture/agent-runtime.md/tool-catalog.md/console.md. Target archive tools, dispatch, search/result-store/coordinator/operations controls and existing changed runtime policy/viewless tests.
- [x] UI helper: UI/Screens/chat_screen.py, settings_screen.py; UI/Console_Modules archive_recovery/archive/prompt_queue/wiring/workspace; Widgets archive cards/console transcript; related archive UI/ownership/workspace regression tests; source CSS and generated bundle. Read design-language.md/component-patterns.md/Architecture/app-shell.md/console.md first. Keep dev's removed screen_agentic_console.tcss removed, compose current source modules, rebuild using build_css.py. Preserve native geometry, token governance, receipt/confirmation/Open behavior and all newer UI handlers. Target affected UI modules and token/build checks after syntax gate.
- [x] Root: console_chat_controller.py, console_trace_service.py, console_semantic_revision.py, console_turn_context.py, capture followup/service/settlement tests; all documentation/ADRs/Backlog and architecture updates. Merge current dev methods with archive admission/rider/reseal and version2 source profile. Preserve exact current owner and immutable predecessors; preserve all current dev clauses/strict validators. Resolve authored doc add/add by retaining current dev content plus feature amendments/history; do not replace large lessons/indexes wholesale.
- [x] Root integration gate: inspect automatic merges and duplicate definitions, parse affected production/tests, ensure no marker remains, stage explicit resolved paths only. Compile/build and run each named targeted cohort with commands/exits/raw output retained under /private/tmp/archive-recovery-pr3162-rebase-20261018; a fresh independent reviewer checks the combined integration changes without rerunning passed tests.
- [ ] Finish: complete rebase, rerun necessary targeted pins against final source after any fixes, record final hashes/results/limits, update TASK35583 and PR body. Force-push only codex/agent-archive-recovery-pr with exact expected fa86ea508a lease; verify current base ancestry, remote head and mergeable status. Keep PR draft unless user requests readiness; no merge/auto-merge/full suite/model call.

## Integration scope amendments

- Current dev moved tool review into console_interrupt_rounds.py; root ports only archive-specific review deltas there and keeps the controller wrapper.
- Received-turn dispatch now bypasses legacy accept_turn; runtime/UI carry the existing opaque user proof through the frozen received intent and checked configuration to the exact current-owner common admission point.
- Workspace uses SQLiteStoreKernel; storage extends its existing optional connection acquisition with the approved operation deadline instead of restoring the old connection owner. Exact recovery catalogs append chat83/workspace9.
- Existing owned teardown completes before re-raising archive shutdown cancellation. Focused regressions verify the new compatibility seams after the global syntax gate.

- Final mounted closed-source completion qualification exposed a legacy test bypassing the current UI close completion refresh. Align that harness with the existing full sync after runtime.close_session and await the exact current projection. A successor session still exists; no adapter or other production UI repair is needed. Verify only the failed closed case and unexecuted final cases.
