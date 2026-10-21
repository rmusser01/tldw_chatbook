# Workspace Project Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer natural project questions using admitted workspace files while preserving saved Library choices and accurate source attribution.

**Architecture:** Extend the existing capability-aware workspace note and give Console mixed-source evidence its own formatter. Preserve dedicated Library grounding, file authority, and the existing retrieval preparation state machine. This deliverable has no indexing dependency or database migration.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, existing agent/file-tool runtime, pytest and Ruff.

**Spec:** [Approved design](../specs/2026-10-20-workspace-aware-project-context-design.md), sections Source selection through Grounding and citations; acceptance items 1-4 and 12.

**Reference:** [Pinned Codex CLI source inspection](../specs/2026-10-20-codex-cli-project-context-reference.md).

**ADR required:** yes, existing accepted decision.
**ADR path:** `backlog/decisions/240-workspace-project-context-and-folder-bound-index.md`.
**Reason:** Implements the approved source-selection/grounding contract; no additional architectural decision is introduced in this delivery.

## Global Constraints

- Preserve every saved Never/Automatic and Blocked/Allowed value and the conversation/workspace scope intersection.
- Build guidance from the owning run's final advertised read tools and admitted roots, never the visible/global active workspace.
- Keep `fs_*` root-relative addressing distinct from built-in private-scratch addressing.
- Preserve project-instruction trust, activation, file permissions, exclusions, and read-only behavior.
- Do not create Library items, index folders, scan on attach, add an intent router, or require embeddings.
- Dedicated Library answers retain supplied-evidence-only grounding and `[S#]` marker validation.
- Use targeted tests only; full-suite execution requires explicit user opt-in.
- Before executing each task, create/read its corresponding Backlog task, move it In Progress, and add its Implementation Plan with the ADR check. G1-G3 below are plan-local labels, not allocated Backlog IDs.

## Review Focus

- A user-customized agent prompt must not remove the capability note: G1 tests the actual first-request plan.
- A named workspace can advertise Git but no read-capable file tools: G1 tests that this does not promise file inspection.
- A Library handoff or retry can reintroduce strict-only wording: G2 tests each real consumer.
- A mixed answer can cite a real file while inventing `[S999]`: G2 preserves the unknown-marker failure.
- A user switches tabs during preparation: G3 verifies the original session retains its own files and Library settings.

## File structure

| Files | Responsibility |
| --- | --- |
| `tldw_chatbook/Tools/workspace_file_roots.py` | Conditional exploration guidance and existing root note. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Supply final offered read-tool capabilities to that note. |
| `tldw_chatbook/Chat/answer_citations.py` | Console mixed-source formatter alongside unchanged strict formatter. |
| `tldw_chatbook/Chat/console_chat_controller.py`, `chat_handoff_models.py` | Use the Console formatter for automatic evidence, retry, and Console handoff. |
| Existing targeted tests plus `Tests/Chat/test_console_workspace_guidance.py` | Actual request payload, policy-preservation, and protocol coverage. |

## G1 Capability aware workspace guidance

**Files:** Modify `tldw_chatbook/Tools/workspace_file_roots.py` and `tldw_chatbook/Chat/console_agent_bridge.py`; extend `Tests/Chat/test_console_workspace_folder_note_contract.py` and `Tests/Tools/test_workspace_file_roots.py`.

**Interfaces:** Keep `workspace_context_note(...) -> str` backward compatible; add keyword-only `workspace_read_tools: Iterable[str] = ()`. The bridge supplies the intersection of the final `allowed_tools` and `{"fs_list", "fs_glob", "fs_grep", "fs_read"}`. Existing root aliases remain filtered through `path_tool_aliases`. No filesystem I/O is added while rendering guidance.

- [ ] **Write the regression tests.** Reuse the real `_workspace_run`, `_local_provider`, and `_plan` helpers in the existing contract file. `test_first_request_guides_project_exploration` pins `assert "before saying project information is unavailable" in note` and the actual admitted alias; that alias still succeeds through a real `fs_list`. `test_guidance_requires_workspace_read_tools` covers empty and Git-only tool lists with the same sentence absent. `test_custom_operating_prompt_keeps_workspace_guidance` uses a customized operating prompt in the real first-request builder. Preserve existing tests for multiple roots, selection narrowing, remote roots, no tools, and scratch wording. Normalize line wrapping before sentence assertions.
- [ ] **Run the new node IDs before editing production code:** `python -m pytest Tests/Chat/test_console_workspace_folder_note_contract.py -q`. Expect the new guidance/capability assertions to fail; distinguish any inherited private-profile failure from that intended red.
- [ ] **Implement the note and bridge argument.** Use the exact behavioral paragraph in the spec. Condition it on an actually reachable readable root and offered workspace read tools. For follow-ups, say to inspect relevant evidence as needed, not to rescan every turn. Keep absence/unavailable/SSH messages truthful. Do not change `agents.console_agent_operating` globally or alter provider grants.
- [ ] **Verify:** `python -m pytest Tests/Chat/test_console_workspace_folder_note_contract.py Tests/Tools/test_workspace_file_roots.py Tests/Agents/test_workspace_context_note_prompt.py -q`. Expect all selected tests to pass; run Ruff on only these changed Python files and `git diff --check`.
- [ ] **Commit only this task's files:** `git commit -m "feat(console): guide project questions toward admitted workspace files"` after staging exact paths. Add evidence and the ADR link to its Backlog notes before closing the task.

## G2 Console mixed evidence without weakening Library answers

**Files:** Modify `tldw_chatbook/Chat/answer_citations.py`, `tldw_chatbook/Chat/console_chat_controller.py`, and `tldw_chatbook/Chat/chat_handoff_models.py`. Extend `Tests/Chat/test_answer_citations.py`; create `Tests/Chat/test_console_workspace_guidance.py`. Keep `tldw_chatbook/Library/library_rag_answer_service.py` on the existing strict formatter and qualify it through its tests.

**Interfaces:** Add `format_console_evidence_for_cited_answer(value: Any) -> str`. It shares bundle/reference rendering with `format_evidence_for_cited_answer(value: Any) -> str` but supplies the Console's mixed-source instructions. No validation payload or persisted citation schema changes. `ChatHandoffPayload.model_context_block()` uses the Console formatter for its `evidence_bundle`.

- [ ] **Write formatter and consumer regressions.** `test_console_evidence_allows_live_file_grounding` asserts that the Console output names workspace file evidence as valid and restricts `[S#]` labels to their supplied references. `test_library_evidence_stays_strict` pins `assert "Use only available evidence references below to ground factual claims." in strict`. `test_console_invalid_bundle_has_no_grounding_block` pins `assert format_console_evidence_for_cited_answer({}) == ""`. Add `test_retry_and_handoff_use_console_grounding` through the actual retry payload and `ChatHandoffPayload.model_context_block()`.
- [ ] **Run:** `python -m pytest Tests/Chat/test_answer_citations.py Tests/Chat/test_console_workspace_guidance.py -q`. Expect the new helper/consumer assertions to fail before implementation.
- [ ] **Implement the separate formatter and wire all Console consumers.** Replace the two controller calls currently used for automatic preparation and dispatch retry; replace the handoff call. Preserve canonical RAG capture text and explicit-stage leases. Keep actual-marker validation strict, including a file-supported answer containing `[S999]`. Interpret uncited status as a fact about the supplied Library bundle; do not relabel it as failure of the entire mixed answer or invent a file citation-validation result.
- [ ] **Verify:** `python -m pytest Tests/Chat/test_answer_citations.py Tests/Chat/test_console_workspace_guidance.py Tests/Chat/test_console_staged_evidence_dispatch_pin.py Tests/Library/test_library_rag_answer_service.py -q`. Inspect exact initial/retry/handoff provider messages, not only a helper's return. Run scoped Ruff and `git diff --check`.
- [ ] **Commit:** `git commit -m "fix(console): allow workspace grounding alongside Library evidence"` with only the task's staged files and update its Backlog notes.

## G3 Qualify natural questions and policy preservation

**Files:** Extend `Tests/Chat/test_console_workspace_guidance.py` and `Tests/Chat/test_console_library_runtime_policy.py`; update `Docs/Architecture/console-file-authority.md`, `Docs/Architecture/console.md`, and `Docs/User_Guide/console/context-and-rag.md`. Add live receipts under `Docs/superpowers/qa/2026-10-20-workspace-project-context/guidance/` during execution.

**Interfaces:** Consume G1's real first-request plan and G2's Console formatter. Exercise existing `ConsoleChatController.submit_draft` and actual dispatch/retry machinery; do not add a parallel routing service.

- [ ] **Write integrated tests.** `test_project_turn_preserves_all_library_policy_combinations` parametrizes all four policy combinations and compares before/after durable policy rows. `test_pending_project_turn_stays_with_owning_tab` holds preparation while switching the viewed session, then asserts the first payload's alias and Library scope still belong to its original session. Cover both native and fenced protocol preparation. Scripted tool calls test transport, not the model's decision to explore.
- [ ] **Run the new node IDs and verify their assertions exercise the production payload.** A missing fixture or setup failure is not the intended regression. The policy guards may already pass; retain them as protection without manufacturing a red result.
- [ ] **Run a real-model Console journey in an isolated profile.** Bind an unindexed project containing a unique README/manifest marker. Ask "summarize this project" with no tool instructions; repeat with unrelated staged Library evidence and saved Automatic, then with two folders, a denied read, and an explicit mixed Library/project question. Record selected model/protocol, actual file calls, final facts, Library values, and the full Console state. A manual `fs_list` success alone is insufficient.
- [ ] **Run the affected selection:** `python -m pytest Tests/Chat/test_console_workspace_guidance.py Tests/Chat/test_console_workspace_folder_note_contract.py Tests/Chat/test_answer_citations.py Tests/Chat/test_console_library_runtime_policy.py Tests/Chat/test_console_staged_evidence_dispatch_pin.py -q`. Update docs with the verified behavior, scoped lint, and `git diff --check`. Do not claim a live scenario that could not run.
- [ ] **Commit:** `git commit -m "test(console): qualify workspace guidance with preserved Library policy"`. Check task ACs, notes, docs, and ADR linkage before marking its Backlog task Done.

## Delivery gate and handoff

This plan ships when natural project questions work in the qualified real-model
journey, all modified paths pass their targeted checks, and saved Library values
remain unchanged. It does not wait for an index implementation.

Review this plan before execution and choose Native or Subagent-driven execution.
Native is recommended for this small, closely connected change; it avoids moving
the formatter and request-plan seams between implementers. Index delivery uses
its separate reviewed plan and does not start automatically when G3 finishes.
