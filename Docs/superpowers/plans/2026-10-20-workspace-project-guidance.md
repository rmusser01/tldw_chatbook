# Workspace Project Guidance Implementation Plan

Current governance: [ADR-241](../../../backlog/decisions/241-workspace-project-context-and-folder-bound-index.md). Renumbered from project ADR-240 during 2026-10-22 dev integration; historical rulings and delivery notes below retain their original source-era labels.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Answer natural project questions using admitted workspace files while preserving saved Library choices and accurate source attribution.

**Architecture:** Extend the existing capability-aware workspace note, give automatic Console evidence its own formatter, and let canonical Console citation repair accept mixed answers with no Library markers. Preserve dedicated Library grounding, actual-marker validation, file authority, and the existing retrieval preparation state machine. This deliverable has no indexing dependency or database migration.

**Tech Stack:** Python >=3.12, Textual >=8.0.0,<9, existing agent/file-tool runtime, pytest and Ruff.

**Spec:** [Approved design](../specs/2026-10-20-workspace-aware-project-context-design.md), sections Source selection through Grounding and citations; acceptance items 1-4 and 12.

**Reference:** [Pinned Codex CLI source inspection](../specs/2026-10-20-codex-cli-project-context-reference.md).

**ADR required:** yes, existing accepted decision.
**ADR path:** `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md`.
**Reason:** Implements the approved source-selection/grounding contract; no additional architectural decision is introduced in this delivery.

## Global Constraints

- Preserve every saved Never/Automatic and Blocked/Allowed value and the conversation/workspace scope intersection.
- Build guidance from the owning run's final advertised read tools and admitted roots, never the visible/global active workspace.
- Keep `fs_*` root-relative addressing distinct from built-in private-scratch addressing.
- Preserve project-instruction trust, activation, file permissions, exclusions, and read-only behavior.
- Do not create Library items, index folders, scan on attach, add an intent router, or require embeddings.
- Dedicated Library answers retain supplied-evidence-only grounding and `[S#]` marker validation.
- A mixed Console answer may contain no Library marker; repair must not invent attribution merely because a Library bundle was staged.
- Use targeted tests only; full-suite execution requires explicit user opt-in.
- Before executing each task, create/read its corresponding Backlog task, move it In Progress, and add its Implementation Plan with the ADR check. G1-G3 below are plan-local labels, not allocated Backlog IDs.

## Review Focus

- A user-customized agent prompt must not remove the capability note: G1 tests the actual first-request plan.
- A named workspace can advertise Git but no read-capable file tools: G1 tests that this does not promise file inspection.
- A canonical Console capture can force missing-marker repair despite correct file evidence: G2 tests actual capture and direct/agent completion.
- A Library handoff or retry can reintroduce conflicting behavior: G2 tests real staging and dispatch, not the unused `model_context_block()` helper.
- Automatic goals narrow tools after the current note is rendered: G1 moves the note after that narrowing and tests the actual final request.
- A mixed answer can cite a real file while inventing `[S999]`: G2 preserves the unknown-marker failure.
- A user switches tabs during preparation: G3 verifies the original session retains its own files and Library settings.

## File structure

| Files | Responsibility |
| --- | --- |
| `tldw_chatbook/Tools/workspace_file_roots.py` | Conditional exploration guidance and existing root note. |
| `tldw_chatbook/Chat/console_agent_bridge.py` | Supply final offered read-tool capabilities to that note. |
| `tldw_chatbook/Chat/answer_citations.py` | Console mixed-source formatter alongside unchanged strict formatter. |
| `tldw_chatbook/Chat/console_chat_controller.py` | Use the Console formatter for automatic evidence and retry. |
| `tldw_chatbook/Chat/citation_repair.py`, `Event_Handlers/Chat_Events/chat_rag_events.py` | Runtime-only missing-marker policy on canonical Console capture; actual markers remain strict. |
| Existing `UI/Screens/chat_screen.py` handoff staging and canonical capture | Integration test path; no unused handoff-helper change is required. |
| Existing targeted tests plus `Tests/Chat/test_console_workspace_guidance.py` | Actual request payload, policy-preservation, and protocol coverage. |

## G1 Capability aware workspace guidance

**Files:** Modify `tldw_chatbook/Tools/workspace_file_roots.py` and `tldw_chatbook/Chat/console_agent_bridge.py`; extend `Tests/Chat/test_console_workspace_folder_note_contract.py` and `Tests/Tools/test_workspace_file_roots.py`.

**Interfaces:** Keep `workspace_context_note(...) -> str` backward compatible; add keyword-only `workspace_read_tools: Iterable[str] = ()`. The bridge supplies the intersection of the final `allowed_tools` and `{"fs_list", "fs_glob", "fs_grep", "fs_read"}`, after both plugin ceilings and `automatic.goal.narrow_config`. Build/recompute the note there and put it on the final config before schema/request construction. Existing root aliases remain filtered through `path_tool_aliases`. No filesystem I/O is added while rendering guidance.

- [x] **Write the regression tests.** Reuse the real `_workspace_run`, `_local_provider`, and `_plan` helpers in the existing contract file. `test_first_request_guides_project_exploration` pins `assert "before saying project information is unavailable" in note` and the actual admitted alias; that alias still succeeds through a real `fs_list`. `test_guidance_requires_workspace_read_tools` covers empty and Git-only tool lists with the same sentence absent. Add an automatic goal whose captured scope removes all workspace readers and assert the final request offers no exploration promise. `test_custom_operating_prompt_keeps_workspace_guidance` uses a customized operating prompt in the real first-request builder. Preserve existing tests for multiple roots, selection narrowing, remote roots, no tools, and scratch wording. Normalize line wrapping before sentence assertions.
- [x] **Run the new node IDs before editing production code:** `python -m pytest Tests/Chat/test_console_workspace_folder_note_contract.py -q`. Expect the new guidance/capability assertions to fail; distinguish any inherited private-profile failure from that intended red.
- [x] **Implement the note and bridge argument.** Use the exact behavioral paragraph in the spec. Condition it on an actually reachable readable root and offered workspace read tools. For follow-ups, say to inspect relevant evidence as needed, not to rescan every turn. Keep absence/unavailable/SSH messages truthful. Do not change `agents.console_agent_operating` globally or alter provider grants.
- [x] **Verify:** `python -m pytest Tests/Chat/test_console_workspace_folder_note_contract.py Tests/Tools/test_workspace_file_roots.py Tests/Agents/test_workspace_context_note_prompt.py -q`. Expect all selected tests to pass; run Ruff on only these changed Python files and `git diff --check`.
- [x] **Commit only this task's files:** `git commit -m "feat(console): guide project questions toward admitted workspace files"` after staging exact paths. Add evidence and the ADR link to its Backlog notes before closing the task.

## G2 Console mixed evidence without weakening Library answers

**Files:** Modify `tldw_chatbook/Chat/answer_citations.py`, `tldw_chatbook/Chat/console_chat_controller.py`, `tldw_chatbook/Chat/citation_repair.py`, and `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py`. Extend `Tests/Chat/test_answer_citations.py`, `test_citation_repair.py`, `test_console_local_citation_boundary.py`, `Tests/RAG/test_local_citation_capture.py`, `Tests/UI/test_console_local_citation_capture.py`, and the actual handoff fixture in `Tests/UI/test_console_live_work_handoffs.py`; create `Tests/Chat/test_console_workspace_guidance.py`. Keep `tldw_chatbook/Library/library_rag_answer_service.py` on the existing strict formatter and qualify it through its tests.

**Interfaces:** Add `format_console_evidence_for_cited_answer(value: Any) -> str`. It shares bundle/reference rendering with `format_evidence_for_cited_answer(value: Any) -> str` but supplies the Console's mixed-source instructions. Add runtime-only `require_markers: bool = True` to immutable `CitationRepairContract`, validating an actual Boolean. Add the same keyword-only option to `_repair_contract_for_local_evidence`, and pass False only from `capture_console_staged_evidence_for_chat`; generic local capture retains True. `decide_citation_repair` accepts no markers when optional while rejecting actual unknown/malformed markers. The same contract governs repaired-output selection, so removal of an invalid marker does not force a fabricated replacement. Preserve exact context, contract object identity, ordinals, evidence IDs and prompt snapshots; no serialized/persisted citation schema changes.

- [x] **Write formatter and consumer regressions.** `test_console_evidence_allows_live_file_grounding` asserts that the Console output names workspace file evidence as valid and restricts `[S#]` labels to their supplied references. `test_library_evidence_stays_strict` pins `assert "Use only available evidence references below to ground factual claims." in strict`. `test_console_invalid_bundle_has_no_grounding_block` pins `assert format_console_evidence_for_cited_answer({}) == ""`. Exercise actual initial/retry payloads and `_stage_handoff_as_console_live_work` through staging, canonical capture and provider dispatch. In both direct and agent completion, a file-only answer with unrelated canonical staged Library evidence triggers no repair call and no forced label/failure notice; `[S999]` remains invalid. Test malformed markers, optional repaired deletion of `[S999]`, strict default missing-marker repair, and preserved capture identity/snapshots.
- [x] **Run:** `python -m pytest Tests/Chat/test_answer_citations.py Tests/Chat/test_console_workspace_guidance.py Tests/Chat/test_citation_repair.py Tests/Chat/test_console_local_citation_boundary.py -q`. Expect the new helper/capture/consumer assertions to fail before implementation.
- [x] **Implement the formatter and construction-time repair policy.** Replace the two controller calls used for automatic preparation and dispatch retry. Canonical handoffs already use source-only capture; add any needed Console wording outside exact source blocks, without changing their identity. Do not change the unused `ChatHandoffPayload.model_context_block()` merely to satisfy a helper test. Set optional-marker policy when constructing Console capture, never by cloning it in normalization. Keep actual-marker validation strict, including a file-supported answer containing `[S999]`, and handle the accepted no-marker decision without a repair request. Interpret uncited status as a fact about the supplied Library bundle; do not relabel it as failure of the entire mixed answer or invent a file citation-validation result.
- [x] **Verify:** Run the new/extended G2 files plus `Tests/Chat/test_console_staged_evidence_dispatch_pin.py` and `Tests/Library/test_library_rag_answer_service.py`; include the canonical-capture and mounted handoff files listed above. Inspect exact initial/retry/handoff provider messages and the actual repair-call count, not only helper return values. Run scoped Ruff and `git diff --check`.
- [x] **Commit:** `git commit -m "fix(console): allow workspace grounding alongside Library evidence"` with only the task's staged files and update its Backlog notes.

## G3 Qualify natural questions and policy preservation

**Files:** Extend `Tests/Chat/test_console_workspace_guidance.py` and `Tests/Chat/test_console_library_runtime_policy.py`; update `Docs/Architecture/console-file-authority.md`, `Docs/Architecture/console.md`, and `Docs/User_Guide/console/context-and-rag.md`. Add live receipts under `Docs/superpowers/qa/2026-10-20-workspace-project-context/guidance/` during execution.

**Interfaces:** Consume G1's real first-request plan and G2's Console formatter. Exercise existing `ConsoleChatController.submit_draft` and actual dispatch/retry machinery; do not add a parallel routing service.

- [x] **Write integrated tests.** `test_project_turn_preserves_all_library_policy_combinations` parametrizes all four policy combinations and compares before/after durable policy rows. `test_pending_project_turn_stays_with_owning_tab` holds preparation while switching the viewed session, then asserts the first payload's alias and Library scope still belong to its original session. Cover both native and fenced protocol preparation. Scripted tool calls test transport, not the model's decision to explore.
- [x] **Run the new node IDs and verify their assertions exercise the production payload.** A missing fixture or setup failure is not the intended regression. The policy guards may already pass; retain them as protection without manufacturing a red result.
- [x] **Run a real-model Console journey in an isolated profile.** Bind an unindexed project containing a unique README/manifest marker. Ask "summarize this project" with no tool instructions; repeat with unrelated staged Library evidence and saved Automatic, then with two folders, a denied read, and an explicit mixed Library/project question. Record selected model/protocol, actual file calls, final facts, Library values, and the full Console state. A manual `fs_list` success alone is insufficient.
- [x] **Run the affected selection:** `python -m pytest Tests/Chat/test_console_workspace_guidance.py Tests/Chat/test_console_workspace_folder_note_contract.py Tests/Chat/test_answer_citations.py Tests/Chat/test_console_library_runtime_policy.py Tests/Chat/test_console_staged_evidence_dispatch_pin.py -q`. Update docs with the verified behavior, scoped lint, and `git diff --check`. Do not claim a live scenario that could not run.
- [x] **Commit:** `git commit -m "test(console): qualify workspace guidance with preserved Library policy"`. Check task ACs, notes, docs, and ADR linkage before marking its Backlog task Done.

## Delivery gate and handoff

This plan ships when natural project questions work in the qualified real-model
journey, all modified paths pass their targeted checks, and saved Library values
remain unchanged. It does not wait for an index implementation.

The user selected Subagent-driven execution; implement G1, G2 and G3 in dependency
order with review at each seam. The [pre-implementation review](../qa/2026-10-20-workspace-project-context/plan-review.md)
records corrections to final tool narrowing, canonical capture repair, and real
handoff qualification. Index delivery uses its separate reviewed plan and does
not start automatically when G3 finishes.

## Execution record

Implemented on `codex/workspace-project-guidance` from `bbe609a676` in the managed worktree. G1 (`8e02f7883d`), the necessary native Windows full-identity prerequisite (`6678eecb0d`), and G2 (`278687992c`) passed independent task spec/quality review. G3 is implemented in `196b0d3a8e`, with receipt provenance corrected in `0cfd2977d2`; independent task review and scoped re-review are approved. The final whole-branch review found one canonical citation-instruction gap and two small evidence/documentation corrections. The single correction wave `c2578bca5d` passed scoped final re-review, with all findings addressed and no new breakage. All guidance/prerequisite tasks are Done.

The five-file G3 selection has final passing evidence for all 55 node IDs across targeted runs. Actual normal-app Send completed through the real native-tool provider with the synthetic unindexed project; saved Library rows and the original-profile comparisons remained unchanged. Broader inherited failures and non-pristine warning/lint limits remain recorded in the [qualification record](../qa/2026-10-20-workspace-project-context/guidance/README.md) and its execution notes. No full suite, indexing implementation, push or merge is claimed.

ADR required: yes. ADR path: `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md` (guidance/mixed grounding); the Windows representation correction follows existing `backlog/decisions/101-one-shot-pinned-workspace-tool-execution.md`. No new storage or authority decision.

The final canonical-path correction has intended RED for eight direct/agent plus one mounted request and GREEN for nine narrow plus 102 covering checks (including 65 strict Library cases). One fresh actual-app mixed journey completed with owning Library labels, separate file references and unchanged saved policy/profile comparisons. [Final approval and execution rulings](../qa/2026-10-20-workspace-project-context/guidance/execution-notes.md) retain the known baseline, model and observation limits. Optional indexing remains a separate deliverable.
