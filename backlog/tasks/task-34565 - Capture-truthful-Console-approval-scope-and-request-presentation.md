---
id: TASK-34565
title: Capture truthful Console approval scope and request presentation
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 06:41'
updated_date: '2026-10-06 20:38'
labels: []
dependencies:
  - TASK-34564
documentation:
  - >-
    Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md
  - backlog/decisions/221-console-approval-interaction-and-feedback.md
  - Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Give approval cards accurate action, target, authority and supported scope information for their captured requests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Presentation preserves every captured target and actual call count without relying on truncated arguments or the currently selected workspace.
- [x] #2 Until Chatbook exits explains exact-profile grants and raw-shell grants limited to this chat and cleared by Disarm.
- [x] #3 More options describes Default inheritance and withholds remembered-input choices that repeated-tool stamp handling cannot honor.
- [ ] #4 Presentation metadata remains ephemeral and cannot widen permissions, rewrite captured arguments or change provider stamp ownership.
- [x] #5 Large grouped target previews stay bounded with explicit omissions, while complete captured redacted targets remain available through Details.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Task 2 Capture request presentation from the actual owners

**Backlog:** TASK-34565. **ADR:** ADR-221/032/093. **Consumes:** Task 1's recorded baseline and defined boundaries.

**Files:** Create `tldw_chatbook/Chat/approval_presentation.py`, `Tests/Chat/test_approval_presentation.py`. Modify `Agents/mcp_tool_provider.py:MCPPendingCall/pending_gate_for`, built-in pending construction in `Chat/console_chat_controller.py:build_tool_review_hook`, `Agents/local_tool_provider.py:pending_gate_for`, `Agents/virtual_cli_provider.py:pending_gate_for`, `Agents/raw_shell_tool_provider.py:pending_gate_for`, and `_build_approval_payload`. Extend `Tests/Chat/test_approval_payload_summary.py` and the affected provider tests.

**Interfaces:** Define frozen `ApprovalAuthority(provider_kind: Literal['mcp','builtin','local','virtual_cli','raw_shell','runtime'], profile_id: str | None, profile_label: str, location_label: str, grant_domain: Literal['profile','console_chat','none'], stamp_domain: Literal['call','tool_name','shared_group'], revocation_label: str, inherits_default: bool = False)`. Add optional `presentation_authority: ApprovalAuthority | None = None` and `requires_individual_review: bool = False` to MCPPendingCall as display-only fields; only raw shell currently supplies the explicit-review flag. TYPE_CHECKING annotations must not introduce a startup import.

Produce frozen `ApprovalRowView` with `verdict_key`, `call_count`, `action_label`, full `targets`, `authority`, `legal_decisions`, `requires_review`, `withheld_scope_copy`, and owner-captured `argument_sets` excluded from repr and hot equality. Produce `ApprovalBatchView(round_id, session_id, run_id, revision, rows, call_count, bulk_once, bulk_deny)`. Exact types are str for IDs/copy, int for counts/revision, bool for flags, tuples for rows/choices/targets, and tuples of Mapping[str, object] for argument sets. Implement `capture_approval_view(pending: Sequence[MCPPendingCall], *, round_id: str, session_id: str, run_id: str, revision: int) -> ApprovalBatchView` on the owning worker and `scope_copy(row: ApprovalRowView, decision: str) -> str` for shared copy. Known action formatting uses provider_kind, not a name that an external MCP tool could imitate.

- [ ] **Step 1: Write failing contract tests.** Test `test_raw_shell_scope_is_chat_and_disarm`, `test_default_persistent_scope_discloses_inheritance`, `test_native_same_tool_mixed_scopes_withhold_matching`, `test_shared_verdict_counts_all_calls_and_argument_sets`, `test_unknown_mcp_action_does_not_infer_effects`, `test_policy_reason_does_not_promise_always_ask`, and `test_capture_uses_run_authority_after_active_chat_changes`. Assert original verdict keys, raw arguments and gate options remain unchanged. Run the new file and affected payload/provider tests to establish the intended failures.
- [ ] **Step 2: Populate metadata from owners.** Use captured run configuration and actual profile/binding, not the active screen. MCP/built-in/local scopes retain their real profile key; virtual CLI retains its command keys; raw shell uses console_session_id and Disarm. Runtime-owned lesson rows receive their own none-domain metadata without a new review floor. Known tools use code-owned formatters; unknown MCP tools retain literal identity and parameters. A generic risk-floored reason says **High risk: current policy requires approval for this call.** It does not infer a local read or promise permanent always-ask behavior that a valid wider grant could change. Effects remain separate, producer-owned facts. Update raw-shell scope_notice to match the new label. Missing metadata never invents a broad domain.
- [ ] **Step 3: Build the captured view and effective choices.** Group only by the existing addressable verdict contract. Intersect supported presentation choices with producer options. For multiple independent MCP rows sharing a tool-name stamp, withhold per-row allow_matching and show its explanation; do not change _APPROVAL_SCOPE_RANK or stamp keys. Keep supported shared-verdict matching count-aware. Capture a semantic revision for payload changes so production UI sync avoids walking large argument bodies; legacy callers retain the existing changed-call guard.
- [ ] **Step 4: Verify the trust boundary.** Run the new tests, `Tests/Chat/test_approval_payload_summary.py`, and affected cases in `Tests/Agents/test_mcp_tool_provider.py`, `test_local_tool_provider.py`, `test_builtin_tool_gate.py`, `test_raw_shell_tool_provider.py` and `Tests/Chat/test_console_virtual_cli_approval.py`. Assert metadata cannot alter file authority, reason codes, options, persistence targets or model/provider serialization. Re-run the relevant startup/import-budget control if imports changed.
- [ ] **Step 5: Document and commit.** Record the existing repeated-tool limitation and provider-specific copy in task notes. Qualify all ACs before Done. This commit adds captured presentation and tests, not new permission policy.

ADR required: yes
ADR path: backlog/decisions/221-console-approval-interaction-and-feedback.md
Reason: Existing ADR-221 with ADR-032/093 governs captured permission-owner scope and ephemeral presentation without new policy, schema or stamp ownership.

Execution ordering: Task1 private control and recorder foundations passed independent review; unchanged transport baseline and final speed qualification remain open under the documented controller ruling.

Final review correction: bound initial grouped target formatting; retain lossless captured targets/original arguments and explicit Details continuation; add mounted many-distinct-long-target regression.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented frozen owner-captured approval authority, row and batch presentation, provider-specific scope copy, and the semantic revision payload interface. Counts preserve addressable calls; independent same-tool MCP remembered-input choices are withheld without changing stamps. Captured permission-profile IDs identify the actual gate profile. Raw grants remain limited to this Console chat and clear on Disarm or exit; Default persistent inheritance is explained.

Local original arguments are captured before safe summary formatting through optional captured_arguments, excluded from repr/equality and every known authorization, persistence, model/provider and advisory projection. Legacy summaries stay intact; argument_sets retain nested originals and empty mappings. Settled, nonactionable finishing projections drop stale view/revision and retain legacy changed-call guards. Generic risk copy is High risk: current policy requires approval for this call.

Targeted initial checks: 88 passed. Canonical-input fix: 32 HEAD checks passed, including the exact three production local pending cases on both guarded BASE and HEAD with the supported bootstrap_profile fixture marker. Independent review and scoped re-review found the original-input defect addressed and no new fix breakage. Capture foundations are available to functional UI consumers under the documented ordering ruling.

Full completion remains open. Actual virtual dispatch is unqualified on both BASE and HEAD: the existing Windows native/stat device identity mismatch causes root_pin_failed. No root-identity or recovery repair is part of this task. Full edited-file lint still reports two RemoteRoot F821 errors independently reproduced on BASE; scoped amended-code checks and formatting pass. Existing pytest warning noise is retained. Native/browser visible timing remains unqualified; no speed or full-green claim. Status remains In Progress.

Existing ADR-221 with ADR-032/093 applies. Source changes are approval_presentation.py, pending/provider metadata, controller payload/capture composition and shared reason copy; targeted provider/payload/card tests, private launch support, plan and lessons document the verified behavior and limits. Reports and comparison receipts are preserved in this plan's private SDD directory.

Functional AC1–3 supported by reviewed captured-view/interaction and integration receipts. Invocation/native visual/full DoD qualifications remain open; status In Progress. Final qualification: Docs/superpowers/qa/2026-10-05-console-approval-ux/task-6/README.md; ADR-221.

Final bundled fix4542052fb5..9d8a1bd29a: batch broader scopes behind More options, neutral Review individually route, bounded complete-identifier preview with explicit omissions and complete captured target Details. Actual80x24 Console clipping reproduction repaired with existing batch-only4/2 viewport tokens and three-action toolbar. Interaction25/Details19/ownership25/compact1/journeys4/budget28/CSS5/token1/refinement1 passed. Independent scoped review all3 addressed, no new material issue; hashes verified. Evidence Docs/superpowers/qa/2026-10-05-console-approval-ux/task-6/final-review.md and final-fix/. Native/browser timing/fullmatrix, actual Windows dispatch and broader baseline failures remain open; In Progress, not Done. ADR-221.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Originally TASK-34412 in the reviewed approval checkout. Renumbered to TASK-34565 during PR integration onto current dev because older unrelated TASK-34411/34412 already landed. The six approval records moved together to preserve dependency order; original verification hashes/commit references retain their historical context.
