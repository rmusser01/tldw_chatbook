# Final whole-branch review — completed findings batch

Reviewer:/root/final_branch_review. Range:9ba96ebb626dd010f14d093b0e8d40f37c715d32..459e666970ef9e7aa4e148705cddafb9621a72d1. Ready to merge:With fixes. Spec compliance:Issues found. Three Important and one Minor; no Critical. All four are required in the single final fix wave. The reviewer is preserving the exact full report/probe commands in adjacent final-branch-review-full.md; this file preserves the completed findings and outputs for immediate dispatch.

## Important 1. Reject archived destinations at creation and launch boundaries

Locations:Chat/console_chat_controller.py:16940, Chat/chat_persistence_service.py:2367, Chat/console_chat_start.py:403.

validate_workspace_target() only checks whether the workspace exists. Archived records still satisfy that check. The approved-token revalidation checks source ownership and source workspace identity, but does not reject an archived destination. Native acceptance also lacks a destination-availability check.

A real SQLite probe prepared and approved a same-workspace request, archived that workspace, then executed the approved request. Exit0:
archived_after_approval {"ok": true, "launch_status": "draft", "destination_archived": true, "saved_scope": "workspace", "saved_workspace": "archivable"}

Required fix: Validate the exact captured destination as available and unarchived immediately before creation and again before native acceptance. Preserve the approved destination; do not retarget. Add focused cases for archival during approval and during asynchronous start preparation.

## Important 2. Disabling the runtime during preparation does not prevent acceptance or dispatch

Locations:Chat/console_chat_start.py:202 and403, Chat/console_chat_controller.py:24965.

start() checks _agent_runtime_enabled before asynchronous preparation. accept() never checks it again. Dispatch subsequently uses the earlier turn_context.tool_configuration value.

The focused barrier probe paused _resolve_for_send_bounded readiness with asyncio Events, called actual controller.update_agent_runtime(enabled=False,bridge=controller._agent_bridge), then resumed preparation. Exit0:
runtime_disabled_before_acceptance {"status": "started", "reason": null, "runtime_enabled": false, "provider_calls": 1, "handoff_state": "consumed", "generation_used": 1}

The live gate was disabled before the ownership cutoff, yet the draft was consumed and provider work ran. This violates the requirement to recheck current policies at acceptance and preserve the draft when the destination runtime is disabled.

Required fix: Recheck current runtime/start eligibility at the acceptance boundary. A disabled gate should refuse before acceptance, preserve the draft, and settle the uncommitted reservation. Cover the configuration change with a readiness-barrier regression.

## Important 3. The approval card does not disclose remembered-body authority or identify an instructions override

Locations:Widgets/Chat_Widgets/chat_create_confirm_card.py:48,128,155.

The card says Allow for this session without explaining that the grant covers later requests in the same mode and destination, including later supplied opening prompts and instructions. It renders explicit instructions as generic System prompt, without naming the override. Backend grant scoping is sound, but with mode=start remembering permits later supplied bodies to run without another card, making this omission material.

Required fix: State the exact remembered scope and coverage of later supplied bodies. Label nonblank tool instructions as an explicit override, while retaining the complete body and markup-disabled rendering. Verify the actual card text.

## Minor 4. Unavailable-Persona fallback notices are discarded

Locations:Chat/console_chat_controller.py:16902,17082.

The canonical resolver returns startup.notice, but creation omits it from the approval payload and restored session. Explicitly supplying the resolved assistant bypasses the ordinary notice-producing defaults path. The corrected probe registered store._on_assistant_default_notice and inspected target.assistant_default_notice. Exit0:
degraded_default {"resolver_notice": "Workspace default Persona unavailable (persona_deleted). Started with None.", "approval_has_notice": false, "notice_callbacks": [], "target_assistant_default_notice": "", "assistant_id": "console"}

Required fix: Carry the canonical notice through the approval/creation presentation path and ordinary session notice mechanism. Add a missing-Persona case.

## Review evidence and dispositions

Underlying acceptance, original-root accounting/local parents, physical drain/manual withdrawal, machine provenance/profile exclusions, revision-fenced drafts and saved-versus-live outcome architecture were assessed as coherent. Full package inspected in passes; diff body matched git diff -U10 exactly528650bytes; all29 preserved-log manifest sizes/SHA verified; actual163-pass and static logs plus real PTY/receipts inspected. Only isolated real-owner deterministic probes ran; no supplied suites rerun, helpers spawned or checkout/index/HEAD mutations.

Inherited escape/temp cleanup warnings and guide conflict markers:Minor separate debt. FD growth:unresolved test-resource debt, not an established feature regression; per-test GC does not prove cleanup. Test-owner consolidation accepted after real native-child-wake coverage. Two reasoning_replay lambda failures proven inherited at exact FIX_BASE and unchanged overall-base blobs; remain failing, no all-green related-group claim. CtrlQ modal routing is unchanged non-priority app binding plus installed Textual ModalScreen behavior; separate usability follow-up, no baseline live replay. Git GC warnings are housekeeping; no prune. Disk-full run invalid infrastructure evidence superseded by passing reruns. PTY rather than native screenshot, streaming unknown usage conservatively retained, nonstream confirmed settlement separate, clean restarts not OS power-loss proof.

Recommendation:Use one final fix wave for all four findings and scoped rereview. No architecture redesign required.
