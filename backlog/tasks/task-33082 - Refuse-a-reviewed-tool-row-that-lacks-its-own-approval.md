---
id: TASK-33082
title: Refuse a reviewed tool row that lacks its own approval
status: Done
assignee:
  - '@claude'
created_date: '2026-09-27 16:40'
updated_date: '2026-10-02 18:06'
labels:
  - agents
  - permissions
  - security
dependencies:
  - TASK-32956
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
During TASK-32956, a probe found a latent fail-open in the Console's tool-batch review path. When the approval card returns a map in which one row is approved and a sibling row is marked `timeout` (or any other value that is neither an approval nor `deny`), the sibling runs as if approved. It runs on the approved row's name-keyed stamp, because the review hook turns every non-deny row into a `proceed` verdict, and the runtime also defaults missing rows to `proceed` (`agent_runtime.py`, `verdicts.get(call.name, "proceed")`).

The shipped approval bridge never produces such a map today: it answers every row, and it fills a missing answer with deny, or with timeout for the whole batch on a deadline. So this cannot currently be reached. But the safety of every mutating tool, `character_save` included, rests on that invariant in a different module rather than on the hook itself. The hook should fail closed on its own.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A reviewed tool call runs only when its own row carries an approval decision. A row that is missing, or marked timeout or any unknown value, is refused, whatever its siblings were answered.
- [x] #2 A test drives a mixed approval map (one row approved, a sibling timed out) through the real review path and shows the sibling does not run.
- [x] #3 Existing approve/deny/timeout behaviour for fully answered batches is unchanged. The failure-name set matches dev's.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In console_chat_controller.py, add one helper that refuses a reviewed row whose own answer is not an approval when a same-name sibling in the batch was approved (the name-keyed stamp would otherwise let it run). Use it in build_mcp_review_hook (MCP + builtin rows) and build_local_review_hook, before the proceed pass.
2. Leave every other case on its existing path: a lone timeout/unknown row keeps a non-approving stamp and the owner's invoke()/check() refuses it with its audit row, exactly as today (AC#3).
3. Tests: mixed map through run_agent_loop + real build_local_review_hook + real LocalToolProvider (sibling never dispatched); unit verdict tests for the MCP/builtin hook (timeout, unknown, missing sibling rows); existing hook tests unchanged.
4. Compare failure names for the touched test files vs dev; preflight.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Both production review hooks (build_tool_review_hook for MCP + built-in rows, build_local_review_hook) now refuse a reviewed call whose OWN row is not an approval whenever a same-name sibling in the batch was approved. The name-keyed stamp keeps the broadest approval of a name, so such a row used to run on its sibling's stamp. One shared helper, _sibling_approval_refusals in console_chat_controller.py, returns the refusals by call id (by name for an id-less row, which fails closed for its siblings too). The copy is the MCP provider's own: TIMEOUT_REFUSAL for timeout, UNRESOLVED_REFUSAL for a missing or unknown answer.

Every other case keeps its existing path, so fully answered batches behave exactly as before (AC#3). With no approved sibling, the name's stamp is non-approving and the owner still refuses at dispatch and writes its audit row (MCP denied-timeout / denied-unresolved, built-in 'requires approval'). A row with no answer and no approved sibling has no stamp, so MCP re-asks with a fresh card and the built-in gate refuses. It cannot run without its own approval. build_mcp_review_hook has no production caller and is unchanged.

Tests:
- test_console_local_review_hook.py: test_sibling_without_its_own_approval_never_dispatches drives run_agent_loop with the real hook and the real LocalToolProvider. On dev the timed-out, unknown or missing sibling is dispatched; here it is not.
- test_console_local_review_hook.py: test_lone_timed_out_row_still_reaches_the_provider_refusal pins the unchanged path.
- test_console_chat_controller.py: test_review_hook_refuses_a_sibling_without_its_own_approval covers the production MCP/built-in builder; it fails on dev for all three answers.
- The two local tests use @pytest.mark.bootstrap_profile: LocalToolProvider's [tools] config read trips the per-test sandbox's config admission (RecoveryRequired raw_source_selection_changed, ADR-126). That trip is why most of that file already fails locally on dev.

Verification: failure names across all 18 files that drive these hooks match dev, apart from test_console_headless_approval.py::test_reprojection_never_renders_an_exhausted_head. Its params flip between runs on both trees (3 reruns each). Preflight passes and ruff counts are unchanged. console_chat_controller.py grows by 54 lines on a row already about 1,190 over budget on dev.

Qodo round 1 (PR #2908), three fixes:
- Audit: a hook-refused sibling is never dispatched, so its owner never recorded the outcome. New record_hook_refusal(name, timed_out=...) seams on MCPToolProvider and LocalToolProvider write the decision dispatch would have written (denied-timeout / denied-unresolved). The hooks call them through _sibling_approval_refusals' record_refusal callback. Built-in rows stay unaudited, as for a Deny. The approval round writes no row for a timeout, and Stop/revoke rows are deny (skipped), so nothing is duplicated.
- One _APPROVING_DECISIONS constant (also _review_decision's default); MCP extends it with allow_matching.
- Args sections on the new tests.
New tests: test_review_hook_refuses_and_audits_an_mcp_sibling_without_its_own_approval (hook-level audit), test_record_hook_refusal_writes_the_dispatch_decision (MCP seam), and the local runtime test now asserts one denied-* row for the refused sibling.

PR2953 integration with hooks/plugins dev e92b01515f selected the existing Stop-mid-approval audit test. It failed before its assertions because its real LocalToolProvider config read used the per-test redirect. Added the existing bootstrap_profile marker to this one node, matching the neighboring real-provider tests; the ordinary exact-node run now passes (1 passed, zero failures/errors/skips), with no production change or admission bypass. ADR required: no; existing ADR-126 config/profile ownership applies. Final-head review and CI remain required in PR2953.
<!-- SECTION:NOTES:END -->
