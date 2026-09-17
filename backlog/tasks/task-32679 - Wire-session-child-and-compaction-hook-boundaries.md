---
id: TASK-32679
title: Wire session child and compaction hook boundaries
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:24'
updated_date: '2026-09-17 01:49'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32678
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Expose the approved lifecycle events where their effects can be applied safely to the owning run and context.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 SessionStart uses cancellable provisional admission and publishes dependent capabilities/context only after controlling requirements succeed; tab focus and package inspection emit no session events.
- [ ] #2 SubagentStart narrows inherited tools and budgets, and SubagentStop contributes only to an active parent checkpoint without restarting settled work.
- [ ] #3 PreCompact supplies required compactor input and PostCompact fences subsequent input after committed summaries; runtime context blocks are reassembled without duplication.
- [ ] #4 Manual-only UserPromptSubmit, idle hook-set replacement, SessionEnd, required-context failures and independent component success are covered at actual Console and agent boundaries.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/162-managed-agent-plugins.md
Reason: Direct implementation of accepted session, child and compaction hook ownership under the existing Console/agent runtime. Consume final H1-H3 APIs and R41 without creating a second runtime or permission owner.

1. Read task12 brief, actual TASK32679 and the completed H4 section of hook-seam-research.md with controller clarifications. Read final H1-H3 reports/APIs and accepted spec lifecycle/context sections. Recheck actual submit, child inline/fleet and compaction entries. Identify exact affected tests before edits; notify controller before additional source/test/doc paths beyond those explicitly recorded here.
2. Establish actual Console submit RED with a valid controlled command hook and successful submission control. Count real accepted turns and root Stop events. Required SessionStart refusal or cancellation must occur before durable turn publication, root-stop arming or leaked capacity; an isolated lifecycle helper is not proof of Console admission.
3. Add hooks_v2/lifecycle.py and context.py and integrate the existing runtime/controller/bridge. Reserve a reversible atomic provisional root slot after actual draft/workspace/parent validation and before awaiting initialization. Keep immutable hook sessions owned by live runtime identity, publish only after controlling success and release every provisional owner on refusal/cancellation. SessionStart applies across valid origins; UserPromptSubmit remains manual-only. Preserve independent capability eligibility when only dependency-scoped initialization fails.
4. Implement idle-only hook-set replacement and app/session-owned SessionEnd using H2 immediate admission sealing and retained cleanup. Tab focus, archived history and package inspection do not create sessions. Preserve required and invalid-required state when master disabled. Cover detach/remount and viewless lifetime through actual runtime boundaries rather than creating a view owner.
5. Wire SubagentStart after inherited restrictions and before either inline or fleet child admission. Apply R41 narrow-only tool and budget intersections, including zero-unlimited dimensions and runtime-tool gates, then rebuild contained child config. Install SubagentStop's active-parent checkpoint before completion can enable parent input; preserve durable child settlement ordering. Late/parentless context is metadata-diagnosed and discarded without reviving old work.
6. Place PreCompact on actual automatic and explicit manual compaction candidates, with required context confined to that pending candidate and discarded on abort. Install PostCompact checkpoint adjacent to the real successful commit before completion publication, outside external-work locks. Required post failure fences future input without rolling back committed summary. Reassemble runtime-owned context separately so summaries cannot duplicate hook blocks.
7. Run the exact four brief feature/compaction files and relevant named production neighbors for fleet, individual settlement, runtime shutdown and viewless ownership if modified: Tests/Agents/test_fleet_runtime.py, Tests/Chat/test_individual_fleet_settlement.py, Tests/Chat/test_console_runtime_shutdown.py, Tests/Chat/test_console_viewless_hooks.py. These test paths are authorized for the brief's stated inline/fleet, SessionEnd and viewless requirements. Use actual positive/refusal controls, deterministic races, isolated profiles/worktree provenance and a source/test freeze for final covering runs; targeted only, no full sweep.
8. Document final lifecycle/context APIs and H5/H6/I7 handoffs in task-12-report.md and Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md and backlog/decisions/163-expanded-console-hook-runtime.md. Run syntax/Ruff/Black shared-file baseline parity and whitespace checks; preserve actual commands/output, limitations and frozen hashes. Controller owns commits, independent review, all four ACs, implementation notes and Done after gates pass.

Verification handoff from H2 review F3: any child bootstrap/network/profile fixture needed by committed tests must be repository-owned and use the actual pytest temporary root. A private implementation runner may isolate the parent, but cannot be a hidden prerequisite for the ordinary exact test command. Reuse the final reviewed H2 fixture where appropriate; controlled HTTP tests must explicitly scope their own loopback allowance rather than disable network checks globally. Record extra helper paths before edits.

H3 composition handoff: read final task11 report/API plus R48/R49 rulings. Reuse the existing checkpoint owner and captured dependency status; do not replace owning-event versus dependent-capability scopes. Normal settlement joins all pending required events. ContextLedger/context.py is the lifecycle-owned context seam introduced by H4, consuming H3's host-injected atomic publisher; avoid a second queue or attribution owner. Preserve actual host dispatch provenance at child/tool boundaries.

9. Compose H4 with the reviewed H3 live owner rather than a parallel checkpoint gate. The current ToolHookRun constructs a tool-specific HookCheckpointStore/currentness callback and has no injected shared store; consume h3-lifecycle-api-handoff.md and final task11 Fix1 API/evidence. Additional exact paths authorized only as needed for this composition are Agents/hooks_v2/tool_pipeline.py, Agents/hooks_v2/checkpoints.py and Agents/agent_runtime.py under tldw_chatbook. Expose the minimal public live-owner/context/checkpoint seam so actual session, active-parent child and compaction producers use correct event-specific currentness under one admission/terminal owner; never feed non-tool events to the existing tool-only metadata probe or preserve a closed/vanished run by creating a second gate. Preserve R48 dependency scope, R49 tool dispatch versus child-admission distinction, R50 attributed complete-send limits, current authority and no hook await under coordination locks. If those shared H3 paths change, include Tests/Agents/test_hooks_v2_tool_pipeline.py, Tests/Agents/test_hooks_v2_post_checkpoints.py and Tests/Agents/test_agent_runtime.py in affected targeted coverage; notify controller before editing another test/source path. Include every amended path in frozen hashes/static baseline checks. Existing ADR163/162 apply; no new architecture owner or ADR.

10. Resolve the concrete pre-run/manual ownership gap under R53 and the actual producer projections under R54 before production edits. Use one runtime-session-owned HookCheckpointStore with explicit host-owned session, provisional-turn/operation and actual-run scopes; inject this shared store/currentness/context seam into ToolHookRun. Ancestor requirements and context acceptance use the same admission coordination, with no external await under locks and no fabricated AgentRuns IDs. Automatic compaction fences the same pending turn. Manual committed compaction installs a session-owned requirement and one bounded context handoff tied to the exact live session, branch and committed memory identity; the first matching accepted turn claims it atomically, with no replay, cross-session reuse or late-result retargeting. Read h4-owner-projection-rulings.md for lifecycle, cleanup and handoff limits. Extend exactly tldw_chatbook/Agents/hooks_v2/validation.py and Tests/Agents/test_hooks_v2_validation.py for closed, typed producer data: SubagentStart child_task/tool_ids/budget_caps/model/provider, PreCompact candidate/reason, PostCompact reason/memory_id/summarized_prefix_digest. Include these in final covering tests, freeze and static checks. Record the ownership and projection clarification in the existing accepted hook spec and ADR163 before relying on it; no second runtime/dispatcher, new durable run, or separate permission owner. Preserve no-commit visual projection behavior and actual manual/automatic summary commit boundaries. Additional source/test paths still require an explicit CLI amendment.

11. Actual Console/SQLite compaction qualification exposed cached newly accepted message parent fields being absent when _durable_context_snapshots builds the branch lineage. Within the already authorized console_chat_controller.py, make only the necessary prerequisite correction to read authoritative persisted branch-parent metadata before producing the guarded memory lineage/currentness projection; never invent a chain from list order or weaken commit_memory_selection_if_current. Verify the claimed pre-existing condition against baseline source or a controlled baseline reproduction, preserve the actual SQLite/cached-acceptance failure and positive case in Tests/Chat/test_hooks_v2_compaction.py, and record it separately from new hook behavior. No storage/schema/ancestry-policy change or additional source/test path is authorized by this step; notify controller if one is necessary. Include final bytes in the same frozen coverage and shared static baseline checks. Existing ADR163 plus the existing branch-memory contract apply; this is a narrow prerequisite to AC3/R53, not a new runtime or ADR. Include unchanged Tests/Chat/test_console_durable_turn_acceptance.py and Tests/Chat/test_console_memory_selection.py as targeted neighbors for that prerequisite: actual durable acceptance and real repository ancestry/refusal controls. This coverage addition does not authorize edits to those two files; include their unchanged hashes in the final evidence.

12. Resolve the implementer-reported R53 retirement tradeoff explicitly under R55 before initial review: retain identifier-only closed-scope tombstones for that hook-session owner's lifetime to reject rebinding of retired host IDs, while removing context bodies, delivery maps, parent/currentness maps and settled checkpoint state. No durable/global closed-owner registry or cross-session reuse is permitted; retained process cleanup keeps only its required original custody. Record the linear metadata cost and ownership limit in the existing hook spec/ADR163 and controller review inputs. This is a documentation/contract clarification for the already-frozen implementation, not a new runtime feature or a claim of constant-memory session metadata. The independent reviewer must still assess actual retirement, lifetime and resource behavior. Source/test hashes and qualified behavior remain unchanged.
<!-- SECTION:PLAN:END -->
