---
id: TASK-32677
title: Execute v2 command hooks with bounded resource ownership
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-16 04:23'
updated_date: '2026-09-16 23:11'
labels:
  - plugins
  - implementation
  - hooks
dependencies:
  - TASK-32676
documentation:
  - Docs/superpowers/plans/2026-09-15-expanded-hooks.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Run reviewed hook commands with fair application-wide limits and cancellation that retains ownership until actual cleanup.

Design: Docs/superpowers/specs/2026-09-15-managed-plugins-design.md; Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md. ADR required: yes. ADR paths: backlog/decisions/162-managed-agent-plugins.md; backlog/decisions/163-expanded-console-hook-runtime.md. This task implements the accepted contracts without creating a separate permission or runtime owner.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Both synchronous agent-thread and asynchronous Console entries execute argv-only handlers with bounded input/output capture and no implicit shell.
- [ ] #2 The exact per-runtime and application execution, reservation and observation limits include provisional, nested, late and cleanup-pending work with fair admission.
- [ ] #3 Cancellation closes admission immediately, keeps local children counted until reaped, and separates notification deadlines from the five-second host-reap allowance.
- [ ] #4 Real controlled child processes establish successful execution and kill/reap controls; timeout, failed launch, cancelled waiters and surviving-child paths preserve honest outcomes and metadata-only diagnostics.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/163-expanded-console-hook-runtime.md; backlog/decisions/162-managed-agent-plugins.md
Reason: Direct implementation of the accepted app-owned v2 command execution and lifetime contract. Preserve legacy ADR148; the injected process-owner boundary consumes final F8/R43 without creating another permission, plugin or root owner.

1. Read task10 brief, actual TASK32677 and h2-execution-research.md including the final controller clarification. Recheck the final H1 public APIs/report and F8 process/root ownership APIs/report before implementation; read hook-seam-research.md for the mapped Console entry points. Notify controller before necessary additional source/test/doc paths or any unsettled contract change.
2. Establish the real command_hook_case public-entry RED in Tests/Agents/test_hooks_v2_execution.py with a valid normalized H1 declaration/event and sys.executable command. Separately record import scaffolding failures; behavioral evidence must include valid launch/execution and post-close refusal controls with actual retained child handles.
3. Add hooks_v2/ownership.py, budgets.py, command_executor.py and engine.py within the brief's boundaries. Keep Plugins imports out. Inject reserve/publish/settle process ownership; preserve launch tasks, pending claims and exact host terminal evidence through cancellation, including spawn publication races. Standalone hooks keep their existing host owner; plugin composition supplies final F8 dirty-before-access root custody later at the designated composition task.
4. Implement one application-owned execution loop and shared HookBudgetOwner with the brief's exact execution, lifetime-ticket, observation-queue and worker limits. Provide a thread-safe synchronous facade for agent threads and an asynchronous Console entry; reject synchronous calls from the owning loop. Apply round-robin runtime fairness and per-runtime FIFO. Suspension releases scarce execution slots while preserving lifetime ownership/tickets, and controlling work has separate observation capacity.
5. Bound argv-only subprocess input and streaming capture at the accepted stdin/stdout/stderr/depth/node limits. Consume H1 strict complete-output decoding, enforce output authority/effect acceptance, expand declared environment references under current authority and set host-reserved variables last. Empty successful output stays distinct from invalid or truncated output. Keep diagnostics metadata-only.
6. Integrate lazy app-loop lifecycle and immediate synchronous admission sealing with Chat/console_runtime.py. Preserve legacy hook behavior and avoid implementing H3-H5 event producers early. Close the actual runtime queue before teardown; honor separate Interrupt/SessionEnd notification deadlines and post-kill reap allowance. Keep pending/unresolved processes and tickets counted after cancelled waiters, failed launch, pipe EOF or incomplete cleanup. Retain the cleanup owner before the application releases its loop. Explicitly expose the still-authorized Interrupt/SessionEnd notification seam after ordinary admission sealing for H4/H5; it must not reopen normal work, prompt/connect, or extend notification/effect deadlines.
7. Add actual multi-runtime budget/fairness/suspension, launch-cancel-publication, queued/active close, own-loop sync refusal and viewless/shutdown controls in the three exact new test files. Use real controlled child processes to establish success, termination and reaping on the current platform; accurately record unsupported platform and escaped-descendant limits. Test authority before launch and acceptance. Use final F8's injected owner contract without inferring quiescence from empty SQL results or importing Plugins.
8. Run the three exact feature files and the relevant existing Tests/Agents/test_run_hooks.py and Tests/Chat/test_console_viewless_hooks.py neighbors. Include any additional modified neighbors only after their paths are recorded. Targeted tests only; no full sweep. Use isolated profiles, null keyring/network refusal and verified worktree package/data-root provenance. Freeze code/tests through the final covering runs and retain actual commands/output, including intermediate failures.
9. Document the final public APIs, lifecycle/shutdown ordering and platform qualifications in task-10-report.md and the existing Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md and backlog/decisions/163-expanded-console-hook-runtime.md surfaces as needed to document the accepted execution seam without changing its settled authority/limits. Run syntax, Ruff/Black baseline parity for existing files and whitespace checks. Report exact changed paths, frozen hashes, covering commands/output and H3-H6/M4/I7 handoffs. Controller owns commits, independent spec/quality review, all four ACs, implementation notes and Done only after gates pass.

10. Expose a bounded handler-scoped execution primitive and agent-thread synchronous facade on the same HookEngine/app loop/budget/process owner so the accepted tool pipeline can invoke transformer A, validate/apply its replacement, invoke B on the changed event and final C on frozen arguments. Invocation is restricted to the engine's immutable reviewed definitions and applies the same matcher/current-authority/configuration/ownership/output gates as ordinary event execution. The pipeline owns ordering and permission-bound candidate preparation; this primitive never dispatches a transformed tool directly, creates parallel engines or hides required/invalid configuration state. Name the concrete API and qualify a same-owner sequential changed-event control; record any further contract or path need before editing it.

11. Repair the pre-existing unbound session_id in Chat/console_chat_controller.py finalize_session_close as the minimal prerequisite for the actual H2 shutdown qualification: remove the closed session's chat-create grants only after the existing ticket and generation checks succeed and the session ID is assigned. Preserve all close fences and ownership checks; no broader controller refactor. Qualify successful close grant removal and stale-ticket refusal without grant mutation in the already-scoped Tests/Chat/test_console_runtime_shutdown.py. Record the ten pre-existing close failures separately from new H2 RED evidence. This implements AC3-4 and existing ADR150/163 lifecycle ownership; no new ADR is required. Include this additional source file in frozen hashes, covering tests and shared static baseline parity.

12. Update only the three stale delivering_conversation_id() assertions/cleanup uses in Tests/Chat/test_console_viewless_hooks.py to the existing delivering_conversation_ids() tuple API. Preserve the exact delivering-conversation precondition and empty-delivery assertions; no production compatibility shim or weakened viewless behavior. Record the unchanged-neighbor API mismatch separately from H2 RED, rerun the five scoped feature/legacy/viewless files and include this additional modified test in frozen/static parity evidence.

13. Address independent review F1-F4 via the existing H2 runtime/engine modules and exact covering tests: keep exact session fences while hook custody remains unresolved; convert authority/dependency accessor exceptions to fixed policy-preserving metadata outcomes with cancellation/custody intact; replace mutable event-scope authority/lifetime fields with a bounded engine-controlled read-only handle; make child isolation self-contained in committed Tests/hooks_v2_process_support.py shared by the Agent/Chat tests. Run the exact five-file coverage without the external h2-profile child shim and with a non-/private/tmp isolated basetemp, preserving provenance/network/null-keyring checks. Record actual incident RED/GREEN in backlog/docs/lessons-testing-evidence.md. Apply controller R47 in existing command engine/spec/ADR surfaces: Windows native v2 command execution refuses before launch/root access until whole-tree proof is qualified; current Darwin controls remain the only platform success evidence, Linux remains unqualified. Preserve all original budget/deadline/authority contracts and defer reviewer Minor F5 to final branch triage. Record extra necessary paths before editing; freeze final covering/static evidence and append Fix1 report before scoped review.

14. Fix the remaining F2 transient dependency-accessor defect in Agents/hooks_v2/engine.py and Tests/Agents/test_hooks_v2_execution.py: carry the already-obtained conservative dependency-required status into failure construction, never erase it by retrying the accessor. Add actual raise-then-false RED plus stable false/true controls; preserve fixed metadata, explicit/event blocking scopes, cancellation and zero custody/ticket leakage. Rerun the exact execution test file and any truly affected neighbor, with committed isolation and final frozen syntax/Ruff/Black/whitespace evidence. Append Fix2 evidence before fresh scoped re-review. F1/F3/F4/R47 are already addressed; Minor F5 remains deferred. No contract or ADR change.
<!-- SECTION:PLAN:END -->
