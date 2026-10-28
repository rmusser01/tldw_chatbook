# Tool Disclosure Refusal Recovery Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline. Steps use checkbox syntax for tracking.

**Goal:** Let an agent distinguish an unloaded tool schema from a tool unavailable to its run and recover by loading eligible tools.
**Architecture:** One service-level refusal helper is reused by ordinary and skill dispatch. Existing permission and disclosure gates still refuse before execution.
**Tech Stack:** Python 3.12+, pytest, real SQLite AgentRunsDB; no dependencies added.
**Spec:** Docs/superpowers/specs/2026-10-28-tool-disclosure-refusals-design.md
**Backlog:** TASK-37146

ADR required: yes
ADR path: backlog/decisions/252-tool-availability-refusals-before-hook-preparation.md; existing: backlog/decisions/104-token-budgeted-agent-tool-disclosure.md; backlog/decisions/078-structured-agent-tool-outcome-provenance.md; backlog/decisions/154-agent-denial-streak-boundary.md
Reason: the related hooks repair extends the preparation return contract under ADR-252 while preserving ADR-104/078/154 authority rules.

## Global Constraints

- Keep allowed-tools checks first; unavailable and unknown names share the existing generic refusal.
- Preserve blocked/not_started results without assigning approval authority.
- Preserve atomic working-set replacement and token budgeting; no automatic schema loading or execution.
- Run targeted tests only, as required by AGENTS.md.

## Review Focus

- A disclosed but disallowed name remains refused without reload guidance.
- A schema removed by working-set replacement can be reloaded and executed.
- Native and fenced tool-call histories retain the correct refusal and recovery results.
- Skill runners never run before schema disclosure and share the same recovery message.
- Recoverable schema refusals do not fabricate user denial or execution effects.

### Task 1: Correct refusal diagnostics and prove recovery

**Files:** `tldw_chatbook/Agents/agent_service.py`, focused tests under `Tests/Agents/`, and `Docs/Architecture/agent-runtime.md`.
**Consumes:** AgentConfig.allowed_tools, the mutable per-run disclosed_names set, ToolResult.blocked.
**Produces:** `_tool_availability_refusal(name: str, allowed_tools: tuple[str, ...], disclosed_names: set[str], registry: ToolCatalogRegistry) -> ToolResult | None` shared by ordinary and skill dispatch.

- [x] Add service regressions for unloaded, disallowed, and unknown names; verify refusal provenance and no early dispatch.
- [x] Add native/fence recovery regressions starting from deferred or replaced schemas; strengthen the existing undisclosed-skill regression.
- [x] Run the new tests and confirm failures are the missing recovery guidance.
- [x] Implement the shared helper and replace only the two duplicate availability checks.
- [x] Run affected service/skill/runtime tests, lint and formatting; inspect the complete diff.
- [x] Update architecture and Backlog notes with actual evidence and limitations; complete the task only when its DoD is satisfied.

## Progress

- Baseline: the existing disallowed-tool and working-set replacement service tests passed (2 tests). Isolated profile and explicit pytest basetemp avoid unrelated Windows temp ACL failures.
- Pre-flight: one implementation task; no conflicting interfaces.
- Ruling: proceed inline under the user's explicit instruction to fix and continue; no additional design approval is needed.

- RED: new refusal/reload tests plus the existing undisclosed-skill test produced 6 expected assertion failures and 3 passing excluded/unknown-tool controls. Failures were the unchanged generic refusal, not setup errors.

- Review found a related hook-preparation path that hides the reload refusal, plus an allow-listed name absent from the registry that receives an impossible reload instruction. Both are covered by added regressions before repair. The preparation return contract requires ADR-252 before implementation.

Related repair: ADR-252 records the typed preparation refusal contract. The shared availability check also verifies catalog membership before promising reload recovery.

- Review RED: the unknown-but-allowed control and both real hooked recovery protocols failed as expected (3 failures); typed preparation refusal failed while 3 malformed-return controls passed. The fix adds catalog membership validation and preserves typed preparation refusals under ADR-252.

- Core GREEN: 17 focused recovery and preparation tests passed; 10 adjacent service/skill tests passed before the hook repair, and 8 final service/skill/note tests passed.
- Baseline qualification: the existing hook/approval batch has the exact same 19 failures and 1 teardown error under both HEAD and the patch (51 old tests pass on HEAD; the patch adds 5 passing controls). An unchanged low-level hook execution test identifies `unsupported_platform`: command hooks deliberately refuse Windows before launch. Do not weaken that gate.
- The existing real folder-worker test fails with a root-pin refusal independently of disclosure. The separate Windows identity investigation found an existing upstream fix in commit 6fe22b80ba; that implementation is retained on current dev.
- Formatter: the service's pre-existing format differences are unchanged against HEAD; all changed test files and agent_runtime.py pass formatting, and E9/F lint plus diff whitespace checks pass.

- Final guard RED: a typed blocked preparation result whose error text was exactly `proceed` incorrectly dispatched. A fixed `ERROR:` prefix prevents refusal text from colliding with the guard allow verdict; the regression checks no acceptance, review, dispatch, or invented denial.

Final verification: 67 integrated targeted tests passed (195.48s), followed by 10 final refusal/guard/hooked-recovery checks (17.60s). E9/F lint passed, seven source/test files passed formatting, existing service formatter differences match HEAD, whitespace checks passed, and the canonical generated worker matches source and imports with the standard library only. No full-repository suite was run.
