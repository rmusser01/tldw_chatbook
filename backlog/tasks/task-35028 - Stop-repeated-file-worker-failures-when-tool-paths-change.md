---
id: TASK-35028
title: Stop repeated file-worker failures when tool paths change
status: Done
created_date: 2026-10-15 16:04
assignee:
- '@codex'
labels:
- agents
- tools
- bug
priority: high
references:
- backlog/decisions/078-structured-agent-tool-outcome-provenance.md
- backlog/decisions/154-agent-denial-streak-boundary.md
updated_date: 2026-10-15 16:15
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
File-tool worker failures are reported as blocked results, so an agent can repeat the same unavailable operation with changing paths without reaching the existing three-failure stop. Bound those execution failures while preserving permission-denial and workspace-authority behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Three consecutive file-worker failures from the same fs_* or virtual CLI tool stop the agent before another model call even when paths change, for native and fenced tool calls.
- [x] #2 Permission denials and root/scratch-authority refusals retain their existing blocked outcomes and denial-counter semantics.
- [x] #3 Worker failure messages stay bounded and credential/path safe, and the final settled failure is recorded before termination.
- [x] #4 Targeted provider/runtime/denial regressions, scoped lint and formatting pass; architecture documentation records the corrected outcome.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md; backlog/decisions/154-agent-denial-streak-boundary.md
Reason: correct execution-failure classification within existing outcome and retry contracts, retaining existing unknown-effect goal stops; no new authority, interface, schema, dependency, or denial policy.
1. Add provider-to-runtime regressions using admitted-root providers and injected worker failures, changing paths across native/fenced calls; observe RED before production edits.
2. Return ordinary failures for worker execution errors in local and virtual CLI providers, preserving uncertain dispatch effects so automatic goals still stop immediately, and retaining root/scratch authority blocks and all denial logic.
3. Update affected outcome expectations and architecture documentation. Verify final failure recording, automatic write safety, explicit-denial/root-authority controls.
4. Run targeted tests and scoped lint/format checks, independently review the diff, record evidence, and complete task hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Corrected the local and virtual CLI worker-error branches to return ordinary failures, reusing the existing three-consecutive-failure guard without changing the runtime or denial counters. Kept worker effects explicitly uncertain: automatic goals stop immediately rather than repeating a potentially unfinished write. Root-pin, scratch-lease and permission refusals remain blocked, with unchanged authority and denial semantics. Worker error copy and redaction remain unchanged.

Regression coverage in `Tests/Agents/test_local_tool_worker_failure_copy.py` uses admitted-root providers and injected executor errors: seven worker codes across fs_list/fs_read/virtual CLI and native/fenced calls; final failure recording; real permission/root/scratch refusal controls; and automatic fs_write/virtual CLI uncertainty stops. Updated directly affected provider expectations, including approved execution-failure provenance, and retained the private bootstrap profile for the three local-provider tests whose real config reader requires admission. Updated `Docs/Architecture/agent-runtime.md` and recorded the incident in `backlog/docs/lessons-testing-evidence.md`.

Evidence: 99 baseline controls passed. All 42 changing-path cases failed for the intended missing stop before production edits. Review caught the automatic-goal uncertainty regression; all 14 new goal cases failed before preserving dispatch uncertainty. Final combined runtime/worker/denial/virtual-CLI/continuation run: 305 passed; affected local-provider selection: 17 passed (322 distinct cases). New/expanded regression module passes Ruff lint and formatting; all edited Python formatting ranges pass and per-file comparison introduces no diagnostics over the 79 inherited findings. `git diff --check` passes. Shared dependency/foreign pytest cleanup warnings are unrelated. Independent review found no remaining actionable findings after the uncertainty correction; self-review complete.

ADR required: no new ADR. Existing ADR-078 and ADR-154 apply; existing unknown-effect handling is preserved. No dependency, schema, permission change, full-suite run or live Qwen/llama.cpp qualification.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Worker execution failures with changing paths now stop ordinary runs after three consecutive failures; uncertain execution stops automatic goals after one attempt. Permission-denial handling is unchanged. Verified with 322 targeted tests and independent review.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
