---
id: TASK-37234
title: Trace persistent project tool exclusions after disclosure recovery
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-28 18:29'
updated_date: '2026-10-28 19:20'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Fedora reporter still cannot inspect a new project after locally integrating PR #3284. The prior change proved schema recovery but did not establish why that run excluded the requested tools. Determine the actual build and refusal path, reproduce matching inputs and correct verified defects without widening tool authority.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The tested commit and current refusal category are distinguished from the still-open GitHub PR state.
- [x] #2 Tool recovery and command controls run on Linux Python 3.14, matching the interpreter family recorded in the original report.
- [ ] #3 Configuration-capture default mismatches are reproduced at their real consumer before a fix is attributed to the report.
- [x] #4 Any implementation preserves explicit disabled tools, persona restrictions, folder authority and approval gates, with focused regression evidence.
- [x] #5 Ordinary app logs reveal bounded tool-step names/outcomes and numeric schema-discovery/load decisions without arguments, schema bodies, result bodies or project content.
- [x] #6 A fitting compact discovery request is advertised before plain fallback, and runtime progress does not undo a plain plan.
- [x] #7 Built-in file tools reject fs_* path/alias arguments instead of silently inspecting or writing scratch; valid existing paths retain their authority checks.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no for investigation and qualification under existing ADR-104 disclosure, ADR-032 local permissions and ADR-069 folder authority; a new boundary decision would be recorded before implementation. 1. Obtain the reporter tested commit and current exact refusal text while tracing composition. 2. Run the existing real recovery/command controls on Linux Python 3.14 as well as 3.12. 3. Reproduce capture-default versus provider/UI mismatches through actual configuration and catalog consumers. 4. Repair only evidenced causes, verify narrow controls and document what remains unconfirmed.

Reporter is unavailable for saved-step extraction. Add privacy-tested message-only diagnostics at the existing model-request, find/load and terminal step boundaries so ordinary Logs exports can identify the next failure without hidden UI or database access. Existing structured metadata policy applies; no authority or execution change.

2026-10-28 follow-up: ADR required: yes. ADR path: backlog/decisions/253-compact-discovery-before-plain-chat.md. Reason: refine provider-facing fallback tiers under ADR-104. Reproduce the actual transcript arguments against real scratch, add compact-plan and runtime parity regressions, implement only the fitting fallback and argument refusal, and verify on Windows plus Linux 3.12/3.14. Existing folder authority ADR-032/069 applies to argument refusal; no new path routing is introduced.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The original Fedora report remains unresolved. The reported error is now Tool schema not loaded and local tools are on, so the named tool should be registered/allowed but absent from the working set. Ordinary app logs omit the discovery entries; the reporting user is unavailable for the validated read-only saved-step extraction. Their exact git line and actual find/load sequence remain pending. No speculative permission or automatic-disclosure change was made.

A real project first-plan probe retained find_tools/load_tools in all four combinations of native/fenced protocol and verified/unverified 32k windows for the reported unknown llama_cpp model. The separate capture-default mismatch does not explain an allowed-and-registered refusal and has not been changed. Linux run 40748792976 on 94f11ed026 passed 144 tests with zero failures/errors/skips on both Python 3.12.15 and 3.14.9. The earlier 3.14 failure was a process-control fixture asserting descendant startup inside a 600 ms lifetime. The repaired control observes descendant/leader readiness, exercises public cancellation, proves OS group absence, and includes an 800 ms slow-start arm.

Added content-free ordinary-message diagnostics for effective schema counts/discovery availability, find match count, schema-load accepted/omitted/invalid counts, bounded accepted names and step kind/name/outcome, correlated by opaque run id. These expose metadata only; no prompts, search queries, arguments, schema/result bodies or project contents enter the new message strings. Five tests were RED for absent diagnostics; the initial diagnostic/recovery run passed 16 and the private query/catalog variant passed five. Final explicit privacy sentinels and a correctly async fake gateway cleanup are included in the Linux matrix. E9/F lint and new-test formatting pass; existing large-module formatter differences are unchanged. Independent review found no source privacy, bounds or semantic issue; its inventory drift exactly matched the three new service calls and revised bridge message, with no sink or unrelated owner drift. Regenerated the canonical inventory; its diff contains only the reviewed three added service calls and revised bridge message, with no unrelated or sink changes. No new ADR is required for this diagnostic representation under existing authority/privacy contracts. The task remains In Progress pending actual failing-run evidence.

The 12:44/12:45 log now confirms active=0/runtime=1/find=0/load=0; the second run loaded list_directory and executed twice. The supplied transcript passes root_alias/path/depth to a tool requiring directory_path/recursive/max_depth. Its missing-directory default resolves scratch and the alias is ignored, explaining the false empty-project evidence. Actual configured context sizes remain pending. The compact planner probe independently reproduces all-tools removal while discovery plus inbox fits; do not infer the reporter window value from that probe.

Follow-up implementation: compact find/load plus gated progress now precedes the plain fallback, with no raised context limit. The frozen compact input ceiling also narrows later schema loading. Plain plans stay plain at runtime. Built-in read/list/write refuse fs_* root_alias/path arguments before scratch access, and list_directory requires directory_path. ADR-253 records the fallback policy; ADR-032/069 authority is unchanged.

Sixteen targeted regressions first failed for the observed missing discovery, progress re-addition, default listing and ignored cross-family arguments. The broader Windows selection passed 71 with two existing symlink setup failures (WinError 1314). The tests now request directory symlinks explicitly and skip only that Windows privilege error; Linux retains full containment checks. Final Windows planner/oversized-schema run: 22 passed (128.42s); file-root plus existing service-planner run: 31 passed, two privilege skips (222.38s). The earlier selection includes 11 recovery and 20 progress controls. Native tests assert actual native tools payloads. Independent review identified the compact-load budget gap; native/fenced oversized-schema tests reproduced it and now pass. E9/F lint, changed-test formatting and whitespace checks pass. Linux qualification is scheduled on the pushed head. No live Fedora retest yet; keep the investigation In Progress.
<!-- SECTION:NOTES:END -->
