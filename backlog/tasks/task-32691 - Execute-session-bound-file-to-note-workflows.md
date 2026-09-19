---
id: TASK-32691
title: Execute session-bound file-to-note workflows
status: Done
assignee:
  - '@codex'
created_date: '2026-09-16 04:59'
updated_date: '2026-09-19 22:19'
labels:
  - workflows
dependencies:
  - TASK-32690
documentation:
  - Docs/superpowers/specs/2026-09-16-workflows-first-run-design.md
  - Docs/superpowers/plans/2026-09-16-workflows-first-run.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deliver the approved local text-file, prompt, llama.cpp, editable review and Local Note workflow using session-owned execution that survives navigation but not restart, without new SQLite execution infrastructure.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A saved immutable workflow revision completes the five-step example through the existing Workflows screen using the selected llama.cpp endpoint and actual model.
- [x] #2 Run state and review edits survive screen navigation but are never resumed or replayed after app restart; saved definitions drafts and committed Notes remain durable.
- [x] #3 File model and Note effects use captured destinations and fresh fail-closed authority; Off kill-switch and rejected or expired review prevent subsequent effects.
- [x] #4 The selected keyless llama.cpp request has zero retries no redirects or proxies bounded response bytes an absolute deadline and physically settled cancellation; no Ollama fallback is used.
- [x] #5 Note creation preserves existing policy and transactions with one attempt ID verified same-destination readback and commit-wins cancellation reporting; no blind retry or remote sync is dispatched.
- [x] #6 Duplicate Run and Accept actions cannot duplicate effects; quit confirms loss fences new actions and drains owned work before dependent services close.
- [x] #7 No new SQLite schema ownership locks PID records helpers or persistent execution writes are introduced and historical workflow rows remain untouched.
- [x] #8 Targeted automated tests real temporary databases actual-app UI checks and an isolated live localhost:9099 file-to-reviewed-Note run pass with truthful lint and performance evidence.
- [x] #9 Source prompt response and review payloads remain absent from ordinary logs and error diagnostics including resolved Note titles on failed writes.
- [x] #10 The rebased integration handles inventory completion and managed GGUF handoff during lazy selector mounting without crashing or losing the latest selection; scoped regressions pass and current-head platform CI remains a mandatory merge gate.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; direct implementation of the user-approved session-bound amendment to ADR-138. ADR path: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md. Reason: storage ownership and service-composition boundaries remain governed by ADR-125 and ADR-036; the approved design already specifies the opt-in model and permission integration. 1. Map exact current source seams and write Docs/superpowers/plans/2026-09-16-workflows-first-run.md from the approved spec. 2. Qualify bounded llama.cpp and strict permission integration with failing/passing targeted tests. 3. Implement bounded local effects and the in-memory sequential session with retained physical operations. 4. Wire existing app lifecycle and Workflows Run/setup/review/Open Note surfaces without redesign. 5. Verify targeted regressions and isolated actual-app/live-model UAT, self-review and independent review, and update docs/task evidence. No implementation code is written during the planning checkpoint.
PR #2743 review follow-up: rebase the existing commits onto dev, preserve upstream behavior, verify each Qodo finding, restore payload-free Note-failure correlation, document the public Run controls, and run targeted workflow/lifecycle/Notes/style checks before publishing replies and the rewritten branch. ADR required: no new ADR. ADR paths: backlog/decisions/138-portable-workflow-definitions-and-local-execution.md and backlog/decisions/029-local-private-data-boundary.md. Reason: routine review corrections preserve existing lifecycle, transport, and private-data boundaries.
Approved follow-up: reproduce the CSS candidate ratchet failure, re-key only the workflow run controls, rebuild generated CSS and verify the existing visual/interaction matrix without raising limits. Verify and correct Qodo's new setup JSON bound, negative review timeout and public API documentation findings with targeted RED/GREEN tests. Review exact diagnostic-statement drift before regenerating the inventory. Publish tested fixes and inline evidence; monitor current-head review/checks, never merge. ADR required: no new ADR. ADR paths: backlog/decisions/097-boot-budget-ratchets.md, backlog/decisions/138-portable-workflow-definitions-and-local-execution.md and backlog/decisions/029-local-private-data-boundary.md. Reason: routine bounded validation, documentation and selector corrections preserve the existing contracts.
Final 2026-09-19 user request supersedes the earlier no-merge restriction: rebase onto latest dev, verify the unchanged patches and targeted integration checks, reconcile current-head Qodo feedback, and merge PR #2743 only after current-head review and CI settle successfully. Preserve unrelated worktree edits. ADR required: no new ADR; this is integration of the existing approved ADR-138/ADR-097/ADR-029 implementation, not a change to its contracts.

Current-head Windows CI follow-up: reproduce the managed GGUF inventory callback racing lazy Select child composition with a controlled mount barrier; defer selector writes and pending handoff until mounted, replaying retained inventory through the existing post-mount callback. Verify the race, exact-reference handoff, existing inventory fencing and compact keyboard cases; publish only the scoped fix and wait for current-head platform checks and Qodo. ADR required: no. ADR path: N/A. Reason: routine lifecycle correction without changing source authority, storage, launch ownership, UI design, or dependencies.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-138's saved-revision, session-bound file -> prompt -> keyless llama.cpp -> editable human review -> Local Note flow in the existing Workflows screen. Reused existing permissions, Notes and SQLite owners; no new schema, persistent execution, locks/PID/helper service, retries or Ollama fallback. Captured exact-effect approvals, navigation-safe review, Open Note and retained physical quit/cancel settlement are covered.

Qualification fixed nested Library mounting and stale editor focus restoration plus privacy-test logging isolation. Eight passing actual localhost:9099/Gemma processes include four same-profile fresh-process restarts without replay. Final branch review found an Open Note/Note-worker circular wait; the existing route guard now supports nonblocking UI checks, manual busy/retry feedback and stale-message cleanup while retaining default blocking worker semantics and destination identity checks. Both fixes have deterministic RED/GREEN. One scoped re-review closed the P1 with no new findings.

Verification: initial final affected run159 passed/1liveSkip; final fix covering229 and separate ProductionApp12 passed; exact final-source controller run11 passed/1liveSkip/1existingwarning in38.16s. No new static debt. Counts overlap and are not additive. Historical merged failures remain recorded, not relabeled green. An unrelated unchanged Skills test-double failure, legacy lint/dependency warning debt and native PTY/font limits are disclosed. No full repository suite or new live requests were run for the final guard fix.

ADR required: no new ADR; existing backlog/decisions/138-portable-workflow-definitions-and-local-execution.md and ADR-125 govern these boundaries. All six task reviews, one whole-branch review and its single fix-wave scoped review are reconciled. Reviewed source head: bcc8290dd2d883893c781a83a3e5cd0f13f27eb5. Evidence: Docs/Developer/Workflows/2026-09-16-first-run-uat.md. Acceptance, rulings and review record: Docs/Developer/Workflows/2026-09-16-first-run-review-record.md. User guide and testing lessons updated. No push, PR or merge performed.

PR #2743 follow-up: rebase onto latest dev and evaluate all nine Qodo findings. Preserve ADR-138 session-only/non-streaming and failed-drain fences. Verify upstream fixes survived, add safe Note-error correlation and public control docstrings, then run targeted rebase/review checks and publish exact-head review dispositions. No merge or full-suite authorization.

PR #2743 follow-up: rebased all 20 patches onto dev cccf0acdad8e939a55cb003588ff1406cef5d1f4; final source head 12fb82b13dd3a3f8769d4257f9da41383d1b37eb. Qodo findings 1/2/3/8/9 resolved by retaining dev; 6/7 fixed safe correlation and API documentation; 4/5 preserve explicit ADR-138 and approved non-streaming boundaries with evidence. Fresh post-rebase UI58, lifecycle11, joined HTTP2 passed (one opt-in live skip). Earlier targeted core/style107, local77 and adjacent6 passed; counts overlap. Reused existing private-profile test helpers; no new production infrastructure. Scoped review has no findings. ADR-138/ADR-029 unchanged. Record: Docs/Developer/Workflows/2026-09-19-pr-2743-review.md. Awaiting exact-head Qodo follow-up; no merge.
Approved PR #2743 follow-up: fixed the CSS candidate ratchet at 274/274 without raising limits, retaining compact session titles and shared editor spacing. Added shared pre-parse UTF-8 setup bounds and recoverable deep-JSON errors; completed public effect/admission docs. Qodo's negative-timeout suggestion was rejected after checking the approved non-positive-unlimited contract; attempted behavior/test changes were removed. Reviewed both exact logging-statement changes and regenerated only their inventory digests. Existing private-profile helpers cover style entry and mounted spacing; no config reset/DB infrastructure. Final checks: Run/style-entry65, spacing/CSS9 and originaldeadline4 passed; seven changed Python files lint/format clean. Groups overlap. Earlier failures and scoped review P2 disposition are in Docs/Developer/Workflows/2026-09-19-pr-2743-review.md. ADR-097/138/029 remain unchanged. Await current-head Qodo/CI; no merge.

Final requested integration: all 22 existing patches rebased unchanged and conflict-free onto latest fetched dev ad0f76e23b8737904f24eb34760bbee9ac01a04c. Fresh rebased-source checks: UI85 passed (including incoming MCP catalog/wide-modal coverage), session/CSS51 passed, app lifecycle11 passed; seven Python files lint/format clean, generated CSS unchanged, diagnostic inventory verified. Existing AC and reviewed implementation are complete; merge remains separately gated on published-head Qodo and CI. The latest user request now explicitly authorizes that gated merge. Unrelated TASK-32601 and UAT files are preserved. ADR-138/097/029 unchanged; no new infrastructure, full sweep, or live-model request. Evidence: Docs/Developer/Workflows/2026-09-19-pr-2743-review.md.

Scoped Windows CI correction: controlled nested-selector mounting reproduced the exact NoMatches, then mounted-control gating and retained post-mount inventory replay fixed it without new infrastructure. Cached hydration now preserves pending fresh-read handoffs; all three race cases join the existing native CI job and exact-node contract. Ten affected legacy inventory/handoff tests reuse the existing private-profile helper, preserving assertions. Final 21 targeted cases passed in 192.48s; three Python files format-clean, CI contract lint-clean, 21 legacy lint diagnostics unchanged, diagnostic inventory verified. Scoped review P2s have RED/GREEN and re-review has no findings. No layout, workflow semantics, storage, dependencies or model calls changed. Existing ADRs remain unchanged; current-head Qodo and platform CI are still mandatory before the separately authorized merge. Evidence: Docs/Developer/Workflows/2026-09-19-pr-2743-review.md and /private/tmp/chatbook-pr2743-gguf-mount.krUWdO.
<!-- SECTION:NOTES:END -->
