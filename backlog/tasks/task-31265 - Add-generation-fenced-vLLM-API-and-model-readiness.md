---
id: TASK-31265
title: Add generation-fenced vLLM API and model readiness
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:32'
updated_date: '2026-09-04 15:25'
labels:
  - vllm
  - lab
  - readiness
dependencies:
  - TASK-31263
  - TASK-31264
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace process-liveness completion with an explicit, privacy-bounded vLLM lifecycle that proves the OpenAI-compatible API and served model are ready.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Lab distinguishes not configured, checking, launching, loading model, ready, stopping, and failed states.
- [x] #2 Ready requires a current-generation bounded models-endpoint probe and an admissible exact served-model identity.
- [x] #3 Cancellation, target edits, process death, recomposition, and newer checks prevent stale results from enabling actions.
- [x] #4 Activity and recovery expose bounded categories without retaining credentials, raw commands, paths, or unrestricted child output outside the Lab-owned boundary.
- [x] #5 Unit, loopback HTTP, lifecycle, privacy, and mounted UI tests cover the state machine.
- [x] #6 Existing-server discovery returns a bounded non-ready candidate set; only an explicit admissible selection followed by an exact current-generation reprobe may publish a verified target.
- [x] #7 Presentation-only recomposition preserves valid exact-claim evidence or exposes a reachable Reverify action, while an active runtime with a dirty checked draft exposes Restart with draft without losing Stop or synchronized network-warning recovery.
- [x] #8 Active credential values are excluded before candidate retention; Chatbook-owned probes/results/snapshots retain no candidate list, while external discovery and exact selection retain only bounded safe IDs necessary for the visible selection flow.
- [x] #9 A newly mounted Models screen preserves an app-scoped READY target only when the restored profile, current owner token, exact claim-bound launch snapshot, and live runtime ownership still agree; profile mismatch invalidates safely and recovery remains reachable without focus theft.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Follow Docs/superpowers/plans/2026-09-03-vllm-lab-console-complete-redesign.md Task 2 and ADR-117.
2. Add RED owner, loopback probe, cancellation, process-exit, privacy, and mounted recomposition tests.
3. Implement the app-scoped VllmConnectionOwner, bounded activity, credential-aware health/models probing, and LLMScreen lifecycle orchestration.
4. Run the focused Task 2 and incumbent lifecycle suites, self-review, and record exact evidence.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already fixes the connection owner, generation fencing, privacy boundary, lifecycle ownership, and rollback behavior.

Task 6 Fix Round 2:
5. Add a RED regression proving stop-failure global notification copy remains bounded/actionable while excluding the process ID.
6. Remove process identity from the global notification at the lifecycle projection seam and run the focused lifecycle/privacy GREEN checks.
7. Run the complete relevant core matrix plus diagnostic-inventory statement review and static/diff gates; append exact evidence before restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This is a narrow enforcement of ADR-117's existing process-identity privacy rule.

Final UX fix round:
8. Add RED connection-contract tests for a discovery-only result, bounded admissible candidates, empty/missing/changed lists, stale selection invalidation, and exact selected-model reprobe/generation fencing.
9. Add RED mounted lifecycle tests for readiness evidence across presentation recomposition, reachable Reverify when evidence is invalid, and edit -> Check draft -> Restart with draft against a safely controlled process owner.
10. Implement the minimal owner/controller distinction between discovery and verification plus semantic invalidation that excludes presentation-only refresh, keeping Stop and exposure recovery synchronized.
11. Run focused RED/GREEN nodes sequentially and the complete primary, compatibility, geometry, privacy, static, inventory, CSS, and diff gates before checking the new ACs and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: The accepted ADR already separates discovery, explicit model selection, exact verification, semantic invalidation, and current-versus-next restart ownership; no new service or security boundary is introduced.

UX Fix Round 2/5:
12. Add sequential RED privacy regressions for credential-shaped model IDs, Chatbook-owned success/failure minimality, external discovery filtering, and exact-selection visibility.
13. Filter the ephemeral active credential before model-ID retention, enforce owner-specific candidate invariants at result construction/settlement, and retain only the safe external IDs required by discovery or exact selection.
14. Run focused RED/GREEN privacy/owner probes and every requested full gate before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This narrows retained readiness evidence under ADR-117's existing credential and bounded-discovery privacy rules without changing the service contract.

UX Fix Round 3/5:
15. Add a RED production-shaped Lab-to-verified-READY-to-Console-use-to-fresh-Models-screen regression plus a mismatched-profile regression, without relying on same-screen recomposition.
16. Reconcile first profile hydration against the current owner fingerprint and exact live claim-bound launch snapshot, while retaining ordinary invalidation for deliberate selection and every mismatch.
17. Run focused readiness/workflow GREEN checks and every requested full gate before checking the AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already requires app-scoped readiness evidence, exact lifecycle ownership, generation fencing, and safe invalidation; this round corrects first-screen hydration at that boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented an app-scoped immutable vLLM connection owner with exact generation/fingerprint/runtime fencing, bounded allowlisted Activity, credential-aware bounded health and model-list probes, exact local alias verification, admissible existing-server model IDs, and sanitized failure categories. Moved Check/Start/Retry/Stop orchestration to LLMScreen while retaining shared server-lifecycle claims and reducing the legacy event module to picker/compatibility glue. Added readiness/Activity UI projection plus source-side suppression for programmatic Textual field updates, and covered draft edits, raw arguments, cancellation, process death, screen detach, recomposition, response bounds, privacy canaries, and stale settlement. Focused evidence: readiness/UI 33 passed; prescribed filtered readiness 7 passed/26 deselected; incumbent lifecycle 31 passed; incumbent vLLM setup/action 34 passed/17 deselected; Ruff and focused mypy passed; git diff --check passed. ADR required: no. ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md. ADR-117 already fixes ownership, fencing, privacy, lifecycle, and rollback.

Fix Round 1 binds the immutable launch snapshot to the exact shared lifecycle claim, retries live processes only from that binding, rejects cancelled claims, and restores exact runtime ownership across draft invalidation and screen replacement. READY results now require a canonical credential-free target, exact `chatbook-vllm` identity for owned launches, and fail-closed owner revalidation. Stop-before-publication settles as cancellation, preflight failures settle into the authoritative owner, and Stop enablement is derived independently from exact live process ownership. Final focused evidence: readiness/workflow 45 passed; lifecycle/status 31 passed; Task 1 setup compatibility 34 passed; deferred-view compatibility 2 passed/7 deselected; Ruff, focused mypy, and `git diff --check` passed.

Fix Round 2 closes the remaining owned-target identity gap: READY settlement for a Chatbook-owned token now derives the canonical completion endpoint from the exact claim-bound launch snapshot, requires the exact bound operation token and fingerprint, and refuses any other canonical endpoint. External-server targets remain claim-independent. The canonical port-8001 mutation against a port-8000 claim failed before the fix and passes after it; final focused evidence is 48 readiness/workflow and 31 lifecycle/status tests passing, with Ruff, focused mypy, and `git diff --check` green.

Task 6 integration fix round: deadline settlement now buckets the total bounded
readiness orchestration window, while individual probe Activity keeps its own
attempt timing. The allowlist matches the configured 30-second deadline
(`under_1s`, `1_to_4s`, `5_to_14s`, `15_to_29s`, `30s_or_more`). A controlled
clock test went RED when a 30-second terminal timeout reported `under_1s`, then the
deadline/loopback timeout nodes went GREEN (`2 passed`). Integrated privacy
canaries now traverse the real profile notifier, app-scoped owner/claim state, and
server-resource lifecycle logger; after restoring pytest capture following the
app's deliberate logging reconfiguration, the node passed (`1 passed`) with no
credential, path, raw-command, URL, or response canary retained or emitted.

Task 6 Fix Round 2 removes process identity from the global stop-failure
notification while retaining bounded recovery copy: `llama.cpp did not stop;
retry Stop.` The regression was RED with the previous `process 4242` copy and
GREEN as `1 passed`. Exact statement review found no new diagnostic owner or
sink; `check_persistent_diagnostic_inventory.py --diff` therefore remained
clean without regenerating the inventory. No new ADR: this directly enforces
ADR-117's existing process-identity privacy boundary.

The final UX fix round separates external discovery evidence from verified
readiness. A discovery probe may retain at most 100 unique admissible model
IDs, has no target, and settles non-ready; empty, missing, changed, or stale
lists cannot preserve selection. Selecting one returned ID advances the owner
generation and launches an exact reprobe, and only its current exact result may
publish READY. Presentation-only recomposition now preserves an otherwise
valid exact claim and target, while detach or semantic edits still fence it.
With an owned runtime active, a dirty draft keeps Stop reachable, exposes Check
draft, and enables Restart with draft only after current-draft validation; the
mounted fake-process test exercises that visible edit -> check -> confirm ->
restart path. Focused discovery, stale-list, recomposition, generation-cancel,
and mounted restart nodes were observed RED before their owning changes and
GREEN afterward. No service, credential, process, or persistence boundary
changed beyond ADR-117's accepted discovery-versus-verification contract.

Final shared qualification for this round: the setup/connection/profile and
mounted workflow/geometry primary passed `308` tests in `428.25s` with no
descriptor-growth warning; the production CSS build/sync/staleness gate passed
`39`; format, critical Ruff, `py_compile`, both profile/diagnostic inventories,
and `git diff --check` passed. The host still has neither a `vllm` executable
nor an importable `vllm` package, so loopback probes remain contract evidence,
not a live-vLLM claim.

UX Fix Round 2/5 filters an exact active credential echo before any model-ID
candidate is retained and adds constructor-level owner invariants: Chatbook
results retain no candidates, external discovery alone may retain the bounded
candidate set, and external READY may retain at most its exact selected
singleton. Probe success/failure, owner snapshots, and logs are covered by
credential-echo and response-canary regressions. The four initial privacy
contracts and the later forged-READY invariant were observed RED before their
production checks and GREEN afterward. Final connection plus mounted workflow
passed `100`; the final five-file primary passed `325`. This narrows ADR-117's
existing evidence-retention contract and requires no new ADR.

UX Fix Round 3/5 distinguishes first profile hydration on a newly mounted Models
screen from a deliberate selection. READY survives navigation only when the
app-scoped owner token, selected profile ID and semantic fingerprint, exact
claim-bound launch snapshot, and still-running Chatbook-owned process agree.
Mismatch or dead/stale ownership invalidates the generation and target, keeps
Stop truthful for a still-running old process, and leaves Check/Retry reachable.
The production-shaped Models -> verified READY -> Use in Console -> fresh Models
regression uses actual navigation and the screen's own asynchronous profile load;
it also proves passive hydration does not steal focus. The placeholder-generation
assertion was RED before reconciliation and the exact/mismatch/navigation cases
are GREEN after it. ADR-117 remains the governing ownership contract; no new ADR
or generalized lesson is required.
Final Round 3 qualification passed the `60`-case workflow, `71`-case geometry,
and `329`-case five-file primary gates under the normal descriptor limit.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31215. During the branch integration sweep,
current `origin/dev` already shipped `task-31215 -
Personas-mount-heavy-center-views-on-first-use.md` at add commit
`2516735cfd27df249ab45e96c96f15b8aee35d15`. The unmerged vLLM task therefore
moved to collision-free TASK-31265, carrying every dependency and documentation
reference with it. The vLLM record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.
