---
id: TASK-34655
title: Bound structured document checks with worker ownership and deadlines
status: In Progress
created_date: 2026-10-07 03:09
dependencies:
- TASK-34654
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- /Users/macbook-dev/.codex/worktrees/structured-document-editing/tldw_chatbook/Docs/superpowers/plans/2026-09-27-structured-document-editing.md
updated_date: 2026-10-11 22:54
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep checks responsive and retire obsolete work safely during edits, navigation and account changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Checks and formatting share one active owner with one executing snapshot and only the latest pending snapshot; checks debounce400ms and inactive owners release resources.
- [ ] #2 Whole-document admission enforces1MiB UTF-8 source, depth100,100 displayed diagnostics and a2-second deadline without false successful or incomplete states.
- [ ] #3 Real child or browser workers terminate and settle pending requests on deadline/crash/close, with no credential or filesystem authority forwarding or source-bearing logs.
- [ ] #4 Immutable authority/document/session/draft/language/request identities reject stale results across navigation and accounts; formatting uses the captured-source safety seam.
- [ ] #5 Targeted real lifecycle, initialization, cancellation, malformed-reply, alias-heavy and scheduler tests plus boundary copy/startup measurements are recorded.
- [ ] #6 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR paths: backlog/decisions/194-structured-note-language-and-local-editor-validation.md and Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (existing programme decisions)
Reason: implement bounded app-owned runtime/worker and privacy boundaries for the reviewed local engines; reuse and amend the programme decisions for any concretely qualified launcher correction before adoption.
1. Read task10 brief, binding worker constraints and preparation context, applicable guidance/lessons/ADRs, actual reviewed E1/E2/E3 exports and I1 captured-source/retirement seams. Confirm exact unit bases before production edits.
2. Trace existing owner/launch/lifecycle mechanisms. Qualify the real application launch for the plan’s spawn recipe, credential/main-module isolation and shutdown; report a concrete minimal boundary correction to controller before changed launcher adoption. Never temporarily mutate parent environment or inherit document-path authority.
3. Add meaningful RED controls with a real successful child/worker beside real stalled/crashed/malformed/cancelled/closed controls and bounded scheduler gates; fake timers may cover debounce only, never OS child death.
4. Implement one lazily owned engine service with active-editor leases, one executing snapshot and latest pending snapshot, a shared check/format lane,400ms check debounce, permanent retirement on authority/editability/conflict loss and full immutable identity matching. No new storage/transport authority or editor adoption.
5. Enforce1MiB UTF-8 admission/excerpt/surrogate bounds before dispatch, strict bounded source-safe protocol/results,100-depth/100-diagnostic incomplete limits and2-second actual termination/cleanup before replacement. Browser module/local-worker resolution and restrictive CSP must be qualified in actual consumers.
6. Run targeted engine/lifecycle/scheduler controls and relevant scoped static/format/type checks; profile actual startup/copy/encoding at1MiB, initialization close/cancellation/rapid navigation, source-safe failures and credential/handle cleanup with dedicated owned data. No full sweep or unchanged test reruns merely for evidence regeneration.
7. Write full task10 report and durable evidence/ADR notes with commands, output, exit statuses, exact runtime versions/support limits and self-review; commit explicit unit paths per runtime and remain In Progress for independent controller review.
Immutable unit bases: C 3ffaf527f304644e005b3c2ae8f529e274987417; S 12ac258bc9700687bec2679be079095349e23e53. E2 final source ba744f9059d30e26b1a025561f9dead3696a030c and E3 final source 4f1f7356c6e3bb69efae92e9361cf8044c1e2a0c have independently reviewed clean closure.
Ruling: Replace E4's multiprocessing spawn-context recipe with a fixed standard-library isolated Python subprocess for Chatbook, using the package-owned worker_process.py bootstrap, explicit empty/minimal allowlisted child environment, neutral launcher-owned cwd, close_fds and bounded JSON stdio frames, with one request per child and terminate/kill/reap plus pipe closure before replacement — actual module-entry TldwCli mounted probe failed first spawn at Textual stderr.fileno=-1; test-only fd-backed stderr allowed a child that inherited the synthetic secret, reimported app.py/config and loaded1782modules in1.949s, contradicting the binding document-only privacy boundary — cost if wrong: per-request interpreter/parser startup may slow checks and direct package bootstrap may fail in a supported install/platform; qualify real current startup/copy/deadline/cancellation/cleanup and installed/wheel paths before unit acceptance, preserve strict typed bounds and permanent stale retirement. This isolates application authority, not OS user privileges; no OS-sandbox claim, arbitrary launcher option, parent-env mutation, document path, custom sys.modules alias, pool, or new save/storage authority. Amend ADR194/065 and record the plan correction before production adoption.
Ruling: Add a trusted host-owned Worker factory callback to the browser DocumentWorkerClient, retaining Next's default static new Worker(new URL('./worker.ts', import.meta.url), {type:'module'}) and letting WXT's compiled host pass its native ?worker constructor — installed WXT0.20.13 defineImportMeta line8 rewrites import.meta.url to self.location.href before Vite detects the shared URL, and the actual extension positive failed requesting absent chrome-extension://.../worker.ts with zero worker events — cost if wrong: host injection may diverge or be omitted during production adoption; qualify both actual bundled factories/CSP/lifecycle paths now and require I3 to wire the one app-owned service/factory in each real host. Keep it a callback returning Worker, no user URL/CDN/blob option, worker-message factory, new runtime/service or plugin/config disable; preserve the shared bounded protocol/one lane/retirement/deadline contract and amend ADR065 before adoption. This browser-only correction creates no Python arbitrary-launch option.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented fixed isolated document children and one app-owned lease/scheduler lane with strict bounded source-safe JSON, permanent retirement and hard reap/pipe cleanup. ADR194/Ruling39;136 targeted tests, scoped Ruff/Black/Bandit/mypy and current mounted/wheel controls pass. Qualification: Docs/superpowers/reviews/2026-10-11-document-worker-ownership.md; full task10 report/evidence under .superpowers/sdd/2026-09-27-structured-document-editing. CPython3.12.11/macOS qualified; private subprocess transport dependency and no OS-sandbox claim explicit. I2 host capture/retirement wiring remains required. Keep In Progress pending independent review.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
