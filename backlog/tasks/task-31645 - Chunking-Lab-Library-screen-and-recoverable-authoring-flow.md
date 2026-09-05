---
id: TASK-31645
title: Chunking Lab - Library screen and recoverable authoring flow
status: In Progress
assignee:
  - '@codex'
created_date: '2026-09-04 23:13'
updated_date: '2026-09-05 18:31'
labels:
  - chunking
  - chunking-lab
  - ui
dependencies:
  - TASK-31641
  - TASK-31642
  - TASK-31643
  - TASK-31644
references:
  - backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
documentation:
  - Docs/superpowers/specs/2026-09-04-chunking-lab-design.md
  - Docs/superpowers/plans/2026-09-04-chunking-lab.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Ship the Library-owned single-sample A/B authoring screen by composing the tested execution, editing, persistence, save, and comparison seams. Covers all user-facing spec sections and AC 1-5, 11, 13, 15-17, 20, 23-26. Reconcile the existing TASK-24404 Settings proposal under ADR-118 before landing; ship one authoring surface. ADR required: yes; ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md; reason: route ownership, app lifecycle, and cross-module UI integration.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Library, palette, and eligible local extracted-text actions open a lazy dedicated Chunking Lab route and return to the opener; direct entry bypasses starter filtering and never creates a Settings or global destination.
- [x] #2 Users can paste, load UTF-8 text, or copy local extracted Library text; edit method controls or JSON; preview B, pin A, compare, and save either locally with no source mutation.
- [x] #3 Reopen, autosave failures, conflict, cancellation, template import/export, recovery export/restore/Undo restore, and confirmed Clear have working visible actions and truthful status; profile switching never mixes sessions.
- [x] #4 Keyboard-only flows work at 80x24, 120x40, and 160x50; typing r/p/s is safe in editors, F6/F1 follow global conventions, and the footer advertises only working actions.
- [x] #5 Saved templates immediately refresh the ingest picker; TASK-24404 is reconciled without a duplicate editor, v2/v3 actions are absent, and real isolated-profile live verification plus targeted regression evidence is recorded.
- [x] #6 Pure record-field edits and saved-record association preserve invalid JSON, pending controls, undo semantics and captured provenance; unsaved pinned records retain authored fields without fabricated catalog identity, and late saves cannot attach to an unrelated draft.
- [x] #7 Fallback to a previous valid recovery checkpoint is visibly explained through a read-only coordinator warning without exposing private sample/path details.
- [x] #8 Exclusive result preparation/inspection workers can be canceled before starting without leaking an unawaited coroutine.
- [x] #9 Catalog save acknowledgments identify exactly the revision written by that operation; an intervening peer update conflicts with the next unchanged local save.
- [x] #10 Malformed known recovery presentation fields are refused before replacement, while valid historical unsupported results remain readable after restore and reopen.
- [x] #11 Recovery import is inspectable after initial local-store failure and requires a validated summary plus explicit Replace current session confirmation before writable replacement.
- [x] #12 Users can explicitly inspect current and retained Previous output per candidate, including after failed or pending reruns, without substituting old output into current comparisons.
- [x] #13 Unfinished raw tag input survives ordinary renders and reopening, and deliberate save produces the intended separate tags.
- [x] #14 Final review refinements cover lazy screen workers, builtin Save-as-new default, visible sample provenance, preservation-only template export, and accurate final documentation status.
- [x] #15 The final targeted feature and compatibility selection passes the runtime mapper/import architecture guards without waiving the single-runtime-seam contract.
- [x] #16 All actionable PR 2416 review findings are verified and addressed with regression evidence; the rebased feature passes required checks before the user-authorized merge.
- [x] #17 The PR UI latency guard passes without raising its broad-selector ceiling, and narrowing Lab action selectors preserves keyboard and viewport behavior.
- [x] #18 Normal live Library entry finishes initialization even when mount handling yields; targeted regression and real A/B, template-save, and reopen/crash-recovery UAT pass.
- [ ] #19 PR 2421 startup budget correction keeps UI-ready imports at or below 972 without raising guard limits, preserves Environment refresh and vLLM handoffs, and passes current-head review and CI before merge.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
User-authorized PR 2421 follow-up: reproduce the inherited 976-module startup breach; add first-use import regressions; defer Environment gatherer construction until an admitted refresh and vLLM setup imports until target validation; run targeted feature, navigation, and full Perf Guard selections; inspect generated artifacts and current-head Qodo/CI before merging. ADR required: no new ADR. ADR path: backlog/decisions/097-boot-budget-ratchets.md; existing ADR-118 continues to govern the Lab. Reason: first-use import correction preserving existing contracts; no raised budget or new runtime boundary. Detailed plan: Docs/superpowers/plans/2026-09-05-chunking-lab-startup-budget.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the Library-owned local recoverable authoring screen under ADR-118 (backlog/decisions/118-chunking-lab-local-execution-and-recovery.md), with async single-flight profile ownership, exact local text handoff, serialized lossless edits, captured A/current B saves, canonical ingest refresh preserving selection, explicit recovery transfer and guarded navigation/quit. Approved narrow extensions add pure record-field/save-lineage transitions, read-only fallback warning and lazy/teardown-fenced ResultsRegion workers. TASK24404 was re-read unimplemented and archived with ADR/design note, not marked Done. User workflow/privacy/runtime limits are in Docs/Chunking_Lab.md. Final affected UI gate:36 passed45.17s; prior release selection103 passed with the subsequently fixed result teardown failure. Full requested integration:294 passed33 failed; exact BASE replay reproduced30, while3 startup/navigation differentials remain unqualified review concerns (not baseline-proven). Exact commands, RED/GREEN chronology, real SQLite/child/crash/failed-write/fresh-profile evidence, two accepted three-size viewport rounds, static audits and AC mapping: .superpowers/sdd/2026-09-04-chunking-lab/task-8-report.md. Scoped Ruff/format/compile/whitespace checks passed; no full sweep or unrelated repair. Status and AC acceptance remain In Progress pending independent review.

Task-level independent review complete after fix cd3e13926a: final-render edit ownership now rechecks the queue, with deterministic RED3/GREEN3 and final screen/recovery27 passing. Re-review found no new blocking issues. Controller explained the three startup differentials through active7-second splash versus readiness polling; exact3 no-splash intervention passes, untouched BASE reproduces same active-splash boundary. Original integration remains non-green; no blanket cold-start qualification. Durable qualifications and verification chronology: Docs/Chunking_Lab_Verification.md. Deferred Minor tag-entry normalization/eager screen workers go to the pending whole-branch review. All Task8 ACs accepted with documented existing-environment/platform limits; ADR118 unchanged.

Reopened for the single final whole-branch correction wave after review at462e5cc30e identified seven Important gaps and five Minor refinements. Task-level review history remains valid, but final completion awaits this correction and scoped re-review. Root startup diagnosis is documented separately and does not authorize unrelated startup repair.

Final correction wave implemented all R1–R7 and Minor1–5 under existing ADR-118
(`backlog/decisions/118-chunking-lab-local-execution-and-recovery.md`). Catalog writes
now acknowledge their exact transaction record. Recovery validates known fields,
previews the immutable selected snapshot and requires explicit replacement; an
unreadable local store still permits inspection. Historical results retain readable
counts/text/config/runtime. Current/Previous choices persist per candidate without
completing failed batches; successful Previous output survives repeated failures.
Raw comma-separated tags persist through edits/reopen/failed Save. Screen workers
are lazy and teardown-fenced, builtin saves default to copies, sample provenance is
visible, and unsupported authored objects export without runnable admission.
Design/ADR/plan/task chronology and user documentation are reconciled.

Final amended-code and directly affected compatibility selection: 344 passed,
1 existing RequestsDependencyWarning, 73.73s. Exact commands, each finding's RED/GREEN
evidence, changed interfaces, self-review and qualifications are in
`.superpowers/sdd/2026-09-04-chunking-lab/final-fix-report.md`; genuine JUnit output:
`final-fix-targeted.xml`. Scoped Ruff/format/whitespace checks passed. New controls
and dialogs have three-size focus/keyboard/geometry assertions; no new visual round,
full suite, environment upgrade, startup repair, merge or push. The original
294-pass/33-fail integration result, startup diagnosis, platform/resource limitations
and earlier privacy incident remain explicit in Docs/Chunking_Lab_Verification.md.
Status stays In Progress and AC9–14 remain unchecked until the controller's scoped
independent final re-review. Earlier pending-review statements above are chronology.

Final scoped re-review at 5d0df113ff accepted all seven Important and five Minor corrections, with no new breakage in that fix diff. AC9-14 are verified; status remains In Progress because the controller final targeted gate found 466 passes and two branch-introduced runtime mapper/import guard failures. Exact two-node rerun reproduces both failures. Task5 lab_runner.py:225 imports TemplateProcessor for resource admission outside the existing guard allowlist; it is not a second flat mapper, but the runtime-seam/guard contract is unresolved and must not be waived as baseline. No second final fix wave was dispatched. ADR118 and ADR078 remain applicable; final-rereview.md, controller-final-targeted.xml and controller-final-guard-repro.xml retain evidence. Docs/Chunking_Lab_Verification.md records current readiness. No Done, merge or push claim.

Final acceptance: runtime admission now calls narrow template_runtime preprocessing/sanitation adapters (9c3f69bd98) without vendor imports in the runner, widened guards, changed budgets or new policy. Independent Task1 review approved. The combined gate then exposed a separate local test-fixture startup-ownership race; measured enabled-splash Lab-to-Chat takeover was corrected only in the two manual-screen fixtures with real callback RED2/GREEN2 regressions (a89f14d6d3), independently reviewed. Final combined targeted feature/compatibility gate: 473 passed, zero failures/errors/skips, 2 known warnings in106.29s. Complete follow-up final review approved with no issues. Scoped lint/format/compile and whole-branch whitespace checks passed; legacy runtime lint debt and Requests/vendor warnings remain explicitly qualified. ADR078 and ADR118 apply; no new ADR for test-only setup. Docs/Chunking_Lab_Verification.md and follow-up plan record exact evidence/history/limits; lessons-live-verification.md records the measured manual-screen ownership incident. AC1-15 complete. No full sweep, production startup/shared factory change, vendor/dependency edit, merge or push. Historical pending notes above are superseded by this acceptance; earlier failures and privacy/platform/resource qualifications remain retained.

PR 2416 review corrections under existing ADR-078/ADR-118: bounded raw recovery admission now precedes full nested validation and pruning; template import refuses FIFOs including replacement races without modifying source permissions; record-field docs, structured comparison types and shared preview ceilings address the other three Qodo findings. The local ingest host now loads existing Library stylesheets, fixing two real compositor assertions without production styling changes. Combined targeted gate: 635 passed, 5 known warnings in157.71s; independent correction review found no findings and independently passed82 recovery/comparison tests and3 import/export cases. Scoped lint/format/whitespace pass. Rebase onto dev93388ba69b is complete. Docs/Chunking_Lab_Verification.md and follow-up plan record evidence and the user-authorized merge gates. AC16/status remain pending current-head remote review and required CI.

Qodo reviewed published4cf7869966 with zero remaining findings and resolved all5 threads; each has an evidence reply. Required CI33974600625 passed Fast Lane and Derived Artifacts. Perf Guard33974600652 separately failed276 broad-selector rules against274, reproduced locally. Narrowed only two fixed action selectors (EditorRegion and Library Lab-entry strip) to their existing IDs and regenerated widget_defaults_self.tcss; declarations and DOM unchanged. Exact RED276/GREEN274; combined full Perf Guard selection plus Lab screen/recovery modules passed75 with10 unsuppressed warnings in117.75s (pr-perf-correction.xml). Independent selector/declaration/composition/generation review found no issues; scoped lint/format/whitespace and all6 preflight checks pass. Verification docs retain warnings and zero headroom. Existing ADR078/118 apply; no new architectural decision. AC16 accepted on remote review/required-CI evidence; AC17 and task Done await current-head remote Perf Guard. No full sweep or merge yet.

Final code acceptance at13441478984bf17cd28f62172bce4f59df0f85d9: Qodo reports zero remaining issues on that exact head and all five threads are resolved. Required workflow33975848478 passed PR Fast Lane and Derived Artifacts (completed2026-09-05T16:02:57Z); Perf Guard33975848601 passed; six GGUF jobs and CSS/Backlog guards passed. Fresh fetch confirms dev93388ba69b ancestry. AC1-17 complete after independent reviews,635 feature/compatibility tests and75 performance/UI follow-up tests, scoped static checks and preflight; exact evidence and warnings remain in Docs/Chunking_Lab_Verification.md. ADR078 and backlog/decisions/118-chunking-lab-local-execution-and-recovery.md apply. Final bookkeeping is docs/task only; its current-head CI/review still gates merge. No full-suite or cross-platform claim; historical qualifications retained.

User-authorized post-merge UAT correction (2026-09-05): real PTY tracing proved the load worker could run during Mount with is_mounted=False and silently abandon initialization. Deferred only lazy worker dispatch with call_after_refresh, retaining teardown checks, exclusivity and error handling. Yielding-Mount regression RED readiness timeout; focused GREEN2; final screen/recovery/results GREEN66 in70.30s after the new test awaited its superseding render via existing settle helper. Scoped Ruff/format/whitespace pass. Independent read-only review: no findings, focused2 independently pass. Uninstrumented live UAT verifies provider-free Library entry, exact sample, advanced JSON/control preservation, six-versus-three chunk A/B with prefix execution and local backend metadata, Media DB template save, reopen, exact complete-state force-kill recovery, and invalid JSON recovery/discard. Docs/Chunking_Lab_UAT_2026-09-05.md records the precise coverage, historical failed run, warning and two unfixed non-blocking presentation findings (stale error strip and clipped Replace A). Added the measured Mount-order lesson. ADR required: no new ADR; backlog/decisions/118-chunking-lab-local-execution-and-recovery.md applies. No global startup/storage/runtime/dependency change, full sweep, existing user-data mutation, commit or publication. Plan: Docs/superpowers/plans/2026-09-05-chunking-lab-live-initialization.md.

PR 2421 user-authorized startup correction under ADR-097 (ADR-118 retained): deferred first-use Environment gatherers, vLLM setup types, Anthropic subscription helpers and custom-PII regex worker without changing ownership or auth/privacy/scheduler policy. Initial976 census; first4 deferrals left timing-dependent973/974; final970/972 after2 further first-use deferrals. Independent reviews report no findings. Bounded final runs: Lab66, Environment/handoff/setup322, subscription/readiness/PII/imports/census274, exact Perf Guard27, gateway customPII1 pass. vLLM workflow12 pass/1 fail: pending handoff line2551 identically fails on clean dev22006e84d; documented unrelated regression, not waived as passing. Broad combined run stalled and was stopped, not acceptance evidence. Six preflight checks and scoped static/whitespace pass; older API-related files retain unchanged baseline lint debt. Exact commands/selections, JUnit artifacts and warnings in Docs/Chunking_Lab_Verification.md; plan Docs/superpowers/plans/2026-09-05-chunking-lab-startup-budget.md. AC19/status remain pending current-head Qodo and CI. User authorizes merge after gates.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Formerly TASK-31428; moved to TASK-31645 during the user-approved
2026-09-05 pre-push bookkeeping correction. Upstream dev independently uses
31421–31424; the complete Lab chain moved together to preserve dependency
ordering without changing upstream tasks. Original creation dates, acceptance
and implementation history are retained. Historical commits and ignored review
artifacts retain the old IDs; current references use the new IDs. See
Docs/Chunking_Lab_Verification.md for the complete mapping and provenance.

## Publication bookkeeping notes (2026-09-05)

User-approved branch-only task renumbering and diagnostic-inventory correction;
no new ADR or implementation change (ADR-118 remains applicable). Reviewed the
new app cleanup diagnostic's exception-class-only argument and the explicit
private export sink before regenerating Docs/security/production-diagnostic-inventory.json.
All six preflight checks pass; artifact-checker tests pass74; source and test
files are unchanged. The prior fresh feature selection passed473. Additional
diagnostic-architecture coverage passed64/failed2/skipped1; both failures have
identical source-assertion failures on the exact feature BASE and remain
disclosed, not repaired or waived as passing. Exact scope, mapping and evidence
are in Docs/Chunking_Lab_Verification.md. PR publication is authorized; merge,
unrelated repairs, and a full-suite sweep are not.
