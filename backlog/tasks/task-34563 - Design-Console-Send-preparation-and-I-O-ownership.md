---
id: TASK-34563
title: Design Console Send preparation and I/O ownership
status: In Progress
assignee:
- '@codex'
created_date: 2026-10-06 21:51
updated_date: 2026-10-15 18:28
labels:
- console
- architecture
- performance
dependencies: []
documentation:
- Docs/Development/2026-10-06-console-send-architecture-review.md
- Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md
- backlog/decisions/230-console-send-preparation-and-io-ownership.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design a simpler and extensible Console Send process that gives immediate terminal feedback and removes repeated preparation I/O while preserving durable acceptance, permissions, recovery and cancellation.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The reviewed design defines one owner for each lifecycle stage, shared input and storage effect, with explicit data and authority freshness rules.
- [ ] #2 The design covers approval, refusal, errors, retry, navigation, concurrent sessions, Stop and shutdown without duplicating existing state authorities.
- [ ] #3 Extension contracts and phased migration preserve supported entry points and specify targeted correctness and real input/render/performance evidence.
- [ ] #4 The written specification and architectural decision are reviewed with the user before an implementation plan or product changes begin.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes.
ADR path: backlog/decisions/230-console-send-preparation-and-io-ownership.md.
Reason: received-request custody, shared domain results, native outcome ownership and unknown-authority handling cross subsystem interfaces. Extends ADR-094/098/126/148/163/197/220.
1. Read current runtime, preparation, storage and relevant ADRs (completed).
2. Compare approaches and review the ownership/data flow with the user (completed).
3. Review freshness, lifecycle/errors, extension and migration/verification contracts; incorporate requested reviews (completed).
4. Write/commit the consolidated spec and ADR; complete the user's written-spec review and requested issue audit, correct the concrete gaps, and validate document scope/links (completed).
Product implementation and implementation-plan execution are outside this design task; implementation-plan review remains required before product changes.

Current PR3023 integration: follow Docs/superpowers/plans/2026-10-11-pr3023-current-dev-integration.md on actual sequential current dev. ADR required: existing accepted230, accepted bounded amendment231, exact126/097semantic/179/197/220 as applicable; general231 remains Proposed. Preserve older source history/qualification above and all76optimizationIDs/14followups. Current acceptance awaits per-owner disposition, original native/caller/lifetime/security controls, unchanged budgets and separately qualified final saved-source/provider/rendered evidence; no whole-stack replacement, blanket closure or raised cap.
Current-session connected footprint closure: faithfully relocate the nine ordinary native Controller save/cancellation/recovery methods to existing console_native_commit and capture_turn_configuration_snapshot to existing console_configuration_preparation; relocate the four Store native owner/accepted-terminal/dispatch recovery methods to console_native_commit. Preserve real self fields, locks, generation, claims, checkpoints, rollback and finally retirement; public signatures/defaults/custom callbacks remain. E pr3023-complete-integration-connected-relocation-r5-proof.json binds exact pre-source/prospective bytes,14 normalized AST-exact bodies, untouched class nodes and live original-module global seams. Capture the relocated stock settlement helper at the existing finite native callback boundary, keeping custom callbacks unchanged. Extend qualification/test reader snapshots with every moved callee/code/default/global seam and exercise held source mutation/custom callback controls; no new blanket production policy. ADR required: no; ADR path: N/A; reason: mechanical relocation within existing owned services without new source authority/service contract. Run connected native/configuration/custody targeted checks, cold imports and differential owned lint, then lower structural ceilings to exact final counts; original ceilings never increase.
Independent connected relocation review51da... found sole default-ABI blocker: captured stock helper partial bypasses preinstalled/held original same-function metadata_json defaults, including invalid nonobject JSON refusal. Preserve R1 proof/binding/63P/2P/2P and passive causal REDs. Narrow existing finite binding repair: keep actual original captured bound callback and live default argument binding; retain only the actual relocated settlement-helper reference inside existing _ConsoleNativeBinding/contextvar at the same finite capture boundary. Wrapper forwards actual bound metadata explicitly to that retained helper for the same actual store; all ordinary/custom signatures/callbacks, custody/source/locks/generation/refusal/rollback/retirement remain. No new blanket source predicate/refusal/authority. Qualify valid/invalid preinstalled and held defaults, explicit metadata override, held helper-slot/custom original controls and appropriate63 connected native/configuration controls. ADR required: no; ADR path: N/A; reason: faithful preservation of original callback/default ABI within existing finite native retention owner.
Held-delete review correction: preserve R2 snapshot and causal ImportError receipt, resolve the existing native binding before any live settlement-helper lookup, and import the live helper only on the original uncaptured fallback. Retain actual bound callback/default/custom/explicit argument semantics and physical retirement. Run the ten connected controls and affected native integration scope; freeze R3 for independent review. ADR required: no; existing faithful connected relocation, no new source predicate or authority.
### Current governed wiring connected relocation, 2026-10-14

ADR required: yes, bounded amendments to existing ADR230 and ADR126. Reason:
actual defining-class/module interfaces and existing consumed source-reader
ownership move, with original service/permission/physical lifetime policy.

1. Preserve exact app_service_wiring before SHAee9acfea and current
configuration preparation before SHAaedf5abb. Independent coherent R2
reviewf17aa6ea accepts exact proposal4be7147c after the causal stock held-pointer
RED2F and prospective3P7.123s; frozen R1/R9 negatives remain evidence.
2. After these ADR/CLI plan amendments, verify all four before hashes and apply
only the four frozen R2 sources: genuine Skill/Collections mixins, actual module
functions at original importing slots, existing source records using real
defining owners and six explicit pointer-ordering corrections. No unrelated
Watchlists body extraction, successor value copies or blanket callable/default
policy. Keep original public properties, MRO/custom shadows, retained callbacks
and actual live original module globals.
3. Re-run affected original Skill/cold trust and Collections source/native/
cancellation/borrowed/first-use/shutdown/custom/import controls, then the causal
held-pointer controls against actual saved source. Preserve original assertions,
deadlines, source/default negatives and physical creator/borrower retirement.
4. Complete affected differential static/import/voice and source disposition
streams. Governed wiring3509 is prospective measurement only; original cap3601
remains until final current-base source pins. Retain all664/971/77/14 rows and
89docs/40tasks; no whole task/status/AC/native/latency completion from this slice.
Current reviewed connected Collections TEST-ONLY observer correction: proposal6b000cd95d223fe5580a3bea91fc860c776846f874cd2b119b1ef4112beefe65 preserves77P3F76.969 negative. Discover exact actual defining AST Raise(Call(Name _CollectionsSetupObsolete OR Attribute(Name _source_wiring,_CollectionsSetupObsolete))) lines with correct inspect source offset; passive callback checks original function/code/globals/frame and records errors without changing producer behavior. Retain every declined/outcome/resource/count/deadline/restoration assertion. After import hygiene, run3 affected negatives and whole80-case connected cohort; R1 downstream short-circuited assertions are not claimed passing.
Reviewed TEST-ONLY static wiring source reader: after exact original3402197/prospective8918704 hashes, install only Tests/app_module_patches.py bounded literal-vars installer publications; retain runtime equality/lazy absence oracle c59b9 unchanged. Run original10 import/census/patch-target controls then final affected233; preserve original9P1F and unchanged1031 ceiling. ADR required: no new ADR; fixture-only source interpretation under existing ADR230/126, no production contract or assertions changed.
Reviewed source-bound namespace importer fixture: preserve original9P1F19.47 raw_source_selection_changed and all runtime equality/lazy assertions; after exact c59b9/792b96 source checks add ONLY existing bootstrap_profile marker to test_source_derived_bindings_match_runtime_globals. Original10 controls run after final233; no production/config authority/census cap change. ADR required:no new; TEST-ONLY existing fixture prerequisite under ADR126.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Designed the Console Send preparation and I/O ownership architecture around existing runtime/controller/store authorities. The specification defines lightweight received-intent custody, domain-owned immutable results, live authority checks, cancellation/commit outcome settlement, recovery, extension contracts and phased qualification. ADR-230 records the architectural decision and intentional unknown-authority hardening.
The requested written-spec audit corrected atomic admission/promotion across entry points, approval/native-custody enablement prerequisites, attempt-scoped tool ceilings versus existing restart recovery, revision-guarded UI clearing and attachment-prefix compatibility, upstream error propagation, demand-driven capture and raw performance accounting. No new schema or stronger cross-process transaction is claimed.
Files: Docs/superpowers/specs/2026-10-06-console-send-preparation-architecture-design.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md; Docs/Development/2026-10-06-console-send-architecture-review.md.
Validation: eight relative document links, placeholder and whitespace checks, source-based contract review and scoped Git diff checks. This is documentation-only work; runtime/unit/integration and performance acceptance belong to the implementation tasks and are not claimed here. No full suite was run. Existing diagnostic/product candidates remain outside this task.

Current-dev integration 2026-10-15: rebased the reviewed Console work onto dev7c8e0e53 while preserving the incoming deletion wave and separate Dreams boundary job. Retained the existing Skills recovery equality guard at its moved skills_trust_modals owner and retargeted its original regression; updated only the exact timer receiver/root declarations and added the twelve existing PR3023 required targets to the response-rule CI inherited argv list. Original targeted RED343 had5 failures/338 passes; the same343 controls now pass with zero failures/errors/skips, followed by33 focused polling/Interop Outputs/fake-audio TTS consumer passes. Scoped Ruff I,F,E9 passes; two unchanged formatter debts remain. Independent source/evidence review72e01837 and Python3.12 data-only compile/import scan339 surviving changed files (zero references to31 removed modules) are recorded in the review evidence. Existing ADR230/126/150 apply; no new ADR, authority/layout/cadence/budget change, task closure, whole-suite or final provider/render/performance/platform acceptance is claimed.
<!-- SECTION:NOTES:END -->
