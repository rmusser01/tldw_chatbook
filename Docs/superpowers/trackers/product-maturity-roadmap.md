# Product Maturity Roadmap

Date: 2026-05-08
Status: Phase 1 verified; Phase 2 verified; Phase 3.0 verified; Phase 3.1 verified; Phase 3.2 verified; Phase 3.3 verified; Phase 3.4 verified; Gate 1 / Phase 3.5 verified; Gate 1.5 / Phase 3.6 verified; Phase 3.7 verified; Gate 1.6 / Phase 3.8 verified; Phase 3.9 verified; destination visual parity correction verified; Phase 4 verified; Phase 5.1 verified; Phase 5.2 verified; Phase 5.3 verified; Phase 5.4 verified; Phase 5.5 verified
Source Branch: `dev`
Source Spec: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
Layout Contract Spec: `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`

## Purpose

Track product-depth maturity after Unified Shell Phase 6 so rendered screens, clickable controls, and complete usable workflows stay distinct.

## Current Verified Baseline

- Unified Shell Phase 0-6 are verified in `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`.
- Product-maturity work starts with a QA baseline before new feature depth.
- Phase 1.1 creates the reusable harness only; it does not complete the full first-run, focus, visual, empty-state, or core-loop audits.
- Phase 1.2 verifies clean first-run launch and setup orientation only; broader Phase 1 gates remain open.
- Phase 1.3 verifies top-level destination reachability only; keyboard/focus, visual, empty/error/setup, and core-loop gates remain open.
- Phase 1.4 verifies keyboard reachability and fallback affordances only; visual, empty/error/setup, and core-loop gates remain open.
- Phase 1.5 verifies visual/chrome integrity across the supported size matrix only; empty/error/setup and core-loop gates remain open.
- Phase 1.6 verifies empty/error/setup-state coverage only.
- Phase 1.7 verifies the remaining narrow core-loop proof gate: Search/RAG result context can stage into Console with visible local source authority.
- Phase 2.1 verifies the first core-agentic-loop contract: staged Search/RAG context reaches the model-bound Console request and remains staged when send is blocked before generation starts.
- Phase 2.2 verifies the next core-agentic-loop contract: completed assistant responses can create local Chatbook artifact records with bounded Console provenance and recoverable failure notifications.
- Phase 2.3 verifies the next core-agentic-loop contract: Console-saved Chatbook artifact records can be recognized in Artifacts and reopened into Console with visible saved-response provenance.
- Phase 2.4 verifies the next core-agentic-loop contract: Home can surface a Console-saved Chatbook artifact as resumable work and route it to Artifacts or Console.
- Phase 2.5 verifies the complete local core-loop closeout: source/question context can reach Console, save to a Chatbook artifact, reopen through Artifacts, and resume from Home without manual state reconstruction.
- Phase 3.0 verifies the required design gate before additional Phase 3 Knowledge/Study visual rewrites: destination layout and IA contracts must be approved for affected screens or deviations must be reviewed.
- Phase 3.1 verifies the first Knowledge/Study entry contract: Library visibly routes users to Study Dashboard, Flashcards, and Quizzes while preserving the requested Study section.
- Phase 3.2 verifies Library-originated Study entry preserves visible source context in Study without changing deck or quiz service scope away from global or workspace.
- Phase 3.3 verifies the Library destination layout shell against the approved contract: mode bar, source browser, detail, inspector, authority, and existing Library actions remain visible across compact, default, and large terminal sizes.
- Phase 3.4 verifies the first source-selected Study generation contract: Library-selected note and media source items carry into Study Dashboard and can queue a server study-pack generation job with local-mode recovery.
- Gate 1 / Phase 3.5 verifies the core product-loop screen adaptation: Home dashboard regions, Console agentic shell regions, and actionable Library modes are mounted and usable enough to continue into required Gate 1.5 and Gate 1.6 follow-ups.
- Gate 1.5 / Phase 3.6 verifies the Console internals decomposition gate: Console now mounts native workbench components for controls, staged context, transcript/session, composer, run inspector, approvals, RAG/source state, and Chatbook artifact actions while retaining focused compatibility coverage for existing chat behavior.
- Phase 3.7 verifies the next source-selected Study generation contract: queued server study-pack jobs can be observed into completed pack metadata and visible reusable Study dashboard state.
- Gate 1.6 / Phase 3.8 verifies Library-native Search/RAG: Library owns the query/evidence workflow, retrieval results preserve snippets/citations/source authority, selected evidence stages into Console, and Console can invoke Library RAG or show a recoverable blocked state.
- Phase 3.9 verifies the Library Collections IA split: Watchlists is the top-level monitored-source destination, Collections is discoverable and locally manageable inside Library, sync is shown as local-only or sync-unavailable, and citations/snippets remain later-stage Library/Search/RAG work.
- The destination visual parity correction verifies the approved mounted shell geometry for Home, Console, Library, Artifacts, Personas, Watchlists, Schedules, Workflows, MCP, ACP, Skills, and Settings at `140x42` and `100x32`.
- Phase 5.4 verifies that Library Collections surfaces read-only sync mirror dry-run status and diagnostics from existing sync contracts without enabling write sync, mutation replay, or automatic merge behavior.
- Phase 5.3 verifies that Home distinguishes local notification queue state from server-owned observed event feed state, including replay-gap, reconnect, and unavailable recovery states without marking server events as read.

## Post-UX Roadmap Handoff

Source Spec: `Docs/superpowers/specs/2026-05-06-post-ux-product-roadmap-design.md`
Status: pending UX/UI completion and tracker activation

This post-UX roadmap is a planning overlay for work that starts after the current UX/UI and destination layout-contract work is complete. It does not create a parallel task tree.

Product Maturity Phase 1 and Phase 2 are not reopened by default. Their existing QA evidence is the pre-UX baseline. New post-UX evidence should record only changed screens, changed workflows, changed layout contracts, or discovered regressions.

| Post-UX Roadmap Stage | Existing Backlog Owner | Execution Rule |
| --- | --- | --- |
| Post-UX Reliability Rebaseline | `TASK-10` only if Phase 3 UX/UI deltas require a distinct gate | Revalidate deltas against verified Phase 1/2 evidence; do not recreate Phase 1/2 child tasks. |
| Source, Knowledge, And Artifact Loops | `TASK-10` | Continue Phase 3 under existing Knowledge/Study ownership. |
| Controlled Agent Configuration And Run Loops | `TASK-11` | Create PR-sized child tasks under Phase 4 when ready. |
| Monitoring And Cross-Loop Recovery | `TASK-10` or `TASK-11` based on the concrete workflow owner | Do not create standalone recovery rewrites without a workflow gate. |
| Server Parity And Live Integrations | `TASK-12` | Prioritize parity by workflow value, not endpoint count. |
| Release Hardening And Distribution | `TASK-13` | Use only after earlier workflow gates have QA evidence. |

## Severity Policy

| Priority | Taxonomy | Exit Rule |
| --- | --- | --- |
| P0 | `blocker` | Must be fixed before the phase or gate can close. |
| P1 | `workflow-degradation` | Must be fixed before phase close unless explicitly accepted with owner, rationale, and follow-up task. |
| P2 | `recoverability` | May remain only with residual risk and a scoped follow-up. |
| P3 | `polish` | May remain as backlog polish if it does not hide status, source authority, or recovery action. |

## Backlog Task Hierarchy

- Phase 1: QA Baseline And Usability Guardrails - `TASK-8`
- Phase 1.1: Canonical QA Harness - `TASK-8.1`
- Phase 1.2: Clean First-Run Launch And Configuration Walkthrough - `TASK-8.2`
- Phase 1.3: Top-Level Navigation Smoke Walkthrough - `TASK-8.3`
- Phase 1.4: Keyboard And Focus Sweep - `TASK-8.4`
- Phase 1.5: Visual Broken-State Audit - `TASK-8.5`
- Phase 1.6: Empty/Error/Setup State Coverage - `TASK-8.6`
- Phase 1.7: Narrow Core-Loop Proof - `TASK-8.7`
- Phase 2: Core Agentic Loop - `TASK-9`
- Phase 2.1: Grounded Console Response Contract - `TASK-9.1`
- Phase 2.2: Console Chatbook Artifact Save Contract - `TASK-9.2`
- Phase 2.3: Saved Chatbook Artifact Reopen Contract - `TASK-9.3`
- Phase 2.4: Home Chatbook Artifact Resume Contract - `TASK-9.4`
- Phase 2.5: Core Loop Closeout Replay - `TASK-9.5`
- Phase 3: Knowledge And Study Workflows - `TASK-10`
- Phase 3.0: Destination Layout And IA Contracts - `TASK-10.0`
- Phase 3.1: Library Study Entry - `TASK-10.1`
- Phase 3.2: Library Source Study Context - `TASK-10.2`
- Phase 3.3: Library Contract Layout Shell - `TASK-10.3`
- Phase 3.4: Source-Selected Study Generation - `TASK-10.4`
- Phase 3.5 / Gate 1: Core Product Loop Screen Adaptation - `TASK-10.5`
- Phase 3.6 / Gate 1.5: Console Internals Decomposition - `TASK-10.6`
  - Phase 3.6.1: Console Native Display-State Contracts - `TASK-10.6.1`
  - Phase 3.6.2: Console Native Controls And Staged Context - `TASK-10.6.2`
  - Phase 3.6.3: Console Native Transcript And Composer Surface - `TASK-10.6.3`
  - Phase 3.6.4: Console Run Inspector Approvals Tools And Artifacts - `TASK-10.6.4`
  - Phase 3.6.5: Console Internals QA Closeout - `TASK-10.6.5`
- Phase 3.7: Source Study-Pack Completion Reuse - `TASK-10.7`
- Phase 3.8 / Gate 1.6: Library Native Search/RAG - `TASK-10.8`
  - Phase 3.8.1: Library Search/RAG Display-State Contracts - `TASK-10.8.1`
  - Phase 3.8.2: Library Native Search/RAG Panel - `TASK-10.8.2`
  - Phase 3.8.3: Retrieval Adapter And Evidence Results - `TASK-10.8.3`
  - Phase 3.8.4: Console RAG Handoff And Invocation - `TASK-10.8.4`
  - Phase 3.8.5: Library Search/RAG QA Closeout - `TASK-10.8.5`
- Phase 3.9: Library Collections IA Split - `TASK-10.9`
  - Phase 3.9.1: Watchlists IA Split And Compatibility Labels - `TASK-10.9.1`
  - Phase 3.9.2: Library Collections Display-State And Local Service Contracts - `TASK-10.9.2`
  - Phase 3.9.3: Library Collections Mounted Management UI - `TASK-10.9.3`
  - Phase 3.9.4: Library Collections QA Closeout And Tracking - `TASK-10.9.4`
- Phase 4: Agent Configuration And Execution - `TASK-11`
- Phase 5: Server-Parity And Live Integrations - `TASK-12`
  - Phase 5.1: Server Parity Current-State Inventory - `TASK-12.1`
  - Phase 5.2: Active Server Auth Live Status - `TASK-12.2`
  - Phase 5.3: Server Events And Notifications Live Feed - `TASK-12.3`
  - Phase 5.4: Sync Mirror Dry-Run Workflow Surfacing - `TASK-12.4`
  - Phase 5.5: High-Value Domain Parity Workflows - `TASK-12.5`
  - Phase 5.6: Server Parity Live Integration Closeout - `TASK-12.6`
- Phase 6: Release Hardening And Documentation - `TASK-13`

## QA Evidence Index

| Phase | Evidence Path | Status |
| --- | --- | --- |
| Phase 1 | `Docs/superpowers/qa/product-maturity/phase-1/` | verified |
| Phase 2.1 | `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-1-grounded-console-response-contract.md` | verified |
| Phase 2.2 | `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-2-console-chatbook-artifact-save-contract.md` | verified |
| Phase 2.3 | `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-3-saved-chatbook-artifact-reopen-contract.md` | verified |
| Phase 2.4 | `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-4-home-chatbook-artifact-resume-contract.md` | verified |
| Phase 2.5 | `Docs/superpowers/qa/product-maturity/phase-2/2026-05-06-phase-2-5-core-loop-closeout-replay.md` | verified |
| Phase 3.0 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md` | verified |
| Phase 3.1 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-1-library-study-entry.md` | verified |
| Phase 3.2 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-2-library-source-study-context.md` | verified |
| Phase 3.3 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-3-library-contract-layout.md` | verified |
| Phase 3.4 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-phase-3-4-source-study-generation.md` | verified |
| Gate 1 / Phase 3.5 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md` | verified |
| Gate 1.5 / Phase 3.6 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md` | verified |
| Phase 3.7 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-phase-3-7-source-study-pack-completion-reuse.md` | verified |
| Gate 1.6 / Phase 3.8 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-6-library-native-search-rag.md` | verified; TASK-10.8.1 through TASK-10.8.5 done |
| Phase 3.9 | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-phase-3-9-library-collections.md` | verified; TASK-10.9.1 through TASK-10.9.4 done |
| Phase 3 visual parity correction | `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-destination-visual-parity-correction.md` | verified; text geometry evidence in `visual-parity/` |
| Phase 4 QA index | `Docs/superpowers/qa/product-maturity/phase-4/README.md` | verified |
| Phase 4.1 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-agent-execution-planning.md` | verified; TASK-11.1 done |
| Phase 4.2 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-2-personas-runtime-launch.md` | verified; TASK-11.2 done |
| Phase 4.3 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-3-skills-attach-validation.md` | verified; TASK-11.3 done |
| Phase 4.4 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-4-mcp-source-scope.md` | verified; TASK-11.4 done |
| Phase 4.5 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-12-phase-4-5-acp-runtime-session.md` | verified; TASK-11.5 done |
| Phase 4.6 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-15-phase-4-6-schedules-workflows-run-control.md` | verified; TASK-11.6 done |
| Phase 4.7 | `Docs/superpowers/qa/product-maturity/phase-4/2026-05-16-phase-4-7-agent-execution-closeout.md` | verified; TASK-11.7 done |
| Phase 5 QA index | `Docs/superpowers/qa/product-maturity/phase-5/README.md` | in-progress |
| Phase 5.1 | `Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-1-current-state-inventory.md` | verified; TASK-12.1 done |
| Phase 5.2 | `Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-2-active-server-auth-live-status.md` | verified; TASK-12.2 done |
| Phase 5.3 | `Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-3-server-events-notifications-live-feed.md` | verified; TASK-12.3 done |
| Phase 5.4 | `Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-4-sync-mirror-dry-run-workflow-surfacing.md` | verified; TASK-12.4 done |
| Phase 5.5 | `Docs/superpowers/qa/product-maturity/phase-5/2026-05-16-phase-5-5-high-value-domain-parity-workflows.md` | verified; TASK-12.5 done |

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Phase 1: QA Baseline And Usability Guardrails | Establish clean-run usability guardrails before feature depth. | verified | `TASK-8`, Phase 1.1 (`TASK-8.1`), Phase 1.2 (`TASK-8.2`), Phase 1.3 (`TASK-8.3`), Phase 1.4 (`TASK-8.4`), Phase 1.5 (`TASK-8.5`), Phase 1.6 (`TASK-8.6`), Phase 1.7 (`TASK-8.7`) | `phase-1/` | Closed; full grounded generation and Artifact/Chatbook persistence move to Phase 2. |
| Phase 2: Core Agentic Loop | Complete source/question to grounded Console to Artifact/Chatbook loop. | verified | `TASK-9`, Phase 2.1 (`TASK-9.1`), Phase 2.2 (`TASK-9.2`), Phase 2.3 (`TASK-9.3`), Phase 2.4 (`TASK-9.4`), Phase 2.5 (`TASK-9.5`) | `phase-2/2026-05-05-phase-2-1-grounded-console-response-contract.md`, `phase-2/2026-05-05-phase-2-2-console-chatbook-artifact-save-contract.md`, `phase-2/2026-05-05-phase-2-3-saved-chatbook-artifact-reopen-contract.md`, `phase-2/2026-05-05-phase-2-4-home-chatbook-artifact-resume-contract.md`, `phase-2/2026-05-06-phase-2-5-core-loop-closeout-replay.md` | Closed for the local core loop; live provider generation, full `.chatbook` export packaging, and full artifact history picking remain later-phase risks. |
| Phase 3: Knowledge And Study Workflows | Mature ingest, organize, retrieve, study, and reuse workflows. | in-progress; Phase 3.0 verified; Phase 3.1 verified; Phase 3.2 verified; Phase 3.3 verified; Phase 3.4 verified; Gate 1 / Phase 3.5 verified; Gate 1.5 / Phase 3.6 verified; Phase 3.7 verified; Gate 1.6 / Phase 3.8 verified; Phase 3.9 verified; destination visual parity correction verified | `TASK-10`, Phase 3.0 (`TASK-10.0`), Phase 3.1 (`TASK-10.1`), Phase 3.2 (`TASK-10.2`), Phase 3.3 (`TASK-10.3`), Phase 3.4 (`TASK-10.4`), Gate 1 / Phase 3.5 (`TASK-10.5`), Gate 1.5 / Phase 3.6 (`TASK-10.6`), Phase 3.7 (`TASK-10.7`), Gate 1.6 / Phase 3.8 (`TASK-10.8`), Phase 3.9 (`TASK-10.9`) | `phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`, `phase-3/2026-05-06-phase-3-1-library-study-entry.md`, `phase-3/2026-05-06-phase-3-2-library-source-study-context.md`, `phase-3/2026-05-06-phase-3-3-library-contract-layout.md`, `phase-3/2026-05-07-phase-3-4-source-study-generation.md`, `phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md`, `phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md`, `phase-3/2026-05-07-phase-3-7-source-study-pack-completion-reuse.md`, `phase-3/2026-05-07-gate-1-6-library-native-search-rag.md`, `phase-3/2026-05-08-phase-3-9-library-collections.md`, `phase-3/2026-05-08-destination-visual-parity-correction.md` | Layout contracts, Library Study entry, Library source context, Library contract layout shell, source-selected server study-pack job launch, core Home/Console/Library screen adaptation, Console-native internals, completed study-pack reuse state, Library-native Search/RAG with Console evidence handoff/invocation, Library-owned local Collections management, and mounted destination visual parity at compact/default sizes are verified. Full server job history, direct generated deck selection, Workspaces, deeper Import/Export, full server sync, collection item membership, deeper Study/Search/RAG flows, citations/snippets in downstream collection workflows, and Citation/snippet carry-through into Chat, artifacts, and exported Chatbooks remain. Post-UX roadmap execution should continue under TASK-10 unless the tracker explicitly creates a rebaseline child task for UX/UI deltas. |
| Phase 4: Agent Configuration And Execution | Mature Personas, Skills, MCP, ACP, Schedules, and Workflows. | verified; TASK-11.1 through TASK-11.7 done | `TASK-11`, Phase 4.1 (`TASK-11.1`), Phase 4.2 (`TASK-11.2`), Phase 4.3 (`TASK-11.3`), Phase 4.4 (`TASK-11.4`), Phase 4.5 (`TASK-11.5`), Phase 4.6 (`TASK-11.6`), Phase 4.7 (`TASK-11.7`) | `phase-4/2026-05-12-phase-4-agent-execution-planning.md`; `phase-4/2026-05-12-phase-4-2-personas-runtime-launch.md`; `phase-4/2026-05-12-phase-4-3-skills-attach-validation.md`; `phase-4/2026-05-12-phase-4-4-mcp-source-scope.md`; `phase-4/2026-05-12-phase-4-5-acp-runtime-session.md`; `phase-4/2026-05-15-phase-4-6-schedules-workflows-run-control.md`; `phase-4/2026-05-16-phase-4-7-agent-execution-closeout.md`; plan `Docs/superpowers/plans/2026-05-12-phase-4-agent-configuration-execution.md` | Verified for local and honest-blocked agent configuration and execution control surfaces. ACP runtime launch, full Schedules and Workflows run-control services, and server parity remain explicit Phase 5 risks. |
| Phase 5: Server-Parity And Live Integrations | Close high-value `tldw_server2` parity gaps. | in-progress; TASK-12.1, TASK-12.2, TASK-12.3, TASK-12.4, and TASK-12.5 verified | `TASK-12`, Phase 5.1 (`TASK-12.1`), Phase 5.2 (`TASK-12.2`), Phase 5.3 (`TASK-12.3`), Phase 5.4 (`TASK-12.4`), Phase 5.5 (`TASK-12.5`), Phase 5.6 (`TASK-12.6`) | `phase-5/2026-05-16-phase-5-1-current-state-inventory.md`; `phase-5/2026-05-16-phase-5-2-active-server-auth-live-status.md`; `phase-5/2026-05-16-phase-5-3-server-events-notifications-live-feed.md`; `phase-5/2026-05-16-phase-5-4-sync-mirror-dry-run-workflow-surfacing.md`; `phase-5/2026-05-16-phase-5-5-high-value-domain-parity-workflows.md`; plan `Docs/superpowers/plans/2026-05-16-phase-5-server-parity-live-integrations.md` | Current dev now has verified active-server/auth status, server event/feed presentation, sync dry-run surfacing, domain-edge, UX-contract foundations, and source-honest Library/Search/RAG to Console handoff authority. Phase 5 remains open for closeout. ACP runtime launch and Schedules/Workflows run-control remain explicit risks; write sync remains deferred. |
| Phase 6: Release Hardening And Documentation | Reach release-candidate usability. | planned | `TASK-13` | not-started | Depends on earlier phase evidence. The post-UX roadmap public-roadmap and release-readiness refresh belongs under TASK-13. |

## Phase 1.1 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`
- `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`
- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`

## Phase 1.2 Evidence

Status: verified

- `Docs/superpowers/plans/2026-05-05-product-maturity-phase-1-2-first-run-walkthrough.md`
- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-2-first-run-walkthrough.md`

## Phase 1.3 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-3-navigation-smoke.md`

## Phase 1.4 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-4-keyboard-focus.md`

## Phase 1.5 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-5-visual-broken-state-audit.md`

## Phase 1.6 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-6-empty-error-setup-states.md`

## Phase 1.7 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-7-core-loop-proof.md`

## Phase 2.1 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-1-grounded-console-response-contract.md`

## Phase 2.2 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-2-console-chatbook-artifact-save-contract.md`

## Phase 2.3 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-3-saved-chatbook-artifact-reopen-contract.md`

## Phase 2.4 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-2/2026-05-05-phase-2-4-home-chatbook-artifact-resume-contract.md`

## Phase 2.5 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-2/2026-05-06-phase-2-5-core-loop-closeout-replay.md`

## Phase 3.0 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-0-destination-layout-contracts.md`

## Phase 3.1 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-1-library-study-entry.md`

## Phase 3.2 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-2-library-source-study-context.md`

## Phase 3.3 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-phase-3-3-library-contract-layout.md`

## Phase 3.4 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-phase-3-4-source-study-generation.md`

## Gate 1 / Phase 3.5 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-06-gate-1-core-product-loop-screen-adaptation.md`

## Gate 1.5 / Phase 3.6 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-5-console-internals-decomposition.md`
- `Docs/superpowers/plans/2026-05-07-gate-1-5-console-internals-decomposition.md`

## Phase 3.7 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-phase-3-7-source-study-pack-completion-reuse.md`

## Gate 1.6 / Phase 3.8 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-07-gate-1-6-library-native-search-rag.md`
- `Docs/superpowers/plans/2026-05-07-gate-1-6-library-native-search-rag.md`

## Phase 3.9 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-phase-3-9-library-collections.md`
- `Docs/superpowers/plans/2026-05-08-phase-3-9-library-collections-ia-split.md`

## Destination Visual Parity Correction Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-3/2026-05-08-destination-visual-parity-correction.md`
- `Docs/superpowers/plans/2026-05-08-destination-visual-parity-correction-implementation.md`
