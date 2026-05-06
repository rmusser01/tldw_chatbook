# Product Maturity Roadmap

Date: 2026-05-06
Status: Phase 1 verified; Phase 2 verified
Source Branch: `dev`
Source Spec: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

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
- Phase 4: Agent Configuration And Execution - `TASK-11`
- Phase 5: Server-Parity And Live Integrations - `TASK-12`
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

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Phase 1: QA Baseline And Usability Guardrails | Establish clean-run usability guardrails before feature depth. | verified | `TASK-8`, Phase 1.1 (`TASK-8.1`), Phase 1.2 (`TASK-8.2`), Phase 1.3 (`TASK-8.3`), Phase 1.4 (`TASK-8.4`), Phase 1.5 (`TASK-8.5`), Phase 1.6 (`TASK-8.6`), Phase 1.7 (`TASK-8.7`) | `phase-1/` | Closed; full grounded generation and Artifact/Chatbook persistence move to Phase 2. |
| Phase 2: Core Agentic Loop | Complete source/question to grounded Console to Artifact/Chatbook loop. | verified | `TASK-9`, Phase 2.1 (`TASK-9.1`), Phase 2.2 (`TASK-9.2`), Phase 2.3 (`TASK-9.3`), Phase 2.4 (`TASK-9.4`), Phase 2.5 (`TASK-9.5`) | `phase-2/2026-05-05-phase-2-1-grounded-console-response-contract.md`, `phase-2/2026-05-05-phase-2-2-console-chatbook-artifact-save-contract.md`, `phase-2/2026-05-05-phase-2-3-saved-chatbook-artifact-reopen-contract.md`, `phase-2/2026-05-05-phase-2-4-home-chatbook-artifact-resume-contract.md`, `phase-2/2026-05-06-phase-2-5-core-loop-closeout-replay.md` | Closed for the local core loop; live provider generation, full `.chatbook` export packaging, and full artifact history picking remain later-phase risks. |
| Phase 3: Knowledge And Study Workflows | Mature ingest, organize, retrieve, study, and reuse workflows. | planned | `TASK-10` | not-started | Depends on Phase 2 core loop and later task slicing. |
| Phase 4: Agent Configuration And Execution | Mature Personas, Skills, MCP, ACP, Schedules, and Workflows. | planned | `TASK-11` | not-started | Depends on service adapters and runtime readiness. |
| Phase 5: Server-Parity And Live Integrations | Close high-value `tldw_server2` parity gaps. | planned | `TASK-12` | not-started | Requires parity inventory. |
| Phase 6: Release Hardening And Documentation | Reach release-candidate usability. | planned | `TASK-13` | not-started | Depends on earlier phase evidence. |

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
