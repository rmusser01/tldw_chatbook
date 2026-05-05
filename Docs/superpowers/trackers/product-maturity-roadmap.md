# Product Maturity Roadmap

Date: 2026-05-05
Status: Phase 1.1 verified; Phase 1 in progress
Source Branch: `dev`
Source Spec: `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`

## Purpose

Track product-depth maturity after Unified Shell Phase 6 so rendered screens, clickable controls, and complete usable workflows stay distinct.

## Current Verified Baseline

- Unified Shell Phase 0-6 are verified in `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`.
- Product-maturity work starts with a QA baseline before new feature depth.
- Phase 1.1 creates the reusable harness only; it does not complete the full first-run, focus, visual, empty-state, or core-loop audits.

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
- Phase 2: Core Agentic Loop - `TASK-9`
- Phase 3: Knowledge And Study Workflows - `TASK-10`
- Phase 4: Agent Configuration And Execution - `TASK-11`
- Phase 5: Server-Parity And Live Integrations - `TASK-12`
- Phase 6: Release Hardening And Documentation - `TASK-13`

## QA Evidence Index

| Phase | Evidence Path | Status |
| --- | --- | --- |
| Phase 1 | `Docs/superpowers/qa/product-maturity/phase-1/` | in_progress |

## Phase Overview

| Phase | Goal | Status | Backlog Tasks | QA Evidence | Residual Risk |
| --- | --- | --- | --- | --- | --- |
| Phase 1: QA Baseline And Usability Guardrails | Establish clean-run usability guardrails before feature depth. | in_progress | `TASK-8`, Phase 1.1 (`TASK-8.1`) | `phase-1/` | Phase 1.1 is harness-only; product walkthroughs remain future Phase 1 gates. |
| Phase 2: Core Agentic Loop | Complete source/question to grounded Console to Artifact/Chatbook loop. | planned | `TASK-9` | not-started | Depends on Phase 1 QA baseline. |
| Phase 3: Knowledge And Study Workflows | Mature ingest, organize, retrieve, study, and reuse workflows. | planned | `TASK-10` | not-started | Depends on Phase 2 core loop and later task slicing. |
| Phase 4: Agent Configuration And Execution | Mature Personas, Skills, MCP, ACP, Schedules, and Workflows. | planned | `TASK-11` | not-started | Depends on service adapters and runtime readiness. |
| Phase 5: Server-Parity And Live Integrations | Close high-value `tldw_server2` parity gaps. | planned | `TASK-12` | not-started | Requires parity inventory. |
| Phase 6: Release Hardening And Documentation | Reach release-candidate usability. | planned | `TASK-13` | not-started | Depends on earlier phase evidence. |

## Phase 1.1 Evidence

Status: verified

- `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-protocol.md`
- `Docs/superpowers/qa/product-maturity/phase-1/walkthrough-template.md`
- `Docs/superpowers/qa/product-maturity/phase-1/2026-05-05-phase-1-1-harness-smoke.md`
