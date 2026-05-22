# Chatbook Product Roadmap

Chatbook is evolving into a local-first agentic knowledge console: a place to ingest sources, reason over them, configure controlled agentic work, monitor progress, and preserve useful outputs as durable artifacts.

This roadmap is directional. It describes current priorities and likely future areas, not delivery dates or commitments.

## Current Release Baseline

The current release baseline is usable as a local-first product loop:

- Home is the default status and notification surface for setup state, active work, notifications, and next actions.
- Console is the live agentic control surface for chat, source handoff, approvals, run follow-up, and durable Chatbook creation.
- Library is the source, Search/RAG, import/export, and Collections surface for preparing grounded work.
- Chatbooks and other durable outputs live under Artifacts so completed work can be reopened and resumed.
- Personas, Skills, MCP, ACP, Watchlists, Schedules, Workflows, and Settings stay visible as distinct product areas instead of hidden advanced settings.

## Current Limits And Recovery

Some capabilities remain intentionally source-honest rather than overpromised:

- ACP runtime launch and write sync remain future work.
- Server-backed functionality is surfaced only when configured; local mode remains a valid default.
- Optional dependency groups must be installed before advanced media, RAG, MCP, or web capabilities are available.
- Missing providers, missing sources, missing runtimes, and unavailable optional features should show the responsible surface and a recovery path.

## Deferred Tranche Gates

These areas are tracked as staged future work after the post-release actual-use audit. They are not counted as shipped behavior until each tranche has its own implementation evidence and actual app QA:

- ACP runtime launch for task/run package handoff.
- write sync promotion with explicit preview, conflict, rollback, and authority controls.
- Workspaces and Library depth, including workspace ownership labels, Collections membership, and deeper Import/Export without hiding cross-workspace Library items.
- citation and snippet carry-through from Library/Search-RAG into Console answers, Artifacts, and exported Chatbooks.
- optional dependency and package polish for clearer advanced-feature recovery paths.

## Now: Reliability And Product Confidence

Current focus:

- first-run setup and configuration clarity.
- stable layouts across terminal sizes.
- keyboard-first navigation and focus behavior.
- understandable empty, error, and blocked states.
- clear local, server, workspace, and runtime authority labels.
- repeatable QA coverage for core workflows.

## Next: Complete Workflow Loops

Next focus:

- ask grounded questions over selected sources.
- turn answers into reusable Chatbooks and artifacts.
- organize knowledge with Workspaces and Collections.
- generate and reuse flashcards, quizzes, reports, and study outputs.
- configure personas, skills, tools, schedules, and workflows.
- launch and monitor controlled agent work through Console.
- recover cleanly when providers, runtimes, or optional capabilities are missing.

## Later: Server-Backed And Live Capabilities

Longer-term focus:

- richer source and RAG integration.
- server-assisted watchlists and collections.
- live status updates for running work.
- import and sync paths for sources, personas, skills, and artifacts.
- clearer collaboration between local and remote runtimes.
- documented residual gaps where local mode remains the better default.

## Always: Local-First Control

Across every area, Chatbook should keep source authority, runtime readiness, approvals, recovery paths, and generated outputs visible.

Console remains the live work surface. Other areas prepare, inspect, organize, resume, or preserve work.
