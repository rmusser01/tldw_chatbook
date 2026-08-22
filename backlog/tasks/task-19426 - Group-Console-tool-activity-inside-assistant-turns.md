---
id: TASK-19426
title: Group Console tool activity inside assistant turns
status: Done
assignee: []
created_date: '2026-08-21 15:56'
updated_date: '2026-08-22 04:43'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make Console transcripts clearly attribute reasoning and tool activity to the assistant response that produced them, reducing ambiguity and transcript clutter.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each user query is followed by one visually coherent Assistant turn container.
- [x] #2 Tool and reasoning activity is rendered inside its owning Assistant turn.
- [x] #3 Tool and reasoning details are collapsed by default and can be expanded independently.
- [x] #4 The final assistant answer remains visible in the same Assistant turn after its activity rows.
- [x] #5 Existing tool-output expansion and message actions remain usable.
- [x] #6 Focused transcript tests and live visual verification cover completed, streaming, failed, and resumed turn shapes.
- [x] #7 Thinking rows never expose hidden chain-of-thought; absent or unsafe summaries render without a dead disclosure control.
- [x] #8 Keyboard selection and transcript pruning follow the rendered turn hierarchy without splitting or reversing a turn.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add session-only structured activity presentation plus the ADR-078 optional `ToolResult.outcome`/`AgentStep.tool_outcome` provenance contract. New runtime steps classify from review verdict and `ToolResult.ok` before result flattening; only legacy/malformed persisted steps fall back to direct-controller and `ERROR:`-wrapped provider-result parsing.
2. Derive privacy-safe intermediate Thinking markers with identical live/resume ordering for every primary step shape that proves tool work.
3. Add pure contiguous-message Assistant-turn grouping and visual selection order.
4. Build focused collapsed activity-disclosure and Assistant-turn widgets.
5. Integrate composite turns while preserving container/answer identity as the activity stack changes.
6. Make navigation, windowing, pruning, and plain export operate on whole rendered turns.
7. Add source/bundled TCSS and verify supported wide/narrow layouts.
8. Run focused, integration, baseline-aware lint/format, full-suite, live Console, self-review, and Backlog completion checks.

ADR required: yes
ADR path: backlog/decisions/078-structured-agent-tool-outcome-provenance.md
Reason: collision-safe status adds an optional internal provider/runtime fact that is serialized by the existing `dataclasses.asdict` -> schemaless steps-JSON path. ADR-078 records the status precedence, safe fallback for old/malformed step dictionaries, and why no SQLite or external provider-wire migration is required. Conversation marker persistence stays unchanged; ADR-031 still applies to keybinding/footer-hint truthfulness.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a single Assistant-owned Console turn surface that renders collapsed Thinking and tool activity before the visible answer while preserving original message identities, actions, Inspector selection, streaming reconciliation, branch ownership, navigation, pruning, export bounds, and resume behavior. Added privacy-safe Thinking derivation, session-only activity presentation, structured tool outcome provenance under ADR-078, and source/generated TCSS for a left-rail grouped surface with readable status and narrow-layout ellipsis.

Key decisions and tradeoffs:
- TOOL markers remain display-only and the persisted conversation tree remains unchanged.
- AgentStep tool outcome is an optional additive fact in existing schemaless run-step JSON; old or malformed rows use a safe compatibility classifier. No SQLite or external provider-wire migration.
- Activity headers use separate literal label/status children so the status stays readable while labels ellipsize.
- Transcript set_messages accepts optional session identity so recycled marker ids cannot retain expansion across session switches.
- Inspector resolves active-session display-only markers through the transcript only after the authoritative store lookup misses.
- Qodo follow-up converts both generic service permission gates to structured blocked results, documents all new public helper contracts, and caches causal spans once per transcript ingest instead of regrouping on each window/prune lookup.

Scoped verification, followed by a clean final rebase onto dev base f278a43c1:
- Complete focused changed-functionality suite: 856 passed, 2 warnings.
- Additional modified-code coverage for continuation persistence, models/review hook, local-server blocked outcomes, Inspector activity selection, and CSS integrity: 127 passed, 2 warnings.
- Adjacent Console suite: 323 passed. The two marker E2E `binding_unavailable` failures reproduced exactly on clean dev before any feature assertion runs.
- Post-rebase presentation, grouping, widget, all-theme contrast, and CSS bundle-integrity smoke: 203 passed, 1 warning.
- Final overlap-sensitive rebase checks for grouped turns, Inspector/session handoff, CSS parity, and the shared ChatScreen Change Review opener: 223 passed.
- Qodo review regressions: 571 affected agent/bridge/grouping tests and 141 transcript windowing/pruning tests passed; targeted Ruff lint and diff integrity passed.
- Changed Python compileall, Ruff lint, intentional new-file Ruff format gate, git diff check, and CSS source/bundle integrity all passed.
- Isolated live Console UAT exercised real fs_list/fs_read activity at wide and narrow sizes, mouse/Enter/Space/o disclosure parity, failed and successful runs, stable selection and scroll during streaming, Ctrl+K resume with identical collapsed ordering, and Inspector attribution for expanded and collapsed display-only activities. Evidence manifest: /private/tmp/task19426-uat-r2zoEDPZ/evidence/manifest.md.

Plan deviations discovered through review:
- Added ADR-078 and structured runtime outcome provenance after proving arbitrary successful tool content can collide with ERROR or denial copy.
- Added the active-session Inspector fallback after live UAT showed display-only marker selection had no Selected Message section.
- Per the final user direction, verification was limited to tests relevant to changed functionality and modified code; the repository-wide suite was not used as completion evidence.

No reusable lesson document was added: the privacy, live-verification, and test-evidence incidents are already covered by existing lessons, while the lasting runtime boundary decision is recorded in ADR-078.
<!-- SECTION:NOTES:END -->
