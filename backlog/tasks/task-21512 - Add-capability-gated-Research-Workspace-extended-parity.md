---
id: TASK-21512
title: Add capability-gated Research Workspace extended parity
status: Done
assignee:
- '@codex'
created_date: 2026-08-24 05:54
updated_date: 2026-11-24 21:34
labels:
- research
- workspace
- parity
- ux
dependencies:
- TASK-21507
- TASK-21508
- TASK-21509
- TASK-21510
- TASK-21511
references:
- Docs/superpowers/specs/2026-08-23-research-workspace-design.md
- Docs/superpowers/plans/2026-08-23-research-workspace-extended-parity.md
- backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Complete the audited Research Workspace control namespace with honest capability-gated More outputs, work products, workspace/help controls, search, diagnostics, and owner links while keeping unsupported and planned features non-deceptive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `More outputs…` presents Learn, Analyze, and Present groups without duplicate commands and adds Mind Map, Timeline, Data Table, Slides, and Audio Summary only where a real generator and canonical owner exist.
- [x] #2 Executive Brief, Literature Matrix, Corpus Gap Finder, Evidence-Bound Hypotheses, and Research Proposal Pack become actions only when their adapter has a real generator and owner; otherwise each displays authority, owner, reason, and recovery.
- [x] #3 Output options are progressively disclosed by type and unsupported controls never execute a substitute action, switch authority, or claim persistence owned by another system.
- [x] #4 Workspace and Help menus, current-authority global workspace search, templates, collections, import/export/BibTeX, recent/pinned/archived state, banner/split preferences, status, onboarding, and keyboard help follow the approved core/context/owner-link/capability/planned classification.
- [x] #5 Create agent task, ACP history, sandbox diagnostics, MCP/ACP/provider/runtime remediation, Console handoff, Settings workspace management, Library, Artifacts, and Study actions navigate to their real owners instead of recreating management UI.
- [x] #6 Research Dossier, Competitive Market Memo, and Technical Project Spec are visibly Planned and are not focusable or dispatchable actions.
- [x] #7 Unknown or stale capabilities fail closed, refresh by capability revision, preserve the selected authority, and expose readable recovery; footer hints advertise only currently implemented actions.
- [x] #8 Wide/medium/narrow/short layouts keep the complete enabled control set keyboard reachable and screen-reader named without hiding authority, processing route, blocked state, or recovery.
- [x] #9 Targeted capability-matrix, owner-link, menu/search, output-owner, inert-planned-control, accessibility, geometry, and live supported-server-output tests pass without a full-suite claim.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes for the approved companion Server Data Tables contract

ADR path: `backlog/decisions/322-research-workspace-authority-and-screen-boundaries.md`

Reason: ADR-322 and the approved design already classify the audited controls and define their owners. The approved Data Tables slice uses the companion tldw_server ADR-099 (Docs/ADR/099-research-workspace-data-tables.md) for exact native Data Tables admission. The existing Chatbook ownership decision remains ADR-322.

Follow `Docs/superpowers/plans/2026-08-23-research-workspace-extended-parity.md` task-by-task with test-first checkpoints and one scoped commit per completed plan task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Data Table, Literature Matrix, and Corpus Gap Finder now use the exact Server route and captured Media revisions under companion Server ADR-099. Their Studio entries reference the native table UUID and published version and reopen native data for preview and export. Local keeps these actions unavailable. The Server feature currently requires server-managed credentials and usage quotas disabled.
- The focused Server suite passed 175 tests with PostgreSQL required. In Chatbook, 106 targeted Studio, geometry, and token tests passed, followed by 116 owner, schema, and keyboard checks with overlapping coverage. A localhost run with a stubbed model adapter recovered one successful launch after dropping its response and reopened and exported all three kinds through Data Tables.
- Server Slides uses the native structured Media generation route and visual-style catalog. Studio checks the selected Media revision before and after generation, then links the exact native presentation version. Localhost verified native preview, style, Markdown, and Reveal.js ZIP with a provider fixture. Thirty-four focused Chatbook tests passed. The structured endpoint has no replay receipt, so Studio blocks repeat submission when the first response is unconfirmed.
- Server Audio Summary drafts a bounded cited script, uses an explicit ready speech gateway and voice, and links the immutable native speech job and output artifact. Retry reconnects to a known job and Stop waiting leaves the native job available. Forty-six focused Chatbook tests passed before the final request-limit check, which passed 24 targeted tests. Localhost with a speech adapter fixture proved native worker dispatch, artifact lookup, and audio download.
- Local Audio Summary uses the current verified audio.cpp voice observation and exact synthesis, saves under Local TTS history, and links the saved history ID as a Workspace output. Failed links can retry without synthesizing again. Reopen, download, and deletion exercised real temporary history and Workspace storage with a speech runtime fixture. The targeted Local Studio, Audio Services, TTS admission, and mounted Studio run passed 113 tests without unsolicited runtime probes.
- Workspace and Help links use their existing owners. Open in Console selects the exact Local workspace and Create agent task starts a blank Console chat there. Status verifies the native Workspace origin before displaying a Run state, with recovery when refresh fails. Server workspace controls show recovery when their owner cannot verify a mutation or bundle.
- Voice preview sends a fixed sample through the selected speech owner. Local preview leaves TTS history untouched. Server Slides and Speech delete through the confirmed native owner, then mark the reference unlinked with a Workspace version check. A failed reference update can be retried. Native Data Tables keeps deletion in its owner.
- The control inventory exercises Workspace, Help, and More outputs at all six terminal sizes. It checks keyboard reach and measures disabled contrast. Painted-frame checks cover all four ASCII handles, including focused states, and the complete authority recovery message.
- Guarded the Research workers across awaits, switched Copy and Run launch timestamps to canonical UTC, and pinned both SQLite Copy receipt indexes using an actual query plan without statistics. Reviewed the persistent Research diagnostics. The added statements carry exception classes only.
- The final targeted passes covered 27 geometry tests, 37 Studio and Console checks, 29 Copy, Sharing, and Runs flows, and 56 owner and interoperability checks, with overlap. Repository preflight and the Ruff gates passed. Localhost exercised Data Tables retry and CSV, JSON, and XLSX export, native Slides export and deletion, and native Speech preview, download, and deletion. The Server suite passed 175 tests with PostgreSQL required.
- The three broader Console menu failures and the file-descriptor warning also reproduced on the starting revision. Ownership follows [ADR-322](../decisions/322-research-workspace-authority-and-screen-boundaries.md). Companion Server admission follows ADR-099 in tldw_server.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
