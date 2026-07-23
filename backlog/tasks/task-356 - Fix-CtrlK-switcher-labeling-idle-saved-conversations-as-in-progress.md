---
id: TASK-356
title: Fix Ctrl+K switcher labeling idle saved conversations as in-progress
status: Done
assignee: []
created_date: '2026-07-20 14:21'
updated_date: '2026-07-22 01:42'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
In the unfiltered Ctrl+K list, all saved conversations (nothing running, no open tabs for them) show the subtitle 'Chats - in-progress'; the rail simultaneously describes them as 'Chats - saved chat - 21m'. 'In-progress' falsely implies activity, and the two surfaces use different vocabulary for identical objects. The switcher also omits the age labels the rail has, weakening recognition where it matters most.

**Repro:** Open Ctrl+K with no query and compare each result's subtitle against the same conversation's subtitle in the rail Chats section.

**Verifier note:** Code-verified: the 'in-progress'→'saved chat' humanization map lives only in console_workspace_context.py (_STATUS_DETAIL_LABELS, lines 38-51); the switcher subtitle joins raw row.status (console_switcher_state.py:90-94), and persisted rows carry status=item.get('state') whose default is the internal lifecycle value 'in-progress' (chat_conversation_service.py:203). Also confirmed: builders set updated_sort but never updated_label, so switcher rows show no ages. The task-179 unification (chats-conversations-unified ledger) covered rail+Library only — the switcher never had the mapping, so NEW not REGRESSION.

**Source:** Console UX expert review 2026-07-20 (finding j2-switcher-mislabels-saved-chats-in-progress; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-35-switcher-unfiltered.png`, `j2-48-switcher-multi-match.png`, `j2-20-boot2-rail.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One consistent state vocabulary (saved chat / open session / active session) plus recency shown in both surfaces
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The rail mapped raw row status ("in-progress"/"workspace-thread") to a
friendly label ("saved chat") via a private table, but the Ctrl+K switcher
built its subtitle from the RAW status — so identical saved conversations
read 'in-progress' in the switcher and 'saved chat' in the rail. The
switcher also omitted the age label (its input rows carry no precomputed
updated_label).

Fix: a shared `console_conversation_status_detail()` in the neutral
`Workspaces/conversation_browser_state` module owns the one vocabulary
(saved chat / active session / open session); the rail's
`_conversation_detail_status` now delegates to it (behavior identical) and
`build_console_switcher_entries` maps through it too. The switcher also
derives recency from `updated_sort` (via `format_console_relative_age`,
new `now` param) when a row lacks a precomputed age label, so both surfaces
show recency.

Verified: 4 new switcher-builder unit tests (saved-chat vocab, membership/
session mapping, recency-from-updated_sort) + an end-to-end pilot test
opening the real switcher modal (renders 'saved chat', not 'in-progress');
rail suites unchanged (84 passed). Pure display logic — no live capture
needed. Files: `Workspaces/conversation_browser_state.py`,
`Chat/console_switcher_state.py`,
`Widgets/Console/console_workspace_context.py`.
<!-- SECTION:NOTES:END -->
