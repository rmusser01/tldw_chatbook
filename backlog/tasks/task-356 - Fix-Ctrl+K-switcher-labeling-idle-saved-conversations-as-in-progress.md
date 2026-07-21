---
id: TASK-356
title: Fix Ctrl+K switcher labeling idle saved conversations as in-progress
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
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
- [ ] #1 One consistent state vocabulary (saved chat / open session / active session) plus recency shown in both surfaces
<!-- AC:END -->
