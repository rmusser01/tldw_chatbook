---
id: TASK-355
title: Order rail conversations by last activity instead of creation time
status: To Do
assignee: []
created_date: '2026-07-20 14:21'
labels: [console, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
After sending a message into 'Long conversation about embeddings...' at 13:19, a fresh boot at ~13:27 listed it third from top in the same creation order as before, labelled '15m' (creation age) with no sign of the recent activity. Returning power users scan for 'the chat I was just in'; creation-ordered lists defeat that and the age label misrepresents recency of use.

**Repro:** Send a message in an old conversation, restart the app, compare rail order/ages: unchanged from creation order.

**Verifier note:** Root cause found in code: persisted rows populate updated_sort from item.get('updated_at') or created_at or last_updated (chat_screen.py:4093-4098), but normalize_conversation_row exposes only last_modified/created_at — no 'updated_at' key (chat_conversation_service.py:243-244) — so sorting (recency-first by design, _sort_normal_rows) and age labels silently degrade to creation time for all persisted rows; the Ctrl+K switcher ordering shares the same field. Present since the browser's introduction (commit 995a3f9ae, 2026-06-27), so NEW latent bug, not a REGRESSION of relative-age-labels (which pinned the label derivation and native-session updated_at, both still correct). One-line-class fix: include last_modified in the fallback chain.

**Source:** Console UX expert review 2026-07-20 (finding j2-rail-order-ignores-recency; P2, verdict NEW, high confidence) — full report `Docs/superpowers/qa/console-ux-expert-review-2026-07-17/README.md`, app under review at origin/dev cad9e271d, J2 returning power user journey. Evidence: `j2-20-boot2-rail.png`, `j2-12-reply-later.png` (regression/P1 captures committed alongside the report; the full 339-capture set is on the review machine).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Most-recently-active ordering (or a clear 'Recent' grouping) and age = last activity
<!-- AC:END -->
