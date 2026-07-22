---
id: TASK-355
title: Order rail conversations by last activity instead of creation time
status: Done
assignee:
  - '@claude'
created_date: '2026-07-20 14:21'
updated_date: '2026-07-22 04:54'
labels:
  - console
  - ux
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
- [x] #1 Most-recently-active ordering (or a clear 'Recent' grouping) and age = last activity
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root cause (code-verified): persisted rail rows populated `updated_sort` from
`item.get("updated_at") or created_at or last_updated`, but the payload comes
from `normalize_conversation_row`, which emits `last_modified`/`created_at` and
has NO `updated_at` key — so every persisted row silently degraded to its
creation time. Both the rail's recency-first sort (`_sort_normal_rows`) and its
age labels (`format_console_relative_age`) read `updated_sort`, and the Ctrl+K
switcher shares the same rows, so a just-used conversation sorted/labelled as if
it had never been touched. Latent since the browser's introduction (995a3f9ae),
not a regression.

Fix: DRY the two identical inline fallback chains (chat_screen.py, the two
persisted-row builders) into one pure helper `console_persisted_row_updated_sort`
in `Workspaces/conversation_browser_state` that inserts `last_modified` into the
chain (`updated_at → last_modified → created_at → last_updated → ""`, None-safe).
`last_modified` is bumped to now() on every conversation write, so it is the
recency field; an explicit `updated_at` still wins for any caller that supplies
one. Native-session rows already sort by the real `session.updated_at` attribute
and are untouched.

Verified: 2 helper unit tests (last_modified beats created_at; full fallback
order incl. None-safety) + 1 integration test that runs a real
`normalize_conversation_row` payload through the helper and asserts both the root
cause (`"updated_at" not in normalized`) and the fix. Pure-data change with a
clear testable seam, so no served capture needed. Files:
`tldw_chatbook/Workspaces/conversation_browser_state.py`,
`tldw_chatbook/Workspaces/__init__.py`, `tldw_chatbook/UI/Screens/chat_screen.py`,
`Tests/Workspaces/test_console_conversation_browser_state.py`.
<!-- SECTION:NOTES:END -->
