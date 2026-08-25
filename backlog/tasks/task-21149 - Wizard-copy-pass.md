---
id: TASK-21149
title: Wizard copy pass
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:15'
updated_date: '2026-08-25 23:24'
labels:
  - ux
  - wizard
  - copy
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings W-2, W-3, W-4, P-3, P-4, V-4, R-1, R-3, S-3, S-5, N-3, N-4, G-5 (findings.md): track options lack effort estimates and outcome descriptions; no app value proposition; key helper text leads with env-var/TOML phrasing and offers no where-to-get-a-key pointer; 'A replacement API key is ready' on fresh installs; RAG install guidance omits the exact pip command; duplicated Speech guidance; summary provider line unnamed; casing drift; Skip/Exit label drift; splash cards shown as raw snake_case with no show-all.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every finding listed in the description has its copy updated per the findings.md recommendation
- [x] #2 Docs/User_Guide setup-wizard page updated (or its Verified-against stamp refreshed)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Implement W-2 (estimates in subtitle), W-3 (value prop), P-3 (key helper inversion + get-key pointers), P-4, V-4, R-1, R-3, S-3 (named provider row), G-5 (humanized card names + show-all); document N-3/N-4/S-5/W-4 dispositions. Update pinned tests; suites; live spot-check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented: W-3 value-prop line + W-2 honest time estimates in the Welcome subtitle (labels stay inside their 61-char no-wrap budget); P-3 key helper inverted — 'An API key is needed — paste it above. New keys: <provider url>. (Already exported X? It's picked up automatically.)' with a _PROVIDER_KEY_URLS map for 8 cloud providers, applied at the wizard surface so Console's shared readiness copy is untouched; P-4 'Key staged — it will be checked when you continue.'; V-4 needs-test family reworded ('Not tested yet — that's fine…' + failure/cancel variants kept consistent); R-1 RAG copy glosses what RAG does and shows the exact pip command; R-3 Speech de-duplicated ('Optional; Next skips it.'); S-3 Summary names the provider ('✓ Provider — OpenAI'); G-5 splash cards render Title Case names with the raw id riding the button (theme-pattern) plus a 'Show all cards…' parity button.

Dispositions documented rather than changed: N-3 (Skip vs Exit labels carry real semantic difference, dialogs explain it), N-4 (moving the rerun entry out of Settings ▸ Diagnostics is a Settings-IA change, out of wizard scope), S-5 ('Explore Home' keeps the proper-noun tab name), W-4 (outcome framing satisfied by the new value-prop + estimates lines; the label itself must stay within the no-wrap budget).

Suites: 882 passed + the known order-dependent flake (passes isolated). Live-verified Welcome + key-helper copy at 140x40. Lessons entry added (visual focus order + hidden-class trap).
<!-- SECTION:NOTES:END -->
