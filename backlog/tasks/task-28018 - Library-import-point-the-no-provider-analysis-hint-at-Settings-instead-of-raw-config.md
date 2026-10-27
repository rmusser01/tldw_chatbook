---
id: TASK-28018
title: >-
  Library import - point the no-provider analysis hint at Settings instead of
  raw config
status: To Do
assignee: []
created_date: '2026-09-02 04:11'
updated_date: '2026-09-02 21:07'
labels:
  - library
  - media-ux
dependencies: []
references:
  - >-
    .impeccable/critique/2026-09-02T04-00-36Z__tldw-chatbook-ui-screens-library-screen-py.md
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The pre-Start hint reads: Set provider under [analysis_defaults] in your config - instructing hand-editing TOML in an app with a Settings screen. First-timers will not find it. Point at the concrete in-app Settings location, ideally with a direct navigation affordance.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The hint names the in-app Settings location for configuring the analysis provider
- [ ] #2 Wording is verified against the actual Settings IA
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RECON (not started — blocked/feature): there is NO in-app Settings control for the ingest analysis provider. analysis_defaults is not referenced anywhere under UI/; the Import canvas has an 'Analyze after import' toggle but no provider picker. So AC#1 ('name the in-app Settings location') cannot be satisfied as filed -- the location does not exist. Real fixes: (a) add a Settings control for [analysis_defaults] provider, then point the hint at it; or (b) fall back to the user's configured chat/default provider when analysis_defaults is unset (a behavior change with cost implications -- wants product sign-off). Hint source: Library/ingest_analysis.py:163-190 (two hints, both instruct raw [analysis_defaults] TOML editing).

Cross-task note (TASK-34100.2, 2026-10-03, coverage-17): option (b) is now partly in place as a read-side rule. While `[analysis_defaults]` provider and model still equal the shipped template (OpenAI / gpt-4o), analysis follows `[chat_defaults]` (`effective_analysis_defaults` in Library/ingest_analysis.py; nothing is written, a user value is never replaced). The not-ready hint now says why a provider is involved instead of asking for its key: "Analyze after import is on, but analysis uses your chat model (Anthropic · …), and Anthropic is not ready (…)" or "… analysis uses its own default (OpenAI · gpt-4.1, set under [analysis_defaults]), …". Still no in-app Settings control for `[analysis_defaults]`; when this task points the hint at Settings, keep the "your chat model" / "its own default" framing so the two hints don't diverge.
<!-- SECTION:NOTES:END -->
