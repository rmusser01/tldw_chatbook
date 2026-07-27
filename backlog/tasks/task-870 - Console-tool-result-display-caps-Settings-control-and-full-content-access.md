---
id: TASK-870
title: Console tool-result display caps — Settings control and full-content access
status: To Do
assignee: []
created_date: '2026-07-27 00:00'
labels: [console, agents, settings, ux]
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
What the Console shows of an agent tool result is governed by a cascade of
hardcoded constants, none of them user-adjustable, and the final display cap is
roughly 1% of what the model itself received. A tool result that reached the
model at up to 16,000 characters is rendered to the user as 160-200.

The cascade, on `origin/dev`:

| Stage | Cap | Site |
| --- | --- | --- |
| Tool returns | unbounded | provider / tool |
| Enters model history | 16,000 | `RunBudget.max_tool_result_chars` |
| Recorded on the step | 2,000 | `agent_runtime.py:652` `result=content[:2000]` |
| Live step summary | **200** | `console_agent_bridge._STEP_SUMMARY_LIMIT` |
| Transcript marker | **160** | `console_agent_bridge._STEP_MARKER_RESULT_LIMIT` |
| Resumed / persisted step | **200** | `console_agent_bridge._summarize_persisted_step` |
| Tool widget fields | 97 / 77 / 197, first 3 items | `Widgets/tool_message_widgets.py` |

Two consequences. First, a user debugging an agent run cannot see what the agent
actually saw, and has no control over the trade-off between transcript
readability and detail. Second, the paths are inconsistent: the live path uses
`_truncate_step_text`, which cuts on a word boundary and appends an explicit
`(+N chars)` affordance (TASK-350), while `_summarize_persisted_step` does a bare
`str(raw)[:200]` — so a resumed or historical run shows a silent mid-word clip,
which is exactly the defect TASK-350 fixed for live steps.

Separately, the run-log work (see the PRO-LONG programmatic run-memory design)
writes the full, untruncated result to a file on disk. Where such a file exists
for a run, the Console should be able to open the full record rather than only
ever showing a preview — raising the cap and linking to the source are two halves
of the same fix, and the second is what makes a conservative default cap
acceptable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The Console tool-result display cap is a single configurable setting, surfaced on the Settings page, not a set of scattered module constants
- [ ] #2 The setting has a documented default that preserves today's transcript readability, and a documented maximum
- [ ] #3 Changing the setting takes effect on newly rendered steps without an app restart
- [ ] #4 Live, transcript-marker, and resumed/persisted step rendering all honour the same setting; no path retains an independent hardcoded cap
- [ ] #5 `_summarize_persisted_step` uses the same word-boundary + `(+N chars)` affordance as the live path, so a resumed run never shows a silent mid-word clip
- [ ] #6 When a run log file exists for the run, the Console offers a way to read the full, untruncated result from it
- [ ] #7 When no run log file exists, the affordance is absent rather than dangling or erroring
- [ ] #8 The Settings control explains the relationship between the display cap and `max_tool_result_chars` (what the model saw), so the two are not confused
- [ ] #9 Tests cover the cap being read from config, applied identically across all three rendering paths, and the present/absent log-file branches of #6 and #7
<!-- AC:END -->
