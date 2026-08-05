---
id: TASK-870
title: Console tool-result display caps — Settings control and full-content access
status: Done
assignee:
  - '@claude'
created_date: '2026-07-27 00:00'
updated_date: '2026-07-29 01:26'
labels:
  - console
  - agents
  - settings
  - ux
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
- [x] #1 The Console tool-result display cap is a single configurable setting, surfaced on the Settings page, not a set of scattered module constants
- [x] #2 The setting has a documented default that preserves today's transcript readability, and a documented maximum
- [x] #3 Changing the setting takes effect on newly rendered steps without an app restart
- [x] #4 Live, transcript-marker, and resumed/persisted step rendering all honour the same setting; no path retains an independent hardcoded cap
- [x] #5 `_summarize_persisted_step` uses the same word-boundary + `(+N chars)` affordance as the live path, so a resumed run never shows a silent mid-word clip
- [x] #6 When a run log file exists for the run, the Console offers a way to read the full, untruncated result from it
- [x] #7 When no run log file exists, the affordance is absent rather than dangling or erroring
- [x] #8 The Settings control explains the relationship between the display cap and `max_tool_result_chars` (what the model saw), so the two are not confused
- [x] #9 Tests cover the cap being read from config, applied identically across all three rendering paths, and the present/absent log-file branches of #6 and #7
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add one config-backed setting (config.DEFAULT/MIN/MAX_CONSOLE_TOOL_RESULT_DISPLAY_CHARS, [console] tool_result_display_chars) replacing _STEP_SUMMARY_LIMIT/_STEP_MARKER_RESULT_LIMIT.
2. Add _console_tool_result_display_cap() in console_agent_bridge.py: env var (TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS) -> get_cli_setting -> default, read fresh every call, mirroring run_log._setting's tier order.
3. Route format_agent_step_marker, _summarize, and _summarize_persisted_step through that one resolver; give _summarize_persisted_step the same _truncate_step_text word-boundary + (+N chars) treatment as the live path.
4. Add run_log.resolve_existing_log_dir(run_id) (read-only counterpart to RunLogWriter.bind) plus ConsoleAgentBridge.run_log_available/load_run_log_text/latest_primary_run_id.
5. Add a read-only ConsoleRunLogModal and wire a "View full log" button into the Console Agent rail (chat_screen.py), visible only when a log exists for the run being shown (drilled-in sub-agent or the conversation's latest primary run).
6. Surface the setting on the Settings page's Console Behavior category, mirroring the existing paste_collapse_threshold field (Input + validation + draft staging + save + field guidance explaining the relationship to max_tool_result_chars).
7. Tests: Tests/Agents/test_run_log_resolve_existing.py, Tests/Chat/test_console_agent_tool_result_cap.py.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: replaced the scattered hardcoded caps (_STEP_SUMMARY_LIMIT=200, _STEP_MARKER_RESULT_LIMIT=160, _summarize_persisted_step's bare [:200]) with one resolver, _console_tool_result_display_cap() in console_agent_bridge.py, that reads env var (TLDW_CONSOLE_TOOL_RESULT_DISPLAY_CHARS) -> [console] tool_result_display_chars (get_cli_setting) -> default (160), mirroring run_log._setting's tier order exactly as instructed. It is called fresh on every render (no caching anywhere), so a Settings save -- which forces get_cli_setting's cache to reload -- takes effect on the very next step rendered without a restart. format_agent_step_marker, _summarize (live), and _summarize_persisted_step (resumed) all call it, so all three paths share one cap. _summarize_persisted_step now goes through _truncate_step_text (word-boundary cut + "(+N chars)" affordance) instead of its old bare str(raw)[:200] slice -- the exact defect TASK-350 fixed for the live path only.

Bounds: default 160 (== the prior transcript-marker cap, so a fresh install's TRANSCRIPT reads unchanged -- but the Agent rail's live-step and resumed/persisted summaries were previously 200, not 160, so those two secondary-panel surfaces now trim 40 chars more than before; a real minor behaviour change, not a no-op, chosen because the transcript is the primary reading surface and "View full log" now covers the gap), min 20, max 2000 -- the max is not arbitrary: agent_runtime.py already caps a step's own recorded `result` field at 2000 chars before any display path sees it, so a higher display cap could not reveal more, only mislead. _STEP_MARKER_RESULT_LIMIT is kept (hardcoded, no longer read by production code) solely because Tests/Utils/test_path_validation_multi.py imports it directly; a comment documents why.

Settings surfacing: new [console] tool_result_display_chars field on the existing Console Behavior category, following the paste_collapse_threshold pattern end to end (constants in config.py, _loaded_/_stage_/_normalise_ helpers, draft staging, Input widget + validation + save + revert-resync, and dedicated field guidance explaining the max_tool_result_chars distinction -- AC#8).

Full-log affordance: run_log.resolve_existing_log_dir(run_id) is the read-only counterpart to RunLogWriter.bind() -- resolves the same root and tries both the undotted and sandbox-fallback-dotted directory names, without creating anything, so an ARBITRARY (possibly long-finished) run's log can be located even outside its own writer instance. ConsoleAgentBridge gained run_log_available/load_run_log_text/latest_primary_run_id on top of it. chat_screen.py's Agent rail gained a "View full log" button (own _console_agent_full_log_run_id/_console_agent_full_log_available helpers, folded into the existing 0.2s-tick equality-guarded sync) that opens a new read-only ConsoleRunLogModal -- visible only when a log exists for whichever run the rail is currently showing (drilled-in sub-agent, or the conversation's latest primary run), absent otherwise. Two pre-existing Tests/UI/test_console_agent_rail.py _FakeBridge doubles lacked the new latest_primary_run_id method; fixed by making that one lookup getattr-tolerant (the file's own pre-existing idiom for exactly this), not by touching the tests.

Files: tldw_chatbook/config.py, tldw_chatbook/Chat/console_agent_bridge.py, tldw_chatbook/Agents/run_log.py, tldw_chatbook/UI/Screens/chat_screen.py, tldw_chatbook/UI/Screens/settings_screen.py, tldw_chatbook/Widgets/Console/console_run_log_modal.py (new), tldw_chatbook/css/components/_agentic_terminal.tcss (+regenerated bundle). Tests: Tests/Agents/test_run_log_resolve_existing.py (new, 6 tests), Tests/Chat/test_console_agent_tool_result_cap.py (new, 21 tests).

Verified: Tests/Agents/ 535/535 passed; Tests/Chat/ 2713 passed (baseline 4 failed in test_chat_functions.py + 13 errors in test_scope_picker_listers.py unchanged, both pre-existing and unrelated). Also spot-checked Tests/UI/test_console_agent_rail.py (21/21), Tests/UI/test_console_parallel_runs.py (28/28), Tests/test_config_console_defaults.py + Tests/Utils/test_config_import_hygiene.py (29/29) -- all green. Tests/UI/test_settings_configuration_hub.py has 22 pre-existing failures (confirmed identical on a clean stash of origin/dev, unrelated to this task -- stale monkeypatch targets, an unrelated PrivatePathError, and a missing TldwCli attribute).
<!-- SECTION:NOTES:END -->
