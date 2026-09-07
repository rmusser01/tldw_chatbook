---
id: TASK-18600
title: >-
  Expose Console agent run-budget limits in Settings and raise defaults for
  long-running sessions
status: Done
assignee: []
created_date: '2026-08-18 20:00'
labels:
  - console
  - agents
  - settings
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The Console agent's run budget is hardcoded in `console_agent_bridge.CONSOLE_RUN_BUDGET`
(30 model turns / 96 steps / 1800s wall / 1M tokens) plus the engine's per-tool-call
ceiling (`RunBudget.max_tool_call_seconds = 300`). A user who wants a long-running,
expensive agent session — a large refactor, a deep research sweep, a slow local model —
has no way to raise any of them without editing source.

Owner intent (2026-08-18): allow long-running, expensive sessions by default, and let
users tune every limit from inside the app.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Settings ▸ Console Behavior has an "Agent run budget" section exposing five limits: model turns, steps, wall-clock seconds, total tokens, and per-tool-call seconds.
- [x] #2 Each limit persists to `[console]` and takes effect on the next agent run with no app restart.
- [x] #3 Shipped defaults are 2000 model turns, 25000 steps, 86400s wall, 25,000,000 tokens, 3600s per tool call.
- [x] #4 Values are floored (turns/steps >= 1, wall >= 1s, tokens/tool-seconds >= 0 where 0 = unlimited) with no upper ceiling; an unparsable value falls back to the default rather than breaking a run.
- [x] #5 The engine's own `RunBudget` dataclass defaults are unchanged (8/240/30/0 stays the conservative floor for non-Console callers).
- [x] #6 Saving a step budget below the derived floor `3*(turns-1)+1` surfaces a non-blocking warning that the step backstop will bind before the turn cap.
- [x] #7 The tokens field's help text states the limit is PER RUN (not per conversation), that sub-agents inherit it unchanged (worst case ~3x per message), and that 0 removes the only runaway-spend backstop.
- [x] #8 User Guide documents the five limits, including that the token ceiling — not the turn cap — is what actually stops a long run, and points at `[agents] run_log_evict_enabled` as the companion knob.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. `config.py`: add DEFAULT_/MIN_ constants for the five keys, a shared
   `coerce_float_setting` helper alongside `coerce_int_setting`, coercion in
   `load_settings`' `[console]` block, and the keys in the shipped TOML template.
2. `console_agent_bridge.py`: replace the module-level `CONSOLE_RUN_BUDGET` with
   `console_run_budget()` (config-resolved per run, same shape as
   `_console_tool_result_display_cap`) plus `DEFAULT_CONSOLE_RUN_BUDGET` for the
   pure defaults; wire the resolver at the `AgentConfig` construction site.
3. `settings_screen.py`: new "Agent run budget" section in Console Behavior with the
   five inputs, the staged-save boilerplate each key needs, a live seconds->duration
   hint for the wall field, and the derived step-floor warning at save time.
4. Tests: config coercion + floors, bridge resolver honouring config, settings
   stage/save round-trip, step-floor warning. Update the one hardcoded
   `max_model_turns == 30` assertion.
5. Docs: `Docs/User_Guide/console/agent-runs-and-tools.md`.
<!-- SECTION:PLAN:END -->

## Notes

<!-- SECTION:NOTES:BEGIN -->
Review findings that shaped the numbers (2026-08-18, established by code reading):

- **The turn cap can never bind at 25M tokens.** `agent_service._make_call_model`
  re-sends the whole history every turn and `bound_history_for_send` is a no-op
  unless `[agents] run_log_evict_enabled` is on (default off, deliberately —
  `run_log_eviction.py` documents that trimmed history makes weaker models repeat
  work and go `stuck`). `ModelTurn.tokens` counts the full re-sent prompt, so spend
  grows quadratically: ~250 turns exhausts 25M at a typical 800-token round.
  Reaching turn 2000 would need ~12 tokens/round. Owner decision: keep 25M as the
  real governor and document it; turns/steps are backstops.
- **Loop detection will not protect a long run.** `_detect_cycle` keys on
  `(name, json.dumps(args))` — exact-args match — so any varying argument escapes
  it. `max_total_tokens` is therefore the only runaway-spend backstop, hence AC #7.
- **Step storage does not scale to 25000.** `AgentRunsDB.append_steps` read-modify-
  writes one JSON blob column (`_persist` runs once at end of run, so O(n) not
  O(n^2), but the blob can reach tens of MB and is re-parsed on every run-log open).
  Shipping 25000 as specified; filed as a follow-up rather than silently lowered.
- Checked and NOT problems: the transcript prunes rows past a virtual-height
  watermark (TASK-1365), so 25000 step markers will not render; and the generation
  HTTP client has a 300s read timeout, so a hung provider cannot eat the 24h wall.
<!-- SECTION:NOTES:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Five `[console] agent_max_*` keys, a config-resolved run budget, and a
table-driven Settings section.

**Approach.** `console_agent_bridge.CONSOLE_RUN_BUDGET` was a module
constant baked into `AgentConfig`. It is now `console_run_budget()`,
resolved per run from `[console]` (same shape as the existing
`_console_tool_result_display_cap`), so a Settings save reaches the next
message with no restart. `DEFAULT_CONSOLE_RUN_BUDGET` holds the shipped
defaults; `CONSOLE_RUN_BUDGET` remains as an alias for the tests that
import it.

**The Settings section is table-driven.** Five settings is where this
screen's per-field boilerplate quintet (loaded / draft / normalise / stage
/ sync) stops paying for itself, so `AGENT_BUDGET_FIELDS` is one spec table
and the compose loop, the single class-selector `@on` handler, the save
normaliser, and the revert sync all iterate it. Adding a sixth limit is one
table row.

**Defaults are an owner decision, and the numbers do not mean what they
look like.** 2000 turns / 25000 steps / 86400s / 25M tokens / 3600s per
tool call. The turn cap is NOT reachable: the whole conversation is re-sent
every turn, so spend is quadratic and 25M is exhausted around turn 250 for
a typical 800-token round. Owner decision (2026-08-18) was to keep spend as
the real governor and document it rather than lower the turn cap to
something reachable. The help text, the config comments, and the User Guide
all say this plainly; the alternative was shipping a number users would
reasonably expect to reach.

**Two pre-existing bugs fixed in passing.** `coerce_int_setting` raised on
`None` (`TypeError` from the bounds comparison) and on a non-finite float
(`OverflowError` from `int(inf)`, uncaught by `_get_typed_value`). Both are
reachable from a hand-edited config.toml and both would abort
`load_settings` -- i.e. app startup -- not merely mis-set one key. Found by
the new tests; fixed at the shared helper since the new keys route through
it.

**Trade-offs.**
- Floors only, no ceilings (owner call), matching the `max_parallel_runs`
  precedent. A below-floor value falls back to the shipped default rather
  than clamping, so a user never silently runs at a number nobody chose.
- Engine `RunBudget` defaults untouched (8/240/30/0/300) -- the Console is
  the only production caller, and a bare `RunBudget()` should stay
  conservative.
- The step-floor warning is non-blocking: a hard step ceiling is a
  legitimate thing to want, just never by accident.
- 25000 steps pushes `AgentRunsDB.append_steps` (one JSON blob column,
  read-modify-write) well past its 96-step sizing. Shipped as specified and
  filed rather than silently lowered.

**Verified live** (tmux, scratch profile): the section renders in Settings
▸ Console Behavior; `86400.0` shows `= 1d.` and `3600.0` shows `= 1h.`;
editing the token budget to 250000000 and pressing `s` wrote
`agent_max_total_tokens = 250000000` to config.toml, and
`console_run_budget()` then returned it; setting steps to 100 raised
"100 steps caps this run at about 34 tool-calling rounds, not 2000".

**Files.** `config.py` (constants, `coerce_float_setting`,
`coerce_int_setting` hardening, `[console]` coercion, TOML template);
`Chat/console_agent_bridge.py` (`console_run_budget()`, budget block
rewritten); `UI/Screens/settings_screen.py` (field table + section);
`Tests/Chat/test_console_agent_run_budget.py` (new, 25);
`Tests/UI/test_settings_agent_run_budget.py` (new, 27);
`Tests/Agents/test_agent_runtime.py` (stale `== 30`; unbounded test clock);
`Tests/UI/test_settings_configuration_hub.py` (ownership tuple);
`Docs/User_Guide/console/agent-runs-and-tools.md`.
<!-- SECTION:NOTES:END -->
