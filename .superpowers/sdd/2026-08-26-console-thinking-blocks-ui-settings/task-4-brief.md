# Task 4 brief — Planning distinction and round-owned suppression

## Scope

Rename the existing privacy-safe intermediate primary model-step marker from
Thinking to Planning. Suppress that session-only marker only when the same model
round owns an actual displayable or proprietary block in the selected generation.

## Contracts

- A safe intermediate primary `STEP_MODEL.summary` without actual evidence yields
  `ConsoleActivityPresentation("planning", "Planning", "done")`.
- Displayable and proprietary evidence suppress Planning only for their exact
  `ThinkingEnvelope.round_ordinal`; capability or provider identity never does.
- Final model rounds yield no synthetic Planning row. Actual Thinking remains owned
  by the Assistant envelope and is not duplicated as a TOOL marker.
- Live and resume consume explicit selected-envelope round sets and stamp TOOL/
  Planning activity rows with their model-round ownership.
- `safe_intermediate_thinking_summary` keeps its conservative rejection behavior
  unchanged; an unsafe/empty summary yields no Planning row.

## Non-goals

No visual inspection, stylesheet change, Settings/export work, new controller,
dependency, binding, footer hint, or persistence schema change. ADR-090 governs.
