# Task 4 report — Planning distinction and round-owned suppression

## Outcome

Renamed the privacy-safe intermediate primary model-step activity from Thinking to
Planning. A safe summary is shown only after that model round proves tool work; unsafe
or empty summaries and final answer rounds produce no synthetic row. Displayable or
proprietary Thinking evidence suppresses Planning only for the same explicit round,
while later planning-only rounds remain visible.

Live activity rows now carry their owning model-round ordinal. Resume receives exact
round ordinals from the selected Assistant generation's validated ThinkingEnvelope,
keyed by its stable Assistant owner, and reproduces the same suppression without
provider/capability inference. The existing conservative summary sanitizer is
unchanged.

## TDD evidence

- RED: the initial Planning/Thinking slice had 10 failures: unsafe summaries still
  fabricated an empty Thinking row, safe summaries retained the old presentation,
  `planning` was outside the activity vocabulary, and the deriver had no exact-round
  evidence input.
- GREEN: the focused pure Planning/Thinking slice passed 10/10.
- RED: the live/resume ownership slice failed 4/4 before round ownership reached the
  store and selected-generation resume handoff.
- GREEN: displayable and proprietary live evidence, exact multi-round resume, and the
  real controller resume handoff passed 4/4.
- The prescribed bridge/activity/UI regression reached 558/560; both failures were
  stale expectations that still demanded a synthetic row from tool-call rounds with
  no safe summary. Those two cases passed 2/2 after correction, and the complete
  bridge/activity files then passed 403/403. The 157 UI disclosure/transcript/resume
  cases in the prescribed run were green.
- Scoped Ruff lint and `git diff --check` passed.

## Architecture and scope

ADR-090 remains the governing accepted decision; no new ADR is required. The bounded
adjacent changes add `planning` to the existing activity vocabulary, expose the
already-modeled session-only round owner through `ConsoleChatStore.append_message`,
and pass selected-envelope round sets through the existing controller resume handoff.
No schema, persistence format, setting, stylesheet, binding, dependency, or export
surface changed.

Visual inspection, the Impeccable detector, Backlog completion, and export Task
18932.4 remain intentionally outside this implementation slice for the parent task.
