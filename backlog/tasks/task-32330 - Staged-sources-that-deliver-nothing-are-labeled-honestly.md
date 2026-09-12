---
id: TASK-32330
title: >-
  Staged sources that deliver nothing are labeled honestly
status: Done
assignee:
  - '@zcode'
created_date: '2026-09-10 12:00'
labels:
  - console
  - ux
  - rail-ux-review
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UX review C1. The Sources tray renders 'ready' for staged handoffs that send nothing to the model: skills, watchlists/collections snapshots, quizzes, personas (task-2375), and media/conversation handoffs currently deliver only a short label (task-2376). The docs disclose this; the UI does not. Until delivery lands, label those rows so the user can see the model will not receive the content.

Filed from the 2026-09-10 Console rail UX review (review item C1).
<!-- SECTION:DESCRIPTION:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Probe live: does a skills-context reference survive send normalization (it did not). 2. RED: non-deliverable available reference renders 'listed'; deliverable stays 'ready'. 3. Add the deliverable predicate + allowlist mirror + sync test; map the status; add tray detail copy + CSS. 4. Update the docs honesty passage; run suites.
<!-- SECTION:PLAN:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Staged rows for handoff kinds that currently deliver no model payload render a distinct not-delivered status (e.g. 'listed - not sent') instead of 'ready'
- [x] #2 Rows whose delivery is partial (short label only) render an honest distinct status
- [x] #3 Status chip / Source Readiness counts remain consistent with the tray (no disagreement between surfaces)
- [x] #4 User-guide context-and-rag.md updated to match the new row vocabulary
<!-- AC:END -->

## Implementation Notes
<!-- SECTION:NOTES:BEGIN -->
### Close-out (2026-09-10)

**Approach — the investigation.** Re-verification found the task-2375
class still live on dev, by execution: a skills handoff stages a bundle
whose reference carries source_type="skills-context" with the DEFAULT
status "available" (chat_screen's handoff path builds it from the
payload's body) — the tray rendered "Ready" — but on send
`normalize_console_evidence_references` -> `normalize_local_result` ->
`_resolve_source` REJECTS any source_type outside `_SOURCE_ALIASES`
(notes/media/conversations only), so the model received nothing.
Reproduced with a live probe (normalized count 0) before changing
anything. Also verified: media/conversation handoffs DO deliver their
short-label snippet (task-2376's partial state is honest enough to
document, not fix here); the generic live-work fallback item delivers
via the same handoff bundle path.

**The fix.** New `reference_can_deliver()` +
`DELIVERABLE_EVIDENCE_SOURCE_TYPES` in citation_evidence_models
(mirroring the capture allowlist; a sync test pins the mirror).
`ConsoleStagedContextState.from_live_work` maps an "available" reference
with a non-deliverable source kind to status "listed" — the primary row
renders "Listed · title · kind" with a muted color class, and expanding
the row adds the explicit line "This handoff kind is listed for the run;
its content is not sent to the model." Ready/Blocked/Warning statuses
unchanged; deliverable kinds unchanged. The user guide's honesty passage
now states the vocabulary and the capture boundary.

**ADR check.** Not required — display honesty inside one projection;
the delivery gap itself remains task-2375's scope.

**Modified.** `Chat/citation_evidence_models.py`,
`Chat/console_display_state.py`,
`Widgets/Console/console_staged_context.py`, both console CSS split
sheets (new `.console-staged-source-primary-listed` class),
`Tests/Chat/test_console_display_state.py` (+3 tests incl. the allowlist
sync guard), `Docs/User_Guide/console/context-and-rag.md`. Verified:
display-state + staged-context suites — 64 passed.

### Walkthrough verification (2026-09-10, dev tip a0b8f96416)

PREMISE MOSTLY GONE on dev: task-2375/2376 handoff kinds (skills/watchlists/quizzes/personas) can no longer be staged; staging funnels through Library RAG evidence (library_rag_state.py:53-59); send delivers only available_references() (citation_evidence_models.py:367-376). RESIDUAL TO CHECK: rows whose reference status is neither available nor blocked/missing render 'Warning' in the tray (console_display_state.py:826-834) - do they deliver? And the generic live-work fallback item (console_live_work.py:169-196) stages a navigation-only item - does it honestly render?
Live evidence: headless tmux run, scratch profile, 160x45 and 124x45 captures; code evidence re-verified in the implementation worktree.
<!-- SECTION:NOTES:END -->
