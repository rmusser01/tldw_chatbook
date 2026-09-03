---
id: TASK-26018
title: 'Compaction: focus-directed summaries'
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:45'
updated_date: '2026-09-01 16:30'
labels:
  - console
  - context
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A compaction summary cannot be steered toward what the user cares about. Verified on origin/dev: a grep for focus across Chat/console_context_compaction.py returns zero, so the summary prompt is fixed and a long debugging session compacts down to whatever the model judged salient. Hermes accepts a topic argument that biases the summary. The change is one string appended to the compaction prompt built for the auxiliary call.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A manual compaction accepts an optional topic that biases what the summary preserves
- [x] #2 With no topic supplied, the produced summary is byte-identical to today's for the same input
- [x] #3 The topic is recorded in the compaction provenance so a later reader knows the summary was steered and how
- [x] #4 The topic is treated as untrusted user text: it is bounded in length and cannot inject instructions that alter the summarizer's role
- [x] #5 A topic that yields an empty or unusable summary falls back to the unsteered path rather than committing a degraded record
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED: sanitize/identity/plan-steering/provenance/fallback tests\n2. sanitize_summary_focus (collapse, 200-cap, reserved-marker refusal) + focus_directed_prompt (identity when empty)\n3. plan_manual_* focus kwarg -> steered auxiliary messages + unsteered fallback set on the plan\n4. summarize_manual attempt loop (steered then unsteered once) + focus_topic provenance marker with applied flag\n5. Preview modal gains optional focus Input (result bool->str|None); controller sanitizes at boundary
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Focus rides the 26017 preview dialog: an optional Input in ConsoleSummarizePreviewModal (result type bool->str|None; None=cancel, ''=unsteered) -> screen -> summarize_from/up_to(focus=) -> _manual_summary_planning sanitizes once at the boundary (sanitize_summary_focus: whitespace collapse, hard 200-char cap, reserved-envelope markers refuse to '') -> plan_manual_prefix/range(focus=) -> focus_directed_prompt appends a role-preserving frame ('user-supplied topic, not an instruction', topic JSON-quoted) to the auxiliary system prompt ONLY. AC#2 is structural: focus='' returns the identical prompt object and a byte-identical plan (pinned against a plan built without the kwarg). AC#5: a focused plan carries the unsteered messages (fallback_auxiliary_messages); summarize_manual retries once unsteered when the steered summary is empty/reserved, both calls under the 26016 timeout. AC#3: the committed record's selected_units_json gains {'kind':'focus_topic','topic':...,'applied':bool} — applied=False when the fallback produced the summary. 6 new tests; compaction suite 135 passed; modal-dismissal + rewind_summarize at exact baseline failure sets. Automatic compaction intentionally NOT steered (the task is manual-scope; automatic has no user at the moment of compaction).
<!-- SECTION:NOTES:END -->
