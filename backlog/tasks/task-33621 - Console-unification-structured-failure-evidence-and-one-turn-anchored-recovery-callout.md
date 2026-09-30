---
id: TASK-33621
title: >-
  Console unification: structured failure evidence and one turn-anchored
  recovery callout
status: To Do
assignee: []
created_date: '2026-09-30 03:03'
labels:
  - console
  - ux-review-2026-09-29
  - unification
dependencies: []
references:
  - qa/console-ux-review-2026-09-29/report.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Why: on user-initiated Console paths, broad except blocks, exit_on_error=False workers and keep-alive handlers turn backend exceptions into silence. The review verified this for tab close, 'Save .md', the 'Choose folder' freeze, compaction lineage, the citation finalizer, the Inspector reader and the capture signature. When copy does appear, it is chosen from a coarse enum or from substring matches, and it states facts nothing measured: 'The provider was not contacted' after a 429, 'on the source device', 'choose another model', 'Private scratch space is unavailable'. Recovery then shows up in four or five unrelated places (top-of-transcript card, control-deck panel, queue shelf, System row, toast), each with its own grammar, and several of their buttons refuse to act. The theme also contains the primary send-loop P0s: the built-in tool-schema HTTP 400 on every default OpenAI or Anthropic send, silent refusal after a character swap or greeting turn, and the compaction loop that bills a summarizer call on every send.

What unifying means here: one rule and one component. The rule: every catch on a user-initiated path logs the exception type and its context, and sets a visible state from a structured failure record (ConsoleFailureEvidence). The record holds phase, provider contacted (yes/no/unknown), HTTP status, category, sanitized provider message, origin (local or remote) and blocker code. User copy is derived from those fields and never asserts a field the record does not hold. The component: one RecoveryCallout anchored to the affected turn, with a single primary action and object-named secondary actions. The composer and header only point to it. A live-stack test with no fakes pins every primary recovery action. This umbrella is the parent of the theme's P0/P1 fixes (G4-01, G4-03, GAP5-01, G4-04, G4-05, G4-10, G2-04, GAP1-05, G3-02, GAP4-01, G3-01, GAP2-01, GAP2-02, GAP5-06, GAP5-07, G2-09, G1-13, G2-06, GAP1-04, GAP5-09, G4-06, G1-09, G3-09).

Evidence: qa/console-ux-review-2026-09-29/report.md (theme 'Failures are swallowed') and qa/console-ux-review-2026-09-29/findings.md / findings.json.

Related P2/P3 in the ledger: GAP4-02, G2-37, G4-07, G4-24, G4-25, G4-27, G4-28, GAP5-23, G1-11, G1-29, G4-30, G4-33.

Source: Console UX review 2026-09-29 — qa/console-ux-review-2026-09-29/report.md (themes) and findings.md (ledger).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 An ordinary send with the default configuration to each supported cloud provider completes end to end, verified by a live-stack test with no fakes
- [ ] #2 Every exception raised on a user-initiated Console path is logged with its type and context and produces a visible state for the user; a guard test fails if a catch on such a path swallows the error with no user-visible result
- [ ] #3 Failure copy is derived from a structured failure record and never states a fact the record does not hold (for example, 'provider was not contacted' appears only when the record says the provider was not contacted)
- [ ] #4 Every recoverable turn failure is shown in one recovery callout anchored to the affected turn, with exactly one primary action, and no second recovery surface for the same failure
- [ ] #5 Every primary recovery action, tested against the live stack with no fakes, either performs its action or explains why it cannot
- [ ] #6 All P0/P1 child tasks under this umbrella are Done
<!-- AC:END -->
