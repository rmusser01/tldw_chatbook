---
id: TASK-418
title: >-
  Skills copy pass - self-referential empty state, approve modal title, jargon
  labels
status: Done
assignee:
  - '@claude'
created_date: '2026-07-21 15:18'
updated_date: '2026-07-21 16:37'
labels:
  - skills
  - ux
  - copy
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 from the 2026-07-21 Skills UX/NNG review. Copy problems verified live: (1) list empty state says 'No skills yet - create them in Library > Skills' which is where the user already is and never names Create > New skill or Import or the on-disk skills directory; (2) pressing Approve opens a modal titled 'Unlock Local Skill Trust' - task/dialog mismatch; (3) jargon: 'disable model invocation: no' double negative, 'context: inline/fork', row flags 'user - agent' with no legend, 'trusted baseline'/'trust-blocked'. NNG heuristics 2 (match with real world) and 10 (help).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Empty state names the actual creation and import paths and no longer points at itself,Approve flow modal title and message describe approving the reviewed version,Toggle and cycle labels read as plain statements without double negatives,Row flags line is either self-explanatory or accompanied by a legend/tooltip,Copy changes covered by snapshot or unit tests where such tests exist
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Copy pass, all watched-fail-first: (1) list empty state now names the real paths ('No skills yet — use Create ▸ New skill in the rail, or Import… above.') instead of pointing at the exact list being viewed; Console picker's identical-looking copy is a SEPARATE constant (console_skill_resolver.SKILLS_EMPTY_LIST_ROW) where the pointer is correct, untouched. (2) SkillTrustPassphraseModal gained title/message purpose overrides (+ #skill-trust-passphrase-title id); the Library approve flow now presents 'Approve Reviewed Skill Version' instead of 'Unlock Local Skill Trust'. (3) Toggle labels: 'User can invoke: yes/no', 'Agent can invoke: yes/no' (display polarity inverted over the stored disable_model_invocation - no more double negative), 'Runs in: inline (this conversation) / fork (sub-agent)' keeping the SKILL.md spec values visible. (4) Row flags line spelled out: 'invocable: user & agent / user only / agent only / not invocable'. (5) trust-blocked save status reworded to point at the trust panel. Rider (labeled): added 'prefill' to _SHADOWED_BUILTIN_NAMES - PR #729's /prefill command was missing and the shadow-name sync test failed on dev tip; a skill named 'prefill' would have silently shadowed the command. Updated pins: skills-state flags tests, canvas empty-copy test, usability-smoke skills tuple. Suites: canvas+state 79 passed, Tests/Skills 129 passed; smoke's remaining 'mcp expected copy' failure is pre-existing dev-tip baseline (no MCP files in this diff).
<!-- SECTION:NOTES:END -->
