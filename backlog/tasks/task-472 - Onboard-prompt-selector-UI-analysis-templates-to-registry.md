---
id: TASK-472
title: 'Onboard prompt_selector UI analysis-prompt templates to the Internal Prompts registry'
status: Won't Do
assignee: []
created_date: '2026-07-22 22:10'
labels:
  - internal-prompts
  - enhancement
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the program (picker UX, different feature). Widgets/prompt_selector.py holds ~30 hardcoded system+user analysis-prompt templates keyed by media type. Decide whether these belong in the Internal Prompts registry or remain a separate picker concern; if onboarded, they need a UX that fits a keyed template set rather than the single-prompt editor.

**2026-08-13 update:** `Widgets/prompt_selector.py` was retired as dead code
(zero importers anywhere in the codebase) by task-15481, commit `0ddd7286c`.
The module no longer exists on disk; its ~30 templates survive only in git
history and are recoverable via
`git show fdffc031a:tldw_chatbook/Widgets/prompt_selector.py` (`fdffc031a`
is the dev commit immediately before the retirement). Any onboarding work
under this task must pull the templates from that git-history snapshot
rather than from a live file.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded on whether prompt_selector templates join the registry or stay separate
- [ ] #2 If onboarded: templates are registry-backed with parity tests and a suitable editing UX
- [ ] #3 If not: the rationale is documented and the templates left as-is
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: WON'T DO** — evidence verified against origin/dev (a3c47c6c72).

Superseded: Widgets/prompt_selector.py was deleted outright — no prompt_selector references remain in tldw_chatbook on origin/dev; provenance cross-referenced via commit 5489cc6a03 'docs: cross-reference retired prompt_selector provenance (task-15481)'. The ~30 picker templates no longer exist to onboard.
