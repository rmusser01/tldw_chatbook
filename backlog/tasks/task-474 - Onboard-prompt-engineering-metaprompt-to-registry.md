---
id: TASK-474
title: 'Onboard the prompt-engineering metaprompt to the Internal Prompts registry'
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
Deferred from the program. Prompt_Management/Prompt_Engineering.py contains the large Anthropic 'metaprompt' used by generate_prompt(). Onboard it as a single PromptSpec so it is user-editable, preserving its exact text and its function-calling example blocks.

**Source moved (TASK-32901, tier-2 review S09, 2026-09-22.)** That module was
deleted: it was unimportable in this repo at all (its line 12 does `from
tldw_Server_API.app.core.Chat.Chat_Functions import chat_api_call`, and
`tldw_Server_API` is not a module here), had zero importers in `tldw_chatbook/`
or `Tests/`, and was not in `Prompt_Management/__init__.py`'s `_LAZY_EXPORTS`.
`generate_prompt()` therefore could never have run. The metaprompt TEXT this
task needs is intact in git and is the only thing that was load-bearing:

    git show 3e564db3918ecd7287d06eb779aa5db11f51224d

(the blob as of `9e33252708`, or equivalently
`git show 9e33252708:tldw_chatbook/Prompt_Management/Prompt_Engineering.py`).
AC#2 will need rewording: there is no longer a `generate_prompt()` to resolve
through the registry, so this becomes "register the spec and give it a live
caller", not "re-point an existing one".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The metaprompt is registered with a byte-identical default (parity test)
- [ ] #2 generate_prompt() resolves it via the registry
- [ ] #3 Its placeholder/token contract is captured so edits cannot break generate_prompt()
<!-- AC:END -->


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: WON'T DO** — evidence verified against origin/dev (a3c47c6c72).

Superseded: Prompt_Management/Prompt_Engineering.py was deleted as unimportable (commit ceeb170bfd, task-32901); no 'metaprompt' references remain anywhere in tldw_chatbook on origin/dev. The module the task proposed to onboard no longer exists.
