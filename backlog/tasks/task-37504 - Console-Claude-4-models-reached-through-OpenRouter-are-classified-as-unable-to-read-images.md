---
id: TASK-37504
title: >-
  Console: Claude 4+ models reached through OpenRouter are classified as unable
  to read images
status: To Do
assignee: []
created_date: '2026-10-29 09:07'
labels:
  - console
  - model-capabilities
  - p2
dependencies: []
references:
  - TASK-33620.2
  - TASK-37031
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found reviewing TASK-33620.2 (2026-10-29). TASK-33620.2 made every Anthropic-routed Claude from the 4 generation on read images, but the rule is gated to the `anthropic` provider on purpose (a local server's `claude-...` alias is not Anthropic's model). Claude reached through OpenRouter is still decided by the OpenRouter pattern list, whose only Claude row is `anthropic/claude-3` (both in `DEFAULT_MODEL_PATTERNS` and in the shipped config template). So the shipped OpenRouter entry `anthropic/claude-sonnet-4`, and current ids such as `anthropic/claude-sonnet-4.5` or `anthropic/claude-haiku-4.5`, block Send as soon as an image is staged, although the model reads images. OpenRouter writes the version with dots (`claude-sonnet-4.5`), not the tier-first dash form the Anthropic rule parses. As with TASK-33620.2, an older config.toml's pattern list replaces the shipped one wholesale, so a pattern-only fix does not reach existing users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every Claude 4+ model in the shipped OpenRouter model list, and current OpenRouter Claude ids such as anthropic/claude-sonnet-4.5, anthropic/claude-haiku-4.5 and anthropic/claude-opus-5, is classified as vision-capable with a fresh config and with a config.toml carrying the old [model_capabilities] block
- [ ] #2 OpenRouter Claude 3.x ids keep their current classification, and Claude 2 / instant ids stay unable to read images
- [ ] #3 A user's explicit vision = false row still wins, and a non-Anthropic provider's model merely named claude-... is unaffected
- [ ] #4 Any other route the app has to Anthropic models (a Bedrock- or Vertex-style id) is either classified the same way or recorded in the task as not routed by the app
- [ ] #5 Tests use the real capability lookup and fail on the code before the fix
<!-- AC:END -->
