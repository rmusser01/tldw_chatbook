---
id: TASK-519
title: 'Refresh default models for DeepSeek, Anthropic, and OpenAI'
status: To Do
assignee: []
created_date: '2026-07-27 03:58'
updated_date: '2026-07-27 03:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Replace stale provider defaults with current vendor-supported balanced general-purpose models while preserving supported alternatives and keeping provider request payloads compatible with the selected model families.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Fresh installations default DeepSeek to deepseek-v4-flash, Anthropic to claude-sonnet-5, and OpenAI to gpt-5.6-terra
- [ ] #2 Provider catalogs retain supported alternatives and exclude retired DeepSeek aliases from active defaults
- [ ] #3 OpenAI GPT-5.6 default requests use a compatible token and reasoning contract without regressing explicit Responses API reasoning flows
- [ ] #4 Anthropic Claude Sonnet 5 requests omit unsupported sampling parameters and honor supported adaptive-thinking settings
- [ ] #5 The selected OpenAI and Anthropic defaults are recognized by capability metadata
- [ ] #6 Focused configuration, provider-payload, and capability tests pass
- [ ] #7 ADR-020 and the approved design are linked; no new ADR is required
<!-- AC:END -->
