---
id: TASK-452
title: 'Local summarizer: drop trailing </s> {{ .Prompt }} Ollama cruft (behavior change)'
status: To Do
assignee: []
created_date: '2026-07-21 20:09'
labels:
  - internal-prompts
  - behavior-change
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The `summarization.local_summarizer_template` registry default (moved verbatim from `Local_Summarization_Lib.py`) still carries `<s>`/`</s>` sentinel tags and a trailing `{{ .Prompt }}` Ollama-modelfile token that get sent verbatim to models today. This is leftover templating cruft, not intended prompt content. Removing it changes the bytes sent to the model, so it needs its own review and before/after evaluation rather than riding a mechanical migration. When done, update the parity/registry default and the pin test that currently asserts the cruft is present.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The `<s>`/`</s>` sentinels and trailing `{{ .Prompt }}` are removed from the local summarizer default
- [ ] #2 The Task-3 pin test asserting `{{ .Prompt }}` presence is updated to reflect the cleaned text
- [ ] #3 A note/PR body records this is a deliberate behavior change with before/after prompt bytes
<!-- AC:END -->
