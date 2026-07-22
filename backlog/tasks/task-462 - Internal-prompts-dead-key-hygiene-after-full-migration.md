---
id: TASK-462
title: 'Internal prompts: dead-key hygiene after full migration'
status: To Do
assignee: []
created_date: '2026-07-21 20:08'
labels:
  - internal-prompts
  - cleanup
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Once the registry migration is fully landed (P1+P2, and P3 Settings UI), remove the now-orphaned prompt config scaffolding in `config.py`: the unconsumed `prompts_strings` loader, the `CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT` constant (no call site anywhere), and the stale `[Prompts]` stub one-liner comments that predate the registry. Verify each is genuinely unreferenced before deletion (the registry's legacy tier reads `[Prompts]` keys, so keep the section itself if any spec still points a `legacy_config_path` at it — only remove the truly dead scaffolding).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 `prompts_strings` loader removed if grep confirms zero consumers
- [ ] #2 `CONFIG_PROMPT_SITUATE_CHUNK_CONTEXT` removed (no call site exists)
- [ ] #3 `[Prompts]` keys still referenced by any spec's legacy_config_path are preserved; only genuinely-dead scaffolding is deleted
- [ ] #4 App imports and the internal-prompts suite stay green
<!-- AC:END -->
