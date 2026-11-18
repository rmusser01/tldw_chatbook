---
id: TASK-37503
title: >-
  Settings: saving a model's context window freezes a defaulted image-support
  flag into config
status: To Do
assignee: []
created_date: '2026-10-29 09:05'
labels:
  - settings
  - console
  - p2
dependencies: []
references:
  - TASK-33620.2
  - TASK-26023
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found reviewing TASK-33620.2 (2026-10-29). Settings ▸ "save context window" builds the saved entry by copying everything `ModelCapabilities.get_model_capabilities` returns for the model and adding the window (`model_context_window_save_entry` in UI/Screens/settings_context_memory.py). Whatever the lookup filled in from a default or an upstream catalog is therefore written into `[model_capabilities.models]` as if the user had set it.

Before TASK-33620.2, claude-haiku-4-5 (the shipped Anthropic default), claude-opus-5 and claude-fable-5 matched no capability row, so the unknown-model default `vision = false` was what got copied. Anyone who saved a context window for one of those models now holds an explicit `vision = false` direct row, and because an explicit row outranks the new Claude family rule (deliberately: a user's own `vision = false` must win), their images stay blocked. With the models.dev gap-fill on, the copy also carries `source = "models.dev"` into the user's row. The TASK-33620.2 fix does not reach these users.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving a context window in Settings leaves the model's image support as it was before the save: no vision value is written that the user did not set, and a model the app classifies as able to read images still does after the save
- [ ] #2 A user who saved a context window for claude-haiku-4-5, claude-opus-5 or claude-fable-5 before TASK-33620.2 can stage an image on that model without hand-editing config.toml, while a vision = false the user wrote deliberately still blocks images
- [ ] #3 Catalog bookkeeping such as the models.dev source marker is not persisted into the user's own model row
- [ ] #4 Tests drive the real save path and the real capability lookup, and fail on the code before the fix
<!-- AC:END -->
