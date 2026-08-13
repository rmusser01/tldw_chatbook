---
id: TASK-15772
title: STTS Select widgets compose options in the wrong tuple order, so set-value calls fail
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - stts
priority: medium
---

## Description

Found during task-15478's review (input-latency burn-down) for one Select
and confirmed here to be a repeated pattern across `UI/STTS_Window.py`.
Textual's `Select.options` expects `(label, value)` tuples, but multiple
Selects in this file are composed with `(id, label)` — the reverse:

- `#import-source-select` (`options=[("file", "Text File"), ...]`, flagged
  in task-15478's notes) — `_import_content`'s
  `if import_source == "file":` branches check the id, but the widget's
  actual `.value` after selection is the display label ("Text File"), never
  the lowercase id. All four "Import From" dispatch branches
  (file/notes/conversation/paste) are non-functional today via this Select.
- `#audiobook-provider-select` (`options=[("openai", "OpenAI"),
  ("elevenlabs", "ElevenLabs"), ("kokoro", "Kokoro (Local)"),
  ("chatterbox", "Chatterbox (Local)")]`) and `#audiobook-format-select`
  (`options=[("mp3", "MP3"), ...]`) — same reversed shape, confirmed by
  reading the compose call. `_initialize_audiobook_defaults` (STTS_Window.py,
  scheduled via `set_timer(0.1, ...)` on mount) then does
  `provider_select.value = "openai"` / `format_select.value = "m4b"` inside a
  bare `try/except Exception: logger.debug(...)` — since "openai"/"m4b" are
  never present among the Select's actual values (the labels are), Textual's
  illegal-value validation fires and the default-selection attempt silently
  no-ops into the debug log on every STTS window mount.

## Acceptance Criteria

- [ ] `#import-source-select`, `#audiobook-provider-select`, and
      `#audiobook-format-select` all compose `options=` in Textual's real
      `(label, value)` order
- [ ] Every `.value` comparison and assignment against these three Selects
      (`_import_content`'s branch checks;
      `_initialize_audiobook_defaults`'s provider/format assignment) is
      updated to match, and actually selects the intended default on mount
      (not silently swallowed by the surrounding `try/except`)
- [ ] All four "Import From" dispatch branches (file/notes/conversation/paste)
      are reachable and functional through the Select, not just through a
      direct method call (test drives the Select, not `_import_content`
      called directly)
- [ ] `_initialize_audiobook_defaults` no longer logs an illegal-select-value
      warning on a fresh STTS window mount (test asserts no such log line)
