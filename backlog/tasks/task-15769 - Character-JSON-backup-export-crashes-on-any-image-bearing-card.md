---
id: TASK-15769
title: Character JSON backup export crashes on any image-bearing card
status: To Do
assignee: []
created_date: '2026-08-13 12:31'
labels:
  - bug
  - characters
priority: medium
---

## Description

Found and confirmed during task-15474 (input-latency burn-down), flagged as
pre-existing and explicitly out of scope for that task's own fix:
`Tools_Settings_Window._export_characters_worker` (the JSON backup dump)
calls `json.dumps()` over character card rows that include the raw `image`
BLOB (`bytes`) whenever a card has an image, and nothing base64-encodes it
first. `bytes` is not JSON-serializable, so exporting any character list
that includes an image-bearing card crashes today.

Task-15474 changed `list_character_cards`/`list_character_cards_page`'s
default projection to exclude `image` (a separate, deliberate perf fix), which
means the export worker's crash is now masked for callers that adopt the new
default — but `_export_characters_worker` itself was not touched, so if it
(or any future caller) opts into `include_image=True` to actually include
images in a backup, the crash reproduces exactly as today. This task is
about fixing the export path itself, not relying on the masking side effect.

## Acceptance Criteria

- [ ] Exporting a character list that includes an image-bearing card via
      `Tools_Settings_Window._export_characters_worker` succeeds (no
      `TypeError: Object of type bytes is not JSON serializable`)
- [ ] The exported JSON either base64-encodes the image (matching the
      compatibility shape `Chat_Functions.load_characters` already uses for
      `image_base64`) or explicitly and intentionally omits it with a
      visible note in the export — pick one and make it correct, not an
      accidental silent drop
- [ ] A round-trip test: export an image-bearing card, confirm the output is
      valid JSON, and (if images are included) confirm the image round-trips
      byte-for-byte through base64
- [ ] Existing character-export tests stay green
