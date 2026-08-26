---
id: TASK-15769
title: Character JSON backup export crashes on any image-bearing card
status: Done
assignee: ['@claude']
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

- [x] Exporting a character list that includes an image-bearing card via
      `Tools_Settings_Window._export_characters_worker` succeeds (no
      `TypeError: Object of type bytes is not JSON serializable`)
- [x] The exported JSON either base64-encodes the image (matching the
      compatibility shape `Chat_Functions.load_characters` already uses for
      `image_base64`) or explicitly and intentionally omits it with a
      visible note in the export — pick one and make it correct, not an
      accidental silent drop
- [x] A round-trip test: export an image-bearing card, confirm the output is
      valid JSON, and (if images are included) confirm the image round-trips
      byte-for-byte through base64
- [x] Existing character-export tests stay green

## Implementation Plan

1. Re-locate at HEAD (`bb91fef73`): confirm `_export_characters_worker`
   (`UI/Tools_Settings_Window.py`) still `json.dumps()`s raw
   `list_character_cards` rows, and that the crash is only masked by the
   task-15474 `include_image=False` default (silent image drop today).
2. Reproduce the crash signature on the worker's own seam: extract the
   serialization into a module-level helper
   `_serialize_character_cards_for_backup(...)`, have the worker opt into
   `include_image=True` (a backup should contain the image), and write a
   born-red round-trip test that fails with
   `TypeError: Object of type bytes is not JSON serializable`.
3. Fix: the helper replaces the raw `image` BLOB with a plain-base64
   `image_base64` string (the `Chat_Functions.load_characters` compatibility
   shape; deliberately NOT the data-URI form `export_character_card_to_json`
   uses, because the import chain `b64decode`s the raw string and a data-URI
   prefix would garble it).
4. Make the export genuinely re-importable: teach `parse_v1_card`
   (`Character_Chat/Character_Chat_Lib.py`) to accept `image_base64` as an
   image source key (today only `char_image`/`image` are read, and an
   unrecognized `image_base64` would be swallowed whole into `extensions`,
   bloating the DB on re-import).
5. Round-trip test: DB card with image bytes -> export JSON (valid, no raw
   bytes) -> base64 decodes byte-for-byte -> full re-import via
   `import_and_save_character_from_file` into a fresh DB restores identical
   image bytes.
6. Keep existing export/import tests green; ruff check + format on touched
   files.

## Implementation Notes

The export worker now includes images and serializes through a new
module-level helper instead of a bare `json.dumps` over raw DB rows.

**Re-location found a SECOND crash class.** The audit diagnosed the `bytes`
BLOB; reproducing at HEAD (`bb91fef73`) showed the masked, image-free
projection ALSO crashes — `list_character_cards` rows carry
`created_at`/`last_modified` as `datetime` objects, so the backup died with
`TypeError: Object of type datetime is not JSON serializable` for EVERY
card, image or not. Both crashes sit on the same `json.dumps` line, so both
are fixed in the one helper (AC 1 is unverifiable otherwise — the worker
could never succeed).

**Approach** (smallest honest diff, both sides of the round trip):

- `UI/Tools_Settings_Window.py`: `_export_characters_worker` opts into
  `list_character_cards(limit=10000, include_image=True)` (a backup should
  contain the avatar; relying on the task-15474 image-free default was the
  accidental-silent-drop the AC forbids) and serializes via new
  `_serialize_character_cards_for_backup()`: the raw `image` BLOB is
  replaced by a plain-base64 `image_base64` string (the
  `Chat_Functions.load_characters` compatibility shape; deliberately NOT
  the data-URI form `export_character_card_to_json` embeds, because the
  import chain b64decodes the raw string and a prefix would garble it),
  and datetimes ISO-format via a `default=` hook that still raises for any
  other unexpected type.
- `Character_Chat/Character_Chat_Lib.py`: `parse_v1_card` now accepts
  `image_base64` as an image source key (after `char_image`/`image`) and
  lists it in the known-keys set — previously the whole base64 payload
  would have been swallowed into `extensions` on re-import and the avatar
  lost.

**Evidence** (born-red first, then green; venv + PYTHONPATH pinned to the
worktree):

- Born-red: with the helper extracted but naive, the new test file failed
  5/5 — 4 with `TypeError: Object of type datetime is not JSON
  serializable` and the parse test on the unrecognized key; the bytes
  signature (`Object of type bytes is not JSON serializable`) demonstrated
  on the same dumps call shape. After the fix: 5/5 green.
- Worker-level test (`Tests/UI/test_tools_settings_window.py::
  test_export_characters_worker_survives_image_bearing_cards`, using the
  file's TASK-927 SimpleNamespace harness) proves AC 1 on the named seam;
  mutation check (helper body temporarily reverted, Edit-based restore)
  fails it with the real production notify:
  `Error exporting characters: Object of type datetime is not JSON
  serializable`.
- Round trip: export -> single-card JSON file ->
  `import_and_save_character_from_file` into a fresh DB restores the image
  byte-for-byte (256-value byte string), with a bloat guard that
  `image_base64` is not duplicated into `extensions`.
- Regression: `Tests/Character_Chat/` 577 passed, 1 failed —
  `test_dictionary_attachment_index.py::TestMigrationBackfill::
  test_v34_database_backfills_existing_attachments` (`assert 37 == 36`)
  reproduced identically on a pristine `bb91fef73` worktree: pre-existing
  dev red (task-16197/15765 migration-rewind family), not this change.
  `Tests/UI/test_tools_settings_window.py` 65 passed (5:04) plus the new
  worker test; full-suite `--collect-only` clean (43,687 collected).
- ruff: the 3 owned files check clean; the two pre-existing modules'
  format debt and the UI test file's E402 reproduce byte-for-byte at base
  (`git show bb91fef73:... | ruff`), left untouched to keep the diff
  minimal.

**Files**: `tldw_chatbook/UI/Tools_Settings_Window.py`,
`tldw_chatbook/Character_Chat/Character_Chat_Lib.py`,
`Tests/Character_Chat/test_character_backup_export_image.py` (new),
`Tests/UI/test_tools_settings_window.py`.
