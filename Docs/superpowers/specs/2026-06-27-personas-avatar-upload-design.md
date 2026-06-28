# Personas Avatar Upload Design

## Task

TASK-100 - Wire avatar upload in the ds-native character editor.

## Goal

Restore avatar upload in the Personas character editor so users can select an image file, see the editor reflect that an avatar is staged, and persist the avatar through the normal Save flow.

## Current Context

The ds-native `PersonasCharacterEditorWidget` replaced the legacy editor with a flat form and a read-only avatar status line. The screen already owns character Save, validation, dirty state, import/export dialogs, and worker-backed file picker flows. Character card storage uses the `image` field as a BLOB/bytes value; character card export handles base64 conversion later when producing JSON or PNG card output.

## UX Decision

Avatar upload is staged until Save.

Selecting a file updates only the active editor session. It does not immediately write to the database. Save persists the staged image through the existing create/update path. Cancel, mode changes guarded by unsaved state, or leaving without saving discard the staged upload exactly like other unsaved editor changes.

## Proposed Flow

1. The character editor avatar row shows the current status and an `Upload` button.
2. Pressing `Upload` posts a `CharacterImageUploadRequested` message.
3. `PersonasScreen` handles the message only while `_edit_mode` is `create` or `edit`.
4. The screen opens `EnhancedFileOpen` in a worker, using image-only filters and a distinct `character_avatar_upload` picker context.
5. A path-based helper validates the selected path, reads image bytes, and passes those bytes to the editor.
6. The editor stages the bytes in `_character_data["image"]`, updates the status line to `Avatar: embedded`, and posts `EditorContentChanged` if the session is not already dirty.
7. The existing Save handler persists `get_character_data()` through `ccp_character_handler.create_character` or `update_character`.

## Data Handling

Store staged avatar data as raw `bytes` in the editor's pending character data. Do not base64-encode at upload time. Existing DB and export code already treat `character_cards.image` as bytes and perform export-time base64 conversion when needed.

If the loaded record contains an existing `image`, untouched saves continue to preserve it because `get_character_data()` starts from `_character_data`. Upload replaces only the pending `image` value for the active editor session.

## Validation

The picker filters should accept common local image formats: `.png`, `.jpg`, `.jpeg`, `.webp`, and `.gif`. The path-based helper must also validate:

- the path exists and is a file
- the suffix is one of the supported image extensions
- the file can be read
- empty files are rejected

Validation errors should notify the user and leave the editor state unchanged.

## Tests

Update the existing regression that asserts the upload button is absent. It should now assert that the button exists and posts `CharacterImageUploadRequested`.

Add focused coverage for:

- selected image bytes update the editor status to `Avatar: embedded`
- `get_character_data()` includes the staged bytes before Save
- staged image upload posts dirty-state notification
- the screen path helper rejects invalid paths/extensions without mutating editor state
- Save receives/persists the staged `image` bytes through the existing character save path

## ADR Check

ADR required: no

ADR path: N/A

Reason: this restores a scoped UI workflow using existing editor, file picker, dirty-state, and character persistence boundaries. It does not change schema, storage policy, sync policy, provider/runtime boundaries, or long-lived application architecture.

## Reviewed Refinements

The design review tightened the initial proposal in three places:

- stage raw bytes, not base64 text
- validate selected paths in the helper instead of relying only on picker filters
- use a dedicated avatar-upload picker context rather than reusing character import semantics
