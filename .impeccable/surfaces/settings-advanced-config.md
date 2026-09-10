# Settings — Advanced Config

Mode: Operate. Disposition: ship for the reviewed draft-recovery scope.

## Authority and ownership

This hardens the incumbent Textual Settings editor. The main project's
PRODUCT.md and DESIGN.md remain the product and visual authority; no new visual
system is introduced. ADR-033 owns the guarded raw-TOML commit model, and ADR-031
owns shortcut honesty and discard confirmation.

The controller in `UI/Screens/settings_advanced_config.py` owns the live draft,
validation revision, exact file/profile baseline, and operation state. The
existing memory-only screen-state store retains that session across Settings
recreation. App-owned workers finish after navigation; views attach/detach
callbacks with ownership checks. No raw draft body enters durable UI metadata.
The config owner compares and replaces under its existing file lock, preserving
backup, encryption and protected-section rules.

## Interaction contract

Preserve the order: pinned unsaved/commit banner, expandable raw editing guide,
Validate/Save and Load Backup/Revert controls, validation/result status, editor.
The raw document owns its scroll position. At 80×24 the inspector is hidden by
the existing compact workbench; all four controls and at least three editor
content rows remain available. Keep native controls and existing semantic tokens.

Typing changes the dirty marker immediately, including invalid and empty text.
Status updates must never erase input waiting for its TextArea change event.
Validate applies to the current revision only; it establishes TOML syntax, not
provider readiness. Save is disabled until that revision validates. A save clears
only its submitted text and preserves later edits. Restore keyboard focus after
an operation only when the user has not moved focus elsewhere.

Load Backup previews text without saving. Load Backup and Revert ask before
replacing an unsaved draft, and a later edit supersedes an earlier confirmation.
The r shortcut is available outside text entry; Esc cancels the discard dialog.
If another writer or profile changes the baseline, retain the draft and block
Save with explicit copy/reload/reapply guidance. Do not merge arbitrary raw TOML
or offer an unconditional overwrite from a stale baseline.

## Evidence and limits

Mounted production-CSS captures under `.impeccable/review/raw-config/` show entry
and dirty state at 120×35 and 80×24. Textual SVG exports were rasterized with a
local Menlo fallback for Fira Code. This is layout and keyboard evidence, not a
participant usability study or live-provider verification.

Independent review reproduced four async/lifecycle defects in the first pass;
all four received regression tests and repairs. Re-review reported no remaining
actionable findings and independently passed 34 raw-draft/snapshot tests. The
final task notes record the broader targeted compatibility gate. User-facing
instructions are in `Docs/User_Guide/settings.md`, Expert — Advanced Config.
