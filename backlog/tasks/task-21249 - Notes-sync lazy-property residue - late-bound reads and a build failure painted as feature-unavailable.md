---
id: TASK-21249
title: >-
  Notes-sync lazy-property residue - late-bound reads and a build failure
  painted as feature-unavailable
status: Done
assignee: []
created_date: '2026-08-23'
labels:
  - notes-sync
  - error-handling
  - technical-debt
dependencies: []
priority: medium
---

## Description

Source: close-out of the 2026-08-22 holistic performance review burn-down; two findings from
the TASK-21108 review, filed together because both are consequences of the same change —
turning `notes_sync_runtime_owner` into a lazily-built property.

Both are against `origin/fix/task-21108-wave5`, which was **not merged into dev** at close-out.
Re-confirm against dev after it lands.

1. **Late-bound reads.** That branch's own new lesson is that a lazily-constructed owner must
   not capture app state eagerly — and it applies that to two of six reads. Four remain in
   `_construct_notes_sync_runtime_owner` (branch `app.py:6117`): `app_config`,
   `_instance_lock_status.acquired`, `notes_user_id`, and the two path getters. A value read
   at first-property-access rather than at the construction the caller believes it triggered is
   ordering dependence that only shows up when the access point moves — which is exactly what
   TASK-21247 proposes to do next.
2. **A build failure shown as feature-unavailable.** `library_screen.py:3241` reads the runtime
   with `getattr(app_instance, "notes_sync_runtime_owner", None)`. Once that name is a property
   that builds on access, the `getattr` default swallows an `AttributeError` raised **inside**
   the build, and the screen paints "awaiting_cutover" — a construction failure presented to
   the user as "this feature is not available yet", with nothing distinguishing the two. On dev
   today the name is a plain attribute (`app.py:6032`) so the swallow is unreachable; it
   becomes reachable the moment the property lands.

## Acceptance Criteria

- [ ] The values the notes-sync runtime owner depends on are read at one well-defined point, and moving the property's first access does not change what it is built with
- [ ] A build failure inside the notes-sync runtime property is not reported to the user as "awaiting cutover" — it is distinguishable in the UI and logged
- [ ] A test raises an `AttributeError` from inside the property build and fails if the Library screen reports feature-unavailable
- [ ] Neither change re-introduces an eager construction on a zero-profile boot

## Implementation Plan

1. Item 1 (late-bound reads): make `_construct_notes_sync_runtime_owner`
   name and document its single read point -- every value the owner is built
   from is captured inside the construct, under the property lock, and
   flows only into the build call. Pin with a test that mutates the app
   slots after first access and asserts the built owner is unchanged.
2. Item 2 (build failure): add `_read_notes_sync_runtime_owner_safe()` to
   the wiring mixin -- property-descriptor check, try/except with the
   failure recorded on the app and logged at ERROR; the Library screen's
   three read sites go through it, and the mount site hands the controller
   an `InertLastingSyncRuntime(status="failed", next_action="review_settings")`
   when a build failure is recorded -- distinguishable from
   ("awaiting_cutover", "finish_upgrade") in both UI and logs.
3. Tests: safe-reader records/memoizes/clears-on-double; failed-vs-cutover
   snapshot discrimination; construction read-point freeze.

## Implementation Notes (2026-10-11, dev `a3c47c6c72`)

- **Item 1**: the construct's body now reads `app_config`, `notes_user_id`,
  the state path, and the watcher intervals into locals at the top of the
  locked construct (the docstring names this THE read point);
  `profile_process_is_sole` stays an inline
  `self._instance_lock_status.acquired` read because the restart-cutover
  fence test (`Tests/Notes/test_notes_sync_cutover.py`) pins that AST shape
  -- it is still a single read inside the same construct.
  `test_owner_inputs_are_read_once_at_construction_not_at_use` pins the
  freeze: post-construction slot mutations never reach the built owner.
  With TASK-21247's gate, the property's first access is now deterministic
  (on_mount when configured, or first-time setup / Library mount), closing
  the ordering-dependence window the task described.
- **Item 2**: `_read_notes_sync_runtime_owner_safe()` on
  `ServiceWiringMixin` (logged once at ERROR, sticky until a double is
  installed through the setter, which clears it). Library screen sites at
  mount wiring, the note-session port lambda, and the folder-sync-root
  probe all read through it (falling back to the old getattr for harness
  doubles that lack the mixin). `InertLastingSyncRuntime` gained
  status/next_action construction so a build failure presents as
  ("failed", "review_settings") -- the failed status branch the controller
  already renders -- not "awaiting_cutover".
- **Evidence**: `Tests/App/test_notes_sync_boot_gate.py` (safe reader +
  freeze tests), `Tests/UI/test_library_notes_lasting_sync_flow.py` +
  `test_library_notes_w4_sync_roots.py` 13 passed; cutover suite identical
  to HEAD baseline (2 pre-existing local env failures).
