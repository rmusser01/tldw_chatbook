---
id: TASK-21249
title: >-
  Notes-sync lazy-property residue - late-bound reads and a build failure
  painted as feature-unavailable
status: To Do
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
