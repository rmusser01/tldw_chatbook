---
id: TASK-34360
title: >-
  Backups taken before a ChaChaNotes schema upgrade cannot be restored
  afterwards
status: To Do
assignee: []
created_date: '2026-10-04 14:26'
labels:
  - backup-recovery
  - database
  - migrations
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The backup and restore policy accepts exactly one ChaChaNotes schema version: the current one. Once an upgrade moves the app to the next version, every backup taken before it is refused with `unsupported_schema_version`. A user who backs up, upgrades, and then needs that backup cannot use it. ChaChaNotes is the store that holds conversations, messages, characters and notes.

Measured 2026-10-04 on dev `9878fd251a` (schema v76). A database built by the v75 constructor (dev `f1f80847a4`, the commit before PR #3002) has the same 533 schema entries the v76 policy declares; only the version stamp differs. The policy still refuses it in all three checks: the owner's own validation, and restore-candidate validation both with and without migration. The v74 to v75 bump left v74 backups in the same position.

Two other stores already keep their previous version restorable: Prompts accepts v4 beside v5, and Evals declares a v5 to v6 restore-time step. ChaChaNotes has never done so.

This needs the owner's decision before any code, because keeping older backups restorable adds a step to every future ChaChaNotes schema bump. Qodo raised it on PR #3002; it was not done there because no earlier bump had done it. What the app shows the user when a restore is refused for this reason has not been traced.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The owner's decision is recorded: whether a backup from an earlier ChaChaNotes schema version stays restorable after an upgrade, and how many versions back
- [ ] #2 What the app shows today when a restore is refused for this reason is established and recorded in this task
- [ ] #3 If yes: a backup of a database built by the v75 constructor restores on the current app with its rows intact, proven by a test that goes through the real restore validation
- [ ] #4 If yes: a database carrying an accepted older version stamp but a schema other than the one declared for that version is still refused
- [ ] #5 If yes: the migration instructions tell the author of the next schema bump how to keep the previous version restorable, and a test fails when a bump drops it
- [ ] #6 If no: the user is told at restore time, in plain words rather than a reason code, that the backup predates a schema upgrade, and the limit is stated in the user documentation
<!-- AC:END -->
