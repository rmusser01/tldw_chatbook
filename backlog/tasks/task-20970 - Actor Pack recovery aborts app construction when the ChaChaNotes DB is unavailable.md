---
id: TASK-20970
title: >-
  Actor Pack recovery aborts app construction when the ChaChaNotes DB is
  unavailable
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - bug
  - app-startup
  - database
  - resilience
  - test-integrity
priority: high
dependencies:
  - TASK-19057
---

## Description

Source: surfaced while baselining **TASK-19561** against a clean `origin/dev`
worktree, and independently while renumbering **TASK-19564**'s migration after
the schema-version collision with TASK-19057. Re-verified at `684c6aba4`.

`TldwCli.__init__` has always tolerated an absent ChaChaNotes database. When
the DB cannot be opened it logs `ChaChaNotesDB (CharactersRAGDB) instance not
found/assigned in app.__init__` (`app.py:5956`), sets `self.chachanotes_db =
None`, and every downstream wiring degrades around it — the sibling
`_wire_chat_conversation_services` guards the same value explicitly
(`app.py:6664`, `:6678`, `:6705`).

TASK-19057's Actor Packs wiring does not. `_wire_character_persona_services`
constructs `ActorPackRepository(self.chachanotes_db)` unconditionally
(`app.py:6605`) and then calls `recover()` (`app.py:6612`). With no database
the repository dereferences `None` (`Actor_Packs/repository.py:276`) and raises
`AttributeError`. The call is wrapped in `except
PersonaActorPackCoordinatorError`, which is a `ValueError` subclass
(`Actor_Packs/persona_coordinator.py:22`) — so the error escapes the guard that
exists, and construction of the application object fails outright.

**This is not confined to the test harness.** `get_chachanotes_db_lazy()`
returns `None` on *any* failure to open the database (`config.py:7011-7016`) —
a corrupt file, a permission error, or a migration that cannot complete. That
was measured directly, without patching anything: pointing an isolated
`HOME`/`XDG` at a ChaChaNotes file containing non-SQLite bytes gives
`CharactersRAGDBError: file is not a database`, the loader returns `None`, and
the app is then in exactly the state its own error branch anticipates. A
degraded-but-running app is now an app that cannot be constructed at all, and
the failure is an unhandled `AttributeError` rather than the operator-legible
message the existing branch was written to produce. TASK-19860 (migration `.sql`
files missing from the wheel) is a shipped example of a real cause.

Measured symptoms at `684c6aba4`, all one root cause — every failure below
terminates in the identical frame chain `app.py:6010` →
`_wire_character_persona_services` `app.py:6612` → `repository.py:276`:

- **294 test failures** across `Tests/App`, `Tests/Scheduling`,
  `Tests/Subscriptions` and `Tests/Watchlists` (`294 failed, 1795 passed`), of
  which **294 of 294** report `AttributeError: 'NoneType' object has no
  attribute 'execute_query'` and nothing else. `Tests/Watchlists/
  test_watchlists_artifacts_pane.py` alone contributes 122.
- **4 collection errors** in `Tests/UI`.
  `test_settings_category_sweep.py:27-29` builds an app at module scope, so it
  errors during collection, and `test_settings_footer_hints.py`,
  `test_settings_model_catalog_layout.py` and
  `test_settings_save_commit_models.py` cascade into `ImportError` because they
  import from it.
- A clean repo-wide `pytest --collect-only` therefore exits non-zero.

## Acceptance Criteria

- [ ] Constructing `TldwCli` with no ChaChaNotes database succeeds, and the app
      reaches the same degraded-but-running state it reached before TASK-19057
- [ ] The Actor Pack wiring makes an explicit decision about a missing
      database rather than dereferencing it, consistent with how the sibling
      wiring in the same constructor treats the same value
- [ ] A missing database produces an operator-legible diagnostic, not an
      unhandled `AttributeError`
- [ ] The recovery call's failure handling covers the errors the repository can
      actually raise, so a repository failure cannot escape the guard that is
      written to contain it
- [ ] A test builds the app with no ChaChaNotes database and fails if
      construction raises — so this specific regression cannot return silently
- [ ] The 294 failures across `Tests/App`, `Tests/Scheduling`,
      `Tests/Subscriptions` and `Tests/Watchlists` are green
- [ ] The 4 `Tests/UI/test_settings_*` collection errors are gone and a
      repo-wide `pytest --collect-only` no longer reports them
- [ ] A degraded start is verified end to end at least once against a real
      unopenable database file, not only against a patched-out service

## Notes

Filed as one task on measured evidence rather than as three. The
`test_app_watchlists_db_wiring` failure, the 8 `Tests/App` + 2
`Tests/Scheduling` reds, and the 4 `Tests/UI/test_settings_*` collection errors
were each reported separately; running them showed 294 of 294 failures ending
in the same three frames, so they are one defect with several shadows.

Two consequences worth keeping in view while this is open. First, this is
currently **masking** other test signal — `test_watchlists_artifacts_pane.py`
is entirely red from this cause, including the test TASK-20978 is about, so
that flake cannot be re-measured until this is fixed. Second, the reason the
None branch exists at all is that it was reachable; nothing about TASK-19057
made it less so.
