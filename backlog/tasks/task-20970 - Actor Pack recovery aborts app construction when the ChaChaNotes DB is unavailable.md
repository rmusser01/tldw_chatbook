---
id: TASK-20970
title: >-
  Actor Pack recovery aborts app construction when the ChaChaNotes DB is
  unavailable
status: Done
assignee: []
created_date: '2026-08-22'
updated_date: '2026-08-22'
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

- [x] Constructing `TldwCli` with no ChaChaNotes database succeeds, and the app
      reaches the same degraded-but-running state it reached before TASK-19057
- [x] The Actor Pack wiring makes an explicit decision about a missing
      database rather than dereferencing it, consistent with how the sibling
      wiring in the same constructor treats the same value
- [x] A missing database produces an operator-legible diagnostic, not an
      unhandled `AttributeError`
- [x] The recovery call's failure handling covers the errors the repository can
      actually raise, so a repository failure cannot escape the guard that is
      written to contain it
- [x] A test builds the app with no ChaChaNotes database and fails if
      construction raises — so this specific regression cannot return silently
- [x] The 294 failures across `Tests/App`, `Tests/Scheduling`,
      `Tests/Subscriptions` and `Tests/Watchlists` are green
- [x] The 4 `Tests/UI/test_settings_*` collection errors are gone and a
      repo-wide `pytest --collect-only` no longer reports them
- [x] A degraded start is verified end to end at least once against a real
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

## Implementation Plan

1. Reproduce at `origin/dev` in a fresh worktree: run the four named suites and
   the four `Tests/UI/test_settings_*` modules, and confirm every failure ends
   in the one frame chain the filing names.
2. Write the born-red regression coverage first, in two shapes: the cheap one
   through the shared app factory (which patches `get_chachanotes_db_lazy` to
   `None`), and an end-to-end one against a genuinely unopenable database file
   on disk with nothing patched out.
3. Fix the layer that is actually wrong. Two separate defects, kept separate:
   the wiring dereferences a value it should decide about, and the repository
   lets a `None` database escape its own error boundary.
4. Mutation-check both halves of the guard independently, then Edit-restore and
   confirm the diff is byte-identical.
5. Re-measure the four suites, the settings modules, and a repo-wide
   `--collect-only`, each against a pinned `origin/dev` baseline.
6. Regenerate the production-diagnostic inventory, reading the changed
   statements with `--statements` before writing.

## Implementation Notes

Two independent repairs, deliberately not collapsed into one.

**The wiring now decides.** `_wire_character_persona_services` reads
`chachanotes_db` once and branches on it, exactly as the sibling
`_wire_chat_conversation_services` does with the same value: when it is `None`,
`actor_pack_repository`, `persona_actor_pack_coordinator` and
`actor_pack_creation_service` are all set to `None`, `actor_pack_recovery_error`
records the new fixed category `actor_pack_recovery_unavailable`, and one
operator-legible ERROR is logged. Nothing else in the method is skipped — the
branch is an `if/else`, not an early `return`, so the chat-dictionary and
scope-service wiring below it still runs. `PersonasScreen` already guards
`actor_pack_creation_service is None` and notifies "Actor Pack creation is
unavailable.", so the degraded app has a real user-facing story rather than a
latent crash. The diagnostic:

    Actor Pack services disabled: the ChaChaNotes database could not be
    opened, so portable-identity recovery was skipped and Actor Pack creation
    is unavailable for this session (actor_pack_recovery_unavailable)

**The repository now raises its own typed error.** `ActorPackRepository.
__init__` rejects a `None` database with
`ActorPackRepositoryError("actor_pack_repository_unavailable")`. This is the
durable half and it was chosen over the two alternatives on purpose. Widening
the call site to `except Exception` would have contained *this* symptom while
also swallowing genuine programming errors in a startup path. Adding
`AttributeError` to each query method's `except` tuple would mean ten sites
each remembering, and would swallow real attribute bugs inside the repository
itself. A single construction-time check converts every method on the class
into the repository's own typed category, and it is a true statement about the
class's contract rather than a patch on one caller. The call site's `except`
was widened to `(PersonaActorPackCoordinatorError, ActorPackRepositoryError)`
as well: `recover()` does convert the repository's failures today, but it sits
above a store with its own error type and only that store knows what it can
raise — naming both is what makes "a repository failure cannot escape this
guard" true rather than incidental.

**Evidence.** Born-red at `origin/dev` `80248f3e4`: both new tests fail with
`AttributeError: 'NoneType' object has no attribute 'execute_query'` through
`app.py:6010` → `app.py:6612` → `persona_coordinator.py:174` →
`repository.py:276`. Four suites `294 failed, 1795 passed, 1 skipped` →
`2091 passed, 1 skipped` (the +2 are the new tests). The four
`Tests/UI/test_settings_*` modules: 4 collection errors → `22 passed`.
Repo-wide `--collect-only`: `56524 collected, 5 errors` → `56549 collected,
1 error`, the survivor being TASK-20972's unrelated parametrize-signature
error; the +25 is 22 settings tests + 2 app tests + 1 repository test.
Mutation: removing only the repository check reds the repository unit test and
leaves the app tests green; disabling only the wiring guard reds both app tests
with the typed `ActorPackRepositoryError`; removing both reproduces the original
`AttributeError` verbatim. Edit-restore returned a byte-identical diff
(sha256 `7bd13d55…`).

**A fifth shadow, baselined and fixed.**
`Tests/Character_Chat/test_character_persona_scope_service.py::
test_app_wires_character_persona_services` was already red at `origin/dev` with
`AttributeError: 'object' object has no attribute 'execute_query'` — it passed
a bare `object()` as `chachanotes_db`, which was fine while the wiring only
forwarded the value and stopped being fine when TASK-19057 made it query.
Replaced with a two-line double that reports no intents, so the test is about
the scope-service wiring it is named for. Not fixed here and not caused here:
`Tests/UI/test_console_runtime_ownership.py::
test_app_fences_console_then_drains_buddy_before_profile_teardown`, red before
and after with `'TldwCli' object has no attribute 'notes_sync_runtime_owner'`
(the other four reds in that module were this defect and are now green).

**Files:** `tldw_chatbook/app.py`,
`tldw_chatbook/Actor_Packs/repository.py`,
`Tests/App/test_app_degraded_without_chachanotes_db.py` (new),
`Tests/Actor_Packs/test_actor_pack_repository.py`,
`Tests/Character_Chat/test_character_persona_scope_service.py`,
`Docs/security/production-diagnostic-inventory.json` (regenerated: +1 call —
the new diagnostic, plus the pre-existing `actor_pack_recovery_failed` call
re-wrapped and one warning re-indented; all constant strings, no interpolation),
`backlog/docs/lessons-testing-evidence.md`.
