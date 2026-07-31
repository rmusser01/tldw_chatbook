# Character TTS Portability Implementation Plan

> **For Codex:** Execute this plan in order with test-driven development. Keep the opt-in portability path local-only; do not add audio.cpp process management, server persistence, or standalone profile import.

**Goal:** Let users explicitly export and import a sanitized audio.cpp generation profile with local character cards, while preserving ordinary card behavior and existing assignments.

**Architecture:** Add one strict portability codec in the TTS domain and one typed local-card workflow over the existing character importer, profile service, and profile repository. The character importer strips the reserved extension before persistence and reports a created-versus-reused outcome. Textual owns user decisions; the service owns collision classification, availability checks, and profile/assignment mutations; the repository owns atomic create-plus-assignment writes.

**Tech stack:** Python 3.11+, dataclasses, Textual, SQLite, pytest/pytest-asyncio.

## Scope and decision record

- ADR required: yes
- ADR path: `backlog/decisions/028-character-tts-generation-profile-ownership.md`
- Reason: Slice 4 activates the ADR's previously deferred portability boundary and fixes the hostile-input, cross-store compensation, and local-only ownership contract.
- Backlog task: `TASK-1626`
- Approved design: `Docs/superpowers/specs/2026-07-25-character-tts-generation-profiles-design.md`

## Task 0: Amend the governing ownership decision

**Files:**

- Modify: `backlog/decisions/028-character-tts-generation-profile-ownership.md`

1. Amend ADR-028 before implementation with the explicit transient payload, hostile import, UUID-as-hint collision authority, local-only scope, and cross-database compensation decision.
2. Add `TASK-1626` to related tasks and keep managed audio.cpp lifecycle, server synchronization, and standalone import excluded.

## Task 1: Add the exact sanitized payload codec

**Files:**

- Create: `tldw_chatbook/TTS/profile_portability.py`
- Create: `Tests/TTS/test_profile_portability.py`

1. Write failing tests for the exact version-1 wire shape and deterministic standalone JSON export.
2. Add hostile-input cases for unknown fields, invalid UUIDs, overlong identifiers, unsafe names, non-finite/non-`1.0` speed, non-WAV format, non-empty audio.cpp options, payloads over 16 KiB, and container depth over four.
3. Add tests proving unknown versions/providers return a typed skip-with-warning outcome, while malformed known payloads return a typed invalid-with-warning outcome.
4. Implement immutable portable-profile values plus strict encode/decode functions by reusing `TTSProfileDraft` validation and adding exact shape/size/depth checks.
5. Run `python -m pytest Tests/TTS/test_profile_portability.py -q`.

## Task 2: Make local character persistence report created versus reused and strip the attachment

**Files:**

- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py`
- Create: `Tests/Character_Chat/test_character_tts_portability.py`
- Modify: `Tests/Character_Chat/test_character_card_lenient_import.py`
- Modify: `Tests/Character_Chat/test_character_file_operations.py`

1. Write failing JSON/PNG import tests proving the reserved namespace is extracted and removed before persistence, unrelated extensions survive, and unknown/invalid attachments still allow character import with a warning.
2. Write failing duplicate-name tests for a structured `created`/`reused` result.
3. Introduce a typed detailed import result while retaining `import_and_save_character_from_file()` as the legacy ID-returning wrapper.
4. Keep parsing and structural TTS validation before the character write; never put the reserved attachment into `extensions` stored in `character_cards`.
5. Add privacy tests that capture import logs and typed outcomes, proving message text, authority, credentials, origins, and full filesystem paths are absent; replace touched full-path logging with bounded file-type/category context.
6. Update the local CCP wrapper to expose the detailed outcome only to the Personas workflow.
7. Run the focused Character_Chat tests.

## Task 3: Add collision reads and atomic create-plus-assignment persistence

**Files:**

- Modify: `tldw_chatbook/TTS/profile_repository.py`
- Modify: `tldw_chatbook/TTS/profile_service.py`
- Modify: `tldw_chatbook/TTS/profile_types.py`
- Modify: `Tests/TTS/test_profile_repository.py`
- Modify: `Tests/TTS/test_profile_service.py`

1. Write failing repository tests for exact UUID and normalized-name collision lookup, caller-selected UUID creation, and one transaction that creates a profile and its exact assignment.
2. Add rollback tests proving a simulated assignment failure leaves neither the new profile nor assignment.
3. Add repository APIs that retain lifecycle-generation fencing and perform combined writes on the serialized worker.
4. Write failing service tests for every UUID/name/generation-tuple collision combination, including two different colliding rows, explicit reuse/copy requirements, collision-safe names, and new UUID generation for copies.
5. Add typed inspect/commit methods. Recheck the caller-held repository generation and expected current assignment; revalidate capability authority before any assignment.
6. Prove unavailable profiles may be persisted for repair but never assigned, and a reused character's current assignment is unchanged.
7. Run the focused repository and service tests.

## Task 4: Add opt-in character-card and standalone export

**Files:**

- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Modify: `tldw_chatbook/UI/stts_profile_library.py`
- Modify: `Tests/Character_Chat/test_character_tts_portability.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify: `Tests/UI/test_stts_profile_library.py`

1. Write failing tests proving default JSON/PNG export remains TTS-free and unchanged, while an explicit payload is added only to a transient deep copy and unrelated extension namespaces survive.
2. Add fail-closed tests for a malformed or occupied reserved namespace.
3. Extend the existing export functions with an opt-in portable payload parameter whose default preserves the existing path.
4. Add explicit Personas controls to include the currently assigned profile; do not infer inclusion from assignment presence.
5. Add an Export action to the STTS profile library that writes the same standalone sanitized payload. Do not add standalone import.
6. Run the focused export and UI tests.

## Task 5: Orchestrate hostile import, prompts, and compensation in Personas

**Files:**

- Create: `tldw_chatbook/Widgets/Persona_Widgets/character_tts_portability_dialogs.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify: `Tests/UI/test_uat_first_time_character_chat.py`

1. Write failing UI tests for collision reuse/copy prompts, existing-character apply confirmation, cancellation, unavailable-profile repair messaging, and partial failure messaging.
2. Parse and structurally validate the attachment and evaluate current profile availability before any character/profile write; add an ordering test that blocks persistence until both checks complete.
3. Persist the character through the detailed local importer, then resolve collision prompts. Revalidate current availability and repository generation at commit before any assignment.
4. For new characters, assign only an available profile. For reused characters, require explicit apply confirmation and compare against the exact existing assignment.
5. On cancellation, perform no profile/assignment mutation. On profile failure, keep a new character imported and unassigned; preserve a reused character's prior assignment.
6. Refresh/select the character using the structured outcome instead of before/after row counts.
7. Add tests proving portability logs, events, notifications, exported payloads, and metrics exclude text, authority, credentials, origins, and full filesystem paths.
8. Add an end-to-end local-card UAT test covering explicit export, import, assignment, and complete-WAV roleplay speech resolution without managing the audio.cpp process.

## Task 6: Close documentation and verification

**Files:**

- Modify: `backlog/tasks/task-1626 - Add-sanitized-TTS-portability-to-local-character-cards.md`

1. Run focused tests for the new codec/workflow and the existing card/profile/UI suites.
2. Run `ruff check` on changed Python files, `mypy` on new/changed domain modules where the project configuration supports it, and `git diff --check`.
3. Self-review for accidental logging/export of text, authority, origins, credentials, or filesystem paths.
4. Check all task acceptance criteria, add concise Implementation Notes with verification evidence, and move `TASK-1626` to Done only after every Definition of Done condition is satisfied.

## Verification commands

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_portability.py \
  Tests/Character_Chat/test_character_tts_portability.py \
  Tests/Character_Chat/test_character_file_operations.py \
  Tests/Character_Chat/test_character_card_lenient_import.py \
  Tests/Character_Chat/test_character_dictionaries_portability.py \
  Tests/TTS/test_profile_types.py \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_service.py \
  Tests/UI/test_stts_profile_library.py \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_uat_first_time_character_chat.py -q

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check <changed-python-files>
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/TTS/profile_portability.py
git diff --check
```

Baseline before implementation: 789 focused existing tests passed on `origin/dev` commit `962bc0698` in 336.31 seconds; one third-party `requests` dependency-version warning was emitted.
