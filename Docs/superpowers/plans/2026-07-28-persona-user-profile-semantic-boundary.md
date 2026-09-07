# Persona/User Profile Semantic Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Roleplay's inverted identity model so Personas are assistant-side profiles, authenticated User Profiles remain the human-account domain, and no Persona is projected as the human in Roleplay or Console.

**Architecture:** Keep the existing local JSON store, server Persona endpoints, scope service, Personas workbench, and Console session machinery. Narrow the server wire DTOs to the exact `tldw_server` contract, add explicitly local Persona mutation DTOs, rename the callable Persona surface without compatibility aliases, remove the obsolete active-human Persona resolver, and reduce Console identity presentation to assistant/character/Persona labels. Compatibility is handled only while reading existing persisted records or settings; no migration, parallel service, new store, or TTS behavior is added.

**Tech Stack:** Python 3.12, Pydantic v2, asyncio, Textual 8, JSON file persistence, httpx through the existing API client, pytest/pytest-asyncio, Ruff, mypy, Backlog.md.

**Task:** `TASK-617.1`

**Parent:** `TASK-617`

**ADR required:** yes
**ADR path:** `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`
**Reason:** ADR-037 already governs the Persona/User Profile boundary, source-specific Persona contracts, inert legacy projections, and the ordered Slice 3A delivery. This task implements only its 3A.1 decision; no new ADR, schema migration, store, dependency, TTS boundary, or process-runtime decision is introduced.

---

## Scope boundary

This plan implements only approved Slice 3A.1:

- reserve `UserProfile*` names for authenticated human/account contracts;
- make the Persona server DTOs exact mirrors of `tldw_server`;
- retain local-only Persona fields behind local mutation DTOs;
- preserve unknown fields in existing local Persona JSON records;
- rename Persona runtime methods, mode IDs, entity kinds, widgets, events, and tests;
- remove Persona-as-human controls, markers, configuration access, and macro substitution;
- remove the Persona-as-human Console chip, row, setting, and `As:` label;
- ignore legacy Console identity projection keys on restore without migrating stored bytes;
- retain current Persona enabled/disabled behavior and existing global TTS behavior.

It does **not** implement:

- Persona chat authority, Persona memory/runtime parity, Persona exemplars, or new macro aliases;
- character authority or `assistant_authority_id`;
- a conversation schema or Sync V2 change;
- trusted Console speech snapshots;
- TTS assignment mutation, resolution, or visible assignment controls;
- Persona TTS inheritance or Persona-specific voices;
- a genuine Roleplay User Profile editor or account-to-roleplay identity mapping;
- saved-workbench-mode migration;
- managed audio.cpp binary discovery, launch, supervision, restart, or shutdown.

## Fixed terminology and contracts

Use these names end to end:

```python
class PersonaProfileCreate(BaseModel): ...
class PersonaProfileUpdate(BaseModel): ...
class PersonaProfileResponse(BaseModel): ...

class LocalPersonaProfileCreate(BaseModel): ...
class LocalPersonaProfileUpdate(BaseModel): ...

async def list_persona_profiles(...): ...
async def get_persona_profile(...): ...
async def create_persona_profile(...): ...
async def update_persona_profile(...): ...
async def delete_persona_profile(...): ...
async def restore_persona_profile(...): ...

PersonaWorkbenchMode = Literal["characters", "personas", ...]
PersonaEntityKind = Literal["character", "persona", ...]
```

There are no Persona-domain `UserProfile*` aliases, no fallback calls to
`*_user_profile`, and no dual mode IDs. Genuine account types under
`tldw_api/auth_user_schemas.py` retain their current `UserProfile*` names.

The server request field sets remain exact to
`tldw_Server_API/app/api/v1/schemas/persona.py`:

```python
SERVER_PERSONA_CREATE_FIELDS = {
    "id",
    "name",
    "archetype_key",
    "character_card_id",
    "mode",
    "system_prompt",
    "is_active",
    "use_persona_state_context_default",
    "voice_defaults",
    "setup",
}

SERVER_PERSONA_UPDATE_FIELDS = {
    "name",
    "character_card_id",
    "mode",
    "system_prompt",
    "is_active",
    "use_persona_state_context_default",
    "voice_defaults",
    "setup",
}

SERVER_PERSONA_RESPONSE_FIELDS = {
    "id",
    "name",
    "archetype_key",
    "character_card_id",
    "origin_character_id",
    "origin_character_name",
    "origin_character_snapshot_at",
    "mode",
    "system_prompt",
    "is_active",
    "use_persona_state_context_default",
    "voice_defaults",
    "setup",
    "created_at",
    "last_modified",
    "version",
    "buddy_summary",
}

LOCAL_PERSONA_CREATE_FIELDS = SERVER_PERSONA_CREATE_FIELDS | {
    "description",
    "personality_traits",
}

LOCAL_PERSONA_UPDATE_FIELDS = SERVER_PERSONA_UPDATE_FIELDS | {
    "description",
    "personality_traits",
}
```

Local create/update DTOs additionally support `description` and the current
freeform `personality_traits`. Local updates serialize with
`exclude_unset=True` and without `exclude_none=True`, then merge into the
stored record. Omitted means unchanged; explicit `None` clears a nullable
field. Unknown existing keys survive the merge and the existing
`tldw_chatbook_personas.json` filename does not change.

All four mutation DTOs reject extra keys. This prevents Pydantic from silently
discarding local-only fields at the server boundary and prevents callers from
mutating reserved local persistence fields such as `version`, `deleted`,
timestamps, or origin snapshots. Local list/get operations remain lossless
dictionary views; they do not coerce a stored record through the narrower
server response model.

The workbench mode is:

```text
id: personas
label: Personas
description: Personas — assistant profiles for roleplay and chat
entity kind: persona
```

`is_active` means enabled/disabled only. The workbench does not expose
**Set as my name**, **Clear my name**, **Chatting as**, or an active-human
marker.

Console and shell identity presentation is:

```text
generic session       -> Assistant: General
character session     -> Character: <name>
existing Persona chat -> Persona: <assistant_name-or-assistant_id>
```

New Console settings contain `character_label` only as the existing
character-handoff display projection. They contain neither `persona_label`
nor `user_profile_label`. Restore explicitly drops both legacy keys before
constructing settings. No migration scans or rewrites stored session blobs or
transcripts. The single presentation item may accept already-available
`assistant_kind`, `assistant_name`, and `assistant_id` values to render an
existing Persona session, but this task does not add or persist those identity
fields to native Console sessions.

## Supported-interpreter baseline

Use the repository's existing Python 3.12 environment:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python --version
```

Baseline result on the docs-only branch rebased to `origin/dev`:

```text
Python 3.12.11
```

`Tests/UI/test_chat_shell_bar.py` currently fails collection because it imports
the already-removed `TabState` from `chat_screen_state.py`. Because this task
must change the shell's Persona label, replace that obsolete test fixture with
a local lightweight session object while updating the shell assertions.

The remainder of the initial focused suite produced:

```text
727 passed, 7 failed, 1 warning in 558.61s
```

The seven failures are inherited and outside Slice 3A.1:

- `Tests/UI/test_personas_workbench.py::TestImportExport::test_import_failure_shows_recovery_copy`
- `Tests/UI/test_console_session_settings.py::test_console_remote_defaults_use_smoke_verified_models`
- `Tests/UI/test_console_session_settings.py::test_console_model_resolution_includes_runtime_discovered_models`
- `Tests/UI/test_console_session_settings.py::test_console_missing_key_recovery_action_is_provider_specific`
- `Tests/UI/test_console_session_settings.py::test_real_journey_settings_save_unblocks_console_without_restart`
- `Tests/UI/test_console_session_settings.py::test_console_resolution_view_suppresses_boot_echo_reactives`
- `Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_choose_model_opens_settings`

Do not silently fix those unrelated failures or claim the pre-existing broad
gate is green. Run task-focused node IDs or deselect the exact inherited tests,
then separately report the unchanged broad baseline.

## File responsibility map

| File | Responsibility |
| --- | --- |
| `backlog/tasks/task-617.1 - Establish-the-Roleplay-Persona-and-User-Profile-semantic-boundary.md` | Atomic acceptance criteria, plan summary, ADR link, and final evidence |
| `Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md` | Approved Slice 3A integration contract |
| `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md` | Canonical domain and compatibility decision |
| `tldw_chatbook/tldw_api/character_persona_schemas.py` | Exact server Persona DTOs and distinct local Persona mutation DTOs |
| `tldw_chatbook/tldw_api/__init__.py` | Persona exports without Persona-domain `UserProfile*` aliases |
| `tldw_chatbook/tldw_api/client.py` | PATCH omission-versus-null serialization |
| `tldw_chatbook/Character_Chat/character_persona_scope_service.py` | Exact mode-aware Persona method surface |
| `tldw_chatbook/Character_Chat/server_character_persona_service.py` | Exact Persona wrapper names and wire DTO types |
| `tldw_chatbook/Character_Chat/local_character_persona_service.py` | Persona-named local storage surface and non-destructive source-specific merges |
| `tldw_chatbook/Character_Chat/persona_list_paging.py` | `page_persona_profiles` paging helper |
| `tldw_chatbook/Character_Chat/active_user_profile.py` | Delete after every import and call is removed |
| `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py` | Persona service calls, copy, and source-specific save requests |
| `tldw_chatbook/UI/Screens/personas_screen.py` | `personas` mode, source-specific editor/save behavior, and removal of active-human actions |
| `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py` | Neutral `User` placeholder and Persona mode name |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py` | `personas` mode and `persona` entity kind |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py` | Persona mode/state labels |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py` | Persona-named edit/save events |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py` | Renamed Persona editor; source-aware local-only fields |
| `tldw_chatbook/Widgets/Persona_Widgets/user_profile_editor_widget.py` | Delete through a file move; do not retain a compatibility module |
| `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py` | Persona edit event |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py` | Remove active-human summary/button/marker behavior |
| `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py` | Persona mode behavior |
| `tldw_chatbook/Chat/console_display_state.py` | One assistant identity label rather than character-plus-human-Persona labels |
| `tldw_chatbook/Chat/console_session_settings.py` | Remove live user-Persona setting and render generic/character identity copy |
| `tldw_chatbook/Widgets/Console/console_workbench_state.py` | Remove Persona-as-human workbench mode |
| `tldw_chatbook/Widgets/Console/console_status_chips.py` | Remove Persona-as-human chip and use assistant identity chip |
| `tldw_chatbook/Widgets/Console/console_control_bar.py` | Remove Persona-as-human label from summary and sync |
| `tldw_chatbook/Widgets/Console/console_settings_modal.py` | Remove read-only User Profile row |
| `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py` | Character/Persona/generic assistant labels without `As:` |
| `tldw_chatbook/UI/Screens/chat_screen.py` | Ignore legacy settings keys, stop Persona-to-human handoff substitution, and stop emitting a user-Persona setting |
| `tldw_chatbook/app.py` | Rename the local Persona store constructor keyword while preserving its path |
| `Tests/tldw_api/test_character_persona_schemas.py` | Exact type ownership and field-set contracts |
| `Tests/tldw_api/test_character_persona_client.py` | PATCH omission-versus-explicit-null request bodies |
| `Tests/Character_Chat/test_character_persona_scope_service.py` | Persona-only dispatch and capability reporting |
| `Tests/Character_Chat/test_local_character_persona_service.py` | Existing-file preservation, unknown fields, and omitted/null merges |
| `Tests/Character_Chat/test_persona_list_paging.py` | Persona pager name and behavior |
| `Tests/Character_Chat/test_active_user_profile.py` | Delete with the removed implementation |
| `Tests/UI/test_ccp_handlers.py` | Persona method names and source-specific requests |
| `Tests/UI/test_persona_profile_widgets.py` | Persona editor/event names and source-aware fields |
| `Tests/UI/test_personas_*.py` | Workbench mode/copy/actions, neutral human fallback, and unchanged restore floor |
| `Tests/Chat/test_console_display_state.py` | Assistant identity presentation state |
| `Tests/Chat/test_console_session_settings.py` | Legacy-key ignore/new-key omission serialization contract |
| `Tests/Chat/test_console_run_status_surfaces.py` | Updated assistant identity state fixtures |
| `Tests/UI/test_chat_shell_bar.py` | Generic/character/Persona shell labels and obsolete fixture repair |
| `Tests/UI/test_console_session_settings.py` | Summary/modal identity copy and no User Profile row |
| `Tests/UI/test_console_status_chips.py` | No Persona-as-human chip |
| `Tests/UI/test_console_workbench_contract.py` | One assistant identity control and no Persona mode |
| `Tests/TTS/test_console_speak_autoplay.py` | Unchanged global Console TTS regression |

## Task 1: Introduce strict local Persona mutation DTOs without removing live APIs

**Files:**

- Modify: `tldw_chatbook/tldw_api/character_persona_schemas.py`
- Modify: `Tests/tldw_api/test_character_persona_schemas.py`

This preparatory task is deliberately additive. The existing wire DTOs,
aliases, services, and consumers remain unchanged until Task 4 can remove them
with every call site in one green commit.

- [ ] **Step 1: Write failing exact local-field and strictness tests**

Add tests for the new local DTOs:

```python
assert set(LocalPersonaProfileCreate.model_fields) == LOCAL_PERSONA_CREATE_FIELDS
assert set(LocalPersonaProfileUpdate.model_fields) == LOCAL_PERSONA_UPDATE_FIELDS
```

Assert both reject unknown or persistence-owned fields such as
`origin_character_id`, `version`, `deleted`, `created_at`, and an arbitrary
`future_extension`. Assert explicit nullable fields remain present in
`model_fields_set` while omitted fields do not.

- [ ] **Step 2: Run the red schema tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/tldw_api/test_character_persona_schemas.py -q
```

Expected: failures because the local mutation DTOs do not exist.

- [ ] **Step 3: Add the local DTOs only**

Add `LocalPersonaProfileCreate` and `LocalPersonaProfileUpdate` with the exact
field sets above and `ConfigDict(extra="forbid")`. Do not narrow
`PersonaProfileCreate`, remove `UserProfile*` aliases, change exports, or
switch a runtime consumer yet.

- [ ] **Step 4: Run the focused tests and an import gate**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/tldw_api/test_character_persona_schemas.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/tldw_api
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/tldw_api/character_persona_schemas.py \
  Tests/tldw_api/test_character_persona_schemas.py
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/tldw_api/character_persona_schemas.py \
  Tests/tldw_api/test_character_persona_schemas.py
git commit -m "refactor(personas): add strict local mutation contracts"
```

## Task 2: Remove Persona-as-human runtime behavior before renaming APIs

**Files:**

- Modify: `tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Delete: `tldw_chatbook/Character_Chat/active_user_profile.py`
- Modify: `Tests/UI/test_personas_preview.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Delete: `Tests/Character_Chat/test_active_user_profile.py`

- [ ] **Step 1: Write failing neutral-human and non-mutation tests**

Replace active-pointer tests with assertions that:

- character greetings replace `{{user}}` with literal `User`;
- local and server runtime modes produce the same neutral human substitution;
- preview user-speaker labels use `User`;
- character-to-Console handoff uses literal `User` and emits no human Persona
  setting;
- the inspector exposes no active-human summary, action, or marker;
- a seeded `character_defaults.active_user_profile` value has no effect.

For the legacy config case, compare the source config mapping and temporary
config-file bytes before and after preview, workbench, and handoff operations.
Install fail-fast spies on config save/repair callbacks. The source value and
bytes must remain identical and no persistence callback may run.

- [ ] **Step 2: Run the red tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_personas_preview.py \
  Tests/UI/test_personas_workbench.py \
  --deselect=Tests/UI/test_personas_workbench.py::TestImportExport::test_import_failure_shows_recovery_copy \
  -q
```

Expected: failures while preview, workbench, and handoff still resolve or
mutate a Persona as the human.

- [ ] **Step 3: Remove the resolver from runtime paths**

Delete `_active_user_name` and its asynchronous equivalent from the preview
controller. Pass `"User"` directly to the existing placeholder replacement
call. Remove active-profile resolution and settings projection from the
character handoff.

Remove the inspector summary/button/marker, screen synchronization,
rename-following pointer writes, and delete-time pointer clearing. Keep
`is_active` as the existing enabled/disabled field.

After every import is unwired, delete
`Character_Chat/active_user_profile.py` and its obsolete test module. Do not
replace it with a dormant compatibility service.

- [ ] **Step 4: Prove the old config value is inert**

Run:

```bash
rg -n "active_user_profile|resolve_active_user_profile|set_active_user_profile|clear_active_user_profile" \
  tldw_chatbook Tests
```

Expected: no production match. Do not edit or migrate any user configuration
file.

- [ ] **Step 5: Run focused tests, import checks, and static checks**

Run the Step 2 tests and:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/Character_Chat \
  tldw_chatbook/UI/Persona_Modules \
  tldw_chatbook/UI/Screens \
  tldw_chatbook/Widgets/Persona_Widgets
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py \
  Tests/UI/test_personas_preview.py
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py \
  tldw_chatbook/Character_Chat/active_user_profile.py \
  Tests/UI/test_personas_preview.py \
  Tests/UI/test_personas_workbench.py \
  Tests/Character_Chat/test_active_user_profile.py
git commit -m "fix(roleplay): keep personas out of the human identity"
```

## Task 3: Correct the Personas workbench mode, events, editor, and actions

**Files:**

- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_state.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py`
- Move: `tldw_chatbook/Widgets/Persona_Widgets/user_profile_editor_widget.py` → `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Modify: `Tests/UI/test_persona_profile_widgets.py`
- Modify: `Tests/UI/test_personas_editor_save_in_place.py`
- Modify: `Tests/UI/test_personas_library_pane.py`
- Modify: `Tests/UI/test_personas_library_pane_paging.py`
- Modify: `Tests/UI/test_personas_library_scale.py`
- Modify: `Tests/UI/test_personas_persona_editor_validation.py`
- Modify: `Tests/UI/test_personas_workbench.py`
- Modify: `Tests/UI/test_personas_workbench_foundation.py`
- Modify: `Tests/UI/test_personas_workbench_state.py`
- Modify: `Tests/UI/test_personas_dictionaries.py`

- [ ] **Step 1: Write failing terminology and action tests**

Update state/message/widget tests to require:

```python
state.switch_mode("personas")
assert state.status_message == "Mode: Personas"
assert selected.entity_kind == "persona"
assert isinstance(message, EditPersonaProfileRequested)
assert isinstance(message, PersonaProfileSaveRequested)
```

Mount the workbench and assert:

- the chip reads `Personas`;
- the description is exactly
  `Personas — assistant profiles for roleplay and chat`;
- the editor title is `Persona Editor`;
- no visible text contains `User Profiles`, `Set as my name`,
  `Clear my name`, or `Chatting as`;
- Persona enabled/disabled controls still work;
- the mode ID and entity kind are `personas` and `persona`;
- the prior saved-mode restore test still falls back to Characters because
  only character mode is currently restored; no migration is added.

- [ ] **Step 2: Write failing source-aware editor tests**

Verify the editor exposes local-only `description` and
`personality_traits` fields in local mode, but hides or disables them with
clear explanatory copy in server mode. A server save must produce a wire DTO
whose dumped keys never contain those fields; a local save must retain them.

- [ ] **Step 3: Run the red workbench tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py \
  Tests/UI/test_personas_workbench.py \
  --deselect=Tests/UI/test_personas_workbench.py::TestImportExport::test_import_failure_shows_recovery_copy \
  -q
```

Expected: failures for the old mode, entity, event, widget, copy, and active
human terminology.

- [ ] **Step 4: Rename the workbench contract**

Use `"personas"` and `"persona"` throughout the mode/entity state. Rename:

```text
UserProfileEditorWidget       -> PersonaProfileEditorWidget
EditUserProfileRequested      -> EditPersonaProfileRequested
UserProfileSaveRequested      -> PersonaProfileSaveRequested
```

Move the editor file; do not leave an import shim. Update all messages,
selectors, handlers, tests, status copy, tooltips, and notifications.

- [ ] **Step 5: Make save payloads source-aware**

Set the editor's runtime source whenever backend mode changes or a record is
loaded. Build the strict local DTOs only in local mode and
`PersonaProfileCreate` / `PersonaProfileUpdate` only in server mode. Explicitly
omit the local-only fields from the server construction before Task 4 makes
the wire models themselves strict. Never silently accept a local-only edit in
server mode.

Add a raw-record editor round trip proving `description` and
`personality_traits` survive list selection, editor load, collection, and
local DTO construction.

- [ ] **Step 6: Run focused tests, import checks, and static checks**

Run the Step 3 command, then:

```bash
rg -n \
  "user_profile_editor_widget|UserProfileEditorWidget|EditUserProfileRequested|UserProfileSaveRequested" \
  tldw_chatbook Tests

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c \
  'from tldw_chatbook.UI.Screens.personas_screen import PersonasScreen; from tldw_chatbook.Widgets.Persona_Widgets.persona_profile_editor_widget import PersonaProfileEditorWidget'

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py \
  Tests/UI/test_personas_workbench.py \
  --deselect=Tests/UI/test_personas_workbench.py::TestImportExport::test_import_failure_shows_recovery_copy \
  --collect-only -q

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py
```

Expected: green, with only the explicitly deselected inherited import/export
failure omitted from the broad workbench file. The old-name search has no
match, the direct imports succeed, and the affected test modules collect.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py
git commit -m "refactor(roleplay): present personas as assistant profiles"
```

## Task 4: Atomically narrow wire DTOs and rename every remaining Persona API

**Files:**

- Modify: `tldw_chatbook/tldw_api/character_persona_schemas.py`
- Modify: `tldw_chatbook/tldw_api/__init__.py`
- Modify: `tldw_chatbook/tldw_api/client.py`
- Modify: `tldw_chatbook/Character_Chat/character_persona_scope_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Modify: `tldw_chatbook/Character_Chat/local_character_persona_service.py`
- Modify: `tldw_chatbook/Character_Chat/persona_list_paging.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py`
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py`
- Modify if the old-name guard finds a remaining mode/pager consumer:
  `tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py`
- Modify: `tldw_chatbook/app.py`
- Modify: `Tests/tldw_api/test_character_persona_schemas.py`
- Modify: `Tests/tldw_api/test_character_persona_client.py`
- Modify: `Tests/Character_Chat/test_character_persona_scope_service.py`
- Modify: `Tests/Character_Chat/test_local_character_persona_service.py`
- Modify: `Tests/Character_Chat/test_persona_list_paging.py`
- Modify: `Tests/Character_Chat/test_persona_personality_traits_roundtrip.py`
- Modify: `Tests/UI/test_ccp_handlers.py`
- Modify: `Tests/UI/test_destination_shells.py`
- Modify: affected `Tests/UI/test_personas_*.py` fakes and assertions

This task performs the hard removals together. Do not commit between removing a
type/method and migrating its final caller.

- [ ] **Step 1: Write failing exact wire-boundary tests**

Replace the old alias assertions with:

```python
assert set(PersonaProfileCreate.model_fields) == SERVER_PERSONA_CREATE_FIELDS
assert set(PersonaProfileUpdate.model_fields) == SERVER_PERSONA_UPDATE_FIELDS
assert set(PersonaProfileResponse.model_fields) == SERVER_PERSONA_RESPONSE_FIELDS
assert not hasattr(character_persona_schemas, "UserProfileCreate")
assert not hasattr(character_persona_schemas, "UserProfileUpdate")
assert not hasattr(character_persona_schemas, "UserProfileResponse")
```

Assert package-level `UserProfileResponse` still resolves to the authenticated
account schema and that the package exports no Persona-domain
`UserProfileCreate` or `UserProfileUpdate`.

All server and local mutation DTOs must raise `ValidationError` for unknown
keys. In particular, passing `description` or `personality_traits` to either
server mutation DTO must fail instead of being silently ignored.

- [ ] **Step 2: Write failing PATCH-body tests**

Capture the API request body and prove:

```python
PersonaProfileUpdate(system_prompt=None)
# -> {"system_prompt": None}

PersonaProfileUpdate(name="Guide")
# -> {"name": "Guide"} and no unrelated defaulted fields
```

- [ ] **Step 3: Write failing exact-dispatch and lossless-local-read tests**

Change fakes and assertions to expose only:

```python
list_persona_profiles
get_persona_profile
create_persona_profile
update_persona_profile
delete_persona_profile
restore_persona_profile
```

Assert the scope service dispatches only those names for both backends and
does not accept a `list_user_profiles`-only backend.

Seed the unchanged `tldw_chatbook_personas.json` with:

```json
{
  "id": "local-persona-1",
  "name": "Archivist",
  "description": "Local description",
  "personality_traits": "patient",
  "future_extension": {"keep": true}
}
```

Verify all of the following:

- constructing the service and calling list/get does not rewrite the file;
- list/get return lossless dictionary views containing local and unknown keys;
- list → editor load → local edit → save preserves `future_extension`;
- an update supplying only `name` preserves every omitted key;
- explicit `description=None` clears only `description`;
- omitted `description` leaves it unchanged;
- restart reloads local fields and the unknown extension;
- the top-level JSON layout and on-disk filename remain unchanged.

- [ ] **Step 4: Run the red schema/service tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/tldw_api/test_character_persona_schemas.py \
  Tests/tldw_api/test_character_persona_client.py \
  Tests/Character_Chat/test_character_persona_scope_service.py \
  Tests/Character_Chat/test_local_character_persona_service.py \
  Tests/Character_Chat/test_persona_list_paging.py \
  Tests/Character_Chat/test_persona_personality_traits_roundtrip.py \
  Tests/UI/test_ccp_handlers.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_workbench.py \
  --deselect=Tests/UI/test_personas_workbench.py::TestImportExport::test_import_failure_shows_recovery_copy \
  -q
```

Expected: failures for widened wire models, alias exports, old method/pager
names, collapsed explicit `None`, and local coercion gaps.

- [ ] **Step 5: Narrow wire DTOs and PATCH serialization**

Make create/update/response fields exact to the server contract. Give both
wire mutation DTOs `ConfigDict(extra="forbid")`. Remove the Persona-domain
alias block and lazy exports; keep genuine account exports untouched.

Change Persona PATCH serialization only to:

```python
request_data.model_dump(exclude_unset=True, mode="json")
```

Do not change unrelated endpoint serialization.

- [ ] **Step 6: Rename the service boundary and every final consumer**

Rename public methods in the scope, server, and local services. Remove fallback
tuples such as `("get_user_profile", "fetch_persona_by_id")`. Rename local
private names (`_user_profiles`, `_load_user_profiles`,
`_persist_user_profiles`, `_find_user_profile`, and the constructor keyword)
to Persona terminology.

Rename `page_user_profiles` to `page_persona_profiles`. Update the CCP handler,
Personas screen, test fakes, and all other call sites in the same change.
Build local DTOs only in local mode and exact wire DTOs only in server mode.
Update user-facing errors from “User profile” to “Persona”.

Preserve only the existing file path:

```python
persona_store_path=get_user_data_dir() / "tldw_chatbook_personas.json"
```

Do not add old-name aliases or method fallbacks.

- [ ] **Step 7: Implement lossless local merges**

Keep list/get as dictionary views. Validate mutations with the strict local
DTOs, then:

```python
changes = request.model_dump(exclude_unset=True, mode="json")
record.update(changes)
```

Update only `last_modified` and `version` afterward. Do not reconstruct the
record, coerce it through `PersonaProfileResponse`, drop unknown keys, rewrite
on read/startup, or use `exclude_none=True`.

- [ ] **Step 8: Run focused tests, a complete import gate, and static checks**

Run the Step 4 command, then:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py -q
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/tldw_api/character_persona_schemas.py \
  tldw_chatbook/tldw_api/__init__.py \
  tldw_chatbook/tldw_api/client.py \
  tldw_chatbook/Character_Chat/character_persona_scope_service.py \
  tldw_chatbook/Character_Chat/server_character_persona_service.py \
  tldw_chatbook/Character_Chat/local_character_persona_service.py \
  tldw_chatbook/Character_Chat/persona_list_paging.py \
  tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/app.py
```

Expected: green and importable with no compatibility aliases.

- [ ] **Step 9: Run the old-name guard before committing**

Run a Persona-domain search scoped away from genuine auth/account modules:

```bash
rg -n \
  "list_user_profiles|get_user_profile|create_user_profile|update_user_profile|delete_user_profile|restore_user_profile|UserProfileEditorWidget|EditUserProfileRequested|UserProfileSaveRequested|page_user_profiles|\"user_profiles\"|entity_kind=\"user_profile\"" \
  tldw_chatbook/Character_Chat \
  tldw_chatbook/UI/CCP_Modules \
  tldw_chatbook/UI/Persona_Modules \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets \
  Tests/Character_Chat \
  Tests/UI/test_ccp_handlers.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_*.py
```

Expected: no Persona-domain match.

- [ ] **Step 10: Commit the coordinated removal**

```bash
git add tldw_chatbook/tldw_api/character_persona_schemas.py \
  tldw_chatbook/tldw_api/__init__.py \
  tldw_chatbook/tldw_api/client.py \
  tldw_chatbook/Character_Chat/character_persona_scope_service.py \
  tldw_chatbook/Character_Chat/server_character_persona_service.py \
  tldw_chatbook/Character_Chat/local_character_persona_service.py \
  tldw_chatbook/Character_Chat/persona_list_paging.py \
  tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py \
  tldw_chatbook/app.py \
  Tests/tldw_api/test_character_persona_schemas.py \
  Tests/tldw_api/test_character_persona_client.py \
  Tests/Character_Chat/test_character_persona_scope_service.py \
  Tests/Character_Chat/test_local_character_persona_service.py \
  Tests/Character_Chat/test_persona_list_paging.py \
  Tests/Character_Chat/test_persona_personality_traits_roundtrip.py \
  Tests/UI/test_ccp_handlers.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_workbench.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py

git status --short
git diff --name-only
git commit -m "refactor(personas): use exact persona contracts end to end"
```

Before committing, stage every task-owned call site reported by Step 9,
including any affected `Tests/UI/test_personas_*.py` file. `git diff
--name-only` must print nothing, proving no task change was left unstaged.
Do not stage unrelated pre-existing worktree changes.

## Task 5: Remove Persona-as-human Console settings and presentation

**Files:**

- Modify: `tldw_chatbook/Chat/console_display_state.py`
- Modify: `tldw_chatbook/Chat/console_session_settings.py`
- Modify: `tldw_chatbook/Widgets/Console/console_workbench_state.py`
- Modify: `tldw_chatbook/Widgets/Console/console_status_chips.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_display_state.py`
- Modify: `Tests/Chat/test_console_session_settings.py`
- Modify: `Tests/Chat/test_console_run_status_surfaces.py`
- Modify: `Tests/UI/test_chat_shell_bar.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_console_status_chips.py`
- Modify: `Tests/UI/test_console_workbench_contract.py`

- [ ] **Step 1: Write failing pure-state and serialization tests**

Require a single assistant identity label:

```python
ConsoleControlState.from_values(character=None).assistant_label
# "Assistant: General"

ConsoleControlState.from_values(character="Ada").assistant_label
# "Character: Ada"

ConsoleControlState.from_values(
    assistant_kind="persona",
    assistant_name="Guide",
    assistant_id="persona-7",
).assistant_label
# "Persona: Guide"

ConsoleControlState.from_values(
    assistant_kind="persona",
    assistant_id="persona-7",
).assistant_label
# "Persona: persona-7"
```

Remove the human-meaning `persona` input and `user_profile_label` output. The
optional assistant-kind/name/ID inputs are presentation-only values already
available on a session-like caller; they do not add fields to
`ConsoleChatSession`, persistence, or conversation schemas.

For Console settings:

```python
source = {
    "provider": "llama_cpp",
    "persona_label": "Legacy A",
    "user_profile_label": "Legacy B",
}
before = deepcopy(source)
restored = ChatScreen._restore_console_settings(source)
assert restored is not None
assert source == before
assert not hasattr(restored, "user_profile_label")
assert "persona_label" not in ChatScreen._serialize_console_settings(restored)
assert "user_profile_label" not in ChatScreen._serialize_console_settings(restored)
```

Assert summary identity is `Assistant: General` without a character and
`Character: Ada` with one. Seed a serialized legacy settings blob, run restore,
and assert its source mapping and encoded bytes remain unchanged and no
persistence/save callback runs.

- [ ] **Step 2: Write failing widget/shell tests**

Assert:

- Console status chips and modes contain one assistant identity item and no
  Persona-as-human item;
- an already-identified Persona session renders `Persona: <name-or-ID>` in
  the status chip and control summary as well as the shell;
- no widget ID `console-persona-chip` or `console-settings-persona-readonly`
  exists;
- settings modal contains no **User Profile** row;
- generic shell copy is `Assistant: General`;
- character shell copy is `Character: <name>`;
- Persona session shell copy is `Persona: <assistant_name-or-assistant_id>`;
- no current identity label begins with `As:`.

Replace `Tests/UI/test_chat_shell_bar.py`'s obsolete `TabState` import with a
small local dataclass or `SimpleNamespace`; do not reintroduce `TabState`.

- [ ] **Step 3: Run the red Console tests**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Chat/test_console_display_state.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/Chat/test_console_run_status_surfaces.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/UI/test_console_status_chips.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_workbench_contract.py \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_remote_defaults_use_smoke_verified_models \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_model_resolution_includes_runtime_discovered_models \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_missing_key_recovery_action_is_provider_specific \
  --deselect=Tests/UI/test_console_session_settings.py::test_real_journey_settings_save_unblocks_console_without_restart \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_resolution_view_suppresses_boot_echo_reactives \
  --deselect=Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_choose_model_opens_settings \
  -q
```

Expected: failures for the old dataclass field, chips, settings row, and `As:`
copy.

- [ ] **Step 4: Simplify live Console identity state**

Replace the control state's parallel `character_label` and
`user_profile_label` presentation with one `assistant_label`. Render
`Persona: <assistant_name-or-assistant_id>` when an existing caller supplies
`assistant_kind="persona"`, `Character: <name>` for a character, and
`Assistant: General` otherwise. Remove the Persona-as-human mode/chip/label
from the control bar, status strip, and workbench state; the remaining single
assistant identity item is not a user profile selector.

`ChatScreen._build_console_control_state` may read these three optional values
with `getattr` from its already-active session-like object. It must not add
them to `ConsoleChatSession`, serialize them, synthesize them from a Persona
setting, or otherwise pull 3A.2 identity persistence into this task.

Keep `ConsoleSessionSettings.character_label`; remove
`ConsoleSessionSettings.user_profile_label` and every replacement/copy site.

- [ ] **Step 5: Handle compatibility only on restore**

Before filtering restored settings:

```python
values = dict(payload)
values.pop("persona_label", None)
values.pop("user_profile_label", None)
```

Do not map either value into a new field. Because serialization is `asdict` of
the narrowed dataclass, new snapshots emit neither key. Do not scan, rewrite,
or delete stored session records or transcripts.

- [ ] **Step 6: Correct settings and shell copy**

Remove the read-only User Profile row and helper. Make the identity summary
`Assistant: General` or `Character: <name>`. In `ChatShellContext`, derive
Persona display from the existing session's `assistant_name` when present,
otherwise `assistant_id`, and render `Persona: ...`. Remove
`ChatShellLabelResolver.user_profile_label`.

- [ ] **Step 7: Run focused tests and static checks**

Run the Step 3 command, then:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook/Chat \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Chat_Widgets \
  tldw_chatbook/UI/Screens/chat_screen.py
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/Chat/console_session_settings.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_display_state.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/Chat/test_console_run_status_surfaces.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/UI/test_console_status_chips.py
```

Expected: green, with only the exact inherited tests deselected.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/Chat/console_session_settings.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  Tests/Chat/test_console_display_state.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/Chat/test_console_run_status_surfaces.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_status_chips.py \
  Tests/UI/test_console_workbench_contract.py
git commit -m "refactor(console): remove persona as human identity"
```

## Task 6: Verify the complete semantic boundary and close task documentation

**Files:**

- Modify: `backlog/tasks/task-617.1 - Establish-the-Roleplay-Persona-and-User-Profile-semantic-boundary.md`

Treat the approved design and ADR-037 as read-only during implementation. If
code reveals a genuine contract or decision correction, stop and request
explicit design reapproval instead of folding it into TASK-617.1.

- [ ] **Step 1: Run the complete focused gate**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/tldw_api/test_character_persona_schemas.py \
  Tests/tldw_api/test_character_persona_client.py \
  Tests/Character_Chat/test_character_persona_scope_service.py \
  Tests/Character_Chat/test_local_character_persona_service.py \
  Tests/Character_Chat/test_persona_list_paging.py \
  Tests/Character_Chat/test_persona_personality_traits_roundtrip.py \
  Tests/UI/test_ccp_handlers.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_editor_save_in_place.py \
  Tests/UI/test_personas_library_pane.py \
  Tests/UI/test_personas_library_pane_paging.py \
  Tests/UI/test_personas_library_scale.py \
  Tests/UI/test_personas_persona_editor_validation.py \
  Tests/UI/test_personas_preview.py \
  Tests/UI/test_personas_workbench_foundation.py \
  Tests/UI/test_personas_workbench_state.py \
  Tests/UI/test_personas_dictionaries.py \
  Tests/UI/test_personas_workbench.py \
  Tests/Chat/test_console_display_state.py \
  Tests/Chat/test_console_session_settings.py \
  Tests/Chat/test_console_run_status_surfaces.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/UI/test_console_status_chips.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/TTS/test_console_speak_autoplay.py \
  --deselect=Tests/UI/test_personas_workbench.py::TestImportExport::test_import_failure_shows_recovery_copy \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_remote_defaults_use_smoke_verified_models \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_model_resolution_includes_runtime_discovered_models \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_missing_key_recovery_action_is_provider_specific \
  --deselect=Tests/UI/test_console_session_settings.py::test_real_journey_settings_save_unblocks_console_without_restart \
  --deselect=Tests/UI/test_console_session_settings.py::test_console_resolution_view_suppresses_boot_echo_reactives \
  --deselect=Tests/UI/test_console_workbench_contract.py::test_console_empty_transcript_choose_model_opens_settings \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Replay the pre-change module baseline without deselection**

Re-run Step 1's complete module list after removing every `--deselect`
argument.

Expected:

- the shell test module collects and passes;
- the same seven inherited test failures remain;
- every other collected test passes;
- there is no new failure or collection error.

If an inherited failure disappears because an assertion was directly updated
by this task, record why. Any new failure must be fixed or reproduced on a
clean `origin/dev` worktree before it may be classified as baseline.

- [ ] **Step 3: Run TTS/profile/application regression suites**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/TTS/test_profile_repository.py \
  Tests/TTS/test_profile_service.py \
  Tests/TTS/test_tts_app_ownership.py \
  Tests/TTS/test_stts_audio_cpp_generation.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/UI/test_stts_playground_audio_cpp.py -q
```

Expected: no regression to global Console speech, complete-WAV audio.cpp
generation, the profile library, or app-owned service lifecycles.

- [ ] **Step 4: Run the broad repository suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q
```

For every failure, compare the exact node ID and failure shape against a clean
`origin/dev` worktree on the same Python environment. The task branch may have
no new failure. Record the full pass/fail/skip counts and the exact inherited
set in Implementation Notes; do not fix unrelated failures in this task.

- [ ] **Step 5: Run terminology guards**

Run:

```bash
rg -n \
  "list_user_profiles|get_user_profile|create_user_profile|update_user_profile|delete_user_profile|restore_user_profile|UserProfileEditorWidget|EditUserProfileRequested|UserProfileSaveRequested|active_user_profile|Set as my name|Clear my name|Chatting as|You: default|As: General" \
  tldw_chatbook/Character_Chat \
  tldw_chatbook/UI/CCP_Modules \
  tldw_chatbook/UI/Persona_Modules \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/Chat/console_session_settings.py \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py \
  Tests/Character_Chat \
  Tests/Chat \
  Tests/UI/test_ccp_handlers.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_*.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_status_chips.py \
  Tests/UI/test_console_workbench_contract.py

rg -n '"user_profiles"|entity_kind="user_profile"|user_profile_label|persona_label' \
  tldw_chatbook/Character_Chat \
  tldw_chatbook/UI/CCP_Modules \
  tldw_chatbook/UI/Persona_Modules \
  tldw_chatbook/UI/Screens/personas_screen.py \
  tldw_chatbook/Widgets/Persona_Widgets \
  tldw_chatbook/Chat \
  tldw_chatbook/Widgets/Console \
  tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py \
  Tests/Character_Chat \
  Tests/Chat \
  Tests/UI/test_ccp_handlers.py \
  Tests/UI/test_destination_shells.py \
  Tests/UI/test_persona_profile_widgets.py \
  Tests/UI/test_personas_*.py \
  Tests/UI/test_chat_shell_bar.py \
  Tests/UI/test_console_session_settings.py \
  Tests/UI/test_console_status_chips.py \
  Tests/UI/test_console_workbench_contract.py
```

Expected:

- no Persona-domain runtime/test match;
- legacy string keys appear only in the explicit Console restore-ignore test
  and restore `pop` boundary;
- genuine account `UserProfile*` symbols remain outside these Persona scopes.

- [ ] **Step 6: Run an executable changed-file allowlist and deferred-scope guard**

Run:

```bash
git diff --name-only origin/dev | \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import sys
exact = {
    "Docs/superpowers/plans/2026-07-28-persona-user-profile-semantic-boundary.md",
    "Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md",
    "backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md",
    "backlog/tasks/task-617.1 - Establish-the-Roleplay-Persona-and-User-Profile-semantic-boundary.md",
    "tldw_chatbook/tldw_api/character_persona_schemas.py",
    "tldw_chatbook/tldw_api/__init__.py",
    "tldw_chatbook/tldw_api/client.py",
    "tldw_chatbook/Character_Chat/character_persona_scope_service.py",
    "tldw_chatbook/Character_Chat/server_character_persona_service.py",
    "tldw_chatbook/Character_Chat/local_character_persona_service.py",
    "tldw_chatbook/Character_Chat/persona_list_paging.py",
    "tldw_chatbook/Character_Chat/active_user_profile.py",
    "tldw_chatbook/UI/CCP_Modules/ccp_persona_handler.py",
    "tldw_chatbook/UI/Persona_Modules/personas_preview_controller.py",
    "tldw_chatbook/UI/Screens/personas_screen.py",
    "tldw_chatbook/UI/Screens/chat_screen.py",
    "tldw_chatbook/Widgets/Persona_Widgets/personas_messages.py",
    "tldw_chatbook/Widgets/Persona_Widgets/personas_state.py",
    "tldw_chatbook/Widgets/Persona_Widgets/personas_pane_messages.py",
    "tldw_chatbook/Widgets/Persona_Widgets/persona_profile_editor_widget.py",
    "tldw_chatbook/Widgets/Persona_Widgets/user_profile_editor_widget.py",
    "tldw_chatbook/Widgets/Persona_Widgets/persona_profile_card_widget.py",
    "tldw_chatbook/Widgets/Persona_Widgets/personas_inspector_pane.py",
    "tldw_chatbook/Widgets/Persona_Widgets/personas_library_pane.py",
    "tldw_chatbook/Chat/console_display_state.py",
    "tldw_chatbook/Chat/console_session_settings.py",
    "tldw_chatbook/Widgets/Console/console_workbench_state.py",
    "tldw_chatbook/Widgets/Console/console_status_chips.py",
    "tldw_chatbook/Widgets/Console/console_control_bar.py",
    "tldw_chatbook/Widgets/Console/console_settings_modal.py",
    "tldw_chatbook/Widgets/Chat_Widgets/chat_shell_bar.py",
    "tldw_chatbook/app.py",
    "Tests/tldw_api/test_character_persona_schemas.py",
    "Tests/tldw_api/test_character_persona_client.py",
    "Tests/Character_Chat/test_active_user_profile.py",
    "Tests/Character_Chat/test_character_persona_scope_service.py",
    "Tests/Character_Chat/test_local_character_persona_service.py",
    "Tests/Character_Chat/test_persona_list_paging.py",
    "Tests/Character_Chat/test_persona_personality_traits_roundtrip.py",
    "Tests/Chat/test_console_display_state.py",
    "Tests/Chat/test_console_session_settings.py",
    "Tests/Chat/test_console_run_status_surfaces.py",
    "Tests/UI/test_ccp_handlers.py",
    "Tests/UI/test_destination_shells.py",
    "Tests/UI/test_chat_shell_bar.py",
    "Tests/UI/test_console_session_settings.py",
    "Tests/UI/test_console_status_chips.py",
    "Tests/UI/test_console_workbench_contract.py",
    "Tests/UI/test_persona_profile_widgets.py",
}
prefixes = ("Tests/UI/test_personas_",)
paths = [line.strip() for line in sys.stdin if line.strip()]
bad = [path for path in paths if path not in exact and not path.startswith(prefixes)]
print("\\n".join(bad))
raise SystemExit(bool(bad))'

git diff -U0 origin/dev -- tldw_chatbook | \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c 'import re, sys
pattern = re.compile(r"^\\+.*(assistant_authority_id|authority_scope_id|speech_revision|speech_snapshot|TTSMessageSpeechSnapshot|SpeechSnapshot|CharacterRef|assignment_revision|set_tts_assignment|detach_tts_assignment|audiocpp_server|managed audio\\.cpp|\\{\\{persona\\}\\}|\\{\\{character\\}\\}|\\{\\{char\\}\\})", re.IGNORECASE)
bad = [line.rstrip() for line in sys.stdin if pattern.search(line)]
print("\\n".join(bad))
raise SystemExit(bool(bad))'
```

Expected: the allowlist script exits zero with no output, and the deferred-scope
search has no match. These comparisons use `origin/dev` against the current
index and working tree, not `origin/dev...HEAD`, so staged or unstaged review
amendments are included. The two previously approved design/ADR files are
allowed because they are already committed on this planning branch;
implementation does not edit them.

- [ ] **Step 7: Run formatting, lint, typing, compile, and diff checks**

Run format and Ruff over every added/copied/modified Python file rather than a
hand-maintained subset:

```bash
git diff --name-only --diff-filter=ACMR origin/dev -- '*.py' | \
  xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff format --check

git diff --name-only --diff-filter=ACMR origin/dev -- '*.py' | \
  xargs /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m ruff check

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall -q \
  tldw_chatbook

/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m mypy \
  tldw_chatbook/tldw_api/character_persona_schemas.py \
  tldw_chatbook/Character_Chat/character_persona_scope_service.py \
  tldw_chatbook/Character_Chat/server_character_persona_service.py \
  tldw_chatbook/Character_Chat/local_character_persona_service.py \
  tldw_chatbook/Chat/console_display_state.py \
  tldw_chatbook/Chat/console_session_settings.py

git diff --check
git status --short
```

If broad pre-existing Ruff or mypy findings remain, record the exact baseline
and prove no task-introduced finding rather than expanding scope.

- [ ] **Step 8: Review the diff against every acceptance criterion**

Confirm:

- server and local Persona contracts are distinct;
- no callable Persona surface uses User Profile terminology;
- local unknown fields and explicit clears are preserved;
- no active-human Persona behavior remains;
- `{{user}}` remains literal `User`;
- Console emits no legacy human-Persona setting;
- global TTS and complete-WAV behavior are unchanged;
- no work from 3A.2, 3A.3, 3A.4, Slice 3B, or managed audio.cpp entered the diff.

- [ ] **Step 9: Request independent code and scope review**

Use `superpowers:requesting-code-review` against the full Slice 3A.1 diff.
Address every verified Critical, Important, and Minor issue. Re-run affected
tests after each amendment. Stage only the task-owned review-amendment files so
new files as well as tracked edits enter the `git diff origin/dev` checks.
Then repeat **Steps 1 through 8 in full**, including the baseline replay,
TTS/profile/application regressions, and broad repository suite.

- [ ] **Step 10: Commit verified review amendments**

If review changed code or tests, inspect the staged diff and commit only after
the repeated Steps 1–8 are green:

```bash
git status --short
git diff --cached --check
git commit -m "fix(personas): address semantic boundary review"
```

Skip this commit when review required no amendment.

- [ ] **Step 11: Update Backlog evidence and mark Done last**

Check every acceptance criterion only after its evidence is green. Add concise
Implementation Notes with:

- source-specific DTO and merge behavior;
- exact renamed surfaces and deleted compatibility code;
- workbench and Console presentation changes;
- legacy compatibility treatment;
- tests and inherited baseline failures;
- ADR-037 applicability and confirmation that no new ADR was needed.

Check all acceptance criteria and Definition-of-Done items, then make the final
state transition:

```bash
backlog task edit 617.1 -s Done --plain
```

Do not mark Done while any criterion, documentation, test, static check, or
review item remains unresolved.

- [ ] **Step 12: Commit final task documentation**

```bash
git add "backlog/tasks/task-617.1 - Establish-the-Roleplay-Persona-and-User-Profile-semantic-boundary.md"
git commit -m "docs(personas): close semantic boundary task"
```

Do not stage the approved design or ADR; this task treats both as read-only.
