# TASK-14801 Roleplay Chat Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give native Console character chats named, theme-aware participants with a display-only global human name, a durable per-chat override, and safe dynamic expansion of trusted character templates.

**Architecture:** Add a pure Console roleplay identity layer that validates display names, expands trusted templates once, parses the task-owned metadata object, and resolves one shared message presentation. Keep protocol roles and ordinary stored content unchanged. Store safe resolved projections in existing conversation/message fields while retaining explicit local provenance, then thread the shared presentation through transcript, exports, speech, previews, and provider context.

**Tech Stack:** Python 3.11+, Textual 8.x, Rich `Content` and `cell_len`, SQLite conversation/message metadata through existing services, pytest, Ruff, generated TCSS bundle.

## Global Constraints

- Run only tests related to files touched and behavior reachable from those files.
- Effective human name precedence is per-chat override, then `[chat_defaults].user_display_name`, then `User`.
- `[general].users_name` remains storage-profile identity and must not be read, written, or migrated by this feature.
- Display names are trimmed, Unicode-capable, control-character-free, and no wider than 48 terminal cells.
- Blank global values resolve to `User`; blank per-chat values clear the override and inherit.
- Character-template expansion is case-sensitive, single-pass, and limited to explicitly trusted template provenance.
- `{{user}}`, `{{random_user}}`, and `<USER>` resolve to the effective human name.
- `{{char}}`, `{{character}}`, `{{persona}}`, and `<CHAR>` resolve to the loaded character name.
- Replacement values are inserted once and are never scanned for more macros.
- Generic and Persona transcript assistant rows remain labeled `Assistant`.
- Character-only row tinting is supplemental to literal speaker labels and never overrides selected, system, tool, streaming, or failure state priority.
- Persist safe projections in ordinary fields; never make raw template source the canonical synced message content.
- Do not add a database migration or change Sync v2 payload contracts.
- New user-facing copy must not use em dashes.
- Edit actions receive the visible resolved projection and clear template provenance on save.
- The generated `tldw_chatbook/css/tldw_cli_modular.tcss` file is rebuilt from `tldw_chatbook/css/components/_agentic_terminal.tcss`, never edited by hand.

## ADR check

ADR required: yes

ADR path: `backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md`

Reason: this feature adds persisted human display identity ownership, source/projection provenance, optimistic metadata merge behavior, and a cross-module presentation contract that extends ADR-037.

## File structure

### New production files

- `tldw_chatbook/Chat/console_roleplay_identity.py`: pure validation, effective-name resolution, single-pass macro expansion, presentation context, and message presentation resolver.
- `tldw_chatbook/Chat/console_roleplay_metadata.py`: guarded parse/serialize/merge helpers for the versioned `console_roleplay_context` conversation metadata object.

### New test files

- `Tests/Chat/test_console_roleplay_identity.py`: pure name, macro, projection, and presentation contracts.
- `Tests/Chat/test_console_roleplay_metadata.py`: versioned metadata parsing, sibling preservation, clearing, and future-version degradation.

### Existing production files to modify

- `tldw_chatbook/config.py`: add the display-only default config key and getter.
- `tldw_chatbook/Chat/message_metadata.py`: add closed seeded-greeting template provenance.
- `tldw_chatbook/Chat/console_chat_store.py`: own live roleplay context, safe projection materialization, first-persist flush, edit-clears-provenance, and speech snapshots.
- `tldw_chatbook/Chat/chat_persistence_service.py`: bounded optimistic merge for task-owned conversation metadata.
- `tldw_chatbook/Chat/console_chat_controller.py`: use live trusted projections for provider and preview context.
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`: expose the existing card-field join as a raw template composer so Console does not duplicate field order or labels.
- `tldw_chatbook/UI/Console_Modules/session.py`: seed raw sources plus safe projections, restore roleplay context, serialize screen state, and apply identity changes.
- `tldw_chatbook/UI/Console_Modules/workspace.py`: restore roleplay metadata with resumed conversations.
- `tldw_chatbook/UI/Console_Modules/message.py`: use shared presentation for actions, exports, and edit entry.
- `tldw_chatbook/UI/Screens/chat_screen.py`: pass presentation context to transcript and character-picker flows, and include identity revision in refresh fingerprints.
- `tldw_chatbook/Widgets/Console/console_settings_modal.py`: add the per-chat name field and a result type that keeps it separate from provider settings.
- `tldw_chatbook/Widgets/Console/console_transcript.py`: render plain/Markdown rows and plain export from the shared presentation.
- `tldw_chatbook/UI/Screens/settings_screen.py`: add the canonical global display-name setting to staged save/revert.
- `tldw_chatbook/css/components/_agentic_terminal.tcss`: add semantic character-roleplay row and label rules.
- `tldw_chatbook/css/tldw_cli_modular.tcss`: regenerated output.

### Existing focused tests to extend

- `Tests/Chat/test_message_metadata.py`
- `Tests/Chat/test_console_chat_store.py`
- `Tests/Chat/test_console_chat_controller.py`
- `Tests/Chat/test_console_speech_snapshots.py`
- `Tests/Chat/test_console_save_targets.py`
- `Tests/Character_Chat/test_compose_character_card_text.py`
- `Tests/test_config_console_defaults.py`
- `Tests/UI/test_character_session_prompt_seed.py`
- `Tests/UI/test_console_session_settings.py`
- `Tests/UI/test_console_native_transcript.py`
- `Tests/UI/test_console_transcript_markdown.py`
- `Tests/UI/test_console_transcript_markdown_widget.py`
- `Tests/UI/test_console_message_controller.py`
- `Tests/UI/test_settings_configuration_hub.py`
- `Tests/UI/test_settings_save_commit_models.py`

---

### Task 1: Establish the pure identity, template, and presentation contracts

**Files:**
- Create: `tldw_chatbook/Chat/console_roleplay_identity.py`
- Create: `Tests/Chat/test_console_roleplay_identity.py`

**Interfaces:**
- Produces: `ChatDisplayNameError`, `normalize_chat_display_name(value, *, blank_means_none)`, `effective_user_display_name(override, global_default)`, `expand_character_template(source, *, user_name, character_name)`, `ConsolePresentationContext`, `ConsoleMessagePresentation`, and `resolve_console_message_presentation(message, context)`.
- Consumes: `ConsoleChatMessage`, `ConsoleMessageRole`, `MessageMetadata`, and Rich `cell_len`.

- [x] **Step 1: Run the pure related baseline tests**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Character_Chat/test_placeholder_aliases.py Tests/Chat/test_message_metadata.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_markdown.py -q
```

Expected: PASS, or record the exact existing failures before changing code and compare the same command after the task.

- [x] **Step 2: Write failing validation and precedence tests**

Add tests with these exact behavioral cases:

```python
def test_effective_name_prefers_override_then_global_then_user():
    assert effective_user_display_name("Captain Rowan", "Default") == "Captain Rowan"
    assert effective_user_display_name(None, "Default") == "Default"
    assert effective_user_display_name(None, "   ") == "User"


def test_name_validation_uses_terminal_cells_and_rejects_controls():
    assert normalize_chat_display_name("  海の人  ", blank_means_none=False) == "海の人"
    assert normalize_chat_display_name("👩‍🚀 Rowan", blank_means_none=False) == "👩‍🚀 Rowan"
    with pytest.raises(ChatDisplayNameError, match="48 terminal cells"):
        normalize_chat_display_name("界" * 25, blank_means_none=False)
    with pytest.raises(ChatDisplayNameError, match="control"):
        normalize_chat_display_name("Rowan\nAdmin", blank_means_none=False)
    with pytest.raises(ChatDisplayNameError, match="control"):
        normalize_chat_display_name("Rowan\u202eAdmin", blank_means_none=False)


def test_blank_override_clears_while_blank_global_falls_back():
    assert normalize_chat_display_name("  ", blank_means_none=True) is None
    assert normalize_chat_display_name("  ", blank_means_none=False) == "User"
```

- [x] **Step 3: Run the new tests and verify RED**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_roleplay_identity.py -q
```

Expected: collection fails because `console_roleplay_identity` does not exist.

- [x] **Step 4: Implement exact validation and effective-name resolution**

Use the following public shape:

```python
from rich.cells import cell_len

CHAT_DISPLAY_NAME_MAX_CELLS = 48


class ChatDisplayNameError(ValueError):
    """A human chat display name is unsafe or too wide."""


def normalize_chat_display_name(value: object, *, blank_means_none: bool) -> str | None:
    if value is None:
        text = ""
    elif not isinstance(value, str):
        raise ChatDisplayNameError("Display name must be text.")
    else:
        text = value.strip()
    if not text:
        return None if blank_means_none else "User"
    if any(
        unicodedata.category(char) in {"Cc", "Cs"}
        or char in {"\u2028", "\u2029"}
        or (unicodedata.category(char) == "Cf" and char not in {"\u200c", "\u200d"})
        for char in text
    ):
        raise ChatDisplayNameError("Display name cannot contain control characters.")
    if cell_len(text) > CHAT_DISPLAY_NAME_MAX_CELLS:
        raise ChatDisplayNameError("Display name must fit within 48 terminal cells.")
    return text


def effective_user_display_name(override: object, global_default: object) -> str:
    local = normalize_chat_display_name(override, blank_means_none=True)
    if local is not None:
        return local
    return normalize_chat_display_name(global_default, blank_means_none=False) or "User"
```

Reject control characters, surrogates, line/paragraph separators, and format controls other than the zero-width non-joiner/joiner needed by ordinary Unicode shaping and emoji sequences. Keep markup characters such as `[` and `]` literal and valid. Reject non-string values instead of stringifying corrupt config or metadata shapes.

- [x] **Step 5: Write failing single-pass macro tests**

```python
@pytest.mark.parametrize("token", ["{{user}}", "{{random_user}}", "<USER>"])
def test_user_aliases_expand_only_once(token):
    result = expand_character_template(
        f"Hello {token}",
        user_name="Archivist {{character}}",
        character_name="Alraune",
    )
    assert result == "Hello Archivist {{character}}"


def test_character_aliases_share_the_loaded_name():
    source = "{{char}}/{{character}}/{{persona}}/<CHAR> greets {{user}}"
    assert expand_character_template(
        source, user_name="Rowan", character_name="Alraune"
    ) == "Alraune/Alraune/Alraune/Alraune greets Rowan"


def test_case_and_unknown_tokens_stay_literal():
    assert expand_character_template(
        "{{User}} {{unknown}}", user_name="Rowan", character_name="Alraune"
    ) == "{{User}} {{unknown}}"
```

- [x] **Step 6: Implement one regex substitution, not sequential replacement**

Use one compiled alternation and one `re.sub` callback:

```python
_TEMPLATE_TOKEN_RE = re.compile(
    r"\{\{user\}\}|\{\{random_user\}\}|<USER>|"
    r"\{\{char\}\}|\{\{character\}\}|\{\{persona\}\}|<CHAR>"
)
_USER_TOKENS = frozenset({"{{user}}", "{{random_user}}", "<USER>"})


def expand_character_template(source: str, *, user_name: str, character_name: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        return user_name if match.group(0) in _USER_TOKENS else character_name

    return _TEMPLATE_TOKEN_RE.sub(replacement, source)
```

- [x] **Step 7: Write failing presentation tests**

Cover these exact cases:

```python
def test_character_rows_use_named_speakers_and_roleplay_classes():
    context = ConsolePresentationContext(
        user_name="Captain [Rowan]",
        assistant_kind="character",
        character_name="Alraune",
        revision=7,
    )
    user = resolve_console_message_presentation(user_message("Hi"), context)
    assistant = resolve_console_message_presentation(assistant_message("Hello"), context)
    assert (user.speaker_label, user.row_class) == (
        "Captain [Rowan]",
        "console-transcript-message-roleplay-user",
    )
    assert (assistant.speaker_label, assistant.row_class) == (
        "Alraune",
        "console-transcript-message-roleplay-character",
    )
    assert user.revision_token[-1] == 7


@pytest.mark.parametrize("assistant_kind", ["generic", "persona", None])
def test_non_character_assistant_label_stays_assistant(assistant_kind):
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind=assistant_kind, character_name="Ada"
    )
    presentation = resolve_console_message_presentation(
        assistant_message("Hello"), context
    )
    assert presentation.speaker_label == "Assistant"
    assert presentation.row_class is None


def test_generic_user_rows_use_custom_name_without_roleplay_tint():
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="generic", character_name=None
    )
    presentation = resolve_console_message_presentation(user_message("Hi"), context)
    assert presentation.speaker_label == "Rowan"
    assert presentation.row_class is None


def test_character_session_without_name_falls_back_to_neutral_assistant():
    context = ConsolePresentationContext(
        user_name="Rowan", assistant_kind="character", character_name="  "
    )
    presentation = resolve_console_message_presentation(
        assistant_message("Hello"), context
    )
    assert presentation.speaker_label == "Assistant"
    assert presentation.row_class is None
```

Also assert seeded greeting metadata is expanded only when `template_kind == "character_greeting"`, the session is a named character session, and `template_source` is nonblank. Ordinary content containing `{{user}}` must remain unchanged.

- [x] **Step 8: Implement the immutable presentation types and resolver**

Use these types and keep Rich/Textual markup parsing outside the resolver:

```python
@dataclass(frozen=True, slots=True)
class ConsolePresentationContext:
    user_name: str = "User"
    assistant_kind: str | None = "generic"
    character_name: str | None = None
    revision: int = 0


@dataclass(frozen=True, slots=True)
class ConsoleMessagePresentation:
    speaker_label: str
    content: str
    row_class: str | None
    revision_token: tuple[object, ...]
```

The resolver selects the current variant content, applies greeting provenance only under the trusted gate, leaves system/tool content unchanged, and includes label, resolved content, row class, and context revision in `revision_token`.

- [x] **Step 9: Run Task 1 tests and mutation checks**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_roleplay_identity.py Tests/Character_Chat/test_placeholder_aliases.py -q
```

Temporarily replace the regex callback with sequential `.replace()` calls and verify `test_user_aliases_expand_only_once` fails. Restore the correct code and rerun green.

- [x] **Step 10: Commit Task 1**

```powershell
git add tldw_chatbook/Chat/console_roleplay_identity.py Tests/Chat/test_console_roleplay_identity.py
git commit -m "feat: add roleplay chat identity resolver"
```

### Task 2: Add versioned metadata and seeded-greeting provenance

**Files:**
- Create: `tldw_chatbook/Chat/console_roleplay_metadata.py`
- Create: `Tests/Chat/test_console_roleplay_metadata.py`
- Modify: `tldw_chatbook/Chat/message_metadata.py`
- Modify: `Tests/Chat/test_message_metadata.py`

**Interfaces:**
- Consumes: `normalize_chat_display_name` from Task 1.
- Produces: `ROLEPLAY_CONTEXT_METADATA_KEY`, `RoleplayContextVersionError`, `ConsoleRoleplayContext`, `parse_console_roleplay_context(raw_metadata)`, and `merge_console_roleplay_context(raw_metadata, context)`.
- Extends: `MessageMetadata.template_kind` and `MessageMetadata.template_source`.

- [x] **Step 1: Write failing conversation metadata tests**

```python
def test_parse_version_one_context():
    raw = json.dumps({
        "console_roleplay_context": {
            "version": 1,
            "user_name_override": "Captain Rowan",
            "character_system_template": "Speak with {{user}}.",
        }
    })
    assert parse_console_roleplay_context(raw) == ConsoleRoleplayContext(
        user_name_override="Captain Rowan",
        character_system_template="Speak with {{user}}.",
    )


@pytest.mark.parametrize("payload", [None, "not json", "[]"])
def test_invalid_metadata_degrades_to_empty_context(payload):
    assert parse_console_roleplay_context(payload) == ConsoleRoleplayContext()


def test_future_version_degrades_without_guessing():
    raw = json.dumps({
        "console_roleplay_context": {
            "version": 2,
            "user_name_override": "Do not trust this build",
        }
    })
    assert parse_console_roleplay_context(raw) == ConsoleRoleplayContext()


def test_write_refuses_to_clobber_future_owned_version():
    raw = json.dumps({
        "sibling": {"kept": True},
        "console_roleplay_context": {"version": 2, "future_field": "keep"},
    })
    with pytest.raises(RoleplayContextVersionError, match="version 2"):
        merge_console_roleplay_context(
            raw, ConsoleRoleplayContext(user_name_override="Rowan")
        )


def test_merge_preserves_siblings_and_removes_empty_owned_object():
    raw = json.dumps({"active_dictionaries": [4], "pinned_response_prefill": "Yes"})
    merged = json.loads(merge_console_roleplay_context(
        raw, ConsoleRoleplayContext(user_name_override="Rowan")
    ))
    assert merged["active_dictionaries"] == [4]
    assert merged["pinned_response_prefill"] == "Yes"
    cleared = json.loads(merge_console_roleplay_context(
        json.dumps(merged), ConsoleRoleplayContext()
    ))
    assert "console_roleplay_context" not in cleared
```

- [x] **Step 2: Run metadata tests and verify RED**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_roleplay_metadata.py -q
```

Expected: import failure for the new module.

- [x] **Step 3: Implement guarded parse and sibling-preserving merge**

Use this closed data shape:

```python
ROLEPLAY_CONTEXT_METADATA_KEY = "console_roleplay_context"
ROLEPLAY_CONTEXT_VERSION = 1


@dataclass(frozen=True, slots=True)
class ConsoleRoleplayContext:
    user_name_override: str | None = None
    character_system_template: str | None = None
```

`parse_console_roleplay_context` must accept a raw JSON string, mapping, or missing value; trust only an object with exact integer version `1`; validate the override; accept only nonblank string system sources; and return an empty context on invalid or future data. `merge_console_roleplay_context` must parse the outer metadata as an object, preserve every sibling, write only version and nonblank owned fields, and remove the owned object when both fields are absent. If the existing owned object has an integer version greater than `1`, raise `RoleplayContextVersionError` instead of overwriting unknown future fields. Restore/send still degrade to safe projections; only the incompatible durable write is refused.

- [x] **Step 4: Write failing message provenance tests**

```python
def test_character_greeting_provenance_round_trips():
    metadata = MessageMetadata(
        template_kind="character_greeting",
        template_source="Hello {{user}}.",
    )
    assert MessageMetadata.from_json(metadata.to_json()) == metadata


def test_unknown_template_kind_degrades_and_drops_source():
    restored = MessageMetadata.from_json(
        json.dumps({"template_kind": "future_kind", "template_source": "secret"})
    )
    assert restored is not None
    assert restored.template_kind == ""
    assert restored.template_source == ""


def test_template_source_requires_the_closed_kind():
    with pytest.raises(ValueError, match="template_source"):
        MessageMetadata(template_source="Hello {{user}}")
```

- [x] **Step 5: Extend `MessageMetadata` with a closed template vocabulary**

Add:

```python
TEMPLATE_KINDS = frozenset({"", "character_greeting"})

template_kind: str = ""
template_source: str = ""
```

Validate `template_kind` and require source and kind to be present together. In `from_json`, drop both values unless the stored kind is exactly `character_greeting` and the source is a nonblank string. Preserve the existing unknown-key and corrupt-data degradation behavior.

- [x] **Step 6: Run Task 2 tests**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_message_metadata.py -q
```

Expected: PASS.

- [x] **Step 7: Commit Task 2**

```powershell
git add tldw_chatbook/Chat/console_roleplay_metadata.py tldw_chatbook/Chat/message_metadata.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_message_metadata.py
git commit -m "feat: persist trusted roleplay template provenance"
```

### Task 3: Persist live per-chat identity and safe projections

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py`
- Modify: `Tests/Chat/test_console_chat_store.py`
- Modify: `Tests/Chat/test_chat_persistence_service.py`

**Interfaces:**
- Consumes: Task 1 name/template helpers and Task 2 metadata helpers.
- Produces: session fields `user_display_name_override`, `character_system_template`, `identity_revision`; store methods `presentation_context(session_id, global_default)`, `set_session_user_display_name_override(session_id, value, *, global_default)`, `refresh_session_roleplay_projections(session_id, *, global_default)`, and `seed_character_roleplay(session_id, *, system_template, greeting_template, global_default)`; persistence method `update_conversation_roleplay_context(...)`.

- [x] **Step 1: Write failing session and first-persist tests**

Cover:

```python
def test_session_override_is_not_console_session_settings():
    session = ConsoleChatSession(user_display_name_override="Rowan")
    assert session.user_display_name_override == "Rowan"
    assert not hasattr(ConsoleSessionSettings(), "user_display_name_override")


def test_first_persist_flushes_roleplay_context_after_id_exists():
    store, persistence, session = make_store_with_character_session()
    session.user_display_name_override = "Rowan"
    session.character_system_template = "Speak with {{user}}."
    conversation_id = store.persist_session_if_needed(session.id)
    assert persistence.roleplay_updates[-1] == {
        "conversation_id": conversation_id,
        "user_name_override": "Rowan",
        "character_system_template": "Speak with {{user}}.",
    }


def test_temporary_session_keeps_override_without_durable_write():
    session.ephemeral = True
    persisted = store.set_session_user_display_name_override(
        session.id, "Rowan", global_default="User"
    )
    assert persisted[1] is True
    assert persistence.roleplay_updates == []
```

- [x] **Step 2: Run the focused store tests and verify RED**

Run the new selectors only:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_chat_store.py -k "roleplay or display_name or character_template" -q
```

Expected: failures for missing fields/methods.

- [x] **Step 3: Add session fields and presentation context generation**

Add to `ConsoleChatSession`:

```python
user_display_name_override: str | None = None
character_system_template: str | None = None
identity_revision: int = 0
```

`presentation_context` resolves the current name on demand, uses character identity already owned by the session, and sets revision to `identity_revision`. Every override, character name, or template-source change increments the revision and the existing payload revision.

- [x] **Step 4: Write failing optimistic metadata tests**

In `Tests/Chat/test_chat_persistence_service.py`, use a fake DB whose first `update_conversation` raises `ConflictError` after replacing the row with a sibling metadata key. Assert:

```python
assert service.update_conversation_roleplay_context(
    conversation_id="conv-1",
    user_name_override="Rowan",
    character_system_template="Speak to {{user}}.",
) is True
assert fake_db.update_attempts == 2
saved = json.loads(fake_db.row["metadata"])
assert saved["concurrent_sibling"] == {"kept": True}
assert saved["console_roleplay_context"]["user_name_override"] == "Rowan"
```

Add a second test where both attempts conflict and assert the second `ConflictError` propagates.

- [x] **Step 5: Implement bounded re-read, merge, and one retry**

Extend the persistence protocol and `ChatPersistenceService`:

```python
def update_conversation_roleplay_context(
    self,
    *,
    conversation_id: str,
    user_name_override: str | None,
    character_system_template: str | None,
) -> bool:
    for attempt in range(2):
        record = self.db.get_conversation_by_id(str(conversation_id))
        if record is None:
            return False
        metadata = merge_console_roleplay_context(
            record.get("metadata"),
            ConsoleRoleplayContext(user_name_override, character_system_template),
        )
        try:
            self.db.update_conversation(
                str(conversation_id),
                {"metadata": metadata},
                expected_version=record["version"],
            )
            return True
        except ConflictError:
            if attempt == 1:
                raise
    return False
```

Keep the loop bounded at exactly two total attempts.

- [x] **Step 6: Write failing projection materialization and edit tests**

```python
def test_rename_rematerializes_system_and_seeded_greeting():
    store, persistence, session, greeting = seeded_character_store(
        system_template="Speak with {{user}}.",
        greeting_template="Hello {{user}}.",
    )
    _, persisted = store.set_session_user_display_name_override(
        session.id, "Captain Rowan", global_default="User"
    )
    assert persisted is True
    assert session.settings.system_prompt == "Speak with Captain Rowan."
    assert store.get_message(greeting.id).content == "Hello Captain Rowan."
    assert persistence.updated_messages[-1]["content"] == "Hello Captain Rowan."


def test_editing_derived_greeting_clears_provenance():
    edited = store.update_message_content(greeting.id, "Hello there.")
    assert edited.metadata is not None
    assert edited.metadata.template_kind == ""
    assert edited.metadata.template_source == ""


def test_editing_system_prompt_clears_character_template_source():
    updated, persisted = store.set_session_system_prompt(session.id, "Be concise.")
    assert persisted is True
    assert updated.character_system_template is None
```

- [x] **Step 7: Implement atomic live materialization behavior**

`set_session_user_display_name_override` must:

1. Validate and update the in-memory override first.
2. Increment identity/payload revisions.
3. Resolve and store the current system prompt from `character_system_template` when the session is a named character session.
4. Resolve every message with closed greeting provenance and persist its safe ordinary content.
5. Persist the roleplay metadata object for durable conversations.
6. Return `persisted=False` if any attempted durable write fails, without rolling back the live session.

`update_message_content` must replace template fields with empty values before persisting an edited derived message. `set_session_system_prompt` must clear `character_system_template` and persist the metadata object alongside the ordinary system prompt.

Implement `refresh_session_roleplay_projections(session_id, *, global_default)` as the same idempotent projection/persistence core without changing the override. It returns immediately when the currently materialized system prompt and all trusted greeting projections already match, so a global-name refresh cannot create repeated version bumps.

- [x] **Step 8: Flush owned context during first persist and promotion**

After `create_conversation` returns its id, call `update_conversation_roleplay_context` once when either owned field is present. Promotion keeps the existing all-or-nothing transaction. A failed context flush on ordinary first persist is logged and observable; a failed temporary promotion stays on the existing rollback path.

- [x] **Step 9: Run Task 3 focused tests**

Run:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_message_metadata.py -k "roleplay or display_name or character_template or message_metadata or system_prompt or first_persist or ephemeral" -q
```

Temporarily skip the second metadata read in the conflict branch and verify the concurrent-sibling test fails. Restore and rerun green.

- [x] **Step 10: Commit Task 3**

```powershell
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_persistence_service.py
git commit -m "feat: persist per-chat roleplay identity"
```

### Task 4: Seed and restore trusted character templates

**Files:**
- Modify: `tldw_chatbook/Character_Chat/Character_Chat_Lib.py`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py`
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/UI/test_character_session_prompt_seed.py`
- Modify: `Tests/Character_Chat/test_compose_character_card_text.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Modify: `Tests/UI/test_console_resume_active_path.py`
- Review-fix scope expansion: `tldw_chatbook/Chat/console_chat_store.py`
- Review-fix scope expansion: `Tests/Chat/test_console_chat_store.py`
- Review-fix round 2 scope expansion: `tldw_chatbook/Chat/chat_persistence_service.py`
- Review-fix round 2 scope expansion: `Tests/Chat/test_chat_persistence_service.py`
- Review-fix round 4 scope expansion: `tldw_chatbook/Sync_Interop/sync_state_repository.py`
- Review-fix round 4 scope expansion: `Tests/Sync_Interop/test_sync_state_repository.py`

**Interfaces:**
- Consumes: Tasks 1 through 3.
- Changes: `_character_session_prompt_seed` returns a source-bearing `CharacterSessionPromptSeed` instead of a tuple.
- Produces: screen-state and durable resume round trips for override/source/revision.

- [x] **Step 1: Write failing seed tests for source and projection**

Replace tuple-only expectations with:

```python
seed = _character_session_prompt_seed(card, user_name="Captain Rowan")
assert seed.name == "Alraune"
assert "{{user}}" in seed.system_template
assert seed.system_prompt == seed.system_template.replace("{{user}}", "Captain Rowan")
assert seed.greeting_template == "Hello, {{user}}."
assert seed.greeting == "Hello, Captain Rowan."
```

Add the recursion guard with a user name containing `{{character}}`, and assert source construction preserves the card's exact macro text while character aliases resolve only in the projection.

- [x] **Step 2: Run seed tests and verify RED**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/UI/test_character_session_prompt_seed.py -q
```

Expected: failures because the helper returns a 3-tuple and hardcodes `User`.

- [x] **Step 3: Extract the shared raw character-card template composer**

Add `compose_character_card_template(...) -> str` beside `compose_character_card_text`. Move only the existing field ordering, labels, and blank-field joining into the raw helper. Keep compatibility by making `compose_character_card_text` delegate to the raw helper and then call its existing placeholder resolver:

```python
def compose_character_card_text(..., user_name: str | None = None) -> str:
    template = compose_character_card_template(
        name=name,
        system_prompt=system_prompt,
        personality=personality,
        description=description,
        scenario=scenario,
        message_example=message_example,
        post_history_instructions=post_history_instructions,
    )
    return replace_placeholders(template, name, user_name)
```

Extend `Tests/Character_Chat/test_compose_character_card_text.py` to assert the raw helper retains `{{user}}`/character aliases, the resolved helper keeps its existing outputs, and field order/labels are byte-identical apart from placeholder resolution.

- [x] **Step 4: Introduce a source-bearing seed dataclass**

```python
@dataclass(frozen=True, slots=True)
class CharacterSessionPromptSeed:
    name: str
    system_template: str
    system_prompt: str
    greeting_template: str
    greeting: str
```

Build the source with `compose_character_card_template`, never by duplicating the join or routing placeholder sentinels through `replace_placeholders`. Then call `expand_character_template` once for each projection. Preserve the fixed `Stay in character.` fallback as both source and projection when the composed source is blank.

- [x] **Step 5: Update every character entry path to seed provenance**

Update Start Chat, character picker new-chat, and character swap to:

```python
seed = _character_session_prompt_seed(card, choice.name, user_name=effective_name)
session = store.create_session(
    ...,
    character_name=seed.name,
    character_system_template=seed.system_template,
)
store.seed_character_roleplay(
    session.id,
    system_template=seed.system_template,
    greeting_template=seed.greeting_template,
    global_default=global_name,
)
```

The greeting append must attach `MessageMetadata(template_kind="character_greeting", template_source=seed.greeting_template)` and store `seed.greeting` as ordinary content. Character swaps seed a greeting only in an empty chat, matching current behavior.

Review correction: a swap must update character identity and trusted template source atomically in the store before a single materialization/persistence pass. It must not durably expose an intermediate old-template/new-character projection, and any refused durable write must be returned to the controller.

- [x] **Step 6: Write failing screen-state and resume tests**

Assert `_console_session_to_state` and `_console_session_from_state` round-trip:

```python
assert restored.user_display_name_override == "Captain Rowan"
assert restored.character_system_template == "Speak with {{user}}."
assert restored.identity_revision == original.identity_revision
```

For durable resume, provide conversation metadata containing `console_roleplay_context`; assert the restored session receives both fields and its safe stored system prompt remains usable when metadata is absent or future-versioned.

- [x] **Step 7: Implement guarded restore and serialization**

Add the three fields to the explicit session-state field list. In resume, call `parse_console_roleplay_context(conversation.get("metadata"))`; copy only valid version-one values; never infer provenance from stored content. Keep `system_prompt` from the ordinary conversation column as the safe fallback.

- [x] **Step 8: Ensure manual `/system` edits clear source across all entry points**

Route the Console Settings and `/system` apply paths through `store.set_session_system_prompt`; do not directly replace settings with a new `system_prompt`. The store owns clearing provenance and persistence.

Review correction: the general Console Settings modal does not own a system-prompt value. Saving unrelated settings must preserve the current prompt and trusted source; only an explicit prompt edit routes through `set_session_system_prompt`.

- [x] **Step 9: Run Task 4 tests**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Character_Chat/test_compose_character_card_text.py Tests/UI/test_character_session_prompt_seed.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_resume_active_path.py -k "character or template or roleplay or display_name or screen_state or system_prompt or resume" -q
```

Expected: PASS.

- [x] **Step 10: Commit Task 4**

```powershell
git add tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/workspace.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Character_Chat/test_compose_character_card_text.py Tests/UI/test_character_session_prompt_seed.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_resume_active_path.py
git commit -m "feat: seed dynamic character chat templates"
```

### Task 5: Add global and per-chat name settings

**Files:**
- Modify: `tldw_chatbook/config.py`
- Modify: `tldw_chatbook/UI/Screens/settings_screen.py`
- Modify: `tldw_chatbook/Widgets/Console/console_settings_modal.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/test_config_console_defaults.py`
- Modify: `Tests/UI/test_settings_configuration_hub.py`
- Modify: `Tests/UI/test_settings_save_commit_models.py`
- Modify: `Tests/UI/test_console_session_settings.py`
- Review-fix scope expansion: `tldw_chatbook/Chat/console_chat_store.py`
- Review-fix scope expansion: `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Produces: `get_chat_defaults_user_display_name(default="User")` and `ConsoleSettingsResult(settings, user_display_name_override)`.
- Consumes: `normalize_chat_display_name` and store identity mutation from prior tasks.

- [x] **Step 1: Write failing config isolation tests**

```python
def test_chat_display_name_uses_chat_defaults_not_general_users_name(monkeypatch, tmp_path):
    write_config(tmp_path, "[general]\nusers_name='storage-owner'\n[chat_defaults]\nuser_display_name='Rowan'\n")
    assert config_module.get_chat_defaults_user_display_name() == "Rowan"


def test_blank_chat_display_name_falls_back_to_user(monkeypatch, tmp_path):
    write_config(tmp_path, "[chat_defaults]\nuser_display_name='   '\n")
    assert config_module.get_chat_defaults_user_display_name() == "User"
```

Add `user_display_name = "User"` to the default `[chat_defaults]` TOML block, not `[general]`.

- [x] **Step 2: Run config tests and verify RED**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/test_config_console_defaults.py -k "display_name or users_name" -q
```

- [x] **Step 3: Implement the getter and default config key**

The getter reads only `get_toml_section("chat_defaults")`, validates with `blank_means_none=False`, and degrades invalid stored values to `User` with a bounded warning that does not echo the value.

- [x] **Step 4: Write failing canonical Settings tests**

Assert the Console Behavior category renders:

```python
name_input = screen.query_one("#settings-console-default-user-display-name", Input)
assert name_input.value == "Rowan"
assert "Default chat display name" in rendered_console_behavior_text(screen)
```

Stage a valid change and assert the save model writes exactly `chat_defaults.user_display_name`. Stage an over-wide CJK value and assert Save is rejected, the config/app snapshot remains unchanged, and the existing category error surface contains the 48-cell message.

- [x] **Step 5: Wire the global field into staged save, revert, and inspector copy**

Add `user_display_name` to the Console Behavior value map and save order. Render `Default chat display name` above sampling fallbacks with help copy `Used for your speaker label. Character chats also use it for trusted {{user}} templates.` Normalize at save time with the shared validator, not on every keystroke. Update the in-memory `app_config["chat_defaults"]` only after durable save succeeds.

- [x] **Step 6: Write failing per-chat modal tests**

```python
modal = ConsoleSettingsModal(
    settings=settings,
    user_display_name_override="Captain Rowan",
    global_user_display_name="Default Name",
    ...,
)
assert modal.query_one("#console-settings-user-display-name", Input).value == "Captain Rowan"
```

Assert blank input produces `ConsoleSettingsResult(..., user_display_name_override=None)`, a valid name stays separate from `ConsoleSessionSettings`, and an invalid value prevents dismissal with an inline summary.

Add a joined screen/store test with two open sessions, one inherited and one overridden. After changing the in-memory global default from `Default One` to `Default Two`, assert the inherited session's presentation label and trusted projections use `Default Two`, while the overridden session remains `Captain Rowan`. Clearing that override must immediately switch it to `Default Two`.

- [x] **Step 7: Add a result dataclass and per-chat field**

```python
@dataclass(frozen=True, slots=True)
class ConsoleSettingsResult:
    settings: ConsoleSessionSettings
    user_display_name_override: str | None
```

Render a `Chat identity` section with label `Your name in this chat`, placeholder equal to the effective global name, and help copy `Leave blank to use the global default.` Change `ConsoleSettingsModal` to return `ConsoleSettingsResult | None`.

- [x] **Step 8: Apply modal results through both owners**

In `ChatScreen`, replace provider settings from `result.settings` and call `store.set_session_user_display_name_override` with `result.user_display_name_override`. On persistence failure, keep the live value and notify: `Name changed for this session, but it may not survive reopening.` Then resync transcript, settings summary, context estimate, and prompt surfaces.

Track the last `(active_session_id, effective_global_name)` projection-refresh key on `ChatScreen`. When a Settings save or screen resume changes that key, update transcript presentation immediately from the live resolver and run `store.refresh_session_roleplay_projections(...)` through an existing async worker/off-thread persistence seam. Coalesce duplicate keys so the 0.2-second transcript tick cannot repeat writes. When an inactive tab becomes active, its key changes and receives the same refresh. On durable failure, notify once: `Your chat name is active, but updated character templates may not survive reopening.` Provider context still uses the live source projection while this worker runs or fails.

Review correction: split refresh into owner-thread live materialization plus an immutable durable persistence plan. Off-thread work must not mutate the live store; rapid changes are serialized/generation-fenced so stale plans cannot win. Canonical global Settings save and real tab activation are explicit triggers, modal results are bound to the opening session, and failure tests use real trusted templates plus a blocked/refused persistence writer.

Review correction round 2: guard each durable projection write optimistically so a generation invalidated after dispatch cannot overwrite a newer manual edit/provenance revocation; ensure the newest queued plan survives screen-worker cancellation or leaves a resume repair marker; and chain Sync v2 base versions across serialized plans so C follows B rather than reusing A. Verify with mounted unmount/cancel-latest, real `ChatPersistenceService`, and Sync producer/outbox tests. Isolate Settings lifecycle-hook exceptions after durable save.

Review correction round 3: defer projection Sync enqueue until owner-thread acceptance of the current generation so concurrent manual edits cannot create sibling A-based envelopes; replace per-refresh retained tasks with a bounded one-active/one-latest drain; and preserve predecessor candidates per system/message component after failed or partial persistence so later generations can repair the real durable ancestor.

Review correction round 4: make Sync outbox insert and returned-entry readback one atomic repository transaction, eliminating committed-row/readback ambiguity; and bound screen unmount cancellation so a hung immutable writer cannot retain the dead screen, while a screen-free writer completion is consumed and an app/resume generation marker repairs the latest current projection.

Review correction round 5: forced repair on a restored store may accept an older durable projection only under exact trusted source/provenance ownership guards and consumes its app marker only after success; immutable writers use a screen-free daemon bridge rather than the asyncio default executor so app exit cannot wait on a hung thread; and identical deterministic Sync outbox upserts preserve dispatched status instead of scheduling a redundant push.

- [x] **Step 9: Run Task 5 tests**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/test_config_console_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/UI/test_console_session_settings.py -k "display_name or console_behavior or session_settings or save or revert" -q
```

Expected: PASS.

- [x] **Step 10: Commit Task 5**

```powershell
git add tldw_chatbook/config.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/UI/Screens/chat_screen.py Tests/test_config_console_defaults.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/UI/test_console_session_settings.py
git commit -m "feat: add global and per-chat display names"
```

### Task 6: Use live presentation in provider context, actions, exports, and speech

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_controller.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `Tests/Chat/test_console_chat_controller.py`
- Modify: `Tests/Chat/test_console_speech_snapshots.py`
- Modify: `Tests/Chat/test_console_save_targets.py`
- Modify: `Tests/UI/test_console_message_controller.py`

**Interfaces:**
- Consumes: `ConsolePresentationContext` and `resolve_console_message_presentation`.
- Produces: optional controller constructor dependency `global_user_display_name: Callable[[], str] | None`, controller helper `_presentation_for(session_id, message)` used by provider/context paths, context-aware `issue_tts_message_speech_snapshot`, and a screen action helper used by Copy/Save/Edit.

- [x] **Step 1: Write failing provider-context tests**

Create a character session with safe stored projection `Hello User`, provenance source `Hello {{user}}`, then change the live override without manually editing message content. Assert:

```python
payload = controller._provider_messages_for_session(session.id)
assert payload[0]["content"] == "Speak with Captain Rowan.\n\nHello Captain Rowan."
assert all("Hello User" not in row["content"] for row in payload)
```

Also assert ordinary user text `Say {{user}} literally` remains unchanged, generated assistant text remains unchanged, strict-provider leading greeting folding uses the current projection, and `build_context_snapshot` matches the send payload.

- [x] **Step 2: Run controller tests and verify RED**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_chat_controller.py -k "character or greeting or context_snapshot or provider_messages or roleplay" -q
```

- [x] **Step 3: Centralize controller presentation lookup**

Add `global_user_display_name: Callable[[], str] | None = None` to `ConsoleChatController.__init__`; default it to `lambda: "User"` for narrow tests, and wire `ChatScreen._global_chat_display_name` in production. The helper calls that accessor, asks the store for `presentation_context`, and resolves each message. Update `_seeded_greeting_text`, `_provider_message_payloads`, `_leading_system_message`, `_provider_messages_for_session`, `_provider_messages_through_message`, and `build_context_snapshot` to consume resolved content. Preserve raw role values, attachment handling, native message ids, failure filtering, and variant selection.

- [x] **Step 4: Write failing action/export tests**

For a character greeting, assert:

```python
assert copy_result.clipboard_text == "Hello Captain Rowan."
assert note_payload["content"] == "Hello Captain Rowan."
assert media_payload["content"] == "Hello Captain Rowan."
assert prompt_payload["user_prompt"] == "Hello Captain Rowan."
assert chatbook_payload["content"] == "Hello Captain Rowan."
assert chatbook_payload["metadata"]["message_role"] == "Alraune"
```

Open the edit modal and assert its initial text is the current projection. Saving the edit must clear provenance through Task 3.

- [x] **Step 5: Replace duplicate role/content action helpers**

Remove `_console_message_role_label` and call one screen helper:

```python
def _console_message_presentation(self, message: ConsoleChatMessage) -> ConsoleMessagePresentation:
    session = self._active_console_session()
    context = store.presentation_context(session.id, self._global_chat_display_name())
    return resolve_console_message_presentation(message, context)
```

Use `speaker_label` and `content` for Copy, excerpts, Note, Media, Prompt, Chatbook, and edit entry. Continue checking raw `message.role` for authorization rules such as Chatbook assistant-only saves.

- [x] **Step 6: Write failing speech snapshot tests**

Assert a speech snapshot for a derived greeting contains the current resolved content, and changing the name invalidates the previous snapshot through the existing speech revision guard.

- [x] **Step 7: Resolve speech content while preserving snapshot fences**

Change the store API to `issue_tts_message_speech_snapshot(message_id, *, presentation_context: ConsolePresentationContext | None = None)`. A missing context preserves neutral legacy behavior for narrow callers; the Console message-action path always passes the active context. Store resolved visible content in `TTSMessageSpeechSnapshot`. Increment message speech revisions when a rename changes a derived greeting projection. Do not change TTS profile/voice selection authority.

- [x] **Step 8: Run Task 6 tests and mutation check**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_speech_snapshots.py Tests/Chat/test_console_save_targets.py Tests/UI/test_console_message_controller.py -k "roleplay or character or greeting or context or copy or save or speech or edit" -q
```

Temporarily make `_seeded_greeting_text` read `message.content` directly and verify the live-name provider test fails. Restore and rerun green.

- [x] **Step 9: Commit Task 6**

```powershell
git add tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/UI/Console_Modules/message.py tldw_chatbook/UI/Screens/chat_screen.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_speech_snapshots.py Tests/Chat/test_console_save_targets.py Tests/UI/test_console_message_controller.py
git commit -m "feat: apply roleplay identity across chat outputs"
```

### Task 7: Render named, tinted transcript rows without remounting

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Modify: `Tests/UI/test_console_native_transcript.py`
- Modify: `Tests/UI/test_console_transcript_markdown.py`
- Modify: `Tests/UI/test_console_transcript_markdown_widget.py`

**Interfaces:**
- Consumes: Task 1 presentation context/resolver.
- Adds: `ConsoleTranscript.set_presentation_context(context)` and context-aware row constructors/sync methods.

- [x] **Step 1: Write failing plain and Markdown transcript tests**

Assert both rendering modes show literal names and current projections:

```python
transcript.set_presentation_context(character_context(user_name="Captain [Rowan]"))
transcript.set_messages([user_message("Hi"), assistant_message("Hello")])
plain = transcript.to_plain_text(width=80)
assert "Captain [Rowan]" in plain
assert "Alraune" in plain
assert "Assistant" not in plain
```

Repeat with a generic session and assert the custom human label still replaces `User` while neither row receives a roleplay tint. Then change only `character_name` plus context revision and assert all current assistant rows relabel without changing their raw roles/content.

For mounted plain and Markdown rows, assert the outer widget has the correct roleplay class. Query the dedicated speaker-label child and assert its text is literal, its role-specific TCSS class is present, and markup-like characters cannot style or inject neighboring body content.

- [x] **Step 2: Write failing in-place revision tests**

Mount a transcript, capture the row widget object and scroll/selection state, call `set_presentation_context` with a new revision/name, refresh, and assert:

```python
assert transcript.query_one(f"#console-message-{message.id}") is original_row
assert transcript.selected_message_id == message.id
assert transcript.is_anchored is original_follow_state
assert "Captain Rowan" in transcript.to_plain_text()
```

- [x] **Step 3: Run transcript tests and verify RED**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_markdown.py Tests/UI/test_console_transcript_markdown_widget.py -k "roleplay or markdown or plain_text or signature or selection or tail" -q
```

- [x] **Step 4: Thread the shared presentation through every row path**

`ConsoleTranscript` stores a default neutral context for backwards-compatible tests. `set_presentation_context` clears only message signature cache entries whose presentation revision changed and schedules reconciliation. Pass presentation to `ConsoleTranscriptMessage` and `ConsoleMarkdownMessage` constructors and `sync_message` methods. Replace `_message_role_label`, `_message_body`, and `_assistant_markdown_header` label/content reads with the resolver output.

Convert the plain `ConsoleTranscriptMessage` from one combined `Static` into a height-auto `Vertical` with dedicated child widgets:

```python
class ConsoleTranscriptMessage(Vertical):
    def compose(self) -> ComposeResult:
        yield Static(
            Content(self._presentation.speaker_label),
            classes=self._speaker_label_classes(),
            markup=False,
        )
        yield Static(
            _message_body_content(self._message, self._presentation),
            classes="console-transcript-message-body",
            markup=False,
        )
```

Keep attachment, citation, status, and selection output in the body child so the visible line order stays unchanged. The Markdown row keeps its existing outer `Vertical` and `Markdown` body, but its header `Static` receives the same base speaker-label class plus the role-specific accent class. Both `sync_message` methods update child content/classes in place. This explicit child seam is required because Rich `Content` spans cannot receive TCSS classes, while the name accent must resolve through live theme tokens rather than hard-coded RGB styles.

- [x] **Step 5: Include identity revision in both refresh fingerprints**

Add the presentation context revision and names to `ConsoleTranscript._message_signature_token` and `ChatScreen._native_console_transcript_fingerprint`. In `_sync_native_console_transcript`, set the presentation context before `set_messages`. A global-name change for a session without override must therefore repaint existing rows even when no raw message changed.

- [x] **Step 6: Add semantic row and label styling**

Add exact source selectors near `.console-transcript-message`:

```tcss
.console-transcript-message-roleplay-user {
    background: $primary 8%;
}

.console-transcript-message-roleplay-character {
    background: $secondary 8%;
}

.console-transcript-speaker-label {
    height: auto;
    color: $ds-text-primary;
    text-style: bold;
}

.console-transcript-roleplay-user-label {
    color: $primary;
    text-style: bold;
}

.console-transcript-roleplay-character-label {
    color: $secondary;
    text-style: bold;
}
```

Place selected-state overrides after these selectors and include child-label overrides so `.console-transcript-message-selected` keeps `$ds-focus-bg`, `$ds-focus-fg`, and bold underline across both the outer row and speaker child. System/tool rows never receive roleplay classes. For failure/streaming rows, keep existing status copy and label while the outer roleplay tint remains lower priority than explicit failure/selected treatments.

- [x] **Step 7: Rebuild and check the CSS bundle**

Run:

```powershell
..\..\.venv\Scripts\python.exe tldw_chatbook/css/build_css.py
..\..\.venv\Scripts\python.exe tldw_chatbook/css/check_bundle_sync.py
```

Expected: both commands exit 0; a second build produces no content diff except the build script's deterministic generated header behavior.

- [x] **Step 8: Add compositor-painted dark/light tests**

Use the existing bundled-CSS Textual harness to mount character user/assistant rows under one dark and one light theme. Assert the painted background differs from the neutral panel for both roleplay rows, the selected row uses focus colors instead of the roleplay tint, and the literal speaker labels remain present. Keep the test tied to semantic colors rather than exact RGB values.

- [x] **Step 9: Run Task 7 tests and mutation check**

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_markdown.py Tests/UI/test_console_transcript_markdown_widget.py Tests/UI/test_console_transcript_selection_contract.py Tests/UI/test_console_transcript_tail_follow.py -k "roleplay or markdown or plain_text or selected or tail or signature or color" -q
..\..\.venv\Scripts\python.exe tldw_chatbook/css/check_bundle_sync.py
```

Temporarily remove the roleplay class assignment and verify the mounted/compositor tests fail. Restore and rerun green.

- [x] **Step 10: Commit Task 7**

```powershell
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/css/components/_agentic_terminal.tcss tldw_chatbook/css/tldw_cli_modular.tcss Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_markdown.py Tests/UI/test_console_transcript_markdown_widget.py
git commit -m "feat: theme named roleplay transcript rows"
```

### Task 8: Focused integration, live verification, and documentation closeout

**Files:**
- Modify: `backlog/tasks/task-14801 - Add-roleplay-chat-identities-and-speaker-theming.md`
- Modify only if an incident produced a reusable lesson: one relevant file under `backlog/docs/lessons-*.md`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: focused verification evidence and completed task hygiene.

- [x] **Step 1: Run the final related automated test set only**

Run the exact reachable suite, not the repository-wide suite:

```powershell
..\..\.venv\Scripts\python.exe -m pytest Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_message_metadata.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_speech_snapshots.py Tests/Chat/test_console_save_targets.py Tests/Character_Chat/test_compose_character_card_text.py Tests/test_config_console_defaults.py Tests/UI/test_character_session_prompt_seed.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_resume_active_path.py Tests/UI/test_console_message_controller.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_markdown.py Tests/UI/test_console_transcript_markdown_widget.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py Tests/UI/test_console_transcript_selection_contract.py Tests/UI/test_console_transcript_tail_follow.py -q
```

If a file has unrelated pre-existing failures, rerun the identical command against the pre-change base and compare exact failure node ids. Do not broaden to the full suite.

- [x] **Step 2: Run focused static analysis and generated-asset verification**

Run Ruff only on touched Python files and tests, then verify CSS sync and diff hygiene:

```powershell
..\..\.venv\Scripts\python.exe -m ruff check tldw_chatbook/Chat/console_roleplay_identity.py tldw_chatbook/Chat/console_roleplay_metadata.py tldw_chatbook/Chat/message_metadata.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_controller.py tldw_chatbook/Character_Chat/Character_Chat_Lib.py tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Console_Modules/workspace.py tldw_chatbook/UI/Console_Modules/message.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Screens/settings_screen.py tldw_chatbook/Widgets/Console/console_settings_modal.py tldw_chatbook/Widgets/Console/console_transcript.py Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_message_metadata.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_chat_controller.py Tests/Chat/test_console_speech_snapshots.py Tests/Chat/test_console_save_targets.py Tests/Character_Chat/test_compose_character_card_text.py Tests/test_config_console_defaults.py Tests/UI/test_character_session_prompt_seed.py Tests/UI/test_console_session_settings.py Tests/UI/test_console_resume_active_path.py Tests/UI/test_console_message_controller.py Tests/UI/test_console_native_transcript.py Tests/UI/test_console_transcript_markdown.py Tests/UI/test_console_transcript_markdown_widget.py Tests/UI/test_settings_configuration_hub.py Tests/UI/test_settings_save_commit_models.py
..\..\.venv\Scripts\python.exe tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

- [x] **Step 3: Run focused live verification with isolated state**

Launch the app with a scratch config/data profile and perform these checks in both dark and light themes at one wide and one narrow terminal size:

1. Open an existing character and start a new Console chat.
2. Confirm the greeting image/avatar still renders, the character row says the character name, the human row says the global name, and both rows have subtle distinct tints.
3. Send a message and confirm the model receives current character context without literal trusted macros.
4. Set `Your name in this chat` and confirm current human rows, seeded greeting, context preview, and subsequent model context update immediately.
5. Clear the override and confirm inheritance from the current global name.
6. Change the global name in canonical Settings and confirm an unoverridden open chat updates while the overridden chat does not.
7. Copy, Save As, speak, and edit the seeded greeting; confirm each sees the current projection and the edit stops following later name changes.
8. Reopen the durable chat and confirm override/source restoration. Open a safe-projection fixture without provenance and confirm it remains readable and sendable.
9. Select a row, stream a reply, show a failure row, and inspect tool/system rows to confirm their higher-priority states remain legible.
10. Inspect `app._notifications` for any persistence warning because screenshots do not include the toast rack in this Textual version.

Record the exact scratch-profile command, terminal sizes, themes, character id/name, and observed notifications in the task Implementation Notes. Do not reuse the developer's real config file.

- [x] **Step 4: Perform a fresh self-review against the spec**

Review every requirement in `Docs/superpowers/specs/2026-08-08-task-14801-roleplay-chat-identity-design.md`. Specifically grep the diff for:

```powershell
rg -n "role\.title\(\)|users_name|replace_placeholders|console_roleplay_context|template_kind|template_source|user_display_name" tldw_chatbook Tests
```

Confirm no new transcript path falls back to duplicate `role.title()`, no display identity path touches `[general].users_name`, no trusted dynamic path uses sequential placeholder replacement, and no ordinary user/generated message receives template provenance.

- [x] **Step 5: Update Backlog task hygiene**

Add concise Implementation Notes covering approach, decisions, modified files, focused test commands/results, live verification, and ADR-046. Check all six acceptance criteria only after their evidence exists. Add a lessons entry only if implementation uncovered a repeatable incident; do not manufacture one.

Then run:

```powershell
npx --yes backlog.md task edit 14801 -s Done --notes "Implemented named and theme-aware character chat identities with global/per-chat human display names, safe trusted-template provenance, shared presentation across transcript/actions/provider context, focused tests, and live dark/light verification. ADR: backlog/decisions/046-roleplay-chat-display-identity-and-template-provenance.md"
```

- [x] **Step 6: Request final code review before integration**

Use `superpowers:requesting-code-review`, address verified findings with `superpowers:receiving-code-review`, rerun only the affected focused tests, and use `superpowers:verification-before-completion` before claiming success.

- [x] **Step 7: Commit documentation closeout**

```powershell
git add "backlog/tasks/task-14801 - Add-roleplay-chat-identities-and-speaker-theming.md" backlog/docs Docs/superpowers backlog/decisions
git commit -m "docs: close roleplay chat identity task"
```

## Plan self-review record

- Spec coverage: every product, persistence, template, presentation, rendering, failure, compatibility, and verification section maps to Tasks 1 through 8.
- Placeholder scan: no deferred implementation marker or unspecified error-handling step remains.
- Type consistency: `ConsolePresentationContext`, `ConsoleMessagePresentation`, `ConsoleRoleplayContext`, `ConsoleSettingsResult`, session field names, metadata keys, and public helper names are consistent across producers and consumers.
- Scope check: no Persona-as-human pointer, User Profile selection, Sync v2 contract, database migration, custom color picker, avatar change, or TTS authority change is included.
- Test scope: commands are limited to touched files and directly reachable Console/settings behavior, honoring the owner directive not to run the full suite.

## Execution closeout

- Tasks 1 through 7 were implemented, tested with RED/GREEN and mutation evidence recorded in the ignored SDD task reports, committed, and independently reviewed. Subsequent review fixes are included through `39233ccbd`.
- Task 8 completed 863 unique related test nodes in bounded fresh-`--basetemp` groups. The final Settings Hub and transcript selection/tail commands used only selectors directly reachable from this feature; no full suite or unrelated all-file Settings run was used as completion evidence.
- Ruff was evaluated as a base-to-head ratchet because the touched scope contains 82 pre-existing findings at both revisions; normalized comparison found no new issue. CSS bundle synchronization and diff hygiene passed.
- Live verification used real Textual widgets and a real isolated scratch SQLite store in dark 160x48 and light 80x24 sessions. Provider-boundary evidence came from the local/fake controller payload seam, not a network or real-provider request. Copy, every Save As projection, speech/edit behavior, durable reopen, and provenance-free safe projections were explicitly exercised.
- The documented `npx backlog.md` command was replaced, with owner approval, by a manual local task-file update because no trusted local Backlog CLI was available and package download/network use was prohibited. Status, acceptance criteria, implementation notes, and ADR links were updated in the same canonical Markdown file.
- Temporary scratch profiles and pytest basetemps remain untracked and are excluded from this closeout. Inaccessible or aborted basetemps and unrelated pre-existing failures/ratchet findings were neither modified nor counted as evidence.
- No lessons file was changed: the only reusable incidents reinforced existing base-comparison and live-compositor guidance rather than adding new knowledge.
