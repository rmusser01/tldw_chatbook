# Console Persona Session Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persona "Chat now" binds a persona-native Console session so the transcript label shows the persona's name instead of "Assistant", persona system prompts expand `{{user}}`/`{{char}}`-family macros dynamically, and resumed persona conversations re-resolve the name — full parity with character sessions, locally and against a server.

**Architecture:** Extends the existing ADR-046 character identity machinery (stored projections, single-pass macro expansion, one shared presentation resolver) with a persona branch, per ADR-149. The session gains `assistant_name` + `persona_system_template` fields; no schema migration — the name is re-resolved from `assistant_id` at resume, and the trusted template rides the version-1 `console_roleplay_context` metadata envelope as an optional key.

**Tech Stack:** Python 3.11+, Textual, SQLite (via `CharactersRAGDB`), pytest (+ pytest-asyncio).

**Spec:** `Docs/superpowers/specs/2026-09-11-console-persona-session-identity-design.md`
**ADR:** `backlog/decisions/149-console-persona-session-identity.md`
**Backlog task:** `backlog/tasks/task-32481 - Console-persona-session-identity-label-parity-and-dynamic-identity-macros.md` (mark In Progress at start, Done at end; add Implementation Notes when done)

## Global Constraints

- **No database schema migration.** Conversations already persist `assistant_kind` / `assistant_id`; nothing else is added to the DB.
- **Persona conversations remain authority-free** (ADR-037): never resolve or store `assistant_authority_id` for a persona session.
- **Single-pass macro expansion** (ADR-046): all expansion goes through `expand_character_template`; never re-scan replacement values.
- **Character behavior must not change**: existing tests in `Tests/Chat/test_console_roleplay_identity.py`, `Tests/Chat/test_console_chat_store.py`, and the character handoff tests in `Tests/UI/test_console_native_chat_flow.py` must stay green unmodified (except where a task below explicitly extends a test helper).
- **One-name-set invariant**: at most one of `character_name` / `assistant_name` is non-blank on any session; every identity write enforces it.
- Repo notes: the `timeout` shell command is unavailable; run targeted tests only (`pytest <file> -v`), never the full suite unless asked.
- The `backlog` CLI is unavailable in this environment; update `backlog/tasks/task-32481 - ....md` by editing the file (status field, Implementation Notes section) if it remains unavailable at execution time.
- **Spec deviation (§2), flagged for plan review:** the spec called for extracting "a shared fenced-fetch spine parameterized by a kind-specific fetch and a kind-specific seed." This plan extracts the security-critical half — the server capture/currency fence — as `_capture_server_handoff_context`, shared verbatim by both starters, but keeps two starter bodies. Rationale: the authority-resolver dance, greeting-matched duplicate detection, reaction clearing, and the seed call signatures are all genuinely kind-specific; a fully parameterized spine would thread ~6 injected seams through the exact fencing code the review finding was protecting. If the full spine is preferred, say so before execution and Task 5 will be restructured.

---

### Task 1: Session identity fields + presentation persona branch

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` — `ConsoleChatSession` (fields at lines 547–551), `create_session` (883–964), `presentation_context` (3313–3325)
- Modify: `tldw_chatbook/Chat/console_roleplay_identity.py` — `ConsolePresentationContext` (118–126), `resolve_console_message_presentation` (141–217), new helper
- Test: `Tests/Chat/test_console_roleplay_identity.py`, `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: session fields `assistant_name: str | None`, `persona_system_template: str | None`; `create_session(..., assistant_name=None, persona_system_template=None)`; `session_assistant_display_name(session: object) -> str | None`; `ConsolePresentationContext.assistant_name`; persona branch in `resolve_console_message_presentation`. Every later task relies on these.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_roleplay_identity.py`:

```python
def assistant_message(content: str) -> ConsoleChatMessage:
    return ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content=content)


def test_persona_session_labels_assistant_rows_with_persona_name():
    presentation = resolve_console_message_presentation(
        assistant_message("Certainly."),
        ConsolePresentationContext(
            assistant_kind="persona", assistant_name="Archivist"
        ),
    )
    assert presentation.speaker_label == "Archivist"
    assert presentation.speaker_tone == "assistant"
    assert presentation.row_class == "console-transcript-message-role-assistant"


def test_persona_session_without_name_falls_back_to_assistant():
    presentation = resolve_console_message_presentation(
        assistant_message("Certainly."),
        ConsolePresentationContext(assistant_kind="persona", assistant_name="  "),
    )
    assert presentation.speaker_label == "Assistant"
    assert presentation.speaker_tone == "assistant"


def test_persona_name_never_gets_roleplay_character_styling():
    presentation = resolve_console_message_presentation(
        assistant_message("Certainly."),
        ConsolePresentationContext(
            assistant_kind="persona",
            assistant_name="Archivist",
            transcript_style=ConsoleTranscriptStyle.IMMERSIVE_RP,
        ),
    )
    assert presentation.speaker_label == "Archivist"
    assert presentation.speaker_tone == "assistant"
    assert presentation.row_class == "console-transcript-message-role-assistant"


def test_session_assistant_display_name_picks_per_kind():
    from types import SimpleNamespace

    from tldw_chatbook.Chat.console_roleplay_identity import (
        session_assistant_display_name,
    )

    character = SimpleNamespace(
        assistant_kind="character", character_name="Alraune", assistant_name=None
    )
    persona = SimpleNamespace(
        assistant_kind="persona", character_name=None, assistant_name=" Archivist "
    )
    generic = SimpleNamespace(
        assistant_kind="generic", character_name=None, assistant_name=None
    )
    assert session_assistant_display_name(character) == "Alraune"
    assert session_assistant_display_name(persona) == "Archivist"
    assert session_assistant_display_name(generic) is None
```

Append to `Tests/Chat/test_console_chat_store.py` (check the file's existing imports; it already imports `ConsoleChatStore`):

```python
def test_presentation_context_carries_persona_display_name():
    store = ConsoleChatStore()
    session = store.create_session(
        title="Chat with Archivist",
        assistant_kind="persona",
        assistant_id="local-persona-abc",
        assistant_name="Archivist",
    )
    context = store.presentation_context(session.id, "")
    assert context.assistant_kind == "persona"
    assert context.assistant_name == "Archivist"
    assert context.character_name is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_chat_store.py -v -k "persona or display_name"`
Expected: FAIL — `TypeError: ConsolePresentationContext() got an unexpected keyword argument 'assistant_name'` and `create_session() got an unexpected keyword argument 'assistant_name'`; `ImportError` for `session_assistant_display_name`.

- [ ] **Step 3: Implement the session fields and store plumbing**

In `tldw_chatbook/Chat/console_chat_store.py`, `ConsoleChatSession` (after the `character_name` field at line 547):

```python
    character_name: str | None = None
    #: Persona-kind assistant display name (ADR-149). Set only when
    #: ``assistant_kind == "persona"``; mutually exclusive with
    #: ``character_name``. Never persisted on the conversation row --
    #: re-resolved from ``assistant_id`` at resume.
    assistant_name: str | None = None
```

and after `character_system_template` (line 551):

```python
    #: Trusted persona system-prompt source (ADR-149); materialized into
    #: ``settings.system_prompt`` via the identity-template expander and
    #: persisted as an optional key in the version-1 roleplay envelope.
    persona_system_template: str | None = None
```

In `create_session` (883), add two keyword parameters after `character_name: str | None = None,`:

```python
        character_name: str | None = None,
        assistant_name: str | None = None,
        persona_system_template: str | None = None,
```

and pass them into the `ConsoleChatSession(...)` constructor after `character_name=character_name,`:

```python
            character_name=character_name,
            assistant_name=assistant_name,
            persona_system_template=persona_system_template,
```

In `presentation_context` (3313–3325), add one line:

```python
        return ConsolePresentationContext(
            user_name=effective_user_display_name(
                session.user_display_name_override, global_default
            ),
            assistant_kind=session.assistant_kind,
            character_name=session.character_name,
            assistant_name=session.assistant_name,
            revision=session.identity_revision,
        )
```

Do NOT change `restore_persisted_session`'s signature — hydration assigns persona fields post-restore (mirroring `character_system_template` at `console_conversation_hydration.py:438`).

- [ ] **Step 4: Implement the identity-module changes**

In `tldw_chatbook/Chat/console_roleplay_identity.py`:

Add the field to `ConsolePresentationContext` (after `character_name`):

```python
@dataclass(frozen=True, slots=True)
class ConsolePresentationContext:
    """Identity inputs used to project one Console transcript message."""

    user_name: str = "User"
    assistant_kind: str | None = "generic"
    character_name: str | None = None
    assistant_name: str | None = None
    revision: int = 0
    transcript_style: ConsoleTranscriptStyle = DEFAULT_CONSOLE_TRANSCRIPT_STYLE
```

Add the helper after `effective_user_display_name`:

```python
def session_assistant_display_name(session: object) -> str | None:
    """Return the session's kind-appropriate assistant display name.

    Character sessions read ``character_name``; persona sessions read
    ``assistant_name`` (ADR-149); anything else has no bound identity.
    Blank values are treated as absent.
    """
    kind = getattr(session, "assistant_kind", None)
    if kind == "character":
        value = getattr(session, "character_name", None)
    elif kind == "persona":
        value = getattr(session, "assistant_name", None)
    else:
        return None
    text = value.strip() if isinstance(value, str) else ""
    return text or None
```

In `resolve_console_message_presentation`, extend the name resolution block (lines 145–156):

```python
    raw_character_name = (
        context.character_name.strip()
        if isinstance(context.character_name, str)
        else ""
    )
    raw_persona_name = (
        context.assistant_name.strip()
        if isinstance(context.assistant_name, str)
        else ""
    )
    is_character_session = (
        context.assistant_kind == "character" and bool(raw_character_name)
    )
    is_persona_session = (
        context.assistant_kind == "persona" and bool(raw_persona_name)
    )
    character_display_name = sanitize_character_display_label(
        raw_character_name,
        max_characters=CHARACTER_SPEAKER_LABEL_MAX_CHARACTERS,
    )
    persona_display_name = sanitize_character_display_label(
        raw_persona_name,
        max_characters=CHARACTER_SPEAKER_LABEL_MAX_CHARACTERS,
    )
```

and the ASSISTANT branch (lines 174–183):

```python
    elif message.role is ConsoleMessageRole.ASSISTANT:
        if is_character_session:
            speaker_label = character_display_name
        elif is_persona_session:
            speaker_label = persona_display_name
        else:
            speaker_label = "Assistant"
        speaker_tone = "character" if is_character_session else "assistant"
        row_class = None
        if role_accents:
            row_class = (
                "console-transcript-message-roleplay-character"
                if is_character_session
                # Persona rows keep the standard assistant treatment:
                # personas are not roleplay characters (ADR-149).
                else "console-transcript-message-role-assistant"
            )
```

(The greeting-template block below it stays gated on `is_character_session` — personas have no greeting templates.)

Add one line to `expand_character_template`'s docstring:

```python
def expand_character_template(
    source: str, *, user_name: str, character_name: str
) -> str:
    """Expand trusted character template aliases in one non-recursive pass.

    This is the identity-template expander for both character and persona
    sessions (ADR-149); ``character_name`` carries the persona name for
    persona-kind content.
    """
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_chat_store.py -v`
Expected: PASS, including all pre-existing tests (character branch unchanged).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/console_roleplay_identity.py Tests/Chat/test_console_roleplay_identity.py Tests/Chat/test_console_chat_store.py
git commit -m "feat: persona display identity on Console sessions + transcript persona branch (task-32481)"
```

---

### Task 2: Roleplay envelope `persona_system_template` persistence

**Files:**
- Modify: `tldw_chatbook/Chat/console_roleplay_metadata.py` (whole file is the envelope: dataclass, parse, merge)
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py` — `update_conversation_roleplay_context` (480–515)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` — Protocol signature (365–372), `_persist_roleplay_context` (4101–4120), first-persist flush condition (5524–5527)
- Test: `Tests/Chat/test_console_roleplay_metadata.py`, `Tests/Chat/test_chat_persistence_service.py`

**Interfaces:**
- Consumes: Task 1's `session.persona_system_template`.
- Produces: `ConsoleRoleplayContext.persona_system_template: str | None = None`; `update_conversation_roleplay_context(..., persona_system_template: str | None)`; envelope key `persona_system_template` under `console_roleplay_context` version 1. Tasks 3 and 6 rely on these.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_roleplay_metadata.py` (it already imports `json`, `ConsoleRoleplayContext`, `merge_console_roleplay_context`, `parse_console_roleplay_context` — verify and extend imports as needed):

```python
def test_persona_template_round_trip_preserves_sibling_keys():
    merged = merge_console_roleplay_context(
        '{"unrelated": true}',
        ConsoleRoleplayContext(persona_system_template="Guide {{user}}."),
    )
    decoded = json.loads(merged)
    assert decoded["unrelated"] is True
    owned = decoded["console_roleplay_context"]
    assert owned["version"] == 1
    assert "user_name_override" not in owned
    assert "character_system_template" not in owned
    assert owned["persona_system_template"] == "Guide {{user}}."
    parsed = parse_console_roleplay_context(merged)
    assert parsed.persona_system_template == "Guide {{user}}."


def test_envelope_without_persona_key_parses_as_before():
    merged = merge_console_roleplay_context(
        None,
        ConsoleRoleplayContext(
            user_name_override="Rowan", character_system_template="Be {{char}}."
        ),
    )
    parsed = parse_console_roleplay_context(merged)
    assert parsed.user_name_override == "Rowan"
    assert parsed.character_system_template == "Be {{char}}."
    assert parsed.persona_system_template is None


def test_blank_persona_template_fails_soft_and_all_empty_pops_envelope():
    assert parse_console_roleplay_context(
        '{"console_roleplay_context": '
        '{"version": 1, "persona_system_template": "  "}}'
    ) == ConsoleRoleplayContext()
    merged = merge_console_roleplay_context(
        '{"console_roleplay_context": '
        '{"version": 1, "persona_system_template": "x"}}',
        ConsoleRoleplayContext(),
    )
    assert json.loads(merged) == {}
```

Append to `Tests/Chat/test_chat_persistence_service.py`, inside `TestChatPersistenceService` (uses the file's `db_instance` fixture; `service.create_conversation` kwargs mirror its existing conversation-creation tests):

```python
    def test_roleplay_context_round_trips_persona_template(
        self, db_instance: CharactersRAGDB
    ):
        service = ChatPersistenceService(db_instance)
        conversation_id = service.create_conversation(
            conversation_title="Chat with Archivist",
            runtime_backend="local",
            assistant_kind="persona",
            assistant_id="local-persona-abc",
        )
        assert service.update_conversation_roleplay_context(
            conversation_id=conversation_id,
            user_name_override=None,
            character_system_template=None,
            persona_system_template="Guide {{user}}.",
        )
        record = db_instance.get_conversation_by_id(conversation_id)
        owned = json.loads(record["metadata"])["console_roleplay_context"]
        assert owned["version"] == 1
        assert owned["persona_system_template"] == "Guide {{user}}."
```

(If `create_conversation`'s exact required kwargs differ, copy the call shape from a neighboring passing test in the same class and keep the identity kwargs above.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_roleplay_metadata.py -v -k persona` and `pytest Tests/Chat/test_chat_persistence_service.py -v -k persona_template`
Expected: FAIL — `TypeError: ConsoleRoleplayContext() got an unexpected keyword argument 'persona_system_template'`; `TypeError` on the service kwarg.

- [ ] **Step 3: Implement the envelope key**

In `tldw_chatbook/Chat/console_roleplay_metadata.py`:

```python
@dataclass(frozen=True, slots=True)
class ConsoleRoleplayContext:
    """Trusted, optional roleplay context stored on a conversation."""

    user_name_override: str | None = None
    character_system_template: str | None = None
    persona_system_template: str | None = None
```

In `parse_console_roleplay_context`, after the `character_system_template` validation block:

```python
    persona_system_template = owned_context.get("persona_system_template")
    if persona_system_template is not None and (
        not isinstance(persona_system_template, str)
        or not persona_system_template.strip()
    ):
        return ConsoleRoleplayContext()

    return ConsoleRoleplayContext(
        user_name_override=user_name_override,
        character_system_template=character_system_template,
        persona_system_template=persona_system_template,
    )
```

In `merge_console_roleplay_context`, after the `character_system_template` normalization:

```python
    persona_system_template = context.persona_system_template
    if (
        not isinstance(persona_system_template, str)
        or not persona_system_template.strip()
    ):
        persona_system_template = None

    if (
        user_name_override is None
        and character_system_template is None
        and persona_system_template is None
    ):
        metadata.pop(ROLEPLAY_CONTEXT_METADATA_KEY, None)
    else:
        metadata[ROLEPLAY_CONTEXT_METADATA_KEY] = {
            "version": ROLEPLAY_CONTEXT_VERSION,
            **(
                {"user_name_override": user_name_override}
                if user_name_override is not None
                else {}
            ),
            **(
                {"character_system_template": character_system_template}
                if character_system_template is not None
                else {}
            ),
            **(
                {"persona_system_template": persona_system_template}
                if persona_system_template is not None
                else {}
            ),
        }
    return json.dumps(metadata, sort_keys=True)
```

(The version stays 1 — an optional key under per-key fail-soft parsing, per ADR-149.)

- [ ] **Step 4: Plumb the persistence seam**

In `tldw_chatbook/Chat/chat_persistence_service.py::update_conversation_roleplay_context`:

```python
    def update_conversation_roleplay_context(
        self,
        *,
        conversation_id: str,
        user_name_override: str | None,
        character_system_template: str | None,
        persona_system_template: str | None = None,
    ) -> bool:
```

and pass it into the context constructor:

```python
            metadata = merge_console_roleplay_context(
                record.get("metadata"),
                ConsoleRoleplayContext(
                    user_name_override=user_name_override,
                    character_system_template=character_system_template,
                    persona_system_template=persona_system_template,
                ),
            )
```

In `tldw_chatbook/Chat/console_chat_store.py`, Protocol method (365–372):

```python
    def update_conversation_roleplay_context(
        self,
        *,
        conversation_id: str,
        user_name_override: str | None,
        character_system_template: str | None,
        persona_system_template: str | None = None,
    ) -> bool:
        """Persist Console-owned roleplay identity context for a conversation."""
```

`_persist_roleplay_context` (4101–4120) — add one kwarg:

```python
                writer(
                    conversation_id=session.persisted_conversation_id,
                    user_name_override=session.user_display_name_override,
                    character_system_template=session.character_system_template,
                    persona_system_template=session.persona_system_template,
                )
```

`persist_session_if_needed` first-persist flush condition (5524–5527):

```python
        if (
            session.user_display_name_override is not None
            or session.character_system_template is not None
            or session.persona_system_template is not None
        ) and not self._persist_roleplay_context(session):
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_chat_persistence_service.py Tests/Chat/test_console_chat_store.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_roleplay_metadata.py tldw_chatbook/Chat/chat_persistence_service.py tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_roleplay_metadata.py Tests/Chat/test_chat_persistence_service.py
git commit -m "feat: persist persona system template in v1 roleplay envelope (task-32481)"
```

---

### Task 3: Store persona identity operations

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py` — `_is_named_character_session` (3573–3579), `_roleplay_projection_is_stale` (3581–3610), `_materialize_roleplay_projections_live` (3626–3690), `seed_character_roleplay` (3395–3425), `swap_session_character_roleplay` (3427–3493), `set_session_character_name` (3533–3551)
- Test: `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: Task 1 session fields; Task 2 `_persist_roleplay_context` plumbing.
- Produces: `ConsoleChatStore._is_named_persona_session(session) -> bool`, `._is_named_identity_session(session) -> bool`, `._identity_display_name(session) -> str`, `._identity_system_template(session) -> str | None`; `seed_persona_roleplay(session_id, *, system_template: str, global_default: object) -> None`; `set_session_assistant_name(session_id, assistant_name: str | None, *, global_default: object) -> tuple[ConsoleChatSession, bool]`. Tasks 5 and 6 call these.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Chat/test_console_chat_store.py`. Imports to add if missing: `from dataclasses import replace`, `from types import SimpleNamespace`, `from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings`.

```python
def _persona_settings() -> ConsoleSessionSettings:
    return ConsoleSessionSettings(provider="anthropic", model="claude-3-haiku")


def _persona_session(store: ConsoleChatStore) -> ConsoleChatSession:
    return store.create_session(
        title="Chat with Archivist",
        settings=_persona_settings(),
        assistant_kind="persona",
        assistant_id="local-persona-abc",
        assistant_name="Archivist",
    )


def test_named_identity_gates_separate_kinds():
    persona = SimpleNamespace(
        assistant_kind="persona", character_name=None, assistant_name="Archivist"
    )
    character = SimpleNamespace(
        assistant_kind="character", character_name="Alraune", assistant_name=None
    )
    generic = SimpleNamespace(
        assistant_kind="generic", character_name=None, assistant_name=None
    )
    assert ConsoleChatStore._is_named_persona_session(persona) is True
    assert ConsoleChatStore._is_named_character_session(persona) is False
    assert ConsoleChatStore._is_named_persona_session(character) is False
    assert ConsoleChatStore._is_named_character_session(character) is True
    assert ConsoleChatStore._is_named_identity_session(persona) is True
    assert ConsoleChatStore._is_named_identity_session(character) is True
    assert ConsoleChatStore._is_named_identity_session(generic) is False


def test_persona_seed_expands_identity_macros_into_system_prompt():
    store = ConsoleChatStore()
    session = _persona_session(store)
    store.seed_persona_roleplay(
        session.id,
        system_template="Guide {{user}} as {{persona}}.",
        global_default="Rowan",
    )
    assert session.persona_system_template == "Guide {{user}} as {{persona}}."
    assert session.settings is not None
    assert session.settings.system_prompt == "Guide Rowan as Archivist."
    # Personas have no greeting: seeding never appends a message.
    assert store.messages_for_session(session.id) == []


def test_persona_seed_without_system_prompt_keeps_default_settings():
    store = ConsoleChatStore()
    session = _persona_session(store)
    store.seed_persona_roleplay(
        session.id, system_template="  ", global_default="Rowan"
    )
    assert session.persona_system_template is None
    assert session.settings is not None
    assert session.settings.system_prompt == _persona_settings().system_prompt
    assert session.assistant_name == "Archivist"


def test_persona_rename_rematerializes_projection():
    store = ConsoleChatStore()
    session = _persona_session(store)
    store.seed_persona_roleplay(
        session.id,
        system_template="I am {{char}} for {{user}}.",
        global_default="Rowan",
    )
    _session, persisted = store.set_session_assistant_name(
        session.id, "Scribe", global_default="Rowan"
    )
    assert persisted is True
    assert session.assistant_name == "Scribe"
    assert session.settings is not None
    assert session.settings.system_prompt == "I am Scribe for Rowan."


def test_set_session_assistant_name_clears_character_name():
    store = ConsoleChatStore()
    session = store.create_session(
        title="Chat with Alraune",
        settings=_persona_settings(),
        assistant_kind="character",
        assistant_id="7",
        character_id=7,
        character_name="Alraune",
    )
    store.set_session_assistant_name(session.id, "Archivist", global_default="Rowan")
    assert session.assistant_name == "Archivist"
    assert session.character_name is None


def test_character_swap_off_persona_session_clears_persona_identity():
    store = ConsoleChatStore()
    session = _persona_session(store)
    store.seed_persona_roleplay(
        session.id, system_template="Guide {{user}}.", global_default="Rowan"
    )
    # The screen sets kind/id fields before calling the store seam; mirror it.
    session.assistant_kind = "character"
    session.assistant_id = "7"
    session.character_id = 7
    _session, _greeting, persisted = store.swap_session_character_roleplay(
        session.id,
        character_name="Alraune",
        system_template="Be {{char}}.",
        greeting_template="",
        global_default="Rowan",
    )
    assert persisted is True
    assert session.character_name == "Alraune"
    assert session.assistant_name is None
    assert session.persona_system_template is None
    assert session.settings is not None
    assert session.settings.system_prompt == "Be Alraune."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/Chat/test_console_chat_store.py -v -k "persona or named_identity or swap_off"`
Expected: FAIL — `AttributeError` for `_is_named_persona_session` / `seed_persona_roleplay` / `set_session_assistant_name`.

- [ ] **Step 3: Implement the gates and kind-aware accessors**

In `console_chat_store.py`, after `_is_named_character_session` (3573–3579):

```python
    @staticmethod
    def _is_named_persona_session(session: ConsoleChatSession) -> bool:
        return (
            session.assistant_kind == "persona"
            and isinstance(session.assistant_name, str)
            and bool(session.assistant_name.strip())
        )

    @classmethod
    def _is_named_identity_session(cls, session: ConsoleChatSession) -> bool:
        return cls._is_named_character_session(
            session
        ) or cls._is_named_persona_session(session)

    @staticmethod
    def _identity_system_template(session: ConsoleChatSession) -> str | None:
        """Return the trusted template for the session's bound kind."""
        if session.assistant_kind == "persona":
            return session.persona_system_template
        return session.character_system_template

    @staticmethod
    def _identity_display_name(session: ConsoleChatSession) -> str:
        """Return the session's bound assistant display name, or ""."""
        return (session_assistant_display_name(session) or "").strip()
```

Add the import at the top of the file, extending the existing identity import block (line 56 imports `expand_character_template` from the same module):

```python
from tldw_chatbook.Chat.console_roleplay_identity import (
    session_assistant_display_name,
)
```

- [ ] **Step 4: Generalize staleness and materialization**

In `_roleplay_projection_is_stale` (3581–3610), change the gate and the two reads:

```python
    def _roleplay_projection_is_stale(
        self, session: ConsoleChatSession, global_default: object
    ) -> bool:
        if not self._is_named_identity_session(session):
            return False
        context = self.presentation_context(session.id, global_default)
        identity_name = self._identity_display_name(session)
        template = self._identity_system_template(session)
        if template and session.settings is not None:
            if session.settings.system_prompt != expand_character_template(
                template,
                user_name=context.user_name,
                character_name=identity_name,
            ):
                return True
```

(The greeting-provenance loop below stays exactly as-is: it keys on
`template_kind == "character_greeting"`, which persona sessions never carry.)

In `_materialize_roleplay_projections_live` (3626–3690), change the gate and reads the same way:

```python
        session = self._session_or_raise(session_id)
        if not self._is_named_identity_session(session):
            return None
        context = self.presentation_context(session_id, global_default)
        identity_name = self._identity_display_name(session)
        template = self._identity_system_template(session)
        system_prompt_write: _RoleplaySystemPromptWrite | None = None
        message_writes: list[_RoleplayMessageProjectionWrite] = []
        if template and session.settings is not None:
            projected_system = expand_character_template(
                template,
                user_name=context.user_name,
                character_name=identity_name,
            )
```

(Everything below — the settings write, the greeting loop, the persistence
plan — is unchanged; the greeting loop is inert for persona sessions.)

- [ ] **Step 5: Implement `seed_persona_roleplay` and `set_session_assistant_name`; harden the swap**

After `seed_character_roleplay` (ends ~3425):

```python
    def seed_persona_roleplay(
        self,
        session_id: str,
        *,
        system_template: str,
        global_default: object,
    ) -> None:
        """Seed the trusted persona system source into a fresh session.

        Persona profiles have no greeting field, so unlike
        ``seed_character_roleplay`` this never appends a message.
        """
        session = self._session_or_raise(session_id)
        source = (
            system_template
            if isinstance(system_template, str) and system_template.strip()
            else None
        )
        source_changed = session.persona_system_template != source
        session.persona_system_template = source
        if source_changed:
            self._bump_identity_revision(session_id)
        context_persisted = self._persist_roleplay_context(session)
        self._materialize_roleplay_projections(
            session_id, global_default=global_default
        )
        if not context_persisted:
            logger.warning("Failed to persist seeded Console persona context.")
```

After `set_session_character_name` (ends ~3551):

```python
    def set_session_assistant_name(
        self,
        session_id: str,
        assistant_name: str | None,
        *,
        global_default: object,
    ) -> tuple[ConsoleChatSession, bool]:
        """Set persona identity through the projection revision seam.

        Setting a non-blank persona name clears any character name, keeping
        the one-name-set invariant (ADR-149).
        """
        session = self._session_or_raise(session_id)
        normalized = assistant_name.strip() if isinstance(assistant_name, str) else ""
        new_name = normalized or None
        if session.assistant_name == new_name:
            return session, True
        session.assistant_name = new_name
        if new_name is not None:
            session.character_name = None
        self._bump_identity_revision(session_id)
        persisted = self._materialize_roleplay_projections(
            session_id, global_default=global_default
        )
        return session, persisted
```

In `swap_session_character_roleplay` (3453–3471), extend `identity_changed` and clear persona fields:

```python
        identity_changed = (
            session.character_name != new_name
            or session.character_system_template != source
            or session.assistant_name is not None
            or session.persona_system_template is not None
        )
        source_changed = session.character_system_template != source
        session.character_name = new_name
        session.character_system_template = source
        # One-name-set invariant (ADR-149): a character swap retires any
        # persona identity bound to this session.
        session.assistant_name = None
        session.persona_system_template = None
        if identity_changed:
            self._bump_identity_revision(session_id)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest Tests/Chat/test_console_chat_store.py -v`
Expected: PASS, including all pre-existing store tests.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py Tests/Chat/test_console_chat_store.py
git commit -m "feat: persona identity operations in Console chat store (task-32481)"
```

---

### Task 4: Persona handoff extraction + prompt seed

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/session.py` — after `_character_session_identity_from_handoff` (304–348) and after `CharacterSessionPromptSeed`/`_character_session_prompt_seed` (351–429)
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Consumes: nothing from earlier tasks (pure functions), though it is only reachable once Task 5 wires the starter.
- Produces: `_persona_session_identity_from_handoff(payload: ChatHandoffPayload) -> tuple[str, str, str, str] | None` returning `(runtime_backend, persona_id, persona_name, assistant_id)`; `PersonaSessionPromptSeed(name: str, system_template: str, system_prompt: str)`; `_persona_session_prompt_seed(profile: Mapping, name_hint: str = "", *, user_name: str = "User") -> PersonaSessionPromptSeed`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_native_chat_flow.py`, next to the character handoff helpers (the file already imports `ChatHandoffPayload`):

```python
def _persona_start_handoff(
    *,
    runtime_backend: str = "local",
    active_server_profile_id: str | None = None,
    persona_id: object = "local-persona-abc",
) -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="personas",
        item_type="persona-card",
        title="Archivist",
        body="Persona summary",
        runtime_backend=runtime_backend,
        source_owner=runtime_backend,
        source_selector_state=runtime_backend,
        active_server_profile_id=active_server_profile_id,
        metadata={
            "intent": "start_chat",
            "selected_kind": "persona",
            "selected_record_id": persona_id,
            "selected_name": "Archivist",
            "selected_target_id": f"{runtime_backend}:persona:{persona_id}",
            "backend": runtime_backend,
        },
    )


def test_persona_handoff_accepts_exact_coherent_payload():
    payload = _persona_start_handoff()
    assert _persona_session_identity_from_handoff(payload) == (
        "local",
        "local-persona-abc",
        "Archivist",
        "local-persona-abc",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        pytest.param("item_type", "character-card", id="character-item-type"),
        pytest.param("runtime_backend", "LOCAL", id="non-exact-runtime-source"),
        pytest.param("source_owner", "server", id="contradictory-owner"),
    ),
)
def test_persona_handoff_rejects_incoherent_envelope(field, value):
    payload = replace(_persona_start_handoff(), **{field: value})
    assert _persona_session_identity_from_handoff(payload) is None


def test_persona_handoff_rejects_character_kind_and_target_mismatch():
    character_payload = _character_start_handoff()
    assert _persona_session_identity_from_handoff(character_payload) is None
    mismatched = _persona_start_handoff()
    mismatched.metadata["selected_target_id"] = "local:persona:someone-else"
    assert _persona_session_identity_from_handoff(mismatched) is None


def test_persona_handoff_preserves_opaque_string_id():
    payload = _persona_start_handoff(persona_id="srv-persona-7")
    payload.metadata["selected_target_id"] = "local:persona:srv-persona-7"
    identity = _persona_session_identity_from_handoff(payload)
    assert identity is not None
    assert identity[1] == "srv-persona-7"  # never int-coerced


def test_persona_seed_expands_user_and_persona_macros():
    seed = _persona_session_prompt_seed(
        {"id": "local-persona-abc", "name": "Archivist",
         "system_prompt": "Guide {{user}} as {{persona}} ({{char}})."},
        user_name="Rowan",
    )
    assert seed.name == "Archivist"
    assert seed.system_template == "Guide {{user}} as {{persona}} ({{char}})."
    assert seed.system_prompt == "Guide Rowan as Archivist (Archivist)."


def test_persona_seed_without_system_prompt_stays_empty():
    seed = _persona_session_prompt_seed(
        {"id": "local-persona-abc", "name": "Archivist"}, user_name="Rowan"
    )
    assert seed.system_template == ""
    assert seed.system_prompt == ""


def test_persona_seed_uses_name_hint_and_stays_single_pass():
    seed = _persona_session_prompt_seed(
        {"id": "p1", "system_prompt": "Hi {{user}}"},
        name_hint=" {{char}} ",
        user_name="{{user}}",
    )
    # The hint is sanitized but never macro-expanded; the user name is
    # substituted literally exactly once (ADR-046 single-pass rule).
    assert seed.name == "{{char}}"
    assert seed.system_prompt == "Hi {{user}}"
```

(`replace` is `dataclasses.replace` — `ChatHandoffPayload` is a frozen dataclass in these tests; if the file's existing reject tests mutate instead, mirror them. Add imports: `from tldw_chatbook.UI.Console_Modules.session import _persona_session_identity_from_handoff, _persona_session_prompt_seed` — place beside the file's existing private imports of that module.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "persona_handoff or persona_seed"`
Expected: FAIL — `ImportError` on both names.

- [ ] **Step 3: Implement the extractor and seed**

In `tldw_chatbook/UI/Console_Modules/session.py`, after `_character_session_identity_from_handoff` (ends line 348):

```python
def _persona_session_identity_from_handoff(
    payload: ChatHandoffPayload,
) -> tuple[str, str, str, str] | None:
    """Return persona session identity for Personas Start Chat handoffs.

    Mirrors ``_character_session_identity_from_handoff`` with one deliberate
    difference: persona IDs are opaque strings (``local-persona-<hex>``
    locally, server IDs remotely), so no integer coercion or canonical
    numeric check applies.

    Returns:
        A tuple of `(runtime_backend, persona_id, persona_name,
        assistant_id)` when the payload is a source-aware Personas persona
        Start Chat handoff; otherwise `None`.
    """
    metadata = payload.metadata
    if not isinstance(metadata, Mapping):
        return None
    if (
        payload.source != "personas"
        or payload.item_type != "persona-card"
        or metadata.get("intent") != "start_chat"
        or metadata.get("selected_kind") != "persona"
    ):
        return None
    runtime_backend = payload.runtime_backend
    if runtime_backend not in {"local", "server"}:
        return None
    if (
        payload.source_owner != runtime_backend
        or payload.source_selector_state != runtime_backend
        or metadata.get("backend") != runtime_backend
    ):
        return None

    persona_id = str(metadata.get("selected_record_id") or "").strip()
    if not persona_id:
        return None
    if (
        metadata.get("selected_target_id")
        != f"{runtime_backend}:persona:{persona_id}"
    ):
        return None

    persona_name = str(metadata.get("selected_name") or payload.title or "").strip()
    return runtime_backend, persona_id, persona_name, persona_id
```

After `_character_session_prompt_seed` (ends ~429). Check whether
`sanitize_character_display_label` is already imported in this module
(`character.py` imports it from `...UI.character_display_text`); if not, add
it to the existing identity imports near line 164:

```python
@dataclass(frozen=True, slots=True)
class PersonaSessionPromptSeed:
    """Trusted persona source and its current safe projection."""

    name: str
    system_template: str
    system_prompt: str


def _persona_session_prompt_seed(
    profile: Mapping[str, Any], name_hint: str = "", *, user_name: str = "User"
) -> PersonaSessionPromptSeed:
    """Return the trusted persona source and its safe projection.

    Unlike character cards there is no multi-field join: a persona profile
    contributes its ``system_prompt`` field verbatim as the trusted
    template, and there is no greeting.
    """
    raw_name = str(profile.get("name") or "").strip() or str(name_hint or "").strip()
    name = (
        sanitize_character_display_label(raw_name, max_characters=180) or "Persona"
    )
    template = str(profile.get("system_prompt") or "")
    system_prompt = (
        expand_character_template(template, user_name=user_name, character_name=name)
        if template.strip()
        else ""
    )
    return PersonaSessionPromptSeed(
        name=name, system_template=template, system_prompt=system_prompt
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "persona_handoff or persona_seed or character_start_handoff"`
Expected: PASS, including the pre-existing character extraction tests.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/session.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: persona start-chat handoff extraction + prompt seed (task-32481)"
```

---

### Task 5: Persona session starter + server-fence helper + repurpose persona branch

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/session.py` — new `_capture_server_handoff_context` (before `_start_character_console_session`, ~2522), character starter server block (2549–2648), new `_start_persona_console_session` (after 2827)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — `_consume_pending_chat_handoff` (11253–11293)
- Modify: `tldw_chatbook/Chat/console_chat_store.py` — `repurpose_pristine_session` (1015–1096)
- Test: `Tests/UI/test_console_native_chat_flow.py`, `Tests/Chat/test_console_chat_store.py`

**Interfaces:**
- Consumes: Task 1 `create_session(..., assistant_name=...)`; Task 3 `seed_persona_roleplay`; Task 4 `_persona_session_identity_from_handoff` / `_persona_session_prompt_seed`.
- Produces: `ConsoleSessionController._capture_server_handoff_context(payload) -> tuple[object, Callable[[], bool]] | None`; `ConsoleSessionController._start_persona_console_session(payload) -> bool`; `repurpose_pristine_session(..., character_name: str | None, assistant_name: str | None = None)` with a validated persona branch. Task 6's resume overlay relies on the same identity fields.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_native_chat_flow.py`, beside the character handoff helpers (`_persona_profile` near `_character_card` at 4405, the rest near `_character_handoff_runtime` at 4417):

```python
def _persona_profile() -> dict:
    return {
        "id": "local-persona-abc",
        "name": "Archivist",
        "system_prompt": "Guide {{user}} as {{persona}}.",
    }


def _persona_start_handoff(
    *,
    runtime_backend: str = "local",
    active_server_profile_id: str | None = None,
) -> ChatHandoffPayload:
    return ChatHandoffPayload(
        source="personas",
        item_type="persona-card",
        title="Archivist",
        body="Persona summary",
        runtime_backend=runtime_backend,
        source_owner=runtime_backend,
        source_selector_state=runtime_backend,
        active_server_profile_id=active_server_profile_id,
        metadata={
            "intent": "start_chat",
            "selected_kind": "persona",
            "selected_record_id": "local-persona-abc",
            "selected_name": "Archivist",
            "selected_target_id": f"{runtime_backend}:persona:local-persona-abc",
            "backend": runtime_backend,
        },
    )


def _persona_handoff_runtime(
    *,
    active_server_id: str | None = None,
    profile=_DEFAULT_HANDOFF_VALUE,
) -> SimpleNamespace:
    """Persona mirror of `_character_handoff_runtime` with sentinel seams."""
    scoped_profile = _persona_profile() if profile is _DEFAULT_HANDOFF_VALUE else profile
    scope_service = SimpleNamespace(
        get_persona_profile=AsyncMock(return_value=scoped_profile),
        # The persona flow must never touch the character fetch seam.
        get_character=AsyncMock(side_effect=AssertionError("character seam touched")),
    )
    db = SimpleNamespace(
        # Personas are authority-free: no local authority lookup either.
        get_local_authority_id=Mock(return_value="local-authority"),
    )
    initial_capture = SimpleNamespace(account="A")
    authority_context_state = {"current": initial_capture}
    capture_context = Mock(
        side_effect=lambda *, expected_server_id: authority_context_state["current"]
    )
    capture_is_current = Mock(
        side_effect=lambda capture: capture is authority_context_state["current"]
    )
    resolver = AsyncMock(
        side_effect=AssertionError("persona flow must not resolve authority")
    )
    app = SimpleNamespace(
        app_config={},
        active_server_id=active_server_id,
        chachanotes_db=db,
        character_persona_scope_service=scope_service,
        server_context_provider=SimpleNamespace(
            capture_character_authority_context=capture_context,
            is_character_authority_context_current=capture_is_current,
            resolve_character_authority_id=resolver,
        ),
    )
    return SimpleNamespace(
        app=app,
        db=db,
        scope_service=scope_service,
        resolver=resolver,
        authority_context_state=authority_context_state,
    )
```

Tests:

```python
@pytest.mark.asyncio
async def test_persona_start_chat_binds_local_persona_session(monkeypatch):
    runtime = _persona_handoff_runtime()
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)

    started = await screen._session._start_persona_console_session(
        _persona_start_handoff()
    )

    assert started is True
    session = store.session
    assert session is not None
    assert session.runtime_backend == "local"
    assert session.assistant_kind == "persona"
    assert session.assistant_id == "local-persona-abc"
    # ADR-037: persona sessions never carry an authority id, local included.
    assert session.assistant_authority_id is None
    assert session.assistant_name == "Archivist"
    assert session.character_name is None
    assert session.character_ref() is None
    assert session.persona_system_template == "Guide {{user}} as {{persona}}."
    assert session.settings is not None
    # app_config={} -> global display name falls back to "User".
    assert session.settings.system_prompt == "Guide User as Archivist."
    # Personas have no greeting: seeding appends no message.
    assert store.messages == []
    runtime.scope_service.get_persona_profile.assert_awaited_once_with(
        "local-persona-abc", mode="local"
    )
    runtime.scope_service.get_character.assert_not_awaited()
    runtime.resolver.assert_not_awaited()
    runtime.db.get_local_authority_id.assert_not_called()


@pytest.mark.asyncio
async def test_persona_start_chat_fails_closed_when_profile_missing(monkeypatch):
    runtime = _persona_handoff_runtime(profile=None)
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)

    started = await screen._session._start_persona_console_session(
        _persona_start_handoff()
    )

    assert started is False
    assert store.session is None
    assert store.create_kwargs is None


@pytest.mark.asyncio
async def test_persona_start_chat_fails_closed_when_profile_fetch_raises(monkeypatch):
    runtime = _persona_handoff_runtime()
    runtime.scope_service.get_persona_profile.side_effect = RuntimeError("boom")
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)

    started = await screen._session._start_persona_console_session(
        _persona_start_handoff()
    )

    assert started is False
    assert store.session is None


@pytest.mark.asyncio
async def test_persona_start_chat_binds_server_persona_without_authority(monkeypatch):
    runtime = _persona_handoff_runtime(active_server_id="server-1")
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)

    started = await screen._session._start_persona_console_session(
        _persona_start_handoff(
            runtime_backend="server", active_server_profile_id="server-1"
        )
    )

    assert started is True
    session = store.session
    assert session.runtime_backend == "server"
    assert session.assistant_kind == "persona"
    assert session.assistant_authority_id is None
    runtime.resolver.assert_not_awaited()


@pytest.mark.asyncio
async def test_persona_start_chat_server_fence_flip_mid_fetch_fails_closed(
    monkeypatch,
):
    runtime = _persona_handoff_runtime(active_server_id="server-1")

    async def flip_then_return(*_args, **_kwargs):
        runtime.authority_context_state["current"] = SimpleNamespace(account="B")
        return _persona_profile()

    runtime.scope_service.get_persona_profile.side_effect = flip_then_return
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)

    started = await screen._session._start_persona_console_session(
        _persona_start_handoff(
            runtime_backend="server", active_server_profile_id="server-1"
        )
    )

    assert started is False
    assert store.session is None
    assert store.create_kwargs is None


@pytest.mark.asyncio
async def test_persona_payload_rejected_by_character_starter_accepted_by_persona_starter(
    monkeypatch,
):
    runtime = _persona_handoff_runtime()
    store = _CharacterHandoffStore()
    screen = _handoff_chat_screen(monkeypatch, runtime.app, store)
    payload = _persona_start_handoff()

    assert await screen._session._start_character_console_session(payload) is False
    assert await screen._session._start_persona_console_session(payload) is True
```

Append to `Tests/Chat/test_console_chat_store.py` (helpers `_pristine_defaults`/`_pristine_session` already exist at 33/37):

```python
def test_repurpose_pristine_session_accepts_persona_identity():
    defaults = _pristine_defaults()
    persona_settings = replace(
        defaults, system_prompt="Guide User as Archivist.", character_label=""
    )
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)

    updated = store.repurpose_pristine_session(
        session.id,
        canonical_settings=defaults,
        trusted_system_prompt="Guide User as Archivist.",
        title="Chat with Archivist",
        settings=persona_settings,
        runtime_backend="local",
        assistant_kind="persona",
        assistant_id="local-persona-abc",
        assistant_authority_id=None,
        character_id=None,
        character_name=None,
        assistant_name="Archivist",
    )

    assert updated is session
    assert updated.assistant_kind == "persona"
    assert updated.assistant_id == "local-persona-abc"
    assert updated.assistant_name == "Archivist"
    assert updated.character_id is None
    assert updated.character_name is None
    assert updated.assistant_authority_id is None
    assert updated.settings == persona_settings


def test_repurpose_pristine_session_rejects_persona_authority():
    defaults = _pristine_defaults()
    store = ConsoleChatStore()
    session = _pristine_session(store, defaults)
    before = replace(session)

    with pytest.raises(ValueError, match="authority-free"):
        store.repurpose_pristine_session(
            session.id,
            canonical_settings=defaults,
            trusted_system_prompt="Guide User.",
            title="Chat with Archivist",
            settings=replace(defaults, system_prompt="Guide User."),
            runtime_backend="server",
            assistant_kind="persona",
            assistant_id="opaque-persona",
            assistant_authority_id="server-user-v1:" + ("a" * 64),
            character_id=None,
            character_name=None,
            assistant_name="Archivist",
        )

    assert store.sessions()[0] is session
    assert session == before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_chat_store.py -v -k "persona_start_chat or persona_payload_rejected or repurpose_pristine_session_accepts_persona or repurpose_pristine_session_rejects_persona"`
Expected: FAIL — `AttributeError: 'ConsoleSessionController' object has no attribute '_start_persona_console_session'`; repurpose tests fail with `TypeError: unexpected keyword argument 'assistant_name'`.

- [ ] **Step 3: Extract the server-fence helper and rewire the character starter**

In `tldw_chatbook/UI/Console_Modules/session.py`, immediately before `_start_character_console_session`:

```python
    def _capture_server_handoff_context(
        self, payload: ChatHandoffPayload
    ) -> tuple[object, Callable[[], bool]] | None:
        """Capture and fence the server authority context for a handoff.

        Returns ``(capture, is_current)`` when the handoff's expected server
        is still active and its context capture is current, else ``None``.
        ``is_current()`` re-checks both conditions, so callers re-fence after
        every suspension point by calling it again.
        """
        expected_server_id = payload.active_server_profile_id
        if (
            type(expected_server_id) is not str
            or not expected_server_id
            or expected_server_id != expected_server_id.strip()
        ):
            return None
        if getattr(self.app_instance, "active_server_id", None) != expected_server_id:
            return None
        provider = getattr(self.app_instance, "server_context_provider", None)
        capture_context = getattr(
            provider, "capture_character_authority_context", None
        )
        context_is_current = getattr(
            provider, "is_character_authority_context_current", None
        )
        if not callable(capture_context) or not callable(context_is_current):
            return None
        try:
            capture = capture_context(expected_server_id=expected_server_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            return None

        def is_current() -> bool:
            if (
                getattr(self.app_instance, "active_server_id", None)
                != expected_server_id
            ):
                return False
            try:
                return context_is_current(capture) is True
            except Exception:
                return False

        if not is_current():
            return None
        return capture, is_current
```

In `_start_character_console_session`, delete the closure defs at 2551–2560 (`server_context_capture`/`server_context_is_current` predeclarations and `exact_server_context_is_current`) and replace the server `else` branch (2581–2648) with:

```python
        else:
            fenced = self._capture_server_handoff_context(payload)
            if fenced is None:
                return False
            server_context_capture, exact_server_context_is_current = fenced
            expected_server_id = payload.active_server_profile_id
            assistant_authority_id = None
            provider = getattr(self.app_instance, "server_context_provider", None)
            resolver = getattr(provider, "resolve_character_authority_id", None)
            if callable(resolver):
                try:
                    resolved_authority_id = await resolver(
                        expected_server_id=expected_server_id,
                        context_capture=server_context_capture,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    resolved_authority_id = None
                if (
                    type(resolved_authority_id) is str
                    and _SERVER_CHARACTER_AUTHORITY_PATTERN.fullmatch(
                        resolved_authority_id
                    )
                    is not None
                ):
                    assistant_authority_id = resolved_authority_id

            # This is both the post-resolver fence and the immediately
            # pre-card-fetch fence. No card from a newly active target may be
            # used for the ID carried by the handoff.
            if not exact_server_context_is_current():
                return False
            local_character_id = None
```

(`Callable` is already imported at line 124.) The helper's `is_current` now folds in the `active_server_id` check the old call sites did separately, so simplify the two later fences: at 2669–2676 and 2694–2699 replace the compound `not exact_server_context_is_current() or active_server_id != ...` with just `if not exact_server_context_is_current(): return False` (same conjunction). Keep the predeclarations `assistant_authority_id: str | None` / `local_character_id: int | None` at 2549–2550. The resolver call kwargs are unchanged, so the existing pin `resolve_captures == [initial_capture]` still holds.

- [ ] **Step 4: Implement the persona starter and the consumption branch**

In `session.py`, after `_start_character_console_session` (ends 2827):

```python
    async def _start_persona_console_session(
        self, payload: ChatHandoffPayload
    ) -> bool:
        """Build a dedicated persona-bound session from a Personas handoff.

        Mirrors ``_start_character_console_session`` minus greeting, local
        numeric projection, and authority resolution: persona sessions stay
        authority-free per ADR-037 (server personas get freshness fencing but
        never an ``assistant_authority_id``).
        """
        identity = _persona_session_identity_from_handoff(payload)
        if identity is None:
            return False
        runtime_backend, persona_id, name_hint, assistant_id = identity

        scope_service = getattr(
            self.app_instance, "character_persona_scope_service", None
        )
        get_persona_profile = getattr(scope_service, "get_persona_profile", None)
        if not callable(get_persona_profile):
            return False

        context_is_current: Callable[[], bool] | None = None
        if runtime_backend == "server":
            fenced = self._capture_server_handoff_context(payload)
            if fenced is None:
                return False
            _capture, context_is_current = fenced

        try:
            profile = await get_persona_profile(persona_id, mode=runtime_backend)
            if hasattr(profile, "model_dump"):
                profile = profile.model_dump(mode="json")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Start Chat: persona profile unavailable; staging context "
                "instead (source={}, backend={}).",
                payload.source,
                runtime_backend,
            )
            return False
        if not isinstance(profile, Mapping) or not profile:
            return False
        profile = dict(profile)
        if str(profile.get("id") or "") != persona_id:
            return False
        if context_is_current is not None and not context_is_current():
            return False

        global_name = _console_global_user_display_name(
            self._provider_readiness_app_config()
        )
        seed = _persona_session_prompt_seed(
            profile, name_hint=str(name_hint or ""), user_name=global_name
        )

        store = self._ensure_console_chat_store()
        canonical_defaults = self._default_console_session_settings()
        settings = replace(
            canonical_defaults,
            system_prompt=seed.system_prompt or None,
            character_label="",
        )
        if context_is_current is not None and not context_is_current():
            return False
        active = next(
            (
                candidate
                for candidate in store.sessions()
                if candidate.id == store.active_session_id
            ),
            None,
        )
        if (
            active is not None
            and active.settings is not None
            and active.settings != canonical_defaults
            and active.canonical_settings_baseline == active.settings
        ):
            try:
                active = store.refresh_pristine_session_settings(
                    active.id,
                    prior_canonical_settings=active.settings,
                    current_canonical_settings=canonical_defaults,
                )
            except ValueError:
                pass
        active_messages = (
            store.messages_for_session(active.id) if active is not None else []
        )
        duplicate_handoff = bool(
            active is not None
            and active.settings == settings
            and active.runtime_backend == runtime_backend
            and active.assistant_kind == "persona"
            and active.assistant_id == assistant_id
            and active.assistant_name == seed.name
            and active.persona_system_template == seed.system_template
            and not active_messages
        )
        if duplicate_handoff:
            session = active
        else:
            session = None
            if active is not None and store.is_pristine_session(
                active.id,
                expected_settings=canonical_defaults,
            ):
                try:
                    session = store.repurpose_pristine_session(
                        active.id,
                        canonical_settings=canonical_defaults,
                        trusted_system_prompt=seed.system_prompt,
                        title=f"Chat with {seed.name}",
                        settings=settings,
                        runtime_backend=runtime_backend,
                        assistant_kind="persona",
                        assistant_id=assistant_id,
                        assistant_authority_id=None,
                        character_id=None,
                        character_name=None,
                        assistant_name=seed.name,
                    )
                except ValueError:
                    session = None
            if session is None:
                session = store.create_session(
                    title=f"Chat with {seed.name}",
                    workspace_id=CONSOLE_GLOBAL_WORKSPACE_ID,
                    settings=settings,
                    runtime_backend=runtime_backend,
                    assistant_kind="persona",
                    assistant_id=assistant_id,
                    assistant_authority_id=None,
                    assistant_name=seed.name,
                )
            try:
                store.seed_persona_roleplay(
                    session.id,
                    system_template=seed.system_template,
                    global_default=global_name,
                )
            except Exception as exc:
                logger.warning(
                    "Start Chat: persona roleplay template seed/persist failed; "
                    "continuing (error_type={}).",
                    type(exc).__name__,
                )
        store.switch_session(session.id)
        if not duplicate_handoff:
            # Same defensive cleanup as the server-character path: a persona
            # session never keys reactions by actor, so clear wholesale.
            self._clear_session_manual_reactions(session.id)
        try:
            await self._sync_native_console_chat_ui()
            self._focus_console_composer_if_needed(force=True)
        except asyncio.CancelledError:
            # The durable commit boundary is above. Report success so the
            # caller acknowledges this handoff instead of replaying it.
            return True
        except Exception:
            # The persona session is already durably created above -- a
            # UI-sync/focus failure here must not propagate, or the caller
            # would never clear ``pending_chat_handoff`` and a later
            # re-consume would build a SECOND durable persona session.
            logger.opt(exception=True).warning(
                "Start Chat: post-seed console sync/focus failed; persona "
                "session was already created and the handoff is still "
                "considered consumed."
            )
        return True
```

(`Mapping` is already imported at line 124; `_console_global_user_display_name`, `CONSOLE_GLOBAL_WORKSPACE_ID`, and `replace` are already used by the character starter in this module.)

In `chat_screen.py` `_consume_pending_chat_handoff` (11253–11293), replace the comment block and staging call at 11267–11277:

```python
            # The native Console composes no legacy tab surface. Personas
            # Start-Chat handoffs get a dedicated identity-bound session:
            # character cards seed a greeting (task-427); persona cards bind
            # name + system template without one (task-32481). Anything else
            # -- or an identity session that failed to build -- stages into
            # the Console live-work lane so the context lands in Staged
            # Context instead of being dropped with a warning.
            if await self._session._start_character_console_session(payload):
                store.acknowledge(claim)
                return
            if await self._session._start_persona_console_session(payload):
                store.acknowledge(claim)
                return
            self._stage_handoff_as_console_live_work(payload)
            store.acknowledge(claim)
```

- [ ] **Step 5: Add the persona branch to `repurpose_pristine_session`**

In `console_chat_store.py` (1015–1096): change the signature to `character_name: str | None` plus `assistant_name: str | None = None`; move the `trusted_system_prompt` non-empty check (1035–1036) into the character branch; widen the kind gate (1039–1040). The validation block becomes:

```python
        if type(trusted_system_prompt) is not str:
            raise TypeError("Trusted roleplay system prompt must be text.")
        if runtime_backend not in {"local", "server"}:
            raise ValueError("Roleplay runtime backend must be local or server.")
        if assistant_kind not in {"character", "persona"}:
            raise ValueError("Repurposed sessions require named identity.")
        if type(assistant_id) is not str or not assistant_id:
            raise ValueError("Roleplay assistant id must be non-empty text.")
        if assistant_authority_id is not None and (
            type(assistant_authority_id) is not str or not assistant_authority_id
        ):
            raise ValueError("Roleplay authority id must be non-empty text or None.")
        if assistant_kind == "character":
            if not trusted_system_prompt.strip():
                raise ValueError("Trusted roleplay system prompt must be non-empty text.")
            if type(character_name) is not str or not character_name.strip():
                raise ValueError("Roleplay character name must be non-empty text.")
            if title != f"Chat with {character_name}":
                raise ValueError("Roleplay title does not match the character identity.")
            expected_roleplay_settings = replace(
                canonical_settings,
                system_prompt=trusted_system_prompt,
                character_label=character_name,
            )
        else:
            # Persona sessions remain authority-free (ADR-037) and never
            # carry character identity. A blank template keeps the canonical
            # default prompt (``None``), matching the persona seed contract.
            if assistant_authority_id is not None:
                raise ValueError("Persona sessions remain authority-free (ADR-037).")
            if character_id is not None or character_name is not None:
                raise ValueError("Persona sessions cannot carry character identity.")
            if type(assistant_name) is not str or not assistant_name.strip():
                raise ValueError("Persona name must be non-empty text.")
            if title != f"Chat with {assistant_name}":
                raise ValueError("Persona title does not match the persona identity.")
            expected_roleplay_settings = replace(
                canonical_settings,
                system_prompt=trusted_system_prompt or None,
                character_label="",
            )
        if settings != expected_roleplay_settings:
            raise ValueError("Roleplay settings contain noncanonical changes.")
        if runtime_backend == "local":
            if assistant_kind == "character" and (
                type(character_id) is not int
                or character_id < 1
                or assistant_id != str(character_id)
            ):
                raise ValueError("Local roleplay identity is inconsistent.")
        elif character_id is not None:
            raise ValueError("Server roleplay identity cannot carry a local id.")
```

At the end of the field assignments (after 1092), add:

```python
        session.assistant_name = assistant_name if assistant_kind == "persona" else None
```

(This also keeps the one-name invariant on the character path: repurposing a tab as a character clears any stale `assistant_name`.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_chat_store.py -v -k "persona_start_chat or persona_payload_rejected or repurpose_pristine_session"`
Expected: PASS, including the pre-existing repurpose tests at 582/651/693/740/778.

Regression for the fencing refactor (existing character server tests):

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "handoff"`
Expected: PASS — the pre-existing `test_persona_start_chat_does_not_create_character_session` (5091) stays green (it calls the character starter directly).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Chat/console_chat_store.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_chat_store.py
git commit -m "feat: persona console session starter + repurpose persona branch (task-32481)"
```

---

### Task 6: Resume persona branch + wiring + hydration

**Files:**
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` — new `_resolve_resumed_persona_name` after `_resolve_resumed_character_name` (7148–7171)
- Modify: `tldw_chatbook/UI/Console_Modules/workspace.py` — ctor param (~164), stored fn (~236), property accessor (after 474), persona overlay branch in `_resume_console_workspace_conversation` (between 2468 and 2469)
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py` — beside `resolve_resumed_character_name` (299–301)
- Modify: `tldw_chatbook/Chat/console_conversation_hydration.py` — after line 438
- Test: `Tests/UI/test_console_native_chat_flow.py`

**Interfaces:**
- Consumes: Task 2 envelope key `persona_system_template`; Task 5 bound persona sessions.
- Produces: `ChatScreen._resolve_resumed_persona_name(persona_id, runtime_backend) -> str`; `ConsoleWorkspaceController` ctor kwarg `resolve_resumed_persona_name` + `_resolve_resumed_persona_name` property; persona overlay branch on resume; hydration restoring `session.persona_system_template`. Task 7's send path relies on both restored fields.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_native_chat_flow.py`, mirroring `test_console_resume_rehydrates_local_character_name_from_local_projection` (9096–9137):

```python
@pytest.mark.asyncio
async def test_console_resume_rehydrates_persona_name_from_profile():
    """A resumed persona session re-resolves its display name, both backends."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "local-persona": {
                "conversation": {
                    "id": "local-persona",
                    "title": "Chat with Archivist",
                    "runtime_backend": "local",
                    "assistant_kind": "persona",
                    "assistant_id": "local-persona-abc",
                    # No assistant_authority_id: personas are authority-free.
                    # No character_id: personas carry no local projection.
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        name_lookup = AsyncMock(return_value="Archivist")
        console._resolve_resumed_persona_name = name_lookup

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "local-persona"
            )
            is True
        )

        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        assert session.runtime_backend == "local"
        assert session.assistant_kind == "persona"
        assert session.assistant_id == "local-persona-abc"
        assert session.assistant_authority_id is None
        assert session.assistant_name == "Archivist"
        assert session.character_name is None
        assert session.character_ref() is None
        name_lookup.assert_awaited_once_with("local-persona-abc", "local")


@pytest.mark.asyncio
async def test_console_resume_persona_name_lookup_failure_stays_unlabeled():
    """A failed profile lookup resumes the session without a persona name."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "local-persona": {
                "conversation": {
                    "id": "local-persona",
                    "title": "Chat with Archivist",
                    "runtime_backend": "local",
                    "assistant_kind": "persona",
                    "assistant_id": "local-persona-abc",
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        console._resolve_resumed_persona_name = AsyncMock(return_value="")

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "local-persona"
            )
            is True
        )

        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        assert session.assistant_kind == "persona"
        assert session.assistant_name is None


@pytest.mark.asyncio
async def test_console_resume_restores_persona_system_template_from_metadata():
    """The trusted persona template rides the roleplay metadata envelope."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    app.chat_conversation_scope_service = StaticConversationTreeService(
        {
            "local-persona": {
                "conversation": {
                    "id": "local-persona",
                    "title": "Chat with Archivist",
                    "runtime_backend": "local",
                    "assistant_kind": "persona",
                    "assistant_id": "local-persona-abc",
                    "metadata": {
                        "console_roleplay_context": {
                            "version": 1,
                            "persona_system_template": "Guide {{user}}.",
                        },
                    },
                },
                "root_threads": [],
            }
        }
    )
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-transcript")
        console._resolve_resumed_persona_name = AsyncMock(return_value="Archivist")

        assert (
            await console._workspace._resume_console_workspace_conversation(
                "local-persona"
            )
            is True
        )

        store = console._ensure_console_chat_store()
        session = store.switch_session(store.active_session_id)
        assert session.persona_system_template == "Guide {{user}}."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "resume_rehydrates_persona or resume_persona_name_lookup or restores_persona_system_template"`
Expected: FAIL — `assistant_name is None` assertions fail (no persona overlay branch); the metadata test fails on `persona_system_template is None` (hydration does not restore it). The mock on `console._resolve_resumed_persona_name` is never awaited.

- [ ] **Step 3: Implement the screen resolver**

In `chat_screen.py`, immediately after `_resolve_resumed_character_name` (ends 7171):

```python
    async def _resolve_resumed_persona_name(
        self, persona_id: str, runtime_backend: str
    ) -> str:
        """Return a resumed persona's display name from its profile, or ``""``.

        Best-effort mirror of ``_resolve_resumed_character_name``: any
        failure (missing scope service, missing profile, fetch error)
        returns an empty string and the caller keeps the session unlabeled.
        """
        scope_service = getattr(
            self.app_instance, "character_persona_scope_service", None
        )
        get_persona_profile = getattr(scope_service, "get_persona_profile", None)
        if not callable(get_persona_profile):
            return ""
        try:
            profile = await get_persona_profile(persona_id, mode=runtime_backend)
            if hasattr(profile, "model_dump"):
                profile = profile.model_dump(mode="json")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.opt(exception=True).warning(
                "Resume: persona profile fetch failed; identity row falls back."
            )
            return ""
        if not isinstance(profile, Mapping):
            return ""
        return str(profile.get("name") or "").strip()
```

(`Mapping` is already imported in `chat_screen.py` at line 3 — no import change needed.)

- [ ] **Step 4: Implement the workspace overlay branch, ctor kwarg, and wiring**

In `workspace.py`'s `_resume_console_workspace_conversation`, insert between the character branch (ends 2468) and the trailing `elif session.settings is not None:` (2469):

```python
        elif session.assistant_kind == "persona" and session.assistant_id:
            # Persona sessions carry no local projection; the display name is
            # re-resolved from the live profile on every resume (ADR-149),
            # for local and server backends alike.
            persona_name = await self._resolve_resumed_persona_name(
                session.assistant_id, session.runtime_backend
            )
            session.assistant_name = persona_name or None
            if persona_name:
                # One-name invariant: a persona session never shows a stale
                # character label.
                session.character_name = None
            if session.settings is not None:
                session.settings = replace(session.settings, character_label="")
```

In the same file's constructor, add after the `wake_retry_poke` parameter (164):

```python
        resolve_resumed_persona_name: Callable[[str, str], Any] | None = None,
```

Store it beside line 236 (`self._resolve_resumed_character_name_fn = ...`):

```python
        self._resolve_resumed_persona_name_fn = resolve_resumed_persona_name
```

Add the accessor beside the `_resolve_resumed_character_name` property (473–475):

```python
    @property
    def _resolve_resumed_persona_name(self) -> Any:
        resolver = self._resolve_resumed_persona_name_fn
        if resolver is None:
            # Bare-controller/test construction: best-effort resume simply
            # leaves the session unlabeled.
            async def _unresolved(_persona_id: str, _runtime_backend: str) -> str:
                return ""

            return _unresolved
        return resolver
```

In `wiring.py`, beside `resolve_resumed_character_name` (299–301):

```python
        resolve_resumed_persona_name=(
            lambda persona_id, runtime_backend: screen._resolve_resumed_persona_name(
                persona_id, runtime_backend
            )
        ),
```

In `console_conversation_hydration.py`, after line 438 (`session.character_system_template = ...`):

```python
    session.persona_system_template = roleplay_context.persona_system_template
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "resume"`
Expected: PASS, including the pre-existing character resume tests (9096+).

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/UI/Console_Modules/workspace.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/Chat/console_conversation_hydration.py Tests/UI/test_console_native_chat_flow.py
git commit -m "feat: persona name re-resolution + template restore on resume (task-32481)"
```

---

### Task 7: Screen-state serialization + send-path persona branch

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/session.py` — `_console_session_to_state` (3084–3125), `_console_session_from_state` (3127–3273)
- Modify: `tldw_chatbook/Chat/console_chat_controller.py` — `_resolved_system_prompt` (12770–12796)
- Test: `Tests/UI/test_console_native_chat_flow.py`, `Tests/Chat/test_console_provider_continuation.py`

**Interfaces:**
- Consumes: Task 1 session fields; Task 5/6 bound-and-restored persona identity.
- Produces: `assistant_name` / `persona_system_template` keys in serialized screen state; kind-keyed one-name invariant on restore; persona branch in `_resolved_system_prompt`. Completes the feature.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_native_chat_flow.py`, mirroring `test_native_console_state_round_trip_preserves_source_aware_character_identity` (10323–10375):

```python
def test_native_console_state_round_trip_preserves_persona_identity():
    """Screen state keeps persona provenance and its trusted template."""
    store = ConsoleChatStore()
    session = ConsoleChatSession(
        id="session-p",
        title="Chat with Archivist",
        runtime_backend="local",
        assistant_kind="persona",
        assistant_id="local-persona-abc",
        assistant_name="Archivist",
        persona_system_template="Guide {{user}} as {{persona}}.",
    )
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen(store)

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    assert {
        key: payload["sessions"][0][key]
        for key in (
            "runtime_backend",
            "assistant_kind",
            "assistant_id",
            "assistant_name",
            "persona_system_template",
        )
    } == {
        "runtime_backend": "local",
        "assistant_kind": "persona",
        "assistant_id": "local-persona-abc",
        "assistant_name": "Archivist",
        "persona_system_template": "Guide {{user}} as {{persona}}.",
    }

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)
    _restore_console_snapshot_with_sessions(restored_screen, payload)

    restored_session = restored_store.sessions()[0]
    assert restored_session.runtime_backend == "local"
    assert restored_session.assistant_kind == "persona"
    assert restored_session.assistant_id == "local-persona-abc"
    assert restored_session.assistant_name == "Archivist"
    assert (
        restored_session.persona_system_template
        == "Guide {{user}} as {{persona}}."
    )
    assert restored_session.character_name is None
    assert restored_session.character_ref() is None


def test_native_console_state_restore_drops_stray_character_name_on_persona():
    """A contradictory payload keeps only the kind-appropriate name."""
    store = ConsoleChatStore()
    session = ConsoleChatSession(
        id="session-p",
        title="Chat with Archivist",
        runtime_backend="local",
        assistant_kind="persona",
        assistant_id="local-persona-abc",
        assistant_name="Archivist",
    )
    store.restore_state(
        sessions=[session],
        messages_by_session={session.id: []},
        active_session_id=session.id,
    )
    screen = _bare_console_screen(store)

    payload = _console_snapshot_with_sessions(screen)
    assert payload is not None
    payload["sessions"][0]["character_name"] = "Stale Character"

    restored_store = ConsoleChatStore()
    restored_screen = _bare_console_screen(restored_store)
    _restore_console_snapshot_with_sessions(restored_screen, payload)

    restored_session = restored_store.sessions()[0]
    assert restored_session.assistant_name == "Archivist"
    assert restored_session.character_name is None
```

Append to `Tests/Chat/test_console_provider_continuation.py` (add `from tldw_chatbook.Chat.console_session_settings import ConsoleSessionSettings` to the imports — not currently present; `ConsoleChatStore` and `ConsoleChatController` are already imported at 19–27):

```python
def test_resolved_system_prompt_re_expands_persona_template() -> None:
    """The send path re-expands the trusted persona template per turn."""
    store = ConsoleChatStore()
    session = store.create_session(
        title="Chat with Archivist",
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-3-haiku"
        ),
        assistant_kind="persona",
        assistant_id="local-persona-abc",
        assistant_name="Archivist",
    )
    store.seed_persona_roleplay(
        session.id,
        system_template="Guide {{user}} as {{persona}}.",
        global_default="Rowan",
    )
    controller = ConsoleChatController(store=store, provider_gateway=object())

    # The bare controller is constructed without a `global_user_display_name`
    # accessor, so the constructor default (`lambda: "User"`, line 1760)
    # applies and the per-turn re-expansion uses "User" — not the seed-time
    # "Rowan". This pins that send-time expansion always wins.
    assert (
        controller._resolved_system_prompt(session.id)
        == "Guide User as Archivist."
    )


def test_resolved_system_prompt_leaves_generic_sessions_untouched() -> None:
    """Non-identity sessions keep the controller's configured prompt."""
    store = ConsoleChatStore()
    session = store.create_session(
        title="Chat",
        settings=ConsoleSessionSettings(
            provider="anthropic", model="claude-3-haiku"
        ),
    )
    controller = ConsoleChatController(
        store=store, provider_gateway=object(), system_prompt="Base prompt."
    )

    assert controller._resolved_system_prompt(session.id) == "Base prompt."
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "round_trip_preserves_persona or stray_character_name"` and `pytest Tests/Chat/test_console_provider_continuation.py -v -k "resolved_system_prompt"`
Expected: FAIL — serialized payload lacks `assistant_name`/`persona_system_template`; `_resolved_system_prompt` returns `None` for the persona session (character-only gate falls through to `self.system_prompt`, which defaults to `None`).

- [ ] **Step 3: Serialize and restore the persona fields with the one-name invariant**

In `_console_session_to_state` (session.py 3091–3125), add `"assistant_name": session.assistant_name` immediately after the `"character_name"` line (3113) and `"persona_system_template": session.persona_system_template` immediately after the `"character_system_template"` line (3115). Explicit field list — nothing else changes.

In `_console_session_from_state` (3127–3273), after the `character_name` block (3238–3240):

```python
        raw_assistant_name = raw_session.get("assistant_name")
        if raw_assistant_name is not None:
            session_kwargs["assistant_name"] = str(raw_assistant_name)
```

After the `character_system_template` block (3249–3254):

```python
        raw_persona_system_template = raw_session.get("persona_system_template")
        session_kwargs["persona_system_template"] = (
            raw_persona_system_template
            if isinstance(raw_persona_system_template, str)
            else None
        )
```

Immediately before `return ConsoleChatSession(**session_kwargs)` (3273):

```python
        # One-name invariant (ADR-149): a session shows exactly one identity
        # label, keyed by kind. A contradictory payload keeps only the
        # kind-appropriate field. Legacy payloads predate `assistant_name`,
        # so the else-branch pop is a no-op for them.
        if session_kwargs.get("assistant_kind") == "persona":
            session_kwargs.pop("character_name", None)
        else:
            session_kwargs.pop("assistant_name", None)
```

- [ ] **Step 4: Add the persona branch to `_resolved_system_prompt`**

In `console_chat_controller.py`, replace the body of `_resolved_system_prompt` (12770–12796) with a kind-keyed pick (behavior on the character path is unchanged):

```python
    def _resolved_system_prompt(self, session_id: str | None) -> str | None:
        """Resolve the trusted identity system template for the session kind."""
        if session_id is None:
            return self.system_prompt
        session = next(
            (
                candidate
                for candidate in self.store.sessions()
                if candidate.id == session_id
            ),
            None,
        )
        if session is None:
            return self.system_prompt
        if session.assistant_kind == "character":
            name = session.character_name
            template = session.character_system_template
        elif session.assistant_kind == "persona":
            name = session.assistant_name
            template = session.persona_system_template
        else:
            return self.system_prompt
        if (
            not isinstance(name, str)
            or not name.strip()
            or not isinstance(template, str)
            or not template.strip()
        ):
            return self.system_prompt
        context = self._presentation_context_for(session_id)
        return expand_character_template(
            template,
            user_name=context.user_name,
            character_name=name.strip(),
        )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest Tests/UI/test_console_native_chat_flow.py -v -k "round_trip or stray_character_name or state_restore"` and `pytest Tests/Chat/test_console_provider_continuation.py -v`
Expected: PASS — new persona tests plus the pre-existing state round trips (10323, 10406) and the whole provider-continuation file (character `_resolved_system_prompt` behavior preserved; no pre-existing direct tests of it — these are the first).

- [ ] **Step 6: Full feature sweep + task hygiene**

Run the complete set of files this plan touches:

Run: `pytest Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_chat_store.py Tests/Chat/test_console_provider_continuation.py Tests/Chat/test_console_conversation_hydration.py Tests/Chat/test_console_roleplay_metadata.py -q`
Expected: PASS.

Then update the backlog task by hand (the `backlog` CLI is not installed here — edit `backlog/tasks/task-32481 - Console-persona-session-identity-label-parity-and-dynamic-identity-macros.md` directly): mark every acceptance criterion `- [x]`, add the `## Implementation Notes` section (approach: persona branch on the ADR-046 identity machinery; authority-free per ADR-037; name re-resolved at resume; template in the version-1 envelope), and set `status: Done` in the frontmatter.

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/UI/Console_Modules/session.py tldw_chatbook/Chat/console_chat_controller.py Tests/UI/test_console_native_chat_flow.py Tests/Chat/test_console_provider_continuation.py "backlog/tasks/task-32481 - Console-persona-session-identity-label-parity-and-dynamic-identity-macros.md"
git commit -m "feat: persona identity in console screen state + send path (task-32481)"
```

---

## Self-Review Notes

Checks run against the approved spec before commit:

- **Spec coverage:** §1 identity model → Task 1; gate hygiene (`_is_named_persona_session` / `_is_named_identity_session`) and swap hardening → Task 3; §2 handoff routing/creation → Tasks 4–5 (with the fenced-fetch deviation flagged in Global Constraints); §3 expansion at seed/send/identity-change → Tasks 3, 5, 7; `expand_character_template` docstring line → Task 1; §4 presentation → Task 1 (chip needs no change — verified `_build_console_control_state` already passes `assistant_name`); §5 persistence/resume → Tasks 2, 6; §6 error handling → Tasks 5–7; §7 testing matrix → per-task test steps. Resume server-mode parameter pinning is covered by the local assertion pattern only, matching the character suite's local-only resume coverage.
- **Placeholder scan:** no TBD/TODO; every line reference and signature was verified against the working tree on 2026-09-11.
- **Type/name consistency:** `assistant_name`, `persona_system_template`, `session_assistant_display_name`, `_is_named_identity_session`, `seed_persona_roleplay(session_id, *, system_template, global_default)`, `set_session_assistant_name`, `_persona_session_identity_from_handoff` (4-tuple), `PersonaSessionPromptSeed`, `_persona_session_prompt_seed(profile, name_hint=..., *, user_name=...)`, `_capture_server_handoff_context`, `_start_persona_console_session`, `_resolve_resumed_persona_name(persona_id, runtime_backend)`, `get_persona_profile(persona_id, mode=...)`, and `ConsoleRoleplayContext.persona_system_template` are spelled identically at every production and consumption site.

