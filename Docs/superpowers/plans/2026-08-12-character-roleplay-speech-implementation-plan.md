# Character Roleplay Speech Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the character-card-to-roleplay journey with clear import behavior, correct chat handoff, persistent opt-in reply speech, visible playback actions, and lossless paste blocks.

**Architecture:** Keep character import display sanitization separate from stored card data. Reuse an untouched initial Console session through a pure store predicate. Add conversation speech preferences to existing conversation metadata, drive automatic speech from one completion observer and pure eligibility decision, and dispatch through the existing trusted speech snapshot/event path.

**Tech Stack:** Python 3.11+, Textual, pytest, pytest-asyncio, existing Console chat store/controller, TTS events and playback sequencer, ChaChaNotes conversation metadata.

## Global Constraints

- Character file picker starts at remembered character-import directory, then `~/Documents`, then home; never process working directory.
- Display sanitization never mutates stored character-card data.
- An untouched initial `Chat 1` is reused; a worked-on or non-default session is preserved.
- Automatic reply speech is opt-in per conversation and starts off.
- Consent is destination-aware and contains no credential or message text.
- Hands-free owns reply speech while active.
- Only completed assistant/character responses are eligible.
- Background conversations never auto-play or replay later.
- One automatic speech failure pauses persistently; no silent retry.
- Manual Speak/Stop stays visible in each eligible message header.
- Adjacent collapsed paste blocks submit with one newline and never concatenate.

---

## File Structure

- Modify `tldw_chatbook/Widgets/enhanced_file_picker.py`: context-aware initial directory and selected-row visuals.
- Create `tldw_chatbook/UI/character_display_text.py`: bounded display-only terminal text sanitizer.
- Modify character/persona widgets to consume sanitized display projections.
- Modify `tldw_chatbook/Chat/console_chat_store.py`: pristine-session reuse and completion subscriptions.
- Modify `tldw_chatbook/UI/Console_Modules/session.py`: character handoff repurposing.
- Create `tldw_chatbook/Chat/console_speech_preferences.py`: metadata serialization and merge contract.
- Create `tldw_chatbook/Chat/console_auto_speak.py`: pure eligibility, consent, and pause decisions.
- Create `tldw_chatbook/Widgets/Console/console_auto_speak_consent.py`: explicit destination confirmation modal.
- Modify Console control/message/transcript modules for toggle, visible Speak/Stop, lifecycle state, and dispatch.
- Modify `tldw_chatbook/Widgets/Console/console_composer_bar.py`: paste block labels and separators.

### Task 1: Correct the character-import picker start and selection state

**Files:**
- Modify: `tldw_chatbook/Widgets/enhanced_file_picker.py:1070-1160`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py:940-990`
- Modify: `tests/UI/test_file_picker_start_dir.py`
- Modify: `tests/UI/test_file_picker_action_tooltips.py`
- Modify: `tests/UI/test_uat_first_time_character_chat.py`

**Interfaces:**
- Produces: `resolve_file_picker_start(context: str, remembered: Path | None, *, home: Path) -> Path`.
- Preserves: `EnhancedFileOpen(..., context="character_import")` and per-context remembered directory.
- Produces distinct row classes for focused and selected paths.

- [ ] **Step 1: Write start-precedence and selected-row tests**

```python
def test_character_import_start_precedence(tmp_path):
    home = tmp_path / "home"
    documents = home / "Documents"
    remembered = home / "cards"
    documents.mkdir(parents=True)
    remembered.mkdir()
    assert resolve_file_picker_start("character_import", remembered, home=home) == remembered
    assert resolve_file_picker_start("character_import", None, home=home) == documents


async def test_character_picker_selected_row_differs_from_focus():
    async with _character_picker_app().run_test() as pilot:
        await _focus_first_file(pilot)
        await pilot.press("space")
        row = _first_file_row()
        assert row.has_class("-focused")
        assert row.has_class("-selected")
```

- [ ] **Step 2: Run picker tests and observe the `.` fallback**

Run: `.venv/bin/python -m pytest tests/UI/test_file_picker_start_dir.py tests/UI/test_file_picker_action_tooltips.py tests/UI/test_uat_first_time_character_chat.py -k "character_import or selected_row" -v`

Expected: FAIL because no remembered directory currently falls back to process working directory.

- [ ] **Step 3: Implement context-aware start resolution**

```python
def resolve_file_picker_start(context, remembered, *, home):
    if remembered is not None and remembered.is_dir():
        return remembered
    documents = home / "Documents"
    if context == "character_import" and documents.is_dir():
        return documents
    return home
```

Resolve this before the base picker composes. Keep explicit caller `location` authoritative when it is not the default `.`. Add stable CSS for focus, selection, and combined focus+selection to `EnhancedDirectoryNavigation.DEFAULT_CSS` in `enhanced_file_picker.py`; do not encode selection only by color.

- [ ] **Step 4: Run picker and UAT import tests**

Run: `.venv/bin/python -m pytest tests/UI/test_file_picker_start_dir.py tests/UI/test_file_picker_action_tooltips.py tests/UI/test_uat_first_time_character_chat.py -k "picker or import" -v`

Expected: PASS; successful selection saves only its parent directory for `character_import`.

- [ ] **Step 5: Commit picker corrections**

```bash
git add tldw_chatbook/Widgets/enhanced_file_picker.py tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py tests/UI/test_file_picker_start_dir.py tests/UI/test_file_picker_action_tooltips.py tests/UI/test_uat_first_time_character_chat.py
git commit -m "fix: start character imports in a useful directory"
```

### Task 2: Sanitize character text for terminal display only

**Files:**
- Create: `tldw_chatbook/UI/character_display_text.py`
- Modify: `tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py:55-85,460-590`
- Modify: `tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py`
- Modify: `tldw_chatbook/Widgets/Console/console_character_picker_modal.py`
- Create: `tests/UI/test_character_display_text.py`
- Modify: `tests/UI/test_personas_character_widgets.py`

**Interfaces:**
- Produces: `sanitize_character_display_text(value: object, *, max_characters: int) -> str`.
- Preserves: original database/card mappings unchanged.
- Replaces: decoding replacement characters, Cc/Cf/Cs controls other than newline/tab, and characters with `wcwidth(character) < 0` by `?`.

- [ ] **Step 1: Write display-only sanitizer tests**

```python
def test_character_display_sanitizer_replaces_invalid_terminal_sequences():
    raw = "A\ufffdB\x00C\ud800D"
    assert sanitize_character_display_text(raw, max_characters=20) == "A?B?C?D"


def test_character_display_sanitizer_does_not_mutate_card():
    card = {"name": "Name\ufffd", "description": "Original\x00value"}
    shown = sanitize_character_display_text(card["description"], max_characters=200)
    assert shown == "Original?value"
    assert card["description"] == "Original\x00value"
```

- [ ] **Step 2: Run sanitizer/widget tests**

Run: `.venv/bin/python -m pytest tests/UI/test_character_display_text.py tests/UI/test_personas_character_widgets.py -v`

Expected: FAIL because the display projection does not exist.

- [ ] **Step 3: Implement one bounded display projection**

```python
def sanitize_character_display_text(value, *, max_characters):
    text = str(value or "")[:max_characters]
    result: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        invalid = character == "\ufffd" or category in {"Cs"} or (
            category in {"Cc", "Cf"} and character not in {"\n", "\t"}
        ) or wcwidth(character) < 0
        result.append("?" if invalid else character)
    return "".join(result)
```

Apply only immediately before `Static.update`, `TextArea.text`, option labels, and transcript labels. Do not sanitize card values before import, export, persistence, prompt construction, or TTS request resolution.

- [ ] **Step 4: Run character import/display/prompt tests**

Run: `.venv/bin/python -m pytest tests/UI/test_character_display_text.py tests/UI/test_personas_character_widgets.py tests/Character_Chat/test_character_card_lenient_import.py tests/Character_Chat/test_compose_character_card_text.py -v`

Expected: PASS; prompt tests prove stored text remains original.

- [ ] **Step 5: Commit display sanitization**

```bash
git add tldw_chatbook/UI/character_display_text.py tldw_chatbook/UI/CCP_Modules/ccp_character_handler.py tldw_chatbook/Widgets/Persona_Widgets/personas_character_card_widget.py tldw_chatbook/Widgets/Console/console_character_picker_modal.py tests/UI/test_character_display_text.py tests/UI/test_personas_character_widgets.py
git commit -m "fix: sanitize character text only at display boundaries"
```

### Task 3: Reuse only the untouched initial Console session

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:420-520,640-810`
- Modify: `tldw_chatbook/UI/Console_Modules/session.py:1542-1765`
- Modify: `tests/Chat/test_console_chat_store.py`
- Modify: `tests/UI/test_console_session_controller.py`
- Modify: `tests/UI/test_uat_first_time_character_chat.py`

**Interfaces:**
- Produces: `ConsoleChatStore.is_pristine_session(session_id: str, *, expected_settings: ConsoleSessionSettings) -> bool`.
- Produces: `ConsoleChatStore.repurpose_pristine_session(session_id: str, **identity) -> ConsoleChatSession`.
- Consumes: canonical `build_default_console_session_settings` from Plan 2.

- [ ] **Step 1: Write pristine-session and handoff tests**

```python
def test_initial_chat_one_is_pristine(store, defaults):
    session = store.ensure_session(title="Chat 1", settings=defaults)
    assert store.is_pristine_session(session.id, expected_settings=defaults)
    store.set_session_draft(session.id, "typed work")
    assert not store.is_pristine_session(session.id, expected_settings=defaults)


async def test_character_handoff_reuses_untouched_chat_one(screen, character_handoff):
    original_id = screen.store.active_session.id
    assert await screen._session._start_character_console_session(character_handoff)
    assert screen.store.active_session.id == original_id
    assert screen.store.active_session.title == "Chat with Alba"
    assert len(screen.store.sessions) == 1
```

- [ ] **Step 2: Run store/session handoff tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_chat_store.py tests/UI/test_console_session_controller.py tests/UI/test_uat_first_time_character_chat.py -k "pristine or untouched or character_handoff" -v`

Expected: FAIL because the current handoff always creates another session.

- [ ] **Step 3: Implement strict repurposing before greeting seed**

`is_pristine_session` requires default title, no persisted conversation, no messages/tree nodes, blank draft, no attachments/prefill/RAG scope/context overrides, generic assistant identity, no character fields, non-ephemeral state, and exact expected default settings. `repurpose_pristine_session` revalidates the predicate, then updates title/settings/identity atomically in memory. The character handoff repurposes the active session when eligible; otherwise it creates a new one. Both paths seed the same greeting and focus the same composer.

- [ ] **Step 4: Run chat store, roleplay, and handoff tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_chat_store.py tests/Chat/test_console_session_settings.py tests/UI/test_console_session_controller.py tests/UI/test_uat_first_time_character_chat.py -v`

Expected: PASS; worked-on tabs remain untouched and active model comes from canonical defaults.

- [ ] **Step 5: Commit session reuse**

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/UI/Console_Modules/session.py tests/Chat/test_console_chat_store.py tests/UI/test_console_session_controller.py tests/UI/test_uat_first_time_character_chat.py
git commit -m "fix: reuse an untouched Console session for roleplay"
```

### Task 4: Persist per-conversation auto-speak and destination consent

**Files:**
- Create: `tldw_chatbook/Chat/console_speech_preferences.py`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:183-340,420-520,3000-3090,3770-3920`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:300-450`
- Modify: `tests/Chat/test_console_chat_store.py`
- Create: `tests/Chat/test_console_speech_preferences.py`
- Modify: `tests/Chat/test_console_chat_store_summary.py`

**Interfaces:**
- Produces: `ConsoleSpeechPreferences(auto_speak: bool=False, paused: bool=False, consent_destination: str | None=None, consent_version: int=1)`.
- Produces: `parse_console_speech_preferences(metadata) -> ConsoleSpeechPreferences` and `merge_console_speech_preferences(metadata, preferences) -> dict[str, object]`.
- Produces store methods `set_auto_speak`, `pause_auto_speak`, `resume_auto_speak`, and `confirm_auto_speak_destination`.
- Persists through conversation metadata with optimistic version checking.

- [ ] **Step 1: Write metadata round-trip and default-off tests**

```python
def test_missing_metadata_defaults_auto_speak_off():
    assert parse_console_speech_preferences(None) == ConsoleSpeechPreferences()


def test_speech_preferences_merge_preserves_roleplay_metadata():
    metadata = {"console_roleplay": {"character_name": "Alba"}, "other": 1}
    merged = merge_console_speech_preferences(
        metadata,
        ConsoleSpeechPreferences(True, True, "sha256:abc", 1),
    )
    assert merged["console_roleplay"] == metadata["console_roleplay"]
    assert merged["console_speech"]["paused"] is True


def test_auto_speak_pause_round_trips_with_conversation(store):
    session = _persisted_session(store)
    store.set_auto_speak(session.id, True)
    store.pause_auto_speak(session.id)
    restored = _restore_session(store, session.persisted_conversation_id)
    assert restored.speech_preferences.paused is True
```

- [ ] **Step 2: Run persistence tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_speech_preferences.py tests/Chat/test_console_chat_store.py tests/Chat/test_console_chat_store_summary.py -k "speech_preferences or auto_speak" -v`

Expected: FAIL because sessions do not own speech preferences.

- [ ] **Step 3: Add bounded metadata parsing and store mutations**

Use the metadata key `console_speech`. Accept only exact booleans, consent version `1`, and a destination string matching `sha256:` plus 64 lowercase hex digits. Invalid/corrupt values fail closed to auto-speak off and no consent. Store mutations update in-memory state first only after the persistence adapter confirms the versioned metadata merge; unsaved sessions stage preferences until first conversation persistence.

- [ ] **Step 4: Run conversation persistence and restore tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_speech_preferences.py tests/Chat/test_console_chat_store.py tests/Chat/test_console_chat_store_summary.py tests/Chat/test_console_roleplay_metadata.py -v`

Expected: PASS with roleplay and speech metadata coexisting.

- [ ] **Step 5: Commit conversation speech preferences**

```bash
git add tldw_chatbook/Chat/console_speech_preferences.py tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Chat/chat_persistence_service.py tests/Chat/test_console_speech_preferences.py tests/Chat/test_console_chat_store.py tests/Chat/test_console_chat_store_summary.py
git commit -m "feat: persist per-conversation reply speech preferences"
```

### Task 5: Add pure auto-speak eligibility and consent decisions

**Files:**
- Create: `tldw_chatbook/Chat/console_auto_speak.py`
- Create: `tests/Chat/test_console_auto_speak.py`
- Modify: `tests/Chat/test_console_hands_free.py`

**Interfaces:**
- Produces: `AutoSpeakDisposition` values `SPEAK`, `DISABLED`, `PAUSED`, `NEEDS_CONSENT`, `HANDSFREE_OWNS`, `BACKGROUND`, `INELIGIBLE`.
- Produces: `AutoSpeakContext(preferences, destination_fingerprint, active_session_id, hands_free_active)`.
- Produces: `decide_auto_speak(message: ConsoleChatMessage, *, session_id: str, context: AutoSpeakContext) -> AutoSpeakDisposition`.

- [ ] **Step 1: Write a complete decision table**

```python
@pytest.mark.parametrize(
    ("enabled", "paused", "consent", "active", "hands_free", "status", "role", "expected"),
    [
        (False, False, DEST, True, False, "complete", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.DISABLED),
        (True, True, DEST, True, False, "complete", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.PAUSED),
        (True, False, None, True, False, "complete", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.NEEDS_CONSENT),
        (True, False, DEST, False, False, "complete", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.BACKGROUND),
        (True, False, DEST, True, True, "complete", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.HANDSFREE_OWNS),
        (True, False, DEST, True, False, "streaming", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.INELIGIBLE),
        (True, False, DEST, True, False, "complete", ConsoleMessageRole.ASSISTANT, AutoSpeakDisposition.SPEAK),
    ],
)
def test_auto_speak_decision_table(enabled, paused, consent, active, hands_free, status, role, expected):
    session_id = "active"
    message = ConsoleChatMessage(role=role, content="Ready.", status=status)
    context = AutoSpeakContext(
        preferences=ConsoleSpeechPreferences(
            auto_speak=enabled,
            paused=paused,
            consent_destination=consent,
        ),
        destination_fingerprint=DEST,
        active_session_id=session_id if active else "other",
        hands_free_active=hands_free,
    )
    assert decide_auto_speak(
        message, session_id=session_id, context=context
    ) is expected
```

- [ ] **Step 2: Run decision tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_auto_speak.py tests/Chat/test_console_hands_free.py -v`

Expected: FAIL because no shared auto-speak policy exists.

- [ ] **Step 3: Implement fail-closed pure policy**

Require non-empty completed assistant/character text, current active session, enabled/unpaused state, exact current destination consent, and hands-free inactive. Exclude system/user/tool/error/cancelled/partial content. The decision function performs no persistence, UI, network, playback, or logging.

- [ ] **Step 4: Run policy and hands-free tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_auto_speak.py tests/Chat/test_console_hands_free.py -v`

Expected: PASS; hands-free exclusion is explicit rather than timing-dependent.

- [ ] **Step 5: Commit the auto-speak policy**

```bash
git add tldw_chatbook/Chat/console_auto_speak.py tests/Chat/test_console_auto_speak.py tests/Chat/test_console_hands_free.py
git commit -m "feat: define reply speech eligibility and consent policy"
```

### Task 6: Wire opt-in auto-speak through trusted completion and TTS paths

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1950-2160,3250-3520`
- Create: `tldw_chatbook/Widgets/Console/console_auto_speak_consent.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py`
- Modify: `tldw_chatbook/UI/Console_Modules/wiring.py`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py:1120-1270`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:5000-5750`
- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:500-845`
- Create: `tests/UI/test_console_auto_speak_wiring.py`
- Modify: `tests/TTS/test_console_speak_autoplay.py`
- Modify: `tests/UI/test_uat_first_time_character_chat.py`

**Interfaces:**
- Produces: `ConsoleChatStore.subscribe_message_completed(callback) -> Callable[[], None]`, emitting one immutable `(session_id, message_id)` token per transition.
- Produces: visible `Switch(id="console-auto-speak")` labeled `Speak replies`.
- Produces: `AutoSpeakConsentModal(provider_label, sanitized_destination, charges_may_apply)` returning bool.
- Consumes: `openai_destination_fingerprint`, effective TTS resolver, `issue_tts_message_speech_snapshot`, and existing TTS request event.

- [ ] **Step 1: Write completion, consent, background, and failure tests**

```python
async def test_enabling_auto_speak_does_not_replay_existing_greeting(screen):
    greeting = _completed_character_message(screen)
    await _enable_and_confirm(screen)
    assert greeting.id not in screen.auto_speech_requests


async def test_new_active_character_reply_dispatches_once(screen):
    await _enable_and_confirm(screen)
    message = await _complete_character_reply(screen, "Welcome back.")
    assert screen.auto_speech_requests == [message.id]


async def test_destination_change_requires_reconfirmation_before_dispatch(screen):
    await _enable_and_confirm(screen, destination=DEST_A)
    screen.set_effective_tts_destination(DEST_B)
    message = await _complete_character_reply(screen, "This must wait.")
    assert screen.auto_speech_requests == []
    assert screen.consent_modal.destination == SANITIZED_DEST_B
    await screen.accept_auto_speak_consent()
    assert screen.auto_speech_requests == [message.id]


async def test_auto_speak_failure_persists_paused_state(screen):
    await _enable_and_confirm(screen)
    screen.tts_service.fail_next()
    await _complete_character_reply(screen, "Try me.")
    assert screen.active_session.speech_preferences.paused
```

- [ ] **Step 2: Run wiring and TTS tests**

Run: `.venv/bin/python -m pytest tests/UI/test_console_auto_speak_wiring.py tests/TTS/test_console_speak_autoplay.py tests/UI/test_uat_first_time_character_chat.py -k "auto_speak or destination_change" -v`

Expected: FAIL because no completion subscription or toggle exists.

- [ ] **Step 3: Implement one completion observer and dispatch coordinator**

Emit completion tokens only on the first live transition to complete, never while hydrating or restoring persisted messages. The screen subscriber resolves the message, active session, current effective TTS destination, and hands-free state, then calls `decide_auto_speak`. Enabling the toggle opens confirmation without attaching any previously completed message, so greetings and history are never replayed. If a newly completed active reply encounters a changed destination, `NEEDS_CONSENT` opens one deduplicated modal and retains only that immutable `(session_id, message_id)` token while the modal is open. Additional completions while that modal is open are not queued. On acceptance, store the destination fingerprint, then re-resolve and revalidate current session, message completion, destination, hands-free ownership, and speech snapshot before dispatching exactly once. Dismissal drops the token. `SPEAK` issues the same trusted speech snapshot/event used by Manual Speak. `BACKGROUND` stores no delayed token and leaves Manual Speak available. On TTS failure, persist paused state and expose Retry speech/Resume auto-speak. Never log text or raw origin.

- [ ] **Step 4: Run auto-speak, speech snapshot, hands-free, and UAT tests**

Run: `.venv/bin/python -m pytest tests/UI/test_console_auto_speak_wiring.py tests/TTS/test_console_speak_autoplay.py tests/TTS/test_console_speech_snapshot_admission.py tests/Chat/test_console_hands_free.py tests/UI/test_uat_first_time_character_chat.py -v`

Expected: PASS with exactly one TTS request for one eligible active reply.

- [ ] **Step 5: Commit opt-in auto-speak wiring**

```bash
git add tldw_chatbook/Chat/console_chat_store.py tldw_chatbook/Widgets/Console/console_auto_speak_consent.py tldw_chatbook/Widgets/Console/console_control_bar.py tldw_chatbook/UI/Console_Modules/wiring.py tldw_chatbook/UI/Console_Modules/message.py tldw_chatbook/UI/Screens/chat_screen.py tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py tests/UI/test_console_auto_speak_wiring.py tests/TTS/test_console_speak_autoplay.py tests/UI/test_uat_first_time_character_chat.py
git commit -m "feat: add opt-in character reply speech"
```

### Task 7: Keep Speak/Stop and playback state visible in message headers

**Files:**
- Modify: `tldw_chatbook/Chat/console_message_actions.py:45-205`
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py:90-120,3100-3260`
- Modify: `tldw_chatbook/UI/Console_Modules/message.py:1120-1270`
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss`
- Modify: `tests/Chat/test_console_message_actions.py`
- Modify: `tests/UI/test_console_transcript_region.py`
- Modify: `tests/TTS/test_console_speak_autoplay.py`

**Interfaces:**
- Produces message-header action state `speak`, `speak-stop`, or disabled generating indicator.
- Produces playback labels `Generating`, `Playing`, `Stopped`, `Failed` adjacent to the action.
- Preserves selected-message row for all non-speech actions.

- [ ] **Step 1: Write visibility and lifecycle tests**

```python
def test_completed_assistant_header_always_contains_speak():
    row = build_console_message_actions(_completed_assistant(), selected=False)
    assert row.header_action.action_id == "speak"


async def test_message_header_tracks_speech_lifecycle(transcript):
    transcript.set_speech_state(MESSAGE_ID, "generating")
    assert _header_copy(transcript, MESSAGE_ID) == "Generating"
    transcript.set_speech_state(MESSAGE_ID, "playing")
    assert _header_action(transcript, MESSAGE_ID) == "speak-stop"
```

- [ ] **Step 2: Run message action/transcript tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_message_actions.py tests/UI/test_console_transcript_region.py tests/TTS/test_console_speak_autoplay.py -k "header or speech_lifecycle" -v`

Expected: FAIL because Speak is currently confined to the selected-message action row.

- [ ] **Step 3: Render a stable icon action and bounded state label**

Reserve fixed header width for the existing speech icon button and one bounded state label. Use the existing icon/action IDs and tooltips; do not duplicate the selected-row Speak action. Swap Speak to Stop only for the message currently generating/playing. On failure, restore Speak and show Failed until the next action or selection change.

- [ ] **Step 4: Run transcript, action, and narrow-layout tests**

Run: `.venv/bin/python -m pytest tests/Chat/test_console_message_actions.py tests/UI/test_console_transcript_region.py tests/UI/test_console_narrow_layout.py tests/TTS/test_console_speak_autoplay.py -v`

Expected: PASS without message text overlap at supported widths.

- [ ] **Step 5: Commit visible speech actions**

```bash
git add tldw_chatbook/Chat/console_message_actions.py tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Console_Modules/message.py tldw_chatbook/css/components/_agentic_terminal.tcss tests/Chat/test_console_message_actions.py tests/UI/test_console_transcript_region.py tests/TTS/test_console_speak_autoplay.py
git commit -m "fix: keep Console speech actions visible"
```

### Task 8: Preserve boundaries between adjacent collapsed paste blocks

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_composer_bar.py:420-690,1320-1410,3238-3310,3980-4065`
- Modify: `tests/UI/test_console_composer_collapse.py`
- Modify: `tests/UI/test_console_composer_undo.py`
- Modify: `tests/UI/test_console_composer_keymap.py`
- Modify: `tests/UI/test_uat_first_time_character_chat.py`

**Interfaces:**
- Produces display label `Pasted text | N characters | Expand`.
- Produces exactly one literal newline segment between adjacent newly collapsed paste segments.
- Preserves canonical content, cursor mapping, delete-as-unit, undo/redo, draft persistence, and expanded editing behavior.

- [ ] **Step 1: Write adjacent-block and copy tests**

```python
def test_adjacent_collapsed_pastes_submit_with_one_newline(composer):
    composer.insert_pasted_text("A" * 80)
    composer.insert_pasted_text("B" * 90)
    assert composer.draft_text() == ("A" * 80) + "\n" + ("B" * 90)
    assert composer._display_draft_text().count("Pasted text |") == 2
    assert "Unfurl" not in composer._display_draft_text()


def test_expanded_paste_can_be_edited_without_hidden_separator_changes(composer):
    composer.insert_pasted_text("A" * 80)
    composer.activate_focused_paste_token()
    composer.insert_text(" edited")
    assert " edited" in composer.draft_text()
```

- [ ] **Step 2: Run collapse/undo tests and reproduce concatenation/copy**

Run: `.venv/bin/python -m pytest tests/UI/test_console_composer_collapse.py tests/UI/test_console_composer_undo.py tests/UI/test_console_composer_keymap.py -k "adjacent or pasted or unfurl" -v`

Expected: FAIL because `_canonical_draft_text()` joins raw segments and display says `Unfurl?`.

- [ ] **Step 3: Insert an explicit boundary and rename expansion states**

When inserting a new paste that will be collapsed, inspect the preceding segment and the new text. If the previous segment is a collapsed/confirm paste, its canonical text does not end with `"\n"`, and the new text does not start with `"\n"`, insert one ordinary literal `"\n"` segment before the new paste in the same undo transaction. Existing spaces or tabs do not count as a visible block boundary. Render collapsed as `Pasted text | {len(text)} characters | Expand` and confirmation as `Expand?`. Keep the literal newline through snapshot/history serialization so send, undo, restore, and session switching agree.

- [ ] **Step 4: Run all composer and UAT paste tests**

Run: `.venv/bin/python -m pytest tests/UI/test_console_composer_collapse.py tests/UI/test_console_composer_undo.py tests/UI/test_console_composer_keymap.py tests/UI/test_console_composer_caret_nav.py tests/UI/test_uat_first_time_character_chat.py -k "paste or draft or adjacent" -v`

Expected: PASS with canonical content matching the visible block boundary.

- [ ] **Step 5: Commit paste block integrity**

```bash
git add tldw_chatbook/Widgets/Console/console_composer_bar.py tests/UI/test_console_composer_collapse.py tests/UI/test_console_composer_undo.py tests/UI/test_console_composer_keymap.py tests/UI/test_uat_first_time_character_chat.py
git commit -m "fix: preserve boundaries between pasted text blocks"
```

### Task 9: Run the complete clean-profile character voice UAT

**Files:**
- Modify: `tests/UI/test_uat_first_time_character_chat.py`
- Create: `tests/integration/test_pocket_tts_character_roleplay.py`

**Interfaces:**
- Consumes: PocketTTS configuration/defaults from Plan 2, profile synchronization from Plan 3, and Tasks 1-8 above.
- Produces: one captured end-to-end request ledger proving exact chat model and TTS payload/auth behavior.
- Produces in the new integration module: `fake_chat`, `fake_pocket_tts`, and `character_png` fixtures plus the journey helpers shown below. The HTTP fixtures use loopback ephemeral servers, append parsed request records, and close in fixture teardown; `fake_chat` always returns `Welcome back.` from model `test-chat-model`.

- [ ] **Step 1: Add one complete automated acceptance journey**

```python
async def test_clean_profile_character_roleplay_uses_pocket_tts(fake_chat, fake_pocket_tts, character_png):
    app = await launch_clean_chatbook()
    await complete_quick_setup(
        app,
        tts_endpoint=fake_pocket_tts.speech_url,
        auth="none",
        model="pocket-tts",
        voice="alba",
        response_format="wav",
        use_as_default=True,
    )
    await import_character_and_start_chat(app, character_png)
    await enable_speak_replies_and_confirm(app)
    await send_roleplay_message(app, "Hello there")
    await wait_for_audio_playback(app)
    assert fake_chat.requests[-1].json()["model"] == "test-chat-model"
    assert fake_pocket_tts.requests[-1].json() == {
        "model": "pocket-tts",
        "input": "Welcome back.",
        "voice": "alba",
        "response_format": "wav",
        "speed": 1.0,
    }
    assert "Authorization" not in fake_pocket_tts.requests[-1].headers
```

- [ ] **Step 2: Run the acceptance journey**

Run: `.venv/bin/python -m pytest tests/integration/test_pocket_tts_character_roleplay.py tests/UI/test_uat_first_time_character_chat.py -v`

Expected: PASS with one active character conversation and one spoken response.

- [ ] **Step 3: Run the full affected regression set**

Run: `.venv/bin/python -m pytest tests/Wizards tests/TTS tests/Chat tests/UI/test_uat_first_time_character_chat.py tests/UI/test_console_auto_speak_wiring.py tests/UI/test_console_transcript_region.py tests/UI/test_console_composer_collapse.py tests/integration/test_first_run_pocket_tts_flow.py tests/integration/test_pocket_tts_character_roleplay.py -v`

Expected: PASS; inspect failures before changing expectations.

- [ ] **Step 4: Perform the real PocketTTS listening check**

Start Chatbook against a real PocketTTS service. Repeat the acceptance journey and record: sample success, first spoken reply latency, intelligibility, expected voice identity, playback stop behavior, and whether a second reply speaks exactly once. This is the only subjective release check; do not replace it with a byte-only assertion.

- [ ] **Step 5: Commit acceptance coverage**

```bash
git add tests/UI/test_uat_first_time_character_chat.py tests/integration/test_pocket_tts_character_roleplay.py
git commit -m "test: cover PocketTTS character roleplay journey"
```

## Plan Verification

Run: `.venv/bin/python -m pytest tests/UI/test_file_picker_start_dir.py tests/UI/test_character_display_text.py tests/Chat/test_console_chat_store.py tests/Chat/test_console_speech_preferences.py tests/Chat/test_console_auto_speak.py tests/UI/test_console_auto_speak_wiring.py tests/Chat/test_console_message_actions.py tests/UI/test_console_transcript_region.py tests/UI/test_console_composer_collapse.py tests/UI/test_uat_first_time_character_chat.py tests/integration/test_pocket_tts_character_roleplay.py -v`

Manual completion gate: no blank tab, correct character/greeting/title, correct chat model, explicit opt-in and destination confirmation, one spoken completed response through the configured PocketTTS endpoint, visible Speak/Stop state, persistent pause on failure, and no concatenated paste blocks.
