# Trusted Console Speech Snapshots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bind Console **Speak** to the exact completed assistant message and selected text variant the user clicked, rejecting stale or mismatched requests before cooldown or provider work while preserving the existing global TTS and complete-WAV path.

**Architecture:** Add one immutable, privacy-safe `TTSMessageSpeechSnapshot` value and make `ConsoleChatStore` the sole authority that issues and validates it. Console posts a dedicated snapshot request event carrying the snapshot and the issuing store's validator; the application and TTS handler validate it before normalization or cooldown, while the existing `TTSRequestEvent` remains the explicit trusted global-speech path for callers with no Console message. Persisted messages additionally compare the captured optimistic-lock row version through a narrow `ChatPersistenceService` read.

**Tech Stack:** Python 3.11+, dataclasses, Textual messages/events, SQLite optimistic versions, pytest/pytest-asyncio, Ruff, mypy.

---

## Scope and governing decision

- Implement only approved Slice **3A.3** from
  `Docs/superpowers/specs/2026-07-28-tts-character-identity-persona-separation-design.md`.
- Preserve global provider/model/voice/format/speed selection and the existing
  `TTSService.synthesize_default()` complete-WAV lifecycle.
- Do not add profile assignment mutation, assignment UI, assigned-profile
  resolution, automatic speech, Persona voice inheritance, portability, or
  managed audio.cpp behavior.

ADR required: yes
ADR path: `backlog/decisions/037-roleplay-assistant-identity-and-persona-user-profile-separation.md`
Reason: ADR-037 already governs immutable Console speech snapshots, authorship validation, pre-cooldown rejection, privacy, and the deferral of assigned-profile resolution. This task implements that accepted decision and requires no new ADR.

## File responsibility map

- Create `tldw_chatbook/Chat/console_speech.py`: immutable snapshot and bounded
  rejection-code contract only.
- Modify `tldw_chatbook/Chat/console_chat_store.py`: process-local speech
  revisions plus snapshot issuance and validation.
- Modify `tldw_chatbook/Chat/chat_persistence_service.py`: narrow current
  message-version lookup.
- Modify `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py`: dedicated
  Console snapshot event and validation before any cooldown mutation.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: issue snapshots from the
  store instead of copying caller-visible text into a global request.
- Modify `tldw_chatbook/app.py`: route the dedicated event without logging
  synthesis text.
- Modify `tldw_chatbook/Chat/console_message_actions.py`: offer **Speak** only
  for completed assistant text, matching admission.
- Create `Tests/Chat/test_console_speech_snapshots.py`: store issuance,
  revisions, authorship, variants, persistence-version, and privacy tests.
- Create `Tests/TTS/test_console_speech_snapshot_admission.py`: handler ordering,
  cooldown isolation, and valid global-flow tests.
- Modify `Tests/Chat/test_console_generation_actions.py`,
  `Tests/Chat/test_console_message_actions.py`, and
  `Tests/UI/test_console_native_chat_flow.py`: Console event wiring and
  user-visible regression coverage.
- Modify `Docs/Development/TTS/TTS_MODULE_GUIDE.md` and
  `Docs/Features/Speech-Services-Guide.md`: document trusted Console admission
  and unchanged global synthesis selection.
- Update `backlog/tasks/task-617.3 - Add-trusted-Console-speech-snapshots.md`
  only after implementation and verification evidence exists.

### Task 1: Define the immutable privacy-safe snapshot contract

**Files:**
- Create: `tldw_chatbook/Chat/console_speech.py`
- Create: `Tests/Chat/test_console_speech_snapshots.py`

- [ ] **Step 1: Write failing snapshot-value tests**

Add tests that construct the desired frozen value and assert:

```python
snapshot = TTSMessageSpeechSnapshot(
    session_id="session-1",
    message_id="message-1",
    persisted_conversation_id=None,
    persisted_message_id=None,
    raw_content="private response",
    selected_variant_id="message-1",
    speech_revision=0,
    persisted_message_version=None,
    role=ConsoleMessageRole.ASSISTANT,
    status="complete",
    assistant_kind="generic",
    character_ref=None,
)
with pytest.raises(FrozenInstanceError):
    snapshot.raw_content = "changed"  # type: ignore[misc]
assert "private response" not in repr(snapshot)
```

Also assert `character_ref` is omitted from `repr`, rejection codes are from a
closed set, and the exception exposes only its safe code and generic
retry-facing copy.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_speech_snapshots.py
```

Expected: collection fails because `tldw_chatbook.Chat.console_speech` does not
exist.

- [ ] **Step 3: Implement the minimal snapshot module**

Add:

```python
@dataclass(frozen=True, slots=True)
class TTSMessageSpeechSnapshot:
    session_id: str
    message_id: str
    persisted_conversation_id: str | None
    persisted_message_id: str | None
    raw_content: str = field(repr=False)
    selected_variant_id: str
    speech_revision: int
    persisted_message_version: int | None
    role: ConsoleMessageRole
    status: ConsoleMessageStatus
    assistant_kind: str | None
    character_ref: CharacterRef | None = field(repr=False)
```

Define `ConsoleSpeechSnapshotRejected` with a validated bounded code and one
generic user-facing message. Do not add serialization or a generic assistant
identity abstraction.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Task 1 command again. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Chat/console_speech.py \
  Tests/Chat/test_console_speech_snapshots.py
git commit -m "feat(tts): define trusted Console speech snapshot"
```

### Task 2: Make the Console store issue and validate snapshots

**Files:**
- Modify: `tldw_chatbook/Chat/console_chat_store.py:299-380`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:1352-1974`
- Modify: `tldw_chatbook/Chat/console_chat_store.py:2717-2775`
- Modify: `tldw_chatbook/Chat/chat_persistence_service.py:19-60`
- Test: `Tests/Chat/test_console_speech_snapshots.py`
- Test: `Tests/Chat/test_console_chat_store.py`

- [ ] **Step 1: Write failing issuance and validation tests**

Cover:

- a completed assistant snapshot carries its owning session, message,
  persisted IDs, exact raw selected content, linear-or-selected variant ID,
  revision, role/status, assistant kind, and scoped `CharacterRef`;
- generic, Persona, and authority-null character sessions carry no
  `CharacterRef`;
- user, system, tool, blank, pending, streaming, stopped, and failed messages
  cannot issue a snapshot;
- switching the active session, moving the message off the active path,
  deleting it, or changing session authorship rejects an already-issued
  snapshot;
- edit then restore identical content remains stale;
- adding, streaming, finalizing, or selecting a text variant makes the old
  snapshot stale;
- an unchanged snapshot validates to its exact raw content.

- [ ] **Step 2: Write failing persisted-version tests**

Add a real `ChatPersistenceService` test proving
`get_message_version(message_id)` returns the current positive row version and
returns `None` for a missing/deleted row. Add a store test that issues a
snapshot, changes the persisted row through a second writer, and confirms
validation rejects it even when visible text is restored to the original.

- [ ] **Step 3: Run the store tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_speech_snapshots.py \
  Tests/Chat/test_console_chat_store.py
```

Expected: snapshot issuance, revision, validation, and message-version APIs are
missing.

- [ ] **Step 4: Implement process-local revision ownership**

In `ConsoleChatStore`:

- initialize `_message_speech_revisions: dict[str, int]`;
- initialize a node's revision to zero only in `_register_tree_node`;
- remove revisions during subtree deletion and session close;
- add `_bump_message_speech_revision(message_id)` and
  `_selected_speech_variant_id(message)`;
- bump after every content/status/text-variant mutation, including stream
  chunks, reset, provisional-body replacement, edit, retry preparation,
  terminal transitions, variant add/begin/finalize/select;
- do not serialize the map or place the revision on `ConsoleChatMessage`.

No exact increment count is externally meaningful; only monotonic inequality is
part of admission.

- [ ] **Step 5: Implement the persisted-version seam**

Add `ChatPersistenceService.get_message_version(message_id) -> int | None`,
backed by `db.get_message_by_id`, accepting only a non-deleted exact positive
integer version. Declare the narrow method on `ConsoleChatPersistence`.

Snapshot issuance and validation call this seam only for a persisted message.
If persistence or a trustworthy version is unavailable, fail closed with a
bounded rejection code rather than assuming the current version.

- [ ] **Step 6: Implement store issuance and validation**

Add:

```python
def issue_tts_message_speech_snapshot(
    self, message_id: str
) -> TTSMessageSpeechSnapshot:
    ...

def validate_tts_message_speech_snapshot(
    self, snapshot: TTSMessageSpeechSnapshot
) -> str:
    ...
```

Issuance requires the message to be on the active session's active path and to
be a complete nonblank assistant message. Validation re-resolves all state from
the store and compares active session, owning session, active path, native and
persisted identity, exact selected variant, raw content, revision, role/status,
assistant kind, `CharacterRef`, and current durable row version. It returns the
captured raw content only after every check succeeds.

- [ ] **Step 7: Run the store tests and verify GREEN**

Run the Task 2 command. Expected: all tests pass with only the existing
dependency warning.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  Tests/Chat/test_console_speech_snapshots.py \
  Tests/Chat/test_console_chat_store.py
git commit -m "feat(console): issue and validate speech snapshots"
```

### Task 3: Validate snapshot requests before cooldown

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:46-59`
- Modify: `tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py:317-400`
- Create: `Tests/TTS/test_console_speech_snapshot_admission.py`
- Test: `Tests/TTS/test_tts_improvements.py`

- [ ] **Step 1: Write failing event-admission tests**

Create a recording handler and assert:

- `TTSMessageSpeechRequestEvent` accepts an exact snapshot plus validator and
  exposes the native message ID without a caller-supplied text field;
- the validator runs before service availability checks, normalization,
  cooldown cleanup, cooldown insertion, active-task creation, or synthesis;
- a bounded rejection posts one `TTSCompleteEvent`, leaves the cooldown mapping
  byte-for-byte unchanged, performs no provider call, and logs no raw content
  or authority;
- an unexpected validator failure also fails closed without exception detail;
- a valid snapshot passes its exact text into the unchanged synthesis path;
- legacy/non-Console `TTSRequestEvent` behavior remains unchanged.

- [ ] **Step 2: Run the event tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_tts_improvements.py
```

Expected: the dedicated event and pre-cooldown admission path are absent.

- [ ] **Step 3: Add the dedicated event and shared admitted-request path**

Add `TTSMessageSpeechRequestEvent` with the snapshot and issuing store's bound
validator. Refactor `handle_tts_request` only enough to resolve either:

- the explicit global event's trusted `text`, `message_id`, and optional
  `voice`; or
- the Console snapshot event's validator-returned exact text, snapshot message
  ID, and no voice override.

For a Console event, call the validator before reading the clock or mutating
cooldown state. Catch only the bounded rejection separately; convert unexpected
validator exceptions into a generic safe outcome without logging exception
values. After admission, reuse the existing length, whitespace, cooldown,
task, synthesis, complete-WAV, playback, and error code unchanged.

- [ ] **Step 4: Register the Textual event entry point**

Add `on_tts_message_speech_request_event` beside
`on_tts_request_event`, both calling `handle_tts_request`.

- [ ] **Step 5: Run the event tests and verify GREEN**

Run the Task 3 command. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_tts_improvements.py
git commit -m "feat(tts): admit Console snapshots before cooldown"
```

### Task 4: Wire Console and application routing

**Files:**
- Modify: `tldw_chatbook/Chat/console_message_actions.py`
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py:15607-15725`
- Modify: `tldw_chatbook/app.py:228-241`
- Modify: `tldw_chatbook/app.py:6450-6470`
- Modify: `Tests/Chat/test_console_message_actions.py:548-637`
- Modify: `Tests/Chat/test_console_generation_actions.py:528-575`
- Modify: `Tests/UI/test_console_native_chat_flow.py:2460-2515`

- [ ] **Step 1: Write failing action and wiring tests**

Update action tests to require **Speak** only for complete nonblank assistant
messages. Update screen tests to assert the posted object is
`TTSMessageSpeechRequestEvent`, contains the store-issued snapshot, contains no
caller text field, and its validator rejects a mutation performed after the
button press. Update the repaired-response test to inspect
`event.snapshot.raw_content`.

Add an application routing test that confirms the dedicated handler does not
log the snapshot text and delegates to the app-owned `TTSEventHandler`.

- [ ] **Step 2: Run the wiring tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/UI/test_console_native_chat_flow.py
```

Expected: Console still posts caller-copied `TTSRequestEvent` text and offers
Speak for user messages.

- [ ] **Step 3: Wire the store-issued event**

In `ChatScreen.handle_console_message_action`, replace:

```python
TTSRequestEvent(text=message.content, message_id=message.id)
```

with store issuance followed by
`TTSMessageSpeechRequestEvent(snapshot, store.validate_tts_message_speech_snapshot)`.
If issuance fails, show the generic retry copy and do not set speaking state.

In `app.py`, register a separate `@on(TTSMessageSpeechRequestEvent)` handler
that logs only a static safe message, ensures the TTS handler, and delegates.
Do not route the snapshot event through the legacy handler that logs
`event.text`.

- [ ] **Step 4: Align the action availability**

Change `ConsoleMessageActionService` so **Speak** is available only for a
complete nonblank assistant message. Preserve stop-toggle order and all
existing playback plumbing.

- [ ] **Step 5: Run the wiring tests and verify GREEN**

Run the Task 4 command. Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/app.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/UI/test_console_native_chat_flow.py
git commit -m "feat(console): post trusted speech requests"
```

### Task 5: Prove unchanged native/global synthesis and document the boundary

**Files:**
- Modify: `Tests/TTS/test_console_audio_cpp_native.py`
- Modify: `Tests/TTS/test_console_speak_autoplay.py`
- Modify: `Docs/Development/TTS/TTS_MODULE_GUIDE.md`
- Modify: `Docs/Features/Speech-Services-Guide.md`

- [ ] **Step 1: Add the valid-snapshot native audio.cpp regression**

Extend the native Console audio.cpp test to pass a real store-issued snapshot
through `TTSMessageSpeechRequestEvent`. Assert the native adapter receives the
same exact `TTSRequest`, the legacy generator remains unused, the response is
closed, the artifact is a complete WAV, and playback/autoplay behavior is
unchanged.

- [ ] **Step 2: Run the native regression and verify RED/GREEN**

First run before the test-support implementation is complete and confirm it
fails at the new event path. After wiring the shared fixtures, run:

```bash
../../.venv/bin/python -m pytest -q \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/TTS/test_console_speech_snapshot_admission.py
```

Expected final result: all tests pass.

- [ ] **Step 3: Update user and developer documentation**

Document that Console **Speak** now:

- issues an ephemeral immutable snapshot from the Console store;
- validates the exact completed assistant message, selected text variant,
  revision, durable row version, and trusted authorship before cooldown;
- asks the user to click Speak again when the message changed;
- keeps global TTS selection and native/legacy complete-WAV behavior unchanged;
- never persists the snapshot or manages the external audio.cpp process.

- [ ] **Step 4: Commit**

```bash
git add Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Docs/Development/TTS/TTS_MODULE_GUIDE.md \
  Docs/Features/Speech-Services-Guide.md
git commit -m "docs(tts): describe trusted Console speech admission"
```

### Task 6: Full focused verification and task closeout

**Files:**
- Modify: `backlog/tasks/task-617.3 - Add-trusted-Console-speech-snapshots.md`

- [ ] **Step 1: Run the complete focused regression union**

```bash
../../.venv/bin/python -m pytest -q \
  Tests/Chat/test_console_speech_snapshots.py \
  Tests/Chat/test_console_chat_store.py \
  Tests/Chat/test_console_message_actions.py \
  Tests/Chat/test_console_generation_actions.py \
  Tests/TTS/test_console_speech_snapshot_admission.py \
  Tests/TTS/test_console_audio_cpp_native.py \
  Tests/TTS/test_console_speak_autoplay.py \
  Tests/TTS/test_tts_improvements.py \
  Tests/UI/test_console_native_chat_flow.py
```

Expected: all task-focused tests pass. Record inherited warnings separately.

- [ ] **Step 2: Run broader TTS and Console regression suites**

```bash
../../.venv/bin/python -m pytest -q Tests/TTS Tests/Chat
```

If a failure occurs, reproduce the exact node on untouched current `origin/dev`
before classifying it as inherited.

- [ ] **Step 3: Run static and hygiene checks**

```bash
../../.venv/bin/python -m ruff check \
  tldw_chatbook/Chat/console_speech.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Chat/chat_persistence_service.py \
  tldw_chatbook/Chat/console_message_actions.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/app.py
../../.venv/bin/python -m ruff format --check \
  tldw_chatbook/Chat/console_speech.py \
  Tests/Chat/test_console_speech_snapshots.py \
  Tests/TTS/test_console_speech_snapshot_admission.py
../../.venv/bin/python -m compileall -q \
  tldw_chatbook/Chat/console_speech.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py
../../.venv/bin/python -m mypy \
  tldw_chatbook/Chat/console_speech.py \
  tldw_chatbook/Chat/console_chat_store.py \
  tldw_chatbook/Event_Handlers/TTS_Events/tts_events.py
git diff --check origin/dev...HEAD
```

Compare any pre-existing whole-file formatter or mypy diagnostics against
exact `origin/dev`; do not broaden this task into repository-wide cleanup.

- [ ] **Step 4: Run privacy and scope guards**

Verify the production diff contains no snapshot serialization, text-bearing
log, assignment resolver, assignment UI, Persona TTS inheritance, or managed
audio.cpp process code:

```bash
git diff origin/dev...HEAD -- tldw_chatbook \
  | rg -n "raw_content|logger|assignment|managed|subprocess|server\\.json"
```

Review each match manually; the snapshot field and comparisons are expected,
but logging it is forbidden.

- [ ] **Step 5: Request independent code review**

Use `superpowers:requesting-code-review` against the complete
`origin/dev...HEAD` range. Address every verified issue and re-run affected
tests.

- [ ] **Step 6: Complete Backlog evidence**

Only after all gates pass:

- check all acceptance criteria and Definition of Done items;
- add concise implementation notes with test counts, static evidence,
  inherited baselines, review outcome, ADR-037 linkage, and any plan deviation;
- set `TASK-617.3` to Done.

- [ ] **Step 7: Commit closeout**

```bash
git add "backlog/tasks/task-617.3 - Add-trusted-Console-speech-snapshots.md"
git commit -m "docs: close trusted Console speech snapshot task"
```
