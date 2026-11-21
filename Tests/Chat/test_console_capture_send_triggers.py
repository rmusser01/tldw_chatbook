"""TASK-33621.2: Capture-On sends that dev refused before contacting the provider.

Every chat with a system row (a session prompt, a character, or a seeded
greeting -- which is how a lone /generate-image result reaches the provider)
and every send after a saved image result was refused before the provider was
contacted. These tests drive the real ``ConsoleChatController`` and
``ConsoleChatStore`` over a real SQLite database, the real durable trace
request builder and the real ``ConsoleTraceBoundaryFactory``; only the
provider adapter at the bottom of ``ConsoleProviderGateway`` is a recorder.
"""

from __future__ import annotations

import io
import re
import sqlite3

import pytest
from PIL import Image

from Tests.Chat.test_console_send_diagnostics import sinks  # noqa: F401
from Tests.Chat.test_console_trace_current_turn_transforms import (  # noqa: F401
    make_database,
    make_gateway,
    trace_harness,
)
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleRunStatus,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.console_trace_native_reader import ConsoleTraceNativeReader

pytestmark = pytest.mark.bootstrap_profile

PROMPT = "You are terse."


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _generate_image(store, session_id: str) -> None:
    """Append a saved image result exactly as /generate-image does."""
    store.append_generation_message(
        session_id,
        content="[image] a red square",
        variants=[
            (
                _png(),
                "image/png",
                GenerationVariantMeta(
                    prompt="a red square",
                    negative_prompt="",
                    backend="test",
                    model=None,
                    seed=1,
                    style=None,
                    params={},
                ),
            )
        ],
        persist=True,
    )


def _character_id(harness, name: str = "Ava") -> int:
    return int(harness.db.add_character_card({"name": name}))


@pytest.fixture
def harness(trace_harness):  # noqa: F811 -- fixture imported above
    """``trace_harness`` that also records the adapter's ``system_message``.

    The gateway sends the leading system rows as ``system_message`` and the
    rest as ``messages_payload``; the shared harness records only the latter.
    """
    adapter = trace_harness.gateway._chat_api_call_fn
    trace_harness.systems = []

    def recording(**kwargs):
        trace_harness.systems.append(kwargs.get("system_message"))
        return adapter(**kwargs)

    trace_harness.gateway._chat_api_call_fn = recording
    return trace_harness


async def _send(harness, text: str, session_id: str = "session-1"):
    """Send one draft; return the result, rows sent, and the system message."""
    before = len(harness.entries)
    result = await harness.controller.submit_draft(text, session_id=session_id)
    sent = harness.entries[before:]
    system = harness.systems[-1] if sent else None
    return result, sent, system


def _assert_answered(harness, result, sent) -> None:
    assert result.terminal_status is ConsoleRunStatus.COMPLETED, result
    assert result.provider_started, result
    assert len(sent) == 1, "the provider must be contacted exactly once"
    reply = harness.store.get_message(result.assistant_message_id)
    assert reply.content == "answer"
    # Capture stayed on: the turn has a durable trace, not a silent Capture Off.
    user = harness.store.get_message(result.user_message_id)
    calls = ConsoleTraceNativeReader(harness.db).read_calls(user.persisted_message_id)
    assert len(calls) == 1
    assert calls[0].capture.request["messages_payload"] == sent[0]


@pytest.mark.asyncio
async def test_session_system_prompt_set_before_the_first_send(harness):
    harness.store.set_session_system_prompt("session-1", PROMPT)

    result, sent, system = await _send(harness, "hello")

    _assert_answered(harness, result, sent)
    assert system == PROMPT
    assert sent[0] == [{"role": "user", "content": "hello"}]
    following, sent, system = await _send(harness, "again")
    _assert_answered(harness, following, sent)
    assert system == PROMPT


@pytest.mark.asyncio
async def test_session_system_prompt_set_mid_chat(harness):
    first, sent, system = await _send(harness, "hello")
    _assert_answered(harness, first, sent)
    assert system is None
    harness.store.set_session_system_prompt("session-1", PROMPT)

    result, sent, system = await _send(harness, "again")

    _assert_answered(harness, result, sent)
    assert system == PROMPT
    assert sent[0] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "again"},
    ]
    following, sent, system = await _send(harness, "third")
    _assert_answered(harness, following, sent)
    assert system == PROMPT


@pytest.mark.asyncio
async def test_character_swapped_into_an_existing_chat(harness):
    store = harness.store
    first, sent, _system = await _send(harness, "hello")
    _assert_answered(harness, first, sent)
    character_id = _character_id(harness)
    # The Console character picker's in-place rebind (session.py), then the
    # store swap it calls. An existing chat gets no greeting.
    session = next(item for item in store.sessions() if item.id == "session-1")
    session.runtime_backend = "local"
    session.assistant_kind = "character"
    session.assistant_id = str(character_id)
    session.assistant_authority_id = None
    session.character_id = character_id
    _, _, persisted = store.swap_session_character_roleplay(
        "session-1",
        character_name="Ava",
        system_template="You are {{char}}.",
        greeting_template="",
        global_default="User",
    )
    assert persisted

    result, sent, system = await _send(harness, "who are you?")

    _assert_answered(harness, result, sent)
    assert system == "You are Ava."


@pytest.mark.asyncio
async def test_new_character_chat_with_its_greeting(harness):
    store = harness.store
    character_id = _character_id(harness)
    settings = next(
        item for item in store.sessions() if item.id == "session-1"
    ).settings
    store.create_session(
        session_id="session-2",
        title="Chat with Ava",
        settings=settings,
        assistant_kind="character",
        assistant_id=str(character_id),
        character_id=character_id,
        character_name="Ava",
    )
    greeting = store.seed_character_roleplay(
        "session-2",
        system_template="You are {{char}}.",
        greeting_template="Hi {{user}}, I am {{char}}.",
        global_default="User",
    )
    assert greeting is not None
    assert store.persist_session_if_needed("session-2")

    result, sent, system = await _send(harness, "hello", session_id="session-2")

    _assert_answered(harness, result, sent)
    assert "You are Ava." in system
    assert "Hi User, I am Ava." in system
    assert sent[0] == [{"role": "user", "content": "hello"}]
    following, sent, _system = await _send(harness, "and?", session_id="session-2")
    _assert_answered(harness, following, sent)


@pytest.mark.asyncio
async def test_chat_whose_only_assistant_history_is_an_image_result(harness):
    _generate_image(harness.store, "session-1")

    result, sent, system = await _send(harness, "describe it")

    _assert_answered(harness, result, sent)
    # A leading assistant row is folded into the system message (task-1531).
    assert "[image] a red square" in system
    assert sent[0] == [{"role": "user", "content": "describe it"}]
    following, sent, _system = await _send(harness, "again")
    _assert_answered(harness, following, sent)


@pytest.mark.asyncio
async def test_sends_after_an_image_generated_following_a_normal_exchange(harness):
    first, sent, _system = await _send(harness, "hello")
    _assert_answered(harness, first, sent)
    _generate_image(harness.store, "session-1")

    result, sent, _system = await _send(harness, "describe it")

    _assert_answered(harness, result, sent)
    # Provider history carries an assistant image result as its text only.
    assert sent[0] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer"},
        {"role": "assistant", "content": "[image] a red square"},
        {"role": "user", "content": "describe it"},
    ]
    following, sent, _system = await _send(harness, "and now?")
    _assert_answered(harness, following, sent)
    assert sent[0][2] == {"role": "assistant", "content": "[image] a red square"}


@pytest.mark.asyncio
async def test_provenance_stage_failure_logs_a_content_free_category(
    harness,
    monkeypatch,
    sinks,  # noqa: F811 -- fixture imported above
):
    """AC#5: the catch that pauses the send leaves a keyed, content-free line."""

    def locked(*_args, **_kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(controller_module, "admit_message_provenance", locked)

    result, sent, _system = await _send(harness, "PRIVATE-DRAFT-33621")

    assert sent == []
    assert result.terminal_status is ConsoleRunStatus.BLOCKED
    lines = sinks.path.read_text().splitlines()
    failure = next((line for line in lines if "phase=trace_provenance" in line), "")
    assert failure, "the trace-provenance catch logged nothing"
    assert "status=failed" in failure
    assert "error_category=database" in failure
    assert "exception_type=OperationalError" in failure
    attempt = re.search(r"attempt_id=([0-9a-f]{32})", failure)
    assert attempt is not None
    submit = next((line for line in lines if "phase=controller_submit" in line), "")
    assert f"attempt_id={attempt.group(1)}" in submit
    text = "\n".join(lines)
    assert "PRIVATE-DRAFT-33621" not in text
    assert "database is locked" not in text


async def _voice_send(harness, transcript: str):
    """One Capture-On speculative voice turn through its production stages.

    The controller prepares the attempt; the gateway dispatches it as a
    provisional voice call with Capture On (its dispatch commit) and seals the
    manifest; the winning promotion commits the pair and imports the trace,
    wired as ``create_console_speculative_voice_session`` wires it. Only the
    provider adapter at the bottom of the gateway is the harness's recorder.

    Returns:
        The rows the provider received, the prepared request's provenance,
        the promotion outcome, and the trace imports (an exception when the
        import refused the turn).
    """
    from Tests.Chat.test_console_voice_promotion import _inline
    from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderCallPurpose
    from tldw_chatbook.Chat.console_trace_provenance import (
        ConsoleRequestRoute,
        ConsoleTraceCaptureMode,
    )
    from tldw_chatbook.Chat.console_trace_service import ConsoleTraceService
    from tldw_chatbook.Chat.console_voice_attempts import VoiceAttemptSnapshot
    from tldw_chatbook.Chat.console_voice_promotion import (
        VoicePromotionContext,
        VoicePromotionOwner,
        VoiceWinningPromotion,
    )
    from tldw_chatbook.Chat.console_voice_trace_gateway import VoiceTraceImportContext

    gateway, store = harness.gateway, harness.store
    prepared = await harness.controller.prepare_speculative_voice_attempt(
        attempt_epoch=1, transcript=transcript, turn_id="voice-turn"
    )
    request, seed = prepared.request, prepared.promotion_seed
    assert prepared.requires_pre_dispatch_authority is False, (
        "the voice turn fell back to an accepted send"
    )
    attempt = request.provisional_trace_attempt
    assert attempt is not None, "the voice turn has no Capture-On trace"
    before = len(harness.entries)
    parts: list[str] = []
    try:
        async for item in gateway.stream_chat(
            request.resolution,
            request.prepared,
            tools=request.prepared.tools,
            dispatch_purpose=ConsoleProviderCallPurpose.VOICE_PROVISIONAL,
            capture_mode=ConsoleTraceCaptureMode.CAPTURE_ON,
            route=ConsoleRequestRoute.FRESH,
            provisional_trace_attempt=attempt,
        ):
            if isinstance(item, str):
                parts.append(item)
        manifest, envelopes = gateway.seal_provisional_voice_trace(attempt)
    except BaseException:
        gateway.abandon_provisional_voice_trace(attempt)
        raise
    reply = "".join(parts)
    persistence = store.persistence
    registry = gateway.provisional_trace_registry
    imports: list[object] = []

    def trace_context(_context, commit):
        return VoiceTraceImportContext(
            import_id=seed.promotion_id,
            conversation_id=commit.conversation_id,
            user_message_id=commit.user_message_id,
            user_revision_id=commit.user_revision_id,
            assistant_message_id=commit.assistant_message_id,
            assistant_revision_id=commit.assistant_revision_id,
            turn_id=commit.user_message_id,
            run_id=f"voice:{seed.attempt_id}",
            policy=seed.capture_policy,
        )

    def import_trace(sealed, sealed_envelopes, import_context):
        try:
            result = ConsoleTraceService(
                repository=persistence.console_trace_repository
            ).import_provisional_voice_trace(
                persistence.db, registry, sealed, sealed_envelopes, import_context
            )
        except Exception as exc:  # recorded: promotion swallows trace failures
            imports.append(exc)
            raise
        imports.append(result)
        return result

    winner = VoiceWinningPromotion(
        VoicePromotionOwner(lambda: store, sync_runner=_inline),
        trace_registry=registry,
        trace_context_factory=trace_context,
        trace_importer=import_trace,
        sync_runner=_inline,
    )
    context = VoicePromotionContext(
        promotion_id=seed.promotion_id,
        attempt_id=seed.attempt_id,
        origin=seed.origin,
        expected_native_leaf_id=seed.expected_native_leaf_id,
        expected_persisted_leaf_id=seed.expected_persisted_leaf_id,
        user_text=transcript,
        assistant_text=reply,
        usage_json=None,
        terminal_boundary_id=seed.terminal_boundary_id,
        capture_eligible_at_dispatch=seed.capture_eligible_at_dispatch,
    )
    outcome = await winner.promote(
        context,
        VoiceAttemptSnapshot(
            attempt_epoch=1,
            response_text=reply,
            trace_manifest=manifest,
            trace_envelopes=envelopes,
        ),
    )
    return harness.entries[before:], request.prepared.provenance, outcome, imports


@pytest.mark.asyncio
async def test_a_voice_send_after_an_image_result_traces_what_was_sent(harness):
    """TASK-33621.46: the voice builder applies the typed path's saved-media rule.

    Provider history sends an assistant image result as its text only, but
    that message's saved revision projects the image too. The typed builder
    (TASK-33621.2) admits such a row as a provider artifact; the voice builder
    admitted every owned row as its saved revision, so the voice turn's trace
    named a revision the provider never received. The voice turn's own
    dispatch commit and trace import pass on both; what the turn recorded is
    what differs, so this reads it back the way the typed tests do.
    """
    from tldw_chatbook.Chat.console_trace_provenance import (
        SavedRevisionTraceProvenance,
    )

    first, sent, _system = await _send(harness, "hello")
    _assert_answered(harness, first, sent)
    _generate_image(harness.store, "session-1")

    sent, provenance, outcome, imports = await _voice_send(harness, "describe it")

    assert sent == [
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "answer"},
            {"role": "assistant", "content": "[image] a red square"},
            {"role": "user", "content": "describe it"},
        ]
    ], "the provider must be contacted exactly once, with the image as text"
    assert outcome._commit is not None, "the voice pair was not committed"
    assert len(imports) == 1
    assert not isinstance(imports[0], Exception), repr(imports[0])
    descriptors = provenance.messages_payload
    assert [type(item) is SavedRevisionTraceProvenance for item in descriptors] == [
        True,
        True,
        False,  # the image result left its image out: not that revision
        False,  # the active voice turn
    ]
    # The trace says what the provider received, not the saved revision.
    calls = ConsoleTraceNativeReader(harness.db).read_calls(
        outcome._commit.user_message_id
    )
    assert len(calls) == 1
    recorded = calls[0].capture.request["messages_payload"]
    assert recorded[:3] == sent[0][:3], recorded
