"""TASK-33621.2: blocked Console sends, through the real ChatScreen.

Each test mounts the real ``ChatScreen`` with its real controller, store,
runtime and a file-backed ChaChaNotes database (so the agent loop and the
durable trace path run as in the app). Only the provider adapter at the bottom
of the real ``ConsoleProviderGateway`` is a recorder. Nothing stubs the durable
trace request builder or the recovery projection; the provenance-stage failure
tests make the SQLite admission step itself raise ``database is locked``.
Every AC#1 trigger (a session prompt set before the first send or mid-chat, a
character swapped in, a Personas "Start chat" character chat with its
greeting, an image result as the only assistant history, and an image after a
normal exchange) was refused before the provider on origin/dev.

Startup trace maintenance uses its ordinary timing. TASK-33621.47 now protects
the exact canonical revision metadata between admission and call reservation;
the separate deterministic GC tests also enforce payload reclamation.
"""

from __future__ import annotations

import asyncio
import io
import sqlite3
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
from PIL import Image
from textual.widgets import Button, Static
from textual.worker import WorkerCancelled

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _persist_console_provider_config
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat import console_chat_controller as controller_module
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleMessageRole,
    ConsoleRunStatus,
    GenerationVariantMeta,
)
from tldw_chatbook.Chat.conversation_archive_actions import conversation_send_refusal
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.prompt_queue import turn_recovery_label
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)
from tldw_chatbook.UI.Console_Modules.session import _character_session_prompt_seed
from tldw_chatbook.Widgets.Console.console_send_authority_summary import (
    project_console_send_authority,
)

pytestmark = pytest.mark.bootstrap_profile

PROMPT = "You are terse."
REPLY = "Provider reply 33621"
IMAGE_TEXT = "[image] a red square"
BLOCKED_REASON = "trace not saved"
SIZES = [(80, 24), (235, 52)]


def _frame_rows(host) -> list[str]:
    """Return the painted screen, one string per terminal row."""
    root = ElementTree.fromstring(host.export_screenshot())
    rows: dict[float, list[tuple[float, str]]] = {}
    for element in root.iter():
        if element.tag.endswith("text") and element.get("y") is not None:
            rows.setdefault(float(element.get("y")), []).append(
                (float(element.get("x") or 0), "".join(element.itertext()))
            )
    # Textual's SVG export writes spaces as U+00A0.
    return [
        "".join(text for _x, text in sorted(parts)).replace("\xa0", " ")
        for _y, parts in sorted(rows.items())
    ]


async def _until(pilot, condition, *, timeout: float = 10.0, what: str = "") -> None:
    for _ in range(int(timeout / 0.25)):
        await pilot.pause(0.25)
        if condition():
            return
    raise AssertionError(what or "condition not reached")


async def _until_completed(h) -> None:
    """Wait out a run that is still finishing on a loaded machine."""
    await _until(
        h.pilot,
        lambda: h.controller.run_state.status is ConsoleRunStatus.COMPLETED,
        timeout=30.0,
        what=f"the run did not complete: {h.controller.run_state}",
    )


async def _settle_workers(h) -> None:
    """Await the app's workers; one cancelled by an exclusive successor is fine.

    ``workers.wait_for_complete()`` raises ``WorkerCancelled`` when the
    exclusive ``console-sync`` worker replaces an earlier run of itself, which
    made the recovery-action test fail at random. Any other failure still
    fails the test.
    """
    for _ in range(3):
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                *(worker.wait() for worker in list(h.host.workers)),
                return_exceptions=True,
            ),
            10,
        )
        assert all(
            not isinstance(outcome, BaseException)
            or isinstance(outcome, WorkerCancelled)
            for outcome in outcomes
        ), outcomes
        await h.pilot.pause()


@asynccontextmanager
async def _mounted_console(tmp_path, monkeypatch, size):
    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "chat.sqlite", "task-33621-2")
    app.chachanotes_db = database
    _persist_console_provider_config(
        app,
        provider="openai",
        model="gpt-4.1",
        provider_settings={"api_key": "synthetic-test-key"},
    )
    # The send boundary checks the saved chat is not archived. The test app
    # has no local conversation service, so a later send would fail closed
    # with "Local conversation storage is unavailable" before the controller.
    app.local_chat_conversation_service = SimpleNamespace(
        get_conversation_archive_states=lambda ids: {}
    )
    host = ConsoleHarness(app)
    host.CSS_PATH = TldwCli.CSS_PATH
    calls: list[dict] = []

    def adapter(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": REPLY}}]}

    console = None
    try:
        async with host.run_test(size=size) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-native-composer")
            controller = console._ensure_console_chat_controller()
            gateway = controller.provider_gateway
            monkeypatch.setattr(gateway, "_chat_api_call_fn", adapter)
            original_resolve = gateway.resolve_for_send

            async def resolve(selection):
                return replace(await original_resolve(selection), streaming=False)

            monkeypatch.setattr(gateway, "resolve_for_send", resolve)
            runtime = console._console_runtime()
            tasks: list[asyncio.Task] = []
            accept = runtime.accept_turn

            def capture(request, **kwargs):
                turn_id = accept(request, **kwargs)
                tasks.append(runtime._turn_custody[turn_id].task)
                return turn_id

            monkeypatch.setattr(runtime, "accept_turn", capture)
            yield SimpleNamespace(
                host=host,
                pilot=pilot,
                console=console,
                controller=controller,
                store=controller.store,
                session=controller.store.ensure_session(),
                database=database,
                calls=calls,
                tasks=tasks,
            )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


async def _send(h, text: str) -> None:
    """Send ``text`` in the active chat through the visible send action."""
    started = len(h.tasks)
    h.console._session._sync_console_session_draft()
    h.console._console_composer_or_none().load_draft(text)
    await asyncio.wait_for(
        h.console._send_console_message_from_visible_action(
            session_id=h.store.active_session_id
        ),
        10,
    )
    for task in h.tasks[started:]:
        await asyncio.wait({task}, timeout=30)
    await h.console._sync_native_console_chat_ui()
    await h.pilot.pause(0.3)


def _assistant_texts(h) -> list[str]:
    return [
        message.content
        for message in h.store.messages_for_session(h.store.active_session_id)
        if message.role is ConsoleMessageRole.ASSISTANT
        and not message.generation_metadata
    ]


def _block_provenance_admission(monkeypatch) -> SimpleNamespace:
    """Make the admission SQL fail like a locked database, until released."""
    state = SimpleNamespace(blocked=True)
    original = controller_module.admit_message_provenance

    def admit(*args, **kwargs):
        if state.blocked:
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(controller_module, "admit_message_provenance", admit)
    return state


def _png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def _generate_image(h) -> None:
    """Commit an image result through the ChatScreen's /generate-image path."""
    h.console._image._append_durable_generation_message(
        h.store,
        h.store.active_session_id,
        content=IMAGE_TEXT,
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
    )


def _add_character(h) -> int:
    return int(
        h.database.add_character_card(
            {
                "name": "Ava",
                "description": "A terse guide.",
                "system_prompt": "Guide {{user}} as {{char}}.",
                "first_message": "Hi {{user}}, I am {{char}}.",
            }
        )
    )


async def _system_prompt_first(h) -> list[str]:
    h.store.set_session_system_prompt(h.store.active_session_id, PROMPT)
    return [PROMPT]


async def _system_prompt_mid_chat(h) -> list[str]:
    await _send(h, "Hello")
    h.store.set_session_system_prompt(h.store.active_session_id, PROMPT)
    return [PROMPT]


async def _character_swapped_in(h) -> list[str]:
    await _send(h, "Hello")
    character_id = _add_character(h)
    card = h.database.get_character_card_by_id(character_id)
    assert h.console._session._swap_console_session_character(
        h.store,
        character_id,
        _character_session_prompt_seed(card),
        global_default="User",
    )
    return ["as Ava."]


async def _new_character_chat(h) -> list[str]:
    """Personas 'Start chat' handoff: a new chat seeded with the greeting."""
    character_id = _add_character(h)

    async def get_character(record_id, mode="local"):
        return h.database.get_character_card_by_id(int(record_id))

    h.console.app_instance.character_persona_scope_service = SimpleNamespace(
        get_character=get_character
    )
    payload = ChatHandoffPayload(
        source="personas",
        item_type="character-card",
        title="Ava",
        body="Character summary",
        runtime_backend="local",
        source_owner="local",
        source_selector_state="local",
        metadata={
            "intent": "start_chat",
            "selected_kind": "character",
            "selected_record_id": str(character_id),
            "selected_name": "Ava",
            "selected_target_id": f"local:character:{character_id}",
            "backend": "local",
        },
    )
    assert await h.console._session._start_character_console_session(payload)
    return ["as Ava.", "I am Ava."]


async def _image_only_history(h) -> list[str]:
    _generate_image(h)
    return [IMAGE_TEXT]


async def _image_after_an_exchange(h) -> list[str]:
    await _send(h, "Hello")
    _generate_image(h)
    return []


TRIGGERS = {
    "system-prompt-before-first-send": _system_prompt_first,
    "system-prompt-set-mid-chat": _system_prompt_mid_chat,
    "character-swapped-into-a-chat": _character_swapped_in,
    "new-character-chat-with-greeting": _new_character_chat,
    "image-result-only-history": _image_only_history,
    "image-after-a-normal-exchange": _image_after_an_exchange,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", list(TRIGGERS))
async def test_each_trigger_chat_sends_and_renders_the_reply(
    tmp_path, monkeypatch, trigger
):
    """AC#1: each chat dev refused reaches the provider and shows its reply."""
    async with _mounted_console(tmp_path, monkeypatch, (120, 40)) as h:
        in_system = await TRIGGERS[trigger](h)
        before = len(h.calls)

        await _send(h, "Who are you?")

        assert len(h.calls) == before + 1, "the provider was not contacted"
        system = h.calls[-1].get("system_message") or ""
        for fragment in in_system:
            assert fragment in system
        await _until_completed(h)
        assert _assistant_texts(h)[-1] == REPLY
        await _until(h.pilot, lambda: REPLY in " ".join(_frame_rows(h.host)))
        assert not h.console.query_one(TraceCallRecoveryCallout).display

        # The chat stays usable: a further send reaches the provider too.
        await _send(h, "And then?")

        assert len(h.calls) == before + 2, "the follow-up send was refused"
        await _until_completed(h)
        assert not h.console.query_one(TraceCallRecoveryCallout).display


async def _assert_reads_blocked(h) -> None:
    """AC#3: the header badge, run chip and Inspect Run line say Blocked."""
    header = h.console._build_console_workbench_state(
        h.console._build_console_control_state(None)
    ).header
    assert (header.status, header.status_label) == ("blocked", "Blocked")
    badge = h.console.query_one(
        "#console-workbench-header #workbench-header-status", Static
    )
    await _until(
        h.pilot,
        lambda: str(badge.render()) == "Blocked",
        what=f"the painted header badge reads {str(badge.render())!r}",
    )
    chip = h.console.query_one("#console-run-chip", Static)
    assert chip.display
    assert str(chip.render()) == f"Run: Blocked — {BLOCKED_REASON}"
    # TASK-33620.5: the hidden compat mode bar reads the chip's copy first, so
    # it says Blocked too. On dev it repeated the run state's own copy ("Trace
    # provenance could not be saved. ...") while the chip said Blocked.
    mode_bar = str(h.console.query_one("#console-mode-bar", Static).renderable)
    assert mode_bar.endswith(f"| Run: Blocked — {BLOCKED_REASON}"), mode_bar
    inspector = h.console._build_console_inspector_state(None)
    assert project_console_send_authority(inspector).run == (
        f"Blocked — {BLOCKED_REASON}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", SIZES)
async def test_provenance_failure_is_visible_everywhere_the_turn_is_shown(
    tmp_path, monkeypatch, size
):
    _block_provenance_admission(monkeypatch)
    async with _mounted_console(tmp_path, monkeypatch, size) as h:
        await _send(h, "Hello")

        assert h.calls == []
        # AC#2: the callout renders at the stuck turn and names the cause.
        callout = h.console.query_one(TraceCallRecoveryCallout)
        await _until(
            h.pilot, lambda: "Trace capture blocked" in " ".join(_frame_rows(h.host))
        )
        assert callout.display
        text = " ".join(" ".join(_frame_rows(h.host)).split())
        assert "This send's trace record could not be saved" in text
        assert "provider was not contacted" in text
        for label in ("Retry capture", "Send without capture", "Cancel send"):
            assert callout.query_one(
                {
                    "Retry capture": "#console-trace-retry",
                    "Send without capture": "#console-trace-send-without",
                    "Cancel send": "#console-trace-cancel",
                }[label],
                Button,
            ).display
        await _assert_reads_blocked(h)

        # AC#4: a later send is parked with the refusal's reason and whole
        # Restore / Discard labels.
        await _send(h, "Second")

        assert h.calls == []
        # AC#3 still holds: that refused send re-stamps the run state (hooks
        # "Initializing hooks.", then "Preparation ended."), which once put
        # the header, run chip and Inspect Run line back to Ready.
        await _assert_reads_blocked(h)
        summary = h.console.query_one("#console-prompt-queue-summary", Static)
        await _until(h.pilot, lambda: summary.region.width > 0)
        assert (
            str(summary.render()) == "Not sent: Last send is blocked; resolve it first"
        )
        restore = h.console.query_one("#console-prompt-queue-manage", Button)
        discard = h.console.query_one("#console-prompt-queue-pause", Button)
        assert (str(restore.label), str(discard.label)) == ("Restore", "Discard")
        shelf_row = next(
            row
            for row in _frame_rows(h.host)
            if "Not sent: Last send is blocked" in row
        )
        assert "Restore" in shelf_row and "Discard" in shelf_row
        assert restore.region.right <= size[0] and discard.region.right <= size[0]
        assert restore.content_region.width >= len("Restore")
        assert discard.content_region.width >= len("Discard")


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["retry", "send-without", "cancel"])
async def test_provenance_failure_recovery_actions_work(tmp_path, monkeypatch, action):
    admission = _block_provenance_admission(monkeypatch)
    async with _mounted_console(tmp_path, monkeypatch, (80, 24)) as h:
        await _send(h, "Hello")
        callout = h.console.query_one(TraceCallRecoveryCallout)
        await _until(h.pilot, lambda: callout.display)
        assert h.calls == []
        admission.blocked = False

        button = callout.query_one(f"#console-trace-{action}", Button)
        button.focus()
        await h.pilot.pause(0.2)
        await h.pilot.press("enter")
        await _settle_workers(h)
        await h.console._sync_native_console_chat_ui()
        await h.pilot.pause(0.3)

        assert not callout.display
        if action == "cancel":
            assert h.calls == []
            assert _assistant_texts(h) == []
        else:
            assert len(h.calls) == 1
            await _until_completed(h)
            assert _assistant_texts(h) == [REPLY]
        assert (
            h.console._build_console_workbench_state(
                h.console._build_console_control_state(None)
            ).header.status_label
            != "Blocked"
        )
        assert not project_console_send_authority(
            h.console._build_console_inspector_state(None)
        ).run.startswith("Blocked")


@pytest.mark.asyncio
async def test_a_send_refused_because_the_chat_was_archived_keeps_its_reason(
    tmp_path, monkeypatch
):
    """AC#4: the runtime's own archive check parks the turn with its reason.

    The send boundary re-reads the saved chat's archive flag before the
    controller runs. A chat archived while its tab stays open is refused
    there, and that refusal's copy must reach the unsent-turn shelf like a
    controller refusal's does, not the generic label.
    """
    async with _mounted_console(tmp_path, monkeypatch, (80, 24)) as h:
        service = ChatConversationService(h.database)
        h.console.app_instance.local_chat_conversation_service = service
        await _send(h, "Hello")
        await _until_completed(h)
        conversation_id = next(
            session.persisted_conversation_id
            for session in h.store.sessions()
            if session.id == h.store.active_session_id
        )
        assert conversation_id
        version = h.database.get_conversation_by_id(conversation_id)["version"]
        archived = service.set_conversations_archived(
            [conversation_id],
            archived=True,
            expected_versions={conversation_id: version},
        )
        assert conversation_id in archived["changed"]
        refusal = await conversation_send_refusal(
            h.console.app_instance, conversation_id
        )
        assert refusal and "archived" in refusal

        await _send(h, "Second")

        assert len(h.calls) == 1, "the archived chat's send reached the provider"
        summary = h.console.query_one("#console-prompt-queue-summary", Static)
        await _until(h.pilot, lambda: summary.region.width > 0)
        assert str(summary.render()).startswith(
            "Not sent: This conversation is archived"
        )
        assert str(summary.render()) == turn_recovery_label(refusal)
        (entry,) = h.console._console_runtime().recoveries_for_session(
            h.store.active_session_id
        )
        assert (entry.draft, entry.reason) == ("Second", refusal)
