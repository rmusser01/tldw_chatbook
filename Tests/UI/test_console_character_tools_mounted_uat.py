"""Mounted Console UAT for the character_* tools (TASK-32954 Task 8).

Modelled on ``test_console_watchlists_mounted_uat.py``: only model planning is
scripted; the controller, LocalToolProvider, permission gate, approval card,
CharacterToolService and ChaChaNotes DB are the real ones. The avatar backend
is the one seam faked (``character_avatar.generate_avatar_bytes``).
"""

from __future__ import annotations

import base64
import copy
import json
import time
from dataclasses import replace

import pytest
from textual.widgets import Button

from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    _visible_text,
)
from tldw_chatbook import config as app_config
from tldw_chatbook.Backup_Recovery.chat_source_participants import (
    build_persona_service,
)
from tldw_chatbook.Character_Chat import character_avatar
from tldw_chatbook.Character_Chat.character_events import CharacterCardChanged
from tldw_chatbook.Chat.console_library_destination import (
    resolve_console_destination,
)
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderResolution
from tldw_chatbook.Chat.provider_setup_persistence import persist_provider_setup
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Tools import character_tool_service
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Wizards import first_run_setup_state as wizard_state
from tldw_chatbook.Widgets.Console import ConsoleComposerBar

#: A real 1x1 PNG, so the local service's image validation accepts it.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
_DESCRIPTION = "CARD-FIELD-TEXT-9C2E a lighthouse keeper who talks to gulls."
_DONE_MARKER = "CHARACTER-UAT-DONE-41B7"
_TOOLS = ("character_search", "character_get", "character_save")


def _tool_fence(name: str, arguments: dict) -> str:
    return (
        "```tool_call\n"
        + json.dumps({"name": name, "arguments": arguments})
        + "\n```"
    )


class _ScriptedCharacterGateway:
    """Script only model planning; every tool and durable effect stays real."""

    def __init__(self) -> None:
        self.calls: list[list[dict]] = []
        self.stage = 0

    def cached_context_window(self, settings):
        """The real gateway's offline metadata fallback (the Console screen
        reads it at compose; the watchlists gateway predates that call)."""
        from tldw_chatbook.Utils.token_counter import resolve_context_window

        return resolve_context_window(settings.provider, settings.model or "")

    async def resolve_for_send(self, selection):
        """Resolve through the same typed destination contract as production."""
        resolution = ConsoleProviderResolution(
            provider=selection.provider,
            base_url=selection.base_url or "http://127.0.0.1:8791",
            model=selection.explicit_model
            or selection.configured_model
            or "scripted-mounted-model",
            ready=True,
            execution_key=selection.provider,
        )
        return replace(
            resolution,
            resolved_destination=resolve_console_destination(resolution),
        )

    async def stream_chat(self, _resolution, messages, tools=None, **_kwargs):
        del tools
        self.calls.append(copy.deepcopy(messages))
        self.stage += 1
        if self.stage == 1:
            yield _tool_fence("find_tools", {"query": "character cards"})
        elif self.stage == 2:
            yield _tool_fence(
                "load_tools", {"ids": [f"local:{name}" for name in _TOOLS]}
            )
        elif self.stage == 3:
            yield _tool_fence("character_search", {"query": "Maren"})
        elif self.stage == 4:
            yield _tool_fence(
                "character_save",
                {
                    "name": "Maren",
                    "description": _DESCRIPTION,
                    "avatar": {"source": "generate"},
                },
            )
        else:
            yield f"Saved Maren. {_DONE_MARKER}"


@pytest.mark.asyncio
@private_profile_test
async def test_mounted_console_creates_a_character_through_an_approved_save(
    request, tmp_path, monkeypatch
):
    prompts: list[str] = []

    def fake_generate(prompt: str) -> bytes:
        prompts.append(prompt)
        return _PNG

    monkeypatch.setattr(character_avatar, "generate_avatar_bytes", fake_generate)
    monkeypatch.setattr(
        character_tool_service, "image_backend_configured", lambda: "test-backend"
    )

    loaded = app_config.load_cli_config_and_ensure_existence(force_reload=True)
    mutation = wizard_state.build_first_run_provider_commit(
        wizard_state.FirstRunProviderDraft(
            provider="llama_cpp",
            endpoint="http://127.0.0.1:8791/v1/chat/completions",
            credential=wizard_state.ProviderCredentialDraft("none", "", 0),
        ),
        "mounted-persisted-model",
        loaded,
    )
    assert persist_provider_setup(mutation).fully_applied is True

    app = _build_test_app(configured_default="chat")
    profile = tmp_path / "profile"
    profile.mkdir()
    db = CharactersRAGDB(profile / "chachanotes.sqlite", client_id="mounted-uat")
    app.chachanotes_db = db
    app.local_character_persona_service = build_persona_service(db)
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "active-conversation-conflict"

    gateway = _ScriptedCharacterGateway()
    app.console_provider_gateway_factory = lambda: gateway
    app.unified_mcp_service.set_global_default("allow")

    posted: list[CharacterCardChanged] = []
    real_post = app.post_message

    def recording_post(message):
        if isinstance(message, CharacterCardChanged):
            posted.append(message)
        return real_post(message)

    app.post_message = recording_post

    async with app.run_test(size=(180, 50)) as pilot:
        console = app.screen
        deadline = time.monotonic() + 10.0
        while not (isinstance(console, ChatScreen) and console.is_mounted):
            assert time.monotonic() < deadline, "ChatScreen did not mount"
            await pilot.pause(0.02)
            console = app.screen
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        composer.load_draft("Make me a lighthouse-keeper character with an avatar.")
        console.query_one("#console-send-message", Button).press()

        approved_round_ids: set[str] = set()
        approved_names: set[str] = set()
        save_card_text = ""
        save_calls: list = []
        deadline = time.monotonic() + 40.0
        while time.monotonic() < deadline:
            console = app.screen
            pending_approval = (
                console._task_resume_state.pending_approval
                if isinstance(console, ChatScreen)
                else None
            )
            round_id = (
                str(pending_approval.get("round_id", ""))
                if isinstance(pending_approval, dict)
                else ""
            )
            call_names = {
                str(call.get("llm_name", "") or call.get("name", ""))
                for call in (
                    pending_approval.get("calls", [])
                    if isinstance(pending_approval, dict)
                    else []
                )
                if isinstance(call, dict)
            }
            matching_button = None
            matched_name = ""
            if isinstance(console, ChatScreen):
                for row in console.query(".approval-row"):
                    header = " ".join(
                        str(item.renderable)
                        for item in row.query(".approval-row-header")
                    )
                    names = [name for name in call_names if name and name in header]
                    buttons = list(row.query(".approval-row-fast-approve"))
                    if names and buttons:
                        matched_name = names[0]
                        matching_button = buttons[0]
                        if matched_name == "character_save":
                            save_calls = copy.deepcopy(
                                [c for c in pending_approval.get("calls", [])
                                 if isinstance(c, dict)]
                            )
                            save_card_text = " ".join(
                                str(item.renderable) for item in row.query("Static")
                            )
                        break
            if (
                round_id
                and round_id not in approved_round_ids
                and matching_button is not None
            ):
                approved_round_ids.add(round_id)
                approved_names.add(matched_name)
                matching_button.press()
            if gateway.stage >= 5 and _DONE_MARKER in _visible_text(console):
                break
            await pilot.pause(0.03)
        else:
            raise AssertionError(
                f"mounted Console loop did not settle: stage={gateway.stage}, "
                f"text={_visible_text(app.screen)!r}"
            )
        await pilot.pause()

    # The save asked (mutates floors to ask even under a global Allow) and its
    # card showed the summary -- name and field sizes -- never the field text.
    # The card clips each value, so the untruncated source it renders (the
    # pending call's ``arguments``) is checked too.
    assert "character_save" in approved_names
    assert "Maren" in save_card_text
    assert '"changed_fields"' in save_card_text
    assert "CARD-FIE" not in save_card_text
    assert [call["tool_name"] for call in save_calls] == ["character_save"]
    card_arguments = save_calls[0]["arguments"]
    assert card_arguments["changed_fields"] == {
        "name": "5 chars",
        "description": f"{len(_DESCRIPTION)} chars",
    }
    assert "CARD-FIELD-TEXT-9C2E" not in json.dumps(card_arguments)

    # The tool results the model saw: search ran, save reported the avatar.
    tool_text = "\n".join(
        str(message.get("content", "")) for message in gateway.calls[-1]
    )
    assert "character_search" in tool_text
    assert '"status":"saved"' in tool_text
    assert '"avatar":"saved"' in tool_text

    # One DB row, text and image, written in a single version.
    row = db.get_character_card_by_name("Maren")
    assert row is not None
    assert row["description"] == _DESCRIPTION
    assert bytes(row["image"]) == _PNG
    assert row["version"] == 1
    assert len(prompts) == 1

    # Personas was told.
    assert [message.character_id for message in posted] == [row["id"]]
