"""TASK-40715 AC #2 in the real app: a second chat's Send while the first replies.

A private child builds the real ``TldwCli`` over the stock config template
(only first-run completion, no splash and an OpenAI fixture key are written on
top; the ``full_tools`` case also turns Full agent tools on), mounts it
headless, sends the first chat through the real composer, and holds that
provider call. A second chat then goes through the Console's own pre-send
capture. ``chat_api_call`` is the only replaced boundary.
"""

import json

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run

pytestmark = pytest.mark.bootstrap_profile

BUSY = "Workspace activity is busy; retry after it finishes."

_SCRIPT = r"""
import asyncio, json, os, sys, threading
from pathlib import Path
from Tests import network_guard, real_profile_guard
from Tests.windows_private_fixture_runner import user_fixture_default_owner
network_guard.install()
real_profile_guard.install()
from loguru import logger
logger.remove()


async def exercise():
    import pytest
    from Tests.Chat.test_automatic_provider_budget import response
    from Tests.Performance._stock_private_app_creators import OriginalPrivateAppCreators
    from Tests.UI.test_console_hook_review_send_freeze import _key, _until
    from tldw_chatbook import app as app_source
    from tldw_chatbook.Chat import Chat_Functions
    from tldw_chatbook.Chat import console_ordinary_tree as ordinary
    from tldw_chatbook.UI.Console_Modules.wiring import _precapture_turn_authority

    facts = {"captures": [], "calls": []}
    original = ordinary._capture_maximum

    def capture(tree, **kwargs):
        footprint = original(tree, **kwargs)
        configuration = tree.configuration
        facts["captures"].append({
            "opaque": footprint.opaque,
            "agent_eligible": tree.agent_eligible,
            "personal_native": tree.personal_native,
            "personal_cached": tree.personal_service is not None,
            "other_providers": sorted(tree.other_maximum),
            "mcp_servers": sorted({row[1] for row in configuration.mcp_definition_sources}),
            "mcp_pending": configuration.mcp_capture_pending,
        })
        return footprint

    entered, release = threading.Event(), threading.Event()

    def reply(**kwargs):
        messages = kwargs.get("messages", kwargs.get("messages_payload", ())) or ()
        text = " ".join(
            str(m.get("content")) for m in messages
            if isinstance(m, dict) and m.get("role") == "user"
        )
        facts["calls"].append(text[-40:])
        if "first default chat" in text:
            entered.set()
            assert release.wait(60), "first chat was never released"
        result = response()
        result["choices"][0]["message"]["content"] = "default reply"
        return result

    patch = pytest.MonkeyPatch()
    patch.setattr(ordinary, "_capture_maximum", capture)
    patch.setattr(Chat_Functions, "chat_api_call", reply)
    creators = OriginalPrivateAppCreators(app_source, Path(os.environ["XDG_DATA_HOME"]).parent)
    creators.install()
    try:
        app = app_source.TldwCli()
    except BaseException:
        creators.close()
        raise
    creators.finish_construction(app)
    app._cli_no_splash = True
    runtime = app.console_runtime
    try:
        async with app.run_test(size=(140, 42)):
            assert await _until(lambda: getattr(app, "_initial_screen_pushed", False), 30)
            await asyncio.wait_for(asyncio.shield(app._initial_screen_setup_task), 30)
            screen = runtime.view
            controller = screen._ensure_console_chat_controller()
            store = controller.store
            assert await _until(
                lambda: screen._console_attach_reconciled
                and not screen._console_attach_reconcile_running, 30)
            first = store.ensure_session()
            facts["first_agent_eligible"] = controller._agent_dispatch_is_eligible(first, prefill=None)
            composer = screen._console_composer_or_none()
            composer.load_draft("first default chat")
            composer.screen.set_focus(composer)
            assert await _until(lambda: app.focused is composer, 5)
            _key(app, "enter", "\r")
            assert await _until(entered.is_set, 60), facts
            other = store.create_session(workspace_id=first.workspace_id)
            try:
                prepared = await _precapture_turn_authority(screen, other.id)
                with prepared():
                    configuration = screen._session._build_console_turn_execution_context(other.id)
                    result = await asyncio.wait_for(controller.submit_draft(
                        "second default chat", session_id=other.id,
                        configuration=configuration), 60)
                facts["second"] = [result.accepted, result.visible_copy]
            finally:
                release.set()
            assert await _until(lambda: not runtime._ordinary_turns, 30)
    finally:
        release.set()
        print("FACTS " + json.dumps(facts, sort_keys=True, default=str), flush=True)
        await runtime.dispose()
        creators.retire()
    print("retired and reopened")

with user_fixture_default_owner():
    selected = Path(os.environ["TLDW_CONFIG_PATH"])
    selected.write_text(
        "[first_run]\nsetup_completed=true\n[_first_run]\nsetup_completed=true\n"
        "[splash_screen]\nenabled=false\n"
        '[api_settings.openai]\napi_key="fixture-not-a-real-key"\n'
        '[chat_defaults]\nprovider="openai"\nmodel="gpt-4o-mini"\n'
        + ("[tools]\nfull_agent_tools_enabled=true\n" if sys.argv[2] == "full_tools" else ""),
        encoding="utf-8",
    )
    selected.chmod(0o600)
    asyncio.run(asyncio.wait_for(exercise(), timeout=300))
"""


@pytest.mark.parametrize("surface", ["default", "full_tools"])
def test_default_config_second_chat_reaches_provider_while_first_reply_runs(
    tmp_path, surface
):
    _run(tmp_path, "default_send", surface, script=_SCRIPT, timeout=360)
    log = (tmp_path / "restart-output.log").read_text(
        encoding="utf-8", errors="replace"
    )
    facts = json.loads(log.split("FACTS ", 1)[1].splitlines()[0])
    first = facts["captures"][0]
    assert facts["first_agent_eligible"] and first["agent_eligible"], facts
    if surface == "default":
        # Nothing the light surface offers is undeclarable: both chats run.
        assert not first["opaque"], facts
        assert facts["second"][0] is True, facts
        assert any("second default chat" in call for call in facts["calls"]), facts
    else:
        # Full agent tools still count Canvas and chatbook's MCP tools up front.
        assert first["opaque"], facts
        assert "canvas" in first["other_providers"], facts
        assert "builtin:tldw_chatbook" in first["mcp_servers"], facts
        assert facts["second"] == [False, BUSY], facts
