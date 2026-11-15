"""A turn's claim widens to OPAQUE only when it uses an undeclarable effect.

TASK-40715: an agent turn that merely could spawn a sub-agent no longer
serializes every other chat; the spawn itself widens the claim, and a spawn
that would collide with another running turn is refused, never waited on.
"""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.Agents.test_agent_service import fence
from Tests.Chat.test_console_ordinary_tree import (
    ordinary_host as ordinary_host,  # noqa: PLC0414 - expose pytest fixture
)
from Tests.Chat.test_console_reference_custody import (
    reference_host as reference_host,  # noqa: PLC0414 - expose pytest fixture
)
from Tests.Chat.test_console_reference_custody import until
from Tests.Chat.test_console_skill_substitution import _RecordingGateway
from tldw_chatbook.Agents.agent_models import SPAWN_TOOL_NAME
from tldw_chatbook.Agents.execution_capacity import CapacityRefused
from tldw_chatbook.Agents.ordinary_tree import OrdinaryTurn
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

BUSY = "Workspace activity is busy; retry after it finishes."


def _chat(h, **kwargs):
    # Project instructions would stop an agent run at their consent prompt.
    return h.store.create_session(
        workspace_id=h.session.workspace_id,
        project_instruction_state=ProjectInstructionControlState.legacy_disabled(),
        **kwargs,
    )


def _last_user(messages):
    return next(
        (str(m.get("content")) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )


class _Held:
    """Holds every provider call whose last user message names ``marker``."""

    def __init__(self, marker):
        self.marker = marker
        self.entered, self.release = threading.Event(), threading.Event()

    async def hold(self, messages):
        if self.marker in _last_user(messages):
            self.entered.set()
            assert await asyncio.to_thread(self.release.wait, 30)


def _install(h, gateway):
    h.controller.provider_gateway = gateway
    h.bridge._gateway = gateway


def _turn(h, session):
    return next(t for t in h.runtime._ordinary_turns if t.session is session)


@pytest.mark.parametrize("second_kind", ["agent", "character"])
async def test_other_chat_send_reaches_provider_while_first_reply_is_held(
    ordinary_host, second_kind
):
    h = ordinary_host
    held = _Held("first chat")

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            await held.hold(messages)
            yield "reply"

    gateway = Gateway()
    _install(h, gateway)
    first_chat = _chat(h)
    assert h.controller._agent_dispatch_is_eligible(first_chat, prefill=None)
    first = asyncio.create_task(
        h.controller.submit_draft("first chat", session_id=first_chat.id)
    )
    h.tasks.append(first)
    if second_kind == "character":
        card = int(h.store.persistence.db.add_character_card({"name": "Alraune"}))
        other = _chat(
            h, assistant_kind="character", assistant_id=str(card), character_id=card
        )
    else:
        other = _chat(h)
    try:
        await until(lambda: held.entered.is_set() or first.done())
        assert held.entered.is_set(), (
            first.result() if first.done() else "no provider call"
        )
        assert not _turn(h, first_chat).footprint.opaque
        result = await h.controller.submit_draft("second chat", session_id=other.id)
        assert result.accepted, result.visible_copy
        await until(lambda: len(gateway.payloads) == 2)
    finally:
        held.release.set()
        await first


async def test_spawn_widens_the_claim_and_other_chats_are_busy_until_it_retires(
    ordinary_host, monkeypatch
):
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    h = ordinary_host
    entered, release = threading.Event(), threading.Event()
    attach = OrdinaryTurn.attach

    def held_attach(tree, owner):
        attach(tree, owner)
        # The child has joined the widened claim. Refuse it before its row,
        # so this host needs no child model run (not under test here).
        entered.set()
        assert release.wait(20)
        raise CapacityRefused("test_stops_child")

    monkeypatch.setattr(OrdinaryTurn, "attach", held_attach)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            if len(self.payloads) == 1:
                yield fence(SPAWN_TOOL_NAME, {"task": "CHILD-TASK read the notes"})
            else:
                yield "parent done"

    _install(h, Gateway())
    spawning = _chat(h)
    task = asyncio.create_task(
        h.controller.submit_draft("spawn chat", session_id=spawning.id)
    )
    h.tasks.append(task)
    other = _chat(h)
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), task.result() if task.done() else "no child"
        assert _turn(h, spawning).footprint.opaque, "spawn did not widen the claim"
        result = await h.controller.submit_draft("other chat", session_id=other.id)
        assert not result.accepted and result.visible_copy == BUSY
    finally:
        release.set()
    assert (await task).accepted
    await until(lambda: not h.runtime._ordinary_turns)
    result = await h.controller.submit_draft("other chat again", session_id=other.id)
    assert result.accepted, "the widened claim outlived its turn"


async def test_spawn_conflicting_with_another_running_turn_is_refused_not_awaited(
    ordinary_host, monkeypatch
):
    monkeypatch.setenv("TLDW_AGENTS_AUTOWAKE_ENABLED", "false")
    h = ordinary_host
    held = _Held("held chat")
    spawn_payloads = []
    attached = []
    attach = OrdinaryTurn.attach

    def guarded_attach(tree, owner):
        attach(tree, owner)
        # Reaching here means the child joined a claim it should not have;
        # stop it before it runs so a regression fails instead of hanging.
        attached.append(tree.footprint.opaque)
        raise CapacityRefused("test_stops_child")

    monkeypatch.setattr(OrdinaryTurn, "attach", guarded_attach)

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            await held.hold(messages)
            if "spawn chat" in str(messages):
                spawn_payloads.append(messages)
                if len(spawn_payloads) == 1:
                    yield fence(SPAWN_TOOL_NAME, {"task": "CHILD-TASK read it"})
                    return
            yield "done without a child"

    _install(h, Gateway())
    blocker = _chat(h)
    blocking = asyncio.create_task(
        h.controller.submit_draft("held chat", session_id=blocker.id)
    )
    h.tasks.append(blocking)
    spawning = _chat(h)
    try:
        await until(lambda: held.entered.is_set() or blocking.done())
        assert held.entered.is_set()
        result = await asyncio.wait_for(
            h.controller.submit_draft("spawn chat", session_id=spawning.id), 30
        )
        assert result.accepted, result.visible_copy
        assert not held.release.is_set() and not blocking.done()
    finally:
        held.release.set()
        await blocking
    assert attached == [], "the spawn joined a claim it could not widen"
    assert len(spawn_payloads) == 2, "the spawning turn did not carry on"
    assert "spawn_subagent is unavailable while another chat's turn is running" in str(
        spawn_payloads[1]
    )
    rows = h.database.list_runs(spawning.persisted_conversation_id)
    assert [row["agent_kind"] for row in rows] == ["primary"]


class _ScriptSkills:
    """Skills service whose granted script run records the claim it ran under."""

    def __init__(self, record):
        self.local_service = SimpleNamespace(
            trust_service=SimpleNamespace(script_execution_granted=lambda name: True)
        )
        self.record = record

    def enforce_run_script(self):
        return None

    async def get_context(self, *, mode="local"):
        return {"available_skills": [], "blocked_skills": []}

    async def describe_skill_script(self, skill_name, script_path, *, mode=None):
        from tldw_chatbook.Skills_Interop.local_skills_service import ScriptPlan

        return ScriptPlan(skill_name, script_path, "interpreter", "python3", False)

    async def run_skill_script(self, *args, **kwargs):
        self.record()
        raise RuntimeError("recorded")


@pytest.mark.parametrize("contended", [False, True])
@pytest.mark.parametrize("tool", ["install_skill", "run_skill_script"])
async def test_skill_effect_widens_the_claim_or_is_refused_while_another_turn_runs(
    ordinary_host, monkeypatch, tool, contended
):
    import tldw_chatbook.Skills_Interop.skill_remote_fetch as fetch
    from Tests.Chat.test_console_agent_bridge import _install_skills_service

    h = ordinary_host
    effects = []
    acting = _chat(h)

    def record():
        effects.append(_turn(h, acting).footprint.opaque)

    if tool == "install_skill":

        async def install(url, *, scope_service, **kwargs):
            record()
            return {"name": "demo"}

        monkeypatch.setattr(fetch, "install_skill_from_url", install)
        h.bridge._skills_service = _install_skills_service()
        h.controller.request_skill_install_confirm = lambda url, session_id=None: True
        call = fence(tool, {"url": "https://github.com/o/r"})
    else:
        h.bridge._skills_service = _ScriptSkills(record)
        h.controller.set_pending_skill_script = lambda *args, **kwargs: None
        call = fence(tool, {"skill": "demo", "script": "scripts/x.py", "args": []})
    held = _Held("held chat")
    acting_payloads = []

    class Gateway(_RecordingGateway):
        async def stream_chat(self, resolution, messages, **kwargs):
            self.payloads.append(messages)
            await held.hold(messages)
            if "acting chat" in str(messages):
                acting_payloads.append(messages)
                if len(acting_payloads) == 1:
                    yield call
                    return
            yield "done"

    _install(h, Gateway())
    blocking = None
    if contended:
        blocker = _chat(h)
        blocking = asyncio.create_task(
            h.controller.submit_draft("held chat", session_id=blocker.id)
        )
        h.tasks.append(blocking)
        await until(lambda: held.entered.is_set() or blocking.done())
        assert held.entered.is_set()
    try:
        result = await asyncio.wait_for(
            h.controller.submit_draft("acting chat", session_id=acting.id), 30
        )
        assert result.accepted, result.visible_copy
    finally:
        held.release.set()
        if blocking is not None:
            await blocking
    assert len(acting_payloads) == 2, "the acting turn did not carry on"
    if contended:
        assert effects == []
        assert f"{tool} is unavailable while another chat's turn is running" in (
            str(acting_payloads[1])
        )
    else:
        assert effects == [True], f"{tool} ran without widening the claim"
