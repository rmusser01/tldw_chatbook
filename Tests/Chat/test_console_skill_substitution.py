"""Tests for the Task 10 provider-payload skill substitution rule.

Spec: `Docs/superpowers/specs/2026-07-14-skills-library-console-design.md`
"Invocation semantics" §5 (the substitution rule) -- when building a
provider payload, the turn's TRIGGERING user message (the final ``role ==
"user"`` message) is re-resolved, re-trust-checked, and re-rendered fresh at
build time if it parses as a resolvable skill command. Earlier history
messages (including earlier raw skill commands) are never substituted.
"""

import pytest

from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Skills_Interop.skill_trust_models import SkillTrustBlockedError


class _Skills:
    def __init__(self, mode="inline", raise_trust=False):
        self._mode = mode
        self._raise = raise_trust
        self.executions = []

    async def get_context(self, *, mode="local"):
        return {"available_skills": [{"name": "code-review", "description": "d",
                                      "user_invocable": True, "trust_blocked": False}],
                "blocked_skills": []}

    async def execute_skill(self, name, *, mode="local", args=None):
        self.executions.append((name, args))
        if self._raise:
            raise SkillTrustBlockedError(skill_name=name, reason_code="skill_modified",
                                         trust_status="quarantined_modified")
        return {"skill_name": name, "rendered_prompt": f"RENDERED[{args}]",
                "allowed_tools": None, "execution_mode": self._mode, "fork_output": None}


class _ReadyResolution:
    ready = True
    provider = "llama_cpp"
    model = "m"
    visible_copy = ""


class _RecordingGateway:
    """Streams one chunk and records the exact payload the provider saw."""

    def __init__(self, *, fail=False):
        self.fail = fail
        self.payloads = []

    async def resolve_for_send(self, selection):
        return _ReadyResolution()

    async def stream_chat(self, resolution, messages):
        self.payloads.append(messages)
        yield "reply"
        if self.fail:
            raise RuntimeError("stream failed")


def _controller(skills):
    store = ConsoleChatStore()
    return ConsoleChatController(store=store, provider_gateway=object(),
                                 provider="llama_cpp", model="m",
                                 skills_service=skills), store


@pytest.mark.asyncio
async def test_inline_substitutes_final_user_message_only():
    controller, _store = _controller(_Skills("inline"))
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "/code-review fix it"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert refuse is None
    assert out[-1] == {"role": "user", "content": "RENDERED[fix it]"}
    assert out[1] == {"role": "user", "content": "earlier"}    # history preserved


@pytest.mark.asyncio
async def test_fork_drops_history_keeps_system():
    controller, _store = _controller(_Skills("fork"))
    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "earlier"},
            {"role": "user", "content": "/code-review go"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert refuse is None
    assert out == [{"role": "system", "content": "sys"},
                   {"role": "user", "content": "RENDERED[go]"}]


@pytest.mark.asyncio
async def test_non_skill_final_message_unchanged():
    controller, _store = _controller(_Skills("inline"))
    msgs = [{"role": "user", "content": "just a question"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert out == msgs and refuse is None


@pytest.mark.asyncio
async def test_edited_skill_refuses_at_build():
    controller, _store = _controller(_Skills(raise_trust=True))
    msgs = [{"role": "user", "content": "/code-review go"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert out == msgs
    assert refuse == ('Skill "code-review" isn\'t trusted (skill_modified) — '
                      "review and approve it in Library ▸ Skills before running it.")


@pytest.mark.asyncio
async def test_no_skills_service_is_a_noop():
    store = ConsoleChatStore()
    controller = ConsoleChatController(store=store, provider_gateway=object(),
                                       provider="llama_cpp", model="m")
    msgs = [{"role": "user", "content": "/code-review go"}]
    out, refuse = await controller._apply_skill_substitution(msgs)
    assert out == msgs and refuse is None


# ---------------------------------------------------------------------------
# End-to-end through the real send paths (the rule wired at build sites).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_submit_sends_rendered_payload_but_stores_raw_command():
    skills = _Skills("inline")
    store = ConsoleChatStore()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway,
                                       provider="llama_cpp", model="m",
                                       skills_service=skills)

    result = await controller.submit_draft("/code-review fix it")

    assert result.accepted is True
    # Provider saw the rendered body as the triggering turn.
    assert gateway.payloads[-1][-1] == {"role": "user", "content": "RENDERED[fix it]"}
    # The store (transcript + persistence source) keeps the RAW command.
    messages = store.messages_for_session(store.active_session_id)
    user_rows = [m for m in messages if m.role is ConsoleMessageRole.USER]
    assert user_rows[-1].content == "/code-review fix it"


@pytest.mark.asyncio
async def test_fork_survives_retry_by_re_rendering_fresh():
    skills = _Skills("fork")
    store = ConsoleChatStore()
    failing = _RecordingGateway(fail=True)
    controller = ConsoleChatController(store=store, provider_gateway=failing,
                                       provider="llama_cpp", model="m",
                                       skills_service=skills)
    await controller.submit_draft("/code-review go")
    messages = store.messages_for_session(store.active_session_id)
    failed = next(m for m in reversed(messages)
                  if m.role is ConsoleMessageRole.ASSISTANT and m.status == "failed")

    retry_gateway = _RecordingGateway()
    controller.provider_gateway = retry_gateway
    result = await controller.retry_message(failed.id)

    assert result.accepted is True
    # The retry re-rendered fresh: execute_skill called once per build.
    assert skills.executions == [("code-review", "go"), ("code-review", "go")]
    # Fork = clean context on the retry too: rendered turn only (no system
    # prompt configured, and the failed assistant row is skipped upstream).
    assert retry_gateway.payloads[-1] == [{"role": "user", "content": "RENDERED[go]"}]


@pytest.mark.asyncio
async def test_submit_refusal_appends_system_row_and_aborts_without_provider_call():
    skills = _Skills(raise_trust=True)
    store = ConsoleChatStore()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway,
                                       provider="llama_cpp", model="m",
                                       skills_service=skills)

    result = await controller.submit_draft("/code-review go")

    assert result.accepted is False
    assert gateway.payloads == []  # the run never reached the provider
    messages = store.messages_for_session(store.active_session_id)
    # Raw command persists (honest record), followed by the refuse system row.
    assert messages[-2].role is ConsoleMessageRole.USER
    assert messages[-2].content == "/code-review go"
    assert messages[-1].role is ConsoleMessageRole.SYSTEM
    assert messages[-1].content == (
        'Skill "code-review" isn\'t trusted (skill_modified) — '
        "review and approve it in Library ▸ Skills before running it."
    )
    assert result.visible_copy == messages[-1].content
    # No dangling empty assistant row was created for the aborted turn.
    assert all(m.role is not ConsoleMessageRole.ASSISTANT for m in messages)
    # The refusal is terminal-blocked, not a wedge: a follow-up send works.
    assert controller.run_state.status is ConsoleRunStatus.BLOCKED
    follow_up = await controller.submit_draft("plain question")
    assert follow_up.accepted is True


@pytest.mark.asyncio
async def test_submit_refusal_never_invokes_accepted_hook():
    """Qodo finding 3 (PR #636 bot review): the accepted-hook must not fire
    before the skill substitution/trust check settles. Previously
    `submit_draft` called `_notify_submission_accepted()` right after the
    USER row was appended -- BEFORE `_apply_skill_substitution` even ran --
    so a refused/untrusted skill still fired the hook. In the real
    ChatScreen that hook is the sole consume point for a staged
    "driving this turn" TOOL marker, so a refused submit still appended a
    marker claiming the skill ran, right before the refuse row."""
    skills = _Skills(raise_trust=True)
    store = ConsoleChatStore()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway,
                                       provider="llama_cpp", model="m",
                                       skills_service=skills)
    accepted_calls = []
    controller.on_submission_accepted = lambda: accepted_calls.append(True)

    result = await controller.submit_draft("/code-review go")

    assert result.accepted is False
    assert accepted_calls == []


@pytest.mark.asyncio
async def test_submit_success_still_invokes_accepted_hook_before_assistant_row():
    """The Qodo-3 reorder must not regress the successful path: the hook
    still fires exactly once, and still fires strictly before the
    ASSISTANT placeholder is appended (store order stays
    [USER, ..., ASSISTANT])."""
    skills = _Skills("inline")
    store = ConsoleChatStore()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway,
                                       provider="llama_cpp", model="m",
                                       skills_service=skills)
    assistant_rows_seen_at_hook_time = []

    def _on_accepted():
        session_id = store.active_session_id
        messages = store.messages_for_session(session_id) if session_id else []
        assistant_rows_seen_at_hook_time.append(
            [m for m in messages if m.role is ConsoleMessageRole.ASSISTANT]
        )

    controller.on_submission_accepted = _on_accepted

    result = await controller.submit_draft("/code-review go")

    assert result.accepted is True
    assert assistant_rows_seen_at_hook_time == [[]]


@pytest.mark.asyncio
async def test_regenerate_refusal_after_skill_edit_keeps_prior_answer():
    skills = _Skills("inline")
    store = ConsoleChatStore()
    gateway = _RecordingGateway()
    controller = ConsoleChatController(store=store, provider_gateway=gateway,
                                       provider="llama_cpp", model="m",
                                       skills_service=skills)
    await controller.submit_draft("/code-review go")
    messages = store.messages_for_session(store.active_session_id)
    assistant = next(m for m in reversed(messages)
                     if m.role is ConsoleMessageRole.ASSISTANT)
    assert assistant.content == "reply"

    skills._raise = True  # the skill was edited since; now trust-blocked
    result = await controller.regenerate_message(assistant.id)

    assert result.accepted is False
    assert "isn't trusted (skill_modified)" in result.visible_copy
    # Only the first (pre-edit) send reached the provider.
    assert len(gateway.payloads) == 1
    # The good prior answer is untouched by the refused regenerate.
    assert store.get_message(assistant.id).content == "reply"
    assert store.get_message(assistant.id).status == "complete"
