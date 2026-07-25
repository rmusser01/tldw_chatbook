"""End-to-end: agent tool call -> real services -> real subprocess.

Exercises the whole trust-gated skill-script-execution chain built across
Tasks 1-7 of the skills-script-execution SDD plan (see
``.superpowers/sdd/task-8-brief.md``) with NOTHING faked except the human
confirm callback:

- A REAL ``ServicePolicyEnforcer`` bound to a REAL ``PolicyEngine`` (over the
  production ``CAPABILITY_REGISTRY``, or a copy with the ``skills.run_script.
  launch.local`` row disabled), wired onto BOTH ``LocalSkillsService`` and
  ``SkillsScopeService`` -- mirroring ``Tests/Skills/conftest.py``'s
  ``script_scope_service_denied`` fixture (an enforcer-less scope service
  silently no-ops every policy check, which would make a "denied" test pass
  vacuously; see that fixture's own docstring for the same lesson applied to
  the install layer's ``test_e2e_install_skill_from_github_tree_url_real_
  services``).
- A REAL ``SkillTrustService`` (unlock + bootstrap + approve), a REAL
  ``LocalSkillsService``/``SkillsScopeService`` pair, and a real skill
  directory on disk with real scripts.
- The REAL ``run_skill_script_tool`` closure ``console_agent_bridge.
  run_reply`` builds -- captured (not reimplemented) by intercepting the
  ``AgentService(...)`` construction site, the exact technique ``Tests/Chat/
  test_console_skill_script_confirm.py``'s ``bridge_closure_env`` fixture
  uses (that file's fake-scope-service version is the sibling of this
  fixture's real-scope-service one).

``ConsoleAgentBridge.run_reply`` is SYNCHRONOUS -- it owns its own event loop
via ``asyncio.run()`` internally (see its own docstring), so every test here
is a plain ``def``, never ``async def``/``@pytest.mark.asyncio``, matching
every other ``bridge.run_reply(...)`` call across ``Tests/Chat/test_console_
agent_bridge.py`` and the sibling install-layer e2e tests in ``Tests/Skills/
test_skill_remote_fetch.py``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

import tldw_chatbook.Chat.console_agent_bridge as console_agent_bridge_module
from tldw_chatbook.Agents.agent_models import ToolResult
from tldw_chatbook.Agents.agent_service import AgentService as _RealAgentService
from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
from tldw_chatbook.Skills_Interop.local_skills_service import LocalSkillsService
from tldw_chatbook.Skills_Interop.skill_trust_service import SkillTrustService
from tldw_chatbook.Skills_Interop.skill_trust_store import (
    FileSkillTrustGenerationMarkerStore,
    SkillTrustStore,
)
from tldw_chatbook.Skills_Interop.skills_scope_service import SkillsScopeService
from tldw_chatbook.runtime_policy.engine import PolicyEngine
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState

_RUN_SCRIPT_ACTION_ID = "skills.run_script.launch.local"
_SKILL_NAME = "demo-skill"
_HELLO_SCRIPT = "print('hello from a real skill script')"


class _PlainTextGateway:
    """Provider gateway that streams one plain-text reply, no tool calls.

    Used only to drive ``run_reply`` far enough to build and hand off the
    real ``run_skill_script_tool`` closure to the intercepted ``AgentService``
    constructor below -- the scripted turn never enters the agent loop's
    tool-call path, so no script executes during capture.
    """

    async def stream_chat(self, resolution, messages, tools=None):
        yield "ok"


@dataclass
class _E2EEnv:
    """Handle for driving the REAL ``run_skill_script`` closure end to end.

    Attributes:
        tool: The exact ``run_skill_script_tool`` closure ``run_reply``
            built, callable as ``tool(skill_name, script_path, args) ->
            ToolResult``.
        confirm_calls: Every payload dict passed to the (faked) human
            confirm callback, in call order.
        marker_path: Path ``scripts/marker.py`` writes to when it actually
            runs -- used to prove a denied confirm never executes it.
    """

    tool: Callable[[str, str, list], ToolResult]
    confirm_calls: list[dict[str, Any]]
    marker_path: Path
    _trust_service: Any
    _skill_dir: Path

    def mutate_script(self, new_content: str) -> None:
        """Overwrite ``scripts/hello.py`` with new content.

        Mirrors a user editing a trusted skill's bundle: the file's on-disk
        fingerprint changes immediately (invalidating any digest-pinned
        script grant), and the skill quarantines (``quarantined_modified``)
        until ``retrust`` re-approves the new content.

        Args:
            new_content: The new full source of ``scripts/hello.py``.
        """
        (self._skill_dir / "scripts" / "hello.py").write_text(
            new_content, encoding="utf-8"
        )

    def retrust(self) -> None:
        """Re-approve the skill's current on-disk content.

        Required before the closure can reach the confirm/run steps again
        after ``mutate_script`` -- an un-retrusted mutation leaves the skill
        quarantined, which refuses (via ``describe_skill_script``) before
        any prompt, a different failure mode than "re-prompts".
        """
        self._trust_service.trust_current_skill(_SKILL_NAME, audit_event="e2e_retrust")


@pytest.fixture
def e2e_bridge_env(tmp_path, monkeypatch) -> Callable[..., _E2EEnv]:
    """Factory fixture: a real service stack plus the real closure it builds.

    Each call builds a brand-new ``SkillTrustService``/``LocalSkillsService``/
    ``SkillsScopeService``/``ServicePolicyEnforcer`` stack rooted under a
    fresh subdirectory of ``tmp_path`` (so multiple calls within one test
    never share state), writes a real ``demo-skill`` bundle with ``scripts/
    hello.py`` and ``scripts/marker.py`` on disk, then drives ONE real
    ``ConsoleAgentBridge.run_reply`` call (a plain-text scripted turn, no
    tool call) to capture the exact ``run_skill_script_tool`` closure it
    constructs. Callers drive that closure directly and repeatedly via
    ``env.tool(...)`` -- exercising the real implementation, never a
    reimplementation of it.

    Args:
        confirm: The dict the faked human confirm callback returns on every
            call, e.g. ``{"allow": True, "remember": False}``.
        policy_enabled: When False, the ``skills.run_script.launch.local``
            capability row is swapped for a disabled copy before the
            ``PolicyEngine`` is built -- proving policy denial through a
            REAL engine evaluation, not merely an absent enforcer (see this
            module's docstring).
        trusted: When False, the skill directory is written but never
            approved (``bootstrap_trust`` runs first, over an empty store,
            exactly like ``test_e2e_install_skill_from_github_tree_url_
            real_services`` in ``test_skill_remote_fetch.py``), so it is
            unambiguously trust-pending rather than merely unreviewed.

    Returns:
        A factory that returns an ``_E2EEnv`` handle.
    """

    def _make(
        *,
        confirm: dict[str, bool],
        policy_enabled: bool = True,
        trusted: bool = True,
    ) -> _E2EEnv:
        trust_service = SkillTrustService(
            skills_dir=tmp_path / "skills",
            trust_store=SkillTrustStore(
                store_dir=tmp_path / "trust",
                marker_store=FileSkillTrustGenerationMarkerStore(
                    tmp_path / "marker.json"
                ),
            ),
        )
        trust_service.unlock_with_passphrase("e2e-passphrase", salt=b"7" * 32)
        trust_service.bootstrap_trust()  # baseline: empty store

        skill_dir = tmp_path / "skills" / _SKILL_NAME
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: demo\n---\nBody.\n",
            encoding="utf-8",
        )
        (skill_dir / "scripts" / "hello.py").write_text(
            _HELLO_SCRIPT, encoding="utf-8"
        )
        marker_path = tmp_path / "marker-ran.txt"
        (skill_dir / "scripts" / "marker.py").write_text(
            f"from pathlib import Path\nPath({str(marker_path)!r}).write_text('ran')\n",
            encoding="utf-8",
        )
        if trusted:
            trust_service.trust_current_skill(_SKILL_NAME, audit_event="e2e_setup")

        registry = CAPABILITY_REGISTRY
        if not policy_enabled:
            registry = dict(CAPABILITY_REGISTRY)
            registry[_RUN_SCRIPT_ACTION_ID] = dataclasses.replace(
                registry[_RUN_SCRIPT_ACTION_ID], enabled=False
            )
        policy_enforcer = ServicePolicyEnforcer(
            state_provider=lambda: RuntimeSourceState(active_source="local"),
            engine=PolicyEngine(registry),
        )
        local_service = LocalSkillsService(
            store_dir=tmp_path,
            trust_service=trust_service,
            policy_enforcer=policy_enforcer,
        )
        scope_service = SkillsScopeService(
            local_service=local_service,
            server_service=None,
            policy_enforcer=policy_enforcer,
        )

        confirm_calls: list[dict[str, Any]] = []

        def confirm_cb(payload: dict[str, Any]) -> dict[str, bool]:
            confirm_calls.append(payload)
            return confirm

        captured: dict[str, Any] = {}
        real_agent_service = console_agent_bridge_module.AgentService

        class _CapturingAgentService(real_agent_service):
            def __init__(self, *args, **kwargs):
                captured.update(kwargs)
                super().__init__(*args, **kwargs)

        monkeypatch.setattr(
            console_agent_bridge_module, "AgentService", _CapturingAgentService
        )

        db = AgentRunsDB(tmp_path / "runs.db", client_id="t")
        store = ConsoleChatStore()
        session = store.ensure_session()
        store.append_message(session.id, role=ConsoleMessageRole.USER, content="hi")
        assistant = store.append_message(
            session.id, role=ConsoleMessageRole.ASSISTANT, content=""
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=_PlainTextGateway(),
            skills_service=scope_service,
        )
        bridge.run_reply(
            conversation_id="conv-e2e-script",
            session_id=session.id,
            resolution=object(),
            assistant_message_id=assistant.id,
            model="test-model",
            session_system_prompt="",
            agent_messages=[{"role": "user", "content": "hi"}],
            should_cancel=lambda: False,
            request_skill_script_confirm=confirm_cb,
        )

        assert real_agent_service is _RealAgentService  # sanity: patched the real class
        tool = captured.get("run_skill_script_tool")
        assert tool is not None, "expected run_reply to build the run_skill_script closure"

        return _E2EEnv(
            tool=tool,
            confirm_calls=confirm_calls,
            marker_path=marker_path,
            _trust_service=trust_service,
            _skill_dir=skill_dir,
        )

    return _make


# ---------------------------------------------------------------------------
# Behaviors from the task-8 brief.
# ---------------------------------------------------------------------------


def test_agent_call_runs_a_real_script_and_returns_its_stdout(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": False})
    result = env.tool("demo-skill", "scripts/hello.py", [])
    assert result.ok is True
    assert "hello from a real skill script" in result.content
    assert "exit_code: 0" in result.content


def test_denied_confirm_never_runs_the_script(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": False, "remember": False})
    result = env.tool("demo-skill", "scripts/marker.py", [])
    assert result.ok is False
    assert not env.marker_path.exists(), "the script must never have executed"


def test_policy_disabled_denies_before_any_prompt(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": False}, policy_enabled=False)
    result = env.tool("demo-skill", "scripts/hello.py", [])
    assert result.ok is False
    assert env.confirm_calls == []


def test_untrusted_skill_is_refused_end_to_end(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": False}, trusted=False)
    result = env.tool("demo-skill", "scripts/hello.py", [])
    assert result.ok is False
    assert env.confirm_calls == []


def test_always_allow_persists_and_the_second_run_does_not_prompt(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": True})
    first = env.tool("demo-skill", "scripts/hello.py", [])
    second = env.tool("demo-skill", "scripts/hello.py", [])
    assert first.ok is True and second.ok is True
    assert len(env.confirm_calls) == 1, "the grant must suppress the second prompt"


def test_mutating_the_skill_after_a_grant_re_prompts(e2e_bridge_env):
    env = e2e_bridge_env(confirm={"allow": True, "remember": True})
    env.tool("demo-skill", "scripts/hello.py", [])
    env.mutate_script("print('changed')")
    env.retrust()
    env.tool("demo-skill", "scripts/hello.py", [])
    assert len(env.confirm_calls) == 2, (
        "a content change must invalidate the standing grant"
    )
