"""Opt-in paid Moonshot and Z.ai Console tool-continuation checks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent
from types import SimpleNamespace

import pytest


_PROVIDERS = {
    "moonshot": {
        "gate": "TLDW_LIVE_MOONSHOT",
        "key": "MOONSHOT_API_KEY",
    },
    "zai": {
        "gate": "TLDW_LIVE_ZAI",
        "key": "ZAI_API_KEY",
    },
}
_TOOL_MARKER_PREFIX = "HOSTED_LIVE_TOOL_RESULT_"
_TOOL_MARKER_SUFFIX = "_END"


def _live_enabled(provider: str, environ: Mapping[str, str]) -> bool:
    settings = _PROVIDERS[provider]
    return environ.get(settings["gate"], "").strip() == "1" and bool(
        environ.get(settings["key"], "").strip()
    )


def _live_subprocess_environment(
    profile: Path,
    provider: str,
    environ: Mapping[str, str],
) -> dict[str, str]:
    env = dict(environ)
    env.update(
        {
            "HOME": str(profile / "home"),
            "XDG_CONFIG_HOME": str(profile / "xdg-config"),
            "XDG_DATA_HOME": str(profile / "xdg-data"),
            "TLDW_CONFIG_PATH": str(profile / "config" / "config.toml"),
            "TLDW_LIVE_HOSTED_DATA_DIR": str(profile / "data"),
            "TLDW_LIVE_HOSTED_PROVIDER": provider,
        }
    )
    return env


@pytest.mark.parametrize("provider", ["moonshot", "zai"])
@pytest.mark.parametrize(
    ("gate", "key", "expected"),
    [
        ("", "", False),
        ("1", "", False),
        ("", "test-key", False),
        ("1", "test-key", True),
    ],
)
def test_live_gate_requires_provider_opt_in_and_key(
    provider: str,
    gate: str,
    key: str,
    expected: bool,
) -> None:
    settings = _PROVIDERS[provider]
    environ = {settings["gate"]: gate, settings["key"]: key}
    assert _live_enabled(provider, environ) is expected


def test_live_subprocess_isolates_profile_before_chatbook_imports() -> None:
    profile = Path("/tmp/hosted-live-structural-test")
    env = _live_subprocess_environment(profile, "moonshot", {})

    assert env["HOME"] == str(profile / "home")
    assert env["XDG_CONFIG_HOME"] == str(profile / "xdg-config")
    assert env["XDG_DATA_HOME"] == str(profile / "xdg-data")
    assert env["TLDW_CONFIG_PATH"] == str(profile / "config" / "config.toml")
    assert env["TLDW_LIVE_HOSTED_DATA_DIR"] == str(profile / "data")
    assert _LIVE_CHILD.index("config_path.write_text") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    assert _LIVE_CHILD.index("config_path.chmod") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    compile(_LIVE_CHILD, "<hosted-live-child>", "exec")


def _build_probe_contract(
    provider: str,
    left: int,
    right: int,
    text_sentinel: str,
) -> tuple[str, str, str]:
    expression = f"{left} + {right}"
    system_prompt = (
        "Run one harmless provider contract probe. Call the calculator tool "
        f"exactly once with expression {expression!r}. Only after the tool returns, "
        f"reply with text marker {text_sentinel!r}, prefix "
        f"{_TOOL_MARKER_PREFIX!r}, the exact integer result as ASCII digits, then "
        f"suffix {_TOOL_MARKER_SUFFIX!r}."
    )
    return system_prompt, f"Run the {provider} contract probe now.", expression


def _validate_probe_observation(
    *,
    provider: str,
    status: str,
    final_text: str,
    steps: Sequence[object],
    text_sentinel: str,
    system_prompt: str,
    user_prompt: str,
    expression: str,
    expected: int,
) -> None:
    tool_marker = f"{_TOOL_MARKER_PREFIX}{expected}{_TOOL_MARKER_SUFFIX}"
    if tool_marker in system_prompt or tool_marker in user_prompt:
        raise ValueError("Derived marker was disclosed before tool execution.")
    if status != "done":
        raise ValueError(f"{provider} live run did not complete.")
    calls = [
        step
        for step in steps
        if getattr(step, "kind", "") == "tool_call"
        and getattr(step, "tool_name", "") == "calculator"
    ]
    results = [
        step
        for step in steps
        if getattr(step, "kind", "") == "tool_result"
        and getattr(step, "tool_name", "") == "calculator"
    ]
    if len(calls) != 1 or len(results) != 1:
        raise ValueError(f"{provider} did not execute exactly one calculator call.")
    if getattr(calls[0], "args", None) != {"expression": expression}:
        raise ValueError(f"{provider} calculator arguments were unexpected.")
    try:
        result_payload = json.loads(str(getattr(results[0], "result", "")))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{provider} calculator result was malformed.") from exc
    observed = (
        result_payload.get("result") if isinstance(result_payload, Mapping) else None
    )
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed != expected
    ):
        raise ValueError(f"{provider} calculator result was incorrect.")
    if text_sentinel not in final_text or tool_marker not in final_text:
        raise ValueError(f"{provider} final answer did not use the tool result.")


def test_live_probe_marker_requires_the_calculator_result() -> None:
    provider = "moonshot"
    left = 1_000_000_003
    right = 2_000_000_033
    expected = left + right
    text_sentinel = "HOSTED_LIVE_TEXT_TEST"
    system_prompt, user_prompt, expression = _build_probe_contract(
        provider, left, right, text_sentinel
    )
    marker = f"{_TOOL_MARKER_PREFIX}{expected}{_TOOL_MARKER_SUFFIX}"
    steps = [
        SimpleNamespace(
            kind="tool_call",
            tool_name="calculator",
            args={"expression": expression},
        ),
        SimpleNamespace(
            kind="tool_result",
            tool_name="calculator",
            result=json.dumps({"result": expected}),
        ),
    ]

    _validate_probe_observation(
        provider=provider,
        status="done",
        final_text=f"{text_sentinel} {marker}",
        steps=steps,
        text_sentinel=text_sentinel,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expression=expression,
        expected=expected,
    )
    steps[1].result = json.dumps({"result": expected + 1})
    with pytest.raises(ValueError, match="incorrect"):
        _validate_probe_observation(
            provider=provider,
            status="done",
            final_text=f"{text_sentinel} {marker}",
            steps=steps,
            text_sentinel=text_sentinel,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expression=expression,
            expected=expected,
        )


_LIVE_CHILD = dedent(
    r"""
    from __future__ import annotations

    import json
    import os
    from pathlib import Path
    import secrets

    provider = os.environ["TLDW_LIVE_HOSTED_PROVIDER"]
    metadata = {
        "moonshot": {
            "display": "Moonshot",
            "model_env": "TLDW_LIVE_MOONSHOT_MODEL",
            "model": "kimi-k3",
            "base_env": "TLDW_LIVE_MOONSHOT_API_BASE_URL",
            "base": "https://api.moonshot.ai/v1",
            "key_env": "MOONSHOT_API_KEY",
        },
        "zai": {
            "display": "Z.ai",
            "model_env": "TLDW_LIVE_ZAI_MODEL",
            "model": "glm-5.2",
            "base_env": "TLDW_LIVE_ZAI_API_BASE_URL",
            "base": "https://api.z.ai/api/paas/v4",
            "key_env": "ZAI_API_KEY",
        },
    }[provider]
    model = os.environ.get(metadata["model_env"], metadata["model"]).strip()
    base_url = os.environ.get(metadata["base_env"], metadata["base"]).strip()
    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    data_dir = Path(os.environ["TLDW_LIVE_HOSTED_DATA_DIR"])
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    config_path.write_text(
        "\n".join(
            (
                "[general]",
                'users_name = "hosted-live-test"',
                "",
                "[paths]",
                f"data_dir = {json.dumps(str(data_dir))}",
                "",
                f"[api_settings.{provider}]",
                f"api_key_env_var = {json.dumps(metadata['key_env'])}",
                f"api_base_url = {json.dumps(base_url)}",
                f"model = {json.dumps(model)}",
                "streaming = true",
                "",
            )
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    from loguru import logger

    logger.remove()

    from Tests.Chat.test_live_moonshot_zai_api import (
        _build_probe_contract,
        _validate_probe_observation,
    )
    from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    text_sentinel = f"HOSTED_LIVE_TEXT_{secrets.token_hex(8).upper()}"
    left = 10**12 + secrets.randbelow(10**11)
    right = 10**12 + secrets.randbelow(10**11)
    expected = left + right
    system_prompt, user_prompt, expression = _build_probe_contract(
        provider, left, right, text_sentinel
    )
    runs_db = AgentRunsDB(
        data_dir / f"agent-runs-{provider}.db", client_id=f"{provider}-live"
    )
    chat_db = CharactersRAGDB(
        data_dir / f"chat-{provider}.db", f"{provider}-live"
    )
    try:
        store = ConsoleChatStore(persistence=ChatPersistenceService(chat_db))
        session = store.create_session(title=f"{provider} live contract")
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=user_prompt,
            persist=True,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
            persist=True,
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=runs_db,
            store=store,
            provider_gateway=ConsoleProviderGateway(),
        )
        resolution = ConsoleProviderResolution(
            provider=metadata["display"],
            base_url=base_url,
            model=model,
            ready=True,
            readiness_key=provider,
            execution_key=provider,
            api_key=os.environ[metadata["key_env"]],
            streaming=True,
            continuation_protocol="chat_completions",
            request_timeout=90.0,
            request_retries=1,
            request_retry_delay=1.0,
        )
        _run_id, outcome = bridge.run_reply(
            conversation_id=f"{provider}-live-conversation",
            session_id=session.id,
            resolution=resolution,
            assistant_message_id=assistant.id,
            model=model,
            session_system_prompt=system_prompt,
            agent_messages=[{"role": "user", "content": user_prompt}],
            should_cancel=lambda: False,
            native_tools_enabled=True,
        )
        _validate_probe_observation(
            provider=provider,
            status=outcome.status,
            final_text=outcome.final_text,
            steps=outcome.steps,
            text_sentinel=text_sentinel,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expression=expression,
            expected=expected,
        )
    finally:
        runs_db.close()
        chat_db.close_connection()
    """
)


@pytest.mark.allow_network
@pytest.mark.integration
@pytest.mark.parametrize("provider", ["moonshot", "zai"])
def test_live_hosted_text_and_native_tool(tmp_path: Path, provider: str) -> None:
    """Make one paid provider request only behind its exact double gate."""
    if not _live_enabled(provider, os.environ):
        settings = _PROVIDERS[provider]
        pytest.skip(f"Set {settings['gate']}=1 and {settings['key']} to opt in.")
    env = _live_subprocess_environment(tmp_path / provider, provider, os.environ)
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _LIVE_CHILD],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=360,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"Live {provider} verification timed out with output suppressed.")
    if completed.returncode != 0:
        pytest.fail(f"Live {provider} verification failed with output suppressed.")
