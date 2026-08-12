"""Opt-in live QwenCloud text and native-tool verification.

The paid checks run in a subprocess whose profile paths are isolated before
any Chatbook module is imported. They require both ``TLDW_LIVE_QWENCLOUD=1``
and ``DASHSCOPE_API_KEY`` and are skipped by default.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import subprocess
import sys
from textwrap import dedent

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.optional, pytest.mark.slow]


def _live_enabled(environ: Mapping[str, str]) -> bool:
    return environ.get("TLDW_LIVE_QWENCLOUD", "").strip() == "1" and bool(
        environ.get("DASHSCOPE_API_KEY", "").strip()
    )


_LIVE_CHILD = dedent(
    r"""
    from __future__ import annotations

    import json
    import os
    from pathlib import Path
    import secrets

    config_path = Path(os.environ["TLDW_CONFIG_PATH"])
    data_dir = Path(os.environ["TLDW_LIVE_QWENCLOUD_DATA_DIR"])
    config_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    base_url = os.environ.get(
        "TLDW_LIVE_QWENCLOUD_API_BASE_URL",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    ).strip() or "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    model = (
        os.environ.get("TLDW_LIVE_QWENCLOUD_MODEL", "qwen3.8-max").strip()
        or "qwen3.8-max"
    )
    api_mode = os.environ["TLDW_LIVE_QWENCLOUD_API_MODE"]
    config_path.write_text(
        "\n".join(
            (
                "[general]",
                'users_name = "qwencloud-live-test"',
                "",
                "[paths]",
                f"data_dir = {json.dumps(str(data_dir))}",
                "",
                "[api_settings.qwencloud]",
                'api_key_env_var = "DASHSCOPE_API_KEY"',
                f"api_base_url = {json.dumps(base_url)}",
                f"api_mode = {json.dumps(api_mode)}",
                f"model = {json.dumps(model)}",
                "",
            )
        ),
        encoding="utf-8",
    )
    config_path.chmod(0o600)

    from loguru import logger

    logger.remove()

    from tldw_chatbook.Agents.agent_models import STEP_TOOL_CALL, STEP_TOOL_RESULT
    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge
    from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    text_sentinel = f"QWENCLOUD_LIVE_TEXT_{secrets.token_hex(8).upper()}"
    left = 10**12 + secrets.randbelow(10**11)
    right = 10**12 + secrets.randbelow(10**11)
    expected = left + right
    tool_sentinel = f"QWENCLOUD_LIVE_TOOL_{expected}"
    expression = f"{left} + {right}"
    system_prompt = (
        "You are running a harmless API contract probe. Call the calculator "
        f"tool exactly once with expression {expression!r}. After its result, "
        f"reply with both exact markers {text_sentinel!r} and {tool_sentinel!r}."
    )
    user_prompt = "Run the requested contract probe now."

    db = AgentRunsDB(data_dir / f"agent-runs-{api_mode}.db", client_id="qwen-live")
    try:
        store = ConsoleChatStore()
        session = store.ensure_session()
        store.append_message(
            session.id,
            role=ConsoleMessageRole.USER,
            content=user_prompt,
        )
        assistant = store.append_message(
            session.id,
            role=ConsoleMessageRole.ASSISTANT,
            content="",
        )
        bridge = ConsoleAgentBridge(
            agent_runs_db=db,
            store=store,
            provider_gateway=ConsoleProviderGateway(),
        )
        resolution = ConsoleProviderResolution(
            provider="QwenCloud",
            base_url=base_url,
            model=model,
            ready=True,
            readiness_key="qwencloud",
            execution_key="qwencloud",
            api_key=os.environ["DASHSCOPE_API_KEY"],
            streaming=True,
            api_mode=api_mode,
        )
        _run_id, outcome = bridge.run_reply(
            conversation_id=f"qwencloud-live-{api_mode}",
            session_id=session.id,
            resolution=resolution,
            assistant_message_id=assistant.id,
            model=model,
            session_system_prompt=system_prompt,
            agent_messages=[{"role": "user", "content": user_prompt}],
            should_cancel=lambda: False,
        )

        if outcome.status != "done":
            raise SystemExit("QwenCloud live run did not complete.")
        if text_sentinel not in outcome.final_text:
            raise SystemExit("QwenCloud live text marker was not returned.")
        if tool_sentinel not in outcome.final_text:
            raise SystemExit("QwenCloud live tool marker did not affect the answer.")
        tool_calls = [
            step
            for step in outcome.steps
            if step.kind == STEP_TOOL_CALL and step.tool_name == "calculator"
        ]
        tool_results = [
            step
            for step in outcome.steps
            if step.kind == STEP_TOOL_RESULT and step.tool_name == "calculator"
        ]
        if len(tool_calls) != 1 or len(tool_results) != 1:
            raise SystemExit("QwenCloud did not execute exactly one calculator call.")
        if str(expected) not in tool_results[0].result:
            raise SystemExit("QwenCloud calculator result marker was not recorded.")
    finally:
        db.close()
    """
)


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.allow_network
@pytest.mark.skipif(
    not _live_enabled(os.environ),
    reason="Set TLDW_LIVE_QWENCLOUD=1 and DASHSCOPE_API_KEY to run paid live checks.",
)
def test_live_qwencloud_text_and_native_tool(
    tmp_path: Path,
    api_mode: str,
) -> None:
    """Require identifying text and a real calculator continuation in each mode."""
    profile = tmp_path / api_mode
    env = os.environ.copy()
    env.update(
        {
            "HOME": str(profile / "home"),
            "XDG_CONFIG_HOME": str(profile / "xdg-config"),
            "XDG_DATA_HOME": str(profile / "xdg-data"),
            "TLDW_CONFIG_PATH": str(profile / "config" / "config.toml"),
            "TLDW_LIVE_QWENCLOUD_DATA_DIR": str(profile / "data"),
            "TLDW_LIVE_QWENCLOUD_API_MODE": api_mode,
        }
    )
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
        pytest.fail(
            f"Live QwenCloud {api_mode} verification timed out without exposing "
            "request or response content."
        )
    if completed.returncode != 0:
        pytest.fail(
            f"Live QwenCloud {api_mode} verification failed without exposing "
            "request or response content."
        )
