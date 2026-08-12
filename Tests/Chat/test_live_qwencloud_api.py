"""Opt-in live QwenCloud text and native-tool verification.

The paid checks run in a subprocess whose profile paths are isolated before
any Chatbook module is imported. They require both ``TLDW_LIVE_QWENCLOUD=1``
and ``DASHSCOPE_API_KEY`` and are skipped by default.
"""

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


_TOOL_MARKER_PREFIX = "QWENCLOUD_LIVE_TOOL_RESULT_"
_TOOL_MARKER_SUFFIX = "_END"


def _live_enabled(environ: Mapping[str, str]) -> bool:
    return environ.get("TLDW_LIVE_QWENCLOUD", "").strip() == "1" and bool(
        environ.get("DASHSCOPE_API_KEY", "").strip()
    )


def _live_subprocess_environment(
    profile: Path,
    api_mode: str,
    environ: Mapping[str, str],
) -> dict[str, str]:
    """Return an isolated environment for one paid live subprocess."""
    env = dict(environ)
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
    return env


@pytest.mark.parametrize(
    ("environ", "expected"),
    [
        ({}, False),
        ({"TLDW_LIVE_QWENCLOUD": "1"}, False),
        ({"DASHSCOPE_API_KEY": "test-key"}, False),
        (
            {"TLDW_LIVE_QWENCLOUD": "1", "DASHSCOPE_API_KEY": "test-key"},
            True,
        ),
    ],
)
def test_live_gate_requires_explicit_opt_in_and_key(
    environ: Mapping[str, str],
    expected: bool,
) -> None:
    assert _live_enabled(environ) is expected


def test_live_subprocess_profile_is_isolated_before_chatbook_imports() -> None:
    profile = Path("/tmp/qwencloud-live-structural-test")
    env = _live_subprocess_environment(
        profile,
        "responses",
        {"DASHSCOPE_API_KEY": "test-key"},
    )

    assert env["HOME"] == str(profile / "home")
    assert env["XDG_CONFIG_HOME"] == str(profile / "xdg-config")
    assert env["XDG_DATA_HOME"] == str(profile / "xdg-data")
    assert env["TLDW_CONFIG_PATH"] == str(profile / "config" / "config.toml")
    assert env["TLDW_LIVE_QWENCLOUD_DATA_DIR"] == str(profile / "data")
    assert env["TLDW_LIVE_QWENCLOUD_API_MODE"] == "responses"
    assert _LIVE_CHILD.index("config_path.write_text") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )
    assert _LIVE_CHILD.index("config_path.chmod") < _LIVE_CHILD.index(
        "from tldw_chatbook"
    )


def _build_probe_contract(
    left: int,
    right: int,
    text_sentinel: str,
) -> tuple[str, str, str]:
    """Build a prompt whose derived tool marker is unavailable before execution."""
    expression = f"{left} + {right}"
    system_prompt = (
        "You are running a harmless API contract probe. Call the calculator "
        f"tool exactly once with expression {expression!r}. Only after the tool "
        f"returns, reply with text marker {text_sentinel!r}, fixed prefix "
        f"{_TOOL_MARKER_PREFIX!r}, the exact integer result copied as plain ASCII "
        f"digits without commas or a decimal point, then suffix {_TOOL_MARKER_SUFFIX!r}."
    )
    user_prompt = "Run the requested contract probe now."
    return system_prompt, user_prompt, expression


def _validate_probe_observation(
    *,
    status: str,
    final_text: str,
    steps: Sequence[object],
    text_sentinel: str,
    system_prompt: str,
    user_prompt: str,
    expression: str,
    left: int,
    right: int,
) -> None:
    """Validate identifying output without exposing request or response content."""
    expected = left + right
    tool_marker = f"{_TOOL_MARKER_PREFIX}{expected}{_TOOL_MARKER_SUFFIX}"
    if tool_marker in system_prompt or tool_marker in user_prompt:
        raise ValueError(
            "Derived calculator marker was disclosed before tool execution."
        )
    if status != "done":
        raise ValueError("QwenCloud live run did not complete.")
    tool_calls = [
        step
        for step in steps
        if getattr(step, "kind", "") == "tool_call"
        and getattr(step, "tool_name", "") == "calculator"
    ]
    tool_results = [
        step
        for step in steps
        if getattr(step, "kind", "") == "tool_result"
        and getattr(step, "tool_name", "") == "calculator"
    ]
    if len(tool_calls) != 1 or len(tool_results) != 1:
        raise ValueError("QwenCloud did not execute exactly one calculator call.")
    if getattr(tool_calls[0], "args", None) != {"expression": expression}:
        raise ValueError(
            "QwenCloud calculator call did not use the requested expression."
        )
    try:
        result_payload = json.loads(str(getattr(tool_results[0], "result", "")))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "QwenCloud calculator result was not structured JSON."
        ) from exc
    observed = (
        result_payload.get("result") if isinstance(result_payload, Mapping) else None
    )
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed != expected
    ):
        raise ValueError("QwenCloud calculator result did not match the derived value.")
    if text_sentinel not in final_text:
        raise ValueError("QwenCloud live text marker was not returned.")
    if tool_marker not in final_text:
        raise ValueError("QwenCloud live tool marker did not affect the answer.")


def test_live_probe_contract_does_not_disclose_derived_tool_marker() -> None:
    left = 1_000_000_003
    right = 2_000_000_033
    system_prompt, user_prompt, _expression = _build_probe_contract(
        left,
        right,
        "QWENCLOUD_LIVE_TEXT_TEST",
    )
    tool_marker = f"{_TOOL_MARKER_PREFIX}{left + right}{_TOOL_MARKER_SUFFIX}"

    assert tool_marker not in system_prompt
    assert tool_marker not in user_prompt
    assert "{tool_sentinel!r}" not in _LIVE_CHILD


def test_live_probe_validation_rejects_mutated_calculator_result() -> None:
    text_sentinel = "QWENCLOUD_LIVE_TEXT_TEST"
    left = 1_000_000_003
    right = 2_000_000_033
    expected = left + right
    tool_marker = f"{_TOOL_MARKER_PREFIX}{expected}{_TOOL_MARKER_SUFFIX}"
    system_prompt, user_prompt, expression = _build_probe_contract(
        left,
        right,
        text_sentinel,
    )
    good_steps = [
        SimpleNamespace(
            kind="tool_call",
            tool_name="calculator",
            args={"expression": expression},
            result="",
        ),
        SimpleNamespace(
            kind="tool_result",
            tool_name="calculator",
            args=None,
            result=json.dumps(
                {
                    "expression": expression,
                    "result": expected,
                    "result_type": "int",
                }
            ),
        ),
    ]
    _validate_probe_observation(
        status="done",
        final_text=f"{text_sentinel} {tool_marker}",
        steps=good_steps,
        text_sentinel=text_sentinel,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        expression=expression,
        left=left,
        right=right,
    )

    mutated_steps = [
        good_steps[0],
        SimpleNamespace(
            kind="tool_result",
            tool_name="calculator",
            args=None,
            result=json.dumps(
                {
                    "expression": expression,
                    "result": expected + 1,
                    "result_type": "int",
                }
            ),
        ),
    ]
    with pytest.raises(ValueError, match="calculator result"):
        _validate_probe_observation(
            status="done",
            final_text=f"{text_sentinel} {tool_marker}",
            steps=mutated_steps,
            text_sentinel=text_sentinel,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expression=expression,
            left=left,
            right=right,
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

    from Tests.Chat.test_live_qwencloud_api import (
        _build_probe_contract,
        _validate_probe_observation,
    )
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
    system_prompt, user_prompt, expression = _build_probe_contract(
        left, right, text_sentinel
    )

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

        _validate_probe_observation(
            status=outcome.status,
            final_text=outcome.final_text,
            steps=outcome.steps,
            text_sentinel=text_sentinel,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            expression=expression,
            left=left,
            right=right,
        )
    finally:
        db.close()
    """
)


@pytest.mark.parametrize("api_mode", ["responses", "chat_completions"])
@pytest.mark.allow_network
@pytest.mark.integration
@pytest.mark.optional
@pytest.mark.slow
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
    env = _live_subprocess_environment(profile, api_mode, os.environ)
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
