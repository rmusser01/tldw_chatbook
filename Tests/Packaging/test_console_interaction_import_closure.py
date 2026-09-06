"""First-use Console services must not consume the boot budget (ADR-097)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "code",
    [
        """
import sys
from tldw_chatbook.UI.Console_Modules.environment import ConsoleEnvironmentController

jobs = []
controller = ConsoleEnvironmentController(
    run_worker=lambda *args, **kwargs: jobs.append(args),
    marshal_to_ui=lambda *args: None,
    workspace_root_accessor=lambda: '/unused-closed-rail',
    rail_open_accessor=lambda: False,
    on_snapshot=lambda snapshot: None,
)
controller.request_refresh(include_net=True)
controller.poll_tick()
assert not jobs
assert controller.snapshot.git.root == ''
for name in (
    'tldw_chatbook.Workspaces.environment_status',
    'tldw_chatbook.Workspaces.git_workspace',
):
    assert name not in sys.modules, name
""",
        """
import sys
from tldw_chatbook.UI.Navigation.vllm_handoff import (
    VllmConsoleIntent, owner_has_current_intent,
)

intent = VllmConsoleIntent('http://127.0.0.1:8000/v1/chat/completions', 'example/model', 1)
assert not owner_has_current_intent(object(), intent)
assert intent.generation == 1
assert 'tldw_chatbook.UI.LLM_Management.vllm_setup' not in sys.modules
""",
        """
import sys
from tldw_chatbook.Chat.provider_readiness import get_provider_readiness
import tldw_chatbook.LLM_Calls.LLM_API_Calls

assert get_provider_readiness('Ollama', {}, environ={}).ready
assert 'tldw_chatbook.LLM_Calls.anthropic_subscription' not in sys.modules
""",
        """
import sys
from tldw_chatbook.Chat.console_trace_custom_pii import (
    CustomPIIRuleset, redact_pii_value_with_custom_rules,
)

ruleset = CustomPIIRuleset(1, 'c1d3b183-3887-4f44-a811-9c68d1cb4911', ())
result = redact_pii_value_with_custom_rules('plain text', ruleset)
assert result.available and result.value == 'plain text'
assert 'tldw_chatbook.Chat.console_trace_regex_worker' not in sys.modules
""",
    ],
    ids=[
        "closed-environment-rail",
        "vllm-handoff-contract",
        "non-anthropic-readiness",
        "no-custom-pii-rules",
    ],
)
def test_console_services_load_only_on_first_use(tmp_path: Path, code: str) -> None:
    """Exercise boot-safe consumers without loading their interaction services.

    Args:
        tmp_path: Isolated profile root for the fresh Python process.
        code: Real consumer scenario and independently specified absent modules.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "USERPROFILE": str(tmp_path),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "TLDW_CONFIG_PATH": str(tmp_path / "config.toml"),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(repo_root),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
