"""Keep vLLM setup and Environment I/O behind their first-use seams."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_console_defers_setup_and_environment_io_until_requested(
    tmp_path: Path,
) -> None:
    """Constructing Console helpers is cheap; real first use remains available."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(mode=0o700)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"[paths]\ndata_dir = {json.dumps(str(data_dir))}\n", encoding="utf-8"
    )
    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "TLDW_CONFIG_PATH": str(config_path),
        "TLDW_CONSOLE_CLOSURE_ROOT": str(tmp_path),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    code = r"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import tldw_chatbook.UI.Screens.chat_screen
from tldw_chatbook.UI.Console_Modules.environment import ConsoleEnvironmentController
from tldw_chatbook.UI.Navigation.vllm_handoff import VllmConsoleIntent, owner_has_current_intent

jobs = []
root = None
controller = ConsoleEnvironmentController(
    run_worker=lambda job, **kwargs: jobs.append(job),
    marshal_to_ui=lambda callback, *args: callback(*args),
    workspace_root_accessor=lambda: root,
    rail_open_accessor=lambda: True,
    on_snapshot=lambda snapshot: None,
)
controller.poll_tick()
deferred = (
    "tldw_chatbook.UI.LLM_Management.vllm_setup",
    "tldw_chatbook.Workspaces.environment_status",
    "tldw_chatbook.Workspaces.git_workspace",
)
assert not [name for name in deferred if name in sys.modules]
assert not jobs

# Explicit first use retains exact target validation and the real scanner.
from tldw_chatbook.UI.LLM_Management.vllm_setup import VllmConnectionTarget, VllmReadinessState
from tldw_chatbook.Workspaces.environment_status import BacklogTaskScanner

target = VllmConnectionTarget("vllm", "http://127.0.0.1:8000/v1/chat/completions", "owner/model", "external", 1, "none")
intent = VllmConsoleIntent.from_target(target)
owner = SimpleNamespace(snapshot=lambda: SimpleNamespace(
    target=target, current_token=SimpleNamespace(generation=1), state=VllmReadinessState.READY))
assert owner_has_current_intent(owner, intent)
try:
    VllmConsoleIntent.from_target(SimpleNamespace(api_url=target.api_url, model_id=target.model_id, generation=1))
except TypeError:
    pass
else:
    raise AssertionError("Noncanonical target accepted")

root = os.environ["TLDW_CONSOLE_CLOSURE_ROOT"]
controller.poll_tick()
assert len(jobs) == 1
scanner = controller._scanner
assert isinstance(scanner, BacklogTaskScanner)
jobs.pop()()
controller.poll_tick()
assert controller._scanner is scanner
assert len(jobs) == 1
print("CONSOLE_INTERACTION_CLOSURE_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
    assert "CONSOLE_INTERACTION_CLOSURE_OK" in result.stdout
