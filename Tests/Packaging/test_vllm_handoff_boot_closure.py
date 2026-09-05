"""vLLM target implementation types are validation-time, not boot-time work."""

from pathlib import Path
import subprocess
import sys


def test_handoff_import_defers_target_types_until_first_validation():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from dataclasses import asdict
from types import SimpleNamespace
from tldw_chatbook.UI.Navigation.vllm_handoff import (
    VllmConsoleIntent, VllmDefaultIntent, owner_has_current_intent,
)

assert "tldw_chatbook.UI.LLM_Management.vllm_setup" not in sys.modules
assert "tldw_chatbook.UI.LLM_Management" not in sys.modules
intent = VllmConsoleIntent("http://127.0.0.1:8000/v1/chat/completions", "model", 1)
assert not owner_has_current_intent(object(), intent)
assert "tldw_chatbook.UI.LLM_Management.vllm_setup" not in sys.modules

from tldw_chatbook.UI.LLM_Management.vllm_setup import (
    VllmConnectionTarget, VllmReadinessState,
)
target = VllmConnectionTarget(
    "vllm", intent.api_url, intent.model_id, "external", 1, "none",
)
assert VllmConsoleIntent.from_target(target) == intent
assert VllmDefaultIntent.from_target(target).generation == 1
snapshot = SimpleNamespace(
    target=target, current_token=SimpleNamespace(generation=1),
    state=VllmReadinessState.READY,
)
owner = SimpleNamespace(snapshot=lambda: snapshot)
assert owner_has_current_intent(owner, intent)
snapshot.current_token.generation = 2
assert not owner_has_current_intent(owner, intent)
try:
    VllmConsoleIntent.from_target(SimpleNamespace(**asdict(target)))
except TypeError:
    pass
else:
    raise AssertionError("lookalike target accepted")
""",
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
