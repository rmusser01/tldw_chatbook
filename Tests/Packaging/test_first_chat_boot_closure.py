"""Import-closure guard: TASK-34100.2's first-chat code stays off first paint.

PR #3517's "UI latency guardrails" lane went red on both ADR-097 boot ratchets
(2026-11-13): ``import tldw_chatbook.app`` loaded 690 own modules against a
limit of 688, and 1,045 were resident at ``_ui_ready`` against 1,040. An
import-parent tracer, diffed against dev ``26b2e677dd``, named two edges this
file pins, because each was deferred rather than paid for with a raise:

1. ``LLM_Calls/LLM_API_Calls.py`` imported ``chat_transport_retry`` (the
   urllib3 retry the requests-based handlers mount) and ``provider_reason``
   (a 429 body's reason) at module scope. ``LLM_API_Calls`` is on the boot
   import path (``Subscriptions.watchlists_operation_coordinator ->
   briefing_service -> Chat.Chat_Functions -> LLM_API_Calls``), but both
   helpers are only needed inside a send (a worker thread) or its 429
   branch, so they are imported there.
2. Readiness asked the observed-window ledger on every build, and
   ``ConsoleSendOutcomes.window_for`` interpreted the endpoint
   (``canonical_connection_identity -> _route_owning_presets``, which imports
   ``Provider_Contract.provider_setup_facts``) before looking for a record.
   At first paint the ledger is always empty, so the endpoint is now
   interpreted only when a recorded window could be refused for it. The
   writer (``record_window``, on the send path) still interprets it, so no
   import moves onto a render path.

Behaviour at first use is covered where it lives:
``Tests/Chat/test_rate_limit_reads_as_rate_limit.py`` (the handlers' retry and
429 reason) and ``Tests/Chat/test_first_chat_readiness.py`` (a window answers
only for the endpoint that reported it).

Subprocess-isolated for the same reason as ``test_app_import_diet_closure.py``
(TASK-21108), whose pattern this file follows: ``sys.modules`` is
process-global.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated_python(tmp_path: Path, code: str) -> subprocess.CompletedProcess[str]:
    """Run a Python snippet in a fresh interpreter with isolated config/data dirs.

    Args:
        tmp_path: Per-test scratch directory for the subprocess's HOME/XDG so
            the app import can never read or write the live user config.
        code: The Python source to execute with ``python -c``.

    Returns:
        The completed process (never raises on nonzero exit).
    """
    data_home = tmp_path / "data"
    config_home = tmp_path / "config"
    home = tmp_path / "home"
    for path in (data_home, config_home, home):
        path.mkdir(parents=True, exist_ok=True)

    env = {
        **os.environ,
        "TLDW_TEST_MODE": "1",
        "XDG_DATA_HOME": str(data_home),
        "XDG_CONFIG_HOME": str(config_home),
        "HOME": str(home),
        "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
        "PYTHONPATH": os.pathsep.join(
            (str(REPO_ROOT), str(REPO_ROOT / "packages/tldw_profile_core/src"))
        ),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )


_APP_CLOSURE_SNIPPET = """
import sys

import tldw_chatbook.app  # noqa: F401

forbidden = (
    "tldw_chatbook.LLM_Calls.chat_transport_retry",
    "tldw_chatbook.LLM_Calls.provider_reason",
)
resident = sorted(m for m in forbidden if sys.modules.get(m) is not None)
assert not resident, (
    "send-time chat helpers resident after `import tldw_chatbook.app`: "
    + repr(resident)
    + " -- import them inside the handler that mounts the retry or raises the "
    "429 (LLM_Calls/LLM_API_Calls.py) rather than raising MAX_TLDW_MODULE_COUNT "
    "(ADR-097)."
)

# Anti-vacuity: LLM_API_Calls itself must still be in the boot closure, or the
# assertion above would pass without exercising the deferral.
assert "tldw_chatbook.LLM_Calls.LLM_API_Calls" in sys.modules, (
    "LLM_API_Calls left the app import closure; this guard no longer tests "
    "the deferral"
)

print("FIRST_CHAT_APP_CLOSURE_OK")
"""


_READINESS_SNIPPET = """
import sys

from tldw_chatbook.Chat import console_session_settings as session_settings
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SEND_OUTCOMES,
    ConsoleSessionSettings,
    build_console_settings_readiness,
)

FACTS = "tldw_chatbook.Provider_Contract.provider_setup_facts"
assert FACTS not in sys.modules, "importing readiness already loads the setup facts"

asked = []
_real_window_for = CONSOLE_SEND_OUTCOMES.window_for


def _counting_window_for(*args, **kwargs):
    asked.append(args)
    return _real_window_for(*args, **kwargs)


CONSOLE_SEND_OUTCOMES.window_for = _counting_window_for

# What the Console header computes at first paint for the census's profile (a
# configured cloud key) with an empty observed-window ledger. A local server
# is deliberately not asserted here: its readiness resolves the send endpoint
# itself (``_send_endpoint -> effective_provider_endpoint``), which reads the
# setup facts' route suffixes by design.
build_console_settings_readiness(
    ConsoleSessionSettings(provider="openai", model="gpt-4o", max_tokens=4096),
    app_config={"api_settings": {"openai": {"api_key": "sk-closure-test"}}},
    environ={},
)

# Anti-vacuity: readiness did consult the ledger with an endpoint, so the
# assertion below exercises the guarded edge.
assert any(len(call) >= 3 and call[2] for call in asked), asked
assert FACTS not in sys.modules, (
    "readiness with an empty window ledger interpreted the endpoint and loaded "
    "the provider setup facts at first paint -- look for a record before "
    "calling _window_endpoint (ConsoleSendOutcomes.window_for)."
)

# The writer still interprets the endpoint, and a recorded window still
# answers only for the server that reported it.
CONSOLE_SEND_OUTCOMES.record_window(
    "llama_cpp", "qwen-small", 4096, True, endpoint="http://127.0.0.1:9099"
)
assert FACTS in sys.modules
assert _real_window_for("llama_cpp", "qwen-small", "http://127.0.0.1:9099/") == (
    4096,
    True,
)
assert _real_window_for("llama_cpp", "qwen-small", "http://127.0.0.1:9431") is None
assert _real_window_for("llama_cpp", "qwen-small") == (4096, True)

print("FIRST_CHAT_READINESS_CLOSURE_OK")
"""


def test_app_import_leaves_the_send_time_chat_helpers_unloaded(tmp_path: Path) -> None:
    """``import tldw_chatbook.app`` does not execute the retry/reason helpers.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _APP_CLOSURE_SNIPPET)
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "FIRST_CHAT_APP_CLOSURE_OK" in result.stdout


def test_readiness_with_no_recorded_window_leaves_setup_facts_unloaded(
    tmp_path: Path,
) -> None:
    """First-paint readiness does not interpret an endpoint it cannot use.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _READINESS_SNIPPET)
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "FIRST_CHAT_READINESS_CLOSURE_OK" in result.stdout
