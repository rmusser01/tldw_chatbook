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

   Rebased 2026-11-19 onto dev ``a6df4d8d48``: dev's TASK-40300
   (``40a1ca25ae``) readiness now checks every provider's endpoint at compose
   (``endpoint_refusal -> resolve_provider_endpoint -> _route_owning_presets``),
   so the setup facts are resident at first paint on dev itself and
   ``sys.modules`` can no longer tell this edge apart. The readiness case
   therefore counts the ledger's endpoint interpretations instead.

PR #3517's round-24 port (owner decision 2026-11-23, "lazy-load, no raises")
deferred three more of the PR's new modules instead of raising ADR-097's
limits, and the last three cases pin each deferral and its first use:
``app_env_key_notice`` (no longer a ``TldwCli`` base; loaded once a notice
is held), ``UI.Console_Modules.setup_card_offers`` (the Get started card's
offers; loaded when a blocking card syncs or an offer is clicked) and
``UI.Settings_Modules.provider_test_rows`` (Settings' Test-row facts; was the
one module that took the screen pre-import pass to 559 against 558).

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

# Importing readiness and the ledger still leaves the setup facts unloaded:
# dev's endpoint check loads them when readiness runs, not at import
# (``_route_owning_presets`` imports them lazily).
FACTS = "tldw_chatbook.Provider_Contract.provider_setup_facts"
assert FACTS not in sys.modules, (
    "importing console_session_settings already loads " + FACTS
    + " -- keep that import inside the endpoint interpretation"
)

asked = []
_real_window_for = CONSOLE_SEND_OUTCOMES.window_for


def _counting_window_for(*args, **kwargs):
    asked.append(args)
    return _real_window_for(*args, **kwargs)


interpreted = []
_real_window_endpoint = session_settings._window_endpoint


def _counting_window_endpoint(provider, endpoint):
    interpreted.append(endpoint)
    return _real_window_endpoint(provider, endpoint)


CONSOLE_SEND_OUTCOMES.window_for = _counting_window_for
session_settings._window_endpoint = _counting_window_endpoint

# What the Console header computes at first paint for the census's profile (a
# configured cloud key) with an empty observed-window ledger. Dev's TASK-40300
# endpoint check loads the setup facts here on its own, so the guarded edge is
# counted rather than read from sys.modules.
build_console_settings_readiness(
    ConsoleSessionSettings(provider="openai", model="gpt-4o", max_tokens=4096),
    app_config={"api_settings": {"openai": {"api_key": "sk-closure-test"}}},
    environ={},
)

# Anti-vacuity: readiness did consult the ledger with an endpoint, so the
# assertion below exercises the guarded edge.
assert any(len(call) >= 3 and call[2] for call in asked), asked
assert interpreted == [], (
    "readiness with an empty window ledger interpreted the endpoint at first "
    "paint -- look for a record before calling _window_endpoint "
    "(ConsoleSendOutcomes.window_for). Interpreted: " + repr(interpreted)
)

# The writer still interprets the endpoint, and a recorded window still
# answers only for the server that reported it.
CONSOLE_SEND_OUTCOMES.record_window(
    "llama_cpp", "qwen-small", 4096, True, endpoint="http://127.0.0.1:9099"
)
assert interpreted == ["http://127.0.0.1:9099"], interpreted
interpreted.clear()
assert _real_window_for("llama_cpp", "qwen-small", "http://127.0.0.1:9099/") == (
    4096,
    True,
)
assert interpreted == ["http://127.0.0.1:9099/"], interpreted
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


def test_readiness_with_no_recorded_window_never_interprets_the_endpoint(
    tmp_path: Path,
) -> None:
    """First-paint readiness does not interpret an endpoint it cannot use.

    Since dev's TASK-40300 the setup facts are loaded at first paint by
    readiness's own endpoint check, so this counts the ledger's endpoint
    interpretations (none with an empty ledger) instead of reading
    ``sys.modules`` after readiness runs. Importing readiness still leaves
    the setup facts unloaded, and that is checked before it runs.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _READINESS_SNIPPET)
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "FIRST_CHAT_READINESS_CLOSURE_OK" in result.stdout


_ENV_KEY_NOTICE_SNIPPET = """
import sys
from types import SimpleNamespace

from tldw_chatbook.app import TldwCli

MOD = "tldw_chatbook.app_env_key_notice"
assert MOD not in sys.modules, "importing the app loaded " + MOD

from tldw_chatbook.Chat.console_onboarding_state import EnvKeyFirstLaunchNotice

events = []
app = SimpleNamespace(
    notify=lambda message, **kwargs: events.append(message),
    _persist_env_key_notice_flag=lambda: events.append("flag-saved"),
    _pending_env_key_notice=None,
)
# A launch without exported keys holds no notice: the flush shows nothing.
TldwCli._show_pending_env_key_notice(app)
assert MOD not in sys.modules, "an empty notice flush loaded " + MOD
assert events == [], events

# First use: a held notice loads the bodies, is shown once, then recorded.
app._pending_env_key_notice = EnvKeyFirstLaunchNotice(
    title="Provider key found", message="Found ANTHROPIC_API_KEY", ready=False
)
TldwCli._show_pending_env_key_notice(app)
assert MOD in sys.modules
assert events == ["Found ANTHROPIC_API_KEY", "flag-saved"], events
assert app._pending_env_key_notice is None
assert app._env_key_notice_message == "Found ANTHROPIC_API_KEY"
print("ENV_KEY_NOTICE_FIRST_USE_OK")
"""


_SETUP_CARD_OFFERS_SNIPPET = """
import asyncio
import sys
from types import SimpleNamespace

from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Workbench.workbench_widgets import WorkbenchActionRequested

MOD = "tldw_chatbook.UI.Console_Modules.setup_card_offers"
assert MOD not in sys.modules, "importing the Console screen loaded " + MOD

# First use: an env-key offer click loads the module and reaches its commit,
# which looks for the card (this stub screen has none, so nothing commits).
asked = []


def query_one(selector, *args):
    asked.append(selector)
    raise LookupError(selector)


receiver = SimpleNamespace(query_one=query_one)
asyncio.run(
    ChatScreen.on_console_workbench_action_requested(
        receiver, WorkbenchActionRequested("use-env-key:anthropic")
    )
)
assert MOD in sys.modules
assert asked == ["#console-setup-modal"], asked
print("SETUP_CARD_OFFERS_FIRST_USE_OK")
"""


_PROVIDER_TEST_ROWS_SNIPPET = """
import sys

from tldw_chatbook.UI.Screens import settings_screen

MOD = "tldw_chatbook.UI.Settings_Modules.provider_test_rows"
assert MOD not in sys.modules, "importing the Settings screen loaded " + MOD

# First use: the Test rows' getter loads the module the rows read.
rows = settings_screen._test_rows()
assert MOD in sys.modules
assert rows is sys.modules[MOD]
assert callable(rows.key_verdict) and callable(rows.model_capacity)
print("PROVIDER_TEST_ROWS_FIRST_USE_OK")
"""


def test_the_env_key_notice_loads_only_once_a_notice_is_held(tmp_path: Path) -> None:
    """The app import leaves the env-key notice unloaded until one is held.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _ENV_KEY_NOTICE_SNIPPET)
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "ENV_KEY_NOTICE_FIRST_USE_OK" in result.stdout


def test_the_setup_card_offers_load_at_first_use(tmp_path: Path) -> None:
    """The Console screen import leaves the card's offers module unloaded.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _SETUP_CARD_OFFERS_SNIPPET)
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "SETUP_CARD_OFFERS_FIRST_USE_OK" in result.stdout


def test_the_settings_test_rows_load_at_first_use(tmp_path: Path) -> None:
    """The Settings screen import leaves the Test-row facts module unloaded.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's HOME/XDG.
    """
    result = _run_isolated_python(tmp_path, _PROVIDER_TEST_ROWS_SNIPPET)
    assert result.returncode == 0, (
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    assert "PROVIDER_TEST_ROWS_FIRST_USE_OK" in result.stdout
