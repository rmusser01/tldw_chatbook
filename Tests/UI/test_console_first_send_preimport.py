"""A Console's first send no longer pays its lazy imports on the UI loop (TASK-33620.15.5).

Measured live on origin/dev 4de437a5c5 (Anthropic haiku, 160x45, a probe
recording every module imported on the UI thread): the first send after launch
imported 54 modules on the UI loop between Enter and the provider call (hook
preparation and the hooks v2 package, the response-rules package, the visual
transcript with PIL, the agent's tool providers, the run-start catalog),
inside run-start callbacks of 40-118 ms, and nine more after the provider
call (the canvas settlement's in one 157 ms-CPU callback). Its configuration
capture also parsed ``MCP/server.py`` for the built-in manifest. The app's
post-UI-ready pre-import pass now does both first, at idle, on its daemon
thread.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.bootstrap_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt"

#: The live first-send record (origin/dev 4de437a5c5, 2026-10-19): the modules
#: the send imported on the UI loop between Enter and the provider call.
LIVE_FIRST_SEND_IMPORTS = (
    "UI.Console_Modules.send_acknowledgement",
    "UI.Console_Modules.sent_draft",
    "Chat.console_configuration_preparation",
    "UI.Console_Modules.turn_admission",
    "MCP.console_snapshot",
    "Chat.console_received_dispatch",
    "Chat.console_send_diagnostics",
    "Chat.console_hook_preparation",
    "Agents.automatic_work_runtime",
    "Chat.console_references",
    "Agents.hooks_v2.lifecycle",
    "Agents.hooks_v2.matching",
    "Agents.hooks_v2.models",
    "Agents.hooks_v2.validation",
    "Agents.hooks_v2.checkpoints",
    "Agents.hooks_v2.engine",
    "Agents.hooks_v2.budgets",
    "Agents.hooks_v2.command_executor",
    "Agents.hooks_v2.ownership",
    "Agents.hooks_v2.context",
    "Chat.console_legacy_flat_roots",
    "Chat.console_context_window",
    "Chat.console_visual_transcript",
    "Chat.console_predispatch_block",
    "Chat.console_native_commit",
    "Chat.console_trace_row_sources",
    "Chat.response_rules.generation",
    "Chat.response_rules.builder",
    "Chat.response_rules.evaluator",
    "Chat.response_rules.models",
    "Chat.response_rules.repository",
    "Chat.response_rules.resources",
    "Chat.response_rules.runtime",
    "Chat.response_rules.corrections",
    "Chat.response_rules.evidence",
    "Chat.response_rules.store",
    "Chat.response_rules.resolution",
    "Chat.response_rules.chat_state",
    "Agents.local_tool_provider",
    "Chat.console_run_start_tools",
    "MCP.console_tool_preparation",
    "Character_Chat.world_info_resolver",
    "Agents.persona_policy",
    "Tools.character_tool_service",
    "Tools.git_tool_impls",
    "Tools.web_tool_impls",
    "Web_Scraping.deep_search_citations",
    "Agents.ask_user_questions",
    "Agents.fs_read_ledger",
    "Chat.conversation_archive_actions",
    "Agents.canvas_tool_provider",
    "Canvas.guide",
    # After the provider call (the stream and the run's end), same record.
    "Chat.provider_rate_limits",
    "Canvas.native_authority",
    "Canvas.compiler",
    "Canvas.gateway",
    "Canvas.capabilities",
    "Canvas.control_protocol",
    "Tools.remote_session_registry",
    "Tools.remote_session_worker",
    "UI.Console_Modules.citation_counts",
)

_PASS_SCRIPT = """
import ast
import json
import sys

import tldw_chatbook.app as app_module
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (the initial route)
from tldw_chatbook.MCP import server

expected = json.loads(sys.argv[1])
cls = app_module.TldwCli
app = cls.__new__(cls)  # the pass reads only these
app._shutting_down = False
loaded_before = [m for m in expected if m in sys.modules]
at_routes = {}


def routes_start(self, routes):
    at_routes.update({m: m in sys.modules for m in expected})


cls._preimport_screens = routes_start
cls._pause_between_preimports = lambda self, gap: None  # pacing: unit-tested below
cls._preimport_heavy_screens(app)

# The first send's configuration capture reads the built-in manifest.
parses = []
real_parse = ast.parse


def counted(*args, **kwargs):
    parses.append(1)
    return real_parse(*args, **kwargs)


ast.parse = counted
server.describe_local_mcp_capabilities()
print(
    "PASS_JSON:"
    + json.dumps(
        {"before": loaded_before, "at_routes": at_routes, "manifest_parses": len(parses)}
    )
)
"""


@pytest.fixture(scope="module")
def pass_outcome(tmp_path_factory) -> dict:
    """Run the app's pre-import pass in a fresh interpreter (sys.modules is global)."""
    tmp_path = tmp_path_factory.mktemp("first_send_pass")
    home, data, config_dir = (tmp_path / n for n in ("home", "data", "config"))
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700)
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_CONFIG_HOME": str(config_dir),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)
    expected = [f"tldw_chatbook.{name}" for name in LIVE_FIRST_SEND_IMPORTS]
    result = subprocess.run(
        [sys.executable, "-c", _PASS_SCRIPT, json.dumps(expected)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    line = next(x for x in result.stdout.splitlines() if x.startswith("PASS_JSON:"))
    return json.loads(line[len("PASS_JSON:") :])


@pytest.mark.integration
def test_the_idle_pass_imports_the_first_send_modules_before_any_screen(pass_outcome):
    """Red on origin/dev: the pass warmed none of them."""
    assert pass_outcome["before"] == [], (
        "no longer lazy at UI-ready, drop from the list: "
        + ", ".join(pass_outcome["before"])
    )
    cold = sorted(m for m, loaded in pass_outcome["at_routes"].items() if not loaded)
    assert not cold, "the first send would import these on the UI loop: " + ", ".join(cold)


@pytest.mark.integration
def test_the_idle_pass_parses_the_manifest_source_for_the_first_send(pass_outcome):
    """Red on origin/dev: the first manifest read after the pass parsed server.py."""
    assert pass_outcome["manifest_parses"] == 0, (
        f"the first send's manifest read parsed server.py "
        f"{pass_outcome['manifest_parses']} time(s) on the UI loop"
    )


def test_every_live_first_send_import_is_listed():
    from tldw_chatbook.Chat.console_first_send_modules import FIRST_SEND_MODULES

    missing = [
        name
        for name in LIVE_FIRST_SEND_IMPORTS
        if f"tldw_chatbook.{name}" not in FIRST_SEND_MODULES
    ]
    assert not missing, missing


class _PassApp:
    """The three things the warm-up reads from the app's pre-import pass."""

    def __init__(self, stop_after: int | None = None) -> None:
        self._shutting_down = False
        self.pauses: list[float] = []
        self._stop_after = stop_after

    def _screen_preimport_pacing(self) -> tuple[float, float]:
        return (2.0, 0.5)

    def _pause_between_preimports(self, gap_seconds: float) -> None:
        self.pauses.append(gap_seconds)
        if self._stop_after is not None and len(self.pauses) >= self._stop_after:
            self._shutting_down = True


def test_the_warm_up_stops_on_quit_and_swallows_a_failed_import(monkeypatch):
    from tldw_chatbook.Chat import console_first_send_modules as warm

    imported: list[str] = []

    def fake_import(name):
        imported.append(name)
        if len(imported) == 2:
            raise ImportError("broken optional dependency")

    monkeypatch.setattr(warm.importlib, "import_module", fake_import)
    warm.preimport_first_send_modules(_PassApp(stop_after=3))

    assert imported == list(warm.FIRST_SEND_MODULES[:3])


def test_the_warm_up_hands_the_loop_a_paced_gap_after_every_import(monkeypatch):
    """Like the screen routes after it: a gap of cost x ratio, capped, per import.

    The pass's pause also parks while a navigation or accepted Console work
    owns the loop, so a send that starts mid-warm-up is not raced.
    """
    from tldw_chatbook.Chat import console_first_send_modules as warm

    clock = iter(float(n) * 0.1 for n in range(10_000))
    monkeypatch.setattr(warm.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(warm.importlib, "import_module", lambda name: None)
    primed: list[bool] = []
    monkeypatch.setattr(
        "tldw_chatbook.MCP.server.prime_local_manifest_source",
        lambda: primed.append(True),
    )
    app = _PassApp()

    warm.preimport_first_send_modules(app)

    # Each import "took" 0.1 s on the fake clock: gap 0.1 x 2.0 = 0.2 s.
    assert app.pauses == [pytest.approx(0.2)] * len(warm.FIRST_SEND_MODULES)
    assert primed == [True]


#: Frames that mark the send's own work (the acknowledgement, admission and run
#: start); the UI's ticks and polls lazily import their own modules meanwhile.
SEND_FRAMES = frozenset(
    {
        "request_visible_send",
        "schedule_acknowledged_send",
        "_start_send",
        "_send_console_message_from_visible_action",
        "_run_custodied_turn",
    }
)


def _in_send() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name in SEND_FRAMES:
            return True
        frame = frame.f_back
    return False


@pytest.mark.asyncio
async def test_a_harness_send_imports_nothing_lazy_beyond_the_list():
    """Every import a send runs inside a function is warm at UI-ready or listed.

    The mounted harness's send (llama.cpp, a held gateway, no agent tools)
    covers admission, hooks and the durable commit; the live record covers
    the agent's tool providers. Function-local import statements under the
    send's own frames are recorded whether or not the module is already
    loaded, so the check does not depend on what earlier tests imported.
    """
    from Tests.UI.test_console_send_acknowledgement import (
        build,
        eager_tasks,
        press,
        ready_console,
        until,
    )
    from tldw_chatbook.Chat.console_first_send_modules import FIRST_SEND_MODULES

    ready = {line for line in SNAPSHOT.read_text().splitlines() if not line.startswith("#")}
    covered = set(FIRST_SEND_MODULES) | {
        name.rsplit(".", depth)[0] for name in FIRST_SEND_MODULES for depth in (1, 2)
    }
    seen: dict[str, str] = {}
    real = builtins.__import__
    main = threading.main_thread()

    def recording(name, globals=None, locals=None, fromlist=(), level=0):
        if threading.current_thread() is main:
            caller = sys._getframe(1).f_code
            if caller.co_name != "<module>" and _in_send():
                full = name
                if level:
                    package = (globals or {}).get("__package__")
                    full = importlib.util.resolve_name("." * level + name, package)
                seen.setdefault(full, caller.co_qualname)
        return real(name, globals, locals, fromlist, level)

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            await ready_console(host, pilot, gateway)
            builtins.__import__ = recording
            try:
                press(host, "enter", "\r")
                await until(gateway.validation_started.is_set)
                gateway.validation_release.set()
                await until(lambda: gateway.stream_calls == 1)
            finally:
                builtins.__import__ = real

    lazy = {
        name: where
        for name, where in seen.items()
        if name.startswith("tldw_chatbook") and name not in ready and name not in covered
    }
    assert not lazy, f"imported on the UI loop by the send, not warmed: {lazy}"
    assert "tldw_chatbook.Agents.hooks_v2.lifecycle" in seen, "the send never ran hooks"
