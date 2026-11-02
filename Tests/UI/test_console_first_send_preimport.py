"""A Console's first send no longer pays its lazy imports on the UI loop (TASK-33620.15.5).

Measured live on origin/dev 4de437a5c5 (Anthropic haiku, 160x45, a probe
recording every module imported on the UI thread): the first send after launch
imported 52 modules and three packages on the UI loop between Enter and the
provider call (hook preparation and the hooks v2 package, the response-rules
package, the visual transcript with PIL, the agent's tool providers, the
run-start catalog), inside run-start callbacks of 40-118 ms, and nine more
after the provider call (the canvas settlement's in one 157 ms-CPU callback).
Its configuration capture also parsed ``MCP/server.py`` for the built-in
manifest. The app's post-UI-ready pre-import pass now does both first, at
idle, on its daemon thread.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from Tests.private_profile import private_profile_test

pytestmark = pytest.mark.bootstrap_profile

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = REPO_ROOT / "Tests/Performance/boot_budget_snapshots/ui_ready_modules.txt"

#: The live first-send record (origin/dev 4de437a5c5, 2026-10-19): the modules
#: the send imported on the UI loop between Enter and the provider call,
#: reconciled with current send seams. console_received_dispatch is now
#: resident through console_spend_projection and is no longer a lazy step.
LIVE_FIRST_SEND_IMPORTS = (
    "UI.Console_Modules.send_acknowledgement",
    "UI.Console_Modules.sent_draft",
    "Chat.console_configuration_preparation",
    "UI.Console_Modules.turn_admission",
    "MCP.console_snapshot",
    "Chat.console_send_diagnostics",
    "Chat.console_plugin_context",  # current native invocation projection
    "Chat.console_hook_preparation",
    "Agents.hook_permissions",  # current prepare_hooks_v2 lazy consent owner
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
    "Chat.console_prompt_mixin_service",
    "DB.console_prompt_mixins",
    "Chat.console_pre_provider_stop",
    # Added after that record (TASK-35582): begin_generation_attempt's lazy
    # continuation-binding import, deferred off first paint.
    "Chat.console_continuation_binding",
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
    "UI.Console_Modules.citation_counts",
)

#: TASK-33620.15.19, live on origin/dev 7536ac3b87 (2026-10-30, Anthropic
#: haiku, 160x45, a note in Library, the conversation's Auto set to Automatic,
#: RAG profile BM25 Only, first send 60 s after launch with the warm-up done):
#: the only modules the UI loop loaded between Enter and the provider call
#: were these two, both inside the send's own run-start callback (5-30 ms wall
#: across five launches), where Library preparation begins. Everything else
#: the retrieval path reaches is resident at UI-ready, and the Library
#: service's runtime build runs on a worker thread. They cost 0.5-3.6 ms wall
#: per first Automatic send, and they stay lazy, out of ``FIRST_SEND_MODULES``:
#: warming them would raise ADR-097's first-send warm-up row, and the owner
#: declined that on 2026-10-31 ("Don't raise it").
LIVE_LIBRARY_FIRST_SEND_IMPORTS = (
    "Chat.console_library_preparation",
    "Chat.console_library_search",
)

_PASS_SCRIPT = """
import ast
import json
import sys

import tldw_chatbook.app as app_module
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (the initial route)

# Loaded by the app and its initial route: resident at UI-ready in a live app.
resident = sorted(
    m for m in sys.modules if m == "tldw_chatbook" or m.startswith("tldw_chatbook.")
)
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
after = [m for m in expected if m in sys.modules]

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
        {
            "before": loaded_before,
            "at_routes": at_routes,
            "after": after,
            "manifest_parses": len(parses),
            "resident": resident,
        }
    )
)
"""


def _run_isolated(
    tmp_path: Path, script: str, *args: str
) -> subprocess.CompletedProcess:
    """Run ``script`` in a fresh interpreter with a scratch HOME (sys.modules is global)."""
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
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result


@pytest.fixture(scope="module")
def pass_outcome(tmp_path_factory) -> dict:
    """Run the app's pre-import pass in a fresh interpreter (sys.modules is global)."""
    expected = [f"tldw_chatbook.{name}" for name in LIVE_FIRST_SEND_IMPORTS]
    result = _run_isolated(
        tmp_path_factory.mktemp("first_send_pass"), _PASS_SCRIPT, json.dumps(expected)
    )
    line = next(x for x in result.stdout.splitlines() if x.startswith("PASS_JSON:"))
    return json.loads(line[len("PASS_JSON:") :])


#: The pass as it runs live: the screen routes, then the warm-up, each
#: warm-up step recording the third-party packages it loaded (TASK-33620.15.18).
_STEPS_SCRIPT = """
import importlib
import json
import sys

from loguru import logger

import tldw_chatbook.app as app_module
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (the initial route)
from tldw_chatbook.Chat import console_first_send_modules as warm

cls = app_module.TldwCli
app = cls.__new__(cls)  # the pass reads only these
app._shutting_down = False
app.loguru_logger = logger
cls._pause_between_preimports = lambda self, gap: None  # pacing: unit-tested
# The routes run first, as live: whatever they load is resident already.
app._preimport_screens(app._screen_preimport_route_order())
stdlib = set(sys.stdlib_module_names)


def third_party():
    # A private top-level module (``_cython_3_2_3``, ``_cffi_backend``) is a
    # compiled package's own runtime, loaded with it.
    return {
        name.split(".")[0]
        for name, module in list(sys.modules.items())
        if module is not None
        and name.split(".")[0] not in stdlib
        and not name.startswith(("tldw_chatbook", "_"))
    }


steps = []


class Recorder:
    def import_module(self, name, package=None):
        before = third_party()
        try:
            return importlib.import_module(name, package)
        finally:
            steps.append([name, sorted(third_party() - before)])

    def __getattr__(self, attr):
        return getattr(importlib, attr)


warm.importlib = Recorder()
warm.preimport_first_send_modules(app, app._finish_first_send_warmup)
print("STEPS_JSON:" + json.dumps(steps))
"""


@pytest.fixture(scope="module")
def warm_up_steps(tmp_path_factory) -> list[list]:
    """``[step, [third-party packages it loaded]]`` for the live pass's warm-up."""
    result = _run_isolated(tmp_path_factory.mktemp("first_send_steps"), _STEPS_SCRIPT)
    line = next(x for x in result.stdout.splitlines() if x.startswith("STEPS_JSON:"))
    return json.loads(line[len("STEPS_JSON:") :])


@pytest.mark.integration
def test_the_idle_pass_imports_the_first_send_modules(pass_outcome):
    """Red on origin/dev: the pass warmed none of them."""
    assert pass_outcome["before"] == [], (
        "no longer lazy at UI-ready, drop from the list: "
        + ", ".join(pass_outcome["before"])
    )
    cold = sorted(set(pass_outcome["at_routes"]) - set(pass_outcome["after"]))
    assert not cold, "the first send would import these on the UI loop: " + ", ".join(
        cold
    )


@pytest.mark.integration
def test_the_idle_pass_warms_every_screen_route_before_the_first_send_modules(
    pass_outcome,
):
    """TASK-33620.15.14: the warm-up runs after the last screen route.

    Run first, it pushed every speculative screen import back by its own
    import time and paced gaps, so a first navigation soon after boot found
    its screen cold where origin/dev had warmed it. Red on the branch's first
    order (warm-up, then routes).
    """
    early = sorted(m for m, loaded in pass_outcome["at_routes"].items() if loaded)
    assert not early, "warmed before the screen routes: " + ", ".join(early)


@pytest.mark.integration
def test_the_idle_pass_parses_the_manifest_source_for_the_first_send(pass_outcome):
    """Red on origin/dev: the first manifest read after the pass parsed server.py."""
    assert pass_outcome["manifest_parses"] == 0, (
        f"the first send's manifest read parsed server.py "
        f"{pass_outcome['manifest_parses']} time(s) on the UI loop"
    )


#: What a listed package imports only when it is installed (aiohttp 3.13:
#: ``resolver.py`` tries aiodns, which loads pycares; ``compression_utils.py``
#: tries brotlicffi or brotli and backports.zstd; ``aiohttp[speedups]``
#: installs them). The measured venv had none of them, so live they never
#: loaded; where they are installed they load inside aiohttp's own step.
_OPTIONAL_ACCELERATORS: dict[str, frozenset[str]] = {
    "aiohttp": frozenset({"aiodns", "pycares", "brotli", "brotlicffi", "backports"}),
}


@pytest.mark.integration
def test_no_warm_up_step_loads_a_third_party_package_it_does_not_name(warm_up_steps):
    """A navigation waits behind one small step at most (TASK-33620.15.18).

    The warm-up parks before each step while a navigation holds the loop, but
    a step it has started runs to its end, sharing the GIL with the
    navigation. Live on origin/dev 7536ac3b87 (160x45, established profiles,
    an idle host), ``Canvas.native_authority`` was one step of 85-222 ms
    (63-70 ms CPU): importing it loaded aiohttp, html5lib, tinycss2 and
    their dependencies, 121 modules. A Library click landing in it drew
    Library's first frame 181-216 ms after the click reached the app on a
    lightly loaded host, against 141-165 ms on origin/dev without the
    warm-up at the same moment (paired median +48.6 ms from the click's
    send, slower in 8 of 8 pairs at all loads). Keys landing in it were not
    slower.

    So each third-party package a send needs is its own step, after its
    dependencies: a step loads no third-party package but the one it names,
    apart from an optional accelerator that package imports only when it is
    installed (the list cannot name a package that may be absent).
    """
    assert len(warm_up_steps) > 60, "the warm-up recorded no steps"
    hidden = {
        step: packages
        for step, packages in warm_up_steps
        if set(packages)
        - {step.split(".")[0]}
        - _OPTIONAL_ACCELERATORS.get(step, frozenset())
    }
    assert not hidden, (
        "these warm-up steps load third-party packages they do not name; list "
        "each package, after its dependencies, before the step: " + repr(hidden)
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
        self.events: list[tuple[str, object]] = []
        self._stop_after = stop_after

    def _finish_first_send_warmup(self):
        pass

    def call_from_thread(self, callback, *args):
        self.events.append(("callback", (callback, args)))

    def _screen_preimport_pacing(self) -> tuple[float, float]:
        return (2.0, 0.5)

    def _pause_between_preimports(self, gap_seconds: float) -> None:
        self.pauses.append(gap_seconds)
        self.events.append(("pause", gap_seconds))
        if self._stop_after is not None and len(self.pauses) >= self._stop_after:
            self._shutting_down = True


@pytest.mark.parametrize("manifest_fault", [False, True])
def test_real_import_completion_queues_one_ui_request_without_boot_freezes(
    monkeypatch, tmp_path, manifest_fault
):
    """A completed pass requests runtime maintenance, even after manifest failure."""
    from tldw_chatbook.Chat import console_first_send_modules as warm
    from tldw_chatbook.MCP import server
    from tldw_chatbook.Utils import ui_responsiveness

    app = _PassApp()
    (tmp_path / "_real_first_send_step.py").write_text("IMPORTED = True\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "_real_first_send_step", raising=False)
    monkeypatch.setattr(
        warm, "FIRST_SEND_MODULES", ("_real_first_send_step", "_missing_optional_step")
    )
    monkeypatch.setattr(warm, "seconds_since_user_input", lambda: float("inf"))
    freezes = []
    # Both the old imported alias and the shared helper would expose an
    # accidental direct freeze. Real reclamation is tested in fresh children.
    monkeypatch.setattr(warm, "freeze_long_lived_heap", freezes.append, raising=False)
    monkeypatch.setattr(ui_responsiveness, "freeze_long_lived_heap", freezes.append)

    def prime():
        if manifest_fault:
            raise ValueError("optional source parse failed")

    monkeypatch.setattr(server, "prime_local_manifest_source", prime)
    try:
        warm.preimport_first_send_modules(app, app._finish_first_send_warmup)
        assert sys.modules["_real_first_send_step"].IMPORTED
        assert "_missing_optional_step" not in sys.modules
        assert freezes == []
        assert [event for event in app.events if event[0] == "callback"] == [
            ("callback", (app._finish_first_send_warmup, ()))
        ]
    finally:
        sys.modules.pop("_real_first_send_step", None)


@pytest.mark.parametrize("state", ["replaced", "detached", "shutdown", "no-runtime"])
def test_completion_callback_resolves_current_attachment_on_ui_loop(monkeypatch, state):
    """Queued callbacks never retain the worker's former view."""
    from types import SimpleNamespace
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Console_Modules import transcript

    old, current = object(), object()
    runtime = SimpleNamespace(view=old)
    app = SimpleNamespace(console_runtime=runtime, _shutting_down=False)
    queued = (TldwCli._finish_first_send_warmup, (app,))
    runtime.view = current if state == "replaced" else None
    if state == "shutdown":
        runtime.view = current
        app._shutting_down = True
    elif state == "no-runtime":
        del app.console_runtime
    requests = []
    monkeypatch.setattr(
        transcript, "request_console_heap_refreeze", lambda *args: requests.append(args)
    )
    queued[0](*queued[1])
    assert requests == (
        [(current, "first_send_preimport")] if state == "replaced" else []
    )


def test_the_warm_up_stops_on_quit_and_swallows_a_failed_import(monkeypatch):
    from tldw_chatbook.Chat import console_first_send_modules as warm

    imported: list[str] = []
    frozen: list[str] = []

    def fake_import(name):
        imported.append(name)
        if len(imported) == 1:
            raise ImportError("broken optional dependency")

    monkeypatch.setattr(warm.importlib, "import_module", fake_import)
    monkeypatch.setattr(warm, "freeze_long_lived_heap", frozen.append, raising=False)
    monkeypatch.setattr(warm, "seconds_since_user_input", lambda: float("inf"))
    app = _PassApp(stop_after=3)
    warm.preimport_first_send_modules(app, app._finish_first_send_warmup)

    # Every import waits behind a pause (the park); quit lands in the third.
    assert imported == list(warm.FIRST_SEND_MODULES[:2])
    assert frozen == []


def test_the_warm_up_parks_before_every_import_and_paces_each_gap(monkeypatch):
    """Like the screen routes before it: a park, then a gap of cost x ratio, capped.

    The warm-up starts right after the last screen route, so its first pause
    has no gap of its own but still parks while a navigation or accepted
    Console work owns the loop: a send that starts as the routes finish
    imports what it needs itself, unraced. When it is done it freezes the
    heap again (ADR-198), since the routes' freeze ran before it.
    """
    from tldw_chatbook.Chat import console_first_send_modules as warm
    from tldw_chatbook.MCP import server as mcp_server

    app = _PassApp()
    clock = iter(float(n) * 0.1 for n in range(10_000))
    monkeypatch.setattr(warm.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(warm, "seconds_since_user_input", lambda: float("inf"))
    monkeypatch.setattr(
        warm.importlib,
        "import_module",
        lambda name: app.events.append(("import", name)),
    )
    # Patched by object, not by dotted path: `warm.importlib` is the global
    # importlib, and pytest 9 resolves a dotted path through
    # importlib.import_module, which the fake above has just replaced
    # (TASK-33620.15.15: the PR lanes install pytest 9).
    monkeypatch.setattr(
        mcp_server,
        "prime_local_manifest_source",
        lambda: app.events.append(("prime", None)),
    )
    monkeypatch.setattr(
        warm,
        "freeze_long_lived_heap",
        lambda reason: app.events.append(("freeze", reason)),
        raising=False,
    )

    warm.preimport_first_send_modules(app, app._finish_first_send_warmup)

    # Each import "took" 0.1 s on the fake clock: gap 0.1 x 2.0 = 0.2 s,
    # before every later import and before the manifest parse.
    assert app.pauses == [0.0] + [pytest.approx(0.2)] * len(warm.FIRST_SEND_MODULES)
    assert app.events[:2] == [("pause", 0.0), ("import", warm.FIRST_SEND_MODULES[0])]
    assert app.events[-2:] == [
        ("prime", None),
        ("callback", (app._finish_first_send_warmup, ())),
    ]


def test_the_warm_up_waits_out_recent_input_then_parks_again(monkeypatch):
    """No step starts within the input-quiet time of any input (TASK-33620.15.18).

    A navigation takes the app's lock only when its click reaches the
    handler. Live (fix head d9320f721e, 160x45, idle host), the pass's park
    checked the lock 1-7 ms after a Library click arrived, found it free and
    started the next step: ``Canvas.gateway``, 46 ms, inside the click's
    navigation; the lock was held by 27-53 ms. At a random moment of the
    window a click drew Library's first frame a paired median +9 ms later
    than without the warm-up (10 launches), keys +0.3 ms.

    So before each step the warm-up waits until no input has arrived for
    ``INPUT_QUIET_SECONDS``, then parks again for a navigation that started
    meanwhile. Negative control: with no recent input it waits for nothing.
    """
    from tldw_chatbook.Chat import console_first_send_modules as warm
    from tldw_chatbook.MCP import server as mcp_server

    app = _PassApp()
    ages = iter([0.04, 0.12, float("inf")])  # a click 40 ms ago, then quiet
    monkeypatch.setattr(
        warm, "seconds_since_user_input", lambda: next(ages, float("inf"))
    )

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return 0.0

        @staticmethod
        def sleep(seconds: float) -> None:
            app.events.append(("sleep", round(seconds, 3)))

    monkeypatch.setattr(warm, "time", Clock)
    monkeypatch.setattr(
        warm.importlib,
        "import_module",
        lambda name: app.events.append(("import", name)),
    )
    monkeypatch.setattr(mcp_server, "prime_local_manifest_source", lambda: None)
    monkeypatch.setattr(
        warm, "freeze_long_lived_heap", lambda reason: None, raising=False
    )

    warm.preimport_first_send_modules(app, app._finish_first_send_warmup)

    quiet = warm.INPUT_QUIET_SECONDS
    assert app.events[:6] == [
        ("pause", 0.0),
        ("sleep", round(quiet - 0.04, 3)),
        ("pause", 0.0),
        ("sleep", round(quiet - 0.12, 3)),
        ("pause", 0.0),
        ("import", warm.FIRST_SEND_MODULES[0]),
    ], "a step started inside the input-quiet time of a click"
    # Negative control: input long past, so each later step waits for nothing.
    assert ("sleep", round(quiet, 3)) not in app.events
    assert app.events[6:8] == [("pause", 0.0), ("import", warm.FIRST_SEND_MODULES[1])]


def test_quit_stops_the_warm_up_while_it_waits_out_input(monkeypatch):
    """Quitting ends the input wait at once, whatever the user does (TASK-33620.15.18).

    Input that never pauses keeps the warm-up waiting before its next step,
    so that wait must watch for quit itself, or the pass's daemon thread
    would outlive the app's shutdown. Here every read says input arrived
    just now; quit lands in the third park, and the warm-up then imports
    nothing and freezes nothing.
    """
    from tldw_chatbook.Chat import console_first_send_modules as warm

    app = _PassApp(stop_after=3)
    imported: list[str] = []
    frozen: list[str] = []
    monkeypatch.setattr(warm, "seconds_since_user_input", lambda: 0.0)

    class Clock:
        @staticmethod
        def monotonic() -> float:
            return 0.0

        @staticmethod
        def sleep(seconds: float) -> None:
            app.events.append(("sleep", round(seconds, 3)))
            assert len(app.events) < 50, "the warm-up kept waiting after quit"

    monkeypatch.setattr(warm, "time", Clock)
    monkeypatch.setattr(warm.importlib, "import_module", imported.append)
    monkeypatch.setattr(warm, "freeze_long_lived_heap", frozen.append, raising=False)

    warm.preimport_first_send_modules(app, app._finish_first_send_warmup)

    quiet = round(warm.INPUT_QUIET_SECONDS, 3)
    assert app.events == [
        ("pause", 0.0),
        ("sleep", quiet),
        ("pause", 0.0),
        ("sleep", quiet),
        ("pause", 0.0),
    ], "the warm-up did not stop waiting when the app quit"
    assert imported == [] and frozen == []


def _raise(*_args, **_kwargs):
    raise RuntimeError("the pass's pacing is unavailable")


@pytest.mark.parametrize(
    "broken", ["_screen_preimport_pacing", "_pause_between_preimports"]
)
def test_a_failing_warm_up_ends_quietly_after_the_screen_routes(monkeypatch, broken):
    """The warm-up runs on the pass's thread after the routes: a raise ends only it.

    Its pacing and park calls are the app's own, so a fault there (or in a
    later refactor of either) must not escape the pass thread, and the
    screen routes must already have run.
    """
    import tldw_chatbook.app as app_module
    from tldw_chatbook.Chat import console_first_send_modules as warm

    cls = app_module.TldwCli
    app = cls.__new__(cls)  # the pass reads only what is patched here
    app._shutting_down = False
    order: list[tuple[str, object]] = []
    monkeypatch.setattr(warm.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(
        warm,
        "freeze_long_lived_heap",
        lambda r: order.append(("freeze", r)),
        raising=False,
    )
    monkeypatch.setattr(cls, broken, _raise)
    monkeypatch.setattr(cls, "_screen_preimport_route_order", lambda self: ["routes"])
    monkeypatch.setattr(
        cls, "_preimport_screens", lambda self, routes: order.append(("routes", routes))
    )
    if broken != "_screen_preimport_pacing":
        monkeypatch.setattr(cls, "_screen_preimport_pacing", lambda self: (1.0, 0.5))

    cls._preimport_heavy_screens(app)

    assert order == [("routes", ["routes"])]


def _warm_at_ui_ready(outcome: dict) -> set[str]:
    """The ``tldw_chatbook`` modules a send's import finds already loaded.

    The ADR-097 snapshot (what a live app has loaded at ``_ui_ready``, mount
    work included) united with what importing the app and its initial route
    loads in a fresh interpreter: the census checks the snapshot's count, not
    its contents, so a module that became resident since its last re-pin is
    missing from it.
    """
    snapshot = {
        line for line in SNAPSHOT.read_text().splitlines() if not line.startswith("#")
    }
    return snapshot | set(outcome["resident"])


def _covered_by_the_warm_up() -> set[str]:
    """``FIRST_SEND_MODULES`` and their parent packages."""
    from tldw_chatbook.Chat.console_first_send_modules import FIRST_SEND_MODULES

    return set(FIRST_SEND_MODULES) | {
        name.rsplit(".", depth)[0] for name in FIRST_SEND_MODULES for depth in (1, 2)
    }


def _not_warmed(
    seen: dict[str, str], warm: set[str], covered: set[str]
) -> dict[str, str]:
    return {
        name: where
        for name, where in seen.items()
        if name.startswith("tldw_chatbook") and name not in warm and name not in covered
    }


def test_the_send_guard_counts_a_module_resident_at_ui_ready_as_warm(pass_outcome):
    """The guard's warm set is what a live app has loaded, not only the snapshot.

    The ADR-097 census enforces the snapshot's count, not its contents, and on
    origin/dev 7700c63a68 it lacks ``pricing_display``, which the listed
    ``console_spend_projection`` imports at module scope. The cost chip's
    rebuild re-imports it inside ``pricing_snapshot_options``; when that
    rebuild fell inside the send's frames, the harness guard below failed on
    correct code (1 run in 26, this record).
    """
    warm = _warm_at_ui_ready(pass_outcome)
    resident_dispatch = "tldw_chatbook.Chat.console_received_dispatch"
    assert resident_dispatch in pass_outcome["resident"]
    assert _not_warmed({resident_dispatch: "spend projection"}, warm, set()) == {}
    flake = {
        "tldw_chatbook.UI.Console_Modules.pricing_display": "pricing_snapshot_options"
    }
    assert _not_warmed(flake, warm, _covered_by_the_warm_up()) == {}
    # Still a guard: the send's own lazy modules are not warm at UI-ready.
    record = {
        f"tldw_chatbook.{name}": "live"
        for name in LIVE_FIRST_SEND_IMPORTS + LIVE_LIBRARY_FIRST_SEND_IMPORTS
    }
    assert _not_warmed(record, warm, set()) == record


#: Frames that mark the send's own work (the acknowledgement, admission and run
#: start); the UI's ticks and polls lazily import their own modules meanwhile.
SEND_FRAMES = frozenset(
    {
        "request_visible_send",
        "schedule_acknowledged_send",
        "_start_send",
        "_send_console_message_from_visible_action",
        "_run_custodied_turn",
        "run_received_intent",
        "_run_received_intent_bound",
    }
)


def _in_send() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name in SEND_FRAMES:
            return True
        frame = frame.f_back
    return False


@contextlib.contextmanager
def _recording_send_imports() -> Iterator[dict[str, str]]:
    """Record every function-local import statement the send's frames run.

    Yields the record (module name -> the importing function's qualname),
    filled while the block runs. Import statements are recorded whether or
    not the module is already loaded, so the record does not depend on what
    earlier tests imported.
    """
    seen: dict[str, str] = {}
    real = builtins.__import__
    main = threading.main_thread()

    def recording(name, globals=None, locals=None, fromlist=(), level=0):
        full = None
        if threading.current_thread() is main:
            caller = sys._getframe(1).f_code
            if caller.co_name != "<module>" and _in_send():
                full = name
                if level:
                    package = (globals or {}).get("__package__")
                    full = importlib.util.resolve_name("." * level + name, package)
                seen.setdefault(full, caller.co_qualname)
        module = real(name, globals, locals, fromlist, level)
        if full is not None:
            # ``from pkg import submodule`` imports the submodule inside the
            # import system, past this hook; record each fromlist name that
            # is a loaded submodule, or the package (covered as a parent of
            # a listed module) would hide it.
            for item in fromlist or ():
                if item != "*" and f"{full}.{item}" in sys.modules:
                    seen.setdefault(f"{full}.{item}", caller.co_qualname)
        return module

    builtins.__import__ = recording
    try:
        yield seen
    finally:
        builtins.__import__ = real


@contextlib.contextmanager
def _recording_cold_send_loads() -> Iterator[dict[str, str]]:
    """Observe successful cold loads on the UI thread under send frames."""
    seen: dict[str, str] = {}
    real = importlib._bootstrap._find_and_load
    main = threading.main_thread()

    def recording(name, import_):
        cold = (
            threading.current_thread() is main
            and name not in sys.modules
            and _in_send()
        )
        if cold:
            frame = sys._getframe(1)
            stack = []
            while frame is not None:
                stack.append(
                    f"{frame.f_globals.get('__name__')}:{frame.f_code.co_qualname}"
                )
                frame = frame.f_back
        module = real(name, import_)
        if cold:
            seen.setdefault(name, " -> ".join(stack))
        return module

    importlib._bootstrap._find_and_load = recording
    try:
        yield seen
    finally:
        importlib._bootstrap._find_and_load = real


def test_cold_send_load_recorder_observes_module_scope_external_dependencies(
    tmp_path, monkeypatch
):
    root = "pr3339_recorder_root"
    package = "pr3339_recorder_external"
    (tmp_path / f"{root}.py").write_text(
        f"from {package} import child\n", encoding="utf-8"
    )
    (tmp_path / package).mkdir()
    (tmp_path / package / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / package / "child.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    names = {root, package, f"{package}.child"}
    assert names.isdisjoint(sys.modules)

    def _start_send(name=root):
        return importlib.import_module(name)

    real_loader = importlib._bootstrap._find_and_load
    with _recording_cold_send_loads() as seen:
        _start_send()
    assert names <= seen.keys(), seen
    assert importlib._bootstrap._find_and_load is real_loader

    (tmp_path / "pr3339_recorder_failure.py").write_text(
        "raise ValueError('loader failure')\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="loader failure"):
        with _recording_cold_send_loads() as failed:
            _start_send("pr3339_recorder_failure")
    assert "pr3339_recorder_failure" not in failed
    assert importlib._bootstrap._find_and_load is real_loader


@pytest.mark.asyncio
async def test_a_harness_send_imports_nothing_lazy_beyond_the_list(pass_outcome):
    """Every import a send runs inside a function is warm at UI-ready or listed.

    The mounted harness's send (llama.cpp, a held gateway, no agent tools)
    covers admission, hooks and the durable commit; the live record covers
    the agent's tool providers. Function-local import statements under the
    send's own frames are recorded whether or not the module is already
    loaded, so the check does not depend on what earlier tests imported.
    Which UI-loop work lands inside those frames depends on timing (the
    cost chip's rebuild sometimes does), so "warm" is everything resident at
    UI-ready, not only the snapshot's possibly stale contents.
    """
    from Tests.UI.test_console_send_acknowledgement import (
        build,
        eager_tasks,
        press,
        ready_console,
        until,
    )

    ready = _warm_at_ui_ready(pass_outcome)
    covered = _covered_by_the_warm_up()

    host, gateway, _timeline = build()
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            await ready_console(host, pilot, gateway)
            with _recording_send_imports() as seen:

                def _start_send():
                    from tldw_chatbook.Chat import console_voice_input

                    return console_voice_input

                # Exercise this installed recorder's fromlist expansion in a
                # send frame; the real send need not perform dictation sync.
                _start_send()
                assert "tldw_chatbook.Chat.console_voice_input" in seen
                seen.clear()
                press(host, "enter", "\r")
                await until(gateway.validation_started.is_set)
                gateway.validation_release.set()
                await until(lambda: gateway.stream_calls == 1)

    lazy = _not_warmed(seen, ready, covered)
    assert not lazy, f"imported on the UI loop by the send, not warmed: {lazy}"
    assert "tldw_chatbook.Agents.hooks_v2.lifecycle" in seen, "the send never ran hooks"


_EVIDENCE = "The Fresnel lens lets a lighthouse beam reach ships far away."


class _EvidenceLibrary:
    """A started Library whose search finds one note, as the live BM25 run did."""

    def __init__(self, note_id: str) -> None:
        self.note_id = note_id
        self.prewarms = 0
        self.searches = 0

    async def prewarm(self) -> bool:
        self.prewarms += 1
        return True

    async def search(self, query, source_types, mode, **kwargs):
        self.searches += 1
        return {
            "runtime_backend": "local",
            "results": [
                {
                    "source_id": self.note_id,
                    "chunk_id": "chunk-1",
                    "title": "Lighthouse keeping notes",
                    "snippet": _EVIDENCE,
                    "score": 0.9,
                    "source_type": "notes",
                }
            ],
        }


@pytest.mark.asyncio
@private_profile_test
async def test_an_automatic_library_send_loads_only_the_two_library_modules_lazily(
    pass_outcome,
    request,
    monkeypatch,
    tmp_path,
):
    """A private provider-direct Automatic send cold-loads only the Library pair.

    Real preimports run first; a file-backed note passes actual source authority
    and evidence capture before its text reaches the provider request. The old
    statement guard remains independent of the all-namespace cold-load guard.
    Historical timings and the owner decision are documented in the task.
    """
    from Tests.UI.test_console_send_acknowledgement import (
        build,
        eager_tasks,
        press,
        ready_console,
        until,
    )
    from tldw_chatbook.Chat.console_library_policy import (
        ConsoleAssistantLibraryAccess,
        ConsoleAutoRetrieve,
        ConsoleLibraryPolicyCandidate,
    )

    from tldw_chatbook import config as config_module
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    private_root = Path(os.environ["TLDW_TEST_CONFIG_ROOT"]).resolve(strict=True)
    assert config_module.get_cli_config_path().resolve().is_relative_to(private_root)
    assert config_module.save_setting_to_cli_config("console", "agent_runtime", False)
    ready = _warm_at_ui_ready(pass_outcome)
    covered = _covered_by_the_warm_up()
    host, gateway, _timeline = build()
    app = host.app_instance
    app.app_config.setdefault("console", {})["agent_runtime"] = False
    db = CharactersRAGDB(tmp_path / "notes.sqlite", client_id="first-send-imports")
    request.addfinalizer(db.close_connection)
    app.chachanotes_db = db
    app._wire_chat_conversation_services()
    note_id = db.add_note(title="Lighthouse keeping notes", content=_EVIDENCE)
    library = _EvidenceLibrary(note_id)
    app.library_rag_search_service = library
    requests: list[object] = []
    held_stream = gateway.stream_chat

    async def recorded_stream(resolution, messages, **kwargs):
        requests.append(messages)
        async for chunk in held_stream(resolution, messages, **kwargs):
            yield chunk

    gateway.stream_chat = recorded_stream
    async with host.run_test(size=(160, 45)) as pilot:
        with eager_tasks():
            console, _composer = await ready_console(host, pilot, gateway)
            store = console._ensure_console_chat_controller().store
            store.stage_session_library_policy(
                store.active_session_id,
                ConsoleLibraryPolicyCandidate(
                    auto_retrieve=ConsoleAutoRetrieve.AUTOMATIC,
                    assistant_access=ConsoleAssistantLibraryAccess.BLOCKED,
                ),
            )
            from tldw_chatbook.Chat import console_first_send_modules as warm
            from tldw_chatbook.MCP import server

            completed = []
            prime = server.prime_local_manifest_source

            def record_prime():
                result = prime()
                completed.append(True)
                return result

            monkeypatch.setattr(server, "prime_local_manifest_source", record_prime)
            monkeypatch.setattr(
                host.app_instance, "_pause_between_preimports", lambda gap: None
            )
            await asyncio.to_thread(host.app_instance._preimport_heavy_screens)
            assert completed == [
                True
            ], "first-send preimport did not finish manifest prime"
            assert (
                warm.FIRST_SEND_MODULES[-1] in sys.modules
            ), "last first-send module absent"
            pair = {f"tldw_chatbook.{name}" for name in LIVE_LIBRARY_FIRST_SEND_IMPORTS}
            assert pair.isdisjoint(
                sys.modules
            ), f"Library modules already resident: {pair & sys.modules.keys()}"
            assert {
                "pr3339_cold_dependency_probe",
                "pr3339_cold_dependency_probe.child",
            }.isdisjoint(sys.modules)
            with _recording_send_imports() as seen, _recording_cold_send_loads() as cold:
                press(host, "enter", "\r")
                await until(gateway.validation_started.is_set)
                gateway.validation_release.set()
                await until(lambda: gateway.stream_calls == 1)

    assert set(cold) == pair, f"unexpected cold UI-thread send loads: {cold}"
    # Non-vacuity: the send searched the Library, found the note and carried it.
    assert (library.prewarms, library.searches) == (1, 1)
    assert len(requests) == 1 and _EVIDENCE in str(requests[0]), requests
    for name in LIVE_LIBRARY_FIRST_SEND_IMPORTS:
        assert f"tldw_chatbook.{name}" in seen, f"the send never imported {name}"
    lazy = _not_warmed(seen, ready, covered)
    assert set(lazy) == {
        f"tldw_chatbook.{name}" for name in LIVE_LIBRARY_FIRST_SEND_IMPORTS
    }, (
        "an Automatic send's UI-loop imports are no longer exactly the two "
        f"Library modules left lazy on 2026-10-31: {lazy}"
    )
