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

    def _screen_preimport_pacing(self) -> tuple[float, float]:
        return (2.0, 0.5)

    def _pause_between_preimports(self, gap_seconds: float) -> None:
        self.pauses.append(gap_seconds)
        self.events.append(("pause", gap_seconds))
        if self._stop_after is not None and len(self.pauses) >= self._stop_after:
            self._shutting_down = True


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
    warm.preimport_first_send_modules(_PassApp(stop_after=3))

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

    warm.preimport_first_send_modules(app)

    # Each import "took" 0.1 s on the fake clock: gap 0.1 x 2.0 = 0.2 s,
    # before every later import and before the manifest parse.
    assert app.pauses == [0.0] + [pytest.approx(0.2)] * len(warm.FIRST_SEND_MODULES)
    assert app.events[:2] == [("pause", 0.0), ("import", warm.FIRST_SEND_MODULES[0])]
    assert app.events[-2:] == [("prime", None), ("freeze", "first_send_preimport")]


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
                press(host, "enter", "\r")
                await until(gateway.validation_started.is_set)
                gateway.validation_release.set()
                await until(lambda: gateway.stream_calls == 1)

    lazy = _not_warmed(seen, ready, covered)
    assert not lazy, f"imported on the UI loop by the send, not warmed: {lazy}"
    assert "tldw_chatbook.Agents.hooks_v2.lifecycle" in seen, "the send never ran hooks"
    # The composer's ``from ...Chat import console_voice_input`` (its
    # dictation-state sync) is a fromlist submodule import every harness send
    # makes; seeing it keeps that path of the guard non-vacuous.
    assert "tldw_chatbook.Chat.console_voice_input" in seen, (
        "the guard no longer records `from package import submodule`"
    )


_EVIDENCE = "The Fresnel lens lets a lighthouse beam reach ships far away."


class _EvidenceLibrary:
    """A started Library whose search finds one note, as the live BM25 run did."""

    def __init__(self) -> None:
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
                    "source_id": "note-1",
                    "chunk_id": "chunk-1",
                    "title": "Lighthouse keeping notes",
                    "content": _EVIDENCE,
                    "score": 0.9,
                    "source_type": "notes",
                }
            ],
        }


@pytest.mark.asyncio
async def test_an_automatic_library_send_loads_only_the_two_library_modules_lazily(
    pass_outcome,
):
    """TASK-33620.15.19: an Automatic send adds exactly the measured pair.

    Same guard as the harness send above, on a send whose conversation has
    Auto set to Automatic and whose Library search returns evidence: Library
    preparation, the bounded search and the evidence bundle run under the
    send's frames. The controller's function-local imports of
    ``console_library_preparation`` and ``console_library_search`` are
    neither warm at UI-ready nor listed, so this send loads them on the UI
    loop, as the live first send did (0.5-3.6 ms wall). They stay lazy by the
    owner's 2026-10-31 decision, so they are the only lazy imports allowed
    here. Red if the path picks up any other unwarmed import, and red if the
    pair stops being lazy, so the record above is revisited either way.
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

    ready = _warm_at_ui_ready(pass_outcome)
    covered = _covered_by_the_warm_up()
    library = _EvidenceLibrary()

    host, gateway, _timeline = build()
    host.app_instance.library_rag_search_service = library
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
            with _recording_send_imports() as seen:
                press(host, "enter", "\r")
                await until(gateway.validation_started.is_set)
                gateway.validation_release.set()
                await until(lambda: gateway.stream_calls == 1)

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
