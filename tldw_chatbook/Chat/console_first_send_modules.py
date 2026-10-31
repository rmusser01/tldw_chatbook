"""Modules a Console's first send imports lazily, warmed at idle (TASK-33620.15.5).

Measured live on origin/dev 4de437a5c5 (Anthropic haiku, 160x45, a probe
recording every module imported on the UI thread): between Enter and the
provider call, a Console's first send after launch imported 52 modules and
three packages on the UI loop, inside the send's own callbacks (56 imports in
all, one by the dictionary-summary sync tick): hook preparation and the hooks
v2 package, the response-rules package, the visual transcript with PIL, the
agent's tool providers (one defines pydantic models at import) and the
run-start catalog. They are lazy on purpose (each is kept off the ADR-097
UI-ready census), so the first send paid them while the user waited, in
run-start callbacks of 40-118 ms. After the provider call it imported nine
more, the canvas settlement's in one callback of 157 ms CPU.

The app's post-UI-ready pre-import pass (``TldwCli._preimport_heavy_screens``,
a daemon thread behind the same on/off switch, off in the census) imports them
after the speculative screen routes, paced like them, then parses the built-in
MCP manifest's source once (every send's configuration capture reads that
manifest) and freezes the heap again (ADR-198). Nothing moves before UI-ready.
TASK-33620.15.14: run first, the warm-up pushed every screen import back by
its own time, so a Library click one second after the Console's first frame
found Library cold in 8 of 8 launches (0 of 8 on origin/dev) and imported
about 200 modules on the UI loop; after the routes, a first navigation sees
origin/dev's timing.

The thread parks before each import while accepted Console work owns the
loop, so a send races the warm-up only when it arrives mid-import. A send that
needs the module being imported blocks on Python's per-module import lock no
longer than its own import would have taken; one that needs a different module
shares the GIL with that import instead.
``Tests/UI/test_console_first_send_import_cycles.py`` keeps the list free of
module-level import cycles, which two threads could otherwise deadlock on.

Keep the list to modules the send path really imports, in the order it does;
``Tests/UI/test_console_first_send_preimport.py`` checks that each is still
lazy at UI-ready and that the harness's send imports nothing lazy beyond it.
"""

from __future__ import annotations

import importlib
import time
from typing import Any

from loguru import logger

from tldw_chatbook.Utils.ui_responsiveness import freeze_long_lived_heap

#: In first-send order. Before acceptance: the acknowledgement, the received
#: intent and admission. After it, run start: hook preparation, compaction
#: preflight, the durable commit, response rules, the agent's tool providers
#: and the run-start catalog; then the stream and the run's end.
FIRST_SEND_MODULES: tuple[str, ...] = (
    "tldw_chatbook.UI.Console_Modules.send_acknowledgement",
    "tldw_chatbook.UI.Console_Modules.sent_draft",
    "tldw_chatbook.Chat.console_received_intent",
    "tldw_chatbook.Chat.console_configuration_preparation",
    "tldw_chatbook.Chat.console_preparation_reads",
    "tldw_chatbook.Chat.console_configuration_capture",
    "tldw_chatbook.UI.Console_Modules.turn_admission",
    "tldw_chatbook.MCP.console_snapshot",
    "tldw_chatbook.Chat.console_received_dispatch",
    "tldw_chatbook.Chat.console_received_turn",
    "tldw_chatbook.Chat.console_send_diagnostics",
    "tldw_chatbook.Chat.console_automatic_submission",
    "tldw_chatbook.Chat.console_hook_preparation",
    "tldw_chatbook.Agents.run_hooks",
    "tldw_chatbook.Agents.automatic_work_runtime",
    "tldw_chatbook.Chat.console_references",
    "tldw_chatbook.Agents.hooks_v2.lifecycle",
    "tldw_chatbook.Agents.hooks_v2.matching",
    "tldw_chatbook.Agents.hooks_v2.models",
    "tldw_chatbook.Agents.hooks_v2.validation",
    "tldw_chatbook.Agents.hooks_v2.checkpoints",
    "tldw_chatbook.Agents.hooks_v2.engine",
    "tldw_chatbook.Agents.hooks_v2.budgets",
    "tldw_chatbook.Agents.hooks_v2.command_executor",
    "tldw_chatbook.Agents.hooks_v2.ownership",
    "tldw_chatbook.Agents.hooks_v2.context",
    "tldw_chatbook.Chat.console_legacy_flat_roots",
    "tldw_chatbook.Chat.console_context_window",
    "tldw_chatbook.Chat.console_visual_transcript",
    "tldw_chatbook.Chat.console_predispatch_block",
    "tldw_chatbook.Chat.console_native_commit",
    "tldw_chatbook.Chat.console_continuation_binding",
    "tldw_chatbook.Chat.console_trace_row_sources",
    "tldw_chatbook.Chat.response_rules.generation",
    "tldw_chatbook.Chat.response_rules.builder",
    "tldw_chatbook.Chat.response_rules.evaluator",
    "tldw_chatbook.Chat.response_rules.models",
    "tldw_chatbook.Chat.response_rules.repository",
    "tldw_chatbook.Chat.response_rules.resources",
    "tldw_chatbook.Chat.response_rules.runtime",
    "tldw_chatbook.Chat.response_rules.corrections",
    "tldw_chatbook.Chat.response_rules.evidence",
    "tldw_chatbook.Chat.response_rules.store",
    "tldw_chatbook.Chat.response_rules.resolution",
    "tldw_chatbook.Chat.response_rules.chat_state",
    "tldw_chatbook.UI.Console_Modules.response_rules",
    "tldw_chatbook.app_served",
    "tldw_chatbook.Web_Server.served_relay_publisher",
    "tldw_chatbook.Guardian.settings",
    "tldw_chatbook.Agents.local_tool_provider",
    "tldw_chatbook.Chat.console_run_start_tools",
    "tldw_chatbook.MCP.console_tool_preparation",
    "tldw_chatbook.Character_Chat.world_info_resolver",
    "tldw_chatbook.Agents.persona_policy",
    "tldw_chatbook.Tools.character_tool_service",
    "tldw_chatbook.Tools.git_tool_impls",
    "tldw_chatbook.Tools.web_tool_impls",
    "tldw_chatbook.Web_Scraping.deep_search_citations",
    "tldw_chatbook.Agents.ask_user_questions",
    "tldw_chatbook.Agents.fs_read_ledger",
    "tldw_chatbook.Chat.conversation_archive_actions",
    "tldw_chatbook.Agents.canvas_tool_provider",
    "tldw_chatbook.Canvas.guide",
    "tldw_chatbook.Chat.console_stream_persistence",
    "tldw_chatbook.Chat.provider_rate_limits",
    # The run's end: canvas settlement (one UI-loop callback of 157 ms CPU on
    # a loaded host), remote sessions and citation counts.
    "tldw_chatbook.Canvas.native_authority",
    "tldw_chatbook.Canvas.compiler",
    "tldw_chatbook.Canvas.gateway",
    "tldw_chatbook.Canvas.capabilities",
    "tldw_chatbook.Canvas.control_protocol",
    "tldw_chatbook.Tools.remote_session_registry",
    "tldw_chatbook.Tools.remote_session_worker",
    "tldw_chatbook.UI.Console_Modules.citation_counts",
    # Reached by the send too; a live app has usually imported them already
    # (the sync tick, the scheduler, the composer's dictation-state sync), so
    # here they are dictionary hits.
    "tldw_chatbook.UI.Console_Modules.config_sync",
    "tldw_chatbook.emergency_stop",
    "tldw_chatbook.Chat.console_voice_input",
)


def preimport_first_send_modules(app: Any) -> None:
    """Import ``FIRST_SEND_MODULES`` in order, then parse the MCP manifest source.

    Runs on the pre-import daemon thread after the screen routes, and never
    raises: a fault anywhere in it, the app's own pacing and park included,
    ends the warm-up quietly. Paced like the routes: before each import the
    thread parks while a navigation or accepted Console work owns the loop
    (so a send that starts first imports what it needs itself, as before),
    then gives the UI loop a pause as long as the previous import took (the
    app's pre-import pacing). A module that fails to import is left to fail
    again, identically, where the send imports it (a failed import is never
    cached). When it finishes it freezes the heap, as the routes' pass did
    before it (ADR-198: these modules live as long as the app).

    Args:
        app: The running ``TldwCli``. Its ``_shutting_down`` stops the
            warm-up; ``_screen_preimport_pacing()`` and
            ``_pause_between_preimports()`` pace it.
    """
    try:
        _warm_first_send(app)
    except Exception as exc:  # noqa: BLE001 -- a speculative warm-up
        logger.debug("First-send warm-up stopped (error_type={})", type(exc).__name__)


def _warm_first_send(app: Any) -> None:
    ratio, cap = app._screen_preimport_pacing()
    previous = 0.0
    for name in FIRST_SEND_MODULES:
        app._pause_between_preimports(min(previous * ratio, cap))
        if getattr(app, "_shutting_down", False):
            return
        started = time.monotonic()
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- a speculative warm-up
            logger.debug(
                "First-send pre-import failed (module={}, error_type={})",
                name,
                type(exc).__name__,
            )
        previous = time.monotonic() - started
    app._pause_between_preimports(min(previous * ratio, cap))
    if getattr(app, "_shutting_down", False):
        return
    try:
        from tldw_chatbook.MCP.server import prime_local_manifest_source

        prime_local_manifest_source()
    except Exception as exc:  # noqa: BLE001 -- a speculative warm-up
        logger.debug(
            "First-send manifest warm-up failed (error_type={})", type(exc).__name__
        )
    freeze_long_lived_heap("first_send_preimport")
