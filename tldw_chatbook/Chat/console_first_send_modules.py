"""Modules a Console's first send imports lazily, warmed at idle (TASK-33620.15.5).

Measured live on origin/dev 4de437a5c5 (Anthropic haiku, 160x45, a probe
recording every module imported on the UI thread): between Enter and the
provider call, a Console's first send after launch imported 54 modules on the
UI loop, inside the send's own callbacks: hook preparation and the hooks v2
package, the response-rules package, the visual transcript with PIL, the
agent's tool providers (one defines pydantic models at import) and the
run-start catalog. They are lazy on purpose (each is kept off the ADR-097
UI-ready census), so the first send paid them while the user waited, in
run-start callbacks of 40-118 ms.

The app's post-UI-ready pre-import pass (``TldwCli._preimport_heavy_screens``,
a daemon thread behind the same on/off switch, off in the census) imports them
first, before the speculative screen routes, then parses the built-in MCP
manifest's source once (every send's configuration capture reads that
manifest). Nothing moves before UI-ready, and a send racing the warm-up blocks
on Python's per-module import lock no longer than its own import would have
taken.

Keep the list to modules the send path really imports, in the order it does;
``Tests/UI/test_console_first_send_preimport.py`` checks that each is still
lazy at UI-ready and that the harness's send imports nothing lazy beyond it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

from loguru import logger

#: In first-send order. Before acceptance: the acknowledgement, the received
#: intent and admission. After it, run start: hook preparation, compaction
#: preflight, the durable commit, response rules, the agent's tool providers
#: and the run-start catalog; then the stream.
FIRST_SEND_MODULES: tuple[str, ...] = (
    "tldw_chatbook.UI.Console_Modules.send_acknowledgement",
    "tldw_chatbook.UI.Console_Modules.sent_draft",
    "tldw_chatbook.Chat.console_received_intent",
    "tldw_chatbook.Chat.console_configuration_preparation",
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
    # Reached by the send too; a live app has usually imported them already
    # (the sync tick and the scheduler), so here they are dictionary hits.
    "tldw_chatbook.UI.Console_Modules.config_sync",
    "tldw_chatbook.emergency_stop",
)


def preimport_first_send_modules(should_stop: Callable[[], bool]) -> None:
    """Import ``FIRST_SEND_MODULES`` in order, then parse the MCP manifest source.

    Runs on the pre-import daemon thread and never raises. A module that
    fails to import is left to fail again, identically, where the send
    imports it (a failed import is never cached).

    Args:
        should_stop: Polled before each step; True stops the warm-up
            (the app is quitting).
    """
    for name in FIRST_SEND_MODULES:
        if should_stop():
            return
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 -- a speculative warm-up
            logger.debug(
                "First-send pre-import failed (module={}, error_type={})",
                name,
                type(exc).__name__,
            )
    if should_stop():
        return
    try:
        from tldw_chatbook.MCP.server import prime_local_manifest_source

        prime_local_manifest_source()
    except Exception as exc:  # noqa: BLE001 -- a speculative warm-up
        logger.debug(
            "First-send manifest warm-up failed (error_type={})", type(exc).__name__
        )
