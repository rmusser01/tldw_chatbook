"""Get started card offers: a detected local server and env-found keys.

TASK-34100.2 AC#12 (E2): when exported keys skipped setup, the Console's
"Get started" card names each provider whose key was found ("Found
ANTHROPIC_API_KEY · Use Anthropic"). Choosing one makes it the chat default
with ``credential_source = environment`` in one atomic provider-setup commit
(the same owner Settings and setup use), then re-derives the card from the
shared readiness verdict. Kept out of ``chat_screen.py`` (size ratchet).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from typing import TYPE_CHECKING

from loguru import logger

from tldw_chatbook.Chat.console_onboarding_state import (
    CONSOLE_SETUP_ENV_KEY_WORKBENCH_ACTION_PREFIX,
    ConsoleSetupCardState,
    build_console_detected_server_action,
    build_console_env_key_actions,
    build_env_key_provider_mutation,
)

if TYPE_CHECKING:
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal


def sync_setup_card_offers(
    screen: ChatScreen, modal: ConsoleSetupModal, card_state: ConsoleSetupCardState
) -> None:
    """Offer (or withdraw) the card's detected-server and env-key actions."""
    modal.sync_detected_server_action(
        build_console_detected_server_action(
            screen._console_detected_local_server, card_mode=card_state.mode
        )
    )
    actions: tuple = ()
    if card_state.mode == "card":
        # The in-memory snapshot, never a fresh disk read: rendering the card
        # needs no new config admission (a backup may have paused it).
        app_config = getattr(screen.app_instance, "app_config", None) or {}
        try:
            actions = build_console_env_key_actions(
                app_config, os.environ, card_mode=card_state.mode
            )
        except Exception as exc:
            logger.debug("Env-key card offers skipped ({})", type(exc).__name__)
    modal.sync_env_key_actions(actions)


def is_env_key_offer(action_id: str) -> bool:
    """Whether a workbench action id is a card env-key offer."""
    return action_id.startswith(CONSOLE_SETUP_ENV_KEY_WORKBENCH_ACTION_PREFIX)


async def apply_env_key_offer(screen: ChatScreen, action_id: str) -> bool:
    """Commit the chosen env-keyed provider and move this chat onto it.

    Args:
        screen: The Console screen whose card offered the action.
        action_id: ``use-env-key:<provider>`` from the card button.

    Returns:
        Whether the provider became the chat default.
    """
    from tldw_chatbook.Chat.console_session_settings import (
        build_default_console_session_settings,
    )
    from tldw_chatbook.Chat.provider_setup_persistence import (
        mirror_provider_setup_mutation,
        persist_provider_setup,
    )
    from tldw_chatbook.Widgets.Console.console_setup_modal import ConsoleSetupModal

    try:
        modal = screen.query_one("#console-setup-modal", ConsoleSetupModal)
    except Exception:
        return False
    action = next(
        (item for item in modal.env_key_actions if item.workbench_action == action_id),
        None,
    )
    if action is None:
        return False
    app_config = screen._provider_readiness_app_config()
    try:
        mutation = build_env_key_provider_mutation(action, app_config)
        result = await asyncio.to_thread(persist_provider_setup, mutation)
    except Exception as exc:
        logger.error(
            "Env-key provider commit failed (provider={}, error_type={})",
            action.provider_key,
            type(exc).__name__,
        )
        result = None
    if result is None or not result.fully_applied:
        screen.app_instance.notify(
            f"Couldn't make {action.label.removeprefix('Use ')} the chat default; "
            "nothing was changed. Try Set up provider.",
            title="Setup not saved",
            severity="error",
            markup=False,  # a provider's display name is text, never markup
        )
        return False
    # Review F2: every reader of the app's config snapshot (Library's
    # analyze-after-import, Reader Generate) sees the new chat default now.
    mirror_provider_setup_mutation(
        getattr(screen.app_instance, "app_config", None), mutation
    )
    settings = build_default_console_session_settings(
        screen._provider_readiness_app_config(), action.provider_key, action.model
    )
    screen._session._replace_active_console_session_settings(
        replace(settings, source="user")
    )
    screen._sync_console_transcript_guidance()
    screen.run_worker(
        screen._sync_native_console_chat_ui(), exclusive=True, group="console-sync"
    )
    dismiss_env_key_notice(screen.app_instance)
    return True


def dismiss_env_key_notice(app: object) -> None:
    """Remove the env-key first-launch toast once a card offer commits.

    Review round 2 (F10): after 'Use Anthropic' the 20 s toast still said
    'new chats use OpenAI, which has no key'. Only that toast is removed
    (``TldwCli._show_pending_env_key_notice`` remembers its message); other
    toasts stay. Textual has no public way to remove one toast, so this reads
    the app's notification collection and is a no-op if that ever changes.

    Args:
        app: The running app (``TldwCli``).
    """
    message = getattr(app, "_env_key_notice_message", None)
    notifications = getattr(app, "_notifications", None)
    unnotify = getattr(app, "_unnotify", None)
    if not message or notifications is None or not callable(unnotify):
        return
    try:
        for notification in [n for n in notifications if n.message == message]:
            unnotify(notification)
    except Exception:  # noqa: BLE001 - a stale toast never fails the commit
        logger.debug("env_key_notice_dismiss_failed")
