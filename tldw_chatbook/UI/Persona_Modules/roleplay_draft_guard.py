"""The Roleplay aggregate draft guard, shared by leaving and quitting.

Leaving the Roleplay screen with unsaved drafts asks one aggregate question --
Save and continue / Discard and continue / Stay -- and a save that fails offers
Retry / Stay instead of dropping the drafts. Quitting the app leaves the screen
too, so ``PersonasScreen.confirm_quit`` asks the same question through the same
flow (TASK-33622.14); before it existed, Ctrl+Q quit straight past the drafts.

The two callers differ only in how a prompt is awaited, so the flow is handed
an ``ask`` and never pushes a screen itself. App navigation passes
``push_screen_wait``. The quit flow passes ``await_quit_prompt``, its one
choke point (TASK-33622.10): Ctrl+Q is a priority binding, so a quit prompt can
sit above a modal whose own ``dismiss()`` pops it unanswered, and a bare wait
on that prompt would hang the quit worker for the rest of the session. A
vanished prompt answers ``None``, which is Stay.

This lives outside ``personas_screen.py``, which is under the module size
ratchet, and both hooks import it lazily, so it stays off the boot path.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger
from textual.screen import ModalScreen

from ...Widgets.confirmation_dialog import await_quit_prompt
from ...Widgets.Persona_Widgets.persona_profile_editor_widget import (
    PersonaProfileEditorWidget,
)
from ...Widgets.Persona_Widgets.personas_character_editor_widget import (
    PersonasCharacterEditorWidget,
)
from ..Navigation.character_conversation_navigation import (
    RoleplayDraftNavigationDialog,
    RoleplayDraftRecoveryDialog,
    RoleplayDraftSnapshot,
)
from .roleplay_frame_state import roleplay_has_unsaved_work

if TYPE_CHECKING:
    from ..Screens.personas_screen import PersonasScreen

logger = logger.bind(module="PersonasScreen")

#: Awaits one draft prompt and returns its answer: ``"save"``, ``"discard"``
#: or ``"retry"``, or ``None`` for Stay.
AskPrompt = Callable[[ModalScreen[str | None]], Awaitable[str | None]]


async def confirm_roleplay_quit(screen: PersonasScreen) -> bool:
    """Ask the Roleplay draft question for the app's quit flow.

    Must run inside a worker; the app's quit flow is one. The flow first
    waits for in-flight Roleplay work (a save, or a reaction generation), so
    it never asks about a draft that is still changing. That wait can be long
    and Ctrl+Q cannot be pressed again meanwhile, so it is announced rather
    than silent.

    Args:
        screen: The Roleplay screen holding the drafts.

    Returns:
        True when the quit may proceed (no drafts, or they were saved or
        discarded); False to stay, including when a prompt vanished.
    """

    waiting_for = screen._aggregate_roleplay_draft_snapshot().inflight_save_domains
    if waiting_for:
        screen.app.notify(
            "Waiting for Roleplay work to finish before quitting: "
            f"{', '.join(waiting_for)}."
        )

    async def ask(prompt: ModalScreen[str | None]) -> str | None:
        return await await_quit_prompt(screen.app, prompt, no_answer=None)

    return await confirm_roleplay_drafts(screen, ask)


async def confirm_roleplay_drafts(screen: PersonasScreen, ask: AskPrompt) -> bool:
    """Run the aggregate Save / Discard / Stay veto over every Roleplay draft.

    Args:
        screen: The Roleplay screen holding the drafts.
        ask: Awaits one prompt and returns its answer (``None`` is Stay).

    Returns:
        True when the drafts are clean, saved or discarded; False to stay.
    """
    snapshot = screen._aggregate_roleplay_draft_snapshot()
    if snapshot.inflight_save_domains:
        await _await_roleplay_save_owners(screen)
        snapshot = screen._aggregate_roleplay_draft_snapshot()
    if not roleplay_has_unsaved_work(snapshot):
        return True
    domains = tuple(
        dict.fromkeys(
            _aggregate_roleplay_dirty_domains(screen, snapshot)
            + snapshot.inflight_save_domains
        )
    )
    try:
        choice = await ask(RoleplayDraftNavigationDialog(domains))
    except Exception:  # noqa: BLE001 - broken presentation fails closed
        logger.opt(exception=True).warning(
            "Could not present aggregate Roleplay navigation guard"
        )
        return False
    if choice == "save":
        while True:
            failures = await _save_aggregate_roleplay_drafts(screen, snapshot)
            if not failures and not roleplay_has_unsaved_work(
                screen._aggregate_roleplay_draft_snapshot()
            ):
                return True
            retry = await ask(RoleplayDraftRecoveryDialog(failures))
            if retry != "retry":
                return False
            snapshot = screen._aggregate_roleplay_draft_snapshot()
    if choice == "discard":
        await screen._drain_visual_identity_authoring()
        await screen._drain_persona_shared_visual_identity_authoring()
        await screen._discard_persona_visual_authoring_async()
        await screen._drain_actor_pack_creation()
        for editor in (
            *screen.query(PersonasCharacterEditorWidget),
            *screen.query(PersonaProfileEditorWidget),
        ):
            editor.discard_unsaved_form()
        screen.state.has_unsaved_changes = False
        screen._set_active_row_unsaved(False)
        return not roleplay_has_unsaved_work(
            screen._aggregate_roleplay_draft_snapshot()
        )
    return False


async def _save_aggregate_roleplay_drafts(
    screen: PersonasScreen, snapshot: RoleplayDraftSnapshot
) -> tuple[str, ...]:
    """Save each incumbent owner and return exact domains still failing."""

    failures: list[str] = []
    form_domain = (
        "Persona form" if screen.state.active_mode == "personas" else "character form"
    )
    character_visual = screen._visual_identity_authoring
    if snapshot.character_visual_dirty and character_visual is not None:
        try:
            await screen._save_visual_identity_pack(character_visual.authoritative_pack)
        except Exception:  # noqa: BLE001 - name every failed domain
            failures.append("character visuals")
    if snapshot.persona_visual_dirty:
        if screen._persona_shared_visual_identity_authoring is not None:
            try:
                await screen._save_persona_shared_visual_identity_pack()
            except Exception:  # noqa: BLE001
                failures.append("Persona visuals")
        persona_visual = screen._persona_visual_authoring
        if persona_visual is not None and persona_visual.dirty:
            try:
                await screen._save_persona_visual_pack()
            except Exception:  # noqa: BLE001
                if "Persona visuals" not in failures:
                    failures.append("Persona visuals")
    if snapshot.form_dirty:
        try:
            prior_owners = (
                screen._character_save_worker_handle,
                screen._actor_pack_save_worker_handle,
                screen._profile_save_completion,
            )
            screen.action_personas_save()
            # The editor's button posts its save request on the next loop
            # turn. Join only the exact owner it creates: Resume is a
            # worker in the same manager and must never wait on itself.
            for _ in range(20):
                await asyncio.sleep(0)
                current_owners = (
                    screen._character_save_worker_handle,
                    screen._actor_pack_save_worker_handle,
                    screen._profile_save_completion,
                )
                if (
                    current_owners != prior_owners
                    or not screen.state.has_unsaved_changes
                ):
                    break
            await _await_roleplay_save_owners(screen)
        except Exception:  # noqa: BLE001
            failures.append(form_domain)
        if screen.state.has_unsaved_changes and form_domain not in failures:
            failures.append(form_domain)
    residual = _aggregate_roleplay_dirty_domains(
        screen, screen._aggregate_roleplay_draft_snapshot()
    )
    failures.extend(domain for domain in residual if domain not in failures)
    return tuple(failures)


async def _await_roleplay_save_owners(screen: PersonasScreen) -> None:
    """Join only incumbent save owners, never the navigation worker."""

    waits: list[Awaitable[Any]] = []
    for worker in (
        screen._character_save_worker_handle,
        screen._actor_pack_save_worker_handle,
    ):
        if worker is not None:
            waits.append(worker.wait())
    completion = screen._profile_save_completion
    if completion is not None and not completion.done():
        waits.append(asyncio.shield(completion))
    current_task = asyncio.current_task()
    for task in (
        screen._visual_identity_operation_task,
        screen._persona_shared_visual_identity_operation_task,
        screen._persona_visual_operation_task,
    ):
        if task is not None and task is not current_task and not task.done():
            waits.append(task)
    if waits:
        await asyncio.gather(*waits, return_exceptions=True)


def _aggregate_roleplay_dirty_domains(
    screen: PersonasScreen, snapshot: RoleplayDraftSnapshot
) -> tuple[str, ...]:
    """Name the mounted form owner while preserving stable visual labels."""

    domains = list(snapshot.dirty_domains)
    if screen.state.active_mode == "personas" and "character form" in domains:
        domains[domains.index("character form")] = "Persona form"
    return tuple(domains)
