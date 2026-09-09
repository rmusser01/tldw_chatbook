"""App-owned workspace Buddy projection and exact-result commands."""

from __future__ import annotations

import asyncio
from typing import Any

from tldw_chatbook.Persona_Buddy.inbox import BuddyInboxEntry, project_workspace_inbox
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding


class BuddyWorkspaceCoordinator:
    """Project existing work; this object never accepts or owns an agent run."""

    def __init__(self, app: Any, binding: BuddyBinding) -> None:
        if binding.kind != "workspace":
            raise ValueError("Choose a workspace for this inbox.")
        self.app = app
        self.binding = binding

    async def snapshot(self) -> tuple[str, tuple[BuddyInboxEntry, ...]]:
        """Read membership off-loop and then project current in-memory activity."""
        registry = getattr(self.app, "workspace_registry_service", None)
        if registry is None:
            raise ValueError("Workspace storage is unavailable.")
        runtime = getattr(self.app, "console_runtime", None)
        initialize = getattr(runtime, "ensure_activity_receipt_service", None)
        receipts_service = (
            await asyncio.to_thread(initialize)
            if initialize is not None
            else getattr(runtime, "activity_receipts", None)
        )
        if receipts_service is None:
            raise ValueError(
                "Result storage is unavailable. Check the active local profile and reopen the inbox."
            )
        if runtime is not getattr(self.app, "console_runtime", None) or getattr(
            runtime, "_disposed", False
        ):
            raise ValueError("The active profile changed. Reopen the Buddy inbox.")
        state_reader = getattr(receipts_service, "hydration_state", None)
        state = await asyncio.to_thread(state_reader) if state_reader else "ready"
        hydrate = getattr(runtime, "ensure_activity_hydration", None)
        if hydrate is not None and state != "ready":
            hydrate()
        if state == "degraded":
            raise ValueError(
                "Result storage could not be read. The inbox will retry; check the active profile if this continues."
            )
        if state != "ready":
            raise ValueError(
                "Loading saved results. The inbox will refresh automatically."
            )
        receipts = await asyncio.to_thread(receipts_service.unseen_snapshot)
        wanted = {
            receipt.conversation_id for receipt in receipts if receipt.conversation_id
        }

        def read_members() -> tuple[str, dict[str, str]]:
            workspace = registry.get_workspace(self.binding.target_id)
            if (
                workspace is None
                or workspace.archived
                or str(getattr(workspace.authority, "value", workspace.authority))
                != "local-only"
            ):
                raise ValueError(
                    "The bound workspace is unavailable. Choose another in Buddy settings."
                )
            service = getattr(self.app, "local_chat_conversation_service", None)
            members = {}
            if wanted:
                for membership in registry.list_workspace_conversations(
                    self.binding.target_id
                ):
                    if membership.item_id not in wanted or service is None:
                        continue
                    row = service.get_conversation_metadata(membership.item_id)
                    if row and not row.get("deleted", False):
                        members[membership.item_id] = str(
                            row.get("title") or membership.title or "Conversation"
                        )
            return workspace.name, members

        title, members = await asyncio.to_thread(read_members)
        # Refresh live owners after I/O: selection, move or deletion may have changed.
        if runtime is not getattr(self.app, "console_runtime", None) or getattr(
            runtime, "_disposed", False
        ):
            raise ValueError("The active profile changed. Reopen the Buddy inbox.")
        store = getattr(runtime, "chat_store", None)
        controller = getattr(runtime, "chat_controller", None)
        sessions = tuple(store.sessions()) if store else ()
        scoped = [s for s in sessions if self.binding.includes(s)]
        states = (
            {s.id: controller.run_state_for(s.id) for s in scoped} if controller else {}
        )
        activities = (
            {s.id: controller.activity_for(s.id) for s in scoped} if controller else {}
        )
        # Use the frozen receipt set with matching membership; the next refresh
        # picks up later outcomes. No result is acknowledged by this read.
        return title, project_workspace_inbox(
            self.binding.target_id,
            sessions=sessions,
            receipts=receipts,
            run_states=states,
            activities=activities,
            member_titles=members,
        )

    async def _validate_entry(self, entry: BuddyInboxEntry) -> BuddyInboxEntry:
        _, rows = await self.snapshot()
        current = next((row for row in rows if row.key == entry.key), None)
        if (
            current is None
            or current.binding != entry.binding
            or current.receipt_ids != entry.receipt_ids
        ):
            raise ValueError(
                "This item is no longer in the workspace inbox. Refresh and select its current entry."
            )
        return current

    async def acknowledge(self, entry: BuddyInboxEntry) -> int:
        """Acknowledge only the exact selected outcome, never an entire conversation."""
        if entry.group != "results" or not entry.receipt_ids:
            raise ValueError(
                "Only a result can be marked seen; questions still need a response."
            )
        await self._validate_entry(entry)
        runtime = getattr(self.app, "console_runtime", None)
        receipts = getattr(runtime, "activity_receipts", None)
        if receipts is None:
            raise ValueError(
                "Result storage is unavailable; the result remains unseen."
            )
        return await asyncio.to_thread(receipts.acknowledge, entry.receipt_ids)

    def request_open_entry(self, entry: BuddyInboxEntry) -> None:
        self.app.run_worker(
            self._open_entry(entry),
            group="buddy-inbox-open",
            exclusive=True,
            exit_on_error=False,
        )

    async def _open_entry(self, entry: BuddyInboxEntry) -> None:
        try:
            await self._validate_entry(entry)
            from .buddy_conversation import open_buddy_conversation

            open_buddy_conversation(self.app, entry.binding, allow_voice=False)
        except ValueError as exc:
            self.app.notify(str(exc), severity="warning")
        except Exception:  # noqa: BLE001 - a failed inbox read must not terminate active work
            self.app.notify(
                "Could not open this workspace item. Reopen the inbox and retry.",
                severity="error",
            )


def open_buddy_workspace(app: Any, binding: BuddyBinding) -> None:
    """Open a workspace inbox without selecting a Console session or microphone."""
    from tldw_chatbook.Widgets.Persona_Widgets.buddy_workspace_modal import (
        BuddyWorkspaceModal,
    )

    from .buddy_speech import ensure_buddy_speech

    current = getattr(app, "_buddy_workspace_modal", None)
    if current is not None and current.is_mounted:
        return
    coordinator = BuddyWorkspaceCoordinator(app, binding)
    modal = BuddyWorkspaceModal(
        snapshot=coordinator.snapshot,
        open_entry=coordinator.request_open_entry,
        acknowledge=coordinator.acknowledge,
        speech=ensure_buddy_speech(app),
    )
    app._buddy_workspace_modal = modal
    app.push_screen(modal)
