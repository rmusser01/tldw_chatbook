"""Versioned archive receipts beside the canonical Library conversation reader."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from ...Utils.input_validation import validate_conversation_archive_scope


async def change_conversation_archive(
    app: Any, ids: tuple[str, ...], **kwargs: Any
) -> dict[str, Any]:
    """Use the shared lifecycle gate (import lazily to avoid UI cycles)."""
    from ...Chat.conversation_archive_actions import (
        change_conversation_archive as change,
    )

    return await change(app, ids, **kwargs)


class LibraryConversationRecovery:
    """Own archive scope and versioned receipts; transcript state stays with Reader."""

    def __init__(self, screen: Any) -> None:
        self.screen = screen
        self._owner = getattr(screen, "_screen", screen)
        self.scope = "active"
        self.receipt_copy = ""
        self.receipt_versions: dict[str, int] = {}
        self.receipt_archived = True
        self.busy = False
        self._change_task: asyncio.Task[None] | None = None

    def sync(self) -> None:
        """Paint only the mounted conversation canvas; navigation owns remounts."""
        if not getattr(self._owner, "is_current", True):
            return
        query = getattr(self.screen, "query", None)
        if callable(query) and not query("#library-conversations-canvas"):
            return
        self.screen._sync_library_conversation_canvas()

    def project(self, state: Any) -> Any:
        """Attach ephemeral reader/receipt state without changing list ownership."""
        return replace(
            state,
            archive_scope=self.scope,
            receipt_copy=self.receipt_copy,
            undo_available=bool(self.receipt_versions) and not self.busy,
            actions_disabled=state.actions_disabled or self.busy or state.loading,
        )

    def set_scope(self, scope: str) -> None:
        """Validate scope and refresh page one while preserving the search query.

        Args:
            scope: Requested local conversation lifecycle scope.
        """
        try:
            scope = validate_conversation_archive_scope(scope)
        except ValueError:
            return
        if self.busy:
            return
        self.scope = scope
        self.screen._start_library_conversation_page_request(
            1,
            self.screen._library_conversation_requested_query,
            focus_after_apply=f"#library-conversations-scope-{scope}",
        )

    async def change(
        self,
        ids: tuple[str, ...],
        *,
        archived: bool,
        expected_versions: dict[str, int],
        undo: bool = False,
    ) -> None:
        """Mutate captured identities and retain only successful versioned changes."""
        if self.busy or not ids:
            return
        self.busy = True
        self.sync()
        state = getattr(self._owner, "_conversations_state", None)
        generation = getattr(state, "request_generation", None)
        self._change_task = asyncio.create_task(
            self._complete_change(
                ids,
                archived=archived,
                expected_versions=expected_versions,
                undo=undo,
                state=state,
                request_generation=generation,
                reader_generation=(
                    state.reader_state.generation,
                    state.reader_state.loaded_generation,
                )
                if state is not None
                else None,
            )
        )
        # The controller owns completion even if the confirmation worker leaves.
        # Keep its strong reference and busy gate until storage and receipt settle.
        await asyncio.shield(self._change_task)

    async def _complete_change(
        self,
        ids: tuple[str, ...],
        *,
        archived: bool,
        expected_versions: dict[str, int],
        undo: bool,
        state: Any,
        request_generation: int | None,
        reader_generation: tuple[int, int | None] | None,
    ) -> None:
        try:
            result = await change_conversation_archive(
                self.screen.app_instance,
                ids,
                archived=archived,
                expected_versions=expected_versions,
            )
            changed = dict(result.get("changed", {}))
            failures = dict(result.get("failures", {}))
            if state is not None and state is self._owner._conversations_state:
                reader = state.reader_state
                updates = {}
                for kind in ("loaded", "selected"):
                    identity = getattr(reader, f"{kind}_id")
                    if (
                        (reader.generation, reader.loaded_generation)
                        == reader_generation
                        and identity in changed
                        and getattr(reader, f"{kind}_version")
                        == expected_versions.get(identity)
                    ):
                        metadata_name = f"reader_{kind}_metadata"
                        setattr(
                            state,
                            metadata_name,
                            {
                                **getattr(state, metadata_name),
                                "archived": archived,
                                "version": changed[identity],
                            },
                        )
                        updates[f"{kind}_version"] = changed[identity]
                if updates:
                    state.reader_state = replace(reader, **updates)
                # Canvas sync reconciles selection against retained page rows.
                # Advance those same versions before it can restart a reader
                # load from the pre-commit lifecycle metadata.
                state.page_records = tuple(
                    {**row, "archived": archived, "version": changed[identity]}
                    if (identity := str(row.get("id") or row.get("conversation_id")))
                    in changed
                    and row.get("version") == expected_versions.get(identity)
                    else row
                    for row in state.page_records
                )
            if undo:
                self.receipt_versions = {
                    key: value
                    for key, value in self.receipt_versions.items()
                    if key not in changed
                }
            else:
                self.receipt_versions = changed
                self.receipt_archived = archived
            verb = "Archived" if archived else "Restored"
            self.receipt_copy = f"{verb} {len(changed)} conversation(s)."
            if failures:
                from ...Chat.conversation_archive_actions import archive_failure_copy

                records = getattr(self.screen, "_conversation_records", lambda: ())()
                titles = {
                    str(row.get("id") or row.get("conversation_id")): str(
                        row.get("title") or "Untitled"
                    )
                    for row in records
                }
                self.receipt_copy += " Unchanged: " + "; ".join(
                    f"{titles.get(key, key)}: {archive_failure_copy(reason)}"
                    for key, reason in failures.items()
                )
            if not archived:
                self.receipt_copy += " Current Console context is unchanged."
        except Exception:  # noqa: BLE001 - retain reader state at the UI boundary
            if not undo:
                self.receipt_versions = {}
                self.receipt_archived = False
            self.receipt_copy = (
                "Undo did not complete. Retry Undo or refresh the list."
                if undo
                else "Could not update conversations. Refresh the list and retry."
            )
        finally:
            self.busy = False
            self.sync()
        query = getattr(self.screen, "query", None)
        if (
            getattr(self._owner, "is_current", True)
            and (not callable(query) or query("#library-conversations-canvas"))
            and state is getattr(self._owner, "_conversations_state", None)
            and request_generation == getattr(state, "request_generation", None)
        ):
            self.screen._start_library_conversation_page_request(
                self.screen._library_conversation_requested_page,
                self.screen._library_conversation_requested_query,
            )

    async def undo(self) -> None:
        await self.change(
            tuple(self.receipt_versions),
            archived=not self.receipt_archived,
            expected_versions=dict(self.receipt_versions),
            undo=True,
        )

    async def annotate(
        self, records: tuple[Mapping[str, Any], ...]
    ) -> tuple[Mapping[str, Any], ...]:
        """Read workspace labels away from the UI thread, including retired groups.

        Args:
            records: Validated conversation metadata mappings from the current page.

        Returns:
            Metadata mappings enriched with workspace display and archive state.
        """
        registry = getattr(self.screen.app_instance, "workspace_registry_service", None)
        if registry is None:
            return records

        def enrich() -> tuple[Mapping[str, Any], ...]:
            output: list[Mapping[str, Any]] = []
            workspaces = {}
            for source in records:
                row = dict(source)
                workspace_id = row.get("workspace_id")
                if workspace_id:
                    if workspace_id not in workspaces:
                        workspaces[workspace_id] = registry.get_workspace(workspace_id)
                    record = workspaces[workspace_id]
                    if record is not None:
                        row["workspace_name"] = record.name
                        row["workspace_archived"] = bool(record.archived)
                else:
                    row["workspace_name"] = "Default"
                output.append(row)
            return tuple(output)

        if getattr(getattr(registry, "db", None), "is_memory_db", False):
            return enrich()
        return await asyncio.to_thread(enrich)
