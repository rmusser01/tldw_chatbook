"""Console conversation and workspace row actions, persisted targets, and Markdown exports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from loguru import logger


logger = logger.bind(module="ChatScreen")


def _active_lineage_rows(db, conversation_id: str, rows: list[dict]) -> list[dict]:
    """Filter fetched rows to the conversation's active branch.

    PR #2262 review: a conversation can hold off-path branches (regenerate
    siblings) and unselected variants; the transcript the user sees is the
    parent-chain from ``active_leaf_message_id``. Falls back to every row
    for legacy conversations with no leaf pointer.
    """
    by_id = {str(row.get("id")): row for row in rows}
    record = None
    try:
        record = db.get_conversation_by_id(conversation_id)
    except Exception:
        record = None
    leaf = str((record or {}).get("active_leaf_message_id") or "") or None
    if leaf is None or leaf not in by_id:
        return rows
    lineage: list[dict] = []
    seen: set[str] = set()
    current: str | None = leaf
    while current is not None and current in by_id and current not in seen:
        seen.add(current)
        lineage.append(by_id[current])
        parent = by_id[current].get("parent_message_id")
        current = str(parent) if parent else None
    lineage.reverse()
    return lineage if len(lineage) == len(seen) else lineage


class ConsoleRowActionsController:
    """Own console conversation and workspace row actions, persisted targets, and markdown exports.

    App identity is stable for this controller lifetime. All other dependencies
    are explicit callables resolved by wiring at use time. No DOM is owned here.
    """

    def __init__(
        self,
        *,
        app_instance_accessor: Callable[[], Any],
        _activate_workspace: Callable[..., Any],
        _archive_workspace: Callable[..., Any],
        _create_session: Callable[..., Any],
        _delete_conversation: Callable[..., Any],
        _ensure_console_chat_store: Callable[..., Any],
        _notify: Callable[..., Any],
        _open_workspace_scope: Callable[..., Any],
        _rename_conversation: Callable[..., Any],
        _rename_workspace: Callable[..., Any],
        _request_workspace_files: Callable[..., Any],
        _save_markdown: Callable[..., Any],
        _set_conversation_state: Callable[..., Any],
        _toggle_star: Callable[..., Any],
        push_screen: Callable[..., Any],
        run_worker: Callable[..., Any],
        _files_availability_by_id_accessor: Callable[[], Any],
    ) -> None:
        self._app_instance_accessor = app_instance_accessor
        self._activate_workspace = _activate_workspace
        self._archive_workspace = _archive_workspace
        self._create_session = _create_session
        self._delete_conversation = _delete_conversation
        self._ensure_console_chat_store = _ensure_console_chat_store
        self._notify = _notify
        self._open_workspace_scope = _open_workspace_scope
        self._rename_conversation = _rename_conversation
        self._rename_workspace = _rename_workspace
        self._request_workspace_files = _request_workspace_files
        self._save_markdown = _save_markdown
        self._set_conversation_state = _set_conversation_state
        self._toggle_star = _toggle_star
        self.push_screen = push_screen
        self.run_worker = run_worker
        self._files_availability_by_id_accessor = _files_availability_by_id_accessor

    @property
    def _files_availability_by_id(self) -> Any:
        return self._files_availability_by_id_accessor()

    @property
    def app_instance(self) -> Any:
        return self._app_instance_accessor()

    def _console_conversation_state(self, conversation_id: str | None) -> str:
        """Return one conversation's PERSISTED state, or the default.

        Qodo review, PR #2233. The row's ``status`` field was used here, and
        it is display copy, not a database state: rows reach the browser
        carrying ``active``, ``open``, ``workspace`` or ``workspace-thread``
        as well as real states, and first-wins deduplication keeps those
        ahead of the canonical persisted row. Every non-canonical value
        normalises to ``in-progress``, so a *resolved* conversation shown as
        an "active session" row offered **Archive** instead of Unarchive and
        marked the wrong current status.

        The menu asks the database instead. This is one primary-key lookup
        taken when the menu opens, not on every rail render.

        Args:
            conversation_id: Persisted conversation id, or None for a chat
                that has never been saved.

        Returns:
            A canonical state, or the default when it cannot be resolved --
            unsaved chats have no state, and the menu gates their
            state-changing entries anyway.
        """
        from tldw_chatbook.Chat.console_conversation_actions import (
            DEFAULT_CONVERSATION_STATE,
        )

        if not conversation_id:
            return DEFAULT_CONVERSATION_STATE
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return DEFAULT_CONVERSATION_STATE
        try:
            record = db.get_conversation_by_id(conversation_id)
        except Exception as exc:  # noqa: BLE001 - falls back, never blocks the menu
            logger.debug(
                "Console conversation state lookup failed: exception_type={}",
                type(exc).__name__,
            )
            return DEFAULT_CONVERSATION_STATE
        if not record:
            return DEFAULT_CONVERSATION_STATE
        return str(record.get("state") or DEFAULT_CONVERSATION_STATE)

    def _workspace_menu_target(self, workspace_id: str):
        """Build the pure menu target from registry truth.

        Args:
            workspace_id: Registry id of the workspace row pressed.

        Returns:
            A ``WorkspaceMenuTarget`` (imported lazily per ADR-097) or None
            when the workspace is no longer resolvable -- the caller then
            just does not open a menu.
        """
        from tldw_chatbook.Chat.console_workspace_actions import (
            WorkspaceMenuTarget,
        )

        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )
        record = (
            registry_service.get_workspace(workspace_id)
            if registry_service is not None
            else None
        )
        if record is None:
            return None
        return WorkspaceMenuTarget(
            workspace_id=workspace_id,
            name=str(getattr(record, "name", "") or ""),
            is_active=bool(getattr(record, "active", False)),
            files_available=bool(
                self._files_availability_by_id.get(workspace_id, False)
            ),
        )

    async def _create_console_chat_in_workspace(self, workspace_id: str) -> None:
        """Activate a workspace, then create the new chat inside it.

        TASK-25712: "New chat" on a non-active workspace composes the two
        existing operations -- activation, then session creation, which
        targets the active workspace -- rather than threading a workspace
        parameter through session creation.

        Review, PR #2255: the activation wrapper deliberately discards
        ``_switch_console_workspace``'s failure result (its other callers
        are fire-and-forget), so success is verified against the registry
        afterwards. A missing workspace or a failed switch must NOT create
        the chat in the previously active workspace -- the user asked for
        it somewhere specific.
        """
        registry_service = getattr(
            self.app_instance, "workspace_registry_service", None
        )

        def _record():
            return (
                registry_service.get_workspace(workspace_id)
                if registry_service is not None
                else None
            )

        record = _record()
        if record is None:
            self.app_instance.notify(
                "This workspace is no longer available.", severity="warning"
            )
            return
        if not getattr(record, "active", False):
            self._activate_workspace(workspace_id)
            record = _record()
            if record is None or not getattr(record, "active", False):
                self.app_instance.notify(
                    f"Could not activate {record.name if record else workspace_id}. "
                    "New chat was not created.",
                    severity="warning",
                )
                return
        await self._create_session()

    def apply_workspace_action(self, action_id: str, target: Any) -> None:
        """Run the chosen workspace command against the captured row.

        Dispatched by handler-name convention (ADR-097 lazy-import rule).
        """
        from tldw_chatbook.Chat.console_workspace_actions import (
            ACTION_ACTIVATE,
            ACTION_ARCHIVE,
            ACTION_NEW_CHAT,
            ACTION_RAG_SCOPE,
            ACTION_RENAME,
            ACTION_SHOW_FILES,
        )

        workspace_id = str(target.workspace_id)
        if action_id == ACTION_ACTIVATE:
            self._activate_workspace(workspace_id)
            return
        if action_id == ACTION_NEW_CHAT:
            self.run_worker(
                self._create_console_chat_in_workspace(workspace_id),
                exclusive=True,
                group="console-workspace-new-chat",
            )
            return
        if action_id == ACTION_SHOW_FILES:
            self.run_worker(
                self._request_workspace_files(
                    workspace_id,
                    expected_available=bool(target.files_available),
                ),
                exclusive=False,
                group="console-workspace-files-open",
            )
            return
        if action_id == ACTION_RENAME:
            self._rename_workspace(workspace_id)
            return
        if action_id == ACTION_RAG_SCOPE:
            self.run_worker(
                self._open_workspace_scope(),
                exclusive=True,
                group="console-workspace-scope-open",
            )
            return
        if action_id == ACTION_ARCHIVE:
            self._archive_workspace(workspace_id)
            return

    def _console_target_has_messages(
        self, native_session_id: str, conversation_id: str | None
    ) -> bool:
        """Cheap open-time probe: does this row have any messages?

        TASK-25886: gates the copy entries. A native session asks the live
        store; a persisted conversation asks the database for a single row.
        """
        if native_session_id:
            try:
                store = self._ensure_console_chat_store()
                return bool(store.read_only_messages_for_session(native_session_id))
            except Exception:
                return False
        if not conversation_id:
            return False
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return False
        try:
            return bool(db.get_messages_for_conversation(conversation_id, limit=1))
        except Exception:
            return False

    def _console_markdown_source_messages(self, target) -> list:
        """Return normalized messages for a copy target, or [].

        Source pick (TASK-25886): an open native session reads the LIVE
        chat store (richest fidelity -- in-flight tool structure never
        needed serializing); a persisted conversation reads the database,
        paginated so long chats are not silently truncated at the default
        page size.
        """
        from tldw_chatbook.Chat.console_conversation_markdown import (
            markdown_messages_from_db_rows,
            markdown_messages_from_store,
        )

        native_session_id = str(getattr(target, "native_session_id", "") or "")
        if native_session_id:
            try:
                store = self._ensure_console_chat_store()
                live = store.read_only_messages_for_session(native_session_id)
            except Exception:
                logger.opt(exception=True).debug("copy-markdown live read failed")
                live = []
            if live:
                return markdown_messages_from_store(live)
        conversation_id = (target.conversation_id or "").strip()
        if not conversation_id:
            return []
        db = getattr(self.app_instance, "chachanotes_db", None)
        if db is None:
            return []
        rows: list[dict] = []
        page_size = 200
        offset = 0
        # PR #2262 review: one logical read, one transaction (reads use the
        # shared context manager too), then a lineage filter below.
        with db.transaction():
            while True:
                page = db.get_messages_for_conversation(
                    conversation_id, limit=page_size, offset=offset
                )
                rows.extend(page)
                if len(page) < page_size:
                    break
                offset += page_size
            active_rows = _active_lineage_rows(db, conversation_id, rows)
        return markdown_messages_from_db_rows(active_rows)

    def _render_console_conversation_markdown(self, target, fidelity: str):
        """Render the target chat as markdown, or None when empty."""
        from tldw_chatbook.Chat.console_conversation_markdown import (
            render_conversation_markdown,
        )
        from datetime import date

        messages = self._console_markdown_source_messages(target)
        if not messages:
            return None
        return render_conversation_markdown(
            title=str(getattr(target, "title", "") or ""),
            rendered_at=date.today().isoformat(),
            messages=messages,
            fidelity=fidelity,
        )

    async def _copy_console_conversation_markdown(self, target, fidelity: str) -> None:
        """Copy one conversation to the clipboard as markdown."""
        import asyncio

        # PR #2262 review: the paginated read + render are blocking work;
        # coroutine workers still run on the UI loop, so push them off it.
        markdown = await asyncio.to_thread(
            self._render_console_conversation_markdown, target, fidelity
        )
        if markdown is None:
            self._notify("This chat has no messages to copy.", severity="warning")
            return
        copy_to_clipboard = getattr(self.app_instance, "copy_to_clipboard", None)
        if not callable(copy_to_clipboard):
            self._notify("Clipboard is unavailable.", severity="warning")
            return
        copy_to_clipboard(markdown)
        size_kb = max(1, round(len(markdown.encode("utf-8")) / 1024))
        label = "Clean markdown" if fidelity == "clean" else "Full transcript"
        self._notify(f"Copied {label} ({size_kb} KB).")

    async def _save_console_conversation_markdown(self, target) -> None:
        """Prompt for a path and write the Clean markdown rendering."""
        from pathlib import Path

        from tldw_chatbook.Widgets.Console.console_save_markdown_modal import (
            ConsoleSaveMarkdownModal,
            markdown_filename_slug,
        )

        import asyncio

        markdown = await asyncio.to_thread(
            self._render_console_conversation_markdown, target, "clean"
        )
        if markdown is None:
            self._notify("This chat has no messages to save.", severity="warning")
            return
        title = str(getattr(target, "title", "") or "")
        default_path = str(
            Path.home() / "Downloads" / f"{markdown_filename_slug(title)}.md"
        )

        def _write(chosen: "str | None") -> None:
            if not chosen:
                return
            self.run_worker(
                self._write_console_markdown_file(chosen, markdown),
                exclusive=True,
                group="console-copy-markdown",
            )

        self.push_screen(
            ConsoleSaveMarkdownModal(default_path=default_path), callback=_write
        )

    async def _write_console_markdown_file(self, path_text: str, markdown: str) -> None:
        """Validate and write one markdown export off the loop."""
        from pathlib import Path

        import aiofiles

        from tldw_chatbook.Utils.path_validation import validate_path_simple

        # expanduser FIRST: validate_path_simple rejects unresolved '~'
        # components, and the expansion is exactly what a user means by it.
        candidate = Path(path_text).expanduser()
        try:
            target_path = validate_path_simple(candidate, require_exists=False)
        except Exception as exc:
            self._notify(f"Invalid path: {exc}", severity="error")
            return
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(target_path, "w", encoding="utf-8") as fh:
                await fh.write(markdown)
        except Exception as exc:
            self._notify(f"Could not write file: {exc}", severity="error")
            return
        self._notify(f"Saved {target_path.name}.")

    def apply_conversation_action(self, action_id: str, target: Any) -> None:
        """Run the chosen row command against the captured conversation.

        Dispatched by handler-name convention -- see
        `on_conversation_action_menu_dismissed` for why.
        """
        from tldw_chatbook.Chat.console_conversation_actions import (
            ACTION_ARCHIVE,
            ACTION_DELETE,
            ACTION_FAVORITE,
            ACTION_RENAME,
            ACTION_UNARCHIVE,
            ACTION_UNFAVORITE,
            ARCHIVED_STATE,
            DEFAULT_CONVERSATION_STATE,
            state_from_action,
        )

        conversation_id = (target.conversation_id or "").strip()
        # TASK-25886: copy/save work for open native sessions too (their
        # messages come from the live store), so they route BEFORE the
        # persisted-id guard below.
        if action_id in ("copy-markdown:clean", "copy-markdown:full"):
            self.run_worker(
                self._copy_console_conversation_markdown(
                    target, "clean" if action_id.endswith("clean") else "full"
                ),
                exclusive=True,
                group="console-copy-markdown",
            )
            return
        if action_id == "save-markdown":
            self.run_worker(
                self._save_markdown(target),
                exclusive=True,
                group="console-copy-markdown",
            )
            return
        if not conversation_id:
            self._notify(
                "Send or save this chat before managing it.", severity="warning"
            )
            return

        if action_id in (ACTION_FAVORITE, ACTION_UNFAVORITE):
            self._toggle_star(
                conversation_id,
                starred=action_id == ACTION_UNFAVORITE,
                conversation_title=target.title,
            )
            return

        if action_id == ACTION_RENAME:
            self._rename_conversation(conversation_id, target.title)
            return

        if action_id == ACTION_DELETE:
            self._delete_conversation(conversation_id, target.title)
            return

        new_state = state_from_action(action_id)
        if action_id == ACTION_ARCHIVE:
            new_state = ARCHIVED_STATE
        elif action_id == ACTION_UNARCHIVE:
            new_state = DEFAULT_CONVERSATION_STATE
        if new_state is None:
            return
        self._set_conversation_state(
            conversation_id, new_state, conversation_title=target.title
        )
