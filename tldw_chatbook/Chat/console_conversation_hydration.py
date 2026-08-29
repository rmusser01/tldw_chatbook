"""Turning a persisted conversation into a Console session, with no view.

task-15860 Task 6 (wake at launch). A wake delivered at process start has
to run in a **session** for a conversation nobody has opened -- and until
this module existed, the only code that could build one lived on
`ChatScreen`: `_resume_console_workspace_conversation`
(`UI/Console_Modules/workspace.py`) interleaved the session-producing
policy with a screen's own work (composer snapshot, toasts, resume-marker
overlay, retrieval-scope warm, transcript repaint, focus).

This module is the **session-producing half, moved verbatim** so the two
callers share one policy instead of two:

| Caller | View work it keeps |
|---|---|
| `ChatScreen._resume_console_workspace_conversation` | draft snapshot, both failure toasts, resume-marker overlay, scope warm, UI sync, focus |
| the launch wake (`console_launch_wake.py`) | none -- it has no view |

**What is deliberately NOT here.** The screen's base settings come from
the currently active session (`_console_session_settings_for_resume` ->
`_active_console_session_settings`); a launch has no active session, so it
uses the config defaults the screen falls back to
(`default_console_session_settings`). That difference is inherent to
having no view, and the part that a *conversation* contributes -- its
saved system prompt and pinned prefill -- IS shared, through
`apply_resume_settings_overrides`.

Nothing here touches the DOM, `app.notify`, or any screen attribute; the
only app members read are `chat_conversation_scope_service` and
`chachanotes_db`.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Mapping, Sequence

from loguru import logger

from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_GLOBAL_WORKSPACE_ID,
    ConsoleChatMessage,
    ConsoleMessageRole,
    MessageAttachment,
)
from tldw_chatbook.Chat.console_chat_fork import (
    parse_console_fork_message_metadata,
)
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    parse_persisted_console_session_settings,
)
from tldw_chatbook.Chat.console_prefill import (
    pinned_prefill_from_conversation_metadata,
)
from tldw_chatbook.Chat.console_roleplay_metadata import (
    parse_console_roleplay_context,
)
from tldw_chatbook.Chat.message_metadata import MessageMetadata
from tldw_chatbook.Chat.provider_usage import ProviderUsage
from tldw_chatbook.Video_Generation.video_metadata import VideoGenerationMetadata

__all__ = [
    "ConversationLoadFailed",
    "ConversationServiceUnavailable",
    "apply_resume_settings_overrides",
    "console_messages_from_conversation_tree",
    "hydrate_console_session",
    "load_console_conversation_tree",
]


class ConversationServiceUnavailable(RuntimeError):
    """The app has no conversation service that can load a tree."""


class ConversationLoadFailed(RuntimeError):
    """The conversation service raised while loading the tree."""


def _console_message_role_from_persisted(
    message: Mapping[str, Any],
) -> ConsoleMessageRole:
    """Return a native Console role for a persisted Chat message row."""
    raw_role = str(message.get("role") or "").strip().lower()
    if raw_role:
        try:
            return ConsoleMessageRole(raw_role)
        except ValueError:
            pass
    sender = str(message.get("sender") or "").strip().lower()
    if sender in {"user", "system", "tool"}:
        return ConsoleMessageRole(sender)
    return ConsoleMessageRole.ASSISTANT


def _batch_fetch_resume_attachments(
    db: Any, messages: Sequence[ConsoleChatMessage]
) -> None:
    """Fill positions >= 1 for resumed multi-attachment messages, once.

    ``get_conversation_tree`` only returns the legacy image columns
    (position 0); the ``message_attachments`` table (positions >= 1) is
    fetched here in a SINGLE batched call covering every message this
    resume produced, then folded into each message's attachments tuple via
    ``_apply_console_message_attachments`` (see that helper for the store
    mirror invariant it replicates by hand).
    """
    from tldw_chatbook.UI.Console_Modules.message import (
        _apply_console_message_attachments,
    )

    ids = [m.persisted_message_id for m in messages if m.persisted_message_id]
    if not ids:
        return
    getter = getattr(db, "get_attachments_for_messages", None)
    if not callable(getter):
        return
    try:
        rows_by_id = getter(ids)
    except Exception:
        logger.opt(exception=True).warning(
            "Console resume attachment batch fetch failed."
        )
        return
    if not isinstance(rows_by_id, dict):
        return
    for message in messages:
        extra_rows = (
            rows_by_id.get(message.persisted_message_id)
            if message.persisted_message_id
            else None
        )
        if not extra_rows:
            continue
        extras = [
            MessageAttachment(
                data=row.get("data"),
                mime_type=row.get("mime_type") or "",
                display_name=row.get("display_name") or "",
                position=int(row.get("position", 0)),
            )
            for row in extra_rows
        ]
        _apply_console_message_attachments(message, list(message.attachments) + extras)


def console_messages_from_conversation_tree(
    tree: Mapping[str, Any], *, db: Any = None
) -> list[ConsoleChatMessage]:
    """Build native Console messages from a persisted conversation tree.

    Task 8: flattens the ENTIRE tree (every node, all branches -- not just
    the ``children[-1]`` latest branch), each message carrying its
    ``persisted_message_id`` and persisted ``parent_message_id`` so the
    store can reconnect the full tree and pick the active branch from the
    stored active-leaf pointer.

    Parenthood is taken from the tree's own NESTING (the id of the node we
    recursed from), not the row's ``parent_message_id`` field: the real DB
    tree sets both consistently, but a node's structural position is the
    authoritative source and stays correct even for trees whose rows omit
    the field. A truly-empty node (no content and no image) is dropped but
    transparent to parenthood -- its children re-parent to the nearest kept
    ancestor -- so a skipped row never orphans a branch.

    Args:
        tree: The persisted conversation tree.
        db: The ChaChaNotes DB used for the one batched attachment fetch
            (positions >= 1). ``None`` skips it, exactly as a DB without
            ``get_attachments_for_messages`` does.

    Returns:
        Every node of the tree, pre-order, as Console messages.
    """
    messages: list[ConsoleChatMessage] = []

    # TASK-22206: an explicit stack, not recursion -- a linear conversation's
    # tree is as deep as it is long, and the old per-node recursion raised
    # RecursionError on resume at ~1000 messages. Children are pushed in
    # reverse so pop order preserves the original pre-order. The visited
    # guard turns a (malformed) self-referential mapping into a skip instead
    # of a hang; a well-formed tree never revisits a node.
    stack: list[tuple[Any, str | None]] = []
    root_threads = tree.get("root_threads")
    if isinstance(root_threads, list):
        stack = [(root, None) for root in reversed(root_threads)]
    visited_node_ids: set[int] = set()
    while stack:
        node, parent_persisted_id = stack.pop()
        if not isinstance(node, dict) or id(node) in visited_node_ids:
            continue
        visited_node_ids.add(id(node))
        content = str(node.get("content") or "")
        raw_image = node.get("image_data")
        image_data = (
            bytes(raw_image) if isinstance(raw_image, (bytes, bytearray)) else None
        )
        raw_mime = node.get("image_mime_type")
        image_mime_type = str(raw_mime) if raw_mime else None
        usage = ProviderUsage.from_json(node.get("usage_json"))
        raw_metadata_json = node.get("metadata_json")
        # task-3401.4: a video generation row's metadata_json carries the
        # namespaced video payload instead of turn provenance -- hydrate
        # it into video_metadata and leave ``metadata`` None (the two
        # shapes never co-write one row; persistence prefers the video
        # payload so a later edit cannot clobber it).
        video_metadata = VideoGenerationMetadata.from_json(raw_metadata_json)
        fork_metadata = (
            None
            if video_metadata is not None
            else parse_console_fork_message_metadata(raw_metadata_json)
        )
        metadata = (
            None
            if video_metadata is not None or fork_metadata is not None
            else MessageMetadata.from_json(raw_metadata_json)
        )
        raw_id = node.get("id")
        node_persisted_id = str(raw_id) if raw_id is not None else None
        generation_state = node.get("assistant_generation_state")
        kept = (
            bool(content)
            or image_data is not None
            or generation_state is not None
            or node.get("provider_continuation_json") is not None
        )
        if kept:
            # The tree only carries the legacy position-0 columns; positions
            # >= 1 (multi-attachment table rows) are batch-fetched below,
            # once for the whole resumed list.
            attachments: tuple[MessageAttachment, ...] = (
                (
                    MessageAttachment(
                        data=image_data,
                        mime_type=image_mime_type or "",
                        display_name=(fork_metadata[1] if fork_metadata else ""),
                        position=0,
                    ),
                )
                if image_data is not None
                else ()
            )
            messages.append(
                ConsoleChatMessage(
                    role=_console_message_role_from_persisted(node),
                    content=content,
                    status=fork_metadata[0] if fork_metadata else "complete",
                    persisted_message_id=node_persisted_id,
                    parent_message_id=parent_persisted_id,
                    image_data=image_data,
                    image_mime_type=image_mime_type,
                    attachment_label=(
                        fork_metadata[1]
                        if fork_metadata is not None and fork_metadata[1]
                        else None
                    ),
                    attachments=attachments,
                    usage=usage,
                    metadata=metadata,
                    assistant_generation_state=(
                        str(generation_state) if generation_state is not None else None
                    ),
                    video_metadata=video_metadata,
                )
            )
        # Children re-parent to this node when kept, else pass the nearest
        # kept ancestor straight through (a dropped empty row is invisible
        # to the tree linkage).
        child_parent_id = node_persisted_id if kept else parent_persisted_id
        children = node.get("children")
        if isinstance(children, list):
            for child in reversed(children):
                stack.append((child, child_parent_id))

    _batch_fetch_resume_attachments(db, messages)
    return messages


async def load_console_conversation_tree(
    app: Any, conversation_id: str
) -> Mapping[str, Any] | None:
    """Load one persisted conversation's WHOLE tree.

    The depth/root caps are policy, not tuning: the service defaults (50)
    truncate a long or branchy conversation, and a truncated tree produces
    a wrong provider payload for whoever sends next. Raising them here
    once is what keeps the screen resume and the launch wake honest about
    the same conversation.

    Args:
        app: The app object; read for ``chat_conversation_scope_service``.
        conversation_id: The durable conversation id.

    Returns:
        The tree, or ``None`` when the conversation record is MISSING (the
        caller owns that failure's UX).

    Raises:
        ConversationServiceUnavailable: No conversation service is wired.
        ConversationLoadFailed: The service raised.
    """
    service = getattr(app, "chat_conversation_scope_service", None)
    get_conversation_tree = getattr(service, "get_conversation_tree", None)
    if not callable(get_conversation_tree):
        raise ConversationServiceUnavailable(
            "Saved conversation resume is unavailable in this build."
        )
    try:
        # Task 8: raise the depth/root caps well past the service defaults
        # (50) so a long or branchy conversation's full tree -- every
        # branch, not just the latest -- is loaded intact, not truncated.
        maybe_tree = get_conversation_tree(
            conversation_id, mode="local", depth_cap=10_000, root_limit=10_000
        )
        tree = await maybe_tree if inspect.isawaitable(maybe_tree) else maybe_tree
    except Exception as exc:  # noqa: BLE001 -- re-raised as this module's type
        raise ConversationLoadFailed(str(conversation_id)) from exc
    if not isinstance(tree, dict) or not tree.get("conversation"):
        return None
    return tree


def apply_resume_settings_overrides(
    settings: ConsoleSessionSettings, conversation: Mapping[str, Any]
) -> ConsoleSessionSettings:
    """Restore one saved settings snapshot plus canonical row-owned fields.

    A complete valid versioned snapshot replaces the caller's provider and
    generation defaults. Invalid metadata is ignored as one unit. The row's
    ``system_prompt`` and top-level pinned-prefill metadata remain canonical.

    Blank/whitespace-only prompt text collapses to "no system prompt";
    anything else is restored verbatim (leading/trailing whitespace and
    internal formatting included) rather than stripped, so a
    formatting-sensitive prompt survives close/resume unchanged.

    Args:
        settings: The base settings snapshot.
        conversation: The persisted conversation row.

    Returns:
        Persisted settings, or the base snapshot on malformed metadata, with
        canonical prompt and pinned-prefill fields applied.
    """
    raw_system_prompt = conversation.get("system_prompt")
    system_prompt = (
        raw_system_prompt
        if isinstance(raw_system_prompt, str) and raw_system_prompt.strip()
        else None
    )
    pinned_prefill = pinned_prefill_from_conversation_metadata(
        conversation.get("metadata")
    )
    persisted = parse_persisted_console_session_settings(conversation.get("metadata"))
    return replace(
        persisted or settings,
        system_prompt=system_prompt,
        pinned_prefill=pinned_prefill,
    )


async def hydrate_console_session(
    *,
    app: Any,
    store: Any,
    conversation_id: str,
    tree: Mapping[str, Any],
    settings: ConsoleSessionSettings | None,
    target_scope_type: str | None = None,
    target_workspace_id: str | None = None,
    activate: bool = True,
) -> Any:
    """Create a Console session from a persisted tree.

    Moved verbatim out of `_resume_console_workspace_conversation`: the
    workspace resolution, the title fallback, the whole-tree node build,
    the durable two-component cursor, the runtime-backend/assistant/character
    field discipline, the `restore_persisted_session` call and the roleplay
    overlay, including saved character-name identity. The screen's own work
    (marker overlay, scope warm, repaint) stays in the screen.

    Args:
        app: The app object; read for ``chachanotes_db`` only.
        store: The Console chat store the session is created in.
        conversation_id: The durable conversation id being resumed.
        tree: The tree from `load_console_conversation_tree`.
        settings: The settings snapshot for the new session.
        target_scope_type: ``"global"`` pins the global workspace.
        target_workspace_id: Requested workspace, used only when the
            conversation carries none.
        activate: Whether to activate the hydrated session after its policy
            state has been restored.

    Returns:
        The newly created `ConsoleChatSession`, activated when ``activate``
        is true.
    """
    target = str(conversation_id or "").strip()
    conversation = tree.get("conversation")
    if not isinstance(conversation, dict):
        conversation = {}
    roleplay_context = parse_console_roleplay_context(conversation.get("metadata"))
    active_workspace_id = str(store.workspace_context.active_workspace_id or "").strip()
    persisted_workspace_id = (
        str(conversation.get("workspace_id")).strip()
        if conversation.get("workspace_id") is not None
        else ""
    )
    target_scope = str(target_scope_type or "").strip()
    requested_workspace_id = (
        str(target_workspace_id).strip() if target_workspace_id is not None else ""
    )
    if target_scope == "global":
        workspace_id = CONSOLE_GLOBAL_WORKSPACE_ID
    else:
        workspace_id = (
            persisted_workspace_id
            or requested_workspace_id
            or active_workspace_id
            or None
        )
    title = str(conversation.get("title") or "Saved conversation").strip()
    if not title:
        title = "Saved conversation"
    # Task 8: load the WHOLE persisted tree (every branch), then reconstruct
    # the active branch from the stored two-component cursor. Loading all
    # branches (not just the latest) is what makes off-path siblings
    # navigable (swipe) right after resume.
    db = getattr(app, "chachanotes_db", None)
    all_nodes = console_messages_from_conversation_tree(tree, db=db)
    cursor_reader = getattr(db, "get_conversation_active_cursor", None)
    if callable(cursor_reader):
        active_leaf_id, active_leaf_before_id = cursor_reader(target)
    else:
        active_leaf_id = getattr(
            db, "get_conversation_active_leaf", lambda _target: None
        )(target)
        active_leaf_before_id = None
    raw_runtime_backend = conversation.get("runtime_backend")
    if type(raw_runtime_backend) is str:
        runtime_backend = raw_runtime_backend
    else:
        runtime_backend = ""
    raw_assistant_kind = conversation.get("assistant_kind")
    assistant_kind = raw_assistant_kind if type(raw_assistant_kind) is str else None
    raw_assistant_id = conversation.get("assistant_id")
    assistant_id = raw_assistant_id if type(raw_assistant_id) is str else None
    raw_assistant_authority_id = conversation.get("assistant_authority_id")
    assistant_authority_id = (
        raw_assistant_authority_id if type(raw_assistant_authority_id) is str else None
    )
    raw_persona_memory_mode = conversation.get("persona_memory_mode")
    persona_memory_mode = (
        raw_persona_memory_mode if type(raw_persona_memory_mode) is str else None
    )
    if assistant_kind is None:
        # Conversation metadata normalizes the legacy/default ``generic`` kind
        # to the canonical unscoped form. Keep the rest of that identity in
        # the same form instead of hydrating an impossible mixed identity.
        assistant_id = None
        assistant_authority_id = None
        persona_memory_mode = None
    raw_character_id = conversation.get("character_id")
    character_id = (
        raw_character_id
        if (
            runtime_backend == "local"
            and assistant_kind == "character"
            and type(raw_character_id) is int
            and raw_character_id > 0
            and assistant_id == str(raw_character_id)
        )
        else None
    )
    character_name = (
        roleplay_context.character_name_snapshot
        if assistant_kind == "character"
        else None
    )
    if settings is not None:
        settings = replace(settings, character_label=character_name or "")
    prior_active_session_id = store.active_session_id
    session = store.restore_persisted_session(
        title=title,
        workspace_id=workspace_id,
        persisted_conversation_id=target,
        all_nodes=all_nodes,
        active_leaf_persisted_id=active_leaf_id,
        active_leaf_before_persisted_id=active_leaf_before_id,
        settings=settings,
        runtime_backend=runtime_backend,
        assistant_kind=assistant_kind,
        assistant_id=assistant_id,
        assistant_authority_id=assistant_authority_id,
        persona_memory_mode=persona_memory_mode,
        character_id=character_id,
        character_name=character_name,
        activate=False,
    )
    try:
        await store.hydrate_session_library_policy(session.id)
        await store.reconcile_pending_workspace_projection(session.id)
        session.user_display_name_override = roleplay_context.user_name_override
        session.character_system_template = roleplay_context.character_system_template
        if activate:
            store.switch_session(session.id)
    except BaseException:
        store.rollback_restored_session(
            session.id,
            expected_session=session,
            prior_active_session_id=prior_active_session_id,
        )
        raise
    return session
