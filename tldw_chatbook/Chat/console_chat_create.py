"""Agent Chat creation through the controller's existing preparation and commit seams."""

from __future__ import annotations

from . import console_chat_controller as host


def execute_agent_chat_create(
    self, payload: dict[str, host.Any]
) -> dict[str, host.Any]:
    """WORKER THREAD: create the confirmed chat and marshal UI completion.

    TASK-32482 Task 7. Runs ONLY after ``request_chat_create_confirm``
    returned an allow (the bridge closure enforces that ordering), so
    refusal before creation makes no chat; the outcome's ``kind``
    tells the model (and the run log) why. A source retired during I/O
    refuses UI placement. Legacy forks discard their orphan row; prepared
    new chats retain their successfully saved draft before launch.

    For ``fork_chat`` the new conversation copies the source's active
    path verbatim (``ChatConversationService.copy_conversation_active_
    path``) and records fork lineage (``parent_conversation_id`` +
    ``forked_from_message_id`` -- both FK-enforced, so only real
    persisted ids are ever passed). For ``new_chat`` a fresh
    conversation is created in the owning session's workspace scope.
    Both stamp the ``console_agent_handoff`` metadata key (``draft``/
    ``created_via``/``source_run_id``) Task 8 rehydrates from.

    On success the UI-thread completion (``complete_agent_chat_create``)
    is marshaled via ``app.call_from_thread`` with the new session's
    placement inputs; the executor itself returns the outcome dict the
    bridge closure renders into the ToolResult.

    Args:
        payload: ``{"tool" ("fork_chat"|"new_chat"), "session_id",
            "run_id", "title", "opening_prompt", "instructions"}``.

    Returns:
        ``{"ok": True, "title", "conversation_id", "workspace_id",
        "copied_messages", "draft_set"}`` on success, else
        ``{"ok": False, "kind", "error"}`` with ``kind`` one of
        ``session_gone``, ``payload_too_large``,
        ``source_not_persisted``, ``character_conflict``,
        ``empty_history``, ``execution_failed``.
    """
    from tldw_chatbook.Agents.agent_models import (
        CHAT_CREATE_PAYLOAD_MAX,
        CHAT_CREATE_TITLE_MAX,
    )
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService

    tool = str(payload.get("tool") or "")
    session_id = str(payload.get("session_id") or "")
    # A committed Close retains its source during drain; an earlier Allow
    # cannot authorize a new durable chat once that Close has committed.
    if session_id in self._session_close_generations:
        return {
            "ok": False,
            "kind": host._CHAT_CREATE_SESSION_GONE,
            "error": "source session is closing",
        }
    if tool == "new_chat":
        return self._execute_prepared_chat_create(payload)

    session = next((s for s in self.store.sessions() if s.id == session_id), None)
    if session is None:
        return {
            "ok": False,
            "kind": host._CHAT_CREATE_SESSION_GONE,
            "error": "source session not found",
        }
    title = str(payload.get("title") or "").strip()
    opening_prompt = str(payload.get("opening_prompt") or "")
    instructions = str(payload.get("instructions") or "")
    # Qodo round finding 4: same strict-type contract as the bridge
    # closures -- the executor is reachable from callers other than the
    # closures, so a non-string routing input is a clear refusal here
    # too, never a str() coercion.
    for _name in ("provider", "model", "preset"):
        _value = payload.get(_name, "")
        if not isinstance(_value, str):
            return {
                "ok": False,
                "kind": "invalid_args",
                "error": f"{_name} must be a string, got {type(_value).__name__}",
            }
    req_provider = str(payload.get("provider") or "").strip()
    req_model = str(payload.get("model") or "").strip()
    req_preset = str(payload.get("preset") or "").strip()
    # Mirrors the bridge closures' cap (CHAT_CREATE_PAYLOAD_MAX in
    # agent_models -- the single shared constant, PR review #5);
    # duplicated here so the executor stays safe even if wired to a
    # caller other than the closures.
    if (
        len(opening_prompt) > CHAT_CREATE_PAYLOAD_MAX
        or len(instructions) > CHAT_CREATE_PAYLOAD_MAX
    ):
        return {
            "ok": False,
            "kind": "payload_too_large",
            "error": f"payload exceeds {CHAT_CREATE_PAYLOAD_MAX} chars",
        }
    persistence = self.store.persistence
    db = getattr(persistence, "db", None)
    if persistence is None or db is None:
        return {
            "ok": False,
            "kind": "execution_failed",
            "error": "persistence unavailable",
        }

    # TASK-32874: optional routing for the CREATED chat. Resolved
    # BEFORE anything is created; every RoutingError surfaces as the
    # outcome's kind with nothing created (ADR-344 vocabulary).
    routed_settings: host.Any = None
    if req_provider or req_model or req_preset:
        from dataclasses import replace as _dc_replace

        from tldw_chatbook.Agents.agent_models import definition_from_row
        from tldw_chatbook.Agents.agent_routing import (
            RoutingError as _RoutingError,
        )
        from tldw_chatbook.Agents.agent_routing import (
            load_agents_routing_config as _load_routing,
        )
        from tldw_chatbook.Agents.agent_routing import (
            resolve_spawn_target as _resolve_target,
        )
        from tldw_chatbook.Chat.console_session_settings import (
            build_default_console_session_settings as _build_settings,
        )
        from tldw_chatbook.config import load_settings as _load_app_config

        app_config = _load_app_config()
        routing_cfg = _load_routing()
        preset_def = None
        if req_preset:
            agent_db = getattr(self._agent_bridge, "agent_runs_db", None)
            rows = (
                agent_db.list_agent_definitions(enabled_only=True)
                if agent_db is not None
                else []
            )
            match = next(
                (r for r in rows if str(r.get("name") or "") == req_preset), None
            )
            if match is None:
                return {
                    "ok": False,
                    "kind": "unknown_preset",
                    "error": f"no enabled agent preset named '{req_preset}'",
                }
            preset_def = definition_from_row(match)
            if not (preset_def.provider or preset_def.model):
                return {
                    "ok": False,
                    "kind": "preset_unrouted",
                    "error": f"preset '{req_preset}' sets no provider/model",
                }
        try:
            # The [agents] SUB-AGENT default level does not apply to
            # chat creation (this is not a spawn): clear it so the
            # levels are exactly ad-hoc -> preset -> parent-fill.
            target = _resolve_target(
                app_config,
                parent_provider=(session.settings.provider if session.settings else ""),
                parent_model=(session.settings.model if session.settings else ""),
                preset=preset_def,
                override_provider=req_provider,
                override_model=req_model,
                # The [agents] sub-agent default level never applies
                # here (uniformly cleared for every chat-create
                # request): levels are ad-hoc -> preset -> parent-fill.
                routing=_dc_replace(
                    routing_cfg,
                    subagent_default_provider="",
                    subagent_default_model="",
                ),
            )
        except _RoutingError as exc:
            return {"ok": False, "kind": str(exc.code), "error": str(exc)}
        # Qodo round finding 2: the resolver's merged params (preset
        # params above registry-entry params above defaults) ride the
        # ADR-344 extra_sources seam so a routed chat keeps the
        # preset/endpoint tuning, not just provider/model.
        routed_settings = _build_settings(
            app_config,
            provider=target.provider,
            model=target.model or None,
            extra_sources=(
                (dict(getattr(target, "params", None) or ()),)
                if getattr(target, "params", None)
                else ()
            ),
        )

    source_conv: str | None = None
    source_row: dict[str, host.Any] | None = None
    new_conv: str | None = None
    saved_producer = None
    copied = 0
    if tool == "fork_chat":
        source_conv = session.persisted_conversation_id
        if session.ephemeral or not source_conv:
            return {
                "ok": False,
                "kind": "source_not_persisted",
                "error": "the current chat is temporary; nothing to fork",
            }
    elif not title:
        title = "New Chat"

    def _handoff_metadata(
        source_metadata: str | None, settings_snapshot: host.Any = None
    ) -> dict[str, host.Any]:
        """PR review #9: MERGE the handoff key into the source's own
        metadata (speech/roleplay/canvas prefs ride along) instead of
        replacing it with a handoff-only mapping."""
        merged: dict[str, host.Any] = {}
        if source_metadata:
            try:
                parsed = host.json.loads(source_metadata)
                if isinstance(parsed, dict):
                    merged.update(parsed)
            except ValueError:
                host.logger.debug("chat_create: source metadata unparseable; replaced")
        # Qodo 2761 finding 5: stamp the TRUE owning run id (the
        # executor runs inside the run context; the closure's payload
        # carries the originating assistant-message id placeholder).
        from tldw_chatbook.Agents.run_context import current_run_id as _cur_run

        merged["console_agent_handoff"] = {
            "draft": opening_prompt,
            "created_via": tool,
            "source_run_id": str(_cur_run() or payload.get("run_id") or ""),
        }
        if settings_snapshot is not None:
            # Qodo round finding 3: the routed generation snapshot is
            # DURABLE -- merged under the same generation-settings
            # metadata key every reopened conversation reads, so the
            # routing survives a failed completion or a later reopen,
            # not just the in-memory restore.
            from tldw_chatbook.Chat.console_generation_settings_metadata import (
                merge_console_generation_settings as _merge_gen,
            )
            from tldw_chatbook.Chat.console_generation_settings_metadata import (
                snapshot_from_session_settings as _snap,
            )

            merged = _merge_gen(merged, _snap(settings_snapshot))
        return merged

    try:
        if tool == "fork_chat":
            tree = ChatConversationService(db).get_conversation_tree(
                str(source_conv), root_limit=10_000, depth_cap=10_000
            )
            source_row = dict(tree.get("conversation") or {})
            if not title:
                title = f"Fork of {source_row.get('title') or 'chat'}"[
                    :CHAT_CREATE_TITLE_MAX
                ]
            if instructions and source_row.get("character_id"):
                return {
                    "ok": False,
                    "kind": "character_conflict",
                    "error": "character chats keep their persona; "
                    "fork without instructions",
                }
            conversation_service = ChatConversationService(db)
            # PR review #3: resolve the EFFECTIVE leaf (pointer, else
            # latest -- always a live row) through the same resolver the
            # copy uses, so the FK-enforced lineage column can never
            # receive a stale/dangling pointer.
            source_leaf = conversation_service.effective_active_leaf(str(source_conv))
            if source_leaf is None:
                return {
                    "ok": False,
                    "kind": "empty_history",
                    "error": "nothing to fork yet; use new_chat",
                }
            new_conv = str(host.uuid4())
            saved_producer = self.app.call_from_thread(
                self.store.reserve_saved_creation, new_conv
            )
            persistence.create_conversation(
                conversation_id=new_conv,
                conversation_title=title,
                scope_type=source_row.get("scope_type") or "global",
                workspace_id=source_row.get("workspace_id"),
                system_prompt=instructions or source_row.get("system_prompt"),
                character_id=source_row.get("character_id"),
                assistant_kind=source_row.get("assistant_kind"),
                assistant_id=source_row.get("assistant_id"),
                assistant_authority_id=source_row.get("assistant_authority_id"),
                metadata=_handoff_metadata(
                    source_row.get("metadata"),
                    settings_snapshot=routed_settings,
                ),
                parent_conversation_id=source_conv,
                forked_from_message_id=source_leaf,
            )
            copy_outcome = conversation_service.copy_conversation_active_path(
                str(source_conv), new_conv
            )
            copied = int(copy_outcome["copied"])
        else:
            workspace_id = session.workspace_id
            scope = (
                "global"
                if workspace_id in (None, host.CONSOLE_GLOBAL_WORKSPACE_ID)
                else "workspace"
            )
            new_conv = persistence.create_conversation(
                conversation_title=title,
                scope_type=scope,
                workspace_id=None if scope == "global" else workspace_id,
                system_prompt=instructions or None,
                metadata=_handoff_metadata(None, routed_settings),
            )
    except ValueError as exc:
        if new_conv is not None:
            self._discard_chat_create_orphan(db, new_conv)
        if saved_producer is not None:
            self.app.call_from_thread(
                self.store.abort_saved_creation, saved_producer, exc
            )
        if "empty_history" in str(exc):
            return {
                "ok": False,
                "kind": "empty_history",
                "error": "nothing to fork yet; use new_chat",
            }
        return {"ok": False, "kind": "execution_failed", "error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        if new_conv is not None:
            self._discard_chat_create_orphan(db, new_conv)
        if saved_producer is not None:
            self.app.call_from_thread(
                self.store.abort_saved_creation, saved_producer, exc
            )
        return {"ok": False, "kind": "execution_failed", "error": str(exc)}

    # PR reviews #6/#10: hydrate the new session's transcript nodes on
    # THIS worker thread -- the UI-thread completion must not run the
    # fork's DB reads (a large fork would freeze the chat UI). A
    # hydration failure degrades to an empty transcript (the durable
    # messages are safe); it never strands the created chat.
    nodes: list[host.Any] = []
    new_leaf: str | None = None
    try:
        from tldw_chatbook.Chat.console_conversation_hydration import (
            console_messages_from_conversation_tree,
        )

        new_tree = ChatConversationService(db).get_conversation_tree(
            new_conv, root_limit=10_000, depth_cap=10_000
        )
        nodes = list(console_messages_from_conversation_tree(new_tree, db=db))
        new_leaf = db.get_conversation_active_leaf(new_conv)
    except Exception:  # noqa: BLE001 -- hydration is best-effort here
        host.logger.opt(exception=True).warning(
            "chat_create: new-chat hydration failed; restoring empty transcript"
        )

    # PR review #8: a globally scoped fork carries workspace_id None;
    # the session restore path expects the Console's GLOBAL workspace
    # id, not None (None is replaced by create_session's default).
    completion_workspace_id = (
        (source_row or {}).get("workspace_id")
        if tool == "fork_chat"
        else session.workspace_id
    ) or host.CONSOLE_GLOBAL_WORKSPACE_ID
    # PR review #1: pass the conversation's identity through so the
    # restored session keeps the fork's assistant/character identity
    # instead of generic Console defaults (settings themselves stay
    # None: the restore path rebuilds provider defaults from the
    # durable generation snapshot like any reopened conversation).
    identity = {
        "assistant_kind": (source_row or {}).get("assistant_kind"),
        "assistant_id": (source_row or {}).get("assistant_id"),
        "assistant_authority_id": (source_row or {}).get("assistant_authority_id"),
        "persona_memory_mode": (source_row or {}).get("persona_memory_mode"),
        "character_id": (source_row or {}).get("character_id"),
        "character_name": (source_row or {}).get("character_name"),
    }
    if tool != "fork_chat":
        identity = {
            "assistant_kind": session.assistant_kind,
            "assistant_id": session.assistant_id,
            "assistant_authority_id": session.assistant_authority_id,
            "persona_memory_mode": None,
            "character_id": session.local_character_id(),
            "character_name": session.character_name
            if session.local_character_id() is not None
            else None,
        }

    completion_admitted = False

    async def complete_if_source_open(**kwargs: host.Any) -> bool:
        """UI THREAD: qualify the source before using the current view sink.

        Args:
            kwargs: Created-chat placement inputs resolved by this worker.

        Returns:
            False if Close/disposal retired the source; True otherwise,
            including a live source whose view is currently detached.
        """
        nonlocal completion_admitted
        if not self._chat_create_source_is_open(session_id):
            return False
        completion_admitted = True
        created_session = await self.store.restore_persisted_session_owned(
            producer=saved_producer,
            title=title,
            workspace_id=completion_workspace_id,
            persisted_conversation_id=new_conv,
            all_nodes=nodes,
            active_leaf_persisted_id=new_leaf,
            settings=routed_settings,
            **identity,
        )
        self.store._project_workspace_membership_after_commit(created_session)
        complete = self.complete_agent_chat_create
        if complete is not None:
            complete(session_id=session_id, **kwargs)
        return True

    completion_allowed = self._chat_create_source_is_open(session_id)
    app = self.app
    if completion_allowed and app is not None:
        try:
            completion_allowed = app.call_from_thread(
                complete_if_source_open,
                conversation_id=new_conv,
                title=title,
                tool=tool,
                opening_prompt=opening_prompt,
                workspace_id=completion_workspace_id,
                nodes=nodes,
                active_leaf_persisted_id=new_leaf,
                settings=routed_settings,
                **identity,
            )
        except Exception:
            # A retiring app can refuse the UI hop before running its gate.
            # Keep unrelated completion failures on their existing path.
            if saved_producer is not None and not saved_producer.ready:
                self.app.call_from_thread(
                    self.store.abort_saved_creation,
                    saved_producer,
                    RuntimeError("Native fork publication failed."),
                )
            if completion_admitted or self._chat_create_source_is_open(session_id):
                raise
            completion_allowed = False
    if not completion_allowed:
        self._discard_chat_create_orphan(db, new_conv)
        if saved_producer is not None:
            self.app.call_from_thread(
                self.store.abort_saved_creation,
                saved_producer,
                RuntimeError("Native source closed before publication."),
            )
        return {
            "ok": False,
            "kind": host._CHAT_CREATE_SESSION_GONE,
            "error": "source session closed before chat completion",
        }
    return {
        "ok": True,
        "title": title,
        "conversation_id": new_conv,
        "workspace_id": completion_workspace_id,
        "copied_messages": copied,
        "draft_set": bool(opening_prompt),
    }
