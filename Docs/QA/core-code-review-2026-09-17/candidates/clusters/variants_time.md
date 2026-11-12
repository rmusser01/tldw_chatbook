
====================================================================================================
# _utc_now: 23 defs, 12 distinct bodies, 11 distinct shapes

--- body 80deec59 (shape c7daaaf1): 4 copies (core 3, interop 1)
   files: Chat/console_trace_runtime.py:48, Notifications/event_state_repository.py:1964, Notifications/server_notification_events.py:268, Sync_Interop/sync_state_repository.py:3383
   | def _utc_now() -> str:
   |     return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

--- body 5beccd49 (shape e7cdf11e): 3 copies (core 3, interop 0)
   files: Chatbooks/local_chatbook_service.py:139, Notes/file_notes_replica.py:766, Subscriptions/local_watchlists_service.py:2085
   | def _utc_now() -> str:
   |         return datetime.now(timezone.utc).isoformat()

--- body 77f69cad (shape f85f0ee3): 2 copies (core 2, interop 0)
   files: Agents/agent_runtime.py:129, Agents/agent_service.py:1185
   | def _utc_now() -> datetime:
   |     """Return the UTC wall clock used to stamp durable agent steps."""
   |     return datetime.now(timezone.utc)

--- body 37c0fcb4 (shape 648ade8b): 2 copies (core 2, interop 0)
   files: Canvas/repository.py:1556, Canvas/staging.py:776
   | def _utc_now() -> str:
   |     return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")

--- body 9fa339a8 (shape 02374e39): 2 copies (core 2, interop 0)
   files: LLM_Provider_Catalog/model_discovery_disk_cache.py:32, TTS/profile_repository.py:564
   | def _utc_now() -> datetime:
   |     return datetime.now(UTC)

--- body 007d065e (shape 36eb239f): 2 copies (core 2, interop 0)
   files: Library/collections_legacy_recovery.py:120, Research_Workspace/source_operation_store.py:59
   | def _utc_now() -> str:
   |     return (
   |         datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
   |     )

--- body fe3335e9 (shape e693dff5): 2 copies (core 2, interop 0)
   files: Library/library_collections_service.py:614, Library/review_set_service.py:32
   | def _utc_now() -> str:
   |     return (
   |         datetime.now(timezone.utc)
   |         .replace(microsecond=0)
   |         .isoformat()
   |         .replace("+00:00", "Z")
   |     )

--- body d184c6da (shape 3b929824): 2 copies (core 2, interop 0)
   files: Research_Workspace/local_adapter.py:107, Research_Workspace/server_adapter.py:101
   | def _utc_now() -> str:
   |     return datetime.now(timezone.utc).isoformat(timespec="seconds")

--- body 03fd93de (shape 36eb239f): 1 copies (core 1, interop 0)
   files: Chat/console_voice_trace_gateway.py:343
   | def _utc_now() -> str:
   |     return (
   |         datetime.now(timezone.utc)
   |         .isoformat(timespec="microseconds")
   |         .replace("+00:00", "Z")
   |     )

--- body 84067691 (shape 2a6814d9): 1 copies (core 1, interop 0)
   files: LLM_Management/snapshot_store.py:69
   | def _utc_now() -> str:
   |     return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

--- body 9df3ed3f (shape 547cf391): 1 copies (core 0, interop 1)
   files: Sync_Interop/notes_organization_sync_service.py:2359
   | def _utc_now() -> str:
   |     return datetime.now(UTC).isoformat()

--- body de10ce25 (shape c9f44e01): 1 copies (core 1, interop 0)
   files: Tool_Packs/binding.py:619
   | def _utc_now(self) -> datetime:
   |         now = self._now()
   |         if type(now) is not datetime or now.tzinfo is None or now.utcoffset() is None:
   |             raise ToolPackError("bind", "confirmation_invalid")
   |         return now.astimezone(timezone.utc)

====================================================================================================
# _utc_now_iso: 3 defs, 3 distinct bodies, 3 distinct shapes

--- body 5beccd49 (shape e7cdf11e): 1 copies (core 1, interop 0)
   files: Chat/console_chat_store.py:1361
   | def _utc_now_iso() -> str:
   |     return datetime.now(timezone.utc).isoformat()

--- body 8cb9c53c (shape a5a88207): 1 copies (core 1, interop 0)
   files: Chatbooks/server_chatbook_service.py:29
   | def _utc_now_iso() -> str:
   |     return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

--- body 80deec59 (shape c7daaaf1): 1 copies (core 0, interop 1)
   files: UX_Interop/server_parity_contracts.py:561
   | def _utc_now_iso() -> str:
   |     return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

====================================================================================================
# _now: 15 defs, 8 distinct bodies, 8 distinct shapes

--- body 5beccd49 (shape e7cdf11e): 5 copies (core 2, interop 3)
   files: Character_Chat/local_character_persona_service.py:160, Character_Chat/local_chat_dictionary_service.py:205, Chat_Grammars_Interop/local_chat_grammars_service.py:40, Feedback_Interop/local_feedback_service.py:40, Kanban_Interop/local_kanban_service.py:146
   | def _now() -> str:
   |         return datetime.now(timezone.utc).isoformat()

--- body 80deec59 (shape c7daaaf1): 3 copies (core 2, interop 1)
   files: Chat/chat_conversation_service.py:321, Chat/conversation_local_marks_service.py:94, Writing_Interop/local_writing_service.py:415
   | def _now() -> str:
   |         return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

--- body f13ca73e (shape 4e066fc0): 2 copies (core 2, interop 0)
   files: Notes/note_import_receipts.py:308, Notes/notes_device_state_store.py:244
   | def _now() -> int:
   |     return max(1, time.time_ns())

--- body 9df3ed3f (shape 547cf391): 1 copies (core 1, interop 0)
   files: DB/agent_worktrees.py:31
   | def _now() -> str:
   |     return datetime.now(UTC).isoformat()

--- body 03fd93de (shape 36eb239f): 1 copies (core 1, interop 0)
   files: Library/collections_capture_repository.py:102
   | def _now() -> str:
   |     return (
   |         datetime.now(timezone.utc)
   |         .isoformat(timespec="microseconds")
   |         .replace("+00:00", "Z")
   |     )

--- body 807ff943 (shape 3bcb73cd): 1 copies (core 1, interop 0)
   files: MCP/server_request_handlers.py:173
   | def _now(self) -> float:
   |         if self._now_fn is not None:
   |             return self._now_fn()
   |         from time import monotonic
   |         return monotonic()

--- body 0a752750 (shape b1a553d4): 1 copies (core 1, interop 0)
   files: Personal_Context/repository.py:318
   | def _now() -> datetime:
   |     now = datetime.now(UTC)
   |     return now.replace(microsecond=now.microsecond // 1000 * 1000)

--- body 1191daba (shape f53ee643): 1 copies (core 0, interop 1)
   files: Research_Interop/local_research_service.py:529
   | def _now() -> str:
   |         """Current UTC time in the shared lease-comparable timestamp format.

   |         Returns:
   |             An ISO-8601 UTC timestamp string produced by
   |             ``_format_timestamp``.
   |         """
   |         return LocalResearchService._format_timestamp(datetime.now(timezone.utc))

====================================================================================================
# _now_iso: 7 defs, 3 distinct bodies, 3 distinct shapes

--- body 5beccd49 (shape e7cdf11e): 4 copies (core 2, interop 2)
   files: Notifications/client_notifications_db.py:598, Skills_Interop/local_skills_service.py:348, Skills_Interop/skill_trust_service.py:64, Widgets/Console/console_scope_picker_modal.py:237
   | def _now_iso() -> str:
   |         return datetime.now(timezone.utc).isoformat()

--- body 25a36d9a (shape 3393c16c): 2 copies (core 2, interop 0)
   files: Agents/agent_service.py:1181, DB/AgentRuns_DB.py:145
   | def _now_iso() -> str:
   |     return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

--- body 1a572270 (shape f7c435e7): 1 copies (core 1, interop 0)
   files: Agents/run_log.py:978
   | def _now_iso() -> str:
   |     from datetime import datetime, timezone

   |     return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

====================================================================================================
# _datetime_to_iso: 6 defs, 2 distinct bodies, 2 distinct shapes

--- body c38e4896 (shape 4f9cae8b): 5 copies (core 5, interop 0)
   files: MCP/local_store.py:140, MCP/server_target_store.py:372, MCP/unified_context_store.py:78, MCP/unified_control_models.py:15, runtime_policy/source_state.py:133
   | def _datetime_to_iso(value: datetime | None) -> str | None:
   |     if value is None:
   |         return None
   |     if value.tzinfo is None:
   |         value = value.replace(tzinfo=timezone.utc)
   |     return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

--- body 2f2ec7e6 (shape 2cdd1fb1): 1 copies (core 0, interop 1)
   files: UX_Interop/server_parity_contracts.py:565
   | def _datetime_to_iso(value: datetime | None) -> str | None:
   |     if value is None:
   |         return None
   |     return value.isoformat().replace("+00:00", "Z")

====================================================================================================
# _timestamp: 3 defs, 3 distinct bodies, 3 distinct shapes

--- body 5a4812d6 (shape 8d64a1a7): 1 copies (core 1, interop 0)
   files: Chunking/_template_conversion.py:303
   | def _timestamp(value: Any) -> Any:
   |     """Normalize a copied timestamp to the storage string form."""
   |     if value is None:
   |         return None
   |     if isinstance(value, str):
   |         return value
   |     return str(value)

--- body 883908ac (shape e525eddb): 1 copies (core 1, interop 0)
   files: Research_Workspace/source_operations.py:177
   | def _timestamp(value: object, field_name: str) -> str:
   |     normalized = _bounded_text(
   |         value,
   |         field_name,
   |         maximum=MAX_TIMESTAMP_CHARS,
   |         required=True,
   |     )
   |     try:
   |         parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
   |     except ValueError:
   |         raise SourceOperationValidationError(
   |             f"{field_name} must be an ISO-8601 timestamp"
   |         ) from None
   |     if parsed.tzinfo is None:
   |         raise SourceOperationValidationError(
   |             f"{field_name} must include an explicit timezone"
   |         )
   |     return normalized

--- body 352f407b (shape 01bae2df): 1 copies (core 1, interop 0)
   files: Tool_Packs/receipt_store.py:121
   | def _timestamp(value: object) -> str:
   |     text = _text(value, max_bytes=64)
   |     if not text.endswith("Z"):
   |         raise _fail("payload_invalid")
   |     try:
   |         parsed = datetime.fromisoformat(text[:-1] + "+00:00")
   |     except ValueError:
   |         raise _fail("payload_invalid") from None
   |     if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
   |         raise _fail("payload_invalid")
   |     return text

====================================================================================================
# _utcnow: 1 defs, 1 distinct bodies, 1 distinct shapes

--- body 007d065e (shape 36eb239f): 1 copies (core 1, interop 0)
   files: Evals/word_bench/capture_client.py:73
   | def _utcnow() -> str:
   |     return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

====================================================================================================
# _iso_now: 0 defs, 0 distinct bodies, 0 distinct shapes

====================================================================================================
# _now_ns: 1 defs, 1 distinct bodies, 1 distinct shapes

--- body 3eaa9e50 (shape 2f82ef83): 1 copies (core 1, interop 0)
   files: Audio/voice_turn_coordinator.py:2183
   | def _now_ns(self) -> int:
   |         value = self._scheduler.now_ns
   |         now = value() if callable(value) else value
   |         if type(now) is not int or now < 0:
   |             raise ValueError("scheduler now_ns must be a non-negative integer")
   |         return now
