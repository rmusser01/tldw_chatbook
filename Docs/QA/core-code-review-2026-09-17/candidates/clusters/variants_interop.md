
====================================================================================================
# _enforce_policy: 51 defs, 6 distinct bodies, 6 distinct shapes

--- body 188097af (shape 0ad1d672): 43 copies (core 12, interop 31)
   files: Audio_Services_Interop/audio_services_scope_service.py:127, Auth_Account_Interop/auth_account_scope_service.py:91, Character_Chat/character_persona_scope_service.py:139, Character_Chat/chat_dictionary_scope_service.py:117, Chat/chat_conversation_scope_service.py:94, Chat_Grammars_Interop/chat_grammars_scope_service.py:59, Claims_Interop/claims_scope_service.py:64, Collections_Interop/collections_feeds_scope_service.py:79 ... +35
   | def _enforce_policy(self, action_id: str) -> None:
   |         if self.policy_enforcer is None:
   |             return
   |         self.policy_enforcer.require_allowed(action_id=action_id)

--- body 58aba4ad (shape 47a7ae41): 4 copies (core 4, interop 0)
   files: Chat/chat_loop_scope_service.py:43, Outputs/server_outputs_scope_service.py:41, Sharing/server_sharing_scope_service.py:41, WebClipper/server_web_clipper_scope_service.py:43
   | def _enforce_policy(self, action_id: str) -> None:
   |         if self.policy_enforcer is not None:
   |             self.policy_enforcer.require_allowed(action_id=action_id)

--- body ffea885d (shape b3051bb8): 1 copies (core 1, interop 0)
   files: Character_Chat/character_persona_scope_service.py:126
   | def _enforce_policy(self, mode: str, action: str) -> None:
   |         if self.policy_enforcer is None:
   |             return
   |         action_prefix = self._ACTION_IDS.get(action)
   |         if action_prefix is None:
   |             return
   |         self.policy_enforcer.require_allowed(action_id=f"{action_prefix}.{mode}")

--- body 513b1489 (shape d5c6a99c): 1 copies (core 0, interop 1)
   files: Evaluations_Interop/evaluation_scope_service.py:106
   | def _enforce_policy(
   |         self, mode: EvaluationBackend, resource: str, action: str
   |     ) -> None:
   |         if self.policy_enforcer is None:
   |             return
   |         self.policy_enforcer.require_allowed(
   |             action_id=f"evaluations.{resource}.{action}.{mode.value}"
   |         )

--- body 7b549809 (shape 3c5273e1): 1 copies (core 0, interop 1)
   files: Study_Interop/study_scope_service.py:130
   | def _enforce_policy(self, mode: StudyBackend, action_id: str) -> None:
   |         if self.policy_enforcer is None:
   |             return
   |         self.policy_enforcer.require_allowed(action_id=f"{action_id}.{mode.value}")

--- body a6b78f41 (shape 88a4d9f4): 1 copies (core 1, interop 0)
   files: Subscriptions/watchlist_scope_service.py:94
   | def _enforce_policy(self, backend: WatchlistBackend, action: str) -> None:
   |         if self.policy_enforcer is None:
   |             return
   |         action_id = self._action_id(backend, action)
   |         require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
   |         require_ui_action_allowed = getattr(
   |             self.policy_enforcer, "require_ui_action_allowed", None
   |         )
   |         if callable(require_allowed):
   |             require_allowed(action_id=action_id)
   |         elif callable(require_ui_action_allowed):
   |             decision = require_ui_action_allowed(action_id=action_id)
   |             if decision is not None and getattr(decision, "allowed", True) is False:
   |                 raise PolicyDeniedError(
   |                     action_id=action_id,
   |                     reason_code=getattr(decision, "reason_code", None)
   |                     or "authority_denied",
   |                     user_message=getattr(decision, "user_message", None)
   |                     or f"{action_id} is not allowed.",
   |                     effective_source=getattr(decision, "effective_source", None)
   |                     or backend.value,
   |                     authority_owner=getattr(decision, "authority_owner", None)
   |                     or backend.value,
   |                 )

====================================================================================================
# _normalize_mode: 46 defs, 46 distinct bodies, 5 distinct shapes

--- body addff98f (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Audio_Services_Interop/audio_services_scope_service.py:103
   | def _normalize_mode(
   |         self, mode: AudioServicesBackend | str | None
   |     ) -> AudioServicesBackend:
   |         if mode is None:
   |             return AudioServicesBackend.LOCAL
   |         if isinstance(mode, AudioServicesBackend):
   |             return mode
   |         try:
   |             return AudioServicesBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid audio services backend: {mode}") from exc

--- body 0b2f024e (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Auth_Account_Interop/auth_account_scope_service.py:64
   | def _normalize_mode(
   |         self, mode: AuthAccountBackend | str | None
   |     ) -> AuthAccountBackend:
   |         if mode is None:
   |             return AuthAccountBackend.SERVER
   |         if isinstance(mode, AuthAccountBackend):
   |             return mode
   |         try:
   |             return AuthAccountBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid auth/account backend: {mode}") from exc

--- body 08dd9e60 (shape a4639b2f): 1 copies (core 1, interop 0)
   files: Character_Chat/character_persona_scope_service.py:107
   | def _normalize_mode(self, mode: str | None) -> str:
   |         normalized_mode = "local" if mode is None else str(mode).strip().lower()
   |         if normalized_mode not in {"local", "server"}:
   |             raise ValueError(
   |                 f"Invalid character/persona mode: {mode!r}. Expected 'local' or 'server'."
   |             )
   |         return normalized_mode

--- body 28ed4e97 (shape a4639b2f): 1 copies (core 1, interop 0)
   files: Character_Chat/chat_dictionary_scope_service.py:46
   | def _normalize_mode(self, mode: str | None) -> str:
   |         normalized_mode = "local" if mode is None else str(mode).strip().lower()
   |         if normalized_mode not in {"local", "server"}:
   |             raise ValueError(
   |                 f"Invalid chat dictionary mode: {mode!r}. Expected 'local' or 'server'."
   |             )
   |         return normalized_mode

--- body 2150f774 (shape 735a853e): 1 copies (core 1, interop 0)
   files: Chat/chat_conversation_scope_service.py:64
   | def _normalize_mode(mode: str | None) -> str:
   |         normalized = str(mode or "local").strip().lower()
   |         if normalized not in {"local", "server"}:
   |             raise ValueError("mode must be 'local' or 'server'")
   |         return normalized

--- body d50d0315 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Chat/chat_loop_scope_service.py:22
   | def _normalize_mode(self, mode: ChatLoopBackend | str | None) -> ChatLoopBackend:
   |         if mode is None:
   |             return ChatLoopBackend.LOCAL
   |         if isinstance(mode, ChatLoopBackend):
   |             return mode
   |         try:
   |             return ChatLoopBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid chat loop backend: {mode}") from exc

--- body 86d44337 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Chat_Grammars_Interop/chat_grammars_scope_service.py:32
   | def _normalize_mode(
   |         self, mode: ChatGrammarsBackend | str | None
   |     ) -> ChatGrammarsBackend:
   |         if mode is None:
   |             return ChatGrammarsBackend.SERVER
   |         if isinstance(mode, ChatGrammarsBackend):
   |             return mode
   |         try:
   |             return ChatGrammarsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid chat grammars backend: {mode}") from exc

--- body a7867860 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Claims_Interop/claims_scope_service.py:45
   | def _normalize_mode(self, mode: ClaimsBackend | str | None) -> ClaimsBackend:
   |         if mode is None:
   |             return ClaimsBackend.SERVER
   |         if isinstance(mode, ClaimsBackend):
   |             return mode
   |         try:
   |             return ClaimsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid claims backend: {mode}") from exc

--- body eec10d43 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Collections_Interop/collections_feeds_scope_service.py:45
   | def _normalize_mode(
   |         self, mode: CollectionsFeedsBackend | str | None
   |     ) -> CollectionsFeedsBackend:
   |         if mode is None:
   |             return CollectionsFeedsBackend.SERVER
   |         if isinstance(mode, CollectionsFeedsBackend):
   |             return mode
   |         try:
   |             return CollectionsFeedsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid collections feeds backend: {mode}") from exc

--- body 0099636d (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Companion_Interop/companion_scope_service.py:50
   | def _normalize_mode(self, mode: CompanionBackend | str | None) -> CompanionBackend:
   |         if mode is None:
   |             return CompanionBackend.SERVER
   |         if isinstance(mode, CompanionBackend):
   |             return mode
   |         try:
   |             return CompanionBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Companion backend: {mode}") from exc

--- body 2ec687e5 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Evaluations_Interop/evaluation_scope_service.py:85
   | def _normalize_mode(
   |         self, mode: EvaluationBackend | str | None
   |     ) -> EvaluationBackend:
   |         if mode is None:
   |             return EvaluationBackend.LOCAL
   |         if isinstance(mode, EvaluationBackend):
   |             return mode
   |         try:
   |             return EvaluationBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid evaluation backend: {mode}") from exc

--- body 0a53f631 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: External_Connectors_Interop/connectors_scope_service.py:34
   | def _normalize_mode(
   |         self, mode: ConnectorsBackend | str | None
   |     ) -> ConnectorsBackend:
   |         if mode is None:
   |             return ConnectorsBackend.SERVER
   |         if isinstance(mode, ConnectorsBackend):
   |             return mode
   |         try:
   |             return ConnectorsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid connectors backend: {mode}") from exc

--- body 8e67756a (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Feedback_Interop/feedback_scope_service.py:43
   | def _normalize_mode(self, mode: FeedbackBackend | str | None) -> FeedbackBackend:
   |         if mode is None:
   |             return FeedbackBackend.SERVER
   |         if isinstance(mode, FeedbackBackend):
   |             return mode
   |         try:
   |             return FeedbackBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid feedback backend: {mode}") from exc

--- body ca7337cc (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Kanban_Interop/kanban_scope_service.py:67
   | def _normalize_mode(self, mode: KanbanBackend | str | None) -> KanbanBackend:
   |         if mode is None:
   |             return KanbanBackend.SERVER
   |         if isinstance(mode, KanbanBackend):
   |             return mode
   |         try:
   |             return KanbanBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Kanban backend: {mode}") from exc

--- body d48c1237 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: LLM_Provider_Catalog/llm_provider_catalog_scope_service.py:64
   | def _normalize_mode(
   |         self, mode: LLMProviderCatalogBackend | str | None
   |     ) -> LLMProviderCatalogBackend:
   |         if mode is None:
   |             return LLMProviderCatalogBackend.LOCAL
   |         if isinstance(mode, LLMProviderCatalogBackend):
   |             return mode
   |         try:
   |             return LLMProviderCatalogBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid LLM provider catalog backend: {mode}") from exc

--- body 19a05347 (shape b77baefe): 1 copies (core 1, interop 0)
   files: Library/library_rag_state.py:805
   | def _normalize_mode(value: Any) -> str:
   |     mode = _sanitize_display_text(value, "rag", max_length=32, escape=False).lower()
   |     return mode if mode in {"rag", "search"} else "rag"

--- body 2e1f12ad (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: MCP_Governance_Interop/mcp_governance_scope_service.py:43
   | def _normalize_mode(
   |         self, mode: MCPGovernanceBackend | str | None
   |     ) -> MCPGovernanceBackend:
   |         if mode is None:
   |             return MCPGovernanceBackend.SERVER
   |         if isinstance(mode, MCPGovernanceBackend):
   |             return mode
   |         try:
   |             return MCPGovernanceBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid MCP governance backend: {mode}") from exc

--- body 586fa12a (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Media/media_reading_scope_service.py:113
   | def _normalize_mode(
   |         self, mode: MediaReadingBackend | str | None
   |     ) -> MediaReadingBackend:
   |         if mode is None:
   |             return MediaReadingBackend.LOCAL
   |         if isinstance(mode, MediaReadingBackend):
   |             return mode
   |         try:
   |             return MediaReadingBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid media backend: {mode}") from exc

--- body d60584b5 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Meetings_Interop/meetings_scope_service.py:45
   | def _normalize_mode(self, mode: MeetingsBackend | str | None) -> MeetingsBackend:
   |         if mode is None:
   |             return MeetingsBackend.SERVER
   |         if isinstance(mode, MeetingsBackend):
   |             return mode
   |         try:
   |             return MeetingsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid meetings backend: {mode}") from exc

--- body 4af49db1 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Notifications/notifications_scope_service.py:55
   | def _normalize_mode(
   |         self, mode: NotificationsBackend | str | None
   |     ) -> NotificationsBackend:
   |         if mode is None:
   |             return NotificationsBackend.SERVER
   |         if isinstance(mode, NotificationsBackend):
   |             return mode
   |         try:
   |             return NotificationsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid notifications backend: {mode}") from exc

--- body 4689e5ed (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Notifications/server_notifications_scope_service.py:28
   | def _normalize_mode(
   |         self, mode: ServerNotificationBackend | str | None
   |     ) -> ServerNotificationBackend:
   |         if mode is None:
   |             return ServerNotificationBackend.LOCAL
   |         if isinstance(mode, ServerNotificationBackend):
   |             return mode
   |         try:
   |             return ServerNotificationBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid notifications backend: {mode}") from exc

--- body 5a7b23a7 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Outputs/server_outputs_scope_service.py:22
   | def _normalize_mode(self, mode: OutputBackend | str | None) -> OutputBackend:
   |         if mode is None:
   |             return OutputBackend.LOCAL
   |         if isinstance(mode, OutputBackend):
   |             return mode
   |         try:
   |             return OutputBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid outputs backend: {mode}") from exc

--- body 90fd5687 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Outputs_Interop/outputs_scope_service.py:69
   | def _normalize_mode(self, mode: OutputsBackend | str | None) -> OutputsBackend:
   |         if mode is None:
   |             return OutputsBackend.SERVER
   |         if isinstance(mode, OutputsBackend):
   |             return mode
   |         try:
   |             return OutputsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid outputs backend: {mode}") from exc

--- body e0b8779c (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Personalization_Interop/personalization_scope_service.py:50
   | def _normalize_mode(
   |         self, mode: PersonalizationBackend | str | None
   |     ) -> PersonalizationBackend:
   |         if mode is None:
   |             return PersonalizationBackend.SERVER
   |         if isinstance(mode, PersonalizationBackend):
   |             return mode
   |         try:
   |             return PersonalizationBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Personalization backend: {mode}") from exc

--- body 2727a805 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Prompt_Management/prompt_chatbook_scope_service.py:106
   | def _normalize_mode(
   |         self, mode: PromptChatbookBackend | str | None
   |     ) -> PromptChatbookBackend:
   |         if mode is None:
   |             return PromptChatbookBackend.LOCAL
   |         if isinstance(mode, PromptChatbookBackend):
   |             return mode
   |         try:
   |             return PromptChatbookBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid prompt/chatbook backend: {mode}") from exc

--- body c94931cd (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Prompt_Management/prompt_scope_service.py:1189
   | def _normalize_mode(self, mode: PromptBackend | str | None) -> PromptBackend:
   |         if mode is None:
   |             return PromptBackend.LOCAL
   |         if isinstance(mode, PromptBackend):
   |             return mode
   |         try:
   |             return PromptBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid prompt backend: {mode}") from exc

--- body d13c9e4a (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Prompt_Studio_Interop/prompt_studio_scope_service.py:47
   | def _normalize_mode(
   |         self, mode: PromptStudioBackend | str | None
   |     ) -> PromptStudioBackend:
   |         if mode is None:
   |             return PromptStudioBackend.SERVER
   |         if isinstance(mode, PromptStudioBackend):
   |             return mode
   |         try:
   |             return PromptStudioBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Prompt Studio backend: {mode}") from exc

--- body 14cc27f6 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: RAG_Admin/rag_admin_scope_service.py:55
   | def _normalize_mode(self, mode: RAGAdminBackend | str | None) -> RAGAdminBackend:
   |         if mode is None:
   |             return RAGAdminBackend.LOCAL
   |         if isinstance(mode, RAGAdminBackend):
   |             return mode
   |         try:
   |             return RAGAdminBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid RAG admin backend: {mode}") from exc

--- body 93ee2e38 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Research_Interop/research_scope_service.py:267
   | def _normalize_mode(self, mode: ResearchBackend | str | None) -> ResearchBackend:
   |         if mode is None:
   |             return ResearchBackend.LOCAL
   |         if isinstance(mode, ResearchBackend):
   |             return mode
   |         try:
   |             return ResearchBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid research backend: {mode}") from exc

--- body 91d55e25 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Research_Interop/research_search_scope_service.py:100
   | def _normalize_mode(
   |         self, mode: ResearchSearchBackend | str | None
   |     ) -> ResearchSearchBackend:
   |         if mode is None:
   |             return ResearchSearchBackend.LOCAL
   |         if isinstance(mode, ResearchSearchBackend):
   |             return mode
   |         try:
   |             return ResearchSearchBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid research search backend: {mode}") from exc

--- body e6039dad (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Server_Runtime_Interop/server_runtime_scope_service.py:45
   | def _normalize_mode(
   |         self, mode: ServerRuntimeBackend | str | None
   |     ) -> ServerRuntimeBackend:
   |         if mode is None:
   |             return ServerRuntimeBackend.SERVER
   |         if isinstance(mode, ServerRuntimeBackend):
   |             return mode
   |         try:
   |             return ServerRuntimeBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid server-runtime backend: {mode}") from exc

--- body d8ff644f (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: Sharing/server_sharing_scope_service.py:22
   | def _normalize_mode(self, mode: SharingBackend | str | None) -> SharingBackend:
   |         if mode is None:
   |             return SharingBackend.LOCAL
   |         if isinstance(mode, SharingBackend):
   |             return mode
   |         try:
   |             return SharingBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Sharing backend: {mode}") from exc

--- body f7f4b8b7 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Sharing_Interop/sharing_scope_service.py:50
   | def _normalize_mode(self, mode: SharingBackend | str | None) -> SharingBackend:
   |         if mode is None:
   |             return SharingBackend.SERVER
   |         if isinstance(mode, SharingBackend):
   |             return mode
   |         try:
   |             return SharingBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid sharing backend: {mode}") from exc

--- body 4b55a77b (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Skills_Interop/skills_scope_service.py:54
   | def _normalize_mode(self, mode: SkillsBackend | str | None) -> SkillsBackend:
   |         if mode is None:
   |             return SkillsBackend.SERVER
   |         if isinstance(mode, SkillsBackend):
   |             return mode
   |         try:
   |             return SkillsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid skills backend: {mode}") from exc

--- body f9f1edad (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Study_Interop/study_scope_service.py:103
   | def _normalize_mode(self, mode: StudyBackend | str | None) -> StudyBackend:
   |         if mode is None:
   |             return StudyBackend.LOCAL
   |         if isinstance(mode, StudyBackend):
   |             return mode
   |         try:
   |             return StudyBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid study backend: {mode}") from exc

--- body a7c4ee1e (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Sync_Interop/sync_scope_service.py:90
   | def _normalize_mode(self, mode: SyncBackend | str | None) -> SyncBackend:
   |         if mode is None:
   |             return SyncBackend.SERVER
   |         if isinstance(mode, SyncBackend):
   |             return mode
   |         try:
   |             return SyncBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid sync backend: {mode}") from exc

--- body 16f6705f (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Text2SQL_Interop/text2sql_scope_service.py:44
   | def _normalize_mode(self, mode: Text2SQLBackend | str | None) -> Text2SQLBackend:
   |         if mode is None:
   |             return Text2SQLBackend.SERVER
   |         if isinstance(mode, Text2SQLBackend):
   |             return mode
   |         try:
   |             return Text2SQLBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Text2SQL backend: {mode}") from exc

--- body edaedc43 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Tools_Interop/tools_scope_service.py:34
   | def _normalize_mode(self, mode: ToolsBackend | str | None) -> ToolsBackend:
   |         if mode is None:
   |             return ToolsBackend.SERVER
   |         if isinstance(mode, ToolsBackend):
   |             return mode
   |         try:
   |             return ToolsBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid tools backend: {mode}") from exc

--- body bf4fc395 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Translation_Interop/translation_scope_service.py:43
   | def _normalize_mode(
   |         self, mode: TranslationBackend | str | None
   |     ) -> TranslationBackend:
   |         if mode is None:
   |             return TranslationBackend.SERVER
   |         if isinstance(mode, TranslationBackend):
   |             return mode
   |         try:
   |             return TranslationBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid translation backend: {mode}") from exc

--- body 073d291f (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: User_Governance_Interop/user_governance_scope_service.py:45
   | def _normalize_mode(
   |         self, mode: UserGovernanceBackend | str | None
   |     ) -> UserGovernanceBackend:
   |         if mode is None:
   |             return UserGovernanceBackend.SERVER
   |         if isinstance(mode, UserGovernanceBackend):
   |             return mode
   |         try:
   |             return UserGovernanceBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid user-governance backend: {mode}") from exc

--- body 30e7c3c9 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Voice_Assistant_Interop/voice_assistant_scope_service.py:68
   | def _normalize_mode(
   |         self, mode: VoiceAssistantBackend | str | None
   |     ) -> VoiceAssistantBackend:
   |         if mode is None:
   |             return VoiceAssistantBackend.SERVER
   |         if isinstance(mode, VoiceAssistantBackend):
   |             return mode
   |         try:
   |             return VoiceAssistantBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Voice Assistant backend: {mode}") from exc

--- body aeafcb18 (shape 4e6205a8): 1 copies (core 1, interop 0)
   files: WebClipper/server_web_clipper_scope_service.py:22
   | def _normalize_mode(
   |         self, mode: WebClipperBackend | str | None
   |     ) -> WebClipperBackend:
   |         if mode is None:
   |             return WebClipperBackend.LOCAL
   |         if isinstance(mode, WebClipperBackend):
   |             return mode
   |         try:
   |             return WebClipperBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid Web Clipper backend: {mode}") from exc

--- body 982c176f (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Web_Clipper_Interop/web_clipper_scope_service.py:57
   | def _normalize_mode(
   |         self, mode: WebClipperBackend | str | None
   |     ) -> WebClipperBackend:
   |         if mode is None:
   |             return WebClipperBackend.SERVER
   |         if isinstance(mode, WebClipperBackend):
   |             return mode
   |         try:
   |             return WebClipperBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid web-clipper backend: {mode}") from exc

--- body 1cbb781c (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Web_Scraping_Interop/web_scraping_scope_service.py:58
   | def _normalize_mode(
   |         self, mode: WebScrapingBackend | str | None
   |     ) -> WebScrapingBackend:
   |         if mode is None:
   |             return WebScrapingBackend.SERVER
   |         if isinstance(mode, WebScrapingBackend):
   |             return mode
   |         try:
   |             return WebScrapingBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid web-scraping backend: {mode}") from exc

--- body dc2ac0aa (shape 69daee18): 1 copies (core 1, interop 0)
   files: Widgets/Library/library_media_content.py:292
   | def _normalize_mode(self, mode: str) -> str:
   |         """Validate a mode and force non-Markdown bodies to their Raw view."""
   |         if mode not in self._VALID_MODES:
   |             raise ValueError(f"Unsupported Library media content mode: {mode!r}")
   |         return mode if self.is_markdown else "raw"

--- body d55873d6 (shape 4e6205a8): 1 copies (core 0, interop 1)
   files: Writing_Interop/writing_scope_service.py:215
   | def _normalize_mode(self, mode: WritingBackend | str | None) -> WritingBackend:
   |         if mode is None:
   |             return WritingBackend.LOCAL
   |         if isinstance(mode, WritingBackend):
   |             return mode
   |         try:
   |             return WritingBackend(str(mode))
   |         except ValueError as exc:
   |             raise ValueError(f"Invalid writing backend: {mode}") from exc

====================================================================================================
# _require_client: 47 defs, 47 distinct bodies, 1 distinct shapes

--- body 7c17e84c (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Audio_Services_Interop/server_audio_services_service.py:71
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server audio/speech operations."
   |         )

--- body bd6fc695 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Auth_Account_Interop/server_auth_account_service.py:76
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server auth/profile/account operations."
   |         )

--- body d20978cc (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Character_Chat/server_character_persona_service.py:57
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server character and persona operations."
   |         )

--- body 7327d8ed (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Character_Chat/server_chat_dictionary_service.py:48
   | def _require_client(self) -> Any:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("Server chat dictionary client is unavailable.")

--- body 29378d0f (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Chat/server_chat_conversation_service.py:58
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server chat conversation operations."
   |         )

--- body da16f583 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Chat/server_chat_loop_service.py:35
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server chat loop operations.")

--- body b67f477e (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Chat_Grammars_Interop/server_chat_grammars_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server chat grammar operations."
   |         )

--- body 81370adf (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Chatbooks/server_chatbook_service.py:226
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server chatbook operations.")

--- body df80724e (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Claims_Interop/server_claims_service.py:69
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server claims operations.")

--- body 036947b8 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Collections_Interop/server_collections_feeds_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server collections feed operations."
   |         )

--- body aa23c1b5 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Companion_Interop/server_companion_service.py:62
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server Companion operations.")

--- body e0f0e3e1 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Evaluations_Interop/server_evaluations_service.py:57
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server evaluation operations."
   |         )

--- body b0da6f88 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: External_Connectors_Interop/server_connectors_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server connector operations.")

--- body 65bc8065 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Feedback_Interop/server_feedback_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server feedback operations.")

--- body 4b170daa (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Kanban_Interop/server_kanban_service.py:654
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server Kanban operations.")

--- body 1e03e85f (shape 2cd39255): 1 copies (core 1, interop 0)
   files: LLM_Provider_Catalog/server_llm_provider_catalog_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server LLM provider/model catalog operations."
   |         )

--- body e9abd51c (shape 2cd39255): 1 copies (core 0, interop 1)
   files: MCP_Governance_Interop/server_mcp_governance_service.py:74
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server MCP governance operations."
   |         )

--- body cdbc14d8 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Media/server_media_reading_service.py:79
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server media operations.")

--- body 62fa8439 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Meetings_Interop/server_meetings_service.py:64
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server meeting operations.")

--- body d6b12f45 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Notes/server_notes_workspace_service.py:70
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server note and workspace operations."
   |         )

--- body 8a6e5c51 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Notifications/server_notifications_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server notification operations."
   |         )

--- body 1e1d5ae6 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Outputs/server_outputs_service.py:42
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server outputs operations.")

--- body 6b639ee3 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Outputs_Interop/server_outputs_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server output operations.")

--- body 80930944 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Personalization_Interop/server_personalization_service.py:62
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server Personalization operations."
   |         )

--- body 92dfe765 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Prompt_Management/prompt_scope_service.py:173
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server prompt operations.")

--- body 161716a7 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Prompt_Management/server_prompt_service.py:57
   | def _require_client(self) -> Any:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("Server prompt client is unavailable.")

--- body 272d6e50 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Prompt_Studio_Interop/server_prompt_studio_service.py:79
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server Prompt Studio operations."
   |         )

--- body 7d15e44a (shape 2cd39255): 1 copies (core 1, interop 0)
   files: RAG_Admin/server_rag_admin_service.py:60
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server retrieval-admin operations."
   |         )

--- body 3658036c (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Research_Interop/server_research_search_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server research search operations."
   |         )

--- body 15900a0f (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Research_Interop/server_research_service.py:57
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server research operations.")

--- body 3f4e7073 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Server_Runtime_Interop/server_runtime_service.py:58
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server runtime/config operations."
   |         )

--- body 62135998 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Sharing/server_sharing_service.py:42
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server Sharing operations.")

--- body e9baaaa6 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Sharing_Interop/server_sharing_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server sharing operations.")

--- body fb470ae3 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Skills_Interop/server_skills_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server skill operations.")

--- body 5a89b9c2 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Study_Interop/server_quiz_service.py:53
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server quiz operations.")

--- body 7033c480 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Study_Interop/server_study_service.py:55
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server study operations.")

--- body 29ce4812 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: Subscriptions/server_watchlists_service.py:65
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server watchlist operations.")

--- body bd3f19e4 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Sync_Interop/server_sync_service.py:180
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server sync operations.")

--- body fcda50a1 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Text2SQL_Interop/server_text2sql_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for Text2SQL operations.")

--- body 6cd61651 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Tools_Interop/server_tools_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server tool operations.")

--- body becf7e14 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Translation_Interop/server_translation_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server translation operations."
   |         )

--- body 40f5a1d6 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: User_Governance_Interop/server_user_governance_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server user-governance operations."
   |         )

--- body e59c71f7 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Voice_Assistant_Interop/server_voice_assistant_service.py:62
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server Voice Assistant operations."
   |         )

--- body c21bf544 (shape 2cd39255): 1 copies (core 1, interop 0)
   files: WebClipper/server_web_clipper_service.py:38
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server Web Clipper operations."
   |         )

--- body bf51be70 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Web_Clipper_Interop/server_web_clipper_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server web-clipper operations."
   |         )

--- body 01828581 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Web_Scraping_Interop/server_web_scraping_service.py:54
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError(
   |             "TLDW API client is required for server web-scraping operations."
   |         )

--- body 6f0cebf2 (shape 2cd39255): 1 copies (core 0, interop 1)
   files: Writing_Interop/server_writing_service.py:67
   | def _require_client(self) -> TLDWAPIClient:
   |         if self.client is not None:
   |             return self.client
   |         if self.client_provider is not None:
   |             return self.client_provider.build_client()
   |         raise ValueError("TLDW API client is required for server writing operations.")
