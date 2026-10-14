
====================================================================================================
# _set_status: 22 defs, 22 distinct bodies, 16 distinct shapes

--- body ea5978d1 (shape 9c006074): 1 copies (core 1, interop 0)
   files: Audio/diarizer_local.py:556
   | def _set_status(self, status: str) -> None:
   |         """`ensure_models`'s `progress` callback -- integers only (spec §8)."""
   |         self._status = status

--- body 44a981d2 (shape 05f4898c): 1 copies (core 1, interop 0)
   files: UI/Research_Window.py:799
   | def _set_status(self, message: str) -> None:
   |         self.status_message = message
   |         if not self.is_mounted:
   |             return
   |         try:
   |             self.query_one("#research-status", Static).update(message)
   |         except Exception:
   |             pass

--- body 3711bbe1 (shape bdbb5ec9): 1 copies (core 1, interop 0)
   files: UI/Screens/image_gen_demo_screen.py:96
   | def _set_status(self, message: str) -> None:
   |         """Update the status line. UI-thread only (call via ``call_from_thread``
   |         from a worker)."""
   |         self.query_one("#imagegen-status", Static).update(message)

--- body 0abd4ef1 (shape bdbb5ec9): 1 copies (core 1, interop 0)
   files: UI/Screens/model_remote_view.py:1145
   | def _set_status(self, message: str) -> None:
   |         self.query_one("#remote-model-status", Static).update(message)

--- body 454fb721 (shape 7031ff1c): 1 copies (core 1, interop 0)
   files: UI/Speech/speech_settings_pane.py:1255
   | def _set_status(
   |         self,
   |         copy: str,
   |         *,
   |         severity: str | None = None,
   |     ) -> None:
   |         """Update visible status and optionally use the app announcement channel."""

   |         self.query_one("#studio-tts-status", Static).update(copy)
   |         if severity is not None:
   |             self.app.notify(copy, severity=severity)

--- body 987e2311 (shape 3f9e9ba9): 1 copies (core 1, interop 0)
   files: UI/Writing_Window.py:257
   | def _set_status(self, message: str) -> None:
   |         self.status_message = message
   |         self.source_panel.set_notice(message)
   |         if not self.is_mounted:
   |             return
   |         try:
   |             self.query_one("#writing-status", Static).update(message)
   |         except Exception:
   |             pass
   |         try:
   |             self.source_panel.query_one("#writing-source-status", Static).update(
   |                 message
   |             )
   |         except Exception:
   |             pass

--- body 2eeb2fc1 (shape 5f5e38f1): 1 copies (core 1, interop 0)
   files: UI/stts_profile_library.py:2759
   | def _set_status(self, copy: str) -> None:
   |         if not self.is_mounted:
   |             return
   |         status = self.query_one("#stts-profile-status-copy", Static)
   |         status.remove_class("selected-detail")
   |         status.update(Text(copy))
   |         identifiers = self.query_one("#stts-profile-identifiers", TextArea)
   |         if identifiers.has_focus:
   |             self.query_one("#stts-profile-table", DataTable).focus()
   |         identifiers.display = False
   |         identifiers.load_text("")
   |         self._render_dependency_actions(())

--- body 3194e220 (shape 4d32d00b): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_capture_policy_dialog.py:800
   | def _set_status(self, message: str, *, error: bool = False) -> None:
   |         self.status_text = message
   |         if self.is_mounted:
   |             status = self.query_one("#capture-policy-status", Static)
   |             status.set_class(error, "-error")
   |             status.update(message)

--- body e64ea42e (shape 91c984fb): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_exchange_export_dialog.py:356
   | def _set_status(self, message: str, *, error: bool = False) -> None:
   |         status = self.query_one("#exchange-export-status", Static)
   |         status.set_class(error, "-error")
   |         status.update(message)

--- body c32ce529 (shape bdbb5ec9): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_fork_chat_modal.py:343
   | def _set_status(self, copy: str) -> None:
   |         self.query_one("#console-fork-chat-status", Static).update(copy)

--- body 0ee6c75c (shape e3e05fd5): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_provider_picker.py:279
   | def _set_status(self, copy: str) -> None:
   |         if self.is_mounted:
   |             self.query_one("#console-settings-provider-picker-status", Static).update(
   |                 copy
   |             )

--- body 5eb16a1b (shape 6abe5701): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_session_switcher_modal.py:2550
   | def _set_status(self, message: str) -> None:
   |         try:
   |             self.query_one("#console-switcher-status", Static).update(message)
   |         except NoMatches:
   |             pass
   |         try:
   |             scope = self.query_one("#console-switcher-scope", Static)
   |         except NoMatches:
   |             return
   |         if self._mode is SwitcherMode.CHARACTER_CHATS:
   |             scope.update("This profile · Local chats")
   |             self.query_one("#console-switcher-divider", Static).update(
   |                 message
   |                 if self._query_pending
   |                 or self._activation_phase is not ConsoleActivationPhase.IDLE
   |                 else "─" * 48
   |             )
   |         else:
   |             self.query_one("#console-switcher-divider", Static).update("─" * 48)
   |             scope.update(
   |                 f"Activity updates unavailable · {message}"
   |                 if self._compact_layout and self._activity_receipt_state == "degraded"
   |                 else message
   |             )

--- body faccbfd1 (shape 13dd5cc6): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_side_chat_modal.py:352
   | def _set_status(self, message: str) -> None:
   |         try:
   |             self.query_one(_STATUS, Static).update(message)
   |         except NoMatches:
   |             return

--- body 902c8a9b (shape 91c984fb): 1 copies (core 1, interop 0)
   files: Widgets/Console/trace_export_dialog.py:369
   | def _set_status(self, message: str, *, error: bool = False) -> None:
   |         status = self.query_one("#trace-export-status", Static)
   |         status.set_class(error, "-error")
   |         status.update(message)

--- body 1cd376f2 (shape a7ea69b2): 1 copies (core 1, interop 0)
   files: Widgets/Persona_Widgets/buddy_character_review.py:209
   | def _set_status(self, text: str) -> None:
   |         if self.is_mounted and not self._review_closed:
   |             self.query_one("#buddy-status", Static).update(text)

--- body f350dfca (shape 887b1d96): 1 copies (core 1, interop 0)
   files: Widgets/Persona_Widgets/personas_policy_rules_editor.py:149
   | def _set_status(self, text: str) -> None:
   |         try:
   |             self.query_one("#personas-policy-status", Static).update(text)
   |         except Exception:
   |             pass

--- body 2255792f (shape e3e05fd5): 1 copies (core 1, interop 0)
   files: Widgets/Settings_Widgets/personal_context_review_modal.py:361
   | def _set_status(self, copy: str) -> None:
   |         if self.is_mounted:
   |             self.query_one("#personal-context-proposal-status", Static).update(copy)

--- body 71fac1d0 (shape 08fbf4a9): 1 copies (core 1, interop 0)
   files: Widgets/Settings_Widgets/personal_context_review_modal.py:809
   | def _set_status(self, copy: str) -> None:
   |         self._status_copy = copy
   |         if self.is_mounted:
   |             self.query_one("#personal-context-review-status", Static).update(copy)

--- body 5eea11d0 (shape 887b1d96): 1 copies (core 1, interop 0)
   files: Widgets/Settings_Widgets/server_switch_modal.py:151
   | def _set_status(self, text: str) -> None:
   |         try:
   |             self.query_one("#server-switch-status", Static).update(text)
   |         except QueryError:
   |             pass

--- body 755c7916 (shape 24adace7): 1 copies (core 1, interop 0)
   files: Widgets/audio_troubleshooting_dialog.py:432
   | def _set_status(self, text: str, status_type: str = "ok"):
   |         """Update status display."""
   |         status_container = self.query_one("#status-container", Container)
   |         status_text = self.query_one("#status-text", Label)

   |         status_text.update(text)

   |         # Update container class
   |         status_container.remove_class("status-ok", "status-warning", "status-error")
   |         status_container.add_class(f"status-{status_type}")

--- body 785a8adf (shape ca433f24): 1 copies (core 1, interop 0)
   files: Widgets/model_search_picker.py:595
   | def _set_status(self, copy: str) -> None:
   |         try:
   |             status = self.query_one("#model-search-picker-status", Static)
   |         except NoMatches:
   |             return
   |         status.update(copy)

--- body 82e293ed (shape bdbb5ec9): 1 copies (core 1, interop 0)
   files: Widgets/settings_agents_panel.py:701
   | def _set_status(self, text: str) -> None:
   |         self.query_one("#agents-status", Static).update(text)

====================================================================================================
# _cancel: 53 defs, 13 distinct bodies, 11 distinct shapes

--- body 363b8c81 (shape c77750a4): 23 copies (core 23, interop 0)
   files: Widgets/Console/console_edit_message_modal.py:167, Widgets/Console/console_edit_message_modal.py:300, Widgets/Console/console_endpoint_template_modal.py:651, Widgets/Console/console_exchange_export_dialog.py:219, Widgets/Console/console_feedback_comment_modal.py:128, Widgets/Console/console_fork_chat_modal.py:373, Widgets/Console/console_rename_session_modal.py:87, Widgets/Console/console_session_switcher_modal.py:2520 ... +15
   | async def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         await self.request_safe_cancel(source="button")

--- body 39c74d03 (shape 10c5d09e): 8 copies (core 8, interop 0)
   files: UI/STTS_Window.py:238, UI/stts_profile_library.py:1238, UI/stts_profile_library.py:1417, Widgets/Console/console_save_markdown_modal.py:99, Widgets/Persona_Widgets/character_tts_portability_dialogs.py:99, Widgets/Persona_Widgets/conversation_attach_picker.py:116, Widgets/Persona_Widgets/dictionary_attach_picker.py:117, Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py:140
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.dismiss(None)

--- body c316ae90 (shape c77750a4): 8 copies (core 8, interop 0)
   files: UI/Screens/skills_screen.py:163, UI/Screens/skills_screen.py:282, UI/Screens/skills_screen.py:381, Widgets/Console/console_auto_speak_consent.py:155, Widgets/Console/console_model_popover.py:1226, Widgets/Console/console_summarize_preview_modal.py:127, Widgets/Library/library_note_folder_dialog.py:62, Widgets/Library/library_note_folder_dialog.py:141
   | async def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         await self.request_safe_cancel(source="visible")

--- body c9c45182 (shape 10c5d09e): 3 copies (core 3, interop 0)
   files: Widgets/Persona_Widgets/buddy_management_modal.py:428, Widgets/Persona_Widgets/dictionary_picker.py:129, Widgets/Persona_Widgets/world_book_picker.py:126
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.dismiss_safe_once(None)

--- body c7142a46 (shape 1cf03147): 2 copies (core 2, interop 0)
   files: UI/stts_profile_library.py:1301, Widgets/Persona_Widgets/character_tts_portability_dialogs.py:177
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.dismiss(False)

--- body 932d216f (shape fe3d39e3): 2 copies (core 2, interop 0)
   files: Widgets/Console/console_project_instructions.py:568, Widgets/Console/console_project_instructions.py:672
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.action_cancel()

--- body f4648c91 (shape 6d19755f): 1 copies (core 1, interop 0)
   files: Actor_Packs/importer.py:1560
   | def _cancel(checker: Callable[[], bool]) -> None:
   |     try:
   |         cancelled = checker()
   |     except Exception:
   |         cancelled = True
   |     if cancelled is True:
   |         raise ActorPackImportError("actor_pack_import_cancelled")

--- body 15df8fb7 (shape 4d5b53b3): 1 copies (core 1, interop 0)
   files: Notes/note_import_executor.py:1149
   | def _cancel(
   |         self,
   |         approval_id: str,
   |         progress_callback: Callable[[ImportExecutionProgress], None] | None,
   |     ) -> ImportExecutionReceipt:
   |         self._receipts.transition_session(approval_id, ImportSessionState.CANCELLED)
   |         receipt = self._receipts.aggregate_receipt(approval_id)
   |         _publish_receipt_progress(receipt, progress_callback)
   |         return receipt

--- body a539f620 (shape 58479005): 1 copies (core 1, interop 0)
   files: STT/dispatch_coordinator.py:509
   | def _cancel(self, capture: _Capture, force: bool) -> bool:
   |         done = None
   |         notify = False
   |         attempt_id = None
   |         with self._lock:
   |             if capture.cancelled:
   |                 return False
   |             if capture.done.is_set():
   |                 if capture.retry_buffer is None:
   |                     return False
   |                 self._mark_cancelled_locked(capture)
   |                 return True
   |             if self._reservation is not capture:
   |                 return False
   |             capture.force_cancel = force
   |             self._mark_cancelled_locked(capture)
   |             if self._active_kind == "dictation":
   |                 attempt_id = self._active_attempt_id
   |             else:
   |                 notify = self._clear_reservation_locked(capture)
   |                 done = capture.done
   |         if attempt_id is not None:
   |             method = self._executor.force_stop if force else self._executor.cancel
   |             method(attempt_id)
   |         self._post(done, notify)
   |         return True

--- body 9eb77a44 (shape fa09ba30): 1 copies (core 1, interop 0)
   files: UI/Library_Modules/library_character_repair_controller.py:499
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.controller.cancel_confirmation()
   |         self.dismiss(None)

--- body 80f28966 (shape c754053e): 1 copies (core 1, interop 0)
   files: UI/Screens/backup_restore_screen.py:2338
   | def _cancel(self):
   |         current = self.service.current()
   |         if current and current["state"] == "running":
   |             self.service.cancel(current["operation_id"])
   |             self.query_one("#backup-message", Static).update(
   |                 "Cancellation requested; waiting for a safe stopping point."
   |             )

--- body 7f9380af (shape a9a805f7): 1 copies (core 1, interop 0)
   files: UI/Speech/speech_settings_pane.py:192
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.dismiss("cancel")

--- body a313157e (shape 3513d412): 1 copies (core 1, interop 0)
   files: Widgets/Library/library_note_import_canvas.py:1359
   | def _cancel(self, event: Button.Pressed) -> None:
   |         event.stop()
   |         self.post_message(self.CancelRequested())

====================================================================================================
# _perform_safe_cancel: 45 defs, 33 distinct bodies, 28 distinct shapes

--- body 245a997e (shape e2769490): 8 copies (core 8, interop 0)
   files: UI/Research_Workspace_Modules/add_source_modal.py:240, UI/Research_Workspace_Modules/overlay_conflict_modal.py:51, UI/Research_Workspace_Modules/quick_note_modals.py:37, UI/Research_Workspace_Modules/quick_note_modals.py:66, UI/Research_Workspace_Modules/source_inspector.py:165, Widgets/Console/console_summarize_preview_modal.py:131, Widgets/ModelArtifacts/runtime_choice_modal.py:109, Widgets/modal_dismissal.py:256
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         """Dismiss safely while leaving the mounted draft untouched for recovery."""

   |         del source
   |         self.dismiss_safe_once(None)

--- body 11e3738a (shape b180627e): 6 copies (core 6, interop 0)
   files: UI/Screens/change_review_screen.py:4440, Widgets/Console/console_auto_speak_consent.py:159, Widgets/Console/console_capture_policy_dialog.py:184, Widgets/Library/library_file_notes_git_panel.py:4270, Widgets/ModelArtifacts/install_modal.py:127, Widgets/cancel_confirmation_dialog.py:101
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         self.dismiss_safe_once(False)

--- body 6961d725 (shape 3c1d5d9d): 1 copies (core 1, interop 0)
   files: UI/Library_Modules/prompt_collection_manager_modal.py:643
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._mutation_in_flight:
   |             if not self._mutation_close_rejected:
   |                 self._mutation_close_rejected = True
   |                 self._outcome = "Finish the current collection change before closing."
   |                 for outcome in self.query("#prompt-collection-manager-outcome").results(
   |                     Static
   |                 ):
   |                     outcome.update(self._outcome)
   |                     break
   |             return
   |         self._request_token += 1
   |         self.dismiss_safe_once(None)

--- body 158a4caa (shape 7d721f1e): 1 copies (core 1, interop 0)
   files: UI/Screens/profile_interview_screen.py:538
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._starting and self._session_id is None:
   |             self._cancel_after_start = True
   |             self.query_one("#profile-interview-state", Static).update(
   |                 "Cancellation pending — securely discarding the new draft."
   |             )
   |             return
   |         if self._cancel_after_start:
   |             if not self._busy:
   |                 self._discard_cancelled_start()
   |             return
   |         if self._session_id is None or self._expired_or_cleanup_pending:
   |             self.dismiss_safe_once(ProfileInterviewResult("cancelled", (), None))
   |             return
   |         if self._session is not None and self._session.status == "committed":
   |             self.dismiss_safe_once(
   |                 ProfileInterviewResult(
   |                     "committed",
   |                     self._session.committed_record_ids,
   |                     self._committed_runtime_enabled(self._session),
   |                 )
   |             )
   |             return
   |         if self._session is not None and self._session.status == "committing":
   |             self.dismiss_safe_once(ProfileInterviewResult("commit_unknown", (), None))
   |             return
   |         # Defer until this screen's one-shot cancellation request has closed;
   |    ...

--- body 2ee9608d (shape def8c7cb): 1 copies (core 1, interop 0)
   files: UI/Screens/video_player_screen.py:446
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         self.action_close_player()

--- body 2c0473ad (shape b33815ce): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_appearance_picker_modal.py:387
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source

   |         async def cancel_filter_timer() -> None:
   |             self._cancel_filter_timer()

   |         await self.run_cancel_effect_once(cancel_filter_timer)
   |         self.dismiss_safe_once(None)

--- body 77ca0e26 (shape 23273058): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_capture_policy_dialog.py:807
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._applying:
   |             self._set_status("Applying")
   |             return
   |         self.dismiss_safe_once(None)

--- body 30a1d8e2 (shape 8e2cab20): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_capture_policy_dialog.py:1036
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if not self._applying:
   |             self.dismiss_safe_once(None)

--- body 98bef274 (shape d25dbffd): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_character_picker_modal.py:347
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source

   |         async def cancel_debounce() -> None:
   |             self._cancel_query_debounce()

   |         await self.run_cancel_effect_once(cancel_debounce)
   |         self.dismiss_safe_once(None)

--- body f775e631 (shape 935c219a): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_citation_sources_modal.py:296
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source

   |         async def invalidate_requests() -> None:
   |             self._request_generation += 1

   |         await self.run_cancel_effect_once(invalidate_requests)
   |         self.dismiss_safe_once(None)

--- body b642fbee (shape 23273058): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_exchange_export_dialog.py:361
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._exporting:
   |             self._set_status("Export is finishing; the destination remains protected.")
   |             return
   |         self.dismiss_safe_once(None)

--- body 43bd964b (shape 97c5b002): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_fork_chat_modal.py:397
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         if source == "escape" and self._disclosure_open:
   |             disclosure = self.query_one("#console-fork-chat-disclosure", Button)
   |             self._disclosure_open = False
   |             self.query_one("#console-fork-chat-exclusions", Static).display = False
   |             disclosure.label = "What is not copied"
   |             disclosure.focus()
   |             return
   |         if self.state == "committing":
   |             self._set_status(
   |                 "Fork creation is finishing and can no longer be cancelled."
   |             )
   |             self.query_one("#console-fork-chat-status", Static).focus()
   |             return
   |         if self._on_cancel is not None:

   |             async def cancel_once() -> None:
   |                 result = self._on_cancel()
   |                 if inspect.isawaitable(result):
   |                     await result

   |             await self.run_cancel_effect_once(cancel_once)
   |         panel = self.query_one(self.SAFE_MODAL_CONTENT)
   |         if panel.is_attached:
   |             await panel.remove()
   |         self.dismiss_safe_once(None)

--- body 83d0368d (shape bf19f626): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_library_access_modal.py:448
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._dirty:
   |             self._set_feedback(
   |                 "Unsaved changes. Save them or choose Discard changes.",
   |                 focus=True,
   |             )
   |             self.query_one("#library-access-discard", Button).display = True
   |             # TASK-25824: with Discard on screen, "Cancel" stops meaning
   |             # "abandon my edits" and starts meaning "stay here" -- the same
   |             # word for two different outcomes. Name the outcome instead, so
   |             # the pair reads Keep editing / Discard changes.
   |             self.query_one("#library-access-cancel", Button).label = "Keep editing"
   |             return
   |         self.dismiss_safe_once(None)

--- body 0a5fbb06 (shape a872e603): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_project_instructions.py:558
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         self.dismiss_safe_once(ProjectInstructionSetupResult("cancel"))

--- body 1f6be496 (shape e447b09b): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_project_instructions.py:662
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         self.dismiss_safe_once("cancel")

--- body 5c6e4a01 (shape a4212357): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_prompts_modal.py:1671
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         guard = self.query_one("#console-prompts-dirty-guard", Vertical)
   |         if guard.display:
   |             if source == "escape":
   |                 self._keep_editing()
   |             return
   |         self._request_close()

--- body 3e786df1 (shape fc68fb67): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_reaction_picker_modal.py:702
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source

   |         async def cancel_pending_updates() -> None:
   |             self._cancel_pending_updates()

   |         await self.run_cancel_effect_once(cancel_pending_updates)
   |         self.dismiss_safe_once(None)

--- body c2a20098 (shape 28b05ca9): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_review_notes_modal.py:379
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._editing_id is not None:
   |             self._cancel_edit(self._editing_id)
   |             return
   |         self.dismiss_safe_once(self._changed)

--- body 5e2ff7ed (shape 9537dd3d): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_session_switcher_modal.py:2505
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._activation_phase is ConsoleActivationPhase.OPENING_CANCELLABLE:
   |             cancellation = self._activation_cancellation
   |             if cancellation is not None:
   |                 cancellation.set()
   |             self._set_status("Cancelling…")
   |             return
   |         if self._activation_phase is ConsoleActivationPhase.COMMITTING:
   |             return
   |         self._request_generation += 1
   |         self._cancel_query_debounce()
   |         self.dismiss_safe_once(None)

--- body 6b381ecb (shape def8c7cb): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_settings_modal.py:3970
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         self._request_settings_close()

--- body 31f69249 (shape ec36841e): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_side_chat_modal.py:300
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         await self.run_cancel_effect_once(self._cancel_sidechat_worker)
   |         self.dismiss_safe_once(None)

--- body cf3d9ba6 (shape d25dbffd): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_style_picker_modal.py:222
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source

   |         async def cancel_debounce() -> None:
   |             self._cancel_search_debounce()

   |         await self.run_cancel_effect_once(cancel_debounce)
   |         self.dismiss_safe_once(None)

--- body e4ddd5ff (shape 0ca9191b): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_video_capacity_modal.py:164
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         """Require explicit confirmation before a generic discard request."""
   |         del source
   |         if (
   |             self._discard_confirmation_open
   |             or not self.is_mounted
   |             or self.app.screen is not self
   |         ):
   |             return
   |         generation = self._safe_mount_generation
   |         guard = CancelConfirmationDialog(
   |             title="Discard generated video?",
   |             message=(
   |                 "Discard this generated video? The generated result will be "
   |                 "lost and cannot be recovered."
   |             ),
   |             confirm_text="Discard",
   |             cancel_text="Continue",
   |         )
   |         if not self.is_mounted or self.app.screen is not self:
   |             return
   |         self._discard_confirmation_open = True
   |         self._discard_confirmation_guard = guard
   |         self._discard_confirmation_generation = generation
   |         self.app.push_screen(
   |             guard,
   |             callback=partial(
   |                 self._apply_discard_confirmation,
   |    ...

--- body 7101bf8b (shape 8599b34b): 1 copies (core 1, interop 0)
   files: Widgets/Console/console_workspace_files_modal.py:1128
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         await self.run_cancel_effect_once(self._teardown)
   |         if self.dismiss_safe_once(None) and self._on_back_to_console is not None:
   |             self._on_back_to_console()

--- body cfd13047 (shape 23273058): 1 copies (core 1, interop 0)
   files: Widgets/Console/trace_export_dialog.py:374
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._writing:
   |             self._set_status("Export is finishing; the destination is still protected.")
   |             return
   |         self.dismiss_safe_once(None)

--- body 457efa0c (shape 60643d2f): 1 copies (core 1, interop 0)
   files: Widgets/Library/prompt_delete_confirmation_modal.py:197
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         """Dismiss safely, preserving the captured stale-result fingerprint."""
   |         del source
   |         self.dismiss_safe_once(PromptDeleteDecision(False, self.request.fingerprint))

--- body 62189006 (shape e65e9c74): 1 copies (core 1, interop 0)
   files: Widgets/Persona_Widgets/buddy_character_review.py:445
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         """Cancel preparation freely; wait for atomic publication once admitted."""
   |         if not self._publishing:
   |             self._review_closed = True
   |             self.dismiss_safe_once(
   |                 BuddyCharacterCreated(self._result, self._created_name, False)
   |                 if self._result
   |                 else None
   |             )

--- body c2d52ce3 (shape e96e6a44): 1 copies (core 1, interop 0)
   files: Widgets/Persona_Widgets/petdex_import_review.py:520
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         self._review_closed = True
   |         self._cancel_event.set()
   |         self.dismiss_safe_once(None)

--- body bedd4eab (shape 8e2cab20): 1 copies (core 1, interop 0)
   files: Widgets/Settings_Widgets/personal_context_review_modal.py:356
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if not self._busy:
   |             self.dismiss_safe_once(None)

--- body 2d023783 (shape e8ace1a1): 1 copies (core 1, interop 0)
   files: Widgets/Settings_Widgets/personal_context_review_modal.py:739
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         del source
   |         if self._busy:
   |             return
   |         if self._commit_unknown:
   |             self.dismiss_safe_once(ReviewCommitUnknownResult())
   |             return
   |         result = (
   |             None
   |             if self._receipt is None
   |             else ReviewCommitResult(self._receipt, self._enable_runtime)
   |         )
   |         self.dismiss_safe_once(result)

--- body 5cb380b7 (shape 1bf203dd): 1 copies (core 1, interop 0)
   files: Widgets/confirmation_dialog.py:148
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         """Run the existing callback once, then return the exact cancel value."""
   |         del source
   |         self.result = False
   |         if self.cancel_callback:
   |             await self.run_cancel_effect_once(self.cancel_callback)
   |         self.dismiss_safe_once(False)

--- body c9fa6448 (shape 08344b08): 1 copies (core 1, interop 0)
   files: Widgets/project_skills_import_modal.py:273
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         # Escape / backdrop-click / "Not now" all route here.
   |         del source
   |         if self._import_in_flight():
   |             return
   |         if self._outcomes is not None:
   |             # Results phase (minor 6): escape/backdrop must dismiss exactly
   |             # like "Close" -- the import already ran, there is nothing left
   |             # to cancel, and mislabeling it "not_now" would make the
   |             # already-completed import look declined to the ledger.
   |             self.dismiss_safe_once(("imported", self._outcomes))
   |             return
   |         self.dismiss_safe_once(("not_now", None))

--- body e08ff1ec (shape f7804c54): 1 copies (core 1, interop 0)
   files: Widgets/workspace_create_modal.py:282
   | async def _perform_safe_cancel(self, *, source: str) -> None:
   |         """Route Cancel/Escape/backdrop to a partial result once created.

   |         Finding 7: a workspace that already exists (folder-binding retry in
   |         progress) is a fact, not something Cancel can undo -- deliver the
   |         current state as a result instead of ``None`` so callers still sync
   |         their workspace list/active-workspace UI.
   |         """
   |         if self._created_workspace_id is None:
   |             await super()._perform_safe_cancel(source=source)
   |             return
   |         self.dismiss_safe_once(
   |             WorkspaceCreateResult(
   |                 workspace_id=self._created_workspace_id,
   |                 name=self._created_workspace_name,
   |                 bound_folders=self._bound_folders,
   |                 failed_folders=tuple(
   |                     (folder, self._failed_folder_messages.get(folder, ""))
   |                     for folder in self._folders
   |                 ),
   |                 make_active=self._make_active_result,
   |                 offer_profile_interview=self._offer_profile_interview_result,
   |                 project_skills=self._project_skills_for(self._bound_folders),
   |             )
   |         )

====================================================================================================
# _toast: 2 defs, 1 distinct bodies, 1 distinct shapes

--- body 65d1d81e (shape 92aa0bad): 2 copies (core 2, interop 0)
   files: UI/MCP_Modules/mcp_inspector.py:403, UI/MCP_Modules/mcp_workbench.py:199
   | def _toast(text: str) -> str:
   |     """Escape a `notify()`-bound message before Rich's markup interpreter
   |     sees it.

   |     A small, separate copy of `mcp_workbench.py`'s own `_toast()` --
   |     `mcp_workbench.py` already imports FROM this module (`_ORIGIN_
   |     SENTENCES`, `MCPInspector`); importing back the other way would create
   |     the exact import cycle PR-T2 shipped a real regression from.
   |     """
   |     return escape_markup(text)

====================================================================================================
# _next_request_token: 4 defs, 1 distinct bodies, 1 distinct shapes

--- body ab28630a (shape b298e5b9): 4 copies (core 4, interop 0)
   files: UI/Library_Modules/library_prompt_browse_controller.py:137, UI/Library_Modules/library_skills_browse_controller.py:185, UI/Library_Modules/prompt_collections.py:157, UI/Library_Modules/prompt_history.py:79
   | def _next_request_token(self) -> int:
   |         self._request_counter += 1
   |         return self._request_counter

====================================================================================================
# _cancel_safe: 1 defs, 1 distinct bodies, 1 distinct shapes

--- body c316ae90 (shape c77750a4): 1 copies (core 1, interop 0)
   files: Widgets/enhanced_file_picker.py:1721
   | async def _cancel_safe(self, event: Button.Pressed) -> None:
   |         """Route the visible Cancel button through terminal safe dismissal."""
   |         event.stop()
   |         await self.request_safe_cancel(source="visible")
