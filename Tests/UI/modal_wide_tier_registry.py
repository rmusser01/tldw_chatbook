"""Shared registry of modal surfaces opted into the responsive wide tier.

Single source of truth for three consumers:

* the coverage contract test (``test_modal_wide_tier.py``) -- every anchor
  below must have an ``App.-wide-viewport`` rule in the built bundle, and
  every rule in that block must have an entry here;
* the spot geometry tests -- the ``WIDE_TIER_CAPS`` tier representatives;
* the QA visual harness -- mount every surface, capture narrow/wide SVGs.

Source: ``.superpowers/modal-inventory.md`` (2026-09-13), with every
anchor re-verified against the tree on 2026-09-19 (ids drifted since the
inventory: renamed ids, files moved into ``Widgets/Library`` /
``Widgets/Persona_Widgets`` / ``Widgets/Note_Widgets``, and several
``_console_panels`` rules relocated to ``_console`` / ``_agentic_terminal``).
Surfaces marked ``skip`` in that inventory (tiny-by-design confirms, the
password dialog, the fixed-column voice blender, receipts) are
deliberately absent, as is the Console session switcher: its
``_sync_modal_max_height`` sets ``width`` imperatively via
``set_styles`` (inline styles outrank every CSS rule), so it joins the
inventory's imperative-geometry exclusions. The Conversation settings
modal keeps its own shipped ``196`` cap and Python toggle (PR #2670);
the Alt+M model popover keeps its own shipped per-surface tier
(PR #2672).

Each entry: ``(anchor, cap, owner_module)``.

* ``anchor`` is the selector fragment appended to ``App.-wide-viewport ``.
  Ids and classes work anywhere the surface mounts. Surfaces whose root
  container is anonymous use a type-scoped selector (``X > Vertical`` /
  ``X #id`` / ``X .class``) -- descendant selectors from ``App`` reach
  them all, and the type prefix keeps generic ids (``#dialog``,
  ``.dialog-container``) from leaking to other widgets.
* ``cap`` is the ``max-width`` (columns) at the wide tier; base tier
  geometry is untouched. Ladder: base width >= 104 -> 170, 84-96 -> 150,
  <= 80 -> 120 (settings modal's 196 stays as shipped).
* ``owner_module`` is the tldw_chatbook module whose Python (or owning
  sheet) defines the anchor, so the contract can fail loudly when an id
  is renamed without updating the tier.
"""

from __future__ import annotations

#: Viewport width (columns) at which the app gains the ``-wide-viewport``
#: class (mirrors the shipped Conversation settings / model popover tier).
WIDE_VIEWPORT_COLUMNS = 150

#: Wide-tier width as a fraction of the viewport (token ``$ds-percent-85``).
WIDE_TIER_WIDTH_PERCENT = 85

#: (anchor, cap, owner_module) -- see module docstring. Sorted by cap desc,
#: then anchor, so the generated CSS block and the tables read top-down
#: from biggest cap to smallest.
MODAL_WIDE_TIER: tuple[tuple[str, int, str], ...] = (
    # cap 196 -- the shipped Conversation settings tier, unchanged (PR #2670)
    ("#console-settings-modal", 196, "Widgets/Console/console_settings_modal.py"),
    # cap 170 -- base width >= 104
    ("#console-inspector-modal", 170, "Widgets/Console/console_conversation_inspector.py"),
    ("#console-workspace-files-modal", 170, "css/features/_console.tcss"),
    ("#file-notes-conflict-dialog", 170, "Widgets/Library/library_file_notes_workspace.py"),
    ("#console-run-log-modal", 170, "css/features/_console_panels.tcss"),
    ("#console-citation-sources-modal", 170, "css/features/_console.tcss"),
    ("#console-side-chat-modal", 170, "Widgets/Console/console_side_chat_modal.py"),
    ("#console-edit-message-modal", 170, "Widgets/Console/console_edit_message_modal.py"),
    ("#console-edit-thinking-modal", 170, "Widgets/Console/console_edit_message_modal.py"),
    ("#console-prompts-modal", 170, "Widgets/Console/console_prompts_modal.py"),
    ("#agent-history-dialog", 170, "css/components/_agentic_terminal.tcss"),
    ("#agent-progress-dialog", 170, "css/components/_agentic_terminal.tcss"),
    ("#worktree-recovery-dialog", 170, "css/features/_console_panels.tcss"),
    ("#internal-prompt-editor-modal", 170, "Widgets/settings_internal_prompts_editor_modal.py"),
    # cap 150 -- base width 84-96
    ("#personal-context-review-modal", 150, "css/components/_profile_interview.tcss"),
    ("#console-scope-picker-modal", 150, "css/features/_console_panels.tcss"),
    ("#console-prompt-comparison-modal", 150, "Widgets/Console/console_prompt_comparison_modal.py"),
    ("#buddy-conversation", 150, "Widgets/Persona_Widgets/buddy_conversation_modal.py"),
    ("#actor-pack-import-review", 150, "Widgets/Persona_Widgets/actor_pack_import_review.py"),
    ("#personal-context-link-modal", 150, "css/components/_profile_interview.tcss"),
    ("#console-system-prompt-modal", 150, "css/features/_console_panels.tcss"),
    ("#server-switch-modal", 150, "Widgets/Settings_Widgets/server_switch_modal.py"),
    ("#buddy-inbox", 150, "Widgets/Persona_Widgets/buddy_workspace_modal.py"),
    ("FileExtractionDialog > Vertical", 150, "Widgets/file_extraction_dialog.py"),
    ("EmojiPickerScreen #dialog", 150, "Widgets/emoji_picker.py"),
    # cap 120 -- base width <= 80
    ("NoteCreationModal > Container", 120, "Widgets/Note_Widgets/note_creation_modal.py"),
    ("#conversation-selection-container", 120, "Widgets/conversation_selection_dialog.py"),
    ("#note-selection-container", 120, "Widgets/Note_Widgets/note_selection_dialog.py"),
    ("AudioTroubleshootingDialog .dialog-container", 120, "Widgets/audio_troubleshooting_dialog.py"),
    ("#file-notes-push-auth-dialog", 120, "Widgets/Library/library_file_notes_git_panel.py"),
    ("#trace-export-dialog", 120, "Widgets/Console/trace_export_dialog.py"),
    ("#console-library-access", 120, "css/features/_console_panels.tcss"),
    ("#capture-policy-dialog", 120, "Widgets/Console/console_capture_policy_dialog.py"),
    ("#trace-privacy-dialog", 120, "Widgets/Console/console_capture_policy_dialog.py"),
    ("#exchange-export-dialog", 120, "css/components/_agentic_terminal.tcss"),
    ("#console-prompt-queue-dialog", 120, "Widgets/Console/console_prompt_queue_modal.py"),
    ("#console-reaction-picker-modal", 120, "Widgets/Console/console_reaction_picker_modal.py"),
    ("#console-review-notes-modal", 120, "Widgets/Console/console_review_notes_modal.py"),
    ("#console-endpoint-template-modal", 120, "Widgets/Console/console_endpoint_template_modal.py"),
    ("#console-rewind-modal", 120, "Widgets/Console/console_rewind_modal.py"),
    ("#video-capacity-dialog", 120, "Widgets/Console/console_video_capacity_modal.py"),
    ("#project-skills-modal", 120, "Widgets/project_skills_import_modal.py"),
    ("#console-prompt-picker-modal", 120, "css/features/_console_panels.tcss"),
    ("#console-style-picker-modal", 120, "css/features/_console.tcss"),
    (".model-install-modal", 120, "Widgets/ModelArtifacts/install_modal.py"),
    (".local-gguf-import-modal", 120, "Widgets/ModelArtifacts/local_gguf_import.py"),
    ("#file-notes-push-endpoint-details-dialog", 120, "Widgets/Library/library_file_notes_git_panel.py"),
    ("#file-notes-root-details-dialog", 120, "Widgets/Library/library_file_notes_workspace.py"),
    ("#console-project-setup-modal", 120, "Widgets/Console/console_project_instructions.py"),
    ("#console-fork-chat-modal", 120, "Widgets/Console/console_fork_chat_modal.py"),
    ("#console-character-picker", 120, "Widgets/Console/console_character_picker_modal.py"),
    ("#console-save-as-modal", 120, "Widgets/Console/console_save_as_modal.py"),
    ("#console-terminal-session-modal", 120, "Widgets/Console/console_terminal_session_modal.py"),
    ("#global-full-confirmation", 120, "Widgets/Console/console_capture_policy_dialog.py"),
    ("#console-generate-image-modal", 120, "Widgets/Console/console_generate_image_modal.py"),
    ("EncryptionSetupDialog > Container", 120, "Widgets/password_dialog.py"),
    ("#character-tts-collision-dialog", 120, "Widgets/Persona_Widgets/character_tts_portability_dialogs.py"),
    ("#character-tts-existing-dialog", 120, "Widgets/Persona_Widgets/character_tts_portability_dialogs.py"),
    (".settings-speech-credential-modal", 120, "Widgets/Settings_Widgets/speech_tts_settings_panel.py"),
    ("DocumentGenerationModal > Container", 120, "Widgets/document_generation_modal.py"),
    ("#console-auto-speak-consent-modal", 120, "Widgets/Console/console_auto_speak_consent.py"),
    ("#console-rag-settings", 120, "Widgets/Console/console_rag_settings_modal.py"),
    ("#console-summarize-preview-modal", 120, "Widgets/Console/console_summarize_preview_modal.py"),
    ("#console-workspace-switcher-modal", 120, "Widgets/Console/console_workspace_switcher_modal.py"),
    ("#prompt-delete-modal", 120, "Widgets/Library/prompt_delete_confirmation_modal.py"),
    ("#persona-visual-custom-dialog", 120, "Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py"),
    ("#console-save-markdown-box", 120, "Widgets/Console/console_save_markdown_modal.py"),
    ("#profile-interview-cancel-modal", 120, "css/components/_profile_interview.tcss"),
    ("#dictionary-picker-dialog", 120, "Widgets/Persona_Widgets/dictionary_picker.py"),
    ("#world-book-picker-dialog", 120, "Widgets/Persona_Widgets/world_book_picker.py"),
    ("DictionaryAttachPicker > Vertical", 120, "Widgets/Persona_Widgets/dictionary_attach_picker.py"),
    ("ConversationAttachPicker > Vertical", 120, "Widgets/Persona_Widgets/conversation_attach_picker.py"),
    ("#profile-dialog-container", 120, "Widgets/voice_profile_dialog.py"),
    ("FeedbackDialog > Container", 120, "Widgets/feedback_dialog.py"),
    ("#console-composer-menu", 120, "Widgets/Console/console_composer_menu_modal.py"),
    ("#prompt-variables-dialog", 120, "Widgets/Console/prompt_variables_dialog.py"),
)

#: Cap -> one representative anchor pinned by a live geometry test.
WIDE_TIER_CAPS: dict[int, str] = {
    170: "#console-prompts-modal",
    150: "#console-system-prompt-modal",
    120: "#console-reaction-picker-modal",
}
