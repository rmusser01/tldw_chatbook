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

The reverse direction is pinned too (PR #2742 review): the coverage
contract sweeps the tree for ``ModalScreen`` subclasses and fails on any
class that is neither reachable from an anchor below nor listed in
``MODAL_WIDE_TIER_SKIPPED`` / ``MODAL_WIDE_TIER_SKIPPED_MODULES`` -- the
review found eight substantial modals the 2026-09-13 inventory never
scoped, so "not in the inventory" is no longer a silent state. Vendored
``Third_Party`` code is out of scope.
"""

from __future__ import annotations

from tldw_chatbook.Constants import WIDE_VIEWPORT_COLUMNS as WIDE_VIEWPORT_COLUMNS

#: ModalScreen subclasses deliberately outside the wide tier, each with its
#: reason. Sourced from the modal inventory's skip column and the rollout
#: exclusions, plus individually verified stragglers; the reverse-sweep
#: contract also fails on stale entries (a skip for a class that no longer
#: exists or is no longer a modal).
MODAL_WIDE_TIER_SKIPPED: dict[str, str] = {
    # -- inventory skip column (tiny-by-design / judgment) ------------------
    "PasswordDialog": "inventory #72: single-input dialog, scaling adds nothing",
    "VoiceBlendDialog": "inventory #74: fixed 15-column slider grid, extra width is dead space",
    "ConfirmationDialog": "inventory #76: tiny-by-design confirm",
    "UnsavedChangesDialog": "tiny-by-design confirm (ConfirmationDialog family)",
    "CancelConfirmationDialog": "inventory #77: tiny-by-design confirm",
    "RecoveryPassphraseDialog": "inventory #78: tiny passphrase entry",
    "RagProfileNameModal": "inventory #79: tiny single-input (settings sheet geometry)",
    "RagProfileSwitchConfirmModal": "inventory #79: tiny confirm (settings sheet geometry)",
    "WorkspaceArchiveReceiptModal": "inventory #80: tiny receipt",
    "ConsoleRenameSessionModal": "inventory #82: tiny single-input rename",
    "ConsoleFeedbackCommentModal": "inventory #83: tiny comment box",
    "LibraryNoteFolderNameDialog": "inventory #84: tiny name dialog",
    "ConsoleWorkspaceRenameModal": "inventory #85: tiny rename dialog",
    "TagFilterPicker": "inventory #86: tiny tag list",
    # -- inventory section B (special surfaces) + rollout exclusions --------
    "ConsoleModelPopover": "ships its own per-surface wide tier (PR #2672)",
    "ConsoleSessionSwitcherModal": "imperative inline width via set_styles outranks every CSS rule",
    "ConsoleImageViewerModal": "content-fit auto geometry, nothing to scale",
    "ProjectInstructionNoticeModal": "notice-style modal, inventory section-B exclusion",
    "DeleteConfirmDialog": "tiny media delete confirm (inventory section B)",
    # -- stragglers verified against the tree 2026-09-19 (not inventoried) --
    "DeleteConfirmationDialog": "tiny delete confirm (base width 70)",
    "ManagedGGUFRuntimeChoiceModal": "compact runtime choice dialog (base width 64)",
    "LibraryNoteFolderTargetDialog": "compact folder target picker (base width 64)",
    "LibraryReviewSetPickerDialog": "compact review-set picker (base width 72)",
    "ConsoleAppearancePickerModal": "compact theme swatch picker (base width 64)",
    "SessionGitTrustDialog": "no pinned width; default modal geometry",
    "_GlobalSpeechTTSLeaveModal": "tiny leave confirmation",
    "ConfirmDialog": "tiny confirm (Voice Cloning window)",
    "ConfirmDisableDialog": "tiny confirm (deprecated legacy settings window)",
    "_VllmProfileDeleteConfirmationDialog": "tiny delete confirm (LLM screen)",
    "_SettlingGuardedConfirmationDialog": "tiny wizard confirmation",
    # -- wave 2 (2026-09-19): skills_screen left the module skip list when
    # SkillRecoveryReviewModal joined the tier; the passphrase dialogs stay
    # tiny-by-design (PasswordDialog family, base width 64) --
    "SkillTrustPassphraseModal": "single-input passphrase dialog (base width 64), PasswordDialog family",
    "SkillTrustBootstrapModal": "two-input passphrase bootstrap dialog (base width 64), PasswordDialog family",
}

#: Whole modules whose ModalScreen subclasses are all outside the tier:
#: subsystems the 2026-09-13 inventory never scoped. Entries marked
#: "follow-up wave" carry fixed base caps that the ladder would relax --
#: they are documented gaps queued for a second wide-tier pass, not verdicts
#: that they never need scaling.
MODAL_WIDE_TIER_SKIPPED_MODULES: dict[str, str] = {
    "UI/Chunking_Lab_Modules/dialogs.py": "chunking-lab dev dialogs -- follow-up wave",
    "UI/CodeRepoCopyPasteWindow.py": "full-viewport paste surface, no width rule",
    "UI/Library_Modules/library_character_repair_controller.py": "repair dialog (base width 76) -- follow-up wave",
    "UI/Library_Modules/skill_import_choice_modal.py": "choice dialog (base width 88) -- follow-up wave",
    "UI/Navigation/character_conversation_navigation.py": "navigation dialogs (base 72) -- follow-up wave",
    "UI/Navigation/nav_overflow_menu.py": "fixed-column overflow menu",
    "UI/Research_Workspace_Modules/add_source_modal.py": "Research workspace subsystem -- follow-up wave",
    "UI/Research_Workspace_Modules/overlay_conflict_modal.py": "Research workspace subsystem -- follow-up wave",
    "UI/Research_Workspace_Modules/quick_note_modals.py": "Research workspace subsystem -- follow-up wave",
    "UI/Research_Workspace_Modules/source_inspector.py": "Research workspace subsystem -- follow-up wave",
    "UI/Screens/artifact_share_dialog.py": "share dialog (base width 76) -- follow-up wave",
    "UI/Screens/change_review_screen.py": "change-review modals -- follow-up wave",
    "UI/Screens/model_catalog_consent.py": "consent dialog (base width 72) -- follow-up wave",
    "UI/Screens/profile_interview_screen.py": "full-viewport interview wizard screen",
    "UI/Screens/scheduling/forms/automation_definition_form.py": "scheduling form (base width 84) -- follow-up wave",
    "UI/Screens/scheduling/forms/new_task_choice_modal.py": "choice modal (base width 64) -- follow-up wave",
    "UI/Screens/scheduling/forms/reminder_form.py": "scheduling form (base width 80) -- follow-up wave",
    "UI/Screens/trajectory_screen.py": "full-viewport trajectory viewer",
    "UI/Screens/video_player_screen.py": "full-viewport video player",
    "UI/Speech/speech_settings_pane.py": "tiny leave confirmation",
    "UI/Watchlists_Modules/briefing_preset_modal.py": "Watchlists subsystem -- follow-up wave",
    "UI/Watchlists_Modules/bulk_sources_modal.py": "Watchlists subsystem -- follow-up wave",
    "UI/Watchlists_Modules/kept_briefings_modal.py": "Watchlists subsystem -- follow-up wave",
    "UI/Watchlists_Modules/opml_dialogs.py": "Watchlists subsystem -- follow-up wave",
    "UI/Watchlists_Modules/snapshot_view_modal.py": "Watchlists subsystem -- follow-up wave",
    "UI/Widgets/trace_filter_bar.py": "compact filter dialog (base width 58)",
    "UI/Workflows_Modules/library.py": "workflow choice modals, no pinned widths -- follow-up wave",
    "UI/Workflows_Modules/reference_picker.py": "workflow picker, no pinned width -- follow-up wave",
    "UI/Workbench/help.py": "help panel (base width 76) -- follow-up wave",
    "UI/Wizards/FirstRunSetupWizard.py": "tiny wizard confirmation",
    "UI/Wizards/first_run_recovery_dialog.py": "recovery dialog (base width 72) -- follow-up wave",
    "UI/stts_profile_library.py": "STTS profile library modals (base caps 64-76) -- follow-up wave",
    "Widgets/workspace_create_modal.py": "workspace create form (base width 72) -- follow-up wave",
    "Widgets/workspace_persona_default.py": "assistant-defaults form (base width 68) -- follow-up wave",
}

#: Viewport width (columns) at which the app gains the ``-wide-viewport``
#: class -- re-exported from ``tldw_chatbook.Constants`` so the production
#: breakpoint (``WideViewportTierMixin``) and this contract share one value
#: (PR #2742 review: the duplicated literals could drift).

#: Wide-tier width as a fraction of the viewport (token ``$ds-percent-85``).
WIDE_TIER_WIDTH_PERCENT = 85

#: (anchor, cap, owner_module) -- see module docstring. Sorted by cap desc,
#: then anchor, so the generated CSS block and the tables read top-down
#: from biggest cap to smallest.
MODAL_WIDE_TIER: tuple[tuple[str, int, str], ...] = (
    # cap 196 -- the shipped Conversation settings tier, unchanged (PR #2670)
    ("#console-settings-modal", 196, "Widgets/Console/console_settings_modal.py"),
    # cap 170 -- base width >= 104
    (
        "#console-inspector-modal",
        170,
        "Widgets/Console/console_conversation_inspector.py",
    ),
    ("#console-workspace-files-modal", 170, "css/features/_console.tcss"),
    (
        "#file-notes-conflict-dialog",
        170,
        "Widgets/Library/library_file_notes_workspace.py",
    ),
    ("#console-run-log-modal", 170, "css/features/_console_panels.tcss"),
    ("#console-citation-sources-modal", 170, "css/features/_console.tcss"),
    ("#console-side-chat-modal", 170, "Widgets/Console/console_side_chat_modal.py"),
    (
        "#console-edit-message-modal",
        170,
        "Widgets/Console/console_edit_message_modal.py",
    ),
    (
        "#console-edit-thinking-modal",
        170,
        "Widgets/Console/console_edit_message_modal.py",
    ),
    ("#console-prompts-modal", 170, "Widgets/Console/console_prompts_modal.py"),
    ("#agent-history-dialog", 170, "css/components/_agentic_terminal.tcss"),
    ("#agent-progress-dialog", 170, "css/components/_agentic_terminal.tcss"),
    ("#worktree-recovery-dialog", 170, "css/features/_console_panels.tcss"),
    (
        "#internal-prompt-editor-modal",
        170,
        "Widgets/settings_internal_prompts_editor_modal.py",
    ),
    # cap 170 -- PR #2742 review: missed by the 2026-09-13 inventory (base 140)
    ("#speech-voice-profile-picker", 170, "UI/STTS_Window.py"),
    # cap 170 -- wave 2 (2026-09-19): skip-list follow-ups the triage report
    # named (base caps 100-120 the ladder relaxes at wide viewports)
    ("#notes-recovery-dialog", 170, "Widgets/Library/notes_recovery_dialog.py"),
    ("#skills-recovery-review", 170, "UI/Screens/skills_screen.py"),
    ("ChatbookCreationWindow > Container", 170, "UI/ChatbookCreationWindow.py"),
    ("ChatbookExportManagementWindow > Container", 170, "UI/ChatbookExportManagementWindow.py"),
    ("ChatbookTemplatesWindow > Container", 170, "UI/ChatbookTemplatesWindow.py"),
    # cap 150 -- base width 84-96
    ("#personal-context-review-modal", 150, "css/components/_profile_interview.tcss"),
    ("#console-scope-picker-modal", 150, "css/features/_console_panels.tcss"),
    (
        "#console-prompt-comparison-modal",
        150,
        "Widgets/Console/console_prompt_comparison_modal.py",
    ),
    ("#buddy-conversation", 150, "Widgets/Persona_Widgets/buddy_conversation_modal.py"),
    (
        "#actor-pack-import-review",
        150,
        "Widgets/Persona_Widgets/actor_pack_import_review.py",
    ),
    ("#personal-context-link-modal", 150, "css/components/_profile_interview.tcss"),
    ("#console-system-prompt-modal", 150, "css/features/_console_panels.tcss"),
    ("#server-switch-modal", 150, "Widgets/Settings_Widgets/server_switch_modal.py"),
    ("#buddy-inbox", 150, "Widgets/Persona_Widgets/buddy_workspace_modal.py"),
    (
        "FileExtractionDialog > Vertical.file-extraction-body",
        150,
        "Widgets/file_extraction_dialog.py",
    ),
    ("EmojiPickerScreen #dialog", 150, "Widgets/emoji_picker.py"),
    # cap 150 -- PR #2742 review: persona-management + tool-pack review
    # surfaces missed by the 2026-09-13 inventory (bases 86-96)
    ("#buddy-review", 150, "Widgets/Persona_Widgets/buddy_character_review.py"),
    ("#petdex-review", 150, "Widgets/Persona_Widgets/petdex_import_review.py"),
    ("#tool-pack-export-review", 150, "Widgets/Settings_Widgets/tool_pack_import_review.py"),
    ("#tool-pack-import-options", 150, "Widgets/Settings_Widgets/tool_pack_import_review.py"),
    ("#tool-pack-import-review", 150, "Widgets/Settings_Widgets/tool_pack_import_review.py"),
    ("#tool-profile-bind-review", 150, "Widgets/Settings_Widgets/tool_pack_import_review.py"),
    # cap 150 -- wave 2 (2026-09-19): skip-list follow-up (base width 96)
    ("#prompt-collection-manager", 150, "UI/Library_Modules/prompt_collection_manager_modal.py"),
    # cap 120 -- base width <= 80
    (
        "NoteCreationModal > Container",
        120,
        "Widgets/Note_Widgets/note_creation_modal.py",
    ),
    # PR #2742 review: persona-management surface missed by the inventory (base 78)
    ("#buddy-management", 120, "Widgets/Persona_Widgets/buddy_management_modal.py"),
    (
        "#conversation-selection-container",
        120,
        "Widgets/conversation_selection_dialog.py",
    ),
    ("#note-selection-container", 120, "Widgets/Note_Widgets/note_selection_dialog.py"),
    (
        "AudioTroubleshootingDialog .dialog-container",
        120,
        "Widgets/audio_troubleshooting_dialog.py",
    ),
    (
        "#file-notes-push-auth-dialog",
        120,
        "Widgets/Library/library_file_notes_git_panel.py",
    ),
    ("#trace-export-dialog", 120, "Widgets/Console/trace_export_dialog.py"),
    ("#console-library-access", 120, "css/features/_console_panels.tcss"),
    ("#capture-policy-dialog", 120, "Widgets/Console/console_capture_policy_dialog.py"),
    ("#trace-privacy-dialog", 120, "Widgets/Console/console_capture_policy_dialog.py"),
    ("#exchange-export-dialog", 120, "css/components/_agentic_terminal.tcss"),
    (
        "#console-prompt-queue-dialog",
        120,
        "Widgets/Console/console_prompt_queue_modal.py",
    ),
    (
        "#console-reaction-picker-modal",
        120,
        "Widgets/Console/console_reaction_picker_modal.py",
    ),
    (
        "#console-review-notes-modal",
        120,
        "Widgets/Console/console_review_notes_modal.py",
    ),
    (
        "#console-endpoint-template-modal",
        120,
        "Widgets/Console/console_endpoint_template_modal.py",
    ),
    ("#console-rewind-modal", 120, "Widgets/Console/console_rewind_modal.py"),
    ("#video-capacity-dialog", 120, "Widgets/Console/console_video_capacity_modal.py"),
    ("#project-skills-modal", 120, "Widgets/project_skills_import_modal.py"),
    ("#console-prompt-picker-modal", 120, "css/features/_console_panels.tcss"),
    ("#console-style-picker-modal", 120, "css/features/_console.tcss"),
    (".model-install-modal", 120, "Widgets/ModelArtifacts/install_modal.py"),
    (".local-gguf-import-modal", 120, "Widgets/ModelArtifacts/local_gguf_import.py"),
    (
        "#file-notes-push-endpoint-details-dialog",
        120,
        "Widgets/Library/library_file_notes_git_panel.py",
    ),
    (
        "#file-notes-root-details-dialog",
        120,
        "Widgets/Library/library_file_notes_workspace.py",
    ),
    (
        "#console-project-setup-modal",
        120,
        "Widgets/Console/console_project_instructions.py",
    ),
    ("#console-fork-chat-modal", 120, "Widgets/Console/console_fork_chat_modal.py"),
    (
        "#console-character-picker",
        120,
        "Widgets/Console/console_character_picker_modal.py",
    ),
    ("#console-save-as-modal", 120, "Widgets/Console/console_save_as_modal.py"),
    (
        "#console-terminal-session-modal",
        120,
        "Widgets/Console/console_terminal_session_modal.py",
    ),
    (
        "#global-full-confirmation",
        120,
        "Widgets/Console/console_capture_policy_dialog.py",
    ),
    (
        "#console-generate-image-modal",
        120,
        "Widgets/Console/console_generate_image_modal.py",
    ),
    ("EncryptionSetupDialog > Container", 120, "Widgets/password_dialog.py"),
    (
        "#character-tts-collision-dialog",
        120,
        "Widgets/Persona_Widgets/character_tts_portability_dialogs.py",
    ),
    (
        "#character-tts-existing-dialog",
        120,
        "Widgets/Persona_Widgets/character_tts_portability_dialogs.py",
    ),
    (
        ".settings-speech-credential-modal",
        120,
        "Widgets/Settings_Widgets/speech_tts_settings_panel.py",
    ),
    (
        "DocumentGenerationModal > Container",
        120,
        "Widgets/document_generation_modal.py",
    ),
    (
        "#console-auto-speak-consent-modal",
        120,
        "Widgets/Console/console_auto_speak_consent.py",
    ),
    ("#console-rag-settings", 120, "Widgets/Console/console_rag_settings_modal.py"),
    (
        "#console-summarize-preview-modal",
        120,
        "Widgets/Console/console_summarize_preview_modal.py",
    ),
    (
        "#console-workspace-switcher-modal",
        120,
        "Widgets/Console/console_workspace_switcher_modal.py",
    ),
    (
        "#prompt-delete-modal",
        120,
        "Widgets/Library/prompt_delete_confirmation_modal.py",
    ),
    (
        "#persona-visual-custom-dialog",
        120,
        "Widgets/Persona_Widgets/personas_persona_visual_pack_widget.py",
    ),
    (
        "#console-save-markdown-box",
        120,
        "Widgets/Console/console_save_markdown_modal.py",
    ),
    ("#profile-interview-cancel-modal", 120, "css/components/_profile_interview.tcss"),
    ("#dictionary-picker-dialog", 120, "Widgets/Persona_Widgets/dictionary_picker.py"),
    ("#world-book-picker-dialog", 120, "Widgets/Persona_Widgets/world_book_picker.py"),
    (
        "DictionaryAttachPicker > Vertical.dictionary-attach-body",
        120,
        "Widgets/Persona_Widgets/dictionary_attach_picker.py",
    ),
    (
        "ConversationAttachPicker > Vertical.conversation-attach-body",
        120,
        "Widgets/Persona_Widgets/conversation_attach_picker.py",
    ),
    ("#profile-dialog-container", 120, "Widgets/voice_profile_dialog.py"),
    ("FeedbackDialog > Container", 120, "Widgets/feedback_dialog.py"),
    ("#console-composer-menu", 120, "Widgets/Console/console_composer_menu_modal.py"),
    ("#prompt-variables-dialog", 120, "Widgets/Console/prompt_variables_dialog.py"),
    # cap 120 -- wave 2 (2026-09-19): base-width-rule gap closed (dialog had
    # no width rule at all; base geometry shipped with this wave)
    ("TemplateSelectorDialog .template-selector-dialog", 120, "Widgets/template_selector.py"),
)

#: Cap -> one representative anchor pinned by a live geometry test.
WIDE_TIER_CAPS: dict[int, str] = {
    170: "#console-prompts-modal",
    150: "#console-system-prompt-modal",
    120: "#console-reaction-picker-modal",
}
