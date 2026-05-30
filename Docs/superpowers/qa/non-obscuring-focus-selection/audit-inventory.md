# Non-Obscuring Focus Selection Audit Inventory

Date: 2026-05-28
Scope: PR 1 foundation plus app-wide inventory

## PR 1 Scope

| Selector | Owner | Screen/widget | Type | Current risk | Target state | PR/status | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `*:focus` | `tldw_chatbook/css/core/_reset.tcss` | Global | fallback | heavy outline can obscure dense labels | visible non-obscuring fallback | PR 1 | source contract |
| `Button:focus` | `tldw_chatbook/css/components/_buttons.tcss` | Global buttons | button | saturated fill and heavy outline | readable text, underline, subtle background, no heavy outline | PR 1 | source contract |
| `Button:hover:focus` | `tldw_chatbook/css/components/_buttons.tcss` | Global buttons | button | stronger hover fill plus heavy outline | readable focused hover without heavy outline | PR 1 | source contract |
| `Input:focus` | `tldw_chatbook/css/components/_forms.tcss` | Native inputs | input | full-field focus can read as warning or obscure compact content | thin border plus bottom emphasis | PR 1 | source contract |
| `TextArea:focus` | `tldw_chatbook/css/components/_forms.tcss` | Native text areas | input | full-field focus can read as warning or obscure compact content | thin border plus bottom emphasis | PR 1 | source contract |
| `Select:focus` | `tldw_chatbook/css/components/_forms.tcss` | Native selects | select | native focus treatment may fall back to heavy outline | thin border plus bottom emphasis where supported | PR 1 | source contract and exception note if needed |
| `.form-input:focus` | `tldw_chatbook/css/components/_forms.tcss` | Shared forms | input | accent fill can obscure compact input text | thin border plus bottom emphasis | PR 1 | source contract |
| `.form-textarea:focus` | `tldw_chatbook/css/components/_forms.tcss` | Shared forms | text area | accent fill can obscure compact input text | thin border plus bottom emphasis | PR 1 | source contract |
| `.nav-button:focus` | `tldw_chatbook/UI/Navigation/main_navigation.py` | Top navigation | nav tab | focus lacks underline contract | underline plus quiet secondary cue | PR 1 | Python source contract |
| `.nav-button.is-active` | `tldw_chatbook/UI/Navigation/main_navigation.py` | Top navigation | nav tab | selected fill can dominate | subtle selected fill, readable label | PR 1 | Python source contract |
| `.nav-button.is-active:focus` | `tldw_chatbook/UI/Navigation/main_navigation.py` | Top navigation | nav tab | combined state unspecified | selected remains readable, focus adds underline | PR 1 | Python source contract |
| `.library-source-action:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Library source browser | button | one-row action can be obscured by global focus | readable text, underline, subtle background | PR 1 visible offender | source contract and PNG |
| `.library-mode-chip:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Library mode bar | chip | focus lacks underline contract | readable text, underline, subtle background | PR 1 | source contract |
| `.library-mode-chip.is-active` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Library mode bar | selected chip | selected fill can dominate | active remains readable | PR 1 | source contract |
| `.library-mode-chip.is-active:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Library mode bar | selected focused chip | combined state lacks focus cue | active remains readable, focus adds underline | PR 1 | source contract |
| `#console-native-composer.console-composer-focused` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Console composer | input container | heavy orange frame reads as warning/selection | thin border plus subtle emphasis | PR 1 visible offender | source contract and PNG |
| `#console-native-transcript:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Console transcript | transcript | tall border can crowd dense transcript rows | solid non-heavy border | PR 1 | source contract |
| `.console-transcript-action-button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Console transcript | button | action fill can obscure compact labels | readable text, underline, subtle background | PR 1 visible offender | source contract and PNG |
| `.console-transcript-message-selected` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Console transcript | selected message | selected row lacks consistent active/selected cue | raised row, border, underline | PR 1 | source contract |
| `.settings-compact-input:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings | compact input | heavy outline can obscure one-row field | thin border plus bottom emphasis | PR 1 | source contract |
| `#settings-shell Button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings | button | reference pattern must stay readable | preserve underline plus raised background | PR 1 reference | existing Settings tests |
| `.settings-action-row Button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings action rows | button | reference pattern must stay readable | preserve underline plus raised background | PR 1 reference | existing Settings tests |
| `#settings-impact-pane Button:focus` | `tldw_chatbook/css/components/_agentic_terminal.tcss` | Settings impact pane | button | reference pattern must stay readable | preserve underline plus raised background | PR 1 reference | existing Settings tests |

## Deferred Inventory

| Selector | Owner | Screen/widget | Type | Current risk | Target state | PR/status | Verification |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `NavigationButton:focus` | `tldw_chatbook/Widgets/base_components.py` | shared widget | button/nav | inline focus state needs review | two-cue non-obscuring focus | PR 2 shared widgets | source contract |
| `NavigationButton.active` | `tldw_chatbook/Widgets/base_components.py` | shared widget | active nav | active fill needs review | active plus focus contract | PR 2 shared widgets | source contract |
| `Collapsible > .collapsible--header:focus` | `tldw_chatbook/css/components/_widgets.tcss` | shared widgets | collapsible header | local focus treatment needs review | readable non-obscuring focus | PR 2 shared widgets | source contract |
| `.message-actions Button:focus` | `tldw_chatbook/css/components/_messages.tcss` | message actions | button | compact action focus needs review | two-cue non-obscuring focus | PR 2 shared widgets | source contract |
| `.message-actions Button:focus:hover` | `tldw_chatbook/css/components/_messages.tcss` | message actions | button | combined hover/focus needs review | two-cue non-obscuring focus | PR 2 shared widgets | source contract |
| `.chat-sidebar-toggle-button:focus` | `tldw_chatbook/css/features/_chat.tcss` | Chat | button | heavy outline feature override | two-cue non-obscuring focus | PR 3 chat focus | source contract |
| `.rag-settings-panel:focus-within` | `tldw_chatbook/css/features/_chat.tcss` | Chat RAG settings | panel | focus-within treatment needs review | non-obscuring container cue | PR 3 chat focus | source contract |
| `.chat-tab.active` | `tldw_chatbook/css/features/_chat_tabs.tcss` | Chat tabs | selected tab | active tab fill needs review | selected plus focus contract | PR 3 chat focus | source contract |
| `.coding-nav-button:focus` | `tldw_chatbook/css/features/_coding.tcss` | Coding | nav button | local focus override needs review | inherit shared two-cue button focus | PR 4 feature focus | shared button contract |
| `.search-query-input-enhanced:focus` | `tldw_chatbook/css/features/_search-rag.tcss` | Search/RAG | input | search field focus needs review | thin non-obscuring input focus | PR 4 feature focus | source contract |
| `Input.search-highlight:focus, TextArea.search-highlight:focus` | `tldw_chatbook/css/features/config_search.tcss` | Config search | input | highlight focus needs review | thin non-obscuring input focus | PR 4 feature focus | source contract |
| `FeatureNotAvailableDialog Button:focus` | `tldw_chatbook/css/features/feature_alerts.tcss` | Feature alert dialog | button | dialog button focus needs review | inherit shared two-cue button focus | PR 4 feature focus | shared button contract |
| `.sidebar *:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | fallback | heavy outline sidebar override | inherit global non-obscuring fallback | PR 5 sidebar focus | shared focus contract |
| `.sidebar Button:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | button | heavy outline sidebar override | inherit shared two-cue button focus | PR 5 sidebar focus | shared button contract |
| `.sidebar Select:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | select | heavy outline sidebar override | inherit shared thin input focus | PR 5 sidebar focus | shared input contract + stable base geometry |
| `.sidebar Input:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | input | heavy outline sidebar override | inherit shared thin input focus | PR 5 sidebar focus | shared input contract + stable base geometry |
| `.setting-input:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | input | local focus override needs review | inherit shared thin input focus | PR 5 sidebar focus | shared input contract + stable base geometry |
| `.preset-button.active` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | active preset | active fill needs review | readable selected preset state | PR 5 sidebar focus | source contract |
| `.sidebar-resize-button:focus` | `tldw_chatbook/css/layout/_sidebars.tcss` | Legacy sidebars | resize button | tiny button focus needs review | inherit shared two-cue button focus | PR 5 sidebar focus | shared button contract |
| `Input:focus, TextArea:focus, Select:focus` | `tldw_chatbook/css/features/_ingestion_rebuilt.tcss` | Ingestion rebuilt | mixed inputs | feature-level input focus override | inherit shared thin input focus | PR 6 ingestion rebuilt focus | shared input contract |
| `Button:focus` | `tldw_chatbook/css/features/_ingestion_rebuilt.tcss` | Ingestion rebuilt | button | feature-level button focus override | inherit shared two-cue button focus | PR 6 ingestion rebuilt focus | shared button contract |
| `.media-card:focus` | `tldw_chatbook/css/features/_new_ingest.tcss` | New ingest | card | card focus needs review | non-obscuring focus cue | PR 7 new ingest focus | source contract |
| `.drop-zone.active` | `tldw_chatbook/css/features/_new_ingest.tcss` | New ingest | active drop zone | active fill needs review | active plus focus/drop contract | Deferred PR 4+ | not yet migrated |
| `.quick-actions Button:focus` | `tldw_chatbook/css/features/_new_ingest.tcss` | New ingest | button | compact action focus needs review | two-cue non-obscuring focus | PR 7 new ingest focus | shared button contract |
| `.metadata-grid Input:focus` | `tldw_chatbook/css/features/_new_ingest.tcss` | New ingest | input | feature-level input focus override | thin non-obscuring focus | PR 7 new ingest focus | shared input contract |
| `Button:focus, Input:focus, RadioButton:focus` | `tldw_chatbook/css/features/_new_ingest.tcss` | New ingest | mixed controls | feature-level focus override | widget-specific non-obscuring focus | PR 7 new ingest focus | shared/global focus contract |
| `.step-number.active` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | step indicator | active state needs review | selected plus progress contract | Deferred PR 4+ | not yet migrated |
| `.step-title.active` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | step title | active state needs review | selected plus progress contract | Deferred PR 4+ | not yet migrated |
| `.wizard-step.active` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | step card | active fill needs review | selected plus progress contract | Deferred PR 4+ | not yet migrated |
| `.content-type-card.selected` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | selected card | selected fill needs review | selected plus focus contract | Deferred PR 4+ | not yet migrated |
| `.preset-card.selected` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | selected card | selected fill needs review | selected plus focus contract | Deferred PR 4+ | not yet migrated |
| `ProgressStep .status-item.active` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | status row | active row needs review | readable selected row | Deferred PR 4+ | not yet migrated |
| `ImportProgressStep .status-item.active` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | status row | active row needs review | readable selected row | Deferred PR 4+ | not yet migrated |
| `SmartContentTree Tree > .selected-node` | `tldw_chatbook/css/features/_wizards.tcss` | Wizards | tree node | selected row needs review | readable selected row | Deferred PR 4+ | not yet migrated |
| `Input:focus, Select:focus, TextArea:focus` | `tldw_chatbook/css/features/_evaluation_unified.tcss` | Evals unified | mixed inputs | feature-level input focus override | widget-specific non-obscuring focus | PR 8 evals/embeddings focus | shared input contract |
| `.embeddings-nav-button:focus` | `tldw_chatbook/css/features/_embeddings.tcss` | Embeddings | nav button | local focus override needs review | two-cue non-obscuring focus | PR 8 evals/embeddings focus | source contract |
| `.embeddings-toggle-button-enhanced:focus` | `tldw_chatbook/css/features/_embeddings.tcss` | Embeddings | button | local focus override needs review | two-cue non-obscuring focus | PR 8 evals/embeddings focus | source contract |
| `Input:focus, Select:focus, TextArea:focus, Button:focus, Checkbox:focus` | `tldw_chatbook/css/features/_embeddings.tcss` | Embeddings | mixed controls | heavy outline feature override | widget-specific non-obscuring focus | PR 8 evals/embeddings focus | shared/global focus contract |
| `.filter-button.active` | `tldw_chatbook/css/features/_embeddings.tcss` | Embeddings | filter button | active fill needs review | active plus focus contract | PR 8 evals/embeddings focus | source contract |
| `#tab-dropdown-select:focus` | `tldw_chatbook/css/features/_tab_dropdown.tcss` | Tab dropdown | select | select focus needs review | thin non-obscuring focus | PR 9 feature nav focus | source and bundled geometry contract |
| `.ingest-nav-pane .ingest-nav-button.active` | `tldw_chatbook/css/features/_ingest.tcss` | Ingest | active nav | active fill needs review | active plus focus contract | PR 9 feature nav focus | source and bundled active-state contract |
| `.tools-nav-pane .ts-nav-button.active-nav` | `tldw_chatbook/css/features/_tools-settings.tcss` | Tools/settings | active nav | active fill needs review | active plus focus contract | PR 9 feature nav focus | source and bundled active-state contract |
| `ChatbooksWindowImproved .search-input:focus` | `tldw_chatbook/css/features/_chatbooks_improved.tcss` | Chatbooks | input | feature-level input focus override | thin non-obscuring focus | PR 11 chatbooks search focus | source and inline widget geometry contract |
| `.keyword-item.selected` | `tldw_chatbook/css/features/_media.tcss` | Media | selected row | selected row needs review | readable selected row | PR 10 media selection focus | source and bundled selected-state contract |
| `.review-item.selected` | `tldw_chatbook/css/features/_media.tcss` | Media | selected row | selected row needs review | readable selected row | PR 10 media selection focus | source and bundled selected-state contract |
| `MediaNavigationPanel .media-type-button.active` | `tldw_chatbook/Widgets/Media/media_navigation_panel.py` | Media | active button | active fill needs review | active plus focus contract | PR 10 media selection focus | inline source contract |
| `MediaListPanel .media-item.selected` | `tldw_chatbook/Widgets/Media/media_list_panel.py` | Media | selected row | selected row needs review | readable selected row | PR 10 media selection focus | inline source contract |
| `.sample-row.selected` | `tldw_chatbook/Widgets/Evals/sample_browser_dialog.py` | Evals dialog | selected row | selected row needs review | readable selected row | PR 12 evals sample selection focus | inline source contract |
| `BaseTamagotchi:focus` | `tldw_chatbook/Widgets/Tamagotchi/base_tamagotchi.py` | Tamagotchi | custom widget | local focus style needs review | non-obscuring custom focus | PR 13 tamagotchi focus | inline source contract |
| `EmojiButton.emoji_button:focus` | `tldw_chatbook/Widgets/emoji_picker.py` | Emoji picker | button | compact button focus needs review | two-cue non-obscuring focus | PR 14 compact widget focus | source contract |
| `PathBreadcrumbs .breadcrumb-button:focus` | `tldw_chatbook/Widgets/enhanced_file_picker.py` | File picker | breadcrumb button | compact button focus needs review | two-cue non-obscuring focus | PR 14 compact widget focus | source contract |
| `ModelCardViewer .file-item.selected` | `tldw_chatbook/Widgets/HuggingFace/model_card_viewer.py` | HuggingFace model card | selected row | selected row needs review | readable selected row | PR 15 model-card focus | inline source contract |
| `NotesToolbar Button.toggle.active` | `tldw_chatbook/Widgets/Note_Widgets/notes_toolbar.py` | Notes toolbar | active toggle | active fill needs review | active plus focus contract | Deferred PR 4+ | not yet migrated |
| `NotesEditorWidget:focus` | `tldw_chatbook/Widgets/Note_Widgets/notes_editor_widget.py` | Notes editor | custom input | local focus style needs review | non-obscuring editor focus | Deferred PR 4+ | not yet migrated |
| `SyncProgressWidget.active` | `tldw_chatbook/Widgets/Note_Widgets/notes_sync_widget.py` | Notes sync | active progress | active state needs review | readable active progress state | Deferred PR 4+ | not yet migrated |
| `SyncProgressSection.active` | `tldw_chatbook/Widgets/Note_Widgets/notes_sync_widget_improved.py` | Notes sync | active progress | active state needs review | readable active progress state | Deferred PR 4+ | not yet migrated |

## Audit Notes

- The inventory command also reports Python state variables such as `selected_*` and `active_*`. Those are not selector contracts unless they emit CSS classes or visible row markers; they are excluded from this table unless backed by a style selector.
- PR 1 only migrates shared/global focus styles plus the visible Console, Library, Settings, and top-navigation offenders. Deferred rows are intentionally not treated as complete.
