# Modal wide-tier visual verification — 2026-09-19

Branch `feat/modal-wide-tier-rollout` (worktree `/tmp/all-modals`, dev 030fd9935d0).

Method: every registry modal (`Tests/UI/modal_wide_tier_registry.py`, 76 surfaces)
was mounted in a consolidated-CSS harness carrying the real
`WideViewportTierMixin` app toggle, at (200, 50) wide and (120, 40) narrow.
Each mount saved an SVG screenshot (151 captures, 8 MB — kept uncommitted at
`/tmp/all-modals/qa-captures/` as `<anchor>-{wide,narrow}.svg`; the harness is
`/tmp/all-modals/qa_modal_wide_tier.py`). OK means the wide-mode container
width measured exactly `min(cap, 85% of 200 columns)`; narrow shows the
untouched base width. Constructors borrow the fixtures of each modal's own
test module where one exists.

Results: 74 OK, 0 CLIPPED, 2 FAILED_TO_MOUNT (both environmental, not CSS):

- `NoteCreationModal > Container` — this venv lacks the tree-sitter markdown
  language pack, so its `TextArea(language="markdown")` fails at construction
  in any harness on this machine.
- `AudioTroubleshootingDialog .dialog-container` — its mount worker raises
  Textual's "Unsupported attempt to run an async worker" under headless
  `run_test` (the suite's own tests for it only parse its CSS, never mount it).

Both surfaces' wide-tier rules are present in the bundle and covered by the
coverage contract test.

| Modal anchor | Narrow (120col) | Wide (200col) | Cap | Status |
|---|---|---|---|---|
| `#console-settings-modal` | 114 | 170 | 196 | OK |
| `#console-inspector-modal` | 120 | 170 | 170 | OK |
| `#console-workspace-files-modal` | 110 | 170 | 170 | OK |
| `#file-notes-conflict-dialog` | 110 | 170 | 170 | OK |
| `#console-run-log-modal` | 96 | 170 | 170 | OK |
| `#console-citation-sources-modal` | 94 | 170 | 170 | OK |
| `#console-side-chat-modal` | 92 | 170 | 170 | OK |
| `#console-edit-message-modal` | 92 | 170 | 170 | OK |
| `#console-edit-thinking-modal` | 92 | 170 | 170 | OK |
| `#console-prompts-modal` | 104 | 170 | 170 | OK |
| `#agent-history-dialog` | 108 | 170 | 170 | OK |
| `#agent-progress-dialog` | 108 | 170 | 170 | OK |
| `#worktree-recovery-dialog` | 108 | 170 | 170 | OK |
| `#internal-prompt-editor-modal` | 108 | 170 | 170 | OK |
| `#personal-context-review-modal` | 96 | 150 | 150 | OK |
| `#console-scope-picker-modal` | 90 | 150 | 150 | OK |
| `#console-prompt-comparison-modal` | 100 | 150 | 150 | OK |
| `#buddy-conversation` | 90 | 150 | 150 | OK |
| `#actor-pack-import-review` | 92 | 150 | 150 | OK |
| `#personal-context-link-modal` | 88 | 150 | 150 | OK |
| `#console-system-prompt-modal` | 84 | 150 | 150 | OK |
| `#server-switch-modal` | 84 | 150 | 150 | OK |
| `#buddy-inbox` | 82 | 150 | 150 | OK |
| `FileExtractionDialog > Vertical` | 96 | 150 | 150 | OK |
| `EmojiPickerScreen #dialog` | 96 | 150 | 150 | OK |
| `NoteCreationModal > Container` | — | — | 120 | FAILED_TO_MOUNT (LanguageDoesNotExist: tree-sitter is available, but no built-in or user-registered language called 'markdown'.
Ensure the language is installed (e.g. ) |
| `#conversation-selection-container` | 80 | 120 | 120 | OK |
| `#note-selection-container` | 80 | 120 | 120 | OK |
| `AudioTroubleshootingDialog .dialog-container` | — | — | 120 | FAILED_TO_MOUNT (WorkerFailed: Worker raised exception: WorkerError('Unsupported attempt to run an async worker')) |
| `#file-notes-push-auth-dialog` | 78 | 120 | 120 | OK |
| `#trace-export-dialog` | 78 | 120 | 120 | OK |
| `#console-library-access` | 76 | 120 | 120 | OK |
| `#capture-policy-dialog` | 76 | 120 | 120 | OK |
| `#trace-privacy-dialog` | 76 | 120 | 120 | OK |
| `#exchange-export-dialog` | 76 | 120 | 120 | OK |
| `#console-prompt-queue-dialog` | 76 | 120 | 120 | OK |
| `#console-reaction-picker-modal` | 76 | 120 | 120 | OK |
| `#console-review-notes-modal` | 76 | 120 | 120 | OK |
| `#console-endpoint-template-modal` | 76 | 120 | 120 | OK |
| `#console-rewind-modal` | 76 | 120 | 120 | OK |
| `#video-capacity-dialog` | 76 | 120 | 120 | OK |
| `#project-skills-modal` | 76 | 120 | 120 | OK |
| `#console-prompt-picker-modal` | 76 | 120 | 120 | OK |
| `#console-style-picker-modal` | 76 | 120 | 120 | OK |
| `.model-install-modal` | 76 | 120 | 120 | OK |
| `.local-gguf-import-modal` | 76 | 120 | 120 | OK |
| `#file-notes-push-endpoint-details-dialog` | 76 | 120 | 120 | OK |
| `#file-notes-root-details-dialog` | 76 | 120 | 120 | OK |
| `#console-project-setup-modal` | 76 | 120 | 120 | OK |
| `#console-fork-chat-modal` | 74 | 120 | 120 | OK |
| `#console-character-picker` | 72 | 120 | 120 | OK |
| `#console-save-as-modal` | 72 | 120 | 120 | OK |
| `#console-terminal-session-modal` | 72 | 120 | 120 | OK |
| `#global-full-confirmation` | 72 | 120 | 120 | OK |
| `#console-generate-image-modal` | 70 | 120 | 120 | OK |
| `EncryptionSetupDialog > Container` | 70 | 120 | 120 | OK |
| `#character-tts-collision-dialog` | 70 | 120 | 120 | OK |
| `#character-tts-existing-dialog` | 70 | 120 | 120 | OK |
| `.settings-speech-credential-modal` | 70 | 120 | 120 | OK |
| `DocumentGenerationModal > Container` | 70 | 120 | 120 | OK |
| `#console-auto-speak-consent-modal` | 68 | 120 | 120 | OK |
| `#console-rag-settings` | 64 | 120 | 120 | OK |
| `#console-summarize-preview-modal` | 64 | 120 | 120 | OK |
| `#console-workspace-switcher-modal` | 64 | 120 | 120 | OK |
| `#prompt-delete-modal` | 64 | 120 | 120 | OK |
| `#persona-visual-custom-dialog` | 64 | 120 | 120 | OK |
| `#console-save-markdown-box` | 64 | 120 | 120 | OK |
| `#profile-interview-cancel-modal` | 64 | 120 | 120 | OK |
| `#dictionary-picker-dialog` | 72 | 120 | 120 | OK |
| `#world-book-picker-dialog` | 72 | 120 | 120 | OK |
| `DictionaryAttachPicker > Vertical` | 72 | 120 | 120 | OK |
| `ConversationAttachPicker > Vertical` | 72 | 120 | 120 | OK |
| `#profile-dialog-container` | 60 | 120 | 120 | OK |
| `FeedbackDialog > Container` | 60 | 120 | 120 | OK |
| `#console-composer-menu` | 56 | 120 | 120 | OK |
| `#prompt-variables-dialog` | 76 | 120 | 120 | OK |

## Addendum — PR #2742 review additions (2026-09-19, same day)

The qodo PR review found eight substantial modals the 2026-09-13 inventory
never scoped; they are now registered and shipped in the tier (registry 84
surfaces): `#speech-voice-profile-picker` (170), `#buddy-review`,
`#petdex-review`, `#tool-pack-import-options`, `#tool-pack-import-review`,
`#tool-pack-export-review`, `#tool-profile-bind-review` (150), and
`#buddy-management` (120). Coverage-contract and owner-module checks cover
all eight (same tests as the table above), and `#buddy-management` is
additionally pinned by a live spot-geometry test (base 78 -> 120 at wide,
both resize directions) in `Tests/UI/test_modal_wide_tier.py`; the visual
SVG capture above was not re-run for the eight. A new reverse-sweep contract
also guarantees no other `ModalScreen` class can sit outside the tier
without an explicit skip entry.
