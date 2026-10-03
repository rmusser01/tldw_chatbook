---
id: TASK-33006
title: 'Phase 6: Chat settings layout — core first, honest fields, pairs-only model change'
status: Done
assignee:
  - '@claude'
created_date: '2026-09-26 11:47'
updated_date: '2026-10-03 14:00'
labels:
  - model-config-redesign
  - phase-6
  - console
  - ux
  - css
  - a11y
dependencies:
  - TASK-33001
  - TASK-33002
  - TASK-33003
  - TASK-33004
  - TASK-33005
references:
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'tldw_chatbook/Widgets/Console/console_settings_modal.py'
  - 'tldw_chatbook/Widgets/Console/console_provider_picker.py'
  - 'tldw_chatbook/css/features/_console_panels.tcss'
  - 'tldw_chatbook/css/core/_variables.tcss'
  - 'Tests/Architecture/test_module_size_ratchet.py'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'backlog/decisions/161-component-pattern-library.md'
  - 'backlog/decisions/150-design-token-system-and-design-language.md'
  - 'Docs/User_Guide/console.md'
  - 'Docs/User_Guide/settings.md'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 6 of the model-configuration redesign "Switchboard with field truth" (judge-synthesis.md §2(b) mock and §4 P6; spec §8). It ships as ONE PR. It is new work: task-32864 explicitly excludes this modal. It closes the UI half of C8(1) (the hidden-for-provider line). For chats that hold work, it also closes C1(b) through 'Use saved defaults', the D1 action the ADR-095 amendment of 2026-09-26 names. Untouched chats have already converged since phase 1.

Why. The Chat settings modal, today titled 'Conversation settings' (console_settings_modal.py:1648), has five problems:
- It opens connection-first. The Connection section holds its own ConsoleProviderPicker (:1719-1735) and ModelSearchPicker (:1794).
- Every tuning field is inside an 'Advanced generation' Collapsible (:1954-1960): Temperature, Top P, Min P, Top K, Response max tokens, Seed, Presence, Frequency and a Streaming toggle button (:1962-2034). The reasoning rows that follow each carry a 'Support not verified for this model.' static (GENERATION_CONTROL_UNKNOWN_COPY, :847).
- Anthropic shows Min P, Seed, Presence and Frequency. Its PROVIDER_PARAM_MAP entry lacks them (Chat_Functions.py:281-300), and project_chat_handler_kwargs drops them silently (:1431-1435).
- A provider can be chosen without a model, because the provider picker (console_provider_picker.py, 455 lines, used only here: :156, :1725) is separate from model search.
- A chat that holds work has no way to adopt newly saved defaults.

'Conversation settings' also appears in 47 strings across the code, many of them user-visible notices (for example chat_screen.py:3747-3835 and :15168, settings_screen.py:12231-12239 and :16976).

In the review, Alex needed Tab ×13 to reach Apply (report.md). The module sits at zero ratchet headroom: 7,807 lines against a 7,807 budget (Tests/Architecture/test_module_size_ratchet.py:68).

Target: judge mock (b), 150x22 over Console at 211x44. The layout is a Model row, Core fields, then Sampling / Connection / Request estimate / name as one-row disclosures, with a Source column and a help line per field, and 'Applies to this chat only'. It builds on earlier phases:
- P1: the single supported-fields function and D1 convergence.
- P2: the field table.
- P3: compact tokens, the Esc dirty guard and the fold-hint fix.
- P4: the switcher's pick-only mode and the Source-word resolver.
- P5: the shared readiness words.

Constraints:
- ADR-150/161: geometry lives in tokens in core/_variables.tcss only.
- ADR-031 rule 2: no Ctrl+C, V, X, S, D, Z, A, R or W.
- ADR-033: keep the commit models.
- ADR-095: Apply is conversation-owned, and explicit source-owned chats never rebase.
- ADR-097: ratchets never rise.
- Target sizes: 211x44 first, then 235x52. Smaller sizes are not redesigned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The phase ships as one PR containing every subtask below
- [x] #2 At 211x44 and 235x52, Chat settings shows the whole Model view without scrolling: the model row, the core fields and the footer actions, core first, each field with a Source word and one help line
- [x] #3 For Anthropic, one line names the fields it does not accept, instead of four editable dead fields (C8(1) UI closed)
- [x] #4 The model is changed only by choosing a provider·model pair in the switcher's pick mode, and ConsoleProviderPicker and its module are deleted
- [x] #5 A chat with any work can adopt newly saved defaults through 'Use saved defaults' and Apply, and no configuration is written by that path (C1(b) for chats with work, D1)
- [x] #6 No user-visible copy names this modal 'Conversation settings'; it is 'Chat settings' everywhere
- [x] #7 console_settings_modal.py shrinks, and its row in Tests/Architecture/test_module_size_ratchet.py is lowered to the measured size in this PR. Boot CSS bytes and the _ui_ready census do not rise (ADR-097)
- [x] #8 New geometry comes only from tokens in core/_variables.tcss. The dimension-literal and Python-style ratchets in Tests/UI/test_component_pattern_governance.py (:266, :291) stay at their floors (ADR-150/161)
- [x] #9 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added, and every key the modal advertises works
- [x] #10 Every existing test that pinned the old layout is rewritten on purpose and listed in the PR description, and the Context and memory view's tests pass unchanged
- [x] #11 Live evidence: tmux captures at 211x44 and 235x52 from a scratch profile (TLDW_CONFIG_PATH), using the production stylesheet and real keypresses. They cover an Anthropic chat, a Not-ready chat and a chat using 'Use saved defaults'
- [x] #12 Docs/User_Guide pages updated: console.md (Chat settings) and settings.md (the Console-modal reference near :373) (content only; verification is recorded in the task notes, never as a "Verified against" paragraph, per CLAUDE.md)
- [x] #13 ./scripts/preflight.sh passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Chat settings now opens core-first in one row grammar. It has these parts:

- The MODEL row comes first. Its **Change Alt+M** opens Switch model in pick-only mode, the only way to change the model. ConsoleProviderPicker and its module are deleted.
- The CORE fields follow: Temperature, Max tokens, Streaming, and the reasoning or thinking controls the model takes. Each row shows a Source word and one help line.
- Then come four closed disclosures, each title one row: Sampling, Connection, Request estimate, and Your name in this chat.
- Fields the provider does not accept are hidden. The Sampling title names them when they fit one row and counts them otherwise, and the opened disclosure lists them all.
- The modal is named **Chat settings** everywhere and says what it changes.
- **Use saved defaults** lets a chat that holds work adopt newly saved defaults through Apply, which writes no configuration.

The branch is rebased onto origin/dev 9b28ce1479. It was never merged, and nothing was pushed.

**Commits per subtask** (SHAs after the rebase):
- Plan and owner ruling: be879f99d6, cd26ec14c3, b29734c902, ae01b1139d, 1d4b2bbdda. 1d4b2bbdda records the 2026-10-02 one-row ruling in the plan's Global Constraints.
- TASK-33006.1, core-first layout and row grammar: 11bcf7651b, 5e5726725f.
- TASK-33006.2, hidden fields: dd780bdb48, 9069fb45b3.
- TASK-33006.3, one-row disclosures: 666e0e47ae, 93fb5294e6, 66f255611c, 7b8da5b7ef.
- TASK-33006.4, pick mode only, picker deleted: 903b2bf7ce, ac5cde4298.
- TASK-33006.5, Chat settings name, scope, footer and Use saved defaults: 2911d9b564, ae2374951c.
- TASK-33006.6, Context labels and the defaults line: 909b6759d7, 2b8861e5b5.
- TASK-33006.7, a view switch opens at the top: 939bf6333b, fb666a0a50, 0bebb015a6.
- Final review fix wave: 5ef85f878d (code, tests, guide), 020379c8c1 (captures retaken), and the task-file and lessons commit that carries these notes. TASK-33006.8 (three tests red on dev) is closed in 5ef85f878d.

**Final review fix wave** (`.superpowers/sdd/plan-2026-10-02-model-config-p6/final-review.md`, findings C1 and I1-I7, every must-fix Minor):

- **C1.** The owner ruling of 2026-10-02 is implemented and pinned. The census covers all 62 `PROVIDER_PARAM_MAP` providers at their default model and with every field hidden; none goes over 141 cells. A real-key test opens Sampling and finds every hidden label. TASK-33006.2 AC#1-2 carry the amendment, and TASK-33006.1 AC#1 now holds as written. Docs and every capture that showed the two-row wrap were retaken.
- **I1.** Rebased with `git rebase --onto origin/dev 06fd85792b`, never merged. Two conflicts were resolved:
  - `lessons-testing-evidence.md`: both entries kept.
  - `Tests/Widgets/test_console_provider_picker.py`: resolved as delete, and TASK-33002.16 is annotated as superseded by TASK-33006.4.
- **I2.** TASK-33006.8 is Done (see its notes).
- **I3.** The Persona-change refusal says "Chat settings are still saving".
- **I4.** Alt+M no longer bypasses the close guard or a default recovery.
- **I5.** Save as model default shows on the same saved-defaults answer as Use saved defaults.
- **I6.** A blank row reads "provider", and a blank choice Select shows "default".
- **I7.** A created entry's listing can no longer orphan it silently, and the status line says the listing is running.
- **Minors.** `UNSAVED_ENDPOINT_COPY` names Save endpoint & use model. The "unapplied edit" line in console.md is corrected. Draft keys read "unsaved key". All titles measure with cell_len against one constant. `context_copy` is public. The required-field and row-id knowledge is shared. Stale comments are fixed. `MODAL_LABEL_WIDTH` is gone. The `/endpoint` test asserts the listed row. The test literals use constants, a misnamed test is renamed, the DB is closed, and a wrong test comment is fixed. Details are in each subtask's "Final fix wave" notes.

**Behaviour changes for the PR description:**
- Chat settings is core-first and 150x22 at 211x44 and 235x52. The Context and memory view keeps its taller frame (94%) and scrolls.
- Fields the provider does not accept are hidden, never shown dimmed, and Apply commits them blank. A blank Temperature or Top P blocks Apply only while the provider accepts it.
- The model changes only by picking a provider·model pair in Switch model's pick mode (Change or Alt+M). New endpoint… and `/endpoint` land the new entry on a pair the same way. Pick mode cannot pick a NEEDS SETUP row.
- "Conversation settings" is now "Chat settings" in every user-visible string. Log lines and internal exception messages keep the old name.
- The footer reads Use saved defaults, Save as model default (only while a save would change the defaults), Default for new chats (Ctrl+N) and Apply to this chat (Ctrl+Enter). Cancel shows in the Context view only.
- `CharactersRAGDB.update_conversation` now starts with an IMMEDIATE transaction. This affects every conversation update, not only this modal: it was a real read-then-write race that lost Apply's snapshot under load (TASK-33006.5 fix round 1).
- The 196-column wide tier is gone.

**Tests rewritten or deleted on purpose** (parent AC#10). The full lists, with the reason for each, are in the subtask notes. In summary:
- TASK-33006.1: the AC#13 order and Tab pins, the Streaming button sites, the layout pins, and the geometry wide-tier pins.
- TASK-33006.2: the unknown-support and enumerated-input tests, and the two native-flow Configure-credential clicks.
- TASK-33006.4: every picker, model Select/Input, Custom model and Keep unverified test was deleted or rewritten to pick mode. `Tests/Widgets/test_console_provider_picker.py` was deleted.
- TASK-33006.5: the header, footer and Cancel pins, and the rename pins.
- TASK-33006.6: the strict xfail became `test_defaults_line_follows_the_view_switch`.
- TASK-33006.7: none.
- Final fix wave:
  - `_expected_line`, `_painted_title` and the Anthropic paint test in `test_console_settings_hidden_fields.py`;
  - `FOLDED` in `test_console_settings_disclosures.py`;
  - `test_blank_field_says_what_it_sends_and_no_placeholder_shows` and `test_every_row_paints_label_value_source_and_help`;
  - `test_labels_fit_the_existing_label_columns` (now strict);
  - `test_console_settings_modal_model_default_action_names_its_scope` (renamed);
  - `test_use_saved_defaults_is_disabled_with_its_reason_while_the_draft_matches`;
  - the four `UNSAVED_ENDPOINT_COPY` pins;
  - `test_native_console_state_keeps_suspended_settings_draft_process_local`;
  - the footer composition in `build_geometry_modal` and `_resize_full_settings`, because their ready drafts now hold a Temperature the saved chain lacks, so all four footer actions show and a blocked chat offers no model-default save.

  The Context and memory view's own tests pass unchanged.

**Delegated** (riders filed as subtasks; none blocks this PR):
- TASK-33006.9: aligned Source column.
- TASK-33006.10: open-focus fallback and the helper's name.
- TASK-33006.11: local_onnx and local_transformers support.
- TASK-33006.12: the neutral copy pushes the help off the row.
- TASK-33006.13: the Endpoint row sync by id.
- TASK-33006.14: coverage for a collapsed Connection on a blocked chat.
- TASK-33006.15: `endpoint_host` '://'.
- TASK-33006.16: phantom edits after a pick.
- TASK-33006.17: two endpoint and click tests red on dev.
- TASK-33006.18: `_switcher_sources` typing.
- TASK-33006.19: per-endpoint served listings.
- TASK-33006.20: the stale app footer after Esc.
- TASK-33006.21: one readiness sync on mount.
- TASK-33006.22: the Context view's reason with an unsaved endpoint.
- TASK-33006.23: the focused tab's Selected mark.
- TASK-33006.24: the anchor geometry flake.

TASK-34000.43 (already on dev) conforms Chat settings to the amended ADR-031, under which Ctrl+S saves in every editor (dev 81d51feefc, 2026-10-03). This plan's and AC#9's "no Ctrl+S" wording predates that amendment. Nobody should add Ctrl+S here outside that task, and the ban should not be read as current.

**Owner decisions recorded, not taken here:**
- The Context view's 22 -> 94% height change (T1 AC#11 ruling).
- Whether the `boot_css_bytes.json` re-baseline is like-for-like. The review recommends regenerating it with `scripts/update_boot_budget_snapshots.py --only css` in its own commit. The boot CSS budget test passes on the rebased tree.
- Pick mode cannot pick NEEDS SETUP rows.
- R14's "a defaults button" wording, against TASK-33006.6 AC#2's ruling (the line follows Save as model default only).

**Pre-existing reds**, identical on origin/dev 9b28ce1479:
- 11 size ratchets: 9 modules (console_chat_controller, console_chat_store, mcp_workbench, llm_screen, watchlists_collections_screen, console_transcript, app_lifecycle, app_service_wiring, tldw_api/client) and 2 screens (chat_screen, library_screen). chat_screen is 25,290 lines here against 25,294 on dev; the controller is unchanged.
- The governance dimension-literal ratchet (_agentic_terminal 1, _settings_splash_theme 2, _workflows 11) and Python-style ratchet (library_skill_work_pane:170), with the same offenders as dev.
- In the 79-file covering set, about 1,100 failures on each side, overwhelmingly the environmental `RecoveryRequired: raw_source_selection_changed` (ADR-126).

**Verification** (final head; real `~/.config/tldw_cli/config.toml` 15c6cb224a6a51c7 and the data listing db7e7faf5bff92d2 unchanged throughout):
- **Covering set.** 79 files: every test file the branch touches plus every test naming the modified modules, ids or CSS, plus component governance and the boot CSS budget. Run with `-n 8` and `PYTHONPATH=<tree>`, against the rebased pre-fix tree 1d4b2bbdda:
  - Head: 1114 failed / 3930 passed. Base: 1104 / 3930.
  - Of 14 head-only names, 12 were the footer-composition pins named above, now rewritten and passing.
  - `test_endpoint_command_create_lands_on_a_pair_in_pick_mode` raced its new row assertion; it now waits for the rows and passes.
  - `test_reopen_keeps_previously_applied_temperature[default-full]` passes 3/3 serially (4/4 params each time).
  - The anchor geometry test fails intermittently on both sides (TASK-33006.24).
  - Four base-only failures pass at head, including the state-store test.
- **Six new Phase 6 test files**, all passing at the final head (132 tests): core_first 39, hidden_fields 22, disclosures 38, model_change 18, saved_defaults 13, saved_defaults_flow 2.
- **Negative controls**, each red as expected:
  - forcing the named Sampling form and hiding the list fails 5 tests;
  - removing the Alt+M guard fails `[alt+m]`;
  - an always-shown save fails the I5 test;
  - the resolve outside the `try` fails the I7 test;
  - a no-op `_cancel_generation_test` fails the 33006.8 AC#1 test.
- **Architecture** (`-p no:xdist`): the Chat settings name guard passes, and the size ratchets fail exactly dev's 11 names. The modal is 6,103 with its row lowered from 6,142.
- **Performance** (serial, `PYTHONPATH=<worktree>`): `_ui_ready` census 1033, conversation-settings boot closure, boot CSS byte budget, and the keystroke census including storage units: 12 passed.
- **Preflight** (`PYTHON=<venv> ./scripts/preflight.sh`): rc 0.
- **Live captures.** At 5ef85f878d, from the real app under `env -i` with scratch profiles at 211x44 and 235x52, with real keys and SGR clicks. They cover an Anthropic chat (T2, T3, T7), not-ready chats (T3 06, T7 07-10) and Use saved defaults (T5). Retaken: T2 01-04, T3 01-06, T4 01-07, T5 01-08, T6 01-04, T7 01-10. No capture shows a wrapped title, and none shows a save offered beside Matches saved defaults.
<!-- SECTION:NOTES:END -->
