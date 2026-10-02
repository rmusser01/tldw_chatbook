# Plan: Phase 6: Chat settings layout — core first, honest fields, pairs-only model change (TASK-33006)

Spec: backlog/docs/spec-2026-09-26-model-config-redesign.md (binding authority; ADR-095 and ADR-012 amendments of 2026-09-26).
Evidence: qa/model-config-ux-review-2026-09-26/ (mockups-211x44.md, verified-claims.md).
Parent task: backlog/tasks/task-33006 - Phase-6-Chat-settings-layout-core-first-honest-fields-pairs-only-model-change.md

## Phase goal (parent)

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

### Parent acceptance criteria

- [ ] #1 The phase ships as one PR containing every subtask below
- [ ] #2 At 211x44 and 235x52, Chat settings shows the whole Model view without scrolling: the model row, the core fields and the footer actions, core first, each field with a Source word and one help line
- [ ] #3 For Anthropic, one line names the fields it does not accept, instead of four editable dead fields (C8(1) UI closed)
- [ ] #4 The model is changed only by choosing a provider·model pair in the switcher's pick mode, and ConsoleProviderPicker and its module are deleted
- [ ] #5 A chat with any work can adopt newly saved defaults through 'Use saved defaults' and Apply, and no configuration is written by that path (C1(b) for chats with work, D1)
- [ ] #6 No user-visible copy names this modal 'Conversation settings'; it is 'Chat settings' everywhere
- [ ] #7 console_settings_modal.py shrinks, and its row in Tests/Architecture/test_module_size_ratchet.py is lowered to the measured size in this PR. Boot CSS bytes and the _ui_ready census do not rise (ADR-097)
- [ ] #8 New geometry comes only from tokens in core/_variables.tcss. The dimension-literal and Python-style ratchets in Tests/UI/test_component_pattern_governance.py (:266, :291) stay at their floors (ADR-150/161)
- [ ] #9 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added, and every key the modal advertises works
- [ ] #10 Every existing test that pinned the old layout is rewritten on purpose and listed in the PR description, and the Context and memory view's tests pass unchanged
- [ ] #11 Live evidence: tmux captures at 211x44 and 235x52 from a scratch profile (TLDW_CONFIG_PATH), using the production stylesheet and real keypresses. They cover an Anthropic chat, a Not-ready chat and a chat using 'Use saved defaults'
- [ ] #12 Docs/User_Guide pages updated: console.md (Chat settings) and settings.md (the Console-modal reference near :373) (content only; verification is recorded in the task notes, never as a "Verified against" paragraph, per CLAUDE.md)
- [ ] #13 ./scripts/preflight.sh passes

## Global Constraints

- Work ONLY in this worktree: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/model-config-p6. Start EVERY shell command with `cd <that path> &&`. Never touch the main checkout (another session's uncommitted work).
- Python: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python (the worktree has no venv); run pytest FROM the worktree cwd. Tests/Architecture needs -p no:xdist.
- NEVER write ~/.config/tldw_cli or ~/.local/share/tldw_cli. Never bypass test isolation (private_profile_test, TLDW_TEST_PRIVATE_PROFILE_NODE, HOME/XDG/TLDW_CONFIG_PATH). Live runs: scratch TLDW_CONFIG_PATH with a unique users_name, FULL SCREEN 211x44 (primary) and 235x52.
- ADR-126 RecoveryRequired in a clean worktree is environmental: compare failure-NAME sets against origin/dev.
- Size ratchets never rise (ADR-097); console_settings_modal.py net lines <= 0 against its current row.
- Geometry only through tokens in css/core/_variables.tcss (ADR-150/161); rebuild the CSS bundle with the repo script.
- ADR-031: never bind Ctrl+C/V/X/S/D/Z/A/R/W; footer hints must match working bindings.
- TDD; real-implementation tests for config/provider surfaces; rewrite pinned tests on purpose and name them.
- Commit per task `fix|feat(model-config): <summary> (TASK-33006.N)` + `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never push; NEVER merge origin/dev into the branch.
- Tick ACs, set Done, add Implementation Notes in each subtask file. Update Docs/User_Guide CONTENT; never add "Verified against" paragraphs (record verification in task notes).
- Lessons (backlog/docs/lessons-*.md): insert new entries MID-FILE near a related section, never appended at the end of the file (dev appends there constantly and every rebase conflicts).
- Evidence: commit captures (.txt/.ansi.txt) under qa/, but NEVER commit one-off driver or probe scripts (.sh/.py with machine-specific paths); describe the capture procedure in a short qa README instead. Captures are reviewed by AI reviewers, so they must not contradict the PR's claims.
- Before DONE: covering tests + `PYTHON=<venv> ./scripts/preflight.sh` (rc 0; never piped through tail).

## Lessons carried from Phases 4-5 (binding)

- The venv editable install points at the MAIN checkout: some tests silently import its code. Before calling anything a regression, compare failure-NAME sets against a clean origin/dev worktree (create under the session scratchpad, remove after). A log line can interleave into a FAILED name; strip timestamps before diffing.
- Qodo flags EVERY new public function/method without Google-style Args:/Returns: sections and every repeated string literal that already has a constant. Write both up front.
- Never let two surfaces contradict each other at the same moment (readiness words, Key rows, Overview status): the Phase 5 checkpoint found two such contradictions. Every readiness word comes from the shared vocabulary module (Chat readiness words) and the shared connection evidence owner.
- Owner rulings in force: only a 401 reads "key rejected" and blocks; a 403 from any model listing is "model listing unavailable" (non-blocking, never verified); "verified" applies only to the model actually tested; untouched shipped local defaults that refuse sit quietly under NOT RUNNING.
- Phase 5 riders TASK-33005.7-.15 are NOT in Phase 6 scope unless a Phase 6 task file names them.
- A census or timing test (storage units, keystroke work) that fails also fails on clean dev intermittently (trace maintenance storage_admissions 2.125 > 2 is a known dev regression): run serially and compare with dev before acting.
- Workers: never run_worker(exclusive=True) without group=; gated UI pilot tests carry bootstrap_profile in the file itself.

## Task 1: Lay out Chat settings core-first with one field-row grammar (TASK-33006.1)

Task file: backlog/tasks/task-33006.1 - Lay-out-Chat-settings-core-first-with-one-field-row-grammar.md
Depends on: TASK-33002, TASK-33003, TASK-33004

### Why

The Model view puts setup ahead of tuning. Connection comes first, and it is about 25 rows at 211x44 (verified-claims C5(4)). Every tuning field then sits inside the 'Advanced generation' Collapsible (console_settings_modal.py:1954-1960).

That order was set by TASK-30012 AC#1 and is pinned by Tests/UI/test_console_session_settings.py:13764 test_console_settings_modal_connection_first_hierarchy_and_title. In the review, a power user needed 45 actions for a switch-and-tune task, including Tab ×13 to reach Apply (report.md).

Streaming is a toggle Button (:2020-2034), although ADR-095:79 makes chat-scope streaming a plain On/Off choice. Labels drift from the other surfaces ('Response max tokens', 'Budget', 'Summary').

The judge's rule 2 gives every editor one row grammar: Label | one-row control sized to its value type | Source word | help line. The Source words and their resolver already exist since Switch model (phase 4); this modal must reuse them rather than keep its own mapping. Spec §6 also requires values, not placeholders: an empty field shows what it inherits. The review measured placeholders at 2.8-3.5:1 and found them indistinguishable from values, and spec §6 puts help and Source text at 4.5:1 or better.

The wide tier (85% capped at 196 columns, _console_panels.tcss:192-203) is pinned by Tests/UI/test_console_settings_geometry.py:62, :168 and :214. The mock is 150 columns.

### Acceptance criteria

- [ ] #1 The Model view shows, in order: the scope line, the MODEL row, the CORE fields (Temperature, Max tokens, Streaming, and the reasoning or thinking controls the model supports), then one-row disclosures for Sampling, Connection, Request estimate and 'Your name in this chat'
- [ ] #2 Every field row shows its label, a one-row control sized to its value type, one Source word and one help line from the shared field table. Labels match the other model surfaces
- [ ] #3 Source words come from the resolver Switch model introduced, with no modal-local mapping
- [ ] #4 An empty field shows the value it inherits and where it comes from; no placeholder looks like a value
- [ ] #5 Help and Source text measure at least 4.5:1 against the modal background in agentic_terminal and one light theme
- [ ] #6 Streaming is one On/Off Select at chat scope (ADR-095:79), replacing the toggle button
- [ ] #7 The 'Advanced generation' Collapsible is gone, and every field keeps its widget id so the Apply and Save tests keep working
- [ ] #8 Focus opens on Temperature. Tab reaches Apply without entering collapsed disclosures, and hidden fields are absent from the Tab order
- [ ] #9 At 211x44 and 235x52 the Model view fits through the footer without scrolling, with disclosures collapsed
- [ ] #10 The modal's width and height come from tokens in core/_variables.tcss only
- [ ] #11 The Context and memory view keeps its content and behaviour, and its tests pass unchanged
- [ ] #12 The existing compact-size reachability tests (Tests/UI/test_console_settings_geometry.py:268 and :314) still pass. No new design work targets sizes below 211x44
- [ ] #13 Tests that pin the old order are rewritten on purpose. In Tests/UI/test_console_session_settings.py: :11888 test_console_settings_modal_focus_order_starts_provider_ends_cancel_and_skips_collapsed, :11916 test_console_settings_keyboard_tab_leaves_provider_results_in_logical_order, :13764 test_console_settings_modal_connection_first_hierarchy_and_title, :13813 test_console_settings_modal_new_and_blocked_disclosures_start_closed, :13880 test_console_settings_modal_tab_order_skips_collapsed_disclosure_children, :13949 test_console_settings_modal_targeted_advanced_control_opens_disclosure and :13976 test_console_settings_modal_restores_non_targeted_disclosure_snapshot. In Tests/UI/test_console_settings_geometry.py: the wide-tier width pins at :62, :168 and :214
- [ ] #14 Paint probes use the production stylesheet and real keypresses to show that every one-row control renders its value. .value-only assertions do not count as evidence

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Chat/console_settings_apply.py
- tldw_chatbook/css/features/_console_panels.tcss
- tldw_chatbook/css/core/_variables.tcss
- Tests/UI/test_console_session_settings.py
- Tests/UI/test_console_settings_geometry.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/tasks/task-30012 - Recompose-Conversation-Settings-around-connection-first-disclosure.md
- backlog/docs/lessons-live-verification.md

## Task 2: Hide fields the provider does not accept behind one summary line (TASK-33006.2)

Task file: backlog/tasks/task-33006.2 - Hide-fields-the-provider-does-not-accept-behind-one-summary-line.md
Depends on: TASK-33001, TASK-33006.1

### Why

C8(1), UI half. For Anthropic, the modal renders Min P, Seed, Presence and Frequency unconditionally (console_settings_modal.py:1976-2017).
- PROVIDER_PARAM_MAP['anthropic'] has none of them (Chat_Functions.py:281-300).
- project_chat_handler_kwargs forwards only mapped keys, with no warning (:1431-1435).
- The data root cause, _supported_console_settings_fields treating every sampler as universal (console_chat_controller.py:774-802), is fixed in P1 by one supported-fields function (capability projection ∩ PROVIDER_PARAM_MAP). This task makes the modal honour it.

Only the reasoning and thinking controls have support treatment today:
- _GENERATION_CONTROL_INPUTS (:848-854).
- PROVIDER_CHOICE_NO_EFFECT_SUFFIX (:846).
- Per-row 'Support not verified' statics (:847).

Constraints:
- ADR-006: an unsupported control is omitted or shows unavailable copy.
- TASK-30012 AC#3: a control is hidden only on authoritative evidence, and unknown support stays available with neutral copy (pinned by Tests/UI/test_console_session_settings.py:5653 test_console_settings_modal_keeps_unknown_support_visible_with_neutral_copy).

### Acceptance criteria

- [ ] #1 For Anthropic, Min P, Seed, Presence and Frequency are not rendered and not in the Tab order. The Sampling disclosure summary reads 'hidden for Anthropic: Min P, Seed, Presence, Frequency (this provider does not accept them)', using display names
- [ ] #2 The same rule hides unsupported reasoning and thinking controls and names them in the same line
- [ ] #3 Support comes from the one shared supported-fields function, and the modal holds no support table of its own
- [ ] #4 A provider with no PROVIDER_PARAM_MAP entry, or a field whose support is unknown, stays visible with neutral help copy in its row (TASK-30012 AC#3). The separate per-row 'Support not verified' statics are gone
- [ ] #5 test_console_settings_modal_keeps_unknown_support_visible_with_neutral_copy (Tests/UI/test_console_session_settings.py:5653) is rewritten to assert the new location of the neutral copy, not deleted
- [ ] #6 Changing the model through pick mode updates the hidden set and the summary line immediately
- [ ] #7 A test through the real apply path shows that applying from Chat settings on Anthropic submits none of the hidden fields
- [ ] #8 A live capture at 211x44 of an Anthropic chat shows the summary line

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Chat/Chat_Functions.py
- tldw_chatbook/Chat/console_chat_controller.py
- backlog/decisions/006-provider-aware-generation-settings.md
- backlog/tasks/task-30012 - Recompose-Conversation-Settings-around-connection-first-disclosure.md
- Tests/UI/test_console_session_settings.py

## Task 3: Fold Connection, Request estimate and name into one-row disclosures (TASK-33006.3)

Task file: backlog/tasks/task-33006.3 - Fold-Connection-Request-estimate-and-name-into-one-row-disclosures.md
Depends on: TASK-33006.1, TASK-33003

### Why

The Connection block opens the modal and fills most of its body at 211x44 (verified-claims C5(4)). It holds:
- Base URL.
- The credential recovery action, whose round trip is the TASK-30010 return contract.
- Test connection and the opt-in paid 1-token test (TASK-30014 AC#1-3).
- The readiness panel.

'Conversation identity' and 'Request estimate' are Collapsibles (console_settings_modal.py:2139 and :2168). Each costs about 7 rows when collapsed (C5(b)); P3 removed the leaked CSS behind that cost.

The judge's mock shows each as a one-row disclosure whose summary already carries the useful value. For example: '▸ Connection · api.anthropic.com · key from env ANTHROPIC_API_KEY · change it in Settings ▸ Providers & Models'. The review also found a 'Base URL' label with no input for cloud providers.

The blocked-chat path must survive the reorder. TASK-30012 AC#2 keeps tuning from competing with an incomplete connection.

### Acceptance criteria

- [ ] #1 Collapsed, Connection is one row that summarises the endpoint host, where the key comes from (saved, the env var's name, or missing; never the key itself) and 'change it in Settings ▸ Providers & Models'
- [ ] #2 Expanded in place, Connection offers what it offers today: Base URL for URL-based providers only (no label without an input), 'Configure credential…', Test connection where a non-generating probe exists, the opt-in paid test with its consent step, and the readiness detail
- [ ] #3 When the chat is Not ready, Chat settings opens with Connection expanded and focus on its recovery action, so setup never starts inside the tuning fields
- [ ] #4 The credential round trip ('Configure credential…' → Settings → back with the draft restored) still works, and its existing tests pass unchanged. This includes test_mounted_suspended_draft_rehydrates_raw_provider_drafts_and_focus at Tests/UI/test_console_session_settings.py:2062
- [ ] #5 Request estimate and 'Your name in this chat' each collapse to one row that shows the current value in its summary
- [ ] #6 A collapsed disclosure measures one row tall in a production-stylesheet harness
- [ ] #7 The existing connection-test and paid-test tests pass unchanged (TASK-30014 AC#1-3 still hold)

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- backlog/tasks/task-30010 - Add-safe-Conversation-Settings-credential-return-contract.md
- backlog/tasks/task-30012 - Recompose-Conversation-Settings-around-connection-first-disclosure.md
- backlog/tasks/task-30014 - Harden-Conversation-Settings-verification-accessibility-and-geometry.md
- Tests/UI/test_console_session_settings.py

## Task 4: Change the model through the switcher's pick mode and delete ConsoleProviderPicker (TASK-33006.4)

Task file: backlog/tasks/task-33006.4 - Change-the-model-through-the-switcher-s-pick-mode-and-delete-ConsoleProviderPicker.md
Depends on: TASK-33004, TASK-33005, TASK-33006.1

### Why

The judge's rule 1 is pairs only: wherever a model is chosen, the user picks a provider·model pair.

In the modal, choosing a provider rebases with model=None (_switch_provider → _rebase_to, console_settings_modal.py:5357-5373). That is the UI path to C7(a); the data root cause is fixed in P1.

ConsoleProviderPicker (console_provider_picker.py, 455 lines) is used only by this modal (:156, :1725). It is also referenced by:
- Tests/Widgets/test_console_provider_picker.py
- Tests/Packaging/test_conversation_settings_boot_closure.py:48-65
- Tests/UI/test_css_parse_cache_modal_probe.py:138
- The generated tldw_chatbook/css/widget_defaults_self.tcss (:1450-1470).
- The boot-CSS snapshot Tests/Performance/boot_budget_snapshots/boot_css_bytes.json (:196, 494 bytes).

P4 provides the switcher's pick-only mode. ADR-031's task-16211 refinement requires dismissal inventory coverage for modal-to-modal launches (Tests/UI/test_console_modal_dismissal.py). Alt on macOS types composed characters unless Option-as-Meta is on (chat_screen.py:1333-1339), so the Change button must work without Alt.

### Acceptance criteria

- [ ] #1 The MODEL row shows the model, the provider's display name, a Source word, the readiness word and the context window, with a '[Change Alt+M]' button
- [ ] #2 Change (the button, or Alt+M inside Chat settings) opens the switcher in pick mode over the modal. Choosing a pair rebases the Chat settings draft to that exact provider and model through the existing controller rebaser, and nothing is applied until Apply
- [ ] #3 Chat settings no longer hosts a provider picker or its own model search, so a provider cannot be chosen without a model. An unlisted or custom model ID can still be chosen through pick mode (TASK-30012 AC#4)
- [ ] #4 Esc in pick mode returns to Chat settings with the draft unchanged and focus on Change. This modal-to-modal path is covered in Tests/UI/test_console_modal_dismissal.py (ADR-031 task-16211)
- [ ] #5 console_provider_picker.py and Tests/Widgets/test_console_provider_picker.py are deleted, and no import or test reference remains; the boot-closure and CSS parse-cache probe lists are updated
- [ ] #6 The generated widget-defaults CSS is regenerated by css/build_css.py, and the boot CSS byte snapshot goes down, never up (ADR-097)
- [ ] #7 Tests that drove the removed pickers are rewritten to drive pick mode
- [ ] #8 console_settings_modal.py's row in Tests/Architecture/test_module_size_ratchet.py (:68) is lowered to the measured size in the same PR

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/Widgets/Console/console_provider_picker.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/css/widget_defaults_self.tcss
- tldw_chatbook/css/build_css.py
- Tests/Widgets/test_console_provider_picker.py
- Tests/Packaging/test_conversation_settings_boot_closure.py
- Tests/UI/test_css_parse_cache_modal_probe.py
- Tests/UI/test_console_modal_dismissal.py
- Tests/Performance/boot_budget_snapshots/boot_css_bytes.json
- Tests/Architecture/test_module_size_ratchet.py
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
- backlog/decisions/097-boot-budget-ratchets.md

## Task 5: Name the modal, its scope and its actions, and add Use saved defaults (TASK-33006.5)

Task file: backlog/tasks/task-33006.5 - Name-the-modal-its-scope-and-its-actions-and-add-Use-saved-defaults.md
Depends on: TASK-33006.1, TASK-33003, TASK-33001

### Why

C1(b) for chats with work: a chat that holds work has no way to adopt newly saved defaults, and nothing in the modal says where defaults live. Untouched chats already converge since phase 1. The ADR-095 amendment of 2026-09-26 gives chats with work 'Use saved defaults'. It rebases the draft onto the saved default chain for the chat's current provider and model: the model profile, then chat_defaults, then the provider. It carries over no deliberate edits, never changes the provider or model, and applies nothing until Apply.

The scope line reads 'Use: this conversation only. Defaults: future provider conversations.' (CONSOLE_SETTINGS_MODEL_SCOPE_COPY, console_settings_modal.py:857-863). It is pinned by Tests/UI/test_console_session_settings.py:12450 test_console_settings_modal_scope_line_names_session_and_default_scopes, which also pins a 'Save as provider defaults' label the code no longer shows. The header 'Conversation settings' (:1648) is pinned at :4407 and :13777. The name also appears in 47 strings across the code, many of them user-visible notices: chat_screen.py:3747-3835 and :15168, and settings_screen.py:12231-12239 and :16976. Log and exception messages use it too.

The footer (:2569-2610) offers Cancel, 'Save as model default', 'Make default for new chats' and 'Apply to this chat'. Ctrl+Enter is bound (:1141) but no button teaches it (report.md), although TASK-30014 AC#5 requires the accelerator to be discoverable. Phase 3 already made the Esc hint describe what Esc does.

The judge made 'adopt defaults' a button because Ctrl+R is banned (ADR-031 rule 2). Other ADR-095 rules that apply:
- Apply never mutates configuration (ADR-095:20).
- 'Save as model default' uses the full field mask (:74-77).

Ctrl+N is unbound in Console and is not a Textual Input binding.

### Acceptance criteria

- [ ] #1 The title reads 'Chat settings · <chat title>' and shows the count of unsaved edits. The scope line reads 'Applies to this chat only · saved with the conversation · defaults live in Settings ▸ Providers & Models (F4)'
- [ ] #2 The footer shows, in order: phase 3's Esc hint, 'Use saved defaults', 'Save as model default', 'Default for new chats (Ctrl+N)' and 'Apply to this chat (Ctrl+Enter)'. Every advertised key works
- [ ] #3 No binding from ADR-031 rule 2 (Ctrl+C, V, X, S, D, Z, A, R or W) is added
- [ ] #4 'Use saved defaults' stages, for the current provider·model, exactly the values a newly created chat on that provider·model would resolve (build_default_console_session_settings: model profile, saved console.provider_defaults, applicable extra sources, chat_defaults, then provider settings). It replaces unapplied edits in the draft, marks each field that differs from the conversation 'edited *', keeps the provider and model, and applies nothing until Apply
- [ ] #5 A parity test shows that 'Use saved defaults' stages the same values as a new blank chat (Ctrl+T) on the same provider and model, including when [console.provider_defaults.<provider>] holds a value
- [ ] #6 'Use saved defaults' is disabled with a visible reason when the draft already equals the saved defaults
- [ ] #7 Apply from Chat settings never writes configuration (ADR-095:20), and 'Save as model default' keeps the full field mask (ADR-095:74-77)
- [ ] #8 An integration test uses the real config writer on a scratch TLDW_CONFIG_PATH. Settings saves new model defaults and a chat with messages keeps its values. 'Use saved defaults' then Apply puts the new values in that conversation's snapshot, and the config file is unchanged by Apply
- [ ] #9 Every user-visible string that names this modal (notices, Settings return-continuation copy, blocker copy, palette entries and help) says 'Chat settings', and a test fails on any user-visible 'Conversation settings' string. Log and exception messages may keep the old name
- [ ] #10 Pinned tests are rewritten on purpose: test_console_settings_modal_scope_line_names_session_and_default_scopes (Tests/UI/test_console_session_settings.py:12450), the header-title assertions at :4407 and :13777, and any test that pins a renamed notice

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/UI/Screens/chat_screen.py
- tldw_chatbook/UI/Screens/settings_screen.py
- backlog/decisions/095-conversation-owned-console-generation-settings.md
- backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md
- backlog/tasks/task-30014 - Harden-Conversation-Settings-verification-accessibility-and-geometry.md
- Tests/UI/test_console_session_settings.py

## Task 6: Context and memory labels clear their inputs and the defaults line needs a defaults action (TASK-33006.6)

Task file: backlog/tasks/task-33006.6 - Context-and-memory-labels-clear-their-inputs-and-the-defaults-line-needs-a-defaults-action.md
Depends on: —

### Why

The Phase 2 capture triage found two defects in the Context and memory view of Conversation settings, at 211x44 and 235x52 (qa/model-config-33002-captures/conversation-settings-context-view-*.txt). origin/dev 89dd84943a has both, so they predate Phase 2.

- The label 'Conversation max tokens' (Widgets/Console/console_settings_modal.py:2255) is exactly as wide as the 23-cell label column (css/features/_console_panels.tcss:67-71). It runs into its input and reads 'Conversation max tokens▊'.
- The defaults-scope line 'Used by future conversations for llama.cpp.' (:3535-3540 at 867762f3ef) shows although this view's footer offers only Cancel and Use for this conversation. It describes defaults actions the view does not have.

task-33006.1 AC#11 keeps this view's content unchanged, and task-33006.5 names only the footer actions, so neither closes these defects.

### Acceptance criteria

- [ ] #1 At 211x44 and 235x52, every Context and memory label has at least one blank cell before its control.
- [ ] #2 The defaults-scope line shows only while a defaults action is visible (Save as provider defaults, Make default for new chats or their Phase 6 successors).
- [ ] #3 The Context and memory view's existing tests pass. Any test that pins the old line is rewritten on purpose and named in the PR.
- [ ] #4 A 211x44 live capture of the Context and memory view is attached to the task notes.

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
- tldw_chatbook/css/features/_console_panels.tcss
- qa/model-config-33002-captures/conversation-settings-context-view-211x44.txt

## Task 7: "A Chat settings view switch opens the new view at its top, not at the other view's scroll" (TASK-33006.7)

Task file: backlog/tasks/task-33006.7 - A-Chat-settings-view-switch-opens-the-new-view-at-its-top-not-at-the-other-views-scroll.md
Depends on: —

### Why

Phase 3's capture triage (2026-09-30) found this in Chat settings at 211x44. Expand Advanced generation, scroll the Model and generation view down, then choose Context and memory. The Context view opens partway down, at Conversation budget, and the Model capacity section is hidden above the fold. A user who does not scroll back up never sees the model window, max tokens or safe input ceiling that the budget rows depend on.

origin/dev 75c06af39a has the same behaviour: it was reproduced live from a scratch detached worktree of that commit with a scratch profile, at 211x44. So it predates Phase 3.

Cause, from reading the code:
- Both views live in one scroll container, #console-settings-body. `_show_settings_view` (Widgets/Console/console_settings_modal.py:3252 at feat/model-config-p3-density) only toggles which sections display, so the body keeps the Model view's scroll offset, clamped to the Context view's height.
- `_show_context_view` (:3838) then focuses the Budget strategy Select. Focus scrolls only far enough to reveal that Select, so the section above it stays hidden.

TASK-33006.1 AC#11 keeps the Context view's content unchanged, and TASK-33006 AC#2 fits the Model view on screen without scrolling. The Context view still scrolls at 211x44, and so does an expanded Model view, so neither criterion closes this.

### Acceptance criteria

- [ ] #1 At 211x44 and 235x52, switching to either view after scrolling the other one opens the new view at its top, or at the scroll position that view was left at. Model capacity is visible when the Context view is first opened after the Model view was scrolled.
- [ ] #2 After the switch, the view's first control still has focus and is visible.
- [ ] #3 A mounted test scrolls one view, switches views, and asserts that the other view's first section is painted. It fails before the fix.
- [ ] #4 A 211x44 live capture of the Context view, taken after the Model view was scrolled, is attached to the task notes.

### References

- tldw_chatbook/Widgets/Console/console_settings_modal.py
