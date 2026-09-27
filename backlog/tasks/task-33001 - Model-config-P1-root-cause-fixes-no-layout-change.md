---
id: TASK-33001
title: 'Model config P1: root-cause fixes, no layout change'
status: Done
assignee: []
created_date: '2026-09-26 11:47'
labels:
  - model-config-redesign
  - phase-1
  - console
  - settings
  - readiness
dependencies: []
references:
  - 'backlog/docs/spec-2026-09-26-model-config-redesign.md'
  - 'qa/model-config-ux-review-2026-09-26/judge-synthesis.md'
  - 'qa/model-config-ux-review-2026-09-26/verified-claims.md'
  - 'qa/model-config-ux-review-2026-09-26/backlog-adr-check.md'
  - 'qa/model-config-ux-review-2026-09-26/report.md'
  - 'backlog/decisions/095-conversation-owned-console-generation-settings.md'
  - 'backlog/decisions/006-provider-aware-generation-settings.md'
  - 'backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md'
  - 'Tests/Architecture/test_screen_size_ratchet.py'
  - 'Tests/Architecture/test_module_size_ratchet.py'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 1 of the model-configuration redesign 'Switchboard with field truth' (backlog/docs/spec-2026-09-26-model-config-redesign.md §8; qa/model-config-ux-review-2026-09-26/judge-synthesis.md §4). Ships as one PR. It fixes the verified root causes under the model-configuration findings before any surface is redrawn, so later phases build on correct data and behaviour.

What it closes (verified at c4225b5d38, re-checked at this worktree's HEAD):
- C7(a): a provider switch borrows the global default model. resolve_effective_chat_configuration ranks chat_defaults.model above the target provider's own model (Chat/console_session_settings.py:1269-1275).
- C8(1), data only: three copies of field support treat every sampler as universal (Chat/console_chat_controller.py:774-802, Chat/console_settings_defaults.py:344, UI/Screens/settings_screen.py:12596), so Anthropic shows four samplers the request silently drops (Chat/Chat_Functions.py:281-300, :1431-1435).
- C3, double append: Provider Test bakes stored evidence into the result (settings_screen.py:15143-15150) and appends fresh evidence again (:15458-15469).
- C8(3): F6 on Settings only toasts, because SettingsScreen (settings_screen.py:2731) has no pane handler and the app falls back to a notice (app.py:20175-20186).
- C1(b) and the C1 first-run gap: the task-177 refresh never converges an untouched chat whose provider already reads Ready (UI/Console_Modules/session.py:3786-3791, :3814), and only the 'Start chatting' exit stages the first-chat handoff (UI/Wizards/FirstRunSetupWizard.py:9936). Owner decision D1, recorded in the ADR-095 amendment of 2026-09-26 (drafted in this worktree with the spec), removes those readiness gates. The spec moved this here from the Chat settings layout phase because it is a root fix with no layout change.
- Palette commands switch provider by a title-cased raw key and never show the model (app.py:1493-1603).
- The review's minor defects on these surfaces that need no layout change (report.md 'Minor observations').

Absorbs TASK-14812: its AC#6 ('cannot retain a model from the previous provider') has regressed, and its In Progress status is stale because every AC is checked.

No CSS, token or widget geometry changes in this phase. Constraints: chat_screen.py has 32 lines of headroom (Tests/Architecture/test_screen_size_ratchet.py:85), console_settings_modal.py has none (Tests/Architecture/test_module_size_ratchet.py:68), and ADR-097 ratchets never rise. Every test and live check that touches configuration uses a scratch TLDW_CONFIG_PATH, never the real profile.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Switching provider on any Console surface never leaves another provider's model in the draft
- [x] #2 The Console draft rebase, the model-default writer and Settings model defaults all give the same answer on which generation fields apply, and that answer matches what the request actually forwards
- [x] #3 Re-running Provider Test on an unchanged draft shows each endpoint fact once
- [x] #4 F6 and Shift+F6 cycle focus through the Settings panes
- [x] #5 An untouched open Console chat follows newly saved defaults after a Settings save or any completing first-run exit, whatever its readiness; chats with messages, edits or user work keep their settings
- [x] #6 No command-palette entry switches provider without its model
- [x] #7 The review's minor model-config defects that need no layout change are fixed: picker count copy, the silent result cap, the URL display break in native terminals, credential keys written for keyless providers, and values hidden on focus
- [x] #8 The phase changes no CSS, design token or widget geometry
- [x] #9 chat_screen.py stays within its screen-size budget, console_settings_modal.py does not grow, and no ADR-097 ratchet value rises
- [x] #10 TASK-14812 is closed as Done with a note naming this phase's fix
- [x] #11 Docs/User_Guide pages updated (settings.md, console.md), including their Verified-against stamps
- [x] #12 ./scripts/preflight.sh passes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Phase 1 shipped as seven root fixes (subtasks TASK-33001.1 to .7, all Done with every AC ticked), one merge of origin/dev, and a final fix wave from the whole-branch review (.superpowers final-review.md, I1-I3 and M1-M3). No CSS, token or widget geometry changed.

**Subtasks and commits**
- TASK-33001.1, provider switch picks the target provider's own model: 26da90a037, c28979b31d. The guard lives in `resolve_effective_chat_configuration` and compares canonical identities, so the Alt+M popover, the Conversation settings modal and every other explicit-provider caller inherit it. TASK-14812 closed Done (its AC#6 holds again).
- TASK-33001.2, one field-support decision: 8ee30099c1, 465f1a5a88, and bca3fd37c1 from the final wave. `supported_generation_fields` (Chat/console_provider_support.py) replaces three drifting projections and is the only field-to-request-key table; the rebase, the model-default writer and Settings' rows all call it.
- TASK-33001.3, Provider Test shows each endpoint fact once: e39f63530f, 8e8a2f309f.
- TASK-33001.4, F6/Shift+F6 cycle the Settings panes: a031a6ac31, 7335d3edad, and d2966e6148 from the final wave.
- TASK-33001.5, untouched open chats follow saved defaults (D1): e9cc14ba3e, fb8fc1854d, and 1c44c83632 from the final wave.
- TASK-33001.6, provider-only palette commands removed: c902f34b69, 68240aba01.
- TASK-33001.7, the review's minor defects: 042422f79b, 16dfb194dd, and a4696bcba6 from the final wave.

**Final fix wave**
- I1, merge of origin/dev 88b61879b9 (e9f9a1bde5). Dev's TASK-32943 had landed a second Settings F6 handler. The merge keeps TASK-33001.4's, deletes dev's `_focus_relative_settings_pane` and the duplicate `shift+f6` Binding, keeps dev's task-32944 j/k guard, and leaves one F6 row and one Shift+F6 row in settings.md. Dev's `test_settings_f6_cycles_rail_detail_inspector` passes on the kept implementation. d2966e6148 pins the control-less inspector stop and adds a static guard for exactly one `shift+f6` and no `f6`. The ChatScreen ratchet row is re-measured on the merged tree (25218/759, under dev's 25363/762). The diagnostic inventory is regenerated from dev's rows (chat_screen.py -2: the two deleted handoff warnings).
- I2 (a4696bcba6): catalog-health warnings outrank the picker's result-cap note on focus.
- M1 (bca3fd37c1): Settings' model-default rows pass the app config, so a `custom-ep:` endpoint is decided by its family, as in Console.
- M2 (1c44c83632): the no-disk sentinel now replaces chat_screen's own `load_settings` binding.
- M3: task-note wording in TASK-33001.2, .5 and .10. Riders: TASK-33001.14 (legacy summarizers ignore `credential_source`, ADR-012), and two phase-1 carry-overs appended to TASK-33002 (local "minimal" reasoning dropped silently; the Test toast's second phrasing and the in-flight line).

**Explicit delegations**
- AC#1: both Console editors and the palette are fixed. The hidden control-bar and first-chat mirror paths that can still pair a provider with no model are TASK-33001.8.
- AC#3: met for the double append and the in-flight line. The reachable-probe toast still states the generation fact in a second phrasing; that belongs to TASK-33002's Test result rows.

**Tests rewritten on purpose** (named per TASK-33001.5 AC#13 and ruling R12)
- TASK-33001.5 AC#13: `test_provider_setup_recovery_keeps_created_persona_prompt` (fixture no longer stubs readiness), and fifteen mounted tests that seeded providers only into `app.app_config` now persist the seed (`Tests/UI/app_factory.persist_seeded_config`); the list is in TASK-33001.5's notes.
- R12 (TASK-33001.7): the zero-width display pins in Tests/UI/test_settings_url_input.py now run on a textual-web test app; the hub's pure-helper pins are kept and annotated.
- Also: TASK-33001.1's two QwenCloud fail-closed tests; TASK-33001.2's `_ANTHROPIC_MODEL_FIELDS`, OpenAI provenance/normalization pins and three Settings copy/save pins; TASK-33001.6's palette, handoff-store and ProductionApp pins; the final wave's three bare-screen Settings unit tests (now set `app_instance = None`).

**Behaviour changes to state in the PR description**
- A keyless save records `credential_source = "none"`. That includes a legacy `custom`/`custom_2` section saved while its env var is unset, so a later export of that variable is ignored (ruling R10).
- Pre-existing, not changed here: the legacy koboldcpp/tabbyapi summarizers ignore `credential_source` (rider TASK-33001.14).
- Removed palette entries: "LLM Provider Management: Switch to …" no longer exists; "change model" opens Conversation settings.

**Known pre-existing reds** (fail on origin/dev or the merge-base too)
- Size ratchets: library_screen and seven module rows (console_chat_controller, mcp_workbench, personas_screen, watchlists_collections_screen, FirstRunSetupWizard, console_settings_modal, console_transcript), each at dev's own measured size.
- Tests/Docs: README and library doc contract checks.
- ADR-126 RecoveryRequired in unwrapped mounted/config tests; ProductionApp writer-gate tests (TASK-33001.11); six roleplay resume-navigation tests (TASK-33001.12).
- Data loss in a keyed legacy section save, TASK-33001.13: shipped in hotfix PR #2847 (merge 8e94a261a5); this branch is rebased onto it. Its `keyless-legacy` row in `test_settings_model_save_keeps_the_stored_key_that_resolves` now expects `credential_source = "none"` (TASK-33001.7), not dev's "environment". Rider TASK-33001.15 (legacy `[API]`-only key vs a recorded "stored") came from the #2847 review.

**Verification (final fix wave, merged tree)**
- The 28 test files this branch changes, run with -n 8 on the merged HEAD and on origin/dev 88b61879b9 (the 26 that exist there, extracted to a scratch tree): HEAD 1720 passed, 568 failed, 31 errors (537 of the 599 carry ADR-126 RecoveryRequired); dev 1344 passed, 574 failed, 31 errors. Comparing junit ids: 0 HEAD-only failures among the tests both runs share; 6 fixed at HEAD (the app.py slack and ChatScreen rows, and 4 Settings/journey tests that are RecoveryRequired on dev); 387 ids exist only at HEAD, of which 2 fail, the ProductionApp writer-gate pair covered by TASK-33001.11.
- Final-wave RED/GREEN and controls: the picker precedence test was red 3/3 before a4696bcba6; the inspector-stop pin fails a filter-out-scroll-bodies mutant; the M2 sentinel fails a mutant that makes the snapshot look disk-loaded; the M1 comparison was red before bca3fd37c1 (the static adapter answered True for Min P on an ollama entry). Tests/Widgets/test_model_search_picker.py 41 passed; the four mounted model-profile Settings tests 5/5 under private_profile_test.
- ./scripts/preflight.sh: rc=0 ("all derived-artifact checks passed", gated Tests/UI census 118/118).
<!-- SECTION:NOTES:END -->
