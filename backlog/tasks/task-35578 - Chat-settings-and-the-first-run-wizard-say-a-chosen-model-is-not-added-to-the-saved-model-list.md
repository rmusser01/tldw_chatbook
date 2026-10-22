---
id: TASK-35578
title: Chat settings and the first-run wizard say a chosen model is not added to the saved model list
status: Done
assignee:
- '@claude'
created_date: 2026-10-19 21:50
updated_date: 2026-10-22 20:16
labels:
- console
- first-run
- model-discovery
- model-config
- copy
- ux-review-2026-10-14
dependencies:
- TASK-33641
- TASK-35011
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Finding C6 of the 2026-10-14 Settings UX review: Console's Chat settings and the first-run wizard's model step never persist a provider's saved model list. A user can choose a model a server listed there, but neither surface adds it to that provider's list in the config file's [providers] section, which Settings ▸ Providers & Models shows as the Saved model list. The review rated this BY-DESIGN at P3, and its critique snapshot (.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md) lists C6 among the findings retired as by design. The behaviour stays. What is left is copy: these surfaces should say that their choice does not join the saved list, and where a model can be added to it.

Why it was deferred. The S1 plan (backlog/docs/plan-2026-10-14-s1-discovered-model-default-saved-lists.md, Out of scope) kept C6 out because other streams owned both surfaces:
- console_settings_modal.py had no size headroom, and Phase 9's TASK-33006.19 was reworking its served listings.
- Phase 8 was rewriting the wizard's model step.

The plan said to revisit C6 after Phase 8 and Phase 9 land. Phase 8 has now merged (#3144), and Phase 9 (#3169) is armed in the merge queue. On 2026-10-19 the owner ruled that C6 stays out of S1 and gets this follow-up.

The gap now matters more. After TASK-35011, choosing a discovered model in Settings' Default model does add it to the saved list, so the two surfaces that do not add it need to say so.

Both files are still exactly on their module size ratchet rows at Phase 9's head: console_settings_modal.py is 6,166 of 6,166, and first_run_model_step.py is 997 of 997. The copy has to fit inside those rows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Where Chat settings offers models from a server's listing, it says that choosing one does not add it to the provider's saved model list, and names where in Settings a model can be added to that list
- [x] #2 The first-run wizard's model step says the same for a model chosen from a server's listing
- [x] #3 Choosing a model in Chat settings or in the wizard still leaves the provider's saved model list in the config file unchanged
- [x] #4 console_settings_modal.py and first_run_model_step.py do not grow past their module size ratchet rows
- [x] #5 The Console and First-Run Setup User Guide pages say that a model chosen in Chat settings or in the wizard is not added to the saved model list, and where to add one
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. One shared sentence, SAVED_LIST_NOTE, in Chat/provider_catalog.py (shared provider display data both surfaces already import), naming Settings ▸ Providers & Models, Discover models, Default model, s and Add to model list as Settings_Modules/saved_model_list.py names them.
2. Chat settings: a listing that found models ends its status line with the note (zero models: unchanged). The modal pays the extra line by deleting the dead _snapshot_model helper (no caller since TASK-33006.4); the import goes on the existing one-line provider_catalog import.
3. Wizard model step: the subtitle moves into first_run_model_discovery.show_provider_line, and _render_models repaints it with the note while the rows came from a server's listing. The curated [providers] fallback (already the saved list) is marked "saved" so it shows no note. Net lines in first_run_model_step.py go down.
4. Tests: Console copy (exact line, no note at zero) and a private-profile pick + Make default that leaves [providers] unchanged; wizard subtitle (listed vs saved fallback vs failure) and a private-profile discovered commit that leaves [providers] unchanged. Update existing exact-equality pins of the status line.
5. Lower both ratchet rows to the new sizes; update console.md and First_Run_Setup.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Copy only; neither surface's persistence changed.

- One sentence, `SAVED_LIST_NOTE` in `Chat/provider_catalog.py` (shared provider display data that Chat settings and the wizard already import), names the controls as `UI/Settings_Modules/saved_model_list.py` does: "A pick here is not added to the saved model list. To add one, use Settings ▸ Providers & Models: after Discover models, choose it in Default model and press s, or check it and press Add to model list." No F4, since the wizard has no Settings shortcut.
- Chat settings (AC#1): a Test connection & list models result that found models ends its status line with the note ("2 models listed. A pick here …"). A listing that found none still reads "No models reported". The Connection test toast is unchanged.
- Wizard (AC#2): the Model step's subtitle moved to `first_run_model_discovery.show_provider_line`. `_render_models` repaints it on every render, adding the note only while the rows came from a server's listing. The curated fallback (the `[providers]` saved list itself) is marked `discovery_state = "saved"`, so it reads "Models for *Provider*." alone, as does a failed listing. `on_show` paints the subtitle only when it will re-render: the first version reset it on every show, so going Back and forward to a listed step lost the note. The new test caught that.
- AC#3: no existing test pinned this, so there are two new private-profile tests. One picks a served-only model in the real Console (Test connection, Change, Make default for new chats). The other commits a discovered pick through `SetupWizardContainer.commit_staged_provider_setup` with the real writer. Each asserts that the file's `chat_defaults.model` is the pick and that its `[providers]` table is unchanged.
- AC#4: `console_settings_modal.py` 6,166 → 6,159. The note costs 1 line (its import joined the existing one-line provider_catalog import). The dead `_snapshot_model` helper went (no caller since TASK-33006.4). `first_run_model_step.py` 997 → 990. Both ratchet rows were lowered to those sizes.
- AC#5: `Docs/User_Guide/console.md` (Connection paragraph) and `Docs/User_Guide/First_Run_Setup.md` (before the Model step's status lines) now say this and point to Advanced ▸ Saved model list.

Red/green evidence:
- AC#1 `test_a_listing_says_its_picks_are_not_added_to_the_saved_model_list[listed]` failed `'2 models listed' == '2 models listed. A pick here…'` before the change.
- AC#2 `test_model_step_says_a_listed_pick_is_not_added_to_the_saved_model_list[listed]` failed `'Models for Anthropic.' == 'Models for Anthropic. A pick here…'`.
- AC#3 tests are green on dev behaviour. Each went red on its last assertion (`{'Llama_cpp': ['model-a', 'served-y']} != {'Llama_cpp': ['model-a']}`) under a temporary source edit: the Console one when a `providers` write was injected into the Make-default mutation (`console_settings_defaults`), the wizard one when it was injected into `apply_settings_mutation_to_cli_config`. Both edits were reverted.

Updated exact pins of the status line: four in `Tests/UI/test_console_session_settings.py` and one in `Tests/UI/test_console_endpoint_discovery.py`.

Files: `tldw_chatbook/Chat/provider_catalog.py`, `tldw_chatbook/Widgets/Console/console_settings_modal.py`, `tldw_chatbook/UI/Wizards/first_run_model_step.py`, `tldw_chatbook/UI/Wizards/first_run_model_discovery.py`, `Tests/UI/test_console_settings_model_change.py`, `Tests/UI/test_console_session_settings.py`, `Tests/UI/test_console_endpoint_discovery.py`, `Tests/Wizards/test_first_run_setup_wizard.py`, `Tests/Wizards/test_first_run_setup_integration.py`, `Tests/Architecture/test_module_size_ratchet.py`, `Docs/User_Guide/console.md`, `Docs/User_Guide/First_Run_Setup.md`, `backlog/docs/lessons-testing-evidence.md`.

Lane note: the Console tests are in a whole-file census entry, so a PR runs them. The wizard tests are in `Tests/Wizards` files that PR lanes run only node by node from `derived-artifacts.yml`, so they run on main pushes and nightly only.
<!-- SECTION:NOTES:END -->
