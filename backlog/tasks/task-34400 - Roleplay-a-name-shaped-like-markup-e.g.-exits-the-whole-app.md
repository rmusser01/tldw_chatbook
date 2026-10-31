---
id: TASK-34400
title: 'Roleplay: a name shaped like markup (e.g. [/]) exits the whole app'
status: Done
assignee:
  - '@claude'
created_date: '2026-10-04 14:39'
updated_date: '2026-10-04 18:52'
labels:
  - roleplay
  - bug
  - crash
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
On dev, Roleplay draws several user-controlled names as Textual markup. A character, persona, dictionary, lore book, tag or conversation title named like markup — '[/]' is enough, and an imported card can carry it — raises MarkupError while the screen is drawn. Render-time errors bypass the app's keep-alive (TASK-32533 covers only message handlers), so the whole app exits and every open draft is lost. A name such as '[@click=app.quit]x' also turns into a live click action. Confirmed sinks so far: the Inspector's 'Selected:' line and validation line (personas_inspector_pane.py), the library's Tag filter button label (personas_library_pane.py set_tag_label), every Roleplay toast (personas_screen.py _notify, which is also the crash path of TASK-33790), and the shared header's subtitle ('Editing {name}', '{name} - unsaved'). Found while planning Roleplay frame slice B1 (TASK-33910.2), whose Task 8 carried the fix; split out on the owner's ruling (2026-10-04) so a crash does not wait on a layout slice. A static sweep of every Roleplay module adds any further sinks before the fix lands.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Selecting, editing, filtering by tag or being notified about a character, persona, chat dictionary, lore book, entry key, tag or conversation title named '[/]', '[b]x', '[@click=app.record(x)]N[/]', '[/' or '[TODO] y' never raises and never exits the app; the name paints exactly as typed
- [x] #2 No Roleplay surface turns an untrusted name into a click action: no painted cell carries @click meta, and clicking every painted copy of the name runs nothing (checked against a positive control that does run)
- [x] #3 Every Roleplay toast shows square brackets literally and never raises, including exception text such as a persona save that fails validation
- [x] #4 Saving a persona whose name is over 200 characters leaves the app running and shows a toast (the crash half of TASK-33790; its UX criteria stay with that task)
- [x] #5 Every markup-parsing sink the static sweep confirms in Roleplay code is fixed or listed with a reason in the implementation notes
- [x] #6 A real-app test (TldwCli, Roleplay route) with a character named '[/]' fails on dev before the fix and passes after; each fix has a named mutation that turns a test red
- [x] #7 personas_screen.py's module-size ratchet row moves only by the measured growth of this fix, with the owner's ruling recorded beside it (owner 2026-10-04: raise the limit rather than squeeze unrelated code)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce on dev through the real app (TldwCli, Roleplay route, character named '[/]') and through each kind of name: Tests/UI/test_roleplay_hostile_names.py, bootstrap_profile, failing first.
2. Fix the four known sinks: Inspector 'Selected:' and validation Statics (markup=False), the Tag filter button (literal Content label), Roleplay toasts (_notify markup=False, which also stops TASK-33790's exit), and the shared header subtitle (escape_markup on the name in _header_subtitle_text; the shared DestinationHeader stays markup-on because other screens already escape into it).
3. Static sweep of every Roleplay module (PS, Persona_Widgets, Persona_Modules) for further untrusted-text-to-markup sinks, verified on Textual 8.2.8; fix each confirmed sink and extend the test.
4. Named mutation per fix; ruff check/format; owner-approved raise of the personas_screen.py size-ratchet row to the measured count (owner 2026-10-04: expand the limit, do not contort code).
5. Paired base/head run of the existing Roleplay suites; preflight; PR to dev.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Every Roleplay surface that shows untrusted text now paints it literally, so a name shaped like markup can no longer exit the app or become a click action.

**Cause.** Markup-on widgets received untrusted text. On Textual 8.2.8 a `MarkupError` raised while drawing (compositor, tooltip timer, compose) is outside TASK-32533's keep-alive, which covers only widget message handlers, so the whole app exits. Reproduced on dev 9878fd251a through a real `TldwCli` on the Roleplay route with a character named `[/]`.

**Approach.**
- Owned surfaces become literal: `markup=False` on Statics; `Content`/`Text` for Select/OptionList/SelectionList/RadioButton prompts, Button labels, DataTable cells and tooltips.
- Shared markup-on widgets keep their markup and Roleplay escapes into them with `Utils.input_validation.escape_markup` at the call site:
  - the `DestinationHeader` subtitle;
  - three `EnhancedFileOpen` titles.
  Making those widgets literal would put stray backslashes on other screens that already escape into them.
- The preview transcript is now built from plain segments plus italic spans. The old path ran `escape_markup` and then Rich's `Text.from_markup`; `escape_markup` is written for Textual's parser, so Rich un-escaped a backslash before `[` and crashed on `a\[/]b` and on LaTeX replies.
- The preview controller no longer pre-escapes provider names, since its readout is now literal.
- Scope came from a static sweep of personas_screen.py, Widgets/Persona_Widgets, UI/Persona_Modules and the CCP handlers Roleplay wires in: four finders, then each sink verified on Textual 8.2.8. The sweep's verifiers hit a usage limit, so verification was done inline.

**Sinks fixed.**
- **Toasts** (`markup=False`):
  - PersonasScreen._notify (about 60 callers quoting names, paths and exception text; also stops TASK-33790's exit);
  - CCPCharacterHandler._notify and CCPPersonaHandler._notify;
  - LoadingManager.start_loading and the with_loading success/error toasts;
  - the validate_input and validate_file_import toasts;
  - BuddyManagementCoordinator._apply_and_report and BuddyWorkspaceModal._mark_seen.
- **Statics** (`markup=False`):
  - Inspector: selected-name, validation summary, readiness line;
  - preview status and provider lines;
  - character editor style readout and validation footer;
  - visual-identity pack title.
- **Literal prompts, labels, cells and tooltips:**
  - library Tag button;
  - dropdowns: preview greeting, persona portrait character, TTS voice profile, Buddy review portrait state, Petdex state;
  - visual-identity asset OptionList;
  - alternate-greetings DataTable cells;
  - Chat now and Send-to-draft tooltips;
  - ChatQuestionCard options (shared with Console).
- **escape_markup at Roleplay's call site:**
  - header subtitle (`Editing {name}`, `{name} - unsaved`);
  - file-picker titles (asset label, persona state key, expression state).

**Not fixed here.** The shared `Widgets/enhanced_file_picker.py` has its own sinks: breadcrumbs, tooltips, labels and toasts that quote directory or bookmark names, typed paths and exception text. It is used app-wide, so it is filed as TASK-34401. The Console staged-handoff strip was not swept.

**Tests.**
- `Tests/UI/test_roleplay_hostile_names.py` (22 tests): real app plus mock-tier flows; positive control, a click on every painted copy, and a `@click` meta scan.
- `Tests/UI/test_roleplay_hostile_text_surfaces.py` (81 tests): one surface per test, seven hostile strings; dropdowns are opened, and tooltips are rendered the way the tooltip timer renders them.
- On dev 9878fd251a all 103 fail. On this branch all 103 pass.
- `Tests/UI/test_ccp_handlers.py`'s exact notify call now includes `markup=False`.

**Named mutations.** Each fix is reverted alone (or its `markup=False` flipped to `True`), the owning tests run, then the fix is restored. All 36 go red:

| Mutation | Result |
|---|---|
| inspector-selected-markup | 5 failed |
| inspector-validation-markup | 7 failed |
| inspector-readiness-markup | 7 failed |
| inspector-attach-tooltip-str | 7 failed |
| inspector-start-tooltip-str | 7 failed |
| library-tag-str | 5 failed |
| toast-markup-on | 6 failed |
| subtitle-edit-unescaped | 5 failed |
| subtitle-unsaved-unescaped | 7 failed |
| preview-status-markup | 7 failed |
| preview-readout-markup | 7 failed |
| preview-greeting-str | 2 failed |
| preview-rich-reparse | 2 failed |
| controller-escape | 7 failed |
| editor-style-markup | 7 failed |
| editor-validation-markup | 7 failed |
| editor-greeting-str-cell | 7 failed |
| portrait-str | 7 failed |
| tts-str | 7 failed |
| vi-title-markup | 7 failed |
| vi-option-str | 3 failed |
| buddy-review-str | 7 failed |
| petdex-str | 3 failed |
| question-str | 7 failed |
| ccp-char-notify | 1 failed |
| ccp-persona-notify | 1 failed |
| loading-start | 1 failed |
| loading-success | 1 failed |
| loading-error | 1 failed |
| validation-input | 1 failed |
| validation-file | 1 failed |
| buddy-mgmt-notify | 1 failed |
| buddy-workspace-notify | 1 failed |
| picker-title-state-key | 7 failed |
| picker-title-asset | 7 failed |
| picker-title-upload | 7 failed |

**Regression.** Paired run of the 87 existing test files that import a changed module (with the bootstrap-profile plugin the mounted Roleplay suites need), same command on dev 9878fd251a and on this branch: 128 failed + 6 errors on BOTH arms. Five tests failed only on the branch; each passes when run alone, and three also fail intermittently on dev when run alone (F6 pane focus 2 of 5 runs on dev vs 3 of 5 here, two Console avatar geometry tests 1 of 3 on dev), so all five are pre-existing flakes. Five other tests failed only on dev.

**Ratchet.**
- The `personas_screen.py` row goes 16,525 → 16,528 (+3: the import, a comment, a wrapped docstring line). It is raised with a dated comment per the owner's ruling to raise the limit rather than squeeze unrelated code.
- The 12 other size-ratchet failures in Tests/Architecture are identical on dev and on this branch, and none involves a touched file.

**Files.**
- **Production:**
  - UI/Screens/personas_screen.py;
  - Widgets/Persona_Widgets: personas_inspector_pane, personas_library_pane, personas_preview_pane, personas_character_editor_widget, personas_character_tts_widget, persona_profile_editor_widget, personas_visual_identity_pack_widget, buddy_character_review, petdex_import_review, buddy_workspace_modal;
  - UI/Persona_Modules/personas_preview_controller;
  - UI/CCP_Modules: ccp_character_handler, ccp_persona_handler, ccp_loading_indicators, ccp_validation_decorators;
  - UI/Navigation/buddy_management;
  - Widgets/Chat_Widgets/chat_question_card.
- **Tests:** the two new modules, Tests/UI/test_ccp_handlers.py and Tests/Architecture/test_module_size_ratchet.py.
- **Docs:** a lessons-textual.md entry.
<!-- SECTION:NOTES:END -->
