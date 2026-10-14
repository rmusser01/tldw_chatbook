# Plan: Settings tells the truth about saving, and every form uses one control grammar (stream S2-save-honesty)

> **Owner rulings, 2026-10-14.**
> - OD-3: **(B)**, a StateCheckbox labelled with the setting's name (owner choice). Amend DESIGN.md's toggle bullet.
> - The recommendation was applied for each of the rest; the owner was told and did not object:
>   - OD-1 **(A)**: the count goes on the clean line only.
>   - OD-2 **(A)**: remove Palette limit and Smooth scrolling.
>   - OD-4 **(A)**: B5 is added as an AC on TASK-33005.15, in the same PR that filed this plan.

Source: the Settings UX review of 2026-10-14 (critique snapshot `.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md`), reviewed on origin/dev 1ef4fcf4ae.

Tasks: TASK-35013 (Settings shows which controls save without s, and where each page's saves go), TASK-35014 (Settings drops the controls nothing reads and corrects copy about when changes apply), TASK-35015 (Every Settings form uses one on/off control), TASK-35023 (Speech & TTS selects render one row tall like other Settings selects).

Review: the verified Settings UX review on origin/dev 1ef4fcf4ae (assessments A–E, adversarial verdicts, completeness critic). origin/dev is now 67136d5dd7. The commits since the review only change backlog files and clean up the repository root (path pointers in two tests and two scripts), so no Settings code changed. Every code anchor below was read on PR #3057's head 29ca19f10d, because this stream lands after it.

Findings closed:
- TASK-35013: D1 (the residual the verifier left after rating it BY-DESIGN P2), D-shell-design-M1, D5 (the storage line), D-shell-design-M4.
- TASK-35014: D7.
- TASK-35015: D6 (the on/off controls and their one-row shape), D-shell-design-M5, B10.
- TASK-35023: D6 (Speech & TTS's three-row selects).

Not here, each with its one owner: D-shell-design-M2 (the Domain Defaults "read-only" tooltip) is TASK-35016's, and D6's Goal runs rows are TASK-35017's.

## Goal

On any Settings page, what is on screen tells the user four things:
- which controls save only when they press `s`, and which save the moment they are touched;
- where the page's saves go;
- that every control they see does something;
- what an on/off setting is set to, shown by one control that looks and behaves the same on every page.

ADR-033 rule 3 already requires the first: "Every instant-apply control MUST carry the shared `INSTANT_APPLY_BEHAVIOR_COPY` label inline or in the focused-field inspector". Console Behavior breaks that rule today. This stream does not change any commit model (ADR-033 rule 1). It makes the labels true and makes them come from one table.

## Assumptions & sequencing

**PR #3057 (Phase 7, TASK-33007) merges first.** It was read at 29ca19f10d; check its state with `gh pr view 3057`. This plan builds on what it ships:
- The copy constants `STAGED_SAVE_BEHAVIOR_COPY` ("edits save with s · r reverts") and `INSTANT_APPLY_BEHAVIOR_COPY` ("applies immediately · no Save needed").
- The clean draft badge "No changes · edits save with s" in `SettingsScreen._category_state_banner_text`.
- Every Settings checkbox built as `StateCheckbox` (`Widgets/state_checkbox.py`: ✓ when on, a blank cell when off), guarded by `test_no_settings_panel_builds_a_stock_checkbox`.
- Status row placement as a one-row Select and Chat images as a `StateCheckbox`, both in `_render_console_behavior_card`. Each has help prose that starts "Applies immediately.", but neither sits in an instant-apply group, and the Inspector still says the staged row while either is focused.
- Providers & Models' instant groups (Catalog refresh, Session summary on quit, Custom endpoints) inside closed Advanced disclosures whose titles say "applies immediately" (`providers_models_card.ADVANCED_DISCLOSURES`, `advanced_title`).
- The Console Behavior one-control-column grammar (`Tests/UI/test_settings_console_behavior_grammar.py`, in the UI census as a whole file).
- The `settings_screen.py` size-ratchet row at 35,767 lines. It has zero headroom, and a shrink of more than 50 lines must lower the row (`_SLACK_TOLERANCE_LINES`).

If #3057 changes again before it merges, re-read every anchor below at its merge commit. Lessons: "A finding verified on the review's commit may already be fixed on your base"; "Compare against the branch's merge base".

**P8 and P9 restack onto #3057 after it merges.** Both are still based on an older #3057 commit, so compare their own commits, not their trees. They touch some of this stream's files; see Overlaps. Whichever PR lands second rebases. Never merge dev into a branch.

**The shell streams land first.** TASK-35016 and TASK-35017 (S3) land before T1, T2 and T3:
- TASK-35017's first commit moves `SettingsScreen._render_console_behavior_card` into the new `UI/Settings_Modules/console_behavior_card.py` (as `compose_console_behavior_card`). T1 and T3 depend on TASK-35017 and edit the card in that module, never in `settings_screen.py`. TASK-35017 also folds Permission summaries and Exchange capture into closed Advanced disclosures whose titles carry their state, so T1 re-counts which save-without-`s` groups no title names before writing the State-line count.
- TASK-35016 owns the rail groups, the Domain Defaults "read-only" tooltip (D-shell-design-M2) and the User Guide's category map and view-only count. T1 depends on it because both edit the guide's "How saving works" badge table (TASK-35016 changes its view-only count); T2 depends on it because both edit `_render_domain_category_detail` and the guide's group-named headings.

**Order inside the stream: TASK-35013 and TASK-35014 are independent, TASK-35015 comes after TASK-35014, and TASK-35023 is independent of all three.**
- T1 and T2 can land in either order. The second rebases: both edit `settings_screen.py`, its ratchet row and settings.md, and T1 moves `_category_state_scope_text`, whose Splash line T2 edits.
- T3 waits for T2 so that it does not convert Smooth scrolling, a control T2 deletes, or rewrite the Appearance block twice.
- TASK-35023 shares only `SpeechTTSSettingsPanel` (its `_row` helper) and the Speech CSS rules with T3, which converts Speech's Switches. Whichever lands second rebases.

**Landing order across the 2026-10-14 Settings streams.** #3057 first. S1 (TASK-35010 → TASK-35011 → TASK-35012) is independent of the shell streams. TASK-35016 and TASK-35017 land before TASK-35013, TASK-35014, TASK-35015, TASK-35018 and TASK-35019. TASK-35020 can land any time after #3057: it fixes the red guards at today's truth (33 categories) and owns the rail-size guard; TASK-35016 adds no category and leaves that guard alone. TASK-35021 lands after TASK-35016, and TASK-35022 after TASK-35021. TASK-35023 can land any time after #3057.

Design target: 211x44 and 235x52 only.

## Scope / Out of scope

In scope:
- How each control on a draft page that saves without `s` declares itself.
- The State line's at-rest summary of those controls.
- The Scope Inspector's Save row and storage line.
- The User Guide and DESIGN.md save-model text.
- The two inert Appearance controls.
- The false Splash, Guardian and Dreams copy.
- One on/off control across Settings category forms, drawn in one row, and DESIGN.md's row-shape line.
- Speech & TTS's three-row selects (TASK-35023).

Out of scope, with the reason for each:
- **Abolishing or merging the commit models.** ADR-033 rule 1 rejects it, and this plan does not recommend it.
- **Providers & Models' Saved model list "Save selected" (finding A2).** It writes at once and has no instant-apply hint. Finding A2 (BY-DESIGN P2) belongs to the review's saved-model-list stream, which renames it and adds the hint in the same group, and #3057 AC#9 keeps its behaviour. TASK-35013 names it as that stream's item and does not label it, so the two PRs do not both edit the group.
- **Privacy & Security's encryption actions and the raw-CLI Arm button.** Each encryption action opens a password dialog that states what it will do. Arm only arms for this launch, saves nothing, and already says "Applies immediately." beside it. Neither is a preference that silently persists, which is what ADR-033 rule 3 governs.
- **One leave policy for every draft page, and an "Unsaved: …" header chip.**
  - Speech & TTS's leave modal is deliberate (task-2708, pinned by `test_speech_tts_dirty_banner_names_leave_resolution`).
  - #3057's `settings_quit_guard` covers quitting.
  - TASK-33002.10 (P9 Task 7) covers saying in the app that drafts survive a category switch.
- **The rest of D5:** the inspector's width, the category name shown four times, and the long "Owns:" lists. These are shell-layout items. S2 fixes only the false storage claim.
- **B5:** see OD-4.
- **Theme's `r` = rename, and Theme having no State banner.** Both are BY-DESIGN and pinned (Excluded).
- **D2, D3, D4, D8, D-shell-design-M2 and D-shell-design-M3.** These belong to the review's IA/page-order (TASK-35016 owns M2's tooltip), copy, keyboard/search and findability/guards streams.
- **Goal runs' rows (the Goal runs part of D6).** TASK-35017 rebuilds that section into Advanced ▸ Agent limits and gives it the shared one-row shape.

## Settled, not asked

- **Status row placement and Chat images stay instant.** ADR-033 rule 3 allows instant-apply for low-risk, independently meaningful, trivially reversible preferences, if the control is labelled. Both handlers (`handle_console_status_row_position_toggle`, `handle_console_remote_images_toggle`) are documented as "immediate write, no category draft". TASK-32761 qualified their write queue across leaving the screen. M1's verified fix is the label, so staging them would undo deliberate, tested work for no verified gain.
- **The one Settings boolean widget is a checkbox.** TASK-2827 (Done) chose Checkbox over Switch and Button toggles for Settings, and #3057 made it the state-glyph `StateCheckbox`. The Speech and Agents Switches and the Appearance Button toggles came back later and are regressions against that task's AC#1. OD-3 settled only how the checkbox is labelled.

## Decisions (ruled 2026-10-14)

Each decision records the options considered and the ruling of 2026-10-14 (the block at the top). The tasks are written to the rulings.

### OD-1: How the State line summarises the controls that save without `s` (TASK-35013, D1)

The banner's content row is 120 cells at 211x44. On #3057:
- Console Behavior's clean line is 91 cells and its worst dirty line ("· 99 unsaved · revert with r") is 109.
- Providers & Models' clean line is 118 cells, pinned by `test_providers_models_state_line_names_its_scope_in_one_row`.

Options:
- **(A)** Add a count to the clean line only, counting only the groups that no closed disclosure's title already names. Today that means Console Behavior, for example "· 5 groups save without s" (116 cells).
- **(B)** Add the count on every draft page, and shorten Providers & Models' scope copy to make room. This reverses #3057 AC#4's "who a save reaches" wording.
- **(C)** Leave the banner alone, and put the count in F1's "Save contract:" line and the Inspector's draft-status row.

**Ruled (A)** (the recommendation; the owner was told and did not object). It fits one row everywhere and does not reopen #3057. It warns on the only page where these exceptions are spread among about 20 groups.

### OD-2: What happens to the two inert Appearance controls, Palette limit (themes) and Smooth scrolling (TASK-35014, D7)

No runtime reader has ever existed. An unfiltered grep finds only `settings_screen.py`, `settings_appearance_defaults.py` and the `config.py` template, and the critic confirmed this with `git log -S palette_theme_limit`.

Options:
- **(A)** Remove them from the form, the field guide, `/` search, the Appearance summary, the defaults model and the shipped template. Users' existing keys are left untouched.
- **(B)** Keep them, labelled "(not used yet)".
- **(C)** Wire Palette limit to the command palette's theme list. This is new feature work.

**Ruled (A)** (the recommendation; the owner was told and did not object). Add a control back when something reads it.

### OD-3: How the one Settings checkbox is labelled (TASK-35015, D6/M5/B10)

DESIGN.md's dense-form convention says: "Toggles and switches: always paired with a text-state word ('On'/'Off', 'Enabled'/'Disabled') — the word is the state". That rule dates from the stock checkbox, which painted X in both states. `StateCheckbox` makes the mark itself the state, and that mark can be read in a plain-text capture. Today Settings mixes three labellings:
- **A name label.** About 25 sites: Console Behavior, Goal runs, Hooks, RAG, Privacy, Providers, Workspaces' filter, Guardian's rule editor.
- **A setting name in the 24-cell label column, beside a checkbox with no label, a Switch, a separate On/Off Static, or a Button whose label is the state.** About 30 sites: Appearance (Animations, Reduce motion, ASCII glyphs, Library rail, Width mode, six "Items pane" toggles, Folder Files tree pane), Splash ×3, Speech (12 `_switch` rows and 3 explicit Switches), Agents ×2, Theme editor dark mode.
- **A "Name: on/off" label.** Image Gen Context LLM and Video Gen Confirm cost (`toggle_label`).

Options:
- **(A)** Keep DESIGN.md as written. The setting name goes in the label column, and the checkbox's label is the live On/Off word. That means about 25 conversions the other way, and long names must wrap or be cut to fit the 24-cell column, including the raw-CLI consent wording "Allow raw CLI and Terminal access on this device" (pinned in `test_settings_raw_cli`).
- **(B)** One `StateCheckbox` whose label names the setting, with the mark as the state and no separate word. Amend the DESIGN.md bullet to say so. In a grid whose rows already name the item, the checkbox may keep a short On/Off label; this covers Catalog refresh's per-provider grid (ruled by #3057) and the Image/Video Gen backend rows.

**Ruled (B)** (owner choice). It is the shape #3057 ships and guards, needs no consent copy rewritten, and is the review's own suggestion ("Checkbox with the label on the right"). Amending DESIGN.md is the design-system owner's call, which is why it went to the owner.

### OD-4: Where B5 goes

B5 was verified by reading code on #3057. `_settings_overview_presentation` reads `_resolve_provider_model_for_settings()`, which overlays an unsaved Providers draft (`draft.values["provider"/"model"]`) on the saved chat defaults. The Overview's "Configuration" row and its Status therefore describe a pair that was never saved. Nothing pins this.

Options:
- **(A)** Add an AC to TASK-33005.15 (P9 Task 2, To Do): "the Overview's Configuration row and its Status describe the saved pair, and name an unsaved Providers draft separately".
- **(B)** Give it its own task in this stream.

**Ruled (A)** (the recommendation; the owner was told and did not object). P9 Task 2 already rewires `_resolve_provider_model_for_settings` (TASK-33001.8 AC#3) and `_provider_overview_readiness_status` (TASK-33005.15). Fixing the shown pair without its Status would create a new contradiction, and a second PR on the same two functions would conflict. This changes another program's task, which is why it went to the owner. The AC is on TASK-33005.15 as AC#4.

## Task-by-task design

### TASK-35013: Settings shows which controls save without `s`, and where each page's saves go

Closes D1 (residual), D-shell-design-M1, D5 (storage line) and D-shell-design-M4.

**Mechanism.**

1. **One save-contract module.** Add `tldw_chatbook/UI/Settings_Modules/settings_save_contract.py`. DESIGN.md's One Home Rule puts Settings collaborators in `UI/Settings_Modules/`. The module holds:
   - The bodies of `SettingsScreen._persistence_badge` and `SettingsScreen._category_state_scope_text`, moved verbatim. The screen keeps one-line delegators, because tests call `screen._persistence_badge(...)` and `screen._category_state_scope_text(...)` (`test_settings_personal_context.py`, `test_settings_provider_save_scope.py`, `test_settings_state_line_unsaved_count.py`).
   - **The save-without-`s` table:** for each draft category, the groups whose controls save without `s`. Each entry records the group's control ids, whether a closed disclosure's title already names it, and how it saves: "instant", or "its own Apply button". It holds:
     - Console Behavior: Show model thinking, Status row placement, Chat images, Permission summaries (`PERMISSION_SUMMARY_FIELD_IDS`), and Exchange capture (its own Apply, `#settings-console-exchange-capture-apply`).
     - Providers & Models: the ids `_provider_field_guidance_rows` already treats as instant (`MODEL_CATALOG_FIELD_IDS`, `SESSION_SUMMARY_FIELD_IDS`, `OPENAI_RECONNECT_REVIEW_ID`), plus Custom endpoints, all marked as titled.
   - A lookup from a focused field id to its Save copy.
   - The banner count.
   - The storage line for each category (step 5).
   - The focused-guide rows for the newly guided Console ids.

   The module imports the screen's constants at module top, as `providers_models_card.py` does. `settings_screen.py` imports the module only inside the functions that use it. ADR-097 pins the Settings route's pre-import payload (`Tests/Performance/boot_budget_snapshots/preimport_payload.json`).

2. **Each Console control that saves without `s` declares itself the ADR-033 way (M1).** In `compose_console_behavior_card`, in `UI/Settings_Modules/console_behavior_card.py` (TASK-35017 moves `_render_console_behavior_card` there in its first commit, and this task lands after it):
   - Wrap Show model thinking, Status row placement and Chat images, each with its header, control and help, in a `Vertical(classes="settings-instant-apply-group")`, as the Permission summaries group already is.
   - Give each help `Static` the `settings-instant-apply-hint` class, and start it with `INSTANT_APPLY_BEHAVIOR_COPY` followed by the rest of today's sentence. This replaces #3057's ad-hoc "Applies immediately." prefix and adds no row.
   - Give Exchange capture's help a lead clause naming its Apply button and saying that `s` does not save it.

   #3057's `#settings-console-behavior-card .settings-instant-apply-group` rules already draw no frame and no padding, so no CSS change is expected. `test_console_behavior_draws_one_frame_and_no_boxes` and `test_console_behavior_prose_starts_where_its_section_headers_do` prove it.

   Providers & Models is not re-laid-out: its instant groups already carry the hint and titled disclosures.

3. **The Inspector's Save row follows the focused field.**
   - Console Behavior's inspector has one Save row: the fixed `_detail_row("Save", STAGED_SAVE_BEHAVIOR_COPY)` under "Override rules" in `_render_impact_pane_body`. Give it an identifier, and have `_refresh_console_behavior_field_guidance` set its text from the table lookup. It reads the staged copy when nothing, or a staged field, is focused.
   - Add the Status row, Chat images and Exchange capture ids to the `console_behavior_field_ids` set in `handle_descendant_focus`.
   - Their guide rows come from the module, with the same four-row shape every Console branch returns (Purpose / Consequences / Saved as / Applies). That keeps the composed and refreshed rows aligned (`_split_config_key_row` still lifts "Saved as" into the config-key disclosure).
   - `_provider_field_guidance_rows` reads its instant id set from the same table, so Providers & Models' behaviour does not change.

4. **The State line summarises (OD-1 A).** In `_category_state_banner_text`, the clean branch for a draft category with untitled save-without-`s` groups appends the count after #3057's clean badge. The dirty branch is unchanged, and so is `test_state_banner_dirty_branch_keeps_priority`. The implementer chooses the words, but both lines must stay one 120-cell row.

5. **A true storage line (D5).** `#settings-local-scope-note` in `_render_impact_pane_header` uses one `else` branch for every category. Replace that branch with the module's per-category line. The table covers every `SettingsCategoryId`, following the `_INSPECTOR_GUIDANCE` precedent, so a new category must decide its line.
   - Overview keeps its paragraph, and Theme keeps its line.
   - Categories that write the config file keep "Local-only: saves write your config file.", which is pinned for Providers & Models and Console Behavior.
   - Categories that write somewhere else name the place in plain words, for example Agents (the Agents database), Workspaces (the workspace records), Tool Profiles (your local tool profiles), My Profile (your encrypted profile), and Dreams and Guardian (config plus their own store).
   - Categories whose ownership record has `writes_allowed=False` say that nothing on the page is saved: Diagnostics, About, and the seven view-only pages.
   - Check each page's real write path before writing its line. `_INSPECTOR_GUIDANCE`'s "Affected config" / "Affected data" rows are the starting list.

6. **Docs drift (M4).**
   - **Docs/User_Guide/settings.md:** rewrite "How saving works — five models, eight badges". The table lists every string `_persistence_badge` can return and the clean draft wording, with each category under the badge it shows. On #3057 the table is missing:
     - "Reviewed actions" (Tool Profiles) and "Pending — save with s" (Network);
     - Hooks under Draft, Video Gen under "save/revert below", and Agents, Schedules, Dreams and Guardian under Applies immediately.

     The table's view-only count (eight → seven) is TASK-35016's, which lands first; keep its count when rewriting the table.
   - Correct every "seven draft categories" claim to the code: the Scope Inspector row of the layout table, the `s`/`r` rows of the keys table, and the Quirks bullet "'s' did nothing at all". `s` saves the nine `GUIDED_SETTINGS_MUTATION_CATEGORIES` plus Network (`action_settings_save_category`).
   - Update the layout table's "Local-only: saves write your config file." sentence for the new storage line.
   - **DESIGN.md:** update the "Persistence badge" bullet, which still lists "Managed in editor" and says five models.

**Files.** `UI/Settings_Modules/settings_save_contract.py` (new); `UI/Settings_Modules/console_behavior_card.py` (TASK-35017's module); `UI/Screens/settings_screen.py` (a net shrink, with the ratchet row lowered to the new size if it shrinks by more than 50 lines); `Docs/User_Guide/settings.md`; `DESIGN.md`; the tests below. No CSS change is expected. If one turns out to be needed, edit `css/features/_settings.tcss` and rebuild with `tldw_chatbook/css/build_css.py`.

### TASK-35014: Settings drops the controls nothing reads and corrects copy about when changes apply

Closes D7. The design follows the OD-2 ruling (A). D-shell-design-M2 (the Domain Defaults "read-only" tooltip) is TASK-35016's, which removes that toggle.

**Mechanism.**

1. **Remove the inert controls.** Delete Palette limit (themes) and Smooth scrolling from:
   - the Appearance branch of `_render_detail_pane`;
   - `handle_appearance_palette_theme_limit_changed` and `handle_appearance_smooth_scrolling_changed`;
   - `_appearance_field_selector`'s map, and the `_appearance_field_guidance_rows_base` branches (including Smooth scrolling's false "toggle the checkbox" text);
   - the two `_appearance_invalid_field_key` branches;
   - `_appearance_summary_text` ("Palette limit N");
   - the revert/sync paths;
   - `settings_search_index.py`'s two Appearance rows;
   - from `settings_appearance_defaults.py`: the `SettingsAppearanceDefaults` fields, the `DEFAULT_*` / `MIN_` / `MAX_PALETTE_THEME_LIMIT` constants, validation, load, and the `build_appearance_save_sections` entries;
   - the two lines in `config.py`'s `CONFIG_TOML_CONTENT`.

   `build_appearance_save_sections` deep-copies each existing section before updating it, so a user's `general.palette_theme_limit` and `appearance.smooth_scrolling` stay as they are (`test_build_appearance_save_sections_preserves_unrelated_config`).

2. **Splash.** The Splash line of `_category_state_scope_text` says that changes save as you make them and show at the next launch. That function lives wherever it is when this lands: in `settings_screen.py`, or in T1's module. The "Auto-saved" badge stays.

3. **Guardian and Dreams mode copy.** `_render_domain_category_detail` prints Schedules' wording ("Immediate global briefing-gate control" / "Yes - global gate only") for every `settings_can_mutate=True` contract it renders. Today that means Schedules, Dreams and Guardian.
   - Add two optional fields to `SettingsDomainCategoryContract` (`settings_config_models.py`) for the "Settings mode" and "Writes allowed" rows. Set them on the three contracts, and have the renderer use them. The view-only wording stays the default.
   - `settings_config_models.py` is in the UI-ready module census, so add fields and no imports.

4. **Docs.**
   - Docs/User_Guide/settings.md: the Appearance paragraph; the Splash, Guardian and Dreams wording; and the Quirks bullet "Four Appearance fields are less than they appear". Animations is now read by `Chat/character_expression_playback.py`, and Palette limit and Smooth scrolling are gone.
   - `Tests/UI/COMMAND_PALETTE_TESTING.md`: remove the stale `palette_theme_limit` step.

**Files.** `settings_screen.py` (a net shrink; lower the row), `settings_appearance_defaults.py`, `settings_search_index.py`, `settings_config_models.py`, `config.py`, `Docs/User_Guide/settings.md`, `Tests/UI/COMMAND_PALETTE_TESTING.md`. If T1 has landed, also `settings_save_contract.py`.

### TASK-35015: Every Settings form uses one on/off control

Closes D6 (the on/off controls and their one-row shape), D-shell-design-M5 and B10. The design follows the OD-3 ruling (B). D6's Goal runs rows are TASK-35017's, and Speech & TTS's three-row selects are TASK-35023's.

**Mechanism.**

1. **One on/off control.** Every on/off setting on a Settings category form is a `StateCheckbox` whose label names the setting. Each converted control keeps its id, saved key, value and save model. Its handler moves from `Button.Pressed` or `Switch.Changed` to `Checkbox.Changed`. Space, Enter and a click toggle it.
   - **Appearance.**
     - Animations has no label today, and with `width: auto` it clips its closing glyph (▐X… on dev, ▐✓… after #3057).
     - The Reduce motion and ASCII glyphs Buttons. Carry over P8's TASK-33008.4 tooltip and Purpose copy for Reduce motion.
     - The value Buttons: Library rail, Width mode ("Automatic widths"/"Custom widths"), the six `LIBRARY_READER_DESTINATIONS` "Items pane" toggles, and Folder Files tree pane.
     - Remove `_appearance_bool_label` and `_appearance_media_layout_label`, along with the Buttons' label cycling.
   - **Splash.** The three label-less checkboxes, their `-state` Statics, and `switch_state_label`.
   - **Speech & TTS.**
     - `SpeechTTSSettingsPanel._switch` (12 callers) and the three explicit Switches (`settings-speech-pipeline-aec-enabled`, `settings-speech-realtime-enabled`, `settings-speech-handsfree-acoustic-barge-in`).
     - Update the `isinstance(widget, Switch)` read in its state collection, `@on(Switch.Changed, ".settings-speech-draft-field")`, `_row`'s tooltip-from-label rule (it lists Input/Select/Switch), and the `.settings-speech-stacked … > Switch` CSS selector.
   - **Agents.** `agents-enabled-switch`, and the label-less `agents-override-enabled-checkbox`.
   - **Image Gen and Video Gen.** Context LLM and "Confirm cost before paid generation" drop `toggle_label`'s ": on/off" suffix.
   - **Theme editor.** Dark mode drops its label-column Static and its live "On"/"Off" label (`settings_theme_editor.py`).

   Left as they are, with the reason for each:
   - Catalog refresh's per-provider grid and the Image/Video Gen backend-row boxes are grids whose rows already name the item, which OD-3 (B) allows. The backend rows' geometry was just fixed by TASK-33641.3.
   - Workspaces' "Show archived" is a list filter that already has a name label.
   - Schedules, Dreams and Guardian's Enable/Disable buttons are actions named by a verb, beside a status line, not state-word toggles. TASK-34792 and TASK-33200 own those pages.
   - Modal dialogs are not category forms.

2. **One CSS rule.** Give every Settings `StateCheckbox` the one-row, borderless shape through one settings-owned class, for example `.settings-toggle`. Tokens only: the dimension-literal ratchet is in `Tests/UI/test_component_pattern_governance.py`.
   - It replaces the id list under "Compact toggles retain the full label and glyph…". Where they measure equal, it also replaces the `#settings-providers-models-card Checkbox` and `#settings-console-behavior-card Checkbox` rules.
   - On Console Behavior the class goes on the checkboxes in `compose_console_behavior_card` (TASK-35017's module); this task lands after TASK-35017.
   - Keep the selector made of settings-owned tokens, so that `build_css.py` routes it to the lazily loaded `screen_agentic_settings.tcss` and not to the boot bundle. Never add it to `components/_settings_splash_theme.tcss`, which is in the boot manifest.
   - The ancestor-scoped bare-type census (`MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` in `Tests/Performance/test_textual_css_fastpath.py`) boots on Console, before the Settings sheet loads. It therefore does not count these rules: leave its constant alone, and add no new bare-type subject anyway.

3. **The row shape.** Every converted on/off control takes the one-row shape from step 2. The two field-shape items the review found elsewhere each have one owner and are not done here:
   - **Goal runs:** TASK-35017, which rebuilds `compose_goal_policy`'s rows into Advanced ▸ Agent limits.
   - **Speech & TTS's three-row Selects and Inputs:** TASK-35023, below.

4. **DESIGN.md.** Amend the dense-form convention's "Toggles and switches" bullet to say:
   - one `StateCheckbox` labelled with the setting, the mark is the state;
   - in a grid whose rows name the item, the label may be On/Off;
   - no Switch, and no Button whose label is the state; an action button is named by its verb.

   Add one line on the row shape.

5. **User Guide.** Update every Settings section that describes a converted control. For example: Console Behavior's "An Enabled/Disabled toggle plus Threshold", Splash's switches, Appearance's toggles and Speech's switches.

**Files.** `Widgets/Settings_Widgets/speech_tts_settings_panel.py`, `Widgets/settings_splash_screen_viewer.py`, `Widgets/settings_agents_panel.py`, `Widgets/settings_image_gen_panel.py`, `Widgets/settings_video_gen_panel.py`, `Widgets/settings_theme_editor.py`, `UI/Settings_Modules/console_behavior_card.py` (TASK-35017's module), `UI/Screens/settings_screen.py` (a net shrink), `css/features/_settings.tcss` (rebuild the bundle), `DESIGN.md`, `Docs/User_Guide/settings.md`.

### TASK-35023: Speech & TTS selects render one row tall like other Settings selects

Closes D6's Speech & TTS part. Split out of TASK-35015, because its cause is unknown and it needs its own investigation.

**Mechanism.**

1. **Find the cause first.** Speech's Selects and Inputs already pass `compact=True` and `settings-compact-*`, yet render three rows tall, so about seven fields fit on a 44-row screen (`02-speech-211.txt`). Measure before changing CSS. The suspects are `.settings-speech-input-row.settings-select-row { min-height: $ds-size-3 }` and the error-stack `Vertical` that `SpeechTTSSettingsPanel._row` wraps each field in. `.value = "x"` in Pilot cannot see a field that paints no text (TASK-13154.1), so read painted strips.
2. **Fix it at the cause.** Keep the `settings-compact-*` classes and every field's id, saved key, value and save model. Keep `.settings-speech-stacked` working, per the lesson "Exercise both sides of a form's actual responsive threshold". Tokens only; add no bare-type subject or raw dimension literal.
3. **Overlap.** T3 converts Speech's Switches in the same panel and edits `_row`'s tooltip rule and the stacked `> Switch` selector. Whichever lands second rebases.

**Files.** `Widgets/Settings_Widgets/speech_tts_settings_panel.py` and/or `css/features/_settings.tcss` (rebuild the bundle), whichever holds the cause; the tests below.

## Pinned tests to update (with the reason)

Run every pin below locally before pushing. Use xdist (`-n 6 --dist loadfile`; Tests/Architecture runs serially) through the scratch HOME/XDG/`TLDW_CONFIG_PATH` wrapper, with `--basetemp` in the scratchpad. Most of these files are in no PR lane. Each test whose expectation changes is rewritten on purpose, with a "rewritten on purpose" note that names the finding.

**TASK-35013**
- `Tests/UI/test_settings_configuration_hub.py::test_state_banner_leads_with_persistence_badge`: its `startswith("State: No changes · edits save with s | ")` for Console Behavior gains the count.
- `Tests/UI/test_settings_state_line_unsaved_count.py`:
  - `::test_categories_without_a_draft_keep_todays_badge` checks Console's clean line for equality.
  - `::test_dirty_state_line_renders_without_a_new_row_at_211x44` ends on Console's clean-line equality. Its invariant, that dirty rows are never more than clean rows, still holds.
- `Tests/UI/test_settings_console_behavior_grammar.py::test_status_row_chat_images_and_summary_prompt_read_as_controls`: the help no longer starts "Applies immediately.". Assert the shared hint and its group instead. Its `parent` and `region.x` checks must still pass.
- `Tests/UI/test_settings_save_commit_models.py`:
  - the TASK-1341 module docstring;
  - `test_guidance_row_builders_keep_a_uniform_row_count`, which adds the Console builder.
- Must stay green (verify):
  - `test_settings_inspector_has_no_write_blocked_contradiction`;
  - `test_reassurance_short_line_off_overview`;
  - `test_local_scope_note_is_pinned_outside_scrollable_body`;
  - `test_state_banner_dirty_branch_keeps_priority`;
  - `test_state_banner_text_has_exactly_one_state_segment`;
  - `Tests/UI/test_settings_provider_save_scope.py::test_providers_models_state_line_names_its_scope_in_one_row`;
  - `test_console_display_name_documents_staged_save_in_the_inspector`;
  - the staged Save-row assertions in `Tests/UI/test_settings_console_rail_labels.py` and `Tests/UI/test_settings_console_side_chat.py`;
  - `test_settings_footer_hints.py`'s "Save contract:" assertions.
- `Tests/Architecture/test_module_size_ratchet.py`: lower `settings_screen.py`'s row if it shrank by more than 50 lines.
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` (it pins `settings_screen.py` entries): re-run it. The moved bodies hold no logger calls.
- Re-run `scripts/check_markup_interpolation.py`. Census rows are keyed by function name, so moving code that holds a tooltip or a notify re-keys its row. The moved bodies hold neither. TASK-35017 already removed the card's one `tooltip_interp` row (it hoists the Frame-rate tooltip), so the card edits add no row.

**TASK-35014**
- `Tests/UI/test_settings_appearance_defaults.py`: these 7 tests build or read the two fields:
  - `test_load_appearance_defaults_uses_safe_defaults`;
  - `…_reads_general_web_and_appearance_sections`;
  - `…_coerces_integer_boolean_values`;
  - `…_falls_back_for_malformed_values`;
  - `test_validate_appearance_defaults_accepts_valid_values`;
  - `…_rejects_invalid_values` (its parametrize rows "Palette theme limit" and "Smooth scrolling");
  - `test_build_appearance_save_sections_preserves_unrelated_config`.
- `Tests/UI/test_settings_configuration_hub.py`:
  - `test_settings_appearance_renders_guided_defaults_and_validates`;
  - `test_settings_appearance_revert_restores_loaded_values`;
  - `test_settings_appearance_focused_input_keeps_typed_text_visible`, which uses the Palette limit Input; switch it to Web font size.
- `Tests/UI/test_settings_theme_picker_screen.py`: the `_save_appearance` helper stages its edit through the Palette limit Input. It is used by `test_appearance_save_round_trip_keeps_launch_default` and `test_launch_default_changed_survives_a_later_appearance_save`; switch it to Web font size. `test_appearance_save_sections_carry_no_default_theme` stays green unchanged, because its input still carries the key.
- `Tests/UI/test_settings_field_guide_in_view.py::test_other_guided_categories_keep_the_guide_in_view_on_resize`: the Appearance row of `SHARED_HOOK_CASES` uses the Palette limit field; switch it to another guided Appearance field. Its parametrize id stays "appearance".
- `Tests/UI/test_settings_save_commit_models.py::test_guidance_row_builders_keep_a_uniform_row_count`: its `appearance_field_ids` tuple.
- The search-index tests (`Tests/UI/test_settings_search_index.py`) must stay green.
- No test pins the Splash or Guardian/Dreams strings (verifier grep, re-checked on #3057). Re-grep before editing.

**TASK-35015**
- `Tests/UI/test_settings_console_behavior_grammar.py::test_no_settings_panel_builds_a_stock_checkbox`: extend it, or add the new guard beside it.
- `Tests/UI/test_settings_configuration_hub.py`. These query the Appearance toggles as `Button`, or read their labels:
  - `test_settings_appearance_renders_guided_defaults_and_validates`;
  - `test_settings_appearance_reads_ascii_glyphs_from_fresh_config`;
  - `test_settings_appearance_defaults_ascii_glyphs_off_in_fresh_config`;
  - `test_search_enter_focuses_reduce_motion`;
  - `test_settings_appearance_library_reader_controls_round_trip_all_destinations`;
  - `test_field_search_folder_files_tree_width_guides_to_custom_widths_when_disabled`;
  - `test_search_landing_on_a_row_hidden_for_the_provider_says_so`;
  - `test_settings_keeps_dormant_library_width_through_resize_mode_and_generation`;
  - `test_settings_appearance_revert_restores_loaded_values`.
- `Tests/UI/test_settings_splash_screen_viewer.py`: `switch_state_label` and the `-state` Statics.
- Speech:
  - `Tests/UI/test_settings_speech_tts_panel.py`;
  - `Tests/UI/test_settings_speculative_voice_panel.py`;
  - `Tests/UI/test_audio_cpp_model_library_handoff.py`;
  - `Tests/UI/test_speech_tts_settings_ownership_closeout.py` (`_assert_field_rows_are_labelled` queries `Input, Select, Switch`);
  - `Tests/UI/test_settings_narrow_layout.py` (stacked Speech).

  Check each `Switch` hit, because many are other screens' Switches. For example, `Tests/Backup_Recovery/test_unsaved_editors.py`'s Switch is Speech Studio's, not Settings'.
- `Tests/UI/test_settings_agents_category.py` (`agents-enabled-switch`).
- `Tests/UI/test_settings_theme_editor.py` and `Tests/UI/test_settings_theme_card_contrast.py` (dark mode).
- `Tests/UI/test_settings_console_behavior_grammar.py::test_every_console_behavior_control_shares_one_width_and_shows_whole_labels`: keep it green (TASK-35017 extends it to the goal Inputs).
- Must stay green: `Tests/UI/test_checkbox_height_render.py` (in the census), and `Tests/Performance/test_screen_preimport_payload_budget.py`.

**TASK-35023**
- `Tests/UI/test_settings_speech_tts_panel.py` and `Tests/UI/test_speech_tts_settings_ownership_closeout.py` (`_assert_field_rows_are_labelled`): re-run; update only geometry assertions the fix moves, each with a "rewritten on purpose" note.
- `Tests/UI/test_settings_narrow_layout.py` (stacked Speech): must stay green.

## New tests (and their lane)

Every new file carries `pytestmark = pytest.mark.bootstrap_profile`, or `@private_profile_test` on tests that build or reload app config. Lesson: "Tests/UI `RecoveryRequired` at setup is a profile-selection trip".

Add each new file, or each new node in a file the census lists only by node (for example `test_settings_save_commit_models.py`), to `scripts/ui_pr_gate_census.txt`. Run each one once the way the lane runs it: no local scratch plugin on `PYTHONPATH`, in census order. If a shard nears its cap, add a shard rather than minutes (the comment in `derived-artifacts.yml`).

**TASK-35013**
- `Tests/UI/test_settings_save_contract.py` (new):
  - Every guided category's clean and dirty State lines render in one row at 211x44 and 235x52 with the production stylesheet.
  - Console Behavior's clean line carries the count; Providers & Models' does not.
  - Every `SettingsCategoryId` has a storage-line entry, and no `writes_allowed=False` category's line claims a save.
  - Every badge `_persistence_badge` can return appears in Docs/User_Guide/settings.md's save-model table, with its categories in the same row. The test reads the doc, following the precedent in `Tests/UI/test_library_notes_files_sync_journey.py`.
  - Every registered instant id is inside an instant-apply group with the shared hint, and gets the instant Save row in the Inspector.
  - A negative control: an injected unregistered instant id fails the check.
- New nodes in `Tests/UI/test_settings_save_commit_models.py`, which extend TASK-1341's file:
  - Focusing Status row placement, Chat images and Show model thinking shows the instant Save row, and a staged neighbour shows the staged one.
  - Changing Status row placement and Chat images writes the scratch config at once with no unsaved count, and the write survives removing and recreating Settings. Lesson: "Settings category navigation does not exercise screen departure".
  - Exchange capture's Save row names its Apply button.

**TASK-35014**
- `Tests/UI/test_settings_copy_honesty.py` (new):
  - Appearance renders neither control, and `/` finds neither.
  - A config pre-seeded with both keys opens Appearance, saves a Web font size change, and both keys keep their values. Compare values, not bytes, because the config writer can reformat the file.
  - The Splash scope line names the next launch.
  - Guardian's and Dreams' mode rows differ from Schedules'.

**TASK-35015**
- `Tests/UI/test_settings_boolean_grammar.py` (new), or new nodes in the census-listed `test_settings_console_behavior_grammar.py`:
  - An AST guard: no Settings module builds a `Switch`, or a `StateCheckbox` without a label. It covers the globs the stock-checkbox guard walks. `toggle_label`, `_appearance_bool_label`, `_appearance_media_layout_label` and `switch_state_label` are gone.
  - At 211x44 and 235x52, every converted category's checkboxes paint their whole mark and label, read from `Screen._compositor.render_strips()`. Lessons: "A clipped region check does not prove a control is painted"; "A terminal capture is not evidence about what rendered".
  - Every converted checkbox is one row tall at both sizes, including in Speech's stacked layout.
  - A staged checkbox still adds to the unsaved count; an instant one still writes at once.

**TASK-35023**
- New nodes in `Tests/UI/test_settings_speech_tts_panel.py` (or a new census-listed file), with the production stylesheet: every Speech Select and Input is one row tall, label-left, at 211x44 and 235x52, read from painted strips. A stacked-Speech case covers the other side of the threshold. A negative control (the cause put back) turns it red.

## Docs

- TASK-35013: Docs/User_Guide/settings.md: "How saving works", the layout table's Scope Inspector row, the keys table, and the Quirks bullet. DESIGN.md: the "Persistence badge" bullet. No ADR change: ADR-033 rule 3 already says what T1 does.
- TASK-35014: Docs/User_Guide/settings.md: Appearance, Quirks, Splash, Guardian and Dreams. `Tests/UI/COMMAND_PALETTE_TESTING.md`.
- TASK-35015: DESIGN.md: the dense-form convention's toggle bullet and a row-shape line. Docs/User_Guide/settings.md: every converted control's description.
- TASK-35023: Docs/User_Guide/settings.md only if its Speech section describes the field layout.
- Never add "Verified against" paragraphs. What was verified, and on which branch and date, goes in each task's Implementation Notes.

## Live verification script

Follow the repo's `verify` skill. This is the sandboxed scratch-profile recipe it needs:

```
WT=<this task's worktree>; SCR=<scratchpad>/s2-live; mkdir -p "$SCR/home"
cat > "$SCR/deny.sb" <<'EOF'
(version 1)
(allow default)
(deny file-write* (subpath "/Users/macbook-dev/.config/tldw_cli")
                  (subpath "/Users/macbook-dev/.local/share/tldw_cli")
                  (subpath "/Users/macbook-dev/.config/tldw-live"))
EOF
touch "$SCR/baseline"
ENVV="env -u TLDW_CONFIG_PATH HOME=$SCR/home XDG_CONFIG_HOME=$SCR/home/.config XDG_DATA_HOME=$SCR/home/.local/share XDG_CACHE_HOME=$SCR/home/.cache PYTHONPATH=$WT"
PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
$ENVV $PY -c 'import tldw_chatbook; print(tldw_chatbook.__file__)'   # must print a path under $WT
tmux -L s2 new-session -d -x 211 -y 44 "cd $WT && $ENVV sandbox-exec -f $SCR/deny.sb $PY -m tldw_chatbook.app"
```

- The pin check matters. The venv's editable install can point at another worktree, and only import order wins (lesson: "The venv's editable install points `tldw_chatbook` at a FOREIGN worktree").
- Do not redirect stderr. Expect a cold start of about 45 s, and pause between an editing chord and the text that follows it (lesson: "A tmux key burst loses typed text").
- On first boot:
  - skip the first-run wizard;
  - answer "Don't check" to the online model-list consent;
  - quit with C-q.
- Then copy `$SCR/home/.config/tldw_cli/config.toml` as the post-boot baseline, because the app rewrites its config on boot. Repeat every step below at `-x 235 -y 52`.
- Afterwards:
  - quit with C-q and run `tmux -L s2 kill-server`;
  - check that `find ~/.config/tldw_cli ~/.local/share/tldw_cli ~/.config/tldw-live -newer "$SCR/baseline"` prints nothing;
  - delete the scratch home;
  - keep the captures in the scratchpad.

**TASK-35013**
1. Open F4 Settings ▸ Console Behavior. Capture the State line: it carries the count in one row.
2. Use `/` "status row" ▸ Enter. Capture the Inspector: its Save row says it applies immediately.
3. Change the placement. Diff the config against the baseline within 2 s. The State line still says No changes, and Save reads "— no changes".
4. Leave Settings for Console, then return: the value holds.
5. Repeat steps 2–4 for Chat images and Show model thinking.
6. Focus "Capture future exchanges": the Save row names "Apply exchange capture". Press Esc, then `s`: nothing about capture is saved.
7. Open Providers & Models. The State line is unchanged and one row, and the Advanced titles still say "applies immediately".
8. Check the storage line on Agents, Workspaces, Workflows (view) and Diagnostics.

**TASK-35014**
1. With the app quit, add `palette_theme_limit = 5` under `[general]` and `smooth_scrolling = false` under `[appearance]` in the scratch config, then relaunch.
2. Open Appearance: neither control is there.
3. Change Web font size and press `s`. Diff the config: both seeded keys keep their values, and the font size changed.
4. Open Splash: the State line names the next launch.
5. On Guardian and Dreams, "How this page works" names each page's own scope.

**TASK-35015**
1. Capture these pages at both sizes: Appearance, Splash, Console Behavior, Speech & TTS, Privacy & Security, RAG, Hooks, Agents (edit form), Image Gen, Video Gen and the Theme editor. Every on/off setting reads "✓ Name" or a blank mark and "Name", with no "…".
2. Press Space on each toggle. Staged ones add to the unsaved count, and instant ones write at once.
3. Read the painted strips (`Screen._compositor.render_strips()`) for any row that looks clipped before calling it a defect.

**TASK-35023**
1. Speech's Global defaults show one-row Selects at both sizes. Count the fields on one screen against the review's capture (`02-speech-211.txt`: 7).
2. Narrow the pane until Speech stacks: every field still shows with its label.

## Overlaps

Each item has one owner. Each entry says what this stream shares with it and the order rule.

- **PR #3057 / TASK-33007 (and .6, .7).** The prerequisite. S2 builds on its copy constants, clean badge, `StateCheckbox` conversion, Console control column and Advanced titles. It keeps #3057 AC#4's Providers scope line (OD-1 A) and AC#9's Save selected behaviour. It does not re-lay-out the Providers & Models card.
- **P8 (`feat/model-config-p8-first-run`).**
  - TASK-33008.4 rewrites Reduce motion's tooltip and Purpose copy in the Appearance block that T3 converts; carry P8's words over.
  - TASK-33008.1/.2 add the Console-return line to Providers & Models and edit settings.md.

  Whichever lands second rebases.
- **TASK-33002.10 (P9 Task 7).** It shares `_category_state_banner_text` and the one-row State-line rule. P9 expects to put "switching categories keeps the draft" in the footer, F1 or the guided-action row; if it uses the State line, the two features compete for the same cells. Whichever lands second re-runs the one-row test at both sizes. P9 Task 7 also edits `settings_search_index.py` Console rows, and T2 removes two Appearance rows there.
- **TASK-33002.17 (P9 Task 14).** It adds `test_settings_state_line_unsaved_count.py` to the census; T1 rewrites two pins in that file. Whichever lands second rebases.
- **TASK-33005.15 and TASK-33001.8 (P9 Task 2).** They own `_resolve_provider_model_for_settings` and the Overview Status. B5 goes there (OD-4).
- **P9 Task 10 (TASK-33641.3, Done on the P9 branch; TASK-33003.22).**
  - TASK-33641.3 changes the Image/Video Gen backend-row geometry; T3 leaves those boxes alone and edits only Context LLM and Confirm cost.
  - TASK-33003.22 restores the dimension-literal ratchet that T3's CSS must keep at zero.
- **P9 Task 3 (TASK-33003.15/.16/.17, TASK-33641.1).** Contrast work in `_settings.tcss`, a shared file. Whichever lands second rebases.
- **TASK-33005.13.** Already fixed by #3057 (P9 plan); nothing here.
- **TASK-2827 (Done).** It settled the checkbox as Settings' on/off idiom. T3 restores it where later work regressed.
- **TASK-1341 / TASK-1372 (ADR-033).** T1 applies rule 3 to the controls that skipped it, with no ADR change.
- **TASK-1378 (split `settings_screen.py`).** T1's move of the badge and scope text is one more slice of it, with no conflict.
- **TASK-34792 and TASK-33200 (Dreams / Guardian, In Progress).** T2 edits only their "How this page works" rows, and T3 leaves their gate buttons alone.
- **TASK-34100.13 (wizard Appearance step).** Reduce motion and ASCII glyphs keep writing the same `[appearance]` keys.
- **TASK-32465 (radios) and TASK-33137 (shared settings vocabulary).** Related, not absorbed.
- **The review's sibling streams.**
  - Saved-model-list stream (A2): it owns Save selected's rename and hint.
  - IA/page-order stream (D2, D3): TASK-35016 owns the Domain Defaults "read-only" tooltip (M2) and the guide's view-only count; T1 and T2 land after it. TASK-35017 owns Goal runs' rows and moves the Console Behavior card into `console_behavior_card.py`; T1 and T3 land after it and edit the card there.
  - Copy stream (D4): it rewrites the Agents scope copy "agent_runs.db (SQLite) …", which lives in T1's module if T1 lands first.
  - Findability/guards stream (M1, A7, D-shell-design-M3): it shares `settings_search_index.py`.

## Risks & traps

- **#3057 moves under you.** Re-read the anchors at its merge commit. P8 and P9 are on older #3057 commits, so compare their own commits, not their trees.
- **No PR lane runs most of the pins.** The UI Fast Lane runs only `scripts/ui_pr_gate_census.txt`. Run the pinned list locally through the scratch wrapper; any pytest under Tests/ imports the app via conftest. A local plugin that stamps the profile marker hides a missing marker until CI (PR #3055 incident).
- **Zero ratchet headroom.** `settings_screen.py` cannot grow. New code goes to `Settings_Modules/` and the existing satellites.
- **Moved code.**
  - Keep the delegators.
  - Tests that read a moved file's source run outside the PR gate (TASK-33011 incident).
  - Re-run `test_persistent_diagnostic_inventory.py` and `check_markup_interpolation.py`, whose census rows are keyed by function name.
- **Imports.**
  - The new module is imported inside functions (ADR-097 pre-import payload).
  - `settings_config_models.py` is in the UI-ready census: add fields, no imports.
  - Speech, Agents, Image Gen and the splash viewer are in the Settings pre-import payload. `StateCheckbox` is already imported by most of them, so re-run `test_screen_preimport_payload_budget.py`.
- **CSS.**
  - Edit `css/features/_settings.tcss`, never `css/screen_agentic_settings.tcss`, and rebuild. `./scripts/preflight.sh` checks the bundle is in sync.
  - Use tokens only.
  - Keep new rules out of the boot-bundled `components/_settings_splash_theme.tcss`.
  - Add no bare-type subject.
- **The State line is 120 cells.** T1's count must fit at both sizes, and so must TASK-33002.10's text if it lands there.
- **Instant writes and leaving the screen.** Switching category does not exercise leaving Settings. Prove instant writes across removing and recreating the screen (TASK-32761).
- **Config rewritten on boot.** Compare config values after a post-boot baseline, never hand-authored bytes.
- **Speech's three-row Selects (TASK-35023).** Find the cause before changing CSS. `.value = "x"` in Pilot cannot see a field that paints no text (TASK-13154.1). Keep the `settings-compact-*` classes, and verify both the horizontal and stacked layouts.
- **Markup.** All new copy is static. The Button-label sites T3 removes only shrink the census.
- **zsh.** Write `git show "${REF}:path"`. `$REF:path` applies the `:t` modifier and mangles the ref, which happened during this review.
- **Process rules from the owner.** Fix every review finding (no capped rounds), and reviewers fix what they find. Use xdist. No paperwork ACs.

## Rollback

Each task is one PR, and reverting its merge commit restores the previous behaviour. None of them changes the schema, migrates data or rewrites user config:
- T2 leaves users' `palette_theme_limit` / `smooth_scrolling` keys in place, so a revert brings the controls back with the users' values.
- T3 keeps every control id and saved key, so config written before and after it has the same shape.
- TASK-35023 keeps every Speech field's id and saved key; a revert restores the three-row height only.
