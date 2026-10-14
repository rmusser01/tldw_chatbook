# Plan: Settings information architecture and page order (stream S3-ia-order)

> **Owner rulings, 2026-10-14.**
> - OD-1: rail map **A** (owner choice), with the group names shown in the option the owner chose: Core, Look & Feel, Console & Agents, Retrieval & media, Data & Privacy, Maintenance ▸ (collapsed), Owned elsewhere ▸ (collapsed).
> - OD-3: **(a)**, one Console Behavior page, re-ordered (owner choice). Sub-ruling: Conversation context and memory stays open (the recommendation; the owner was told and did not object).
> - OD-2: **(a)**, view-only pages stay as `(view)` rows in "Owned elsewhere", each leading with "Open <owner>". This was the recommendation; the owner was told and did not object.

Source: the Settings UX review of 2026-10-14 (critique snapshot `.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md`), reviewed on origin/dev 1ef4fcf4ae.

Tasks: TASK-35016 (Group the Settings rail by what people come to change, and give view-only pages a way to their owner), TASK-35017 (Open Console Behavior on chat defaults and fold agent, capture and replay sections into Advanced).

Source: the verified Settings UX review on origin/dev 1ef4fcf4ae (assessments A–E, adversarial verdicts, completeness critic). Planning base: origin/dev **67136d5dd7**. Since the review, dev has only had backlog commits and a repository-root cleanup (#3102); no Settings code changed. PR #3057 (TASK-33007, Phase 7) was read at **29ca19f10d** and must merge first (owner ruling); check its state with `gh pr view 3057`, since this plan does not record a live status. Code is anchored by **symbol** in a file. Line numbers are left out on purpose.

## Goal

Make the Settings rail and its largest page put first what people come to change:

- **Rail.** Every page Settings can edit sits in a group named for what the user wants to do. Only the four maintenance pages and the seven pages owned by other screens are folded away.
- **View-only pages.** Each view-only page leads to the screen that owns it. It no longer shows the "Planned" roadmap row, and its Inspector line says where to change its settings instead of repeating a roadmap note. The other roadmap rows in those pages' contract tables are S4's (TASK-35019).
- **Console Behavior.** The page opens on chat defaults, not on an expert feature that is off by default. Agent, capture and replay controls sit at the end, in closed one-row disclosures.

Findings closed:

- **D2** (rail IA). The verifier corrected the count: there are **7** "(view)" Domain Defaults pages, not 8.
- **D-shell-design-M2** (the "read-only" claim covering editable pages).
- D-shell-design-M3 (the rail-size guard asserts 31, but the enum has 33) is owned by TASK-35020; this stream adds no category and leaves that guard alone.
- **D3 + B16**, Console part only (page order; "Start here" at row 39 of 44).
- **D6**, Goal runs rows only (TASK-35017 converts them to the shared one-row grammar).

The Providers & Models part of A4/B7/D3 is done by #3057 and is excluded here.

## Assumptions & sequencing

- **#3057 merges first, and both tasks branch from dev after it lands.** Four things depend on it:
  - The settings_screen.py row in Tests/Architecture/test_module_size_ratchet.py (TASK-33007.8, 35,767 at 29ca19f10d). After the merge the file may not grow.
  - The frameless one-row rules for `#settings-console-behavior-card Collapsible`, in css/features/_settings.tcss (TASK-33007.5/.6).
  - The open-state memory for `.settings-advanced-disclosure`: `SettingsScreen._advanced_disclosures_open` and `SettingsScreen.handle_advanced_disclosure_toggled`, plus the `ADVANCED_DISCLOSURE_CLASS`/`advanced_title`/`advanced_disclosure`/`refresh_advanced` pattern in UI/Settings_Modules/providers_models_card.py.
  - The global fallback rows from `settings_field_rows.compose_console_fallbacks` and `refresh_console_fallbacks` (TASK-33007.7).
- **If #3057 slips,** TASK-35016 may be prototyped on dev, because `_category_groups`, the rail collapse code, `_render_domain_category_detail`, `_render_overview_detail` and `on_key` are byte-identical on dev and 29ca19f10d. It still lands after #3057 (it depends on TASK-33007) and re-pins the ratchet row then. TASK-35017 must wait.
- **P8 and P9 restack onto #3057 after it merges.** I checked feat/model-config-p8-first-run (19a9117e9e) and feat/model-config-p9-riders (841361671c). Every difference from #3057 in the S3 functions comes from their older bases (no Dreams/Guardian, an older P7 head). Neither branch changes the rail, Overview's action row, the domain detail renderer or the Console card's order.
  - P9 Task 7 (TASK-33003.14) edits settings_search_index.py. S3 does not touch that file and keeps every Console field id.
  - P9 Task 2 (TASK-33005.15) edits the Overview readiness line, not the action row.
  - P9 Task 3 (TASK-33641.1) changes the contrast of the shared bare-type `Collapsible` frame and of the **Console** rail rules. The Console card's disclosures are frameless under #3057's scoped rules.
  - The expected conflicts are mechanical: the settings_screen.py ratchet row, the census tail plus `MINIMUM_FILES`, and Docs/User_Guide/settings.md. Whoever lands second re-measures the row and keeps both census blocks.
- **TASK-35016 and TASK-35017 are independent PRs.** Either can land first. The second rebases (`git pull --rebase origin dev`, never a merge) and re-measures the ratchet row.
- **TASK-35017 lands before the streams that edit the Console Behavior card.** Its first commit moves `SettingsScreen._render_console_behavior_card` into the new `UI/Settings_Modules/console_behavior_card.py`. TASK-35013, TASK-35015 and TASK-35019 depend on TASK-35017 and edit the card in that module, never in `settings_screen.py`.
- **TASK-35016 lands before TASK-35013, TASK-35014, TASK-35018 and TASK-35021.** It owns the rail groups and toggles, the Domain Defaults "read-only" tooltip (D-shell-design-M2), the view-only pages' "Planned" row and Inspector owner line, and the User Guide's category map and view-only count.
- **The rulings are in** (the block at the top). OD-1 (rail map A, with the group names shown in the option the owner chose) and OD-3 (a) are owner choices. OD-2 (a) and the sub-ruling that Conversation context and memory stays open were recommendations the owner was told about and did not object to. The ACs are written to those rulings.
- **Landing order across the 2026-10-14 Settings streams.** #3057 first. S1 (TASK-35010 → TASK-35011 → TASK-35012) is independent of the shell streams. TASK-35016 and TASK-35017 land before TASK-35013, TASK-35014, TASK-35015, TASK-35018 and TASK-35019. TASK-35020 can land any time: it fixes the red guards at today's truth (33 categories) and owns the rail-size guard; TASK-35016 adds no category and leaves that guard alone. TASK-35021 lands after TASK-35016, and TASK-35022 after TASK-35021. TASK-35023 can land any time after #3057.

## Scope

- **TASK-35016, the rail regroup (OD-1):**
  - Collapse works from group membership instead of the one hard-coded "Domain Defaults" title.
  - The seven view-only pages get an Open \<owner\> button and lose the "Planned" row and the Inspector's roadmap line (OD-2).
  - Overview gains Run setup wizard.
  - The two guards that 1b769eb757 turned red are fixed.
  - The User Guide category map is rewritten.
- **TASK-35017, Console Behavior page order (OD-3):**
  - Goal runs moves next to Agent run budget and uses one-row grammar.
  - Expert sections fold into closed one-row disclosures under an Advanced heading.
  - The card's composition moves into UI/Settings_Modules/.
  - The User Guide Console section and DESIGN.md follow.

## Out of scope (other streams or existing tasks)

- Save-model honesty: D1, D-M1, D5 and D-M4. The one exception is the view-only pages' Inspector "Read-only" roadmap line, which TASK-35016 takes as part of D2.
- D7, including the Dreams/Guardian "Settings mode: Immediate global briefing-gate control" strings: S2's TASK-35014.
- Copy pass (S4): D4, including the Goal runs note's wording. TASK-35017 only shortens the goal labels so that they fit the label column.
- Boolean widget grammar and the rest of D6's row shape (D-M5/B10): S2 (TASK-35015, and TASK-35023 for Speech & TTS's selects). TASK-35017 does D6's one-row shape for Goal runs only.
- The rail as one Tab stop, arrow order following screen order (S5-N1), Ctrl+S, and moving guards into required CI: S5.
- The rest of the settings_screen.py split: TASK-1378 stays open.
- Category-switch remount cost: TASK-33281.
- Focus-intent races: TASK-2831.
- Retrofitting the page-order rule to other categories: no verified finding asks for it.

## Decisions (ruled 2026-10-14)

OD-1 and OD-3 are owner choices; OD-2 and the OD-3 sub-ruling were recommendations the owner was told about and did not object to (the block at the top). Each decision states its ruling; the options not chosen are kept as the record.

### OD-1: the rail map (all 33 SettingsCategoryId members placed)

**Measured rail window.** The 235x52 capture with Domain Defaults expanded (V-shell-design/captures/03-domain-expanded-235.txt) paints rows 12–48 with content still below, so the window is 37 rows. The pane is 8 rows shorter at 211x44, which gives **29 rows**. Each group heading or toggle costs one row. The new fit test (TASK-35016) settles the exact number. If it measures 30, A has one spare row.

**Option A (ruled, owner choice): five open groups and two collapsed groups.**

| Group | State | Members (enum) |
|---|---|---|
| Core | open | OVERVIEW, PROVIDERS_MODELS, WEB_SEARCH, SPEECH_TTS |
| Look & Feel | open | APPEARANCE, THEME, SPLASH_SCREEN |
| Console & Agents | open | CONSOLE_BEHAVIOR, AGENTS, HOOKS, SCHEDULES, DREAMS, GUARDIAN |
| Retrieval & media | open | LIBRARY_RAG, IMAGE_GENERATION, VIDEO_GENERATION |
| Data & Privacy | open | STORAGE, WORKSPACES, TOOL_PROFILES, PRIVACY_SECURITY, NETWORK, PERSONAL_CONTEXT |
| Maintenance ▸ (4) | collapsed | DIAGNOSTICS, ABOUT, INTERNAL_PROMPTS, ADVANCED_CONFIG |
| Owned elsewhere ▸ (7) | collapsed | ARTIFACTS, PERSONAS, SKILLS, WATCHLISTS, WORKFLOWS, MCP_DEFAULTS, ACP_DEFAULTS |

- **Totals.** 4+3+6+3+6+4+7 = 33 categories. At rest the rail needs 5 headings + 22 rows + 2 toggles = **29 rows**, so it fits 211x44 without scrolling and with no spare row. The next category added to an open group makes the rail scroll, and the new fit test then forces that to be a deliberate decision, which is what M3's guard was meant to do.
- **At 235x52.** Maintenance open costs 33 rows and fits. Both groups open cost 40 rows and scroll by 3, which is the user's choice.
- **Pages behind a toggle.** Every editable page is visible except Internal Prompts and Advanced Config, the two expert editors in Maintenance.
- **Cost.** "Settings › Diagnostics › Run setup wizard" (README, pinned by test_product_maturity_phase6_packaging_data_safety.py; app.py; first_run_summary_step.py) still names a real category and button, but Diagnostics now sits behind "Maintenance ▸". Overview's new **Run setup wizard**, `/diag` and the palette's "Setup: Run setup wizard…" cover it, so the copy stays as it is.
- **If the window is smaller than measured.** If the measured rail window at 211x44 is under 29 rows, stop and re-ask the owner before changing the map. The map is the owner's choice, so no member moves without a new ruling.

**Option A2 (not chosen).** As A, but Schedules, Dreams and Guardian get their own "Background features" group. That needs 30 rows, so the rail scrolls by 1 at 211x44.

**Option B (not chosen).** As A, but Maintenance stays open. Every Settings-owned page is visible, and the README path is visible without opening anything. The rail needs 33 rows, so it scrolls by 4 at 211x44 (Maintenance's tail and the toggle sit below the fold) and fits at 235x52. Overview's Run setup wizard becomes optional.

**Option C (minimal; not chosen).**
- Keep today's six groups.
- Move AGENTS to Expert.
- Rename "Domain Defaults" to "Feature defaults" and drop "(read-only ownership contracts)".
- Leave all 13 collapsed.

C closes M2 and part of D2. Console Behavior stays under Interface, and six editable pages stay hidden.

**Ruled: A, with the group names shown in the option the owner chose** (Core, Look & Feel, Console & Agents, Retrieval & media, Data & Privacy, Maintenance ▸, Owned elsewhere ▸). The names are part of the ruling. History only: before the ruling the plan also listed alternative names ("Generation defaults" for Retrieval & media, "Other screens' defaults" for Owned elsewhere, "Agents & automation" for Console & Agents); none of them is open.

### OD-2: the seven view-only pages (Artifacts, Roleplay, Skills, Watchlists, Workflows, MCP Defaults, ACP Defaults)

- **(a) Ruled** (the recommendation; the owner was told and did not object). The pages stay as "(view)" rows in the collapsed "Owned elsewhere" group.
  - Each page leads with **Open \<owner\>**, which lands on the owning screen. Artifacts and Skills land on their Library views: Library takes `{"mode": "artifacts-all"}` and `{"mode": "skills"}` (`LIBRARY_NAV_MODE_TO_ROW_ID`).
  - The "Planned: …" row is removed. The Inspector's "Read-only:" line stops repeating the same roadmap note.
- **(b) Not chosen.** Remove them from the rail, so they are reachable only through category search and deep links.
  - This saves the toggle row (28 of 29 under A).
  - It changes what `test_settings_category_groups_cover_every_category_id_exactly_once` means, and it needs a "no rail row" concept in search and in restore.
- **(c) Not chosen.** Leave them as they are and only add the button. The roadmap copy stays.

### OD-3: the shape of Console Behavior

- **(a) Ruled (owner choice).** One category, re-ordered.
  - One draft, one Save/Revert, the same search rows and the same atomic [console]/[chat_defaults]/[agents] save batch (`SettingsScreen._save_console_behavior_values`).
  - **Open sections, in this order:**
    1. Global fallback defaults
    2. Conversation context and memory
    3. Model thinking (renamed from "Start here")
    4. Status row placement
    5. Chat images
    6. Rail presentation
    7. Composer paste handling
    8. Background effects
  - **Then an "Advanced" heading with closed one-row disclosures, each titled with its state:**
    1. **Agent limits:** Agent run budget, then Goal runs, Parallel agent runs and Agent tool-result display cap.
    2. **Permission summaries:** its instant-apply group, unchanged.
    3. **Selection side chat**
    4. **Local reasoning history:** with its nested Reasoning replay override.
    5. **Exchange capture**
  - The "Conversation context and memory ↓" jump button and the "Also in this category: …" line are deleted. The jump was CM-007's mitigation for that section being buried (Docs/superpowers/qa/console-context-memory-uat-2026-08-10), and it is now second on the page.
- **(b) Not chosen.** Local modes (Chat / Context / Display / Agents) as panes inside the category. Search has to switch modes, and this adds a fourth navigation level. Medium risk.
- **(c) Not chosen.** Split into new rail categories, for example "Agent limits". High risk, because:
  - The enum grows to 34, which costs a rail row (30 under A, so the rail scrolls).
  - New drafts, badges, inspector guidance and search-index keys are needed, and the keys collide with P9's TASK-33003.14.
  - The one atomic save batch splits.
  - ADR-006/ADR-052 keep the fallbacks and memory defaults in Console Behavior.
- **Sub-ruling:** Conversation context and memory. Ruled: stays open.

The page-order rule ("most common task first; expert or newly added controls in closed one-row disclosures under an Advanced heading at the end, each title saying its state") needs no separate ruling. DESIGN.md's Interaction Quality Contract already sets the principle ("Progressive disclosure: keep the common path focused…"), and #3057 applied it to Providers & Models. TASK-35017 records the concrete Settings instance in DESIGN.md.

## Task-by-task design

### TASK-35016: Group the Settings rail by intent and give view-only pages a way to their owner

Closes D2 and D-shell-design-M2.

Files:
- tldw_chatbook/UI/Screens/settings_config_models.py
- tldw_chatbook/UI/Screens/settings_screen.py
- the test files listed under "Pinned tests"
- new Tests/UI/test_settings_rail_groups.py
- scripts/ui_pr_gate_census.txt, plus `MINIMUM_FILES` in scripts/check_ui_pr_gate_census.py
- Tests/Architecture/test_module_size_ratchet.py
- Docs/User_Guide/settings.md

No CSS change.

Mechanism:

1. **The map becomes data in an existing module.** Add these to settings_config_models.py:
   - `RAIL_GROUPS`: the OD-1 table as `(title, ids)` tuples.
   - `COLLAPSED_RAIL_GROUPS`: title → a literal tooltip.
   - `OWNER_JUMPS`: category → (literal button label, route id, context or None). The entries:
     - Artifacts → "Open Artifacts", "library", {"mode": "artifacts-all"}
     - Roleplay → "Open Roleplay", "personas"
     - Skills → "Open Skills", "library", {"mode": "skills"}
     - Watchlists → "Open Watchlists", "watchlists_collections"
     - Workflows → "Open Workflows", "workflows"
     - MCP Defaults → "Open MCP", "mcp"
     - ACP Defaults → "Open ACP", "acp"

   settings_screen.py already imports this module at module scope, so this adds no module and no lazy import, and the UI-ready census and pre-import payload do not move.

   `SettingsScreen._category_groups()` keeps its name and shape, because `_category_group_title`, `_apply_category_search_filter`, `_render_category_buttons` and the tests call it. Its body becomes `return RAIL_GROUPS`. Moving the ~70-line tuple out pays for the code added below.
2. **Collapse follows group membership.**
   - These go: `DOMAIN_DEFAULTS_GROUP_TITLE`, `_domain_group_expanded: bool`, `_domain_group_is_expanded()` and `_domain_group_toggle_label()`. Today they key on `DOMAIN_SETTINGS_CATEGORY_IDS`.
   - They are replaced by `_expanded_rail_groups: set[str]`, `_rail_group_is_expanded(title)` and `_rail_group_toggle_label(title)`, keyed on the group's own members.
   - `_render_category_buttons` and `_apply_category_search_filter` test `title in COLLAPSED_RAIL_GROUPS` instead of `== DOMAIN_DEFAULTS_GROUP_TITLE`.
   - A collapsed group shows its rows when:
     - the user opened it;
     - the active category is a member (navigation, `restore_state`, or a `NavigateToScreen("settings", {"category": …})` deep link);
     - a category search matches a member (today's force-show rule).
   - `handle_domain_group_toggle_pressed` becomes one `@on(Button.Pressed, ".settings-category-group-toggle")` handler. It finds the group whose `_category_group_dom_id(title)` equals the button id. The toggle ids become `settings-category-group-maintenance` and `settings-category-group-owned-elsewhere`.
   - `DOMAIN_SETTINGS_CATEGORY_IDS`, `SETTINGS_DOMAIN_CATEGORY_CONTRACTS` and `DOMAIN_CONTRACT_BY_CATEGORY` stay as they are, because they still drive the inspector, the banner and `_render_domain_category_detail`. Only the rail stops reading them.
   - Tooltips (closes M2):
     - Maintenance's tooltip names its four pages.
     - Owned elsewhere's tooltip may say the pages are view-only, because every member is.
3. **View-only pages lead to their owner.**
   - In `SettingsScreen._render_domain_category_detail`, a category in `OWNER_JUMPS` first yields `Horizontal(Button(label, id=f"settings-owner-jump-{category.value}", classes="settings-owner-jump"), classes="settings-action-row")`. The existing `.settings-action-row Button` rule keeps the button one row tall. The label is a literal, so the markup census gains no row.
   - The `self._detail_row("Planned", contract.follow_up)` line is deleted.
   - In `SettingsScreen._domain_category_ownership_records`, the generic (view-only) branch currently sets `read_only_reason=contract.follow_up`. It changes to a user line such as "Change these in \<owner\>.", so the Inspector's "Read-only:" row stops showing the roadmap note.
   - `follow_up` stays on the contract: `test_settings_domain_category_contracts_are_explicit_about_mutation_scope` asserts it is non-empty.
   - One screen handler on `.settings-owner-jump` posts `NavigateToScreen(route, context)`. This is the precedent already used for `NavigateToScreen("mcp", {...})` in settings_screen.py.
   - Enter on the focused button must press it. `SettingsScreen.on_key` stops Enter only for an id set and for rail rows, so the Button's own binding should fire. The keyboard test decides; add the id to that set only if the test shows Enter is swallowed.
4. **Overview: Run setup wizard.** `SettingsScreen._render_overview_detail` adds `Button("Run setup wizard", id="settings-overview-run-setup-wizard")` to the top `settings-action-row`, beside Backup & Restore. It costs no row at full screen. `handle_run_setup_wizard`'s `@on` takes both ids, and `on_key`'s Enter id set gains the new id if the keyboard test needs it.
5. **The guards that 1b769eb757 (Dreams + Guardian, 2026-10-12) turned red,** both in Tests/UI/test_settings_configuration_hub.py:
   - `test_settings_category_summaries_cover_every_category_id_exactly_once` (31 vs 33) belongs to TASK-35020, which sets it to 33 with the Dreams and Guardian history (1b769eb757; at 2bce8ffa0f the count was 31, HOOKS the 31st). TASK-35016 adds no category and does not edit it.
   - `test_settings_domain_category_contracts_are_explicit_about_mutation_scope` is red by reading the code, not by a run: it expects 11 contracts, and `SETTINGS_DOMAIN_CATEGORY_CONTRACTS` has 13. Add DREAMS and GUARDIAN to its expected and mutable sets (both have `settings_can_mutate=True`). Run it at the merge base first to confirm.
6. **Ratchet.** Measure settings_screen.py at the PR head (`len(path.read_text().splitlines())`) and lower its row to that number. `test_budget_is_not_left_slack` fails at more than 50 lines of slack.

### TASK-35017: Open Console Behavior on chat defaults and fold expert sections into Advanced

Closes D3 (Console part), B16, and D6 for the Goal runs rows.

Files:
- new tldw_chatbook/UI/Settings_Modules/console_behavior_card.py
- tldw_chatbook/UI/Screens/settings_screen.py
- tldw_chatbook/UI/Screens/settings_goal_policy.py
- tldw_chatbook/css/features/_settings.tcss, plus the rebuilt bundle css/screen_agentic_settings.tcss (via tldw_chatbook/css/build_css.py)
- scripts/markup_interpolation_census.tsv
- new Tests/UI/test_settings_console_behavior_order.py
- the census and `MINIMUM_FILES`
- the ratchet row
- DESIGN.md
- Docs/User_Guide/settings.md

Mechanism:

1. **Commit 1: move with no behaviour change** (Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md, "Migration safety").
   - The body of `SettingsScreen._render_console_behavior_card` becomes `compose_console_behavior_card(screen)` in the new module, calling the same `screen._…` helpers as `compose_providers_models_card` does.
   - The `elif category is SettingsCategoryId.CONSOLE_BEHAVIOR:` branch of the detail compose lazily imports it inside the existing `#settings-console-behavior-detail` wrapper, so the pre-import payload does not grow (ADR-097).
   - The unused `compact` parameter goes; its only caller passes `False`.
   - Before pushing, grep Tests/ for `_render_console_behavior_card` and the moved ids (lesson "Tests that read a moved file's source run outside the PR gate"). Docs/superpowers/qa/console-context-memory-uat-2026-08-10/uat_context_memory.py queries the jump button; it is a historical QA script, not a test, so leave it.
   - Re-pin the ratchet row.
2. **Commit 2: re-order to the OD-3 ruling.**
   - Open sections stay direct children of `#settings-console-behavior-card`.
   - Then `Static("Advanced", id="settings-console-advanced-title", classes="destination-section")`, mirroring #3057's `#settings-advanced-title`.
   - Then one disclosure per folded section. Build it with a local `CONSOLE_ADVANCED = {id: (name, summary)}` table and a twin of `advanced_title`/`advanced_disclosure`:
     - `Collapsible(title=Content(f"{name} · {summary(screen)}"), collapsed=id not in screen._advanced_disclosures_open, id=id, classes=ADVANCED_DISCLOSURE_CLASS)`.
     - Import `ADVANCED_DISCLOSURE_CLASS` from providers_models_card.py and do not add code there; P9's plan records zero headroom for that module.
   - The shared class lets #3057's handler remember the open state across rebuilds and category switches. `_land_search_focus_on_field` already opens collapsed ancestors, and the `Expanded` message that posts is remembered too.
   - Each folded section's old `.destination-section` header Static goes; the disclosure title is the header.
   - "Start here" becomes "Model thinking".
   - These are deleted together with their CSS rule:
     - the jump `Button` `#settings-console-context-memory-jump`;
     - its handler `SettingsScreen.handle_console_context_memory_jump`;
     - the `#settings-console-behavior-section-index` Static;
     - the `#settings-console-context-memory-jump` rule in css/features/_settings.tcss. Rebuild the bundle; the rule sits in the lazy Settings sheet, so boot bytes do not move.
3. **Alignment without new CSS.** #3057 already makes every `#settings-console-behavior-card Collapsible` frameless with a one-row title. Its prose rule, `#settings-console-behavior-card > Static`, matches direct children only. Every prose Static that moves inside a disclosure, including the unclassed help lines (for example `#settings-console-sidechat-prompt-template-help` and the goal note), therefore gains `settings-detail-row` (padding 0 1 in css/components/_forms.tcss). `test_console_behavior_prose_starts_where_its_section_headers_do` checks this, because it walks `Contents` children. Do not add a `Contents > Static` rule: it is a bare-type subject. Keeping it out is a convention, not a guarded limit: the ancestor-scoped bare-type census (`MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` in Tests/Performance/test_textual_css_fastpath.py) boots Console only, so it never counts the lazily loaded Settings sheet and would not catch such a rule.
4. **Goal runs use the shared row grammar.**
   - `settings_goal_policy.compose_goal_policy` yields `Horizontal(classes="settings-input-row")` containing `Static(label, classes="settings-input-label")` and `Input(…, classes="settings-goal-policy settings-compact-input")`.
   - These stay as they are: the ids, the `.settings-goal-policy` selector in `SettingsScreen._goal_policy_changed`, `sync_goal_policy` and `normalize_goal_setting`. The note Static gains `settings-detail-row`.
   - `GOAL_SETTING_FIELDS` labels shorten to fit the 24-cell label column, for example "Model calls" and "Time limit (s)". The dropped qualifiers ("including helpers", "including waits") move into the note. S4's copy owner reviews the final words.
   - The block sits inside Agent limits, directly after Agent run budget's rows (D3: the two budgets are now side by side).
5. **Titles carry state.**
   - The summaries return one row. Examples:
     - "Agent limits · goal runs Off"
     - "Permission summaries · applies immediately · Off"
     - "Selection side chat · session model"
     - "Local reasoning history · Automatic"
     - "Exchange capture · Off"
   - A `refresh_console_advanced(screen)` (a twin of `refresh_advanced`: id lookups and plain walks, because it runs on every keystroke) re-sets the titles from three places:
     - the CONSOLE_BEHAVIOR branch of `SettingsScreen._update_draft_status_widgets`, next to `refresh_console_fallbacks`, for edits, saves and reverts;
     - the permission-summary write result path;
     - `handle_console_exchange_capture_apply`'s result path, for instant writes.
6. **Markup census.** The moved Background effects Frame-rate tooltip is an f-string over constants. Under its new module and symbol it would count as a new `tooltip_interp` site and fail preflight. Hoist it to a module constant, then regenerate scripts/markup_interpolation_census.tsv **after reading the diff**: the only change allowed is that the `SettingsScreen._render_console_behavior_card tooltip_interp` row goes away.
7. **Docs and ratchet.** Add the concrete page-order instance to DESIGN.md's "Dense-form control convention (one-row fields)", citing the existing Progressive disclosure principle. Lower the settings_screen.py row to its measured size.

## Pinned tests to update (rewritten on purpose; name each in the PR with its reason)

Test names are as on 29ca19f10d. `Button.press()` returns early when the button's own `display` is False (Textual 8.2.8 source). `Widget.focus()` ignores a closed ancestor. Those two facts decide which tests below break.

TASK-35016 (Tests/UI/test_settings_configuration_hub.py unless noted):

| Test | Change and reason |
|---|---|
| `test_settings_domain_category_contracts_are_explicit_about_mutation_scope` | Add DREAMS and GUARDIAN to its expected/mutable sets (same cause, read from the code). |
| `test_settings_category_navigation_is_grouped_for_scan` | New group titles. |
| `test_internal_prompts_appears_in_settings_sidebar_expert_group` | "Expert" → "Maintenance"; rename the test. |
| Tests/UI/test_settings_about_2775.py::`test_about_category_registered_read_only` | "Troubleshooting" → "Maintenance". |
| `test_video_generation_category_is_final_domain_default_and_count_matches_rail` | Video Gen is last of "Retrieval & media"; check the toggle count per collapsed group. |
| `test_settings_category_search_escape_clears_filter` | Expected visible rows are everything outside the collapsed groups, not everything outside `DOMAIN_SETTINGS_CATEGORY_IDS`. |
| `test_settings_domain_defaults_group_toggle_expands_and_collapses` | Becomes per collapsed group. |
| `test_settings_category_search_reveals_domain_matches` | Toggle id becomes `#settings-category-group-owned-elsewhere`. |
| `_reveal_settings_category_button` (helper behind `_open_settings_category` and `_select_settings_category`) | Presses the toggle of whichever collapsed group hides the target. This keeps the Diagnostics and Advanced Config hub tests and Tests/UI/test_internal_prompts_search_survives_save.py working. |
| `test_search_finds_reduce_motion_with_scope_text`, `test_search_ambiguous_theme_disambiguates_with_scope`, `test_search_next_segment_is_dropped_on_short_terminals` | "(Interface)" → "(Look & Feel)". The longer name can change the echo's wrap in the 30-column rail, so re-measure the short-terminal case rather than editing by delta. |
| Tests/UI/test_first_run_wizard_live_contract.py::`test_full_track_skip_everything_leaves_app_usable` | Its `_open_settings_diagnostics` calls `.press()` on `#settings-category-diagnostics`, which is now hidden, so the press does nothing. Open Maintenance first, or select the category the way the user does. |
| Tests/UI/test_destination_visual_parity_correction.py::`test_settings_advanced_config_controls_use_action_and_status_rows` | Its local `_click_settings_category` clicks Advanced Config without revealing it. Open Maintenance first. |
| Tests/UI/test_settings_category_sweep.py (`_settle_settings`) | Sets `_domain_group_expanded`; switch to opening every collapsed group. |
| Tests/UI/test_settings_narrow_layout.py::`test_all_category_labels_render_fully`, `test_dirty_marker_renders_fully` | Same attribute; same switch. |

Expected to pass unchanged (verify):
- `test_settings_domain_group_expands_when_restored_to_domain_category`
- `test_settings_active_category_uses_explicit_nav_marker`
- `test_settings_domain_category_renders_read_only_owner_contract`
- `test_settings_domain_categories_are_grouped_and_have_ownership_records`: `read_only_reason` stays non-empty.
- Tests/UI/test_settings_personal_context.py::`test_my_profile_is_registered_in_data_privacy_and_settings_contracts`
- Tests/UI/test_settings_speech_tts_panel.py and Tests/UI/test_settings_web_search.py (Core membership)
- Tests/UI/test_settings_panel_scoped_updates.py::`test_same_category_rebuild_keeps_focus_on_the_control_used`
- Tests/UI/test_product_maturity_phase6_packaging_data_safety.py (README path)
- Tests/UI/test_settings_image_gen_panel.py and Tests/UI/test_console_watchlists_mounted_uat.py: their `if not button.display:` branch that presses `#settings-category-group-domain-defaults` is no longer taken. Leave them as they are; P9's wave forbids edits to the image-gen file.

TASK-35017:

| Test | Change and reason |
|---|---|
| Tests/UI/test_settings_context_memory_controls.py::`test_console_memory_controls_mount_stage_and_fit_narrow_settings` (in #3057's census) | The jump button is gone. Reach `#settings-console-context-budget-mode` directly (it is now the second section) and keep every label and fit assertion. |
| Tests/UI/test_settings_agent_budget_journeys.py (the mounted tests that `_edit` budget fields and `_assert_painted` their labels) | The fields move into closed Agent limits. Open it with the keyboard first; that is the user's path now. |
| Tests/UI/test_settings_capture_journeys.py (`_apply` asserts the Apply button is painted; `_choose_viewer`) | Open Exchange capture first. |
| Tests/UI/test_settings_narrow_layout.py::`test_console_capture_controls_are_reachable_at_80x24` | It asserts the capture controls' region heights. Open the disclosure first: "reachable" now means one Enter on its title. |
| `test_settings_console_behavior_saves_max_parallel_runs`, `test_settings_console_behavior_rejects_invalid_tool_result_display_chars` | They would still pass by typing into a field nobody can see. Open Agent limits first, so they use the path a user can reach. |
| `test_settings_console_behavior_focus_auto_scrolls_to_field_guide`, `test_settings_console_behavior_focus_reveals_full_guide_when_purpose_starts_flush_with_bottom_fold` | Their subject is the Inspector's scroll; `.focus()` still reaches `#settings-console-max-parallel-runs` in the closed disclosure. Retarget them to an open field with a multi-row guide, such as `#settings-console-paste-collapse-threshold`, and keep the guide-row count the test measures. |

Expected to pass unchanged, and serving as evidence:
- Tests/UI/test_settings_console_behavior_grammar.py (in the census): `_open_card` opens every Collapsible. The one-width test then also covers the goal Inputs (AC#3), and the prose test covers the moved help lines.
- Tests/UI/test_console_goal_settings.py::`test_mounted_goal_policy_save_revert_and_fleet_independence`: it sets values by id, so it proves folded values still save.
- test_settings_console_side_chat.py and test_settings_console_side_chat_field_width.py.
- test_settings_console_reasoning_history.py::`test_override_controls_show_only_beside_the_model_they_remember`.
- test_settings_permission_summary_journeys.py (its census node sets `control.value`).
- test_settings_state_line_unsaved_count.py.
- test_settings_console_rail_labels.py and test_settings_console_status_row.py.
- test_model_config_field_labels.py.
- test_destination_shells.py and test_destination_visual_parity_correction.py's paste-toggle test (that section stays open).
- The hub tests `test_console_capture_settings_*` and `test_console_trace_viewer_full_uses_explicit_confirmation`.

## New tests and their lane

Both files go into scripts/ui_pr_gate_census.txt under a comment naming the task, and `MINIMUM_FILES` rises by the number of entries added. The census has no fixed position rule; its header records both tail appends and mid-file inserts. Appending conflicts only at the tail.

The UI Fast Lane rules:
- Serial, 9 shards, minimal dependency set (`-e . pytest pytest-asyncio pytest-timeout packaging`; no xdist, no hypothesis), with `--timeout=180`.
- A 60 s per-entry rule.
- Each file carries its own profile marker (`private_profile_test` or `bootstrap_profile`).

Qualify each entry before enrolling it:
1. Pass alone in the scratch wrapper's serial mode.
2. Pass in a scratch venv with exactly the lane's install line (S5 plan's recipe: `uv venv` + `uv pip install`, then `check_ui_pr_gate_census.py --shard`).
3. If a file runs over 60 s, enroll its nodes by id (the TASK-33621.27 precedent).

`Tests/UI/test_settings_rail_groups.py` (TASK-35016) uses the real stylesheet through `_SettingsCssHarness` (test_settings_narrow_layout.py):
- `RAIL_GROUPS` equals the OD-1 table, and every id is in exactly one group (static).
- Parametrised over (211, 44) and (235, 52), at rest: every open-group row and both toggles lie inside `#settings-category-list`'s `scrollable_content_region`, and `max_scroll_y == 0`. Use the painted window, not the pane (lesson TASK-33003.7).
- A collapsed group opens for an active member (restored state and deep link) and for a search hit until the search is cleared.
- No rail heading or tooltip says read-only over a group with an editable page.
- Owner jumps:
  - Static: every `OWNER_JUMPS` route resolves through `screen_registry` to the owner's screen class.
  - Real keyboard landings (focus the button, press Enter, assert `app.screen` and Library's selected row, never the posted message; lesson task-32245) for Artifacts and Skills, the two context routes, and for MCP, one plain route.
  - No "Planned" row on the page, and no roadmap text in the Inspector.
- Overview's Run setup wizard pushes the re-run wizard: monkeypatch `FirstRunSetupWizard` to a stub and assert `rerun=True`.
- The search echo names the new group.

`Tests/UI/test_settings_console_behavior_order.py` (TASK-35017) uses the real stylesheet at (211, 44) and (235, 52):
- At scroll 0, the Global fallback defaults header and its Temperature, Max tokens and Streaming rows lie inside the detail body's `scrollable_content_region`.
- The order of the top-level headers and disclosure ids matches OD-3.
- The goal rows come directly after Agent run budget and are one row each.
- Each closed title is one row and follows a staged edit, `s`, `r`/Discard and an instant permission-summary write.
- An opened disclosure stays open after a save, a revert, and a category switch and back.
- '/' "Max parallel agent runs" Enter opens Agent limits and focuses the field.
- A goal value edited and then folded away is saved by `s`.

## Docs

**TASK-35016, in Docs/User_Guide/settings.md:**
- Rewrite "The category map" to cover all 33 categories (today it omits Hooks, Tool Profiles, Network, Dreams, Guardian and Video Gen).
- Rename the "### \<Group\> — \<Category\>" headings to the new groups.
- Update the layout-tour rail row ("Domain Defaults ▸ (10)") and the search-echo example, and mention Overview's Run setup wizard.
- Replace "Domain Defaults — the eight view-only pages" with "Owned elsewhere — the seven view-only pages": an Open \<owner\> button per page, no "What is still missing" column, and Schedules removed (it is editable).
- Update the badge table's "eight view-only" count to seven.
- Content only; never add "Verified against" paragraphs.
- Inbound anchors from Docs/ and tldw_chatbook/ are only #custom-endpoints, #inference-clouds, #model-makers-own-apis and #provider-setup, and none of them changes. The only internal anchor that changes is the map's link to the view-only table.

**TASK-35017:**
- Re-order the Console Behavior table to match the page.
- Move the misplaced top section "Goal policy in Console behavior" (it names F9, which is not a binding; Settings is F4) into that table under Agent limits, with the path F4 ▸ Console Behavior ▸ Advanced ▸ Agent limits.
- Add the DESIGN.md line described in TASK-35017 step 7.

## Live verification script (keyboard only; sandboxed scratch profile; a fresh tmux session per size)

1. **Profile and launch** (the 2026-10-11 recipe; build PROFILE before HOME is overridden, and keep the single quotes inside the tmux string):
   ```
   SP=<scratchpad>/s3-verify-<task>-<sha>; mkdir -p $SP/home $SP/config $SP/data $SP/cache; touch $SP/baseline
   PROFILE="(version 1)(allow default)(deny file-write* (subpath \"$HOME/.local/share/tldw_cli\") (subpath \"$HOME/.config/tldw_cli\") (subpath \"$HOME/.config/tldw-live\"))"
   cd <worktree> && tmux -L s3-<W> new-session -d -x <W> -y <H> "env -u TLDW_CONFIG_PATH HOME=$SP/home XDG_CONFIG_HOME=$SP/config XDG_DATA_HOME=$SP/data XDG_CACHE_HOME=$SP/cache OPENAI_API_KEY=sk-scratch-not-a-real-key PYTHONPATH=<worktree> sandbox-exec -p '$PROFILE' <repo>/.venv/bin/python -m tldw_chatbook.app"
   ```
   - Do not set `TLDW_CONFIG_PATH` to a file that does not exist yet.
   - Do not redirect stderr.
   - The placeholder key silences the first-run wizard. Never send a message with it.
   - Cold start takes ~45 s.
   - Open Settings with `send-keys F4`; Ctrl+digit and Alt+digit cannot be sent.
   - Answer "Check model lists online?" with "Don't check".
   - Capture with `capture-pane -p` and `-p -e`.
2. **One fresh session per size,** 211x44 then 235x52. Never resize a running session.
3. **Moving around.** Until S5 lands, Tab walks every rail row (D8), so reach the detail pane with **F6** and then Tab inside it. Pause between an editing chord and the typed text that follows it, and tell a value from its placeholder with the `-e` capture (lesson "A tmux key burst loses typed text").
4. **TASK-35016 captures,** at each size:
   - (01) Overview with the rail at rest.
   - (02) `/` `diag` Enter: Maintenance is open and Diagnostics is active.
   - (03) `/` `mcp` Enter: MCP Defaults shows Open MCP and no Planned row; the Inspector has no roadmap line.
   - (04) F6, then Tab to Open MCP, then Enter: the MCP screen.
   - (05) F4, `/` `skills` Enter, F6, Tab to Open Skills, Enter: Library ▸ Skills.
   - (06) F4, `/` `overview` Enter, F6, Tab to Run setup wizard, Enter: the wizard opens. Capture it, then end the session; the wizard ignored tmux keys before.
   - (07, 235x52 only) both collapsed groups open.
5. **TASK-35017 captures,** at each size:
   - (08) Console Behavior at rest.
   - (09) F6, Tab to the Agent limits title, Enter: it opens.
   - (10) Edit Iterations: the title and banner show the edit. Then Esc and `s`; the title shows the saved value.
   - (11) Enter on the Agent limits title to close it, then `/` "Max parallel agent runs" Enter: it opens and the field has focus.
   - (12) Open Permission summaries and change its mode: the title follows. This writes only the scratch config.
6. **Teardown and isolation check:**
   - C-q, then `tmux -L s3-<W> kill-server`.
   - `find ~/.config/tldw_cli ~/.local/share/tldw_cli ~/.config/tldw-live -newer $SP/baseline` must print nothing.
   - Remove $SP.
7. **Covering tests:**
   - Run every touched test file through the scratch wrapper (scratch HOME/XDG/TLDW_CONFIG_PATH plus `--basetemp`) with xdist (`-n 6 --dist loadfile`); run Tests/Architecture with `-p no:xdist`.
   - Run the new files in serial mode, as the lane does.
   - Compare FAILED-name **sets** with a clean worktree at the merge base (a unique scratch name, removed afterwards).
   - Run Tests/Performance/test_boot_css_byte_budget.py, test_ui_ready_module_census.py (with PYTHONPATH set to the worktree), test_screen_preimport_payload_budget.py and test_textual_css_fastpath.py.
   - Run `PYTHON=<venv> ./scripts/preflight.sh` and require rc 0, unpiped.

## Risks & traps

- **zsh `:t` modifier.** `git show "$R:tldw_chatbook/…"` mangles the ref in this shell; write `"${R}:path"`. It happened during this planning.
- **Profile isolation.** Run pytest only through the scratch wrapper; Tests/conftest.py imports the app. `RecoveryRequired('raw_source_selection_changed')` at setup is environmental, so compare failure names, not counts (lessons-testing-evidence). Never bypass Tests/real_profile_guard.py.
- **Hidden controls are reachable in tests only.**
  - `Widget.focus()` and typing work through a closed ancestor.
  - `Button.press()` silently does nothing on a button whose own `display` is False, which is the rail rows in a collapsed group.
  - Back every "the user can see or reach it" claim with a painted-window check, or open the group or disclosure first.
- **Disclosure state.** A Collapsible built without `ADVANCED_DISCLOSURE_CLASS` closes again on every same-category rebuild with the user's focus inside it (the TASK-2831 family). Reuse the class and the set.
- **Detached widgets.** Re-query after anything that rebuilds the pane (lesson "A widget reference captured before a structural recompose is a silent key sink").
- **The markup census keys on module and qualified symbol, so moved code counts as new.** Hoist the Frame-rate tooltip, keep owner labels literal, and read the regenerated census before writing it.
- **The ratchet works in both directions.** The row fails if the file grows and if it is left more than 50 lines slack. Re-measure after every rebase; P8, P9, S4 and S5 all edit the same row.
- **No headroom.** Option A uses all 29 rows at 211x44. The fit test will fail the next category added to an open group. That is intended; the PR that adds the category decides.
- **Toggle ids come from titles.** Renaming "Domain Defaults" changes `#settings-category-group-domain-defaults` everywhere. TASK-35021's rail keyboard code should treat `.settings-category-group-toggle` (a class) as a rail row, not the old id.
- **Arrow order.** `_move_category_focus` walks `_category_summaries()` order, not screen order (S5-N1). The regroup makes the mismatch larger until TASK-35021 lands. S3 does not reorder the summaries, because search tie-breaks read that order.
- **Narrow widths.** Overview's new button sits in an action row that stacks at compact widths, so it can cost a row there. Run test_settings_narrow_layout.py and the Overview tests (lesson "A layout pin proves one width").
- **CSS.** Only TASK-35017 changes CSS, and only by deleting one rule. Edit css/features/_settings.tcss, rebuild, keep `check_bundle_sync` green, add no bare-type subjects, and stay inside the boot CSS headroom.
- **Concurrent sessions.** Use your own worktree. Never run a bare `git stash`. Assign any new backlog id against origin/dev.
- **Captures.** Commit no driver or probe scripts with machine paths, and no capture that contradicts the PR's claims.

## Rollback

Each task is one PR; revert its merge commit.
- Neither task changes a config key, the schema or persisted state. The rail's expanded groups and the disclosure open set are in memory only; neither is in `save_state`.
- A revert of TASK-35016 restores the old groups, the "Planned" rows and the Inspector's roadmap line.
- A revert of TASK-35017 restores the old order and the jump button, and moves the card back.
- After a revert, re-pin the ratchet row to the measured size.
