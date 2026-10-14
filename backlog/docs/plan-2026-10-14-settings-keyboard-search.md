# Plan: Settings keyboard path and "/" setting search (stream S5-keyboard-search)

> **Owner ruling, 2026-10-14.** OD1 **(a)**: curated Providers & Models phrases ("discover models", "model discovery", "test provider") rank above Dreams. This was the recommendation; the owner was told and did not object.

Source: the Settings UX review of 2026-10-14 (critique snapshot `.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md`), reviewed on origin/dev 1ef4fcf4ae.

Tasks: TASK-35020 (Settings search finds every rendered setting, and its drift guards fail pull requests), TASK-35021 (Tab goes rail to page to inspector: the Settings rail is one Tab stop), TASK-35022 (Ctrl+S saves a Settings page from inside its fields).

**Source.** The 2026-10-14 Settings UX review, run on origin/dev 1ef4fcf4ae. It covered assessments A to E, with adversarial verification of A, C and D.

**Base.** origin/dev is now 67136d5dd7, 8 commits later. Those commits touch only:
- backlog task files;
- the root-file cleanup, which moved `qa/` to `Docs/QA/`;
- two comment path pointers.

No anchor below moved. Code is anchored by symbol and file, never by line number.

## Goal

A keyboard-only Settings user should be able to:
- reach a page from the rail in one Tab and come back in one Shift+Tab;
- move along the rail with the arrow keys in the order shown;
- save a page without first escaping the field they are typing in;
- find any rendered setting by typing its name after "/".

The guards that keep "/" complete (the search-index drift guard and the rail category count) are green. They fail the pull request that breaks them, not a workflow no PR runs.

## Findings this stream closes

| Id | Verdict | Verified cause | Task |
|---|---|---|---|
| A-pm-design-M1 | Found by the A verifier; node run red | 29 rendered value controls have no `FIELD_SEARCH_INDEX` row (`tldw_chatbook/UI/Screens/settings_search_index.py::build_field_search_index`). By category: console-behavior 12, providers-models 5, speech-tts 4, dreams 3, privacy-security 2, web-search 2, theme 1. That is 7 categories, not the 8 the verifier wrote. `Tests/UI/test_settings_search_index.py::test_every_rendered_setting_is_in_the_search_index` names them. It runs only in `.github/workflows/test.yml`, which triggers on push to `main` and on dispatch. `scripts/ui_pr_gate_census.txt` does not list it. | T1 |
| A7 | CONFIRMED | The session-summary Checkbox and Input are unindexed value widgets. Discover models has no row, because it is an action outside `VALUE_WIDGET_TYPES`. "/discover" opens Dreams because Dreams' description ("daily discovery digest") matches at the description tier (rank 2). In `SettingsScreen._category_search_rank` that tier outranks both field tiers (3 and 4). The discovered list is excluded on purpose. | T1 |
| D-shell-design-M3 | Found by the D verifier; node run red | `test_settings_category_summaries_cover_every_category_id_exactly_once` asserts `== 31`, but `SettingsCategoryId` has 33 members. The guard was last edited at 2bce8ffa0f, at 31 with HOOKS; Dreams and Guardian (1b769eb757) then took the enum to 33. #3057 still asserts 31. | T1 |
| D heuristic 6 | Review note | Search has no synonyms. "dark mode" and "shortcut" both return 0 matches. | T1, "dark mode" only |
| D8, Tab half | CONFIRMED, live | Rail rows are separate focusable `Button`s (`SettingsScreen._render_category_buttons`, class `settings-category-button`). `SettingsScreen.on_key` handles Tab only from nothing focused or from the nav bar. | T2 |
| B17 | B had no verifier | The `VerticalScroll` body `#settings-detail-pane-body` is a Tab stop placed before its own controls, because Textual adds a container to the chain ahead of its children. Only one edge of the outline shows. | T2 (reproduce first) |
| S5-N1 | New; inferred from code, not seen live | `SettingsScreen._move_category_focus` walks `_filtered_category_values()`. That list follows `_category_summaries()` order (HOOKS first, CONSOLE_BEHAVIOR after PERSONAL_CONTEXT, Domain Defaults before Diagnostics), not the rail's group order, and it is not limited to shown rows. Textual's `Widget.focusable` checks visibility, not `display`, so `.focus()` on a `display = False` row succeeds. | T2 (RED test first) |
| D8, in-field save half | "Nothing explains s" REFUTED; residual CONFIRMED | The footer already reads "Esc, s save category" (task-1560), and s typing into a field is pinned (task-1367). The confirmed residual: no save key works inside a field, and an integer field silently drops "s". | T3 |

## Assumptions & sequencing

- **PR #3057 (TASK-33007, Phase 7) merges first.** It was still open at 29ca19f10d during review. Everything here is designed against that head and must be re-measured on post-merge dev. What #3057 changes that S5 relies on:
  - **Search index.** It indexes `settings-provider-search` as Provider and `model-search-picker-input` as Model. It adds `settings-snapshot-enabled` and `settings-snapshot-keep`, and renames "Credential env var" to "Env var".
  - **`NON_SETTING_CONTROLS`.** It gains the hidden adapters `settings-provider-value` and `settings-model-value`, plus `settings-test-provider`.
  - **The Providers & Models card**, now in `UI/Settings_Modules/providers_models_card.py`:
    - Discover models (`settings-discover-provider-models`) sits in the "Saved model list" disclosure.
    - Session summary has its own disclosure: Checkbox "Show session usage summary when quitting", and row "Duration (seconds)".
    - The Custom endpoints editor has its own "Save name" and "Save endpoint" buttons.
    - Test is `can_focus = False`.
  - **Size ratchet.** `Tests/Architecture/test_module_size_ratchet.py` pins `settings_screen.py` at 35,767 lines, its exact size at #3057's head, with 50 lines of slack. The pin is gated in pr-fast-lane. After the merge the file may not grow.
  - **UI Fast Lane.** It runs **9 shards** (`derived-artifacts.yml` matrix), and `MINIMUM_FILES` is 282. The census already gates these files: `test_settings_connect_rows.py`, `test_settings_connect_tab_budget.py`, `test_settings_default_model_picker.py`, `test_base_app_screen_tab_region.py`. It also gates several hub nodes, including `test_settings_focus_help_line_never_describes_the_nav_bar`.
  - **Code #3057 leaves alone.** These symbols are identical on dev, #3057 and the P8/P9 branches: `on_key`, `_move_category_focus`, `_focus_category`, `_render_category_buttons`, `_workbench_focus_targets`, `_footer_shortcut_entries`, `_jk_category_navigation_blocked`, `compose_content`, `BINDINGS`. Two more are identical on dev and #3057: `_category_search_rank` and `_top_field_match`.
- **P8 (TASK-33008) and P9 (TASK-33641)** are local branches based on older dev and not yet restacked. S5 depends on neither.
  - Land TASK-35020 before P9 Task 7 if possible. TASK-33003.14 AC#2 needs `test_settings_search_index.py` to "stay green", and it is red today.
  - Both edit the CONSOLE_BEHAVIOR tuple. S5 appends rows at its end; P9 rewrites the `settings-console-default-*` rows. A rebase resolves this.
  - P9 Task 14 also appends to the census. Its plan says six shards, but #3057's head has nine, so read the matrix at rebase.
- **S3 (rail IA).** TASK-35021 lands after TASK-35016, which regroups the rail and turns the one Domain Defaults toggle into one toggle per collapsed group (class `.settings-category-group-toggle`). T2 derives arrow order from the rendered rail, so it treats every group toggle as a rail row. TASK-35020 sets the category-count literal to today's 33 and PR-gates it; TASK-35016 adds no category and leaves it alone.
- **Order.** T1 and T2 are independent. T3 follows T2, because both use the new keyboard module and the footer. Each task is one PR, rebased (never merged) onto dev and landed under CLAUDE.md's merge-queue rules.
- **Landing order across the 2026-10-14 Settings streams.** #3057 first. S1 (TASK-35010 → TASK-35011 → TASK-35012) is independent of the shell streams. TASK-35016 and TASK-35017 land before TASK-35013, TASK-35014, TASK-35015, TASK-35018 and TASK-35019. TASK-35020 can land any time after #3057: it fixes the red guards at today's truth (33 categories) and owns the rail-size guard; TASK-35016 adds no category and leaves that guard alone. TASK-35021 lands after TASK-35016, and TASK-35022 after TASK-35021. TASK-35023 can land any time after #3057.

## Scope

- **T1.** Index rows for every rendered setting the guard names, plus the Discover models action and Theme synonyms. The "/discover" ranking per the OD1 ruling. The rail-count literal. The search-index guard and the two rail guards (count and group coverage) in the required UI lane. The User Guide's "/" text.
- **T2.** One Tab stop for the rail. Arrows and j/k in the order shown, including the collapsed groups' toggles. No Tab stop on a pane body that holds controls. The User Guide's Tab path, including its Tab sentence.
- **T3.** Ctrl+S on the Settings pages that offer s, and in the in-page editors on those pages. The footer in a field. The User Guide's keys.

## Out of scope

- **Console Behavior generation-fallback search**: TASK-33003.14 (P9 Task 7). S5 does not touch the `settings-console-default-*` rows or `_top_field_match`'s choice within a category.
- **Focus after the Revert dialog**: TASK-33007.11 (on the local P8/P9 branch, not yet on dev; P9 Task 7). The Tab filter does not change what Textual focuses when a modal closes, because the nav bar stays first in the chain.
- **Discovered-list cursor (B3)**: TASK-33003.17, Done on the P9 branch (266776c54b, P9 Task 3).
- **Ctrl+S in editors on pages without s**: TASK-34000.43. These are the Advanced Config raw editor, the Theme editor, the My Profile record editor, Internal Prompts, the Image Gen and Video Gen panels, Workspaces rename, and every other screen.
- **Rail regrouping and the Domain Defaults "read-only" tooltip**: TASK-35016. **Page order**: TASK-35017. Copy and User Guide drift outside the keyboard and "/" paragraphs: S4.
- **A "shortcut" synonym.** No Settings control owns shortcuts (F1 and the footer list them), so the synonym would route to an unrelated page.
- **A focus cue for a pane body that has no control**, such as Theme's inspector. B17 reported the stop in front of controls, which T2 removes. Add the cue if a live capture shows the remaining stop unreadable.

## Decisions (ruled 2026-10-14)

**OD1: how "/discover" reaches Providers & Models. Ruled: (a), curated phrases** (the recommendation; the owner was told and did not object). T1 is written to it.

Why it was a decision: Dreams wins today because its description matches at rank 2, above every field tier. The description tier is a substring match. A general "search terms" haystack would therefore re-route every single word inside a term: "/mode", "/key", "/test", "/check", "/models".

The options, kept as the record:
- **(a) Curated phrases (ruled).** Add a few phrases for Providers & Models: "discover models", "model discovery", "test provider". They rank with the description, but match only when the query is the start of a phrase. Only queries that begin one of those phrases change rank.
  - Providers & Models then ties Dreams at rank 2 and wins on rail order (index 2 against 21).
  - Its new Discover models row gives the field landing.
  - Chosen because it is the smallest change that fixes the evidenced query and re-ranks nothing else. T1's corpus probe checks that.
- **(b) The review's suggestion (not chosen).** A Domain Defaults category's description-tier hit would rank below field-label hits in other categories. This is general, and it would re-rank every query whose description hit is a Domain Defaults page. For example, "/session" would move from ACP to Providers & Models › Session summary.
- **(c) Index rows only (not chosen).** "/discover" would have kept opening Dreams, with "Next: Providers & Models › Discover models", and T1 AC#3 would have read "'/model discovery' lands on Discover models".

"dark mode" needed no ranking change under any option. T1 adds Theme alias rows.

## Design decisions the repo already settles (not owner calls)

- **The rail becomes one stop through a `SettingsScreen.focus_chain` override.** Settings keeps `TAB_REGION = None` and the `region_focus_*` binding, so ADR-212 §4's stock walk still runs: `app.action_focus_next` → `Screen._move_focus` → `self.focus_chain`. The ADR's own oracle, `Tests/UI/test_base_app_screen_tab_region_routes.py::test_tab_from_the_first_content_control_follows_the_stock_walk[SettingsScreen]`, compares against `app.action_focus_next()` and walks the same chain.
  - **Rejected: a Settings-owned Tab binding** (the Chat/Library precedent). It would contradict §4's "None keeps the stock walk on every route" and add a third allowlisted screen.
  - **Rejected: rebuilding the rail as an `OptionList`.** It would rewrite every `#settings-category-<id>` pin (the `_click_settings_category` helpers have 113 call sites) for no difference a user could see.
  - **Rejected: `TAB_REGION`.** Textual selectors have no `:not()`, so a region selector cannot drop rail rows or pane bodies.
- **Guard lane: the UI census.** `scripts/ui_pr_gate_census.txt` feeds the UI Fast Lane. That lane is required through `derived-artifacts.yml::derived-artifacts` ("Require successful UI fast lane"), and `check_ui_pr_gate_census.py` runs in preflight.
  - `./scripts/preflight.sh` is stdlib-only static checks, and the drift guard needs a mounted app.
  - Editing `.github/workflows/derived-artifacts.yml` would risk the queue's `evict-workflow-file` refusal, since #3057 also edits that file.
  - Fallback: if the lane qualification below fails, use pr-fast-lane's admission-sensitive step.
- **Ctrl+S.** ADR-031's 2026-10-03 refinement (an owner ruling) defines it, and TASK-34000.43 names Settings forms among its examples. Personas (`personas_screen.py`, `Binding("ctrl+s", "personas_save", ...)`) and Library already bind it at screen level, non-priority. TASK-35022 delivers the Settings slice of TASK-34000.43 AC#2 and AC#4. That task keeps the inventory, the architecture test and the other editors.

## Global constraints (verified against the repo)

- **Module size ratchet** (ADR-097).
  - `settings_screen.py` nets ≤ 0 lines in every S5 PR. When it shrinks by more than 50 lines, lower its row to the measured size (`test_budget_is_not_left_slack`).
  - New code goes in `tldw_chatbook/UI/Settings_Modules/settings_keyboard.py` (new; T2 and T3) and in `settings_search_index.py` (T1). Follow the screen-decomposition rules (`Docs/superpowers/specs/2026-08-02-screen-decomposition-design.md`) and the `settings_quit_guard.py` pattern: plain functions that take `screen`, imported inside the function that uses them, as #3057 does.
  - The helper module gets no ratchet row. Its size is small, and a row would need a workflow edit to be PR-gated.
- **CSS.** No task in this stream changes CSS. If review asks for a cue on a pane body that has no control, that is a scope change:
  - edit `css/features/_settings.tcss` only, with an id-scoped rule (precedent: `#console-native-transcript:focus` in `_console.tcss`);
  - rebuild with `python tldw_chatbook/css/build_css.py`;
  - never edit `css/screen_agentic_settings.tcss`.
- **Markup.** No new interpolated `notify`, tooltip or Button label. Moved functions own no row in `scripts/markup_interpolation_census.tsv`.
- **Diagnostic inventory.** Do not move `SettingsScreen._focus_category`: it owns a `logger.debug`, and moving it changes the production diagnostic inventory. Run `./scripts/preflight.sh` and read any drift rows before `--write`.
- **Lanes.**
  - Every new test file or node is appended to the end of `scripts/ui_pr_gate_census.txt`, so earlier entries keep their shard (round-robin).
  - Raise `MINIMUM_FILES` in `scripts/check_ui_pr_gate_census.py` by the measured number of new entries, with a comment.
  - A new mounted file carries `private_profile_test` or `pytestmark = pytest.mark.bootstrap_profile`. It qualifies under a plain `pytest <file> -n0` from the worktree root, with no helper plugin on `PYTHONPATH` (PR #3055).
- **Runs.**
  - Every pytest goes through the scratch-isolated runner: scratch HOME, XDG and `TLDW_CONFIG_PATH`, and `--basetemp` under the scratchpad, removed afterwards.
  - Use xdist (`-n 6 --dist loadfile`) for everything except `Tests/Architecture` and the census-shard qualification. The qualification mirrors the lane: serial, in census order.
- **Design target.** Live checks run at 211x44 and 235x52 only.

---

## Task TASK-35020: Settings search finds every rendered setting, and its drift guards fail pull requests

Closes A-pm-design-M1, A7, D-shell-design-M3 and D heuristic 6 ("dark mode").

### Design

1. **Measure first, on post-#3057 dev.** Run the drift-guard node through the scratch runner and record its missing list. Dev listed 29; #3057 indexes the two snapshot ids, so about 27 should remain. Work from the measured list, because #3057 may have added unindexed controls.
   - The guard's later assertions have not run since the missing-list assertion went red. Those are label drift, stale allowlist rows, unresolved indexed ids, and the `PER_CATEGORY_MIN_SETTINGS` floors. Clear the list, re-run, and fix whatever they report, by measurement.
2. **Rows in `build_field_search_index`.** Append each row at the end of its category's tuple, so `_top_field_match`'s order-stable landing within a category cannot move.
   - **The label rule.** A label must contain, or be contained in, the rendered `settings-input-label` text, because the guard's label sweep checks both ways. Where the bare label collides with an earlier category, qualify it with the page name.
   - **The dev list, with rendered labels:**
     - **CONSOLE_BEHAVIOR:**
       - `settings-console-reasoning-history`, `-reasoning-native-tools`, `-reasoning-override`
       - `settings-goal-runs-enabled` ("Enable goal runs")
       - `settings-max-goal-budget-tokens`, `-generations`, `-model-calls`, `-output-tokens`, `-wall-seconds`
       - `settings-permission-summary-mode`, `-model`, `-provider`
       - Read the labels off the mounted page.
     - **PROVIDERS_MODELS:**
       - `settings-provider-auth-source`: "Sign in with".
       - `settings-session-summary-enabled`: "Show session usage summary when quitting", plus alias "Session summary".
       - `settings-session-summary-duration`: "Session summary duration (seconds)". This contains the rendered "Duration (seconds)" and does not start with "duration".
       - **A7 action row**: `settings-discover-provider-models`, "Discover models", plus alias "Model discovery". For a provider without discovery the button is disabled, and the existing disabled-landing explanation applies (`test_search_landing_on_disabled_field_explains_instead_of_no_op`).
     - **DREAMS** (new section): `settings-dreams-provider`, `-model`, `-region` as "Dreams provider", "Dreams model" and "Dreams region". The rendered labels are "Provider", "Model" and "Region". The qualifier keeps Providers & Models' own Provider and Model on top.
     - **WEB_SEARCH** (new section): `web-search-default` as "Default search backend", `web-search-backend` as "Configure backend".
     - **PRIVACY_SECURITY:** `settings-canvas-enabled`, `settings-canvas-auto-open`, labelled with their checkbox text.
     - **SPEECH_TTS:** "Acoustic barge-in (headphones)", "Send delay (seconds, pipeline engine)", "Echo cancellation", "Response eagerness (ms)".
     - **THEME:**
       - `settings-theme-preset-target` ("Presets fill") goes to `NON_SETTING_CONTROLS` with this reason: a transient pick of which colour a swatch fills, read by `SettingsThemeEditor._preset_target` and never saved.
       - D heuristic 6: add alias rows `("settings-theme-list", "Dark mode")` and `("settings-theme-list", "Light mode")` at the tuple's end. Today "theme" lands on that list (TASK-32948).
3. **OD1 (a), as ruled.** Add `CATEGORY_SEARCH_PHRASES: dict[SettingsCategoryId, tuple[str, ...]]` to `settings_search_index.py`, holding only the three Providers & Models phrases.
   - `SettingsScreen._category_search_rank` returns 2 when any phrase for the category starts with the query, next to the existing Web Search `SEARCH_TERMS` case. Leave that case and its lazy import as they are.
   - "/discover": Providers & Models and Dreams both rank 2. `_filtered_category_summaries` breaks ties on `_category_summaries()` order, so Providers & Models wins. `_search_match_is_field_landing` then promises "Providers & Models › Discover models (Core)", and the Next segment names Dreams.
   - "/test provider" opens Providers & Models plainly, where `t` runs the check. The Test button is deliberately not a Tab stop (#3057).
4. **Ratchet offset.** Move `SettingsScreen._field_match_rank_tier` into `settings_search_index.py` as `field_match_rank_tier(query, category)`. Its only caller is `_category_search_rank`, and no test calls it, so no alias is needed. `settings_screen.py` nets ≤ 0 lines.
5. **Rail count.** Set the literal in `test_settings_category_summaries_cover_every_category_id_exactly_once` to the measured count: 33 on dev and on #3057. Extend its docstring history with the step that took it from 31 to 33: Dreams and Guardian (1b769eb757), and fill in the 27→31 steps from `git log -L` so the history has no gap.
6. **Gate.** Append these under a comment naming this task:
   - `Tests/UI/test_settings_search_index.py` (whole file; module-level `bootstrap_profile`; the census already carries bootstrap files, such as `test_settings_anthropic_auth_source.py`)
   - `Tests/UI/test_settings_configuration_hub.py::test_settings_category_summaries_cover_every_category_id_exactly_once`
   - `Tests/UI/test_settings_configuration_hub.py::test_settings_category_groups_cover_every_category_id_exactly_once`

   Then raise `MINIMUM_FILES` by 3. Qualify them in the lane's own environment:
   1. Create a scratch venv with exactly the lane's install line, `-e . pytest pytest-asyncio pytest-timeout packaging`. The repo `.venv` is uv-managed, so use `uv venv` and `uv pip install`.
   2. Read the shard count from the `ui-fast-lane` matrix (9 at #3057's head).
   3. Print each receiving shard with `python scripts/check_ui_pr_gate_census.py --shard <index> <total>`.
   4. Run that shard serially, in order, with `--timeout=180`.

   It must pass, and the search-index file must stay under the lane's 60 s per-file rule. The one local timing (24.4 s) was a red run on the full venv. Under minimal dependencies the category floors (`image_generation` 40, `speech-tts` 15) may differ; this run is the evidence that counts. If it fails, use the pr-fast-lane admission-sensitive step instead.
7. **Floors.** Add `"dreams"` and `"web-search"` to `PER_CATEGORY_MIN_SETTINGS`, using measured values. A section that stops composing is then caught, not silently skipped.

### Known intended re-routes

- "/discover": Dreams → Providers & Models › Discover models (the OD1 ruling).
- "/duration": Speech & TTS → Providers & Models › Session summary duration. A Providers & Models label now contains "duration", and it ranks earlier at the same tier. Speech stays the Next match.
- Name both in the PR description. Any other moved landing found by the probe is a defect to fix before review.

### Pinned tests to update (with reason)

**Update deliberately:**
- `Tests/UI/test_settings_search_index.py::test_every_rendered_setting_is_in_the_search_index`: turns green. `NON_SETTING_CONTROLS` gains the theme preset target, and the floors gain dreams and web-search.
- `::test_non_setting_allowlist_has_no_stale_rows`: must hold with the new allowlist row.
- `Tests/UI/test_settings_configuration_hub.py::test_settings_category_summaries_cover_every_category_id_exactly_once`: 31 → measured 33.

**Must hold unchanged.** Re-run each; any change is a regression.

- `Tests/UI/test_settings_configuration_hub.py`:
  - `test_settings_category_search_filters_and_enter_opens_first_match` and `test_settings_category_search_reports_ranked_matches_and_enter_target` ("priv", exact "2 matches", Overview's secondary-match class)
  - `test_filter_word_boundary_match_outranks_substring` ("rag")
  - `test_filter_rank_tiers_keep_relative_order`
  - `test_filter_matches_owned_config_keys`
  - `test_filter_clears_after_opening_a_match` ("console b")
  - `test_field_search_surfaces_category_and_names_the_field` and `test_field_search_enter_focuses_the_field` ("threshold")
  - `test_search_finds_reduce_motion_with_scope_text` and `test_search_enter_focuses_reduce_motion`
  - `test_search_ambiguous_theme_disambiguates_with_scope` ("theme": the Theme alias rows must not change it)
  - `test_search_description_tier_match_still_lands_on_the_field` ("context window")
  - `test_search_token_keeps_its_pre_existing_intra_category_landing` ("token")
  - `test_search_landing_expands_enclosing_collapsibles` ("temperature")
  - `test_search_next_segment_is_dropped_on_short_terminals`
  - `test_search_landing_on_disabled_field_explains_instead_of_no_op`
  - `test_search_landing_on_a_row_hidden_for_the_provider_says_so`
  - `test_model_thinking_visibility_has_search_guidance_and_device_ownership`
  - `test_field_search_finds_and_focuses_folder_files_tree_controls`
  - `test_field_search_folder_files_tree_width_guides_to_custom_widths_when_disabled`
  - `test_settings_category_search_reveals_domain_matches` ("personas")
  - `test_settings_category_search_escape_clears_filter`
  - `test_video_gen_filter_exposes_category_and_enter_opens_existing_panel`
  - `test_video_generation_category_is_final_domain_default_and_count_matches_rail`
- `Tests/UI/test_settings_web_search.py::test_category_search_finds_backend_terms`
- `Tests/UI/test_settings_console_rail_labels.py` ("layout scope", "vertical")
- `Tests/UI/test_settings_console_side_chat.py` ("side chat model")
- `Tests/UI/test_settings_image_gen_panel.py` ("swarmui")
- `Tests/UI/test_settings_personal_context.py` ("my profile": an exact `_filtered_category_values() == ["personal-context"]`)
- `Tests/UI/test_settings_speech_tts_panel.py` ("Kokoro")

### New tests and lane

In `Tests/UI/test_settings_search_index.py`, which this task gates:
- **Unmounted**, using `SettingsScreen(_build_test_app())` like the hub's status tests:
  - "/discover" promises "Providers & Models › Discover models" and offers "Next: Dreams".
  - "/model discovery", "/summary duration", "/sign in", "/region" and "/search backend" each name their field.
  - "/dark mode" and "/light mode" open Theme.
- **One mounted case:** Enter on "discover" opens the Saved model list disclosure and focuses `#settings-discover-provider-models`.

**Verification steps (not committed):**
- **Corpus probe.** An uncommitted probe under `Tests/`, run through the scratch runner. A probe outside the tree runs with no conftest. On base and branch it records `_category_search_status_text(q)` for every indexed label, every word of every label, every category title word, and the review's queries. Only the intended re-routes above may differ.
- **Negative controls.** Delete one new row: the drift guard fails and names it. Set the literal back to 31: the rail guard fails. Both must be seen red in the lane-equivalent run.

### Docs

In `Docs/User_Guide/settings.md`, the "Layout tour" filter paragraph ("The filter knows every rendered setting's visible label") gains:
- that "/" also finds the Discover models action and the "model discovery", "test provider" and "dark mode" phrasings;
- that "/discover" lands on Discover models.

---

## Task TASK-35021: Tab goes rail to page to inspector; the Settings rail is one Tab stop

Closes D8 (Tab half) and B17, and fixes S5-N1.

### Design

1. **RED first.** In a new file, `Tests/UI/test_settings_keyboard_path.py` (`private_profile_test`, plain `DestinationHarness`):
   - **S5-N1.** These cases were measured on dev's grouping. TASK-35021 lands after TASK-35016's regroup, so re-derive each case on the regrouped rail before writing it. The cause (the walk follows `_category_summaries()` order, not the rendered rail, and `.focus()` succeeds on a hidden row) survives the regroup, because TASK-35016 does not reorder the summaries.
     - Down from Splash Screen should reach Console Behavior (on dev: Storage).
     - With a group collapsed, Down from the row above its toggle should reach the toggle, never a hidden member row (on dev: a hidden Library RAG row).
     - Up from Overview should stay on Overview (on dev: Hooks).
   - **D8:** one Tab from the open row should land on the page's first control (today: the next rail row).
   - **B17:** on post-#3057 dev at 211x44, Tab from the rail stops on `#settings-detail-pane-body` before the first control.
2. **New `tldw_chatbook/UI/Settings_Modules/settings_keyboard.py`.** Functions that take `screen`:
   - `rail_rows(screen)`: the displayed `Button`s under `#settings-category-list`, in DOM order. That is the order shown on screen: category rows plus the collapsed groups' toggles (`.settings-category-group-toggle`).
   - `rail_tab_stop(screen, rows)`: the focused rail row if there is one, else the open category's row if shown, else the first shown row.
   - `move_rail_focus(screen, delta)`: moves along `rail_rows`, clamped at the ends as today.
   - `tab_chain(screen, chain)`:
     - drops every widget under `#settings-category-list`, including the list's own `VerticalScroll`, except the stop;
     - drops `#settings-detail-pane-body` and `#settings-impact-pane-body` while the chain holds a widget inside them;
     - never drops the focused widget, because `Screen._move_focus` restarts from the first widget when the focused one is missing.
   - **Why bodies with controls can go:** Textual applies the `VerticalScroll`'s PageUp/PageDown bindings to its focused descendants, so the page still scrolls from its controls.
   - **Why bodies without a control stay:** F6 (`_workbench_focus_targets`) and `test_workbench_pane_focus.py` (Theme's inspector body) rely on them.
3. **`settings_screen.py` changes:**
   - **A `focus_chain` property override** returning `settings_keyboard.tab_chain(self, super().focus_chain)`. Textual's Tab path (`BaseAppScreen.action_region_focus_next` → `App.action_focus_next` → `Screen._move_focus`) and `_reset_focus` read it. Clicks, `.focus()`, F6 and every `#settings-category-<id>` helper are unaffected.
   - **`_move_category_focus`**'s body becomes a call to `move_rail_focus`.
   - **`on_key`:**
     - The up/down/j/k branches treat a group toggle (class `.settings-category-group-toggle`, not a toggle id) as a rail row. Today they require `_focused_category_value()`.
     - The Tab-from-nothing/nav-bar branch focuses `rail_tab_stop` instead of Overview. On first arrival that is still Overview, so the existing pins hold.
     - Enter on the toggle already falls through to the Button's own press.
   - **No change to `compose_content`, no CSS.** `_focus_category` stays where it is (diagnostic inventory). The file nets ≤ 0 lines; lower the ratchet row if it shrinks by more than 50.
4. **Docs** (below).

### Pinned tests to update (with reason)

**Docstring only:** `Tests/UI/test_settings_connect_tab_budget.py::_focus_chain_stops_to_model` says "Settings takes Tab over only from the nav bar or with nothing focused". That is no longer true. The Provider→Model slice of the chain is unchanged, so the assertions hold.

**Must hold, re-run.** Each of these could break:
- `Tests/UI/test_base_app_screen_tab_region_routes.py::test_tab_from_the_first_content_control_follows_the_stock_walk[SettingsScreen]`: its oracle `app.action_focus_next()` walks the same overridden chain. It is not in the census (slow), so run it explicitly.
- `Tests/UI/test_base_app_screen_tab_region.py::test_every_production_screen_inherits_the_region_binding_or_is_allowlisted`: no binding changes.
- `Tests/UI/test_workbench_pane_focus.py`:
  - Down-walk from Overview to Appearance, then Down to Theme;
  - the Theme inspector's chain equals `["settings-impact-pane-body"]`;
  - F6 mid-swap.
- `Tests/UI/test_tool_profile_focus.py::test_disappearing_action_has_a_safe_visible_continuation[unavailable]`: the detail body is the continuation when no control remains.
- `Tests/UI/test_settings_configuration_hub.py`:
  - `test_settings_tab_focus_and_enter_select_categories`
  - `test_settings_keyboard_category_focus_survives_selection_recompose`
  - `test_settings_jk_category_navigation_is_rail_scoped`
  - `test_settings_jk_leaves_theme_editor_focus_alone`
  - `test_settings_jk_never_steals_keys_from_text_input`
  - `test_settings_jk_never_steals_keys_from_select`
  - `test_settings_f6_cycles_rail_detail_inspector`
  - `test_settings_focus_surfaces_tooltip_in_focus_help_line`
  - `test_settings_focus_help_line_never_describes_the_nav_bar`
- `Tests/UI/test_settings_category_sweep.py::test_every_settings_category_renders_and_restores_focus` (focus on the clicked row).
- `Tests/UI/test_settings_connect_tab_budget.py` and `Tests/UI/test_settings_default_model_picker.py` (`focus_chain` membership).
- Detail-pane Tab walks:
  - `test_settings_console_rail_labels.py`
  - `test_settings_agents_category.py`
  - `test_settings_scope_inspector_focus.py`
  - `test_settings_field_guide_in_view.py`
  - `test_settings_provider_keyboard_journeys.py`
  - `test_settings_speech_keyboard_journeys.py`
  - `test_settings_generation_default_journeys.py`
  - `test_settings_raw_draft.py`
  - `test_settings_workspace_folder_journeys.py`
  - `test_settings_workspace_assistant_journeys.py`
  - `test_settings_qwencloud_api_mode.py`
- `Tests/Architecture/test_module_size_ratchet.py`: lower the row if the file shrinks.

### New tests and lane

`Tests/UI/test_settings_keyboard_path.py` joins the census (append; raise `MINIMUM_FILES`). Keep it under 60 s with one mount sweeping every category at 211x44, plus one spot check at 235x52. Cases:
- For every category, one Tab from the open row reaches the page's first control, or the page itself when it has no control. Shift+Tab from that control returns to the open row.
- Tab from the nav bar, and with nothing focused, lands on the open row, or on the first match while a filter hides it.
- Arrows and j/k follow the rail as shown, never land on a hidden row, and reach the toggle; Enter on the toggle toggles it. With a filter active they walk only the matches.
- After moving with the arrows, Tab leaves from the moved-to row with no other rail stop. A click on any row opens and focuses it.
- On Console Behavior (a long page), Tab never lands on `#settings-detail-pane-body`, and PageDown on one of its checkboxes scrolls the body.

### Docs

`Docs/User_Guide/settings.md`:
- **The "Moving around" paragraph in "Layout tour":** the rail is one stop; Tab goes rail → page → inspector; ↑/↓ and j/k move along the rail as shown.
- **"Keyboard & commands" rows:** Tab, ↑/↓ (which now include the collapsed groups' toggles) and j/k.
- This task owns the guide's Tab sentence (today "Tab again walks into the detail pane's fields"); S4 does not edit it.

---

## Task TASK-35022: Ctrl+S saves a Settings page from inside its fields

Closes the D8 in-field save half. Conforms the Settings pages that offer s to ADR-031's 2026-10-03 refinement.

### Design

1. **RED first.** New file `Tests/UI/test_settings_ctrl_s_save.py`, `private_profile_test`, writing only to the scratch config:
   - **Appearance text field.** Edit it, press Ctrl+S. The scratch config changes, the result line matches what the Save (s) button gives, and focus is still in the field.
   - **Appearance checkbox.** Toggle it, press Ctrl+S: the same holds.
   - **Providers & Models invalid Custom ID** (helpers from `Tests/UI/test_settings_default_model_picker.py`). Type an invalid Custom ID, press Ctrl+S. The saved model is the previously committed one, never "" (`DefaultModelPicker._restore_committed_display_after_blur`).
   - **Session summary Duration** (instant-apply; commits on submit or on leaving). Type a value, press Ctrl+S: it is written (ADR-031 b).
   - **A custom endpoint's Edit form.** Change the URL, press Ctrl+S. The endpoint is saved, as with Save endpoint, and the page draft is not saved.
   - **Ctrl+S does nothing** (no save, no toast; ADR-031 d) on a rail row, in `#settings-category-search`, in the inspector and with nothing focused.
   - **This task's binding saves nothing on Theme and Workspaces**, pages whose footer does not offer s. Ctrl+S in those pages' editors is TASK-34000.43's, so the test asserts only that the Settings binding performs no save there, and TASK-34000.43 may later change that page's own behaviour.
   - **Footer in a text field.** It shows `("ctrl+s", "save category")`, or the editor's own verb inside an in-page editor, and no `("Esc, s", ...)`. `("Esc, r", ...)` and the `t` hint keep their "Esc," form.
2. **`settings_screen.py` changes:**
   - Add `Binding("ctrl+s", "settings_save_from_field", "Save", show=False)` to `SettingsScreen.BINDINGS`. It is non-priority, like the Personas and Library precedents, so a widget's own future Ctrl+S (TASK-34000.43) still wins.
   - Nothing app-wide binds ctrl+s, and no Settings widget does. Textual's driver clears IXON (`LinuxDriver._patch_iflag`), so the key arrives instead of freezing the terminal.
   - `action_settings_save_from_field` delegates to `settings_keyboard.save_from_field(self)`.
   - `_footer_shortcut_entries`' text-entry branch calls a module helper for the mapping. The file nets ≤ 0 lines; offset by moving that branch's body.
3. **`settings_keyboard.save_from_field(screen)`:**
   1. Act only while focus is inside `#settings-detail-pane` and the open category's `_category_footer_shortcuts` offers `s`. That is `GUIDED_SETTINGS_MUTATION_CATEGORIES` plus Network, the same set the footer teaches.
   2. If the focused widget belongs to an in-page editor that has its own commit button, press that button and stop.
      - The table is `IN_PAGE_EDITOR_COMMITS`, mapping a container or field id to a commit button id, measured on post-#3057 dev.
      - Known entries: `settings-cep-rename-value` → `settings-cep-rename-save`; the `settings-cep-edit-*` fields → `settings-cep-edit-save`.
      - Candidates to confirm: the Console Behavior exchange-capture group → `settings-console-exchange-capture-apply`, and the RAG profile switch → `settings-rag-profile-switch-save`.
   3. Otherwise:
      1. Remember the focused widget, then release focus as Esc does (`set_focus(None)`), so every commit-on-leave and rollback runs.
      2. After `model_search_picker._BLUR_RESTORE_DELAY_SECONDS` (via `set_timer`), run `action_settings_save_category(allow_text_entry_focus=True)`. That is exactly what the Save button's `handle_save_category` calls.
      3. Return focus to the remembered widget if it is still mounted and nothing else took focus. P8 Task 1's focus on "Return to Console" after a successful save must win.
   4. Mark the timer coupling with a `ponytail:` comment. It is tied to the picker's blur delay. The upgrade is an explicit commit hook if a field ever commits later than that.
4. **Footer.** Map `s` to `ctrl+s` only while a text field inside the detail pane has focus. Outside a field, `s` works and the footer keeps "s save category". In the filter box it keeps "Esc, s". F1 is unchanged (ADR-031 rule 4, refinement e).

### Pinned tests to update (with reason)

**Rewrite on purpose:**
- `Tests/UI/test_settings_connect_rows.py::test_provider_is_one_row_one_tab_stop_and_filters_by_name_or_id` (gated): it asserts `("Esc, s", "save category")` with the provider list open in a text field. It becomes `("ctrl+s", "save category")`. Its post-Esc `("s", "save category")` assertion holds.

**Docstring only:**
- `Tests/UI/test_settings_default_model_picker.py::test_an_invalid_custom_id_is_rolled_back_when_the_field_is_left`: "footer's 'Esc, s'" becomes "Esc, s (or Ctrl+S)". The behaviour is unchanged.

**Must hold, re-run.** In `Tests/UI/test_settings_footer_hints.py`:
- `test_printable_shortcut_keys_type_into_focused_text_fields`
- `test_profile_footer_explains_escape_hatch_while_editor_owns_keys` (My Profile offers no s)
- `test_footer_hints_follow_the_active_category`
- `test_category_footer_shortcuts_only_advertise_working_keys`
- `test_advertised_capabilities_match_real_action_branches`
- `test_shortcut_keys_fire_actions_outside_text_entry`
- `test_narrow_footer_collapses_but_f1_help_stays_truthful`
- the `test_f1_help_*` tests

### New tests and lane

`Tests/UI/test_settings_ctrl_s_save.py` joins the census (append; raise `MINIMUM_FILES`).

### Docs

- `Docs/User_Guide/settings.md`:
  - "Keyboard & commands": the "Read this first" paragraph (Ctrl+S saves from inside a field; the letters still need Esc), plus a new Ctrl+S row.
  - "Quirks & troubleshooting": "'s' typed a letter" names Ctrl+S.
- `Docs/User_Guide/settings/rag.md`: its "Esc, s / Esc, r" line.

---

## Live verification script (each PR runs its own part)

**Setup** (lessons-live-verification, "Isolate the profile before touching runtime state"; the `verify` skill):
- **Profile.** A scratch HOME with XDG dirs, and the splash off. The review's verifiers saw the app write its config to `$HOME/.config/tldw_cli/config.toml`, so diff that file. Point `TLDW_CONFIG_PATH`, if set, inside the scratch tree.
- **Sandbox.** Launch under `sandbox-exec` with writes to the real `~/.config/tldw_cli` and `~/.local/share/tldw_cli` denied. Fingerprint the real profile before and after.
- **Interpreter.** Pin `PYTHONPATH` to the worktree and record the loaded module paths, because a worktree probe can import the main checkout.
- **No provider calls.** Answer the "Check model lists online?" modal with "Don't check".
- **Size.** Use `tmux new-session -d -x 211 -y 44`, then a fresh session at `-x 235 -y 52`. Never resize a live session: the capture keeps the first size's scroll state.
- **Reading focus.** Read focus from `capture-pane -e -p` by the focus style's SGR token, not by line diffs.
- **Typing.** Pause between a chord and the text after it, because tmux bursts reorder keys.

**Steps:**
1. **T1.** F4, then `/discover`. The status line should read "Filter: discover | 2 matches | Enter opens Providers & Models › Discover models (Core) | Next: Dreams (<Dreams' group>)", where the group name is whatever the rail shows at the time (Domain Defaults before TASK-35016, Console & Agents after it). Press Enter: the Saved model list disclosure is open and Discover models is focused. Repeat with `/model discovery`, `/summary duration`, `/sign in`, `/region`, `/search backend` and `/dark mode`, and capture each.
2. **T2.**
   - Click Appearance and press Tab once: the first Appearance control shows the focus cue. Shift+Tab: the Appearance row shows its focus bar.
   - Down from Splash Screen lands on Console Behavior, the first row of Console & Agents; each further Down lands on the next row shown.
   - Walk Down to the last Data & Privacy row, then Down once more: the Maintenance ▸ toggle. Enter expands it; Down lands on Diagnostics. With it collapsed again, Down skips its hidden rows and lands on the Owned elsewhere ▸ toggle.
   - Open Console Behavior, Tab to a checkbox and press PageDown: the page scrolls, and Tab never lands on the page body.
   - F6 still cycles the three panes.
3. **T3.**
   - On Appearance, change a text field and press Ctrl+S. The result line reports the save, the scratch config diff shows the key, and the cursor is still in the field.
   - Type "s": it types a letter.
   - Press Ctrl+S on a rail row: nothing happens.
   - On Providers & Models, enter an invalid Custom ID and press Ctrl+S: the config keeps the earlier model.
4. Diff the real-profile fingerprint (it must not change), then remove the scratch tree.

## Risks & traps

- **A pin below an already-red assertion goes stale unseen** (lessons-testing-evidence, TASK-33622.15). The drift guard's later assertions and the rail literal have not run since they went red. Measure them on both trees; never edit by delta.
- **A finding verified on the review's commit may differ on your base.** #3057 renames, excludes and adds Providers & Models ids. Re-run the guard on post-merge dev before writing rows.
- **The lane is not your venv.**
  - It installs minimal dependencies, runs serially in census order, and has 9 shards on #3057.
  - A plugin that stamps `bootstrap_profile` hides a missing marker (PR #3055).
  - `RecoveryRequired: raw_source_selection_changed` at setup is a profile-selection trip, not a broken test. Compare failure names with dev before calling anything environmental.
- **Tests that read a moved symbol's source run outside the gate.** Before moving `_field_match_rank_tier` or the footer branch, grep `Tests/` for the symbol and run every hit. Today only `_category_search_rank` calls it.
- **Mutation-test every guard.** A green guard proves nothing until it has been seen red on a broken input (T1's negative controls).
- **Search re-routes.** New labels can move unrelated landings, so the corpus probe is the check. Phrase matching is prefix-only for that reason. Console Behavior's default-* labels stay untouched (TASK-33003.14).
- **ADR-212 §4.** Settings keeps `TAB_REGION = None` and Textual's stock walk over its own chain. If a reviewer reads §4 as forbidding a filtered chain, the alternative is a Settings-owned Tab binding (the Chat/Library allowlists in both tab-region tests). That needs §4's "Library and Console keep their own Tab bindings" line amended, which is an owner call.
- **Focus semantics.**
  - `Widget.focus()` is deferred, so tests pause before reading `app.focused`.
  - `focusable` ignores `display` (the root of S5-N1), so never rely on it to skip hidden rows.
  - `_move_focus` resets when the focused widget is missing from the chain, so `tab_chain` always keeps it.
  - Keyboard scrolling must be checked by pressing PageDown (lessons-live-verification, "CSS overflow alone does not provide keyboard scrolling").
- **Ctrl+S.**
  - ADR-031 (d) forbids it outside an editor; the focus guard enforces that.
  - The timer is coupled to the picker's blur delay.
  - P8's Return-to-Console focus must win over T3's focus restore.
  - The in-page editor table must be measured, not guessed; a missed editor would save the page instead.
  - TASK-34000.43's future architecture test must recognise this as an editing-form binding.
- **Ratchet churn.** P8, P9, S3, S4 and S5 all touch the `settings_screen.py` row. Re-measure at every rebase (`len(path.read_text().splitlines())`).
- **Census merge.** P9 Task 14 also appends and raises the floor. Resolve the census tail and re-measure.
- **Disk.** Remove `--basetemp` dirs after each scratch run (the 78 GB incident).

## Rollback

Each task is one PR and reverts cleanly on its own:
- **T1:** the revert removes the rows, phrases and census entries, and lowers `MINIMUM_FILES` with them. The guards return to ungated.
- **T2:** the revert restores the old Tab walk and arrow order. Restore the ratchet row to the measured pre-T2 size.
- **T3:** the revert removes the binding and the footer mapping, and the "Esc, s" pin returns.

No task changes config schema or user data, so no migration rollback is needed.

## Overlaps

- PR #3057 / TASK-33007: assumed merged.
- P9 Task 7 (TASK-33003.14; TASK-33007.11, on the local P8/P9 branch, not yet on dev): excluded and coordinated.
- P9 Task 3 (TASK-33003.17 = B3): excluded.
- P9 Task 14: census tail and floor.
- P8 Task 1 (TASK-33008.1): focus after save.
- TASK-34000.43: T3 is its Settings slice.
- TASK-2831: hidden-row focus edge.
- TASK-32385: Folder Files labels in the same index file.
- TASK-1378: split of `settings_screen.py`.
- ADR-212 §4; ADR-031 (2026-10-03 refinement).
- S3: TASK-35016 (rail groups, group toggles) lands before T2; TASK-35017 (page order). S4 (other User Guide copy; the Tab sentence stays T2's).
