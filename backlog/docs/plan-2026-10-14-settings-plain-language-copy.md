# Plan: Settings speaks plain language: the Inspector, F1 help, page copy and the User Guide (stream S4-copy)

> **Owner rulings, 2026-10-14.** The recommendation was applied for both; the owner was told and did not object:
> - OD-1 **(a)**: F1 gets an "Includes:" line drawn from the `/` search index.
> - OD-2 **(a)**: the raw-CLI card title becomes "Danger: raw CLI and Terminal run commands as your user, with no sandbox".

Source: the Settings UX review of 2026-10-14 (critique snapshot `.impeccable/critique/2026-10-14T15-26-24Z__tldw-chatbook-ui-screens-settings-screen-py.md`), reviewed on origin/dev 1ef4fcf4ae.

Tasks: TASK-35018 (Scope Inspector and F1 help explain each Settings page in plain words), TASK-35019 (Settings pages, banners and the User Guide speak plain language).

Review: the verified Settings UX review at origin/dev 1ef4fcf4ae. origin/dev is now 67136d5dd7; the commits since then touch only backlog files and stray root files, so no Settings code anchor moved. Every symbol below was re-checked on PR #3057's head (origin/feat/model-config-p7-settings-pm, 29ca19f10d), the base this stream builds on. The plan uses verified causes only.

Findings closed:
- D4: implementation vocabulary used as user copy.
- D5: only the half about what the Inspector says. S2 owns the half about whether the storage line is true.
- A-pm-design heuristic 2: the Inspector's "Owns" list of raw config paths.
- D minor 7: F1 only repeats the save contract.
- D minor 14 and A minor 11: User Guide drift, for the copy this stream changes. The guide's category map and view-only count are TASK-35016's, the Goal runs section's move under Console Behavior is TASK-35017's, and the Tab sentence is TASK-35021's.
- D question 7: the raw-CLI "DANGER!!!" copy.

B11 is excluded (see Out of scope).

The owner chose this stream. Its copy rule is this plan's extension of the approved model-config spec §4 rule 2 (`backlog/docs/spec-2026-09-26-model-config-redesign.md`, "Raw config keys never appear in UI copy"), which Phase 7 applied to Providers & Models.

## Goal

A first-time user can read every Settings page, banner, Inspector and F1 panel in plain words. Config keys, TOML section names, database file names, Python class names, enum identifiers, ADR or task ids and roadmap notes no longer appear as copy. An expert can still find the key for the field they are editing, in one known place. F1 says what the page is for and what is on it, instead of being a fourth copy of the save contract.

## The Settings copy rule (adopted by this stream)

1. Labels, headings, banners, status lines and help text are in plain language.
2. Some things appear only inside a closed **config key** disclosure: a config key, a TOML section name (`[guardian]`), a database file name (`agent_runs.db`), a Python class or function name (`HiggsSettings`, `RawCliRuntime`), or an enum identifier (`not_tested`).
   - For an editable field, that is the "Saved as" line of the focused-field guide.
   - For read-only rows that are edited somewhere else, it is one "Saved as" disclosure under that block.
   - This extends the model-config spec §4 rule 2 ("Raw config keys never appear in UI copy") from Providers & Models to the whole shell.
   - It reuses what PR #3057 built: `SettingsScreen._config_key_disclosure`, `SettingsScreen._split_config_key_row` and `CONFIG_KEY_ROW_LABEL = "Saved as"`.
   - Settled, not an owner decision: P7 ships this exact grammar for Providers & Models as its way of meeting spec §4 rule 2, and this plan reuses it. Categories without a focused-field guide therefore show no keys. Experts use Advanced Config.
3. Enum values reach the screen through a words map, never through `.value`.
4. The UI carries no roadmap notes ("Planned", "not available yet", "future defaults may cover…", "(v1)") and no ADR or task ids.
5. A warning states the consequence in a calm sentence. Its force comes from the semantic `$error` card and the explicit disclosure lines, not from capitals or "!!!".
6. Exempt from the rule:
   - File paths the user may open.
   - The config file's name, `config.toml`.
   - Product and provider names, including `llama.cpp`.
   - User data.

## Assumptions & sequencing

- **PR #3057 (TASK-33007, head 29ca19f10d) must merge before S4 starts.** S4 reuses these pieces from it:
  - `_split_config_key_row`, `_config_key_disclosure` and `CONFIG_KEY_ROW_LABEL`.
  - `compose_providers_models_inspector` in `UI/Settings_Modules/providers_models_card.py`.
  - Console Behavior's config-key disclosure (TASK-33007.7).
  - The exact-count `settings_screen.py` row in `Tests/Architecture/test_module_size_ratchet.py` (TASK-33007.8).

  If #3057 changes again before it merges, re-check every symbol named here on its final head first (`git grep` each name).
- **What #3057 already does, which S4 does not repeat:**
  - It drops the Providers & Models "Owns" row (capture finding 18).
  - It moves `chat_defaults.streaming is canonical` into the disclosure.
  - It rewrites discovered rows as "selected · id · saved/not saved · discovered · capabilities X".
  - It moves the endpoint-key row into the Inspector disclosure, which closes B11 for that row.
- **Re-check every quoted string on the post-#3057 base before editing it** (lesson "A finding verified on the review's commit may already be fixed on your base").
- **P8** (`feat/model-config-p8-first-run`, TASK-33008.x):
  - It edits only the Reduce motion purpose strings in `_appearance_field_guidance_rows_base` and in `_render_detail_pane`'s Appearance branch (TASK-33008.4). S4 does not touch those strings, so the two collide only as text in the same functions.
  - P8 also edits `settings.md` (about 130 lines) and the ratchet row.
- **P9** (`feat/model-config-p9-riders`, TASK-33641 and riders): none of S4's functions differ from P8 there.
  - P9 Task 7 (TASK-33002.10) may put a "draft is kept when you switch categories" note in F1. Whichever lands second keeps that note.
  - P9 Task 14 edits `scripts/ui_pr_gate_census.txt`, so the census collides only as text.
  - P9 Task 2 (TASK-33005.7, .11, .15) owns the LLM readiness vocabulary. S4 never edits readiness words or rows.
- **Stream S2 (save-model honesty) runs beside this one. The boundary in one sentence: S4 owns what the Inspector body, F1 and page copy say about fields and pages; S2 owns every claim about where and when a save lands.**
  - S2's pieces: the Inspector header's `#settings-local-scope-note`, the draft-status row, the guided-action rows' truthfulness, Save/Revert, `_persistence_badge`, the Splash scope timing, and the domain "Settings mode" / "Writes allowed" rows.
  - TASK-35019 makes one name-only edit inside `_guided_action_message` (OVERVIEW).
  - TASK-35019 rewrites four branches of `_category_state_scope_text`: AGENTS, STORAGE, DREAMS and GUARDIAN.
  - Both streams edit `_render_domain_category_detail`, so whichever lands second rebases.
- **Order:** TASK-35019 depends on TASK-35018, because T1 creates the copy module and the guard file that T2 extends. Both edit `settings_screen.py` and `settings.md`.
- **The shell streams land first (S3).**
  - TASK-35018 depends on TASK-35016, which owns the view-only pages' "Planned" row and their Inspector read-only line (it rewrites `read_only_reason` to "Change these in \<owner\>"), plus the User Guide's category map and view-only count. S4 leaves all of these as TASK-35016 ships them.
  - TASK-35019 depends on TASK-35017. TASK-35017's first commit moves `SettingsScreen._render_console_behavior_card` into `UI/Settings_Modules/console_behavior_card.py`, and it reshapes `compose_goal_policy`; TASK-35019 edits the card and the goal help in their new places. TASK-35017 also moves the guide's Goal runs section under Console Behavior.
  - TASK-35021 owns the guide's Tab sentence.
- **Landing order across the 2026-10-14 Settings streams.** #3057 first. S1 (TASK-35010 → TASK-35011 → TASK-35012) is independent of the shell streams. TASK-35016 and TASK-35017 land before TASK-35013, TASK-35014, TASK-35015, TASK-35018 and TASK-35019. TASK-35020 can land any time: it fixes the red guards at today's truth (33 categories), and TASK-35016 then updates the rail-size guard deliberately. TASK-35021 lands after TASK-35016, and TASK-35022 after TASK-35021. TASK-35023 can land any time after #3057.
- **Merge queue is on.**
  - Arm auto-merge only after Qodo has reviewed the current head and every thread is resolved.
  - Never `update-branch` an armed PR.
  - Before pushing to an armed PR, run `--disable-auto` first, then `git pull --rebase`, then push with `--force-with-lease`.

## Scope / Out of scope

**In scope:**
- **T1:** the Scope Inspector body for every category, plus the header's "Selected category" row; the F1 help body; the ownership-record copy fields, which then render only in F1; and the focused-field guide rows outside "Saved as".
- **T2:** the State-banner scope text and page copy listed below, the Speech word map, the Guardian and Dreams word maps, the raw-CLI title, the Canvas limits disclosure, the domain pages' source rows and contract-table roadmap rows, the Agents scope notes, Console Behavior's budget and goal help, and the User Guide's plain-language pass.

**Out of scope, each with its owner:**
- **S2 (save-model honesty):** D1, D7, D-M1, D-M4 and the storage-line half of D5. The boundaries are listed above. D-M2 (the Domain Defaults "read-only" tooltip) is TASK-35016's.
- **Shell IA, page order and keyboard (D2, D3, D6, D8):**
  - Regrouping the rail, and the guide's category map and view-only count (TASK-35016).
  - "Open <Destination>" buttons on view-only pages, their "Planned" row and their Inspector read-only line (TASK-35016).
  - Goal runs placement, in the page and in the guide (TASK-35017).
  - The "Mode:" strip row.
  - Tab behaviour and the guide's Tab sentence (TASK-35021).
- **B11 (dotted config keys wrapping onto a second line in the wide detail pane): excluded.** It was never corroborated, and its only evidence row (the Providers endpoint key) moved to the Inspector on #3057, so no verified cause remains.
- **Findability (A7, A-M1, D-M3):** the search-index rows and the red index and count guards. S4 adds only the alias its own rename needs.
- **Discovery residuals (A2, C-M2):** the "Save selected" label and `MODEL_DISCOVERY_CAPABILITY_WARNING`.

## Decisions (ruled 2026-10-14)

Both were ruled on 2026-10-14 (the block at the top); the options are kept as the record.

- **OD-1: How F1 names the settings on a page. Ruled: (a)** (the recommendation; the owner was told and did not object).
  - (a) **Ruled.** An "Includes:" line drawn from the `/` search index (`FIELD_SEARCH_INDEX[category]`): the first visible label per field id, skipping labels that contain `.`, `_` or `<`. It is capped at 8 labels, followed by "and N more — press / to find one". Chosen because it is derived data, works on an unmounted screen, and needs no new strings.
  - (b) Not chosen. The page's section headings, read from the mounted detail pane.
  - (c) Not chosen. One to three hand-written common tasks per category, about 66 strings with no guard.
  - (d) Not chosen. No list; F1 points to the Inspector's focused-field guide.
  - Why a cap: the index holds 60 auto-refresh rows on Providers & Models and dozens of rows on Console Behavior and Image Gen, so an uncapped line would be a wall of text.
- **OD-2: The raw CLI and Terminal card's title. Ruled: (a)** (the recommendation; the owner was told and did not object).
  - (a) **Ruled.** "Danger: raw CLI and Terminal run commands as your user, with no sandbox".
  - (b) Not chosen. "Unsandboxed host access: raw CLI and Terminal".
  - (c) Not chosen. Keep "DANGER!!! RAW CLI + TERMINAL HOST ACCESS".
  - Why it went to the owner: the approved plan `Docs/superpowers/plans/2026-08-26-raw-cli-user-command.md` (Task 4) fixed the exact text "DANGER!!! RAW CLI HOST ACCESS". ADR-093 and ADR-094 require the feature to be "unmistakably dangerous", while PRODUCT.md asks for "confident restraint".
  - (a) keeps one plain danger word, states the real scope (the OS user, per `RAW_CLI_DISCLOSURE_LINES`, not "the whole computer"), and keeps the `settings-raw-cli-danger` card styling and every disclosure line unchanged.

## Task TASK-35018: The Scope Inspector and F1 help explain the page in plain words

Closes: D5 (Inspector content), A-pm-design heuristic 2 (the Owns list), D minor 7 (F1).
It also closes the D4 items that render in the Inspector or F1: class names in ownership records, `agent_runs.db`, `[tools]`, `[hooks]`, `[image_generation]`, `[scheduling]…`, `[dreams]`, `[guardian]`, `splash_screen` and `[internal_prompts]`.

**Files:**
- `tldw_chatbook/UI/Screens/settings_screen.py`
- New `tldw_chatbook/UI/Settings_Modules/settings_help_copy.py` (data only)
- New `Tests/UI/test_settings_plain_copy.py`
- `scripts/ui_pr_gate_census.txt`
- `Tests/Architecture/test_module_size_ratchet.py` (lower the row)
- `Docs/User_Guide/settings.md`

**Mechanism:**
1. **New data module `UI/Settings_Modules/settings_help_copy.py`.**
   - It receives `_INSPECTOR_GUIDANCE` and `_INSPECTOR_GUIDANCE_FALLBACK`, moved verbatim from `settings_screen.py` and renamed `INSPECTOR_GUIDANCE` and `INSPECTOR_GUIDANCE_FALLBACK`. The row label "Affected config" / "Affected data" becomes "Affects", and every value is rewritten to the rule. Examples:
     - Agents: "agent_definitions table in agent_runs.db (DB, not config.toml)" → "your named sub-agent definitions".
     - Agents boundary: "[tools] gates" → "tool switches in MCP".
     - Hooks: "[hooks] and standalone Console lifecycle commands" → "your Console hook definitions".
     - Personal Context: "…to PersonalContextService" → "…through the encrypted profile".
     - Internal Prompts: "config.toml [internal_prompts] overrides…" → "your edits to the built-in prompts".
     - Splash: "splash_screen section defaults…" → "startup splash defaults and the chosen card".
     - Workspaces recovery: "…create a workspace in Library" → "switch, rename or archive in Console (Alt+W); create one here with Create workspace…". The page has that button (D5 evidence).
   - The module holds data only. `SettingsScreen._inspector_guidance` stays in `settings_screen.py`, because its THEME, IMAGE_GENERATION, SCHEDULES, DREAMS, GUARDIAN and domain branches need `_display_path`, `_theme_save_target` and `self._domain_category_contract`. Moving the method would create a circular import.
   - `_inspector_guidance` imports the module inside the function. This is P7's lazy pattern; the Settings route's pre-import payload must not grow.
   - The explicit branches are rewritten in place. Examples: "[image_generation] backend enable/default…" → "image backends, their keys and generation defaults"; "[scheduling].briefing_schedules_enabled" → "whether scheduled briefings run".
   - Why a module at all: `settings_screen.py` is pinned at its exact line count, and T2 adds word maps that would otherwise grow the file.
2. **`SettingsScreen._render_impact_pane_body`.**
   - Delete the generic tail's "Runtime owner", "Writes allowed" and "Owns" rows on every category, and its "Read-only" row on every category except the seven view-only pages.
   - On the seven view-only pages the "Read-only" row is TASK-35016's: it rewrites `read_only_reason` from `contract.follow_up` (a roadmap sentence) to "Change these in \<owner\>". This task keeps that row as TASK-35016 ships it and does not reword it.
   - The else-branch drops its `summary.description` echo of the rail text.
   - Every non-guide category keeps the plain Affects/Recovery/Boundary block from `_inspector_guidance`. The Boundary row keeps id `settings-boundary-note`.
   - **The five guide categories show the focused-field guide (plus the blocks each already has) and no Affects/Recovery/Boundary block.** This follows the copy rule above (this plan's extension of spec §4 rule 2); the facts are kept in F1 through the ownership notes.
     - Providers & Models ends after `compose_providers_models_inspector`. This matches spec mock (c), which ends at "Key".
     - Console Behavior keeps P7's branch unchanged.
     - Appearance keeps its guide and the "Open Theme picker" button. It drops "Settings owns" / "Theme owns", because the ownership `boundary_copy` already says the same in F1.
     - Storage keeps its guide and one "Takes effect" row: "next launch; files are not moved or reconnected". It drops "Affected config", "Restart required … active handles stay unchanged", "No migration", "Runtime owner", "Writes allowed" and "Recovery".
     - RAG keeps its "Affects…" line, its guide and its "Save" row. It drops "Library owns", "Runtime owner", "Writes allowed", "Config keys: 10 editable defaults…" and "Recovery".
   - **Appearance and Storage move "Saved as" into the disclosure.**
     - Both compose with `self._split_config_key_row(...)` and then `self._config_key_disclosure(...)`. The disclosure ids are `settings-appearance-config-key` and `settings-storage-config-key`; the rows are `…-config-key-saved-as`.
     - `_refresh_appearance_field_guidance` and `_refresh_storage_field_guidance` split the same way and update the saved-as row, copying P7's `_refresh_provider_field_guidance`. Every Appearance and Storage branch has exactly one "Saved as" row, so the in-place index refresh stays aligned. `_with_save_behavior_row` keeps the count at 5.
   - **RAG gets no disclosure.** Its group "Saved as" values are already words ("the profile's embedding settings"), and its 5-row fallback has no "Saved as" row: a split would break the fixed-length in-place refresh that `_RAG_GROUP_GUIDANCE` documents. Only the "profile" group's value "the [rag.service].profile pointer" changes, to "which profile is active".
   - **Two other Inspector rows hold section names.**
     - Splash's "Config section: splash_screen" row is deleted.
     - Internal Prompts' "Save target: <path>  [internal_prompts]" becomes "Saved to: <path>". The path is exempt.
3. **Focused-field guide rows outside "Saved as"** are rewritten wherever the T1 guard flags them. Known sites:
   - The Appearance Library-rail purpose ("…saved under library.<destination>_reader").
   - The Console Behavior tool-result display cap ("max_tool_result_chars ([agents] config…)" → "the separate limit on what the model sees (16,000 characters by default)").
   - The permission-summary purpose "(ADR-090)" is removed.
   - The QwenCloud API-mode Validation row "responses or chat_completions" → "Responses or Chat Completions", the labels in `QWENCLOUD_API_MODE_OPTIONS`. This one is P7 residue.
4. **`SettingsScreen._render_impact_pane_header`:** delete the "Selected category: <title>" Static. The rail's `>` mark and the detail title already name the category. Every other header row belongs to S2.
5. **Ownership copy fields** in `_category_ownership_records` and `_domain_category_ownership_records`.
   - Rewrite `runtime_owner`, `boundary_copy` and `recovery_copy` to the rule. After this task they render only in F1, so `runtime_owner` is phrased as who uses the settings. Examples:
     - Privacy: "Canvas global execution gate plus Settings unlock and RawCliRuntime launch-local arm state" → "Canvas, and the raw CLI and Terminal unlock".
     - Storage: "Settings persisted defaults; storage services active handles" → "Chatbook's databases, from the next launch".
     - Tool Profiles: "ToolPackService (immediate reviewed actions)" → "Tool profiles; reviewed actions apply at once".
     - Network: "…the persisted [network] ssl_verify default" → "…the saved certificate-checking default".
   - **Unchanged:** `owns_config_sections`, `reads_runtime_state_from`, `writes_allowed` and `read_only_reason`. The `/` owned-key ranking tier reads `owns_config_sections`, and data tests pin the rest.
   - Storage's `recovery_copy` keeps the word "restart". Overview's boundary and recovery copy keep Console, MCP, ACP, sync and workspace. Both are pinned.
6. **F1: `SettingsScreen._category_help_notes`, edited in place.** The notes come in this order:
   - "What this page is for: <summary.description>".
   - "Includes: …", per OD-1.
   - "Saving: <badge>. <scope text>". This replaces "Save contract:" and "Scope:". It reads S2's functions, so their changes flow through.
   - "Used by: <runtime_owner>". The read-only domain pages keep their single "Owned by X" sentence, and Overview keeps its owner rows.
   - "Boundary: …" and "Recovery: …", under the existing de-duplication rule.
   - For the five guide categories only: "In the Scope Inspector: focus a setting to see what it does; its config key is under 'config key'."
   - The existing extras stay: "No shortcut keys…", "Test provider (t)…" and the Theme list-keys note.
   - "Writes here:" is dropped, because the badge says it.
   - `_workbench_help_state` and `WorkbenchHelpState` are unchanged.
7. **Size.** The deletions and the moved table shrink `settings_screen.py`. Lower its ratchet row to the measured count in the same PR; `test_budget_is_not_left_slack` requires it. The new module gets no ratchet row; that list is hand-picked.

## Task TASK-35019: Settings pages, banners and the User Guide speak plain language

Closes: D4 (the page, banner and status half), D question 7 (the raw-CLI copy), and D minor 14 and A minor 11 for the copy this task changes (User Guide drift). Depends on TASK-35018 and TASK-35017.

**Sites, by symbol:**
1. **`SettingsScreen._category_state_scope_text`:**
   - AGENTS "agent_runs.db (SQLite) — immediate CRUD, no draft" → "Definitions save as soon as you create or change them."
   - STORAGE "…; active handles stay unchanged." → "Changes apply the next time Chatbook starts."
   - DREAMS and GUARDIAN "…keys stay editable in config.toml only" → "…are changed in Advanced Config".
   - Nothing else changes; the badge and the Splash line belong to S2.
2. **`SettingsScreen._render_detail_pane`.**
   - **STORAGE:**
     - "Activation … active database handles keep current paths" → "Takes effect: next launch; open databases keep their current files until then".
     - "Check mode: non-mutating; reports parent readiness for the current config runtime" → "Check: reads only; reports whether each folder exists and can be written".
   - **PRIVACY_SECURITY:**
     - The `#settings-raw-cli-title` copy follows OD-2. The card classes, `RAW_CLI_DISCLOSURE_LINES` and the armed-state lines are unchanged.
     - The "Effective hard quotas — read-only" block moves into a closed `Collapsible` titled "Canvas limits (read-only)". It reuses the existing `.settings-config-key-disclosure` class (a plain class selector in `css/features/_settings.tcss`), so no CSS is added.
     - The label "Skill trust keyring convenience" → "Remember skill trust in the system keychain".
   - **DIAGNOSTICS:** "Diagnostics writes: not available yet - raw edits remain gated in Advanced Config" → "Changes: none here — edit raw settings in Advanced Config".
3. **`settings_privacy_security.build_privacy_posture_rows`:** the same keyring label.
4. **`settings_storage_defaults.STORAGE_FIELD_LABELS["chachanotes_db_path"]`:** "ChaChaNotes DB" → "Conversations & notes DB".
   - That field's validation messages interpolate the label, so they change with it. The other labels stay.
   - `settings_search_index.build_field_search_index` gains the alias row `("settings-storage-chachanotes-db-path", "ChaChaNotes DB")`, so `/` still finds the field by its old name. The index docstring allows several rows per field.
5. **Speech.**
   - Add `SPEECH_TTS_CONNECTION_WORDS` beside `SpeechTTSConnectionState` in `UI/Speech/speech_settings_contracts.py`: Reachable, Unreachable, Not tested, Not supported. The panel already imports that module.
   - Use it at every `.connection.value`, `.catalog.value` and `.sample.value` interpolation in `SpeechTTSSettingsPanel` (`Widgets/Settings_Widgets/speech_tts_settings_panel.py`): `_connection_status_copy`, the `#settings-speech-provider-current-status` refresh, and `_compose_provider_setup_body`.
   - This is not LLM readiness (TASK-33005.7).
6. **`SettingsScreen._render_domain_category_detail`.**
   - Delete the "Where the data lives" heading and its "Source N" rows on every domain page. The "Planned" (`follow_up`) row is TASK-35016's, which deletes it; this task does not touch it.
   - In `DOMAIN_CONTRACT_BY_CATEGORY`, delete the roadmap and developer-instruction rows: Roleplay "What Settings might add later"; Skills, Watchlists and Workflows "Settings role: future defaults…"; Artifacts "Settings role: …do not move artifact operations here".
   - Reword these rows: Dreams "Budgets & cadence: read-only here; edit the remaining [dreams] keys in config.toml (v1)" → "read-only here; change them in Advanced Config", and Guardian "Dreams gate: only rules opted in with feeds_discovery…" → "only rules you allow to feed Dreams do; crisis rules never can".
   - `source_of_truth` and `follow_up` data stay non-empty, because they are pinned. The view-only pages' `read_only_reason` ("Change these in \<owner\>") is TASK-35016's; leave it as it ships. The MCP "Runtime owner" row stays (pinned).
   - The "Settings mode" and "Writes allowed" rows belong to S2 (D7).
7. **`SettingsScreen._render_guardian_controls` and `_render_dreams_controls`.**
   - Add `GUARDIAN_READ_ONLY_WORDS` and `DREAMS_READ_ONLY_WORDS` to `settings_help_copy.py`, imported inside the functions. Examples: `fixation_share_threshold` → "Fixation alert share"; `stories_per_cycle` → "Stories per cycle". The rows read "<words>: <value>".
   - The headings say "read-only here; change them in Advanced Config".
   - Under each block, add one `_config_key_disclosure` holding "Saved as: [guardian] fixation_share_threshold, …".
   - **Pending owner ruling (asked 2026-10-14): supersede TASK-34792 R1 — plain-language labels with the keys in a 'Saved as' disclosure — or keep R1.** R1 is that task's controller ruling that the Dreams budget and cadence rows are labelled by their TOML key. This step and TASK-35019 AC#3 are written for superseding it; R1's purpose, letting users find the key, is kept by the disclosure. If the owner keeps R1, the Dreams rows are excluded from this step and from AC#3 (they keep their key labels), and Guardian's rows still change.
8. **Console Behavior help** (after TASK-35017; the card lives in `UI/Settings_Modules/console_behavior_card.py`).
   - In `AGENT_BUDGET_FIELDS`, rewrite the `help_text` of `agent_max_total_tokens`, `agent_max_wall_seconds` and `agent_max_tool_call_seconds` in plain words, keeping every safety fact:
     - Spend grows faster each turn, because the whole conversation is re-sent.
     - Sub-agents have their own time limit.
     - A tool that timed out may still finish, so a limit under about three minutes can run an MCP tool twice.
   - "199,999" and the "at least" hint stay (pinned).
   - In `settings_goal_policy`, the `compose_goal_policy` hint and the `normalize_goal_setting` error drop "admission". For example: "0 blocks new goal runs".
9. **`Widgets/settings_agents_panel.py`:**
   - "(stored in agent_runs.db, not config.toml)" → "(saved in Chatbook's agents database, not your config file)".
   - The Routing note "(stored in config.toml [agents], applied at spawn)" → "(saved in your config file; used when a sub-agent starts)".
10. **`SettingsScreen._guided_action_message`, OVERVIEW entry:** "choose Providers or Console" → "choose Providers & Models or Console Behavior". This fixes the names only and makes no claim about saving (S2).
11. **`Docs/User_Guide/settings.md` content: the plain-language pass.** Four items are owned elsewhere and not touched here: the category map and the view-only page count (TASK-35016), moving the Goal runs section under Console Behavior with its F4 path (TASK-35017), and the Tab sentence (TASK-35021).
    - **Goal runs wording:** in the section TASK-35017 moved, drop "zero disables admission". The guide may keep the key names in its table, since the rule governs the UI, not the guide.
    - **Quirks:** Animations is read by character expressions. Leave the Smooth scrolling and Palette limit halves to S2 (D7).
    - **Quote as they now render:** the Privacy raw-CLI passage, the Agents notes, the Storage label list, and the Speech, Guardian and Dreams rows.

## Pinned tests to update

On the #3057 base, several Settings nodes are in the UI PR lane, but most of the pins below are not. For each task:
- Run every `scripts/ui_pr_gate_census.txt` row that starts `Tests/UI/test_settings` or is `Tests/UI/test_profile_owned_settings_paths.py`.
- Run every file named below, on the merge base and on head, with xdist (`-n 6 --dist loadfile`).
- Compare the sets of failure names.

**TASK-35018:**
- `Tests/UI/test_settings_configuration_hub.py`:
  - `test_inspector_guidance_covers_every_settings_category`: reads `settings_screen_module._INSPECTOR_GUIDANCE`. Point it at `settings_help_copy.INSPECTOR_GUIDANCE`.
  - `test_settings_copy_names_the_rail_category_and_the_real_control` (TASK-33002.6):
    - It reads `settings_screen.py`'s source for "console defaults" and indexes `_INSPECTOR_GUIDANCE[PROVIDERS_MODELS]["Boundary"]`.
    - Make it read both files' sources, then assert that the Console Behavior route survives in the Providers F1 recovery note.
    - Lesson: "Tests that read a moved file's source run outside the PR gate".
  - `test_settings_provider_inspector_excludes_console_sampling_ownership`: asserts the "Affected config: provider, model…" row and "Global fallbacks are in Console Behavior." in the Inspector. The block leaves the Inspector, so assert its absence there, and assert that F1's notes name "Console Behavior" and never "Console Defaults".
  - `test_settings_inspector_has_no_write_blocked_contradiction`: asserts "Writes allowed: Yes". The row is gone. Keep the task-181 intent: no "Writes remain blocked" and no "Mutation replay". The Local-only line is S2's.
  - `test_settings_inspector_uses_category_specific_guidance`: asserts Storage's "Affected config:" and "Recovery: Validate paths…" in the Inspector. Assert them in Storage's F1 notes instead.
  - `test_settings_library_rag_renders_guided_defaults_and_validates` and `test_settings_library_rag_inspector_uses_shortened_terse_guidance`: assert "Config keys: 10 editable defaults…". The row is deleted; the terse-guidance assertions stay.
  - `test_settings_advanced_config_guided_path_buttons_escape_raw_toml`: uses "Selected category: Providers & Models" as its landing evidence. Use `active_category` and the detail title.
  - `test_settings_console_behavior_inspector_explains_visible_controls`: asserts that "Affected config: chat_defaults fallbacks…" is NOT visible. It stays green but becomes vacuous; swap it for the plain-copy guard's check on that page.
- `Tests/UI/test_settings_narrow_layout.py::test_inspector_does_not_wrap_mid_word`: the Overview row expects "Read-only:". Swap it for "Affects:".
- `Tests/UI/test_profile_owned_settings_paths.py` (in the lane):
  - `test_settings_theme_copy_uses_effective_profile_paths`: asserts the `("Affected config", "custom theme files under …")` tuple. Relabel it "Affects".
  - `test_settings_internal_prompt_copy_uses_effective_profile_path`: asserts "Save target: <path>  [internal_prompts]". It becomes "Saved to: <path>".
  - Both tests drive `_render_impact_pane_body` on a `MagicMock(spec=SettingsScreen)`, so keep the Theme and Internal Prompts branches free of new `self.` calls.
- `Tests/UI/test_settings_footer_hints.py`:
  - `test_f1_help_has_contract_content_for_every_category`:
    - It asserts `len(members) == 30` against 33 members, so it is red today and nothing under it has run since.
    - Drop the literal and loop over the enum.
    - Assert "Saving: " and "Used by: " / "Owned by ".
    - Move the de-dup exemption from "Runtime owner: " to "Used by: ".
    - Lesson: "A pin below an already-red assertion goes stale unseen".
  - `test_f1_help_video_gen_states_a_coherent_write_contract`: asserts "Save contract: …" and "Writes here: yes.".
  - `test_f1_help_renders_literal_brackets_in_notes`: relies on Agents' "[tools] gates", which the rule removes. Keep the markup-safety guarantee by passing a literal-bracket note to `WorkbenchHelpPanel` in the test itself.
  - `test_f1_help_panel_body_carries_category_contract_when_mounted`: asserts the exact Storage and Schedules runtime-owner strings and "Writes here: yes.".
- `Tests/UI/test_settings_theme_picker_screen.py::test_theme_help_copy_describes_the_picker`: asserts "Save contract: Applies immediately." Change it to the "Saving:" line.
- **Run, and update only if row indexes or heights move:** `Tests/UI/test_settings_field_guide_in_view.py` (including the lane nodes `test_a_focused_field_keeps_applies_to_and_next_new_chat_in_view` and `test_the_inspector_starts_at_a_heading_and_says_when_more_is_above`), `Tests/UI/test_settings_scope_inspector_focus.py`, and `Tests/UI/test_settings_save_commit_models.py`. Its row-count test calls the `_rows()` builders, which keep 5 rows, so it should stay green.

**TASK-35019:**
- `Tests/UI/test_settings_raw_cli.py::test_raw_cli_unlock_and_arm_are_separate_confirmed_gates`: asserts the exact title. Use the OD-2 title, and keep its border, background and disclosure assertions.
- `Tests/UI/test_settings_canvas.py::test_privacy_settings_present_canvas_controls_status_and_read_only_quotas`: checks "Effective hard quotas — read-only" and the limit rows. Expand the "Canvas limits (read-only)" disclosure first.
- `test_settings_configuration_hub.py::test_settings_privacy_and_diagnostics_label_unsupported_mutations_as_wip`: checks "Diagnostics writes: not available yet". Use the new Diagnostics row.
- `test_settings_configuration_hub.py::test_settings_read_only_overview_hides_actions_and_clean_privacy_disables_them`: checks "Guided edits: choose Providers or Console." twice. Use the new names.
- `Tests/UI/test_settings_guardian.py::test_guardian_renders_read_only_trend_key_rows`: checks the "<key>: <value>" rows, "read-only" and "config.toml". Use the word labels, expand the disclosure to read the keys, and assert "Advanced Config".
- `Tests/UI/test_settings_dreams.py::test_dreams_category_registered_with_toggle_and_budget_rows`: checks that "cadence_hours" and "max_llm_calls_per_day" are visible. Same change.
- `Tests/UI/test_settings_privacy_security.py::test_privacy_posture_reports_skill_trust_without_leaking_paths` and `::test_privacy_posture_reports_skill_trust_disabled_without_mapping`: check the keyring label.
- `Tests/UI/test_settings_speech_tts_panel.py::test_settings_connection_uses_sample_over_unsupported_catalog` ("sample reachable") and `::test_settings_connection_invalidates_evidence_at_changed_revision` ("not_tested", "catalog not_tested"): use the words.
- `Tests/UI/test_settings_field_guide_in_view.py`: its `SHARED_HOOK_CASES` table pins "Focused setting: ChaChaNotes DB". Use the new label.

**Must stay green unchanged.** They prove the data, search and geometry are untouched:
- The `owns_config_sections` data pins:
  - `Tests/UI/test_settings_hooks.py`
  - the ownership tuple in `test_settings_raw_cli.py`
  - `Tests/UI/test_settings_qwencloud_api_mode.py`
  - `test_settings_ownership_records_cover_categories_and_runtime_boundaries`, which needs "restart" in the Storage recovery and the Overview owners
  - the domain-record test that asserts `read_only_reason`, `recovery_copy` and `runtime_owner` are non-empty
- `test_settings_domain_category_contracts_are_explicit_about_mutation_scope`.
- `test_settings_domain_category_renders_read_only_owner_contract`, which needs "Owner destination: MCP" and "MCP owns server/tool runtime".
- `test_settings_overview_ownership_rows_are_sourced_from_record`.
- `Tests/UI/test_destination_visual_parity_correction.py::test_settings_boundary_note_is_fully_visible_at_rest` and the settings case of `test_runtime_and_settings_default_states_preserve_workbench_geometry`. Overview keeps `#settings-boundary-note`.
- `test_settings_inspector_boundary_is_structured_without_duplicate_copy`.
- `test_detail_row_folds_long_config_keys`, `test_fold_long_tokens_breaks_at_separators_not_mid_word` and `test_fold_prefers_slash_boundaries_over_extension_dots`. The fold default is unchanged.
- `test_settings_storage_defaults_load_validate_and_build_save_payload`, whose "Media DB" messages are unchanged.
- `Tests/UI/test_settings_rag_profile_region.py`, which reads `_RAG_GROUP_GUIDANCE`. That table is not moved.
- `Tests/UI/test_settings_search_index.py`. It is red on dev (A-M1): compare failure names and add none.
- `Tests/UI/test_settings_agent_run_budget.py` and `Tests/UI/test_settings_agent_budget_journeys.py`.
- `Tests/Performance/test_screen_preimport_payload_budget.py` and `Tests/Architecture/test_module_size_ratchet.py`, the latter after lowering the row.

## New tests and their lane

`Tests/UI/test_settings_plain_copy.py` is created by T1 and extended by T2. It is appended to `scripts/ui_pr_gate_census.txt`, so it runs in the sharded `ui-fast-lane` job. Measure the file's time first. If the mounted cases strain a shard, list the unmounted tests plus one mounted node id instead of the whole file; node ids are allowed.

- **`jargon_hits(text)`, a small helper in the test file.** It flags:
  - `.db` file names and "SQLite".
  - CamelCase identifiers ending in Settings, Runtime, Service, Policy, Limits, Engines, Store or Manager.
  - snake_case identifiers.
  - Dotted keys, except an allowlist of `config.toml` and `llama.cpp`.
  - `[section]` references.
  - `ADR-<n>` and task ids.
  - "Planned", "not available yet" and "!!!".

  Tokens containing `/` or `\` count as paths and are exempt. UI names such as "Providers & Models" are not substrings the patterns can match (lesson "Classifying user copy by loose substring…").
- **Permanent negative control.** A parametrized test feeds the verbatim D4 strings to `jargon_hits`, and each must be flagged:
  - "agent_runs.db (SQLite) — immediate CRUD, no draft"
  - "…RawCliRuntime launch-local arm state"
  - "Current status: not_tested."
  - "fixation_share_threshold: 0.6"
  - "SearchSettings.search_provider_default"
  - "DANGER!!! RAW CLI + TERMINAL HOST ACCESS"
  - "Diagnostics writes: not available yet"
  - "[tools] gates"

  Controls that must not be flagged: "Providers & Models", "config.toml", "llama.cpp", "~/.config/tldw_cli/config.toml".
- **Unmounted (`SettingsScreen(_build_test_app())`).** For every `SettingsCategoryId` it checks:
  - **T1:**
    - The `WorkbenchHelpState.render_text()` of F1.
    - Every `_inspector_guidance` value.
    - The ownership `runtime_owner`, `boundary_copy` and `recovery_copy`.
    - Every non-"Saved as" row of the Appearance, Storage, RAG, Console Behavior and Providers & Models guides, at every field id. Iterate the field ids as `test_settings_save_commit_models.py` does.
  - **T2:**
    - `_category_state_scope_text`.
    - The domain contract rows.
    - `STORAGE_FIELD_LABELS`.
    - The `AGENT_BUDGET_FIELDS` help and the goal-policy copy.
    - The Speech, Guardian and Dreams word maps. Each map must cover every member of its enum or tuple, and no value may contain "_".
- **Mounted (Pilot, production stylesheet, `@private_profile_test` like the neighbouring Settings tests).**
  - Read only painted text, skipping widgets inside a collapsed `Collapsible` (lesson "A clipped region check does not prove a control is painted").
  - **T1, at 211x44:**
    - On Overview, Appearance (Density focused), Storage, Agents, Privacy & Security, Splash, Internal Prompts and Web Search, the Inspector's painted text has no `jargon_hits`, no "Runtime owner", no "Writes allowed", no "Owns" and no "Selected category".
    - On Appearance, expanding "config key" shows "Saved as: appearance.density".
  - **T2:**
    - On Privacy, the raw-CLI title has no "!!!" and the card keeps its danger class.
    - Diagnostics, Workflows (view), Guardian, Dreams, Storage and Speech & TTS (scope inspector expanded) paint no `jargon_hits`.

## Docs

`Docs/User_Guide/settings.md` gets content changes only, with no new "Verified against" paragraphs. Record verification in each task's Implementation Notes.

- **T1:**
  - The Layout tour's "Scope Inspector" row: what is pinned (no "Selected category"), the focused-field guide, the "config key" disclosure, and the plain Affects/Recovery/Boundary block on the other categories.
  - The F1 paragraph under "Keyboard & commands": what the panel now holds.
- **T2:** the items in T2 step 11.
- **Both:**
  - Run `python scripts/check_guide_claim_strings.py Docs/User_Guide/settings.md` before and after. It is a read-list, not a gate. No new misses may appear for strings this stream changed.
  - Describe composed rows (F1 notes, Inspector rows) as rules, not as quoted specimens (lesson "A claim-string checker is blind by construction to composed copy").

## Live verification script (per task, a fresh session per size)

Use the sandboxed recipe that has been verified to work. Never point the app at the real profile.

1. **Set up the scratch dirs.**
   - `V=$SCRATCH/s4-t<N>-<size>`, then `mkdir -p $V/home $V/config $V/data $V/cache`, then `touch $V/baseline`.
2. **Write the sandbox profile.**
   - `PROFILE='(version 1)(allow default)(deny file-write* (subpath "/Users/macbook-dev/.local/share/tldw_cli") (subpath "/Users/macbook-dev/.config/tldw_cli") (subpath "/Users/macbook-dev/.config/tldw-live"))'`
   - Use absolute real paths: `~` inside the app resolves to the scratch home.
3. **Launch from the worktree:**

   `tmux -L s4t<N> new-session -d -x 211 -y 44 "env -u TLDW_CONFIG_PATH HOME=$V/home XDG_CONFIG_HOME=$V/config XDG_DATA_HOME=$V/data XDG_CACHE_HOME=$V/cache OPENAI_API_KEY=sk-scratch-not-a-real-key PYTHONPATH=<worktree> sandbox-exec -p '$PROFILE' <repo>/.venv/bin/python -m tldw_chatbook.app"`

   Then repeat with `-x 235 -y 52` in a new session; never resize a running one.
   - Do not set `TLDW_CONFIG_PATH`: a missing file starts the app in Recovery required.
   - Do not redirect stderr: Textual draws there.
   - The placeholder key skips the first-run wizard. Never send a message with it.
   - Cold start takes about 45 s.
   - Answer "Check model lists online?" with "Don't check".
4. **Visit each category and capture it.** Press F4 (Settings), then visit:
   - Overview, Providers & Models, Web Search, Speech & TTS (expand its "Scope inspector").
   - Appearance: Tab to Density, then expand "config key".
   - Storage, Privacy & Security, Diagnostics, Agents, Guardian, Dreams, Workflows (view), Splash, Internal Prompts and Console Behavior.
   - Press F1 on Appearance, Agents and Workflows.
   - Type `/chachanotes` (T2).
   - Capture each with `tmux -L s4t<N> capture-pane -p > $V/captures/<nn>-<category>.txt`.
5. **What to read in the captures.**
   - **T1:**
     - The Inspector has no "Runtime owner", "Writes allowed", "Owns" or "Selected category".
     - No dotted key appears outside an expanded disclosure.
     - The whole focused-field guide is visible without scrolling.
     - The Overview boundary note is fully visible.
     - F1 shows "What this page is for" and "Includes", and states the save contract once.
     - Note whether F1's empty bordered region (B20, out of scope) persists.
   - **T2:**
     - No "DANGER!!!", "not_tested", "agent_runs.db", "ChaChaNotes", "Planned" or "Source 1".
     - The Guardian and Dreams rows are in words.
     - `/chachanotes` lands on the Storage field.
6. **Back captures with Pilot.** A capture is a read, not proof of what was painted. Every claim made in the PR also has a Pilot assertion in `test_settings_plain_copy.py`.
7. **Clean up.**
   - Quit with `C-q`, then `tmux -L s4t<N> kill-server`.
   - `find /Users/macbook-dev/.config/tldw_cli /Users/macbook-dev/.local/share/tldw_cli -newer $V/baseline` must print nothing.
   - Delete `$V`. Implementation Notes cite the commands and test names, not the capture files.

## Risks & traps

- **Moved table, source-reading tests.**
  - `test_settings_copy_names_the_rail_category_and_the_real_control` reads `settings_screen.py`'s source. Grep `Tests/` for `_INSPECTOR_GUIDANCE`, `settings_screen_module.` and `__file__` reads, and run every hit on both trees.
  - The PR lane collects few of them (lesson "Tests that read a moved file's source run outside the PR gate").
- **A literal count that is already red.** `test_f1_help_has_contract_content_for_every_category` asserts `== 30` against 33 members, so its loop has never run against the newer categories. Expect real F1 failures to surface once the literal goes. Measure; never edit by delta.
- **`RecoveryRequired: raw_source_selection_changed` at setup is a profile-selection trip, not a regression.**
  - Compare the failure-name sets at the merge base and at head.
  - Run every pytest through the scratch wrapper: scratch HOME, XDG dirs, `TLDW_CONFIG_PATH`, and a unique `--basetemp` you delete afterwards. Use `-n 6 --dist loadfile`, and run `Tests/Architecture` serially.
  - With a shared venv, set `PYTHONPATH=<worktree>`: its editable install may point at another checkout.
- **The guard has two ways to fail.**
  - False positives on product names, UI names and paths: use the allowlist plus the negative and positive controls above.
  - False negatives on composed copy: it also runs the builders and the painted text, not only literals.
- **Field-guide refresh by index.** The Appearance and Storage splits must remove exactly one row in every branch. RAG must not be split. Pins: `test_settings_scope_inspector_focus.py` and `test_settings_field_guide_in_view.py`.
- **Pre-import payload.** `settings_help_copy` is imported only inside functions. Measure `test_screen_preimport_payload_budget.py` with `PYTHONPATH=<worktree>`.
- **Module size ratchet.**
  - Lower the `settings_screen.py` row to the measured count in each PR.
  - T2's word maps live in `settings_help_copy.py`, so T2 must not grow `settings_screen.py`. If it does, move more data out; never raise the number.
- **CSS.** S4 adds no CSS. If a rule turns out to be unavoidable, it lands in the lazily loaded Settings sheet, which the bare-type census (it boots Console) does not count, so add no bare-type subject anyway, and keep it out of the tight boot CSS budget:
  - Edit `css/features/_settings.tcss`, never the generated `css/screen_agentic_settings.tcss`.
  - Rebuild with `tldw_chatbook/css/build_css.py`.
  - Run `./scripts/preflight.sh`.
- **Markup.**
  - `_detail_row` already uses `markup=False`. Any new Static that interpolates runtime text does too.
  - Run `python scripts/check_markup_interpolation.py`; it should show no new sites.
- **Superseded rulings, said out loud in the PR.**
  - TASK-34792's R1 (keys as Dreams row labels): pending owner ruling (asked 2026-10-14) — supersede R1 (plain-language labels with the keys in a 'Saved as' disclosure) or keep it. If it is superseded, say so in the PR; if it is kept, the Dreams rows stay as R1 has them and the PR says so.
  - P7 review R1 #27's Providers Inspector boundary line: the fact moves to F1.
  - The raw-CLI plan's exact title (OD-2).
- **Branch conflicts.**
  - S2 edits `_render_impact_pane_header`, `_category_state_scope_text` and `_render_domain_category_detail` nearby, so keep to the lines named above.
  - P8 edits nearby Reduce-motion strings.
  - P8, P9 and S2 all touch the ratchet row and `settings.md`. Whoever lands second rebases (`git pull --rebase origin dev`, push with `--force-with-lease`) and re-measures the ratchet.
- **zsh.**
  - Quote `git show "${ref}:path"`, because zsh's `:t` modifier mangles an unquoted ref.
  - Never pass an unquoted `$FILES` to pytest: zsh does no word splitting, so pytest gets one argument and runs nothing.

## Rollback

Each task is one PR holding copy, rendering and test changes only. There is no schema, config or migration change, and the ownership data, contract data and search ranking are untouched. Reverting a merge commit restores the old copy and the old ratchet row. T2 can be reverted alone. To revert T1, revert T2 first, because T2 extends T1's copy module and guard file.
