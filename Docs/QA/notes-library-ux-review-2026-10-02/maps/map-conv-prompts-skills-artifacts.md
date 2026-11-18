# UX map — Library ▸ Conversations, Prompts, Skills, Artifacts/Reports, Study handoffs, character repair/return, cross-destination handoffs

- Code under review: `.worktrees/notes-library-ux-review` @ origin/dev `2d34cbf80d` (2026-10-02).
- Method: static code reading only (no live run). Every claim cites `file:line`. Paths are relative to
  `tldw_chatbook/` unless they start with `Docs/`.
- Abbreviations: `LS` = `UI/Screens/library_screen.py`; `LM/` = `UI/Library_Modules/`; `WL/` = `Widgets/Library/`;
  `L/` = `Library/` (pure state).
- Everything in "Suspected issues" is **SUSPECTED** — derived from code, not observed. Each has a live probe.
- No design recommendations in this file (by brief).

---

## 1. Inventory

### 1.1 Where these surfaces live in the Library rail

Rail rows are built in `L/library_shell_state.py:493-761`. Sections, in order: **Browse**, **Artifacts**,
**Create**, **Study**, **Import / Export** (`L/library_shell_state.py:747-756`).

| Rail row (DOM id `#library-row-<row_id>`) | Visible title / short title / gloss | Target | Cite |
|---|---|---|---|
| `browse-conversations` | "Conversations" / short "Chats" / count `(N)` | canvas `conversations` | `L/library_shell_state.py:507-522` |
| `browse-prompts` | "Prompts" / gloss "reuse" / count | canvas `prompts` | `L/library_shell_state.py:523-547` |
| `browse-skills` | "Skills" / gloss "AI add-ons" / count | canvas `skills` | `L/library_shell_state.py:548-561` |
| `artifacts-all` | "All artifacts" (no count) | canvas `artifacts-all` | `L/library_shell_state.py:610-617` |
| `artifacts-chatbooks` | "Chatbooks" (no count) | canvas `artifacts-chatbooks` | `L/library_shell_state.py:618-624` |
| `artifacts-reports` | "Reports" (no count) | canvas `artifacts-reports` | `L/library_shell_state.py:625-631` |
| `create-prompt` | "New prompt" | **same** canvas `prompts`, editor in create mode | `L/library_shell_state.py:644-652`, `L/library_shell_state.py:44-52` |
| `create-skill` | "New skill" | **same** canvas `skills`, editor in create mode | `L/library_shell_state.py:653-661` |
| `create-study` | "Study decks" `(N)` | handoff canvas `study` | `L/library_shell_state.py:671-680` |
| `create-flashcards` | "Flashcards" / short "Cards" / `due: N` | handoff canvas `flashcards` | `L/library_shell_state.py:681-706` |
| `create-quizzes` | "Quizzes" `(N)` | handoff canvas `quizzes` | `L/library_shell_state.py:707-716` |

Legacy route aliases that land here: `notes|prompts|skills|search|media|ingest|artifacts → library`
(`UI/Navigation/screen_registry.py:216-283`); bare-alias default contexts `artifacts→artifacts-all`,
`prompts→prompts`, `skills→skills` (`app.py:2442-2449`).

### 1.2 Per-surface component inventory

| Surface | Items pane widget | Work pane widget | Shell / grip labels | Controllers |
|---|---|---|---|---|
| Conversations | `LibraryConversationsCanvas` (`WL/library_conversations_canvas.py:33`) | `LibraryConversationReader` (`WL/library_conversation_reader.py:130`) | `LibraryAdaptiveReaderShell`, `library_label="Library"`, `items_label="Items"` (`LS:15395-15406`) | `LM/library_conversations_controller.py`, `LM/library_conversation_reader_controller.py`, `LM/library_conversation_recovery.py` |
| Prompts | `LibraryPromptsListCanvas` mode=list (`WL/library_prompts_canvas.py:96`) | `LibraryPromptWorkPane` (`WL/library_prompt_work_pane.py:13`) | items_label="Prompts" (`LS:15268-15279`) | `LM/library_prompts_controller.py`, `LM/library_prompt_browse_controller.py`, `LM/prompt_collection_manager_modal.py`, `LM/prompt_history_region.py` |
| Skills | `LibrarySkillsListCanvas` mode=list (`WL/library_skills_canvas.py:691`) | `LibrarySkillWorkPane` (`WL/library_skill_work_pane.py:25`) | items_label="Skills" (`LS:15316-15325`) | `LM/library_skills_controller.py`, `LM/library_skills_builtin_controller.py`, `LM/library_skill_import_controller.py`, `LM/skill_import_choice_modal.py` |
| Artifacts (All/Chatbooks/Reports) | `ArtifactItems` (`WL/library_artifacts_widgets.py:32`) | `ArtifactWork` (`WL/library_artifacts_widgets.py:176`) | `LibraryArtifactsReaderShell`, "Library"/"Items" (`WL/library_artifacts_reader_shell.py:27-41`) | `LM/library_artifacts_controller.py`, `LM/library_artifacts_navigation.py`, `LM/library_artifacts_share_controller.py` |
| Study handoffs | `LibraryStudyHandoffCanvas` (`WL/library_entry_canvases.py:526`) | — | — | `LS:14274-14296`, `LS:14437-14471`, `LS:35712-35733` |
| Character return | `LibraryCharacterReturn` (`WL/library_character_return.py:10`) | — | composed above header (`LS:15016`) | `LM/library_unavailable_navigation.py:150-188` |
| Character repair | `LibraryCharacterRepairDialog` modal (`LM/library_character_repair_controller.py:275`) | — | — | `LM/library_navigation_controller.py:136-184` |

Modals used in scope:
- `ConfirmationDialog` (archive/restore conversations) — `LM/library_conversations_controller.py:1669-1678`.
- `PromptDeleteConfirmationModal` — `WL/prompt_delete_confirmation_modal.py:81-213`.
- `PromptCollectionManagerModal` ("Manage Prompt collections") — `LM/prompt_collection_manager_modal.py:138-280`.
- `PromptVariablesDialog` (Use in Console with variables/System) — `LS:26970-26992`.
- `FileOpen` / `FileSave` pickers (prompt import/export, report export) — `LM/library_prompts_controller.py:1680-1683`, `:3714-3726`; `LM/library_artifacts_controller.py:757-766`.
- `SkillImportChoiceModal` — `LM/skill_import_choice_modal.py:14-107`.
- `KeptBriefingsModal` (artifact "Scripts…") — `LM/library_artifacts_controller.py:658-668`.
- `ArtifactShareDialog` — `LM/library_artifacts_share_controller.py:170-176`.
- `LibraryCharacterRepairDialog` — `LM/library_character_repair_controller.py:275-518`.

---

## 2. States

### 2.1 Conversations — list (Items pane)

| State | What renders | Cite |
|---|---|---|
| Loading (initial broad snapshot) | Static "Loading local Library sources…" in place of canvas | `LS:15346-15351` |
| Empty (fresh, active scope) | Title "Conversations (0)"; scope toolbar; "No conversations yet. Chat in Console and it appears here."; button **Start in Console** | `WL/library_conversations_canvas.py:114-144`, `L/library_conversations_state.py:18-20`, `:365-374` |
| Empty (archived scope) | "No archived conversations. Archive saved chats from Active." + the **same** "Start in Console" button | `L/library_conversations_state.py:370-374`, `WL/library_conversations_canvas.py:134-143` |
| No matches (query) | Filter input kept; "No conversations match '<q>'."; button **Clear filter** | `WL/library_conversations_canvas.py:122-143`, `L/library_conversations_state.py:367-368` |
| Populated | Title "Conversations (N)"; [Export…] [Select]; filter "Filter conversations… (Enter)"; status line (e.g. "3 matches for 'x'"); rows `▸ <title>\n    <N messages · 2d · <Workspace> · Active>`; pager | `WL/library_conversations_canvas.py:145-422`, `L/library_conversations_state.py:220-227`, `:328-338`, `:356-363` |
| Row secondary when message count unknown | literally "conversation" | `L/library_conversations_state.py:220-222` |
| Stale page | `actions_disabled=True`, rows/actions disabled with "○" and tooltip = pager status ("List may be out of date") | `L/library_conversations_state.py:305-307`, `:422`; `WL/library_conversations_canvas.py:145-161`, `:332-338` |
| Request error | Pager status + **Try again** (`#library-conversations-retry`) | `WL/library_conversations_canvas.py:406-412` |
| One page | Pager controls hidden by shared one-page rule | `WL/library_conversations_canvas.py:357-394` |
| Select mode | "N selected" · **Select all N shown** / **Clear** · **Export selected** / **Archive selected** · **Restore selected**; rows prefixed ☐/☑ | `WL/library_conversations_canvas.py:184-277`, `:309-312` |
| Busy (archive in flight) | `actions_disabled` forced true by recovery projection | `LM/library_conversation_recovery.py:46-54` |
| Archive/restore receipt | "Archived N conversation(s)." (+ " Unchanged: <title>: <reason>" + " Current Console context is unchanged." for restores) + **Undo** + **View archived** | `LM/library_conversation_recovery.py:175-192`, `WL/library_conversations_canvas.py:96-113` |
| Archive failure | "Could not update conversations. Refresh the list and retry." / "Undo did not complete. Retry Undo or refresh the list." | `LM/library_conversation_recovery.py:193-201` |
| Unavailable-character projection | Title "Unavailable character conversations (N)"; no scope toolbar (scope "") | `LM/library_unavailable_navigation.py:70-84` |

### 2.2 Conversations — reader (Work pane)

| State | Status line / controls | Cite |
|---|---|---|
| Nothing selected | "Select a conversation to read it here." | `WL/library_conversation_reader.py:588` |
| Loading | "Loading <title> (<id>)…" or "Loading <A> (<idA>); showing <B> (<idB>) until ready." | `WL/library_conversation_reader.py:567-574` |
| Loaded | "Loaded <title> · X of Y messages · complete/loading more." (+ " · Find: N exact matches.") | `WL/library_conversation_reader.py:575-587`, `:523-532` |
| Error | error copy + " Selected <title> (<id>). Showing <title> (<id>)." + **Try again** | `WL/library_conversation_reader.py:553-566`, `:369-376`; copies `LM/library_conversation_reader_controller.py:546-776` |
| Unavailable | "Conversation unavailable." + ids | `WL/library_conversation_reader.py:546-552` |
| Bulk selection active | "Bulk selection: N conversations. The retained transcript is (not) included and remains read-only." | `WL/library_conversation_reader.py:534-545` |
| Workspace-blocked | "Use as source" enabled with reason line "This conversation is <blocked>. Pressing this adds it to the active workspace first, and you can undo that." OR disabled "○ Use as source" + detail sentence; **Link to workspace** shown when linkable | `WL/library_conversation_reader.py:80-127`, `:296-346` |
| Linked receipt | "✓ linked · <workspace> · this conversation can now be used in Console" + **Undo link** | `WL/library_conversation_reader.py:351-368` |
| Archived conversation | primary label becomes **Restore and resume**; **Restore only** replaces **Archive conversation** | `WL/library_conversation_reader.py:282-292`, `:324-338` |
| Info mode | "Title / Conversation ID / Version / Messages / Workspace / Updated / Keywords / Authority: local saved conversation" | `WL/library_conversation_reader.py:471-503` |

### 2.3 Prompts — list (Items pane)

| State | Render | Cite |
|---|---|---|
| Empty library | "Prompts (0)"; "No prompts yet. Create or import a prompt to begin."; **New prompt** (primary) + **Import…** | `WL/library_prompts_canvas.py:600-662`, `:64` |
| Empty collection | "collection: <name>" label; "This collection has no prompts. Choose another collection or add prompts."; **All prompts** | `WL/library_prompts_canvas.py:628-636`, `:65-67` |
| No matches | Filter kept; `No prompts match "<q>". Clear the search or try different words.`; **Clear filter** | `WL/library_prompts_canvas.py:615-627` |
| Populated | "Prompts (N)"; filter "Filter prompts… (Enter)"; **collection: All prompts** chooser; **sort: Newest**, **Select**; **Import…**, **Export…**; rows `<name>\n<Local · has system and user text>\n<description>` | `WL/library_prompts_canvas.py:712-934`, `:1021-1075`; row parts `L/library_prompts_state.py:2148-2168` |
| Loading / error (no pager) | "Loading prompts…" / `browse_result.error` + **Retry** | `WL/library_prompts_canvas.py:955-1000` |
| Stale page | rows/actions disabled with tooltip "List may be out of date. Retry or change the scope." | `WL/library_prompts_canvas.py:80`, `:744-755` |
| Mutation in flight | "Updating selected items…" and every control disabled with that tooltip | `WL/library_prompts_canvas.py:77`, `:932-952` |
| Select mode | "N selected · M on this page"; **Select page** / **Clear all**; **Done**; **Export selected**; **Delete selected**; reason line "Select one or more items to use bulk actions." | `WL/library_prompts_canvas.py:756-866` |
| Delete receipt | "✓ deleted · Prompt · <name>" or "✓ deleted · N items" + **Undo** / **Dismiss** | `WL/library_prompts_canvas.py:664-711` |
| Sort strip open | toolbar row swapped for Newest / Name chooser | `WL/library_prompts_canvas.py:859-908` |

### 2.4 Prompts — work pane

| State | Render | Cite |
|---|---|---|
| No selection | "Select a prompt to edit it here." | `WL/library_prompt_work_pane.py:31-38` |
| Import open | "Import prompts" heading + path input "File or folder path…" + **Browse…** **Import** **Cancel** + outcome line | `WL/library_prompt_work_pane.py:23-30`, `WL/library_prompts_canvas.py:1169-1226` |
| Loading detail | "Loading prompt…" (+ **Retry** on failure) | `WL/library_prompts_canvas.py:235-249` |
| Editor — new | **‹ Back to list**; Name; Description; Basic/Advanced/Info; **Save prompt** + **Cancel** | `WL/library_prompts_canvas.py:1294-1310`, `:1628-1695` |
| Editor — saved & clean | header **Use in Console**; **More actions** (Export…, Copy Markdown, Duplicate, Collections, History, Delete) | `WL/library_prompts_canvas.py:1360-1382`, `:1647-1718` |
| Editor — dirty | meta line gets unsaved marker; **Save changes** + **Discard changes**; Use in Console and More actions hidden | `WL/library_prompts_canvas.py:500-553`, `:1628-1695` |
| Conflict | "This item changed elsewhere — Reload the current version or save your kept blocks as a new item." + **Save as new** / **Reload** | `WL/library_prompts_canvas.py:1546-1553`, `:1654-1671` |
| Name-in-use | status "Name already in use — pick another or open the existing prompt." + **Open existing** | `LM/screen_constants.py:199-204`, `WL/library_prompts_canvas.py:1554-1571` |
| Bulk read-only preview | "Read-only preview · (Not) included in bulk selection" | `WL/library_prompts_canvas.py:1268-1278` |
| Identity mismatch | detail notice + **Retry**, fields read-only | `WL/library_prompts_canvas.py:1279-1293` |
| Unsupported artifact | compatibility reason + **Convert and save as new Prompt** | `WL/library_prompts_canvas.py:1440-1460` |
| History disclosure | "Loading retained history…", "No retained versions are available.", rows "vN · change C · <timestamp> · Prompt", "Restore selected version…" | `LM/prompt_history_region.py:265-530` |

### 2.5 Skills — list and work pane

| State | Render | Cite |
|---|---|---|
| Empty | "No skills yet — use Create ▸ New skill in the rail, or Import skill… above." | `WL/library_skills_canvas.py:77-79`, `:1297-1302` |
| No matches | "No skills match your filter." | `WL/library_skills_canvas.py:80`, `:1297-1302` |
| Populated | "Skills (N)"; trust header; filter "Filter skills… (Enter)"; **sort: Name**, **Import skill…**; rows `<✓/⚠> <name> · <trust label>[ · Built-in[ · Disabled]][ · overrides built-in]` + secondary Static | `WL/library_skills_canvas.py:1226-1358` |
| Trust header postures | "Skill trust isn't set up, so every skill you added reads "needs review" — built-in skills don't need it. Set it up to review and use yours." + **Set up skill trust**; locked → **Unlock** + **Reset skill trust…**; error → "Skill trust can't be verified — set it up again."; ready+blocked → "N skills need review" + **Review** | `L/library_skills_state.py:866-900`, `WL/library_skills_canvas.py:143-150`, `:607-660` |
| Reset confirm | "Reset skill trust? Every skill will need re-approval. Your skills are not deleted." + **Reset** / **Cancel** | `WL/library_skills_canvas.py:240-242`, `:640-660` |
| Work — nothing selected | "Select a skill to inspect it here." | `WL/library_skill_work_pane.py:70-77` |
| Work — loading | "Loading skill…" (+ **Retry**) | `WL/library_skills_canvas.py:855-869` |
| Work — built-in | "<name> · Built-in (read-only)"; **Customize**; " Enabled " + `Switch`; Markdown body | `WL/library_skill_work_pane.py:146-176` |
| Work — user skill | mode strip **Overview / Edit / Trust / Files** | `WL/library_skill_work_pane.py:178-198` |
| Overview | name, description or "No description set.", invocation copy, "Version: N", "Trust: <raw status with spaces>" | `WL/library_skill_work_pane.py:81-109` |
| Edit | Name (disabled for existing, "Rename isn't supported — create a new skill instead."), Description, **Show advanced/Show basic**, **User can invoke: ✓ yes ⇄ no**, **Agent can invoke**, Body; advanced: Allowed tools picker, **Runs in: inline (this conversation) / fork**, Imported model | `WL/library_skills_canvas.py:1551-1760` |
| Edit — action row lifecycle | create: **Save skill** + **Cancel**; dirty: **Save changes** + **Discard changes**; clean: **Back to list** + **More actions** (→ **Delete**); busy: "Saving changes…" + "Editor actions are unavailable until saving finishes."; conflict: **Reload** | `WL/library_skills_canvas.py:1085-1160`, `:1775-1866` |
| Delete armed | `Delete "<name>"? This removes the skill's directory and N supporting files and cannot be undone.` + **Delete** / **Cancel** | `WL/library_skills_canvas.py:449-465` |
| Trust tab | "Trust: <state>"; **View details** (healthy) or remediation + script grant line + **Revoke script access**; **Unlock** / **Review changes** / **Approve**; first run: explanation + **Set up skill trust** | `WL/library_skills_canvas.py:1869-2035` |
| Files tab | "Supporting files"; "Read-only in Library. File contents are not editable here."; list "<file> (N bytes)" | `WL/library_skill_work_pane.py:128-144`, `WL/library_skills_canvas.py:558-576` |
| Import row (work pane) | "Import skills"; input "SKILL.md file or skill folder path… or GitHub/zip URL"; **Browse…**, **Browse folder…**; **Import**, **Cancel**; status; structural **Cancel**; recovery bullets; **Retry**; **Review "<name>"…** | `WL/library_skill_work_pane.py:55-63`, `WL/library_skills_canvas.py:1448-1549` |
| Import outcomes | "Inspecting/importing…", `Imported "<name>" · re-review it in the trust panel`, `Skipped — a skill named "<name>" already exists.`, "Could not find that file or folder.", "No SKILL.md found in that folder.", "Remote skill import is disabled by policy.", "Import cancelled · check the skills list before retrying." | `LM/library_skill_import_controller.py:169`, `:409`, `:454`, `:563`, `:761`, `:770-789` |

### 2.6 Artifacts (All / Chatbooks / Reports)

| State | Render | Cite |
|---|---|---|
| Heading | "Reports · Local" / "Chatbooks · Local" / "All artifacts · Local" | `WL/library_artifacts_widgets.py:66-72` |
| Loading | count line "Loading…" (initial compose text "Loading reports…" in every view) | `WL/library_artifacts_widgets.py:52`, `:112` |
| Empty (reports, and All) | "No reports yet. Open Watchlists or try the report demo." | `WL/library_artifacts_widgets.py:113-122` |
| Empty (chatbooks) | "No Chatbooks yet. Open Manage Chatbook packs." | `WL/library_artifacts_widgets.py:119-120` |
| No matches | "No matches. Clear search or change the filter." | `WL/library_artifacts_widgets.py:117-118` |
| Populated | count "1–20 of N copies"; OptionList rows `<title>\n<timestamp>\n<Live / Kept / Saved response / Registered> · <status> · #<native id>` | `WL/library_artifacts_widgets.py:92-104`, `:114-116`; labels `L/library_artifacts_state.py:198`, `Chatbooks/artifact_registry_snapshot.py:137` |
| Error | error + **Retry** | `WL/library_artifacts_widgets.py:124-126` |
| Reader — none | "Select a report" → "Select an artifact"; body "Choose an item to read it here." | `WL/library_artifacts_widgets.py:190-197`, `:220-222` |
| Reader — retention line | Chatbook: "<Saved response/Registry metadata> · Original was truncated · <sharing>"; kept: "Kept in Library · independent saved copy"; live: "Watchlist copy · disappears if its Watchlist is deleted" | `WL/library_artifacts_widgets.py:223-243` |
| Reader — no body | "This report is <status>. Open Watchlists for its status." | `WL/library_artifacts_widgets.py:254-257` |
| Reader — stale | "This copy is no longer available. Retry to refresh the list." / "This copy changed. Retry to read its latest version." | `LM/library_artifacts_controller.py:627-635` |
| Share strip | "Sharing N Chatbooks · <url…>" + **Manage** / **Stop** | `LM/library_artifacts_share_controller.py:21-56`, `:292-303` |

### 2.7 Study handoff canvases

| State | Render | Cite |
|---|---|---|
| With sources | header (Study decks/Flashcards/Quizzes); purpose; "Carries forward: …" (titles, cap 3 + "and N more"); "This page shows what carries over; generation and review run in Study."; "Source snapshot is ready."; **Continue in Study** | `WL/library_entry_canvases.py:535-562`, `LM/screen_constants.py:325-365`, `LS:14437-14471` |
| No sources (blocked) | recovery line styled `ds-recovery-callout is-blocked`: "Import sources or create notes first, or open <Study Dashboard/Flashcards/Quizzes> globally without Library context."; button unchanged and still enabled | `LS:14463-14470`, `WL/library_entry_canvases.py:546-562` |

### 2.8 Character return / repair

| State | Render | Cite |
|---|---|---|
| Arrived via Console character inspection/browse | **Back to Console** button above header (only while route admission current and Conversations row selected) | `LM/library_unavailable_navigation.py:150-162`, `WL/library_character_return.py:10-30` |
| Suspended (navigated away) | return cleared on suspend | `LS:9231`, `LM/library_unavailable_navigation.py:179-188` |
| Repair dialog | "Repair saved character link"; "Historical identity: …"; Select "Choose a replacement; nothing is preselected" (options "<name> · local character <id>"); status; **Next 20 characters**; **Refresh** **Repair**→**Confirm repair** **Cancel** | `LM/library_character_repair_controller.py:299-338`, `:450-476` |
| Repair statuses | "Refreshing authoritative repair choices…", "Refresh failed. Retry or cancel.", "Replace <old> with <new>. Press Repair to review.", "Confirm the old and selected identities before applying.", "Repair failed. Retry or cancel.", "Character link repaired." | `LM/library_character_repair_controller.py:368`, `:384`, `:431`, `:461`, `:490`, `:266` |
| Data Profile changed | toast "The active Data Profile changed. Repair was not applied." | `LM/library_navigation_controller.py:152-158` |

---

## 3. Controls & copy

### 3.1 Conversations

| Control (id) | Label as rendered | Behaviour | Cite |
|---|---|---|---|
| `#library-conversations-scope-{active,archived,all}` | "✓ Active" / "Archived" / "All" | reload page 1 with query, focus the scope button | `WL/library_conversations_canvas.py:86-95`, `LM/library_conversation_recovery.py:56-73` |
| `#library-conversations-export` | "Export…" (hidden in select mode) | open Export canvas scoped `kind="conversations"` | `WL/library_conversations_canvas.py:151-161`, `LM/library_conversations_controller.py:1421-1432` |
| `#library-conversations-select-toggle` | "Select" / "Done"; disabled tooltip "Nothing here to select yet." | toggle select mode, clears selection | `WL/library_conversations_canvas.py:165-183`, `LM/library_conversations_controller.py:1361-1384` |
| `#library-conversations-select-all` | "Select all N shown" | select rendered rows | `WL/library_conversations_canvas.py:213-224` |
| `#library-conversations-select-clear` | "Clear" | clear selection | `WL/library_conversations_canvas.py:230-239` |
| `#library-conversations-export-selected` | "Export selected" (○ + tooltip when 0) | Export canvas with selected ids | `WL/library_conversations_canvas.py:240-263`, `LS:22801-22817` |
| `#library-conversations-archive-selected` / `-restore-selected` | "Archive selected" / "Restore selected" — no ○ marker, no tooltip; both always shown | ConfirmationDialog "Archive N conversation(s)?" / "Restore N conversation(s)?" | `WL/library_conversations_canvas.py:265-277`, `LM/library_conversations_controller.py:1643-1678` |
| `#library-conversations-filter` | placeholder "Filter conversations… (Enter)" | Enter-submit only (no as-you-type) | `WL/library_conversations_canvas.py:283-288`, `LM/library_conversations_controller.py:1434-1446` |
| `.library-conversation-row` | `▸ <title>\n    <secondary>`; tooltip "<title> · <secondary>" | select → reader loads; in select mode toggles ☐/☑ | `WL/library_conversations_canvas.py:306-341`, `LS:22690-22740` |
| `#library-conversations-previous/-next/-retry` | "Previous" / "Next" / "Try again" | page request | `WL/library_conversations_canvas.py:395-422` |
| `#library-conversations-undo` / `-view-archived` | "Undo" / "View archived" | reverse last change / switch scope to Archived | `WL/library_conversations_canvas.py:102-113`, `LM/library_conversations_controller.py:1618-1640` |
| `#library-conversations-empty-console` | "Start in Console" | `open_console_for_live_work(source="library-conversations-empty", title="Start a conversation")` | `LS:34182-34192` |
| `#library-conversation-reader-read/-info` | "Read" / "Info" | switch reader mode | `WL/library_conversation_reader.py:257-274` |
| `#library-conversation-open-console` | "Resume conversation" / "Restore and resume" — no ○ marker | `resume_console_conversation` → Console restore dialog when archived → Console | `WL/library_conversation_reader.py:282-295`, `LM/library_conversations_controller.py:1727-1744`, `UI/Console_Modules/archive.py:18-220` |
| `#library-conversation-use-source` | "Use as source" / "○ Use as source" | stage `ChatHandoffPayload(item_type="conversation")`, linking to workspace first when that is the only block | `WL/library_conversation_reader.py:296-304`, `LS:35420-35439`, `LS:13601-13669` |
| `#library-conversation-archive` / `-restore` | "Archive conversation" / "Restore only" (no tooltip when disabled) | same ConfirmationDialog, single id | `WL/library_conversation_reader.py:324-338` |
| `#library-conversation-link-workspace` | "Link to workspace" | add membership to active workspace | `WL/library_conversation_reader.py:339-346`, `LS:35451-35461` |
| `#library-conversation-link-undo` | "Undo link" | remove that membership | `WL/library_conversation_reader.py:361-368`, `LS:35441-35449` |
| `#library-conversation-reader-find` | placeholder "Find in complete transcript…" (Enter) | find after full hydrate | `WL/library_conversation_reader.py:382-388`, `LM/library_conversation_reader_controller.py:882-906` |
| `#library-conversation-find-previous/-next` | "Find previous" / "Find next"; position "Match i of n · message m" | cycle matches | `WL/library_conversation_reader.py:389-402`, `:711-755` |

Confirm dialog copy (`LM/library_conversations_controller.py:1673-1678`): title "Archive N conversation(s)?" / "Restore N conversation(s)?"; message "Saved messages stay available in Archived. Active work or unsaved drafts are skipped." / "Restore these saved conversations to Active. Console context stays unchanged."; confirm label "Archive" / "Restore".
Console-side restore dialog (`UI/Console_Modules/archive.py:205-218`): "Restore workspace and resume?" / "Restore conversation?", confirm "Restore & resume".

### 3.2 Prompts

| Control (id) | Label | Behaviour | Cite |
|---|---|---|---|
| `#library-prompts-filter` | "Filter prompts… (Enter)" | debounced search on change (0.25 s); Enter flushes | `WL/library_prompts_canvas.py:712-718`, `LM/library_prompts_controller.py:1560-1565`, `LS:24426-24432`, `LM/screen_constants.py:85` |
| `#library-prompts-collection` | "collection: All prompts"; tooltip "Press to pick the prompt scope: All prompts, or one collection." | opens Manage Prompt collections (browse mode) | `WL/library_prompts_canvas.py:719-740` |
| `#library-prompts-sort` | "sort: Newest" / "sort: Name" | opens chooser strip | `WL/library_prompts_canvas.py:859-908` |
| `#library-prompts-select` | "Select" | enter select mode | `WL/library_prompts_canvas.py:859-897` |
| `#library-prompts-import` | "Import…" | opens import row in work pane (dirty veto first) | `WL/library_prompts_canvas.py:909-931`, `LM/library_prompts_controller.py:1601-1630` |
| `#library-prompts-export` | "Export…" | Export canvas, `kind="prompts"` (Chatbook .zip) | `LM/library_prompts_controller.py:5137-5151` |
| `#library-prompts-export-selected` / `-delete-selected` | "Export selected" / "Delete selected" | Export canvas with ids / delete modal | `WL/library_prompts_canvas.py:827-853`, `LM/library_prompts_controller.py:4078-4098` |
| `#library-prompts-delete-undo` / `-receipt-dismiss` | "Undo" / "Dismiss" | restore batch / dismiss receipt | `LM/library_prompts_controller.py:4472-4531` |
| `#library-prompts-import-path/-browse/-run/-cancel` | "File or folder path…" / "Browse…" / "Import" / "Cancel" | file-only picker; folder must be typed | `WL/library_prompts_canvas.py:1169-1226`, `LM/library_prompts_controller.py:1654-1683` |
| Import outcome | "N imported · M skipped (duplicate name)[ · K failed]", "Could not find that file or folder.", "Unsupported file type.", "No supported prompt files found." | (copy) | `LS:26432-26586` |
| `#library-prompt-back` | "‹ Back to list" | guarded exit (dirty veto) | `WL/library_prompts_canvas.py:1294-1302` |
| `#library-prompt-name/-details` | labels "Name", "Description" | | `WL/library_prompts_canvas.py:1303-1320` |
| `#library-prompt-mode-basic/-advanced/-info` | "Basic" "Advanced" "Info" | presentation switch | `WL/library_prompts_canvas.py:1319-1346` |
| `#library-prompt-insert-console` | "Use in Console" (only saved+clean) | Prompt: variables dialog or direct append → Console; Recipe: converts to unsaved Prompt copy (no Console) | `WL/library_prompts_canvas.py:1360-1382`, `LS:26994-27056`, `LM/library_prompts_controller.py:3506-3553` |
| Basic fields | "Instructions" (hint "Instructions the model always follows.") / "Message template" (hint "The message inserted into the composer.") | | `WL/library_prompts_canvas.py:1463-1496`, `:68-71` |
| Advanced extras | artifact status "<Prompt> · <Local> · <format>"; Keywords; Author; "Include current text as starter content"; **Convert and save as new Prompt** | (copy) | `WL/library_prompts_canvas.py:1391-1525` |
| Info | "Persisted source: <Local> · <Prompt> · <format>"; "History and collection memberships describe the saved Prompt; unsaved Basic or Advanced edits remain draft-only until Save." | (copy) | `WL/library_prompts_canvas.py:1526-1544` |
| Memberships | "Collections" + summary ("No collections" / "Current: … · Staged: …"); **Manage collections**/**Retry memberships**; **Apply memberships**; status "Membership changes staged — apply separately from Prompt Save." | (copy) | `WL/library_prompts_canvas.py:1572-1626`, `:1737-1766` |
| Actions | **Save prompt** / **Save changes**; **More actions**; **Save as new**; **Reload**; **Cancel** / **Discard changes** | (copy) | `WL/library_prompts_canvas.py:1628-1695` |
| More actions region | **Export…** (Markdown FileSave) **Copy Markdown** **Duplicate** **Collections** **History** **Delete** | (copy) | `WL/library_prompts_canvas.py:1696-1718`, `LM/library_prompts_controller.py:3654-3726` |

Delete modal (`WL/prompt_delete_confirmation_modal.py:149-195`): title "Delete Prompt?" / "Delete N items?"; body `The saved Prompt "<name>" will be discarded. You can Undo from the Prompts list after deletion.` (dirty: "…and this unsaved working copy will be discarded…"); buttons **Cancel** **Delete**.
Collection manager (`LM/prompt_collection_manager_modal.py:138-280`): "Manage Prompt collections"; "Local only · Prompt Save is separate"; "Search collections… (Enter)"; **All prompts** row (browse mode); checkboxes (membership mode); "No collections match this search."; **Load more**; "Collection name"; **New collection** **Rename selected** **Retry**; **Done** **Cancel**. No delete-collection action anywhere (also stated in `Docs/User_Guide/library/prompts.md:172-173`).

### 3.3 Skills

| Control (id) | Label | Behaviour | Cite |
|---|---|---|---|
| `#library-skills-filter` | "Filter skills… (Enter)" | Enter-submit only | `WL/library_skills_canvas.py:1258-1262`, `LS:24533` |
| `#library-skills-sort` | "sort: Name" / "sort: Status" | chooser strip | `WL/library_skills_canvas.py:1267-1293` |
| `#library-skills-import` | "Import skill…" | opens import in work pane | `WL/library_skills_canvas.py:1279-1284` |
| `#library-skills-trust-action` | "Set up skill trust" / "Unlock" / "Retry" / "Review" / "Review restored skills" | trust posture action | `WL/library_skills_canvas.py:143-150`, `:607-628` |
| `#library-skills-trust-reset` | "Reset skill trust…" (danger) | arms inline confirm | `WL/library_skills_canvas.py:629-660` |
| `.library-skill-row` | `<glyph> <name> · <trust> · Built-in · Disabled` | built-in → read-only preview; user → Overview | `WL/library_skills_canvas.py:1304-1358`, `LM/library_skills_builtin_controller.py:56-75` |
| `#library-skill-mode-*` | "Overview" "Edit" "Trust" "Files" (all but Edit disabled while creating, no reason) | switch work mode | `WL/library_skill_work_pane.py:178-198` |
| `#library-skill-builtin-customize` | "Customize" | copy to user skills, reset editor, return to list; toast "Copied to your skills — edit your copy." (+ " It needs review before the assistant can use it.") | `LM/library_skills_builtin_controller.py:148-197` |
| `#library-skill-builtin-enabled` | Switch beside " Enabled " | writes `[skills] disabled_builtins`; toast on persist fail "Changed for this session, but could not be saved." | `LM/library_skills_builtin_controller.py:199-256` |
| `#library-skill-editor-mode` | "Show advanced" / "Show basic" | | `WL/library_skills_canvas.py:1562-1566` |
| invoke toggles | "User can invoke: ✓ yes ⇄ no", "Agent can invoke: …" | | `WL/library_skills_canvas.py:495-526` |
| context toggle | "Runs in: inline (this conversation) ⇄ fork" | | `WL/library_skills_canvas.py:529-552` |
| `#library-skill-save` etc. | "Save skill"/"Save changes", "Cancel", "Discard changes", "Back to list", "More actions", "Delete" | see 2.5 | `WL/library_skills_canvas.py:1775-1866` |
| Trust panel | "Unlock" / "Review changes" / "Approve" / "Revoke script access" / "View details" / "Set up skill trust" — disabled states have tooltips but no "○" marker | | `WL/library_skills_canvas.py:1869-2035`, tooltips `:386-441` |
| Hint | "ctrl+s Save · esc Back to list" | | `WL/library_skills_canvas.py:446` |
| Import choice modal | "Choose one skill to import"; "This repository contains multiple installable skills. Only the selected subdirectory will be copied for trust review."; **Import skill** **Cancel** | (copy) | `LM/skill_import_choice_modal.py:60-86` |

Save outcomes: `LIBRARY_SKILL_SAVE_STATUS_COPY` "Saved." / "A skill with this name already exists." / "Skill name must use lowercase letters, numbers, and hyphens." / "This skill is blocked by trust review — approve it in the trust panel before saving." (`LM/screen_constants.py:248-254`); post-save "Saved. Review trust before using this Skill with the agent." (`LM/library_skills_controller.py:2246`). Dirty veto toast "Unsaved skill changes — Save or Discard changes first." (`LM/screen_constants.py:240`). Warnings: shadow-name and "Saving marks this skill "needs review" — re-approve it in the trust panel after saving." (`WL/library_skills_canvas.py:109-116`).

### 3.4 Artifacts

| Control (id) | Label | Behaviour | Cite |
|---|---|---|---|
| `#library-artifacts-search` | "Find reports by Watchlist…" / "Find artifacts by title…" | search on change; Escape clears query first | `WL/library_artifacts_widgets.py:22-29`, `:43-47`, `:73-77`, `:148-151` |
| `#library-artifacts-all` / `-kept` | "✓ All reports" / "Kept" (reports view only) | kept-only filter | `WL/library_artifacts_widgets.py:49-50`, `:79-80`, `:136-141` |
| `#library-artifacts-sort` | "Newest" / "A–Z" (bare; no "sort:" prefix) | toggles newest/title | `WL/library_artifacts_widgets.py:51`, `:142-144`, `LM/library_artifacts_controller.py:550-553` |
| `#library-artifacts-list` | OptionList | highlight = select; Enter = focus reader | `WL/library_artifacts_widgets.py:153-166` |
| `#library-artifacts-first/-prev/-next/-last` | "First" "Prev" "Next" "Last" (disabled, no ○, no tooltip) | page | `WL/library_artifacts_widgets.py:54-58`, `:127-135` |
| `#library-artifacts-retry` | "Retry" | | `WL/library_artifacts_widgets.py:59` |
| `#library-artifacts-manage` | "Manage Chatbook packs…" (non-reports views) | navigate `chatbooks` screen | `WL/library_artifacts_widgets.py:60-62`, `LM/library_artifacts_controller.py:560-563` |
| `#library-artifacts-back` / `-preview` / `-details` | "‹ Items" / "Preview" / "Details" | | `WL/library_artifacts_widgets.py:186-189` |
| `#library-artifacts-keep` | "Keep in Library" (live report, complete) | keep → toast "Kept in Library · N scripts added", opens kept copy | `LM/library_artifacts_controller.py:636-657` |
| `#library-artifacts-export` | "Export…" (reports) | FileSave "Export report as Markdown" → "Report exported to <file>" | `LM/library_artifacts_controller.py:730-786` |
| `#library-artifacts-scripts` | "Scripts…" (kept report) | KeptBriefingsModal | `LM/library_artifacts_controller.py:658-668` |
| `#library-artifacts-play` | "Play" | play audio or "This report's audio is no longer available." | `LM/library_artifacts_controller.py:788-806` |
| `#library-artifacts-share` | "Share…" (chatbook with usable ZIP) | share dialog; "Install tldw_chatbook[web] to share Chatbooks." if extra missing | `LM/library_artifacts_share_controller.py:99-118` |
| `#library-artifacts-console` | "Use in Console" (chatbooks only) | `open_console_for_live_work(**launch)` | `LM/library_artifacts_controller.py:681-697` |
| `#library-artifacts-source` | "Open source" (chatbook with source conversation) | `resume_console_conversation(id)` | `LM/library_artifacts_controller.py:672-680` |
| `#library-artifacts-watchlists` / `-demo` | "Watchlists" / "Try report demo" | navigate Watchlists (briefing context for live report) / run demo | `LM/library_artifacts_controller.py:566-616` |
| `#library-artifacts-share-manage` / `-stop` | "Manage" / "Stop" | reopen dialog / stop share immediately, no confirm | `LM/library_artifacts_share_controller.py:47-56`, `:305-314` |
| Action failure toast | `Could not {name} this report. Please retry.` with `name` ∈ keep/export/scripts/play/source/console | | `LM/library_artifacts_controller.py:699-708` |

### 3.5 Study handoffs & rail-level "Use in Console"

- Study canvases: one primary button labelled **Continue in Study** on all three (`LM/screen_constants.py:337`, `:343`, `:349`), ids `#library-open-study|flashcards|quizzes` (`LS:14286-14290`); tooltip "Open <Study Dashboard|Flashcards|Quizzes> with the current Library source snapshot, or globally when none is available." (`WL/library_entry_canvases.py:556-562`).
- Rail Details "Workspace" group: "Active: <ws>", "Handoff: <summary>" (e.g. "N items can't be used in Console yet · <reason> · <remedy>") (`LS:14701-14734`, `LS:14622-14700`). Actions group: **Create local workspace**, **Import sources**, **Use in Console** (`#library-use-in-console`, stays pressable when blocked, dimmed via class) — stages a whole-library "Local Library Sources" snapshot (`LS:14760-14834`, `LS:35735-35778`).

---

## 4. Key bindings

Screen BINDINGS: `LS:1010-1222`; gates: `check_action` `LS:25253-25600`; manual keys: `on_key` `LS:8749-8955`; footer sets `LS:4193-4520`.

| Key | Where active | Action / footer chip | Cite |
|---|---|---|---|
| `tab` / `shift+tab` | all Library | cycle within `#screen-content` (not nav bar) | `LS:1020-1021`, `LS:8980-9008` |
| `F6` / `shift+F6` | all | next/previous workbench pane; chip "F6 next pane" | `LS:1022-1028`, `LS:8960-8972`, `LS:9154-9160` |
| `/` | Conversations (Items open) | focus `#library-conversations-filter`; chip "/ focus filter" | `LS:8862-8872`, `LS:4278-4279` |
| `/` | Prompts list, Media | focus that canvas's filter; chip still reads "/ focus search" | `LM/screen_constants.py:457-460`, `LS:8885-8894`, `LS:1352-1356` |
| `/` | Skills, Study, Export, others | focus rail `#library-search-input` | `LS:8908-8920` |
| `/` | Artifacts (focus in shell) | focus artifacts search (shell `on_key`) | `WL/library_artifacts_reader_shell.py:109-112` |
| `i` | any canvas except Ingest, not typing | open Import… | `LS:8929-8939` |
| `n`, `ctrl+n` | landing and Notes only | New note (no New prompt / New skill key) | `LS:8940-8955`, `LS:25304-25334` |
| `up` / `down` | focused `.library-conversation-row` / `.library-prompt-row` / `.library-skill-row` | move row focus | `LS:8853-8858`, `LM/screen_constants.py:434-450` |
| `c` | Conversations with a conversation selected/loaded | Resume conversation; when not ready, toast with block sentence; chip "c resume conversation" only when ready | `LS:1185-1195`, `LS:25453-25465`, `LS:35469-35481`, `LS:4285-4286` |
| `c` | Media reader | Use in Console (different semantics, same key) | `LS:1184`, `LS:4411` |
| `ctrl+f` | Media reader only (not Conversations reader) | Find | `LS:1202` |
| `ctrl+s` | Skill editor with save available | Save skill; chip "ctrl+s save skill" | `LS:1047`, `LS:25243-25251`, `LS:4497-4509` |
| `ctrl+s` | Prompt editor | **unbound** (explicit-save-only editor) | `LS:1010-1222` (no prompt save binding) |
| `escape` | Prompt editor (not select mode) | back to list; chip "esc back to list" / "esc save or discard first" / "esc busy, try again" | `LS:1074`, `LS:25420-25424`, `LS:4466-4495`, `LM/screen_constants.py:226`, `:232` |
| `escape` | Prompt editor with More actions open | close region (canvas `on_key`) | `WL/library_prompts_canvas.py:564-580` |
| `escape` | Skill editor (view=editor incl. Overview/Trust/Files) | back to skills list / "close more actions" | `LS:1048`, `LS:4497-4509` |
| `escape` | Skill built-in preview (view=preview), wide layout, focus not in a text field | no surface-specific binding matches (`library_skill_back` needs view=editor; focus-rail needs view=list); no esc chip. Only the narrow-stage/emergency returns can fire | `LS:25365-25368`, `LS:10452-10453`, `LS:25518-25530` |
| `escape` | Prompts/Skills list, Conversations Items/Work | focus hop: "esc focus rail" (lists); Conversations "esc focus Items" / "esc focus Library" | `LS:25518-25530`, `LM/library_conversations_controller.py:774-782` |
| `escape` | Study handoff canvas | back to hub; chip "esc back to hub" | `LS:1093`, `LS:25435-25439`, `LS:4263-4268` |
| `escape` | Artifacts shell | focus Items (after clearing search text first) | `WL/library_artifacts_reader_shell.py:98-108` |
| `escape` | in a text field (last resort) | blur to canvas | `LS:1114-1128` |
| `escape` | delete modal / collection manager / skill import choice | cancel | `WL/prompt_delete_confirmation_modal.py:128`, `LM/prompt_collection_manager_modal.py:46`, `LM/skill_import_choice_modal.py:17` |
| `escape` | Character repair dialog | **no binding declared** | `LM/library_character_repair_controller.py:275-298` |
| `enter` / `space` | Buttons; OptionList Enter on artifacts → focus reader | | `WL/library_artifacts_widgets.py:163-166` |
| `u`, `enter`, `o` | Search/RAG only (out of scope) | | `LS:1029`, `LS:1140-1143` |

Footer sets are mostly hidden bindings (`show=False`); the footer is the only advertisement (`LS:4193-4520`).

---

## 5. Flows (first-timer / power user)

Each flow lists the steps as the code drives them. "How to reach live" names the anchors.

### F1. Conversations — first-timer, empty library
1. Library opens on landing; "Get started" shows Import a file / Find it / Use it in Console steps — no mention of Conversations, Prompts, Skills or Artifacts (`WL/library_entry_canvases.py:262-428`).
2. Rail ▸ Browse ▸ **Conversations** (may read **Chats** in a narrow rail) (`L/library_shell_state.py:507-522`).
3. Empty state: "No conversations yet. Chat in Console and it appears here." + **Start in Console** (`WL/library_conversations_canvas.py:114-144`).
4. Start in Console → Console opens with a live-work card titled "Start a conversation" (`LS:34182-34192`, `app_destinations.py:224-248`).
- Reach live: fresh profile → nav Library → `#library-row-browse-conversations` → `#library-conversations-empty-console`.

### F2. Conversations — first-timer, read and resume
1. Select row → canvas sync + reader load ("Loading <title> (<id>)…") (`LS:22734-22740`, `WL/library_conversation_reader.py:567-574`).
2. Read transcript (messages show "<sender> · <age>") (`WL/library_conversation_reader.py:448-458`); Info shows IDs/version (`:471-503`).
3. **Resume conversation** (or `c`) → Console; if archived, Console shows "Restore conversation?" / "Restore workspace and resume?" with **Restore & resume** (`UI/Console_Modules/archive.py:205-218`).
4. Return: navigating back to Library reuses the suspended screen; Conversations re-requests its page on resume (`LS:9262-9337`).
- Reach live: populated profile → `#library-row-browse-conversations` → `.library-conversation-row` → `#library-conversation-open-console`.

### F3. Conversations — power user, bulk archive with undo
1. From Console: `#console-search-all` / `#console-open-archive` → Library Conversations with scope all/archived (`UI/Screens/chat_screen.py:17440-17444`, `app_destinations.py:142-160`).
2. Filter (type, Enter) → **Select** → **Select all N shown** → **Archive selected** → ConfirmationDialog → receipt "Archived N conversation(s)." + **Undo** + **View archived** (`LM/library_conversations_controller.py:1643-1678`, `LM/library_conversation_recovery.py:175-192`).
3. **Export selected** → Export canvas; Esc returns to Conversations (`LS:22801-22817`, `LS:4248-4262`).
4. Find in transcript → **Find next**/**Find previous** (`WL/library_conversation_reader.py:711-755`).
- Reach live: Console ▸ "Search all" button → filter → `#library-conversations-select-toggle` → `#library-conversations-select-all` → `#library-conversations-archive-selected`.

### F4. Conversations — use as source (workspace-gated)
1. Loaded conversation; if blocked by workspace membership, reason line under "Use as source" says pressing will add it to the active workspace (`WL/library_conversation_reader.py:115-127`).
2. **Use as source** → link (if needed) → `open_chat_with_handoff(ChatHandoffPayload(item_type="conversation"))` → Console staged-context lane (`LS:35420-35439`, `LM/library_conversations_controller.py:1681-1725`, `UI/Screens/chat_screen.py:17531-17600`).
3. What travels: an excerpt of the **first** ≤3,000 chars of the transcript (oldest messages first, senders capped 80 chars) + title, id, message count, workspace, updated, "Source authority: local" (`LS:13612-13669`, `LM/screen_constants.py:263-264`); Console further caps snippet at 4,000 chars (`UI/Screens/chat_screen.py:17549-17552`).
4. Back in Library: receipt "✓ linked · <ws> · this conversation can now be used in Console" + **Undo link** (`WL/library_conversation_reader.py:347-368`).

### F5. Prompts — first-timer, create and use
1. Rail ▸ **Prompts · reuse** → "No prompts yet. Create or import a prompt to begin." → **New prompt** (routes to rail row `create-prompt`) (`WL/library_prompts_canvas.py:637-662`, `LS:24318-24324`).
2. Editor: Name, Description, Basic (Instructions / Message template) → **Save prompt** (`WL/library_prompts_canvas.py:1294-1493`, `:1628-1636`).
3. After save, **Use in Console** appears in the header (`WL/library_prompts_canvas.py:1360-1382`).
4. Use in Console → if no Console target yet: toast "Open Console once, then retry Use in Console." (`LS:26913-26920`); else PromptVariablesDialog (variables/System) or direct append → navigates to Console (`LS:26944-26992`, `app_destinations.py:202-221`).
- Reach live: `#library-row-browse-prompts` → `#library-prompts-empty-new` → fill `#library-prompt-name`, `#library-prompt-user` → `#library-prompt-save` → `#library-prompt-insert-console`.

### F6. Prompts — power user
1. Filter as you type; **collection:** chooser → Manage Prompt collections modal (search, Load more, New collection, Rename selected, Done) (`LM/prompt_collection_manager_modal.py:138-280`).
2. **sort:** strip; **Select** → **Select page** → **Delete selected** → modal ("You can Undo from the Prompts list after deletion.") → receipt with **Undo** (`WL/prompt_delete_confirmation_modal.py:154-172`, `WL/library_prompts_canvas.py:664-711`).
3. Editor More actions: Duplicate / History (restore retained version) / Collections (manage + **Apply memberships**, separate from Save) / Copy Markdown / Export… (Markdown) / Delete (`WL/library_prompts_canvas.py:1696-1718`, `:1751-1765`).
4. Import… → type folder path (Browse… is file-only) → "N imported · M skipped (duplicate name)" (`LM/library_prompts_controller.py:1654-1683`, `LS:26583-26586`).
5. Inbound: Console "save recipe/prompt" → Library Prompts opened on that item (`UI/Console_Modules/prompts.py:1550-1572`).
- Keyboard: no save key; Escape back to list is vetoed while dirty (chip "esc save or discard first") (`LS:4466-4495`).

### F7. Skills — first-timer, built-in → customize
1. Rail ▸ **Skills · AI add-ons**; trust header may read "Skill trust isn't set up, so every skill you added reads "needs review" — built-in skills don't need it…" + **Set up skill trust** (`L/library_skills_state.py:871-883`).
2. Select a built-in row → "<name> · Built-in (read-only)" with **Customize** and **Enabled** switch (`WL/library_skill_work_pane.py:146-176`).
3. **Customize** → toast "Copied to your skills — edit your copy."; work pane resets to the list (copy not opened) (`LM/library_skills_builtin_controller.py:190-197`).
4. Find the user copy ("overrides built-in") → Overview / Edit / Trust / Files.
- Reach live: `#library-row-browse-skills` → a `.library-skill-row` containing "Built-in" → `#library-skill-builtin-customize`.

### F8. Skills — first-timer, import
1. **Import skill…** → work pane "Import skills" → path/URL → **Import** ("Inspecting/importing…") (`WL/library_skills_canvas.py:1448-1549`, `LM/library_skill_import_controller.py:169`).
2. Repo with multiple skills → "Choose one skill to import" modal (`LM/skill_import_choice_modal.py:60-86`).
3. Success `Imported "<name>" · re-review it in the trust panel` + **Review "<name>"…** → Trust tab (`LM/library_skill_import_controller.py:769-777`).
4. Trust not set up → "Set up skill trust" passphrase modal (reused from retired SkillsScreen) (`WL/library_skills_canvas.py:1975-1987`, `UI/Navigation/screen_registry.py:233-242`); then **Review changes** → **Approve**.

### F9. Skills — power user
1. Rail ▸ Create ▸ **New skill** → Edit tab only (other tabs disabled) → **Show advanced** (Allowed tools picker, Runs in) → `ctrl+s` → "Saved. Review trust before using this Skill with the agent." (`WL/library_skill_work_pane.py:196-198`, `LM/library_skills_controller.py:2246`).
2. Trust tab → Review changes → Approve; Revoke script access.
3. Delete: Back to list state → **More actions** → **Delete** → inline confirm "…cannot be undone." (`WL/library_skills_canvas.py:449-465`, `:1851-1866`).
4. Sort by Status; built-in Enabled switch persisted to config (`LM/library_skills_builtin_controller.py:213-256`).

### F10. Artifacts — first-timer, reports
1. Rail ▸ Artifacts ▸ **Reports** → "No reports yet. Open Watchlists or try the report demo." → **Try report demo** → list refreshes on completion (`WL/library_artifacts_widgets.py:121`, `LM/library_artifacts_controller.py:568-583`).
2. Select row → Preview (Markdown) / Details; **Keep in Library** → toast "Kept in Library · N scripts added", reader moves to the kept copy (`LM/library_artifacts_controller.py:636-657`).
3. **Export…** → FileSave → "Report exported to <file>"; **Play** for audio.
- Reach live: `#library-row-artifacts-reports` → `#library-artifacts-demo` → `#library-artifacts-list` → `#library-artifacts-keep`.

### F11. Artifacts — power user, Chatbooks share / use
1. Rail ▸ Artifacts ▸ **Chatbooks** → select → **Share…** (needs `[web]` extra) → share dialog → strip "Sharing N Chatbooks · <url>" with **Manage** / **Stop** (`LM/library_artifacts_share_controller.py:99-314`).
2. **Use in Console** → Console live-work launch; **Open source** → resume source conversation in Console (`LM/library_artifacts_controller.py:672-697`).
3. **Manage Chatbook packs…** → Chatbooks screen (`LM/library_artifacts_controller.py:560-563`).
4. Inbound exact target: Console live-work card → `ARTIFACT_CHATBOOK_TARGET` handoff → Library switches to Chatbooks row and opens the record; failure → Retry (`LM/library_artifacts_navigation.py:56-205`, `app_destinations.py:307-315`).

### F12. Study handoffs
1. Rail ▸ Study ▸ **Study decks** / **Flashcards** (Cards due: N) / **Quizzes** → staging canvas (2.7).
2. **Continue in Study** → `open_study_screen(scope_context?, initial_section=dashboard|flashcards|quizzes)`; `origin` not passed (Study defaults to Library origin) (`LS:35712-35733`, `app_destinations.py:68-115`).
3. Study Escape → `NavigateToScreen(TAB_LIBRARY, {mode: "study"})` → always the **Study decks** row, whichever row the user left from (`UI/Screens/study_screen.py:1352-1356`, `LM/screen_constants.py:368-410`).
- Reach live: `#library-row-create-flashcards` → `#library-open-flashcards` → Esc in Study.

### F13. Character return / repair (from Console)
1. Console character context → unavailable conversation "view" → Library Conversations with title "Unavailable character conversations (N)" and **Back to Console** (`UI/Console_Modules/character_context.py:1221-1278`, `LM/library_unavailable_navigation.py:70-84`, `:150-176`).
2. Repair → `LibraryCharacterRepairDialog`: pick → **Repair** → **Confirm repair** → "Character link repaired." → navigate back to the return anchor (Console) (`LM/library_character_repair_controller.py:450-513`, `LM/library_navigation_controller.py:192-205`).
3. Leaving Library (suspend) clears the Back to Console control (`LS:9231`).

### F14. Rail-level "Use in Console" (whole library)
1. Rail Details ▸ Actions ▸ **Use in Console** (pressable even when blocked) → stages "Local Library Sources" snapshot (`LS:14826-14834`, `LS:35735-35778`).

---

## 6. Cross-links

### 6.1 OUT of Library

| From (Library control) | Mechanism | Destination | Context that travels | Cite |
|---|---|---|---|---|
| Conversation **Resume conversation** / `c` | `resume_console_conversation` → `request_conversation_resume` | Console (after optional restore dialog) | conversation id only; Console reloads | `LM/library_conversations_controller.py:1737-1744`, `app_destinations.py:163-172` |
| Conversation **Use as source** | `open_chat_with_handoff` (CHAT channel) | Console staged-context lane | first ≤3k chars excerpt + metadata | `LS:13601-13669`, `app_destinations.py:175-199` |
| Conversations empty **Start in Console** | `open_console_for_live_work` | Console live-work card | title "Start a conversation" | `LS:34182-34192` |
| Prompt **Use in Console** | `stage_console_prompt_insert` (CONSOLE_PROMPT_INSERT) | Console composer append | rendered System/User text + target session id | `LS:26882-26992`, `app_destinations.py:202-221` |
| Chatbook **Use in Console** | `open_console_for_live_work(**launch)` | Console live-work card | `ArtifactsScreen._build_chatbook_console_launch(record)` | `LM/library_artifacts_controller.py:681-697` |
| Chatbook **Open source** | `resume_console_conversation` | Console | source conversation id | `LM/library_artifacts_controller.py:672-680` |
| Rail **Use in Console** | `open_chat_with_handoff` | Console | whole local source snapshot | `LS:35735-35778` |
| Report **Watchlists** | `NavigateToScreen("watchlists_collections", ctx)` | Watchlists (artifacts section, briefing id for live report) | backend/section/briefing id | `LM/library_artifacts_controller.py:601-616` |
| **Manage Chatbook packs…** | `NavigateToScreen("chatbooks")` | Chatbooks screen | none | `LM/library_artifacts_controller.py:560-563` |
| Study **Continue in Study** | `open_study_screen` | Study (section) | `StudyScopeContext` or none; no origin | `LS:35712-35733` |
| **Back to Console** (character) | `NavigateToScreen(return_target.screen_id, {return_focus})` | Console (focus `console-context-character`) | focus id | `LM/library_unavailable_navigation.py:165-176` |
| Repair success | `return_from_repair` | return anchor | focus id | `LM/library_navigation_controller.py:192-205` |
| Workspaces (rail) | **Create local workspace** / **Import sources** | Library Import canvas / workspace registry | — | `LS:14780-14818` |

### 6.2 INTO Library

| Origin | Context keys | Lands on | Cite |
|---|---|---|---|
| Console "Search all" / "Open archive" | `mode=conversations`, `conversation_archive_scope`, `conversation_query` | Conversations, scoped | `UI/Screens/chat_screen.py:17440-17444`, `app_destinations.py:142-160` |
| Console archive toast "view" action | same (archived) | Conversations ▸ Archived | `UI/Console_Modules/archive.py:444-446` |
| Console staged-source detail / citation modal | `open_source_type` ∈ media/notes/conversations/prompt + id | that canvas, opens item | `UI/Console_Modules/right_rail.py:672-685`, `UI/Screens/chat_screen.py:17754-17770`, `Widgets/Console/console_citation_sources_modal.py:461` |
| Console prompt/recipe save | `mode=prompts`, `open_source_type=prompt`, id | Prompts editor for that id (dirty veto: "Can't open Prompt <id> — Save or Discard the open Prompt first.") | `UI/Console_Modules/prompts.py:1550-1572`, `LM/screen_constants.py:219-221` |
| Console character context | `LIBRARY_NAV_CONTEXT_CHARACTER_{INSPECTION,BROWSE,REPAIR}` | Conversations (unavailable projection) + Back to Console / repair modal | `UI/Console_Modules/wiring.py:136-180`, `LM/library_unavailable_navigation.py:564-640` |
| Console live-work chatbook action | `ARTIFACT_CHATBOOK_TARGET` channel | Artifacts ▸ Chatbooks, record opened | `app_destinations.py:307-315`, `LM/library_artifacts_navigation.py:56-147` |
| Personas conversations | `mode=conversations`, `conversation_id` | Conversations, that item | `UI/Persona_Modules/personas_conversations_controller.py:1278-1292` |
| Home resume (note / media) | `note_id` / `open_source_type=media` | Notes / Media (conversations resume goes to Console, not Library) | `UI/Screens/home_screen.py:1056-1080` |
| Study Escape | `mode=study` | Study decks staging row | `UI/Screens/study_screen.py:1352-1356` |
| ArtifactsScreen "Open Library" | none | Library as last left | `UI/Screens/artifacts_screen.py:1395-1397` |
| Legacy aliases | `artifacts`, `prompts`, `skills`, `search`, `media` defaults | respective row | `app.py:2442-2449` |
| Command palette / nav bar | none | last state (screen reused) | `LS:9205-9337` |

"What the user sees on return": the Library screen is reusable and suspended rather than unmounted (`LS:9205-9230`); on resume the selected row is kept and its data surface re-fetches (`LS:9262-9337`). Only character-inspection arrivals get an explicit "Back to …" control; no other inbound handoff renders a return affordance.

---

## 7. Suspected issues with live probes

Severity guess: **High** = likely blocks/misleads a core task; **Med** = confusing or inconsistent with siblings; **Low** = polish/jargon.
All items are SUSPECTED from code; verify live before reporting.

### Conversations

1. **Restore selected is live in the Active scope (and Archive selected in Archived)** — Med.
   Evidence: both buttons always rendered in select mode, enabled on any selection (`WL/library_conversations_canvas.py:265-277`); handler captures all selected ids regardless of state (`LM/library_conversations_controller.py:1661-1666`); service reports "Already active." per row (`Chat/conversation_archive_actions.py:41-47`).
   Probe: Conversations ▸ Active ▸ Select ▸ pick 2 ▸ `#library-conversations-restore-selected` ▸ confirm. Expect dialog "Restore 2 conversation(s)?" then receipt "Restored 0 conversation(s). Unchanged: …: Already active. Current Console context is unchanged."

2. **Receipt toolbar always offers "View archived", even after a Restore or an Undo** — Med.
   Evidence: `WL/library_conversations_canvas.py:102-113` renders "View archived" unconditionally with the receipt; restore copy `LM/library_conversation_recovery.py:175-192`.
   Probe: Archived scope ▸ open an archived conversation ▸ `#library-conversation-restore` ▸ confirm ▸ check `#library-conversations-view-archived` is present next to "Restored 1 conversation(s)…".

3. **Archive/restore receipt never clears** (no Dismiss; persists across scope switches and visits because the screen is reused) — Med.
   Evidence: `receipt_copy` only assigned in `_complete_change` (`LM/library_conversation_recovery.py:31`, `:175-201`); no reset in `set_scope` (`:56-73`); screen reuse `LS:9205-9230`.
   Probe: archive one ▸ click "Archived" scope ▸ "All" ▸ go to Home ▸ back to Library ▸ Conversations: is "Archived 1 conversation(s)." still shown?

4. **Archived empty state offers "Start in Console"** — Low.
   Evidence: empty CTA chosen only by query presence (`WL/library_conversations_canvas.py:134-143`), copy for archived scope `L/library_conversations_state.py:370-374`.
   Probe: profile with no archived chats ▸ Conversations ▸ `#library-conversations-scope-archived`.

5. **Bulk Archive/Restore and reader Archive/Restore buttons break the "○ + reason" disabled convention** — Low.
   Evidence: no `library_disabled_action_label`/tooltip on `#library-conversations-archive-selected|restore-selected` (`WL/library_conversations_canvas.py:265-277`) or `#library-conversation-archive|restore` (`WL/library_conversation_reader.py:324-338`); "Resume conversation" also lacks ○ (`:282-295`) while sibling "Use as source" has it (`:206-217`).
   Probe: select mode with 0 selected; and a conversation still loading — compare labels/tooltips of the four buttons.

6. **"(s)" pluralisation in confirm and receipt** — Low.
   Evidence: `LM/library_conversations_controller.py:1673`, `LM/library_conversation_recovery.py:176`.
   Probe: archive a single conversation from the reader.

7. **Use as source stages only the oldest ~3,000 characters** of a long conversation — High (silent context loss).
   Evidence: excerpt loop walks `reader.messages` from index 0 and stops at 3,000 chars (`LS:13612-13629`, `LM/screen_constants.py:263`); Console caps at 4,000 (`UI/Screens/chat_screen.py:17549-17552`).
   Probe: conversation with >30 long messages ▸ Use as source ▸ in Console inspect the staged item content: does it contain the latest messages?

8. **Same key `c`, different meaning across sibling readers** — Med.
   Evidence: Media `c` = "use in Console" (stage source) (`LS:1184`, `:4411`); Conversations `c` = "Resume conversation" (`LS:1190-1195`, `:4286`); the Conversations equivalent of Media's action is "Use as source".
   Probe: Media reader press `c` vs Conversations reader press `c`; compare destination/result.

9. **Reader status and Info expose raw IDs and storage jargon** — Low.
   Evidence: "Selected <title> (<uuid>). Showing …" (`WL/library_conversation_reader.py:546-566`); Info "Conversation ID", "Version", "Authority: local saved conversation" (`:492-502`); bulk copy "retained transcript" (`:534-545`).
   Probe: select a conversation then force an error (or select a second while first loads); open Info.

10. **Filter behaviour differs from Prompts** — Low.
    Evidence: Conversations/Skills filter only on Enter (`LM/library_conversations_controller.py:1434-1446`, `LS:24533`) while Prompts filters as you type (`LM/library_prompts_controller.py:1560-1565`), yet all three placeholders say "(Enter)".
    Probe: type in each filter without pressing Enter.

11. **No Find key in the Conversations reader** — Low.
    Evidence: `ctrl+f` gated to Media (`LS:1202`); Conversations Find is an always-visible Input with Enter (`WL/library_conversation_reader.py:382-402`).
    Probe: open a conversation, press ctrl+f.

12. **Unavailable-character projection may persist after leaving and returning without its Back to Console** — Med.
    Evidence: suspend clears the return control (`LS:9231`, `LM/library_unavailable_navigation.py:179-188`) but resume re-requests the unavailable page while `_library_unavailable_browse_scope` is set (`LS:9337-9346`).
    Probe: Console character context ▸ view unavailable group in Library ▸ nav to Home ▸ nav back to Library: is the title still "Unavailable character conversations (N)" with no Back to Console and no scope toolbar? How does the user get back to normal Conversations?

### Prompts

13. **Prompt editor has no keyboard save** while being explicit-save-only — Med.
    Evidence: only `ctrl+s → library_skill_save` exists (`LS:1047`); prompt editor described as "explicit-Save-only (no autosave)" (`LS:26609-26612`); footer offers only esc (`LS:4466-4495`).
    Probe: open a prompt, edit, press ctrl+s.

14. **"Use in Console" disappears (not disabled) whenever the prompt is new or dirty, with no reason** — Med.
    Evidence: `use_console.display = not conflict and prompt_id is not None and not dirty` and the whole header row hides (`WL/library_prompts_canvas.py:1372-1382`, `:527-532`).
    Probe: open a saved prompt, type one character in Name: the primary action vanishes.

15. **"Use in Console" on a Recipe does not go to Console; it converts to an unsaved Prompt copy** — Med (label vs behaviour).
    Evidence: `LS:27026-27032` → `LM/library_prompts_controller.py:3517-3550` ("Recipe converted to an unsaved Prompt copy; nothing was applied.").
    Probe: open a Recipe artifact in Prompts ▸ Use in Console.

16. **First "Use in Console" can fail with "Open Console once, then retry Use in Console."** when Console has not published a target (e.g. `default_tab` set to library/notes, or after a runtime switch) — Med.
    Evidence: `LS:26913-26920`, `app.py:2825-2837`; default tab is chat (`config.py:3884`).
    Probe: set `[general] default_tab = "library"` in an isolated config, launch, open a saved prompt ▸ Use in Console.

17. **Two "Export…" actions with different results** (list → Chatbook .zip canvas; editor More actions → Markdown file) — Low/Med.
    Evidence: `LM/library_prompts_controller.py:5137-5151` vs `:3654-3726`; guide admits it (`Docs/User_Guide/library/prompts.md:532-534`).
    Probe: press both and compare.

18. **Delete, Duplicate, History, Collections and Export are unreachable while the prompt is dirty** (More actions hidden) — Low.
    Evidence: `WL/library_prompts_canvas.py:533-553`, `:1647-1653`.
    Probe: edit a saved prompt, look for More actions.

19. **Collection membership is a second, separate commit** ("Apply memberships", "Membership changes staged — apply separately from Prompt Save.") — Med (two Save concepts in one editor).
    Evidence: `WL/library_prompts_canvas.py:1572-1626`, `:1751-1765`; modal subtitle "Local only · Prompt Save is separate" (`LM/prompt_collection_manager_modal.py:145-149`).
    Probe: open saved prompt ▸ Info ▸ Manage collections ▸ tick one ▸ Done ▸ leave without Apply: is the membership lost? any warning?

20. **Zero collections reads "No collections match this search."** with an empty search box — Low.
    Evidence: catalog status "empty" whenever no items (`L/library_prompts_state.py:592`), copy at `LM/prompt_collection_manager_modal.py:182-187`.
    Probe: fresh profile ▸ Prompts ▸ `#library-prompts-collection`.

21. **Collections can be created and renamed but never deleted** — Low/Med (power user).
    Evidence: modal actions `LM/prompt_collection_manager_modal.py:238-260`; guide `Docs/User_Guide/library/prompts.md:172-173`.
    Probe: open the collection manager, look for delete.

22. **Terminology drift for the two prompt parts**: editor "Instructions" / "Message template"; list "System only" / "User only" / "has system and user text"; history "Stored System lane" / "Stored User lane"; Info "Persisted source" — Low/Med.
    Evidence: `WL/library_prompts_canvas.py:1463-1496`, `L/library_prompts_state.py:2150-2156`, `LM/prompt_history_region.py:410-425`, `WL/library_prompts_canvas.py:1529-1535`.
    Probe: compare a row, the Basic editor, Advanced, Info and History for one prompt.

23. **Footer advertises "/ focus search" on the Prompts list but `/` focuses the Prompts filter** — Low.
    Evidence: `LIBRARY_LIST_SHORTCUTS` (`LS:1352-1356`) vs `_LIBRARY_SLASH_CANVAS_FILTERS` (`LM/screen_constants.py:457-460`); Conversations relabels to "focus filter" (`LS:4278-4279`).
    Probe: Prompts list, press `/`, read footer.

24. **Import: Browse… picks files only; folder must be typed; duplicates are skipped silently with no preview** — Low.
    Evidence: `LM/library_prompts_controller.py:1654-1683`; `LS:26583-26586`; "Unsupported file type." does not list formats (`LS:26458`).
    Probe: Import… ▸ Browse…; import a folder with an existing name.

25. **Save disabled without a reason** when `can_update_original` is false — Low.
    Evidence: `WL/library_prompts_canvas.py:1629-1644` (no tooltip).
    Probe: open a prompt whose block state is unavailable/legacy and edit it; hover Save changes.

26. **User Guide contradicts code on delete recovery**: footer stamp says delete "cannot be undone from Library"; code shows "You can Undo from the Prompts list after deletion." — Low (docs).
    Evidence: `Docs/User_Guide/library/prompts.md:543-546` vs `WL/prompt_delete_confirmation_modal.py:155`.
    Probe: delete a prompt and read the modal.

27. **Rail highlights "New prompt" / "New skill" while browsing the list after a create** — Low.
    Evidence: create flows keep `_library_selected_row_id` at CREATE_* (`LS:10430-10453`, `LM/screen_constants.py:462-484`).
    Probe: Create ▸ New prompt ▸ Save ▸ ‹ Back to list: which rail row is selected?

### Skills

28. **Customize returns to the list instead of opening the new copy** although the toast says "edit your copy" — Med.
    Evidence: `LM/library_skills_builtin_controller.py:190-197` (`_reset_library_skill_editor_state` then refresh).
    Probe: select a built-in ▸ Customize; where is focus/work pane afterwards?

29. **Skill delete is irreversible and inline, while Prompts uses a modal + Undo and Conversations a dialog + Undo** — Med (inconsistent destructive patterns).
    Evidence: `WL/library_skills_canvas.py:449-465`; prompts `WL/prompt_delete_confirmation_modal.py`; conversations `LM/library_conversations_controller.py:1669-1678`.
    Probe: delete one item in each canvas.

30. **"More actions" on a clean skill reveals exactly one item (Delete)** — Low.
    Evidence: `WL/library_skills_canvas.py:1851-1866`.
    Probe: open a user skill ▸ Edit ▸ More actions.

31. **Overview shows raw trust status** ("Trust: quarantined modified", "Trust: trust uninitialized") instead of the friendly copy used elsewhere — Low/Med.
    Evidence: `WL/library_skill_work_pane.py:104-108` vs `WL/library_skills_canvas.py:94-103`.
    Probe: open a skill whose files changed since trust; compare Overview vs Trust tab.

32. **Built-in preview has no Escape/back and uses a `Switch`** the canvas family documents as render-unreliable — Med (likely-to-fail-live).
    Evidence: no gate matches view "preview" (`LS:25365-25368`, `LS:10452-10453`); Switch at `WL/library_skill_work_pane.py:169-175`; render caveat `WL/library_skills_canvas.py:12-22`.
    Probe: select a built-in at 100x30 and 235x52; press Escape; toggle Enabled with mouse and keyboard; check rendering.

33. **No in-canvas "New skill" button; empty copy points to the rail** ("use Create ▸ New skill in the rail"), which is hidden in single-pane narrow layouts — Low/Med.
    Evidence: `WL/library_skills_canvas.py:77-79`, list toolbar `:1264-1284` (only sort + Import); Prompts has an in-canvas New prompt (`WL/library_prompts_canvas.py:637-662`).
    Probe: empty skills profile at 60x24 and 100x30.

34. **`/` on Skills goes to the rail search, not the skills filter** (Prompts and Conversations go to their own filter) — Low.
    Evidence: `LM/screen_constants.py:457-460` lacks Skills.
    Probe: Skills list, press `/`.

35. **Two "Cancel" buttons during an in-flight import** (row Cancel disabled + structural Cancel enabled) — Low.
    Evidence: `WL/library_skills_canvas.py:1500-1525`.
    Probe: import a slow GitHub/zip URL; inspect buttons while "Inspecting/importing…".

36. **"re-review" on a first import; trust jargon (baseline, manifest, trust store, quarantined)** — Low.
    Evidence: `LM/library_skill_import_controller.py:772`; `WL/library_skills_canvas.py:94-103`, `:192-230`.
    Probe: import a new skill; read status and Trust tab.

37. **Disabled trust buttons and creation-mode tabs lack the "○" marker** (tooltips exist for trust buttons, none for tabs) — Low.
    Evidence: `WL/library_skills_canvas.py:2005-2035`; `WL/library_skill_work_pane.py:192-198`.
    Probe: New skill ▸ hover Overview/Trust/Files; trusted skill ▸ Trust ▸ inspect Unlock/Approve.

38. **User Guide drift**: says "scroll to the Trust panel" below the editor, and "You can invoke"; code has a separate Trust tab and "User can invoke" — Low (docs).
    Evidence: `Docs/User_Guide/library/skills.md:340`, `:392`, `:405`, `:259` vs `WL/library_skill_work_pane.py:40-44`, `WL/library_skills_canvas.py:495-526`.
    Probe: follow the guide's Common tasks literally.

### Artifacts

39. **Action-failure toast grammar**: "Could not console this report.", "Could not source this report." and "report" for Chatbooks — Low.
    Evidence: `LM/library_artifacts_controller.py:699-708`.
    Probe: Chatbooks ▸ select a record whose source conversation was deleted after load ▸ Open source / Use in Console (or simulate failure).

40. **"Sharing 1 Chatbooks"** pluralisation — Low.
    Evidence: `LM/library_artifacts_share_controller.py:165-167`, `:299`.
    Probe: share a single chatbook.

41. **"All artifacts" empty state talks only about reports** — Low.
    Evidence: `WL/library_artifacts_widgets.py:113-122` (only "chatbooks" has its own copy).
    Probe: profile with no reports and no chatbooks ▸ `#library-row-artifacts-all`.

42. **Pager pattern differs from every other Library list** ("First/Prev/Next/Last", no one-page suppression, no "○", no reason tooltip) — Low/Med.
    Evidence: `WL/library_artifacts_widgets.py:54-58`, `:127-135`; shared rule `L/library_pager_state.py:9-10`, `:59-60`.
    Probe: Reports with < 1 page of items.

43. **Sort control label is a bare value ("Newest"/"A–Z")** unlike sibling "sort: …" choosers; All/Kept is a ✓ toggle pair — Low.
    Evidence: `WL/library_artifacts_widgets.py:51`, `:142-144` vs `L/library_shell_state.py:296-302`.
    Probe: Reports toolbar.

44. **Rows show internal ids and storage labels** ("#123", "Registered", "Registry metadata", "copies") — Low.
    Evidence: `WL/library_artifacts_widgets.py:96-99`, `:114-116`; `Chatbooks/artifact_registry_snapshot.py:137`; `L/library_artifacts_catalog.py:391-397`.
    Probe: Chatbooks list and Details.

45. **"Open source" is ambiguous and actually resumes a conversation in Console** — Low/Med.
    Evidence: `WL/library_artifacts_widgets.py:207`, `LM/library_artifacts_controller.py:672-680`.
    Probe: Chatbooks ▸ saved response ▸ Open source.

46. **Share Stop has no confirmation and no receipt** — Low/Med.
    Evidence: `LM/library_artifacts_share_controller.py:305-314`.
    Probe: start a share ▸ `#library-artifacts-share-stop`.

47. **Initial count text "Loading reports…" in Chatbooks/All views** — Low.
    Evidence: `WL/library_artifacts_widgets.py:52` (compose) before first sync.
    Probe: open Chatbooks on a slow DB; watch the first frame.

### Cross-destination / Study / character

48. **Four different behaviours share the label "Use in Console"** (prompt → composer insert; chatbook → live-work card; rail → whole-library snapshot; Media → staged source) while Conversations calls its staging action "Use as source" and Study uses "Continue in Study" — Med (IA consistency).
    Evidence: `LS:26994-27056`; `LM/library_artifacts_controller.py:681-697`; `LS:35735-35778`; `WL/library_conversation_reader.py:296-304`.
    Probe: trigger each and compare what Console shows on arrival.

49. **All three Study buttons say "Continue in Study"; Escape from Study always returns to "Study decks"** — Low/Med.
    Evidence: `LM/screen_constants.py:337-349`; `UI/Screens/study_screen.py:1352-1356`; `LM/screen_constants.py:368-410`.
    Probe: Flashcards ▸ Continue in Study ▸ Esc: which rail row is selected?

50. **Blocked Study canvas keeps the same enabled primary button** (recovery says "open … globally without Library context") — Low.
    Evidence: `LS:14463-14470`, `WL/library_entry_canvases.py:546-562`.
    Probe: empty profile ▸ Study decks.

51. **"Source snapshot" / "carries forward" jargon on Study canvases and tooltip** — Low.
    Evidence: `LS:14446-14470`, `WL/library_entry_canvases.py:556-560`.
    Probe: read Study staging canvas.

52. **Character repair dialog: no Escape binding; "Retry or cancel" but the button is "Refresh"; forward-only "Next 20 characters"; jargon ("Historical identity", "authoritative", "local character <id>")** — Med.
    Evidence: `LM/library_character_repair_controller.py:275-338`, `:368`, `:384`, `:404-412`, `:444-448`.
    Probe: Console character context ▸ unavailable conversation ▸ Repair ▸ press Escape; page Next twice; read copy.

53. **"Back to Console" label is hard-coded** though the route's return target can be a Personas anchor — Low (latent).
    Evidence: `WL/library_character_return.py:20-22` vs `UI/Navigation/character_conversation_navigation.py:63-85` (personas targets exist); current senders all use Console (`UI/Console_Modules/character_context.py:1221-1278`).
    Probe: check whether any Personas path emits a Library character route.

54. **No return affordance for non-character inbound handoffs** (Console citation "open source", Personas "open in Library", Console prompt save) — Low/Med.
    Evidence: only `compose_character_return` renders a back control (`LM/library_unavailable_navigation.py:150-162`).
    Probe: Console citation modal ▸ open source in Library ▸ look for a way back other than the nav bar.

55. **First-timer landing does not mention Conversations, Prompts, Skills or Artifacts** (Get started = Import / Find / Use; quick actions = Import…, New note, Search) — Low/Med.
    Evidence: `WL/library_entry_canvases.py:262-428`.
    Probe: fresh profile ▸ Library landing.

56. **No keyboard "new" for Prompts/Skills; `n`/ctrl+n only create notes** — Low.
    Evidence: `LS:25313-25334`, `LS:8940-8955`.
    Probe: Prompts list press ctrl+n.

57. **Grip/pane naming inconsistent**: Prompts/Skills items grip "Prompts"/"Skills", Conversations/Artifacts "Items" — Low.
    Evidence: `LS:15276`, `LS:15323`, `LS:15403`, `WL/library_artifacts_reader_shell.py:38-39`.
    Probe: collapse Items in each at 120 cols; read grip tooltip.
