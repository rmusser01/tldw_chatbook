# J4: Library power-user journey ("Morgan")

- Code under test: `.worktrees/notes-library-ux-review` @ origin/dev `2d34cbf80d`, Textual 8.2.8
- Harness: nlrev, socket `nl-j4-1`, GOLDEN profile, 235x52 primary, 120x36 for the resize checks. Run 2 was a `REUSE=1 REGEN=1` relaunch that added `[analysis_defaults]` (see caveats).
- Evidence: `evidence/j4-poweruser-library/NN-<what>-<cols>x<rows>.{txt,ansi}`. PNGs exist for 09, 25, 43, 60, 67 and 73. PNGs are approximate; the .txt and .ansi files are authoritative.
- Read before driving: PRODUCT.md, DESIGN.md (skim), maps `map-media-ingest-search.md`, `map-conv-prompts-skills-artifacts.md`, `map-shell-nav.md`, and `Docs/User_Guide/library.md`.
- Every cause below cites `file:line` under `tldw_chatbook/`. A cause I could not trace is labelled **hypothesis**.

## Persona and goals

Morgan is a solo builder and operator who works in the terminal and drives everything from the keyboard. Library is Morgan's source and context hub for agent work in Console. Morgan triages a lot of material and expects to see authority, cost and consequences before acting. Goals for this session:

1. Triage media quickly: filter, sort, select in bulk, trash and restore, then export a bundle that can be trusted.
2. Read sources, find text inside them, and keep notes and analysis next to them.
3. Ground a question in chosen sources and hand the evidence to Console at a known model and cost.
4. Move between Library and Console without losing place.

## Step log

Keys are the keystrokes actually sent. "Tab×N" is the number of Tab presses it really took. Friction is rated none, minor, major or blocker.

| # | Intent | Keys / clicks | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Open Library | Ctrl+P "Switch to Library" ↓ ⏎ (Ctrl+3 cannot be sent through tmux) | Landing | Landing with counts, "From your Library" and Quick actions | none | 01 |
| 2 | Get to Media | F6, Tab×3, ⏎ | Media list | F6 lands in the rail search box. 3 Tabs reach the "Media" row | minor | 02, 03 |
| 3 | Filter by keyword | `/` "transcript" | Live filter | Filters as you type: "Media (3)". The first hit loads into the Reader | none | 04 |
| 4 | Filter by type | Tab×2 ⏎ ↓↓ ⏎ | audio only | The "type:" chooser works. The scope line cuts off as `filter "transcr…` and hides the type and sort | minor | 05, 06 |
| 5 | Sort | Shift+Tab×7 ⏎ ↓↓ ⏎ | Title A-Z | Works. Two of the focus stops on the way are invisible | minor | 07 |
| 6 | Clear the scope | Shift+Tab×9 ⏎ (scope-line "Clear") | Filter and type cleared | Also reset **sort** from "Title A-Z" to "Newest" | minor | — |
| 7 | Select across pages | `s` Space ↓ Space ↓ Space, Tab×18 to "Next", ⏎ | Selection kept on page 2 | Toast "Selection cleared." Select mode exits | **major** | 08, 09 |
| 8 | Bulk tag | — | A keyword action | None exists. Select mode only offers Export / Review / Analyze / Delete | major | 08 |
| 9 | Trash 3 | `s`, Space×3, Shift+Tab×8 to "Delete", ⏎, Shift+Tab×5 to "Delete" in the confirm, ⏎ | Confirm, then receipt | "Delete 3 selected items? You can undo right away, or restore later from Trash." Receipt "✓ deleted · 3 items · in Trash" with focus on Undo | none | 10, 11 |
| 10 | Restore 1 | Shift+Tab×3 "Trash" ⏎, ↓ (does nothing), Tab ⏎, `r` | Restore | ↓ does not move rows in Trash; Tab does. "Restored 'Server logs excerpt 2026-09-27'." Escape returns focus to the Trash button | minor | 12, 13 |
| 11 | Export the selection | `s`, select 2, Tab to "Export" ⏎, then 29 Tabs through the whole rail to "Choose destination…" ⏎ ⏎, Tab ⏎ | Bundle written | Bundle written, "Last export: …/Library export 2026-10-02.zip · just now" | **major** (keyboard) | 14-16 |
| 12 | Inspect the bundle | unzip | Text, metadata and keywords | 2× .txt + metadata JSON. **Keywords are missing**: `"tags": []` and `media_keywords: null`, while the DB has learning,study and reference,markdown. Manifest says `"quality": "original"` | **major** | (on disk) |
| 13 | Find in content | Ctrl+F "evidence" | Match count | No count until ⏎. After a new query "caution", the old "Match 1 of 8" stays up (6 is correct) until ⏎ | minor | 18-20 |
| 14 | Analysis tab | Tab path ⏎ | Generate | "○ Generate · No analysis provider is configured · Set one in Settings ▸ Providers & Models." This shows even with `[analysis_defaults] provider="OpenAI"` set | **blocker** | 21 |
| 15 | Add analysis by hand | ⏎ "Add analysis", type | Text goes into the editor | Focus stays on the Items row. Typed letters fire list keys: `i` opened Import, and the text was lost | **major** | 23-25 |
| 16 | Type into the editor, then Escape | 18 Tabs to the TextArea, type, Esc | Leave the field and keep the text | Text discarded with no prompt; "No analysis yet." | **major** | 26, 27 |
| 17 | Save analysis | 9 Tabs, type, Tab×2, Shift+Tab, ⏎ | Saved | Saved. The row gains "analysed" | minor | 28 |
| 18 | Open original | More ⏎, Tab×2 ⏎ | Opens in browser | The URL went to the browser stub. No in-app feedback at all | minor | 29 |
| 19 | Edit metadata, add keyword | Tab ⏎ … (Tab order leaves the form, so the Keywords field took a click) | Saved | "Keywords: learning, study, srs" | minor | 30, 31 |
| 20 | Search | F6, type "retrieval practice" ⏎ | Results | 1 result. Media cards show no snippet ("Matched media · pdf") | minor | 32, 33 |
| 21 | Get into the canvas | F6 (inert), Esc, Tab×6 | Focus the canvas | F6 does nothing on Search/RAG. Esc moves focus into the canvas | major | — |
| 22 | Scope sources | ⏎ on 3 toggles | Media only | Re-runs at once with "Scope: Media (Notes, Conversations, Prompts off)" | none | — |
| 23 | Select 2 evidence | ⏎ card 2, ⏎ card 1 | Both selected | Selecting card 1 deselects card 2 (single-select only) | major | 34 |
| 24 | RAG answer | Shift+Tab×7 mode ⏎, edit query ⏎ | Answer from the configured model | Answer arrived, but the provenance line reads "openai · **gpt-5.6-terra** · $0.0006". Console's model is gpt-4.1-mini. The pre-run notice said only "To openai: question + evidence" | major | 35 |
| 25 | Replay history | Recent searches ▸ click "retrieval practice" | Re-run as a Search | Re-ran as a **paid RAG Answer** (mock request count went from 11 to 12) | major | 36, 37 |
| 26 | Stage evidence | `u` with nothing selected, then Select evidence, then `u` | Staged in Console | First press: toast "Run a query and select usable evidence…". Then staged: "Staged for next send · 1 source" | none | 38, 39 |
| 27 | Round trip | Ctrl+P → Library | State kept | Query, scope, results and the selected card were all kept | none | 40 |
| 28 | Find conversation | `/` "kubernetes" ⏎ | Reader follows the filter | "1 match", but the Reader still shows the **Thesis** conversation | major | 42 |
| 29 | Resume | Esc Esc `c` | Resume the visible match | Resumed "Thesis: chapter 2 structure", which the filter hides | **major** | 43 |
| 30 | Use as source | Tab into row ⏎, F6, Shift+Tab×4, Tab ⏎ | Staged | Staged into the current Console tab (Thesis). The composer was pre-filled. Back in Library: "✓ linked · Default … Undo link" | minor | 45 |
| 31 | Archive | Tab ⏎, Tab ⏎ | Confirm | "Archive 1 conversation(s)?" does not name the conversation. Receipt "Archived 1 conversation(s)." with Undo / View archived; the receipt never clears | minor | 46, 47 |
| 32 | Export conversation | View archived → Select → Select all → Export selected → destination | Bundle | "Overwrites Library export 2026-10-02.zip" (the same-day default name). The bundle drops the conversation keyword "homelab" | major | 48, 49 |
| 33 | Find prompt | rail Tab×6 ⏎, type "summar" | Filter | Filters as you type, although the placeholder says "(Enter)" | minor | 50 |
| 34 | Edit prompt | Tab, type, Ctrl+S | Save | Ctrl+S does nothing; "Use in Console" disappears while the prompt is dirty. Saved through Tab×6 to "Save changes" | minor | 52, 53 |
| 35 | Use in Console | Shift+Tab×5 ⏎, then the dialog | Prompt with its `{{notes}}` variable | "Prompt variables" dialog with no variable fields. The composer received "…Action Items:\n{notes}". The Instructions (System) text was dropped because the System-lane checkbox defaults to Off | major | 54-57 |
| 36 | Built-in skill: enable/disable | Tab to the switch, Space | Visible state | The switch box is empty in both states; the label still reads "Enabled" after turning it off. Only the list row says "· Disabled" | major | 59, 60 |
| 37 | Customize built-in | ⏎ Customize | Opens the copy | The work pane resets. The built-in row disappears behind "⚠ character-creator · needs review · overrides built-in" | minor | 61 |
| 38 | Review trust | Trust tab | Trust state | "Trust: not initialized" here; Overview says "Trust: trust uninitialized". The baseline consequence is stated clearly. Not set up (null keyring) | minor | 62 |
| 39 | Saved search | click "Unread favourites" | Filtered | 2 captures. The list header still says "Collections". The rail count changes to "Collections (2)" | minor | 63, 64 |
| 40 | Quick capture | click | Form | Placeholder-only fields and an unlabelled TextArea. Cancelled without saving (it would have needed network access) | minor | 65 |
| 41 | Reports | click "Reports", then "Try report demo" | Demo | Empty-state copy points to buttons at the bottom-right of the other pane. The demo created a live watchlist with 3 RSS feeds, made a paid call, and scheduled a daily run: "It refreshes daily from now on." | **major** | 66, 67 |
| 42 | Media round trip | click Media, ⏎, Read tab, `c` | Console with the item | The row says "loaded" while the Reader is empty. Then `c` showed the toast "Copy or link this media into workspace workspace-default before using it in Console." | major | 68, 71 |
| 43 | Real round trip | Ctrl+P Console → Ctrl+P Library | State kept | Filter, item, tab and the More strip were all kept | none | 72 |
| 44 | Resize 235→120 | resize | Nothing lost | Nav collapses to a grip; Reader and select mode survive; the Nav grip is reachable | none | 73, 74 |
| 45 | Search at 120x36 | resize | Evidence visible | Evidence is below the fold. Each Sources toggle takes 3 rows | minor | 76 |
| 46 | Resize 120→235 | resize | Restored | Nav, filter and selection restored | none | 75 |

## Task outcomes

Counts are what I actually sent, logged by the driver: keys + typed characters + mouse clicks. They include detours forced by focus problems, which is the point. Where the brief says "keyboard-only", every click is a place where the keyboard route failed or was unreasonable.

| Task | Outcome | Steps | Keystrokes (keys + typed chars) | Clicks | Note |
|---|---|---|---|---|---|
| 1 Media triage | partial | 12 | 194 + 27 = 221 | 0 | Multi-select across pages is impossible (cleared on page turn). There is no bulk tag. Export needed 29 Tabs through the rail. The bundle silently drops keywords. Trash and restore worked well. |
| 2 Reader | partial | 7 | 133 + 282 = 415 | 2 | Generate is blocked by a false "no provider" message. Analysis text was lost twice (focus trap, then Escape). Find needs Enter and shows a stale count. Edit metadata needed a click. |
| 3 Search/RAG | partial | 9 | 79 + 89 = 168 | 5 | No per-item scope. Single evidence only. The answer used a model other than the configured one. Replay silently made a paid call. Staging and round trip were good. |
| 4 Conversations | success (with a wrong-item hazard) | 6 | 44 + 27 = 71 | 5 | `c` resumed a conversation hidden by the filter. Archive and export worked; export drops keywords and overwrites the same-day bundle by default. |
| 5 Prompts | partial | 4 | 65 + 34 = 99 | 2 | No Ctrl+S. Variables not detected; System instructions dropped by default; the composer got the literal "{notes}". |
| 6 Skills | success | 4 | 27 + 17 = 44 | 2 | Switch state invisible. Customize hides the built-in. Trust not set up (harness caveat). |
| 7 Collections / Artifacts | partial | 4 | 0 | 6 | Mouse only. No way to create a saved search. "Try report demo" makes persistent, recurring, paid changes. |
| 8 Round trips | success | 3 | 18 + 85 = 103 | 4 | State was preserved everywhere I checked. Media `c` was refused because of the workspace link. |
| 9 Resize | success | 3 | 0 | 3 | Nothing was lost. At 120x36 the Search evidence falls below the fold. |

## Emotional journey

- **Start: confident.** The landing is calm and dense. Filter-as-you-type plus Reader auto-load feels fast, and the type and sort choosers are crisp.
- **First valley:** turning the page wipes my 3-item selection with only a toast. The bulk tag I came for does not exist.
- **Peak:** bulk delete. The confirm copy is honest, the receipt names the count, focus waits on Undo, and Restore says exactly what came back.
- **Deep valley:** Export. F6 goes nowhere and Tab walks every rail row (29 presses). Then the bundle on disk has no keywords. I stop trusting Export as a backup.
- **Worst moment:** Analysis says "No analysis provider is configured" while Console happily uses OpenAI. When I write the analysis by hand, my typing goes to the list: `i` jumps me to Import and the paragraph is gone. On the second try, Escape eats it.
- **Suspicion:** RAG Answer bills `gpt-5.6-terra` when I picked gpt-4.1-mini. One click on a history row spends money. "Try report demo" quietly signs me up for a daily paid job.
- **Alarm:** I filter Conversations to "kubernetes", press `c`, and land in my thesis chat.
- **Relief:** every round trip to Console and every resize comes back exactly where I left it.
- **End:** I trust Library as a store and as a place to come back to. I do not trust it yet as a keyboard cockpit, and I would avoid Analysis, the demo and history replay until they show their cost.

## Strengths

1. **Honest receipts with undo at the point of action.** "Delete 3 selected items? You can undo right away, or restore later from Trash." Then "✓ deleted · 3 items · in Trash" with focus on **Undo** (11). Restore says "Restored 'Server logs excerpt 2026-09-27'." (13). Export warns "Overwrites Library export 2026-10-02.zip" before running (49). Conversations shows "✓ linked · Default · this conversation can now be used in Console" with **Undo link**. The Info tab states "Use in Console sends: Stored text excerpt (truncated)". These match the PRODUCT principle of making consequences visible.
2. **State survives destination switches and resizes.** Library → Console → Library brought back the Search query, scope, results and the selected evidence card (40); the Media filter, open item, Read tab and More strip (72); and the Conversations filter and select mode. 235→120→235 restored the Nav pane, filter and selection (73-75). For a hub that feeds Console, this is what makes round trips cheap.
3. **Context-aware footer and immediate scope feedback on Search/RAG.** The footer names what Enter will do on the focused control ("enter toggle Conversations", "enter select evidence", "enter switch mode"). Toggling a source re-runs at once and restates the scope in plain words: "Scope: Media (Notes, Conversations, Prompts off)".

## Findings

Ordered by severity.

### F1 (P0): Library analysis always reports "No analysis provider is configured", even when one is configured
- **Evidence:** I relaunched with `[analysis_defaults] provider="OpenAI" model="gpt-4.1-mini"`; it is in `runs/nl-j4-1/config.toml` line 39. The Analysis tab still shows "○ Generate · No analysis provider is configured · Set one in Settings ▸ Providers & Models." (21, 28). Select mode's "○ Analyze" shows the same (08). An isolated probe (`seedrun.sh` + `config.load_settings()`) returned `"analysis_defaults" in settings → False`. The same resolver given the raw TOML returned `ready: True`.
- **Cause:** `config.py:2590-2620`. The settings dict that `load_settings()` returns passes through `chat_defaults`, `character_defaults` and others, but not `analysis_defaults`. `UI/Screens/library_screen.py:33075-33088` resolves against `self.app_instance.app_config` (= `load_settings()`, `app.py:1124`). `Library/ingest_analysis.py:158-176` therefore always takes the "no provider" branch. This affects every user, including the shipped default config (`config.py:5514`).
- **Who it hurts:** anyone who wants Generate, bulk Analyze or Analyze-after-import. The message is false and points to a Settings page that cannot fix it.
- **Fix:** add `"analysis_defaults": copy.deepcopy(toml_config_data.get("analysis_defaults", {}))` to the `load_settings()` return dict. Add a regression test asserting `resolve_ingest_analysis_provider(load_settings())` is ready when the TOML names a ready provider.

### F2 (P1): Export bundles silently drop keywords for media and conversations
- **Evidence:** the bundle for media 6 and 8 has `"tags": []` and `metadata.media_keywords: null`. The DB has learning,study and reference,markdown. The conversation bundle has `"tags": []` and `"metadata": {}`; the DB keyword is "homelab". The canvas only said "Bundle: 2 media items · text only · about 4 KB".
- **Cause:** `Chatbooks/chatbook_creator.py:1620` reads `media_item.get("media_keywords")`. `get_media_by_id` is `SELECT * FROM Media` (`DB/Client_Media_DB_v2.py:~7201`), which has no keywords column; keywords live in `MediaKeywords`. The importer reads the same key (`chatbook_importer.py:2772`), so a round trip loses tags.
- **Who it hurts:** Morgan, who uses export as a portable or backup copy of a triaged set; the tags were the triage.
- **Fix:** fetch keywords with `fetch_keywords_for_media_batch` and write them to `metadata.media_keywords` and `ContentItem.tags`. Do the same for conversation keywords. Add a "Keywords" line to the bundle consequence list, and a creator→importer round-trip test.

### F3 (P1): "Add analysis" leaves focus on the Items row, so typing fires single-letter commands
- **Evidence:** after ⏎ on "Add analysis", the ANSI capture shows the `█` focus bar on "Spaced repetition, explained" in Items, with the TextArea unfocused (25). Typing "Key takeaway: spacing…" opened **Import media** at the `i` and the paragraph was lost (23, 24). Typing "xxx" put nothing in the editor.
- **Cause:** `UI/Library_Modules/library_media_controller.py:4607-4617` sets `_library_media_editing_analysis = True` and re-syncs, but never focuses `#library-media-analysis-edit-text`. The pressed button unmounts and focus falls back to the list, where `t`/`c`/`i`/`s`/`l` are live.
- **Who it hurts:** keyboard users. Lost text, and stray trash arms, Console hand-offs or Import jumps.
- **Fix:** after the sync, `call_after_refresh(lambda: self.query_one("#library-media-analysis-edit-text", TextArea).focus())`. Apply the same to Edit metadata (focus Title).

### F4 (P1): Escape in the analysis editor discards typed text without asking
- **Evidence:** typed a sentence (26) and pressed Esc; the result was "No analysis yet." (27). The footer while typing says "esc close". Elsewhere in Library, Esc in a field means "leave field" and keeps the text.
- **Cause:** `UI/Screens/library_screen.py:30950-30990` (`action_library_media_viewer_back` sets `editing_analysis = False` at :30986) has no dirty check.
- **Fix:** if the TextArea differs from the saved analysis, make the first Esc only blur the field and show "Unsaved analysis — Save or Discard". That matches the Prompts editor's "esc save or discard first" veto. Change the footer chip to match.

### F5 (P1): Conversations filter leaves the Reader on a hidden conversation, and `c`/Resume act on it
- **Evidence:** the filter "kubernetes" showed "1 match for 'kubernetes'" while the Reader still said "Loaded Thesis: chapter 2 structure" (42). Esc, Esc, `c` resumed **Thesis: chapter 2 structure** in Console (43).
- **Cause:** `UI/Library_Modules/library_conversations_controller.py:1434-1446`. Filter submit only requests page 1; the loaded reader is never re-pointed or cleared. Media does the opposite (`library_screen.py:16635-16642`).
- **Fix:** when the loaded conversation is not in the new result set, either load the first match (like Media) or clear the Reader to "Select a conversation…". Gate `c`, Resume, Use as source and Archive on the loaded item being visible.

### F6 (P1): F6 never enters the Export or Search/RAG canvas; keyboard users must Tab through the whole rail
- **Evidence:** on Export, F6 only toggles focus in the rail search box (ANSI trace). From that box it took 25 Tabs to reach the Export name field and 29 to reach "Choose destination…". On Search/RAG, F6 does nothing. The footer still advertises "F6 next pane" on both.
- **Cause:** the default target list in `UI/Screens/library_screen.py:1420-1435` only names `library-hub-*` and `library-ingest-path` for `library-canvas`. `Widgets/workbench_focus.py:70-82` has no first-focusable fallback.
- **Fix:** add Export (`#library-export-name` / Choose destination) and Search (`#library-rag-query` or the mode toggle) to the canvas candidates. Also make the resolver fall back to the pane's first focusable descendant.

### F7 (P1): RAG Answer bills a different model than the configured one, and the pre-run notice hides it
- **Evidence:** the configured Console model is gpt-4.1-mini. The answer's provenance line reads "openai · gpt-5.6-terra · $0.0006 (90 tok)" (35). The mock log shows `model=gpt-5.6-terra`. The app log shows "dropping explicit temperature/top_p for reasoning model 'gpt-5.6-terra'". The pre-run line was only "To openai: question + evidence".
- **Cause:** `Library/library_rag_answer_service.py:163-189` returns `(default_api_endpoint, None)`, so the handler uses its legacy default (`config.py:2753`) instead of `[chat_defaults] model`.
- **Fix:** resolve the model from `[chat_defaults]` (or a `[rag] answer_model`). Show it before Run: "To OpenAI · gpt-4.1-mini: question + evidence". Offer a model chooser next to the mode toggle.

### F8 (P1): "Try report demo" creates persistent, recurring, paid, networked setup with no warning
- **Evidence:** one click produced a "Daily Brief" report and the toast "Your first daily report is ready -- see Artifacts → Reports. It refreshes daily from now on." (67). The report body shows live RSS items ("Ars Technica").
- **Cause:** `Subscriptions/daily_report_demo.py:1-60, 334+` seeds a watchlist with 3 live RSS sources (hnrss.org, BBC, Ars Technica), a preset, and an 86,400 s cadence (ADR-079: "the demo IS the product").
- **Also:** the empty state says "Open Watchlists or try the report demo.", but the buttons sit at the bottom-right of the Reader pane, about 38 rows and 150 columns away (66).
- **Fix:** rename the button to "Set up a daily brief…" and confirm first: "Creates a Daily Brief watchlist (3 RSS feeds), fetches now, calls OpenAI daily". Place the buttons directly under the empty-state sentence.

### F9 (P1): Media selection is page-local and is wiped by any page turn; no bulk keyword action
- **Evidence:** with 3 checked, "Next" gave the toast "Selection cleared." and exited select mode (09). The select toolbar offers only Clear / Export / Review / Analyze / Delete (08).
- **Cause:** `UI/Screens/library_screen.py:16855-16873` (`_clear_library_media_selection_for_scope_change`).
- **Fix:** keep the selection as a set of ids across pages and filters, with an "N selected (M on this page)" chip, the way Prompts already shows "N selected · M on this page". Add "Keywords…" (add/remove) to the select toolbar.

### F10 (P1): Built-in skill "Enabled" switch shows no state; its label still says "Enabled" when off
- **Evidence:** the ANSI capture of the switch shows an empty interior, `▊        ▎`, before and after toggling (59, 60). The only change is the background colour of cells 132-137. The label beside it stays " Enabled ". The list row gained "· Disabled", and config wrote `disabled_builtins = ["character-creator"]`.
- **Cause:** the `Switch` at `Widgets/Library/library_skill_work_pane.py:169-175`. `Widgets/Library/library_skills_canvas.py:12-22` already documents that Switch renders unreliably here.
- **Fix:** replace it with the house text toggle "Enabled: ✓ on ⇄ off" (the same pattern as "User can invoke: ✓ yes ⇄ no"), so the state is text, not colour.

### F11 (P2): Reader "Use in Console" refusal is toast-only, shows a raw id, and contradicts Search's `u`
- **Evidence:** `c` showed the toast "Copy or link this media into workspace workspace-default before using it in Console." (71). The footer still offered "c use in Console". The same kind of unlinked media item staged fine through Search's `u` (39).
- **Cause:** `UI/Screens/library_screen.py:35518-35561`. Conversations instead offers an inline "Link to workspace" control.
- **Fix:** show an inline reason under "Use in Console" with a "Link to workspace" button, as in the Conversations reader. Use the workspace display name ("Default"). Apply the same gate to Search `u`, or explain why evidence is exempt.

### F12 (P2): Reader Find needs Enter and keeps showing the previous query's count
- **Evidence:** with "caution" typed, the bar still read "Match 1 of 8", the count for "evidence" (19). Only after ⏎ did it update to "Match 1 of 6" with highlights (20). The placeholder "Search content…" does not mention Enter, while the list filter a few columns away updates as you type.
- **Cause:** `Widgets/Library/library_media_content.py:451-488` only updates on a submitted query.
- **Fix:** run Find on input changes (debounced, as the list filter does). Otherwise hide the status as soon as the input differs from the submitted query and show "Enter to search".

### F13 (P2): Returning to Media through the rail shows a row marked "loaded" next to an empty Reader
- **Evidence:** reproduced twice. The row read "▸ Spaced repetition, explained · article · updated 5d · loaded" while the Reader said "Select a media item to read it here." (24, 68).
- **Cause (hypothesis):** the rail-row reset (`library_screen.py:22223-22410`) closes the viewer but keeps the row's loaded flag.
- **Fix:** clear the "loaded" fact when the viewer resets. Or, since the filter is kept anyway, reopen the item.

### F14 (P2): Prompt "Use in Console" drops the Instructions by default and treats `{{notes}}` as literal text
- **Evidence:** the "Prompt variables" dialog showed no variable fields; the Tab cycle was checkbox → Cancel → Use original placeholders → Apply (55). The composer received "Summarize the following notes into Decisions and Action Items:\n{notes}", collapsed into a "Pasted text | 71 characters" chip (56, 57). The status bar stayed "System Prompt: off".
- **Cause:** `Prompt_Management/prompt_variables.py:376-383`. `{{` is an escape, so only `{name}` is a variable. The rest of the app uses `{{user}}` and `{{char}}`, and imported Obsidian templates use `{{date}}`. The System checkbox defaults to False (`Widgets/Console/prompt_variables_dialog.py:120-127`).
- **Fix:** in the editor, mark detected variables and say "Variables: {name}". Show a warning in the dialog when `{{name}}` is found ("Did you mean {name}?"). Call the System lane "Instructions", matching the editor, and default it on when the prompt has Instructions.

### F15 (P2): Recent searches replays under the current mode and silently spends money
- **Evidence:** "retrieval practice" was originally a free Search. One click on it while in RAG Answer mode made a paid call (mock requests 11→12) (37). The row gives no cue; the mode is visible only higher up the canvas.
- **Cause:** `Widgets/Library/library_search_rag_panel.py:1329-1338`.
- **Fix:** save the mode and scope with each history entry and replay under that saved mode. Mark RAG entries "RAG · paid".

### F16 (P2): Export canvas promises things the bundle does not do, and the default name collides
- **Evidence:** "quality: original — copies full media files into the zip" appears next to "Bundle: 2 media items · text only"; the zip has only .txt and .json (14, 16). The manifest says `"media_quality": "original"`. The default name "Library export 2026-10-02.zip" was reused for the second export of the day, and the second export overwrote the media bundle (49).
- **Cause:** `Library/library_export_state.py:63-97, 347`.
- **Fix:** hide or disable the quality chooser for text-only bundles, or change its caption to "text only — no files stored". Default the name to `Library export 2026-10-02 2035.zip`.

### F17 (P2): Customize hides the built-in skill and leaves the copy unusable by the agent, without saying so first
- **Evidence:** after Customize, the work pane reset to "Select a skill…". The built-in row was replaced by "⚠ character-creator · needs review · overrides built-in" (61), with Trust "not initialized" (62).
- **Cause:** `UI/Library_Modules/library_skills_builtin_controller.py:148-197`.
- **Fix:** open the new copy in Edit. Keep the built-in row visible as "Built-in · overridden by your copy" with a "Reset to built-in" action. Before copying, say "Your copy needs trust review before the agent can use it."

### F18 (P3): Restore and archive reset "updated" to now, which reorders Newest and the landing
- **Evidence:** after restore, "Server logs excerpt 2026-09-27 · updated just now" jumped to the top. The landing's "From your Library" then showed "Media · Server logs excerpt 2026-09-27". The archived conversation now reads "8m".
- **Fix:** keep `last_modified` for content edits. Store trash and archive state times separately, and sort Newest by content time.

### F19 (P3): Search/RAG at 120x36 pushes all evidence below the fold
- **Evidence:** the query block takes rows 7-17 and four Sources toggles take rows 20-31, each with two blank rows. Evidence starts after 8 wheel ticks (76).
- **Fix:** put the Sources toggles on one row ("☑ Notes 122 ☑ Media 21 ☐ Conversations 11 ☐ Prompts 10") and drop the blank rows around Run.

### F20 (P3): Scope-line "Clear" also resets the sort
- **Evidence:** "sort: Title A-Z" became "sort: Newest" after pressing the scope line's "Clear" (step 6).
- **Cause (hypothesis):** the handler calls `_clear_library_media_filter(clear_type=True)` (`library_media_controller.py:2328-2335`); the sort reset probably happens downstream in the request path.
- **Fix:** keep the sort, or relabel the button "Reset view" and state that it resets type, filter and sort.

## Improvement opportunities

1. **Selection basket:** a persistent cross-page, cross-filter set of selected items with a count chip. Bulk Keywords, Export, Review, Delete and "Search within selection" would all act on it. This would also cover "scope Search/RAG to specific items", which does not exist today.
2. **Multi-evidence staging:** ☑ several evidence cards and "Stage 3 in Console". Also "Stage answer + citations", so a paid RAG answer is not thrown away when the mode changes back to Search.
3. **Cost and authority line before every paid action:** "OpenAI · gpt-4.1-mini · ~$0.001 · question + 2 evidence" on RAG Answer, Generate analysis, bulk Analyze and history replay. The same line could carry the network sources on the report demo.
4. **Library jump keys:** for example `g m` Media, `g c` Conversations, `g s` Search, and an `F6` target for every canvas. That turns 6-29 Tabs into 2 keys.
5. **Export preview with a fidelity checklist:** "Includes: text ✓, keywords ✓, analysis ✓, highlights ✗, original files ✗". Show it before Run, and include it in the README.

## Nielsen heuristic scores

These cover the Library shell and its destinations: Media, Trash, Export, Search/RAG, Conversations, Prompts, Skills, Collections and Artifacts. I did not use Notes in this journey, so its rows are not assessed (-1).

| Surface | Heuristic | Score (0-4) | Key issue |
|---|---|---|---|
| Library | 1 Visibility of system status | 2 | Good receipts, but a false "No analysis provider is configured", a stale "Match 1 of 8", a "loaded" row beside an empty Reader, and a switch with no state |
| Library | 2 Match with the real world | 2 | "System lane", "trust uninitialized", "workspace-default", "Canonical ID", "conversation(s)", "analysed" |
| Library | 3 User control and freedom | 2 | Undo on delete, archive and link; but Esc discards analysis, a page turn discards selection, and the demo installs a recurring job |
| Library | 4 Consistency and standards | 1 | Filter live vs Enter, both labelled "(Enter)"; Find Enter-only; `c` means stage vs resume; `{x}` vs `{{x}}`; four focus styles |
| Library | 5 Error prevention | 1 | The focus trap lets letters fire commands; a filter-hidden item gets resumed; replay spends money; the default export name collides |
| Library | 6 Recognition rather than recall | 2 | Footer chips help, but invisible focus stops, hidden `i`, and no visible Tab order |
| Library | 7 Flexibility and efficiency | 1 | F6 dead on Export and Search; 25-29 Tabs through the rail; no cross-page select, bulk tag or multi-evidence |
| Library | 8 Aesthetic and minimalist design | 2 | Dense and calm at 235, but Search pushes evidence off-screen at 120, Collections rows are centered, and the Reports buttons are far from their copy |
| Library | 9 Recognize, diagnose, recover from errors | 1 | Wrong "no provider" diagnosis; toast-only workspace refusal with a raw id; "Run a query and select usable evidence" right after a query ran |
| Library | 10 Help and documentation | 3 | F1 and the footer are context-aware, and the User Guide is extensive (a few drifts, e.g. Media filter "until Enter") |
| Notes | 1-10 | -1 | Not used in this journey |

## Harness caveats

- Ctrl+digit cannot be sent through tmux, so I used Ctrl+P "Switch to Library" / "Switch to Console" (≈20 keystrokes instead of 1). Those keystrokes are included in the counts.
- The harness config has no `[analysis_defaults]` (the shipped template has one). Run 2 added it through `REUSE=1 REGEN=1 EXTRA_TOML` (mock + `[analysis_defaults] provider="OpenAI" model="gpt-4.1-mini"`). This is the setup that exposed F1.
- `BROWSER` was set to a logging stub for run 2, so "Open original" could not open a real browser. Its log recorded `https://example.net/learning/spaced-repetition`.
- Null keyring: skill trust was deliberately not set up.
- The shared mock LLM log interleaves requests from other agents. I only attributed requests whose timestamp matched my own action, and checked the model from the app log.
- "Try report demo" made real outbound RSS fetches (hnrss.org, BBC, Ars Technica) from the isolated run. The embedding model also contacted the Hugging Face Hub (app log: "sending unauthenticated requests to the HF Hub" at 20:15:43 and 20:29:57). These calls are harmless to the review, but the review was not offline.
- Two of my own keyboard slips are excluded as product defects: a click intended for the "Read" tab hit "Read later" (undone), and a mis-targeted Tab put text in a prompt's Name field (undone).
- Isolation check: the real profile is unchanged against the baseline. Socket `nl-j4-1` was killed and `pgrep -fl runs/nl-j4` is empty. The probe dir `runs/nl-j4-2` was removed.
