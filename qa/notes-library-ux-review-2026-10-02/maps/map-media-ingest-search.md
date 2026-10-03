# UX map — Library ▸ Media, Trash, Ingest, Search/RAG, Export, Collections, Review sets

- Worktree: `.worktrees/notes-library-ux-review` @ origin/dev `2d34cbf80d` (2026-10-02)
- Method: static code read only. Every claim cites `file:line` (paths relative to `tldw_chatbook/`).
  Docs under `Docs/User_Guide/library*` were read for intended behaviour only; where the docs and the
  code disagree, the code is quoted and the drift is listed under Suspected issues.
- Nothing here was run live. Every item in the last section is a **suspected** issue with a probe a
  live tester can run. No design recommendations are made here.

Abbreviations: `LS` = `UI/Screens/library_screen.py`; `MC` = `UI/Library_Modules/library_media_controller.py`;
`MCanvas` = `Widgets/Library/library_media_canvas.py`; `MV` = `Widgets/Library/library_media_viewer.py`;
`MT` = `Widgets/Library/library_media_trash_canvas.py`; `IC` = `Widgets/Library/library_ingest_canvas.py`;
`ICtl` = `UI/Library_Modules/library_ingest_controller.py`; `IS` = `Library/library_ingest_state.py`;
`RP` = `Widgets/Library/library_search_rag_panel.py`; `RCtl` = `UI/Library_Modules/library_rag_search_controller.py`;
`RS` = `Library/library_rag_state.py`; `EC` = `Widgets/Library/library_export_canvas.py`;
`ES` = `Library/library_export_state.py`; `CR` = `Widgets/Library/library_collections_capture_reader.py`;
`SS` = `Library/library_shell_state.py`; `Rail` = `Widgets/Library/library_rail.py`.

---

## 1. Inventory

### 1.1 Entry points into this area

| Entry | Label as rendered | Where | Goes to |
|---|---|---|---|
| Nav bar / global key | `⌃3 Library`, Ctrl+3 | app nav | Library landing (no row selected) |
| Rail primary button (top of rail, `variant="primary"`) | `Import…` | `LS:14864-14881` | Import media canvas |
| Rail section "Import / Export" | `Import…`, `Export` | `SS:723-755`, `SS:762-764` | Import canvas / Export canvas (Export disabled in server mode, tooltip "Export packages local content only." `SS:69`) |
| Rail section "Browse" | `Media (N) — your files`, `Collections (N) — saved captures` (short form `Captures`), `Search / RAG — find all` | `SS:494-607` | Media canvas / Collections canvas / Search/RAG canvas |
| Rail search box | placeholder `Search Library…`, clear button `x` (tooltip "Clear the Library search box") | `Rail:1100-1127`, `RCtl:821-832` | Submitting **always** switches to Search/RAG canvas and forces Search mode (`RCtl:776-804`) |
| Landing "Get started" (empty Library only) | `Import a file`, `Find it`, `Use it in Console` | `Widgets/Library/library_entry_canvases.py:380-428` | Import / Search / stage selected evidence |
| Landing "Quick actions" (non-empty Library) | `Import…`, `New note`, `Search` | `library_entry_canvases.py:343-373` | Import / Notes create / Search |
| Key `i` (any Library canvas, no text field focused) | not advertised outside landing/hub | `LS:8922-8939` | Import canvas |
| Starter-lifecycle rail (empty Library) | only `Import…`, `New note`, `Explore all tools` | `Rail:1092-1096` | — (no Media/Search rows until content exists or Explore pressed) |
| Legacy route ids `media`, `ingest`, `search` | — | `UI/Navigation/screen_registry.py:250-282` | all alias to `library` |

### 1.2 Surfaces in scope

| Surface | Widget / owner | Pane layout |
|---|---|---|
| Media list ("Items") | `LibraryMediaCanvas` (`MCanvas:216`) | Library rail · Items · Reader (three-pane adaptive shell; see `library_browse_reader_shell.py`) |
| Media Reader | `LibraryMediaViewer` (`MV:95`) | permanent right pane, modes Read / Analysis / Highlights / Info |
| Media Trash | `LibraryMediaTrashCanvas` (`MT`) | replaces Items pane; Reader keeps the live item |
| Review-set picker | `LibraryReviewSetPickerDialog` modal (`Widgets/Library/library_review_set_picker.py:40`) | modal |
| Import media | `LibraryIngestCanvas` + `LibraryIngestPreflightSummary` + `LibraryIngestQueuePanel` (`IC`) | single scrolling canvas with docked commit bar |
| Search / RAG | `LibrarySearchRagPanel` (`RP`) | single scrolling canvas |
| Export bundle | `LibraryExportCanvas` (`EC`) | single form |
| Collections (Quick Capture reading list) | `LibraryCollectionsScopeRows` (rail sub-rows), `LibraryCollectionsItemsPane`, `LibraryCollectionsWorkPane` (`CR`) | rail · items · work pane |
| Re-chunk (bulk, in Search/RAG) | `Widgets/Library/library_rechunk_run.py` + button in `RP:209-241` | inside Search/RAG "Sources" region |

### 1.3 Production-dead code paths (not user-reachable)

- Media list "preview" half and its `Open in viewer` button: production passes `show_preview=False`
  (`MC:2051`), so `has_preview` is always False and `#library-media-preview` never displays
  (`MCanvas:1816-1931`). The canvas-level meeting-speaker legend inside it is likewise unreachable;
  the reachable one is the Reader's (`MV:754-803`).

---

## 2. States

### 2.1 Media list (Items pane)

| State | Rendered text / controls | Source |
|---|---|---|
| Title | `Media` or `Media (N)` + `Sets` button (hidden in select mode) | `MCanvas:993-1040` |
| Scope line (applied scope) | e.g. `Media · 1 of 11 · filter “notes” · all types · sort: Newest`; `Clear` only while filter or type applied | `Library/library_media_state.py:1250-1271`, `MCanvas:1047-1068` |
| Fresh empty source | `No media in your Library yet. Import something to see it here.` + `Import media` | `library_media_state.py:20-22`, `MCanvas:1092-1124` |
| Empty by type | `No media of type 'pdf'.` + `Show all types` | `library_media_state.py:1241` |
| Empty by filter | `No media matched “q” in titles, content or keywords.` (no recovery button; `Clear filter` stays) | `library_media_state.py:1235-1239` |
| Empty by filter+type | `No media of type 'pdf' matched “q” in titles, content or keywords.` + `Show all types` | `library_media_state.py:1226-1233` |
| Single filter hit | status `1 result · Enter opens` (the hit is already auto-loaded into Reader) | `library_media_state.py:1210-1220`, `LS:16635-16642` |
| Loading row | row fact line ends `· loading`; open row ends `· loaded` | `MCanvas:188-213` |
| Row anatomy | marker cell (`▸` current, `☑/☐` select mode, `·/✓` review set) + title, then `type · updated 5m[ · analysed][ · keyword: term]` | `MCanvas:164-213`, `library_media_state.py:1349-1421` |
| Load failure | bordered callout `Couldn't load page 1 · <reason>` + `Retry` inside; `○ Export…`/`○ Review these` gated; `Trash` never gated | `MCanvas:935-977`, `MCanvas:1762-1776` |
| Stale page | rows still openable; Select/Export/sort/Select-all/Delete gated with stale reason; pager `Retry` | `MCanvas:797-811`, `MCanvas:1896-1903` |
| Write in flight | `Media change in progress.` gates everything incl. Retry | `MC:2034-2038` |
| Select mode | `N selected` + `Select all N shown`; row `Clear` `Export` `Review`; reason `Select items to enable.`; row `Analyze` (+ provider reason line); row `Delete` (danger); row `Done` | `MCanvas:1357-1536`, `SS:107` |
| Bulk delete armed | `Delete N selected items? You can undo right away, or restore later from Trash.` + `Delete` / `Cancel` | `MCanvas:1358-1423` |
| Bulk delete receipt | `✓ deleted · N items · in Trash` + `Undo` / `Dismiss`; failed undo `✗ undo failed · …` + `Retry undo` | `MCanvas:1550-1607` |
| Review-set dismissed receipt | `✓ dismissed · <name>` + `Undo` / `Dismiss` | `MCanvas:1613-1655` |
| Bulk analyze | choice `N of M already analyzed` + `Skip them` / `Overwrite`; running `Analyzing i of M[ · k failed]`; settled `✓/✗ analyzed · d of M[ · k failed]` + `Retry failed` / `Dismiss` | `MCanvas:1663-1751` |
| Pager | range always; `Page x of y` and `Previous`/`Next` only when >1 page; 20 rows/page | `MCanvas:1994-2064`, `library_media_state.py:43` |
| Selection lost on scope change | toast `Selection cleared.` (any filter keystroke, type pick, page turn exits select mode) | `LS:16855-16873` |

### 2.2 Media Reader

| State | Rendered text / controls | Source |
|---|---|---|
| Nothing open | `Select a media item to read it here.` | `MV:54`, `MV:295-304` |
| Nothing open, list failed | `Nothing loaded — the list could not be loaded.` | `MV:55` |
| Loading | `Loading media…` banner (prior item may stay visible) | `MV:305-318` |
| Detail error | error line + `Retry` | `MV:282-294` |
| Server item | `Server item · not in local Media list`; no mode row; no Read later / Edit / Move to trash | `MV:327-339`, `MV:501-504` |
| Beside Trash | `Showing a Media item · not in Trash` | `MV:327-332` |
| Review set active | banner e.g. `Reviewing: All media — 2 of 14 · 1 reviewed · ✓ reviewed` | `MV:340-349` |
| Header | `‹ Back` (only when layout has an exit), title, byline (Author, else URL), primary toolbar, mode row | `MV:350-376` |
| Primary toolbar | `Find`/`○ Find` (+ inline reason), `Read later`↔`Remove later` (local only), `Use in Console`, `More`↔`More ▴` | `MV:403-457` |
| More strip | `Edit metadata`, `Open original` (if any URL), `Open manager`, `Move to trash` (danger) | `MV:461-499` |
| Mode row | `Read`, `Analysis`, `Highlights`, `Info`; active one reads `Read (selected)` etc. | `MV:501-517` |
| Read | optional image preview + `Hide preview`/`Show preview` or status + `Retry preview`; `Rendered (selected) | Raw` strip only when content sniffs as Markdown; Find bar when opened; body | `MV:519-580`, `MV:616-704` |
| Find bar | input `Search content…` / `Search content (raw text)…`; `Match i of N` / `No matches`; `◀ Prev` `Next ▶` (`○ Prev` `○ Next` with no matches) | `Widgets/Library/library_media_content.py:33-34,416-550` |
| Analysis | text or `No analysis yet.`; `Edit analysis`/`Add analysis`; `Generate`/`Regenerate` (`○` + reason `No analysis provider is configured · Set one in Settings ▸ Providers & Models.`); while generating `Generating analysis…`; Rendered/Raw strip forced to Raw while a Find query is active with reason line | `MV:1093-1235`, `MV:74-77`, `Library/ingest_analysis.py:36-43` |
| Analysis edit | full TextArea + `Save` / `Cancel` | `MV:1237-1267` |
| Highlights | list of `● “quote”` cards with `Color: …`/`Note: …` and `✕ Delete`; empty `No highlights yet.`; collapsed `Add highlight` fold with placeholder-only inputs `Quote`, `Note (optional)`, `Color (optional)` + `Add highlight` | `MV:1341-1421` |
| Info | metadata lines (`Type:`, `Author:`, `URL:`, `Keywords:`, `Updated:`), optional `No Markdown formatting to render — showing the stored text`, provenance block `Backend:`, `Canonical ID:`, `Original source:`, `Stored representation:`, `Use in Console sends:` | `MV:589-614`, `Library/library_media_viewer_state.py:361-388` |
| Edit metadata (forces Info mode) | title becomes `Edit media details`; labelled inputs Title, Author, URL, Keywords (`Keywords (comma-separated)`) + `Save` / `Cancel` | `MC:3406-3418`, `MV:1054-1091` |
| Delete armed | `Delete this media? You can undo right away, or restore later from Trash.` + `Delete` / `Cancel` (no danger class) | `MV:378-399` |
| Speaker rename (finished meeting only) | `Rename speakers` + one `Rename…` input per speaker | `MV:754-803` |

### 2.3 Media Trash

| State | Rendered text / controls | Source |
|---|---|---|
| Heading | `‹ Media` + `Local Trash · N items` / `· N matching` / bare `Local Trash` | `MT:253-288` |
| Filters | input `Search Trash`; `Type: All` opener; scope label; while chooser open: a one-row plain `OptionList` | `MT:290-363` |
| Status (priority order) | `Finishing this action…` > error > `Loading Trash…` > notice > pager status > `Trash is empty. Items you delete from Media land here.`; overflow `▼ more status` | `MT:371-414`, `library_media_state.py:27-29` |
| Rows | `▸`/blank + title, then `type · trashed 2h` | `MT:433-453` |
| Pager | range; page + Previous/Retry/Next only when >1 page | `MT:455-543` |
| Actions | `Restore` and `Delete forever` (danger), each `○` + reason when unavailable | `MT:649-731`, `library_media_state.py:38-41` |
| Delete-forever armed | `This cannot be undone.` / title / type / `trashed 6h` (tooltip exact time) / `Cancel` `Delete permanently` | `MT:545-647` |

### 2.4 Import media

| State | Rendered text / controls | Source |
|---|---|---|
| Header + target | `Import media`; `Imports run on this machine.` / `Imports run on the server.`; switch `Import on the server`↔`Import on this machine` (only when a server exists) | `IC:1832-1862`, `IS:66-76` |
| Unavailable | `Import is unavailable in this runtime.` / `Media database is unavailable.` | `IS:78-79` |
| Path | label `File, folder or URL to import`; input `Path to a local file or a URL…`; `Browse…` (FileOpen, file or folder); `Clear` once typed | `IC:1875-1909`, `ICtl:1973-2011` |
| Intro (fresh) | `Import a file, a whole folder, or a URL. Supported: …` / `Imported items are searchable in your Library and can be used as context in chat.` | `IS:314-318` |
| Pre-check | `Checking…`; errors + `Choose a file…` or `Retry`; advisory notes; tooling summary + `Copy install command` + fold `What's missing` (`⚠ …` lines, per-extra copy buttons); type breakdown; estimate; unsupported/empty/duplicate lines (duplicates: `… appear to already be in your Library — they'll be matched, not re-imported.`) | `IC:342-493`, `IS:3060-3072` |
| Options | `Expand all`/`Collapse all` (only >1 group); one Collapsible per type group titled by changes; fields from capability schema (e.g. `PDF engine`, `Recursive summary (map-reduce)`, `Voice activity detection (VAD) filter`, `Chunking template` `None (manual settings)`/`Auto`/…); `Reset to defaults` per group | `IC:1720-1830`, `Library/ingest_capabilities.py:492-1128` |
| Metadata | labelled `Title (optional)` (`Defaults to source name`), `Author (optional)`, `Keywords (optional)` (`comma-separated`) | `IC:1963-2002` |
| Commit bar | forecast line; gate line (`Enter a file path or URL to start.` / confirm copy); analysis hint; external-prep status + `Cancel external preparation`; `Start import` | `IC:2019-2069`, `IS:85-130` |
| Two-press Start | e.g. `Import active. Start again to queue a duplicate.` | `IS:115-130` |
| Queue | heading; `Latest run: …`; `This queue: 1 parsing · 2 queued · 1 done`; empty `No import jobs yet.` / `Queue is empty.`; `Analyze N skipped`; per-row line + progress phase (`Transcribing audio`, `Saving to Library`, …) + actions `Open in Library`, `View on server`, `Show details`/`Hide details`, `Choose another GGUF…`, `Retry with faster-whisper`, `Retry`/`Retry Research source`, `Cancel`, `Force stop`, `Dismiss`; grouped runs `Show the N files`/`Hide…`, `Retry all`, `Dismiss all`; `Clear finished` (two-press `Press again to clear N finished…`) | `IC:556-954`, `IS:1873-1882,2181-2186,2979` |
| Detail lines | `Reason: …`, `Details: …`, `Underlying: …` | `IS:2273-2323` |
| Recent imports | collapsed fold; `name — done · 3m` + full path line; no actions | `IC:699-737` |
| Retry this batch | `Retry this batch` → `Press again to replace form` | `IC:2085-2092`, `IS:747-753` |
| Fold hint | `▼ more — scroll for the rest` | `IC:1240`, `IC:2098-2104` |
| Completion toast | `Import finished — 2 imported · 1 matched · 1 failed · 1 skipped` | `ICtl:1357-1384` |

### 2.5 Search / RAG

| State | Rendered text / controls | Source |
|---|---|---|
| Title | `Search / RAG` | `RP:307-312` |
| Query row | mode toggle `mode: ✓ Search ⇄ RAG Answer` (tooltip `Cycle Search/RAG mode. Next: RAG Answer — calls a paid provider.`); input `Ask or search Library sources`; quiet line; `Run`/`Searching…`/`Answering…` | `RP:317-339`, `RP:965-1013`, `RS:203-207` |
| Quiet gates | `Enter a question or search query.`; `Select at least one source.`; RAG ready: `To <provider>: question + evidence` | `RP:838-861`, `RS:307-327` |
| Hard blocks (callout) | `Install or enable Search/RAG dependencies.`; `Index selected Library sources before querying.`; RAG no provider `No analysis provider is configured · Set one in Settings ▸ Providers & Models.` + `Open Settings ▸ Providers`; unsafe query | `RS:1296-1345`, `RS:49-51`, `RP:938-961` |
| Retrieval notice | `Retrieval failed. Run again to retry.` / `Retrieval unavailable. …` / `Answer failed. Run again to retry.` | `RP:864-878` |
| Sources | heading, scope summary (`Scope: all local sources` / `Scope: Notes, Conversations (Media, Prompts off)`), toggles `☑ Notes (n)` `☑ Media (n)` `☑ Conversations (n)` `☑ Prompts (n)` (disabled at 0); legacy line `Chunked by an older engine: N items` + `Re-chunk older-engine items` + summary; no-sources `No Library sources yet — import media or create notes, then search.` + `Open Import media` | `RP:341-366`, `RS:71-76,199-202`, `RAG_Admin/local_rag_admin_service.py:380` |
| Answer (RAG only) | `Answer`; `Asking <provider>…`; text + `Citations resolve to staged evidence.`; caution callout when citations do not validate; abstain/no-evidence quiet text; `Answer failed: …` + `Run the query again to retry.`; provenance/cost line | `RP:580-725` |
| Evidence | heading `Evidence · top K per source` (Search) / `Evidence · top K` (RAG); `N results for 'q'.`; coverage notes (`No strong semantic matches — results below are weak.`, `Semantic search found nothing from: …`); cards `1. Title (score)` / badges (`media · 2 citations`) / snippet / `Citations: …` / `Open` + `Select evidence`↔`Selected evidence`; `Use in Console` under the ONE selected card | `RP:1016-1238`, `RS:1819,2042-2047` |
| Searching | searching line | `RP:1239-1245` |
| Empty result | `No evidence matched 'q'.` + `Try broader terms or turn on more sources.` | `RS:1132-1147` |
| Idle | `No evidence yet. Run Search/RAG to populate results.` + `Add or import sources, run a query, then select evidence for Console.` | `RP:1274-1284` |
| Recent searches | fold; `No recent searches.` / `Select an entry to run it again.` + one button per entry (tooltip `Re-runs under the current mode (…)`) + `Clear history` | `RP:391-397`, `RP:1287-1349` |
| Re-chunk run | `Re-chunking…`; `Re-chunk finished: N re-chunked, M skipped, K failed[; re-index …]`; `Re-chunk could not start: <exc>`; `Re-chunk failed: <exc>` | `Widgets/Library/library_rechunk_run.py:36-81`, `Library/library_rechunk_service.py:465-486` |

### 2.6 Export bundle

| State | Rendered text / controls | Source |
|---|---|---|
| Header | `Export bundle (.zip)` | `ES:38` |
| Scope line | `Counting…` then e.g. `Everything: …` / scoped label | `ES:328-331` |
| Empty | `Nothing to export in this scope.` | `ES:40` |
| Fields | placeholder-only `Export name`, `Description (optional)` | `EC:120-131` |
| Quality (media scopes only) | `quality: original` opener + choice strip `thumbnail`/`compressed`/`original`; helper `copies full media files into the zip` etc. | `EC:132-157`, `ES:63-79` |
| Destination | `Choose destination…` (FileSave, home dir, default `<name>.zip`); `No destination chosen` / path / refusal; `Overwrites <file>` | `EC:158-176`, `Library_Modules/library_export_controller.py:1318-1360` |
| Consequence | `Bundle: 2 media items · text only · about 4 KB before compression` or `· size known once it runs`; contents list ≤20 + `+ N more` | `ES:347-389` |
| Run | status line; `Cancel`; error line | `EC:177-192`, `EC:233-239` |
| Receipt | `✓ exported · 12 items · 348 KB · /path/out.zip`; empty run `✗ export produced no content · …` + button `Retry export` | `ES:43-48`, docs |
| Submit gate | `○ Export bundle (.zip)` + reason line (`Choose a destination before exporting.` etc.) | `EC:217-232` |

### 2.7 Collections

| State | Rendered text / controls | Source |
|---|---|---|
| Rail sub-rows | `▸ All Captures (N)`, `Reading`, `Archived`, `Favorites` (+ saved searches, `Searches a–b of n`, `Retry searches`, `Previous`, `More searches…`) | `CR:66-71`, `CR:241-300` |
| Items toolbar | `Collections`; `Quick Capture`; `Filters`; `Sort: saved desc` (cycles 7 raw values) | `CR:395-420`, `Library/collections_capture_models.py:14-22`, `UI/Library_Modules/library_collections_controller.py:1004-1021` |
| Quick Capture form | placeholder-only `https://example.com/article`, `Title (optional)`, `Tags, comma separated (optional)`, unlabeled TextArea; `Save capture`/`Saving…`/`Retry save…`/`Retry anyway`; `Cancel`/`Back`; unknown-outcome + `Refresh capture list`; retry warning about "canonical URL" | `CR:421-489` |
| Filters form | placeholder-only `Domain`, `Tags, comma separated`, `From date (YYYY-MM-DD)`, `To date (YYYY-MM-DD)`; `Apply filters`, `Clear` | `CR:490-523` |
| Filter box | `Filter captures` (applies on Enter) | `CR:524-528`, `library_collections_controller.py:633` |
| Empty | 3 variants (`No captures match these filters · clear them…`, `Nothing in this scope yet · choose All Captures…`, `No saved captures yet · press Quick Capture above…`) | `CR:307-341` |
| Errors | `Showing the last good page. Refresh failed; …` / `Captures could not be loaded: …` + `Retry`; `Loading captures…` | `CR:530-552` |
| Rows | `▸ [Selected · loading |Loaded in Reader ]Title` / `domain · date · Status · Favorite · Extraction failed` | `CR:345-369` |
| Work pane | identity `Local Collections · domain`; title; byline `… · N min read · Status · Authority`; `Mark Read`, `Favorite`, `Move to Archive`/`Archived`; `Open Original`, `More`; modes `✓ Read`/`Highlights`/`Notes`/`Info`; More: `Summarize`, `Listen`, `Save Offline Copy`, `Retry Extraction`, `Delete Permanently…`, legacy disclosure; archive receipt `Moved to Archive · was Reading.` + `Undo` | `CR:765-971` |
| Highlights / Notes / Info | unlabeled quote TextArea + `Highlight note (optional)` + `Add highlight`; `Delete highlight`; `Capture note` + `Save capture note`; `Linked Notes` + `Note ID` input + `Link Note` / `Unlink`; Info `Canonical URL`, `Submitted URL`, `Extraction`, `Authority`, `Backing Media <id>` | `CR:1027-1176` |

### 2.8 Review sets

| State | Rendered | Source |
|---|---|---|
| Picker | `Review sets`; empty `No saved review sets. Use “Review these” on the media list to start one.`; rows `✓ name — progress` + `Dismiss`; `Review read-later`; `Close` | `library_review_set_picker.py:73-125` |
| Errors | `All items in this set were removed.`; `Couldn't open review sets.`; `No media items to review.` | `LS:31703`, `LS:31836-31852` |

---

## 3. Controls & copy (notable labels, behaviours, inconsistencies)

### 3.1 Media list
- `Title/keyword…` filter applies **as you type** after a 0.12 s debounce (`MC:2293-2299`, `Library/library_media_reader_state.py:26`), not only on Enter. A non-empty applied filter re-points the Reader at the first hit (`LS:16635-16642`); a 0-hit keystroke tears the Reader down to its placeholder via a whole-screen recompose (`LS:16655-16690`).
- Three different "Clear" controls in one pane: scope-line `Clear` (filter **and** type, `MC:2328-2335`), toolbar `Clear filter` (filter only, `MC:2323-2326`), select-mode `Clear` (selection, `MCanvas:824-832`).
- `Export…` exports the **type** scope and ignores the text filter (`MC:3181-3191`); `Review these` pins the **filtered** list (`MCanvas:1225-1246`). Both sit together as "whole list" actions.
- Select-mode bulk labels are short (`Export`, `Review`, `Analyze`, `Delete`; `MCanvas:884-933`); the user guide calls them "Export selected"/"Delete selected".
- Type chooser opener `type: All types`, sort opener `sort: Newest` (lowercase, colon). Trash: `Type: All` (title case). Collections: `Sort: saved desc` (title case, raw enum, press-to-cycle).
- Row fact word `analysed` (British, `library_media_state.py:1389`) vs `analyze`/`analyzed` everywhere else (`MCanvas:1675,1688`, `SS:89-99`).
- No bulk keyword/tag action exists in select mode (only Export/Review/Analyze/Delete). Keywords are editable only one item at a time via `More ▸ Edit metadata` (`MV:1054-1091`).
- No read-later indicator or filter on list rows (no `read_later` reference in `library_media_browse_controller.py` / `library_media_state.py` row builder); the queue is only reachable via `Sets ▸ Review read-later`.

### 3.2 Media Reader
- Selected-mode idiom: `Read (selected)` text suffix (`MV:512-517`) vs Collections `✓ Read` (`CR:896-902`) vs choosers `✓ value`.
- `Read later` ↔ `Remove later` (`MV:430-434`).
- `Open manager` posts `NavigateToScreen("media")` (`MC:4712-4733`), and `media` aliases to `library` (`screen_registry.py:282`) — i.e. it navigates to the screen the user is already on.
- `Open original` appears whenever `viewer.original_source` (= stored URL) is non-empty (`MV:484-485`, `library_media_viewer_state.py:423`), but the handler only opens `http(s)://` (`MC:4196-4202`); local imports store `file://…` (`Local_Ingestion/local_file_ingestion.py:1034,1801`).
- Byline falls back to the `URL:` line when no author (`MV:357-374`); `URL:` is shown for any url not starting `local://` (`library_media_viewer_state.py:379-384`) — so a local file shows `file:///…` as its byline.
- Delete wording drift: action `Move to trash` → confirm `Delete this media?` with buttons `Delete`/`Cancel` (no danger class, `MV:387-399`). Bulk confirm `Delete` does carry the danger class (`MCanvas:1410-1416`). The guide calls the single delete "title-specific"; the copy is generic.
- Find counts matching **lines** (one per line, `library_media_viewer_state.py:428-440`). In the Read tab a Markdown item stays Rendered while searching and shows the count with no on-screen mark; the Analysis tab forces Raw while searching (`MV:1141-1163`).
- Highlights: no way to capture a quote from the reading body; inputs are placeholder-only (unlike the labelled edit form); blank quote is silently ignored (`MC:3475-3500`); `✕ Delete` deletes immediately with no confirm/undo (`MC:3565-3577`).
- Info provenance copy uses internal terms: `Backend:`, `Canonical ID:`, `Stored representation:` (`MV:604-614`).
- Analysis edit / metadata edit have no dirty tracking; Escape (and Cancel) discard unconditionally (`LS:30950-30990`).
- `Use in Console` is always enabled; refusals arrive only as toasts (`Open a media item before…`, workspace reason, `Console handoff is unavailable…`, `LS:35518-35561`). Conversations' sibling action uses an inline "link to workspace" flow (user guide).
- Generate/Analyze blocked reasons are text only (no Settings deep-link button), whereas the RAG no-provider block offers `Open Settings ▸ Providers` (`RP:953-961`).

### 3.3 Trash
- Trash type chooser is a plain one-row-high `OptionList` (`MT:311-320`), not the `LibraryChoiceOptionList` with the `█` cursor the Media list uses (`MCanvas:1319-1328`).
- Pressing a trash row only moves the `▸` marker; the Reader keeps the live item (`LS:23698-23716`, `MV:327-332`) — trashed content cannot be read before Restore/Delete forever.
- Single-item only: no multi-select, no bulk restore, no empty trash.
- Footer chip for `x` reads `delete` (`LS:4448`) while the button reads `Delete forever`.

### 3.4 Import
- Rail shows `Import…` twice (primary top button `LS:14874-14880` and the Import / Export row `SS:734`); landing adds a third.
- Several two-press confirms with no explicit Cancel control (Start, `Retry this batch`, `Clear finished`; `IS:115-130,747-753,2979`).
- Enter in the path field: first Enter checks, second starts; footer swaps `enter check this path` ↔ `enter start import` (`ICtl:981-1014`, `ICtl:2144-2162`).
- Escape returns to the Library **hub**, not to the canvas you came from (`ICtl:1547-1568`); Export's Escape returns to its origin canvas (`library_export_controller.py:1233-1260`).
- Jargon in options/rows: `pymupdf4llm`, `Recursive summary (map-reduce)`, `VAD`, `Parakeet precision`, `GGUF`, `faster-whisper`, `Force stop`, `Underlying:`, `matched`.
- `Recent imports` rows are inert text (no Open/Retry) (`IC:705-737`).
- `Keep original file` defaults off (`ingest_capabilities.py:1064-1066`).

### 3.5 Search / RAG
- Evidence selection is single-valued (`RCtl:1098-1122`); `Use in Console` stages one row with `action_label="Review evidence in Console"` (`RCtl:1239-1290`). The generated answer itself is not stageable.
- `u` is advertised in the footer whenever the Search canvas is selected (`LS:1243-1249`, gate `LS:25531-25537`), and refuses via toast with nothing selected (`RCtl:1241-1247`).
- Two synchronized query inputs: the rail `Search Library…` box and the canvas box share one state (`LS:34041-34055`). On the Search canvas `/` focuses the **rail** box (Search is not in `_LIBRARY_SLASH_CANVAS_FILTERS`, `UI/Library_Modules/screen_constants.py:457-460`), and Enter there forces `mode = "search"` (`RCtl:792-794`).
- RAG Answer's no-provider reason reuses the Media-analysis noun: `No analysis provider is configured …` (`RS:49-51`).
- Paid notice is terse `To <provider>: question + evidence` (`RS:327`); the guide quotes a longer sentence.
- Legacy chunk line `Chunked by an older engine: {n} items` has no singular (`local_rag_admin_service.py:380`); re-chunk tooltip and failure lines expose internals and raw exception text (`RP:226-230`, `library_rechunk_run.py:49,81`).
- Collections captures are not a Search/RAG source (`RS:71-76`; `RP:434-435` "Capture search remains inside Collections").
- `Recent searches` re-run under the current mode, not the original (`RP:1329-1338`).

### 3.6 Export
- The `quality:` chooser is inert: the writer stores text + metadata regardless (`ES:82-97` comment), and the helper captions still promise media-file behaviour (`ES:75-79`); the consequence line meanwhile says `text only` (`ES:347`).
- Name/Description inputs are placeholder-only (`EC:120-131`), unlike Import's labelled metadata fields (`IC:1963-2002`).
- Header and submit button share the same text `Export bundle (.zip)` (`ES:38,43`).
- Size estimate is KB-only (`ES:100-123`).

### 3.7 Collections vs Media (sibling inconsistencies)
- Row state prefix `Loaded in Reader ` / `Selected · loading ` **before** the title (`CR:351-369`); Media moved state to the fact line (`MCanvas:203-213`).
- `Open Original` (Title Case) vs Media `Open original`; `Move to Archive` vs `Move to trash`; `Delete highlight` vs `✕ Delete`; `Mark Read` is one-way (`library_collections_controller.py:1525-1530`); `Favorite` toggles but its label never changes (`CR:870-875`, controller `1532-1542`).
- Sort is a 7-value press-to-cycle with raw labels; Media uses a chooser list.
- Filter applies on Enter; Media applies as you type.
- Link Note requires typing an internal `Note ID` (`CR:1131-1143`).
- No UI to create a saved search (only list/page/retry; `UI/Library_Modules/library_collections_saved_search_controller.py:1-49`).

---

## 4. Key bindings

Screen bindings: `LS:1010-1222`. Gates: `check_action` `LS:25253-25652`. Footer sets: `LS:1243-1394`, `LS:4193-4470`, `LS:4662-4700`. Manual keys: `on_key` `LS:8749-8954`.

| Key | Action | Active when | Footer chip |
|---|---|---|---|
| `/` | Focus canvas filter (Media `#library-media-filter`, Prompts) else rail `Search Library…` | not in a text field; not in starter lifecycle (`LS:8859-8920`) | `/ focus search` |
| `i` | Open Import canvas | any Library canvas, no text field focused (`LS:8922-8939`) | only on landing (`i import content`) |
| `F6` / `shift+F6` | Next / previous pane | always / with a row selected | `F6 next pane` |
| `tab`/`shift+tab` | Focus cycle inside Library content | always (`LS:1020-1021`) | — |
| `↑`/`↓` | Move between list rows | a list row focused (`LS:8853-8858`) | — |
| `enter` | Media row: load immediately; select mode: toggle | row focused | — |
| `s` | Enter/leave Media select mode | Media list surface, rows present, no armed delete (`LS:25543-25562`) | `s select` / `s done selecting` |
| `space` | Toggle focused Media row (priority binding) | select mode, Media row or Media grip focused (`LS:25563-25598`) | `space toggle selection` |
| `]` / `[` | Next / previous item (page-local), or walk review set | plain Reader with neighbour (`LS:25601-25616`) | `] next item`, `[ prev item`; review: `] next (marks reviewed)`/`] finish review`, `[ prev in set` |
| `m` / `R` | Toggle reviewed / exit review | review set active | `m toggle reviewed`, `R exit review` (or `R finish review` when complete) |
| `ctrl+f` | Open/close Reader Find | plain Reader, settled, tab has text | `ctrl+f find` / `close find` |
| `l` | Read later toggle | local plain Reader | `l read later` |
| `c` | Use in Console (Media) / Resume (Conversations) | plain Reader | `c use in Console` |
| `t` | Arm Move to trash | local plain Reader, Find closed | `t trash` |
| `r` | Trash Restore / Ingest Retry this batch / Notes trash restore | per surface | `r restore` / `r retry` |
| `x` | Arm Delete forever | Trash row actions live | `x delete` |
| `esc` | 12 ordered bindings: skill/notes/media viewer back (one level), trash back, editors, ingest back (to hub; first press disarms Start consent), export back (to origin), handoff back, bulk-delete cancel, emergency return, narrow-stage return, blur text field, focus rail (`LS:1043-1129`) | first gate that passes | `esc close` / `esc focus Items` / `esc focus Library` / `esc back` / `esc back to <origin>` / `esc cancel delete` / `esc focus rail` |
| `enter` (Search) | On focused evidence card: select | card focused (`LS:1140-1142`) | chip label follows focus |
| `o` (Search) | Open focused evidence | card focused | `o open evidence` |
| `u` (Search) | Stage selected (or focused) evidence in Console | Search row selected (`LS:25531-25537`) | always `u use Library context in Console` |
| `ctrl+n` / `n` | New note | landing or Notes | landing `ctrl+n new note` |
| Type/sort chooser | Up/Down/Home/End/Enter, Escape cancels | chooser open | `enter choose…`, `esc close` |
| Review-set picker | Escape cancels | modal (`library_review_set_picker.py:45`) | — |

When a text field is focused, single printable chips collapse into `after esc: …` (`LS:4735-4800`).

---

## 5. Flows

### 5.1 First-timer: "I have a PDF/article — get it in, find it, ask about it"

1. Ctrl+3 → landing. Empty Library = `Get started` with three step buttons; rail shows only `Import…`, `New note`, `Explore all tools` (`Rail:1092-1096`). `Find it`/`Use it in Console` are pressable but refuse with a toast and a hint line until content exists (`library_entry_canvases.py:388-428`, `LS:34088-34150`).
2. `Import a file` (or rail `Import…`, or `i`) → Import media. Type a path/URL or `Browse…` (file or folder picker) → `Checking…` → breakdown/estimate/warnings. Optionally open the type fold (e.g. `Analyze after import`, default off).
3. Press `Start import` (or Enter twice). Queue scrolls into view; row shows phase; toast `Import finished — 1 imported`.
4. Done row → `Open in Library` → Media Reader opens the item. Or rail `Media` → Items list → row.
5. Read in Reader; `Find`/Ctrl+F to search in the item.
6. "Ask about it": rail `Search / RAG` (or rail box Enter). Mode defaults to Search. To ask a question: toggle `mode:` to RAG Answer; blocks if embeddings extra missing, index empty, or no provider (`RS:1313-1345`). Run → `Answer` + evidence cards.
7. To continue in Console: `Select evidence` on one card → `Use in Console` under it (or `c` from the Reader to stage the whole item).

Lifecycle note: once any user content exists the landing becomes GRADUATED (`Library/library_rail_state.py:118-130`) and the `Get started` steps are replaced by Continue / Needs attention / From your Library / Quick actions (`library_entry_canvases.py:262-375`).

### 5.2 Power user: triage dozens of items by keyboard

1. Ctrl+3 → rail `Media`; `F6` to Items; `↑/↓` rows (settle delay), `Enter` loads now.
2. `/` → filter (applies as you type; Reader jumps to first hit); `type:` / `sort:` choosers via Tab+Enter.
3. In Reader: `]`/`[` walk the **current page only** (gate needs an adjacent mounted row, `MC:3203-3240`); `l` read later; `t` arm trash then Enter/`Delete`; `c` to Console; Ctrl+F find.
4. Sequential review: `Review these` (filtered list, ≤500) or select → `Review`; `]` marks and advances across pages; `m`, `R`; `Sets` to resume.
5. Bulk: `s` → `space` per row / `Select all N shown` (≤20, current page) → `Export` / `Review` / `Analyze` / `Delete`. Any filter keystroke, type change, or page turn exits select mode with `Selection cleared.`
6. Bulk tag: not available (per-item `More ▸ Edit metadata ▸ Keywords`).
7. Trash: `Trash` → rows → `r` restore / `x` arm delete forever; `‹ Media` / Escape back.
8. Export: `Export…` (type scope) or select → `Export` → form → `Choose destination…` → `Export bundle (.zip)`; Escape returns to Media.
9. Scoped RAG with evidence: rail `Search / RAG`; toggle sources (`☑ Notes` etc.); RAG Answer; Tab to cards; Enter select, `o` open, `u` stage (one evidence row at a time).
10. Console hand-off: Reader `c` (one item), or evidence `u` (one row). Multi-item staging from Library uses the rail Details "Use in Console" for the workspace set (out of this map's scope).

### 5.3 Collections / Quick Capture
1. Rail `Collections` → `Quick Capture` → URL (+ title, tags, unlabeled note) → `Save capture`.
2. Rail sub-rows `All Captures`/`Reading`/`Archived`/`Favorites`/saved searches scope the list; `Filters` form (Apply on button), `Filter captures` (Enter), `Sort:` cycle.
3. Work pane: `Mark Read`, `Favorite`, `Move to Archive` (+ `Undo` receipt); More ▸ `Summarize`, `Listen`, `Save Offline Copy`, `Retry Extraction`, `Delete Permanently…`.

---

## 6. Cross-links

| From | To | Mechanism | Source |
|---|---|---|---|
| Media empty state `Import media` | Import canvas | rail row switch | `LS:22894` |
| Ingest done row `Open in Library` | Media Reader with item | `_open_job_in_library` | `ICtl:2444-2459` |
| Ingest server row `View on server` | Reader with server detail | `_open_library_external_media_detail` | `ICtl:2461-2482` |
| Media `Export…` / select `Export` | Export canvas (Escape returns to Media) | `_open_library_export_canvas` | `MC:3181-3191`, `library_export_controller.py:1233` |
| Media Reader `Use in Console` / `c` | Console with staged item | `open_chat_with_handoff` | `LS:35518-35561` |
| Media Reader `Open manager` | `NavigateToScreen("media")` → Library | alias | `MC:4712-4733` |
| Media Reader `Open original` | system browser (http/https only) | `webbrowser.open` | `MC:4196-4202` |
| Search evidence `Open` / `o` | Media/Note/Conversation/Prompt surface | `_open_library_item_by_id` | `RCtl:1124-1152` |
| Search `Use in Console` / `u` | Console live-work (staged, "Review evidence in Console") | `open_console_for_live_work` | `RCtl:1239-1290` |
| Search no-provider `Open Settings ▸ Providers` | Settings ▸ Providers & Models | `NavigateToScreen("settings", …)` | `RP:248-264` |
| Search no-sources `Open Import media` | Import canvas | — | `RP:478-483` |
| Rail `Search Library…` Enter | Search canvas, mode forced to Search | — | `RCtl:776-804` |
| Landing `Needs attention` `Review` | Import queue | — | user guide; `library_entry_canvases.py:313-329` |
| Collections Info `Backing Media <id>` | text only (no link) | — | `CR:1165-1176` |
| Collections → Search/RAG | none (captures not a RAG source) | — | `RS:71-76` |

---

## 7. Suspected issues with live probes

Severity guesses: **P1** blocks or misleads a core job; **P2** friction/inconsistency on a common path;
**P3** copy/polish. All are unverified until probed live (Textual 8.2.8, scratch profile with
`TLDW_CONFIG_PATH` isolation; seed ≥25 media items incl. one local PDF, one Markdown doc, one URL import).

| # | Suspected issue | Evidence | Sev | Live probe |
|---|---|---|---|---|
| S1 | **Get started steps 2–3 can never be used.** `Find it`/`Use it in Console` only render in STARTER/UNKNOWN; the first import makes the Library GRADUATED and replaces them. | `library_entry_canvases.py:217-221,380-428`; `library_rail_state.py:118-130` | P1 | Empty profile → Ctrl+3 → `Import a file` → import a PDF → Escape to hub. Expect the steps gone (Quick actions instead). Check whether any route shows step 2/3 enabled. |
| S2 | **`Export…` ignores the active text filter** while `Review these` beside it honours it. | `MC:3181-3191`; `MCanvas:1225-1246` | P1 | Media → type `quokka` (3 hits of 25) → `Export…` → read `Bundle: N media items`; expect 25 not 3. |
| S3 | **Export `quality:` chooser is inert and its helper contradicts the bundle line.** | `ES:75-97,347` | P1 | Export from Media → set `quality: thumbnail` → read helper ("keeps a small preview…") vs `Bundle: … · text only`; export and unzip, compare with `original`. |
| S4 | **Search/RAG stages only one evidence row**; no multi-select, answer not stageable. | `RCtl:1098-1122,1239-1290`; `RP:1133-1140,1225-1237` | P1 | Run a query with ≥3 results → `Select evidence` on #1 then #2 → observe #1 deselects; press `Use in Console` → Console shows one evidence item. |
| S5 | **Select mode is destroyed by any scope change**, and selection is page-local (≤20). Filter box, `type:` and pager stay visible while selecting. | `LS:16855-16873`; `MCanvas:1285-1303`; `MC:2293-2299` | P1 | Media → `s` → check 3 rows → type one character in `Title/keyword…` → wait 0.2 s → expect select mode gone + toast `Selection cleared.`; repeat with `Next` page. |
| S6 | **Escape silently discards Analysis/metadata edits** (no dirty guard), unlike Notes/Prompts editors. | `LS:30950-30990`; `MC:4607-4630` | P1 | Reader → Analysis → `Edit analysis` → type a paragraph → Escape → expect edits gone with no prompt. Same for `More ▸ Edit metadata`. Also try clicking another Items row mid-edit. |
| S7 | **Filter-as-you-type replaces the open document**; a 0-hit keystroke clears the Reader (reading position lost). | `LS:16635-16690`; `MC:2293-2299` | P2 | Open item, scroll mid-text → type `zzz` in filter → Reader shows `Select a media item…`; clear → which item returns and at what scroll? |
| S8 | **`Open manager` is a dead/no-op control** (navigates to Library from Library). | `MC:4712-4733`; `screen_registry.py:282` | P2 | Reader → `More` → `Open manager`; observe whether anything changes (state reset? flicker?). |
| S9 | **`Open original` shown for local files but does nothing** (handler opens http/https only; local imports store `file://`). | `MV:484-485`; `MC:4196-4202`; `local_file_ingestion.py:1034` | P2 | Import a local PDF → open → `More` → `Open original` → expect no action, no message. |
| S10 | **Local file's absolute `file://` path becomes the Reader byline** when author is absent/"Unknown". | `MV:357-374`; `library_media_viewer_state.py:372-384` | P2 | Open the locally imported PDF; read the row under the title. Check Info `Original source:` too. |
| S11 | **Media `Use in Console` refuses only via toast**; Conversations offers inline link-to-workspace. | `LS:35518-35561` | P2 | With an active workspace that does not contain a media item: Reader `Use in Console` → toast only? Compare Conversations `Use as source`. |
| S12 | **Rail search Enter silently flips RAG Answer → Search**; `/` on Search canvas targets the rail box, not the canvas query. | `RCtl:792-794`; `screen_constants.py:457-460`; `LS:8859-8920` | P2 | Search canvas → set `mode: … ✓ RAG Answer` → run a query → press `/` (focus leaves the canvas box) → Enter → mode label shows `✓ Search`, answer cleared. |
| S13 | **`u` advertised but refuses with nothing selected.** | `LS:1243-1249,25520-25527`; `RCtl:1239-1247` | P3 | Search canvas, no selection, focus off inputs → press `u` → warning toast; footer still shows `u use Library context in Console`. |
| S14 | **Trash type chooser is a one-row plain OptionList** (no `█` cursor; options hidden). | `MT:290-321` | P2 | Media → `Trash` (with ≥3 trashed types) → `Type: All` → count visible options; arrow through. |
| S15 | **Cannot preview a trashed item** before Restore / Delete forever. | `LS:23698-23716`; `MV:327-332` | P2 | Trash → press a row → Reader still shows the live item labelled `Showing a Media item · not in Trash`. |
| S16 | **Trash footer says `x delete` for a permanent delete**; button says `Delete forever`. | `LS:4448`; `MT:697-721` | P3 | Trash with rows → read footer chips. |
| S17 | **Highlight UX gaps**: quote must be typed/pasted, blank quote ignored silently, `✕ Delete` immediate with no undo, placeholder-only fields. | `MV:1341-1421`; `MC:3475-3577` | P2 | Highlights → expand `Add highlight` → press with empty Quote (no feedback?) → add one → `✕ Delete` (gone, no undo). |
| S18 | **Find on a Rendered Markdown item shows `Match i of N` with nothing marked**; counts lines not occurrences; Analysis tab instead forces Raw. | `library_media_viewer_state.py:428-440`; `MV:1141-1163`; user guide | P2 | Open Markdown item (Rendered) → Ctrl+F → term occurring twice on one line + elsewhere → compare count and marks; repeat on Analysis tab. |
| S19 | **Single-item delete wording/styling drift**: `Move to trash` → `Delete this media?` → `Delete` (no danger ink); bulk `Delete` has danger ink; guide says "title-specific". | `MV:387-399,494-499`; `MCanvas:1410-1416` | P3 | Reader → `t` → read copy; compare colours with select-mode `Delete` confirm. |
| S20 | **Hidden `i` key navigates away from any canvas** (not advertised outside landing). | `LS:8922-8939`; `LS:1361-1373` | P3 | Media Reader with focus on content → press `i` → lands on Import. Escape → lands on hub, not Media (see S21). |
| S21 | **Import Escape returns to the hub, Export Escape returns to origin.** | `ICtl:1547-1568`; `library_export_controller.py:1233-1260` | P2 | From Media press `i` → Escape → hub. From Media `Export…` → Escape → Media. |
| S22 | **Get-started / landing `/` and footer honesty in STARTER**: rail has no search box; check footer does not advertise `/`. | `LS:8906-8910`; `LS:4776-4793` | P3 | Empty profile → Ctrl+3 → read footer; press `/`. |
| S23 | **RAG no-provider block says "No analysis provider is configured"** in a RAG-answer context. | `RS:49-51`; `ingest_analysis.py:36-43` | P3 | No provider configured → Search canvas → RAG Answer → type query → read callout. |
| S24 | **`Chunked by an older engine: 1 items`** (no singular) and raw exception text in re-chunk failures. | `local_rag_admin_service.py:380`; `library_rechunk_run.py:49,81` | P3 | Seed one pre-stamp chunked item → Search canvas → Sources region. |
| S25 | **Collections not searchable from Search/RAG or `/`**; on Collections `/` focuses the rail box whose search excludes captures. | `RS:71-76`; `screen_constants.py:457-460`; `LS:1352-1356` | P2 | Save a capture with a unique word → rail `Search Library…` the word → no capture evidence; on Collections press `/` → focus goes to rail box. |
| S26 | **Collections sibling inconsistencies**: row prefix `Loaded in Reader` before title, `Sort: saved desc` raw cycle, Enter-only filter, one-way `Mark Read`, static `Favorite` label, `✓ Read` mode idiom, `Note ID` linking, no saved-search creation. | `CR:345-369,414-420,524-528,864-902,1131-1143`; `library_collections_controller.py:1004-1021,1525-1542` | P2 | Collections → open a capture → press `Mark Read` twice, `Favorite` twice; press `Sort:` 7 times; look for any "Save search". |
| S27 | **Quick Capture / Filters / Export forms rely on placeholders** (identity lost once filled; capture note TextArea unlabeled). | `CR:421-523`; `EC:120-131` | P3 | Fill each field; check whether a field's purpose is still readable. |
| S28 | **Read-later state invisible in the list**; queue only via `Sets ▸ Review read-later`. | `MV:430-434`; `library_media_state.py` row builder | P2 | Mark 3 items `Read later` → back to Items → look for a marker or filter; then `Sets` → `Review read-later`. |
| S29 | **No bulk keyword/tag action** in select mode. | `MCanvas:884-933` | P2 | `s` → select 5 → look for tag/keyword action. |
| S30 | **`analysed` vs `analyzed`** spelling within the same pane. | `library_media_state.py:1389`; `MCanvas:1675,1688` | P3 | Analyze 2 items in bulk → compare row fact line and receipt. |
| S31 | **Header chrome redundancy in Items pane**: `Media (N)` + scope line + `type:`/`sort:` + 3 toolbar rows before first row. | `MCanvas:1024-1303` | P3 | 100x30 and 235x52: count rows above first item. |
| S32 | **Mode-row `(selected)` suffix may clip in a narrow Reader** (`Highlights (selected)` = 21 cells). | `MV:506-517` | P3 | 100x30, open item, select Highlights; check full mode row paints. |
| S33 | **`]`/`[` stop at the page edge** outside a review set; no auto page-advance. | `MC:3203-3240`; `LS:25601-25616` | P2 | Media with 25 items, open row 20 → `]` → nothing (chip disappears); need `Next` page. |
| S34 | **Generate/Analyze blocked reason has no Settings deep-link** while RAG has `Open Settings ▸ Providers`. | `MV:1225-1235`; `RP:953-961` | P3 | No provider → Reader Analysis tab vs Search RAG Answer. |
| S35 | **Export size estimate KB-only** ("about 1048576 KB"). | `ES:100-123` | P3 | Export Everything on a large profile; read Bundle line. |
| S36 | **Ingest jargon** in options, queue actions and details (`pymupdf4llm`, `GGUF`, `faster-whisper`, `Force stop`, `Underlying:`, toast `matched`). | `ingest_capabilities.py:492-1128`; `IC:886-944`; `IS:2273-2323`; `ICtl:1357-1384` | P3 | Import an audio file with no STT configured and a duplicate PDF; read queue rows, Show details, toast. |
| S37 | **Two-press confirms with no visible cancel** (Start, Retry this batch, Clear finished). | `IS:115-130,747-753,2979` | P3 | Arm each; check what disarms it (Escape only disarms Start per `ICtl:1558-1565`). |
| S38 | **Recent imports rows are inert** (cannot reopen or retry from history). | `IC:699-737` | P3 | After several imports + `Clear finished`, expand `Recent imports`; try to open an item. |
| S39 | **User Guide drift**: guide says filter "keeps a draft until Enter" (code debounces), bulk labels "Export selected/Delete selected" (code "Export/Delete"), paid notice wording, single delete "title-specific". | `MC:2293-2299`; `MCanvas:884-933`; `RS:327`; `MV:387-391` vs `Docs/User_Guide/library/media-and-conversations.md:194-271,369,563` and `search-and-rag.md:80-85` | P3 | Compare live strings with the guide sentences cited. |
