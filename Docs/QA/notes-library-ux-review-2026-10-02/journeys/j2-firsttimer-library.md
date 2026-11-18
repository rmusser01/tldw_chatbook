# J2: Jordan, a first-time user in Library (empty profile)

- Code under test: worktree `notes-library-ux-review` @ origin/dev `2d34cbf80d`, Textual 8.2.8
- Harness: nlrev `launch.sh`, sockets `nl-j2-1` (main run, relaunched 3x with `REUSE=1`) and `nl-j2-2` (clean repro of the freeze)
- Sizes: 160x45 for the main pass, 100x30 for the key moments
- Provider: first launch had NO provider (`EXTRA_TOML=/dev/null`). It was switched on with `REUSE=1 REGEN=1 EXTRA_TOML=mock_llm.toml`, which points OpenAI / gpt-4.1-mini at the shared mock on :18777.
- Evidence: `../evidence/j2-firsttimer-library/NN-<what>-<cols>x<rows>.{txt,ansi}`, plus PNGs for 14, 15, 25, 60 and 76.
- Blind pass = Phase 1. Findings first seen only in Phase 2 (after reading the maps and the User Guide) are marked **[P2-probe]**. Map IDs (S#, 7.#) are cited when a live probe confirmed a suspected issue from a map.

## Persona and goals

Jordan is new to Chatbook and reads every label literally. They have a folder of mixed documents
(pdf, md, html, docx, txt) and want to:

- get the documents in;
- read one and find a phrase in it;
- search across them;
- ask a question about them;
- carry a document into Console;
- keep something for later;
- export a bundle.

Jordan does not know what "RAG", "workspace", "staged evidence" or "skill" mean, and does not read
the docs before trying.

## Step log

| # | Intent | Keys / clicks | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Find Library from boot | Boot lands on Console "Get started"; click `⌃3 Library` | A page that says what Library is | "Get started · Add something useful, then use it in Console or Study." Three step buttons: `Import a file` `Find it` `Use it in Console`. The rail has only Import… / New note / Explore all tools. | minor (does not say what Library *is*; "Study" is not a visible destination) | 01, 02 |
| 2 | Ask for help | F1 | An explanation | "Library Shortcuts — Landing", with 3 shortcuts (i, ctrl+n, F6) inside a ~30-row empty box | minor | 03 |
| 3 | See everything | `Explore all tools` | Full menu | Full rail: Browse / Artifacts / Create / Study / Import-Export, with counts and short suffixes ("— your files", "— reuse", "— AI add-ons"). Jordan now gets it: "my files, chats, notes and reusable bits, plus search." | none | 04 |
| 4 | Import the folder | `Import…` → `Browse…` → click `📁 fixtures` → Enter → `Select folder` | Picker, then a summary | "1 PDF document, 3 plain text files, 1 Word/Office document · 5 files · 46.0 KB · 5 will import" | none | 05, 06, 07 |
| 5 | Start and watch | `Start import` | Progress | Per-file rows went `● parsing · Preparing import` → `● writing · Saving to Library` → `✓ done · 3s` with `Open in Library`. The rail count went Media (2) → (5) live. | none | 08, 09 |
| 6 | Check it finished | scroll | Done state | All 5 done, but the screen still shows `Retry this batch` and the footer `r retry` | minor (did something fail?) | 10 |
| 7 | Open the PDF | `Open in Library` on the PDF row | The PDF in a reader | The PDF opens and the row reads "pdf · … · loaded", **but the list cursor `█▸` sits on team-handbook** | minor (two "current" markers) | 11 |
| 8 | Keep it for later | click `Read later` | The PDF is saved | **The reader swaps to team-handbook, now showing "Remove later". The DB shows media 5 (team-handbook) saved; the PDF is not saved.** Same result 3/3, including with focus in the reader. | **major** | 12, 13, 14 |
| 9 | Find a phrase | click PDF row → `Find` → type `7-day retention` → Enter | Matches highlighted | "Match 1 of 2". The phrase occurs 3 times, and **nothing is highlighted** in the default Rendered view. Raw view highlights only the first occurrence on each line. | major | 15, 16 |
| 10 | Search the Library | rail `Search Library…` → `spaced repetition` → Enter | Results | "1 result for 'spaced repetition'", card "web-page-spaced-repetition · Matched media · html". The card has no snippet. | minor | 17, 18 |
| 11 | Ask a question, no provider | `RAG Answer` | Clear block and fix | Callout "No analysis provider is configured · Set one in Settings ▸ Providers & Models." with `Open Settings ▸ Providers`. Run is dimmed (#737373). | none (wording "analysis" vs Console's "provider") | 19 |
| 12 | Ask again via the rail box | type question in rail box → Enter | Answer or block | Mode silently flipped to `✓ Search`, then "No evidence matched 'How much did retrieval practice improve retention?'. Try broader terms." The PDF literally contains the answer. | major | 20, 21 |
| 13 | Follow the fix | `Open Settings ▸ Providers` | Settings | Lands on Settings ▸ Providers & Models. `⌃3` returns to Library with the question kept. | none | 22 |
| 14 | Use the PDF in Console | reader `Use in Console` | Console with the doc | Toast "Copy or link this media into workspace workspace-default before using it in Console." **No control on screen does that.** Same from a direct row open, and from Details ▸ `Use in Console`. | **blocker** on this path | 24, 25, 64, 65 |
| 15 | Find another way | `More` | A link or copy action | Edit metadata / Open original / Open manager / Move to trash. Nothing about workspaces. | major | 26 |
| 16 | Use via Search | Search `retrieval practice` → `Select evidence` → `Use in Console` | Console | Worked. Console card: "Library Search/RAG evidence staged — finish provider setup to use it." | minor (needed 4 extra steps; focus jumped to `Media` toggle) | 27, 28, 29 |
| 17 | Come back | `⌃3 Library` | Same place | Query, results, selection and scroll all preserved | none | 30 |
| 18 | Keep something | `Collections (0)` → `Quick Capture` | Save an item | Web-URL capture only. The 4-row text box has no label. A file:// URL gets "Enter a valid http or https URL before saving." The earlier "Read later" items do not appear under Collections ▸ Reading. | major (3 different "save for later" concepts) | 31, 32, 33 |
| 19 | Find my Read later list | Media → `Sets` | Saved list | Overlay "Review sets · No saved review sets…" with `Review read-later`. The overlay spills over the reader text. | minor | 34 |
| 20 | Export a bundle | rail `Export` → `Choose destination…` → `Save` → `Export bundle (.zip)` | A .zip | Written to `~/Library export 2026-10-02.zip`. The copy says "quality: original · copies full media files into the zip", but the zip has only `content/media/media_1..5.txt` + JSON + README (no PDF/DOCX). "size known once it runs" still shows after the run. | minor | 35–38 |
| 21 | Learn the other rows | Conversations / Prompts / Skills / Notes / All artifacts | Each says what it is | Conversations is clear ("Chat in Console and it appears here."). Prompts: "No prompts yet. Create or import a prompt to begin." (what is a prompt?). Skills shows the "Skill trust isn't set up…" banner although Jordan added no skills, plus "invocable: user & agent". Artifacts lists the export as "Registered · Registered · #1", "1–1 of 1 copies". | minor | 39–43 |
| 22 | 100x30 check | resize | Usable | Nav collapses to a grip and Items + Reader fit. The reader top is a 5-line `file:///private/tmp/...` path, leaving ~12 lines of text. The nav bar is clipped to "⌃8 Workflo". | minor | 44 (All artifacts), 45, 46, 47 |
| 23 | Turn provider on, ask | relaunch with mock → Search/RAG → `RAG Answer` → question → Enter | Grounded answer | "To openai: question + evidence", then Answer + "The answer does not cite available staged evidence." Cost line "openai · gpt-5.6-terra · $0.0006 (90 tok)". **The configured model is gpt-4.1-mini.** There are 5 evidence cards with snippets and "match: strong / weak (0.16)". | minor | 48–52 |
| 24 | Pick evidence for Console | `Select evidence` on card 2 | Card selected, stay in place | Card selected, **but the panel jumps to the top and focus lands on the `Media` source toggle** (footer "enter toggle Media") | major | 53 |
| 25 | Send with source | `Use in Console` → type → Enter | Answer using source | "Staged for next send · 1 source · web-page-spaced-repetition — media [Un-stage]". The inspector shows "Sources: 1 staged". Sent OK. Library place was preserved on return. | none | 54, 55, 56 |
| 26 | RAG at 100x30 | Run / Enter | See the answer | The viewport stays on query + Sources. Only the "Answer" heading shows, on the last row. | major | 57, 58, 59 |
| 27 | Import a typo'd path | `~/fixtures/import-docs/missing-report.pdf` → Enter | "Not found" | Three messages at once: "Invalid path: Path contains dangerous pattern: ~/", "Can't find that path — check it, or use Browse…", and toast "Could not find that file." | major | 60 |
| 28 | Import an existing `~/` path | `~/fixtures/import-docs/team-handbook.docx`, wait 3 s → Enter | Import | "Invalid path: Path contains dangerous pattern: ~/", then "Can't find that path…" **for a file that exists**. Pressing Enter immediately (before the pre-check) imports instead: job 6 "Already in Library — matched". | major | 61, 62, 63 |
| 29 | Export from Media **[P2-probe]** | Media → `Export…` | Export form | **App froze: no repaint, no keys (F1, Ctrl+P, Ctrl+Q) and no resize. The process sat idle at 0% CPU and had to be killed.** Reproduced 3/3, including on a fresh profile with 1 import. | **blocker / data-loss risk** | 68, 69, 70, 71, 72 |
| 30 | Open from Search **[P2-probe]** | Search `local-first` → card "1. article-local-first" → `Open` | That article | On a reused screen the **reader loads team-handbook** (the first list row, "· loaded"). Read later then saves team-handbook. `Move to trash` gives an untitled "Delete this media?" for whatever is shown (cancelled). | **major** | 73, 74, 75, 76 |
| 31 | Map S8/S9 **[P2-probe]** | `More ▸ Open original`, `Open manager` on the local PDF | Something | No visible change and no message for either | minor | 66, 67 |

## Task outcomes

Steps count distinct intents. Keystrokes count clicks + key presses + typed characters (approximate).

| Task | Outcome | Steps | Keystrokes | Note |
|---|---|---|---|---|
| 1. Find Library and say what it is for | success | 3 | 3 | Only "Explore all tools" made the purpose clear. The landing alone did not. |
| 2. Import the 5 documents, watch, handle failure | partial | 6 (+3 failure probes) | ~8 (+~90 typed for failure probes) | The happy path is excellent. The failure path is misleading: `~/` is called "dangerous" and "can't find" for an existing file. |
| 3. Find the PDF, read it, find a phrase | partial | 5 | ~22 | Opening the PDF is easy, but Find gives a wrong count with no visible marks. A side effect saved the wrong item for later. |
| 4. Search across the Library | partial | 3 | ~20 | Keyword search works. The natural-language question returns 0 in Search mode. |
| 5. Ask with no provider, then with provider | success | 4 + 4 | ~55 + ~55 | The block is clear and the fix link works. With the provider: egress line and cost shown, but the model differs from the configured one and the answer is below the fold at 100x30. |
| 6. Use a document in Console, come back | partial | 9 | ~15 | Reader `Use in Console` is always blocked (workspace). Search ▸ Select evidence ▸ Use in Console works. Place is preserved. |
| 7. Keep something for later + export a bundle | partial | 10 | ~25 | Read later hit the wrong item. Quick Capture is web-only. The rail Export wrote a text-only zip. **Media Export… froze the app.** |
| 8. Say what Conversations / Prompts / Skills / Collections are | partial | 5 | 5 | Conversations and Collections are clear. Prompts are unexplained. Skills shows trust jargon for zero user skills. |

## Emotional journey

- **Start (curious, slightly lost):** boot lands on Console, not Library. The Library "Get started" page names steps but not the purpose. F1 helps little.
- **Peak 1 (confident):** the import picker, the type summary and the live per-file queue are the best moment. "It tells me exactly what it's doing."
- **Valley 1 (distrust):** pressing `Read later` on the PDF swaps the reader to a different document and saves that one instead. Jordan stops trusting the buttons.
- **Valley 2 (stuck):** `Use in Console` says "Copy or link this media into workspace workspace-default". Jordan has never heard of a workspace and finds no control that copies or links anything.
- **Recovery (relief):** the no-provider callout plus `Open Settings ▸ Providers` is clear. The Search ▸ Select evidence ▸ Use in Console detour works, and Library remembers exactly where Jordan was.
- **Valley 3 (alarm):** `Export…` from the Media list freezes the whole app. Even Ctrl+Q is dead.
- **End (wary):** Jordan got the content in and asked a grounded question. Jordan would not trust delete or export without checking, and would describe Library as "great at importing, unpredictable after that".

## Strengths

1. **The import pipeline narrates itself.** Before starting it summarises what it found ("1 PDF document, 3 plain text files, 1 Word/Office document · 5 files · 46.0 KB · 5 will import"). Each file then shows a live phase (`● parsing · Preparing import` → `● writing · Saving to Library` → `✓ done · 3s`) with its own `Open in Library`, and the rail counts update live. This is design principle 4 done right: state, progress and the next step all sit at the point of need (captures 07–09).
2. **Honest blocked and egress states in RAG.** With no provider the RAG Answer mode shows a full-width callout naming the problem and the fix ("No analysis provider is configured · Set one in Settings ▸ Providers & Models.") plus a working `Open Settings ▸ Providers` button, and Run is visibly dimmed. With a provider it says before you spend what leaves the machine ("To openai: question + evidence"). Afterwards it shows cost ("$0.0006 (90 tok)") and a caution when citations do not validate. This meets principle 8, "make advanced capabilities honest" (captures 19, 50, 51).
3. **Round-trips to Console keep your place, and staging is legible.** Twice, Library → Console → `⌃3` restored the Search/RAG query, results, selected card and scroll position. Console shows exactly what will be sent: "Staged for next send · 1 source · web-page-spaced-repetition — media", with `Un-stage` and inspector "Sources: 1 staged" (captures 30, 54, 56).

## Findings

Severity: P0 = blocks, data loss, crash, or a lie about saved/synced state; P1 = significant difficulty; P2 = annoyance with a workaround; P3 = polish.

### F1 — P0 — Media list `Export…` freezes the whole app [P2-probe] (defect, high confidence)

**Evidence**
- Captures 68–72. Three reproductions, two of them after a fresh relaunch; one on a fresh empty profile with one imported .md (`nl-j2-2`).
- After the click the panes re-layout once. After that F1, Ctrl+P, rail clicks and Ctrl+Q do nothing, and a `tmux resize-window` produces no repaint.
- The process is idle (0.0% CPU). A SIGUSR2 faulthandler dump shows the main thread idle in `selectors.select` / `asyncio _run_once`, and the Textual input and writer threads idle.
- No traceback or error in the log.

**Repro:** Library → `Media (N)` → click `Export…` in the Items toolbar (with or without a filter).

**Why:** this is the single most "save my stuff" action, and it bricks the session. Any unsaved Notes/Prompt edits in the same session are lost when the user has to kill the terminal.

**Cause (hypothesis, traced):**
1. `Widgets/Library/library_media_canvas.py:537-546` makes `handle_library_media_export` an **async handler on the LibraryMediaCanvas itself**. It awaits `MC.handle_library_media_export` → `library_export_controller.py:741-786 _open_library_export_canvas` → `library_screen.py:12316+ _apply_library_open_item_surface`.
2. Because the mounted reader is `.library-media-route` and the destination (export) has none, that function takes `await self.recompose()` (`library_screen.py:12404-12408`). That removes the very canvas whose message pump is still awaiting inside the handler: a self-removal deadlock.
3. The rail `Export` row (a different path) did not hang. The handler's docstring says "the controller's Export handler awaits a modal", which no longer matches what it does.

**Fix:**
- Make the canvas forwarder non-awaiting: `self.app.call_later(actions.handle_library_media_export, event)`, or `run_worker(...)`.
- Or have `_open_library_export_canvas` schedule the surface swap with `call_after_refresh` instead of awaiting it from inside a handler owned by the canvas being replaced.
- Add a Pilot test that presses `#library-media-export` and then asserts the export canvas mounts and a later key is processed.

### F2 — P1 — Opening an item by link leaves the list cursor on row 1, so the reader and its actions target a different document (defect, high confidence)

**Evidence**
- Captures 11–14 and 73–76; DB table `MediaReadItLaterState`.
- After the import queue's `Open in Library`, the PDF shows in the reader with "· loaded", while the list cursor `█▸` is on team-handbook.
- Clicking the reader's `Read later` saved **media_id 5 (team-handbook)** 3/3 times. The PDF/article (ids 3/2) were not saved, and the reader swapped to team-handbook showing "Remove later".
- Search/RAG evidence `Open` on "1. article-local-first" has the same effect. On a screen that already had Media mounted it **loaded team-handbook into the reader outright** (capture 76: header "team-handbook", row "document · … · loaded").
- Control: clicking the PDF row itself and then `Read later` saved id 3 correctly.
- `More ▸ Move to trash` then asks the untitled "Delete this media? You can undo right away…" (capture 75; cancelled), so a user can trash the wrong item.

**Repro:**
1. Import a folder, then click `Open in Library` on any row except the newest.
2. Click `Read later`.
3. `sqlite3 media_v2.db "select * from MediaReadItLaterState"` shows the first list row saved, not the opened item.

**Why:** actions silently apply to an item the user did not choose. This breaks trust in every reader button (Read later, Move to trash, Use in Console), and the delete confirm does not name the item.

**Cause (hypothesis):**
- `library_screen.py` `_open_library_item_by_id` (media branch, ~34787-34857) sets `selected_media_id` and then calls `_request_library_media_browse(...)`.
- The re-rendered list's cursor stays at index 0, and the settle-delay highlight selection (`library_media_controller.py:2176-2201 _select_library_media_reader_row`, `immediate=False`) then re-points `_selected_media_id` to row 0. `_start_library_media_read_later_toggle` (`library_media_controller.py:4533-4552`) reads that id.

**Fix:**
- In the media branch of `_open_library_item_by_id`, move the Items list cursor/highlight to `record_id` (scroll it into view) before the browse refresh lands.
- Have toolbar actions use the reader session's loaded id, not `_selected_media_id`.
- Put the title in the delete confirm: "Move 'team-handbook' to Trash?".

### F3 — P1 — "Use in Console" is always refused for imported media because the built-in Default workspace is active and nothing can link media into it (defect / missing capability, high confidence)

**Evidence**
- Captures 24, 25, 64, 65.
- Reader `Use in Console` (and `c`) shows only the toast "Copy or link this media into workspace workspace-default before using it in Console.". The internal id is shown instead of the name "Default".
- Details reads "Active · Local Default · Handoff · 5 items can't be used in Console yet · not in this workspace · Copy or link them into this workspace". Details `Use in Console` toasts "Copy or link blocked Library sources into the active workspace before using them in Console."
- The `More` strip has no link/copy action. `workspace_records` holds only `workspace-default` (active=1, created by the app on first boot) and `workspace_memberships` = 0.
- In code, `registry.link_membership(` is only ever called for conversations (`library_screen.py:13906`), notes (`library_notes_controller.py:4494`) and chat persistence (`Chat/chat_persistence_service.py:2441`). There is no media path.
- Meanwhile the Import screen promises "Imported items are searchable in your Library and can be used as context in chat." (capture 05), and Search ▸ Select evidence ▸ Use in Console *does* stage the same media (capture 29). Map S11 was confirmed and is worse than suspected: it affects every first-time user, not only multi-workspace ones.

**Repro:** empty profile → import any file → open it → `Use in Console`.

**Why:** the core loop "import a source → reason over it in Console" (PRODUCT.md purpose) dead-ends on the most obvious button. The recovery copy names an action that does not exist, using jargon the user has never seen.

**Fix:**
- Treat the built-in `workspace-default` like the "no active workspace" branch (`Workspaces/display_state.py:582-600`): local Library sources are eligible.
- Or auto-link new imports to the active workspace at ingest commit.
- Or add a `Link to workspace` action to the media reader, mirroring the conversation reader's `#library-conversation-link-workspace`, and render the button as "○ Use in Console · not in this workspace" with that action beside it.
- Use the workspace *name* in copy, never `workspace-default`.

### F4 — P1 — `~/` paths are called "dangerous", then "can't find that path", even when the file exists (defect, high confidence)

**Evidence**
- Captures 60–63; log `WARNING … Rejected Library ingest path '~/fixtures/import-docs/missing-report.pdf'`.
- Typing `~/fixtures/import-docs/team-handbook.docx` (exists) and waiting for the pre-check shows "Invalid path: Path contains dangerous pattern: ~/". Enter then shows "Can't find that path — check it, or use Browse… to pick a file or folder."
- If Enter is pressed before the pre-check finishes, the same path imports (job 6, "Already in Library — matched"): the behaviour is a race.

**Cause (traced):** `Library/ingest_preflight.py:421-424` calls `validate_path_simple(path_or_url, require_exists=False)` without `expanduser`. `Utils/path_validation.py:559-571` lists `"~/"` as a dangerous pattern.

**Why:** terminal-first users type `~/…` by reflex. The app accuses their input of being a security risk and then claims a real file is missing, and sends them to the mouse picker.

**Fix:** call `os.path.expanduser` on the import field value before preflight and submit, in both the preflight and `_resolve_ingest_source` (`library_screen.py:~28752`), and keep the `~/` check only for non-expanded internal paths. When validation fails, show one message, not three.

### F5 — P2 — Search mode returns nothing for a question that the documents answer verbatim; the box invites questions (usability, high confidence)

**Evidence:** captures 20, 21, 50–52.
- The placeholder is "Ask or search Library sources".
- In Search mode, "How much did retrieval practice improve retention?" gives "No evidence matched … Try broader terms.", although the PDF says "Retrieval practice improved 7-day retention by 21 percentage points".
- The same question in RAG Answer mode retrieves 5 results, #1 being that PDF "match: strong".
- The rail `Search Library…` box silently forces Search mode (map S12 / 7.12 confirmed live: the mode label flipped from `✓ RAG Answer` to `✓ Search`).

**Why:** a first-timer asks in natural language, gets "no evidence", and concludes the import failed.

**Fix:**
- When keyword search yields 0 for a query with ≥4 words, run the semantic/hybrid retrieval the RAG path already uses (or retry with OR-of-terms), and label it "No exact matches — showing related passages".
- Add an inline "Ask this as a question (RAG Answer)" button in the 0-results notice.
- Make the rail box preserve the canvas's current mode instead of resetting it.

### F6 — P2 — In-document Find: wrong count and no visible match in the default view (defect, high confidence)

**Evidence**
- Captures 15, 16. Map S18 confirmed.
- In the default `Rendered (selected)` view, "7-day retention" reports "Match 1 of 2" with no highlighted cells: every reader row is plain `#e0e0e0 on #242f38` in the ANSI.
- The stored content contains the phrase **3** times. In Raw only the first occurrence on each source line is reverse-video, and the second on the same line is unstyled.

**Cause (traced):** `Widgets/Library/library_media_content.py:465` documents `matches` as "Source-line indexes", and `_status_text` (`:544-550`) counts lines.

**Fix:**
- Count and step through occurrences (line, column), and highlight every occurrence.
- When Rendered cannot mark matches, auto-switch to Raw while a query is active (the Analysis tab already does this) and say so: "Showing raw text to mark matches".

### F7 — P2 — RAG Answer bills a different model than the one the user configured, and the pre-run disclosure omits the model (defect, high confidence)

**Evidence:** captures 50, 51; mock request log `20:23:21 … model=gpt-5.6-terra` and `20:25:14 … model=gpt-5.6-terra` for my RAG runs.
- Console sends with `Model: gpt-4.1-mini` (capture 54), and `chat_defaults.model = "gpt-4.1-mini"` in the run config.
- Before Run the panel says only "To openai: question + evidence". After Run it says "openai · gpt-5.6-terra · $0.0006".

**Cause (traced):** `Library/library_rag_answer_service.py:163-189 resolve_library_rag_answer_provider` returns `(endpoint, None)` on purpose ("No model is resolved: the provider handler picks its own default").

**Why:** a user who picked a cheaper model is charged for the provider default without warning, which breaks "source authority/cost visible before action".

**Fix:** resolve the model from `chat_defaults.model` when its provider matches `default_api_endpoint` (or add a model picker on the RAG panel), and show it before Run: "To OpenAI · gpt-4.1-mini: question + evidence".

### F8 — P2 — `Select evidence` throws focus onto a Sources checkbox and can scroll the panel to the top (defect, high confidence)

**Evidence**
- Captures 28, 53.
- After clicking `Select evidence` on card 2 (row 27), the panel jumped back to the query field and the footer read "enter toggle Media". Another time it read "enter toggle Conversations". Pressing Enter at that point would switch a source off.

**Cause (hypothesis):** `library_rag_search_controller.py:1103-1122 _select_library_rag_result_by_index` calls `_refresh_search_rag_panel_state_widgets()`, which rebuilds the result cards. The pressed button disappears and Textual moves focus to the next focusable (the source toggle), and the scroll follows.

**Fix:** after the refresh, restore focus to the same card's (now "Selected evidence") button by id, and keep the scroll offset. Or patch the button label in place instead of rebuilding the card.

### F9 — P2 — Export copy promises full files; the bundle is text-only with opaque names (copy, high confidence)

**Evidence**
- Captures 35–38 and `unzip -l "Library export 2026-10-02.zip"`: `README.md`, `manifest.json`, `content/media/media_1..5.txt`, `content/media/metadata/media_N.json`. No PDF or DOCX is included.
- The form says "quality: original · copies full media files into the zip" and, two lines below, "Bundle: 5 items · text only · size known once it runs". The last line is unchanged after the export finished.
- Map S3 confirmed (helper contradicts bundle).

**Fix:**
- When no originals are stored, hide or disable `quality:` and say "Text and metadata only — original files are not stored in this Library".
- Name files by slugged title (`paper-retrieval-practice.txt`).
- Replace "size known once it runs" with the actual size after the run ("7.1 KB").

### F10 — P2 — Web-page (HTML) import keeps page chrome and ignores `<title>` (defect, high confidence)

**Evidence**
- `Media.content` for id 1 ends with "Site navigation that should be stripped". The title row text is duplicated ("Spaced repetition, explained" ×2).
- The table is flattened to one cell per line ("Interval / Days / 1 / 1 / 2 / 3").
- That chrome surfaces in RAG evidence snippets (capture 52) and in the export.
- The item title is the filename stem "web-page-spaced-repetition", not the page `<title>` "Spaced repetition, explained" (same for the md's `# The case for local-first software`).

**Fix:**
- Run local HTML through the same readability/article extraction used for URL imports: drop `nav`/`header`/`footer`/`aside` and keep table rows as `a | b`.
- Default Title to `<title>`/first `<h1>` (or the Markdown H1) and fall back to the filename.

### F11 — P2 — `More ▸ Open original` and `Open manager` do nothing for a local import [P2-probe] (defect, high confidence)

**Evidence:** captures 66, 67. Map S8/S9 confirmed live. No visible change and no toast for either.

**Cause (traced):** `library_media_controller.py:4196-4202` opens only `http(s)` URLs, and local imports store `file://…`.

**Fix:**
- For `file://` sources, open the file with the OS handler (`open`/`xdg-open`), or reveal it in Finder, or hide the action.
- Remove `Open manager` from the Library reader (it navigates to Library from Library).

### F12 — P2 — At 100x30 the RAG answer lands off-screen (usability, high confidence)

**Evidence**
- Captures 57–59. After Run/Enter the viewport still shows the query and the Sources block, which uses 3 rows per checkbox (12 rows for 4 toggles). Only "Answer" is visible, on the last row.
- At 160x45 only ~8 answer rows are visible (capture 50).

**Fix:**
- After an answer arrives, scroll the Answer heading to the top of the panel, even when the query input has focus.
- Compact the Sources toggles to one row each (or one line "Sources: ☐ Notes (0) ☑ Media (5) ☑ Conversations (1) ☐ Prompts (0)").

### F13 — P2 — The reader is the narrowest pane, and a raw absolute `file://` path eats its top lines (usability, high confidence)

**Evidence**
- Captures 11, 47. At 160 cols, rail 34 + list 56 leaves the reader at ~54 cols, with the PDF wrapped at ~50 chars.
- Under the title, a 4-line (5 at 100x30) byline `file:///private/tmp/…/import-docs/paper-retrieval-practice.pdf` appears. At 100x30 only ~12 lines of document remain.
- Map S10 confirmed.

**Fix:**
- Show the byline as `paper-retrieval-practice.pdf · local file`, with the full path only in Info.
- When a document is opened, auto-collapse the Nav rail once, as Notes already does.

### F14 — P3 — "Keep for later" is split across three unconnected places, and Quick Capture has an unlabeled field (usability, high confidence)

**Evidence**
- Captures 31–34. Read later (Media reader) is not shown on list rows and does not appear in Collections ▸ `Reading`. It is reachable only via Media ▸ `Sets` ▸ `Review read-later`, an overlay that also clips the reader text.
- Collections Quick Capture accepts only http(s) URLs.
- The Quick Capture 4-row `TextArea` (`library_collections_capture_reader.py:438-441`) has no label or placeholder.
- Map S27/S28 confirmed.

**Fix:**
- Add a `· later` fact to list rows and a `Read later (N)` scope under Media.
- Label the text area "Note (optional)".
- In the Collections empty state, say "Captures are web pages saved by URL. To keep a Library file for later, use Read later in Media."

### F15 — P3 — A fully successful import still offers retry (copy, high confidence)

**Evidence:** captures 09, 10. With "This queue: 5 done" the footer shows `r retry` and the canvas shows `Retry this batch`. Per `library_ingest_canvas.py:2079-2092` it re-stages the batch into the form rather than retrying anything.

**Fix:** after an all-success batch, rename it to "Import these again…" or hide it, and drop `r retry` from the footer unless a row failed.

### F16 — P3 — Help and empty states do not explain what things are (copy, high confidence)

**Evidence:** captures 03, 40, 41.
- F1 on the Library landing lists three shortcuts in a ~30-row empty frame, with no sentence on what Library is.
- Prompts says "No prompts yet. Create or import a prompt to begin." with no definition.
- Skills shows "Skill trust isn't set up, so every skill you added reads 'needs review'…" with zero user-added skills, and its only row says "invocable: user & agent".

**Fix:**
- Add one purpose line to F1/landing: "Library holds everything you've imported, chatted or written, so you can find it, ask about it, and send it to Console."
- Prompts empty state: "A prompt is reusable instruction text you can insert into Console."
- Show the trust banner only when at least one user skill exists, and render "Can be run by: you or an agent".

## Improvement opportunities (beyond defects)

1. **"Ask about this document" in the reader.** A button that opens Search/RAG pre-scoped to the open item (sources = this item) in RAG Answer mode. It turns "find a phrase" and "ask a question" into one move, and removes the 4-step Search detour Jordan needed to get a document into Console.
2. **First-run "it just works" workspace.** Make the built-in Default workspace mean "all my local stuff" so newcomers never meet the word workspace until they create a second one. Show workspace scoping as an opt-in.
3. **One "Saved for later" surface.** Unify Read later (media), Review sets and Collections captures under one rail row with type filters. Jordan expected "Read later" to show up in Collections ▸ Reading.
4. **Search that degrades gracefully.** When keyword search finds 0, show semantic matches and a one-click "Ask as a question". Also show a matched-text snippet on Search-mode cards; today a card says only "Matched media · html".
5. **Make answer provenance concrete.** Each answer sentence could carry `[S1]` markers linking to the evidence card. Today every card says "Media · 1 citation / Citations: <its own title>", which repeats the card and does not tie claims to evidence.

## Nielsen heuristic scores

Scale: 4 = fully meets, 3 = minor issues, 2 = notable issues, 1 = major issues, 0 = fails.

### Library shell and destinations

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | 2 | The import queue is exemplary, but Media `Export…` freezes with no indication. The RAG answer lands off-screen at 100x30. The "loaded" row disagrees with the cursor `▸`. |
| 2 | Match with the real world | 2 | "RAG Answer", "analysis provider", "staged evidence", "workspace workspace-default", "invocable: user & agent", "Registered · Registered". |
| 3 | User control and freedom | 2 | Place is preserved across Console round-trips and Un-stage exists, but the freeze forces a kill, filter-as-you-type replaces the open doc (S7), and Select evidence yanks focus. |
| 4 | Consistency and standards | 1 | `Use in Console` works from Search evidence but is refused from the reader and Details. Read later acts on another item. The rail box and canvas box run different modes. There are three "save for later" concepts. |
| 5 | Error prevention | 1 | Actions hit the wrong item after open-by-link. The delete confirm has no title. `~/` paths are mis-rejected. A one-click freeze. |
| 6 | Recognition rather than recall | 3 | Footers are contextual and accurate, and the rail shows counts and short purposes. Read-later items are hidden under `Sets`. |
| 7 | Flexibility and efficiency | 3 | Keyboard accelerators, F6 panes, and `u`/`o`/`c` keys. Only one evidence row can be staged at a time. |
| 8 | Aesthetic and minimalist design | 2 | The reader is the narrowest pane, a 4–5-line `file://` byline, 3-row checkbox spacing, and doubled grips ("N a v" beside an open Navigation). |
| 9 | Recognize, diagnose and recover from errors | 1 | "dangerous pattern: ~/" plus "Can't find that path" for a real file. The workspace toast names a non-existent action. Open original is a silent no-op. |
| 10 | Help and documentation | 2 | F1 lists 3 shortcuts in an empty frame. Empty states explain Conversations and Collections but not Prompts or Skills. |

### Notes

Only the empty state was seen, so most heuristics were not assessed (-1).

| # | Heuristic | Score | Key issue |
|---|---|---|---|
| 1 | Visibility of system status | -1 | Not assessed (only the empty state seen). "Library notes · Ready · Next: Create a note or add from files." reads well. |
| 2 | Match with the real world | -1 | Not assessed |
| 3 | User control and freedom | -1 | Not assessed |
| 4 | Consistency and standards | -1 | Not assessed |
| 5 | Error prevention | -1 | Not assessed |
| 6 | Recognition rather than recall | -1 | Not assessed |
| 7 | Flexibility and efficiency | -1 | Not assessed |
| 8 | Aesthetic and minimalist design | -1 | Not assessed |
| 9 | Recognize, diagnose and recover from errors | -1 | Not assessed |
| 10 | Help and documentation | -1 | Not assessed; the empty-state explanation of Library notes vs Folder files was clear (capture 42) |

## Harness caveats

- The null keyring backend applies, so the Skills trust banner state is not judged as keychain behaviour. Only "banner shown with zero user skills" is reported.
- The mock LLM echoes its prompt, so answer quality is not judgeable, and "The answer does not cite available staged evidence." is expected for an echo. Citation *display* is judged, not citation correctness.
- `mock_llm.requests.log` is shared with other agents. My requests were matched by timestamp (20:23:21 and 20:25:14 `model=gpt-5.6-terra`, tools=0).
- The empty profile's `workspace-default` (active) was created by the app's own first boot, not seeded, so F3 reflects product behaviour. A profile that completed the first-run wizard was not tested (`[first_run] setup_completed=true` skips it).
- Import triggered an embeddings download warning from HF Hub (`huggingface_hub … unauthenticated requests`). Network egress during import is not surfaced in the UI, but this may be an artifact of the shared venv/model cache.
- `py-spy` needs root on macOS. The freeze stacks came from the app's own SIGUSR2 faulthandler hook (main thread idle in `select`), which cannot show asyncio task stacks, so the F1 cause is a traced hypothesis.
- Capture naming: `44-media-reader-100x30` actually shows All artifacts at 100x30. `63-import-tilde-enter-imports-anyway` shows the "Can't find that path" refusal; the immediate-Enter import is in `61`.
- PNGs are approximate (no box borders). Colour and focus claims come from `.ansi` via `rowstyles.py`.
- The real profile was untouched (`snap.sh` diff clean), and no `runs/nl-j2-*` processes remain.
