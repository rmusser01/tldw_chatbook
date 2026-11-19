# J7 — Researcher loop ("Priya"): import → read + note → quote → ask → study → find → export

- **Code under test:** `.worktrees/notes-library-ux-review` @ origin/dev `2d34cbf80d`, Textual 8.2.8. Live on 2026-10-02 (20:09–20:31 PDT).
- **Harness:** `nlrev` isolated profile GOLDEN plus the import-docs fixture. Sockets `nl-j7-1` (main run) and `nl-j7-2` (fresh-launch re-check of one finding). Shared mock LLM on `:18777` (OpenAI / gpt-4.1-mini).
- **Sizes:** 160x45 primary, 120x36 for the side-by-side checks, and one 160x70 probe.
- **Evidence:** `../evidence/j7-researcher-loop/NN-*` (`.txt` and `.ansi`; PNG for 06, 31, 40, 42, 44, 56). Exported bundle artefacts: `68-exported-*.txt`.
- **Method:** a 10-minute **blind** attempt at steps 1–3 (20:09–20:13), then a read of `maps/map-notes.md`, `map-media-ingest-search.md` and `map-conv-prompts-skills-artifacts.md`, then the full loop. Every cause I give is traced to `file:line` or labelled as a hypothesis.

## Persona and goals

Priya is a PhD researcher, one of the expansion users named in PRODUCT.md. She reads papers and takes notes while she reads. She asks questions that are grounded in her sources, and she produces summaries and study material for her advisor. She is moderately technical. Her non-negotiable is provenance: every quote and every AI answer must lead back to the paper it came from.

Her goals for this session:

1. Get `paper-retrieval-practice.pdf` into the Library.
2. Read it and take notes at the same time.
3. Quote a passage with a link back to the paper.
4. Ask a question scoped to this paper, and keep the answer and its citations as a note.
5. Make flashcards or a quiz from the note or the paper.
6. Later, find everything about "retrieval practice" in one place.
7. Export her notes for her advisor with provenance intact.

## Step log

| # | Intent | Keys / clicks | Expected | Observed | Friction | Capture |
|---|---|---|---|---|---|---|
| 1 | Open Library (blind) | click `⌃3 Library` | Landing with an import entry | Landing: "Search everything, pick a section, or add something new." Quick actions offer Import… / New note / Search | none | 01 |
| 2 | Start the import | click rail `Import…` | Import form | "Import media · File, folder or URL to import · Browse…" | none | 02 |
| 3 | Pick the PDF | `Browse…` → click `📁 fixtures` → Enter → click the PDF → `Open` | Picker that remembers nothing yet, then a path | Picker opens at HOME ("1 of 2 entries shown"). After Open the pre-check reads "1 PDF document · 1 file · 3.0 KB" and "1 will import" | minor | 03, 04 |
| 4 | Import | click `Start import` | Progress, then done | "✓ done · paper-retrieval-practice.pdf · 2s", toast "Import finished — 1 imported", `Open in Library` | none | 05 |
| 5 | Open the paper | click `Open in Library` | Reader showing the paper | The reader opens, but the title is **"paper-retrieval-practice"** (the filename) and the byline is a 4-row `file:///private/…/paper-retrieval-practice.pdf`. For several seconds a raw log line, `[WARNING ] huggingface_hub.utils._http:904 - Warning: You are sending unauthenticated requests to the HF Hub…`, was painted over the top nav bar | minor | 06 (PNG), 07 |
| 6 | Look for "take a note" in the reader | `More` | A note action | Only "Edit metadata · Open original · Open manager · Move to trash" | major | 08 |
| 7 | Try Highlights as a clipping tool | `Highlights` → `▶ Add highlight` → type the quote and a note → `Add highlight` | A highlight that can be sent to a note | A card “Retrieval practice improved 7-day retention by 21 percentage points.” / "Note: Key effect size for ch.2" whose **only action is `✕ Delete`**. The quote had to be retyped, because the Read text is on a different tab | major | 09, 10, 11 |
| 8 | Take a note | rail `New note` → Enter (Blank note) | Editor next to the reader | The reader is **replaced** by the Notes list and editor. No link to the paper is recorded | major | 12, 13 |
| 9 | Write notes and a quote | click Title, type; click Body, type a bullet and `> quote (paper-retrieval-practice.pdf)` | Autosave | "Saved 20:12 · Next: Keep editing…". The attribution is hand-typed plain text | minor | 14 |
| 10 | Look for a source field | `Info` | A "Source" or "Linked media" property | Only Created / Modified / Version / Words / "Linked from — checking…" / Copy / Export Markdown / Export text / Delete | major | 15 |
| 11 | (Mis-step) click the clipped `Use in` button | click `Use in` | A menu (the label is truncated) | It navigated straight to Console and staged the note. Back in Library the status reads "Use in Console complete — **Linked to Local Default** · staged in Console." The workspace link was written silently | minor | 16, 17, 19 |
| — | *End of blind attempt (≈4 min). Step 1 succeeded. Step 2 is impossible side by side. Step 3 worked only as a hand-typed quote with no link.* | | | | | |
| 12 | Measure one note-taking cycle, Notes → Media | `--->` grip, click `Media (24)` | The paper is still open where she left it | The list row says **"▸ paper-retrieval-practice · pdf · updated 4m · loaded"**, but the reader says **"Select a media item to read it here."**, still true 7 s later. Enter reloads the item on the **Highlights** tab, not Read | major | 20, 21 |
| 13 | Measure the cycle, Media → Notes | click `Notes (123)` | Her note is still open | Notes list with "Select a note to edit it here." The note is closed and must be found and clicked again | major | 22 |
| 14 | Copy a passage instead of retyping it | drag-select two lines in Read, then `Ctrl+C` | Feedback that the text was copied | The selection highlights (ANSI `on #125385`). `Ctrl+C` gives no toast and no footer hint. Pasting with `Ctrl+V` in the note body does work | minor | 23, 24, 25 |
| 15 | Ask a question scoped to this paper | reader `Use in Console` | Paper staged in Console | Toast: **"Copy or link this media into workspace workspace-default before using it in Console."** Nothing in Library can do that: rail Details says "Handoff · 135 items can't be used in Console yet · … Copy or link them into this workspace" but offers only "Create local workspace" | blocker | 26, 27, 28 |
| 16 | Work around via Search/RAG | rail `Search / RAG` → `mode:` → turn off ☑ Notes, ☑ Conversations, ☑ Prompts → type the question → `Run` | An answer from this paper only | Scope is per source **type** ("Scope: Media (Notes, Conversations, Prompts off)"), not per item. Before Run: "To openai: question + evidence". After: "openai · **gpt-5.6-terra** · $0.0006" (Console's model is gpt-4.1-mini). The callout says "The answer does not cite available staged evidence." There is **no save-as-note** on the answer | major | 29–32 |
| 17 | Hand the evidence to Console | `Select evidence` → `Use in Console` | The paper staged | Works for the **same paper** that step 15 refused. The composer still holds the stale prompt "Use this note as context and help me work with it." from step 11 | minor | 33–35 |
| 18 | Ask in Console | `End`, `Ctrl+U`, type the question again, Enter | Grounded answer | The mock answered. The first request carried "Evidence: [S1] MEDIA — paper-retrieval-practice" | none | 36 |
| 19 | Keep the answer as a note | click message → `More…` → `Capture as note` → `Open note` | A note with the answer, the question and its citation | "Saved to Notes — The answer is now a note in Library ▸ Notes, tagged with this conversation." The note holds the answer text only. Keywords are `console, conversation:71d30fb1-…, message:1f56b655-…`. **The question, the paper and the citation list are not recorded** | major | 37–40 |
| 20 | Make flashcards | rail `Flashcards due: 0` | Cards from this note or paper | Handoff canvas: "Carries forward: *(mock reply)*…, Paper notes: Retrieval practice, Ideas inbox **and 134 more.** … Source snapshot is ready." There is no way to choose just this paper or note | major | 41 |
| 21 | Continue in Study | `Continue in Study` → Dashboard → Flashcards | Generate, or at least hand-make cards | The Dashboard shows a summary of "+7 more" (it was "134 more" in Library) and **no action buttons** below "0 due today", even at 160x70. The Flashcards tab is a box containing only "Decks:" — no deck picker, Create Deck, Front/Back or Create Card | blocker | 42–47 |
| 22 | Return to Library | Esc | Back to Flashcards | Lands on "Study decks" | minor | — |
| 23 | Find everything about the topic | rail `Search Library…` → type "retrieval practice" → Enter | Notes, media and conversations | **"2 results"**, Media only. The source toggles from step 16 were silently kept, and the scope line was scrolled out of view | major | 48 |
| 24 | Fix the scope and re-run | `☐ Notes`, `☐ Conversations`, `Run` | Everything | "5 results": my note (with snippet), the paper, the conversation, the captured answer, "Make It Stick — chapter 2". Media cards show no snippet ("Matched media · pdf") | minor | 49, 50 |
| 25 | Find the captured note by its provenance tag | Notes filter "71d30fb1" + Enter | The captured note | "filter: 71d30fb1 · 0 results". Keywords are not searchable | major | 64 |
| 26 | Export for the advisor | Notes `Select` → click 2 rows → `e` | An "Export selected" button | The select strip reads "2 selected   Done   Select all 100 shown   Clear". **No Export selected button is visible.** Only the footer chip `e export selected` works | minor | 51 |
| 27 | Export | `Choose destination…` → `Save` → `Export bundle (.zip)` | A file the advisor can open, with provenance | A Chatbook zip containing README.md, manifest.json and one `.md` per note. `manifest.json` has `"tags": []` for the note that has 3 keywords. **Both** front-matter blocks are invalid YAML: `title: Paper notes: Retrieval practice` gives "mapping values are not allowed here", and `title: *(mock reply)*…` gives "while scanning an alias". The bundle line still reads "size known once it runs" after the run | major | 52–54, 68 |
| 28 | Check the media's own provenance | reader `Info` | Title, author, source | "Canonical ID: local:media:26", "Original source: file:///…". No Author. The PDF's own Title and Author metadata were ignored | minor | 55 |
| 29 | Side by side at 120x36 | resize and repeat | Reader and note together | The Items list and the Reader share the stage (reader ≈50 cols, 10 body rows). The note editor again replaces the reader. The Notes list's Unfiled folder is below the fold and **cannot be scrolled** (wheel, `g`+↓ and Tab all fail) | blocker (for browsing) | 56–59 |
| 30 | Re-check the list scroll on a fresh launch (`nl-j7-2`, 160x45) | wheel over the list rows | List scrolls | No change. Only 7 of 24 Unfiled notes are reachable without the Filter | blocker (for browsing) | 61–63 |
| 31 | A follow-up question in Console | type and Enter | Still grounded in the paper | The strip says "Staged for next send · 1 source / paper-retrieval-practice — media". The Inspect rail says "Sources: None staged" and also "Sources — next send 1". The status bar says "Sources: 0". The log shows `[ERROR] Console RAG capture unavailable; reason=capture_provider_failure; draft_length=36`, and nothing was shown in the UI | major | 65–67 |

## Task outcomes

| Task | Outcome | Steps | Keys/clicks | Note |
|---|---|---|---|---|
| 1 Import the PDF | success | 9 | ≈9 | Fast and honest. The title defaults to the filename and the PDF metadata is ignored |
| 2 Read and take notes together | fail | 8 per cycle | ≈13 per cycle plus re-scrolling | The reader and the note editor never coexist. Every round trip costs 2 context switches, and state is lost in both directions: the note closes, the reader blanks while its row says "loaded", the reader reopens on the last tab, and the scroll position is gone |
| 3 Quote with source attribution | partial | 9 | ≈10 plus hand-typed attribution | Drag-select with `Ctrl+C`/`Ctrl+V` works but is undiscoverable. Highlights are a dead end. No link back to the media item is possible |
| 4 Scoped question, answer and citations kept as a note | partial | 17 | ≈19, question typed twice | The reader's handoff is refused. The workaround is Search/RAG, then evidence handoff, then Console, then Capture. The saved note has no question, no source and no citations, only UUID keywords |
| 5 Study material | fail | 5 | 6 | The handoff carries the whole Library. The Study dashboard and flashcard editor do not render their controls, and generation is server-only |
| 6 Find everything on one topic | success, after recovery | 6 | ≈6 plus the query | One place exists (Search/RAG), but it silently kept the previous narrowed scope |
| 7 Export for the advisor | partial | 9 | 9 | Zip only, with Markdown inside. Keywords are dropped and the YAML front matter is invalid. No links to sources |

Over the whole loop: about 70 clicks and keystrokes, 11 destination or canvas switches, and the question retyped once. Provenance is lost at four points:

- **(a) Quoting.** There is no link type from a note to a media item.
- **(b) Capture.** The question and the cited source are not saved.
- **(c) Search.** Provenance keywords are not filterable.
- **(d) Export.** Keywords are dropped.

## Emotional journey

- **Peak at the start: import (≈1 min).** The pre-check, the "✓ done · 2s" row and "Open in Library" all inspire confidence.
- **First valley: "where do I write?"** Neither More nor Highlights leads to a note. New note *replaces* the paper.
- **A small lift.** Drag-select and `Ctrl+C` turned out to work, found by guessing.
- **Deep valley: the refusal.** "Copy or link this media into workspace workspace-default…" names an internal id and gives no button to fix it. It reads like the app does not trust her paper.
- **Recovery through Search/RAG.** Trust rose when the app flagged "The answer does not cite available staged evidence" instead of passing off the mock reply as grounded.
- **A peak.** "Saved to Notes → Open note" landed exactly on the captured answer. Then she reads the keywords: `conversation:71d30fb1-…`. The paper is not there.
- **The lowest point: Study.** She is told "Source snapshot is ready", then lands on a Flashcards page showing only "Decks:".
- **Relief.** Search/RAG lists the note, the paper and the conversation together, once she spots the stale scope.
- **End: wary.** The export is a zip whose YAML breaks in her Markdown tools and whose tags are gone. She would keep a separate reference manager and would not trust Chatbook as the system of record for citations.

## Strengths

1. **Import is fast, previewed and closes the loop.** The pre-check ("1 PDF document · 1 file · 3.0 KB", "1 will import") prevents surprises. The queue row and the toast confirm the outcome. "Open in Library" opens the reader on the new item, so there is no hunting. (Captures 04–06.)
2. **RAG honesty and cost visibility.** The answer panel says "The answer does not cite available staged evidence." and prints "openai · gpt-5.6-terra · $0.0006 (90 tok)". Evidence cards show "Citations: paper-retrieval-practice" and a match strength. This is exactly the source-authority transparency PRODUCT.md asks for. (Captures 31, 32.)
3. **One cross-source search, and a direct capture round trip.** With all sources on, a single query returns note, media and conversation hits together, and the note hit shows a snippet. Console's "Capture as note" pops a "Saved to Notes" dialog whose "Open note" deep-links into the editor on the new note. (Captures 39, 40, 50.)

## Findings

Ranked most severe first. P1 means significant difficulty, the kind that makes a user give up.

### F1 (P1) — Reader and note can never be visible together, and every switch discards the other side's state
- **Who it hurts:** anyone who takes notes while reading, which is this persona's core loop.
- **Evidence:** 12, 13, 20, 21, 22, 56.
  - Rail "New note" or "Notes (N)" replaces the Media reader in the single work pane.
  - Back on Media, the row reads "▸ paper-retrieval-practice · pdf · updated 4m · loaded" while the reader says "Select a media item to read it here." (still true after 7 s).
  - Enter reloads the item on the last tab (Highlights), not Read.
  - Back on Notes, "Select a note to edit it here." — the note is closed.
  - At 120x36 the reader is ≈50 cols with ≈10 body rows. The note body at that size is about 4 rows (per the HARNESS editor anchors).
- **Cost:** one note-taking cycle is about 8 clicks plus about 5 keys plus re-scrolling, with 2 context switches.
- **Fix:**
  - Add a "Take note" action to the Media reader toolbar, next to `Find · Read later · Use in Console`. It should open a note editor *beside* the reader (split the work pane at ≥140 cols; at narrower widths toggle with one key) and pre-bind the note to the media item.
  - Keep per-row state when switching rail rows: the open media id, the tab and the scroll offset; the open note id and the caret.
  - Never render the "Select a media item…" placeholder next to a row marked "loaded". Re-hydrate the reader from the selected row.

### F2 (P1) — Media "Use in Console" is refused with no in-app remedy, though the same paper stages fine through Search/RAG and notes auto-link
- **Who it hurts:** every user who imports a file and wants to ask about it. This blocks step 4 from its most natural entry point.
- **Evidence:** 26 (toast "Copy or link this media into workspace workspace-default before using it in Console."), 27–28 (rail Details "Handoff · 135 items can't be used in Console yet … Copy or link them into this workspace", with actions limited to "Create local workspace" and "Use in Console"), 35 (the same paper staged via the evidence `Use in Console`).
- **Code:**
  - Refusal: `UI/Screens/library_screen.py:35540-35548`, with copy from `Workspaces/eligibility.py:79` (it interpolates the raw id).
  - Notes auto-link instead (`UI/Library_Modules/library_notes_controller.py:4594-4616`). Conversations have a "Link to workspace" button (`library_screen.py:13855-13910`). Media has neither.
- **Fix:**
  - Apply the notes behaviour to media: link the item to the active workspace on handoff and say so in the reader status ("Linked to Local Default · staged in Console").
  - Or render an inline "Link to Local Default and use" button next to `Use in Console`.
  - Always show the workspace display name, never `workspace-default`.

### F3 (P1) — No way to quote a passage into a note with a link back to the media item
- **Who it hurts:** researchers, who need every quote to be traceable.
- **Evidence:**
  - 11: the highlight card's only action is "✕ Delete", and the quote has to be retyped because the Read text sits on another tab.
  - 23–25: drag-select plus `Ctrl+C` copies text with no feedback, and the paste carries no attribution.
  - 15: note Info has no source field.
  - 55: media Info shows "Canonical ID: local:media:26", but notes support only `note://` links (`MCP/resources.py` defines `media://` for MCP only; no Notes or Preview handler).
- **Fix:**
  - Add "Quote to note…" in two places: on a reader text selection, and on each highlight card next to "✕ Delete".
  - It inserts `> <quote>` followed by `— [paper-retrieval-practice](media://26)` into a chosen or new note, and records a note↔media relation.
  - Render `media://` links in note Preview so they open the reader at that item.
  - List "Notes citing this item" in media Info.
  - Show a "Copied N characters" toast on `Ctrl+C` in the reader, and add `ctrl+c copy` to the reader footer when a selection exists.

### F4 (P1) — Kept AI answers lose their provenance: no question, no source, UUID keywords nobody can search, and no save path at all from Library RAG
- **Who it hurts:** researchers who need to prove where an answer came from.
- **Evidence:**
  - 40: the note's title and body are the answer text. Keywords: `console, conversation:71d30fb1-a206-…, message:1f56b655-…`. Nothing names paper-retrieval-practice or the question.
  - 39: the dialog promises "tagged with this conversation".
  - 64: filter "71d30fb1" returns "0 results", because keywords are not FTS-indexed (map S-02).
  - 32: the Library RAG answer has no save action.
- **Code:** `UI/Console_Modules/message.py` `_capture_console_answer_as_note` (keywords from `console_note_provenance_keywords`; content is the message only).
- **Fix:**
  - Capture writes a provenance block into the note body or front matter: the question, the model, the date, and "Sources:" with links to each staged or cited item (`media://26`, `note://…`).
  - Render the conversation link as "From conversation: <title>" (clickable), not a UUID keyword.
  - Add "Save answer as note" under the RAG Answer panel, with the same provenance block.
  - Index keywords in the notes filter.

### F5 (P1) — Bulk notes export drops every keyword and writes invalid YAML front matter
- **Who it hurts:** the advisor hand-off, and anyone who opens the export in Obsidian, Pandoc or a static site.
- **Evidence:** 68 (`manifest.json` `"tags": []` for a note with 3 keywords in `note_keywords`).
  - `title: Paper notes: Retrieval practice` fails `yaml.safe_load` with "mapping values are not allowed here".
  - `title: *(mock reply)*…` fails with "while scanning an alias".
- **Code:**
  - `Chatbooks/chatbook_creator.py:1434` uses `db.get_note_by_id`, which runs `SELECT * FROM notes` (`DB/ChaChaNotes_DB.py:17826`). The notes table has no keywords column, so `note.get("keywords")` at `:1454` is always empty.
  - `:1465` writes `f"title: {note['title']}"` unquoted.
  - The single-note export has the same unquoted pattern (`Library/library_notes_state.py:915`).
- **Fix:**
  - Load keywords through the note_keywords join before writing.
  - Emit front matter with `yaml.safe_dump` (or JSON-quote every scalar) in both writers.
  - Add a round-trip test that exports titles containing `:`, `*`, `#` and quotes, and parses them back.
- **Related:** the select strip at 160x45 and 120x36 never shows the "Export selected" button (51, 57). Only the footer chip `e` reaches it. Wrap the strip, or move "Export selected" ahead of "Select all N shown".

### F6 (P1) — The Study handoff dead-ends: whole-Library snapshot, invisible Study controls, server-only generation
- **Who it hurts:** students and researchers making study material, who reach a blank page.
- **Evidence:**
  - 41: "Carries forward: … and 134 more · Source snapshot is ready · Continue in Study". The snapshot cannot be scoped to the open paper or note.
  - 44–46: the Study Dashboard ends at "0 due today". "Resume last session / Open flashcards / Open quizzes / Generate source pack" and the status line never render, even at 160x70.
  - 47: the Flashcards tab shows a box containing only "Decks:".
  - Generation needs server mode: `UI/Screens/study_screen.py:620-625` ("Source generation requires server mode.").
  - The counts disagree: Library says "134 more", Study says "+7 more" (`STUDY_MATERIAL_TITLES_LIMIT = 10`).
- **Cause (hypothesis):** the `Horizontal()` and `Vertical()` containers default to `height: 1fr` with hidden overflow. See `Widgets/Study/study_dashboard.py:73` (the columns row has no `height: auto`) and `UI/Study_Window.py:117-121` (`.card-editor` has no `height: auto`).
- **Fix:**
  - Set `height: auto` on those containers.
  - In local mode, the Library Study canvas should say "Generating from sources needs a server; you can still make cards by hand" instead of "Source snapshot is ready".
  - Add "Make flashcards from this note / paper" on the note editor and the media reader, scoped to that item and using the configured local provider.

### F7 (P1) — The Notes list cannot be scrolled, so notes below the fold are unreachable except through Filter
- **Who it hurts:** anyone browsing more than about a screen of notes, at both tested sizes.
- **Evidence:**
  - Fresh launch `nl-j7-2` at 160x45: wheel over the list (cols 60 and 80, rows 40–42) changes nothing (61→62 identical). Unfiled shows 7 of 24.
  - 60: ↓ moves focus off-screen without scrolling.
  - 120x36: no Unfiled note is visible, and wheel, `g`+↓ and Tab all fail to reveal any (58, 59).
- **Cause:** untraced. `#library-notes-list` is a plain `Vertical` (`Widgets/Library/library_notes_canvas.py:2345`) with no scroll container in the canvas.
- **Fix:**
  - Put the tree (or the whole list column below the toolbar) in a `VerticalScroll`.
  - Call `scroll_visible()` on the focused row.
  - Add a visible "▼ more — scroll" hint like the rail's.

### F8 (P2) — Search/RAG silently keeps the previous query's source toggles
- **Who it hurts:** step 6 users, who conclude that nothing else matches.
- **Evidence:**
  - 48: rail "Search Library…" for "retrieval practice" returned "2 results" (Media only).
  - The scope line "Scope: Media (Notes, Conversations, Prompts off)" sat above the viewport, while the view scrolled to the toggles and results.
  - 49–50: after re-enabling Notes and Conversations there were 5 results.
- **Fix:**
  - Put the scope in the results header ("2 results · Media only — Search all sources") with a one-click reset.
  - Reset to all sources when a search starts from the rail box.

### F9 (P2) — The note editor's header row overflows at 160x45: "Use in Console" shows as "Use in", the status prints one letter per row, and "Discard new note" is off-screen
- **Who it hurts:** everyone writing a note in the wide layout.
- **Evidence:**
  - 13/14/19: ANSI rowstyles show a single-column Static at col 77 printing "E / n / —" for "Empty note…", "U / c" for "Unsaved changes" and "S" for "Saved".
  - The last button reads " Use in " at cols 151–158.
  - The new-note "Discard new note" button is not on screen.
  - Clicking "Use in" navigated to Console without warning (16).
- **Code:** `Widgets/Library/library_notes_canvas.py:2819-2870` places `#library-note-status` in `#library-note-header-second-row` next to `#library-note-task-actions`, which has `min-width: 61` (`css/screen_agentic_library.tcss:1189`). The status only gets its own row in compact mode (`:1380`).
- **Fix:**
  - Apply the compact rule (status on its own row) whenever the work pane is narrower than about 100 cols. Or drop this Static, since the authority line above already shows the status.
  - Never truncate "Use in Console".
  - Keep "Discard new note" visible for a new note.

### F10 (P2) — Console contradicts itself about what is staged after a send, and the follow-up's retrieval failure appears only in the log
- **Who it hurts:** researchers who ask follow-ups and believe they are still grounded.
- **Evidence:**
  - 65/66/67: after sending, the strip says "Staged for next send · 1 source — paper-retrieval-practice — media".
  - The Inspect rail says both "Sources: None staged" and "Sources — next send 1".
  - The status bar says "Sources: 0".
  - The follow-up request carried the paper only through history (mock log `n_msgs=4`; the reply echo lacks "Evidence: [S1]").
  - The log shows `[ERROR] Console RAG capture unavailable; reason=capture_provider_failure; draft_length=36`. The exception is swallowed with no detail (`Chat/console_chat_controller.py`, the `except Exception` around the RAG `provider(...)` call), and the transcript showed nothing.
- **Fix:**
  - Clear the "Staged for next send" strip once a send consumes it, and show "In this conversation: paper-retrieval-practice" instead.
  - When retrieval fails, add an inline turn notice ("Library retrieval failed — this answer used conversation history only · Retry with sources").
  - Log the exception type.
- **Also:** the evidence handoff left the stale composer text "Use this note as context…" from an earlier note handoff. Replace the composer prompt when the staged source changes.

### F11 (P2) — Import ignores PDF Title and Author metadata
- **Who it hurts:** researchers citing from the Library, because every paper shows up under its filename.
- **Evidence:**
  - 05/06/55: the title is "paper-retrieval-practice" and the byline is a 4-row `file://` path, though the PDF metadata has Title "Retrieval practice improves long-term retention" and an Author.
- **Code:** `Local_Ingestion/local_file_ingestion.py:1066-1067` sets `title = file_path.stem` before processing. `Local_Ingestion/PDF_Processing_Lib.py:677` would prefer `raw_metadata["title"]` only when no override is passed (cause inferred from the order of these lines).
- **Fix:**
  - Leave the title unset until the processor returns, then prefer the PDF Title, falling back to the first heading and then the filename.
  - Fill Author from metadata.
  - Show the file path in Info only, not as the byline.

### F12 (P2) — RAG Answer uses a different model from the user's chat model, and names it only after the paid call
- **Who it hurts:** budget-conscious users who chose a cheap model in Console.
- **Evidence:** 30 shows "To openai: question + evidence" before Run. 32 shows "openai · gpt-5.6-terra · $0.0006 (90 tok)" after. Console's status reads "Model: gpt-4.1-mini".
- **Code:** `Library/library_rag_answer_service.py:235-252` resolves its own provider and model (the config default `gpt-5.6-terra`, `config.py:2753`).
- **Fix:** name the model in the pre-run line ("To openai · gpt-5.6-terra: question + evidence") and offer "Use Console model (gpt-4.1-mini)".

### F13 (P3) — The first import silently downloads an embedding model, and a third-party warning is painted over the nav bar
- **Who it hurts:** local-first users who expect to be told about network egress, and anyone watching the screen during the first import.
- **Evidence:**
  - 06: the nav row was overwritten by `2026-10-02 20:11:01 [WARNING ] huggingface_hub.utils._http:904 - Warning: You are sending unauthenticated requests to the HF Hub…`.
  - The log shows `_build: Loaded model default in 15.74s` after the import, from the RAG auto-index.
- **Cause (hypothesis):** a WARNING-level log sink still writes to the terminal after the TUI starts. Candidates are `Utils/startup_logging.py:53` and `tldw_chatbook/__init__.py:97`.
- **Fix:**
  - Remove terminal sinks once the App is running, so third-party warnings go to the log file only.
  - Show "Preparing search index — one-time model download (~90 MB) from Hugging Face" on the import queue row, with a setting to defer it.

## Improvement opportunities (beyond defects)

1. **Reading desk mode.** Media reader and note editor side by side, with the note bound to the media item. The note header shows "About: paper-retrieval-practice", and media Info lists the bound notes. This turns a 13-input cycle into zero switches.
2. **Citation-grade quotes.** "Quote to note" inserts a block quote plus a `media://id#chunk` link. Preview renders it as a chip that opens the reader at the passage. Highlights become the staging area ("Send 3 highlights to note").
3. **Per-item scope everywhere.** A "This item only" chip in Search/RAG that the reader can set ("Ask about this paper"), so "scoped to that paper" is literal, not "Media only".
4. **Provenance-preserving export for advisors.** Offer "Export as Markdown folder / single combined document (Markdown or PDF)" with front matter `sources:` and a generated reference list, alongside the Chatbook zip, which is only useful to another Chatbook user.
5. **Study from here.** "Make flashcards / quiz from this note" in the note editor's Info tab or task actions, with a local provider path. Pre-fill the Study deck name with the note title, and carry provenance into each card's back side.

## Nielsen scores (0 = fails the heuristic, 4 = fully meets it)

| Heuristic | Library shell (Media, Import, Search/RAG, Export, Study handoff) | Notes |
|---|---|---|
| 1 Visibility of system status | **2**: import status is excellent, but the row says "loaded" while the reader is blank, and Study says "Source snapshot is ready" when nothing can be generated | **2**: autosave "Saved 20:12" is clear, but the squeezed status prints "E/n/—" one letter per row, and the silent "Linked to Local Default" only shows up after the fact |
| 2 Match with the real world | **2**: "Canonical ID", "Stored representation", "Carries forward / Source snapshot", "workspace-default" | **2**: "placement", and provenance keywords shown as raw UUIDs |
| 3 User control and freedom | **2**: switching rows discards reader state; Study Esc lands on "Study decks", not where she came from | **2**: Esc discards an empty new note (good), but notes close on every switch and the list cannot be scrolled |
| 4 Consistency and standards | **1**: media `Use in Console` is refused while evidence handoff and notes succeed; RAG uses a different model; three different "Use in Console" behaviours | **2**: the toolbar "New" opens a chooser while `n` creates a note; the "Use in" label is clipped; single-note and bulk exports differ in keyword handling |
| 5 Error prevention | **2**: stale source toggles narrow searches silently; there is no pre-run model disclosure | **2**: export writes invalid YAML and drops keywords with no warning |
| 6 Recognition rather than recall | **2**: `Ctrl+C` in the reader is invisible; "Export selected" only exists as the `e` chip | **2**: no link insertion and no visible ids; Highlights cannot be reused |
| 7 Flexibility and efficiency | **2**: good single-surface keys (`[ ] c l t`, `/`), but the cross-surface cycle costs about 8 clicks | **1**: no side-by-side, no clip-to-note, ↓ does not walk folder rows, no keyboard save |
| 8 Aesthetic and minimalist design | **3**: dense and calm; the 4–5 row `file://` byline wastes the narrow reader | **3**: a clean editor, but the always-visible purpose paragraph costs 4 rows of list |
| 9 Error recognition and recovery | **2**: the media handoff toast names an internal id with no fix control; the RAG "does not cite" callout is a strong positive | **2**: the "tagged with this conversation" claim cannot be acted on; the filter returns 0 for the tag |
| 10 Help and documentation | **2**: F1 help and footers are accurate per surface, but nothing guides a read-and-note workflow | **2**: the footer chips are accurate, but there is no guidance on linking sources |

## Harness caveats

- The mock LLM's replies are nonsense ("The wind shifts, a lantern flickers…"). Answer quality, citation correctness and RAG validation verdicts cannot be judged. Only the plumbing (what was sent, what was kept) is evidence.
- Every fresh run has an empty Hugging Face cache, so the embedding-model download in F13 happens on every harness launch. A real user sees it once per profile. The stderr paint-over is still real.
- Null keyring backend. No workspaces were seeded; "Local Default" is the app's own default. No study decks or sync roots were seeded.
- PNGs are approximate (no box borders). Every claim above rests on the `.txt` and `.ansi` captures.
- Three of my own mis-clicks were recovered, and none is reported as a finding:
  - a `Read later` toggle hit while targeting the Read tab;
  - the first `Use in` click (it still exposed F9);
  - a Study header click that I mistook for the Flashcards tab.
- The F7 re-check used a second fresh socket, `nl-j7-2`, so the result does not depend on my session's state.
- Both sockets were killed at the end. `pgrep -fl runs/nl-j7-*` is empty, the real-profile snapshot is unchanged, the run log has no real-profile paths, and no tracked files were touched.
