# Notes + Library UX review — master findings (2026-10-02)

Target: origin/dev `2d34cbf80d`, worktree `.worktrees/notes-library-ux-review`. Synthesis of 7 live persona journeys (Assessment A: j1–j7) and 2 detector/static passes (Assessment B: B-mechanical, B-static). 134 raw findings were consolidated into 103 deduplicated findings.

Severity: P0 = blocks task completion, data loss, crash, or a lie about saved/synced state; P1 = significant difficulty or confusion; P2 = annoyance with a workaround; P3 = polish. Where sources disagreed, the chosen level and the reason are given in the entry's evidence and in the contradictions section.

**Counts:** P0 5, P1 36, P2 50, P3 12 · Notes 37, Library 38, Shell/cross-destination 28.

**Sources cited:** j3-poweruser-notes (24), j6-stress (22), j5-a11y-consistency (20), j4-poweruser-library (20), j7-researcher-loop (17), j2-firsttimer-library (17), j1-firsttimer-notes (15), B-mechanical (12), B-static (8)

## Summary table

| ID | Sev | Surface | Title | Sources | Live |
|---|---|---|---|---|---|
| [N-01](#n-01) | P0 | Notes | Ctrl+Q quits with no flush or prompt: unsaved note text is lost (and a new note becomes an empty 'Untitled' orphan) | j1-firsttimer-notes, j3-poweruser-notes, j6-stress, B-static | yes |
| [N-02](#n-02) | P0 | Notes-sync/import/folder-files | One ordinary edit to a synced note can wedge lasting sync permanently (postcondition_failed), while Notes keeps saying 'Saved' / 'Sync managed · Ready' | j3-poweruser-notes, j6-stress | yes |
| [N-03](#n-03) | P0 | Notes-sync/import/folder-files | Deleting a synced note leaves the root at '✓ Up to date' although the file still exists | j3-poweruser-notes | yes |
| [N-04](#n-04) | P1 | Notes | At ≥120 columns the Notes tree cannot scroll: notes, 'Load more notes' and 'Recently deleted' below the fold are unreachable, and focus walks onto unseen rows | j1-firsttimer-notes, j3-poweruser-notes, j5-a11y-consistency, j6-stress, j7-researcher-loop, B-mechanical | yes |
| [N-05](#n-05) | P1 | Notes | Note editor header overflows between 120 and ~200 columns: Save and 'Use in Console' clipped or missing, save state crushed to one column, Tab/F6 land on hidden Save | j1-firsttimer-notes, j3-poweruser-notes, j5-a11y-consistency, j6-stress, j7-researcher-loop, B-mechanical | yes |
| [N-06](#n-06) | P1 | Notes | Opening any modal (F1 help, Move note / Add to folder) cancels the pending autosave and never re-arms it, while the status keeps promising automatic save | j1-firsttimer-notes, j6-stress | yes |
| [N-07](#n-07) | P1 | Notes | An autosave validation veto (trailing space in title, duplicate keyword) yanks focus mid-typing, so the next words land in the wrong field | j1-firsttimer-notes, j6-stress | yes |
| [N-08](#n-08) | P1 | Notes | Typed [[wikilinks]] are dead text, and Preview hands note:// links to the OS URL handler | j1-firsttimer-notes, j3-poweruser-notes | yes |
| [N-09](#n-09) | P1 | Notes | Notes filter matches only exact whole-word phrases in title/body: no prefix, no reordered words, keywords not searchable | j1-firsttimer-notes, j3-poweruser-notes, j7-researcher-loop | yes |
| [N-10](#n-10) | P1 | Notes | Inline delete confirmation renders off-screen; Cancel/Delete are pressed blind | j1-firsttimer-notes, j5-a11y-consistency | yes |
| [N-11](#n-11) | P1 | Export | Note and prompt Export silently overwrite an existing file, starting at ~ with a title-derived name | j6-stress, B-static | yes |
| [N-12](#n-12) | P1 | Notes | Cancelling a note export is reported as 'Export failed' and focus lands on Delete | j5-a11y-consistency | yes |
| [N-13](#n-13) | P1 | Notes | Arriving via the palette 'New Note' shows the Notes tree with blank rows instead of its folders | j3-poweruser-notes | yes |
| [N-14](#n-14) | P1 | Notes | Notes select mode is export-only, 'Select all N shown' miscounts, and selecting costs two keys per row | j3-poweruser-notes, j5-a11y-consistency | yes |
| [N-15](#n-15) | P1 | Notes-sync/import/folder-files | Sync review rows for files deleted or renamed (on disk or in-app) can never be resolved, and 'Apply reviewed' is disabled with no visible reason | j3-poweruser-notes, j6-stress | yes |
| [N-16](#n-16) | P1 | Notes-sync/import/folder-files | One non-UTF-8 or mixed-newline file blocks the whole sync folder with 'Check failed — NotesSyncRootRefused' | j6-stress | yes |
| [N-17](#n-17) | P1 | Notes-sync/import/folder-files | 'Add from files…' gets stuck on a stale lasting-sync phase: an old receipt after activation, or a blank page after a failed Recovery | j3-poweruser-notes, j6-stress | yes |
| [N-18](#n-18) | P1 | Notes-sync/import/folder-files | Every synced folder is titled 'Sync folder (name unavailable before cutover)'; roots and receipts cannot be told apart | j3-poweruser-notes, j6-stress, B-static | yes |
| [N-19](#n-19) | P1 | Notes-sync/import/folder-files | Sync Receipts says 'No writes yet.' when the write history could not be read | B-static | code-only |
| [N-20](#n-20) | P2 | Export | Exported note front matter is invalid YAML for ordinary titles (':' or '*') | j7-researcher-loop | yes |
| [N-21](#n-21) | P2 | Notes | No fast path from a query to a note, and no note history | j3-poweruser-notes | yes |
| [N-22](#n-22) | P2 | Notes | Arrow keys skip folder rows and there is no expand/collapse key in the Notes tree | j3-poweruser-notes, B-mechanical, j5-a11y-consistency | yes |
| [N-23](#n-23) | P2 | Notes | Folder pickers (Move note, Move folder, Add to folder) list only folders currently expanded in the tree | j3-poweruser-notes, j6-stress | yes |
| [N-24](#n-24) | P2 | Notes | 'Move note' fails with a false 'That folder changed elsewhere — refresh and try aga…' for Unfiled or just-deleted notes, and the notice sticks | j3-poweruser-notes, j6-stress | yes |
| [N-25](#n-25) | P2 | Notes | Leave-veto toast names a 'Discard new note' button that is not there and blames the title for any veto | j6-stress | yes |
| [N-26](#n-26) | P2 | Notes | Info 'Linked from' never resolves on a new note, or shows the previous note's backlinks | j1-firsttimer-notes | yes |
| [N-27](#n-27) | P2 | Notes-sync/import/folder-files | 'View 10 imported notes' lands on the list with the import folder collapsed | j1-firsttimer-notes | yes |
| [N-28](#n-28) | P2 | Notes | Moving a folder into a collapsed folder hides the target's own notes until collapse/re-expand | j6-stress | yes |
| [N-29](#n-29) | P2 | Notes | Notes 'Recently deleted' cannot delete anything permanently | j6-stress | yes |
| [N-30](#n-30) | P2 | Notes | Notes sort chooser opens with focus left in the Filter field | j5-a11y-consistency | yes |
| [N-31](#n-31) | P2 | Notes | Notes text fields never switch the footer to 'typing in field', so bare-letter chips stay advertised and collide | j5-a11y-consistency, B-mechanical | yes |
| [N-32](#n-32) | P2 | Notes | F6 in the note body selects the current line instead of moving to the next pane | B-mechanical | yes |
| [N-33](#n-33) | P2 | Notes | Ctrl+S saves only Skills; the Notes save action exists but is unbound, and Prompts/Folder files have no save key | B-static | yes |
| [N-34](#n-34) | P2 | Notes-sync/import/folder-files | Escape in Add-from-files cancels a running import (Back keeps it running) and silently discards a sync review's staged choices | B-static | code-only |
| [N-35](#n-35) | P2 | Notes-sync/import/folder-files | Sync and comparison copy shows raw nanosecond timestamps, ISO times, exception class names and stale lines | j3-poweruser-notes | yes |
| [N-36](#n-36) | P2 | Notes-sync/import/folder-files | Folder files' Session Git shows status but never a diff | j3-poweruser-notes | yes |
| [N-37](#n-37) | P2 | Notes-sync/import/folder-files | Folder files with no folder linked is 61–78% blank, the only next step is an inline link, and the Library rail drops below the header | B-mechanical | yes |
| [L-01](#l-01) | P0 | Export | Media list 'Export…' freezes the whole app (no input, no repaint, Ctrl+Q dead) | j2-firsttimer-library | yes |
| [L-02](#l-02) | P0 | Media | Library analysis always says 'No analysis provider is configured', even when [analysis_defaults] names a ready provider | j4-poweruser-library | yes |
| [L-03](#l-03) | P1 | Media | Opening an item by link (import 'Open in Library', Search 'Open') leaves the list cursor on row 1, so reader actions - and sometimes the reader itself - target the wrong item | j2-firsttimer-library | yes |
| [L-04](#l-04) | P1 | Export | Export bundles silently drop keywords for media, conversations and notes | j4-poweruser-library, j7-researcher-loop | yes |
| [L-05](#l-05) | P1 | Ingest | Import rejects '~/' paths as a 'dangerous pattern', then says 'Can't find that path' for a file that exists | j2-firsttimer-library | yes |
| [L-06](#l-06) | P1 | Search-RAG | RAG Answer calls the provider's default model instead of the configured one, and names it only after the paid call | j2-firsttimer-library, j4-poweruser-library, j7-researcher-loop | yes |
| [L-07](#l-07) | P1 | Media | 'Add analysis' leaves focus on the Items row, so typing fires single-letter list commands | j4-poweruser-library | yes |
| [L-08](#l-08) | P1 | Media | Escape in the analysis editor discards typed text without asking | j4-poweruser-library | yes |
| [L-09](#l-09) | P1 | Conversations | Conversations filter leaves the Reader on a conversation that is no longer listed; 'c'/Resume act on it | j4-poweruser-library | yes |
| [L-10](#l-10) | P1 | Artifacts | 'Try report demo' creates a persistent live-RSS watchlist with a daily paid run, disclosed only afterwards | j4-poweruser-library | yes |
| [L-11](#l-11) | P1 | Skills | Built-in skill 'Enabled' switch shows no state; the label still reads 'Enabled' when off | j4-poweruser-library | yes |
| [L-12](#l-12) | P1 | Library-shell | At 60–80 columns Media, Skills and Conversations open to an empty Reader with the list collapsed | B-mechanical | yes |
| [L-13](#l-13) | P2 | Search-RAG | Search mode returns 'No evidence matched' for a plain-language question the documents answer verbatim; the rail box silently flips the mode | j2-firsttimer-library | yes |
| [L-14](#l-14) | P2 | Media | In-document Find counts lines, not matches, and marks nothing in the default Rendered view | j2-firsttimer-library | yes |
| [L-15](#l-15) | P2 | Media | Reader Find needs Enter and keeps showing the previous query's count | j4-poweruser-library | yes |
| [L-16](#l-16) | P2 | Media | Returning to Media via the rail shows a row marked 'loaded' beside an empty Reader | j4-poweruser-library, j7-researcher-loop | yes |
| [L-17](#l-17) | P2 | Search-RAG | 'Select evidence' moves focus onto a Sources checkbox and can jump the panel back to the top | j2-firsttimer-library | yes |
| [L-18](#l-18) | P2 | Export | Export canvas promises 'copies full media files' but writes text only, with opaque media_N.txt names; a second same-day export overwrites the first by default | j2-firsttimer-library, j4-poweruser-library | yes |
| [L-19](#l-19) | P2 | Media | Media selection is page-local and cleared by any page turn; there is no bulk keyword action | j4-poweruser-library | yes |
| [L-20](#l-20) | P2 | Ingest | Imports are titled by filename stem, ignoring PDF Title/Author, HTML <title> and Markdown H1 | j2-firsttimer-library, j7-researcher-loop | yes |
| [L-21](#l-21) | P2 | Ingest | Local HTML import keeps site navigation, duplicates the heading and flattens tables | j2-firsttimer-library | yes |
| [L-22](#l-22) | P2 | Media | Reader 'More ▸ Open original' and 'Open manager' do nothing for a local import | j2-firsttimer-library, j3-poweruser-notes | yes |
| [L-23](#l-23) | P2 | Search-RAG | Search/RAG Sources toggles take ~3 rows each and push the answer or evidence below the fold at 100x30 and 120x36 | j2-firsttimer-library, j4-poweruser-library | yes |
| [L-24](#l-24) | P2 | Search-RAG | Search/RAG silently keeps the previous query's source toggles | j7-researcher-loop | yes |
| [L-25](#l-25) | P2 | Media | The Media reader is the narrowest pane, and an absolute file:// path takes its first 4–5 lines | j2-firsttimer-library, j7-researcher-loop | yes |
| [L-26](#l-26) | P2 | Prompts | Prompt 'Use in Console' drops Instructions by default and passes {{notes}} through as literal text | j4-poweruser-library | yes |
| [L-27](#l-27) | P2 | Search-RAG | Recent searches replays an entry under the current mode, silently making a paid RAG call | j4-poweruser-library | yes |
| [L-28](#l-28) | P2 | Skills | Skill 'Customize' hides the built-in and leaves a copy the agent cannot use, without saying so first | j4-poweruser-library | yes |
| [L-29](#l-29) | P2 | Media | Selecting an item in Media Trash leaves an unrelated live item in the reader | j6-stress | yes |
| [L-30](#l-30) | P2 | Ingest | Batch media import has no cancel, and Enter imports while the footer says 'check this path' | j6-stress | yes |
| [L-31](#l-31) | P2 | Conversations | Conversations reader stacks 16–20 rows of chrome above the first message | B-mechanical | yes |
| [L-32](#l-32) | P2 | Prompts | Prompts list gets half the pane; the other half is blank | B-mechanical, j5-a11y-consistency | yes |
| [L-33](#l-33) | P3 | Ingest | First import silently downloads an embedding model and paints a Hugging Face warning over the nav bar | j7-researcher-loop, j1-firsttimer-notes | yes |
| [L-34](#l-34) | P3 | Collections | 'Keep for later' is split across three unconnected places, and the Quick Capture note box is unlabeled | j2-firsttimer-library | yes |
| [L-35](#l-35) | P3 | Ingest | A fully successful import still offers 'Retry this batch' and footer 'r retry' | j2-firsttimer-library | yes |
| [L-36](#l-36) | P3 | Library-shell | Help and empty states never say what Library, Prompts or Skills are | j2-firsttimer-library | yes |
| [L-37](#l-37) | P3 | Media | Restore and archive reset 'updated' to now, reordering Newest and the landing | j4-poweruser-library | yes |
| [L-38](#l-38) | P3 | Media | Media scope-line 'Clear' also resets the sort | j4-poweruser-library | yes |
| [S-01](#s-01) | P1 | Cross-destination | Media 'Use in Console' is always refused for imported items ('Copy or link this media into workspace workspace-default…') with no in-app remedy, while Search evidence stages the same item | j2-firsttimer-library, j4-poweruser-library, j7-researcher-loop | yes |
| [S-02](#s-02) | P1 | Cross-destination | Reader and note can never be visible together; every rail switch discards the reading position and the open note, and Media has no capture key | j7-researcher-loop, j3-poweruser-notes | yes |
| [S-03](#s-03) | P1 | Cross-destination | No way to quote a passage into a note with a link back to the source | j7-researcher-loop | yes |
| [S-04](#s-04) | P1 | Cross-destination | Kept AI answers lose provenance (no question, no source, UUID keywords), and Library RAG answers cannot be saved | j7-researcher-loop | yes |
| [S-05](#s-05) | P1 | Cross-destination | Study hand-off dead-ends: a whole-Library 'ready' snapshot, invisible Study dashboard controls, and server-only generation | j7-researcher-loop | yes |
| [S-06](#s-06) | P1 | Cross-destination | A second 'Use in Console' silently replaces the first staged note | j3-poweruser-notes | yes |
| [S-07](#s-07) | P1 | Library-shell | Closing F1 or any modal drops keyboard focus on the landing, Notes, Conversations and Skills | j5-a11y-consistency | yes |
| [S-08](#s-08) | P1 | Library-shell | Single-letter shortcuts fire while typing in the Notes filter during its re-render; 'i' opens Import media | j6-stress | yes |
| [S-09](#s-09) | P1 | Library-shell | F6 never enters the Export or Search/RAG canvas; reaching 'Choose destination…' takes ~29 Tabs | j4-poweruser-library, j5-a11y-consistency | yes |
| [S-10](#s-10) | P1 | Library-shell | Keyboard input stopped entirely at 80x24 (Tab/F6/F1/Ctrl+P) until a mouse click - intermittent | j5-a11y-consistency | yes |
| [S-11](#s-11) | P2 | Library-shell | Escape means different things per destination; advertised 'esc focus rail' does nothing when the rail is collapsed | j5-a11y-consistency, B-mechanical | yes |
| [S-12](#s-12) | P2 | Cross-destination | Active state is invisible or misleading on mode and segment controls (Edit/Preview/Info, Library notes\|Folder files, Read/Info, Skills tabs) | j5-a11y-consistency | yes |
| [S-13](#s-13) | P2 | Library-shell | Arrival focus and Tab order do not follow the screen: palette arrival focuses '⌃1 Home', list entry focuses the Nav grip, and the source strip is the last Tab stop | j5-a11y-consistency | yes |
| [S-14](#s-14) | P2 | Cross-destination | Six different focus-indicator styles, several below 2:1 or a single cell | j5-a11y-consistency | yes |
| [S-15](#s-15) | P2 | Library-shell | Footer and F1 disagree, F1 is a bare key list with a malformed row and no 'how it works', and Search/RAG advertises an inert 'o open evidence' | j1-firsttimer-notes, B-mechanical, j5-a11y-consistency | yes |
| [S-16](#s-16) | P2 | Cross-destination | Labels are cut mid-word with no ellipsis across Library, hiding whole reader modes at 80 columns | B-mechanical, j5-a11y-consistency | yes |
| [S-17](#s-17) | P2 | Cross-destination | A vetoed nav-bar switch is silent and leaves the nav bar highlighting the wrong destination | j6-stress | yes |
| [S-18](#s-18) | P2 | Cross-destination | A deleted note stays staged in Console as 'Ready' and the turn sends anyway | j6-stress | yes |
| [S-19](#s-19) | P2 | Cross-destination | After a send, Console contradicts itself about staged sources, and a follow-up's retrieval failure appears only in the log | j7-researcher-loop | yes |
| [S-20](#s-20) | P2 | Cross-destination | Console 'Search Library' modal fails a first-timer: typing lost, 'staged' shown before results exist, a jargon send-block, and results that re-stage after Un-stage | j1-firsttimer-notes | yes |
| [S-21](#s-21) | P2 | Library-shell | The empty landing's only 'Import…' goes to Media import, with no pointer to importing notes | j1-firsttimer-notes | yes |
| [S-22](#s-22) | P2 | Library-shell | A relaunch drops the working context (staged sources, open note, filter, expanded folders) | j3-poweruser-notes | yes |
| [S-23](#s-23) | P3 | Library-shell | Unexplained chrome: ASCII '--->'/'<---' grips with vertical letter labels, an unexplained 'Agent_Lessons' system folder, and literal '[ ]' task boxes in Preview | j1-firsttimer-notes | yes |
| [S-24](#s-24) | P3 | Cross-destination | Internal vocabulary and raw exception text on screen ('rail', 'placement', 'owner review', 'lane', 'cutover', 'authority', 'PermissionError', 'Open failed: {error}') | B-static, j6-stress | yes |
| [S-25](#s-25) | P3 | Cross-destination | Layout and copy polish: wasted list space, toast covering its own buttons, stale 'Next' guidance and receipts, 'hub' jargon | j5-a11y-consistency | yes |
| [S-26](#s-26) | P3 | Library-shell | User Guide keyboard and label claims disagree with the live app (8 of 12 checked) | j5-a11y-consistency | yes |
| [S-27](#s-27) | P3 | Library-shell | After visiting Notes, Media's Items grip is painted 'Notes' | j3-poweruser-notes | yes |
| [S-28](#s-28) | P3 | Library-shell | Select-mode and hand-off keys differ between Media and Notes (s/Space vs button/Enter; e only in Notes; c only in Media/Conversations) | B-static, j3-poweruser-notes | yes |

## Contradictions and harness caveats

- A-vs-A severity, N-04 (Notes tree cannot scroll): j3 rated P0, j1/j5/j6/j7 and B rated P1. Chose P1: Filter plus the filtered 'Recently deleted' link is a (poor) workaround, so it does not fully block completion.
- A-vs-A cause, N-04: j7 says the cause is untraced. j6 cites the generated bundle css/screen_agentic_library.tcss:1242-1249. j1, j3, j5 and B cite the source css/features/_library.tcss:1004-1013. These agree: the bundle is generated from that source. The synthesis confirmed that the wide-layout rule (_library_panels.tcss:543-548, bundle :2034-2039) sets widths only, although its comment calls the list 'its scroll owner'.
- A-vs-A and A-vs-B severity, N-05 (editor header overflow): j3 and j6 rated P3, j7 P2, j1, j5 and B P1. Chose P1 because at 120 cols explicit Save and Use in Console are unreachable (j1's task 9 failed).
- A-vs-B, N-05/N-32 (F6 in the note editor): j5 reports that F6 lands on the hidden Save (library_screen.py:1484-1494 lists library-note-save first). B reports that F6 pressed in the note body never leaves the TextArea and selects the line instead (4/4). Both can be true: j5 likely pressed F6 from a non-TextArea field, while B pressed it from the body. Left unresolved; each is kept with its own finding.
- A-vs-A severity, N-06 (modal cancels autosave): j1 rated P0 as data loss, j6 rated P1. j1's own control capture 55 shows Ctrl+Q alone loses the text, so the data loss belongs to N-01. N-06 is P1, as a standing false 'save automatically' promise.
- A-vs-A trigger, N-02 (sync wedge): j3 reached postcondition_failed with a single in-app edit that has no trailing newline. j6 reached it with a both-sides edit. Neither contradicts the other: the same failing postcondition (notes_sync_executor.py:5384) is hit by two different paths. j6's log shows 'reason=unclassified error_type=RuntimeError'.
- A-vs-guide, N-03: j3 notes that the User Guide (notes.md 'What this covers, exactly') documents that deleting a note does not notify sync. The behaviour is documented, but the '✓ Up to date' label is still a false sync state, so it stays P0.
- A-vs-A-vs-B severity, N-18 (sync roots titled 'name unavailable before cutover'): j6 rated P3, j3 P2 and B-static P1. Chose P1 because with two roots no name or path tells them apart, and Pause, Resume and Recovery can then target the wrong folder.
- A-vs-A severity, N-17 (Add from files stuck): j3 rated P2 at medium confidence for the stale receipt, and j6 rated P1 for the blank page. Chose P1 because j6's variant blocks Import once for the whole session.
- A-vs-A severity, S-01 (media Use in Console refused): j2 and j7 rated P1, j4 P2 at medium confidence. Chose P1. j4 also saw that 'Use as source' on a conversation links that conversation to Default, while j2 found zero memberships on a fresh profile. Both are consistent with the code: only conversations, notes and chat link memberships.
- A-vs-A severity, L-06 (RAG Answer model): j2 and j7 rated P2, j4 P1. Chose P1 because PRODUCT.md requires cost and authority to be visible before action. j7 cites library_rag_answer_service.py:235-252 while j2 and j4 cite :163-189. These are compatible: both describe the provider/model resolution that returns no model.
- A-vs-A, L-14 vs L-15 (reader Find): j2 says the default Rendered view highlights nothing ('Match 1 of 2', every row plain). j4 says that after Enter 'Match 1 of 6' showed 'with highlights'. j4 did not record which Read view was active, so j4 may have been in Raw. Left unresolved, and the two findings are kept separate because their causes differ (j2: line-based counting and no Rendered marking; j4: the status only updates on Enter).
- A-vs-A, L-01 (Export freezes): j2 froze the app 3 times out of 3 from the Media Items-toolbar 'Export…' button, but j4 and j7 reached the Export canvas through select mode ('s'/'e' > Export) without a freeze. These are different entry paths, not a contradiction: the freeze is specific to the list-toolbar handler.
- A-vs-A severity downgrades: j3 rated N-21 (no fast path to a note) P1 and j4 rated L-19 (page-local selection) P1. Both were normalized to P2 because a slower workaround exists. j4 rated L-23 (Search/RAG evidence below the fold) P3 and j2 rated it P2; P2 was chosen.
- A-vs-B scope, N-01 (Ctrl+Q): B-static extends the loss to dirty prompt and skill drafts and to Folder files autosave. Only the note path was reproduced live (by j1, j3, j6 and B). The prompt, skill and Folder-files parts are code-traced only.
- B-only, not live-verified: N-19 (Receipts 'No writes yet' on read failure) and N-34 (Escape cancels an import or abandons a review) are code-traced with suggested repro steps. They are marked live_reproducible=false and kept at B's severity rather than raised, because the false state was never observed.
- Harness caveat (partially dropped): L-36 / j2's Skills banner 'Skill trust isn't set up…' and L-28 / j4's 'Trust: not initialized' / 'needs review' are amplified by the harness's null keyring backend. The trust-state copy itself is not reported as a defect. Only the product behaviour around it is kept: the banner shows with zero user skills, and Customize hides the built-in without warning.
- Harness caveat (partially dropped): in S-05 (Study hand-off), j7's empty Flashcards 'Decks:' box reflects that no study decks were seeded. It is not counted as a defect. The clipped dashboard controls and the misleading 'Source snapshot is ready' are kept.
- Harness caveat: the reduced-protection skill trust marker seen in Skills (j2, j4) comes from the null keyring and is not reported.

## Notes (incl. sync, import, folder files)

<a id="n-01"></a>
### N-01 · P0 · Ctrl+Q quits with no flush or prompt: unsaved note text is lost (and a new note becomes an empty 'Untitled' orphan)

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j3-poweruser-notes, j6-stress, B-static · **Personas:** Jordan, Alex, Riley
- **Heuristics:** H1, H3, H5
- **Evidence:** Reproduced live by four independent runs. j3: typed ' UNSAVED-TAIL-…' with the header reading 'Unsaved changes', C-q, process exited in <3 s, DB still version 1 without the tail. j6: 04-before-ctrlq-body shows 'Unsaved changes · Next: Keep editing; changes save automatically.'; after C-q the DB stays at v3; a new note titled 'Riley new ctrlq' is persisted as title 'Untitled', content '' (10-relaunch-orphan-untitled). B-static (nl-sb-1): 'ZZPROBE2' typed, Ctrl+Q, ro sqlite shows has1=1 has2=0. j1 control 55: typing then Ctrl+Q within 0.4 s loses the text with no modal involved. Code: quit flow calls only confirm_quit/prepare_for_quit on the active screen (app_lifecycle.py:1829-1923; Widgets/confirmation_dialog.py:174-238); LibraryScreen defines neither; flush_pending_work (UI/Screens/library_screen.py:11109-11141) is awaited only by navigation (app_navigation.py:513); on_unmount invalidates the pending save (library_screen.py:9738 -> library_notes_controller.py:3697-3702); autosave is a 2.0 s debounce re-armed on every keystroke, so continuous typing is never saved until a pause. B-static extends the same gap to dirty prompt/skill drafts and Folder files autosave (LibraryFileNotesWorkspace.shutdown stops its timer without saving, library_file_notes_workspace.py:2428-2455) - code-traced, not separately reproduced live.
- **Repro:** Golden, 160x45: Library > Notes > open 'Ideas inbox' > Ctrl+End > type a word > Ctrl+Q within 2 s > relaunch with REUSE=1 > reopen: the word is gone. New-note variant: Ctrl+N, type title, Tab Tab, body, Ctrl+Q -> an empty 'Untitled' row remains.
- **Why it matters:** Silent data loss on the universal quit key while the status line promises 'changes save automatically'. Every user who types a last line and quits inside the debounce window loses it; fast typists are never saved mid-burst.
- **Fix:** Add async LibraryScreen.prepare_for_quit() that awaits self.flush_pending_work() (notes session, prompt and skill editors, Folder files autosave). Add confirm_quit() that runs the same flush and, when it is vetoed or fails, shows 'Quit and discard unsaved changes to "<title>"? Stay · Discard' with Stay focused. Run the untouched-blank-note GC in the same hook. Make on_unmount flush before invalidating. Regression test: type into #library-note-body, call app.action_quit(), assert the DB row contains the text.

<a id="n-02"></a>
### N-02 · P0 · One ordinary edit to a synced note can wedge lasting sync permanently (postcondition_failed), while Notes keeps saying 'Saved' / 'Sync managed · Ready'

- **Surface / kind:** Notes-sync/import/folder-files · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j6-stress · **Personas:** Alex, Riley
- **Heuristics:** H1, H5, H9
- **Evidence:** Two different triggers, same failure. j3: an edit whose note content does not end with a newline (file ends 0A) -> notes_sync_operations 'update_file|needs_attention|postcondition_failed'; an edit ending in a newline completes (clean repro 55/57/58). j6: an app edit plus a different disk edit to 'Budget 2026' -> same row; editor shows 'Saved 20:33' and 'In a synced folder' while DB ends '| riley-app | 1 | app2' and the file ends '| riley-disk | 2 |' (57, 63). Afterwards Review says 'A recovery is still open for that folder. Resolve it, then Check again.' with '0 safe · 0 need attention'; Recovery says 'Recovery failed — RuntimeError. Next: Check changes'; the loop survives restart (j6 93) and later disk appends never arrive (j3). The tree reads '▸ Vault3  ⇄ Sync managed' and the list 'Library notes · Ready'. Code: serialize appends a final newline (Notes/notes_sync_filesystem.py:259-260) but the postcondition compares file.text == note.content (Notes/notes_sync_executor.py:5384, raises :5398-5400); runtime refuses checks while any operation is incomplete (notes_sync_runtime.py:2468-2484); exception class name used as copy (Library/library_notes_lasting_sync_state.py:1189-1204). Disconnect is 'unavailable — not in this release', so there is no exit.
- **Repro:** Keep a folder synced on a vault whose .md files end with a newline > Activate > open a synced note > Ctrl+End > type a word without Enter > wait 5 s > Manage sync folders > Review > Recovery. (Or: append different lines to the same note in the app and on disk.)
- **Why it matters:** Typing at the end of a note is the most normal edit there is. After it, sync silently stops in both directions for the whole folder, the only offered actions loop, and every surface outside Manage sync folders says all is well - users keep writing on both sides and diverge.
- **Fix:** Compare like with like in the postcondition (serialize(note.content, profile) == file.raw_bytes, or normalise the trailing newline on both sides). Treat any remaining postcondition_failed update_file as an ordinary conflict row with Keep file / Keep note / Keep both. Let Recovery accept the on-disk state when bytes match. Replace exception-name copy with a plain reason and a non-looping next action. While a binding has an incomplete operation, the editor location line must read 'Not synced — review in Manage sync folders' and the tree folder row must show attention. Ship Disconnect (or 'Pause and keep both') as an escape hatch.

<a id="n-03"></a>
### N-03 · P0 · Deleting a synced note leaves the root at '✓ Up to date' although the file still exists

- **Surface / kind:** Notes-sync/import/folder-files · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H1
- **Evidence:** j3 run nl-j3-3, captures 64-66: after Info > Delete on 'second', second.md is still on disk, yet Manage sync folders shows '✓ Up to date · Next: Check changes'. Only a manual Check changes reveals 'One side was deleted (1) · second.md' (which then dead-ends, see N-15). The User Guide (notes.md 'What this covers, exactly') documents that note deletion does not notify sync, so this is a known limitation - but the row still asserts a false synced state, which is P0 by definition.
- **Repro:** Activate a lasting root > delete one synced note in the app > Notes > Manage sync folders > read the row.
- **Why it matters:** 'Up to date' is the one status a sync user must be able to trust. The deletion is invisible until the user happens to run a check.
- **Fix:** Emit a sync intent when a note in a managed folder is deleted, restored or created so the row flips to '◌ Changes available'. Until that ships, never print '✓ Up to date' for a root whose managed notes changed since the last check - print 'Not checked since your last change · Check changes'.

<a id="n-04"></a>
### N-04 · P1 · At ≥120 columns the Notes tree cannot scroll: notes, 'Load more notes' and 'Recently deleted' below the fold are unreachable, and focus walks onto unseen rows

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j3-poweruser-notes, j5-a11y-consistency, j6-stress, j7-researcher-loop, B-mechanical · **Personas:** Jordan, Alex, Sam, Riley, Priya
- **Heuristics:** H1, H3, H7, WCAG 2.4.7, WCAG 2.4.11
- **Evidence:** Seen by six sources. B: 10 wheel events over the tree change nothing at 120x36, 160x45 and 235x52 but do scroll at 80x24 and 100x30 (probes/notes-scroll-*); at 120x36 0 of 122 notes are visible under 19 chrome rows. j1 (empty profile + import): 10 wheel events = 0-line ansidiff; Tab + 8×Down leaves no visible highlight and Enter opens the 9th, never-shown row 'Ideas'. j5: 11 Tabs with 0 changed cells; Enter opened hidden 'Daily log 2026-09-20'. j6: Down×12 then Enter opened 'Daily log 2026-09-21'. j3 at 200x50: Meetings, Projects, Research, Study, Unfiled and 'Load more notes' are past the border; filter 'Method' reports 20 results but the Vault3 match is never visible. j7: 7 of 24 Unfiled visible at 160x45. Cause (traced, verified in this synthesis): #library-notes-list is a Vertical (Widgets/Library/library_notes_canvas.py:2345; Textual default overflow hidden); only the compact rule sets overflow-y:auto (css/features/_library.tcss:1004-1013, compact below LIBRARY_NOTES_COMPACT_BREAKPOINT=120, UI/Library_Modules/screen_constants.py:275); the wide rule css/features/_library_panels.tcss:543-548 sets widths only, even though its own comment calls the list 'its scroll owner'. Severity: j3 rated P0; normalized to P1 because the Filter and the 'Recently deleted' link reached via a filter are a (poor) workaround.
- **Repro:** Golden, 160x45: Library > Notes (122) > expand Projects > Thesis > wheel or press Down over the tree past the pane bottom > Enter.
- **Why it matters:** Anyone with more notes than fit cannot browse to them, the pager or Trash; keyboard and low-vision users get focus on rows that are not painted and open notes they never saw.
- **Fix:** In css/features/_library_panels.tcss:544 add 'height: 1fr; min-height: 0; overflow-y: auto; overflow-x: hidden;' to the non-compact #library-notes-list rule (mirroring _library.tcss:1004), regenerate the bundle with css/build_css.py, and call scroll_visible() on the focused row in _move_library_list_row_focus (UI/Library_Modules/canvas_sync.py:61-107) and on Tab focus. Pilot test at 120x36 and 160x45 golden: list.max_scroll_y > 0 and the 30th focused row lies inside the visible region.

<a id="n-05"></a>
### N-05 · P1 · Note editor header overflows between 120 and ~200 columns: Save and 'Use in Console' clipped or missing, save state crushed to one column, Tab/F6 land on hidden Save

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j3-poweruser-notes, j5-a11y-consistency, j6-stress, j7-researcher-loop, B-mechanical · **Personas:** Jordan, Alex, Sam, Riley, Priya
- **Heuristics:** H1, H6, H8, WCAG 1.4.10
- **Evidence:** 120x36: the row shows only 'Edit  Preview  Info'; no Save or Use in (j1 41, B caps/notes-editor-120x36 r14, j5 19-editor-tab05/06 where Tab lands on Save/Use in with 0 cells changed and only the footer says 'enter save note'). 160x45: '… Save · Use in' cut at the pane edge (j1 49 ANSI cols 151-158, j6, j7, B). 200x50 with Nav open: 'Use in C' (j3 16). The status Static is a 1-column strip painting 'S', 'E/n/—', 'U/c' (j1, j5, j6, j7 rowstyles col 77); at 60/80/235 it is painted twice (B). j7 clicked the clipped 'Use in' and was sent to Console without understanding why; 'Discard new note' is off-screen on a new note (j7). Code: status + mode controls + task actions share one Horizontal #library-note-header-second-row (Widgets/Library/library_notes_canvas.py:2814-2870); #library-note-task-actions min-width 61 (css/features/_library.tcss:951-953) applies whenever the shell is ≥120 cols although the editor pane is ~50 cols at 120 and ~86 at 160; F6 prefers library-note-save (UI/Screens/library_screen.py:1484-1494). Severity: j3/j6 rated P3 and j7 P2; normalized to P1 because at 120 cols the explicit Save and the primary Console hand-off are unreachable (j1 task 9 failed).
- **Repro:** Golden, 120x36 and 160x45: Library > Notes > open any note with the list visible; read the row under the title; Tab from Body ×5.
- **Why it matters:** The explicit Save the User Guide tells users to use and the primary hand-off are invisible at common big-terminal sizes; the save-state indicator is unreadable.
- **Fix:** Choose the compact editor header from the editor pane width (reuse _effective_pane_width()), not the 120-cell shell breakpoint. Put #library-note-status on its own full-width row or remove it (the authority line already shows 'Saved · Next: …'). Let #library-note-task-actions wrap under the mode controls (height:auto) instead of min-width 61. Never truncate 'Use in Console'; keep 'Discard new note' visible. Make F6 skip a target with no visible region.

<a id="n-06"></a>
### N-06 · P1 · Opening any modal (F1 help, Move note / Add to folder) cancels the pending autosave and never re-arms it, while the status keeps promising automatic save

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j6-stress · **Personas:** Jordan, Riley
- **Heuristics:** H1, H5
- **Evidence:** j1: typed title and body, F1, Esc; 25 s later still 'Unsaved changes · Next: Keep editing; changes save automatically.' (16) and the DB row is 'Untitled||1'; fresh repro 53. j6: ' A2' typed, Move note within 1 s, Cancel; 'Unsaved changes …' for 15+ s with the DB unchanged (85-86); control with a resize and no modal saved (v5). Code (j1): LibraryScreen.on_screen_suspend stops _notes_state.autosave_timer (UI/Screens/library_screen.py:9256-9260); ScreenSuspend fires on modal push; on_screen_resume (:9262+) never re-arms. j6's exact caller is untraced but consistent with the same suspend path. Severity: j1 rated P0 (data loss with Ctrl+Q) and j6 P1; normalized to P1 because j1's own control (55) shows the loss is caused by N-01 - the edit is still flushed on navigation - but the status line is a standing false promise.
- **Repro:** Open a note > type a line > within 2 s press F1 (or Move note) > Esc/Cancel > wait 15 s > query the DB: unchanged, status still 'Unsaved changes … save automatically'.
- **Why it matters:** Looking up help or filing a note right after typing silently stops saving; combined with N-01 or a crash the edit is lost.
- **Fix:** In on_screen_resume, call _schedule_library_note_autosave() when the session is dirty - or better, await _flush_library_note_save() in on_screen_suspend instead of stopping the timer. Derive the 'save automatically' copy from 'a save is scheduled', not from dirty alone.

<a id="n-07"></a>
### N-07 · P1 · An autosave validation veto (trailing space in title, duplicate keyword) yanks focus mid-typing, so the next words land in the wrong field

- **Surface / kind:** Notes · usability · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j6-stress · **Personas:** Jordan, Riley
- **Heuristics:** H3, H5, H9
- **Evidence:** j1 (63, ANSI row 18): 'Groceries ' Tab Tab 'milk', wait 3.5 s, ' eggs' -> focus jumped to Title and the note saved as 'Groceries  eggs' / body 'milk'. j6 (12): same; and with Keywords 'ok, x, X' the pane switched to Info and body text became keywords 'ok, x, X body2' (16-17). Status contradicts itself: 'Title begins or ends with whitespace — remove it to save. · Next: Keep editing; changes save automatically.' Code: veto at Library/library_notes_session.py:582-597; focus routed even for explicit=False autosaves at UI/Library_Modules/library_notes_controller.py:3945-3978; keywords force Info at :3865-3874.
- **Repro:** New note > Blank > title 'Groceries ' (trailing space) > Tab Tab > 'milk' > wait 3 s > keep typing.
- **Why it matters:** An invisible character silently redirects the user's text into the title or keywords, corrupting the note.
- **Fix:** Trim title whitespace and dedupe keywords at the save boundary (with a one-line notice) instead of vetoing. Never move focus on an autosave veto; mark the field inline and move focus only on explicit Save or a leave attempt. Drop the 'save automatically' suffix while a veto is showing.

<a id="n-08"></a>
### N-08 · P1 · Typed [[wikilinks]] are dead text, and Preview hands note:// links to the OS URL handler

- **Surface / kind:** Notes · missing-capability · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j3-poweruser-notes · **Personas:** Jordan, Alex
- **Heuristics:** H2, H6, H7
- **Evidence:** No link control, '[[' completion, F1 entry or palette command (palette 'link' -> Chunking Lab / Artifacts / Skills). Hand-typed '[[Thesis meeting agenda]]' renders as plain #e0e0e0 text in Preview (j1 18) and the target's Info says 'Linked from (0)' (j1 19); j3: 'See [[Trip planning: Lisbon]]' is not counted in Lisbon's 'Linked from (1) · Index — start here'. Only '[[Title]](note://<uuid>)' counts (User Guide notes.md:463) and the UUID is shown nowhere; in Edit, imported links show raw UUIDs wrapping across lines. Preview builds Markdown(...) with default open_links and no LinkClicked handler (Widgets/Library/library_notes_canvas.py:2909-2913), so a click reaches app.open_url('note://…'). The click path is code-traced only - neither journey clicked, to avoid launching the macOS handler.
- **Repro:** Create note A; create note B with body 'See [[A]]'; Preview; open A > Info > Linked from.
- **Why it matters:** Linking is the core PKM action; users from Obsidian/Logseq cannot link without a UUID the UI never shows (j1 task 5 failed).
- **Fix:** On save, resolve bare [[Title]] and [[Title|alias]] by unique title using the Import-once resolver and record the edge (store the canonical form). Add a '[[' title-completion popup and an Info 'Copy link to this note'. Construct Preview's Markdown with open_links=False and handle Markdown.LinkClicked: note:// opens the note in-app, Enter follows a focused link. In Edit, display stored links as [[Title]].

<a id="n-09"></a>
### N-09 · P1 · Notes filter matches only exact whole-word phrases in title/body: no prefix, no reordered words, keywords not searchable

- **Surface / kind:** Notes · usability · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j3-poweruser-notes, j7-researcher-loop · **Personas:** Jordan, Alex, Priya
- **Heuristics:** H2, H7
- **Evidence:** 'filter: meet · 0 results' vs 'meeting · 3 results' (j1 59-60); 'filter: Lisb · 0 results' (j3 11); keyword 'zebra' (in note_keywords) -> 0 (j1 67); 'homelab', 'keyword:homelab' -> 0; 'size chunk' 0 vs 'chunk size' 4 (j3 12-13); conversation-id keyword '71d30fb1' -> 0 (j7 64). Code: notes_fts indexes title,content only (DB/ChaChaNotes_DB.py:1096-1101); the filter builds one quoted phrase (Utils/fts5_match_forms.py:368-391; Notes/note_folder_repository.py:808-826).
- **Repro:** Notes list > / > 'meet' Enter, then 'meeting'; add keyword 'zebra' to a note and filter 'zebra'.
- **Why it matters:** Keywords users add are decorative; type-ahead and tag habits find nothing, so 'find it again' fails for first-timers and power users alike.
- **Fix:** AND the tokens instead of one phrase, add * to the last token, keep phrase matching only for quoted input, and include notes whose keywords match (join note_keywords or support '#tag' / 'tag:'). Zero-result copy: 'No notes match "meet" — try fewer words'.

<a id="n-10"></a>
### N-10 · P1 · Inline delete confirmation renders off-screen; Cancel/Delete are pressed blind

- **Surface / kind:** Notes · a11y · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, j5-a11y-consistency · **Personas:** Jordan, Sam
- **Heuristics:** H1, H5, WCAG 2.4.11
- **Evidence:** 120x36: after Info > Delete only the footer changes ('enter cancel | tab switch button'); wheel-scrolling Info stops at 'Delete this note? Undo will be available' and the buttons never render (j1 20-21; j5 33-36, Tab toggles 'enter cancel'/'enter delete' with 0 cells changed). 80x24: buttons appear only after a manual wheel (j5 171-172). 160x45: '┃ Cancel ┃ Delete' visible (j1 52). Cause hypothesis: the confirm mounts inside the Info VerticalScroll (Widgets/Library/library_notes_canvas.py:2990-3006, 3099-3123) and focus moves to Cancel before it has a region, so it is never scrolled into view.
- **Repro:** 120x36: open a note > Info > Delete > look for Cancel/Delete; Tab between them.
- **Why it matters:** A destructive confirmation the user cannot see invites accidental deletes or abandonment; keyboard and low-vision users rely only on a footer chip.
- **Fix:** After the confirm mounts, call_after_refresh(cancel.scroll_visible) and focus Cancel; or render 'Delete note? [Cancel] [Delete]' in place of the Delete button under 'Danger' so it never depends on scrolling.

<a id="n-11"></a>
### N-11 · P1 · Note and prompt Export silently overwrite an existing file, starting at ~ with a title-derived name

- **Surface / kind:** Export · defect · confidence high · live-reproducible
- **Sources:** j6-stress, B-static · **Personas:** Riley
- **Heuristics:** H5
- **Evidence:** j6 72: exporting to an existing exp/precious.md -> toast 'Note exported successfully to precious.md' and the file that held 'PRECIOUS USER FILE' now starts with the note's front matter. B-static: two seeded notes titled 'Reading list' both export to ~/Reading list.md with no prompt. Code: FileSave(location=str(Path.home()), default_file=f'{safe_title}.md') with default can_overwrite=True (UI/Library_Modules/library_notes_controller.py:4300-4323; Third_Party/textual_fspicker/file_save.py:37); plain write_text (UI/Screens/library_screen.py:21158-21180); prompts use the same pattern (library_prompts_controller.py:3709-3712). The Export-bundle canvas does check destination_exists (library_export_controller.py:1408), so this is inconsistent within Library.
- **Repro:** Create ~/exp/precious.md > open any note > Info > Export Markdown > type that path > Enter.
- **Why it matters:** A file outside Chatbook (e.g. the user's own ~/TODO.md) is destroyed with no confirmation and no undo.
- **Fix:** Pass can_overwrite=False, or intercept an existing path with 'Replace “X.md” in ~/exp? · Replace · Choose another name' (Cancel focused). Remember the last export directory, and toast the full destination path.

<a id="n-12"></a>
### N-12 · P1 · Cancelling a note export is reported as 'Export failed' and focus lands on Delete

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H1, H9
- **Evidence:** j5 169 (80x24) and 185 (160x45): after Esc in the file dialog the status reads 'Saved · Export failed — choose another destination and try again. · Next: Review the error, then keep editing.', Info repeats it, the toast says 'Note export cancelled.', and the footer reads 'enter delete note' with ┃ Delete ┃ focused. Code: UI/Screens/library_screen.py:21136-21144 treats selected_path None as success=False. The focus move to Delete is a hypothesis (the operation disables Info actions, so focus falls through).
- **Repro:** Open a note > Info > Export Markdown > Enter > Esc in the file dialog.
- **Why it matters:** Changing your mind is reported as a failure, and a keyboard user is one Enter from the delete confirm.
- **Fix:** Add a 'cancelled' outcome to _finish_library_notes_operation that writes no failure copy (status returns to 'Saved'); on FileSave dismiss, re-focus #library-note-context-export-md / -txt.

<a id="n-13"></a>
### N-13 · P1 · Arriving via the palette 'New Note' shows the Notes tree with blank rows instead of its folders

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H1
- **Evidence:** j3 04, 06, 53, 54, reproduced in two fresh runs: Console > Ctrl+P 'New Note' > Blank note > Esc -> two blank rows, then '▾ Unfiled'; Agent_Lessons, Journal, Meetings, Projects, Recipes, Research, Study are missing and 'g go to folder' lands on Unfiled. They appear after a rail Notes press or another create. Cause untraced; hypothesis: the notes_create entry composes the tree before requesting the root-folder branch.
- **Repro:** Fresh golden launch > Console > Ctrl+P 'New Note' > Enter > Enter > type a title > Esc > read the tree.
- **Why it matters:** Right after a capture is when the user files the note, and the folders look deleted.
- **Fix:** Request the root folder branch on the notes_create and note_id entries exactly as the rail-row entry does, and render 'Loading folders…' instead of blank rows.

<a id="n-14"></a>
### N-14 · P1 · Notes select mode is export-only, 'Select all N shown' miscounts, and selecting costs two keys per row

- **Surface / kind:** Notes · missing-capability · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j5-a11y-consistency · **Personas:** Alex, Sam
- **Heuristics:** H1, H7
- **Evidence:** j3 28, 31: strip '10 selected   Done   Select all 20 shown   Clear   Export selected'; selecting took Enter+Down ×10 and Space does nothing; after returning from Export the label read 'Select all 100 shown' but pressing it gave '26 selected'. j5 124: 'Select all 100 shown' with ~13 rows visible. Code: the label falls back to list_state.rows without a tree_projection (Widgets/Library/library_notes_canvas.py:1786-1795) while the handler builds a fresh projection (UI/Screens/library_screen.py:30125-30138). The User Guide itself says an imported vault must be deleted one note at a time.
- **Repro:** Notes > filter 'Daily log' > Select > Enter+Down ×10 > look for Move/Tag/Delete > Export > Esc > Select all.
- **Why it matters:** Re-filing or re-tagging ten notes takes ~120 keys; the Select-all label promises a count it does not deliver.
- **Fix:** Add Move to folder…, Add/remove keyword… and Delete (with undo) to the select strip; make Space toggle a row and Shift+Up/Down extend; compute 'Select all N loaded' from the same projection the handler uses.

<a id="n-15"></a>
### N-15 · P1 · Sync review rows for files deleted or renamed (on disk or in-app) can never be resolved, and 'Apply reviewed' is disabled with no visible reason

- **Surface / kind:** Notes-sync/import/folder-files · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j6-stress · **Personas:** Alex, Riley
- **Heuristics:** H3, H9
- **Evidence:** j6 98 (png): 'ideas.md · One side was deleted' and 'quotes renamed.md · Preview explicit filesystem move', each 'Resolution unavailable for this item. No changes can be staged.' with disabled '○ Restore missing side / ○ Delete/archive counterpart / ○ Disconnect item / ○ Apply once / ○ Leave unchanged'; 99: 'Apply reviewed' is dim (#9e9e9e on #0d0d0d) without a reason and the two safe creates never apply. j3 66: the same dead end after an in-app delete. Code: Widgets/Library/library_notes_add_from_files_canvas.py:780-800 renders these choices disabled=True unconditionally.
- **Repro:** In a synced folder: rm ideas.md and mv quotes.md 'quotes renamed.md' > Check changes > try any choice and Apply reviewed.
- **Why it matters:** Routine deletes and renames permanently stop the folder from syncing; unrelated safe changes are held hostage.
- **Fix:** Wire Restore missing side (recreate from the other side) and Delete counterpart (move the file to the vault's .trash / soft-delete the note) to the runtime; accept a rename as a move. Let Apply reviewed apply the safe set while attention rows stay pending. Print the blocker reason as a visible line, and point to Recently deleted > Restore when an action stays disabled.

<a id="n-16"></a>
### N-16 · P1 · One non-UTF-8 or mixed-newline file blocks the whole sync folder with 'Check failed — NotesSyncRootRefused'

- **Surface / kind:** Notes-sync/import/folder-files · defect · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H2, H9
- **Evidence:** j6 97: '⚠ Needs attention · Check failed — NotesSyncRootRefused · Next: Check changes'; log reason=unsupported_encoding, then reason=mixed_newlines after removing the first file; deletes and renames were not processed until both files were gone (98). Code: Notes/notes_sync_filesystem.py:136-150 raises a root-level refusal; _CHECK_REFUSAL_COPY (Library/library_notes_lasting_sync_state.py:1007+) lacks these reasons so the type name is shown (:1185, :1201-1204).
- **Repro:** In a synced folder add a latin-1 file (byte 0xE9) > Check changes; replace it with a CRLF/LF-mixed file > Check changes.
- **Why it matters:** One legacy or Windows-edited file stops the entire folder, the message does not name the file, and the next action repeats the failure.
- **Fix:** Report unreadable files as per-file review rows ('Skipped — not UTF-8: latin1.md', 'Skipped — mixed line endings: …') and sync the rest; add copy for every refusal reason code.

<a id="n-17"></a>
### N-17 · P1 · 'Add from files…' gets stuck on a stale lasting-sync phase: an old receipt after activation, or a blank page after a failed Recovery

- **Surface / kind:** Notes-sync/import/folder-files · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j6-stress · **Personas:** Alex, Riley
- **Heuristics:** H1, H3
- **Evidence:** j3 59-60: after activating a root, Add from files… reopens 'Sync root activated. 64 applied · listed under Receipts' with only '○ Resolution history' and '‹ Notes'; Esc and re-entry show the same receipt. j6 92, 94 (reproduced twice): after a failed Recovery, Add from files shows only 'Add files to Library notes.' and 'Recovery failed — RuntimeError. Next: Check changes.' - no Import once, no Keep synced, no ‹ Notes. Only a restart clears either. Code (j6): library_notes_sync_controller.py:1527-1529 leaves phase='roots'; library_notes_add_from_files_canvas.py:356-595 and 1026-1156 have no 'roots' branch. j3 rated P2 (medium confidence); normalized to P1 because Import once is fully blocked for the session in the j6 variant.
- **Repro:** Activate a root (or trigger a failed Recovery) > Esc to the list > Add from files… twice.
- **Why it matters:** Import once and every new sync setup are blocked until restart; a user with a second folder cannot add it.
- **Fix:** Reset the lasting phase to 'choose' whenever Add from files is opened or a receipt/recovery is left, add 'Set up another folder' to the receipt bar, and render the chooser as the fallback for any unknown phase.

<a id="n-18"></a>
### N-18 · P1 · Every synced folder is titled 'Sync folder (name unavailable before cutover)'; roots and receipts cannot be told apart

- **Surface / kind:** Notes-sync/import/folder-files · copy · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j6-stress, B-static · **Personas:** Alex, Riley
- **Heuristics:** H2, H6
- **Evidence:** j6 96: roots named 'Riley Vault' and 'Riley MD' both read 'Sync folder (name unavailable before cutover)'; Receipts mix both roots unlabeled. j3: same after naming 'PowerVault'. Code: hard-coded at UI/Library_Modules/library_notes_sync_controller.py:817-819; rendered as the row heading at Widgets/Library/library_notes_sync_roots_canvas.py:72-76; the setup field '#notes-sync-display-name' is collected and never shown. Severity: j6 P3, j3 P2 (copy), B P1; normalized to P1 because with two roots Pause/Resume/Check/Recovery cannot be aimed (no path is shown either), and it leaks the internal term 'cutover'.
- **Repro:** Activate two Keep-synced roots with different display names > Manage sync folders.
- **Why it matters:** Users cannot tell which folder is failing or which one they are pausing.
- **Fix:** Carry the root's display name (and its folder path) into LastingSyncRootRow's title; fall back to 'Synced folder N' in creation order, never to the literal. Prefix each receipt with the root name.

<a id="n-19"></a>
### N-19 · P1 · Sync Receipts says 'No writes yet.' when the write history could not be read

- **Surface / kind:** Notes-sync/import/folder-files · defect · confidence high · code-only (not observed live)
- **Sources:** B-static
- **Heuristics:** H1, H9
- **Evidence:** Code-traced, not reproduced live: refresh_receipts catches each root's runtime.write_receipts() failure with 'except Exception: logger.warning(...); continue' and publishes write_receipts=() (UI/Library_Modules/library_notes_sync_controller.py:875-913); the canvas then renders 'No writes yet. Sync writes are listed here when you open this list.' (Widgets/Library/library_notes_sync_roots_canvas.py:122-127). Realistic trigger: write_receipts -> _require_cutover raises 'notes_sync_cutover_not_admitted' while admission is closed (Notes/notes_sync_runtime.py:2687, 3711-3714). The screen's own comment calls Receipts 'the only trace of the writes lasting sync performs on its own' (library_screen.py:27661-27664). Kept at B's P1 rather than P0 because the false state was not observed live.
- **Repro:** Hypothesised: with an active root that has applied writes, relaunch (REUSE=1) and open Manage sync folders while the root still reads '◌ Starting'; or chmod 000 the notes device-state DB and reopen the list.
- **Why it matters:** A false statement about what sync did to the user's files, on the only surface that records it; a partial failure looks complete.
- **Fix:** Collect failed root ids in refresh_receipts and render 'Couldn't read sync history for N folder(s) · Retry'; show the empty state only when every read succeeded, worded 'No sync writes recorded.' Add a test that a raising write_receipts never yields the empty-state copy.

<a id="n-20"></a>
### N-20 · P2 · Exported note front matter is invalid YAML for ordinary titles (':' or '*')

- **Surface / kind:** Export · defect · confidence high · live-reproducible
- **Sources:** j7-researcher-loop · **Personas:** Priya
- **Heuristics:** H4, H5
- **Evidence:** j7 68-exported-note-paper-notes.md: yaml.safe_load fails on 'title: Paper notes: Retrieval practice' ('mapping values are not allowed here') and on 'title: *(mock reply)*…' ('while scanning an alias'). Code: Chatbooks/chatbook_creator.py:1465 writes f'title: {note["title"]}' unquoted; the same pattern is at Library/library_notes_state.py:915 for single-note export. (Keyword loss in the same bundle is L-04.) j7 rated the combined finding P1; this half is P2 because the text survives and only front-matter consumers break.
- **Repro:** Notes > Select > check a note titled 'Paper notes: Retrieval practice' > e > Export bundle > unzip > yaml.safe_load the front matter.
- **Why it matters:** Files break in Obsidian, Pandoc and static-site tools; research titles commonly contain colons.
- **Fix:** Emit front matter with yaml.safe_dump (or quote every scalar) in both chatbook_creator and build_note_export_content; add a round-trip test for titles with ':', '*', '#' and quotes.

<a id="n-21"></a>
### N-21 · P2 · No fast path from a query to a note, and no note history

- **Surface / kind:** Notes · usability · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H3, H7
- **Evidence:** j3 step 14: from the filter, the first result is 8 Tabs away (New, Select, Add from files…, Export, New folder, Clear filter, ▾ Unfiled, row); Esc from the filter jumps to the rail search; ']' and '[' are bound only for Media (library_screen.py BINDINGS); there is no back/forward. j3 rated P1; normalized to P2 because the path exists, just slowly.
- **Repro:** Press /, type 'start here', Enter, count Tabs to the 'Index — start here' row; follow a backlink and try to return.
- **Why it matters:** Finding a note costs ~11 keys plus the query; returning after following a backlink means searching again.
- **Fix:** Let Down/Enter in the filter focus the first result row; add a 'Go to note…' fuzzy title switcher (palette entry plus a Notes key such as o); add Alt+Left/Right note history and reuse ]/[ for next/previous note.

<a id="n-22"></a>
### N-22 · P2 · Arrow keys skip folder rows and there is no expand/collapse key in the Notes tree

- **Surface / kind:** Notes · a11y · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, B-mechanical, j5-a11y-consistency · **Personas:** Alex, Sam
- **Heuristics:** H4, H7
- **Evidence:** j3: 'g' focuses '▸ Agent_Lessons', then Down and Right produce an empty ansidiff. B: Down ×6 on '▸ Recipes' = empty diff; Tab order reaches the first note 'Ideas inbox' only after ~17 stops. j5 24-list-g-down1…3 dead on folder rows, contradicting the User Guide's '↑/↓ inside a … Notes list'. Code: folder rows use 'library-notes-folder-row' (Widgets/Library/library_notes_canvas.py:2387), missing from _LIBRARY_LIST_ROW_CLASSES (UI/Library_Modules/screen_constants.py:432-450). B rated P3, j3 P2.
- **Repro:** Notes list > g > Down, Right.
- **Why it matters:** Arrow-key tree navigation is basic muscle memory; keyboard users pay a Tab per row.
- **Fix:** Add 'library-notes-folder-row' and 'library-notes-tree-pager' to _LIBRARY_LIST_ROW_CLASSES; Right/l expands or enters a folder, Left/h collapses or goes to parent; separate selecting a folder from toggling it.

<a id="n-23"></a>
### N-23 · P2 · Folder pickers (Move note, Move folder, Add to folder) list only folders currently expanded in the tree

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j6-stress · **Personas:** Alex, Riley
- **Heuristics:** H6
- **Evidence:** j3 21: picker lists Agent_Lessons, Journal, Journal / Daily, Meetings, Projects, Recipes, Research, Study but not Projects / Thesis, Projects / App launch or Study / Japanese. j6 43: Move-folder targets omit every nested folder; 27 lists 'Projects / Thesis' only while Projects is expanded. Code: options built from loaded tree_branches (UI/Library_Modules/library_notes_controller.py:3250-3273). The dialog is titled only 'Move note' and does not name the note (j3).
- **Repro:** Collapse Projects > select a note or folder > Move > open the target list.
- **Why it matters:** Users cannot file into nested folders without first expanding them, which is not discoverable.
- **Fix:** Query all active folders for the picker independent of tree expansion, add type-to-filter, and title the dialog 'Move “<title>” to…'.

<a id="n-24"></a>
### N-24 · P2 · 'Move note' fails with a false 'That folder changed elsewhere — refresh and try aga…' for Unfiled or just-deleted notes, and the notice sticks

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes, j6-stress · **Personas:** Alex, Riley
- **Heuristics:** H5, H9
- **Evidence:** j3 22-23: on an Unfiled note, Choose -> 'That folder changed elsewhere — refresh and try aga…' (truncated) although nothing changed; Add to folder works for the same note. j6 26-32: after deleting 'Trip planning: Lisbon', Add to folder and Move note stay enabled; Move gives the same message, and it still shows after a successful Undo until restart. Code: FolderConflictError mapped to generic copy (UI/Screens/library_screen.py:18797-18801); placement selection not cleared on delete (hypothesis).
- **Repro:** Focus an Unfiled note > Move note > pick a folder > Choose. Or delete a note via Info > Delete, then Move note > Choose.
- **Why it matters:** The user hunts for a concurrent edit that never happened, and is told to 'refresh' with no refresh control.
- **Fix:** Disable Move note for Unfiled rows with the reason 'Unfiled has no folder to move from — use Add to folder'; clear tree_selected_placement_id when the selected note is deleted and say 'That note was deleted — Undo to restore it'; wrap the error text; clear notices on the next successful operation.

<a id="n-25"></a>
### N-25 · P2 · Leave-veto toast names a 'Discard new note' button that is not there and blames the title for any veto

- **Surface / kind:** Notes · copy · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H9
- **Evidence:** j6 13: on an existing note with a trailing-space title, Esc -> 'Can't leave yet — fix the title or press Discard new note.'; 0 occurrences of 'Discard' on screen. Code: UI/Screens/library_screen.py:909-921 returns one fixed string for every VALIDATION_VETO, including keyword vetoes.
- **Repro:** Existing note > set a trailing-space title > Esc.
- **Why it matters:** Users hunt for a non-existent control; keyword problems are misattributed to the title.
- **Fix:** Name the field and fix ('Remove the trailing space from the title' / 'Remove duplicate keyword x'); offer 'Revert changes' on existing notes and 'Discard new note' only on new ones.

<a id="n-26"></a>
### N-26 · P2 · Info 'Linked from' never resolves on a new note, or shows the previous note's backlinks

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes · **Personas:** Jordan
- **Heuristics:** H1
- **Evidence:** j1 12, 14: a new Blank note's Info reads 'Linked from — checking…' until reopened from the list. j1 65-66: after opening 'Thesis meeting agenda' ('Linked from (1) · Advisor meeting follow-up'), pressing n shows the same backlink on the new note. Code: the create path (UI/Screens/library_screen.py:30847-30882) never resets backlinks/backlinks_status nor runs _load_library_note_backlinks (only _begin_library_note_open does, library_notes_controller.py:3452-3468); default status is 'loading' (library_notes_state.py:524).
- **Repro:** Open a note with an inbound link > Esc > n > Info.
- **Why it matters:** Info states a false fact about the note's relationships, or spins forever.
- **Fix:** In the create path set backlinks=() and backlinks_status='ready' (a new note has no inbound links), or run the same backlinks worker.

<a id="n-27"></a>
### N-27 · P2 · 'View 10 imported notes' lands on the list with the import folder collapsed

- **Surface / kind:** Notes-sync/import/folder-files · usability · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes · **Personas:** Jordan
- **Heuristics:** H1
- **Evidence:** j1 32-33: receipt 'Import completed. 10 notes created' > 'View 10 imported notes' shows '▸ md-folder' collapsed beside Unfiled; none of the 10 notes are visible.
- **Repro:** Add from files > Import once > md-folder > Check selection > Import selected items > View 10 imported notes.
- **Why it matters:** The button promises the notes and delivers a closed folder; a literal reader thinks the import vanished.
- **Fix:** When arriving from an import receipt, expand and select the destination folder and scroll it into view (or open the list filtered to the imported notes).

<a id="n-28"></a>
### N-28 · P2 · Moving a folder into a collapsed folder hides the target's own notes until collapse/re-expand

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H1
- **Evidence:** j6 44-46: after Move Recipes > Research, '▾ Research' shows only '▾ Recipes' and its 2 recipes though the DB has 5 notes in Research; collapse and re-expand shows all.
- **Repro:** Click ▸ Recipes > Move > Research > Choose.
- **Why it matters:** Users may believe the target folder's notes were lost.
- **Fix:** Reload the destination branch fully after move_folder in _reconcile_library_notes_tree_mutation.

<a id="n-29"></a>
### N-29 · P2 · Notes 'Recently deleted' cannot delete anything permanently

- **Surface / kind:** Notes · missing-capability · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H3, H4
- **Evidence:** j6 40: 'Deleted notes stay here until you restore them… nothing is removed for good from here.' Restore is the only action, while Media Trash offers 'Delete forever' with a 'Delete permanently' confirm (80).
- **Repro:** Filter anything > 'Recently deleted (N)'.
- **Why it matters:** A note with sensitive content can never be purged; inconsistent with Media Trash.
- **Fix:** Add 'Delete forever' (key x, as in Media) with the same 'This cannot be undone' confirm.

<a id="n-30"></a>
### N-30 · P2 · Notes sort chooser opens with focus left in the Filter field

- **Surface / kind:** Notes · usability · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H4
- **Evidence:** j5 128-129: the '✓ Newest  Oldest  Title' strip opens, focus goes to Filter, the footer says 'enter choose sort' (Enter would submit the filter), and Down/Right do nothing. Handler: UI/Library_Modules/library_notes_controller.py:4916-4952.
- **Repro:** Notes list > Tab to 'Sort: Newest' > Enter > Down.
- **Why it matters:** The footer's promise is wrong and arrow keys are dead in a chooser.
- **Fix:** Focus the current '✓' option when the strip opens (Esc already returns to the opener).

<a id="n-31"></a>
### N-31 · P2 · Notes text fields never switch the footer to 'typing in field', so bare-letter chips stay advertised and collide

- **Surface / kind:** Notes · consistency · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency, B-mechanical · **Personas:** Sam
- **Heuristics:** H4, H5
- **Evidence:** j5 14, 29, 56: no 'typing in field' in the Notes filter, Title or Keywords, unlike Conversations/Prompts/landing (77, 81, 04); at 80x24 'g' typed into the filter (164). B probes/slash-*: after '/' focuses the filter, the footer stays 'esc focus rail | F1 help · F6 next pane', and Esc cleared the unsubmitted 'zq'. Transform: UI/Screens/library_screen.py:4736-4831.
- **Repro:** 80x24 Notes > Ctrl+P > Esc (focus returns to Filter) > press g.
- **Why it matters:** Single-letter keys type into the field without warning; breaks the footer contract the rest of Library follows.
- **Fix:** Include #library-notes-filter, #library-note-title, #library-note-keywords and #library-note-context-keywords in the typing-in-field footer transform; make Esc in the filter blur without clearing unsubmitted text.

<a id="n-32"></a>
### N-32 · P2 · F6 in the note body selects the current line instead of moving to the next pane

- **Surface / kind:** Notes · defect · confidence high · live-reproducible
- **Sources:** B-mechanical · **Personas:** Sam
- **Heuristics:** H4, H5
- **Evidence:** B footer/key-outcomes.txt: 4/4 runs, focus stays in the body, line 1 gets selection style '#e0e0e0 on #27496f' and the position goes '1:1' -> '1:40'; the footer advertises 'F6 next pane'. Cause: Textual 8.2.8 TextArea binds f6 to select_line, and the app's F6 binding (tldw_chatbook/app.py:935, verified) is not priority=True; Shift+F6 is priority and works.
- **Repro:** Golden 160x45 > Notes > open 'Ideas inbox' > press F6.
- **Why it matters:** Keyboard users cannot leave the editor with the advertised key, and the next keystroke replaces the selected line.
- **Fix:** Make the app-level F6 binding priority=True (as Shift+F6 is), or remove f6/f7 from the note, prompt and skill TextArea subclasses' BINDINGS.

<a id="n-33"></a>
### N-33 · P2 · Ctrl+S saves only Skills; the Notes save action exists but is unbound, and Prompts/Folder files have no save key

- **Surface / kind:** Notes · consistency · confidence high · live-reproducible
- **Sources:** B-static · **Personas:** Alex
- **Heuristics:** H4, H6
- **Evidence:** UI/Screens/library_screen.py:1047 ('ctrl+s', 'library_skill_save') is the only ctrl+s binding; action_library_notes_save (:8688-8719) and its check_action gate (:25358-25363) have no binding and no caller. Prompts have only a 'Save changes' button (library_prompts_canvas.py:1630-1640); Folder files autosaves with no Save. Live: Ctrl+S in a note leaves 'Unsaved changes' until the debounce fires.
- **Repro:** Open a note, type, press Ctrl+S at once; open prompt 'Rewrite for clarity', edit, Ctrl+S.
- **Why it matters:** Four editors on one screen, four save models; a key that works in Skills silently does nothing next door.
- **Fix:** Bind ctrl+s to library_notes_save and to a library_prompt_save action (gates are already disjoint) and to 'save now' in Folder files; add ('ctrl+s','save') to the editor footer sets.

<a id="n-34"></a>
### N-34 · P2 · Escape in Add-from-files cancels a running import (Back keeps it running) and silently discards a sync review's staged choices

- **Surface / kind:** Notes-sync/import/folder-files · usability · confidence high · code-only (not observed live)
- **Sources:** B-static
- **Heuristics:** H3, H4, H5
- **Evidence:** Code-traced: Escape while CHECKING/IMPORTING calls cancel() (UI/Library_Modules/library_notes_controller.py:3098-3102; library_note_import_controller.py:754-769; receipt 'Cancelled. Finished items were not rolled back.'), while the visible '‹ Notes' button returns and lets the import continue (:5407-5412). In a lasting review, Escape -> _exit_library_notes_lasting_sync -> abandon_setup sets _review_plan=None (library_notes_sync_controller.py:1166-1178), dropping every 'Choice staged' with no confirm. Repro steps given by B; not captured live.
- **Repro:** Import once on the vault > Import selected items > Escape immediately vs '‹ Notes'. Keep-synced review with a staged 'Keep both' > Escape.
- **Why it matters:** Escape means 'back' everywhere else in Library; here it irreversibly stops an import or throws away a long conflict review.
- **Fix:** Make Escape behave like '‹ Notes' (import continues; cancelling stays on 'Cancel import'). In a review, Escape first collapses an open comparison; with staged choices, confirm 'Leave review? N staged choices will be discarded' or keep the review resumable from Manage sync folders.

<a id="n-35"></a>
### N-35 · P2 · Sync and comparison copy shows raw nanosecond timestamps, ISO times, exception class names and stale lines

- **Surface / kind:** Notes-sync/import/folder-files · copy · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H2, H9
- **Evidence:** j3 62, 41, 43: 'File modified 1790998978887732948 ns · 8 lines/82 chars' (Widgets/Library/library_notes_add_from_files_canvas.py:855); 'Note v2, updated 2026-10-03T03:43:24.542000+00:00'; 'Recovery failed — RuntimeError'; 'Resolution history unavailable — it starts after this root is activated' still shown after activation.
- **Repro:** Pause a root, edit both sides, Resume > Review > View comparison.
- **Why it matters:** PRODUCT.md rules out interfaces that require reading logs; a comparison the user must decode cannot support a keep-file/keep-note decision.
- **Fix:** Format both times as local 'YYYY-MM-DD HH:MM' with 'newer' marked; replace exception names with plain reasons; drop lines that no longer apply once the root is active.

<a id="n-36"></a>
### N-36 · P2 · Folder files' Session Git shows status but never a diff

- **Surface / kind:** Notes-sync/import/folder-files · missing-capability · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H1, H7
- **Evidence:** j3 46-47: 'EDITED Projects/Thesis/Method.md · READY TO STAGE · Git: unstaged · Status: CURRENT · READY — 1 can be staged' with 'Stage' and 'Show bulk · 1 stage', no diff control; grep of Widgets/Library/library_file_notes_git_panel.py finds no diff.
- **Repro:** Folder files > link a git vault > edit a file > Manage > Review session changes > Trust and check status.
- **Why it matters:** A git-backed vault user always reviews the diff before staging or committing.
- **Fix:** Add 'View diff' for the selected row, reusing the conflict-compare unified-diff box, and show the staged diff in Review commit.

<a id="n-37"></a>
### N-37 · P2 · Folder files with no folder linked is 61–78% blank, the only next step is an inline link, and the Library rail drops below the header

- **Surface / kind:** Notes-sync/import/folder-files · usability · confidence medium · live-reproducible
- **Sources:** B-mechanical
- **Heuristics:** H8
- **Evidence:** B caps/folderfiles-*: largest blank rectangle 78% (60x24) to 61% (120x36, 88x25) and 73% (235x52). At ≥100 cols three full-width lines ('Folder files · No folder selected', 'Choose a notes folder.  Choose folder…   Review recovered pairing…', explanation) push the rail box to start at r9-r10 with nothing to its right; at 60x24 'Review recovered pa' is cut.
- **Repro:** Golden 160x45 > Notes > 'Folder files' on the source strip.
- **Why it matters:** The single next action is buried in a dense header line; the shell's stable left column jumps.
- **Fix:** Keep the rail in the shell's left column as on every other route; render the unlinked state as a centred empty state in the work area with one sentence, a primary 'Choose folder…' button and 'Review recovered pairing…' as a secondary link.


## Library destinations

<a id="l-01"></a>
### L-01 · P0 · Media list 'Export…' freezes the whole app (no input, no repaint, Ctrl+Q dead)

- **Surface / kind:** Export · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H1, H3, H5
- **Evidence:** j2 68-72, reproduced 3/3 including a fresh empty profile with one .md (nl-j2-2): after clicking 'Export…' in the Media Items toolbar, F1, Ctrl+P, rail clicks and Ctrl+Q do nothing and a tmux resize does not repaint; process at 0.0% CPU; SIGUSR2 faulthandler shows the main thread idle in selectors.select; no traceback. Code (traced, hypothesis for mechanism): the async Button.Pressed handler on LibraryMediaCanvas (Widgets/Library/library_media_canvas.py:537-546) awaits handle_library_media_export -> _open_library_export_canvas (library_export_controller.py:741-786) -> _apply_library_open_item_surface which awaits self.recompose() (UI/Screens/library_screen.py:12404-12408), removing the canvas whose message pump is awaiting - a self-removal deadlock. Note j4 reached the Export canvas via select mode ('s' > Export) without a freeze, so the trigger is the list-toolbar button path.
- **Repro:** Library > Media (N) > click 'Export…' in the Items toolbar > press F1 / Ctrl+P / Ctrl+Q.
- **Why it matters:** The 'save my stuff' action bricks the session; the user must kill the terminal and loses any unsaved edits elsewhere (compounded by N-01).
- **Fix:** Do not await the surface swap from inside the canvas's own handler: forward with self.app.call_later(actions.handle_library_media_export, event) or run_worker, or have _open_library_export_canvas schedule _apply_library_open_item_surface via call_after_refresh. Pilot test: press #library-media-export, assert #library-export-canvas mounts and a later key is processed. Fix the stale 'awaits a modal' docstring.

<a id="l-02"></a>
### L-02 · P0 · Library analysis always says 'No analysis provider is configured', even when [analysis_defaults] names a ready provider

- **Surface / kind:** Media · defect · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H9
- **Evidence:** j4 21: '○ Generate · No analysis provider is configured · Set one in Settings ▸ Providers & Models.' with [analysis_defaults] provider='OpenAI' model='gpt-4.1-mini' in the run config; 08: bulk '○ Analyze' shows the same. Isolated probe: load_settings() has no 'analysis_defaults' key, while the resolver on the raw TOML returns ready=True. Code: config.py:2590-2620 return dict omits analysis_defaults; library_screen.py:33075-33088 resolves against app_config (= load_settings(), app.py:1124); Library/ingest_analysis.py:158-176. Affects the shipped default config (config.py:5514).
- **Repro:** Configure [analysis_defaults] with a ready OpenAI key > Library > Media > open any item > Analysis tab.
- **Why it matters:** Generate, bulk Analyze and Analyze-after-import are dead for every user, and the message falsely sends them to a Settings page that cannot fix it.
- **Fix:** Add 'analysis_defaults': copy.deepcopy(toml_config_data.get('analysis_defaults', {})) to the load_settings() return dict; add a test that resolve_ingest_analysis_provider(load_settings()) is ready when the TOML names a ready provider.

<a id="l-03"></a>
### L-03 · P1 · Opening an item by link (import 'Open in Library', Search 'Open') leaves the list cursor on row 1, so reader actions - and sometimes the reader itself - target the wrong item

- **Surface / kind:** Media · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H1, H4, H5
- **Evidence:** j2 11-14, 73-76: after 'Open in Library' on the PDF the reader shows the PDF but '█▸' sits on team-handbook; 'Read later' saved media_id 5 (team-handbook) 3/3 times (MediaReadItLaterState) and the reader swapped to team-handbook. Search 'Open' on 'article-local-first' loaded team-handbook (76). 'More ▸ Move to trash' asks the untitled 'Delete this media? You can undo right away…' (75). Control: clicking the PDF row first saves id 3 correctly. Code: _open_library_item_by_id media branch (UI/Screens/library_screen.py ~34787-34857) sets selected_media_id then requests a browse; _start_library_media_read_later_toggle acts on _selected_media_id (library_media_controller.py:4533-4552), which the settle-delay highlight (_select_library_media_reader_row, :2176-2201) re-points to row 0.
- **Repro:** Import a folder > 'Open in Library' on any row except the newest > Read later > check MediaReadItLaterState.
- **Why it matters:** Read later, Move to trash and Use in Console silently act on a document the user did not choose.
- **Fix:** In the media branch of _open_library_item_by_id, move the list highlight to record_id and scroll it into view before the browse refresh lands; make reader toolbar actions use the reader session's loaded id; name the item in the trash confirm ("Move 'team-handbook' to Trash?").

<a id="l-04"></a>
### L-04 · P1 · Export bundles silently drop keywords for media, conversations and notes

- **Surface / kind:** Export · defect · confidence high · live-reproducible
- **Sources:** j4-poweruser-library, j7-researcher-loop · **Personas:** Morgan, Priya
- **Heuristics:** H1, H5
- **Evidence:** j4: manifest content_items tags [] and media_6.json metadata.media_keywords null while MediaKeywords has 'learning,study' / 'reference,markdown'; the conversation has tags [] while its DB keyword is 'homelab'. j7 68-exported-manifest.json: 'tags': [] for a note with 3 note_keywords rows. The canvas mentions nothing ('Bundle: 2 media items · text only · about 4 KB'). Code: chatbook_creator reads media_item.get('media_keywords') (Chatbooks/chatbook_creator.py:1620) and note.get('keywords') (:1454) from SELECT * rows that have no keyword column (DB/Client_Media_DB_v2.py ~7201; DB/ChaChaNotes_DB.py:17826); the importer reads the same keys (chatbook_importer.py:2772).
- **Repro:** Select 2 media items with keywords (or notes with keywords) > Export bundle > unzip > read manifest.json and metadata/*.json.
- **Why it matters:** Tags are the triage and provenance; a backup or advisor hand-off loses them with no warning, and a round-trip import cannot restore them.
- **Fix:** Fetch keywords via fetch_keywords_for_media_batch and the note_keywords / conversation keyword joins, write them to metadata and ContentItem.tags, list 'Keywords' in the export consequence line, and add a creator->importer round-trip test per item type.

<a id="l-05"></a>
### L-05 · P1 · Import rejects '~/' paths as a 'dangerous pattern', then says 'Can't find that path' for a file that exists

- **Surface / kind:** Ingest · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H5, H9
- **Evidence:** j2 60-63; log 'WARNING … Rejected Library ingest path ~/fixtures/…'. For an existing '~/fixtures/import-docs/team-handbook.docx': 'Invalid path: Path contains dangerous pattern: ~/', then on Enter 'Can't find that path — check it, or use Browse…'. A missing file shows three messages at once. Pressing Enter before the pre-check finished imported the same path (ingest_jobs seq 6) - a race. Code: Library/ingest_preflight.py:421-424 calls validate_path_simple without expanduser; Utils/path_validation.py:559-571 lists '~/' as dangerous.
- **Repro:** Library > Import… > type '~/fixtures/import-docs/team-handbook.docx' > wait 3 s > read the error > Enter.
- **Why it matters:** Terminal users type ~/ by reflex; the app calls it a security risk and claims a real file is missing.
- **Fix:** Apply os.path.expanduser to the field value before preflight and submit (ingest_preflight.py and _resolve_ingest_source, library_screen.py ~28752); keep the '~/' check only for unexpanded internal paths; show one accurate error.

<a id="l-06"></a>
### L-06 · P1 · RAG Answer calls the provider's default model instead of the configured one, and names it only after the paid call

- **Surface / kind:** Search-RAG · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j4-poweruser-library, j7-researcher-loop · **Personas:** Jordan, Morgan, Priya
- **Heuristics:** H1, H3, H4
- **Evidence:** All three: pre-run line 'To openai: question + evidence'; post-run 'openai · gpt-5.6-terra · $0.0006 (90 tok)' while Console and [chat_defaults] use gpt-4.1-mini; mock log model=gpt-5.6-terra; app log 'dropping explicit temperature/top_p for reasoning model'. Code: resolve_library_rag_answer_provider returns (default_api_endpoint, None) by design (Library/library_rag_answer_service.py:163-189), so the handler default (config.py:2753) applies. Severity: j2/j7 P2, j4 P1; normalized to P1 because PRODUCT.md requires authority and cost visible before action and the substituted model is a reasoning model the user never chose.
- **Repro:** Set [chat_defaults] model=gpt-4.1-mini > Search / RAG > RAG Answer > ask > read the provenance line and the request log.
- **Why it matters:** Users pay for a model they did not pick and learn of it only afterwards.
- **Fix:** Resolve the model from [chat_defaults] when its provider matches (or a [rag] answer_model), show 'To OpenAI · gpt-4.1-mini: question + evidence' before Run, and add a model chooser next to the mode toggle.

<a id="l-07"></a>
### L-07 · P1 · 'Add analysis' leaves focus on the Items row, so typing fires single-letter list commands

- **Surface / kind:** Media · defect · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H3, H5
- **Evidence:** j4 25 (ANSI): after Enter on 'Add analysis' the █ bar stays on 'Spaced repetition, explained' and the TextArea is unfocused; typing 'Key takeaway: spacing…' opened Import media at the 'i' and lost the text (23). Code: UI/Library_Modules/library_media_controller.py:4607-4617 sets the editing flag and re-syncs but never focuses #library-media-analysis-edit-text.
- **Repro:** Media > open an item > Analysis > Tab to 'Add analysis' > Enter > type a sentence containing i, t, c, s or l.
- **Why it matters:** Keyboard users lose their text and can arm trash, hand off to Console, start Import or toggle read-later by accident.
- **Fix:** After the sync in handle_library_media_analysis_edit, call_after_refresh to focus the TextArea; do the same for Edit metadata (focus Title).

<a id="l-08"></a>
### L-08 · P1 · Escape in the analysis editor discards typed text without asking

- **Surface / kind:** Media · defect · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H3, H5
- **Evidence:** j4 26-27: a typed takeaway, Esc -> 'No analysis yet.'; the footer while typing read 'esc close'. Code: action_library_media_viewer_back sets editing_analysis=False with no dirty check (UI/Screens/library_screen.py:30950-30990, :30986). The Prompts editor vetoes the same case ('esc save or discard first').
- **Repro:** Analysis > Add analysis > focus the editor > type a paragraph > Esc.
- **Why it matters:** A reflex keypress destroys authored content; inconsistent with Prompts.
- **Fix:** When the TextArea differs from the saved analysis, the first Esc only blurs and shows 'Unsaved analysis — Save or Discard'; change the footer chip to match.

<a id="l-09"></a>
### L-09 · P1 · Conversations filter leaves the Reader on a conversation that is no longer listed; 'c'/Resume act on it

- **Surface / kind:** Conversations · defect · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H5
- **Evidence:** j4 42: '1 match for 'kubernetes'' listed while the Reader shows 'Loaded Thesis: chapter 2 structure · 40 of 40 messages'; 43: Esc Esc c opened Console on 'Thesis: chapter 2 structure'. Code: filter submit only requests page 1 and never re-points or clears the reader (UI/Library_Modules/library_conversations_controller.py:1434-1446); Media does re-point (library_screen.py:16635-16642).
- **Repro:** Library > Conversations (Thesis auto-loads) > / kubernetes Enter > Esc Esc > c.
- **Why it matters:** Resume, Use as source and Archive silently hit a different conversation than the one the list shows.
- **Fix:** When the loaded conversation is not in the new result set, load the first match or clear the Reader to 'Select a conversation…'; gate c / Resume / Use as source / Archive on the loaded item being visible.

<a id="l-10"></a>
### L-10 · P1 · 'Try report demo' creates a persistent live-RSS watchlist with a daily paid run, disclosed only afterwards

- **Surface / kind:** Artifacts · usability · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H3, H5
- **Evidence:** j4 67: toast 'Daily brief ready: … It refreshes daily from now o[n]'; the report cites Ars Technica. Subscriptions/daily_report_demo.py:1-60 seeds hnrss.org, BBC, Ars Technica with DEMO_CADENCE_SECONDS=86_400. j4 66: the empty-state sentence 'Open Watchlists or try the report demo.' is at row 12 while the buttons are at row 50 in the other pane.
- **Repro:** Library > Reports (empty) > Try report demo.
- **Why it matters:** 'Demo' implies no commitment; it actually adds recurring network egress and recurring LLM spend.
- **Fix:** Rename to 'Set up a daily brief…' and confirm first ('Creates a Daily Brief watchlist (3 RSS feeds), fetches now, calls <provider · model> daily'); place the buttons directly under the empty-state sentence.

<a id="l-11"></a>
### L-11 · P1 · Built-in skill 'Enabled' switch shows no state; the label still reads 'Enabled' when off

- **Surface / kind:** Skills · a11y · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H4, WCAG 1.4.1
- **Evidence:** j4 59-60 (ANSI): the switch interior '▊        ▎' is empty in both states; only the background of cols 132-137 changes; ' Enabled ' never changes; only the list row gains '· Disabled' and config disabled_builtins=['character-creator']. Code: Widgets/Library/library_skill_work_pane.py:169-175; library_skills_canvas.py:12-22 already notes Switch renders unreliably here.
- **Repro:** Skills > character-creator (Built-in) > Tab to the switch > Space; compare captures.
- **Why it matters:** State is carried by colour alone (contrary to PRODUCT.md); a user can disable the agent's built-in without seeing it.
- **Fix:** Replace the Switch with the house text toggle 'Enabled: ✓ on ⇄ off' (the 'User can invoke: ✓ yes ⇄ no' pattern).

<a id="l-12"></a>
### L-12 · P1 · At 60–80 columns Media, Skills and Conversations open to an empty Reader with the list collapsed

- **Surface / kind:** Library-shell · defect · confidence high · live-reproducible
- **Sources:** B-mechanical · **Personas:** Sam
- **Heuristics:** H1, H6
- **Evidence:** B caps/media-list-80x24: arrival after a rail click shows only grips 'N a v' / 'I t e m s' and 'Select a media item to read it here.'; 0 items visible, 84% blank, footer still offers 's select'. skills-60x24/80x24: 'Select a skill to inspect it here.', 0 skills, 81-84% blank. conversations-60x24/80x24: list collapsed, chips/Export/Select missing, 0 transcript rows. Notes and Prompts open list-first at the same sizes. Code: only Notes (UI/Screens/library_screen.py:6668) and Prompts (library_prompts_controller.py:801) request priority='items'; Skills only above a width floor (library_skills_controller.py:812-818); Media list and Conversations request none.
- **Repro:** Golden 80x24 > ⌃3 Library > click 'Media (23)' (or 'Skills (4)' at 60x24).
- **Why it matters:** On a standard 80x24 terminal the destination opens with nothing to pick and only tiny vertical grips as a way forward.
- **Fix:** Request priority='items' for the Media list view and Conversations when nothing is open (mirroring library_prompts_controller.py:801), drop Skills' width-floor condition, and add a test that for W in [64, 98] the list pane is open on arrival for every browse route.

<a id="l-13"></a>
### L-13 · P2 · Search mode returns 'No evidence matched' for a plain-language question the documents answer verbatim; the rail box silently flips the mode

- **Surface / kind:** Search-RAG · usability · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H4, H9
- **Evidence:** j2 20-21, 50-52: placeholder 'Ask or search Library sources'; in Search mode 'How much did retrieval practice improve retention?' -> 'No evidence matched … Try broader terms.' though the PDF says 'Retrieval practice improved 7-day retention by 21 percentage points'; RAG Answer retrieves it as '#1 match: strong'. Submitting from the rail 'Search Library…' box flipped '✓ RAG Answer' to '✓ Search'.
- **Repro:** Library > rail 'Search Library…' > type a natural-language question about an imported doc > Enter.
- **Why it matters:** A first-timer asks in plain language, gets 'no evidence', and concludes the import failed.
- **Fix:** When keyword search returns 0 for a 4+ word query, fall back to the hybrid retrieval the RAG path uses and label it 'No exact matches — showing related passages'; add an inline 'Ask this as a question (RAG Answer)' button; make the rail box keep the canvas's current mode.

<a id="l-14"></a>
### L-14 · P2 · In-document Find counts lines, not matches, and marks nothing in the default Rendered view

- **Surface / kind:** Media · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H1, H6
- **Evidence:** j2 15-16 (rowstyles): in 'Rendered (selected)', '7-day retention' shows 'Match 1 of 2' while every row stays plain '#e0e0e0 on #242f38'; the content has 3 occurrences; Raw reverse-videos only the first per line. Code: Widgets/Library/library_media_content.py:465 documents matches as 'Source-line indexes'; _status_text (:544-550) counts them.
- **Repro:** Open the imported PDF > Find > '7-day retention' > Enter > look for highlights > switch to Raw.
- **Why it matters:** The user is told a match exists but cannot see where, and the count is wrong whenever a line holds two.
- **Fix:** Track (line, column) occurrences, highlight each, and step through them; while a query is active in Rendered, switch to Raw (as Analysis already does) with 'Showing raw text to mark matches'.

<a id="l-15"></a>
### L-15 · P2 · Reader Find needs Enter and keeps showing the previous query's count

- **Surface / kind:** Media · usability · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H4
- **Evidence:** j4 19: field 'caution' (6 occurrences) with status 'Match 1 of 8' from 'evidence'; 20: after Enter, 'Match 1 of 6'. Placeholder 'Search content…' does not mention Enter. Code: Widgets/Library/library_media_content.py:451-488 updates only on submit.
- **Repro:** Open a media item > Ctrl+F > 'evidence' Enter > replace with 'caution' without Enter.
- **Why it matters:** The status lies about the query on screen; the list filter a few columns away updates as you type.
- **Fix:** Run Find on input change (debounced like the list filter), or hide the status when the input differs from the submitted query and show 'Enter to search'.

<a id="l-16"></a>
### L-16 · P2 · Returning to Media via the rail shows a row marked 'loaded' beside an empty Reader

- **Surface / kind:** Media · defect · confidence medium · live-reproducible
- **Sources:** j4-poweruser-library, j7-researcher-loop · **Personas:** Morgan, Priya
- **Heuristics:** H1
- **Evidence:** j4 24, 68 (twice) and j7 20-22: '▸ paper-retrieval-practice · pdf · updated 4m · loaded' while the Reader says 'Select a media item to read it here.' (still after 7 s). Hypothesis: the rail-row reset (UI/Screens/library_screen.py:22223-22410) closes the viewer but keeps the row's loaded flag. (The loss of reading position itself is part of S-02.)
- **Repro:** Media > open an item > press another rail row > press Media again.
- **Why it matters:** A false status about what is open; c or t then act on nothing or on the wrong assumption.
- **Fix:** Clear the row's 'loaded' fact when the viewer resets, or re-hydrate the reader from the selected row (preferred, with S-02).

<a id="l-17"></a>
### L-17 · P2 · 'Select evidence' moves focus onto a Sources checkbox and can jump the panel back to the top

- **Surface / kind:** Search-RAG · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H3, H5
- **Evidence:** j2 28, 53: after 'Select evidence' on card 2 the panel scrolled to the query field and the footer read 'enter toggle Media' (another time 'enter toggle Conversations'); the next Enter would turn that source off. Hypothesis: _refresh_search_rag_panel_state_widgets() rebuilds the cards (library_rag_search_controller.py:1103-1122) and focus falls to the next focusable.
- **Repro:** Search / RAG > query with ≥2 results > scroll to card 2 > Select evidence > read the footer and position.
- **Why it matters:** Users lose their place and the next Enter silently changes search scope.
- **Fix:** Restore focus by id to the same card's (now 'Selected evidence') button and keep the scroll offset, or patch the label in place without rebuilding the card.

<a id="l-18"></a>
### L-18 · P2 · Export canvas promises 'copies full media files' but writes text only, with opaque media_N.txt names; a second same-day export overwrites the first by default

- **Surface / kind:** Export · copy · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j4-poweruser-library · **Personas:** Jordan, Morgan
- **Heuristics:** H2, H5
- **Evidence:** j2 35-38 and j4 14: 'quality: original · copies full media files into the zip' two lines above 'Bundle: 5 items · text only · size known once it runs'; unzip lists only README.md, manifest.json, content/media/media_1..5.txt and metadata JSON, and 'size known once it runs' never updates. j4 49: a second same-day export defaults to the same file and says 'Overwrites Library export 2026-10-02.zip' (warned, but one Enter away). Code: Library/library_export_state.py:63-97, 347.
- **Repro:** Export any media selection > read the quality caption > export > unzip -l > export again with defaults.
- **Why it matters:** Users believe they backed up their originals when they hold only extracted text, and one default Enter replaces today's earlier bundle.
- **Fix:** For text-only bundles hide or disable 'quality:' and say 'Text and metadata only — original files are not stored in this Library'; name entries by slugged title; show the real size after the run; default the name with a time ('Library export 2026-10-02 2035.zip').

<a id="l-19"></a>
### L-19 · P2 · Media selection is page-local and cleared by any page turn; there is no bulk keyword action

- **Surface / kind:** Media · missing-capability · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H3, H7
- **Evidence:** j4 08-09: '3 selected   Select all 20 shown'; after Next, toast 'Selection cleared.' and select mode exits; toolbar offers Clear/Export/Review/○ Analyze/Delete only. Code: _clear_library_media_selection_for_scope_change (UI/Screens/library_screen.py:16855-16873). j4 rated P1; normalized to P2 because per-page triage works.
- **Repro:** Media (21+ items) > s > Space on 3 rows > Next.
- **Why it matters:** Triage across more than one page or filter needs repeated round trips; tagging is one item at a time via More > Edit metadata.
- **Fix:** Keep the selection as an id set across pages and filters with an 'N selected (M on this page)' chip (as Prompts does), and add 'Keywords…' (add/remove) to the select toolbar.

<a id="l-20"></a>
### L-20 · P2 · Imports are titled by filename stem, ignoring PDF Title/Author, HTML <title> and Markdown H1

- **Surface / kind:** Ingest · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j7-researcher-loop · **Personas:** Jordan, Priya
- **Heuristics:** H2
- **Evidence:** j7 05-06, 55: 'paper-retrieval-practice' though the PDF metadata title is 'Retrieval practice improves long-term retention' with an author; j2: 'web-page-spaced-repetition' / 'article-local-first' instead of the page <title> or '# The case for local-first software'. Code: Local_Ingestion/local_file_ingestion.py:1066-1067 sets title = file_path.stem before processing, while PDF_Processing_Lib.py:677 prefers metadata only without an override (ordering inferred).
- **Repro:** Import fixtures/import-docs/paper-retrieval-practice.pdf (and the .html/.md) with Title blank.
- **Why it matters:** Every paper appears under its filename, which is useless for citing and scanning.
- **Fix:** Leave the title unset until the processor returns, then prefer PDF Title / HTML <title> / first H1, falling back to the filename; fill Author from metadata.

<a id="l-21"></a>
### L-21 · P2 · Local HTML import keeps site navigation, duplicates the heading and flattens tables

- **Surface / kind:** Ingest · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H2, H8
- **Evidence:** media_v2.db id 1 ends with 'Site navigation that should be stripped'; 'Spaced repetition, explained' appears twice; the table is one cell per line (Interval / Days / 1 / 1 / 2 / 3); the boilerplate appears in RAG snippets (j2 52) and in the export.
- **Repro:** Import fixtures/import-docs/web-page-spaced-repetition.html > open it > RAG for 'spaced repetition' > read the snippet.
- **Why it matters:** Boilerplate pollutes retrieval and answers.
- **Fix:** Run local HTML through the same readability extraction used for URL imports (drop nav/header/footer/aside; keep table rows as 'a | b').

<a id="l-22"></a>
### L-22 · P2 · Reader 'More ▸ Open original' and 'Open manager' do nothing for a local import

- **Surface / kind:** Media · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j3-poweruser-notes · **Personas:** Jordan
- **Heuristics:** H1, H9
- **Evidence:** j2 66-67: neither click changes anything or shows a message. Code: library_media_controller.py:4196-4202 opens only http(s), but local imports store file://…; 'Open manager' navigates from Library to Library. j3 08 lists the same More menu (Edit metadata, Open original, Open manager, Move to trash).
- **Repro:** Import a local PDF > open it > More > Open original; then More > Open manager.
- **Why it matters:** Dead controls teach that buttons in this app may silently do nothing.
- **Fix:** For file:// sources open with the OS handler (open/xdg-open) or reveal in Finder, else hide the action; remove 'Open manager' from the Library reader.

<a id="l-23"></a>
### L-23 · P2 · Search/RAG Sources toggles take ~3 rows each and push the answer or evidence below the fold at 100x30 and 120x36

- **Surface / kind:** Search-RAG · usability · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j4-poweruser-library · **Personas:** Jordan, Morgan
- **Heuristics:** H1, H8
- **Evidence:** j2 57-59 (100x30): after Run, only the 'Answer' heading shows on the last row; 160x45 shows ~8 answer rows. j4 76 (120x36): query rows 7-17, four Sources toggles rows 20-31 with blank rows between, 'Evidence · top 15 per source' only after 8 wheel ticks. j4 rated P3, j2 P2.
- **Repro:** Resize to 100x30 > Search / RAG > RAG Answer > ask > Enter.
- **Why it matters:** Users cannot tell anything happened and must scroll past configuration after every run.
- **Fix:** Put the Sources toggles on one line ('☑ Notes 122 ☑ Media 21 ☐ Conversations 11 ☐ Prompts 10'), drop the blank rows around Run, and scroll the Answer/Evidence heading to the top when results arrive.

<a id="l-24"></a>
### L-24 · P2 · Search/RAG silently keeps the previous query's source toggles

- **Surface / kind:** Search-RAG · usability · confidence high · live-reproducible
- **Sources:** j7-researcher-loop · **Personas:** Priya
- **Heuristics:** H1, H5
- **Evidence:** j7 48: a rail search for 'retrieval practice' returned '2 results' (Media only); the line 'Scope: Media (Notes, Conversations, Prompts off)' was above the viewport; re-enabling sources gave '5 results' (49-50).
- **Repro:** Run a RAG Answer with only Media on, then search a topic from the rail box.
- **Why it matters:** The user concludes her notes and conversations do not match and misses her own material.
- **Fix:** Show the scope in the results header ('2 results · Media only — Search all sources') with one-click reset; reset to all sources when a search starts from the rail box.

<a id="l-25"></a>
### L-25 · P2 · The Media reader is the narrowest pane, and an absolute file:// path takes its first 4–5 lines

- **Surface / kind:** Media · usability · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j7-researcher-loop · **Personas:** Jordan, Priya
- **Heuristics:** H8
- **Evidence:** j2 11, 47: at 160 cols the reader is ~54 cols (rail 34 + list 56), text wraps at ~50; a 4-row (5 at 100x30) byline 'file:///private/tmp/…/paper-retrieval-practice.pdf' leaves ~12 document lines at 100x30. j7: ~50 cols and ~10 body rows at 120x36.
- **Repro:** Import a local PDF > open it at 160x45 and 100x30.
- **Why it matters:** Reading - the reader's main job - gets the least space, and the path is noise that exposes the local filesystem.
- **Fix:** Show the byline as 'paper-retrieval-practice.pdf · local file' (full path in Info); auto-collapse the Nav rail once when a document opens, as Notes already does.

<a id="l-26"></a>
### L-26 · P2 · Prompt 'Use in Console' drops Instructions by default and passes {{notes}} through as literal text

- **Surface / kind:** Prompts · usability · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H2, H4, H5
- **Evidence:** j4 55: 'Prompt variables' dialog with no fields; 57: composer got '…Action Items:\n{notes}', status 'System Prompt: off'. Code: Prompt_Management/prompt_variables.py:376-383 treats '{{' as an escape; prompt_variables_dialog.py:120-127 defaults the System checkbox off.
- **Repro:** Prompts > 'Summarize meeting notes' ({{notes}}) > Use in Console > Apply > expand the pasted chip.
- **Why it matters:** The rest of the app and imported templates use {{name}}; the prompt silently loses its variable and its Instructions.
- **Fix:** Show detected variables in the editor ('Variables: {name}'); in the dialog warn on {{name}} ('Did you mean {name}?'); label the lane 'Instructions' and default it on when the prompt has Instructions.

<a id="l-27"></a>
### L-27 · P2 · Recent searches replays an entry under the current mode, silently making a paid RAG call

- **Surface / kind:** Search-RAG · usability · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H5
- **Evidence:** j4 37: clicking 'retrieval practice' (a free Search) while in RAG Answer raised the mock request count 11 -> 12 (gpt-5.6-terra). The history row shows neither mode nor cost. Code: library_search_rag_panel.py:1329-1338.
- **Repro:** Run a Search > switch to RAG Answer > expand Recent searches > click the earlier entry.
- **Why it matters:** A one-click cost the user did not choose, from what looks like a free recall list.
- **Fix:** Store mode and scope with each history entry and replay under the saved mode; mark RAG entries 'RAG · paid'.

<a id="l-28"></a>
### L-28 · P2 · Skill 'Customize' hides the built-in and leaves a copy the agent cannot use, without saying so first

- **Surface / kind:** Skills · usability · confidence high · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H1, H3
- **Evidence:** j4 61-62: the work pane resets to 'Select a skill…'; the list shows '› ⚠ character-creator · needs review · overrides built-in' with no built-in row; 'Trust: not initialized'. Code: library_skills_builtin_controller.py:148-197. The 'needs review' part is amplified by the harness's null keyring, but the hidden built-in, the lost selection and the missing up-front warning are product behaviour.
- **Repro:** Skills > character-creator (Built-in) > Customize.
- **Why it matters:** A working built-in is replaced by a copy that may not run, and the original cannot be compared or restored from the list.
- **Fix:** Open the new copy in Edit; keep the built-in row as 'Built-in · overridden by your copy' with 'Reset to built-in'; before copying, state 'Your copy needs trust review before the agent can use it'.

<a id="l-29"></a>
### L-29 · P2 · Selecting an item in Media Trash leaves an unrelated live item in the reader

- **Surface / kind:** Media · usability · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H1
- **Evidence:** j6 79: selected row 'Old draft — delete candidate' while the reader shows 'bulk-59' with Find / Read later / Use in Console; Enter does not change it.
- **Repro:** Media > Trash > click 'Old draft — delete candidate' > Enter.
- **Why it matters:** Users may act on or permanently delete the wrong item, believing the reader matches the selection.
- **Fix:** Show a read-only preview of the trashed item, or clear the reader to 'Trashed · Restore to read'.

<a id="l-30"></a>
### L-30 · P2 · Batch media import has no cancel, and Enter imports while the footer says 'check this path'

- **Surface / kind:** Ingest · usability · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H3, H4
- **Evidence:** j6 67b: 'Import finished — 1 imported' after Enter with the footer 'enter check this path'; 68: 'bulk — 60 files · active · 3 parsing · 1 writing · 56 queued' with no Cancel; Esc leaves while it runs to 60/60. Code: can_cancel only for active local STT jobs (Library/library_ingest_state.py:2240-2250; Widgets/Library/library_ingest_canvas.py:924-933).
- **Repro:** Library > Import… > path of a folder with 60 .md files > Enter > look for Cancel.
- **Why it matters:** A mistaken folder import cannot be stopped, and the footer understates what Enter does.
- **Fix:** Relabel the chip 'enter import'; add a batch 'Cancel remaining (N)' that marks queued jobs cancelled; confirm before importing folders with more than 10 files.

<a id="l-31"></a>
### L-31 · P2 · Conversations reader stacks 16–20 rows of chrome above the first message

- **Surface / kind:** Conversations · usability · confidence high · live-reproducible
- **Sources:** B-mechanical
- **Heuristics:** H8
- **Evidence:** B: first transcript row at r26 of 30/36/45 rows (0 visible at 60x24 and 80x24, 3 at 100x30, 5 at 120x36, 8 at 160x45, 12 at 235x52). Above it: 'Read  Info', 'Resume conversation', 'Use as source' plus a 2-3 row workspace explanation, 'Archive conversation', 'Link to workspace', a 2-3 row 'Loaded … 40 of 40 messages · complete.', and a 3-row Find field with Find previous/next.
- **Repro:** Golden 120x36 > Library > Conversations (12).
- **Why it matters:** The reader exists to read the transcript; on 24-row terminals none of it is visible.
- **Fix:** Put Resume / Use as source / Archive / Link to workspace on one toolbar row; move the workspace explanation into Use as source's disabled reason; fold Find behind ctrl+f as the Media reader does; shorten the status to '40 of 40 messages · complete'.

<a id="l-32"></a>
### L-32 · P2 · Prompts list gets half the pane; the other half is blank

- **Surface / kind:** Prompts · defect · confidence high · live-reproducible
- **Sources:** B-mechanical, j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H8
- **Evidence:** B caps/prompts-160x45: 5 of 10 prompts in a scrolling list, '1-10 of 10', then 14 blank rows (120x36: 3 of 10, 9 blank; 235x52: 6 of 10, 13 blank). j5 81: 3 of 10 visible with ~10 empty rows below. Code: #library-prompts-pager is a Vertical (Widgets/Library/library_prompts_canvas.py:1082) with default height:1fr, splitting space equally with VerticalScroll#library-prompts-list (:1021).
- **Repro:** Golden 160x45 > Library > Prompts (10).
- **Why it matters:** Users scroll a half-height list beside empty space; a small set should be visible at a glance.
- **Fix:** Add '#library-prompts-pager { height: auto; }' to css/features/_library_panels.tcss and rebuild the bundle with css/build_css.py.

<a id="l-33"></a>
### L-33 · P3 · First import silently downloads an embedding model and paints a Hugging Face warning over the nav bar

- **Surface / kind:** Ingest · defect · confidence medium · live-reproducible
- **Sources:** j7-researcher-loop, j1-firsttimer-notes · **Personas:** Priya, Jordan
- **Heuristics:** H1, H8
- **Evidence:** j7 06: the nav row was overwritten by '… [WARNING ] huggingface_hub.utils._http:904 - Warning: You are sending unauthenticated requests to the HF Hub…'; the log shows '_build: Loaded model default in 15.74s' after the import. j1's log shows the same 15.69 s model load during a Console Library search (see S-20). Cause hypothesis: a WARNING-level terminal sink survives TUI start (Utils/startup_logging.py:53 or tldw_chatbook/__init__.py:97).
- **Repro:** Fresh profile with an empty HF cache: import any PDF and watch the top rows for ~15 s.
- **Why it matters:** Undisclosed network egress in a local-first product, and raw log text corrupts the UI.
- **Fix:** Remove terminal log sinks once the App is running so third-party warnings go to the log file only; show 'Preparing search index — one-time model download from Hugging Face' on the import queue row, with an option to defer.

<a id="l-34"></a>
### L-34 · P3 · 'Keep for later' is split across three unconnected places, and the Quick Capture note box is unlabeled

- **Surface / kind:** Collections · usability · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H4, H6
- **Evidence:** j2 31-34: Media 'Read later' is not shown on list rows and is absent from Collections ▸ 'Reading'; reachable only via Media ▸ 'Sets' ▸ 'Review read-later' (an overlay that clips the reader). Quick Capture accepts only http(s) ('Enter a valid http or https URL before saving.'); its 4-row TextArea has no label or placeholder (library_collections_capture_reader.py:438-441).
- **Repro:** Media > open item > Read later > Collections ▸ Reading (empty) > Quick Capture.
- **Why it matters:** A first-timer cannot find what they saved and does not know what the unlabeled box is for.
- **Fix:** Add a '· later' fact on list rows and a 'Read later (N)' scope under Media; label the text area 'Note (optional)'; in the Collections empty state explain 'Captures are web pages saved by URL. To keep a Library file for later, use Read later in Media.'

<a id="l-35"></a>
### L-35 · P3 · A fully successful import still offers 'Retry this batch' and footer 'r retry'

- **Surface / kind:** Ingest · copy · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H2
- **Evidence:** j2 09-10: 'This queue: 5 done' with footer 'r retry' and button 'Retry this batch', which re-stages the batch into the form (library_ingest_canvas.py:2079-2092).
- **Repro:** Import a folder where every file succeeds > read the queue bottom and footer.
- **Why it matters:** A literal reader assumes something failed.
- **Fix:** After an all-success batch rename the button 'Import these again…' (or hide it) and show 'r retry' only when a row failed.

<a id="l-36"></a>
### L-36 · P3 · Help and empty states never say what Library, Prompts or Skills are

- **Surface / kind:** Library-shell · copy · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library · **Personas:** Jordan
- **Heuristics:** H2, H10
- **Evidence:** j2 03, 40, 41: F1 on the landing shows 'Library Shortcuts — Landing' with 3 shortcuts in a ~30-row empty frame; Prompts empty state 'No prompts yet. Create or import a prompt to begin.'; Skills shows 'Skill trust isn't set up, so every skill you added reads "needs review"…' with zero user skills, and its row reads 'invocable: user & agent'. (The trust banner is partly a harness caveat - null keyring - but showing it with zero user skills is product behaviour.)
- **Repro:** Empty profile > Library > F1; then rail Prompts (0) and Skills (1).
- **Why it matters:** A newcomer cannot learn the destination's purpose from the destination itself.
- **Fix:** Add one purpose line to F1 and the landing ('Library holds everything you've imported, chatted or written, so you can find it, ask about it, and send it to Console.'); Prompts empty state 'A prompt is reusable instruction text you can insert into Console.'; show the trust banner only when a user skill exists; render 'Can be run by: you or an agent'.

<a id="l-37"></a>
### L-37 · P3 · Restore and archive reset 'updated' to now, reordering Newest and the landing

- **Surface / kind:** Media · usability · confidence medium · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H2
- **Evidence:** j4: after restore, 'Server logs excerpt 2026-09-27 · plaintext · updated just now' tops Newest and the landing 'From your Library'; an archived conversation reads '… 8m · … · Archived'.
- **Repro:** Trash an old item, restore it, return to Media (Newest) and the landing.
- **Why it matters:** Housekeeping pollutes recency, so 'recent' no longer means recently worked on.
- **Fix:** Keep last_modified for content edits; store trash/archive state times separately and sort Newest by content time.

<a id="l-38"></a>
### L-38 · P3 · Media scope-line 'Clear' also resets the sort

- **Surface / kind:** Media · usability · confidence medium · live-reproducible
- **Sources:** j4-poweruser-library · **Personas:** Morgan
- **Heuristics:** H3, H4
- **Evidence:** j4 07: 'sort: Title A-Z' becomes 'sort: Newest' after the scope-line Clear. Hypothesis: handle_library_media_scope_clear (library_media_controller.py:2328-2335) calls _clear_library_media_filter(clear_type=True) and the downstream request resets sort.
- **Repro:** Media > filter + type + sort Title A-Z > scope-line Clear.
- **Why it matters:** An unlabelled side effect; users re-sort after every reset.
- **Fix:** Keep the sort on Clear, or relabel the button 'Reset view' and say it resets type, filter and sort.


## Shell and cross-destination

<a id="s-01"></a>
### S-01 · P1 · Media 'Use in Console' is always refused for imported items ('Copy or link this media into workspace workspace-default…') with no in-app remedy, while Search evidence stages the same item

- **Surface / kind:** Cross-destination · defect · confidence high · live-reproducible
- **Sources:** j2-firsttimer-library, j4-poweruser-library, j7-researcher-loop · **Personas:** Jordan, Morgan, Priya
- **Heuristics:** H2, H4, H9
- **Evidence:** j2 24-25, 64-65; j4 71; j7 26-28: toast 'Copy or link this media into workspace workspace-default before using it in Console.'; rail Details 'Handoff · N items can't be used in Console yet · not in this workspace · Copy or link them into this workspace' with no link/copy action anywhere for media; footer still offers 'c use in Console'. The workspaces DB holds workspace-default (active=1, created at first boot) and 0 memberships; registry.link_membership is called only for conversations (UI/Screens/library_screen.py:13906), notes (library_notes_controller.py:4494/4594-4616) and chat persistence, never media. Gate: Workspaces/eligibility.py:73-82 (raw id interpolated at :79) via display_state.py:908-948. The same paper stages fine via Search ▸ Select evidence ▸ Use in Console / 'u' (j2 29, j4 39, j7 35), and the Import screen promises items 'can be used as context in chat'. j4 rated P2 (medium); normalized to P1 because the PRODUCT.md core loop dead-ends on the most obvious button for every fresh user.
- **Repro:** Empty or golden profile > Library > Import any file > open it in the Media reader > Use in Console (or c).
- **Why it matters:** Ingest -> reason in Console is the product's core loop; the recovery copy names an action that does not exist and an id the user has never seen.
- **Fix:** Treat built-in workspace-default like the 'no active workspace' branch (Workspaces/display_state.py:582-600), or mirror notes: link media to the active workspace on hand-off and report 'Linked to Default · staged in Console'. Failing that, render an inline 'Link to Default and use' button next to Use in Console. Always show the workspace display name, never 'workspace-default'. Make Search 'u' and reader 'c' obey the same rule.

<a id="s-02"></a>
### S-02 · P1 · Reader and note can never be visible together; every rail switch discards the reading position and the open note, and Media has no capture key

- **Surface / kind:** Cross-destination · usability · confidence high · live-reproducible
- **Sources:** j7-researcher-loop, j3-poweruser-notes · **Personas:** Priya, Alex
- **Heuristics:** H3, H6, H7
- **Evidence:** j7 12-13, 20-22, 56: rail 'New note' or 'Notes (N)' replaces the Media reader in the single work pane; back on Media the reader is empty (see L-16), Enter reloads on the last tab; back on Notes 'Select a note to edit it here.'. One round trip costs ~8 clicks + 5 keys plus re-scrolling. j3 08-10: in the Media list and reader, n and Ctrl+N do nothing with no feedback; the palette route leaves Media and returning shows 'Select a media item to read it here.'; More has no 'Note on this item'.
- **Repro:** 160x45: import a PDF, Open in Library > rail Notes (N) > open a note, type > '--->' > Media: reader blank > Notes: note closed.
- **Why it matters:** Note-taking while reading is the researcher's and PKM user's core loop; both reading position and editing context are lost on every switch.
- **Fix:** Add 'Take note' (key n) to the Media reader that opens a note editor beside the reader (split the work pane at ≥140 cols, one-key toggle below) pre-linked to the item; keep per-rail-row state (open media id, tab, scroll; open note id, caret) across switches; Esc from a capture returns to the originating reader.

<a id="s-03"></a>
### S-03 · P1 · No way to quote a passage into a note with a link back to the source

- **Surface / kind:** Cross-destination · missing-capability · confidence high · live-reproducible
- **Sources:** j7-researcher-loop · **Personas:** Priya
- **Heuristics:** H6, H7
- **Evidence:** j7 11: a highlight card's only action is '✕ Delete'; 23-25: drag-select + Ctrl+C gives no feedback and the pasted text has no attribution; 15: note Info has no Source field; 55: media Info shows 'Canonical ID: local:media:26' but notes support only note:// (media:// exists only in MCP/resources.py).
- **Repro:** Reader > Highlights > Add highlight > look for 'to note'; select text in Read > Ctrl+C > paste in a note.
- **Why it matters:** Provenance is lost at the moment of capture; quotes cannot later be traced to the paper.
- **Fix:** Add 'Quote to note…' on a reader selection and on each highlight card, inserting '> quote' + '— [title](media://26)' into a chosen or new note and recording a note↔media relation; render media:// links in note Preview to open the reader; list 'Notes citing this item' in media Info; toast 'Copied N characters' and show 'ctrl+c copy' when a selection exists.

<a id="s-04"></a>
### S-04 · P1 · Kept AI answers lose provenance (no question, no source, UUID keywords), and Library RAG answers cannot be saved

- **Surface / kind:** Cross-destination · usability · confidence high · live-reproducible
- **Sources:** j7-researcher-loop · **Personas:** Priya
- **Heuristics:** H1, H9
- **Evidence:** j7 39-40: Console 'Capture as note' writes the answer text as title and body with keywords 'console, conversation:71d30fb1-a206-…, message:1f56b655-…'; nothing names the paper or the question; filtering '71d30fb1' gives 0 results (64, see N-09). 32: the Library RAG Answer has no save action. Code: UI/Console_Modules/message.py _capture_console_answer_as_note (content = message only).
- **Repro:** Stage the paper via Search evidence > ask in Console > More… > Capture as note > Open note > read Keywords.
- **Why it matters:** A kept answer cannot be traced to the paper it was grounded in; the only provenance is unsearchable and dropped on export (L-04).
- **Fix:** Write a provenance block on capture (question, model, date, 'Sources:' with media://, note:// or conversation links for each staged or cited item); show 'From conversation: <title>' as a link, not a UUID keyword; add 'Save answer as note' to the RAG Answer panel with the same block.

<a id="s-05"></a>
### S-05 · P1 · Study hand-off dead-ends: a whole-Library 'ready' snapshot, invisible Study dashboard controls, and server-only generation

- **Surface / kind:** Cross-destination · defect · confidence high · live-reproducible
- **Sources:** j7-researcher-loop · **Personas:** Priya
- **Heuristics:** H1, H4
- **Evidence:** j7 41: 'Carries forward: … and 134 more · Source snapshot is ready · Continue in Study' (Study itself shows '+7 more', STUDY_MATERIAL_TITLES_LIMIT = 10). 44-46: the Study Dashboard ends at '0 due today'; 'Resume last session / Open flashcards / Open quizzes / Generate source pack' never render, even at 160x70. UI/Screens/study_screen.py:620-625 returns 'Source generation requires server mode.' Cause hypothesis: height:1fr with overflow hidden on the Horizontal at Widgets/Study/study_dashboard.py:73 and .card-editor (UI/Study_Window.py:117-121). The empty 'Decks:' box (47) is partly a harness caveat (no decks seeded).
- **Repro:** Library rail > 'Flashcards due: 0' > Continue in Study > Dashboard, then Flashcards, at 160x45 and 160x70.
- **Why it matters:** Making study material from a paper is a named job for this persona; every path ends blank while Library claims 'ready'.
- **Fix:** Set height:auto on the dashboard columns row and .card-editor; in local mode replace 'Source snapshot is ready' with 'Generating from sources needs a server; make cards by hand in Study'; add 'Make flashcards from this note / paper' scoped to one item using the configured provider.

<a id="s-06"></a>
### S-06 · P1 · A second 'Use in Console' silently replaces the first staged note

- **Surface / kind:** Cross-destination · defect · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H1, H4
- **Evidence:** j3 48-49: after note A, 'Staged for next send · 1 source · Thesis outline — …'; after note B, '· 1 source · Trip planning: Lisbon — note' and Inspect 'Sources: 1 staged'. Select mode has no Use in Console. Hand-offs use the single HandoffChannel.CHAT slot (app_destinations.py:175-197).
- **Repro:** Note A > Use in Console > back to Library > note B > Use in Console > read the staged strip.
- **Why it matters:** Context built from several notes silently drops all but the last, so the send is under-grounded.
- **Fix:** Append to the staged sources, de-duplicated by id (or ask 'Replace or add?'), and add 'Use in Console (N)' to Notes select mode.

<a id="s-07"></a>
### S-07 · P1 · Closing F1 or any modal drops keyboard focus on the landing, Notes, Conversations and Skills

- **Surface / kind:** Library-shell · a11y · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H3, WCAG 2.4.3
- **Evidence:** j5 06-09: landing focus moves from the recent row to the nav bar's '⌃1 Home'; 37-40: Notes focus on '┃ Undo ┃' moves to Filter; 77-79: Conversations focus lands on the Nav grip; 97: Skills Tab walks the nav bar. Media keeps focus (62 vs 64, no diff). Code: on_screen_resume runs _refresh_library_visit_surfaces on every ScreenResume, which fires when a modal pops (UI/Screens/library_screen.py:9262-9299; notes tree reload :9418-9422; prompts/skills focus_identity=None :9423-9439; conversations :9320-9330).
- **Repro:** 120x36 Notes: delete a note so focus sits on Undo > F1 > Esc > focus is in Filter.
- **Why it matters:** Each trip to help or a dialog costs up to 14 Tabs to return, and the Undo the user was about to press loses focus.
- **Fix:** Skip the visit refresh when the resume follows a ModalScreen pop (set the suspended flag only for real route changes), or capture self.focused before push_screen and restore it after the refresh, as Media already does.

<a id="s-08"></a>
### S-08 · P1 · Single-letter shortcuts fire while typing in the Notes filter during its re-render; 'i' opens Import media

- **Surface / kind:** Library-shell · defect · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H3, H5
- **Evidence:** j6 75: after 'Lisbon' Enter plus immediate 'xyzi', Import media opens with the footer 'typing in field' and 'xyz' is dropped; 64: the same class of escape led to Enter in the ingest path and a 10-file Media import with no confirm. Hypothesis: the filter apply recomposes the canvas so focus is briefly None and on_key routes 'i' to Import (UI/Screens/library_screen.py:8929-8940).
- **Repro:** Notes filter > type 'Lisbon' > Enter > immediately type 'xyzi'.
- **Why it matters:** Fast keyboard users refining a search are thrown into another destination, and one more Enter imports files.
- **Fix:** Keep the filter Input mounted across apply (update only the tree) or restore its focus synchronously; ignore single-letter shortcuts while screen.focused is None and for ~300 ms after an Input submit.

<a id="s-09"></a>
### S-09 · P1 · F6 never enters the Export or Search/RAG canvas; reaching 'Choose destination…' takes ~29 Tabs

- **Surface / kind:** Library-shell · a11y · confidence high · live-reproducible
- **Sources:** j4-poweruser-library, j5-a11y-consistency · **Personas:** Morgan, Sam
- **Heuristics:** H4, H7, WCAG 2.1.1
- **Evidence:** j4 (ANSI): on Export F6 only blinks in the rail 'Search Library…' box; 25 Tabs reach the bundle name and 29 'Choose destination…'; on Search/RAG F6×3 changes nothing, while both footers say 'F6 next pane'. j5 113: F6 stuck on rail search; 42: F6 cycles invisible targets. Code: library-canvas candidates are only library-hub-* and library-ingest-path (UI/Screens/library_screen.py:1420-1435); Widgets/workbench_focus.py:70-82 has no first-focusable fallback.
- **Repro:** Media > s > check 1 > Export > F6 repeatedly; Tab-count to Choose destination; repeat on Search / RAG.
- **Why it matters:** Keyboard-first users cannot reasonably operate Export or Search, the two power paths, without a mouse.
- **Fix:** Add #library-export-name / Choose destination and the Search query input and mode toggle to the library-canvas WorkbenchPaneTarget candidates; make _resolve_focus_target fall back to the pane's first visible focusable descendant.

<a id="s-10"></a>
### S-10 · P1 · Keyboard input stopped entirely at 80x24 (Tab/F6/F1/Ctrl+P) until a mouse click - intermittent

- **Surface / kind:** Library-shell · defect · confidence low · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H3, WCAG 2.1.1
- **Evidence:** j5 151-157: after resizing 120->80 on Search/RAG with focus on a Sources checkbox, Shift+Tab ×27 to rail 'Notes (122)' + Enter, then 45 Tabs = 0 cells changed, F1 no panel, Ctrl+P no palette, process idle at 0.1% CPU; a mouse click on '▸ Journal' restored input. Two exact replays did not reproduce. Cause hypothesis: focus left on a rail widget removed by the resize recompose. Kept at P1 (keyboard-only user is stuck) with low confidence.
- **Repro:** 120x36 Search/RAG with focus on a Sources checkbox > resize to 80x24 > Shift+Tab to rail 'Notes (122)' > Enter > Tab / F1 / Ctrl+P.
- **Why it matters:** To a keyboard-only user this is a hang with no recovery.
- **Fix:** When a resize or recompose removes the focused widget (_apply_library_notes_stage_visibility_for_resize and the rail collapse path), move focus to the first visible row of the active pane; add a Pilot test resizing across 120 with focus in the rail asserting app.focused is attached and visible.

<a id="s-11"></a>
### S-11 · P2 · Escape means different things per destination; advertised 'esc focus rail' does nothing when the rail is collapsed

- **Surface / kind:** Library-shell · consistency · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency, B-mechanical · **Personas:** Sam
- **Heuristics:** H3, H4
- **Evidence:** j5 41: Notes Esc expands Nav and collapses the list; 70, 91-92: Media/Prompts 'esc focus rail' does nothing with Nav collapsed; 67 vs 127: Esc inert in Media select mode but exits Notes select. B probes/slash-*: Esc in the Notes filter clears unsubmitted text and jumps to rail search. Code: UI/Screens/library_screen.py:27214-27240 focuses #library-search-input even when collapsed.
- **Repro:** 120x36: Media list with an item loaded > Esc ×2; Prompts > Esc; Media > s > Esc.
- **Why it matters:** Users cannot build one model of 'Esc goes back one step'.
- **Fix:** When the rail is collapsed, open it consistently (or relabel 'esc open rail'); make Esc leave Media select mode; make Esc in a filter blur without clearing; publish the per-level Esc contract in F1.

<a id="s-12"></a>
### S-12 · P2 · Active state is invisible or misleading on mode and segment controls (Edit/Preview/Info, Library notes|Folder files, Read/Info, Skills tabs)

- **Surface / kind:** Cross-destination · a11y · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H1, H4, WCAG 1.4.1
- **Evidence:** j5 21: in Preview, Edit/Preview/Info are all 'bold #e1e1e1 on #1e1e1e' (library_notes_canvas.py:3536-3541 sets is-active, which no stylesheet styles); 131/141: 'Library notes | Folder files' identical (css/widget_defaults_scoped.tcss:874 '.-selected { text-style: bold }' is a no-op on already-bold Buttons); 76: Conversations Read/Info identical; 101: active Skills 'Overview' dimmed to #a2a2a2 on #1a1a1a (looks disabled), focus brackets 1.98:1. Media 'Read (selected)', Collections '✓ Read' and '✓ Newest' do it right.
- **Repro:** Open a note > Preview > compare the three tabs; switch the Notes source strip.
- **Why it matters:** Low-vision users cannot tell which mode or source they are in; Skills makes the current mode look unavailable.
- **Fix:** Use the house '✓ ' prefix on every mode/segment button via the canvas builders; remove the dimmed-active Skills style; give .is-active a real style.

<a id="s-13"></a>
### S-13 · P2 · Arrival focus and Tab order do not follow the screen: palette arrival focuses '⌃1 Home', list entry focuses the Nav grip, and the source strip is the last Tab stop

- **Surface / kind:** Library-shell · a11y · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H7, WCAG 2.4.3
- **Evidence:** j5 03-04: after the palette, focus is on ⌃1 Home with 13 nav stops before Library content; 13/76/108: Notes, Conversations, Collections entry focuses the Nav grip; 81: Prompts focuses the filter; 140: Shift+Tab from the rail's top Import… wraps to 'Folder files'; 132: the source strip is reached only after the work pane. The User Guide promises list entry focuses the first row. Strip mounted at library_browse_route_swap.py:109-160 / library_screen.py:15047-15056.
- **Repro:** Ctrl+P 'Switch to Library' > Tab; open Notes and Tab once; Shift+Tab on the rail's Import….
- **Why it matters:** Focus order does not follow visual order (a DESIGN.md contract), and every arrival costs extra Tabs.
- **Fix:** After palette or nav-bar navigation focus the destination's first content control; on list entry focus the first row; move the source strip before #library-shell-grid in the focus chain.

<a id="s-14"></a>
### S-14 · P2 · Six different focus-indicator styles, several below 2:1 or a single cell

- **Surface / kind:** Cross-destination · a11y · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H4, WCAG 2.4.13
- **Evidence:** j5: rail 'x' focus is a one-cell underline (04-tab17); landing quick actions, Conversations scope chips and Search sources use underline + tint at 1.37 / 1.64 / 1.34:1 (04-tab45, 77-tab02, 117-tab4); Skills rows lack the █ bar (99); the reader uses an amber border vs blue elsewhere (72); container stops show only a 1-col │ or ┐ (82, 98, 136); '┃ New ┃' brackets vs underline-only quick actions.
- **Repro:** Tab through landing quick actions, Conversations scope chips, Search sources and Skills rows; ansidiff consecutive captures.
- **Why it matters:** Low-vision users lose focus on tint-only or one-cell stops; six dialects are unlearnable.
- **Fix:** Adopt the '┃ label ┃' bracket (or DESIGN.md's heavy $accent outline) for every Button and toggle; use the █ bar on every row type including Skills; make container scrollers non-focusable unless they are a pane's only target.

<a id="s-15"></a>
### S-15 · P2 · Footer and F1 disagree, F1 is a bare key list with a malformed row and no 'how it works', and Search/RAG advertises an inert 'o open evidence'

- **Surface / kind:** Library-shell · consistency · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes, B-mechanical, j5-a11y-consistency · **Personas:** Jordan, Sam
- **Heuristics:** H4, H10
- **Evidence:** j1 15, 35: editor F1 'Library Shortcuts — Notes' / '- : typing in field' / 'esc: back to list' / 'F6: next pane' / 'ctrl+n: New note' then ~20 empty rows; nothing on autosave, Save, Delete location, links or Use in Console; list F1 says 'ctrl+n' and '/: focus search' while the footer says 'n new note' and '/ find note'. B: footer 'esc back to notes' vs F1 'esc: back to list'; 'ctrl+end end of note' (works) absent from F1; Search/RAG arrival footer 'o open evidence' with 0 cells changed on o (2/2); Conversations arrival footer only 'F6 next pane' while F1 lists working '/' and 'c'. j5 112: same inert 'o'. Code: the panel is footer chips plus active bindings (UI/Screens/library_screen.py:25730-25768); the key-less 'typing in field' chip (:4827) leaks in; 'o' gate near :25538.
- **Repro:** Notes editor > F1; Notes list > compare footer with F1; Search / RAG > press o before any query.
- **Why it matters:** Help fails at the moment of need, and inert or mismatched chips teach users to distrust the footer (PRODUCT.md requires help to reflect what is available).
- **Fix:** Build footer chips and F1 from the same check_action predicate per key and one label per action; drop key-less chips from F1; show 'o' only when a result card exists; add a 4-6 line 'How Notes works' block (autosave 2 s, Save/Ctrl+S, Preview, Delete in Info ▸ Danger, [[Title]] links, Use in Console).

<a id="s-16"></a>
### S-16 · P2 · Labels are cut mid-word with no ellipsis across Library, hiding whole reader modes at 80 columns

- **Surface / kind:** Cross-destination · defect · confidence high · live-reproducible
- **Sources:** B-mechanical, j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H6, H8
- **Evidence:** B (each confirmed against a wider capture): media-reader-80x24 'Read (selected)  Analysis  Highligh' with Info fully off-screen, actions ending 'Use in Console  M' (More), 'sort: New', placeholder 'Title/keyw'; notes-syncsetup '○ Server n', '← Notes to folde'; folderfiles-60x24 'Review recovered pa'; landing-empty-80x24 'Use it in Consol'; import 'Defaults to source nam'; nav bar 'F5 Researc' at 160 and '⌃7 S' at 80. j5 02, 112, 158: rail heading 'Navigat…' at 120, '▾ scroll for'.
- **Repro:** Golden 80x24 > Media > expand Items > open 'A field guide to Markdown tables' > read the tab row.
- **Why it matters:** A cut label gives no sign more exists; Info and More become unreachable at 80 columns.
- **Fix:** Give Library toolbar Buttons/Statics 'text-overflow: ellipsis; text-wrap: nowrap' (as .library-notes-trash-row-copy does, css/features/_library_panels.tcss:573-577); below ~90 cols wrap the Media reader's mode tabs and actions to two rows like the Notes compact stage; in the nav bar move whole items into 'More ▾' instead of cutting labels.

<a id="s-17"></a>
### S-17 · P2 · A vetoed nav-bar switch is silent and leaves the nav bar highlighting the wrong destination

- **Surface / kind:** Cross-destination · defect · confidence high · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H1
- **Evidence:** j6 14-15: with a vetoed note, clicking ⌃2 Console leaves Library on screen while the nav bar frames '⌃2 Console'; only the log says 'Navigation to chat vetoed by the outgoing screen's pending-work flush'. Code: app_navigation.py:529-535 returns False with only logger.info.
- **Repro:** Set a trailing-space title, wait for the veto, click ⌃2 Console.
- **Why it matters:** Users think the app froze or that they are on Console.
- **Fix:** Toast _library_note_editor_exit_veto_message(kind) and re-select the current tab in the nav bar when navigation is vetoed.

<a id="s-18"></a>
### S-18 · P2 · A deleted note stays staged in Console as 'Ready' and the turn sends anyway

- **Surface / kind:** Cross-destination · defect · confidence medium · live-reproducible
- **Sources:** j6-stress · **Personas:** Riley
- **Heuristics:** H1
- **Evidence:** j6 29-30: 'Sources — next send 1 · Trip planning: Lisbon (note · Ready', 'Sources: 1'; Send proceeds with no notice after the note was deleted. Whether its text was transmitted is unverified (the mock log has no bodies).
- **Repro:** Note > Use in Console > back to Library > delete the note > Console > Send.
- **Why it matters:** Users may send content they deleted, or believe a source grounds the answer when it no longer exists.
- **Fix:** Revalidate staged sources at send time; mark them 'Deleted — remove from this send' and require acknowledgement.

<a id="s-19"></a>
### S-19 · P2 · After a send, Console contradicts itself about staged sources, and a follow-up's retrieval failure appears only in the log

- **Surface / kind:** Cross-destination · defect · confidence medium · live-reproducible
- **Sources:** j7-researcher-loop · **Personas:** Priya
- **Heuristics:** H1, H9
- **Evidence:** j7 65-67: strip 'Staged for next send · 1 source / paper-retrieval-practice — media', Inspect 'Sources: None staged' and 'Sources — next send 1', status bar 'Sources: 0'. The follow-up reply lacks 'Evidence: [S1]'; log '[ERROR] Console RAG capture unavailable; reason=capture_provider_failure; draft_length=36', exception swallowed in Chat/console_chat_controller.py; nothing in the transcript. The composer kept a stale 'Use this note as context…' prompt.
- **Repro:** Stage the paper via Search evidence > send a question > send a follow-up > compare strip, Inspect and status bar; check the log.
- **Why it matters:** The user cannot tell whether the follow-up is grounded; a failure visible only in logs is exactly what PRODUCT.md rejects.
- **Fix:** Clear 'Staged for next send' once consumed and show 'In this conversation: <source>'; on retrieval failure add an inline turn notice ('Library retrieval failed — answered from conversation history · Retry with sources') and log the exception type; replace the composer prompt when the staged source changes.

<a id="s-20"></a>
### S-20 · P2 · Console 'Search Library' modal fails a first-timer: typing lost, 'staged' shown before results exist, a jargon send-block, and results that re-stage after Un-stage

- **Surface / kind:** Cross-destination · defect · confidence medium · live-reproducible
- **Sources:** j1-firsttimer-notes · **Personas:** Jordan
- **Heuristics:** H1, H9
- **Evidence:** j1 42: the 'Library search' modal opens without focus in the query, so typing goes nowhere; 43: 'Staged for next send · 1 source — Library Search/RAG retrieval' while the embeddings model was still loading ('_build: Loaded model default in 15.69s' at 20:22:55); 44: send at 20:22:43 -> 'Console send blocked: Library search has no available evidence. Review source authority before sending.'; 45: two results staged themselves after Un-stage (log console-unstage-evidence 20:22:54, search completed 20:22:55).
- **Repro:** Empty profile with 2 notes, 120x36: Console > Search Library > type a query > click the field, type, Search > immediately send a question.
- **Why it matters:** This was the first-timer's only route to 'ask about a note' at 120 cols (Use in Console hidden, N-05).
- **Fix:** Focus the query input on open; label the chip 'Searching… (first run loads a model)' and keep Send disabled until it settles; discard results that arrive after Un-stage; replace the block copy with 'No matching notes for "<q>" yet — wait for the search to finish or Un-stage'.

<a id="s-21"></a>
### S-21 · P2 · The empty landing's only 'Import…' goes to Media import, with no pointer to importing notes

- **Surface / kind:** Library-shell · consistency · confidence high · live-reproducible
- **Sources:** j1-firsttimer-notes · **Personas:** Jordan
- **Heuristics:** H2, H4
- **Evidence:** j1 61: 'Import a file' / [Import…] opens 'Import media — … Supported: PDF documents, … plain text files, web pages.'; scrolled to the end it never mentions notes or 'Add from files…'. Code: UI/Screens/library_screen.py:8929-8940; library_entry_canvases.py:349-356.
- **Repro:** Fresh empty profile > ⌃3 Library > Import… on the landing.
- **Why it matters:** A first-timer's Markdown folder becomes read-only Media items instead of editable notes.
- **Fix:** On the empty landing offer 'Import documents…' and 'Import notes…'; on Import media add 'Bringing in Markdown notes? Notes ▸ Add from files… keeps them editable' with a button.

<a id="s-22"></a>
### S-22 · P2 · A relaunch drops the working context (staged sources, open note, filter, expanded folders)

- **Surface / kind:** Library-shell · usability · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H3, H7
- **Evidence:** j3 50-51: after REUSE=1 relaunch the app lands on Console with 'Sources: 0'; the Library landing has no Continue and 'From your Library' shows only 'Notes · Method'; open note, filter and expanded folders are gone; ScreenStateStore is memory-only.
- **Repro:** Open a note with a filter and expanded folders > stage a note > quit > relaunch > Library > Notes.
- **Why it matters:** Every daily session starts by re-navigating.
- **Fix:** Persist the last Notes route, note id, filter and expanded branches alongside the library.reader config; offer 'Continue: <note>' on the landing and restore on the first Notes visit.

<a id="s-23"></a>
### S-23 · P3 · Unexplained chrome: ASCII '--->'/'<---' grips with vertical letter labels, an unexplained 'Agent_Lessons' system folder, and literal '[ ]' task boxes in Preview

- **Surface / kind:** Library-shell · copy · confidence medium · live-reproducible
- **Sources:** j1-firsttimer-notes · **Personas:** Jordan
- **Heuristics:** H2, H8
- **Evidence:** j1 13: '--->'/'<---' grips with vertical 'N a v' / 'N o t e s', three arrows in one 120x36 list view; tooltips need hover; Nav expands/collapses between views at 120 (24 vs 25). 10: '▸ Agent_Lessons' appears in a brand-new tree after the first Console visit. 09: '- [ ]' renders as '• [ ] Email draft by Friday'.
- **Repro:** Empty profile 120x36: create a note, visit Console, return; write '- [ ] x' and Preview.
- **Why it matters:** Mystery glyphs and system folders add doubt for a hesitant first-timer.
- **Fix:** Replace grips with labelled toggles ('‹ Hide list' / 'Show list ›'); gloss or hide Agent_Lessons while empty; render task items as ☐/☑ in Preview.

<a id="s-24"></a>
### S-24 · P3 · Internal vocabulary and raw exception text on screen ('rail', 'placement', 'owner review', 'lane', 'cutover', 'authority', 'PermissionError', 'Open failed: {error}')

- **Surface / kind:** Cross-destination · copy · confidence high · live-reproducible
- **Sources:** B-static, j6-stress · **Personas:** Riley
- **Heuristics:** H2, H9
- **Evidence:** B: footer 'esc focus rail' for a pane titled 'Navigation'; 'Folders & placement' / '○ Remove placement' (library_notes_canvas.py:118, 2629-2650); '! Needs owner review' (library_notes_tree_state.py:718, 1004); 'Stored System lane' (prompt_history_region.py:412, 422); 'Persisted source:' (library_prompts_canvas.py:497); 'does not match the retained transcript' (library_conversation_reader.py:45); 'fresh projection is unavailable' (library_notes_sync_controller.py:1977); Folder files 'Git mutation in progress…', 'save authority changed', raw f'Open failed: {error}' (library_file_notes_workspace.py:4908, 6333, 6055…). j6 71: export to an unwritable dir shows 'Export failed — check the destination…' three times plus toast 'Error exporting note: PermissionError'. Sync-specific leaks are N-35.
- **Repro:** At 160x45 read the Notes list header and footer; Prompts > History > open a version; Info > Export Markdown into a chmod 555 dir.
- **Why it matters:** PRODUCT.md asks for plain language and no log-reading; raw exception text exposes paths and class names without a next step.
- **Fix:** Replace with 'esc navigation', 'Folders' / 'Remove from folder', '! Sync folder missing — review in Manage sync folders', 'System prompt (saved)', 'Saved in:', 'Still loading this conversation — try again in a moment', 'Git is updating…'; map exception types to sentences with a next step ('Can't write to <dir> — you don't have permission. Choose another folder.'), show one message not three, never interpolate str(error); add B's s2_vocab.py with an allowlist as a copy-lint test.

<a id="s-25"></a>
### S-25 · P3 · Layout and copy polish: wasted list space, toast covering its own buttons, stale 'Next' guidance and receipts, 'hub' jargon

- **Surface / kind:** Cross-destination · copy · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H2, H8
- **Evidence:** j5 87: toast 'Unsaved Prompt changes — Save or Discard changes first.' covers Save changes / Discard changes; 29: 'Saved 20:19 · Next: Start typing.' after typing; 175: '✓ deleted · J5 temp note' still shown ~40 min later; 182: Study F1 'esc: back to hub' under a generic 'Library Shortcuts' title. (Rail truncation is S-16; Prompts list space is L-32.)
- **Repro:** See the captures listed.
- **Why it matters:** Small frictions compound for a low-vision user reading a dense screen.
- **Fix:** Move toasts away from action rows; after the first save change 'Next' to 'Keep editing; changes save automatically'; expire delete receipts on navigation; say 'Library home' instead of 'hub' and add Study to _LIBRARY_HELP_SURFACE_LABELS.

<a id="s-26"></a>
### S-26 · P3 · User Guide keyboard and label claims disagree with the live app (8 of 12 checked)

- **Surface / kind:** Library-shell · copy · confidence high · live-reproducible
- **Sources:** j5-a11y-consistency · **Personas:** Sam
- **Heuristics:** H10
- **Evidence:** library.md: 'never by tabbing' to the nav bar (live 04, 97 Tab reaches it); 'Entering a … list… focuses the list's first row' (13, 81); 'Escape… never… changes what's shown' (41, 70, 91); '↑/↓ inside a … Notes list' (dead on folder rows); 'footer switches … typing in field' (Notes never does). notes.md: 'Use the visible Save button' (hidden, N-05); 'Info's footer is fixed… enter run action' (32: 'enter copy note' etc.); Esc 'one press, from Edit, Preview, or Info' vs 'From Info it goes back to the editor first' (self-contradiction); 'Export…' at notes.md:391 vs live 'Export'.
- **Repro:** Read Docs/User_Guide/library.md 'Keyboard & commands' and Docs/User_Guide/library/notes.md 'Editor keys' against the captures.
- **Why it matters:** Keyboard-only users depend on the guide for keys they cannot discover by pointing.
- **Fix:** Correct the listed sentences after the related fixes land (N-05, N-22, N-31, S-11, S-13) and re-verify the keyboard tables at 80x24 and 120x36.

<a id="s-27"></a>
### S-27 · P3 · After visiting Notes, Media's Items grip is painted 'Notes'

- **Surface / kind:** Library-shell · consistency · confidence high · live-reproducible
- **Sources:** j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H4
- **Evidence:** j3 08: the grip column reads N/o/t/e/s on the Media list until the next refresh. Code: Widgets/Library/library_browse_reader_shell.py:194-198 assigns pane_label without refreshing; LibraryAdaptiveReaderPaneGrip.sync_open repaints only when the arrow changes (library_adaptive_reader_shell.py:142-170).
- **Repro:** Open Notes, then click Media (23) and read the second grip.
- **Why it matters:** A mislabelled pane undermines trust in where you are.
- **Fix:** Call self.items_grip.refresh() after changing pane_label.

<a id="s-28"></a>
### S-28 · P3 · Select-mode and hand-off keys differ between Media and Notes (s/Space vs button/Enter; e only in Notes; c only in Media/Conversations)

- **Surface / kind:** Library-shell · consistency · confidence high · live-reproducible
- **Sources:** B-static, j3-poweruser-notes · **Personas:** Alex
- **Heuristics:** H4, H7
- **Evidence:** B: library_screen.py:1158-1196 binds s (media select), space (media row toggle), c, t, l; Notes binds g and e (:1039-1042); footers differ ('s select' / 'space toggle selection' vs 'enter select note' / 'e export selected' / 'esc done'); Notes has no key to enter select mode and the note editor's Use in Console has no key. j3 28: Space does nothing in Notes select.
- **Repro:** Media: s > Space > s. Notes: press s (nothing) > Select > Space (nothing) > Enter. Note editor (Preview focused): c (nothing).
- **Why it matters:** Learning one list canvas gives a no-op or a different action on the next, contrary to DESIGN.md's 'same key means the same thing'.
- **Fix:** One grammar on every list canvas: s toggles select mode, Space toggles the row, e exports the selection, Esc leaves select mode; bind c to Use in Console in the note editor when no text field has focus; gate through check_action.

Evidence paths are relative to `Docs/superpowers/qa/notes-library-ux-review-2026-10-02/` (`evidence/<journey>/NN-*` for A, `assessment-b/` for B). Code paths are relative to `tldw_chatbook/` unless stated otherwise.
