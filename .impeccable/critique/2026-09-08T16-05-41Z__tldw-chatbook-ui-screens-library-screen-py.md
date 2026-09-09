---
target: "Library screen + all subscreens (critique #8, whole-Library scope, first-time + power-user journeys)"
total_score: 21
max_score: 40
na_heuristics: 
p0_count: 1
p1_count: 7
timestamp: 2026-09-08T16-05-41Z
slug: tldw-chatbook-ui-screens-library-screen-py
---
Method: dual-agent (A: design-review sub-agent · B: detector/evidence sub-agent, isolated on separate scratch profiles and tmux sockets; parent reconciled the two and re-tested every disagreement in two further diagnostic sessions). Target `tldw_chatbook/UI/Screens/library_screen.py` at dev 1c022378cb, scope = the whole Library (landing, rail, Media, Conversations, Notes, File Notes, Prompts, Skills, Collections, Search/RAG, Import, Export, Study hand-offs). Live at 235x52, 100x30 and 60x24; a brand-new profile (first-time journey) and a seeded profile (11 media, 6 conversations, 7 notes, 5 prompts, 2 skills); no LLM provider configured. Earlier snapshots under this slug scored Library ▸ Media only; this run is the first whole-Library score.

## Design Health Score

| # | Heuristic | Score | Key issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 2 | Opening an existing note stays on "Loading note…" for ever with no timeout, no failure state (P0); rail count vs list vs export scope disagree |
| 2 | Match System / Real World | 2 | "[Errno 28] No space left on device" on a disk with 80 GB free; "Copy or link this conversation into workspace workspace-default"; a "Chunking Lab" strip on every canvas; raw UUID and ISO timestamps in the conversation reader |
| 3 | User Control and Freedom | 2 | Escape cannot leave any text box (the next letter is typed into it); a hung load has no Cancel; the File Notes wedge swallowed Escape, the back cue, the palette and Ctrl+Q |
| 4 | Consistency and Standards | 2 | "Send to Console" lives in four different places; `/` means three things; untitled = "New Chat" here, "Untitled source" there; one-page pagers shown on two lists and hidden on a third |
| 5 | Error Prevention | 3 | Armed two-press deletes, forecasts, receipts with focused Undo; but Tab walks out to the nav bar where Enter switches screens, and one fast-typed first title was corrupted |
| 6 | Recognition Rather Than Recall | 2 | Focus is invisible on grips, Run and Blank note; "Open in Console" sits below 30 messages; the Collections row opens a captures browser |
| 7 | Flexibility and Efficiency | 2 | Media has a rich key model, but the New-note canvas is keyboard-dead, evidence cards cannot be reached by Tab, and `o`/`u` are inert while advertised |
| 8 | Aesthetic and Minimalist Design | 2 | Notes status sentence painted twice per screen; a 130-character run-on header on Add from files; "Library tools are now available." lingers as a bare line; canvases 70% empty at 235 wide |
| 9 | Error Recovery | 2 | Raw errno with no "Show details"; hung loads offer nothing; the hand-off refusal names no action |
| 10 | Help and Documentation | 2 | 103 documented claims checked: 74 hold, 11 contradicted, 9 partial; Collections and Conversations pages describe surfaces that no longer exist |
| **Total** | | **21/40** | **Acceptable** |

## Design Specificity Verdict

Specific in vocabulary, generic in composition. Rail rows carry counts and glosses, Study rows are honest two-step hand-offs with a "Carries forward" staging canvas, the Media reader has modes, `]`/`[` walking and blocked states that name the next step, and deletes are graded with focused Undo. Those are authored for this product. The shell around them is the default rail + canvas + key-hint footer of any TUI file manager in one grey ramp with a single selection blue; nothing on screen supports "cyberpunk, efficient, cozy", and the one strip unique to this app under every header ("Chunking Lab | Try selected text") is an engineering tool, not brand.

Deterministic scan: the detector cannot read Python, so both `library_screen.py` and `Widgets/Library/` returned an empty result with zero files scanned; the one TCSS file it accepted yielded two advisory palette-colour findings. No browser overlay exists for a terminal UI, so all evidence below is live-measured (154 captures from A, 233 from B, plus my two diagnostic sessions).

## Overall Impression

The Media subscreen is a genuinely good product surface and the hand-offs into Console are the best idea in Library. Everything around it is uneven: the Notes core loop is broken on this build, the keyboard model has holes a mouse-free user cannot get past, and three subscreens (Collections, Conversations, Import) no longer match their own guide. The single biggest opportunity is to make the rest of Library as honest and keyboard-complete as Media already is, starting with the note loader.

## What's Working

- **Destructive-action grammar on Media.** In-place armed confirm naming the count, "✓ deleted · N items · in Trash" with Undo focused so Enter undoes, a Trash view with `r`/`x` advertised only while pressable, and the DB verified to flip back on Undo. This should be the template for Notes, Prompts and Collections.
- **Blocked states that carry their next step.** "○ Generate · No analysis provider is configured · Set one in Settings ▸ Providers & Models." and the RAG Answer block that names the missing key verbatim. No hover needed.
- **The reader keeps its mode while walking items**, and the CJK 98-character title is clipped in the list and complete in the header.
- **Adaptive layout holds.** 100x30 keeps list and reader side by side with no overlap; a live resize mid-edit recovered cleanly; the 60x24 single stage works for the landing.

## Priority Issues

- **[P0] Existing notes never open.** Any note opened from the Notes list stays on "Loading note… · Next: Wait for loading to finish." with no timeout, no failure state and no Cancel. Reproduced by A on both profiles, by B on the 35 KB note twice, and by me in two fresh sessions where a 60-byte note was the first thing opened; after the first hang every later open hangs too; Escape returns to the list. Notes created in the session are editable only until you leave them. A thread dump during the hang shows the event loop idle and no worker thread running the load, so the load outcome is being dropped or the coroutine cancelled before it paints; the coordinator's own unit tests pass with fakes, and the shell test that opens a row is a pre-existing baseline red, so neither can tell us. The notes guide was last verified 2026-09-06 on `fix/library-uat-31796-31797`, which bounds the regression to the dev commits since then. **Why it matters:** the researcher persona's whole loop (write, reopen, summarise) is dead; Alex's "open the long note" task dead-ends; it looks like the user's fault because the list still works. **Fix:** (1) a 3-second deadline that flips `_library_note_load_state` to "failed" through the existing "Unable to load note. Press Retry." path so the hang is at least visible; (2) instrument the three return paths of `_refresh_library_note_detail` (the generation guards, the STALE branch, the LOADED projection) and `_run_library_service_call`, which runs the async service inside `asyncio.run` on a `to_thread` worker, and find which one swallows the outcome. **Suggested command:** /impeccable harden.

- **[P1] Library is not keyboard-complete.** Four gaps, all measured: Escape cannot leave any text box (rail search, Search/RAG query) and the next printable key is inserted into it; the New-note canvas has nothing focused on entry so Enter and Down are no-ops while the footer says "enter create note", and the first Tab leaves Library for the nav bar where Enter switches to Home; Search/RAG evidence cards never take focus across 14 Tabs and `o`/`u` do nothing while the footer advertises them; focus is invisible on pane grips, the Run button and the Blank-note row. **Why:** Sam cannot complete the first-time journey; Jordan left the screen without noticing. **Fix:** focus-on-entry for every canvas the way lists already do; Escape in a text box blurs to the canvas; keep Tab inside the screen (nav bar only via F6/Ctrl+digit); make evidence cards focusable with the same `█▸` cursor Media uses; paint the same focus ring on grips and every compact Button; let the footer follow the focused control. **Suggested command:** /impeccable harden.

- **[P1] Import failures leak a raw errno, offer no diagnosis, and contradict the guide.** "✗ failed · reading-notes.md · Parse pool could not start: [Errno 28] No space left on device" on a disk with 80 GB free; the documented "Show details" row action is absent; unsupported files that the forecast said "will skip" also become "failed" with the pool reason; per-job "Import finished — 1 failed" toasts stack for a 6-file batch; Retry writes "· attempt 2" where the guide says "· retry 1". The trigger on this Mac is environmental (POSIX semaphore exhaustion, see Environment below), the copy is the product defect. **Fix:** map pool-start failures to "The import worker couldn't start on this machine (system resource limit) · Restart the app, then Retry"; restore "Show details"; keep skip-vs-fail truthful; one toast per batch. **Suggested command:** /impeccable clarify.

- **[P1] Structural waits have no exit.** A found the File Notes folder change stuck on "Changing folder…" with Escape, the "‹ Library / Notes" cue, a palette deep link and Ctrl+Q all swallowed (once, in the session already wedged by the notes hang; B linked the same folder fine). Whether or not the trigger is the same as the P0, a gate that blocks Quit is a design fault on its own. **Fix:** every structural wait (note load, folder change, skill import, export) gets a deadline and a visible Cancel after 3 s; the gate vetoes the write, never the exit or quit. **Suggested command:** /impeccable harden.

- **[P1] Conversation hand-off to Console refuses without a remedy, from a buried button.** "Open in Console" sits at row 49 of 52, under 30 messages, beside a pager that says "Already on the first page. · No more re…"; pressing it toasts "Copy or link this conversation into workspace workspace-default before using it in Console." (both assessors, same row) and nothing on the screen performs that copy or link. The seeded rows are not workspace-linked, so app-created conversations may pass, but imported or restored ones will hit this wall. **Fix:** move the action into the reader header beside Read/Info where Media puts "Use in Console"; render the ineligible state as "○ Open in Console · not in this workspace" with a "Link to workspace" action beside it, or drop the gate for local conversations. **Suggested command:** /impeccable clarify.

- **[P1] Collections is a different product than the one documented, and visiting it rewrites the rail.** The row opens a "Quick Capture" captures browser (Sort: saved desc, Filter captures, "0–0 of 0"), the local Collections service now rejects every write as "legacy_read_only", and `collections.md` still describes create/rename/delete records. The row shows no count until visited, then sprouts six sub-rows and collapses the Create section, and that collapse persists into the next launch. **Fix:** decide what the row is, rewrite the guide to match, give the row a count from the start, and never mutate another section's disclosure as a side effect of selecting a row. **Suggested command:** /impeccable document.

## Full issue register

Seven register rows are P1 (rows 2 to 8); the Priority Issues section above groups rows 2, 3 and 4 into the single keyboard-completeness item, so it lists five P1 bullets.

| # | Sev | Where | Issue | Evidence | Fix | Already tracked |
|---|---|---|---|---|---|---|
| 1 | P0 | Notes | Existing notes never open; no timeout/failed state; all later opens hang | A 85/87/88/132, B 71/110, C 02/04, C2 02 | deadline + failed state; trace dropped outcome | no |
| 2 | P1 | Rail, Search/RAG, Notes | Escape cannot leave a text box; next key typed into it | B 06b, 21 | Escape = blur to canvas | no |
| 3 | P1 | New note | Nothing focused on entry; Enter/Down no-op; Tab exits to nav bar, Enter there switches screen | B 13–19, A 13 | focus-on-entry; keep Tab in-screen | no |
| 4 | P1 | Search/RAG | Evidence cards unreachable by Tab; `o`/`u` inert while advertised; Tab from query lands on the sole source toggle and Enter wipes results | B 28/98/99, A 31/33 | focusable cards with cursor; footer follows focus | no |
| 5 | P1 | Import | Raw errno, no "Show details", skip-vs-fail displaced, per-job toasts, "attempt" vs "retry" | A 09/10/111–113, B 09/102 | reason mapper + Show details | no (task-31944 mapped DatabaseError only) |
| 6 | P1 | File Notes | "Changing folder…" wedge with all exits swallowed (seen once, after the P0) | A 95–100 | deadline + Cancel; never block quit | no |
| 7 | P1 | Conversations | Open in Console buried and refuses with workspace jargon, no remedy | A 82/83, B 69 | header placement + remedy action | no |
| 8 | P1 | Collections | Undocumented captures browser; service recovery-only; rail side effects persisted | A 37/38/115/130, B 32/96 | decide, document, count, no side effects | no |
| 9 | P2 | Export, Skills | "Everything" counted 0 conversations vs rail 6; "No skills yet" vs rail (2) (once); skill import does not refresh count in place | A 114/105, B 95/95b | one enumerator for count, list, scope | no |
| 10 | P2 | Landing | Get started only appears when the config file was created in the same run; a user who quits after setup and relaunches gets the full rail and never sees it (inferred from `coerce_library_lifecycle`; both scratch profiles pre-wrote config so both assessors skipped it) | A 05/06, B 04, code | persist "unknown" at profile creation | no |
| 11 | P2 | Media | `s` inert from a focused row while a Reader item is loaded; select strip labels clipped; confirm copy clipped at the Items floor | A 66–68/75 | check_action gate; wrap confirm | partly: 32045, 15140 |
| 12 | P2 | Notes | Editor auto-collapses the Notes list and it stays collapsed after Escape | B 24–26 | reopen list on exit | no |
| 13 | P2 | Notes | One fast-typed first title was corrupted (body lost) on the fresh profile, not reproduced on power; likely the graduation recompose resetting the Input mid-typing | B 23 + DB | never recompose a focused editor | no |
| 14 | P2 | Notes, Import | Duplicated status sentence; 130-char Add-from-files header; "Library tools are now available." bare line, and it fires on a populated profile's first visit | A 11/23/25/89, B 40 | one status line; toast, not line; no notice on UNKNOWN→GRADUATED | no |
| 15 | P2 | All canvases | "Chunking Lab / Try selected text" strip on every canvas with no gloss; Escape does not leave the Lab | A 05/41, B 107 | move under Details ▸ Actions with a gloss; Escape = Back | no |
| 16 | P2 | Media at 60x24 | Stage is an empty Reader with no items and no "‹ Library" return | B 119/120 | open the Items pane on exit | media page says follow-up pending |
| 17 | P2 | Landing at 100 cols | Landing still paints where the guide says it hides | B 113 | docs or code, pick one | no |
| 18 | P3 | Conversations | Raw UUID "Loaded bf20fab2…" and ISO timestamps; one-page pager painted | A 80/82 | age strings; hide one-page pagers | 2376 nearby |
| 19 | P3 | Media reader | "No Markdown formatting to render" on every plain item; "Unknown" byline at 100x30 | A 56/63/122 | say once in Info; no byline without author | no |
| 20 | P3 | Rail | Stale search query persists across canvases; 20 simultaneous choices; six Study rows for three destinations | A 38/05/39 | clear affordance; collapse Study to three | no |
| 21 | P3 | Footer | Wrapped/duplicated footer at 235 after a search; select-mode hints appear only after a round trip; "enter select evidence" while typing in the query box | A 30/69/74, B 21/97 | footer follows focus | 2520 nearby |
| 22 | P3 | Nav bar | The same box marks the active tab and a tab that merely has focus | B 15b/19 | distinct focus ring | outside Library |
| 23 | P3 | Wizard hand-off | Summary offers "Explore Home" and never mentions Library; "1 Add · 2 Find · 3 Use" names steps with no Find/Use controls | A 02/03/06 | add "Add your first document" | no |
| 24 | P3 | Docs | 11 contradicted claims across library.md, media, notes, collections, search, import pages | B §7 | sweep | no |
| 25 | P3 | Prompts | Variables dialog checkbox has no state glyph; "Use in Console" at row 49 | A 102–104 | glyph + header placement | no |

Already on the board from critique #7 and not re-filed here: tasks 32041 (235-wide arrow focus leak), 32042 (Conversations select mode), 32043 (reader desync), 32044 (flag-emoji frame drift), 32045 (zero-selection reason), 32046 (`/` targets the rail search); 15140 (bulk toolbar overflow below 110 cols); 28024 (review sets).

## Persona Red Flags

**Jordan (first-timer, empty profile):** the wizard's last page sends them to Home and never says where their content lives; the first thing Get started tells them to do (Import) fails with a disk-space error on a disk with 80 GB free and Retry just adds "attempt 2"; `n` opens a New-note canvas where Enter does nothing; Tab, Tab, Enter and they are on Home without noticing; an `Agent_Lessons` folder appears in their notes they never made; "Chunking Lab" is the first clickable thing under the header and drops them into a full-screen A/B tool. Abandons at import, or at the first reopen of a note.

**Alex (power user, seeded profile):** Media is a peak: filter, `]`/`[`, select two, export a 15 KB zip, delete two, Enter to undo, `r` to restore, every step honest. Then the long note hangs, then every note hangs; "Open in Console" is a scroll away and refuses; "Everything" export claims 0 conversations of 6; no "Continue" on relaunch. They will keep Media and stop trusting the rest.

**Sam (keyboard-only, low vision):** cannot create the first note or leave a text box without a mouse; cannot reach an evidence card; focus is invisible on grips, Run and Blank note; Delete and Cancel differ by brightness only, inverted; the hung states have no text-labelled cancel.

**Riley (stress):** 0-result filter, whitespace title, CJK/emoji rows, resize mid-edit and the Media Escape ladder all pass; the 5 k note is the first thing that hangs; the 60x24 Media stage has no way back; Escape does not leave the Chunking Lab.

**Solo builder/operator:** "Library | Local" and the RAG block name the missing key verbatim, good; but rail counts, list rows and export scope disagree, and Collections claims a feature its own service refuses.

**Researcher/student:** the Study rows are clear hand-offs and analysis blocks explain themselves; the notes they would write summaries in cannot be reopened.

## Docs vs Live

103 concrete claims from the nine guide pages were exercised: 74 verified, 11 contradicted, 9 partial, 9 not exercisable on this host. The contradictions that matter: Collections page (whole), Conversations detail is a transcript reader not a preview, "Show details" and the retry suffix on Import, evidence-card keyboard flow on Search/RAG, the Get-started compact rail, the F6 Reader heavy border, the Trash heading, the select-strip labels, the landing at compact widths, and the whitespace-title keep rule. Undocumented but shipped: Notes "New / New folder / Add to folder / Move", Prompts "Info" tab, the Chunking Lab strip, the captures browser, `ctrl+n` and `/ find note`.

## Minor Observations

Footer wrapped once at 235 after a search. The rail search box keeps the last query across canvases. A 1-hit media filter auto-loads the Reader. The export destination picker keeps Library's footer while open. "Cards due: 0" at 100 wide loses the word "Flash". The wizard Summary sprouted a stray tooltip fragment. The Import queue lists six identical failure rows where one grouped row would do. F1 is four lines in a 40-row box.

## Improvement opportunities beyond the fixes

1. **Timeout + Cancel for every structural wait** with a "Still working… · Cancel" line after 3 s. Alex, Sam. S.
2. **One "Send to Console" verb and placement** across Media, Conversations, Notes, Prompts and evidence: a reader-header button plus `c` everywhere, with ineligible states carrying their remedy. Alex, solo operator. M.
3. **Focus-ring parity**: the same bar or bold+underline on grips, Run, Blank note and every compact Button; the footer names the focused control's Enter action. Sam, Jordan. S.
4. **A Library-owned first-import walkthrough** replacing "1 Add · 2 Find · 3 Use" with three live buttons that unlock in sequence, plus a wizard Summary action "Add your first document". Jordan, researcher. M.
5. **One enumeration service** behind rail counts, list rows, export scope and search source counts; disagreements render as a callout with Retry. Solo operator. M.
6. **Reader-style Notes**: open notes in the Media reader shell (mode row, byline, Find, `]`/`[`), Preview as a mode; fewer status sentences, one grip style. Alex, researcher. L.
7. **A visible palette**: one accent for selection/primary, one danger hue only for destructive commits, a dim tier for glosses, through the existing tokens, so Delete vs Cancel reads and "cyberpunk, cozy" is at least legible. Everyone. S–M.
8. **Demote engineering surfaces**: Chunking Lab under Details ▸ Actions with a gloss; Session Git as a File Notes mode only when the folder is a repo; capture sub-rows only when captures exist. Jordan, Sam. S.

## Questions to Consider

- Is Library a content hub or the app's junk drawer? It currently owns a Chunking Lab, a Session Git panel, a trust passphrase store and a Quick Capture reading list. Would Jordan ever say "I'll go to Library to commit"?
- If every hand-off ends in Console, what would one "Send to Console" verb with one placement cost?
- The reader is the best surface here. What if Notes, Conversations and Prompts were reader modes over one Items list, with the type filter as the source?
- The rail shows 20 choices to a user with one note. What would it look like if it only listed sources with content, and everything else lived behind "New…" and "Import…"?
- Two of the findings are mine to own, not to ask about: the "Library tools are now available." graduation notice and the lifecycle rule that skips Get started for any profile whose config pre-exists came from the Home/Library redesign; the select-mode entry gate that leaves `s` inert with a Reader item loaded is from the wave-5 focus work. Proposals: fire the notice only on a real COMPACT→GRADUATED transition and paint it as a toast; persist "unknown" at profile creation; let `s` enter select mode from any focused Items row regardless of Reader state.

## Environment note

Every local import in both runs failed with "[Errno 28] No space left on device" because POSIX semaphores are exhausted on this Mac: a bare `multiprocessing.Pool(1)` in the venv raises the same error while the disk has 80 GB free. I did not kill anything: 34 four-day-old `/bin/bash --noprofile --norc -c set -euo pipefail repo_root=…` harness processes are still running and are the likely holders. Until they are cleared or the machine is rebooted, Import cannot be exercised end to end here, and any product path that starts a process pool will fail the same way.
