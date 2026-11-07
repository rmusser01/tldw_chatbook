# Improvements: first run and learnability (Notes + Library)

**Lens:** first run and learnability for Library and Library ▸ Notes. That covers onboarding that teaches through action; empty states that lead to a first success; progressive disclosure of the three notes worlds and of Library destinations; an in-product glossary; help at the moment of need; and recovery from first mistakes.

- **Target:** origin/dev `2d34cbf80d` (worktree `.worktrees/notes-library-ux-review`), Textual 8.2.8.
- **Inputs:** `master-findings.md` (103 findings), journeys j1–j7 (their improvement opportunities and emotional journeys), `maps/*`, `maps/known-context.md` §4 (ADR constraints), `PRODUCT.md` and `DESIGN.md`.
- **IDs:** ideas are numbered **LRN-01…LRN-12**, so they cannot be confused with finding ids (N-/L-/S-).
- **Paths:** code paths are relative to `tldw_chatbook/` at `2d34cbf80d`. Evidence paths are relative to this review folder.
- **Two ADR-027s:** the repo has two ADRs numbered 027, so this document always names which: *ADR-027 (default-workspace chats)* (`027-default-workspace-chats-in-chats-section.md`) or *ADR-027 (note session coordinator)* (`027-portable-database-note-session-coordinator.md`).

## New live evidence for this lens

I made one probe on socket `nl-learn-1`: an empty profile at 120x36, run through the nlrev harness. The app logged 0 tracebacks, the real profile was untouched, and the socket was killed afterwards. The captures are in `improvements/evidence-learnability/`. They settle three facts the ideas below depend on:

1. **Get started teaches only step 1.** The empty landing shows the steps `Import a file · Find it · Use it in Console`, with the hint "Find it needs something to search — Import a file first." (`01`).
   - After one successful import, Escape lands on the full graduated rail: Browse 7 rows, Artifacts 3, Create 3, Study 3, Import/Export 2. Get started is gone and no message is shown (`04`).
   - So "Find it" and "Use it in Console" are never seen unlocked. This confirms map 7.4, which had been a hypothesis.
   - The code agrees: `_set_library_lifecycle` (`UI/Screens/library_screen.py:21572`) persists the transition and sends no notice.
2. **The palette does not lead to Notes.**
   - Typing `notes` lists "Switch to Library" first. Its description mentions notes, but it lands on the Library landing (`05`).
   - Typing `workspace` describes Library as "for Workspaces, source material, …" (`06`), but nothing in Library defines what a workspace is.
   - `LIBRARY_SUBROUTE_COMMANDS` (`app_command_providers.py:327-338`, task-423) deep-links only Artifacts and Skills.
3. **The Notes (0) state hides its own call to action.**
   - At 120x36, "No notes yet. Create your first note." is plain text on row 28. Above it are 11 list controls, 5 of them disabled (`○ Select`, `○ Add to folder`, `○ Move note`, `○ Remove placement`, and "Note actions unavailable — select a note in the list").
   - The work pane says "Select a note to edit it here." when there is nothing to select. The footer offers `/ focus search` over zero notes (`07`).
   - The same probe saw L-30 again: one Enter in the path field imported the file while the footer said `enter check this path` (`03`).

## Summary

| ID | Improvement | Impact (1-5) | Effort | Main findings removed | ADR fit |
|---|---|---|---|---|---|
| [LRN-01](#lrn-01) | Workspaces appear only when there are two | 5 | M | S-01, S-20, L-31 | Extends ADR-027 (default-workspace chats) to Library; amend it |
| [LRN-02](#lrn-02) | Typing is never punished: a Library input-safety contract | 5 | M | N-01, N-06, N-07, N-25, N-12, L-05, L-08, N-34, S-08, L-07, N-31, L-30 | ADR-031 refinement; ADR-055 D; ADR-027 (note session coordinator) |
| [LRN-03](#lrn-03) | Carry the first loop past graduation | 4 | M | S-21, L-35, L-13, S-01, L-36, map 7.4 | Fits ADR-076; one persisted flag needs an ADR-076 note |
| [LRN-04](#lrn-04) | Zero states that perform the first action | 4 | M | N-37, L-36, S-21, L-34, L-10, L-12 | Exactly what ADR-076 assigns to source canvases |
| [LRN-05](#lrn-05) | F1 says how the surface works, not only which keys exist | 4 | S | S-15, L-36, S-26, S-11, N-33, N-08 (help half), S-25 | Fits ADR-031, ADR-011 |
| [LRN-06](#lrn-06) | One Library vocabulary with an in-product Terms list | 4 | M | L-36, S-24, S-23, N-18, N-35, S-20, S-01 (copy), L-28, L-34, prior L-p3 | Fits; lazy-load for ADR-097 |
| [LRN-07](#lrn-07) | No dead ends: every blocked action names its prerequisite and offers it | 4 | M | S-01, N-15, N-24, N-17, N-16, N-02 (exit), L-02, L-22, S-17, map 7.18 | Fits; reuses the TASK-716 pattern |
| [LRN-08](#lrn-08) | Teach Search vs Ask by consequence at the point of choice | 4 | M | L-13, L-06, L-27, L-24, L-23, S-20, S-01 (detour) | Fits ADR-003, ADR-079 |
| [LRN-09](#lrn-09) | The three notes worlds: choose by consequence, always see which one you are in | 3 | M | S-21, S-12, N-37, N-18, N-27, S-23, N-02/N-03 (visibility) | Fits ADR-059, ADR-021/029; respects the 32627 deferral |
| [LRN-10](#lrn-10) | Find Notes from wherever people look | 3 | S | j1 steps 1–2, N-13, N-21, map 7.6, S-13 | Extends the task-423 exception to ADR-015 |
| [LRN-11](#lrn-11) | A sample to learn on (local, free, removable) | 3 | M | L-10 (contrast), L-33, S-20, j1/j2 task gaps | ADR-076 already treats samples as non-graduating |
| [LRN-12](#lrn-12) | Graduate the rail gradually | 3 | S | L-36, S-16, prior L-p3, map 7.4 | Section prefs only; no lifecycle change (ADR-076, ADR-034) |

Suggested order:
1. The S-effort wins that need no ADR work: LRN-05, LRN-10 and LRN-12.
2. The two core-loop unblockers: LRN-01 and LRN-03.
3. The two contracts that remove whole classes of findings: LRN-02 and LRN-07.
4. The rest.

---

<a id="lrn-01"></a>
## LRN-01 · Workspaces appear only when there are two

**User problem.** The first thing a newcomer tries after importing a document fails.
- In the Media reader, Use in Console is refused with "Copy or link this media into workspace workspace-default before using it in Console." No control anywhere does what that sentence asks (S-01, P1; j2 steps 14–15; j7 "the app does not trust her paper").
- The raw id is interpolated at `Workspaces/eligibility.py:79`.
- The same item stages fine through Search ▸ Select evidence ▸ Use in Console, so the rule looks arbitrary.
- The Conversations reader spends 2–3 rows explaining workspaces above the first message (L-31).
- Console blocks a first-timer's send with "Review source authority before sending." (S-20).
- ADR-027 (`027-default-workspace-chats-in-chats-section.md`) already decided the principle that Library breaks: *"Everyday chatting must not demand workspace vocabulary. The Default workspace exists so the data model is uniform, not so users think about it … The storage identity (`workspace-default`) stays an implementation detail."*

**Proposal (what the user sees and presses).**
1. **One workspace = no workspace vocabulary.** While Default is the only workspace, every local Library item is eligible for Console. The eligibility check takes the existing "no active workspace" branch (`Workspaces/display_state.py:582-600`, "Workspace: Local Default", all local sources eligible).
   - Reader `Use in Console`, Details ▸ Use in Console, Conversations `Use as source` and Notes `Use in Console` all just stage, then report "Staged in Console".
   - The Details "Workspace" row and the Conversations reader's workspace paragraph are not rendered in this state.
2. **The concept arrives with the second workspace.** The shared New Workspace dialog (the one Console, Settings and Library ▸ Details ▸ Create local workspace already use) gains one sentence: "Everything already in your Library stays usable from Default. Items you link to this workspace are the ones Console sees while it is active."
3. **With ≥2 workspaces, the block is a one-press fix.**
   - A blocked control reads `Use in Console · not in Thesis` (the task-32056 short-label pattern) with an adjacent `Link to Thesis and use` button.
   - The button links, stages and leaves an Undo receipt (the task-32388 workspace-hop receipt precedent).
   - Copy always uses the display name, never an id.

**User value.** This removes the single biggest first-run dead end on the product's core loop (ingest → reason in Console; PRODUCT.md purpose). A term is learned when it starts to mean something, not before.

**Effort:** M. **Impact:** 5.

**Precedent.** Git needs no branch concept until a second branch exists; the default just works. VS Code uses single-folder mode until you add a second root, and only then a `.code-workspace` appears. Slack hides the workspace switcher until you join a second workspace.

**Risks / ADR.**
- This extends ADR-027 (default-workspace chats) from Console chats to Library sources. Record it as an amendment to that ADR, or as a new ADR that cites it.
- PRODUCT.md principle 7 ("Treat Workspaces as global context") is kept. Workspaces stay global; only a one-element concept is no longer taught.
- ADR-028 is unaffected: Default stays tool-less, because this is staging eligibility, not tool or folder access.
- The owner must confirm that the eligibility gate's purpose is scoping, not a security boundary. S-01's own fix already assumes so.
- Moving from one to two workspaces must keep Default's items usable exactly as before.

---

<a id="lrn-02"></a>
## LRN-02 · Typing is never punished: a Library input-safety contract

**User problem.** The mistakes newcomers make first are the ones Library punishes hardest, and each is a separate seam today:
- **Trailing space in a title** vetoes autosave and moves focus mid-typing, so the next words land in the title (N-07). The advice then names a button that is not there (N-25).
- **F1 mid-edit** silently stops autosave (N-06), and **Ctrl+Q** then discards the text (N-01, P0).
- **Escape** discards a typed analysis (L-08) and cancels a running import, while `‹ Notes` keeps it running (N-34).
- **Cancelling** an export is reported as "Export failed" (N-12).
- **`~/`** is called a "dangerous pattern" (L-05).
- **Single letters** fire commands while the user is typing or just after a filter submit, so `i` opens Import (S-08, L-07). The Notes footer never switches to `typing in field` (N-31).
- **Enter** imports while the footer says `enter check this path` (L-30; seen again in `evidence-learnability/03`).

j1 sums it up: *"powerful but fragile; I'd keep a copy of anything important elsewhere"*. j6: *"Riley now checks the DB after every save."*

**Proposal.** Add six rules to DESIGN.md's Interaction Quality Contract and enforce them with one shared Pilot suite. The suite runs against every Library text field and editor (note, prompt, skill, analysis, Folder files, import path, filter):

1. **Normalize, don't veto.** Trim title whitespace, dedupe keywords and expand `~`, then say what changed inline under the field ("Removed a trailing space from the title").
2. **Autosave never moves focus.** A validation problem marks the field; focus moves only on explicit Save or a leave attempt.
3. **Escape never destroys typed text.** With a dirty draft the first Esc blurs and shows `Unsaved — Save · Discard`. Discard follows ADR-055 pattern D (confirm, no receipt). Escape on a running job means "back", never "cancel".
4. **Every exit flushes or asks.** Quit, navigation, modal push and unmount all await one seam (`prepare_for_quit` → `flush_pending_work`). ADR-027 (`027-portable-database-note-session-coordinator.md`) keeps the single note-session coordinator, so this adds no second save path.
5. **Cancel is neutral.** A cancelled picker or job returns the status to what it was before, with no failure copy, and focus returns to the opener.
6. **Letters are commands only off text.** Single-letter keys act only when a list row or non-text control has focus.
   - A letter that leaves the surface toasts "Opened Import (i) — Esc to return", and Esc returns to the same field with its text.
   - The footer shows `typing in field` for every text field.

The suite asserts these rules per field: type → trigger (F1, modal, Ctrl+Q, Esc, a letter after Enter) → DB row and focus unchanged.

**User value.** A newcomer can experiment safely, and experimenting is how terminal apps get learned. The suite stops these regressions from coming back one field at a time.

**Effort:** M. The contract and suite are new; the per-field fixes are already itemized in the findings. **Impact:** 5.

**Precedent.** macOS autosave and state restoration; Google Docs never loses a keystroke; GOV.UK form guidance ("trim whitespace, don't reject it"); vim swap files.

**Risks / ADR.**
- Rule 6 is a refinement of ADR-031 (single-letter screen actions; footer advertised == working). Record it there.
- Normalizing may surprise someone who wanted a trailing space. The inline notice makes it visible and reversible.
- Keep the suite cheap: one parametrized test per field kind, not per surface.

---

<a id="lrn-03"></a>
## LRN-03 · Carry the first loop past graduation

**User problem.** Get started is Library's only teach-by-doing device, and it stops teaching after step 1.
- The first import graduates the profile and silently replaces Get started with a 25-row rail (live: `evidence-learnability/01` → `04`). "Find it" and "Use it in Console" are never seen unlocked (map 7.4).
- Step 3's tooltip, "Send a search result to Console as evidence." (`Widgets/Library/library_entry_canvases.py:385`), names the one route to Console that works for media today. The user never meets it and tries reader `Use in Console` instead (S-01).
- After a fully successful import, the queue offers `Retry this batch` and `r retry` (L-35), not a next step.
- The landing `Import…` goes to Media import with no pointer to notes (S-21).
- A plain-language question in the rail box returns "No evidence matched" (L-13).
- j2: "Only 'Explore all tools' made the purpose clear. The landing alone did not."

**Proposal.**
1. **Offer the next step where the last one finished (primary carrier).**
   - **Import queue, after an all-success batch:** replace `Retry this batch` with `Next: ┃ Find it ┃ search what you just imported · Ask about "article-local-first"`. `Find it` opens Search/RAG with the scope chip `Just imported (1)` and focus in the query field. `Retry` stays only when a row failed (L-35).
   - **First Search result:** its evidence card shows `Next: Use in Console (u)`.
   - **First staging in Console:** the existing staged strip ("Staged for next send · 1 source …") already completes the lesson.
2. **Keep a one-row "First loop" strip on the expanded landing** until all three steps are done or the user hides it:
   ```
   First loop   ✓ Import a file   ▸ Find it   ○ Use it in Console        Hide
   ```
   - Completion comes from real events, never counters: an ingest job that completed (a failed import does not count), a Search/RAG run with ≥1 result, and a Library-originated Console staging.
   - At <30 rows the strip folds into the counts line: `First loop 1/3 · Next: Find it`.

**User value.** Every new user walks the PRODUCT.md core loop on the path that works, and learns "a search result is evidence you send to Console" by doing it once.

**Effort:** M. **Impact:** 4.

**Precedent.** GitHub's new-repository "Quick setup"; Linear's and Stripe's dashboard "Get started" checklists, which persist until done or dismissed; VS Code Walkthroughs, whose steps tick from real actions.

**Risks / ADR.**
- ADR-076 is kept: the rail still graduates permanently and silently (task-32555), and there is no wizard and no Beginner mode. The strip is landing content owned by the landing.
- Persisting "first loop hidden or done" adds one key beside `library.rail_state`. Note it in ADR-076 so it is not mistaken for a fifth lifecycle state.
- The landing is hard to reach once you leave it (map 7.3). That is why the point-of-completion hand-offs in (1) carry the lesson and the strip is secondary.

---

<a id="lrn-04"></a>
## LRN-04 · Zero states that perform the first action

**User problem.** ADR-076 itself records that "several empty canvases retain list mechanics that cannot yet do useful work", and it defers per-source empty states to later atomic tasks. Live, they still do:
- **Notes (0):** the call to action is plain text on row 28 under 11 controls, 5 of them disabled. The work pane says "Select a note to edit it here." (`evidence-learnability/07`).
- **Prompts (0):** "No prompts yet. Create or import a prompt to begin." with no definition (L-36).
- **Skills:** the trust banner shows when the user has zero skills of their own (L-36).
- **Folder files, unlinked:** 61–78% blank, with one inline link (N-37).
- **Collections:** the Quick Capture box has no label, and "keep for later" exists in three places (L-34).
- **Reports:** the `Try report demo` button sits 38 rows from its sentence (L-10).
- **At 60–80 columns:** Media, Skills and Conversations open to an empty Reader (L-12).
- j1's first improvement asked for exactly this ("Empty-list starter card").

**Proposal.** Every source owns one zero-state grammar, as ADR-076 requires: what this is (one sentence, user words) → one focused primary action → at most two secondary actions → what becomes possible next.
- Controls that act on nothing (Filter, Sort, Select, Export, Folders & placement) are not rendered until the first item exists.
- The work pane, which is permanent under ADR-086, hosts the explanation in place of "Select a … to … here.".

Notes (0), list pane and work pane:
```
Notes (0)                          Notes you write in Chatbook
No notes yet.                      ────────────────────────────────────────
┃ New note ┃  n                    Kept in Chatbook's own database and saved
  Bring in Markdown files…         as you type. Preview shows headings, lists
  Edit a folder in place…          and checkboxes. Use in Console asks
                                   Chatbook about a note.
▸ Agent_Lessons — where Console    Already keep notes in a folder? Bring them
  agents file lessons for reuse    in (a copy) or edit them in place (no copy).
```
- **Prompts (0):** "A prompt is reusable instruction text you insert into Console." `┃ New prompt ┃` `Import…`
- **Skills, no user skills:** "Skills are instruction packs an agent can load. You have 1 built-in." `┃ Import skill… ┃`. The trust banner appears only once a user skill exists.
- **Folder files, unlinked:** a centred "Edit Markdown files where they already are — nothing is copied into Chatbook." with `┃ Choose folder… ┃` and a secondary `Review recovered pairing…` (the N-37 fix). The rail stays in the shell's left column.
- **Collections (0):** "Captures are web pages saved by URL. To keep a Library file for later, use Read later in Media." `┃ Capture a URL… ┃`. The text box gets the label "Note (optional)".
- **Reports (0):** the sentence, then directly under it `Set up a daily brief…`, which confirms the feeds, the provider and model, and the daily cadence (L-10).
- **Media / Conversations / Skills at 64–98 columns:** open with the list pane, as Notes and Prompts already do (L-12).

**User value.** The first screen of each destination produces the first item instead of describing an empty list. Disabled clutter disappears exactly when a newcomer is reading every label.

**Effort:** M (six sources, each S). **Impact:** 4.

**Precedent.** Obsidian's empty vault ("Create new note"); Linear and Things empty views, with one action plus a sentence; Material Design and NN/g empty-state guidance (explain, then one starting action).

**Risks / ADR.**
- Controls appearing when the first item lands is a state change, not a focus or hover shift, so the DESIGN.md "no layout shift on focus" rule is kept.
- Keep the copy in each source's state module (e.g. `Library/library_notes_state.py:20` `_EMPTY_NOTES_COPY`), not in a shared compositor (ADR-076: "one generic lifecycle controller/canvas" was rejected).
- Respect the list action budget (`library_notes_canvas.py:125-139`).

---

<a id="lrn-05"></a>
## LRN-05 · F1 says how the surface works, not only which keys exist

**User problem.** F1 is where a hesitant user goes, and it does not help.
- **Landing:** 3 keys inside a ~30-row empty frame (j2 `03`).
- **Note editor:** `- : typing in field` (a malformed row), `esc: back to list`, `F6`, `shift+f6` and `ctrl+n`. Nothing on autosave, Save, Preview, where Delete lives, linking or Use in Console. j1's task 5 (linking) failed and task 8 (learning from help) was only partial (S-15).
- **Footer and F1 disagree** (`n` vs `ctrl+n`; `/ find note` vs `/: focus search`), and Search/RAG advertises an inert `o` (S-15).
- The Escape contract differs per destination and is published nowhere (S-11).
- Four editors use four save models (N-33).
- The User Guide's key claims are wrong 8 times out of 12 (S-26).

The help panel already has a free-text section that Library never fills: `WorkbenchHelpState.notes_heading` / `notes` (`UI/Workbench/help.py:30-34`). The Library help builder (`UI/Screens/library_screen.py:25683-25770`) passes only `shortcuts`.

**Proposal.** Give each Library surface a "How <surface> works" block of 3–6 lines.
- **Build it from copy the surface already renders**, as Settings does: `_category_help_notes` (`UI/Screens/settings_screen.py:4222-4240`, TASK-23110) reuses on-screen copy so help cannot drift.
- **Drop key-less chips** from the shortcut list.
- **Spell keys as the footer does.**
- **End each block with this surface's Escape step.**

Note editor example:
```
Library Shortcuts — Note editor
How notes work
  Saves by itself 2 s after you stop typing; "Saved 20:11" confirms it.
  Edit is Markdown · Preview shows it formatted · Info has keywords,
    links, export, and Delete (under Danger).
  Use in Console stages this note for your next question.
  Esc here: back to the notes list. Esc again: Navigation.
Shortcuts
  n new note · / find note · F6 next pane · …
```
Other surfaces, one example line each:
- **Landing:** "Library holds what you've imported, chatted or written, so you can find it, ask about it, and send it to Console." (the L-36 copy)
- **Search/RAG:** "Search finds passages locally and free. RAG Answer sends your question and the top passages to your model." (pairs with LRN-08)
- **Prompts:** "Prompts save when you press Save changes; Notes save by themselves." (states the save model, N-33)
- **Manage sync folders:** "Each row is one folder kept in step with Library notes. Check changes looks; nothing is written until you Activate or Apply."

Help states only what works. Until typed `[[Title]]` links resolve (N-08), the Notes block says "Links between notes come from Import once; a typed [[Title]] stays plain text for now." rather than teaching a convention that fails.

**User value.** Help answers "what is this, and how do I do X here?" at the moment of need, which j1's tasks 5, 8 and 9 asked for. Footer and help stop teaching wrong keys.

**Effort:** S (the panel field exists; this is copy plus wiring per surface). **Impact:** 4.

**Precedent.** Settings' "How this category works" (TASK-23110); Console's "Agents" primer (`UI/Screens/chat_screen.py:4919`); lazygit `?` shows contextual commands with descriptions; vim `:help quickref`.

**Risks / ADR.**
- Fits ADR-031 (F1 reserved; advertised == working) and ADR-011 (contextual help).
- Add a test that every `_LIBRARY_HELP_SURFACE_LABELS` surface has non-empty `notes`, and that every listed key passes `check_action`.
- The help module is already lazy-imported on F1, so ADR-097 boot cost is unchanged.
- Fix the S-26 guide drift by checking the guide's key tables against the same predicates, not by hand.

---

<a id="lrn-06"></a>
## LRN-06 · One Library vocabulary with an in-product Terms list

**User problem.** j2's persona "does not know what 'RAG', 'workspace', 'staged evidence' or 'skill' mean". The app defines none of them, and it uses internal words in their place:
- **Internal words on screen:** "rail", "placement", "owner review", "lane", "cutover", "authority", and exception class names (S-24, N-35).
- **Leaked state names:** "Sync folder (name unavailable before cutover)" (N-18).
- **Jargon in errors:** "Review source authority before sending" (S-20); "workspace workspace-default" (S-01).
- **Unexplained items:** "Agent_Lessons" (S-23); trust words with no explanation (L-28); three unlinked "keep for later" ideas (L-34).
- **Too many names:** five for the left pane (map 7.19), and "hub" vs "Landing" for the landing.
- **Glosses that disappear:** the rail's only definitions are row suffixes that drop by width ("Collections — saved captures" needs 34 cells and is hidden at the default 33; `Library/library_shell_state.py:584`).
- **No glossary anywhere:** none in the app or in `Docs/User_Guide/` (grep: none). The palette calls Library the place "for Workspaces" (`evidence-learnability/06`).
- **Unowned:** prior critique item L-p3 ("no in-product definition of RAG, Skill, Collection, workspace") is still present and has no owner.

**Proposal.**
1. **One terms table:** `LIBRARY_TERMS`, about 22 rows of (term, one-sentence definition, where you manage it, banned synonyms). Rows: Library, Library notes, Folder files, Synced folder, Import once, Unfiled, Agent_Lessons, Media, Read later, Collection / Capture, Prompt, Skill, Built-in skill, Skill trust, Search, Ask (RAG Answer), Evidence, Staged (for next send), Workspace, Source (formerly "source authority"), Chatbook, Report, Recently deleted / Trash.
2. **F1 "Terms on this screen"**, shown under LRN-05's block. It lists only the terms rendered on the current surface:
   ```
   Terms on this screen
     Evidence — a passage Search found; select it to send it to Console.
     Staged — attached to your next Console message only; Un-stage removes it.
     Ask (RAG) — sends your question and the selected passages to your model.
   ```
3. **Palette "What is …":** one command per term ("Library term: Workspace") that opens F1 at that entry. It is found by typing the term, which is what j1 tried with "link".
4. **One name per concept**, enforced by a copy-lint test. Seed it with B's `s2_vocab.py` (the S-24 fix).
   - User-facing strings under `Widgets/Library`, `UI/Library_Modules` and `Library/*_state.py` may not contain the banned words (cutover, placement, lane, owner review, projection, mutation, bare "authority", `{error}` interpolation, exception class names) unless the word is a defined term.
   - The canonical names are "Navigation" (not rail/Nav/Library pane) and "Library home" (not hub/Landing).
5. **Rail glosses move to section headings** (see LRN-12), so they survive any width.

**User value.** A newcomer can look up any word on screen without leaving it. The lint stops new jargon at review time, so the glossary stays short.

**Effort:** M. **Impact:** 4.

**Precedent.** GitHub Docs glossary and inline definitions; Stripe docs glossary; style-guide word lists enforced as CI lint (Vale, alex); Apple HIG ("use terminology consistently").

**Risks / ADR.**
- A glossary can become an excuse for jargon. The lint, not the glossary, is the primary control, and only domain terms a user must learn get entries.
- Load the table lazily on F1 or palette use (ADR-097 boot ratchets).
- Renaming "RAG Answer" to "Ask" touches Console and the guide; do it in one pass (LRN-08).

---

<a id="lrn-07"></a>
## LRN-07 · No dead ends: every blocked action names its prerequisite and offers it

**User problem.** Newcomers learn an app's model by pressing things. Library often answers with a silent no-op or a sentence that points nowhere:
- **S-01:** names an action that does not exist.
- **N-15:** `Apply reviewed` is dim with no reason, and the resolution choices are disabled unconditionally.
- **N-24:** "That folder changed elsewhere — refresh and try aga…" when nothing changed, and there is no refresh control.
- **N-17:** after a failed Recovery, Add from files is blank until restart.
- **N-16:** one non-UTF-8 file blocks a whole sync folder with "Check failed — NotesSyncRootRefused".
- **N-02:** a wedged sync loops between Review and Recovery, and "Disconnect" is "unavailable — not in this release".
- **L-02:** sends users to a Settings page that cannot fix the problem.
- **L-22:** `Open original` / `Open manager` do nothing.
- **S-17:** a vetoed nav switch is silent.
- **Map 7.18:** some reasons exist only as tooltips (mouse-only).
- **Notes (0):** shows five separate `○` controls whose only reason is one generic line (`evidence-learnability/07`).

**Proposal.** Make the house "pressable blocked" pattern (TASK-716; `Widgets/Library/library_entry_canvases.py:416-421`, where a blocked step stays pressable and explains on press) the rule everywhere in Library, and test it:
1. **Every `○` control is focusable and pressable.** Focus or press shows `<Action> needs <prerequisite> — <action>`, with the prerequisite as a button when one exists:
   - `Add to folder needs a note — ┃ New note ┃`
   - `Generate needs a provider — ┃ Open Settings ▸ Providers ┃` (only when that page can actually fix it; L-02)
   - `Use in Console · not in Thesis — ┃ Link to Thesis and use ┃` (LRN-01)
2. **Identical reasons collapse into one line** under the group, e.g. "Folder actions need a note — create one first". This replaces five `○` rows in Notes (0) (see also LRN-04).
3. **An action that can never work in this context is hidden, not left inert** (L-22 `Open manager` inside Library).
4. **Every long-lived failure state has a safe exit:** sync wedge, recovery failure and refused check each offer `Pause and keep both copies` (the N-02 fix). "Next:" never names the same failing action twice. Vetoed navigation toasts the reason (S-17).
5. **Architecture test:** enumerate Library Buttons that are `disabled` or `library-source-action-blocked`, and assert a non-empty, visible (non-tooltip) reason. No "Next:" may point at the action that just failed.

**User value.** Each refusal teaches the dependency ("folders organise notes, so make a note first") and gets the user moving again. Nothing leaves a newcomer stuck or restarting the app.

**Effort:** M. **Impact:** 4.

**Precedent.** GitHub's merge box, which lists each unmet requirement with a link; Stripe's disabled buttons with an explanation; Apple HIG ("explain why an item is unavailable").

**Risks / ADR.**
- Focusable disabled controls add Tab stops. Keep them in visual order (DESIGN.md focus contract), or move focus to the single reason line when reasons are grouped.
- Reason copy uses LRN-06 terms.
- The Notes-sync "keep both" exit must respect ADR-073 (no automatic winner): it pauses, it does not resolve.

---

<a id="lrn-08"></a>
## LRN-08 · Teach Search vs Ask by consequence at the point of choice

**User problem.** Newcomers ask questions in plain language.
- **Search mode fails on questions.** The rail box flips the panel to Search and says "No evidence matched 'How much did retrieval practice improve retention?'. Try broader terms." The PDF says it verbatim; RAG Answer finds it as "#1 match: strong" (L-13).
- **The modes are named by mechanism** (`mode: ✓ Search ⇄ RAG Answer`), in a word the persona does not know.
- **Cost and model come too late.** They appear only after the paid call, and the model is the wrong one (L-06).
- **Replays and scope surprise.** A history click replays a free search as a paid Ask (L-27); the scope silently carries over (L-24); the answer lands below the fold (L-23).
- **No direct path from a document.** The reader has no "ask about this", so the working path to Console is a 4-step Search detour (S-01; j2 improvement 1; j7 improvement 3).

**Proposal.**
1. **Name the modes by what they do, with the consequence before Run:**
   ```
   mode: ✓ Find passages (local, free) ⇄ Ask (sends question + top passages to OpenAI · gpt-4.1-mini)
   ```
   The model shown is the one that will be billed, so this depends on the L-06 fix.
2. **Steer a question to Ask without spending.** A zero-result Find for a query of ≥4 words, or one ending in `?`, shows: `No exact matches. This looks like a question — ┃ Ask it ┃ (sends to OpenAI · gpt-4.1-mini) · Show related passages (local)`. Nothing runs until the user presses.
3. **Show the scope in the results header:** `2 results · Media only — All sources`. A search started from the rail box resets the scope to all sources (L-24).
4. **Add "Ask about this" in the Media reader** (toolbar or More). It opens the panel in Ask mode with the scope chip `This item only · paper-retrieval-practice`; Select evidence → Use in Console works from there.
5. **History rows carry their mode** (`Ask · retrieval practice`) and replay in that mode (L-27).
6. **Scroll Answer/Evidence into view** when results arrive (L-23).

**User value.** The difference between a free local lookup and a paid model call is learned by reading the control, before the money is spent. The question a newcomer actually types leads to an answer instead of "no evidence".

**Effort:** M. **Impact:** 4.

**Precedent.** Perplexity and Kagi separate searching from asking; Raycast AI and GitHub Copilot Chat show the model before running; Google's separate "AI Mode" tab.

**Risks / ADR.**
- Fits ADR-003 (per-run choices stay in Library) and ADR-079 (Console's three Library mechanisms are untouched).
- The suggestion must never auto-run a paid call.
- Rename consistently in Console, the guide and LRN-06's terms.

---

<a id="lrn-09"></a>
## LRN-09 · The three notes worlds: choose by consequence, always see which one you are in

**User problem.** Library notes, Folder files and synced folders meet in one screen (map-notes §1.1). The one decision point is good: j1 found the Add from files copy "literal and clear", and j3 called the review a peak. Everywhere else the worlds are unlabelled:
- **The landing's `Import…` goes to Media** (S-21).
- **The source strip is unreadable.** "Library notes | Folder files" shows no active state (S-12) and is reachable only by Shift+Tab (task 32649).
- **Unlinked Folder files is mostly blank** (N-37).
- **Synced folders are indistinguishable.** Every one is titled "Sync folder (name unavailable before cutover)" (N-18).
- **Sync health is not shown.** The tree says "⇄ Sync managed · Ready" while sync is wedged or a deletion is unsynced (N-02, N-03), so a user cannot learn which notes really have a file.
- **Folders appear unexplained** (Agent_Lessons; S-23), and an import lands in a collapsed folder (N-27).

Task 32627 ruled "fold the three-worlds decision into one screen" **DEFERRED**, so this proposal keeps the existing screens.

**Proposal.**
1. **Label the strip by consequence and mark the active side:** `✓ Library notes · in Chatbook  |  Folder files · on disk`.
2. **Show every note's world.**
   - The task-32640 "where this note lives" line carries the world for every note: `◆ In Chatbook only`, `⇄ Synced with ~/Vault3 · checked 3 m ago` or `⚠ Sync paused — Review`, and in Folder files `▤ File: ~/notes/x.md`.
   - Synced folder rows in the tree show the root's display name and path (the N-18 fix) and the same health word.
3. **Put a consequence table at the single decision point** (Add from files…), above the three choices:
   ```
                         Where the text lives      Edits made outside Chatbook
   Import once           Chatbook (a copy)         not picked up
   Keep a folder synced  Both, kept in step        picked up after you review
   Folder files          Your folder only          always — it is the same file
   ```
4. **Point to it from the wrong door.** Import media and the landing say "Bringing in Markdown notes? ┃ Add to Notes… ┃ keeps them editable" (S-21). The import receipt's `View N imported notes` expands and selects the destination folder (N-27).

**User value.** A user can answer "if I edit this in Obsidian, will Chatbook see it?" from the screen, before choosing and on every note afterwards. That is PRODUCT.md principle 2 ("make source authority visible") taught without the word "authority".

**Effort:** M. **Impact:** 3.

**Precedent.** macOS iCloud Drive and Dropbox per-file cloud/local glyphs; Obsidian's "Open folder as vault" vs import; Notion's import dialog, which states what is copied.

**Risks / ADR.**
- The table states exactly ADR-059 (manual vs sync-managed membership) and ADR-021/029 (disk is the sole authority for Folder files), so no ADR changes.
- Never show "in step" before sync signals exist for every write path (task 32633) and N-02/N-03 are fixed. Until then the line must say "Not checked since your last change".
- Glyphs are always paired with words (PRODUCT.md: colour or glyph is never the only carrier).

---

<a id="lrn-10"></a>
## LRN-10 · Find Notes from wherever people look

**User problem.**
- **Nav and More menu:** j1's first two actions were to look for "Notes" in the nav bar and in More ▾. Neither has it (j1 steps 1–2).
- **Palette search:** `notes` lists "Switch to Library", which lands on the landing, not on Notes (`evidence-learnability/05`).
- **Palette deep links:** `LIBRARY_SUBROUTE_COMMANDS` (`app_command_providers.py:327-338`, task-423) deep-links only Artifacts and Skills.
- **Palette "New Note"** lands on a tree with blank rows (N-13).
- **No quick switcher:** there is no "Go to note…", and reaching the first filter result takes 8 Tabs (N-21).
- **Wrong landing row:** palette and legacy routes do not land on the promised row (map 7.6), and arrival focus misses the content (S-13).

**Proposal.**
1. **Extend `LIBRARY_SUBROUTE_COMMANDS`** with `Library — Notes`, `Library — Media`, `Library — Conversations`, `Library — Prompts`, `Library — Collections`, `Library — Search / RAG` and `Library — Import`. Each lands on its rail row with focus on the first list row (S-13). Typing `notes` must rank `Library — Notes` first.
2. **Add "Go to note…"**, a fuzzy title switcher with recent notes first, as a palette command and as a Notes-list key. `o` is suggested: today it only opens a focused Search/RAG evidence card (gate at `UI/Screens/library_screen.py:25538-25542`), so a Notes-gated `o` keeps the single meaning "open" (ADR-031 consistency).
3. **Add "Notes (in Library)" to More ▾ "All destinations"**, routing through the same subroute.
4. **Add `Notes (N)` to the landing's quick actions** next to `New note`.

**User value.** The word a user already has ("notes") gets them to their notes in one step, from the nav bar, the palette or Home.

**Effort:** S. **Impact:** 3.

**Precedent.** VS Code `Ctrl+P` go to file; Obsidian's quick switcher; Slack `Ctrl+K`.

**Risks / ADR.**
- ADR-015 keeps one navigation command per destination. Subroute commands are task-423's existing, labelled exception; this extends it and adds no destinations.
- Keep the list to Library rows with real content so the palette stays scannable.

---

<a id="lrn-11"></a>
## LRN-11 · A sample to learn on (local, free, removable)

**User problem.**
- **Nothing to practise on.** Get started's three steps need the user's own files. A newcomer with no ready folder, or one wary of putting private files into an unknown app, cannot try Find it or Use it in Console.
- **The only "try" commits the user.** It is `Try report demo`, which silently creates a persistent live-RSS watchlist with a daily paid run (L-10).
- **The first import is a bad first experience.** It also starts an undisclosed embedding-model download that paints a log line over the nav bar (L-33), and a first Console search waited 15.7 s on it (S-20).

**Proposal.**
1. **Offer the sample from the empty states.** Get started and the Notes / Media zero states (LRN-04) offer a secondary `Try with a sample`.
2. **Ship a small, offline, bundled pack**, labelled throughout as `Sample · remove anytime` (keyword `sample`, notes in a `Sample` folder):
   - 2 short documents, one of them answering a stated question.
   - 2 notes, already linked, one demonstrating Preview, checkboxes and keywords.
   - 1 prompt.
3. **Find works offline and needs no download** (keyword index only). Ask works only if a provider is configured; otherwise it shows the existing no-provider callout and its working fix button.
4. **One `Remove sample` action** removes everything as a single ADR-055 receipt with Undo.

**User value.** A newcomer can walk Import → Find → Ask → Use in Console, and see a linked note, in under a minute, risk-free, before trusting the app with real material.

**Effort:** M. **Impact:** 3.

**Precedent.** Obsidian's sandbox vault; Logseq's demo graph; Notion's "Getting Started" page; Airtable sample bases.

**Risks / ADR.**
- ADR-076 already states that "bundled/system/sample content … are not positive evidence". The sample therefore does not graduate the rail, and Get started stays to teach the loop on the user's own files afterwards.
- The pack loads only on press (ADR-097).
- Sample rows must be excluded from Continue / From your Library once the user has real content.
- Add a drift test so the pack still imports cleanly.

---

<a id="lrn-12"></a>
## LRN-12 · Graduate the rail gradually

**User problem.**
- **One import changes everything at once.** A single import turns a 3-row rail into roughly 25 rows (Browse, Artifacts, Create, Study, Import / Export) with no explanation (`evidence-learnability/01` → `04`).
- **Empty rows compete with real ones.** `Study decks (0)`, `Quizzes (0)`, `Collections (0)`, `Prompts (0)` and the three Artifacts rows sit beside the one item the user has.
- **Glosses depend on width.** The only glosses are row suffixes that drop by width (prior L-p3; Collections' gloss needs 34 cells), and the section heading itself is cut to "Navigat…" at 120 columns (S-16).
- **j2 had to press "Explore all tools"** to learn what Library is for (L-36).

**Proposal.**
1. **Collapse empty sections after graduation.** When the rail first renders graduated, sections whose rows are all zero (Artifacts, Study) default to collapsed, with a summary on the heading row: `Study · nothing yet ▸`, `Artifacts · nothing yet ▸`.
2. **Open a section the first time it gets content.** A section auto-expands once, then the user's own expand/collapse choice (the existing `library.rail_state.sections` pref) always wins.
3. **Move glosses onto a second line under each section heading**, so they survive any width: "Browse — everything you've added", "Create — new notes, prompts, skills", "Study — decks and quizzes from your sources", "Artifacts — reports and Chatbooks you've made". Rows keep their counts.
4. **Restore the heading at every supported width.** The section heading reads "Navigation" at every width (S-16: no mid-word cut).

**User value.** After graduation the rail shows what the user has, plus clearly labelled places they have not used yet, rather than an undifferentiated wall of zeroes.

**Effort:** S. **Impact:** 3.

**Precedent.** macOS Finder's collapsible sidebar sections; Gmail labels set to "show if unread"; VS Code's views that start collapsed until they have content.

**Risks / ADR.**
- ADR-076 keeps section preferences "coerced independently" of lifecycle and says expanded and graduated render the same rail. This changes default section state only, not lifecycle; note it as a clarification.
- ADR-034 owns the `▸`/`▾` glyphs.
- A collapsed section must stay keyboard-reachable and show its count, so Study does not become invisible.

---

## Coverage: findings each idea removes

| Finding | Idea(s) | | Finding | Idea(s) |
|---|---|---|---|---|
| N-01 (P0) | LRN-02 | | L-13 | LRN-08, LRN-03 |
| N-02 (P0) | LRN-07 (exit), LRN-09 (visibility) | | L-22 | LRN-07 |
| N-03 (P0) | LRN-09 (visibility) | | L-23, L-24, L-27 | LRN-08 |
| N-06, N-07, N-12 | LRN-02 | | L-28 | LRN-06 |
| N-08 | LRN-05 (help states it honestly) | | L-30 | LRN-02 |
| N-13, N-21 | LRN-10 | | L-31 | LRN-01 |
| N-15, N-16, N-17, N-24 | LRN-07 | | L-33 | LRN-11 |
| N-18 | LRN-09, LRN-06 | | L-34 | LRN-04, LRN-06 |
| N-25, N-31, N-34 | LRN-02 | | L-35 | LRN-03 |
| N-27 | LRN-09 | | L-36 | LRN-04, LRN-05, LRN-06, LRN-12 |
| N-33 | LRN-05 | | S-01 | LRN-01, LRN-07, LRN-08, LRN-03 |
| N-35 | LRN-06 | | S-08 | LRN-02 |
| N-37 | LRN-04, LRN-09 | | S-11, S-15, S-26 | LRN-05 |
| L-02 | LRN-07 | | S-12 | LRN-09 |
| L-05, L-07, L-08 | LRN-02 | | S-13 | LRN-10 |
| L-06 | LRN-08 | | S-16 | LRN-12 |
| L-10 | LRN-04, LRN-11 | | S-17 | LRN-07 |
| L-12 | LRN-04 | | S-20 | LRN-01, LRN-06, LRN-08 |
| | | | S-21 | LRN-03, LRN-04, LRN-09 |
| | | | S-23, S-24 | LRN-06 |

## ADR summary for triage

- **Needs an ADR amendment or note.**
  - LRN-01 extends ADR-027 (default-workspace chats) to Library sources.
  - LRN-03 adds one persisted "first loop" flag beside `library.rail_state` (an ADR-076 note).
  - LRN-02 rule 6 is an ADR-031 refinement.
- **Explicitly within existing ADRs.**
  - LRN-04 is the source-owned empty-state work ADR-076 defers to atomic tasks.
  - LRN-11 relies on ADR-076's "sample content is not positive evidence".
  - LRN-09 states ADR-059 and ADR-021/029 and keeps 32627's deferral of a single three-worlds screen.
  - LRN-10 extends task-423's labelled exception to ADR-015's one-command-per-destination rule.
  - LRN-12 changes section defaults only.
- **Constraints respected throughout.**
  - No Beginner/Expert mode and no second wizard (ADR-076).
  - No `ctrl+s`-class bindings and F1 stays reserved (ADR-031).
  - The work pane stays permanent (ADR-086).
  - One note-session coordinator (ADR-027, note session coordinator).
  - New copy tables and packs load lazily (ADR-097).
