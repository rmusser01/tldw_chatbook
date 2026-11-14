# Improvements: information architecture and the source-to-output loop

- **Lens:** how Library, Notes, Console, Study and Workspaces relate. Covers reading while note-taking, provenance from source to note to answer to study artifact, one search, one hand-off vocabulary, fewer modes and duplicated patterns, and visible source authority and sync state (PRODUCT.md principles 1, 2, 4, 7, 8).
- **Target:** origin/dev `2d34cbf80d`, worktree `.worktrees/notes-library-ux-review`.
- **Inputs:** `master-findings.md` (103 findings); journeys j1–j7, including their emotional journeys and "improvement opportunities"; the four maps and `maps/known-context.md` (ADR constraints); PRODUCT.md; DESIGN.md.
- **Live check:** one short golden run at 160x45 on socket `nl-imp-ia-1` confirmed four things that the findings do not quote directly:
  - The Library header reads only `Library | Local`.
  - The only workspace surface is buried at the bottom of the rail.
  - Collections links a note by a typed `Note ID`.
  - The Media reader has no note action.

  Captures are in `improvements/ia-loop-evidence/` (`.txt` and `.ansi`). Isolation check: socket killed, real profile untouched, 0 real-profile paths in the run log.
- **Code citations:** paths are relative to `tldw_chatbook/` at `2d34cbf80d`. A cause I could not trace is labelled as a hypothesis.

## Why the loop breaks today

The loop PRODUCT.md promises is *ingest → read and note → ask with sources → keep the answer → study → find again → share*. j7 ("Priya") walked it end to end: it took about 70 inputs and 11 destination switches. Provenance was lost at four points: quoting, capture, search and export. Most of the 47 findings this lens touches come from five structural causes rather than five dozen separate bugs:

1. **There is no hand-off model, only a set of buttons.**
   - Eight surfaces send things to Console, through four hand-off channels and three labels (map-conv #48).
   - `c` means "stage" in Media and "resume" in Conversations (`UI/Screens/library_screen.py:1184` vs `:1190-1194`).
   - Console holds exactly one pending launch (`_set_pending_launch`, `UI/Console_Modules/retrieval.py:634-644`), and every hand-off channel is a typed single slot (`UI/Navigation/pending_handoff_store.py:88-94`). Each new hand-off therefore silently replaces the last one.
2. **Provenance has nowhere to live.**
   - Notes can only link to notes (the `note_links` table).
   - Media, conversations and captures have no link type that a note can hold.
   - Collections links notes by a raw ID that Notes never shows.
   - Four separate annotation stores (Highlights, Analysis, Capture note, Linked Notes) do not connect to each other.
3. **Status is computed in several places and they disagree.**
   - The tree badge, the editor location line and the sync root row each derive "synced" independently.
   - Console's strip, Inspector and status bar each count staged sources independently.
4. **Workspace is a hidden gate, not visible context.**
   - The built-in Default workspace refuses media.
   - The only explanation sits under a collapsed "Details ▸" at the bottom of the rail.
5. **Every destination has its own search, its own selection and its own memory.**
   - Each uses a different grammar.
   - None of them survives a switch.

The 13 improvements below each target one of these causes. Each idea is scoped so it can be designed on its own; the dependencies between them are listed at the end.

## Summary

| ID | Improvement | Effort | Impact | Main findings removed | ADR consequence |
|---|---|---|---|---|---|
| [IA-01](#ia-01) | One hand-off vocabulary and an accumulating Console context | M | 5 | S-06, S-18, S-19, S-20, S-28, L-09, L-26, N-05 | Respects ADR-079; key changes under ADR-031 |
| [IA-02](#ia-02) | The Default workspace means "everything local"; workspaces appear only when they matter, then behave one way | M | 5 | S-01, S-24 | Extends ADR-027 (default-workspace-chats) to Library |
| [IA-03](#ia-03) | Source links: one provenance edge from notes to anything in Library, written by every capture | L | 5 | S-03, S-04, N-08, N-09, L-07, L-08, L-34 | Schema migration; ADR-105 (portable notes) must include or exclude the edge explicitly; locators follow ADR-024 |
| [IA-04](#ia-04) | A reading desk: a companion note beside the reader | L | 5 | S-02, S-03, L-16, L-25 | **Amends ADR-086 and ADR-084** |
| [IA-05](#ia-05) | One health projection for "where this note lives", read by every surface | M | 5 | N-02, N-03, N-15, N-16, N-17, N-18, N-19, N-35 | Within ADR-059/073 |
| [IA-06](#ia-06) | One Library search: one grammar, a cross-type "Go to…", scope always visible | L | 5 | N-09, N-21, L-13, L-24, L-27 | Within ADR-013/067 |
| [IA-07](#ia-07) | One answering engine: Library questions become Console turns | M | 4 | L-06, L-27, S-04, L-23 | Within ADR-005/079; applies PRODUCT principle 1 |
| [IA-08](#ia-08) | A context bar: workspace, staged context and sync health in the Library header | M | 4 | S-01, S-17, S-19, L-24, N-02 | Extends ADR-015 DestinationHeader |
| [IA-09](#ia-09) | One selection, one grammar, every verb | M | 4 | N-14, L-19, S-28, S-06 | Within ADR-055/067 |
| [IA-10](#ia-10) | Pick up where you left off: per-destination memory, back/forward, a reachable Continue | M | 4 | S-02, S-22, L-16, N-21, N-27 | Within ADR-104/076 |
| [IA-11](#ia-11) | Study from chosen sources, with an honest capability line and a way back | M | 4 | S-05 | None (local generation would be a new capability) |
| [IA-12](#ia-12) | Export that keeps provenance outside Chatbook | M | 3 | L-04, L-18, N-11, N-20, S-04 | None |
| [IA-13](#ia-13) | One door in, routed by what you want to do with the material | M | 3 | S-21, N-17, N-37, L-34, L-36 | Must stay within ADR-076 (no second wizard) and ADR-113 |

Together these ideas address 47 of the 103 findings: 14 S, 17 N and 16 L. The full list is at the end.

---

<a id="ia-01"></a>
## IA-01 · One hand-off vocabulary and an accumulating Console context

**Problem.** "Send this to Console" is eight surfaces with four behaviours:

| Surface | What it does today |
|---|---|
| Media reader **Use in Console** / `c` | Stages one source, or is refused (S-01) |
| Note editor **Use in Console** (no key) | Auto-links the note to the workspace, then stages it |
| Conversations `c` | **Resumes** the chat |
| Conversations **Use as source** | Stages the chat |
| Search evidence **Use in Console** / `u` | Stages it as "Review evidence in Console" |
| Prompt **Use in Console** | Inserts into the composer |
| Chatbook **Use in Console** | Opens a live-work card |
| Rail Details **Use in Console** | Stages the whole Library snapshot |

Study adds a fifth label: three different buttons all read "Continue in Study". Sources: map-conv #48–49; `library_screen.py:1184`, `:1190-1194`.

The consequences, by finding:

- **S-06:** the second hand-off replaces the first. Console keeps one pending launch, and each hand-off calls `_set_pending_launch` (`UI/Console_Modules/retrieval.py:641`) through `_stage_handoff_as_console_live_work` (`UI/Screens/chat_screen.py:17531`).
- **S-18:** a deleted note stays "Ready" and is sent anyway.
- **S-19:** after a send, the strip says 1 source, Inspector says "None staged" and the status bar says 0.
- **S-20:** the Search Library modal says "staged" before any results exist.
- **L-09:** `c` resumes a conversation that the filter has hidden.
- **N-05:** the note editor's label clips to "Use in".
- **S-28:** the note editor's hand-off has no key.
- **L-26:** the prompt insert drops Instructions by default.

**Proposal.**

1. **Five verbs.** Each has one label, one key and one meaning on every Library surface. Footer, F1 and buttons use the same strings.

   | Verb | Key | What it does | Replaces |
   |---|---|---|---|
   | **Add to context** | `c` | Appends the focused item, the selection, or the selected evidence cards to Console's context for the next send. De-duplicated by canonical id (`note://`, `media://`, `conversation://`). **You stay in Library.** Toast: "Added 'paper-retrieval-practice' · Console context: 3 · ⌃2 opens Console". | Media `c`, Conversations "Use as source", Search `u`, the note editor's "Use in Console" |
   | **Ask in Console** | `a` | Add to context, then open Console with the composer focused (prefilled with the query when started from Search). The only verb that navigates. | — |
   | **Resume in Console** | button only | Conversations reader only. `c` stops meaning Resume, which removes the L-09 trap. | Conversations `c` |
   | **Insert in composer** | — | Prompts only. The dialog shows its lanes, with **Instructions ☑** on whenever the prompt has instructions. | Prompt "Use in Console" |
   | **Make study set…** | — | See IA-11. | "Continue in Study" ×3 |

   The rail-level "Use in Console" (whole-Library snapshot) is retired. It is replaced by asking with the scope chip set to "All sources" (IA-06/IA-07).
2. **One context list, owned by Console.** It replaces the single pending launch. Library posts "add" intents, and each one appends an `EvidenceReference` to the staged bundle. `EvidenceBundle` already holds many references; the Search Library path builds one from all its results. Console renders the list in the Inspector with authority and freshness on every row:
   - `note · Thesis outline · edited 2m ago`
   - `media · paper-retrieval-practice · 3.0 KB`
   - `⚠ note deleted — will not be sent · Remove`

   The header shows a size: "Context for next send · 3 items · ~6k tokens".
3. **Send revalidates every row (S-18).** After a send, the rows move to "Used in this conversation (3) · Use again next send". The strip, the Inspector and the status bar all read this one list, so they cannot disagree (S-19).
4. **Library shows the count** in the context bar (IA-08): "Console context: 3 ▸" opens the list, where each row has Remove.

**User value.** A user can build a grounded question from several notes and papers without bouncing to Console after each one. A staged source is never lost silently. There is one key to learn instead of three meanings for `c`.

**Effort:** M · **Impact:** 5 · **Addresses:** S-06, S-18, S-19, S-20, S-28, L-09, L-26, N-05 (the label no longer needs to fit)

**Precedent.** NotebookLM's source checkboxes; Cursor's and Claude Code's `@` context chips; "Add to cart" versus "Buy now". In this repo:
- Console's Search Library path already stages a multi-reference evidence bundle (`build_library_rag_evidence_bundle(outcome.results)`, `UI/Console_Modules/retrieval.py:685`) with Un-stage.
- Console's Inspector already shows "Sources — next send N".

**Risks and constraints.**
- ADR-079: this is the *manual* mechanism. It must never switch on `auto_retrieve_on_send` or assistant Library access as a side effect.
- ADR-031 key changes:
  - `c` changes from "stage and navigate" to "stage and stay", so the toast must name the way to Console.
  - `a` is unbound in Library at `2d34cbf80d` (checked against `BINDINGS` and `on_key`); confirm again at implementation time.
  - The footer may advertise only keys that work.
- Handoff channels are single-slot by design (`pending_handoff_store.py:88-89`: "Typed single-slot channels"). Because Add to context keeps the user in Library, several adds can land before Console claims the channel, and a single slot would keep only the last. The adds must therefore write to an accumulating, Console-owned staged-context store (directly, or through a new append-only channel), never through `CHAT`. The other channels can stay single-slot.
- A large context needs a visible size and a cap before send.

---

<a id="ia-02"></a>
## IA-02 · The Default workspace means "everything local"; workspaces appear only when they matter, then behave one way

**Problem.** On a fresh profile, Media **Use in Console** is always refused: "Copy or link this media into workspace workspace-default before using it in Console." (S-01). The built-in Default workspace is active with 0 memberships, and `Workspaces/eligibility.py:73-82` refuses any item that is not linked, interpolating the raw id.

Three link policies coexist:

| Item type | Policy |
|---|---|
| Notes | Auto-link on use: "Linked to Local Default · staged in Console" (`UI/Library_Modules/library_notes_controller.py:4594-4616`) |
| Conversations | Link through Use as source, with an Undo receipt |
| Media | Refused, with no remedy |
| Search evidence | Stages the same paper with no check at all |

The only workspace surface in Library sits at the bottom of the rail, under a collapsed "Details ▸" (live `ia-loop-evidence/02-details-160x45.txt`): "Workspace · Active · Local Default · Handoff · 135 items can't be used in Console yet · not in this workspace · Copy or link them into this workspace". Its only action is "Create local workspace". S-24 covers the raw-id vocabulary.

**Proposal.**

1. **Default = everything local.** Treat `workspace-default` exactly like the "no active workspace" branch: every local item is eligible. While Default is the only workspace, Library never uses the word "workspace". This extends to Library the rule that ADR-027 (default-workspace-chats) sets for Console: "Everyday chatting must not demand workspace vocabulary".
2. **When a user-created workspace is active, it becomes visible context:**
   - The context bar (IA-08) shows "Workspace: Thesis ▾", with Switch, Create workspace… and "Show only Thesis items".
   - List rows carry a quiet fact: "· in Thesis".
   - Search gets a scope chip: "Thesis only ✕".
3. **One link policy for every item type.** If Add to context or Ask (IA-01) is used on an item outside the active workspace, the item is linked and the receipt says so: "Added to Thesis and to Console context · Undo link".
   - A button is blocked only when the item genuinely cannot join. It then reads "○ Add to context · server item — not in Thesis", with the fix button beside it.
4. **Retire the Details ▸ Workspace group.** Show display names only, never `workspace-default`.

**User value.**
- A first-timer's core loop (import → ask) works on the first press.
- Workspace users see and control scope where they act, instead of hitting an unexplained gate.
- Principle 7 ("Workspaces as global context") becomes visible.

**Effort:** M (the Default eligibility change alone is S) · **Impact:** 5 · **Addresses:** S-01, S-24

**Precedent.** ADR-027 (default-workspace-chats): the Default workspace exists "so the data model is uniform, not so users think about it", and the switcher shows "Default (everyday chats)". VS Code needs no workspace to open a folder. Slack and Notion put the workspace switcher in the header. In this repo: the workspace-hop Undo receipt (task-32388) and the conversation receipt "✓ linked · <ws> · Undo link".

**Risks and constraints.**
- Auto-linking writes membership, so the Undo receipt is mandatory.
- Server-backed items must keep refusing honestly when they really cannot join.
- The Library switcher must be the same control and the same state as Console's workspace switcher, not a copy (principle 7).

---

<a id="ia-03"></a>
## IA-03 · Source links: one provenance edge from notes to anything in Library, written by every capture

**Problem.** Provenance is lost at the moment of capture.

- **S-03:**
  - A highlight card's only action is "✕ Delete".
  - Copy and paste carries no attribution.
  - Note Info has no Source field.
  - Notes support only `note://` links; `media://` exists only for MCP (`MCP/resources.py:254`).
- **N-08:** a typed `[[Title]]` is dead text, and Preview hands `note://` links to the OS URL handler.
- **Collections links notes by typing a raw ID** that Notes never displays. Live `ia-loop-evidence/05`: "Linked Notes · No Notes are linked to this capture. · [Note ID] · Link Note".
- **Annotation is split across four unconnected stores:**
  - Media Highlights, with an optional note.
  - Media Analysis, a second TextArea without the note editor's guarantees: focus is not moved into it (L-07), and Escape discards it (L-08).
  - The Collections "Capture note" box (live `04`).
  - Collections "Linked Notes".
- **S-04 / N-09:** captured answers record their provenance as UUID keywords (`conversation:71d30fb1-…`) that the filter cannot find.

**Proposal.**

1. **One edge type.** Generalise `note_links` (note → note; `DB/ChaChaNotes_DB.py` around `:17714` and the backlink query at `:17961`) to note → source:
   - Target kind: note, media, conversation, capture or prompt.
   - An optional locator in ADR-024's `SourceLocatorEnvelope` form: chunk or character range, or message id.
   - The link lives in the note's own Markdown as a standard link, e.g. `[paper-retrieval-practice](media://26#chunk=4)`. Folder-files notes therefore keep disk authority (ADR-021/029), and the edge index is rebuilt from text.
2. **Capture verbs on any reader selection.** Reuse the floating selection menu Console already has (ADR-068): `q Quote to note · h Highlight · c Add to context`.
   - **Quote to note** inserts `> passage` and `— [title](media://26#…)` at the caret of the companion note (IA-04), or into a picked note.
   - Highlight cards gain **→ Note**.
3. **Two panels, everywhere:**
   - Every item's Info shows **Notes about this (2)**: Enter opens a note; the panel offers "New note about this" and "Link a note… (pick by title)", which replaces the Note ID box.
   - Note Info shows **Sources (2)** beside **Linked from (1)**.
4. **Links that work in Preview.**
   - Source links render as labelled chips ("▸ paper-retrieval-practice · pdf") and open in-app. Use `open_links=False` plus a `LinkClicked` handler; today's construction is at `Widgets/Library/library_notes_canvas.py:2909-2913`.
   - A typed `[[Title]]` resolves to `note://` on save.
5. **One editor for hand-written annotation.**
   - Media "Add analysis" becomes "New note about this", which opens the coordinator-backed note editor instead of a second TextArea.
   - AI-generated analysis keeps its tab and gains "Save as note about this".
   - This removes the class of defects behind L-07 and L-08 rather than patching each one.

**User value.** Every quote, note and kept answer leads back to its source in one key. Annotation lives in one place that can be searched, linked and exported. The UUID keyword (S-04) becomes a readable link.

**Effort:** L · **Impact:** 5 · **Addresses:** S-03, S-04, N-08, N-09 (provenance no longer hides in unsearchable UUID keywords), L-07, L-08, L-34

**Precedent.**
- Zotero child notes and "Add note from annotation".
- Readwise: highlight → note.
- Obsidian's backlinks pane; Logseq block references.
- In this repo:
  - `note_links` and the backlink query (task-32186).
  - Collections `link_note` / `unlink_note` (`Library/collections_capture_repository.py:1065-1110`).
  - ADR-068's selection menu with "Add to chat".
  - MCP `media://` resources.

**Risks and constraints.**
- **Schema change.** Follow `DB/migrations/README.md`: a `VALID_TABLES` entry, and an index-plan pin captured with `sqlite_stat1` absent.
- **ADR-105.** Portable notes organization syncs as one capability, so source edges must either be in that capability or be declared device-local.
- **Dangling targets.** A trashed item renders as "source in Trash · Restore" (ADR-055 Pattern A).
- **Server mode.** Server-mode Collections capture notes stay server-authoritative (ADR-113). Show them in the same panel with an authority label; do not migrate them.

---

<a id="ia-04"></a>
## IA-04 · A reading desk: a companion note beside the reader

**Problem.**
- **S-02:** the rail's "New note" *replaces* the reader. One round trip costs about 8 clicks plus 5 keys, and it discards both the reading position and the open note. j7's task 2 ("read and take notes together") failed outright.
- **L-16:** back on Media, the row says "loaded" beside "Select a media item to read it here."
- **No note action in the Media reader.**
  - `n` does nothing (S-02, j3 08).
  - Live `ia-loop-evidence/07`: the toolbar is "Find · Read later · Use in Console · More". The footer offers `] [ ctrl+f l c t s` and no note key.
- **L-25:** the reader is the narrowest pane.

**Proposal.**

- **A Note action in every reader.** Add **Note** (key `n`, footer "n note on this item") to the Media, Conversations and Collections readers. It opens the item's *companion note* inside the work pane, next to the reader:
  - The reader stays on the left and keeps width priority.
  - The note is on the right, at least 40 cells wide.
- **Which note opens.**
  - If the item already has notes about it (IA-03), `n` opens the most recent one. Its header shows "About: paper-retrieval-practice · 2 notes ▾".
  - Otherwise `n` creates "Notes — <item title>", already linked to the item and filed in a "Reading notes" folder.
- **Narrow layout.** When the work pane is too narrow for both (below about 100 cells, e.g. 120x36), the desk stacks instead of splitting:
  - `n` flips between the **Reading** and **Note** faces of the same desk.
  - A one-row header always names both faces: "Reading: paper-retrieval-practice · Note: Notes — paper… (saved 20:12)". Nothing closes.
- **Quoting.** `q` (Quote to note, IA-03) on a reader selection inserts the quote and its source link at the note's caret, without moving focus.
- **Leaving.** Esc in the note returns focus to the reader. A second Esc closes the desk with the receipt "Saved to 'Notes — paper…' · n to reopen".
- **Panes.** While the desk is open the Items list collapses (the list collapses before the work pane, per ADR-086), and it returns when the desk closes.
- **Saving.** The companion uses the Notes session coordinator (ADR-027, portable-database-note-session-coordinator: one coordinator owns draft and save): autosave and flush-on-leave come for free.

**User value.** The user reads and writes in one view. The 13-input cycle of the note-taking loop becomes one key, with no state lost in either direction.

**Effort:** L · **Impact:** 5 · **Addresses:** S-02, S-03 (with IA-03), L-16, L-25

**Precedent.** Zotero 7's reader with its notes pane; the notebook sidebar in Readwise Reader; LiquidText; split editors in Obsidian and VS Code.

**Risks and constraints.**
- **Requires amending ADR-086 and ADR-084.** The permanent work pane gains a defined two-document layout and a collapse order:
  1. Items collapses.
  2. The companion stacks.
  3. Only then does the reader shrink.

  The Reader remains the width priority.
- **Two text surfaces raise the single-letter-key risk** (S-08, N-31). Bare letters must be inert while the companion's text field has focus.
- **ADR-097:** mount the companion lazily.
- **ADR-104:** restoring the desk on return must be event-driven.

---

<a id="ia-05"></a>
## IA-05 · One health projection for "where this note lives", read by every surface

**Problem.** Three independent projections describe a synced note, and they disagree:

| Surface | What it shows | Where it comes from |
|---|---|---|
| Tree folder badge | "⇄ Sync managed" | Membership protection, not health (`Library/library_notes_tree_state.py:998-1004`) |
| Editor location line | The fixed prefix "In a synced folder" plus a path | `Widgets/Library/library_notes_canvas.py:228`, `:261` |
| Sync root row | "✓ Up to date" | The sync controller (`UI/Library_Modules/library_notes_sync_controller.py:250`) |

Result:
- **N-02 (P0):** sync is wedged while the editor says "Saved", the tree says "⇄ Sync managed" and the list says "Ready".
- **N-03 (P0):** "✓ Up to date" after a synced note is deleted.
- **N-18:** every root is titled "Sync folder (name unavailable before cutover)".
- **N-19:** "No writes yet." when the write history could not be read.
- **N-15, N-16, N-17:** dead-end states.
- **N-35:** raw nanosecond times and exception class names.

**Proposal.**

**One projection per sync root.** It carries:
- the root's name and folder path;
- a state, one of: In sync · Changes to check · Not checked since your edit · Needs review (N files) · Paused · Stopped — <plain reason>;
- the last check time;
- exactly one next action that works.

**Every surface renders that projection, in the same words:**

| Surface | Example |
|---|---|
| Tree folder row | `▸ Vault3  ⇄ in sync` / `▸ Vault3  ⚠ not syncing — review` |
| Editor location line | `In synced folder Vault3 · ⚠ not syncing since 20:33 · Review` |
| Manage sync folders heading | `Vault3 · ~/Notes/Vault3`, with every receipt prefixed by the root name |
| Context bar (IA-08) | `Notes sync ⚠ 1` |

**Invariants**, each with a test:
- No surface may render a healthier state than the projection.
- "In sync" requires a check newer than the last local change to a managed note. This fixes N-03.
- An unreadable history reads "History unavailable — Retry", never "No writes yet." (N-19).
- One blocked file blocks only itself. Its row names the file and a plain reason, e.g. "not UTF-8 — skip or convert" (N-16).
- Every state that is not green offers one action that works, and always includes "Pause and keep both copies".

**User value.** The one status a sync user has to trust reads the same everywhere and is never optimistic.

**Effort:** M · **Impact:** 5 · **Addresses:** N-02, N-03, N-15, N-16, N-17, N-18, N-19, N-35

**Precedent.** Source-control decorations and the sync indicator in the VS Code status bar; per-file badges in Dropbox and iCloud; Obsidian Sync's status icon.

**Risks and constraints.**
- This does not fix N-02's root cause (the postcondition compare at `Notes/notes_sync_executor.py:5384`). It removes the class of false status around that failure, and any failure like it.
- ADR-073 (no automatic winner) and ADR-059 (membership classes) are unchanged.
- The projection must update from events, not from per-row polling (open tasks 281 and 33286).
- Folder files (ADR-021/029, disk authority) gets its own states in the same vocabulary: "Saved to disk", "Changed on disk — Compare".

---

<a id="ia-06"></a>
## IA-06 · One Library search: one grammar, a cross-type "Go to…", scope always visible

**Problem.** Every list searches differently:

| Surface | How its search behaves | Finding |
|---|---|---|
| Notes filter | Exact whole-word phrase over title and body only: "meet" finds 0 where "meeting" finds 3; keyword "zebra" 0; "71d30fb1" 0 | N-09 |
| Media filter | Filters as you type | map-media 3.7 |
| Collections filter | Applies on Enter | map-media 3.7 |
| Search/RAG, keyword mode | "No evidence matched" for a plain question the document answers word for word; the rail box silently flips the mode | L-13 |
| Search/RAG, between queries | Keeps the previous query's source toggles, with the scope line scrolled above the viewport | L-24 |
| Recent searches | Replays under the current mode, which can make a paid call | L-27 |

Two more gaps:
- Captures are not a search source (map-media 3.5).
- From the Notes filter, reaching a note takes 8 Tabs, and there is no quick switcher (N-21).

**Proposal.**

1. **One query grammar.** It is parsed once and used by every list filter and by Find:
   - words are ANDed, and the last word matches as a prefix;
   - `"quoted phrase"` matches exactly;
   - `#keyword` searches note and media keywords;
   - `type:note|media|chat|prompt|capture`;
   - `in:"Folder"`;
   - `cites:"paper title"`, using IA-03's edges.

   The placeholder teaches it ("words, \"phrase\", #tag"), and F1 lists it once.
2. **Go to… (`ctrl+k` in Library; also a palette entry).** The key mirrors Console's `ctrl+k` session switcher (`UI/Screens/chat_screen.py:1936`).
   - An overlay shows live title matches across all item types, recent first. Each row shows a type badge and its folder or source.
   - Enter opens the item in its home view.
   - The last row, "Search all of Library for '<q>' ↵", goes to Find.
3. **Scope is always in the results header:** "5 results · All sources ▾ · This item ✕ · Thesis ✕".
   - A search started from the rail or from Go to… resets the scope to All.
   - History rows store their mode and scope and show both (RAG rows are marked "RAG · paid"). They replay exactly as saved.
4. **A gentler zero result.** When a 4+ word query gets zero keyword hits, Find falls back to hybrid retrieval under the label "No exact matches — related passages" (L-13).

**User value.** Tag and type-ahead habits work. There is one mental model of search across every destination, and going from a query to a note takes 3 keys instead of about 11.

**Effort:** L (the grammar alone is M; Go to… alone is M; adding captures to the index is L) · **Impact:** 5 · **Addresses:** N-09, N-21, L-13, L-24, L-27

**Precedent.** Obsidian's quick switcher; VS Code's ctrl+P; GitHub and Gmail search qualifiers; Zotero's quick-search modes.

**Risks and constraints.**
- **ADR-013:** the raw text is preserved and FTS is safely quoted at the boundary, so the parser must emit quoted FTS tokens.
- **ADR-067:** Go to… shows the top N matches, not a page.
- **Per-keystroke cost:** see open task 32804.11.
- **ADR-113:** captures stay under Collections authority, so the index is read-only.
- **ADR-031:** ctrl+k is not reserved or forbidden. At `2d34cbf80d` only Console binds it, to its session switcher, so the meaning carries over: ctrl+k means "switch to…" in both places.

---

<a id="ia-07"></a>
## IA-07 · One answering engine: Library questions become Console turns

**Problem.** Library's RAG Answer is a second answering surface with its own rules:

- **L-06:** it calls the provider's default model (`gpt-5.6-terra`) instead of the configured `gpt-4.1-mini`, and names the model only after the paid call (`Library/library_rag_answer_service.py:163-189`).
- **L-27:** Recent searches replays a paid call.
- **S-04:** its answer cannot be saved, staged or resumed (map-media 3.5: "The generated answer itself is not stageable").
- **L-23:** the source toggles push the answer below the fold.

Console's answers, meanwhile, already have a model chooser, a cost line, CitationTrace provenance (ADR-024) and "Capture as note". But that capture drops the question and the sources and writes UUID keywords instead (S-04; `UI/Console_Modules/message.py:2169-2206`). PRODUCT principle 1 says Console is the live work surface.

**Proposal.**

1. **Library Search/RAG becomes Find:** free retrieval, evidence cards and multi-select, plus one paid button, **Ask in Console**.
   - The pre-run line names everything: "Ask in Console · OpenAI gpt-4.1-mini · question + 3 passages · ≈$0.001".
   - It opens Console in a new conversation titled with the question, with the passages in context (IA-01). The question is sent, or prefilled, depending on a setting.
2. **Ask about this.** `a` in any reader, note or selection does the same, scoped to those items. `Chat/rag_scope.py` `EffectiveScope` already supports per-id allowlists for notes and media (`Library/library_local_rag_search_service.py:433-453`).
3. **Library keeps a trace of each question.**
   - A receipt row: "Asked in Console · How much did retrieval practice… · Open".
   - Recent searches lists Finds, which are free, and questions, which open their conversation and never re-run.
4. **Capture as note writes provenance from the CitationTrace:**
   - the question, the model and the date;
   - "Sources:" as IA-03 links;
   - "From: [conversation title](conversation://…)".

   It writes no UUID keywords.

**Option B**, if an inline Library answer must stay: run it through the Console send pipeline into a persisted "Library questions" conversation. Model, cost, trace and capture are then shared, and the inline panel only shows the answer.

**User value.** Answers come from one place, with one model and one cost line. Every answer can be resumed, continued and kept with its sources.

**Effort:** M · **Impact:** 4 · **Addresses:** L-06, L-27, S-04, L-23

**Precedent.** NotebookLM (sources → chat with inline citations → "Save to note"); Perplexity Spaces. In this repo: Console's Search Library modal, and "Capture as note" (task-32146).

**Risks and constraints.**
- Users who like the inline Library answer lose it; Option B mitigates this.
- Console must carry over the caution Library shows today, "The answer does not cite available staged evidence." (j7 strength 2).
- ADR-079: the passages are manual staging. Asking must never turn on auto-retrieve or assistant tools.
- ADR-005 is unaffected, because retrieval stays local.
- Library's Generate analysis should use the same provider resolution. Its separate config path is the root of L-02.

---

<a id="ia-08"></a>
## IA-08 · A context bar: workspace, staged context and sync health in the Library header

**Problem.** Library's header says only `Library | Local` (live `ia-loop-evidence/01-landing`). The facts that decide what an action will do live elsewhere:

- **Workspace:** the active workspace and "135 items can't be used in Console yet" sit under a collapsed Details section at the bottom of the rail (live `02`).
- **Staged context:** visible only in Console, where three counters disagree (S-19).
- **Sync health:** visible only inside Manage sync folders (N-02, N-03).
- **Search scope:** scrolled above the viewport (L-24).
- **Vetoed navigation:** silent, and it leaves the nav bar highlighting the wrong destination (S-17).

**Proposal.**

- **Extend Library's DestinationHeader (ADR-015)** to: `Library | Local · Workspace: Thesis ▾ · Console context: 3 ▸ · Notes sync ⚠ 1 ▸`.
- **Each fact is a focusable button** that opens its own surface: the workspace switcher, the context list with Remove, or the sync review.
- **A fact stays silent at its default.** The Default workspace, an empty context and healthy sync add nothing, so a first-timer sees today's header.
- **The facts read the IA-01, IA-02 and IA-05 projections,** so Library and Console cannot disagree.
- **A vetoed navigation** shows its reason in the bar's status slot, and the nav bar re-selects the current destination (S-17).
- **Below about 100 columns** the bar shortens to `Library | Local · Ctx 3 · Sync ⚠1`. It is always text, never colour alone.

**User value.** One glance before acting. Nothing important hides behind Details or in the log.

**Effort:** M (S once the IA-01, IA-02 and IA-05 projections exist) · **Impact:** 4 · **Addresses:** S-01 (visibility part), S-17, S-19, L-24, N-02 (visibility part)

**Precedent.** The VS Code status bar (branch, sync, problems); the context switchers in the Figma and Linear headers. In this repo, the DestinationHeader status badge (ADR-015).

**Risks and constraints.**
- **Anti-reference "control-room theater":** cap the bar at four facts, silent by default.
- **ADR-097:** the header must not import the sync runtime at boot; fill it lazily.
- **80-column fit:** must be verified live at 80x24 and 120x36.

---

<a id="ia-09"></a>
## IA-09 · One selection, one grammar, every verb

**Problem.**
- **N-14:** Notes select mode can only export. Selecting costs Enter+Down per row, and "Select all 100 shown" selects 26.
- **L-19:** Media selection is page-local. Next clears it with only a toast, and there is no bulk keyword action.
- **S-28:** Media uses `s`/Space while Notes uses a button and Enter; `e` works only in Notes; `c` only in Media and Conversations.
- **S-06:** there is no way to use several notes as context.
- The same gap is j4 idea 1 and j3 idea 6.

**Proposal.**

- **The same keys on every list canvas:**
  - `s` enters and leaves select mode;
  - Space toggles the row;
  - Shift+↑/↓ extends the selection;
  - Esc leaves select mode;
  - "Select all 34 matching" is counted from the same projection the action uses.
- **The selection is an id set.** It survives paging, filtering and switching rail rows. A chip in the list header reads "7 selected · 3 notes · 4 media ▸".
- **Verbs act on the whole set,** with counts and reasons: `Add to context (7) · Make study set (7) · Export (7) · Keywords… (7) · Move to folder… (3 notes) · Trash (7)`.
- **Trash uses the single-delete seam and its receipt** (ADR-055).

**User value.** Triage, re-file, tag and gather context at keyboard speed, with one grammar learned once.

**Effort:** M · **Impact:** 4 · **Addresses:** N-14, L-19, S-28, S-06

**Precedent.** Gmail's "Select all conversations that match this search"; Lightroom's Quick Collection; multi-select in Finder and Zotero. In this repo, Prompts' "N selected (M on this page)".

**Risks and constraints.**
- **ADR-055:** bulk delete reuses the single-delete seam.
- **ADR-067:** with pages capped at 20, "matching" and "loaded" must be labelled explicitly.
- **Mixed types:** verbs that apply to only part of the set must say so.
- **Persistence:** the selection lasts for the session only.

---

<a id="ia-10"></a>
## IA-10 · Pick up where you left off: per-destination memory, back/forward, a reachable Continue

**Problem.**
- **S-02 / L-16:** switching rail rows closes the open note and blanks the reader, while the row still says "loaded".
- **S-22:** a relaunch drops the open note, the filter and the expanded folders. `ScreenStateStore` is memory-only.
- **N-21:** there is no note history, so following a backlink is a one-way trip.
- **map-shell 7.3:** once you leave the landing (and its Continue), you cannot get back to it, and Continue can be stale.
- **N-27:** "View 10 imported notes" lands on the list with the import folder collapsed.

**Proposal.**

1. **Per-row memory.** Each rail row keeps its own work state: open item, tab or mode, scroll, caret, filter and expanded folders. Switching never closes anything, so the row's "loaded" fact is true by construction.
2. **Library history.**
   - Alt+← / Alt+→ walk back and forward through the items opened in Library. That includes links followed from Preview, backlinks and source links.
   - The footer names the target: "alt+← back to 'Index — start here'".
   - The palette offers "Back" and "Forward".
3. **Persistence across relaunch.** Save the last item per destination, the filter and the expanded folders, next to the existing `library.reader` config.
4. **A reachable home.** "Library home" becomes the rail's first row, with "Continue: Paper notes · paper-retrieval-practice · Thesis chat".
5. **Arrivals expand their target.** An arrival such as "View 10 imported notes" expands and scrolls to the folder it names (N-27).

**User value.** Starting the day, or moving round the read ↔ note ↔ ask loop, costs no re-navigation.

**Effort:** M · **Impact:** 4 · **Addresses:** S-02, S-22, L-16, N-21, N-27

**Precedent.** Browser back and forward; VS Code's Go Back / Go Forward and restored editors; Zotero restoring open reader tabs; Obsidian's workspace restore.

**Risks and constraints.**
- **ADR-104:** restoration must be event-driven, two-gate and authority-fenced, with no timers.
- **Deep links win** over remembered state (ADR-076).
- **Missing targets:** a deleted or trashed item needs a named fallback, e.g. "'Thesis outline' was moved to Trash · Restore".
- **Terminal support:** Alt+arrow delivery varies by terminal, hence the palette entries.

---

<a id="ia-11"></a>
## IA-11 · Study from chosen sources, with an honest capability line and a way back

**Problem.**
- **S-05:** the Study hand-off carries the whole Library: "Carries forward: … and 134 more · Source snapshot is ready · Continue in Study". Then:
  - the Study dashboard's controls never render;
  - local mode says "Source generation requires server mode." (`UI/Screens/study_screen.py:620-625`), after Library said "ready".
- **map-conv #49–51:**
  - All three buttons say "Continue in Study".
  - Escape from Study always lands on "Study decks" (`study_screen.py:1352-1356`).
  - The copy uses jargon ("Source snapshot", "carries forward").

**Proposal.**

- **Make study set… (an IA-01 verb)** is offered in the note editor, in every reader and on a selection.
- **The hand-off lists exactly the chosen items:** "From: Paper notes: Retrieval practice (note) · paper-retrieval-practice (pdf)". `StudyScopeContext.source_items` already carries per-item sources (`UI/Screens/study_scope_models.py:83-94`).
- **One capability line before the button:**
  - server mode: "Generate cards on server · 2 sources";
  - local mode: "Card generation needs a server here — write cards by hand with the sources beside you", with **Write cards…** as the primary action.
- **Every card stores a source link (IA-03),** shown on the back of the card as "From: paper-retrieval-practice".
- **Esc returns to the item you came from.**
- **The whole-Library snapshot becomes an explicit choice,** "Use all of Library".

**User value.** The study step of the loop stops dead-ending, and every card can be traced back to its source.

**Effort:** M (L if a local generation path through the configured provider is added) · **Impact:** 4 · **Addresses:** S-05

**Precedent.** NotebookLM's flashcards and quizzes from selected sources; cards made from highlights in RemNote and Anki, with source links.

**Risks and constraints.**
- **Local generation is a new capability.** Study_Interop is server-only today (`_server_only_service`), so adding it needs principle-8 cost disclosure.
- **Keep Study a destination,** not a Library mode (anti-reference: "study-only app").

---

<a id="ia-12"></a>
## IA-12 · Export that keeps provenance outside Chatbook

**Problem.**
- **L-04:** bundles drop keywords for media, conversations and notes (manifest `"tags": []`).
- **N-20:** front matter is invalid YAML for titles that contain ':' or '*'.
- **L-18:** the canvas promises "copies full media files" but writes text-only `media_N.txt` files, and a second export on the same day overwrites the first.
- **N-11:** note and prompt Export overwrite an existing file silently.
- **S-04 / j7 task 7:** the zip for the advisor carries no sources.

**Proposal.**

- **A fidelity line before Run,** computed from what the writer will actually write: "Includes: text ✓ · keywords ✓ · source links ✓ · highlights ✗ · original files ✗ (text only)".
- **Three formats:**
  - "Chatbook (.zip — re-importable into Chatbook)";
  - "Markdown folder (any editor)";
  - "One combined document (.md)".
- **What the Markdown outputs contain:**
  - valid, quoted YAML front matter with `keywords:` and `sources:` (from IA-03's edges, as titles plus the original path or URL);
  - a References section;
  - filenames based on titles.
- **Collisions ask before writing:** "Add -2 / Replace".

**User value.** The output step of the loop works for people without Chatbook, and the citations survive.

**Effort:** M · **Impact:** 3 · **Addresses:** L-04, L-18, N-11, N-20, S-04 (export part)

**Precedent.** Markdown export from Zotero and Obsidian; Readwise exports with metadata; Pandoc reference lists.

**Risks and constraints.**
- **Round-trip tests** from creator to importer are needed for each item type (`Chatbooks/chatbook_creator.py:1454`, `:1620`; `chatbook_importer.py:2772`).
- **Privacy:** absolute file paths in `sources:` leak local paths, so they are off by default.

---

<a id="ia-13"></a>
## IA-13 · One door in, routed by what you want to do with the material

**Problem.**
- **S-21:** the empty landing's only Import… goes to Media, so a Markdown folder becomes read-only media.
- **Too many Import controls.** Live `ia-loop-evidence/01-landing` shows two "Import…" in the rail and a third in Quick actions, all for Media (map-shell 7.4).
- **Notes has its own "Add from files…",** which can get stuck on a stale phase (N-17).
- **Folder files' unlinked state is 61–78% blank** (N-37).
- **Saving for later is split three ways:** Media Read later, Review sets and Collections (L-34). Quick Capture takes only URLs.
- **L-36:** nothing says what each destination is for.

**Proposal.** One **Add to Library…** entry (rail top button, landing, `i`) opens a single chooser that names *outcomes*, not mechanisms:

| Option | Goes to |
|---|---|
| Read and cite documents (PDF, Word, web page, audio or video, transcript) | Media |
| Edit as notes (Markdown or text, copied in) | Notes · Import once |
| Keep a folder and notes in sync | Notes · Keep synced |
| Edit files where they are | Folder files |
| Save a web page to read later | Collections |

- **Paste or browse first,** and the chooser recommends an option: "12 Markdown files — Edit as notes is recommended".
- **Each option states its consequence in one line,** e.g. "copies files; later changes to the originals are not tracked".
- **Notes' Add from files…, the empty Folder files state and Quick Capture become presets of this chooser.**
- **The chooser always opens at its first step,** so N-17's stale phase cannot recur.
- **Each option's one line doubles as the purpose copy L-36 asks for.**

**User value.** Material lands in the right place the first time, and there is one place to learn import.

**Effort:** M · **Impact:** 3 · **Addresses:** S-21, N-17, N-37, L-34, L-36

**Precedent.** Notion's Import menu; Obsidian's "Open folder as vault / Create new vault"; Zotero's "Add item by identifier / Attach file".

**Risks and constraints.**
- **ADR-076 forbids a second onboarding wizard.** This is one chooser screen, not a wizard.
- **ADR-113 and ADR-059:** the chooser only routes; the authorities stay separate.
- **Keep Import media's pre-check summary,** which j2 and j7 both rated as a strength.

---

## Dependencies and order

- **Wave 1 — unblock the loop (S–M).** These three remove both P0 status lies and the first-timer's dead end:
  - IA-02 (Default eligibility; ships alone as S);
  - IA-05 (health projection);
  - IA-01 (verbs and context list).
- **Wave 2 — the provenance spine, in this order:**
  1. IA-03 (source links);
  2. IA-04 (reading desk), which needs IA-03 to bind its note;
  3. IA-07 (Ask in Console), which needs IA-01's context list and IA-03 for capture provenance;
  4. IA-12 (export), which needs IA-03's `sources:`.
- **Wave 3 — consolidation:**
  - IA-06 (search; `cites:` needs IA-03);
  - IA-09 (selection; its verbs come from IA-01);
  - IA-10 (memory);
  - IA-11 (Study; needs IA-01's verb and IA-03's card links);
  - IA-13 (import door);
  - IA-08 (context bar; renders IA-01, IA-02 and IA-05).

## ADR touch-points

| ADR | Effect |
|---|---|
| 086, 084 (library-media-reader-ia) | **Amend** for IA-04: the permanent work pane gains a companion-note layout, and the collapse order becomes Items → companion stacks → reader shrinks |
| 027 (default-workspace-chats) | Its rule ("the Default workspace exists so the data model is uniform") is extended to Library for IA-02. This could be an amendment or a short new ADR |
| 027 (note session coordinator) | Respected: the IA-04 companion note and IA-03's "New note about this" reuse the one coordinator; no second save path |
| 031 | IA-01, IA-06 and IA-10 assign keys: `c` (Add to context, stay), `a` (Ask in Console), `ctrl+k` (Go to…), `n`/`q` in readers, Alt+←/→ (history). None is on the reserved or forbidden list; each needs a conflict check at implementation time |
| 079 (console-library-conversation-authority) | Respected: all staging is the manual mechanism (IA-01, IA-07) |
| 024 | Reused: the CitationTrace and `SourceLocatorEnvelope` give IA-03 its locators and IA-07 its provenance |
| 105 | IA-03 must state whether source edges join the portable-organization capability |
| 013, 067, 055, 104, 076, 113, 059, 073, 021 / 029 (file-notes-disk-authority), 097 | Constraints each idea is designed within (see each idea's Risks) |

## Findings addressed (47 of 103)

- **Shell / cross-destination (14):** S-01, S-02, S-03, S-04, S-05, S-06, S-17, S-18, S-19, S-20, S-21, S-22, S-24, S-28
- **Notes (17):** N-02, N-03, N-05, N-08, N-09, N-11, N-14, N-15, N-16, N-17, N-18, N-19, N-20, N-21, N-27, N-35, N-37
- **Library (16):** L-04, L-06, L-07, L-08, L-09, L-13, L-16, L-18, L-19, L-23, L-24, L-25, L-26, L-27, L-34, L-36

Each of these still needs its own defect fix where the master entry gives one. For example, the postcondition compare behind N-02, the YAML quoting behind N-20, and the CSS overflow behind N-05. The ideas above remove the *class* of each finding so that a new surface cannot reintroduce it.
