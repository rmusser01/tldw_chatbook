# User Guide G2 (Library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the Library section of the User Guide — `library.md` plus
eight child pages plus SVG captures — per the approved spec
`Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`, from the
completed live IA survey of dev @ `bd05a692a` (2026-07-31).

**Architecture:** Pure-docs change. Branch `claude/user-guide-g2-library`
(worktree `/private/tmp/tldw-guide-g1`), one PR at the user merge gate.
Authoring inputs: session-scratchpad artifacts `g2_survey.md`,
`g2_inventory_shell.md`, `g2_inventory_panels.md`,
`g2_inventory_search_study.md` (code-exact labels + file:line + live-walk
record). Drafting by subagents; live verification, captures, stamps, and
commits stay in the controller session (single stamp story). G1's process
carries over verbatim, including the SVG font-strip step now in
`_template.md`.

## Survey deltas vs the spec (survey wins)

1. **Eight children, not five.** The spec's provisional tree
   (notes/media/skills/prompts/collections) missed Conversations, the
   Search/RAG canvas, the File Notes workspace (+ Session Git), and the
   Import/Export canvases. Final tree: `library.md` +
   `library/media-and-conversations.md`, `library/notes.md`,
   `library/file-notes.md`, `library/prompts.md`, `library/skills.md`,
   `library/collections.md`, `library/search-and-rag.md`,
   `library/import-and-export.md`.
2. **Study is a hand-off, not a Library surface.** StudyScreen is its own
   screen (palette "Study"); Library's Study decks / Flashcards / Quizzes
   rows are thin hand-off canvases → folded into the parent page as one
   section with a "Study is its own screen" pointer. The Study screen
   itself is OUT of G2 scope.
3. **Standalone Search screen ≠ Library Search/RAG canvas.** Palette-only;
   owns the index Maintenance UI. G2 documents the Library canvas and adds
   one disambiguation note pointing at it.
4. Prompts and skills stay SEPARATE pages (spec agreed; the skills trust
   panel dominates); import + export share ONE page (one rail section, one
   chatbook format).

## Global constraints

Same as G1 (template order, verbatim labels, honest limitations with
backlog refs — 197, 673, 291, 414-424, 449 available; collision sweep
before filing NEW tasks; commit trailer; absolute paths / `git -C`;
`PYTHONPATH=/private/tmp/tldw-guide-g1` + codebase guard for every by-path
script; strip cdnjs fonts from every new SVG; delete scratch data at
program end). Config-key hygiene: the embeddings table is
`[embedding_config]` — never cite `[embeddings]`.

## Task list

### Tasks 1-9: page drafts (subagents, one file each)
1. `library.md` (parent) — from shell inventory + survey: header line
   ("Library | Local" / server variant), rail tour (all four sections,
   every row verbatim incl. counts and "in Library" secondaries), landing
   copy, getting there (Ctrl+3, palette, legacy routes), Details block
   (Status/Workspace/Actions; "Use in Console" snapshot handoff;
   "Server sync WIP · local only"), Study/Flashcards/Quizzes hand-off
   section (verbatim five-element canvas + "Continue in Study"), child
   directory, `u` key note (Search/RAG-only).
2. `library/media-and-conversations.md` — media list (type filter,
   select/export grammar), media viewer (Content search, Analysis,
   Highlights, action row incl. "Open in Media manager" jump), the
   conversations canvas (filter, select/export, "Open in Console"
   staging semantics vs Console's own rail, NO delete).
3. `library/notes.md` — list/sort/Sync/Import note/Export…, editor
   (autosave, conflict banner, action row), create (Blank + templates),
   Notes sync panel (+ `[notes]` keys, deep-dive link
   ../Features/notes_bidirectional_sync.md), Database↔Files strip pointer.
4. `library/file-notes.md` — workspace (folder link/Change…, navigator
   trees + content search, editor pane statuses, both toolbars,
   `[file_notes] root`), Session Git panel (trust dialog verbatim,
   row states, stage/commit flow), folder-details dialog.
5. `library/prompts.md` — list + inline import row (dedupe = skip), editor
   (fields, save-status copies, conflict banner), Use in Console = INSERT
   into composer draft, Duplicate/Export .md, Delete-no-confirm quirk,
   task-197 gap.
6. `library/skills.md` — list + trust header states/actions, import
   (file/folder/URL → trust-pending), editor (name rules, invoke/Runs-in
   toggles, disabled Model override note, warnings), Trust panel (states,
   review preview caps, Approve + passphrase, script grant/revoke,
   reset), dirty-veto.
7. `library/collections.md` — short honest page: CRUD + two-press delete,
   sync-status labels, the explicit deferred matrix quoted.
8. `library/search-and-rag.md` — canvas walk (mode toggle, scope
   toggles, quiet gates, Evidence rows, Recent searches), retrieval
   inspector blocks, `u`/"Use in Console" → "Review evidence in Console"
   staging, recovery/embeddings-missing copy, config pointers
   (`[library.search]`, `[rag]`, `[rag_search]`, `[embedding_config]`),
   standalone-Search-screen disambiguation note.
9. `library/import-and-export.md` — ingest end-to-end (path/URL,
   pre-flight, guardrail modal, per-type option groups incl. transcription
   + Parakeet modal, metadata, Start ingest, Queue + row actions,
   `library.ingest*` keys, task-673 note if user-visible), Export chatbook
   canvas + scoped entry points + server-mode disable.

### Task 10: captures + live verification + stamps (controller)
- One SVG per page (9): library overview (populated rail), media viewer or
  list, notes editor, file-notes workspace, prompts editor, skills list or
  editor, collections panel, search-rag with evidence, ingest canvas.
  Capture profile must be populated by pilot-driving (create note, ingest
  demo file) before shots; strip cdnjs fonts (template step 5).
- Execute every page's Common tasks live (tmux or pilot battery à la
  verify_tasks_g1.py); file backlog tasks for confirmed new quirks
  (collision sweep first); stamp all pages `bd05a692a — 2026-07-31`
  (or re-stamp if drift forces re-verification).

### Task 11: index wiring + link sweep
Library rows go live in index.md (nav map, Quick Start if touched, legacy
rows Notes/Prompts/Skills/Research → deep links where a page now exists);
sharpen the Library one-liner if it implies Study lives here; full link
sweep `BROKEN: []`; template conformance check per page (all 8 headings in
order — G1's critical finding, don't repeat it).

### Task 12: whole-branch review + PR (user gate)
Same reviewer brief as G1 (seams, template, honesty, ~15 label
spot-checks, captures, keyboard tables). Fix wave. Merge-time drift check:
`git log bd05a692a..origin/dev -- tldw_chatbook/UI/Screens/library_screen.py
tldw_chatbook/Widgets/Library/ tldw_chatbook/Library/` (+ study/search
files if cited); re-verify + re-stamp if moved. Push, open PR against dev.
**Do NOT merge — user gate.**
