---
id: TASK-32129
title: >-
  Library Notes Import once: Obsidian mode (skip config and trash, map
  frontmatter, rewrite wikilinks) — user decision 2026-09-09
status: Done
assignee:
  - '@claude'
created_date: '2026-09-08 21:39'
updated_date: '2026-09-09 06:30'
labels:
  - library
  - notes
  - ux
  - critique-notes-2026-09
  - import
  - obsidian
  - feature
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The user chose 'Adopt (Obsidian mode)' over a format-agnostic copy. Today (PROVEN): discovery walks `.obsidian/` and `.trash/` (no ignore rule in note_import_discovery.py); `.json` is a supported type so five config files land in 'Failed (5) · could not be imported safely'; `_parse_text` never parses YAML frontmatter (title is the first '# ' heading or the file stem) so `tags:` produce zero keywords and the block stays in the body; `Templates/Daily.md` becomes a note titled `{{date:YYYY-MM-DD}}`; `[[wikilinks]]` and `[[link|alias]]` stay literal even though every target was imported in the same batch. Import once remains byte-safe on disk (verified by hash). Evidence: Library ▸ Notes critique snapshot `.impeccable/critique/2026-09-09T04-24-50Z__w-chatbook-widgets-library-library-notes-canvas-py.md` (dev c4a7b1911f, 2026-09-08/09, dual-agent live review at 235x52/100x30/60x24 on a fresh and a seeded profile with a 71-file Obsidian-style vault; parent re-tested every disagreement on a third profile).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When `.obsidian/` exists at the selected root, the review shows a default-on 'Obsidian vault' toggle and states what it will do
- [x] #2 With the toggle on, `.obsidian/`, `.trash/` and `Templates/` are listed as skipped with a plain reason, never as Failed
- [x] #3 YAML frontmatter is parsed: title to the note title, tags to keywords, aliases kept in Info, and the block is removed from the body
- [x] #4 `[[wikilink]]` and `[[link|alias]]` targets created in the same batch become note links; unresolved links stay as text
- [x] #5 Template placeholders such as `{{date}}` never become a note title
- [x] #6 The review shows the resulting titles and keywords before import; the vault on disk stays byte-identical
- [x] #7 Covered by tests with a fixture vault carrying each of the above
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Discovery: detect a vault (top-level .obsidian dir) during the walk; with obsidian_mode skip .obsidian/.trash/Templates at the vault root as SKIP entries with reasons (never Failed).
2. Models: add ImportClassification.SKIPPED (Skip-only contract) and ParsedNotePayload.wikilinks.
3. Parsers: obsidian_mode parses leading YAML frontmatter (title/tags/aliases -> title/keywords), strips it from the body, records [[wikilinks]]; a {{...}} placeholder never becomes a title (unconditional).
4. Planner: skipped discovery entries become SKIPPED preview items carrying their own reason.
5. Executor: build the deterministic note-id map for CREATE_NEW items once per execute() and rewrite resolvable [[links]] to [text](note://id) as each payload is written; unresolved links stay literal.
6. UI: obsidian_mode/vault_detected on the import state, a default-on 'Obsidian vault' toggle in the review (re-runs the read-only check), review rows show the resulting title, keyword and link counts.
7. Tests (new Tests/Notes/test_note_import_obsidian.py + executor/UI extensions), docs, live verify on the fresh profile, backlog.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Import once now reads an Obsidian vault as one, without touching it.

**Discovery** (`note_import_discovery.py`): the walk records `vault_detected` when the selected root holds an `.obsidian/` directory, and with `obsidian_mode=True` returns `.obsidian/`, `.trash/` and `Templates/` as one `skips` entry each (reason codes `obsidian_config`/`obsidian_trash`/`obsidian_template`) instead of descending into them — so a whole config tree is three review rows, not five failures. `obsidian_mode` defaults to False in the library functions, so the other caller of discovery (lasting-sync change detection) is untouched; only the import controller passes True.

**Parser**: a leading `---` YAML block is parsed with the existing `_UniqueKeySafeLoader` (same anchor/alias guard as .yaml sources) into title + keywords and removed from the body; anything that is not a clean mapping is left in the body exactly as written. `tags` and `aliases` both become keywords — there is no per-note metadata store, and keywords are what makes a note findable by its alternate names, which is what an alias is for. A `{{…}}` placeholder is never accepted as a title (frontmatter or heading), unconditionally, since no caller wants `{{date:YYYY-MM-DD}}` as a note name. Recorded `[[targets]]` land on the new `ParsedNotePayload.wikilinks` field.

**Link form and where the rewrite happens**: `note://<note_id>` — the URI scheme the app already uses for a note (`MCP/resources.py`, `local_control_service.py`, `gateway_runtime.py`); `agent_lessons.py`'s `note:` prefix is a lesson id and `note_folder_models.py`'s `note:<folder>:<note>` is a placement cursor, neither is a link. Rendered as a Markdown link `[text](note://id)`, which the note Preview shows as a link. The rewrite happens as each note is written, not in a second pass: note ids are already deterministic (`_deterministic_note_id(approval_id, item_id, payload_index)`), so `_wikilink_note_ids(approved)` builds the whole batch's map up front and `rewrite_wikilinks` transforms the payload once, inside `_execute_item`. No extra writes, no second failure mode, and the receipt's stored fingerprint stays the fingerprint of the parsed file, so re-importing the same vault still detects an unchanged repeat. `payload.wikilinks` is the executor's only licence to rewrite, so a plain import containing `[[…]]` text keeps it literal (pinned by its own test).

**Review**: `ImportClassification.SKIPPED` (Skip-only, like UNSUPPORTED/FAILED) gives the canvas a "Skipped (N)" group, and skip items keep their own reason instead of the generic failure copy. `_effect_summary` now names what will be created — `Content: create 1 new note: Library ▸ Notes review · 4 keywords · 3 links.` A default-on `✓ Obsidian vault` toggle with a one-line explanation is mounted in `#notes-import-review-options`; toggling re-runs the read-only check.

**Not done, deliberately**: no `N links resolved` count in the receipt — that number would have to travel through the durable receipt ledger (schema + validated projection) for one line of copy, and no acceptance criterion asks for it; the review row already shows the per-note link count. The Windows discovery adapter never reports a vault, so on Windows the feature is simply absent rather than half-applied.

**Files**: `Notes/note_import_{discovery,parsers,plan_models,planner,executor}.py`, `Library/library_note_import_state.py`, `UI/Library_Modules/library_note_import_controller.py`, `UI/Library_Modules/library_notes_controller.py` + `UI/Screens/library_screen.py` (handler + delegator), `Widgets/Library/library_note_import_canvas.py` (+ rebuilt `css/widget_defaults_self.tcss`), `Docs/User_Guide/library/notes.md`, new `Tests/Notes/test_note_import_obsidian.py`, extensions to `Tests/Notes/test_note_import_executor.py` / `Tests/UI/test_library_note_import_flow.py` / `Tests/UI/Library_Modules/test_library_note_import_controller.py`, and two re-pinned rows in `Tests/Architecture/test_library_modules_size_ratchet.py`.

**Fix round 1** (review): a frontmatter-only note (an Obsidian Properties-only file) kept its original text as the body instead of becoming a Failed row -- the mode must never turn a file it understands better into a failure. The link scanner is now one regex that matches a code span OR a link, so a `[[Target]]` inside a fence or backticks is neither recorded nor rewritten (it was being rewritten into the note body, corrupting sample code); the parser and the executor share that one definition. The review row now names the keywords instead of counting them. Also: a single-segment display path can no longer abort the link map, the module docstring and the Skip-only contract message mention SKIPPED, and the guide states that the placeholder-title rule applies with the toggle off too and that toggling rebuilds the review.

**Fix round 2** (Qodo on PR #2539): link discovery now reads the parsed body rather than the text kept as content, so a frontmatter-only note's properties (`source: "[[README]]"`) are metadata again and are never rewritten -- every other shape is unaffected because there the body *is* the content. `rewrite_wikilinks` escapes backslashes in the label: `[` and `]` are already outside the target/alias grammar, so the only way a label could close the link text early was a trailing backslash escaping the `]`, which let the link swallow the following text and the next link's destination. Rejected in the same round, with the evidence in the PR replies: a Pydantic schema for three optional frontmatter fields (each manual check is total and has a documented fallback), rewriting a title that merely *contains* `{{…}}` (AC #5 and the guide make the placeholder rule unconditional; a half-filled `Meeting {{date}}` is a worse note name than the file stem), and guaranteeing a target is durably created before its link is written (that needs either target-before-source ordering, which cyclic links make impossible, or a second write pass over committed notes -- and a dangling `note://` is inert unclickable text whose uuid5 id can never resolve to a different note).

**Live-verified** on the fresh profile at 120x40 (captures in scratchpad `notes-crit/wave/obsidian/caps/`): 66 review items with the toggle on, "Skipped (2)" (.obsidian, .trash) and "Skipped (1)" (Templates) with their reasons and no Failed rows for them, import completed 59 imported / 8 skipped, the note opens as `Library ▸ Notes review` with keywords `lib-review, notes-review, project, ux`, a body starting at its heading, and `[yesterday's meeting](note://a381ad8c-…)`; the vault's 71-file content hash is `840ddf4d…` before and after.
<!-- SECTION:NOTES:END -->
