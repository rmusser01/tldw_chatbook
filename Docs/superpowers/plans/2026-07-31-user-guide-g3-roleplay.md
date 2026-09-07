# User Guide G3 (Roleplay & Chat Dictionaries) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the Roleplay & Chat Dictionaries section of the User Guide —
`roleplay-chat-dictionaries.md` plus three child pages plus captures — per the
approved spec `Docs/superpowers/specs/2026-07-25-user-guide-by-screen-design.md`,
from the completed IA survey of dev @ `207053253` (2026-07-31).

**Architecture:** Pure-docs change on branch `claude/user-guide-g3-roleplay`
(worktree `/private/tmp/tldw-guide-g1`), one PR at the user merge gate.
Authoring inputs (session scratchpad): `g3_survey.md`, `g3_inventory_shell.md`,
`g3_inventory_characters_lore.md`, `g3_inventory_dictionaries.md`. Drafting by
subagents; live verification, captures, stamps, and commits stay with the
controller. G1/G2 process carries over, including the SVG cdnjs font-strip step
in `_template.md`.

## Survey decisions

1. **Parent + three children** (matches the spec's provisional tree):
   - `roleplay-chat-dictionaries.md`
   - `roleplay-chat-dictionaries/characters-and-personas.md`
   - `roleplay-chat-dictionaries/lore-books.md`
   - `roleplay-chat-dictionaries/chat-dictionaries.md`
   **Personas is folded into the characters page** (thin surface: read-only
   card + a six-field editor, no Import/Duplicate/PNG export) and shares the
   card → editor → Console-handoff grammar; the page is titled for both.
2. **Naming stays `roleplay-chat-dictionaries`** — locked in G0 from the
   screen banner; the nav bar's short label is "Roleplay".
3. Screen file is ~9k lines: **grep, never read whole**. The pure-state
   files (`Widgets/Persona_Widgets/personas_state.py`, `personas_library_pane.py`,
   `personas_inspector_pane.py`) are small and authoritative.

## HEADLINE CORRECTION this phase must ship (cross-page seam)

`Docs/User_Guide/index.md:49` and `:61` claim **Ctrl+1 … Ctrl+0** switch
screens "from anywhere". On this screen that is false for the first four:
`PersonasScreen` binds **Ctrl+1–4 to the four modes**, shadowing the
destination hotkeys; **Ctrl+5–Ctrl+0 still navigate.** Verified twice
headlessly (Ctrl+1/Ctrl+2 kept us on PersonasScreen and changed mode;
Ctrl+6 left for Watchlists). Fix the index globals row AND state it on the
parent page. This is the same class of error the G1 review caught in the
nav table — do not ship without it.

## Global constraints

Template order and authoring rules as G1/G2; verbatim labels; honest
limitations with backlog refs; captures at 200×50 with fonts stripped;
`PYTHONPATH=/private/tmp/tldw-guide-g1` + codebase guard on every by-path
script; collision sweep before filing new tasks (dev reached 1625, worktrees
1631; this session already filed 1640–1642); commit trailer.

## Task list

### Tasks 1-4: page drafts (subagents, one file each)
1. **Parent** — what the screen is; the four modes (chips + descriptor lines
   verbatim) and how to switch (Ctrl+1–4, chips, and the `[`/`]` caveat that
   text fields swallow them); three-pane tour (Library rail / detail /
   Inspector, collapse `<` `>` + handles); the Library rail's per-mode control
   visibility; the Inspector (rows, readiness copy, and the per-kind action
   matrix: character = all five, persona = no Export PNG, dictionary/lore =
   Delete only); Console handoffs overview (Attach to Console vs Start Chat,
   including the per-intent gate); the unsaved-changes dialog; getting there
   (Ctrl+5, nav "Roleplay", palette entries, legacy routes ccp/characters/
   conversations_characters_prompts/roleplay); the Ctrl+1–4 shadowing note.
2. **characters-and-personas.md** — character card fields; the editor
   (generation toolbar + per-field Generate + Accept/Regenerate/Discard,
   Advanced section, alternate greetings list editor, avatar upload/generate
   with the 5 MB cap and type copy, expressions block); validation and save
   copy; create/edit/duplicate (no confirm)/delete (confirm); import formats
   (PNG `chara`/`ccv3`, WebP EXIF, JSON/YAML/MD) and the deliberately generic
   failure copy; export JSON/PNG; Console handoffs with their exact prompts;
   the embedded "Dictionaries (copied into this character)" and "World Books
   (copied into this character)" snapshot panels (link the sibling pages);
   then a **Personas** section (card, six-field editor, what differs: no
   Import/Duplicate/PNG); and a short "what {{char}}/{{user}}/{{persona}}
   mean" explainer — NOTE the app has no UI surface explaining them, so the
   guide is the only place a user learns it.
3. **lore-books.md** — what world books do; list/rail; Entries tab (keys,
   content, position with its four labels, priority 0–100, enabled,
   case-sensitive, selective + secondary keys, regex + its 500-char/complexity
   limits); Settings tab (scan depth 3, token budget 500, recursive, enabled);
   Attachments tab (attach to conversation); the "Try it — injection preview"
   pane incl. the permanently-disabled "Include recent turns (soon)" switch;
   import (10 MB cap, SillyTavern/character-book shapes accepted) and export;
   character-embedded world books are SNAPSHOTS; link
   ../../Features/World-Lore-Books-Documented.md as the concepts deep dive
   while flagging that its UI walkthrough describes the retired CCP tab.
4. **chat-dictionaries.md** — from the dictionaries inventory: entry model,
   five tabs, Try-It preview + near-miss reasons, validation findings,
   versions/revert, attachments, import/export (lossy-markdown confirm, 10 MB
   cap), enable/disable incl. Space on a row, Ctrl+S does NOT save, the
   character-embedded snapshot seam, and where dictionaries take effect in
   Console (link `../console/context-and-rag.md`, which currently forward-refs
   this page). Flag the stale UI section of
   ../../Features/ChatDictionaries-Documented.md and the `[group]key|33`
   markdown syntax its importer does not parse.

### Task 5: captures + live verification + stamps (controller)
Scenes: parent overview (populated), character card, character editor
(Advanced open), lore Entries + Try-it, dictionary Entries + Try-it. Seed demo
content first (the profile has three stray "Untitled world book" items to
clean up). Execute each page's Common tasks live or by pilot; file backlog
tasks for confirmed new defects; stamp all four pages `207053253 — 2026-07-31`.

### Task 6: index + console wiring, link sweep
`index.md`: Ctrl+5 row → link; "Conversations / CCP" legacy row → link; the
**Ctrl+digit globals exception**. `console/context-and-rag.md:162-163`:
replace "coming in a later guide phase" with the real link. Full sweep
`BROKEN: []`; template conformance per page (all eight headings in order).

### Task 7: whole-branch review + PR (user gate)
Same reviewer brief as G1/G2 **plus an explicit "report and STOP — do not
edit, commit, push, or open PRs" instruction** (the G2 reviewer overstepped).
Merge-time drift check: `git log 207053253..origin/dev --
tldw_chatbook/UI/Screens/personas_screen.py tldw_chatbook/Widgets/Persona_Widgets/
tldw_chatbook/Character_Chat/`. Push, open PR against dev.
**Do NOT merge — user gate.**
