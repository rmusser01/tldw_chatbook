# Roleplay ← Library layout redesign: prior art and constraints

Base: worktree `roleplay-library-ux` @ origin/dev `84247cb843` (2026-10-01). Paths are relative to the repo root. Sizes today: `personas_screen.py` is 16,533 lines and `library_screen.py` is 35,902.

**Correction to the brief.** The public name is now **"Roleplay"** everywhere: nav label, `full_label`, header and palette (`UI/Navigation/shell_destinations.py:77-88`, `UI/Screens/personas_screen.py:1549`). F-034 (`5df5cbd7bd`, 2026-08-04) retired "RP&CD" and "Roleplay & Chat Dictionaries". Those names survive only in the User Guide, which has drifted (see §5). The shortcut is still `ctrl+4` (`shell_destinations.py:202`).

## 1. Binding rules

**Product and design system**
- Console is the live work surface. Other destinations "prepare, inspect, organize, configure, or hand off". Personas, Library and the rest must not be collapsed into one "agents" bucket (`PRODUCT.md:33,37`; `DESIGN.md:109,345,359`). This argues *for* a Library-style management layout and against making Roleplay a second chat surface.
- Canonical screen grammar: global nav → destination header → local mode bar → primary list → main workspace → optional inspector → footer status (`DESIGN.md:107`).
- Interaction contract: recognition over recall, consistency across destinations, no layout shift on focus or hover, and the reason for any disabled action is always shown (`DESIGN.md:121-128,226,363`). Density is valued and excess whitespace is an anti-pattern (`PRODUCT.md:31`).
- Panels and inspectors use `round` borders. `tall` borders are for destination headers only. Padding is `1 2` for headers and `1` for inspectors (`DESIGN.md:236-240`). No side-stripe accent borders on cards or rows (`DESIGN.md:361`). Scrolling inspectors reserve a "▼ more" fold row (`DESIGN.md:301-304`).
- **Layout law** (`backlog/docs/design-language.md:132-136`): the standard sidebar collapses "to zero width (never a sliver)", and "feature geometry stays local until a second consumer appears; then it promotes". The first rule *conflicts* with the 5-cell grips in ADR-084/086. The second rule is the hook for reusing the Library geometry in Roleplay.
- Code shape (`DESIGN.md:376-386,534-544`): a region widget owns pixels and a controller does not. Collaborators go in `UI/<Screen>_Modules/`, which for this screen is the existing `UI/Persona_Modules/`. `personas_screen.py` has **no size-ratchet row yet** (TASK-32809.2), and `library_screen.py` is already over budget (TASK-32463).
- Tokens and patterns: all styling uses `$ds-*` tokens (ADR-150) and catalog classes from `css/patterns.json` (ADR-161). Never hand-edit the CSS bundle. ADR-097 boot ratchets never rise, and the boot CSS is at 584,038 / 608,090 bytes with broad selectors at **274/274, i.e. full** (`Docs/superpowers/qa/2026-09-18-roleplay-recovery/README.md`).

**Decisions (ADRs)**

| ADR | Binding rule for this redesign |
|---|---|
| 004 (`:10-15,26-32`) | One destination-native workbench for Characters and Personas. Roleplay owns per-character conversation **browse/preview/resume** (ADR-120 amendment). Library stays the global conversation archive, owns admin and repair, and keeps reused CCP widget IDs (`#ccp-character-card-view`, `#ccp-character-editor-view`). |
| 007 (`:10,25,31`) | The `personas` route is the durable owner of characters, personas, dictionaries, lore, import/export and Console handoff. Dictionaries and lore must **not** move to Settings. UI migrates in "small screenshot-approved slices". |
| 011 (`:42,44,46`) | Shared primitives take state snapshots and never query the DB. **Responsiveness gate:** a replacement cannot land if it increases stalls, worker buildup or mount churn. The command palette is never the only route to a core workflow. |
| 015 (`:13,34`) | Persona is one of 13 fixed destinations. It uses the shared `DestinationHeader`. |
| 017 (`:10,35`) | The Console left rail is text-only with bordered sections. This is scoped to Console but is the house rail style. |
| 031 (`:7-11`) | Reserved globals: ctrl+q, ctrl+p, f1, f6. **Never bind ctrl+c/v/x/s/d/z/a/r/w.** Screen actions use single letters, and destructive ones need a confirmation. Footer hints match `BINDINGS` 1:1. Note: `personas_screen.py:1098` binds `ctrl+s`, and so does `library_screen.py:1047`. Both are pre-existing violations; do not spread them. |
| 034 (`:46-50`) | The disclosure glyphs `▾`/`▸` are owned by `Widgets/destination_rail.py`. Import them from there. |
| 037 (`:12-16,183-185`) | **A Persona is assistant-side.** User Profile (the human) and Persona are separate domains, and the "Persona-as-human" pointer is inert. Copy: "Assistant: General" / "Persona: _name_", never "As: _name_". |
| 042 amendment (`:33-47`) | Watchlists reader grammar: a permanent centre pane, three collapsible side panes with full-height clickable ASCII grips, **preferred vs effective** layout (responsive collapse is never persisted), and a responsive collapse priority. |
| 046 (`:15-19,106-113`) | Human display name: per-chat → global → "User". This is never a Persona. Any deep link or pane change that would unmount an editor must take the **aggregate draft veto** (Save / Discard / Stay) across forms, visuals, attachments and in-flight saves. |
| 084 (`:9-22`) | Library Media: Library rail + Items list + **permanent Reader** (never collapsed). 5-column grips. Collapse Library before Items. Hysteresis. Persisted preferred state vs transient effective state. Selected and loaded identity are kept separate and fenced by a generation. |
| 086 (`:10-23,32-37,68-81,102-104`) | **The adaptive reader shell is Library-local.** It is shared only by Media, Conversations, Notes, Prompts, Skills and Collections, and is explicitly *not* app-wide. Default rail width is `floor((3W+8)/16)+5`, clamped to 29–39. Items list defaults to 50 cells. Custom widths 24–48 are set in Settings, not by dragging grips. Prefs live in `[library.reader]` plus per-destination sections. Below 64 columns there is an emergency single stage with "‹ Library"/Esc. Geometry is pure: `Utils/adaptive_reader_state.py:217`, `Widgets/Library/library_adaptive_reader_shell.py:73,232`. **Reusing this shell in Roleplay needs a new ADR or an amendment to 086.** |
| 115 (`:10-16,24-28,91-99`) | Four heavy centre bodies (character editor, persona editor, dictionary detail, lore detail) are demand-mounted through one screen-owned first-use boundary, with generation fencing and retry. Card, rail, preview, inspector and conversations stay eager. A redesign must keep the boundary and its verification contract, including no stall above 250 ms. |
| 120 (`:19-23,33`) | Character conversations come from a local Data Profile only. "Roleplay complete browse" is a surface Roleplay owns. Personas get no Conversations view. |
| 148 (`:28-36`, **Proposed**) | MCP precedent: below 120 columns, stack the inspector *under* the canvas. Hiding the inspector behind a toggle was rejected as a discoverability failure. |
| 149 (`:10-17`) | "Chat now" on a persona creates a persona-bound Console session. The **persona name is the assistant's speaker label.** |
| 152 (`:30-41`) | Screens never bind the shell nav layer (ctrl+digit, f2–f5, f7). Mode keys are `c`/`p`/`d`/`l` (`personas_screen.py:436,445`), plus `[`/`]` to cycle. A guard test enforces this. |

Also relevant: ADR-028/074/122/139/144/146 (TTS profiles, Actor Packs, Buddy, expressions). These own sub-panels inside the character and persona editors, not the layout.

## 2. Prior Roleplay UX work

**North-star contract** (`Docs/superpowers/specs/2026-07-13-roleplay-p0-reframe-northstar-design.md:18-48`). This is the last *approved* layout contract for Roleplay. Every mode uses **List (left) → Detail (centre, tabbed, Simple⇄Advanced) → Try-it (right, "what does this DO?")**. List rows have inline enable toggles, badges, Duplicate and + New. Empty states offer starter templates. Validation is a jump-to-entry list. Cross-mode "What's in play" and lore "near-misses" were planned. **As shipped, Try-it sits below the tabs in the centre and the right pane is the Inspector** (`Docs/User_Guide/roleplay-chat-dictionaries/chat-dictionaries.md:37-41`, `lore-books.md:31-36`). The redesign has to decide whether the north-star or the Library grammar wins.

**rp-ux-review 2026-07-21** (`Docs/superpowers/qa/rp-ux-review-2026-07-21/report.md`): 2 P0, 5 P1, 7 P2 and a P3 batch, filed as TASK-425–445. **All are Done.** TASK-442 ("active persona = user") is archived because TASK-617 superseded it with the concept flipped. Fixes include the provider fallback, real Start Chat, lorebook import, file picker, selection persistence, empty state, streaming preview, mode-aware Inspector and plain-language copy.

**Library/Roleplay/MCP review 2026-08-03** (`Docs/superpowers/qa/2026-08-03-library-roleplay-mcp-ux-review.md`):
- Roleplay scored 27/40, then 29/40 after fixes.
- F-030–F-041 are all fixed (PR #1321): toolbar clipping at 100×30, the void and dead-control first paint, the three Console CTAs collapsed to **"Chat now" + "Send to Console draft"**, the five-strip header band reduced, naming (F-034) and accelerator discoverability.
- Round-2 items TASK-2231–2234 are fixed (PR #1336).
- **Open Question 3 (`:135`) is still unanswered:** "Roleplay is named for play but architected for administration (list/detail/inspector). What if the preview conversation were the centre pane and the card/editor lived in a rail?" The owner's Library direction answers it toward administration.
- That review scored Library lower than Roleplay (21→27 vs 27→29). Copying Library means copying its post-fix shell, not its 08-03 state.

**Other QA**
- `Docs/superpowers/qa/2026-09-18-roleplay-recovery/README.md` (TASK-32820): partial-save recovery dialog repaired. TASK-31243 and the broader component review are still open.
- `.impeccable/critique/2026-09-09T00-37-01Z__widgets-persona-widgets-buddy-management-modal-py.md`: Buddy modal scored 24/40 with 4 P1s (CB-1–CB-4). CB-8 asks for progressive disclosure of import and geometry fields.
- No other `.impeccable` critiques cover Roleplay.

## 3. Open backlog touching Roleplay or the Library shell

There are no drafts (`backlog/drafts` is empty).

| Task | Status | Relevance |
|---|---|---|
| 22988 | In Progress | Resume prior character chats from Roleplay. Its browse/resume UI lives in the Inspector and centre. |
| 31243 | In Progress | Trusted character navigation recovery plus full **Roleplay browse**. Directly shapes the panes. |
| 31244 | In Progress | Character conversations in the Console Context rail. Cross-surface. |
| 1532 | In Progress | Avatar thumbnails blank under non-graphics terminals. Affects card and list visuals. |
| 617 | To Do | **Persona parity with tldw_server**: persona is assistant-side, and the "persona = who you are" copy is to be removed (AC#5). |
| 19577 | To Do | User Guide drift. Asserts the *opposite*: the doc's assistant-side wording is wrong and the code's "who you play" is right. **This contradicts 617, ADR-037 and ADR-149.** |
| 496 | To Do | Residual "Personas"-as-destination copy in the Settings ownership audit. |
| 488 | To Do | Restore Dictionaries/Lore/Personas modes across navigation. Possibly stale: F-040/TASK-2091 claims the fix and `personas_screen.py:1833` restores the mode. Verify. |
| 1650 / 20975 | To Do | Deleted world book keeps its name reserved; hard-deleting a world book raises. |
| 1651 | To Do | Character editor validation messages show internal field IDs. |
| 464 / 465 | To Do | Lorebook import follow-ups (integer `position`, re-import, double-inject). |
| 31247 / 31248 | To Do | Character chat search controls and Roleplay "Meaning" search. |
| 33001.12 | To Do | Six roleplay-resume navigation tests are red. |
| 33281 | To Do | **PERF-22: a Personas visit costs 0.7–1.0 s; lists remount every row on each search or page.** |
| 32809.2 / .3 | In Progress | No ratchet row for `personas_screen.py`; decomposition maps recorded. |
| 26983 / 27000 | To Do | Ruff formatter batches for the Personas screen and Persona UI. Expect churn. |
| 21236 | To Do | Persona Buddy lazy-controller residue. |
| 33622.14 | Pending (PR #2952) | Ctrl+Q quits past unsaved Roleplay drafts. |
| Library shell: 32235, 32309, 32576, 32463, 281, 31249, 2520 | Mixed | Glyph legend (`○`/`▸` overloaded), `apply_pane_width` is Notes-only, size ratchet over budget, targeted updates instead of recompose, red UI tests, landing footer key story. Do not copy these defects along with the shell. |

## 4. In-flight collisions

`gh pr list` returned 36 open PRs. Four PR file lists were truncated by the CLI and were re-read through the API.

| PR | State | Overlap |
|---|---|---|
| **#2862** `codex/personal-context-memory-dev` | Ready, **CONFLICTING**, updated 2026-10-01 | **Highest risk.** `personas_screen.py` +3/−117, `Persona_Modules/personas_preview_coordinator.py` +121/−2, `library_screen.py` +86/−183, `Library_Modules/screen_helpers.py` +113, `css/screen_agentic_library.tcss` +1. |
| #2885 `feat/persona-buddy-workbench` | Draft, 2026-09-28 | New `Widgets/Persona_Widgets/persona_buddy.py`, `personas_persona_visual_pack_widget.py` +9 ("Buddy workbench…" action), `UI/Navigation/persona_buddy_host.py`. Unwired. |
| #2563 `codex/native-goal-runs` | Draft, 2026-09-27 | `persona_buddy.py` +491, `personas_persona_visual_pack_widget.py` +33/−4, and `personas_screen.py`/`library_screen.py` at ±0 (likely mode-only). |
| #2952 | Ready | Backlog only (adds task-33622.14). |
| #2904/#2911/#2913/#2914 | Ready | Backlog files only (33264, 33281). No code. |
| #2868 | Ready | `Tests/Character_Chat/test_character_backup_export_image.py` only. |

**Churn.** Since 2026-08-15 there are 117 commits (107 excluding merges) on `personas_screen.py`, `Widgets/Persona_Widgets/` and `UI/Persona_Modules/`. The latest are:
- `1873a7386a` 09-30: preview uses the one model-selection builder (TASK-33004.2).
- `c40f3a7872`, `b1389edd8c` 09-25/26: reload a card saved from Console (TASK-32954).
- `c59c2a0250` 09-12: c/p/d/l mode keys.
- ADR-161 token-class migration, 09-14.
- Buddy work, 09-07 to 09-10.
- `12836ce11c` 09-05: trusted navigation.
- `2516735cfd` 09-03: demand-mount (ADR-115).

This is a hot file. Land the redesign in small slices, consistent with ADR-007.

## 5. User Guide promises (must be rewritten)

`Docs/User_Guide/roleplay-chat-dictionaries.md`, with child pages `characters-and-personas.md`, `chat-dictionaries.md` and `lore-books.md`, currently promises:
- **Layout** (`:59-75`): three panes (Library rail with New/Import/Duplicate/Sort/Tag, a centre detail/editor with "Try a test chat (nothing saved)", and an Inspector). The `<`/`>` header buttons collapse to the "Library"/"Inspector" handles. **Both rails start open.** F6/Shift+F6 cycle handle → Library → centre → Inspector. In code the handles are 13 and 11 cells wide (`personas_screen.py:608-609`) and the compact split is at ≤90 columns (`:607`).
- **Rail collapse lasts only for the current visit** (`:246-248`). The Library grammar persists preferred state, so this behaviour change must be documented.
- An Inspector action/visibility matrix (`:118-124`) and Readiness strings (`:131-143`).
- Dictionaries: five tabs plus Try-it (`chat-dictionaries.md:37-41`). Lore: three tabs plus Try-it (`lore-books.md:31-36`).
- Keys (`:194-214`): c/p/d/l, `[`/`]`, Ctrl+N/F/Enter/S, Esc ("never leaves the screen"), Space toggles a dictionary, F6.

Drift already present, to fix as part of the redesign:
- The title "Roleplay & Chat Dictionaries" (`:1`, `:39`).
- An impossible palette string (`:23`; `characters-and-personas.md:15`).
- The retired "Status row" (`:47-48`, removed by F-033).
- "Attach to Console/Start Chat" (`:118`, `:151`, `:187`). The code says "Chat now"/"Send to Console draft" (`Widgets/Persona_Widgets/personas_inspector_pane.py:287,294`).
- Ctrl+1–4 for modes (`:175-178`; `characters-and-personas.md:308`), which contradicts the page's own `:196-200`.
- The persona tooltip (`:55`) does not match the code (`personas_screen.py:453`).

**Doc convention.** Do **not** add "Verified against" paragraphs. Record verification in task Implementation Notes (`CLAUDE.md:143-147`). The existing stamps are legacy (`:256`, `chat-dictionaries.md:252`, `lore-books.md:228`). The Library guide's layout section (`Docs/User_Guide/library.md:168-215`) is the model for the new text.

## 6. Ecosystem conventions (GENERAL KNOWLEDGE, not verified against current releases)

- **SillyTavern.** Chat is always the centre. Management lives in top-bar drawers:
  - The right-side **Character Management** drawer has a list or grid with search, sort, tags, tag folders, favourites and bulk edit.
  - Selecting a character opens its card editor: description, first message, an "Advanced definitions" popup, alternate greetings and creator notes.
  - **World Info / Lorebooks** has its own drawer with a world selector, global activation settings (scan depth, budget, recursion) and an inline list of collapsible entry rows (keys, content, strategy, position, depth, order, probability). Lorebooks bind globally, per character (primary plus extra), per chat and per persona.
  - **Persona Management** is a grid of *user* avatars (name plus description). One is the default, and personas can be locked per chat or character.
  - The **Regex** extension is the "dictionary" analogue: find/replace scripts, global or character-scoped, targeting user input, AI output or world info, with optional "alter display only".
- **Agnai(stic).** A left nav lists Characters, Chats, **Memory Books** (lorebooks), Presets, Settings and Profile. The character editor is a full page with an advanced section. Your Profile, or an "impersonated" character, is your persona.
- **RisuAI.** A vertical strip of character avatars on the left. The character settings panel has tabs (basic, display, **lorebook**, **regex scripts**, advanced). Lorebooks and regex can be global, per character or per chat, and "Modules" bundle lore with regex and are toggled per chat. Personas are *user* identities.
- **NovelAI / Kobold Lite.** NovelAI's Lorebook is the classic two-pane layout: an entry/category list on the left and an entry editor on the right with Entry/Placement/Advanced tabs. Kobold groups Memory, Author's Note and World Info in a context panel.
- **Implications.**
  1. Across the ecosystem, "persona" means *who you play*. That clashes with ADR-037/149 and TASK-617's assistant-side Persona. The owner has to rule on this before the redesign fixes the meaning into layout or copy.
  2. Lore editing is universally list-of-entries → entry editor, which maps well onto Library's rail → items → work pane.
  3. Users expect attachment scope at global, character and chat level. Chatbook has character and conversation scope plus "What's in play".
  4. "Chat dictionaries" is a tldw-specific term. ST and RisuAI users know these as "regex scripts", so they need a gloss.
  5. These apps keep authoring peripheral to chat. Chatbook deliberately separates it (Console vs Roleplay), which supports a Library-style management screen.

## 7. Decisions the redesign must surface

1. **Shell reuse.** Either amend ADR-086 or write a new ADR before the Library adaptive shell is used outside Library (ADR-084/042 forbid an app-wide split-pane without one).
2. **Inspector.** Library readers have no right inspector. Roleplay's inspector carries the CTAs, validation, readiness and ADR-120 conversations. Choose one: fold it into the work pane, stack it below at narrow widths (ADR-148 precedent), or keep it as a fourth pane.
3. **Modes vs rail.** Moving Characters/Personas/Dictionaries/Lore from the chip strip into a Library-style rail must keep c/p/d/l and `[`/`]` (ADR-152) and truthful footer hints (ADR-031 rule 4).
4. **North-star Try-it vs Library grammar.** Decide where Try-it and the test chat live.
5. **Persistence.** Library persists preferred pane state (`[library.reader]`). Roleplay is per-visit today. A new config section needs the preferred/effective split.
6. **Preserve** the ADR-115 first-use boundary, the ADR-046 aggregate draft veto, and the reused CCP widget IDs (ADR-004).
7. **Performance gate.** ADR-011 responsiveness, PERF-22 (TASK-33281) and the full broad-selector budget (274/274) all apply.
8. **Persona semantics.** The owner must rule between TASK-617 (assistant-side) and TASK-19577 (user-side) first.
