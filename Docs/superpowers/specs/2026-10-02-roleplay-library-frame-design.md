# Roleplay on the Library frame (sub-project B): design

- **Status:** Approved design, 2026-10-02. The owner approved all five design sections as presented in the brainstorming session of 2026-10-01/02 and ruled on the twelve open questions (§8). Next step: one implementation plan per slice (§5.13).
- **Owner:** the project owner (@rmusser01).
- **Programme:** the Roleplay layout review of 2026-10-01. Sub-project **A** (fix-first) is filed; this document is sub-project **B**; sub-projects **C** (world-info semantics) and **D** (persona semantics) are separate (§0.5).
- **Evidence** (lands with PR #2957, branch `backlog/roleplay-fix-first-tasks`), all under `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/`:
  - the review report, [`report.md`](../qa/roleplay-library-layout-review-2026-10-01/report.md): findings RP-001..RP-097 and default decisions D1-D15. This spec cites its sections as "report §N" and its Appendix C lens sections as "C.N";
  - the findings, [`findings.json`](../qa/roleplay-library-layout-review-2026-10-01/findings.json);
  - the code maps and prior art, [`map-roleplay.md`](../qa/roleplay-library-layout-review-2026-10-01/map-roleplay.md), [`map-library.md`](../qa/roleplay-library-layout-review-2026-10-01/map-library.md), [`prior-art.md`](../qa/roleplay-library-layout-review-2026-10-01/prior-art.md), and the `evidence/` captures.
- **Sub-project A tasks** (filed in PR #2957): TASK-33781..TASK-33791 are Roleplay's fix-first tasks (RP-001, 002, 003, 005, 009, 014, 022, 073, 075, 078, 079); TASK-33792 (RP-084) and TASK-33793 (RP-097) are Console's. §5.5 says which B slice needs which.
- **Review:** the design was adversarially reviewed through four lenses (consistency, governance, feasibility, user advocate). The rulings table (§0.6) names the lens that raised each ruling, and Appendix B maps every review issue to its resolution.
- **Base:** `origin/dev @ 84247cb843`, worktree `.worktrees/roleplay-library-ux`. `path:line` is relative to the repo root.
- **Path shorthand:** `PS` = `tldw_chatbook/UI/Screens/personas_screen.py`; `LS` = `tldw_chatbook/UI/Screens/library_screen.py`; `PW/` = `tldw_chatbook/Widgets/Persona_Widgets/`; `PM/` = `tldw_chatbook/UI/Persona_Modules/`; `AT` = `tldw_chatbook/css/components/_agentic_terminal.tcss`; `ARS` = `tldw_chatbook/Utils/adaptive_reader_state.py`.
- **Verified at design time:**
  - Every width in §2 was run through the real resolver (`resolve_adaptive_reader_layout`, `ARS:217`) with the Roleplay profiles.
  - Every mockup line is exactly its stated width.
  - Code facts re-checked: `MODE_CHIP_ORDER`/`MODE_HOTKEYS` zip (`PS:436, 445, 1111`); the guard's "Stay" button (`UI/Navigation/character_conversation_navigation.py:236, 270, 309`); `library_emergency_return` is imported only by `LS:488`; the Library notes work-session reducer (`UI/Library_Modules/library_notes_work_session.py:48-70`); `ITEMS_TARGET_WIDTH = 50` already on dev; today's Ctrl+S presses the visible Save of the character or persona editor only (`PS:1098, 16295-16312`), and the other Roleplay commit buttons are `#personas-visual-identity-save` ("Save", disabled until a change), `#personas-persona-visual-save` ("Save Pack"), `#personas-policy-save`, `#personas-{lore,dict}-settings-save` and `#personas-{lore,dict}-entry-{add,update}`.

---

## Summary for the owner

**What changes.** Roleplay (Ctrl+4) stays its own destination but is rebuilt on the Library's frame.

- **A one-row header** replaces five rows: `Roleplay   Characters`, the data source on the right, and words (not a badge) for unsaved work or a blocked provider. The false "Ready" badge goes.
- **A Roleplay rail** on the left lists the four kinds as equals (Characters, Personas, Lore books, Chat dictionaries), each with a live count and a one-line gloss, plus Start, New…, Import…, a "You" row and Buddy.
- **A full-height items list** in the middle. At 120 columns every item takes one line, so all 28 sample characters are visible instead of 5. Rows load from the database in windows, so 5,000 items stay fast. Marks survive search, and the toolbar is one row.
- **A permanent work pane** on the right. The Inspector column is removed and folds in: a title row with the item, its state, the primary action (Chat now or Run preview) and a readiness word; a view strip (for a character: Card · Edit · Chats · World · Test chat · Info); a "More actions" region; and an anchored Save bar in every editor.
- **Work sessions.** `e` (edit) or `t` (test) starts a session that closes the side panes so the editor or test chat gets the width (110 columns at 120x36). Panes move only when a session starts or ends, never on a view switch or a Save.
- **Test chat and Try it become full-height views.** At 200+ columns they can also sit beside the editor (the "Try column"): on by default for characters, lore books and chat dictionaries, off for personas.
- **Lore and dictionary entries** get a full-height list with a filter and a position count, side by side with the entry editor once the work pane has 100 columns.
- **A landing** appears when there is nothing to restore: Import…, four New buttons, "Continue a character chat" (the 5 most recent), Recently edited and Needs attention.
- **Keyboard.** Arrival lands on the list; Escape never leaves the screen; single letters act only on navigation surfaces; the footer and F1 follow focus; `o` is Chat now on the loaded row; Ctrl+S saves in every Roleplay editor.

**Expected gains** (resolver-verified; §0.1, §2.10):

| Measure | Today 120x36 / 160x45 / 220x55 | This design |
|---|---|---|
| Chrome rows above + below content | 17 / 17 / 17 | **7 / 7 / 7** |
| First list item row | 23 / 23 / 23 | **7 / 7 / 7** |
| Characters visible (of 28) | 5 / 9 / 14 | **28 / 18 / 23** |
| Editor width while authoring | 58 / 78 / 108 | **110 / 108 / 129** (at 220 by default: body 107 + a 60-column Try column, Q2; 129 with it off) |
| Read-only card width (the accepted trade, Q12) | 58 / 78 / 108 | 50 / 73 / 129 |
| Lore entry rows (250-entry book) | 4 / 10 / 10 | **23 / 33 / 43** |
| Test chat transcript | unusable (120x36) / one exchange (160x45) | **≥20 / ≥29 / ≥39 rows**, full height |
| Projected NN/g score (of 40) | 11 (Library 27) | **≈29** (30 once B6's aggregates land) |

**Out of scope** (B leaves a slot or a guard for each; §0.5):

- Sub-project A's fixes (TASK-33781..TASK-33791). B assumes them, and B2 absorbs RP-009 and RP-022 if they are still open.
- World-info semantics (C): whether a character's lore copies apply in chats (D6), entry-level search, reorder speed.
- Persona semantics (D): the Enabled flag, which persona fields reach the model, a "you" description, a draft-aware persona preview.
- Console defects: TASK-33621.2 (gates only B5c), TASK-33792, TASK-33793.
- Shell and design-system defects (the nav highlight after Stay, app-wide button focus) and a batch of copy fixes with no other owner.
- Not in B at all: pane widths in Settings (Q9), a true single-stage layout below 64 columns, persona Duplicate, entry autosave, one "search all Roleplay" box (Q8).

**Slices** (§5.1-§5.2; one `PS` slice in review at a time):

| Slice | What ships | Size | Waits for |
|---|---|---|---|
| B0 | Shared frame primitives and the new ADR; no visible change | M | — |
| B1 | One-row header; the lazy Roleplay stylesheet | S | — |
| B2 | Line-based items list, marks bar, one delete modal, New chooser, one Import picker | L | D14 dry-run census |
| B3 | Keyboard foundations: arrival, Escape ladder, letter scope, `/`, footer and F1; the ADR-031 Ctrl+S amendment | M | — |
| B4 | Test chat and Try it as full-height views | M | — |
| B5a | Chats view | M | panel API agreed with TASK-22988 / TASK-31243 |
| B5b | Work header, view strip, More actions, Info; the Inspector is removed | L | A: TASK-33781, TASK-33789, TASK-33791 |
| B5c | `o` = Chat now, the provider end-to-end test | S | TASK-33621.2 on dev (D13) |
| B6 | Rail, counts and per-kind memory | L | — |
| B7 | Frame geometry: grips, work sessions, saved pane preferences | L | A: TASK-33781; ADR-011 evidence |
| B8 | Try column at ≥200 | S | — |
| B9 | Entries split and drill-in; entry and Options drafts; Ctrl+S there | L | A: TASK-33781-33783, 33785, 33786 |
| B10 | Landing, Start row, kind empty states, Continue block | M | ADR-120/ADR-004 amendment (G5) |
| B11 | Editors: growth caps, sections, Ctrl+S in persona Look and Tool policy | M | A: TASK-33784, TASK-33789 |
| B12 | User Guide consolidation | S | all |

Order: B0 ∥ B1 → B2 → B3 → B4 → B5a → B5b → B6 → B7 → B8 → B9 → B10 → B11 → B12. B5c merges whenever TASK-33621.2 is on dev.

---

## 0. Goals and non-goals

### 0.1 Goal

Rebuild Roleplay (Ctrl+4) on the Library's frame. The frame is a Roleplay navigation rail, an items list and a permanent work pane with views and one action bar. The right-hand Inspector goes, and its contents fold into the work pane. Four jobs are equally important:

- **J1:** import cards and chat fast.
- **J2:** author characters in depth.
- **J3:** build world info (lore books and chat dictionaries).
- **J4:** manage personas (assistant-side, ADR-037) and "you".

The design centre is 120x36 to 160x45 and 200+ columns. 80x24 and below 64 columns must only degrade gracefully.

What the frame is expected to buy, measured against today (report §3) and resolver-verified (§2.10):

| Measure | Today 120x36 / 160x45 / 220x55 | This design |
|---|---|---|
| Chrome rows above + below content | 17 / 17 / 17 | **7 / 7 / 7** (5 + 2) |
| First list item row | 23 / 23 / 23 | **7 / 7 / 7** |
| Characters visible (of 28) | 5 / 9 / 14 | **28 / 18 / 23** |
| Editor width while authoring | 58 / 78 / 108 | **110 / 108 / 129** (work session; at 220 by default: body 107 + a 60-column Try column, Q2; 129 with it off) |
| Lore entry rows (250-entry book) | 4 / 10 / 10 | **23 / 33 / 43** |
| Test chat | unusable / one exchange | full-height view; ≥20 transcript rows at 120x36 |
| Projected NN/g score | 11/40 (Library 27) | **≈29/40** (§7) |

### 0.2 Owner decisions (binding)

| # | Decision | Where it is realised |
|---|---|---|
| O1 | Roleplay stays its own Ctrl+4 destination, rebuilt on the Library frame (Approach 1): rail + items list + work pane with views and one action bar. | §1, §2, §3 |
| O2 | The Inspector is removed; its contents fold into the work pane. | §3.7 |
| O3 | J1-J4 are equally important. | Rail IA §1.4 (four kinds as equal rows); landing §3.10 (four equal New verbs); journeys §7.2 |
| O4 | Design centre 120x36-160x45 and 200+; 80x24 and <64 degrade gracefully only. | §2.8 (no new single-stage machinery below 64) |

The twelve design questions raised while designing were ruled by the owner on 2026-10-02 (§8). Each ruling is applied throughout; none is pending.

### 0.3 Default decisions: adopted, refined or deviated

| D | Default | This design | Why, if refined |
|---|---|---|---|
| D1 = A | Inspector folds: header CTA "Chat now" + readiness word; "More actions ▸"; Info view; Chats view; Buddy as a rail row | **Adopted.** Readiness is never dropped while the primary is disabled, and a destination-wide block shows a header chip (§1.3, §3.2). | At 160x45 a dropped readiness word would fall into a tooltip. |
| D2 | Test chat / Try it is a view at every size, plus a Try column at 200+ | **Refined.** The view works at every size. The column appears only beside **authoring** views (Edit, Look, Entries), never beside Card or Profile. Per-kind defaults: characters, lore and dictionaries **on**, personas **off** (Q2). | A default-on column beside Card would close the rail at 200-222 and flap it on view switches. |
| D3 | Narrow "You" rail row showing the effective `{{user}}`, deep-linking to Settings | **Adopted** (§1.4). Gloss "your name in chats" (report D3's text for the no-user-description case). | — |
| D4 = A | Promote the Library resolver, grips and normaliser to shared code (ADR-086 amendment), with Roleplay width profiles | **Adopted.** The resolver is byte-identical, there is one new shared module, and the Library rail-width coupling is recorded deliberately (§2.1). | — |
| D5 = B | `[roleplay.reader]` stores preferred state only, plus per-mode memory | **Adopted.** No custom-width keys in B (no Settings surface); `last_kind` and `sort` sit in a destination-owned carve-out (§2.11). | Governance: hidden config is not justified. |
| D7 = A | One-row inline DestinationHeader | **Adopted,** with a DESIGN.md §5 amendment for the inline variant (§5.9 G12). | DESIGN.md:328-330's contract (purpose line, readiness, tall border) needs the amendment. |
| D8 = B | Letters work only from list, rail and view controls, with a focus-aware footer | **Adopted** (§4.6). | — |
| D9 | Two rule calls: amend ADR-031 for an editor-scoped Ctrl+S (keeping the visible Save); invalid regex warns, visibly | **Adopted** (Q5): ADR-031 is amended and Ctrl+S is extended to every Roleplay editor (R41, G9a). Invalid regex stays a visible warning (§3.5.4, RP-015). | — |
| D11 = B | Landing in the work pane with a live primary action | **Adopted** (§3.10). Its cross-character "Continue a character chat" block is **shown**, authorised by a bounded ADR-120 amendment mirrored in ADR-004 (Q7, G5). | — |
| D12 = A | Entries list + entry editor in the work pane: side by side at ≥100 work-pane columns, drill-in below | **Adopted** (§3.9). | — |
| D14 | Virtual list vs fixed paging, decided with evidence | **A: one `OptionList`**, loading **database windows** (search, sort and tag stay in SQL). B, fixed paging, is the fallback (§1.5.6). | The local list is not in memory: it is paged from SQL. |
| Rail IA (C.5) | C.5 starting point and canonical vocabulary | **Adopted with recorded deviations** (§1.4.7): no rail search, no Continue section, a Start row, Create & import as two rows. | Row budget and one home per verb. |

### 0.4 Preserve list (report §6): where each item lands

| Preserved item | Where it is kept | Status |
|---|---|---|
| Try-it previews (struck-through originals, fired/skipped reasons, `_SKIP_REASON_COPY`) | §3.8; summary first, 1fr results | kept. **The run key changes:** Enter runs and Ctrl+J inserts a newline (Console composer grammar); Ctrl+Enter stays as an unadvertised alias (§4.6). |
| ADR-046 leave guard **Save and continue / Discard and continue / Stay** | §3.12; the same dialog family for in-screen transitions | kept. **The label stays "Stay"**. |
| Generate-then-Accept; labelled character editor with per-field Generate | §3.11 | kept; only the ✨ is dropped from labels (RP-071) |
| Single Escape owner that never leaves the screen | §4.5 (12 rungs, one binding) | kept |
| Whole-item delete confirmation; dictionary versions with confirmed Revert | §3.6 (one modal for every delete), §3.5.4 (History view) | kept; inline delete confirmation retired (report §10 #9) |
| Save/Cancel anchored outside the editor scroll | §3.1 commit bar, same ids | kept; Ctrl+S presses the *visible* Save (`PS:16295-16312`, which checks the editor view's `display` before pressing; `Button.press()` checks only the button's own `display`, `textual/widgets/_button.py:429`), now in **every** Roleplay editor (Q5, R41, §4.6) |
| Ctrl+F > name > Enter > Enter | §1.6 | kept; `/` is an alias from navigation surfaces |
| c/p/d/l + `[ ]` (ADR-152) | §4.6 | kept, keys bound to the same nouns. The order follows the rail (c p l d), and `MODE_CHIP_ORDER` and `MODE_HOTKEYS` are reordered in lockstep (R2). |
| Bulk marks | §1.5.5 | kept; marks by id with ✓ (recorded in the glyph legend) |
| Quick import (auto-select, remembered folder, embedded-lore toast, re-import message) | §1.4.6 | kept; the unified picker **reuses the remembered folder** |
| One-click Chat now hand-off | §3.6 | kept; the button moves unchanged in B5b, and only its promotion is D13-gated (B5c) |
| Mode/selection restore on leave and return | §4.11 | kept and extended to per-kind memory |
| Footer hints rebuilt from live state | §4.8 | kept; extended with focus |
| ADR-115 demand-mount boundary | §3.8.4, §5.9 G6 | kept; compose once, display-only toggles |
| Reused `#ccp-*` ids (ADR-004) | §3.5, §3.7 | kept; the two container renames are recorded exceptions (§3.7) |
| Stable pane geometry across modes (§6 "good bones") | §2.3 work sessions | **kept.** Pane geometry changes only at the start or end of a work session, never on a view switch. |

### 0.5 Non-goals and out of scope

B may reference these surfaces, but it must not depend on their outcome. Each one has a slot or guard in B, named in §6.

| Sub-project | Findings | What B does |
|---|---|---|
| **A, fix-first** (TASK-33781..TASK-33791, filed in PR #2957) | RP-001, 002, 003, 005, 009, 014, 022, 073, 075, 078, 079 | Assumes they land first or alongside. Hard prerequisites per slice are in §5.5. B adds guards: hidden slots have height 0, the generation fence, the EntriesAnchor hook, markup-off prompts and toasts. B absorbs RP-009 and RP-022 structurally in B2 if they are still open. |
| **C, world-info semantics** | RP-004 / D6, RP-012 (data), RP-013, RP-056 (internals at scale), RP-083, RP-096 | Honest copy from one runtime flag (`roleplay_character_world_info_applied`, false today); the "Used by" slot. |
| **D, persona semantics** | RP-011 (beyond the rail gloss), RP-077, RP-082, RP-086, RP-087, RP-088, RP-089 (semantics), RP-090 (semantics), D15; D3's "user description" ruling | Containers only (Look and Tool policy sections); no on/off word on persona rows; the persona Test chat header says "Uses the saved persona". |
| **Console** | RP-072 and RP-074 (TASK-33621.2), RP-084 (TASK-33792), RP-097 (TASK-33793), RP-047 (payload contract) | Only Chat-now *promotion* (B5c) is gated on TASK-33621.2 (D13). Hand-offs RP-047 says are inert are hidden behind a code flag, as ADR-031 rule 4 requires. |
| **Shell / design system** | RP-061 (nav highlight after Stay), RP-065 (button focus, D10), RP-024's "‹ le" nav fragment | B mitigates: no button is an arrival or F6 target, and the footer's Enter chip names the action. The nav-fragment task targets `UI/Navigation/main_navigation.py`; it is filed with the parent task (§5.13) and B3 links it. |
| **Content batch** (no owner after A-D) | RP-023, 040 (rest), 045 (speaker labels), 046, 048, 051, 057, 058, 059, 069, and the parts of RP-041 outside Roleplay (palette aliases, Console Inspector headings) | Filed as one batch so it does not fall between sub-projects. B provides slots: Example dialogue, the picker's format list. |

### 0.6 Design rulings (R1-R41)

Each ruling names the section that owns the topic and the review lens that raised it. R41 records the design consequence of owner ruling Q5.

| # | Topic | Ruling | Owner § | Raised by |
|---|---|---|---|---|
| R1 | Key `i` | `i` = **Import…**. Info is reached with `< >` (they **wrap**, so `<` from the default view reaches Info in one key) or ←→ on the strip. | 4.6 | — |
| R2 | Kind order | Rail order **c p l d**. `MODE_CHIP_ORDER` = (characters, personas, lore, dictionaries) **and** `MODE_HOTKEYS` = (c, p, l, d) are reordered **in lockstep**, so `l` → lore and `d` → dictionaries never swap (`PS:1111` zips them positionally). A test pins `l`→lore and `d`→dictionaries next to `test_personas_workbench.py:12729`. | 4.6 | consistency major; governance minor |
| R3 | Glyphs | Marks are `✓` (ASCII `[x]`), added to the glyph legend because `glyph_fallback.py:36-40` already uses ✓ for "done". **● is not used anywhere in Roleplay.** Unsaved is the *word* "unsaved". The active view chip uses a semantically named active-marker constant, not `GLYPH_COLLAPSED`. | 1.5, 3.3 | consistency, governance |
| R4 | Search box | One items-toolbar box per kind, id `#personas-filter` (the existing `RoleplayReturnTarget.personas_filter()` target, `character_conversation_navigation.py:33, 90`). No rail search box. | 1.6 | feasibility minor |
| R5 | Rail sections | Start · Cast · World info · Create & import (**two rows: New…, Import…**) · Buddy. | 1.4 | user-advocate minor |
| R6 | Landing trigger | The **Start** rail row (was "Overview") and arrival with nothing to restore. `‹ Roleplay` exists only below 64 columns. | 1.4, 3.10 | — |
| R7 | Arrival focus | The items list. `Import…` is the landing's first F6 stop. | 4.3 | — |
| R8 | Readiness wording | The model name or the block reason, **never "Ready"**. Never dropped while the primary is disabled. | 3.2 | user-advocate major |
| R9 | Geometry is session-scoped | Pane geometry follows the **work session** (§2.3), never the active view. Save keeps the session. | 2.3 | **both blockers** |
| R10 | F6 in reading views | The view strip (it scrolls the body). | 4.4 | — |
| R11 | Escape | The 12 rungs of §4.5: rung 1 has no inline delete confirmation, and rung 7 ends the work session. | 4.5 | — |
| R12 | Header content | `Roleplay   <Kind>[ · editing]` · optional chips · `Local` / `Server: … · read-only`. The item name appears in the header only on the list stage below 64 columns (it is in the work-pane title row everywhere else). | 1.3 | user-advocate minor (H8) |
| R13 | Landing content | §3.10, including the Continue block (Q7). B's landing has no Chat-now element (§3.10.2); any future Chat-now landing element is D13-gated. | 3.10 | — |
| R14 | Persona More actions | No persona Duplicate (new capability; own task). | 3.6 | — |
| R15 | Rail kind-row ids | `Button`s `#personas-mode-{characters,personas,lore,dictionaries}` with `.is-active` (41 refs in 7 files). View chips are `#personas-work-mode-<slot>`. | 1.4, 3.3 | — |
| R16 | Marks lifetime | Per kind, for the **screen visit**. They survive kind switches and are **not** saved in `save_state`, so leaving Roleplay clears them (bulk-delete safety). | 1.5.5 | consistency minor |
| R17 | Shared code | One new shared module, `Widgets/adaptive_pane_shell.py` (shell, grip, messages, `StageReturnBar`, rail-row primitives, compact search input). | 2.1 | — |
| R18 | Lazy CSS split | Created in B1. Prefixes: `roleplay-shell`, `roleplay-rail`, `roleplay-nav`, `roleplay-items`, plus narrow `personas-header`, `personas-library`, `personas-work`, `personas-more`, `personas-try`, each after the compose-site audit (`css/build_css.py:434-440`). **`pinned = {"personas-library-rows"}`**, so `#personas-library-rows:focus` stays in boot (`Tests/UI/test_personas_library_rail_focus_outline.py:20-37`). `roleplay-try` is dropped; bare `personas` is never claimed. | 2.12 | consistency, governance minors |
| R19 | Pane borders | **Bordered** (Q1): one `round` border per pane, with zero-row border titles (`Characters 28`, `4 of 28`) and the zero-row fold cue `▾ more`. Borderless panes were considered and rejected; no DESIGN.md:236 exception (G13 dropped). | 1.1 | owner ruling Q1 |
| R20 | Try column | Authoring views only (Edit, Look, Entries), per-kind preference. Defaults (Q2): characters **on**, lore **on**, dictionaries **on**, personas **off**. | 2.7 | blocker |
| R21 | Work-pane header rows | Row 1: crumb · name · state word. Row 2: primary · readiness · More actions; it **joins row 1 when `w` ≥ 100** and the row still fits. Then the view strip. So the primary is always in work rows 1-2 and above the strip. Tab order: primary → More actions → strip. | 3.1 | user-advocate major |
| R22 | View strip fitting | (1) full labels with counts (chip = label + 2) → (2) drop counts other than warnings → (3) tight chips (label + 1) → (4) short labels (`Test chat`→`Test`, `Tool policy`→`Policy`, `Info (1 warning)`→`Info !`, `Used by`→`Used`, `Try it`→`Try`) → (5) only if step 4 still overflows: keep the active chip and fold the trailing chips into `More ▾`. Measured: every kind fits by step 4 at content 46 (120x36 browse), and the character strip fits at content 34 (80x24). No chip is hidden in any BROWSE layout ≥92 (content ≥46), in any work session at ≥92, or on the XS work stage (content ≥50). On the XS list stage (work 22-49) the strip may fold trailing chips into `More ▾` (step 5); at 80x24 this applies to dictionaries, and below ≈79 columns to every kind. | 3.3 | consistency, feasibility, user-advocate |
| R23 | Footer vocabulary | Rail kinds are **"kind"** (`c/p/l/d [ ] kind`). Work-pane modes are **"view"** in user-facing copy (`< > view`, `←→ view`), and this document uses the same two words. | 4.8 | consistency major |
| R24 | Unsaved signals | Exactly three, all driven by **one predicate**, the ADR-046 aggregate snapshot (`_aggregate_roleplay_draft_snapshot()`, `PS:15979-16027`, plus in-flight saves): the header chip `Unsaved changes`, the title-row word `unsaved`, and the commit-bar summary `2 unsaved: Description, Personality`. No chip ●, no list-row word, no per-field marker. | 1.3, 3.11 | consistency major; governance major |
| R25 | Empty kinds | The **work pane** owns the explanation and actions (`PersonasKindEmptyState`). The list shows one muted line (`No lore books yet`). | 1.5.4, 3.10 | consistency major |
| R26 | `/` | **Pane-scoped.** From the rail, a grip or the items list → the items search (reopening the list if a work session closed it: a manual items priority). From the work pane → that view's filter (Entries, Chats), else the items search. Ctrl+F always → the items search. Armed whenever a target box exists. A Library follow-up is filed to converge `_library_slash_would_land` (`LS:7211`). | 1.6, 4.6 | consistency, governance, user-advocate |
| R27 | One Import label | **`Import…`** everywhere (rail, landing, empty states, F1, palette). Never "Import card…" or "Import lorebook…". | 1.4.6 | consistency major |
| R28 | Hand-off verbs | `Chat now` (new chat), `Continue in Console` (an existing conversation: Chats toolbar, landing rows, `o` on a conversation row), `Add to Console message` (hidden behind a code flag until RP-047). "Resume conversation" and "Send to Console draft" are retired as labels. | 3.6 | consistency major |
| R29 | D13 scope | TASK-33621.2 gates **promoting** Chat now (B5c: `o` and the e2e test), not moving it (B5b). Owner ruling Q11. | 5.2 | consistency, user-advocate major |
| R30 | New draft domains | The entry form and lore/dictionary **Options** (was Settings) become ADR-046 aggregate domains (`entry_form_dirty`, `lore_settings_dirty`), via a dated ADR-046 amendment (B9). | 3.12 | **governance blocker** |
| R31 | `del` / `space` / `n` in work-pane lists | Armed **only on the entries list**. Never on the conversations list, the versions list or tool-policy rules (ADR-004:22-24, ADR-120:195-197). | 4.6 | governance major |
| R32 | Delete confirmation | One modal for every Roleplay delete, naming up to 5 items plus "and N more". No inline delete confirmations. | 3.6 | consistency minor |
| R33 | Toast markup | Every prompt and toast that carries a user-supplied name is built with markup off (`markup=False` or a `Content` object). One undo copy string: `<name> turned off · space to undo`. | 1.5.6, 4.12 | consistency minor (RP-075 class) |
| R34 | Fold cue | Panes are bordered (Q1), so `▾ more` (through `resolve_glyph`) sits in the bottom border (zero rows), plus a DESIGN.md:301-304 note. | 3.1 | consistency, governance minor |
| R35 | Footer API | The `ShortcutContext` path (`set_shortcut_context`, non-persisting; `UI/Navigation/shortcut_context.py:9-25`), keeping the no-recompose pin (`Tests/UI/test_screen_footer_hints.py:599`). | 4.8 | feasibility minor |
| R36 | Book "Settings" view | Renamed **Options** in the strip only (tab ids kept), so it no longer collides with the You row's "Settings ›". | 3.5 | user-advocate minor |
| R37 | XS stages | 64-91 columns: the list stage and the work stage switch on **events** (Enter, `e`, `t` → work; Esc rung 7, the crumb, `/` → list), never on focus (DESIGN.md:125, I10). | 2.8 | consistency minor |
| R38 | Below 64 | No new single-stage machinery in B. The resolver's existing list-first rule plus grips is used. A true single stage is filed as a separate, estimated follow-up. | 2.8 | feasibility major |
| R39 | Shell id | The shell is `#roleplay-shell` (new). `#personas-workbench` is retired with its container (15 refs re-pinned in B7), a recorded exception to "moved, never renamed". | 2.1 | consistency |
| R40 | Mockups | The mockups in this spec (MK1-MK9b) follow these rulings at true width. A slice's approval screenshots come from the live harness (§5.7.5), and a plan that changes a mockup regenerates it from these rulings (§5.13). | 5.13 | consistency major |
| R41 | Ctrl+S | An editor-scoped Save in **every** Roleplay editor: character Edit; persona Edit, Look and Tool policy; lore and dictionary Options; the entry editor. It presses the **visible** commit button of the editor that holds focus, never a hidden one. Elsewhere it is a no-op: never advertised, and a stray press shows the footer state chip `ctrl+s: nothing to save here`. ADR-031 is amended (G9). Binding order: B3 (amendment; character and persona Edit), B9 (entry editor; lore and dictionary Options), B11 (persona Look and Tool policy). | 4.6 | owner ruling Q5 |

---

## 1. Frame, header, rail and items list

### 1.1 Anatomy and chrome

```text
row 1-3     nav bar (shell, unchanged)
row 4       DestinationHeader, one row (§1.3)
row 5       pane top borders: rail title "Roleplay", list title "<Kind> <count>", work pane plain
row 6..H-2  [rail R][grip 5][items L][grip 5][work W]   (list: row 6 toolbar, rows 7.. items)
row H-1     pane bottom borders: list position "4 of 28", work "▾ more"
row H       footer (focus-aware projection, §4.8)
```

- **One framing layer.** The shell (`#roleplay-shell`, an `AdaptivePaneShell`) spans the full terminal width. There is no outer workbench border or padding. Each pane has one `round` border (Q1; DESIGN.md:236-240).
- **Chrome.** 5 rows above the first content row and 2 below, at every width ≥64. Content rows inside the panes: `R = H − 7` (17 / 29 / 33 / 38 / 48 at 24 / 36 / 40 / 45 / 55 rows).
- **Grips.** Full-height 5-cell `AdaptivePaneGrip`s, reserved whether the pane is open or closed. The rail grip paints **"Nav"** (the Library's word; tooltip "Collapse Roleplay navigation"). The items grip paints the **kind noun** (`Characters`, `Personas`, `Lore`, `Dictionaries`), so a closed list still says what it holds. Arrows are `<---` (open) and `--->` (closed). They replace the 13- and 11-cell handles and the 3×1 `<` button (`PS:607-609`).
- **Considered and rejected (Q1):** borderless tonal panes (4 + 1 chrome rows, 6 more columns), which would need a DESIGN.md:236 exception and lose the zero-row border titles and fold cue (R34).

**Fixes:** RP-007 (13 + 4 → 5 + 2 rows), RP-032 (one grip grammar), RP-042 (no "Library" title), RP-066 (labelled grips; no "L"/"In" slivers).

### 1.2 Grips and the collapse grammar

| Property | Rule |
|---|---|
| Control | `AdaptivePaneGrip` (§2.1). The pane name is painted down the column, with arrows on the approved rows (`Widgets/Library/library_adaptive_reader_shell.py:172-222`). |
| Activation | Click or Enter toggles the pane. It writes the **preferred** `*_open` flag (persisted, §2.11) and resolves with that pane as priority. Responsive and work-session closes are never written. |
| Focus | Closing a focused pane moves focus to its grip. A manual reopen restores the pane's last-focused descendant (`library_adaptive_reader_shell.py:317-326, 445-455`). |
| F6 | A closed pane's grip takes the pane's place in the F6 cycle. An open pane's grip is not an F6 stop, but it is a Tab stop in its visual position. |
| Retired | `#personas-library-rail-handle`, `#personas-library-rail-open`, `#personas-library-rail-collapse`, `#personas-inspector-rail-handle`, `#personas-inspector-rail-open` (re-pinned: `Tests/UI/test_personas_workbench.py:586-652`). |
| Not a grip | The ≥200 Try column is a placement of the Try view, toggled from its chip and its header (`Hide ×` / `Keep beside ›`). It is never a collapsible shell pane and has no grip. |

**Fixes:** RP-032, RP-064 (focus evacuation), RP-025 (grip substitution in F6).

### 1.3 Header (D7 = A)

```text
 Roleplay   Characters                                                                                                                                    Local 
 Roleplay   Characters · editing                                                                                                        Unsaved changes   Local 
 Roleplay   Characters                                                                                                            Server: home-tldw · read-only 
 Roleplay   Characters                                                                                                    No chat provider · Settings ›   Local 
```

(160 columns: browsing; an editing session with a draft; server runtime; a destination-wide block.)

| Slot | Content | Rules |
|---|---|---|
| Title | `Roleplay` | Constant. |
| Subtitle | `<Kind>`, plus ` · editing` while a work session is active and an editor body is visible: character or persona Edit, Look, Tool policy, Options, or the entry editor (split or drill-in). It is absent in Test chat, Try it, read views and the drill-in entries list (MK6). | The kind is never cut. It names the active kind when the rail is closed (RP-029). The **item name is not repeated here**: it is in the work-pane title row (R12; H8). The one exception is the list stage below 64 columns, where the title row is off-screen: there the subtitle reads `<Kind> › <item>`. During B1-B5a (before B5b's title row exists), the subtitle carries `› <item>` as an interim (§5.3). |
| `before_status` chip 1 | `Unsaved changes` (short `Unsaved` below 100 columns), `$ds-status-warning`, word only | Shown **iff** `not _aggregate_roleplay_draft_snapshot().is_clean` or a save is in flight (R24). This is the same function `_run_guarded` uses from B5b (§3.12). |
| `before_status` chip 2 | `No chat provider · Settings ›` | Only when the **whole destination** is blocked (no provider configured for character chats). It deep-links like the You row and passes the leave guard. **Never "Ready".** |
| Status chip | `Local`, or `Server: <label> · read-only` | Data source and authority (DESIGN.md:115). |

- **Build.** `DestinationHeader(WorkbenchHeaderState(title="Roleplay", subtitle=…, status=…), before_status=<chips>, id="personas-header", classes="personas-header-inline")`. The `before_status` slot already exists (`UI/Workbench/workbench_widgets.py:164-245`).
- **CSS.** The CSS copies `#console-workbench-header.console-header-inline` (`css/features/_console_panels.tcss:490-525`) as an id-scoped rule in the lazy sheet, and keeps the subtitle visible at ≤24 rows (unlike `UI/Navigation/base_app_screen.py:62-65`).
- **Neutral status.** `WorkbenchStatus` has no neutral value (`UI/Workbench/workbench_state.py:11-19`). Style the chip neutral by id; no new status value.
- **Governance.** DESIGN.md:328-330 and front matter :81-85 require a purpose line, readiness and a `tall` border. B1 amends them with an inline one-row DestinationHeader variant used by Console, Lab and Roleplay: readiness may sit beside the destination's primary action in the work pane (§5.9 G12).

**Fixes:** RP-007 (5 header rows → 1), RP-067 (the false "Ready" badge is gone; a block is shown in words), RP-038 (cross-pane unsaved signal), RP-029 (kind named), RP-052 (interim, until the title row lands), RP-071 (`›` through `resolve_glyph`).

### 1.4 Roleplay navigation rail

#### 1.4.1 Rows

Rail text width is pane width − 4 (border + padding): **24 / 31 / 35** at 120 / 160 / ≥180.

| Row (section) | Label / fallback | Count | Gloss (line 2, ≤18 cells) | Key | Activation |
|---|---|---|---|---|---|
| **Start** | `Start` | – | `import · recent` | – | An **action**, not a location. It unloads the open item (guarded), **ends the work session**, and shows the landing in the work pane. Kind and list stay as they are; the current kind row keeps its `▸`. Focus goes to the landing's first action, `Import…`. |
| **Cast** › Characters | `Characters` | `(28)` | `who the AI plays` | `c` | Switch kind (guarded); focus moves to the list at the restored cursor. |
| Cast › Personas | `Personas` | `(4)` | `reusable AI roles` | `p` | Same. |
| Cast › You | `You: <effective name>` … `Settings ›` | – | `your name in chats` | – | Deep link to Settings › Console Behavior through the leave guard. In `LEAVES_SCREEN_IDS`, so it is never an arrival or F6 stop. |
| **World info** › Lore books | `Lore books` / `Lore` | `(3 · 2 on)` / `(3)` | `facts on keywords` | `l` | Switch kind. |
| World info › Chat dictionaries | `Chat dictionaries` / `Dictionaries` | `(3 · 2 on)` / `(3)`, then ` !` when any dictionary has an invalid rule (B6) | `find/replace rules` | `d` | Switch kind. |
| **Create & import** › New… | `New…` | – | `pick a kind` | `n` (from navigation surfaces) | One chooser; the current kind is preselected. Character: Blank · From a concept · From a portrait (Actor Pack). Persona: Blank · From a portrait. Lore book. Chat dictionary. Enter Enter gives a blank draft of the current kind. |
| Create & import › Import… | `Import…` | – | `any Roleplay file` | `i` | The one Import picker (§1.4.6). |
| **Buddy** | `Buddy: shown/hidden/off` / `Buddy` | – | `app companion` | – | A small chooser: Manage… · Show · Close · Disable. It moves the Inspector's four buttons (`#personas-buddy-use/-show/-close/-disable`, ids kept) and the `_buddy_action_pressed` handler. |

- **Words.** The canonical C.5 nouns. The left pane is never called "Library" (RP-042). "World Books" and "Dictionaries" become "Lore books" and "Chat dictionaries" (RP-041).
- **Glosses.** Built from the descriptors that already work (`PS:449-457`), with the persona gloss corrected (RP-011). Tooltips carry the long forms, for example "Lore books: facts added to a chat when a keyword appears (also called world info or lorebooks)".
- **Rows the rail leaves out.**
  - *Export* lives in the item's More actions and in the marks bar. It acts on items, and RP-020 wants one home per verb.
  - *Duplicate* goes to More actions.
  - *Sort* and *Tag* go to the list toolbar.

#### 1.4.2 Counts

| State | Rendering | Notes |
|---|---|---|
| Loading | `(…)` dim | All four kinds load at arrival, **off-thread, after first paint**. This ends the `· 0` that lasted 14 s or more (RP-028). |
| Known | `(28)`; lore and dictionaries `(3 · 2 on)` = total · enabled | One code path renders the rail count, the list header and the rows. |
| Search or tag active in that kind | `(3 of 28)` | Makes a search left in another kind visible from the rail (RP-021). |
| Server, total unknown | `(100+)` | Never a silent cap (RP-093's lesson). |
| Failed | `(?)`, with the gloss replaced by `couldn't load · enter` | Enter retries. Words, not colour. |

- **Data.** `COUNT` queries only, never the lore N+1 entry-content load (`PS:4354`, PERF-22 P2). B6's **scent aggregates** run in the same worker with explicit invalidation hooks (§1.5.2): per-character chat counts and last activity, per-dictionary invalid-regex counts, per-book entry/off counts. Any new index needs a plan captured with `sqlite_stat1` absent and a row in `scripts/index_plan_pin_census.tsv` (CLAUDE.md gotcha 1).

#### 1.4.3 Fitting, vertical fit, current and focus

- **Horizontal fitting** (pure `fit_rail_row_label`, ported from `LibraryRail._row_label`, `Widgets/Library/library_rail.py:1002-1063`): title + count + key → drop the key → short count → short title → short title + short count → ellipsis. The **canonical noun outlives the key hint**, because keys are always in the footer and F1. The count is never clipped. Results:
  - at 24 cells: `▸ Characters (28)  c`, `Lore books (3 · 2 on)`, `Chat dictionaries (3)`;
  - at 31 cells: `Chat dictionaries (3 · 2 on)`.
- **Vertical fitting**, all or nothing, so rows never shift one at a time:
  1. Glosses drop when rows are short.
  2. Then *Create & import* folds (effective state only).
  3. Then the bottom border carries `▾ more`.
- **Room.** With glosses the rail needs **25 rows**, against 29 at 120x36. At 80x24 (17 rows) the glosses drop and it needs 15.
- **Section state.** Open or closed per section, persisted in `[roleplay.reader.rail_sections]` (§2.11).
- **Current row:** `▸` + bold (`resolve_glyph`, ASCII `>`). The current kind stays marked while the landing shows, because the list still holds that kind.
- **Focused row:** `border-left: thick $ds-action-focus` in place of the 1-cell padding (the house rule, `css/features/_library.tcss:433-437`). It is never the same token as current (RP-063).
- **Navigation:** ↑/↓ rove, ←/→ collapse or expand a section, Enter activates.
- **Disabled rows** say why on their gloss line, for example `New…` → `local profiles only` in server runtime (DESIGN.md:287).
- **Build.** Reuse `DestinationRailSectionHeader` (`Widgets/destination_rail.py:163-289`) and the shared `DestinationRailRow` / `DestinationRailRowButton` (B0). Rows are patched, never recomposed (the Home precedent, `Widgets/Home/home_rail.py:52-90`). Do **not** subclass `LibraryRail` (`library_rail.py:579`).

#### 1.4.4 Kind keys (D8 = B)

- **Where they act.** `c p l d [ ]` act only from navigation surfaces: rail rows and section headers, grips, the items list, the entries list, the conversations list, the dictionary versions list, the tool-policy rules (WLIST) and the view strip. They are inert on buttons (including the items toolbar), text fields, form controls and reading bodies (§4.6).
- **Order.** `[ ]` follow rail order. `MODE_CHIP_ORDER` and `MODE_HOTKEYS` are reordered together (R2).
- **Hints.** Dim right-aligned letters on the kind rows (dropped first when space is short), and always in the footer chip `c/p/l/d [ ] kind` and in F1.
- **Guard.** Every kind switch (rail click, key, a New… that changes kind) goes through `_run_guarded` (`PS:16179`). A kind switch also **ends the work session** (§2.3).

#### 1.4.5 The You row (D3)

- **What it shows.** The effective `{{user}}` name: per chat → global → "User" (ADR-046:15-19). Outside a chat this is the global *Default chat display name*. If none is set it reads `You: User (default)`.
- **What it does.** Enter or click calls `NavigateToScreen('settings', {'category': CONSOLE_BEHAVIOR})`, the deep link the preview controller already uses (`PM/personas_preview_controller.py:390-395`), through the leave guard. The return restores Roleplay's state (§4.11).
- **What it never is.** A list kind, a persona, or a user-profile editor (ADR-037/046). A future user description (D3's extra ruling) belongs to sub-project D.

#### 1.4.6 New… and one Import… picker

- **New…** and the list's `+ New` use one path. `n` and `ctrl+n` open the current kind's chooser with Blank preselected.
  - "From a portrait" is today's New Actor Pack (`PersonaActionRequested("create_actor_pack")`), which folds the peer noun away (RP-043).
  - "From a concept" opens the character editor with its concept / "Generate whole character…" row focused (`PW/personas_character_editor_widget.py:400-711`).
  - New lore book and New chat dictionary open the **Options view as an unsaved draft** (Name focused; commit bar `[ Create lore book ]  Cancel`) instead of saving "Untitled" at once. After Create, the new book opens in Entries with `+ New entry` focused (§3.9.6, RP-040 for books).
- **Import…** (also `i`) is one picker for character cards (PNG/JSON V2/V3), `.tldw-actor-pack`, lorebook JSON and dictionary JSON/MD.
  - The type comes from the extension plus a content sniff. A V2/V3 `spec` field wins. A JSON that still fits more than one kind asks "Import as: Character card / Lore book / Chat dictionary".
  - It routes to the existing handlers, including the Actor Pack review dialog behind `ActorPackImportRequested`.
  - It **reuses the remembered folder**. The result is selected and loaded in its kind (quick import kept). The title lists the accepted formats.
  - RP-058's error copy is content work.

**Fixes (rail):** RP-029, RP-028, RP-008 (verbs leave the toolbar), RP-018 (Buddy leaves item actions), RP-020 (collection verbs in one place), RP-041, RP-042, RP-043, RP-011 (gloss + You row), RP-055 (Start, glosses), RP-063, RP-071, RP-033 and RP-034 (scope and order), RP-021 (`3 of 28`), RP-040 (books' New draft).

#### 1.4.7 Deviations from C.5 (recorded in the ADR)

1. **No rail search box.** Two boxes would give `/` two meanings. The rail has about 12 rows to filter, and a query shared across kinds makes other kinds look empty (RP-021's hidden filter). Per-kind search plus `Also in:` covers cross-kind need (§1.6; Q8).
2. **No "Continue" section.** It would not fit (25 of 29 rows are used at 120x36), it moves with activity, and it duplicates the landing and Chats. The **Start** row brings the landing back from anywhere, which also avoids the Library's landing-only-once defect (report §10 #10).
3. **Create & import is two rows** (`New…`, `Import…`) instead of C.5's Create and Import / Export sections. This saves 4 rows, keeps one home per verb (Export lives on items), and the chooser still lists every C.5 create variant.
4. **Glosses on a dim second line.** Inline glosses need 38 cells and never fit 24-35.
5. **`[ ]` follow rail order** (c p l d), changing today's c p **d l** cycle.
6. **Gloss for You is "your name in chats"**, report D3's wording for the no-user-description case (C.5 had "your name for {{user}}").

### 1.5 Items list

#### 1.5.1 Header and toolbar

| Kind | Border title | Toolbar row (one row; labels shorten, never wrap) |
|---|---|---|
| Characters | `Characters 28` · `3 of 28` · `(…)` · `20 of 348 · tag fantasy` | `▎Search characters… /` (short `Search… /`) · `Name` (sort) · `Tag` · `+ New` |
| Personas | `Personas 4` | search · sort · `+ New` |
| Lore books | `Lore books 3 · 2 on` | search · sort (Name / Recently edited) · `+ New` |
| Chat dictionaries | `Chat dictionaries 3 · 2 on` | search · sort · `+ New` |

- **The search box** (`#personas-filter`) is one row with the dense-form left edge (`settings-compact-input`, `css/patterns.json:25-28`), so focus never changes its height. This keeps sub-project A's RP-009 fix structural. Labels fall back to short forms first, and the box keeps at least 14 cells.
- **Sort and tag.** `s` cycles sort from the list. Tag opens `TagFilterPicker`, which gets RP-095's Enter/Down fix (`PW/tag_filter_picker.py:58-63, 87-115`).
- **Counts.** All four kinds pass `filtered`, so the title says `3 of 28` whenever a search or tag narrows the list (`PS:4329-4335` against `:3690`).
- **The bottom border** carries the position of the loaded row (`4 of 28`).
- **Today's toolbar** had 7 rows (New, New Actor Pack, Import, Import Actor Pack, Duplicate, Sort, Tag; `PW/personas_library_pane.py:200-242`). Its verbs move to the rail, More actions and the marks bar.

#### 1.5.2 Row format and density

| Kind | Line 1: gutter · name · right side (state words never drop) | Line 2 (2-line density only, dim) |
|---|---|---|
| Character | name; 1-line density: `2 chats` only if the full name still fits | `2 chats · 3d · <first tag> · <description snippet>` |
| Persona | name (no on/off word until sub-project D rules on Enabled, RP-086) | first words of the **System prompt** (the field that reaches the model); `· portrait: <character>` when linked (RP-090) |
| Lore book | name · `on`/`off` (1-line density adds `250 · ` only if the full name fits) | `250 entries · 28 off` (`· used by N` stays hidden until sub-project C supplies it) |
| Chat dictionary | name · `warning · on` / `on` / `off` (the `warning` word never drops; ASCII `!`) | `150 rules · 31 regex · 1 invalid` |

- **Gutter.** 3 cells: col 1 `▸` (loaded in the work pane), col 2 `✓` (marked), col 3 a space. In ASCII mode the gutter is 5 cells (`>[x] `). A **click on col 2 toggles the mark** (tooltip "Click to mark"), which is the mouse path to bulk actions (§4.10).
- **Names.** In 2-line density, line 1 holds only the name and its state word, so names get **36 cells at 160x45** (RP-094's ≥32 target). Long names use a middle ellipsis that keeps the last 9-10 cells (`Highspire Gazette…om card) 2`), with the full name in a hover tooltip (from `MouseMove`, `event.style.meta["option"]`, `textual/widgets/_option_list.py:740-746`).
- **Density.**
  - **2-line rows** when the list's inner width is ≥40 and the kind holds ≤40 items.
  - **1-line rows** otherwise: at 120 for every kind, and at 160 or 220 once a kind passes 40 items (43 lore books).
  - No blank separators (RP-068). No user toggle in v1.
- **v1 scent (B2)** uses only fields the page query already returns: tags, description, `last_modified`, dictionary `entry_count`, and lore entry counts from one `GROUP BY` (not N+1). The chat counts and dictionary warnings arrive with **B6's aggregates**; until then those segments are absent, never `0`.
- **Plurals and ages.** `1 entry`, `1 chat`; relative ages (`3d`) instead of raw UTC (RP-070).

#### 1.5.3 State rendering

| State | Signal (never colour alone) |
|---|---|
| Loaded in the work pane | `▸` + bold name |
| Keyboard cursor, list focused | The list's focus edge (`thick $ds-action-focus`) plus a `$ds-focus-bg` cursor row, bold, painted **only while the list has focus**. Id-scoped override `#personas-library-rows:focus > .option-list--option-highlighted`, because the global rule paints regardless of focus (`css/components/_lists.tcss:112-119`). Text ≥4.5:1, edge ≥3:1 in every theme (`test_theme_contrast` extended). |
| Cursor, list not focused | Not painted (RP-062's 3.83:1 border-colour cursor is retired). |
| Marked | `✓` in gutter col 2, plus the marks bar |
| Off | the `off` word + a muted name |
| Validation failing | the `warning` word on line 1 |
| Hover | Textual's hover background; no layout change |

The list carries **no unsaved marker**. Kind and item switches are guarded, so only the loaded item can be dirty, and it already shows `unsaved` in its title row (R24).

#### 1.5.4 Empty, no-match, loading, error

- **Empty kind (unfiltered total 0).** The list shows one muted line: `No lore books yet`. The work pane's `PersonasKindEmptyState` owns the explanation and the actions: `[ New lore book ]` and `[ Import… ]` (R25).
- **Search with no match.**
  ```text
   No characters match "kepler".
   28 in total · esc clears the search
   Also in:  Lore books 1 ›
  ```
  - `Also in:` appears only when another kind matches. Enter on it switches kind with the query applied.
  - The cross-kind counts come from one count query per kind, off-thread and debounced. For characters it is the same FTS expression; it is not computed from in-memory sets.
  - An empty-handed search says `Not found in other kinds either.`
  - While a search shows zero rows, the work pane keeps whatever it showed: the loaded item, the landing, or the kind empty state without its "pick one on the left" line. The work pane never comments on the list's search (RP-022's three contradicting messages).
- **Loading.** `Loading characters…` only after 250 ms.
- **Failure.** The existing recovery row (`PW/personas_library_pane.py:423-430`) with Retry.

#### 1.5.5 Bulk marks

- **Marking.** `m` marks the cursor row, or click gutter col 2. The footer advertises `m mark` whenever the list has focus (RP-034).
- **Marks bar.** Pinned to the bottom row of the items pane, one row, shown only while marks exist: `✓ 2 marked · Export… · Delete… · Clear esc`. It takes the bottom row, so the cursor row never moves.
- **What marks survive.** Marks are stored **by id**. They survive search, sort, tag, scrolling and kind switches; hidden ones are counted (`✓ 3 marked (2 hidden)`). They are **not** saved in `save_state`, so leaving Roleplay clears them (R16).
- **Delete.** `del` / `Delete…` opens the one delete modal (R32), which names up to 5 items plus "and N more" (RP-039's correction keeps F-040's "never hit unseen rows" through disclosure). Bulk delete never retargets silently: the label says `Delete 2 marked`.
- **Moved logic.** The Inspector's `set_marked_count` logic (`PW/personas_inspector_pane.py:193-201, 1095-1143`) moves to the marks bar.

#### 1.5.6 D14: one virtual list, loading database windows (decision A)

**Today.**
- `PersonasLibraryPane` mounts `ListItem > Vertical > Static×2` per row (`PW/personas_library_pane.py:443-475`).
- It clears and remounts on every keystroke, page, sort or refresh (`:418`; PERF-22 P1).
- It pages at 50 (`PS:486`). The page boundary causes RP-080 (about 41 names skipped), RP-039 (marks die) and RP-092 (a 1-3-row wrapping pager with no keys).
- **The local list is fetched from the database one page at a time**, with FTS search, sort and tag applied in SQL: `_reload_character_page` → `count_character_page` + `get_character_page_for_ui(db, limit=PERSONAS_LIBRARY_PAGE_SIZE, offset, order_by, search_term, tag)` (`PS:3828-3890`).
  - `fetch_all_characters()` (`UI/CCP_Modules/ccp_character_handler.py:400-412`) fills a Select dropdown, not the items list.
  - So the list is not in memory, and design A must window the database.

**Design of A.**

| Aspect | Rule |
|---|---|
| Widget | `RoleplayItemList(OptionList)` keeps the id `#personas-library-rows`. Option ids are `"<kind>:<id>"`. Public `OptionList` API only (`set_options :343`, `add_options :362`, `get_option_index :448`, `replace_option_prompt :647`; never `_lines` or `_clear_caches`). |
| Prompts | `Content` with markup **off** (R33; the RP-075 class). The middle ellipsis is computed at the content width. Prompts are rebuilt only when the width actually changes (`_on_resize` clears caches, `:712`). |
| Local loading | **Database windows.** The first window is `get_character_page_for_ui(limit=200)`, appended with `add_options` when the cursor or scroll comes within one screen of the loaded end. `count_character_page` feeds the header and rail counts. Search, sort and tag stay in SQL. Each window fetch runs in a worker (exclusive, group `roleplay-list-window`); no step passes 250 ms. |
| Server | Windows of 100 (`_SERVER_CHARACTER_SEARCH_MAX_RESULTS`, `PS:487`). The header shows `100+` until the total is known. |
| Personas | Page `list_persona_profiles` until exhausted (precedent `UI/Navigation/buddy_management.py:463-475`), which fixes RP-093's newest-100 cap. |
| State | After `set_options` (which resets scroll and highlight, `:343-360`), restore the cursor and anchor by id. Marks are an id set; a mark changes one prompt (`replace_option_prompt` clears the whole line cache, so a mark is O(n): 51 ms at 1,000 rows, 189 ms at 5,000, under the 250 ms gate). |
| Messages | `OptionSelected` → the existing `PersonaEntitySelected` (`PW/personas_messages.py`). Arrows only highlight; Enter or a click loads (I6). |
| Space | `space` toggles on/off for lore books and dictionaries, with the undo toast `<name> turned off · space to undo` (R33). |
| Test seams | Tests patch `personas_screen_module.count_character_page` (7×), `get_character_page_for_ui` (3×) and `PERSONAS_LIBRARY_PAGE_SIZE`. These names stay importable from `PS` (re-exported); #2862's re-export identity convention applies (§5.11). |

**Measured** (bench, Textual 8.2.8, headless, 2-line rows; each figure includes about 21 ms of idle wait):

| Operation | 50 rows | 348 rows | 1,000 | 5,000 |
|---|---|---|---|---|
| `ListView` clear + extend (today) | 182 ms | 583 ms | – | – |
| `OptionList.set_options` | 42 ms | 38 ms | 66 ms | 417 ms |
| Mark one row | 25 ms | 41 ms | 51 ms | 189 ms |

These numbers time **only** the widget. B2's acceptance is end to end: a keystroke → SQL window → painted rows ≤250 ms at 348 and at 5,000 characters, with `set_options` ≤50 ms at 348 and **0 widget mounts** per keystroke.

**Costs and risks.**
- **Test migration** (own line item in B2):
  - 91 `#personas-library-row-*` refs in 12 files;
  - **46** `query_one("#personas-library-rows", ListView)` refs in 6 files, about 70 ListView-API uses (`.children`, `highlighted_child`, `.index`; 26 refs in `test_personas_dictionaries.py` alone);
  - 345 `personas-library-*` refs in 24 files;
  - `_row_dom_id` imported 11×.
  - The harness helpers cover: select by id, highlighted id, visible ids, item at an index, empty/recovery state (§5.7).
- **Fallback B** is decided **after a dry-run census on a scratch branch** and before B2 merges.

**Fallback B** (fixed paging):
- keep `ListView`;
- a one-row, non-wrapping pager in the border title (`Characters 348 · ‹ 2/7 ›`);
- scroll to the top on every page change;
- PgDn on the last row and PgUp on the first turn the page;
- marks by id across pages;
- a cue when the loaded item is on another page;
- one `Static` per row.

B fixes RP-080, RP-092 and RP-039, but not PERF-22 AC#4.

**Fixes (items list):** RP-008, RP-009 (structural), RP-022 (extended), RP-028, RP-039, RP-062, RP-068, RP-070, RP-080, RP-092, RP-093, RP-094, RP-095, RP-015 (row state word, with B6), PERF-22 AC#4.

### 1.6 Search and find

- **`/` is pane-scoped** (R26).
  - From the rail, a grip or the items list → the items search. If a work session closed the list, this reopens it as a manual items priority for the rest of the session.
  - From the work pane → the active view's filter (Entries: `/ filter entries`; Chats: `/ filter chats`). Any other view → the items search.
- **Ctrl+F** always focuses the items search and works while typing.
- **The 4-key find is kept:** `/ isolde ⏎ ⏎` (or Ctrl+F) opens the card (0.34 s across 348 characters). With `o` (B5c) it becomes `/ isolde ⏎ ⏎ o`: 5 keys (RP-035).
- **Entries:** `/ grand ⏎ ⏎ e / bellw ⏎` reaches entry #200 in a 250-entry book (§3.9).
- **It searches more than names.** Characters use FTS over names and descriptions, so the word is "Search", not "Filter". Entry and chat filters are "Filter".
- **Library convergence.** The Library's `/` stands down when its box is in a closed pane (`LS:7211`). A Library follow-up task converges it on this rule (DESIGN.md:125: the same key means the same thing).

**Fixes:** RP-035, RP-021, RP-022, RP-056 (reach).

### 1.7 Mockups: browse

Legend for all mockups:

| Mark | Meaning |
|---|---|
| `▸` in the rail | Current kind |
| `▸` in a list gutter | Loaded item |
| `▸` on a view chip | Active view |
| `✓` | Marked |
| `▎` | One-row input's left edge |
| `N a v` / `C h a r…` columns | 5-cell grips; `<---` collapses, `--->` expands |
| `┄┄┄ rows a-b elided` | Rows left out of the mockup; not a screen row |

The keyboard cursor and focus edges cannot be drawn in plain text, so captions name where focus is.

**MK1 · 120x36 · Characters, browse, Captain Isolde Varga loaded, list focused on the loaded row.** Layout rail 28 · grip 5 · list 32 · grip 5 · work 50; full height. All 28 one-line characters are visible (today 5). `w` = 50 < 100, so the action bar takes row 2 of the work pane and the strip drops counts (R22 step 2). The 12×6 portrait thumbnail sits beside an at-a-glance block, because `w` < 84 (§3.4). The footer is L1. Fixes shown: RP-007, RP-008, RP-019, RP-028, RP-029, RP-042, RP-052, RP-067, RP-068, RP-094.

```text
                                      ╭─────────────╮                                                                   
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP     More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Characters                                                                                            Local 
╭─ Roleplay ───────────────╮  N  ╭─ Characters 28 ──────────────╮  C  ╭────────────────────────────────────────────────╮
│ Start                    │  a  │ ▎Search… /  Name  Tag  + New │  h  │ ‹ Characters  Captain Isolde Varga    saved 2d │
│     import · recent      │  v  │   Ada (Debugging Partner)    │  a  │ [ Chat now ]  gpt-4.1-mini              More ▸ │
│                          │     │   ARIA-7                     │  r  │  ▸Card  Edit  Chats  World  Test chat  Info    │
│ Cast                   ▾ │     │   Brother Anselm             │  a  │ ┌──────────┐ Tags   sci-fi · captain · survey  │
│ ▸ Characters (28)      c │     │▸  Captain Isolde Varga       │  c  │ │          │ Greet  1 primary · 2 alternates   │
│     who the AI plays     │     │   Chef Auguste        1 chat │  t  │ │ portrait │ Voice  Default narrator voice     │
│   Personas (4)         p │     │   Coach Pemberton            │  e  │ │  12 × 6  │ World  1 lore book (a copy)       │
│     reusable AI roles    │     │   Default Assistant  4 chats │  r  │ │          │ Chats  2 · last 3d ago            │
│   You: Alex   Settings › │     │   Detective Rosa Calderón    │  s  │ └──────────┘ Saved  2d ago · v3                │
│     your name in chats   │<--- │   Dungeon Master Vex         │     │                                                │
│                          │     │   Grumpy Barista Bot         │     │ BASICS                                         │
│ World info             ▾ │     │   Hana Kobayashi             │     │ Description                                    │
│   Lore books (3 · 2 on)  │     │   Kestrel                    │     │ Captain Isolde Varga commands the deep-survey  │
│     facts on keywords    │     │   Lady Evangeline Thornwood  │     │ frigate Meridian's Wake, a patched-together    │
│   Chat dictionaries (3)  │     │   Luna — Night Mar…ne Teller │<--- │ vessel that has outlived three refits and two  │
│     find/replace rules   │     │   Marla the Ferrywoman       │     │ mutinies. Raised on the orbital shipyards of   │
│                          │     │   Mira Okonkwo               │     │ Kepler-9, she learned navigation before she…   │
│ Create & import        ▾ │     │   Old Tom the Ligh…se Keeper │     │ Show all ▾                                     │
│   New…                   │     │   Professor Hargreaves       │     │ First message                                  │
│     pick a kind          │<--- │   Quartermaster Bell         │     │ "Welcome aboard. Mind the cat; he outranks     │
│   Import…                │     │   Rook                       │     │ you."                                          │
│     any Roleplay file    │     │   Samira “Sammy” Vadem       │     │ Personality                                    │
│                          │     │   Ser Corwin of th…shen Vale │     │ Dry, precise, loyal, quietly haunted.          │
│   Buddy: hidden          │     │   Sister Agnes               │     │                                                │
│     app companion        │     │   Theo the Cartographer      │<--- │ PROMPTING                                      │
│                          │     │   Vesper                     │     │ System prompt                                  │
│                          │     │   Wren Halloway       1 chat │     │ You are Captain Isolde Varga. Speak            │
│                          │     │   Yusuf the Spice Merchant   │     │ economically; never discuss the Hadley Gap     │
│                          │     │   Zephyr                     │     │ unless pressed twice.                          │
╰──────────────────────────╯     ╰──────────────────── 4 of 28 ─╯     ╰─────────────────────────────────────── ▾ more ─╯
  enter go to card | o chat now | e edit | t test | / search | c/p/l/d [ ] kind | n new | F1 · F6 · Ctrl+P · Ctrl+Q     
```

**MK2 · 160x45 · Characters, browse, Card (rows 1-28 and 43-45).** Layout rail 35 · list 42 (2-line rows; names get 36 cells) · work 73. The work pane is **73 columns, 5 fewer than today's 78** (Q12): the read-only card trades width for an always-visible rail and list. Footer L1 at 160.

```text
                                      ╭─────────────╮                                                                                                           
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP   ⌃0 ACP   F2 Lab   F3 Logs   F4 Settings   More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Characters                                                                                                                                    Local 
╭─ Roleplay ──────────────────────╮  N  ╭─ Characters 28 ────────────────────────╮  C  ╭───────────────────────────────────────────────────────────────────────╮
│ Start                           │  a  │ ▎Search characters… /  Name  Tag  + New│  h  │ ‹ Characters  Captain Isolde Varga                           saved 2d │
│     import · recent             │  v  │   Ada (Debugging Partner)              │  a  │ [ Chat now ]  gpt-4.1-mini                             More actions ▸ │
│                                 │     │   4 chats · 1h · coding · A patient s… │  r  │  ▸Card  Edit  Chats 2  World 1  Test chat  Info                       │
│ Cast                          ▾ │     │   ARIA-7                               │  a  │ ┌──────────┐ Tags        sci-fi · captain · survey                    │
│ ▸ Characters (28)             c │     │   3w · sci-fi · A ship AI that has qu… │  c  │ │          │ Greetings   1 primary · 2 alternates                     │
│     who the AI plays            │     │   Brother Anselm                       │  t  │ │ portrait │ Voice       Default narrator voice                       │
│   Personas (4)                p │     │   1mo · A monastery brewer with stron… │  e  │ │  12 × 6  │ World       1 lore book · a copy, not applied in chats   │
│     reusable AI roles           │     │▸  Captain Isolde Varga                 │  r  │ │          │ Chats       2 · last 3d ago                              │
│   You: Alex          Settings › │     │   2 chats · 3d · sci-fi · Commands th… │  s  │ └──────────┘ Saved       2d ago · v3                                  │
│     your name in chats          │<--- │   Chef Auguste                         │     │                                                                       │
│                                 │     │   1 chat · 5d · A temperamental Paris… │     │ BASICS                                                                │
│ World info                    ▾ │     │   Coach Pemberton                      │     │ Description    Captain Isolde Varga commands the deep-survey          │
│   Lore books (3 · 2 on)       l │     │   2w · A relentlessly upbeat running … │     │                frigate Meridian's Wake, a patched-together vessel     │
│     facts on keywords           │     │   Default Assistant                    │     │                that has outlived three refits and two mutinies.       │
│   Chat dictionaries (3 · 2 on)  │     │   4 chats · 1h · built in · Used for … │<--- │                Raised on the orbital shipyards of Kepler-9, she       │
│     find/replace rules          │     │   Detective Rosa Calderón              │     │                learned navigation before she learned to read…         │
│                                 │     │   1 chat · 2w · noir · A sharp, sardo… │     │                Show all ▾                                             │
│ Create & import               ▾ │     │   Dungeon Master Vex                   │     │ First message  "Welcome aboard. Mind the cat; he outranks you."       │
│   New…                          │     │   6 chats · 1d · fantasy · A theatric… │     │ Personality    Dry, precise, loyal, quietly haunted.                  │
│     pick a kind                 │<--- │   Grumpy Barista Bot                   │     │                                                                       │
│   Import…                       │     │   3w · comedy · A coffee-machine AI t… │     │ PROMPTING                                                             │
│     any Roleplay file           │     │   Hana Kobayashi                       │     │ System prompt  You are Captain Isolde Varga. Speak economically,      │
│                                 │     │   1mo · A ceramicist in Kyoto who tea… │     │                ask pointed questions, and never discuss the Hadley    │
┄┄┄ rows 29-42 elided: card body continues in its own scroll ┄┄┄                                                                                                
│                                 │     │   Quartermaster Bell                   │     │                                                                       │
╰─────────────────────────────────╯     ╰────────────────────────────── 4 of 28 ─╯     ╰────────────────────────────────────────────────────────────── ▾ more ─╯
  enter go to card | o chat now | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | s sort | F6 next pane | F1 · Ctrl+P · Ctrl+Q                 
```

---

## 2. Geometry and persistence

### 2.1 Shared code (D4 = A)

| Responsibility | Module after B0 | Notes |
|---|---|---|
| Pure resolver, profile, preferences, effective layout, normaliser | `ARS`: **same file, behaviour byte-identical** | Adds neutral aliases only: `AdaptivePaneProfile`, `AdaptivePanePreferences`, `AdaptivePaneLayout`, `resolve_adaptive_pane_layout`, read-only `nav_open` / `nav_width`. A new `Utils` module would add one module to the boot and UI-ready censuses (1,032 / 1,033, `Tests/Performance/test_ui_ready_module_census.py:162`). |
| Grip, shell, messages, stage bar, rail-row primitives, compact search input | **new** `tldw_chatbook/Widgets/adaptive_pane_shell.py` | Holds `AdaptivePaneShell`, `AdaptivePaneGrip` (`painted_names` mapping plus an equality-guarded `sync_label()`, because Roleplay's list grip renames with the kind), `PaneToggleRequested`, `AdaptivePaneShellResized`, `PaneVisibilityChanged`, `StageReturnBar`, `DestinationRailRow` + `fit_rail_row_label` + `DestinationRailRowButton`, and the promoted `SelectAllOnFocusingClickInput` with its `/` re-arm (from `Widgets/Library/library_rail.py:380-509`; `swallow_slash_on_focus=False`, because names contain `/`). Imported lazily by the two screens. **Never** placed in the UI-ready-resident `Widgets/destination_rail.py` (`ui_ready_modules.txt:967`). |
| Library compatibility | `Widgets/Library/library_adaptive_reader_shell.py` becomes a thin subclass + aliases | The aliases are the **same class objects** (`LibraryPaneVisibilityChanged is PaneVisibilityChanged`). All six Library handlers bind with `@on(...)` (`LS:7329, 7615, 7845`), so nothing re-routes. `library_emergency_return.py` subclasses `StageReturnBar`; `library_rail.py` re-exports the input. |
| Roleplay orchestration | **new** `PM/roleplay_frame_controller.py` | Layout controller (resolver → `sync_layout`, the work-session reducer, persistence) and view controller (chips, body and Try classes, entries split) as two classes in one module (§5.11). |

- **Roleplay never imports from `Widgets/Library/`.** Its package `__init__` (209 lines) eagerly imports every Library widget, and Library's route payload is 176 modules / 125k LOC.
- **Recorded coupling.** With custom widths off, the resolver projects the rail from the Library rail policy (`project_default_library_width`, `Utils/library_rail_width.py:46-62`, called at `ARS:275-279`). Roleplay **deliberately** shares that policy: one house rail width, and "Nav" means the same thing in both destinations. The ADR records this, and B7's golden test pins Roleplay's resolved table, so a Library rail-policy change fails a Roleplay test and gets reviewed for both.
- **Stays Library-local:**
  - the ordinary-route rail contract and its 3-cell handle;
  - `LibraryRail` and `LibraryShellState`;
  - all `LS` glue: emergency receipt and restore (`:5674-5870`), the pane-toggle if-ladder (`:7615-7700`), static F6 tuples (`:1402-1515`), the 15-entry Escape chain (`:1043-1129`);
  - `[library.reader]` and its Settings rows.

  Report §10 #6-#7: copy the contracts, not the code.

**Fixes:** enabler for RP-017, RP-024, RP-032, RP-066. Side benefit: Library grips stop depending on visit order, because the neutral `.adaptive-pane-*` rules move to a boot component (§2.12).

### 2.2 Width model

| Constant | Value | Source / reason |
|---|---|---|
| Grip | 5 cells each; both always reserved at T ≥ 64 | §1.2 |
| Rail | Library policy `floor((3T+8)/16)+5`, clamped 29-39; 24 floor under explicit priority | policy 29 at 120 (resolved 28: the BROWSE list 32 + work floor 50 + grips 10 leave 28), 31 at 140, 35 at 160, 39 from 180 |
| List | automatic **42** (min 32, comfort 42, max 56); built by the controller, **never routed through the normaliser** (pinned) | 1-line rows below 40 columns put all 28 characters on screen at 120 |
| Work floor, BROWSE | **50** | Keeps three panes open from T = 116 (reopen at exactly 120 when growing) |
| Work floor, EDIT | **100** (content 96) | One field at a ~96-character measure; the entries split threshold |
| ENTRIES | EDIT floor, resolved with the list as priority | The books list stays (book hunting is frequent, C.9c) |
| +TRY (Edit, Look) | work floor **147** | Smallest `w` with `w − try_w − 1 ≥ 90`, where `try_w = clamp(round(0.36·w), 56, 72)` |
| +TRY (Entries) | work floor **158**, list priority | `w − try_w − 1 ≥ 100` |
| Pane chrome | work 4 (border 2, padding 2); rail and list 2 (border) | One framing layer |
| Body measure | prose lines ≤100 characters; landing 96 (`$ds-size-96`) | Report §10 #16 |
| Hysteresis | 4 cells (`LAYOUT_HYSTERESIS_WIDTH`, `ARS:37`) | Reopen only; resizes only |
| `list_first_when_empty` | **True** in every Roleplay profile | Bounded to < 64 by the resolver, so it does not change ≥64 behaviour (§2.8) |
| `reader_has_item` | **True at T ≥ 64** (no donation); below 64 it is true only on the work stage | §2.6, §2.8 |

### 2.3 Work sessions: the posture follows the session, never the view (R9)

Picking the posture from the active view would reopen the rail and list on every Edit ↔ Test chat switch, starve Test chat to 46 columns at 120x36, collapse the editor from 110 to 50 columns on every Ctrl+S, and close the rail on Card at 200+. Roleplay instead ports the Library's **work session** (`UI/Library_Modules/library_notes_work_session.py:48-70`, where view changes, item changes, Save and resize fall through without changing the phase) as a pure reducer in `PM/roleplay_frame_state.py`.

| Phase | Profile | Entered by | Left by |
|---|---|---|---|
| `INACTIVE` | BROWSE | arrival; the end events below | an **authoring intent**, listed below |
| `ACTIVE` (cast) | EDIT; +TRY when T ≥ 200 and the kind's `try_column` is on | an authoring intent on a character or persona | the end events |
| `ACTIVE` (world info) | ENTRIES; +TRY-E when T ≥ 200 and the kind's `try_column` is on | an authoring intent on a lore book or dictionary | the end events |
| `ACTIVE + priority` | same profile with `priority=<pane>` | a **manual grip reopen** or `/` reopening the list during a session | the end events |

- **Authoring intents** (explicit keys and clicks only, never focus, I10):
  - `e`;
  - the Edit, Look, Tool policy or Options chip;
  - `New…`, `n` or `+ New`;
  - Enter or a click on an **entry** row, or `+ New entry`;
  - `t` or the Test chat / Try it chip;
  - `Run preview` from the title row.
- **No phase change on:**
  - any view switch (including Done, Cancel and Discard, which return to the default view);
  - **Save**;
  - loading another item of the same kind from the list or prev/next (Library precedent: `ITEM_CHANGED` falls through);
  - resize;
  - the Try column showing or hiding with width.
- **End events (→ `INACTIVE`, panes return per preference):**
  - the `‹ Kind` crumb;
  - Escape rung 7 ("esc to list", §4.5);
  - the Start row;
  - a kind switch;
  - deleting the loaded item;
  - leaving Roleplay;
  - a runtime source change.
- **Hide × / Keep beside ›** on the Try column re-resolve within the session (EDIT ↔ +TRY), because the user asked for it.

**Consequences, verified with the resolver:**
- Within one loaded item, read views (Card, Profile, Chats, World, Used by, Info, History) never start a session. Once a session is active, cycling **every** view at 120, 140, 160 and 174 changes **no pane width**. The geometry pin in §5.7 tests both.
- Save keeps the editor at 110 (120x36) and 108 (160x45). A Pilot test asserts it.
- Test chat at 120x36 runs at 110 columns (content 106), and Try it at 78. The Try view's 56-column minimum is met everywhere.
- At ≥196, EDIT and ENTRIES resolve exactly like BROWSE, so starting a session moves nothing unless the Try column applies (§2.7).
- Pane geometry changes at most twice per authoring task: at session start and at session end, both on explicit actions.

### 2.4 Resolved widths (resolver-verified; `rail | list | work`, "–" = closed to its grip; grips add 10)

| T | BROWSE (`INACTIVE`) | EDIT session | ENTRIES session | +TRY session (Edit, Look) | +TRY-E session (Entries) |
|---|---|---|---|---|---|
| 80 | – \| – \| 70 (list stage: – \| 32 \| 38) | – \| – \| 70 | – \| 32 \| 38 | – | – |
| 92 | – \| 32 \| 50 | – \| – \| 82 | – \| 32 \| 50 | – | – |
| 100 | – \| 40 \| 50 | – \| – \| 90 | – \| 32 \| 58 (drill-in) | – | – |
| 116 | 24 \| 32 \| 50 | – \| – \| 106 | – \| 32 \| 74 | – | – |
| **120** | **28 \| 32 \| 50** | **– \| – \| 110** | **– \| 32 \| 78** (drill-in) | – | – |
| 140 | 31 \| 42 \| 57 | – \| – \| 130 | – \| 32 \| 98 (drill-in) | – | – |
| 142 | 32 \| 42 \| 58 | – \| 32 \| 100 | – \| 32 \| 100 (split) | – | – |
| **160** | **35 \| 42 \| 73** | **– \| 42 \| 108** | **– \| 42 \| 108** (split) | – | – |
| 175 | 38 \| 42 \| 85 | 33 \| 32 \| 100 | 33 \| 32 \| 100 | – | – |
| 180 | 39 \| 42 \| 89 | 38 \| 32 \| 100 | 38 \| 32 \| 100 | – | – |
| 196 | 39 \| 42 \| 105 | 39 \| 42 \| 105 | 39 \| 42 \| 105 | – | – |
| **200** | **39 \| 42 \| 109** | 39 \| 42 \| 109 | 39 \| 42 \| 109 | – \| 42 \| 148 → body 91 · Try 56 | – \| 32 \| 158 → body 100 · Try 57 |
| **220** | **39 \| 42 \| 129** | 39 \| 42 \| 129 | 39 \| 42 \| 129 | – \| 42 \| 168 → body 107 · Try 60 | – \| 42 \| 168 → body 107 · Try 60 |
| 223 | 39 \| 42 \| 132 | 39 \| 42 \| 132 | 39 \| 42 \| 132 | 34 \| 32 \| 147 | – \| 42 \| 171 |
| 234 | 39 \| 42 \| 143 | 39 \| 42 \| 143 | 39 \| 42 \| 143 | 39 \| 38 \| 147 | 34 \| 32 \| 158 |
| 238 | 39 \| 42 \| 147 | 39 \| 42 \| 147 | 39 \| 42 \| 147 | 39 \| 42 \| 147 | 38 \| 32 \| 158 |

"body" = `w − try_w − 1` (it includes the pane's 4 chrome cells). Notes on the table:
- In an EDIT or ENTRIES session at 175-195, the list narrows to 32 (fit width) while the rail reopens.
- At 142-145 the EDIT list is 32, widening to 42 by 160.

**Thresholds** (fresh resolve / reopen when growing):

| Posture | Rail open | List open |
|---|---|---|
| BROWSE | ≥116 / ≥120 | ≥92 / ≥96 |
| EDIT | ≥175 / ≥180 | ≥142 / ≥146 |
| ENTRIES | ≥175 / ≥180 | always (list priority) |
| +TRY (Edit, Look) | ≥223 / ≥227 | ≥189 / ≥193 |
| +TRY-E (Entries) | ≥234 / ≥238 | always |
| Try column | shown at T ≥ 200 when the gate passes; hidden below 196 | |

**Manual reopen inside a session** (priority, kept until the session ends):
- the list at 120 → `– | 32 | 78`;
- the rail at 120 → `24 | – | 86`;
- the rail at 160 → `35 | – | 115`.

### 2.5 The trade against today (Q12)

| Size | Today (rail · centre · Inspector) | Browse here | Session here |
|---|---|---|---|
| 120 | 28 · **58** · 30 | 28 · 32 · **50** (−8) | editor **110** (+52) |
| 160 | 39 · **78** · 39 | 35 · 42 · **73** (−5) | editor **108** (+30) |
| 220 | 54 · **108** · 54 | 39 · 42 · **129** (+21) | 129, or body 107 + Try 60 |

- **Read-only views are narrower at the design centre.** Card and Profile get 69 content columns at 160 (a comfortable reading measure) and 46 at 120.
- **What that buys:** an always-visible rail with counts and glosses, a list that shows 28 / 18 characters instead of 5 / 9, the whole Inspector folded in, and editors at 110 / 108.
- **Alternatives:**
  - Raising the BROWSE floor to 58 closes the rail at 120, against C.5's "rail open at 120".
  - A Roleplay-owned narrower rail policy (28-33) returns 2 columns at 160.
- **Ruling:** the trade is accepted (Q12).
- **Interim slices are wider.** B6's interim fr layout (§5.2) keeps ≥58 / ≥78 / ≥108, so the final B7 browse figures (50 / 73) are a deliberate step down from B6 for read-only views. They are stated in B7's acceptance and screenshots.

### 2.6 No donation; nothing open

- **Never empty.** Roleplay's work pane is never as empty as the Library's: it always holds the landing, a kind empty state, or an item with its title row and primary action. So `reader_has_item=True` at T ≥ 64.
- **Nothing open.** "Nothing open in a kind" resolves with `priority="items"`, which seats the list at every width ≥64: `– | 32 | 38` at 80 and `– | 40 | 50` at 100.
- **Why not donate.** Uncapped donation would make a 119-column list beside a 50-column landing at 220.

### 2.7 The Try column at ≥200 (D2, R20)

- **Eligibility.** The column appears only **beside authoring views**:
  - character **Edit**;
  - persona **Edit** and **Look**;
  - lore/dictionary **Entries**.

  Card, Profile, Chats, World, Info, Used by, History and Options never show it. That keeps the 200+ browse layout identical to 160's (rail open) and keeps Roleplay a preparation surface (PRODUCT.md:33).
- **Conditions** (all must hold):
  1. T ≥ 200 (hidden below 196);
  2. a work session is active;
  3. the active view is eligible;
  4. `w − try_w − 1` ≥ 90 (Edit, Look) or ≥ 100 (Entries);
  5. the kind's `try_column` preference is on.
- **Posture.** With the preference on at T ≥ 200, the session resolves +TRY / +TRY-E for its whole length. In non-eligible views of that session the column slot is hidden and the body takes the full `w`, so **the rail never flaps between views**.
- **Rail cost** (accepted with Q2). Starting an authoring session with the column on closes the rail at 200-222 (Edit/Look) and 200-233 (Entries); at 223+ / 234+ the rail stays. The column is gone when the session ends, and so is the cost.
- **Per-kind preference** (`[roleplay.<kind>_reader] try_column`). Defaults (Q2): characters **on**, lore **on**, dictionaries **on**, personas **off**. The persona preview reads the *saved* record (`PM/personas_preview_controller.py`), so a side-by-side column would look live but test stale text until sub-project D makes it draft-aware.
- **Widths.** At 200: body 91 · Try 56 (Edit), body 100 · Try 57 (Entries; the list narrows to 32). At 220: body 107 · Try 60 for both.
- **Compose once** (ADR-115). The preview pane and both Try-it widgets live in `#personas-try-column` from compose time (§3.8.4). View versus column, and resizes across 196/200, change only `display` and widths.

**Fixes:** RP-010, RP-006, RP-081 (side by side at XL), RP-017 (XL width).

### 2.8 Responsive tiers

| Tier | Width | Browse | Session | Navigation and return |
|---|---|---|---|---|
| **XL** | ≥200 | rail 39 · list 42 · work 109-147 | as BROWSE; with the Try column: rail closes below 223/234, body 91-107 + Try 56-60 | All panes visible. F6: rail → list → body → Try column |
| **L/M** (design centre) | 120-199 | rail 28-39 · list 32-42 · work 50-105 | EDIT: rail closes below 175, list below 142 (editor 100-131). ENTRIES: list kept; split from 142 | Grips "Nav" and the kind |
| **S** | 92-119 | 116-119: rail 24-27 · list 32 · work 50; below 116: rail grip · list 32-42 · work 50-63 | EDIT: both closed (editor 82-109); ENTRIES: list 32, drill-in | A rail row picked from the Nav grip hands focus to the list |
| **XS** | 64-91 | **event-keyed stages** (R37). *List stage* (arrival, Esc rung 7, the crumb, `/`, a kind switch) resolves with items priority: list 32 + work 22-49. *Work stage* (Enter, `e`, `t`): work 54-81 + two grips | work only | Focus never changes geometry. Enter on a row moves focus to the work stage's F6 target |
| **XXS** (degrade only) | <64 | **No new single-stage code** (R38). Grips stay. *List stage*: `reader_has_item=False` triggers the resolver's list-first rule (`ARS`, task-32065): list 42 + a residual work strip whose body is hidden while narrower than 20 cells (display-only class `-residual`). *Work stage*: work 50 + two grips. The rail opens from its grip (`24 \| – \| 26` at 60) | as browse | The `‹ <Kind>` crumb and Esc rung 7 return to the list stage. The header subtitle names the item on the list stage. A true single stage (no grips; a rail stage with `‹ Roleplay`) is filed as a separate estimated follow-up. |

Two things hold at every tier ≥64: the header is one row, and the work pane's primary action is in work rows 1-2 (R21), so Chat now is never below a fold (RP-019).

**Fixes:** RP-066, RP-064, RP-017.

### 2.9 Hysteresis

1. A pane the resolver closed for width reopens only at threshold + 4.
2. Hysteresis applies to **resizes within a posture**. Posture changes (session start or end, Hide ×, Keep beside ›) resolve fresh, so the panes match the new posture at once.
3. An explicit grip activation bypasses hysteresis for that pane (resolver behaviour).
4. The Try column appears at T ≥ 200 and disappears below 196.
5. Narrowing from 200 to 195 while editing with the column hides it. Because the posture is still `ACTIVE` and 195 ≥ 180, the EDIT profile reopens the rail. That is one resolve and one `sync_layout`.
6. The 64 boundary has no hysteresis (the Library's tested contract; width-only changes are equality-guarded, `Tests/UI/test_library_shell.py:5264`).

### 2.10 Chrome arithmetic and what becomes visible

| Measure | 80x24 | 120x36 | 160x45 | 220x55 | Today (80 / 120 / 160 / 220) |
|---|---|---|---|---|---|
| First list item row | 7 | **7** | **7** | **7** | 21 / 23 / 23 / 23 |
| Content rows R | 17 | 29 | 38 | 48 | – |
| List rows (R − toolbar) | 16 | **28** | **37** | **47** | 1 / 10 / 19 / 29 |
| Characters visible | 16 (1-line) | **28 of 28** (1-line) | **18** (2-line) | **23** (2-line) | 1 / 5 / 9 / 14 |
| Work-pane header rows (R21) | 3 (`w` 70) | 3 browse (`w` 50) / 2 session (`w` 110) | 3 browse (`w` 73) / 2 session (`w` 108) | 2 | 13 rows of stacked chrome |
| Editor scroller rows (session: R − 2 header − rule − commit bar) | 12 | **25** | **34** | **44** | – |
| Editor fields visible (Jump row + Name + 4-line fields, 7 rows each) | ≈2 | **≈4** | **≈5** | **≈6.5** | – / 2.5 / 5 / – |
| Lore/dictionary entry rows (R − header − filter − count − column header) | – | **23** (drill-in, `w` 78) | **33** (split, `w` 108) | **43** (split) | – / 4 / 10 / 10 |
| Test chat transcript rows (session: R − header − 3 test-header rows − rule − status − composer) | 8 | **≥20** (`w` 110) | ≥29 | ≥39 | unusable / one exchange |
| Try-it results rows (session: R − header − title, summary, rule − rule + 3-row sample) | 7 | 19 | **29** (summary never clipped) | 38 (or a 60-column side column) | summary clipped at 160x45 |

The editor rows assume fields can grow, which is RP-002 (sub-project A) plus B11's growth caps.

**Fixes:** RP-007, RP-008, RP-016, RP-002 (rows available), RP-006, RP-010, RP-081, RP-056 (rows), RP-076 (width).

### 2.11 Persistence (D5 = B)

**Preferred versus effective.**

| Written to `config.toml` (preferred) | Never written (effective or session) |
|---|---|
| `nav_open`; per kind `items_open` and `try_column`; rail section states; `last_kind` and per-kind `sort` (the destination-owned carve-out below) | responsive collapse; work sessions and their priorities; the Try column's resolved width and visibility; XS/XXS stages; every resolved width; selection, cursor, search text, tag, anchors, marks, view and item memory |

**Schema** (the rail-section key is `create_import`):

```toml
[roleplay.reader]
# Preferred choices only. Responsive collapse, work sessions, the Try column's
# resolved state and the <64 stages are never written here.
# Environment overrides: TLDW_ROLEPLAY_READER_<KEY>.
nav_open = true
last_kind = "characters"        # destination-owned preference (carve-out)

[roleplay.reader.rail_sections]
cast = true
world_info = true
create_import = true

[roleplay.characters_reader]     # TLDW_ROLEPLAY_CHARACTERS_READER_<KEY>
items_open = true
try_column = true                # Q2 default
sort = "name_asc"                # destination-owned preference (carve-out)

[roleplay.personas_reader]
items_open = true
try_column = false               # Q2 default: the persona preview reads the saved record
sort = "name_asc"

[roleplay.lore_reader]
items_open = true
try_column = true
sort = "name_asc"

[roleplay.dictionaries_reader]
items_open = true
try_column = true
sort = "name_asc"
```

- **No custom-width keys in B.** ADR-086:29-30 says custom widths change through Settings, and B adds no Settings rows (Q9). The automatic list width (42) is never stored.
- **Carve-out.** `last_kind` and `sort` are destination-owned, non-geometry preferences. They are read and written by Roleplay's own adapter, **outside** the shared normaliser. ADR-084:79-80 keeps "active mode" transient for Library. The new ADR records Roleplay's carve-out (§5.9 G1c).
- **Where it is loaded.** `config.py`'s `_load_settings_uncached`, beside the Library block (`:2361-2470`). Booleans go through the existing coercion (`ARS:154-163`); enumerations must be known values or the default is used. Tables that are missing or are not mappings give defaults. Unknown keys are preserved by deep copy (`config.py:2393`). There is no new import on the config path (`Tests/Packaging/test_config_import_closure.py`).
- **Writes.**
  - Only explicit user changes write: a grip, Hide × / Keep beside ›, a rail-section toggle, a kind switch, a sort change.
  - Memory is updated first. `save_setting_to_cli_config("roleplay", …)` runs in a worker (`group="roleplay-pane-preferences"`), debounced and equality-guarded.
  - On failure: the toast "Pane preference could not be saved." and the mounted layout is kept.
  - Resizes never write.
- **No migration.** The section is new. The guide line "rail collapse lasts only for the current visit" is rewritten.
- **Session memory** (§4.11) lives in `save_state` (`PS:1748, 1768`), or on the instance if PERF-22 makes the route reusable (TASK-33281). It is never in `config.toml`.

**Fixes:** RP-031, RP-030 (with §4.11), RP-021.

### 2.12 CSS strategy

1. **Broad selectors (274/274, full; `Tests/Performance/test_textual_css_fastpath.py:326`).**
   - Every new rule's subject is an **id or a class, never a bare type**, in `_roleplay.tcss` **and** in widget `BUNDLED_CSS`. `build_css.py` auto-scopes `BUNDLED_CSS` with the class name (`:1051, 1152`), so a type subject there becomes a counted broad selector.
   - Slices that touch counted rules re-key them to class subjects and **lower `MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` to the measured count in the same PR**:
     - B2: about 6 rules in `PersonasLibraryPane`, such as `#personas-library-rows ListItem Static` and the pagebar, toolbar and filterbar `Button` rules;
     - B4: `PersonasPreviewPane .ds-toolbar Button`, `css/widget_defaults_self.tcss:2973`;
     - B5a: `PS:1291, 1304`;
     - B5b: the Inspector's 3 rules, deleted.
   - Up to **12** rules can be banked (274 → ≈262).
2. **Neutral shell rules in one boot component.**
   - New `css/components/_adaptive_panes.tcss`: `.adaptive-pane-shell/-nav/-items/-work/-grip` with `:hover`, `.-active` and `:focus` (`reverse`, task-32053), plus `-residual`.
   - They must be in boot, because lazy sheets cannot share selectors (`build_css.py:438, 894-906`). The same block leaves `css/features/_library.tcss:109-176`.
   - No width rules: widths stay inline from the resolver (`# ds-runtime:` marker).
3. **The Roleplay lazy sheet, created in B1** (R18). New `css/features/_roleplay.tcss` → `screen_feature_roleplay.tcss` through a `ScreenOwnedSplit` (`build_css.py:440-500` pattern), loaded from `PersonasScreen.CSS_PATH`.
   - **All new layout and visual rules go here.** Widget `BUNDLED_CSS` holds only harness-fallback minimums with class or id subjects, because `BUNDLED_CSS` is lifted into the boot bundle (`boot_css_bytes.json:249-261`).
   - **Zero numeric dimension literals** (`Tests/…/test_component_pattern_governance.py:22-29, 266-286`): use catalog classes (`h-full`, `w-fill`, `p-0`, `border-none`) or `$ds-roleplay-*` tokens in `css/core/_variables.tcss`. That includes B11's field caps (ADR-161 decision 4).
4. **Shrink the Roleplay boot segments.**
   - `PersonasScreen.BUNDLED_CSS` (2,996 B, `boot_css_bytes.json:145`): delete the mode-strip and compact rules (`PS:1171-1197, 1232-1252`).
   - In `AT`, **remove only the `#personas-*` selector items** from the grouped rules shared by 10 destinations (`AT:636-700`). Delete the personas-only blocks (`AT:709-757`: the Inspector's in B5b, the rest in B7) and the never-composed ids (`AT:357, 408-409, 664-665, 689-690, 804-805`; B1). `#personas-inspector-pane` (`AT:410, 666, 681, 691`) is composed today (`PS:1718`) and leaves with the Inspector in B5b.
   - Boot bytes fall in B5b (−1,898 B Inspector segment, minus the new widgets' boot segments, which go in the ledger) and in B7. The ceiling (608,090 B, `Tests/Performance/test_boot_css_byte_budget.py:117`) never rises. Snapshots are refreshed only through `scripts/update_boot_budget_snapshots.py`.
5. **Tokens and patterns.**
   - `$ds-*` only (ADR-150).
   - The `panes` family is registered in `css/patterns.json` **and** `backlog/docs/component-patterns.md` **and** `Widgets/pattern_gallery.py` (`test_component_pattern_governance.py:15-17`).
6. **Generated files** are never hand-edited. Rebuild, then `./scripts/preflight.sh`. PR #2862 also regenerates every sheet: rebase and regenerate, never hand-merge.

### 2.13 Performance guards

| Guard | Requirement | How it is verified |
|---|---|---|
| ADR-011 responsiveness gate (`011-chatbook-workbench-ui-system.md:44`) | No increase in stalls, worker backlog, timer leaks or mount churn when a legacy path is retired or a worker or timer is added | `UIResponsivenessMonitor` (`Utils/ui_responsiveness.py`) captured **before and after** in **every** slice that retires a legacy path or adds a worker or timer: B2, B3, B4, B5a, B5b, B6, B7, B9, B10 (§5.4) |
| Display-only layout | Pane toggles, session start/end, view switches, the Try mode ↔ column switch and tier changes mount and remove **0** widgets; `sync_layout` is equality-guarded; no `recompose=True` (`Tests/UI/test_screen_footer_hints.py:599`) | Mount/remove counters around each transition; `#personas-preview-pane` identity stable across 199↔200 and view ↔ column |
| ADR-115 | The four demand bodies are composed once in their final container; a session never mounts anything; F6 never mounts | `Tests/UI/test_personas_deferred_center_views.py` (six-point contract, no stall >250 ms) |
| Resize does no data work | No DB, config, worker or timer on a width-only change | Modelled on `Tests/UI/test_library_shell.py:5264` |
| Persistence off the loop | Explicit changes only; a resize sweep writes nothing; a grip writes once | Counters |
| Boot / UI-ready census | +0 on the config path | `test_ui_ready_module_census.py`, `test_app_import_weight.py`, `test_config_import_closure.py` |
| PERF-22 (TASK-33281) | No per-row widgets; no per-visit config disk reads; arrival within today's 0.35 s header / ≈1 s list at 348 | §5.7 harness at volume |
| Lazy CSS | `screen_feature_roleplay.tcss` parsed once on the first Ctrl+4, inside the visit budget | The arrival measurement |

---

## 3. Work pane

Every work-pane threshold uses the pane's own width `w` (content = `w − 4`), never the terminal width. The one exception is the Try column's T ≥ 200 gate (D2).

### 3.1 Layout (R21)

```text
╭──────────────────────────────── work pane (w columns) ───────────────────────────────╮
│ ‹ Kind  Item name  state            [Primary] readiness   More actions ▸   (w ≥ 100) │ row 1
│ ▸View1  View2  View3 n  View4  Test chat  Info                                       │ row 2: view strip
│ (More actions region: 0 rows closed, 2 rows open)                                    │
│ body: the active view's content, exactly one scroll owner                            │ 1fr
│ ──────────────────────────────────────────────────────────────────────────────────── │
│ [Save …]  Cancel                                        2 unsaved: Description, …    │ commit bar (draft views only)
╰───────────────────────────────────────────────────────────────────────────── ▾ more ─╯
   when w < 100: row 1 = crumb · name · state; row 2 = [Primary] readiness … More ▸; row 3 = strip
```

- **Header rows.** 2 at `w` ≥ 100 (if the joined row fits), 3 below. The primary is always in work rows 1-2 and above the strip. Tab order inside the work pane: primary → More actions → strip → body → commit bar.
- **Body.** `#personas-detail-stack` stops being a `VerticalScroll` and becomes a plain `Vertical`, so each view body owns its scroll (RP-054). Body scrollers are `can_focus=False` (no invisible Tab stops, RP-027). World gets its own `VerticalScroll`.
- **Fold cue.** Bordered panes carry `▾ more` in the bottom border (zero rows, R34).
- **Commit bar.** Only in views that hold a draft: character Edit, persona Edit (and Look or Tool policy while the profile fields are dirty), lore/dictionary Options, and the entry editor. It is each editor's existing anchored toolbar, restyled, with the **same ids**: `#personas-char-editor-save`/`-cancel` (`PW/personas_character_editor_widget.py:700-711`), `#personas-editor-save`/`-cancel` (`PW/persona_profile_editor_widget.py:172-175`), the entry button rows, and the moved `#personas-{dict,lore}-settings-save`.
  - A validation line sits above the bar only when it has text.
  - It is not merged into the title row: Ctrl+S presses the **visible** Save (`PS:16295-16312`). `editor_save` returns only a button that is displayed together with every ancestor up to its editor region (§4.6), because `Button.press()` checks only the button's own `display` and `disabled` (`textual/widgets/_button.py:429`). There are 22 test refs to the two Save ids. Ctrl+S mirrors the commit bar in every Roleplay editor (R41, §4.6).

**Fixes:** RP-052, RP-018, RP-020, RP-019, RP-054, RP-038, RP-007 (work-pane share).

### 3.2 Title row and readiness

- **Left side:**
  - the crumb `‹ Characters` (`#personas-work-crumb`): moves focus to the list at the loaded row and **ends the work session**;
  - the loaded item's name (`#personas-selected-name`, kept; middle ellipsis plus tooltip; always the *loaded* record, ADR-084);
  - the state word: `new` · `saved 2d` · `unsaved` · `saving…` · `saved just now` · `read-only · server`.
  - The source is **not** repeated; the header chip says `Local`.
- **Right side (the action bar, `#personas-work-actions`):**
  - the primary, which keeps the `console-action-primary` class;
  - the readiness word, muted;
  - `More actions ▸` (`#personas-more-actions-toggle`).
- **Fit order** when the row is too narrow:
  1. `More actions ▸` → `More ▸`;
  2. **wrap the action bar to its own row**;
  3. shorten the crumb to `‹`;
  4. middle-ellipsize the name;
  5. truncate the readiness word with `…`, **only while the primary is enabled** (a model name).

  The primary, the state word and a *disabled* reason are never cut.
- **Readiness word (R8).**
  - Enabled: the model it will use, for example `gpt-4.1-mini`. The tooltip and Info give the sentence: "Opens a new Console chat as Captain Isolde Varga · OpenAI gpt-4.1-mini".
  - Disabled: the reason, for example `save first` · `blocked: no API key` · `read-only · server` · `add an entry first`.
  - This retires "Ready to chat in Console." and the Readiness header (RP-067).
  - A destination-wide block also shows in the header chip (§1.3).

**Fixes:** RP-052, RP-019, RP-067, RP-038, RP-094 (title).

### 3.3 View strip (R22)

- **Widget.** One `DestinationModeStrip` (`Widgets/destination_workbench.py:69`), `#personas-work-modes`, holding at most six chips `#personas-work-mode-<slot>`. Chips are relabelled and shown or hidden per kind, never recomposed.
- **Roving focus.** Only the active chip has `can_focus=True`, and ←/→ move along the strip, so the strip is one Tab stop.
- **Chip states.**
  - Active: a named active-marker glyph (`▸`, ASCII `>`; R3) plus bold underline, whether or not the strip is focused.
  - Focused: the strip's focus edge plus `$ds-focus-bg` on the active chip. A focused inactive chip never looks active (RP-063).
  - Counts are inline: `Chats 2`, `World 1`, `Entries 250`, `Used by 1`, `History 4`, `Tool policy 2`.
  - Validation: `Info (1 warning)`.
  - **No unsaved marker on chips** (R24).
  - A chip that cannot apply is disabled, with the reason in its tooltip and, on selection, in the body.
- **Fitting** (pure `fit_view_strip(chips, width)`, beside `fit_rail_row_label`; R22):
  1. full labels with counts (chip = label + 2 cells of padding);
  2. drop the counts except warnings;
  3. tight chips (class `-tight`: label + 1);
  4. short labels: `Test chat`→`Test`, `Tool policy`→`Policy`, `Info (1 warning)`→`Info !`, `Used by`→`Used`, `Try it`→`Try`, `Test chat (beside)`→`Test ⇥`;
  5. only if step 4 still overflows: keep the active chip and fold the trailing chips into `More ▾`.

  Measured against content widths:

  | Strip | Full width | At content 46 (120 browse) | At content 69 (160 browse) | At content 74 (`w` 78) |
  |---|---|---|---|---|
  | Character | 48 | 44 (step 2) | full | full |
  | Persona | 54 | 46 (step 3) | full | full |
  | Lore | 48 | 42 (step 2) | full | full |
  | Dictionary | 71 | 41 (step 4) | 63 (step 2) | full |

  The character strip fits at content 34 (80x24) at step 4 (33). **No chip is hidden in any BROWSE layout ≥92 (content ≥46), in any work session at ≥92, or on the XS work stage (content ≥50).** On the XS list stage (work 22-49) the strip may fold trailing chips into `More ▾` (step 5); at 80x24 this applies to dictionaries, and below ≈79 columns to every kind. Pilot tests pin all four kinds at content 46, 69 and 74; the character strip at 34; and step 5 for the dictionary strip at 34.
- **Not the rail's kind rows** (`#personas-mode-{kind}`, R15).

**Fixes:** RP-063, RP-016 (view grammar), RP-015 and RP-044 (Info chip).

### 3.4 Work-pane thresholds (all keyed to `w`, 4-column reopen hysteresis)

| Element | Rule |
|---|---|
| Action bar on row 1 | `w` ≥ 100 and the joined row fits; otherwise row 2 |
| Card/Profile portrait | `w` ≥ 84: 24×10 portrait beside the fields. Below: a **12×6 thumbnail** at the top-left of the body beside the at-a-glance block (Tags, Greetings, Voice, World, Chats, Saved); never taller than 6 rows at any terminal height |
| Card label column | `w` ≥ 60: labels in a fixed muted column (14). Below: labels above values |
| Editor short fields in pairs (Creator \| Version, Position \| Order) | `w` ≥ 100 |
| Entries list and editor side by side | `w` ≥ 100: list 46 (about 45%, 44-60), editor ≥ 55. Below: drill-in |
| Chats list and transcript side by side | `w` ≥ 100: list 36-40, transcript fills. Below: drill-in (one number to learn: 100) |
| Landing Start buttons | `w` ≥ 72: the four New buttons in one row. Below: 2 × 2 |
| Test-chat header | `w` ≥ 60: one line. Below: two lines |
| Try column | T ≥ 200 + session + eligible view + `w − try_w − 1` ≥ 90 / 100 + preference on (§2.7) |
| Landing and prose measure | ≤ 96 / ≤ 100 columns |

### 3.5 Views per kind

**Notation:** *eager* = composed at screen build; *demand* = mounted on first use through `_ensure_center_view` (`PS:1997`, ADR-115). A view never mounts a body it does not show.

#### 3.5.1 Character: Card · Edit · Chats (n) · World (n) · Test chat · Info

| View | Body (ids kept) | Contents | Fixes |
|---|---|---|---|
| **Card** (default; or the last *read* view used for characters; never straight into Edit) | `#ccp-character-card-view`, eager | **Contents:** fields in the **editor's order and sections** (Basics · Prompting · Greetings · Voice & look · Details), labels in a fixed muted column, long values clamped to 6 lines with `Show all ▾`. The portrait `#personas-inspector-avatar-thumb` moves here (5 refs; keep `_THUMB_BOX_*` and `AVATAR_THUMB_*` in sync, `PW/personas_inspector_pane.py:58-59`). **Removed:** the card-bottom Edit (`#personas-card-edit-character`, 7 refs) and "Conversations (N)" (`#personas-card-conversations`, 2 refs) buttons. | RP-052, RP-053 (order), RP-019, RP-049 (dead button), RP-054 |
| **Edit** | `#personas-character-editor-slot` → `#ccp-character-editor-view`, demand | §3.11, plus the commit bar. An authoring intent: it starts the session. | RP-016, RP-017, RP-038, RP-085 |
| **Chats (n)** | new `PersonasCharacterChatsPanel` (`#personas-character-chats`), eager (ADR-115 keeps conversation surfaces eager), in `PW/personas_conversation_transcript_widget.py` | **Moved in:** `#personas-conversations-header`, `#personas-conversations-search`, `#personas-conversations-list` (53 refs; the deep-link return target, `character_conversation_navigation.py:32, 80`, switches to Chats **before** focusing it), the transcript `#personas-conversation-transcript-view`, and `#personas-conversation-actions` as a **one-row toolbar** `‹ Chats` · `[ Continue in Console ]` · `Open in Library`. **List:** full height, no 10-row cap, rows `title · 12 messages · 16m · source`, tail paging kept (ADR-120). Split at `w` ≥ 100. "Send transcript to Console draft" is hidden (RP-047). | RP-049, RP-050, RP-018 |
| **World (n)** | `#personas-character-attachments` (`PS:1642-1644`), eager, wrapped in its own `VerticalScroll` | The character's copies with honest copy (§3.13); attach and detach buttons, ids kept. | RP-012 (honesty), RP-054 |
| **Test chat** | the Try container in its view layout (§3.8), eager | An authoring intent. | RP-010, RP-045 (partly) |
| **Info** | new `PersonasInfoPanel` (`#personas-info-panel`), eager, Statics only (snapshots, ADR-011) | The readiness sentence; validation only when there are findings (`#personas-validation-summary` moves here; the "OK" copy is retired); source, id and version; created and modified; avatar; voice profile; Actor Pack eligibility and the reason. | RP-044 (slot), RP-067 |

#### 3.5.2 Persona: Profile · Edit · Look · Tool policy (n) · Test chat · Info

Checked against `PW/persona_profile_editor_widget.py:103-176`, which is one scroll.

| View | Body | Contents | Fixes |
|---|---|---|---|
| **Profile** (default) | `#ccp-persona-card-view`, eager | **Copy:** the gloss "A reusable AI role: in a Console chat the assistant speaks as <name>." **Fields:** System prompt, Description, Personality. **Links:** "Portrait: from character Wren [Look ›]", the existing link made visible (RP-090, visibility only). **Summaries:** the tool-policy summary (`#personas-policy-rules-summary`, moved from the Inspector) and Enabled · source. Whether each field is "sent" is left to sub-project D (RP-077, D15). | RP-018, RP-090 (partly), RP-011 (card line) |
| **Edit** | `#personas-persona-editor-slot` → `#ccp-persona-editor-view`, demand, class `-show-profile` | Name, System prompt, Description, Personality traits, Mode, Enabled (plus the portrait picker on create); commit bar `[ Save persona ]`. | RP-076 (caps), RP-085 |
| **Look** | the same editor body, `-show-look` | The shared visual identity host and `PersonasPersonaVisualPackWidget`, each with its own existing save controls. The persona commit bar appears only if the profile fields are dirty, and then says so. PR #2885's "Buddy workbench…" action lands here unchanged. | RP-089 (container) |
| **Tool policy (n)** | the same editor body, `-show-policy` | `PersonasPolicyRulesEditor` at full height. The chip is disabled with "Save the persona first" or "Server personas are read-only here" when `_sync_runtime_source_controls` hides the editor. | RP-073 (home; the fix is A's), RP-087 (container) |
| **Test chat** | the Try container | Header: "Uses the saved persona; save to test your edits" (true today). | RP-010 |
| **Info** | `PersonasInfoPanel`, persona variant | Source, version, portable UUID, portrait link, Actor Pack eligibility, validation. | RP-044 |

- **Sections.** `PersonaProfileEditorWidget` gains `#personas-editor-group-profile/-look/-policy` and `show_section(name)`. There is one widget, one draft and one demand mount; moving between Edit, Look and Tool policy never unmounts anything.
- **What personas do not get.** No Chats view (ADR-004/120). No Duplicate (R14).

#### 3.5.3 Lore book: Entries (n) · Options · Used by (n) · Try it · Info

The strip sets `TabbedContent.active` on `#personas-lore-tabs`. The tab row is hidden with `#personas-lore-tabs > ContentTabs { display: none; }`; `ContentTabs` is not a counted type key, and the prototype showed that `.active` still switches. Tab ids are kept (16 refs for dictionaries and 6 for lore). Hiding the row also removes its Tab stops (RP-027).

| View | Body | Contents | Fixes |
|---|---|---|---|
| **Entries (n)** (default) | `personas-lore-tab-entries` (lore detail is one of ADR-115's four demand bodies) | §3.9 | RP-006, RP-056, RP-091, RP-003 (re-host) |
| **Options** | `personas-lore-tab-settings` | A labelled form. `[ Save options ]` (`#personas-lore-settings-save`) moves into an anchored commit bar inside the detail widget. Export moves to More actions. **Now an ADR-046 draft domain** (R30), with a new `LoreSettingsEdited` message so the guard sees it. | RP-005, RP-020 |
| **Used by (n)** | `personas-lore-tab-attachments`, relabelled | **Linked chats:** the table, plus `Link to chat…` and `Unlink…` (relabelled `#personas-lore-attach-add` / `-detach`). **Character copies:** a block with honest copy (§3.13). | RP-012 (partly) |
| **Try it** | the Try container with `#personas-lore-tryit` | §3.8 | RP-006, RP-081 |
| **Info** | new `personas-lore-tab-info` | Entries total · off · with secondary keys; version and modified (`DB/ChaChaNotes_DB.py:1808-1821`); id; a provenance line once sub-project C decides RP-013. There is no History view: lore has no version table. | RP-044 |

#### 3.5.4 Chat dictionary: Entries (n) · Options · Used by (n) · Try it · History (n) · Info

Checked against `PW/personas_dictionary_detail.py:201-324` (five tabs today); mapped the same way as lore.

| View | Body | Notes | Fixes |
|---|---|---|---|
| **Entries (n)** (default) | `personas-dict-tab-entries` | §3.9. The validation `OptionList` (`#personas-dict-validation`) moves inline, under the Pattern field of the rule being edited. | RP-006, RP-056, RP-015 |
| **Options** | `personas-dict-tab-settings` | Labelled strategy names; `[ Save options ]` anchored; Export JSON/MD to More actions. It already feeds `has_unsaved_changes` (`PS:5630-5640`) and joins the aggregate (R30). | RP-005, RP-020 |
| **Used by (n)** | `personas-dict-tab-attachments` | As for lore. | RP-012 |
| **Try it** | the Try container with `#personas-dict-tryit` | §3.8 | RP-006, RP-081 |
| **History (n)** | `personas-dict-tab-versions` | View and a confirmed `Revert…`, kept (preserve). **No `del`** (R31). | preserve |
| **Info** | `personas-dict-tab-stats` (renamed in the strip only) | The stats body (`load_statistics`, `:419-444`), the validation summary ("1 warning: rule #9 is not a valid regex [ Open rule ]"), source and version. | RP-015, RP-044 |

### 3.6 Action bar and More actions

| Kind | Primary (id) | Enabled when | Disabled reason (readiness word, tooltip, Info) |
|---|---|---|---|
| Character | **Chat now** (`#personas-start-chat`, 30 refs) | saved and the provider ready (the gate at `PW/personas_inspector_pane.py:964-1094`, moved unchanged) | `save first` · `blocked: no API key` · `read-only · server` |
| Persona | **Chat now** | the same | the same |
| Lore book / Chat dictionary | **Run preview** (new `#personas-work-run-preview`) | saved, and ≥1 entry | `add an entry first` |

- **Chat now (D13, R29).**
  - **B5b** moves the button, gate and handler **unchanged**: per-intent gating (task-523), F-037 disabled reasons, `console-action-primary`.
  - **B5c** promotes it: the `o` key and the end-to-end provider test. It ships only when TASK-33621.2 is on dev and the test passes.
- **Run preview** from another view switches to Try it (an authoring intent) and runs the remembered sample; with an empty sample it focuses the sample. In Try it the slot is hidden: the docked `[ Run ]` beside the sample (`#personas-dict-tryit-run` / `#personas-lore-tryit-run`, ids kept) is the primary there.
- **More actions ▸.** An inline region, `#personas-more-actions`, directly under the strip (the Library's `#library-prompt-more-actions-region` pattern, `library_prompts_canvas.py:1696-1720`).
  - It costs 0 rows closed and 2 open.
  - It closes on Escape (rung 1), a selection change or a view change.
  - It is a region in document order, not a popup.
  - A one-line status names why things are disabled ("Exports read the last saved state; save first.").

| Row | Character | Persona | Lore book | Chat dictionary |
|---|---|---|---|---|
| **Export** | JSON (`#personas-export-json`) · PNG card (`#personas-export-png`) · Actor Pack (`#personas-export-actor-pack`) · `[ ] include voice profile` (`#personas-export-include-tts`, only when assigned) | JSON · Actor Pack | JSON (`#personas-lore-export`, from Options) | JSON (`#personas-dict-export-json`) · Markdown (`#personas-dict-export-md`) |
| **Use / manage** | Add to Console message (`#personas-attach-to-console`, 36 refs) **hidden behind a code flag** · Duplicate (`#personas-library-duplicate`) · **Delete…** (`#personas-delete`) | Add to Console message (hidden) · **Delete…** | Link to chat… · Duplicate · **Delete…** | Link to chat… · Duplicate · **Delete…** |

- **Delete.** `Delete…` sits at the right end after a gap, with the danger token class (the ADR-161 pattern). It acts on the **open item only** and opens the one delete modal (R32).
- **Bulk.** Bulk Export and Delete live in the marks bar (§1.5.5). This removes RP-039's silent retargeting.
- **Hidden hand-offs.** "Add to Console message", its `u` key, the screen-level `ctrl+enter` (`PS:1097`) and the test chat's "Send to Console draft" (`#personas-preview-open-console`) are hidden together. ADR-031 rule 4 mandates it: an inert advertised action loses its binding and its hint. The ids stay, so the feature returns by flipping one constant once RP-047 is fixed.

**Fixes:** RP-018, RP-019, RP-020, RP-039, RP-043 (Actor Pack as an export format), RP-047 (hidden), RP-067.

### 3.7 Inspector fold

Source: `PW/personas_inspector_pane.py:212-363` (compose) and `:964-1159` (visibility). The module is `git mv`-ed to **`PW/personas_work_pane.py`** (history kept; census-neutral), which holds `PersonasWorkHeader`, `PersonasInfoPanel`, `PersonasLanding` and `PersonasKindEmptyState`.

| Inspector element (id) | Today | New home | Id | Refs / notes |
|---|---|---|---|---|
| Rail header "Inspector" + `>` (`#personas-inspector-rail-collapse`) | collapses the pane | removed | — | 5; re-pin `test_personas_workbench.py:586-652` |
| Handle (`#personas-inspector-rail-handle`, `-open`, `-badge`) | 11-column handle | removed (frees 11-39 columns) | — | `PERSONAS_INSPECTOR_RAIL_HANDLE_WIDTH` (`PS:609`) retired |
| "Selected: X" (`#personas-selected-name`) | summary line | title-row name | kept | 14; "Selected: " prefix dropped |
| "Type: character" (`#personas-selected-kind`) | summary line | removed; the crumb names the kind | — | 1 |
| Portrait (`#personas-inspector-avatar-thumb`) | 24×10, pushes Chat now down | Card body (§3.4) | kept | 5 |
| "Validation: OK" (`#personas-validation-summary`) | constant OK | Info (findings only); count on the Info chip; editor errors in the commit bar's validation line | kept | 10 |
| "Tool policy: …" (`#personas-policy-rules-summary`) | persona only | persona Profile; count on the Tool policy chip | kept | 2 |
| Conversations header/search/list | 10-row cap | Chats view | kept | 6 / 1 / 53 |
| Older/retry tail rows; render lock (`:571-952`) | tail ListItem | Chats list tail, moved unchanged | — | — |
| Readiness header (`#personas-readiness-header`) | "Readiness" | removed | — | 2 |
| Readiness line (`#personas-readiness-console`) | sentences | the readiness word; the sentence in Info | kept on the word | 21; copy re-pinned (C.7) |
| Action container (`#personas-inspector-actions`) | vertical stack | action bar **`#personas-work-actions`** | **renamed, a recorded exception** (container replaced; 8 refs re-pinned in B5b) | — |
| Chat now (`#personas-start-chat`) | primary in the stack | title-row primary | kept | 30 |
| Send to Console draft (`#personas-attach-to-console`) | secondary | More actions, "Add to Console message", hidden | kept | 36 |
| Include voice profile, Export JSON/PNG/Actor Pack | stack | More actions, Export row | kept | 3 / 18 / 13 / 4 |
| Delete (`#personas-delete`) | looked like an export button | More actions, danger, `Delete…` | kept | 20 (also serves dictionaries and lore) |
| Buddy ×4 | in every state | the rail Buddy row (B5b parks them in an interim `Buddy ▸` chooser; B6 moves it) | kept | 4 / 7 / 6 / 4 |
| "Console chat is for characters and personas." / no-selection guidance | guidance text | removed / landing and kind empty states | — | — |
| `set_card_actions_visible` | hid actions over the transcript | removed; the action bar stays visible in Chats | — | — |

**Method → new owner** (18 distinct methods, 40 `query_one(PersonasInspectorPane)` sites in `PS`, 8 in `PM/personas_conversations_controller.py:47, 118, 136, 230, 257, 478, 500, 715`):

| Method (calls in `PS`) | New owner | Slice |
|---|---|---|
| `show_conversations` (3), `show_conversations_loading`, `show_older_conversations_loading`, `append_conversations`, `set_conversation_total`, `set_conversation_snapshot`, `focus_conversation` | `PersonasCharacterChatsPanel` | B5a |
| `show_selection` (6), `clear_selection`, `set_unsaved` (9; now the aggregate predicate), `set_console_actions_enabled`, `set_tts_export_available`, `set_actor_pack_export_state` (4), `include_tts_profile_in_export` | `PersonasWorkHeader` (title row, action bar, More actions) | B5b |
| `show_validation` (6), `show_validation_editing` (5) | `PersonasInfoPanel` / the editor's commit-bar validation line | B5b |
| `set_avatar_thumbnail` (11) | Card body | B5b |
| `show_policy_rules` | persona Profile | B5b |
| `set_buddy_status` | the Buddy chooser (B5b), then the rail row (B6) | B5b → B6 |
| `set_marked_count` | the marks bar | B2 |

### 3.8 Test chat, Try it and the Try container

#### 3.8.1 Test chat as a view (characters, personas)

```text
TEST CHAT · nothing saved                                                     [ Hide × ]   ← column only
Uses your unsaved edits · OpenAI gpt-4.1-mini                             [ Configure ]
Greeting  ‹ 1/3 ›                                                              [ Reset ]
──────────────────────────────────────────────────────────────────────────────────────
transcript (1fr, its own scroll, auto-scrolls to the newest turn)
status line (one row: "Thinking…" / provider error + [ Configure ])
──────────────────────────────────────────────────────────────────────────────────────
▎Test message…                                                                 [ Send ]   ← docked
```

- **CSS** (in the lazy sheet, id-scoped):
  - `PersonasPreviewPane` `height: 1fr` (was `auto` / `max-height: 60%`, `:48-49`);
  - `#personas-preview-transcript` `1fr` (was `max-height: 10`);
  - `#personas-preview-body` is always displayed;
  - `#personas-preview-toggle` gets `display: none` (id kept; `expand()` stays as the entry hook).
- **Layout.**
  - The greeting `Select` becomes a one-row cycler bound to `PreviewGreetingSelected`.
  - "Test Reply" → **Send** (id `#personas-preview-test-reply` kept).
  - The composer is docked last (`#personas-preview-composer`).
- **Header copy is kind-aware and truthful.** Characters: "Uses your unsaved edits" (the preview is draft-aware, `PM/personas_preview_controller.py:456-494`). Personas: "Uses the saved persona; save to test your edits".
- **A test chat needs a loaded item.** The strip exists only for a loaded item (RP-045, partly).
- **Content work** outside B: speaker labels (RP-045) and the stale greeting (RP-046).

#### 3.8.2 Try it as a view (lore, dictionaries)

```text
TRY IT · injection preview · nothing saved
5 injected · 64 dropped by token budget (919/1000 tokens) · 3 disabled     ← summary first, always visible
──────────────────────────────────────────────────────────────────────────────────────
results (1fr, own scroll):
  Before character (2)   The Hadley beacon blinks red every 9 seconds; …   [▾]   ← 2-line collapse
  Fired (5) …            (dictionary: struck-through originals, underlined replacements)
  Dropped by token budget (64): sorted by priority, cut-off shown · budget full: next entry needs 41 tokens
  Disabled (3) · Secondary key missing (2)
──────────────────────────────────────────────────────────────────────────────────────
┌ sample (3-8 rows, grows) ─────────────────────────────────────────┐   [ Run ]
└───────────────────────────────────────────────────────────────────┘   enter runs · ctrl+j new line
```

- **Kept:** struck-through originals, underlined replacements, fired/skipped reasons, `_SKIP_REASON_COPY`, every result id.
- **Changed in B4:**
  - **Layout:** `max-height: 60%` (`PW/personas_lore_tryit.py:78`, `PW/personas_dictionary_tryit.py:42`) and the details' `max-height: 12` are removed; the results become the 1fr scroller.
  - **Removed:** the disabled "Include recent turns (soon)" switch (`PW/personas_lore_tryit.py:77-85`).
  - **RP-081's content half is in B4's scope:**
    - non-fired records are grouped (`Dropped by token budget (n)` sorted by priority with the cut-off shown, `Disabled (n)`, `Secondary key missing (n)`);
    - `near-miss` / `over budget` is replaced by `budget full: next entry needs N tokens`;
    - long injected text collapses to 2 lines, expandable.
- **Keys.** The sample becomes a `TextArea` subclass: **Enter runs**, Ctrl+J or Shift+Enter inserts a newline (the Console composer grammar, `Widgets/Console/console_composer_bar.py:4701-4708`), and Ctrl+Enter is an unadvertised alias. A priority binding on the whole Try-it container is **not** used, because it would capture Enter on its buttons.

#### 3.8.3 The column at ≥200

Eligibility, posture and rail cost are in §2.7.
- **The chip.** While the column is visible, the Test chat / Try it chip reads `Test chat (beside)` / `Try it (beside)`. Choosing it, or `t`, focuses the column's input instead of switching view.
- **Hide and show.** `[ Hide × ]` in the column header turns the kind's preference off. In the full view at ≥200, the header offers `[ Keep beside › ]` to turn it on.
- **Focus.** If the column hides while focused (resize, view change), focus moves to the body's first control (named target, §4.11).
- **F6.** The visible column is an F6 target after the body. It has **no grip**.

#### 3.8.4 Container strategy (ADR-115: compose once, never move)

```text
#personas-work-area            Vertical (existing id, 14 refs)
├─ #personas-work-header       PersonasWorkHeader     eager  title row + action bar
├─ #personas-work-modes        DestinationModeStrip   eager  ≤6 chips
├─ #personas-more-actions      Vertical               eager  display:none until opened
└─ #personas-work-body         Horizontal             eager  classes: -try-off | -try-mode | -try-column
   ├─ #personas-detail-stack   Vertical (was VerticalScroll)
   │   ├─ landing, kind empty states                  eager  (§3.10)
   │   ├─ #ccp-character-card-view                     eager
   │   ├─ #personas-character-editor-slot → editor     demand
   │   ├─ #personas-character-chats (+ transcript)     eager
   │   ├─ #personas-character-attachments (World)      eager
   │   ├─ #ccp-persona-card-view                       eager
   │   ├─ #personas-persona-editor-slot → editor       demand
   │   ├─ #personas-dictionary-detail-slot → detail    demand
   │   ├─ #personas-lore-detail-slot → detail          demand
   │   ├─ #personas-info-panel                         eager
   │   └─ #personas-character-link-recovery            eager (existing)
   └─ #personas-try-column      Vertical              eager  composed once, here, for good
       ├─ #personas-preview-pane   (moved here at compose time from PS:1620)
       ├─ #personas-dict-tryit     (from PS:1710-1715)
       └─ #personas-lore-tryit
```

| Body class | `#personas-detail-stack` | `#personas-try-column` | Used for |
|---|---|---|---|
| `-try-off` | 1fr | `display: none` | every view, wherever the column rule fails |
| `-try-mode` | `display: none` | 1fr | the Test chat / Try it view at any size |
| `-try-column` | 1fr | `width: try_w`, left rule | §2.7 |

- Exactly one Try widget is displayed, chosen by kind. The controller patches styles only when they change.
- Preview turns survive leave and return (`save_state`, `PS:1748-1768`). `test_personas_workbench.py:10183` still holds.
- A Textual 8.2.8 prototype confirmed that widget identity survives all three classes and that the docked composer sits on the last row.

**Fixes:** RP-010, RP-006 (Try it leaves the Entries tab), RP-081, D2.

### 3.9 Entries for lore books and chat dictionaries (D12)

#### 3.9.1 Compose (inside the existing Entries `TabPane`, composed once)

```text
personas-lore-tab-entries (TabPane)
└─ #personas-lore-entries-split      Horizontal   classes: -split | -list | -editor
   ├─ #personas-lore-entries-listcol Vertical
   │   ├─ row: [+ New entry]  #personas-lore-entry-filter (one-row Input)
   │   ├─ #personas-lore-entries-count  "#200 of 250 · showing 250 · 28 off"
   │   └─ #personas-lore-entries-table  DataTable, height 1fr (12-row cap removed)
   └─ #personas-lore-entry-editor    Vertical
       ├─ heading: "Entry #200 · Saltmarsh Bellwarden"   ‹ prev  next ›
       ├─ VerticalScroll: the labelled form (sub-project A's RP-003 labels); Content 1fr
       └─ entry commit row: [ Update entry ]  Revert   Move ↑ ↓   Delete…   (anchored)
```

The dictionary uses the same shape, under `#personas-dict-entries-*`; its user-facing noun is "rule", so the dictionary labels are `[+ New rule]`, `[ Update rule ]`, `[ Add rule ]` (ids unchanged).
- **Ids kept:** `#personas-lore-entries-table`, all `#personas-lore-entry-*` field ids, and the Add / Update / Delete / Move button ids.
- **Retired:** `#personas-lore-entries-scroll`.
- **Removed:** the `max-height: 12` rules (`PW/personas_lore_detail.py:100`, `PW/personas_dictionary_detail.py:147`). `DataTable` is already line-based.

#### 3.9.2 Split or drill-in

- **Split** (`w` ≥ 100): list 46 (about 45%, 44-60), editor ≥55. The editor always shows the anchored entry. **33 rows at 160x45 and 43 at 220x55** (today 10).
- **Drill-in** (`w` < 100):
  - Enter (or a click) on a row opens the editor full width, with the heading `‹ Entries · #200 of 250 · Saltmarsh Bellwarden` plus prev/next.
  - Escape first leaves the field (rung 3), then returns to **the same row** with the same scroll anchor (rung 4).
  - **23 rows at 120x36** (today 4).
- **Layout switches** only change the class, so a draft is never lost across `w` 99/100.

#### 3.9.3 Filter, count, columns

- **Filter.** One row: `Filter keys, content…` (lore: keys, secondary keys, content) or `Filter pattern, replacement…` (dictionaries). Debounced 0.2 s (`PERSONAS_SEARCH_DEBOUNCE_SECONDS`). Reached by `/` from the work pane (R26).
- **Count line.** `#200 of 250 · showing 3 of 250 · 28 off`. With nothing selected: `showing 250`. With no match: `No entries match "bellw" · esc clears the filter`.
- **Columns** (RP-091):
  - lore: an `on`/`off` word (3), the first key (18-20, `+2` when there are more keys), the first line of content (fill), Ord (4);
  - dictionaries: `on`, `text`/`regex`, Pattern (20), Replacement (fill), Pri (3).
  - Off rows say `off` as well as being dimmed.
  - A column-header row names the columns.

#### 3.9.4 Selection anchor (hook for sub-project A's RP-014)

```text
EntriesAnchor(entry_id: str | None, top_entry_id: str | None, filter_text: str)
detail.entries_anchor() -> EntriesAnchor
detail.load_entries(entries, *, anchor: EntriesAnchor | None) -> None   # re-select by id, restore the top row, re-apply the filter
```

- **Rule (A owns it; B enables it):** after every Add, Update, Delete, Move, reload or filter change, re-select by id. After Delete, select the next neighbour. After Add, select the new row and scroll it into view.
- **Memory.** One anchor per open book in item memory (§4.11).
- **Code it builds on:** `selected_entry_id` (`PW/personas_lore_detail.py:406-412`, `PW/personas_dictionary_detail.py:597-603`) and `move_cursor(row=ids.index(entry_id))` (`:759`).

#### 3.9.5 Entry drafts (ADR-046 domain, R30)

- **New entry.** `[+ New entry]` (or `n` on the entries list) clears the form, sets the heading to `New entry` and the commit row to `[ Add entry ]  Cancel`, and leaves no row selected. Lore Add never overwrites the loaded entry (preserve). The dictionary labels are `[+ New rule]`, `[ Update rule ]`, `[ Add rule ]` (§3.9.1).
- **Dirty entries.** The entry form is a draft while it differs from the loaded entry (`entry_form_dirty`, part of the aggregate snapshot). While dirty:
  - the title row says `unsaved`, and the entry commit row says `1 unsaved: Content`;
  - choosing another entry, prev/next or `+ New entry` asks inline: `Unsaved entry changes: [ Update ] [ Discard ] [ Stay ]`;
  - a kind, item or screen change goes through the aggregate dialog (§3.12).
- **Out of scope:** autosave (RP-040's alternative) is content work. Reorder speed (RP-096) is sub-project C's.

#### 3.9.6 New lore book / chat dictionary

`New…` → Lore book (or `n` in the Lore kind) opens the **Options view as an unsaved draft**:
- Name is focused, and the commit bar reads `[ Create lore book ]  Cancel`.
- This is an authoring intent, so the session starts.
- The draft is guarded (`lore_settings_dirty`).
- After Create, the book is selected and opens in Entries with `+ New entry` focused.

This replaces today's instant "Untitled world book" save (RP-040, books half; RP-041's default name).

**Fixes (§3.9):** RP-006, RP-056 (B's half), RP-091, RP-003 (re-host), RP-014 (hook), RP-015 (inline), RP-025 (the editor is an F6 target), RP-040 (entry draft, new-book draft), RP-005 (anchored Save).

### 3.10 Landing and kind empty states (D11 = B)

#### 3.10.1 When the landing shows (Q3)

- **On arrival with nothing to restore:** the first visit per launch, and any return whose saved state has no loaded item. **Restore always wins.**
- **The Start rail row:** guarded unload, ending the session.
- **Not on a kind switch:** the kind reopens its remembered item, or shows its empty state.

F-031's first-paint auto-select (`PS:1878-1924`) is retired. It opened "Ada" only because she sorts first. Re-pin `test_personas_workbench.py:12315-12471` after a census of tests that rely on it as an implicit fixture (comments at `:528, 712, 884`).

#### 3.10.2 Contents

There are four sections, capped at 96 columns, built from Statics and compact buttons (about 20 widgets, eager; listed in the ADR-115 note, §5.9 G6).

1. **Start** (live at compose; needs no data).
   - `[ Import… ]` is the primary (J1). Its hint lists the formats. It is the landing's first F6 stop.
   - `[New character] [New persona] [New lore book] [New chat dictionary]`: the four jobs as equals, running the same guarded path as `New…`. They sit on one row at `w` ≥ 72 and in a 2 × 2 grid below.
2. **Continue a character chat** (shown; owner ruling Q7, authorised by the ADR-120 amendment mirrored in ADR-004, G5, which lands with B10).
   - The 5 most recent *character-bound* local conversations, through the shared projection and behind the Data-Profile fence; rows `title · character · N msgs · age`.
   - Enter continues in Console through the canonical opener with typed results (`PM/personas_conversations_controller.py:1120-1226`). Rows whose identity does not resolve are shown non-activatable. The block is local only: it is absent in server runtime.
3. **Recently edited.** The last 4 items across all four kinds, with kind and age. Enter opens the item in its default view.
4. **Needs attention.** Only real, already-detectable conditions, each with one action:
   - a dictionary with invalid regex rules (`[ Open rule ]`, RP-015);
   - characters holding lore or dictionary copies while the D6 flag is false (`[ Why? ]` opens the World explanation).

   The section is hidden when it has no items.

B's landing has no Chat-now element; its primary is `Import…` (J1). D13 would gate any future one.

**Non-void first paint.** Start is live in the first frame. Sections 2-4 render `(…)` and fill from **one** off-thread worker after the list load (ADR-011 gate). Arrival focus stays on the list (R7).

#### 3.10.3 Kind empty states (`PersonasKindEmptyState`, takes over `#personas-characters-empty`, 11 refs)

| Situation | Copy (lore books shown) | Actions |
|---|---|---|
| The kind has items, nothing loaded | "**Lore books**: facts added to a chat when a keyword appears. Pick a book on the left." (The last sentence is omitted while a search shows zero rows.) Then the 3 most recently edited. | `[ New lore book ]` `[ Import… ]` |
| The kind has no items | "No lore books yet. A lore book adds world facts to a chat when a keyword appears. To use one in a chat: open it › Used by › Link to chat…" | `[ New lore book ]` `[ Import… ]` |

The list shows only `No lore books yet` (R25).

**Fixes:** RP-055, RP-021 (a return without a selection now teaches), RP-024 (landing half).

### 3.11 Editors inside the work pane

- **Field heights grow with content up to a cap and never change on focus** (DESIGN.md:121-128). They use `height: auto` with min and max set by **`$ds-roleplay-*` tokens** (zero literals in `_roleplay.tcss`).

  | Field | Min rows | Max rows |
  |---|---|---|
  | First message | 3 | 10 |
  | Description | 4 | 14 |
  | Personality | 2 | 6 |
  | System prompt | 3 | 12 |
  | Scenario, Post-history | 2 | 8 |
  | Creator notes | 2 | 6 |
  | Persona System prompt | 3 | 14 |
  | Persona Description, Personality traits | 2 | 8 |
  | Lore Content, dictionary Replacement | 5 | 1fr |

  Sub-project A's RP-002 fix sets the floors; B11 sets growth and caps.
- **Sections, Jump row, Advanced.** Headings are Static rows inside the editor body; every field id below them is unchanged.
  ```text
  Jump:  Basics · Prompting · Greetings · Voice & look · Details        [ Generate… ]
  BASICS        Name · First message · Description · Personality
  PROMPTING     System prompt · [Example dialogue: slot for RP-023] · Advanced ▸ (Scenario, Post-history)
  GREETINGS     Alternate greetings (table + editor, out of Advanced)
  VOICE & LOOK  Voice & Speech · Avatar · Expressions · Visual identity
  DETAILS       Advanced ▸ (Creator notes · Creator · Version · Tags)
  ```
  - The Jump buttons call `scroll_to_widget`. `[ Generate… ]` expands the existing concept / "Generate whole character…" row (`:407-462`) in place.
  - `#personas-char-editor-advanced-toggle` is kept; its container splits into the two disclosure spots with the same field ids.
  - **Generate-then-Accept and per-field Generate are unchanged** (preserve). The `✨` is dropped from the Generate labels (`PW/personas_character_editor_widget.py:614, 649, 686`; RP-071). Per-field Generate buttons come **after** their field in Tab order (RP-027).
- **Validation.** Editor errors map field ids to labels and render under the field, keeping the red outline. A test asserts that no `#personas-` / `personas-char-editor` string reaches user-facing text (closes TASK-1651, RP-044).
- **Unsaved and save feedback (R24).**
  - Signals: the header chip, the title-row `unsaved`, and the commit-bar summary `2 unsaved: Description, Personality` (field names when the editor can diff against the loaded record, else `Unsaved changes`).
  - After Save: the title row reads `saved just now`, and the commit bar shows `Saved 14:02` for 3 s while Cancel becomes **Done**. The session continues (§2.3).
  - **Ctrl+S** presses the editor's visible commit button in every editor (R41; §4.6 lists each button and the slice that binds it).
- **Opening an editor.** Entering Edit for a different item, or New: `scroll_home(animate=False)`, then focus Name after refresh (RP-085, both editors). Re-entering Edit for the same item restores its scroll offset and focus; the draft is still there.

**Fixes:** RP-016, RP-076 (caps), RP-085, RP-038, RP-053 (sections), RP-027, RP-071 (✨), RP-044, RP-023 (slot only).

### 3.12 ADR-046 draft veto: one predicate, one dialog, the complete trigger table

- **One predicate.** `roleplay_has_unsaved_work()` = `not _aggregate_roleplay_draft_snapshot().is_clean` (`PS:15979-16027`: form edits, the three visual authorings, attachments, in-flight saves), extended in B9 with `entry_form_dirty` and `lore_settings_dirty`. It drives:
  - `_run_guarded` (`PS:16179`, from B5b; today it checks only `has_unsaved_changes` plus the visual authorings, `:16179-16190`);
  - the header chip (B1);
  - the title-row word;
  - the commit-bar summary.
- **One dialog.** The `RoleplayDraftNavigationDialog` family (`UI/Navigation/character_conversation_navigation.py:233-283`) replaces `UnsavedChangesDialog`'s "close the tab" copy (`Widgets/confirmation_dialog.py:164-192`, used at `PS:16223-16243`).
  - The title names the transition ("Switch to Personas?", "Open Wren?", "Close the editor?").
  - The body names the item.
  - The buttons stay **Save and continue · Discard and continue · Stay**. Escape and default focus are **Stay**.
  - After Stay, focus returns to the control that held it. The nav-bar highlight after Stay on *leaving* the screen (RP-061) is a shell defect B depends on but does not fix.

| Transition | Does a draft-holding body get other data? | Veto? |
|---|---|---|
| View switch within the loaded item (Edit → Test chat → Edit; persona Edit → Look) | No: a display toggle; the character test chat even reads the draft | **No.** Card/Profile while dirty shows a one-row notice: "Showing the saved version · 2 unsaved changes in Edit [ Back to Edit ]" |
| Selecting another item | Yes | **Yes** |
| Kind switch (`c p l d [ ]`, rail rows, New… into another kind) | Yes (`switch_mode` clears; `PW/personas_state.py:54-65`) | **Yes** |
| Start row, landing buttons, Recently edited, Continue rows | Yes | **Yes** |
| Chat now while dirty | — | **No dialog**: Chat now is disabled with `save first` |
| Deep link into a character conversation | Yes | **Yes** (ADR-046 names deep links) |
| Leaving Roleplay | Yes | **Yes** (the existing aggregate guard; Ctrl+Q is TASK-33622.14) |
| Entries: another entry, prev/next, `+ New entry` while the entry form is dirty | the form is reloaded | **Yes, inline** `[ Update ] [ Discard ] [ Stay ]` |
| Options (lore or dictionary) dirty → another item, kind or screen | Yes | **Yes** (lore now posts `LoreSettingsEdited`; the dictionary already feeds `PS:5630-5640`) |
| Resize, pane collapse/expand, session start/end, Try column show/hide, More actions toggle | No | **No** |

**Governance:** a dated ADR-046 amendment (cross-referenced from ADR-120:123-127) names the two new domains and the single predicate (§5.9 G4).

**Fixes:** RP-037, RP-038, RP-040 (partly).

### 3.13 World, Used by, Info: copy that does not depend on D6

One runtime flag: `roleplay_character_world_info_applied: bool`. It is **false today**, because `capture_prompt_transform_inputs` passes `None` for the character (`Chat/console_chat_controller.py:1273-1297`). Sub-project C sets it true when D6 ships.

| Surface | Flag false (today) | Flag true (after C) |
|---|---|---|
| Character › World heading | "**Copied into this character** · not applied in chats yet. To use a book in a chat now: open it › Used by › Link to chat…" | "**Applied in this character's chats**" (C supplies copy vs link) |
| Lore/dictionary › Used by › Character copies | "Characters can hold a copy of this book (Character › World). Copies are **not applied in chats**." | the list of characters, supplied by C |
| Used by › Linked chats | the table, with "applied" (proven end to end for lore, C.7 §1) | unchanged |
| Landing › Needs attention | "N characters hold lore copies that are not applied in chats [ Why? ]" | row hidden |

Not in B: the "In this chat" summary (D15, RP-082), an "Update copy" button, the conversation picker (RP-083, C).

**Fixes:** RP-012 (honesty), RP-004 (honesty only).

### 3.14 Mockups: work pane

**MK3 · 160x45 · Character Edit in a work session, caret in Description (rows 1-30 and 42-45).**
- **Layout:** rail closed to its grip, list 42, work 108 (EDIT session at 160: the rail closes below 175, the list stays from 142).
- **Header rows:** `w` ≥ 100, so the action bar joins row 1 (R21).
- **Unsaved signals:** exactly three (R24): the header chip, the title-row word and the commit bar. No ● anywhere.
- **Footer:** T1.

Fixes shown: RP-016, RP-017, RP-038, RP-019, RP-067 (`save first` as words).

```text
                                      ╭─────────────╮                                                                                                           
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP   ⌃0 ACP   F2 Lab   F3 Logs   F4 Settings   More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Characters · editing                                                                                                        Unsaved changes   Local 
  N  ╭─ Characters 28 ────────────────────────╮  C  ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
  a  │ ▎Search characters… /  Name  Tag  + New│  h  │ ‹ Characters  Captain Isolde Varga  unsaved                     [ Chat now ] save first   More actions ▸ │
  v  │   Ada (Debugging Partner)              │  a  │  Card  ▸Edit  Chats 2  World 1  Test chat  Info                                                          │
     │   4 chats · 1h · coding · A patient s… │  r  │ Jump:  Basics · Prompting · Greetings · Voice & look · Details                             [ Generate… ] │
     │   ARIA-7                               │  a  │ BASICS                                                                                                   │
     │   3w · sci-fi · A ship AI that has qu… │  c  │ Name                                                                                                     │
     │   Brother Anselm                       │  t  │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
     │   1mo · A monastery brewer with stron… │  e  │ │Captain Isolde Varga                                                                                  │ │
     │▸  Captain Isolde Varga                 │  r  │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
     │   2 chats · 3d · sci-fi · Commands th… │  s  │ Description                                                                                     Generate │
---> │   Chef Auguste                         │     │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
     │   1 chat · 5d · A temperamental Paris… │     │ │Captain Isolde Varga commands the deep-survey frigate *Meridian's Wake*, a patched-together           │ │
     │   Coach Pemberton                      │     │ │vessel that has outlived three refits and two mutinies. She trusts instruments over people and        │ │
     │   2w · A relentlessly upbeat running … │     │ │keeps a paper logbook because paper cannot be hacked. Twenty years in the Outer Survey have made      │ │
     │   Default Assistant                    │     │ │her terse, exact and quietly haunted by the crew she lost at Kepler-9.▏                               │ │
     │   4 chats · 1h · built in · Used for … │<--- │ │                                                                                                      │ │
     │   Detective Rosa Calderón              │     │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
     │   1 chat · 2w · noir · A sharp, sardo… │     │ Personality                                                                                     Generate │
     │   Dungeon Master Vex                   │     │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
     │   6 chats · 1d · fantasy · A theatric… │     │ │Dry, precise, loyal, quietly haunted. Answers questions with questions when tired.                    │ │
---> │   Grumpy Barista Bot                   │     │ │                                                                                                      │ │
     │   3w · comedy · A coffee-machine AI t… │     │ └──────────────────────────────────────────────────────────────────────────────────────────────────────┘ │
     │   Hana Kobayashi                       │     │ First message                                                                                   Generate │
     │   1mo · A ceramicist in Kyoto who tea… │     │ ┌──────────────────────────────────────────────────────────────────────────────────────────────────────┐ │
     │   Kestrel                              │     │ │*Isolde doesn't look up from the logbook.* "You're late. Debris field's shifted again, and the        │ │
     │   2mo · fantasy · A terse sky-pirate … │<--- │ │nav AI thinks it's being clever."                                                                     │ │
┄┄┄ rows 31-41 elided: editor fields continue in the one scroller (Scenario, Greetings, Voice & look, Details) ┄┄┄                                              
     │   2mo · A distracted xenolinguist      │     │ ──────────────────────────────────────────────────────────────────────────────────────────────────────── │
     │   Quartermaster Bell                   │     │ [ Save changes ]  Cancel                                             2 unsaved: Description, Personality │
     ╰────────────────────────────── 4 of 28 ─╯     ╰───────────────────────────────────────────────────────────────────────────────────────────────── ▾ more ─╯
  typing in field | esc leave field | ctrl+s save | after esc: < > view  c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit               
```

**MK4 · 220x55 · Chat dictionary, Entries in a work session with the Try column (rows 1-30 and 53-55).**
- **Layout:** +TRY-E. The rail is closed below 234, then list 42 and work 168, whose content of 164 splits into entries 46, rule, editor 56, rule and Try 60.
- **Why the rail closed:** the dictionaries' `try_column` preference is on (Q2), and an entry is open.
- **Outside a work session, Card and Profile at 220 keep the rail.**
- **Footer:** E1 at 220 (dictionaries: rule).

Fixes shown: RP-006, RP-056, RP-081, RP-015 (`warning` row word, `Info (1 warning)` chip), RP-091, RP-010/D2.

```text
                                      ╭─────────────╮                                                                                                                                                                       
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP   ⌃0 ACP   F2 Lab   F3 Logs   F4 Settings   F5 Research   F7 Help                                       More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Chat dictionaries · editing                                                                                                                                                                               Local 
  N  ╭─ Chat dictionaries 3 · 2 on ───────────╮  D  ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
  a  │ ▎Search dictionaries… /   Name  + New  │  i  │ ‹ Chat dictionaries  Grand Dialect Pack  saved 1m                                                                                   [ Run preview ]   More actions ▸ │
  v  │   Archaic Speech                    on │  c  │  ▸Entries 150  Options  Used by 1  Try it (beside)  History 4  Info (1 warning)                                                                                      │
     │   12 rules · 1 regex                   │  t  │ [+ New rule]  ▎Filter pattern, replacement… / │Rule #12 · regex                         ‹ prev   next ›│TRY IT · nothing saved                            [ Hide × ] │
     │▸  Grand Dialect Pack      warning · on │  i  │ #12 of 150 · showing 150 · 3 off · 1 warning  │Pattern                                                 │24 fired · 2 skipped · 24/1000 tokens                        │
     │   150 rules · 31 regex                 │  o  │  on   Pattern              → Replacement      │▎/\b(?:hi|hello)\b/                                     │──────────────────────────────────────────────────────────── │
     │   Pirate Cant                      off │  n  │  on   phone                → speaking-stone   │Replacement                                             │Sample                                                       │
     │   40 rules                             │  a  │  on   car                  → carriage         │▎[archaic-1]                                            │OK hello guys, my phone and car and gun broke;               │
     │                                        │  r  │  on   gun                  → crossbow         │Match       ( ) text   (•) regex                        │the computer, television and radio too. Email                │
     │                                        │  i  │  on   computer             → thinking-engine  │Case        [x] ignore case                             │the airplane pilot, yeah.                                    │
---> │                                        │  e  │  on   television           → seeing-mirror    │Probability 100 %     Priority 0                        │                                                             │
     │                                        │  s  │  on   radio                → whisper-shell    │Max uses    0 (unlimited)                               │Result                                                       │
     │                                        │     │  on   email                → raven-post       │Group       greetings                                   │[archaic-0] [archaic-1] [archaic-3], my speaking-stone       │
     │                                        │     │  on   airplane             → sky-galleon      │Enabled     [x] on                                      │and carriage and crossbow broke; the thinking-engine,        │
     │                                        │     │  on   rocket               → fire-lance       │                                                        │seeing-mirror and whisper-shell too. raven-post the          │
     │                                        │<--- │  on   camera               → memory-box       │                                                        │sky-galleon pilot, [archaic-8].                              │
     │                                        │     │  on   laptop               → slate            │                                                        │                                                             │
     │                                        │     │ ▸on   /\b(?:hi|hello)\b/   → [archaic-1]      │                                                        │Fired (24)                                                   │
     │                                        │     │  on   /\bOK\b/             → [archaic-0]      │                                                        │/\b(?:hi|hello)\b/ → [archaic-1]  ×1 · 1 tok                 │
     │                                        │     │  on   /\bguys\b/           → [archaic-3]      │                                                        │/\bOK\b/ → [archaic-0]  ×1 · 1 tok                           │
---> │                                        │     │  off  /(unclosed/          → [archaic-9]      │                                                        │phone → speaking-stone  ×1 · 1 tok                           │
     │                                        │     │  on   /\bcoffee\b/         → [archaic-5]      │                                                        │Skipped (2)                                                  │
     │                                        │     │  on   /\bminutes?\b/       → [archaic-6]      │                                                        │/(unclosed/ · off                                            │
     │                                        │     │  on   /\b(?:yeah|yep)\b/   → [archaic-8]      │                                                        │/\bminutes?\b/ · probability roll                            │
     │                                        │     │  on   weekend              → [archaic-4]      │                                                        │                                                             │
     │                                        │<--- │  on   mph                  → [archaic-2]      │                                                        │                                                             │
┄┄┄ rows 31-52 elided: same columns continue: rules 20-36, Try results, editor blank ┄┄┄                                                                                                                                    
     │                                        │     │                                               │[ Update rule ]  Revert   Move ↑ ↓               Delete…│▎Test sample… (enter runs, ctrl+j new line)          [ Run ] │
     ╰─────────────────────────────── 2 of 3 ─╯     ╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────── ▾ more ─╯
  enter edit rule | n new rule | space on/off | del delete… | / filter entries | esc to list | t try it | < > view | c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                               
```

**MK5 · 160x45 · Lore book Entries in a work session, split, editing entry #200 (rows 1-30 and 42-45).**
- **Layout:** ENTRIES at 160: rail closed, list 42 (1-line rows, because 43 books > 40), work 108, with entries 46 and the editor 57.
- **Entries:** 33 entry rows (today 10).
- **Run preview** is the primary.
- **Footer:** T5. The caret is in the entry's Content field, so Ctrl+S presses `[ Update entry ]` (R41).

Fixes shown: RP-006, RP-056, RP-091, RP-003, RP-014 (anchor), RP-041 (names).

```text
                                      ╭─────────────╮                                                                                                           
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP   ⌃0 ACP   F2 Lab   F3 Logs   F4 Settings   More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Lore books · editing                                                                                                        Unsaved changes   Local 
  N  ╭─ Lore books 43 · 34 on ────────────────╮  L  ╭──────────────────────────────────────────────────────────────────────────────────────────────────────────╮
  a  │ ▎Search lore books… /       Name  + New│  o  │ ‹ Lore books  Grand Codex of the Shattered Realms  unsaved              [ Run preview ]   More actions ▸ │
  v  │   Aetheria Atlas               12 · on │  r  │  ▸Entries 250  Options  Used by 1  Try it  Info                                                          │
     │   Ashfall Chronicle            31 · on │  e  │ [+ New entry]  ▎Filter keys, content… /       │Entry #200 · Saltmarsh Bellwarden          ‹ prev  next › │
     │   Brinewood Almanac            8 · off │     │ #200 of 250 · showing 250 · 28 off            │Keys        ▎Saltmarsh Bellwarden, Bellwarden             │
     │   Coldharbour Customs          15 · on │     │  on   Key                Content           Ord│Also needs  ▎(optional secondary keys)                    │
     │   Emberfall Gazette            22 · on │     │  on   Saltmarsh Abbey    The abbey stand…  195│Content                                                   │
     │▸  Grand Codex of the Shat…ed Realms on │     │  on   Saltmarsh Archive  Ledgers of ever…  196│┌───────────────────────────────────────────────────────┐ │
     │   Highspire Gazetteer (im…rom card) on │     │  off  Saltmarsh Beacon   Lit only when t…  197││The Bellwarden of Saltmarsh rings the drowned bell of  │ │
     │   Highspire Gazetteer (im…m card) 2 on │     │  on   Saltmarsh Bell     The drowned bel…  198││St Ives-under-Tide at each turning of the tide. The    │ │
---> │   Hollow Moon Saga             19 · on │     │  on   Saltmarsh Bellhou… Three storeys o…  199││office passes to whoever hears the bell answer from    │ │
     │   Kepler-9                      3 · on │     │ ▸on   Saltmarsh Bellwar… The Bellwarden …  200││under the water; the current Bellwarden, Agathe Moss,  │ │
     │   Lantern Coast                11 · on │     │  on   Saltmarsh Causeway Passable two ho…  201││heard it at nine. Ships that miss the ringing drift    │ │
     │   Meridian Survey Notes         6 · on │     │  on   Saltmarsh Charter  Granted by the …  202││onto the Hulks.▏                                       │ │
     │   Night Market Rules           9 · off │     │  on   Saltmarsh Eel-wiv… A guild of fish…  203││                                                       │ │
     │   Obsidian Reach Primer        14 · on │<--- │  off  Saltmarsh Ferry    Retired after t…  204││                                                       │ │
     │   Old Harbour Ledger            7 · on │     │  on   Saltmarsh Fog      Comes in from t…  205││                                                       │ │
     │   Pale Lantern Rites           5 · off │     │  on   Saltmarsh Gate     The only landwa…  206││                                                       │ │
     │   Quill & Cinder Society       18 · on │     │  on   Saltmarsh Hulks    Prison ships mo…  207││                                                       │ │
     │   Red Tide Almanac              9 · on │     │  on   Saltmarsh Lantern  Green glass, so…  208││                                                       │ │
---> │   Sable Court Annals           27 · on │     │  on   Saltmarsh Market   Held on the Cau…  209│└───────────────────────────────────────────────────────┘ │
     │   Sky Archipelago Atlas        33 · on │     │  on   Saltmarsh Reeve    Elected each sp…  210│Position    [Before character ▾]   Order ▎200             │
     │   Starfall Concordance         12 · on │     │  on   Saltmarsh Saltpans Terraced pans a…  211│Match       [x] on   [ ] case-sensitive   [ ] regex       │
     │   Sunken Library Index         41 · on │     │  on   Saltmarsh Tithe    One fish in twe…  212│            [ ] selective (needs a secondary key)         │
     │   Tamsin's Field Notes          4 · on │     │  on   Sable Crown        The regalia of …  213│                                                          │
     │   The Drowned Bell Cycle       16 · on │<--- │  on   Sable Court        Ruling council …  214│                                                          │
┄┄┄ rows 31-41 elided: entries continue: 33 entry rows at 160x45 ┄┄┄                                                                                            
     │                                        │     │  on   Slate Quarry       Abandoned when …  226│                                       1 unsaved: Content │
     │                                        │     │  on   Smugglers' Stair   Cut into the cl…  227│[ Update entry ]  Revert   Move ↑ ↓               Delete… │
     ╰────────────────────────────── 6 of 43 ─╯     ╰───────────────────────────────────────────────────────────────────────────────────────────────── ▾ more ─╯
  typing in field | esc leave field | ctrl+s update entry | after esc: enter edit entry  n new entry  < > view  c/p/l/d [ ] kind | F1 · Ctrl+P · Ctrl+Q         
```

**MK6 · 120x36 · Lore book Entries in a work session, drill-in list (full height).**
- **Layout:** ENTRIES at 120: rail closed, list 32, work 78 (< 100, so drill-in).
- **Entries:** 23 entry rows (today 4). Enter opens the entry editor full width; Escape returns to the same row.

```text
                                      ╭─────────────╮                                                                   
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP     More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Lore books                                                                                            Local 
  N  ╭─ Lore books 43 · 34 on ──────╮  L  ╭────────────────────────────────────────────────────────────────────────────╮
  a  │ ▎Search… /        Name  + New│  o  │ ‹ Lore books  Grand Codex of the S…ed Realms                      saved 1m │
  v  │   Aetheria Atlas     12 · on │  r  │ [ Run preview ]                                             More actions ▸ │
     │   Ashfall Chronicle  31 · on │  e  │  ▸Entries 250  Options  Used by 1  Try it  Info                            │
     │   Brinewood Almanac  8 · off │     │ [+ New entry]  ▎Filter keys, content… /                                    │
     │   Coldharbour Customs     on │     │ #200 of 250 · showing 250 · 28 off · enter opens the entry                 │
     │   Emberfall Gazette  22 · on │     │  on   Key                  Content                                     Ord │
     │▸  Grand Codex o…ed Realms on │     │  on   Saltmarsh Abbey      The abbey stands on stilts above the        195 │
     │   Highspire Gaz…rom card) on │     │  on   Saltmarsh Archive    Ledgers of every drowned parish, kept       196 │
     │   Highspire Gaz…m card) 2 on │     │  off  Saltmarsh Beacon     Lit only when the bell is silent; a         197 │
---> │   Hollow Moon Saga   19 · on │     │  on   Saltmarsh Bell       The drowned bell of St Ives-under-Tide      198 │
     │   Kepler-9            3 · on │     │  on   Saltmarsh Bellhouse  Three storeys of salt-white stone that      199 │
     │   Lantern Coast      11 · on │     │ ▸on   Saltmarsh Bellwarden The Bellwarden of Saltmarsh rings the       200 │
     │   Meridian Survey Notes   on │     │  on   Saltmarsh Causeway   Passable two hours either side of low       201 │
     │   Night Market Rules     off │     │  on   Saltmarsh Charter    Granted by the Tide Court in the year       202 │
     │   Obsidian Reach Primer   on │<--- │  on   Saltmarsh Eel-wives  A guild of fisher-women who read the        203 │
     │   Old Harbour Ledger  7 · on │     │  off  Saltmarsh Ferry      Retired after the Causeway opened; its      204 │
     │   Pale Lantern Rites     off │     │  on   Saltmarsh Fog        Comes in from the east in the second        205 │
     │   Quill & Cinder Society  on │     │  on   Saltmarsh Gate       The only landward gate; its lintel is       206 │
     │   Red Tide Almanac    9 · on │     │  on   Saltmarsh Hulks      Prison ships moored in the outer reach      207 │
---> │   Sable Court Annals      on │     │  on   Saltmarsh Lantern    Green glass, sold to travellers who         208 │
     │   Sky Archipelago Atlas   on │     │  on   Saltmarsh Market     Held on the Causeway at low water only      209 │
     │   Starfall Concordance    on │     │  on   Saltmarsh Reeve      Elected each spring tide by the guilds      210 │
     │   Sunken Library Index    on │     │  on   Saltmarsh Saltpans   Terraced pans along the southern flats      211 │
     │   Tamsin's Field Notes    on │     │  on   Saltmarsh Tithe      One fish in twelve to the abbey, paid       212 │
     │   The Drowned Bell Cycle  on │<--- │  on   Sable Crown          The regalia of the Shattered Realms' last   213 │
     │   Thornwood Heraldry      on │     │  on   Sable Court          Ruling council of the eastern realms;       214 │
     │   Umber Steppe Tribes    off │     │  on   Sandglass Order      Monks who keep time by falling sand and     215 │
     │   Veyra Skyport Guide     on │     │  off  Scarlet Ferry        A ghost-boat said to carry oath-breakers    216 │
     │   Vex's Campaign Bible    on │     │  on   Scour Winds          Autumn gales that strip the high moors      217 │
     ╰──────────────────── 6 of 43 ─╯     ╰─────────────────────────────────────────────────────────────────── ▾ more ─╯
  enter edit entry | n new entry | space on/off | del delete… | / filter entries | esc to list | F1 · Ctrl+P · Ctrl+Q   
```

**MK7 · 160x45 · Arrival: landing, nothing loaded, list focused on row 1 (rows 1-31 and 43-45).**
- **Layout:** rail 35, list 42, work 73. The landing's Start buttons sit on one row (`w` ≥ 72).
- **Continue:** shown (Q7; G5).
- **Rail:** the current kind (Characters) keeps `▸`.
- **Footer:** L0.

Fixes shown: RP-055, RP-024, RP-021, RP-028, RP-029.

```text
                                      ╭─────────────╮                                                                                                           
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP   ⌃0 ACP   F2 Lab   F3 Logs   F4 Settings   More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Characters                                                                                                                                    Local 
╭─ Roleplay ──────────────────────╮  N  ╭─ Characters 28 ────────────────────────╮  C  ╭───────────────────────────────────────────────────────────────────────╮
│ Start                           │  a  │ ▎Search characters… /  Name  Tag  + New│  h  │ START                                                                 │
│     import · recent             │  v  │   Ada (Debugging Partner)              │  a  │ Roleplay prepares what a chat knows: who the AI plays, reusable AI    │
│                                 │     │   4 chats · 1h · coding · A patient s… │  r  │ roles, and the world info they share. Pick one on the left, or:       │
│ Cast                          ▾ │     │   ARIA-7                               │  a  │                                                                       │
│ ▸ Characters (28)             c │     │   3w · sci-fi · A ship AI that has qu… │  c  │ [ Import… ]   character cards (.png/.json), .tldw-actor-pack,         │
│     who the AI plays            │     │   Brother Anselm                       │  t  │               lore book or chat dictionary JSON/MD                    │
│   Personas (4)                p │     │   1mo · A monastery brewer with stron… │  e  │ [New character] [New persona] [New lore book] [New chat dictionary]   │
│     reusable AI roles           │     │   Captain Isolde Varga                 │  r  │                                                                       │
│   You: Alex          Settings › │     │   2 chats · 3d · sci-fi · Commands th… │  s  │ CONTINUE A CHARACTER CHAT                               5 most recent │
│     your name in chats          │<--- │   Chef Auguste                         │     │   Shore leave on Kepler-9 · Captain Isolde Varga        12 msgs · 16m │
│                                 │     │   1 chat · 5d · A temperamental Paris… │     │   The Emberfall gates · Dungeon Master Vex               31 msgs · 3h │
│ World info                    ▾ │     │   Coach Pemberton                      │     │   Debris field salvage · Captain Isolde Varga             4 msgs · 2d │
│   Lore books (3 · 2 on)       l │     │   2w · A relentlessly upbeat running … │     │   Cold case: the Lido · Detective Rosa Calderón          18 msgs · 5d │
│     facts on keywords           │     │   Default Assistant                    │     │   Kitchen nightmares · Chef Auguste                       7 msgs · 1w │
│   Chat dictionaries (3 · 2 on)  │     │   4 chats · 1h · built in · Used for … │<--- │   enter continues in Console · a character's Chats view lists all     │
│     find/replace rules          │     │   Detective Rosa Calderón              │     │                                                                       │
│                                 │     │   1 chat · 2w · noir · A sharp, sardo… │     │ RECENTLY EDITED                                                       │
│ Create & import               ▾ │     │   Dungeon Master Vex                   │     │   Grand Codex of the Shattered Realms                  lore book · 2h │
│   New…                          │     │   6 chats · 1d · fantasy · A theatric… │     │   Captain Isolde Varga                          character · yesterday │
│     pick a kind                 │<--- │   Grumpy Barista Bot                   │     │   Grand Dialect Pack                             chat dictionary · 3d │
│   Import…                       │     │   3w · comedy · A coffee-machine AI t… │     │   Cozy Storyteller                                       persona · 3d │
│     any Roleplay file           │     │   Hana Kobayashi                       │     │                                                                       │
│                                 │     │   1mo · A ceramicist in Kyoto who tea… │     │ NEEDS ATTENTION                                                     2 │
│   Buddy: hidden                 │     │   Kestrel                              │     │   ! Grand Dialect Pack: 1 rule is not a valid regex     [ Open rule ] │
│     app companion               │     │   2mo · fantasy · A terse sky-pirate … │<--- │   ! 2 characters hold lore copies not applied in chats       [ Why? ] │
│                                 │     │   Lady Evangeline Thornwood            │     │                                                                       │
┄┄┄ rows 32-42 elided: landing ends at row 30; list continues (18 characters, 2-line rows) ┄┄┄                                                                  
│                                 │     │   Quartermaster Bell                   │     │                                                                       │
╰─────────────────────────────────╯     ╰────────────────────────────── 1 of 28 ─╯     ╰───────────────────────────────────────────────────────────────────────╯
  enter open | / search | n new | i import | c/p/l/d [ ] kind | m mark | s sort | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                         
```

**MK8 · 120x36 · Arrival at the small design centre (full height).**
- **Layout:** work 50 (content 46). The Start buttons fall back to 2 × 2, and Continue rows take two lines.

```text
                                      ╭─────────────╮                                                                   
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists   ⌃7 Schedules   ⌃8 Workflows   ⌃9 MCP     More ▾ 
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 Roleplay   Characters                                                                                            Local 
╭─ Roleplay ───────────────╮  N  ╭─ Characters 28 ──────────────╮  C  ╭────────────────────────────────────────────────╮
│ Start                    │  a  │ ▎Search… /  Name  Tag  + New │  h  │ START                                          │
│     import · recent      │  v  │   Ada (Debugging Partner)    │  a  │ Roleplay prepares what a chat knows: who the   │
│                          │     │   ARIA-7                     │  r  │ AI plays, reusable AI roles, and the world     │
│ Cast                   ▾ │     │   Brother Anselm             │  a  │ info they share. Pick one on the left, or:     │
│ ▸ Characters (28)      c │     │   Captain Isolde Varga       │  c  │                                                │
│     who the AI plays     │     │   Chef Auguste        1 chat │  t  │ [ Import… ]  cards, actor packs, lore books,   │
│   Personas (4)         p │     │   Coach Pemberton            │  e  │              chat dictionaries                 │
│     reusable AI roles    │     │   Default Assistant  4 chats │  r  │ [New character]  [New persona]                 │
│   You: Alex   Settings › │     │   Detective Rosa Calderón    │  s  │ [New lore book]  [New chat dictionary]         │
│     your name in chats   │<--- │   Dungeon Master Vex         │     │                                                │
│                          │     │   Grumpy Barista Bot         │     │ CONTINUE A CHARACTER CHAT                      │
│ World info             ▾ │     │   Hana Kobayashi             │     │   Shore leave on Kepler-9                      │
│   Lore books (3 · 2 on)  │     │   Kestrel                    │     │     Captain Isolde Varga · 12 msgs · 16m       │
│     facts on keywords    │     │   Lady Evangeline Thornwood  │     │   The Emberfall gates                          │
│   Chat dictionaries (3)  │     │   Luna — Night Mar…ne Teller │<--- │     Dungeon Master Vex · 31 msgs · 3h          │
│     find/replace rules   │     │   Marla the Ferrywoman       │     │   Debris field salvage                         │
│                          │     │   Mira Okonkwo               │     │     Captain Isolde Varga · 4 msgs · 2d         │
│ Create & import        ▾ │     │   Old Tom the Ligh…se Keeper │     │                                                │
│   New…                   │     │   Professor Hargreaves       │     │ RECENTLY EDITED                                │
│     pick a kind          │<--- │   Quartermaster Bell         │     │   Grand Codex of th…d Realms         lore · 2h │
│   Import…                │     │   Rook                       │     │   Captain Isolde Varga          character · 1d │
│     any Roleplay file    │     │   Samira “Sammy” Vadem       │     │   Grand Dialect Pack           dictionary · 3d │
│                          │     │   Ser Corwin of th…shen Vale │     │                                                │
│   Buddy: hidden          │     │   Sister Agnes               │     │ NEEDS ATTENTION                              2 │
│     app companion        │     │   Theo the Cartographer      │<--- │   ! Grand Dialect Pack: 1 rule is not a        │
│                          │     │   Vesper                     │     │     valid regex                  [ Open rule ] │
│                          │     │   Wren Halloway       1 chat │     │   ! 2 characters hold lore copies that         │
│                          │     │   Yusuf the Spice Merchant   │     │     are not applied in chats          [ Why? ] │
│                          │     │   Zephyr                     │     │                                                │
╰──────────────────────────╯     ╰──────────────────── 1 of 28 ─╯     ╰─────────────────────────────────────── ▾ more ─╯
  enter open | / search | n new | i import | c/p/l/d [ ] kind | m mark | s sort | F6 next pane | F1 · Ctrl+P · Ctrl+Q   
```

---

## 4. Interaction: keys, focus, Escape, footer, help, state and memory

### 4.1 Invariants (each becomes a test, §5.7)

- **I1. Focus stays in the screen.** No arrival, Tab, collapse, restore, delete, view switch or async completion leaves focus on the nav bar or on `None`.
  - In Textual, hiding a focused container sets `app.focused = None` (prototype); it does not move focus. So **every transition that hides the focused widget's ancestor names its focus target** (§4.11).
- **I2. No first stop leaves the screen.** No arrival target, F6 target or restored focus may be a control whose Enter leaves Roleplay: Chat now, Continue in Console, a landing Continue row, the You row, the destination-blocked chip. These ids form one set, `LEAVES_SCREEN_IDS`.
- **I3. Escape never leaves the screen**, has one owner, and its advertised label is the rung that will run (ADR-031 refinement, task-16211).
- **I4. One projection.** `check_action`, the footer, F1 and the Escape label come from one pure function of (focus context, state). Advertised ⊆ armed, and advertised == working.
- **I5. Letters act only on navigation surfaces** (§4.6).
- **I6. Highlight never loads.** Arrows move a cursor; Enter or a click loads.
- **I7. The work pane renders the loaded record**, never the selection (ADR-084).
- **I8. View switches are lossless and never move panes** (R9). Drafts belong to the loaded item.
- **I9. Every transition names where focus goes** (§4.11).
- **I10. Focus never changes geometry** (DESIGN.md:125), and **no state is colour-only** (DESIGN.md:361).

### 4.2 Focus contexts (the projection's input)

The projection is pure and lives in `PM/roleplay_frame_state.py` (§5.11):

```python
focus_context(focused, layout, state) -> FocusContext
armed_keys(ctx, state) -> tuple[KeyHint, ...]                 # key, label, available, reason
escape_rung(ctx, state) -> EscapeRung | None                  # §4.5
footer_context(ctx, state) -> ShortcutContext                 # UI/Navigation/shortcut_context.py (R35)
help_state(ctx, state) -> WorkbenchHelpState                  # UI/Workbench/help.py
f6_targets(layout, state) -> tuple[WorkbenchPaneTarget, ...]  # Widgets/workbench_focus.py
editor_save(ctx, state) -> SaveTarget | None                  # ctrl+s: the visible commit button (§4.6, R41)
```

| Context | Holds focus | Letters armed |
|---|---|---|
| `NAV` | rail rows, rail section headers, either grip | kind, find, create |
| `LIST_TEXT` | the items search box | none (Enter goes to the list) |
| `LIST` | the items list | all list verbs |
| `ENTRIES` | the entries list | kind, find, `n`, `t`, `space`, `del`, `<` `>` |
| `WLIST` | the conversations list, the dictionary versions list, persona tool-policy rules | kind, find, `<` `>`, `o` on conversation rows; **never `del`, `space` or `n`** (R31) |
| `STRIP` | the view strip | kind, item verbs (`o e t < >`; `u` only while the RP-047 code flag is on) |
| `WORK_BTN` | any Button in the work pane | **none** |
| `WORK_TEXT` | inputs and TextAreas in the work pane (fields, test-chat box, Try-it sample, filters) | none |
| `WORK_FORM` | Select, Switch, Checkbox | none |
| `TRY` | the ≥200 Try column | none |
| `LANDING` | landing buttons and rows | none |
| `NONE` | nothing | must not occur (I1) |

Reading bodies (card, Info, transcript) are not focusable. They are scrolled from the strip (↑/↓ by line, PgUp/PgDn/Home/End).

Ctrl+S is not a letter and is not armed per context: `editor_save` resolves the one commit button it presses from where focus is (the editor regions of §4.6). Only the footer and F1 read that value, to decide whether to advertise Ctrl+S. `check_action('roleplay_save')` returns True everywhere on the screen (bound screen-wide, ADR-031's task-1340 pattern), so a stray press runs and shows L6's state chip instead of falling through. The action presses `editor_save(...)`'s button, or shows the state chip (`nothing to save here`, `no changes`, `saving…`).

### 4.3 Arrival and the Tab region

- **First paint.** The screen's `AUTO_FOCUS` is the items list. The list is eager, so focus lands before the rows arrive; when they load, the cursor parks on row 1 and nothing is loaded (I6). The landing is one F6 away.
- **Return.** After `_apply_pending_restore` (`PS:1808-1876`), focus the remembered `focus_id` only if it is:
  - mounted and focusable;
  - inside `#screen-content`;
  - not in `LEAVES_SCREEN_IDS`;
  - and no key or click has happened since arrival (a focus-intent generation, the `LS:5616-5632` pattern).

  Otherwise focus the list at the restored cursor.
- **Empty kind.** The list still takes focus; the footer offers `n new` and `i import`.
- **Below 64.** The stage that restore selects, focused on that stage's F6 target.
- **Tab region.** `BaseAppScreen` gains an opt-in `TAB_REGION: ClassVar[str | None] = None` with `region_focus_next/previous` actions and an `arrival_focus_target()` hook (fallback `BaseAppScreen._first_focusable_in_content()`, `UI/Navigation/base_app_screen.py:218-232`). Roleplay sets `"#screen-content, #screen-content *"`.
  - Textual's binding merge (`DOMNode._merge_bindings`, `dom.py:671`) makes this behaviour-neutral for screens that do not opt in. Library and Console keep their own overrides; Library adoption is a follow-up (`LS` is in PR #2862).
  - Guard test: on every route with `TAB_REGION is None`, Tab from the first content control reaches the same widget as before.
  - B3 also runs the nav-bar mount tests (for example `test_master_shell_navigation_keeps_active_destination_visible_on_mount`), because the nav bar's `_mount_settled` logic assumes the first automatic focus may land on a `NavigationButton` (`main_navigation.py:740-775`).
- **Editing region.** While an editor view is open (character or persona Edit, Look, Tool policy, Options, the drill-in entry editor), Tab cycles inside the work pane only. F6 and Escape are the exits.
- **Order follows the visual order** (DESIGN.md:125): rail rows → rail grip → items (search, toolbar, list) → items grip → work pane (primary, More actions, strip, body, commit bar) → Try column. Rail rows and the strip are each **one roving Tab stop**. Grips are Tab stops in their visual position.

**Fixes:** RP-024, RP-027, RP-021.

### 4.4 F6 / Shift+F6

Targets are computed per call (the Artifacts method, `Widgets/Library/library_artifacts_reader_shell.py:43-72`) and fed to `focus_relative_workbench_pane` (`Widgets/workbench_focus.py:20-53`). This replaces the static `_WORKBENCH_FOCUS_TARGETS` (`PS:1122-1155`).

| Pane / state | F6 lands on |
|---|---|
| Rail (open / closed) | last-focused rail control not in `LEAVES_SCREEN_IDS`, else the current kind row / the Nav grip |
| Items (open / closed) | last-focused items control, else the list at the cursor / the kind grip |
| Work: landing | `Import…` (never a Continue row, I2) |
| Work: Card, Profile, World, Used by, Info, History | the view strip |
| Work: Edit, Options, Look, Tool policy | the first invalid field after a failed save, else this item's last-focused field, else Name (scrolled into view) |
| Work: Chats | the conversations list at the selected conversation |
| Work: Test chat / Try it | the chat input / the sample |
| Work: Entries split / drill-in editor | the entries list at the anchored entry / the editor's first field |
| Try column (visible) | its input; absent when hidden (no grip) |

- **Chat now is deliberately not an F6 target.** It has its own key (`o`, B5c) and is the work pane's first Tab stop.
- A target inside a demand body that is not mounted yet is skipped, never mounted.
- **Fixes:** RP-025, RP-032 (grip half).

### 4.5 The Escape ladder (one binding, twelve rungs, R11)

`Binding("escape", "roleplay_escape", "Back", show=False)` replaces `PS:16318-16351`. `check_action` returns `escape_rung(ctx, state) is not None`. Rungs are evaluated against the pane that holds focus; the first match wins.

| # | Applies when | Escape does | Focus after | Footer chip | Fixes |
|---|---|---|---|---|---|
| 0 | A modal is open (guard, Import picker, tag picker, New chooser, delete modal) | The modal's safe negative: **Stay** / Cancel | the modal's opener | the modal's own | — |
| 1 | A transient in-pane surface is open: More actions, the Generate suggestion toolbar (`personas_character_editor_widget.py:440-465`), the inline entry guard | Close it safely: the menu closes; a suggestion is discarded (author text untouched); the entry guard = Stay | its opener | `esc close menu` · `esc discard suggestion` | — |
| 2 | A cancellable operation from this pane is in flight (resume opening, `PM/personas_conversations_controller.py:1226-1233`) | cancel it | unchanged | `esc cancel` | preserve |
| 3 | Focus is in a text field | **leave the field** | items search → list; entries/chats filter → its list; an entry-editor field → the entries list at the anchored row; any other work-pane field → the strip; a Try field → the strip | `esc leave field` | RP-026 |
| 4 | Work pane: a drill-in is open (entry editor, transcript) | close the drill-in | the same row in the list beneath | `esc back to entries` · `esc back to chats` | — |
| 5 | Work pane in Edit, Options, Look or Tool policy | cancel the edit: if clean, close; if dirty, the aggregate dialog (§3.12) | the invoker (the list row if the edit began there, else the strip) | `esc cancel edit` / `esc cancel edit…` | RP-037 |
| 6 | Work pane in any other non-default view (Chats, World, Used by, Test chat, Try it, Info, History) | return to the default view; nothing is discarded | the strip | `esc back to card` · `esc back to entries` | — |
| 7 | Work pane in the default view, or the Try column | go to the items list **and end the work session** (panes return per preference) | the list at the loaded row | `esc to list` | R9 |
| 8 | Items list with a search or tag active | clear the search box **and** its filter together; selection and marks kept | the list at the loaded row | `esc clear search` | RP-021 |
| 9 | Items list with marks | clear the marks; toast "12 marks cleared" | unchanged | `esc clear marks` | RP-039 |
| 10 | Items list otherwise | focus the rail (reopening it if it is closed: a manual priority) | the current kind row | F1 only (`esc: rail`) | — |
| 11 | Rail or grip | nothing | — | none | I3 |

- **Below 64.** Rung 7 shows the list stage and rung 10 opens the rail from its grip. The esc chip leads the footer there (recovery before navigation, `LS:4808-4834`).
- **Two presses cancel an edit** (rung 3, then rung 5). The first Escape is safe everywhere and arms the letters for the next key, and the guard still protects the second. This is a deviation from RP-026's editor carve-out (§4.13).

### 4.6 Key map

"Guard" means the action runs through `_run_guarded` and therefore the aggregate dialog when anything is dirty.

| Key | Action | Armed in | Available when | Footer | Guard | Fixes |
|---|---|---|---|---|---|---|
| `F6` / `shift+F6` | next / previous pane (§4.4) | anywhere (F6 app-global, `app.py:929`; Shift+F6 a screen priority binding) | always | `F6 next pane` | — | RP-025 |
| `tab` / `shift+tab` | next / previous control in the region | anywhere | always | F1 only | — | RP-024, RP-027 |
| `esc` | the ladder | anywhere | a rung applies | `esc <rung>` | rung 5 | RP-026 |
| `enter` | the focused control's action (§4.7) | native | — | `enter <label>` (first chip) | per action | RP-035 |
| `↑ ↓ PgUp PgDn Home End` | list cursor; rail rove; strip scrolls the body | `LIST ENTRIES WLIST NAV STRIP` | — | `↑↓ scroll` on the strip | — | RP-027 |
| `← →` | strip: previous/next view; rail: collapse/expand a section | `STRIP NAV` | — | `←→ view` | none | RP-063 |
| `c` `p` `l` `d` | Characters · Personas · Lore books · Chat dictionaries | `NAV LIST ENTRIES WLIST STRIP` | the target is not the active kind | `c/p/l/d [ ] kind` | yes; ends the session | RP-033, RP-034 |
| `[` `]` | previous / next kind, in rail order | same | — | (same chip) | yes | RP-034 |
| `/` | pane-scoped search (R26) | `NAV LIST ENTRIES WLIST STRIP` | a target box exists | `/ search` · `/ filter entries` · `/ filter chats` | — | RP-035, RP-056 |
| `ctrl+f` | the items search; works while typing | anywhere | — | F1 only (`/ or ctrl+f`) | — | preserve |
| `n` | New… for the active kind; on the entries list, a new entry | `NAV LIST ENTRIES` | create allowed (`PS:16468-16475`) | `n new` · `n new book` · `n new entry` · `n new rule` | yes | RP-040 |
| `ctrl+n` | the same; works while typing | anywhere | same | F1 only | yes | preserve |
| `i` | Import… | `NAV LIST` | import allowed | `i import` | — | RP-043 |
| `e` | Edit (cast) or Entries (books) for the cursor row, loading it first; **starts the work session** | `LIST STRIP` | editable kind | `e edit` · `e entries` | yes, when the item changes | RP-035, RP-016 |
| `t` | Test chat (cast) or Try it (books); **starts the session**; with the column visible, focuses it | `LIST ENTRIES STRIP` | the cursor row or the loaded item | `t test` · `t try it` · `t test draft` | yes, when the item changes | D2 |
| `o` (**B5c**) | Chat now (cast); Continue in Console (conversation row) | `STRIP`; `LIST` only when the cursor row **is** the loaded row; `WLIST` conversation rows | the button is enabled (saved, ready, not blocked) | `o chat now` · `o continue in Console` | none (it leaves; state kept) | RP-035, RP-019 |
| `u` | Add to Console message | `STRIP`; `LIST` (loaded row), only while the code flag is on | **never in B:** unbound, `check_action` False, absent from the footer and F1 until the flag flips (ADR-031 rule 4) | — (F1 only once enabled) | — | RP-035 |
| `<` `>` | previous / next view, **wrapping** | `LIST ENTRIES WLIST STRIP` | an item is loaded | `< > view` | none | RP-016 |
| `m` | mark / unmark the cursor row | `LIST` | marking supported | `m mark` · `m unmark` | — | RP-034, RP-039 |
| `s` | cycle sort | `LIST` | sortable kind | `s sort` | — | preserve |
| `space` | toggle on/off (book rows; entries), undo toast (R33) | `LIST` (book kinds), `ENTRIES` | row kind toggleable | `space on/off` | — | RP-033, RP-035, RP-091 |
| `del` | delete the cursor row, or the marked rows when any; one modal naming up to 5 (R32) | `LIST`, `ENTRIES` only (R31) | delete allowed | `del delete` · `del delete 2 marked` | modal (ADR-031 rule 3) | RP-039, RP-014 |
| `ctrl+s` | Press the **visible** commit button of the editor that holds focus (the Ctrl+S table below; R41) | bound screen-wide; acts only inside an editor region | `editor_save` names a visible, enabled button; elsewhere a no-op with the state chip `ctrl+s: nothing to save here` | the button's verb: `ctrl+s save` · `ctrl+s save pack` · `ctrl+s update entry` · `ctrl+s add entry` · `ctrl+s update rule` · `ctrl+s add rule` · `ctrl+s create lore book` · `ctrl+s create chat dictionary` (today's hint, `PS:16477`, generalised) | — | RP-035, H4 (Q5) |
| `enter` / `ctrl+j` (or `shift+enter`) in the Try-it sample | Run preview / newline (a TextArea subclass, §3.8.2); `ctrl+enter` is an unadvertised alias | Try-it sample | — | `enter run` · `ctrl+j new line` | — | RP-035 |
| `enter` in the test-chat box | send | test-chat box | — | `enter send` | — | — |
| `pgup` / `pgdn` at a list edge | previous / next page, **only if D14 falls back to B** | `LIST` | more than one page | `pgup/pgdn page` | — | RP-092 |

- **Retired:**
  - the screen-level `ctrl+enter` "Send to Console draft" (`PS:1097`);
  - `space`, `m` and `s` on the pane container (`personas_library_pane.py:80-86`); they move onto the list widget, so Space on a focused button no longer toggles a row (RP-033).
- **Never bound:**
  - `ctrl+1…0`, `f2-f5`, `f7` (ADR-152; its guard test, `Tests/UI/test_master_shell_navigation.py:560`, keeps running);
  - `ctrl+c/v/x/d/z/a/r/w` (ADR-031 rule 2). `ctrl+s` is rule 2's one exception, under the ADR-031 amendment (G9a): it is editor-scoped and mirrors a visible button.
- **No key** for Export, Duplicate or reorder.
- **House consistency:**
  - `n` new, `/` search, `i` import and `u` use-in-Console match Schedules, Library and Settings;
  - `t` test matches MCP;
  - pre-existing divergences stay (Library `t` = trash; Schedules `d` delete, `x` mark).

**Ctrl+S (R41; owner ruling Q5; ADR-031 amended, G9a).** One screen binding (`ctrl+s` → `roleplay_save`, the successor of `action_personas_save`, `PS:16295-16312`). It presses the **visible** commit button of the editor that holds focus, through the same `Button.press()` path as a click, so validation and messages are unchanged. `editor_save` returns a button only if it is enabled and it and every ancestor up to the editor region are displayed (walk `button.ancestors_with_self`, as `action_personas_save` checks `view.display` today, `PS:16306`). `Button.press()` itself checks only the button's own `display` and `disabled` (`textual/widgets/_button.py:429`), so it is not the guard: a commit button inside a hidden container (the persona commit bar hidden in Look, a hidden `TabPane`, a hidden drill-in entry editor) would still be pressed.

| Editor region (focus anywhere inside it) | Ctrl+S presses | Footer chip | Bound in |
|---|---|---|---|
| Character Edit: the work pane while Edit is the active view (title row, strip, body, commit bar). **Until B5b:** focus inside `#ccp-character-editor-view` / `#ccp-persona-editor-view` (fields, buttons, commit bar; in the persona view, outside the visual-identity, pack and rules sections); the mode strip is "elsewhere". | `#personas-char-editor-save` (`[ Save changes ]`) | `ctrl+s save` | today (`PS:1098`); brought under the focus rule in B3; the title row and strip join the region in B5b |
| Persona Edit (same region rule) | `#personas-editor-save` (`[ Save persona ]`) | `ctrl+s save` | today; B3; title row and strip in B5b |
| Persona Look | the section holding focus: `#personas-visual-identity-save` ("Save") or `#personas-persona-visual-save` ("Save Pack"); from the profile commit bar, the title row or the strip: `#personas-editor-save` while the persona commit bar is shown; with the commit bar hidden, Ctrl+S from the title row or strip is a no-op with `ctrl+s: nothing to save here` | `ctrl+s save` · `ctrl+s save pack` | B11 |
| Persona Tool policy | `#personas-policy-save` (from the rules editor, the title row or the strip); `#personas-editor-save` from the persona commit bar while it is shown | `ctrl+s save` | B11 |
| Lore or dictionary Options, including a new-book draft | `#personas-lore-settings-save` / `#personas-dict-settings-save` (`[ Save options ]`; `[ Create lore book ]` / `[ Create chat dictionary ]` on a draft) | `ctrl+s save` · `ctrl+s create lore book` · `ctrl+s create chat dictionary` | B9 |
| Entry editor, split column or drill-in (not the entries list or its filter) | Lore: `#personas-lore-entry-update` (`[ Update entry ]`), or `#personas-lore-entry-add` (`[ Add entry ]`) for a new entry. Dictionaries: `#personas-dict-entry-update` (`[ Update rule ]`), or `#personas-dict-entry-add` (`[ Add rule ]`) for a new rule | `ctrl+s update entry` · `ctrl+s add entry`; `ctrl+s update rule` · `ctrl+s add rule` | B9 |

- **Elsewhere** (rail, grips, items list, entries list, conversations list, landing, read views, Test chat, Try it): Ctrl+S changes nothing. The footer never advertises it there. A stray press puts the state chip `ctrl+s: nothing to save here` at the head of the footer until the next key or focus change (footer line L6). This is ADR-031's task-1340 pattern: bound screen-wide, advertised only where it works.
- **A visible but disabled button** (for example the visual-identity "Save" before any change, or a save in flight): no chip is advertised, and a press shows the state chip with the reason, for example `ctrl+s: no changes` or `ctrl+s: saving…`.
- **Before its slice binds it**, an editor's own buttons follow the "elsewhere" rule. The persona commit bar is the one exception, because B3 binds it: from B3, while `#personas-editor-save` is shown, Ctrl+S presses it from the persona profile region (its fields and the commit bar; from B5b also the title row and the strip of Edit, Look and Tool policy). Focus inside the visual-identity, pack or tool-policy-rules sections is "elsewhere" (a no-op with the L6 chip) until B11 binds those sections' own saves, so a section that does not hold focus is never saved (K22). Before B5b these sections sit in the one persona editor scroll; the rule is the same.
- **Tests.** A contexts × editors projection test asserts, for every row, the pressed id, the footer chip and the no-op path. A Pilot test proves that a button whose own `display` is True but whose container is hidden (the persona commit bar in Look, a hidden entry editor) is never pressed.

### 4.7 What Enter does

| Focus | Enter |
|---|---|
| Items row, not loaded | Load it (guarded). Focus stays on the list at ≥92 columns; at 64-91 it moves to the work stage (R37); below 64, to the work stage. |
| Items row, already loaded | Go into the work pane at its F6 target (never a hand-off, I2). |
| Rail kind row | Switch kind (guarded); focus the list. |
| Rail Start · New… · Import… · Buddy | Show the landing (guarded unload) · open the chooser · open the picker · open the Buddy chooser |
| Rail You row | Open Settings › Console Behavior (leaves the screen; guarded) |
| Section header · grip | Collapse/expand · toggle its pane |
| View strip | Go into the view's content (its F6 target) |
| Entries row | Open the entry (an authoring intent): split → focus the editor's first field; narrower → open the drill-in |
| Conversation row | Preview the transcript (drill-in below `w` 100); `o` continues it in Console |
| Search or filter box | Go to its list, with the first match under the cursor (keeps the 4-key find) |
| Button · Try-it sample · test-chat box | Press · run · send |

The first footer chip names Enter's action for the focused control. That also mitigates the weak button focus style (RP-065, D10).

### 4.8 Footer projection (R23, R35)

1. **One source.** `footer_context(ctx, state)` builds a `ShortcutContext` through the non-persisting `set_shortcut_context` path. The same key list drives `check_action` and F1 (Ctrl+S excepted: it stays enabled screen-wide, §4.2).
2. **Order.** The default precedence is: state chip (`typing in field`, `2 marked`) → the Enter chip → primary item verbs (`o`, `e`, `t`) → strip navigation (`←→ view`, `↑↓ scroll`) → the esc chip → find and create (`/`, `n`, `i`) → `< > view` → `c/p/l/d [ ] kind` → list verbs (`m`, `s`, `space`, `del`) → `F6 next pane`. `AppFooterStatus` keeps the highest-priority prefix when width runs out (`Widgets/AppFooterStatus.py:480-560`), so at 120 columns the kind chip survives in list, rail and strip contexts and `s` / `F6` fall back to F1. Named exceptions:
   1. A chip that resolves the current state (`ctrl+s <verb>`, `esc cancel edit…`, `del delete N marked`, `m unmark`, `esc clear marks`, `esc clear search`) goes directly after the state chip, or after Enter (L3, L4, S2).
   2. When item verbs are shown, the create chips (`n`, `i`) follow the kind chip (L1-L3, L5, L6); otherwise they precede it (L0, R1).
   3. In the entries and conversations lists (E1, E2) and the Lore list (L5), the context's own verbs (`n new entry`, `space`, `del`) precede `/`. In E1 and E2 the filter chip precedes the esc chip, and `< > view` (and in E1 `t try it`) follows the esc chip, so at 120 columns E1 and E2 drop the kind chip (and E1 `t try it`) to F1.

   The pinned lines below are normative; this order is the default that generates them.
3. **Typing swap** (the Library rule, `LS:4741-4835`). In a text field:
   - drop printable chips and keep `esc`, `enter`, `ctrl+j`, `F6`, and `ctrl+s` only inside an editor region (item 6);
   - lead with `typing in field`, then the field's own `enter` / `ctrl+j` chips when it has them (T2-T4), then `esc leave field`;
   - park the printable verbs that will be armed **where rung 3 lands** in one `after esc: …` chip.
4. **Cheap.** Re-register on `DescendantFocus` only when (context, Enter label, availability tuple) changes. Never recompose; no workers.
5. **Words.** Rail kinds are "kind" and work-pane modes are "view" (R23). Keys are spelled as typed (`esc`, `del`, `ctrl+s`).
6. **Ctrl+S chip** (R41). It appears only where Ctrl+S will press a visible, enabled button, and its label is that button's verb (`ctrl+s save`, `ctrl+s save pack`, `ctrl+s update entry` / `ctrl+s add entry`, `ctrl+s update rule` / `ctrl+s add rule` for dictionaries, `ctrl+s create lore book`, `ctrl+s create chat dictionary`). In an editor's strip or buttons it is the primary verb and leads (S2); while typing it follows `esc leave field` (T1, T5). A stray or blocked press shows a state chip instead (L6).

Footer lines per context (120 columns, then 160; generated with retain-prefix truncation; verify live before pinning). The shipped footer (`Widgets/AppFooterStatus.py`) re-adds `F6` to the compact global cluster whenever the context's own `F6 next pane` chip is truncated, so a live capture can differ from these lines in that cluster (MK1 shows that form).

**L0 · list, nothing loaded (arrival)**

```text
  enter open | / search | n new | i import | c/p/l/d [ ] kind | m mark | s sort | F6 next pane | F1 · Ctrl+P · Ctrl+Q   
  enter open | / search | n new | i import | c/p/l/d [ ] kind | m mark | s sort | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                         
```

**L1 · list, cursor on the loaded row**

```text
  enter go to card | o chat now | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | F1 · Ctrl+P · Ctrl+Q 
  enter go to card | o chat now | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | s sort | F1 help · Ctrl+P palette · Ctrl+Q quit              
```

**L2 · list, cursor on another row**

```text
  enter open | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | s sort | F1 · Ctrl+P · Ctrl+Q           
  enter open | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | s sort | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                  
```

**L3 · list, search active**

```text
  enter open | esc clear search | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | F1 · Ctrl+P · Ctrl+Q 
  enter open | esc clear search | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit        
```

**L4 · list, 2 rows marked**

```text
  2 marked | del delete 2 marked | m unmark | esc clear marks | enter open | c/p/l/d [ ] kind | F1 · Ctrl+P · Ctrl+Q    
  2 marked | del delete 2 marked | m unmark | esc clear marks | enter open | c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit           
```

**L5 · list, Lore books**

```text
  enter open | e entries | t try it | space on/off | / search | c/p/l/d [ ] kind | n new book | F1 · Ctrl+P · Ctrl+Q    
  enter open | e entries | t try it | space on/off | / search | c/p/l/d [ ] kind | n new book | m mark | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit  
```

**L6 · list, after a stray Ctrl+S** (the state chip lasts until the next key or focus change)

```text
  ctrl+s: nothing to save here | enter go to card | o chat now | e edit | t test | / search | F1 · Ctrl+P · Ctrl+Q      
  ctrl+s: nothing to save here | enter go to card | o chat now | e edit | t test | / search | c/p/l/d [ ] kind | n new | m mark | F1 · Ctrl+P · Ctrl+Q          
```

**R1 · rail, Characters row**

```text
  enter open | / search | n new | i import | c/p/l/d [ ] kind | F6 next pane | F1 · Ctrl+P · Ctrl+Q                     
  enter open | / search | n new | i import | c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                                           
```

**R2 · rail, You row**

```text
  enter open Settings › | / search | c/p/l/d [ ] kind | F6 next pane | F1 · Ctrl+P · Ctrl+Q                             
  enter open Settings › | / search | c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                                                   
```

**S1 · strip, character Card**

```text
  o chat now | e edit | t test | ←→ view | ↑↓ scroll | esc to list | c/p/l/d [ ] kind | F1 · Ctrl+P · Ctrl+Q            
  o chat now | e edit | t test | ←→ view | ↑↓ scroll | esc to list | c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                   
```

**S2 · strip, character Edit, unsaved**

```text
  ctrl+s save | esc cancel edit… | t test draft | ←→ view | c/p/l/d [ ] kind | F6 next pane | F1 · Ctrl+P · Ctrl+Q      
  ctrl+s save | esc cancel edit… | t test draft | ←→ view | c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                            
```

**B1 · Chat now focused**

```text
  enter chat now | esc to list | F6 next pane | F1 · Ctrl+P · Ctrl+Q                                                    
  enter chat now | esc to list | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                                                                          
```

**T1 · typing in an editor field** (character or persona Edit, the visual-identity section of Look, Tool policy, Options; in Look's pack section the chip reads `ctrl+s save pack`)

```text
  typing in field | esc leave field | ctrl+s save | after esc: < > view  c/p/l/d [ ] kind | F1 · Ctrl+P · Ctrl+Q        
  typing in field | esc leave field | ctrl+s save | after esc: < > view  c/p/l/d [ ] kind | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit               
```

**T2 · typing in the items search**

```text
  typing in field | enter go to list | esc leave field | after esc: c/p/l/d n i | F6 next pane | F1 · Ctrl+P · Ctrl+Q   
  typing in field | enter go to list | esc leave field | after esc: c/p/l/d n i | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                         
```

**T3 · typing in a Try-it sample**

```text
  typing in field | enter run | ctrl+j new line | esc leave field | after esc: < > view | F1 · Ctrl+P · Ctrl+Q          
  typing in field | enter run | ctrl+j new line | esc leave field | after esc: < > view | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                 
```

**T4 · typing in the test-chat box**

```text
  typing in field | enter send | esc leave field | after esc: o < > view | F6 next pane | F1 · Ctrl+P · Ctrl+Q          
  typing in field | enter send | esc leave field | after esc: o < > view | F6 next pane | F1 help · Ctrl+P palette · Ctrl+Q quit                                
```

**T5 · typing in an entry-editor field** (lore or dictionary; dictionaries: rule; the after-esc chip has a short form at 120)

```text
  typing in field | esc leave field | ctrl+s update entry | after esc: enter n < > c/p/l/d | F1 · Ctrl+P · Ctrl+Q       
  typing in field | esc leave field | ctrl+s update entry | after esc: enter edit entry  n new entry  < > view  c/p/l/d [ ] kind | F1 · Ctrl+P · Ctrl+Q         
```

**E1 · entries list (side by side; dictionaries: rule)**

```text
  enter edit entry | n new entry | space on/off | del delete… | / filter entries | esc to list | F1 · Ctrl+P · Ctrl+Q   
  enter edit entry | n new entry | space on/off | del delete… | / filter entries | esc to list | t try it | < > view | F1 help · Ctrl+P palette · Ctrl+Q quit   
```

**E2 · conversations list (Chats view)**

```text
  enter preview | o continue in Console | / filter chats | esc back to card | < > view | F1 · Ctrl+P · Ctrl+Q           
  enter preview | o continue in Console | / filter chats | esc back to card | < > view | c/p/l/d [ ] kind | F1 help · Ctrl+P palette · Ctrl+Q quit              
```

**What the set guarantees:**
- Arrival (L0) teaches `/`, `n`, `i`, the kind keys, and `m`/`s` (RP-034).
- Chat now has a visible key once a row is loaded (L1, RP-035).
- A hidden filter announces its exit (L3, RP-021).
- Marks announce their target before `del` (L4, RP-039).
- A focused button arms no letters (B1, RP-033).
- Every typing context names its way out (T1-T5, RP-026).
- Ctrl+S is named only where it saves, with the verb of the button it presses (S2, T1, T5); a stray press says so (L6).

### 4.9 Contextual F1 help (RP-036)

`action_show_workbench_help` reuses `WorkbenchHelpPanel` with `shortcut_groups` (`UI/Workbench/help.py:15-57`). It replaces the generic "PersonasScreen Shortcuts" dump (`app_lifecycle.py:1688-1702`, `app.py:5646-5658`).

- **Title:** `Roleplay keys — <kind> · <focus surface> (<loaded item> open)`.
- **Groups:**
  - "Now (<surface>)": the full footer set, including chips the width dropped;
  - "Find and move";
  - "Escape (never leaves Roleplay)": prose;
  - notes, including the letter-scope rule and where Ctrl+S works (R41).
- **Truthful:** every key row passes `check_action` in the focus F1 was opened from (F1 ⊇ footer). F1 also groups the three previous/next pairs (`[ ]` kind, `< >` view, ←→ on the strip).

```text
╭──────────────────────────────────────────────────────────────────────────╮
│                                                                          │
│  Roleplay keys — Characters · list (Captain Isolde Varga open)           │
│  Now (Characters list):                                                  │
│    enter: go to the card        o: chat now (opens Console)              │
│    e: edit    t: test chat      < >: previous/next view                  │
│    m: mark    s: sort           del: delete row or marked (asks)         │
│    n: new…    i: import…        /: search characters                     │
│  Find and move:                                                          │
│    / or ctrl+f: search this kind      F6 / shift+F6: next/prev pane      │
│    c p l d: Characters, Personas, Lore books, Chat dictionaries          │
│    [ ]: previous/next kind            tab: stays inside Roleplay         │
│  Escape (never leaves Roleplay):                                         │
│    field > leave it · view > back to Card · work pane > list             │
│    list > clear search, then marks > rail · rail > nothing               │
│  Letters act on the rail, a list, a grip or the view strip only:         │
│  never while typing, never on a button. More: User Guide > Roleplay.     │
│  ctrl+s saves the editor you are in; anywhere else it does nothing.      │
│                                                                          │
│        [ Close ]                                                         │
╰──────────────────────────────────────────────────────────────────────────╯
```

### 4.10 Mouse targets

| Target | Minimum | Behaviour | Fixes |
|---|---|---|---|
| Grip | 5 × full pane height | Toggles its pane; tooltip names it ("Open Characters list") | RP-032 |
| Rail row | full rail width × 1-2 rows | Click = Enter | RP-008 |
| Items row | full width × row height | Click loads (= Enter) and focuses the list; it never marks | RP-033 |
| Gutter col 2 of an items row | 1 × row height (tooltip "Click to mark") | Toggles the mark (the mouse path to bulk actions) | RP-039 |
| View chip | label + 1-2 cells × 1 | Switches view | RP-063 |
| Action bar, title row, More actions | label + 2 × 1; ≥1 cell between neighbours; Delete only inside More actions, with danger styling | — | RP-020, RP-039 |
| Search clear `x` (shown when the box has text) | ≥3 cells | Clears the box and the filter together | RP-009 |
| Scrollbars | one per pane body | — | RP-054 |

No interactive target is narrower than 3 cells, except scrollbars and the mark gutter (which has a keyboard twin, `m`). Nothing changes size on hover or focus. A click moves focus with it.

### 4.11 State and memory

**Model.** `PersonasWorkbenchState` (`PW/personas_state.py`) stays the serialized carrier and gains the dataclasses below **in the same module**. That module is UI-ready-resident, so this costs zero census modules, but **it must import no new module** (UI-ready is at 1,032 / 1,033).

```python
@dataclass(slots=True)
class ModeMemory:                       # one per kind; session (save_state)
    selected_id: str | None = None      # the loaded item (row shows ▸)
    cursor_id: str | None = None
    search: str = ""                    # restored INTO the box (RP-021)
    sort_key: str = "name_asc"          # also a persisted preference (§2.11)
    tag_filter: str | None = None
    anchor: tuple[str, int] | None = None   # (row id, rows from the viewport top)
    view: str | None = None             # last work-pane view used for this kind
    # marks: kept on the live screen per kind for the visit; NOT serialised (R16)

@dataclass(slots=True)
class ItemMemory:                       # per (kind, id); LRU of 16 per kind; session
    view: str
    focus_id: str | None
    entry_anchor: "EntriesAnchor | None" = None   # books
    conversation_id: str | None = None            # Chats

@dataclass(slots=True)
class RoleplayMemory:
    active_kind: str
    kinds: dict[str, ModeMemory]
    items: dict[tuple[str, str], ItemMemory]
    focus_id: str | None
```

- `switch_mode` stops resetting sort, tag and page (`personas_state.py:54-70`) and swaps `ModeMemory`.
- `_apply_mode` stops clearing the search box, marks and preview (`PS:4459-4549`), and restores them.
- Test-chat turns stay in `save_state["personas_preview"]` (`PS:1752-1767`), keyed by item.
- The work-session phase is **not** saved: a return starts `INACTIVE`.

**Selected vs loaded (ADR-084 generation fence; hardens RP-078/079).**
- `selected` changes at once and drives `▸`. `loaded` (kind, id, version, record) drives the title row, readiness, action bar, Info, tool-policy summary, footer and F1.
- Every load increments `load_generation`. A result applies only if its generation is current **and** the ADR-115 screen generation matches; late results are dropped.
- While `selected ≠ loaded`:
  - the title row reads "Loading Ada…";
  - item actions are disabled with that reason;
  - `o`, `e` and `t` from the list queue behind the load and run only if no newer intent arrived.
- **Save** returns the saved record; `loaded` becomes it (a new version and generation).
- **External change** (a card saved from Console, TASK-32954): reload if clean; with a draft, show "Changed elsewhere · Reload" and never overwrite the draft.

**Transitions and focus postconditions.**

| Event | Guard | Memory | Work pane / session | Focus after |
|---|---|---|---|---|
| Kind switch (`c p l d [ ]`, rail click) | ADR-046 if dirty | snapshot the current `ModeMemory`; restore the target's (**even with no selection**, RP-021) | load the target's `selected_id` (fenced) or show its empty state; **session ends** | the list at the restored cursor |
| Load another row (Enter, click, `e`/`t` from the list) | ADR-046 if dirty | stash the old `ItemMemory`; apply the new | load (fenced) in the remembered view, else the kind default; **session unchanged** | the list (≥92) / the work stage (<92); `e`/`t` → their view's target |
| View switch (`< >`, ←→, chip, `e`, `t`) | none (I8) | `ItemMemory.view` | switch the body (demand-mounted on first use); `e`/`t`/authoring chips **start** the session if inactive | the strip for `< >`/←→; the view's target for `e`/`t` |
| Save | — | — | re-render from the saved record; `Saved 15:23`; Cancel → Done; **session unchanged** | unchanged |
| Cancel / Discard / Done | dialog if dirty | the draft is dropped | the default view from the loaded record; **session unchanged** | the invoker (list row or strip) |
| Start row | ADR-046 if dirty | — | unload; the landing; **session ends** | the landing's `Import…` |
| `‹ Kind` crumb / Escape rung 7 | — | — | **session ends**; panes return per preference | the list at the loaded row |
| Delete (`del`, More actions) | the delete modal | ids leave marks and memory | the neighbour at the same index is loaded, never row 0; if the kind is now empty, its kind empty state (R25); **the session ends** | the list at that neighbour, or the empty list with its one-line message (§4.3) |
| Entry action (add, update, toggle, delete) | per action | the entry anchor is kept (A's rule) | — | the acted-on entry, or its neighbour |
| A pane closes (grip, resize, session start) | — | requested changes persist; automatic ones do not | — | focus inside it moves to its grip |
| A pane is reopened manually | — | — | the session gains that priority | the pane's last-focused control, else its F6 target |
| Session start (Edit at 120, which closes both panes) | — | transient | — | **the editor field** (explicit intent beats evacuation, through the focus-intent generation) |
| Any transition that hides the focused widget's ancestor (view switch, Try column hide, drill-in close) | — | — | — | a **named** target: the strip, the list row, or the body's first control; never `None` |
| Leave Roleplay | ADR-046 aggregate | `save_state(RoleplayMemory)` | — | — |
| Return | — | restore everything | reload the loaded item (fenced); `INACTIVE` | §4.3 |
| Runtime source change | ADR-046 | reset all memory | landing; session ends | the list |

**Async focus never steals.** Load completion, first-use mounts, restore and save completion may move focus only if:
- the focus-intent generation is unchanged since they were scheduled (every key and click bumps it);
- and `_focus_steal_blocked()` (`PS:16371-16385`) allows it.

**Fixes:** RP-030, RP-021, RP-031 (with §2.11), RP-064, RP-085, RP-078/079 (hardening).

### 4.12 Single-letter safety, focus visibility, contrast

- **Scope.** Enforced by `check_action` from the projection, not by declaration order. A letter on a button, reading body or form control is inert (RP-033's "spell" and Space-on-Duplicate cases).
- **Misfires are cheap.** View switches are lossless; `space` toasts with an undo cue (R33); `del` always confirms and names its targets.
- **`o` is the one letter that leaves the screen.** It is armed in the list only when the cursor row is the loaded row (Q4). A stray `o` opens a Console chat but sends nothing, and Ctrl+4 returns with state restored.
- **The persona-buddy overlay's `c`** remains a documented shadow (ADR-152 decision 2).
- **Focus visibility, with existing tokens only and id-scoped rules in the lazy sheet:**

  | Surface | Focus cue (shape first) | Selected/active cue (independent of focus) |
  |---|---|---|
  | Items, entries and conversations lists | `thick $ds-action-focus` edge; cursor row `$ds-focus-bg`/`$ds-focus-fg`, bold underline, **only while focused** | `▸` + bold on the loaded row |
  | Rail rows | `border-left: thick $ds-action-focus` replacing the padding (`css/features/_library.tcss:433-437`) | `▸` + the selected background |
  | View strip | the strip edge; `$ds-focus-bg` on the active chip while focused | bold + underline + the active marker |
  | Text fields | unchanged input focus box | — |
  | Grips | the house grip focus (`reverse`) under neutral classes | arrow direction |
  | Buttons | unchanged (D10 is app-wide); mitigated: no arrival or F6 target is a button, and the footer names Enter | — |

  The thick edge is a focus cue that replaces padding, so it causes no shift. DESIGN.md:361 forbids side stripes as **state** accents. If the design system reads :361 as covering focus edges too, that is a ruling for Library and Roleplay together.
- **Not colour alone.** On/off is a word on every row. Marks are `✓` plus "N marked". Unsaved is a word in three places. Readiness is a model name or reason. Disabled actions show their reason as text.
- **ASCII.** Every glyph goes through `resolve_glyph`. B1 adds the missing map entries in `Widgets/glyph_fallback.py:30-60` (`›`→`>`, `‹`→`<`, `…`→`...`, `·`→`-`, `—`→`-`, `⇥`→`>|`, `→`→`->`, `←`→`<-`, `↑`→`^`, `↓`→`v`, `×`→`x`; the map has `✕` but not `×`), pinned by `Tests/Chat/test_console_glyphs.py`. An ASCII render test covers the frame.
- **Contrast.**
  - Cursor-row text ≥4.5:1 on `$ds-focus-bg`; every focus edge ≥3:1 against the adjacent surface, in every shipped theme.
  - Roleplay's search placeholder (today 3.52:1) and inactive view chips (3.81:1) get AA tokens in Roleplay's id-scoped rules (B2, B5b). `Tests/UI/test_theme_contrast.py` is extended to the new rules.
  - An app-wide token lift is the design system's (D10).

**Fixes:** RP-033, RP-062, RP-063, RP-065 (mitigation), RP-071, RP-091 (state cue).

### 4.13 Deviations recorded (interaction)

| From | This design | Why |
|---|---|---|
| RP-035: "put Chat now first in the work-header F6 target" | Chat now is never an F6 or arrival target; it gets `o` (B5c) | I2; invisible button focus (RP-065) would hide that Enter leaves |
| RP-026: "in editors keep Escape = cancel-with-guard" | Escape leaves the field first, then cancels | one grammar; no reflexive dialog while typing |
| RP-037: two choices for an explicit Cancel | the one three-choice dialog everywhere, labelled **Stay** | one grammar (H4); preserved wording |
| RP-039: "marks cleared on mode switch" | marks survive kind switches for the visit; cleared on leave, by `esc` or Clear | lossless kinds (RP-030); disclosure keeps the safety |
| RP-062: "avoid side-stripe accents" | the shipped thick focus edge | the house focus convention (task-31983, task-32359) |
| ADR-152 wording "list/button focus" | rail, lists, grips and the strip; never buttons | D8 = B; ADR text amended (§5.9 G7) |
| D5 "per-mode memory of selection, search, sort and scroll" | sort persists as a preference; the rest is session-only | stale positions across restarts (Q6) |
| Try-it run key Ctrl+Enter | Enter runs; Ctrl+J newline; Ctrl+Enter alias | terminal-safe keys; Console composer grammar |
| ADR-031 rule 2 (no `ctrl+s`) | Editor-scoped Ctrl+S in every Roleplay editor, pressing the visible commit button; a no-op with a footer hint elsewhere | owner ruling Q5; recorded as a dated ADR-031 amendment, not a silent deviation (G9a) |
| Today's Ctrl+S saves from any focus while an editor is displayed (`PS:16303-16311`) | Saves only with focus inside the editor region; elsewhere the L6 chip | owner ruling Q5: Ctrl+S mirrors the focused editor's visible Save |

### 4.14 Mockups: degrade (off the design centre)

**MK9a · 80x24 · XS list stage** (rail grip · list 32 · grip · work 38). This is reached by arrival, Esc rung 7, the crumb or `/`; Enter moves to the work stage (`– | – | 70`). The character strip fits at step 4 (tight chips, short labels); a dictionary strip would fold trailing chips into `More ▾` here (step 5, §3.3).

```text
                                      ╭─────────────╮                           
   ⌃1 Home   ⌃2 Console   ⌃3 Library  │ ⌃4 Roleplay │   ⌃5 Watchlists    More ▾ 
────────────────────────────────────────────────────────────────────────────────
 Roleplay   Characters                                                    Local 
  N  ╭─ Characters 28 ──────────────╮  C  ╭────────────────────────────────────╮
  a  │ ▎Search… /  Name  Tag  + New │  h  │ ‹ Captain Isolde Varga    saved 2d │
  v  │   Ada (Debugging Partner)    │  a  │ [ Chat now ]  gpt-4.1-mini  More ▸ │
     │   ARIA-7                     │  r  │  ▸Card Edit Chats World Test Info  │
     │   Brother Anselm             │  a  │ Captain Isolde Varga commands the  │
     │▸  Captain Isolde Varga       │  c  │ deep-survey frigate Meridian's     │
     │   Chef Auguste        1 chat │  t  │ Wake, a patched-together vessel    │
     │   Coach Pemberton            │  e  │ that has outlived three refits.    │
     │   Default Assistant  4 chats │  r  │                                    │
     │   Detective Rosa Calderón    │  s  │ First message                      │
---> │   Dungeon Master Vex         │     │ "Welcome aboard. Mind the cat; he  │
     │   Grumpy Barista Bot         │     │ outranks you."                     │
     │   Hana Kobayashi             │     │                                    │
     │   Kestrel                    │     │                                    │
     │   Lady Evangeline Thornwood  │     │                                    │
     │   Luna — Night Mar…ne Teller │<--- │                                    │
     │   Marla the Ferrywoman       │     │                                    │
     │   Mira Okonkwo               │     │                                    │
     ╰──────────────────── 4 of 28 ─╯     ╰─────────────────────────── ▾ more ─╯
  enter card | o chat now | / search | F6 next pane | F1 · Ctrl+P · Ctrl+Q      
```

**MK9b · 60x24 · below 64, work stage** (two grips + work 50; no new single-stage machinery, R38). The crumb or Esc returns to the list stage (list 42 + a hidden residual strip). The header names the item only on the list stage.

```text
                                   ╭─────────────╮          
   ⌃1 Home  ⌃2 Console  ⌃3 Library │ ⌃4 Roleplay │   More ▾ 
────────────────────────────────────────────────────────────
 Roleplay   Characters                                Local 
  N    C  ╭────────────────────────────────────────────────╮
  a    h  │ ‹ Characters  Captain Isolde Varga    saved 2d │
  v    a  │ [ Chat now ]  gpt-4.1-mini              More ▸ │
       r  │  ▸Card  Edit  Chats  World  Test chat  Info    │
       a  │ Captain Isolde Varga commands the deep-survey  │
       c  │ frigate Meridian's Wake, a patched-together    │
       t  │ vessel that has outlived three refits and two  │
       e  │ mutinies.                                      │
       r  │                                                │
       s  │ First message                                  │
--->      │ "Welcome aboard. Mind the cat; he outranks     │
          │ you."                                          │
          │                                                │
          │ Personality                                    │
          │ Dry, precise, loyal, quietly haunted.          │
     ---> │                                                │
          │                                                │
          │                                                │
          ╰─────────────────────────────────────── ▾ more ─╯
  esc back to list | o chat now | F1 · Ctrl+P · Ctrl+Q      
```

---

## 5. Delivery: slices, testing, documentation, governance, risks

### 5.1 Slice plan

```text
 B0 shared primitives + ADR ──┬───────────────────────────────────────────────────────────────┐
 B1 one-row header (∥ B0) ────┤                                                               │
                              ▼                                                               ▼
 B2 items list ─► B3 keyboard ─► B4 Try views ─► B5a Chats ─► B5b work header + fold ─► B6 rail ─► B7 frame ─► B8 Try column
   ▲ A 33787/33788 (absorbable)               ▲ A 33781, 33789, 33791 (B5b)  │                     │
                                                                             ▼                     ▼
                                    B5c Chat-now promotion ◄── TASK-33621.2 on dev (D13), any time after B5b
 B9 entries + new draft domains  ◄─ A 33781-33783, 33785, 33786 (hard); B5b; B7 for the split at 160
 B10 landing + Start row          ◄─ B3 (arrival); B6 (the rail B10 adds Start to)
 B11 editors                      ◄─ B5b; A 33784, 33789
 B12 User Guide consolidation     ◄─ all (each slice also ships its own guide delta)
```

- **Primary order:** B0 ∥ B1 → B2 → B3 → B4 → B5a → B5b → B6 → B7 → B8 → B9 → B10 → B11 → B12.
- **B5c** merges whenever TASK-33621.2 is on dev and its end-to-end test passes (R29, Q11). Nothing else waits for it.
- **Serialisation:** one `PS`-touching slice in review at a time. B0 touches no `PS`.
- **Why the fold precedes the rail.** Adding a 28/35-column rail while the Inspector stays would leave the work pane at 34 columns at 120 (below its own 40 minimum) and 60 at 160, for at least one release. With the Inspector folded first (B5b), the rail costs nothing (B6 interim floors ≥58 / ≥78 / ≥108). Buddy needs a home before the Inspector goes (B5b's interim chooser).

| Slice | One line | Size | Gate | Main RP ids |
|---|---|---|---|---|
| B0 | ADR + shared geometry, grip and rail primitives; no visible change | M | Q10 (census +1, measured) | enabler: 017, 024, 032, 066 |
| B1 | One-row inline header (aggregate predicate, blocked chip); lazy CSS sheet; glyph-map entries; DESIGN.md header amendment | S | — | 007, 067, 038, 029, 052, 071 |
| B2 | Line-based items list (DB windows), one-row toolbar, marks bar + gutter click, delete modal, New chooser, one Import picker, search-aware empty states | **L (XL risk: test migration)** | D14-A, or B after the dry-run census | 008, 039, 080, 092, 093, 094, 095, 062, 068, 070, 042, 043, 022, 028 |
| B3 | Keyboard foundations: arrival, Tab region, Escape rungs, letter scope, `/`, footer + F1 projection; the ADR-031 Ctrl+S amendment (character and persona Edit) | M | — | 024, 026, 033, 034, 035, 036, 063, 065 (mitigation) |
| B4 | Test chat / Try it as full-height views; RP-081's content; Try-it TextArea keys | M | — | 010, 006, 081, 045 |
| B5a | Chats view: `PersonasCharacterChatsPanel`, conversation methods, deep link, one-row toolbar | M | API agreed with TASK-22988/31243 | 049, 050, 018 |
| B5b | Work header: title row, view strip, action bar, More actions, Info, Inspector deleted, one guard dialog, generation fence, item memory | L | A 33781, 33789, 33791 | 018, 019, 020, 037, 052, 054, 067, 030, 025, 027, 044, 047, 015 |
| B5c | Chat-now promotion: `o` = Chat now, the provider end-to-end test | S | **D13: TASK-33621.2 on dev** | 035, 019 |
| B6 | Rail + per-kind memory + scent aggregates + Roleplay-owned vocabulary strings | L | — | 029, 028, 041, 042, 011, 021, 030, 055, 015, 043 |
| B7 | Frame: shell, grips, work sessions, profiles, `[roleplay.reader]`, tiers, dynamic F6 | L | A 33781; ADR-011 evidence before deleting the legacy path | 007, 017, 031, 032, 064, 066, 016, 019 |
| B8 | Try column at ≥200 (authoring views; per-kind preference, Q2 defaults) | S | — | 010, 006, 081 (XL) |
| B9 | Entries split/drill-in, filter, count, anchor; entry and Options draft domains (ADR-046 amendment); New-book draft; Ctrl+S in the entry editor and Options | L | A 33781-33783, 33785, 33786 | 006, 056, 091, 003, 014, 040, 005, 015 |
| B10 | Landing (with the Continue block), Start row, kind empty states | M | G5 amendment lands with it | 055, 021, 024 |
| B11 | Editors: growth caps (tokens), sections, Jump row, ✨ removal, field-label errors, save feedback; Ctrl+S in persona Look and Tool policy | M | A 33784, 33789 | 016, 076, 085, 038, 053, 044, 071, 027 |
| B12 | User Guide consolidation | S | — | 060 |

### 5.2 Slice cards

Every card shares the definition of done in §5.4.

#### B0 · Shared frame primitives and the ADR (no visible change)

- **Scope.**
  - The new ADR (§5.9 G1).
  - `Widgets/adaptive_pane_shell.py` (R17).
  - Library compatibility: subclass + same-object aliases; `library_emergency_return.py` subclasses `StageReturnBar`; `library_rail.py` re-exports the input.
  - Neutral aliases in `ARS`.
  - `BaseAppScreen.TAB_REGION` and `arrival_focus_target()` (opt-in, default `None`).
  - `css/components/_adaptive_panes.tcss` (boot), deleting `css/features/_library.tcss:109-176`.
  - The `panes` family in `css/patterns.json` + `backlog/docs/component-patterns.md` + `Widgets/pattern_gallery.py`.
- **Touches Library compatibility modules** (`library_adaptive_reader_shell.py`, `library_emergency_return.py`, `library_rail.py`, `_library.tcss`). **Never touches** `PS`, `LS`, `Library/library_shell_state.py`, `UI/Library_Modules/screen_helpers.py` (all in #2862).
- **Acceptance.**
  - The resolver's golden outputs are identical over a normalised-preference grid.
  - `LibraryPaneVisibilityChanged is PaneVisibilityChanged`.
  - Library pure and shell suites pass unchanged except the moved CSS-ownership pins (`Tests/UI/test_library_adaptive_reader_shell.py:594-700`).
  - Tab is unchanged on every `TAB_REGION is None` route.
  - Live: Library Notes at 120x36 and 160x45 on a fresh launch with no earlier Library visit paints its grips styled.
- **Tests.**
  - `Tests/UI/test_adaptive_pane_shell.py`;
  - `Tests/UI/test_destination_rail_row.py` (`fit_rail_row_label` at **24, 31, 35** cells, the bordered rail; the count is never clipped; the fallback order of §1.4.3);
  - alias identity and the golden comparison in `Tests/Library/test_library_adaptive_reader_state.py`;
  - `Tests/UI/test_base_app_screen_tab_region.py`.
- **Gates.**
  - Boot CSS +≈1.1 KB under the 608,090 ceiling.
  - Broad selectors stay 274 (class subjects only).
  - UI-ready stays 1,032.
  - Pre-import: **+1 module, measured**, then shed or deferred, else an owner-signed ledger row (Q10).
  - `Tests/Packaging/test_config_import_closure.py` passes.
- **Rollback:** revert.

#### B1 · One-row header (D7 = A)

- **Scope.**
  - The §1.3 header, with the subtitle carrying `› <item>` as an interim until B5b.
  - New `PM/roleplay_frame_state.py` (pure; header state first).
  - `roleplay_has_unsaved_work()` (R24) drives the chip.
  - The blocked-destination chip.
  - The lazy `css/features/_roleplay.tcss` + `ScreenOwnedSplit` (R18) + `PersonasScreen.CSS_PATH`.
  - Glyph-map entries (§4.12).
  - Delete the dead selector items (`AT:357, 408-409, 664-665, 689-690, 804-805`).
  - `#personas-purpose` and `#personas-mode-strip` **stay until B6** (ADR-015:34).
  - The DESIGN.md:328-330 inline-header amendment (G12).
- **Acceptance.**
  - The header is exactly 1 row at 80x24, 120x36, 160x45 and 220x55, with the subtitle visible at ≤24 rows.
  - First list item on row 19.
  - The chip shows iff the aggregate predicate is true (parity test against `_aggregate_roleplay_draft_snapshot()`, **not** `has_unsaved_changes`).
  - A long name ellipsises and the kind is never cut.
  - Journey: Edit → type → chip appears → Ctrl+S → chip goes.
- **Re-pin.**
  - `test_personas_workbench.py:548` (y 10 → 6 at 170x50), `:871/8131/8154` (the title and Blocked badge);
  - `#personas-header` (22 refs in 5 files, including `Tests/UI/test_destination_shells.py`);
  - re-check `test_destination_visual_parity_correction.py:997-1007`.
- **Gates.**
  - CSS sources after the destination tour gain +1, staying below 56 (`Tests/Performance/test_ui_latency_guardrails.py:52`).
  - Boot bytes do not rise.
  - `PS` shrinks, to ≤16,436 if B1 merges before #2862.
  - Pre-import **+1** (`roleplay_frame_state.py`), measured (Q10).
- **Rollback:** revert.

#### B2 · Items list (D14 = A)

- **Scope.** §1.5:
  - `RoleplayItemList(OptionList)`, keeping `#personas-library-rows`, with markup-off prompts, density, the gutter (click on col 2 marks) and marks by id;
  - **database windows** of 200 with `add_options`;
  - personas paged until exhausted;
  - the marks bar and the one delete modal (R32);
  - the border-title counts and a one-row toolbar with `#personas-filter`;
  - search-aware empty states with `Also in:` (cross-kind count query);
  - the New chooser (books keep today's create path until B9) and the one Import picker (format sniff, remembered folder; an interim `Import…` in the toolbar until B6);
  - `space`, `m` and `s` moved onto the list widget;
  - `TagFilterPicker` Enter/Down;
  - Duplicate moved interim into the Inspector stack;
  - the placeholder contrast lift;
  - about 6 counted broad rules in `PersonasLibraryPane` re-keyed, with the ratchet lowered.
- **Files:**
  - `PW/personas_library_pane.py`, rewritten in place;
  - `PM/roleplay_frame_state.py`: prompt builder, density, count text;
  - `PS`: paging, `:486`, `:3690`, `:3828-3890`, `:4273-4360` shrink; `count_character_page`, `get_character_page_for_ui` and `PERSONAS_LIBRARY_PAGE_SIZE` stay importable from `PS`;
  - `PW/tag_filter_picker.py`;
  - `PW/personas_inspector_pane.py`;
  - the lazy sheet.
- **Acceptance.**
  - Toolbar ≤2 rows at 120x36 (interim layout) and 1 row at ≥160.
  - First item on row 13 at 120 and row 12 at ≥160 (interim, §5.3).
  - Volume profile (348 characters):
    - `/ isolde ⏎ ⏎` opens the card;
    - scroll to the end, then search and clear: no name is skipped (RP-080);
    - marks survive search and sort (`✓ 3 marked (2 hidden)`);
    - `Personas 114` with Default Narrator present.
  - **End to end:** keystroke → SQL window → painted rows **≤250 ms at 348 and at 5,000 characters**; `set_options` ≤50 ms at 348; **0 widget mounts** per keystroke.
- **Tests.**
  - `Tests/UI/test_roleplay_item_list.py`: the prompt builder (middle ellipsis; markup injection `[b]x[/b]`), density, cursor and anchor restored by id, marks by id, persona exhaustion paging, empty / no-match / `Also in:`, the marks bar, the delete modal naming ≤5 items.
  - `Tests/UI/roleplay_frame_harness.py`:
    - `StyledRoleplayTestApp` built on `Tests/UI/app_factory.py`'s `_build_test_app`, with no cleanup of its own (#2862 adds `drain_created_runtimes`). It consolidates `test_personas_workbench.py:218-269` and `test_personas_dictionaries.py:690-722`.
    - Helpers: `select_roleplay_item`, `highlighted_item_id`, `visible_item_ids`, `item_at(index)`, `list_state()` (empty/recovery), `assert_painted_inside(widget, pane)` against `scrollable_content_region` (`backlog/docs/lessons-testing-evidence.md:120`), and a size-matrix fixture.
- **Re-pin (its own line item; dry-run census first):**
  - 91 `#personas-library-row-*` refs in 12 files;
  - 46 `ListView`-typed `#personas-library-rows` refs in 6 files (≈70 ListView-API uses);
  - `test_personas_library_pane.py` (24);
  - `test_personas_library_pane_paging.py` (8, deleted with paging);
  - `test_personas_library_toolbar_layout.py`;
  - `#personas-library-count` (14 in 4 files) and `#personas-library-pagebar` (7 in 2);
  - `test_personas_dictionaries.py:875-884`.
  - `test_personas_library_rail_focus_outline.py` keeps its rule (pinned, R18).
- **Gates.**
  - The `PersonasLibraryPane` boot segment (1,928 B) does not grow.
  - Pre-import +0.
  - ADR-011 capture (it retires the ListView path and adds a window worker).
- **Fallback.** Choose D14-B before merge if the dry-run census shows the migration cannot ride the slice.

#### B3 · Keyboard foundations (on the current layout)

- **Scope.**
  - The projection in `PM/roleplay_frame_state.py` (§4.2; no separate module).
  - `PS` adopts `TAB_REGION`; arrival goes to the list.
  - `check_action` scoping.
  - The Escape rungs available before the frame: 0, 2, **3**, 5, 7-11; rung 10 → the old mode chips until B6.
  - The footer projection replaces `PS:16455-16526`.
  - F1 → `action_show_workbench_help`.
  - Keys: `/` (pane-scoped), `ctrl+f`, `n`, `ctrl+n`, `i`, `del` (modal).
  - Retire the screen `ctrl+enter`.
  - The chip focus style differs from active.
  - The ADR-152 letter-scope line (G7a).
  - **The ADR-031 amendment (G9a) and `ctrl+s` through `editor_save`** (R41): character and persona Edit come under the focus rule; every other editor follows the "elsewhere" path until B9 or B11 binds it; the stray-press state chip (L6).
  - Link the shell task for the `‹ le` nav fragment (RP-024) and the Library `/` convergence follow-up (R26), both filed with the parent task (§5.13).
- **Acceptance** (§5.7 journeys, keyboard only, at 120x36 and 160x45):
  - After mount and after a Library round trip, `app.focused` is the list inside `#screen-content`.
  - Enter never changes screen; 60 Tabs never reach a `NavigationButton`; 15 Escapes leave the screen stack unchanged.
  - Every advertised key passes `check_action` and changes state; no printable key is advertised while typing.
  - Letters are inert on buttons, text fields and form controls.
  - F1 ⊇ footer.
  - `/ isolde ⏎ ⏎`; Edit → type → `esc` → `esc` → **Stay** restores focus; `l` from the list switches to Lore.
  - Ctrl+S in character and persona Edit presses the visible Save from a field and from a button of the editor. From the list, the mode strip or a read view it changes nothing, is never advertised, and shows `ctrl+s: nothing to save here`.
- **Re-pin.**
  - Footer hints: `test_personas_workbench.py:698, 743, 8291, 8439, 11967, 12688`.
  - Escape: `:12536-12603`. Mode keys: `:12729`.
  - `ctrl+enter`: `:6409-6410, 7556, 7879, 8240`.
  - The `ctrl+s` hint (`PS:16477`) and `action_personas_save` callers.
  - `pilot.press("ctrl+s")` tests: `test_personas_editor_save_in_place.py:172, 219, 241, 321, 411` and `test_personas_workbench.py:1480, 12631-12681, 12942, 12967`. Census them first and focus a field of the editor before the press where the test assumes today's focus-agnostic save (§4.13).
  - `test_screen_footer_hints.py:599` stays green.
  - **Census first:** run the Roleplay suites with the new arrival focus on a scratch branch to list tests whose early `pilot.press` relied on nav-bar focus.
- **Gates.** A focus change causes 0 recomposes and starts 0 workers. Re-registration is skipped when nothing changed. Pre-import +0.

#### B4 · Test chat and Try it as full-height views (D2, view half)

- **Scope.**
  - `#personas-work-body` (`-try-off` / `-try-mode`) with an eager `#personas-try-column`. The three Try widgets move into it **at compose time** (from `PS:1620, 1710-1715`).
  - The §3.8 layouts, including **RP-081's content**: regrouping, `budget full: next entry needs N tokens`, and 2-line collapse.
  - The Try-it `TextArea` subclass (Enter runs, Ctrl+J newline).
  - Entry is by `t` or the interim switches (`#personas-preview-toggle` relabelled `Test chat ▸`; a compact `#personas-try-toggle` at the right end of `#personas-mode-strip`).
  - Adds Escape rung 6 for Test chat / Try it (return to the default view; focus the interim toggle until B5b's strip).
  - The preview toolbar's counted rule is re-keyed and the ratchet lowered.
- **Acceptance.**
  - At 120x36, 160x45 and 220x55, after 3 mock replies, the input and Send are painted inside the pane.
  - The transcript has ≥10 rows at 120x36 on the interim layout (≥20 after B7).
  - With a 3-line sample, the Try-it summary is visible at 120x36 and 160x45.
  - `#personas-preview-pane` keeps its identity across toggles and kind switches, and turns survive leave and return.
- **Re-pin:** `#personas-preview-toggle` (8 refs in 3 files), the lore "(soon)" switch, Try-it visibility per view. `test_personas_workbench.py:10183` and `test_personas_deferred_center_views.py` stay unchanged.
- **Gates:** a toggle mounts 0 widgets; the Try widget segments (952 B / 951 B) do not grow; ADR-011 capture.

#### B5a · Chats view

- **Scope.**
  - `PersonasCharacterChatsPanel` in `PW/personas_conversation_transcript_widget.py`: the conversation list + render lock (`PW/personas_inspector_pane.py:571-952`), the transcript host, and the one-row toolbar with `Continue in Console`.
  - The conversation methods move (§3.7). `PM/personas_conversations_controller.py`'s 8 references are retargeted.
  - The deep link switches to Chats before focusing the list (`character_conversation_navigation.py:32, 80`).
  - The debounced Chats filter, with no `BEGIN IMMEDIATE` per keystroke (PERF-22 AC#4, second half).
  - Retire `PS:1291, 1304` (broad rules) and lower the ratchet.
  - The Chats view is reached by the interim card button until B5b's strip lands.
  - Adds Escape rung 4 for the transcript drill-in and rung 6 for Chats (back to Card; focus the card's Conversations button until B5b's strip).
- **Acceptance.**
  - The list is full height (no 10-row cap), with tail paging kept.
  - Split at `w` ≥ 100, drill-in below.
  - The deep link lands on the conversation.
  - Re-pin `test_roleplay_character_conversation_browse.py` (5).
- **Dependencies.** The panel API is agreed with the owners of TASK-22988 and TASK-31243 before merge.

#### B5b · Work header and Inspector fold

- **Scope.**
  - `git mv PW/personas_inspector_pane.py PW/personas_work_pane.py` (`PersonasWorkHeader`, `PersonasInfoPanel`).
  - The §3.1-§3.7 title row (R21), view strip (R22, roving focus), action bar `#personas-work-actions`, More actions, Info.
  - Persona Look/Tool policy via `show_section`; the lore/dictionary tab rows hidden; Settings relabelled **Options**.
  - The card's Edit and Conversations buttons removed. `#personas-detail-stack` → `Vertical`.
  - **Chat now moved unchanged** (no `o`).
  - Binds `e` (Edit / Entries for the cursor row or the loaded item) and `<` `>` (wrapping view cycle) with the view strip, armed per §4.2.
  - Ctrl+S is otherwise unchanged here. The character and persona Edit Ctrl+S regions extend to the new title row and view strip (R41). In Look and Tool policy it presses only the persona commit bar (`#personas-editor-save`), from that bar, the title row or the strip, while the bar is shown; inside the visual-identity, pack and rules sections it is the "elsewhere" no-op until B11 binds their saves (§4.6).
  - "Add to Console message", `u` and the preview's "Send to Console draft" hidden behind one constant (ADR-031 rule 4).
  - `_run_guarded` routed through `roleplay_has_unsaved_work()`, with the one dialog (Save and continue · Discard and continue · **Stay**).
  - The generation fence; item memory.
  - Escape rung 1, and rung 6 extended to World, Used by, Info and History.
  - The interim `Buddy ▸` chooser.
  - List 1fr / work 3fr (interim).
  - The Inspector CSS deleted: the `#personas-inspector-pane` items at `AT:410, 666, 681, 691`, and the `#personas-inspector-pane`, `#personas-readiness-console` and `#personas-inspector-pane.personas-workbench-compact-pane` blocks within `AT:709-757`. The inactive-chip contrast lift.
  - The ADR-115 eager-set note (G6).
- **Acceptance.**
  - Work pane ≥86 / 115 / 160 at 120 / 160 / 220 (interim).
  - The primary is in work rows 1-2 at every size.
  - A view switch mounts nothing and never prompts, and a draft survives Edit → Test chat → Edit.
  - More actions closes on Escape, a selection change and a view change.
  - Readiness is never dropped while disabled (at 160x45 with a 20-character name).
  - The strip fits all four kinds at content 46 and 69 with no hidden chip.
  - `e` opens Edit (cast) or Entries (books) for the cursor row or the loaded item, and `<` `>` cycle the views with wrapping (`<` from the default view reaches Info), each only where §4.2 arms it.
  - Ctrl+S from the title row and the view strip of character and persona Edit presses the commit-bar Save; the contexts × editors Ctrl+S test gains the title-row and strip arms.
- **Tests.**
  - Rewrite `test_personas_inspector_pane.py` (60) as `Tests/UI/test_personas_work_header.py` (keeping the CTA labels, gating and disabled reasons).
  - The fence tests.
  - The §3.12 trigger table, parametrised.
  - A projection test that `del` is unarmed on conversations and versions (R31).
- **Re-pin.**
  - `PersonasInspectorPane` (101 refs in 7 files);
  - `test_personas_workbench.py:586-652`;
  - `test_destination_visual_parity_correction.py:997-1007, 1590` (the Watchlists exemption is the precedent);
  - `test_workbench_pane_focus.py:106-126`;
  - `#personas-readiness*` (23 in 4 files) and "Ready to chat in Console" (6 in 2);
  - `#personas-card-edit-character` (7) and `#personas-card-conversations` (2);
  - `#personas-inspector-actions` (8, renamed);
  - `test_persona_policy_rules_editor.py` (4);
  - `test_personas_center_canvas_layout.py`.
- **Gates.**
  - The new widgets' boot segments go into the ledger against the freed 1,898 B (`boot_css_bytes.json:255`).
  - Broad rules fall (Inspector ×3 deleted).
  - ADR-011: about +25 widgets at settle (design estimate); plus the destination tour and `Tests/Performance/test_screen_leaks.py` (Ctrl+4 × 10).
  - Pre-import **+1** (`PM/roleplay_frame_controller.py`), measured (Q10).
- **Dependencies:** A 33781, 33789 and 33791 (hard).

#### B5c · Chat-now promotion (D13-gated)

- **Scope.**
  - The `o` key (strip; list only on the loaded row; conversation rows = Continue in Console).
  - **`Tests/UI/test_roleplay_chat_now_end_to_end.py`:** a real ChatScreen, controller and store on the **default, capture-on** config, stubbing neither the durable trace builder nor the recovery projection. It uses a character with a greeting and a persona, and asserts the provider received **exactly one** request carrying the expected system prompt and greeting.
- **Gate.** TASK-33621.2 merged to dev, and the test green on B5c's head rebased onto it. Live check: the harness's body-logging mock (`gapcheck/mock_llm_body.py` + `mock_body.toml`).
- **Acceptance.** `o` is inert on rows that are not loaded. `/ isolde ⏎ ⏎ o` is 5 keys.

#### B6 · Rail and per-kind memory

- **Scope.**
  - `RoleplayRail` (in `PW/personas_library_pane.py`, census-neutral) with the §1.4 rows.
  - Kind rows `#personas-mode-{kind}` (R15).
  - The counts worker plus the **scent aggregates** (chat counts and last activity, dictionary invalid-regex counts, book entry/off counts), with invalidation hooks: on return to Roleplay, after an entry write, and when Roleplay itself hands off Chat now / Continue in Console (Roleplay-side only; no Console change).
  - The You row (`LEAVES_SCREEN_IDS`). The Buddy chooser moves to its row.
  - Remove `#personas-purpose`, `#personas-mode-strip` and the interim `Import…` / `Buddy ▸`.
  - `MODE_CHIP_ORDER` + `MODE_HOTKEYS` lockstep (R2).
  - `ModeMemory` (in `personas_state.py`); `switch_mode` stops clearing; restore applies the kind even without a selection.
  - The `NAV` context; rung 10 → the rail.
  - Interim layout: rail at the policy width, list 1fr, work 3fr.
  - RP-041's **Roleplay-owned strings**: default names ("Untitled lore book"), toasts, picker titles, tooltips, the guard dialog copy.
  - The DESIGN.md:107, ADR-015:34 and ADR-152 order notes.
- **Acceptance.**
  - All four kinds show counts at 120x36, 160x45 and 220x55, with glosses on line 2 (**22 of 29 rail rows** at 120x36; 25 once B10 adds Start). At 80x24 the glosses drop.
  - A populated list never shows `· 0`.
  - Characters → Lore → Characters restores selection, search text (in the box), sort, anchor, marks and view.
  - Leaving with nothing selected and returning shows the populated list.
  - The You row passes the guard and returns with state restored.
  - `l` → Lore and `d` → Dictionaries (pinned).
- **Tests:** pure `build_roleplay_rail_state`; memory transitions; `LEAVES_SCREEN_IDS` is never an arrival or F6 target.
- **Re-pin:**
  - `test_personas_workbench.py:548/563` (purpose), `:761/838` (tooltips move to the rail rows), `:12729`;
  - `test_destination_shells.py:1306` ("who the ai plays" becomes a gloss);
  - `#personas-purpose` (12) and `#personas-mode-strip` (2);
  - `test_personas_workbench_state.py`.
- **Gates.** One counts worker, no stall >250 ms. Rows are patched, never recomposed. Pre-import +0. Any new index needs a plan pin (CLAUDE.md gotcha 1). ADR-011 capture.

#### B7 · Frame geometry

- **Scope.**
  - `AdaptivePaneShell` (`#roleplay-shell`; rail │ grip │ list │ grip │ work), the profiles of §2.2, and the **work-session reducer** (§2.3), with no donation.
  - `[roleplay.reader]` (§2.11), projected in `load_settings()`; env overrides; off-thread, debounced, equality-guarded writes.
  - Tiers (§2.8): the event-keyed XS stages, and the list-first degrade below 64 with `-residual`.
  - Dynamic F6 with grip substitution; focus evacuation and restore.
  - Drop the outer workbench border (id-scoped).
  - `e`, `t` and the authoring chips start the work session (§2.3).
  - **Delete** the legacy layout:
    - `_sync_responsive_workbench` (`PS:2316-2393`), the rail toggles (`:4427-4457`) and the constants (`:607-609`);
    - the handles' compose (`:1590-1603, 1725-1738`);
    - the remaining personas items of `AT:636-700` (`#personas-workbench`, `:639`) and the `#personas-library-pane` / `#personas-work-area` / `#personas-workbench` blocks in `AT:709-757`;
    - `PS:1171-1197, 1232-1252`.
  - `#personas-workbench` retired (15 refs, R39).
- **Acceptance (resolver-verified, §2.4).**
  - BROWSE 28·32·50 / 35·42·73 / 39·42·129.
  - Session (EDIT) –·–·110 / –·42·108 / 39·42·129.
  - First content row 6, first list item row 7 at every width ≥64.
  - Thresholds 116/120, 92/96, 175/180, 142/146.
  - **Within an active session, cycling every view of one loaded item at 120, 140, 160 and 174 changes no pane width; read views never start one.**
  - `e`, `t` and the authoring chips start the work session; `< >` and the read-view chips never do.
  - **Save keeps the editor at 110 / 108.**
  - A resize sweep 60 → 220 → 60 writes no config and mounts 0 widgets; a grip writes once.
  - Preferred state survives a restart (save → reload → read; `lessons-testing-evidence.md:61, 333`).
  - **The golden table fails if `project_default_library_width` or its constants change** (the recorded coupling).
  - The B7 browse widths are stated in the PR as below B6's interim and today's at 120/160 (Q12).
- **Tests.**
  - A pure profile table at 80, 91, 92, 96, 115, 116, 120, 140, 141, 142, 146, 160, 174, 175, 180, 196, 200, 220, 222, 223, 233, 234, 238 for every posture, including hysteresis and priority.
  - The reducer events.
  - "Controller-built preferences never pass through the normaliser".
  - Shell isolation; painted geometry at five sizes; zero mounts; the no-data-work resize (modelled on `Tests/UI/test_library_shell.py:5264`).
- **Re-pin:**
  - `test_personas_workbench.py:510, 532, 586-652`;
  - `test_workbench_pane_focus.py:106-126`;
  - `test_personas_center_canvas_layout.py`;
  - `test_destination_visual_parity_correction.py:997-1007`;
  - `test_personas_library_toolbar_layout.py` (1 row at all sizes).
- **Gates.** ADR-011 evidence **before** the legacy path is deleted. Boot CSS falls (re-estimated after removing only the personas items from the grouped rules). Pre-import +0.
- **Dependencies:** A 33781 (hard); B6.
- **Rollback:** revert. Older builds ignore `[roleplay]` keys.

#### B8 · Try column at ≥200

- **Scope.** The `-try-column` class, the +TRY / +TRY-E session postures (147 / 158), the per-kind `try_column` preference (Q2 defaults), the `(beside)` chip, `Hide ×` / `Keep beside ›`, the column as an F6 target, and focus evacuation.
- **Acceptance.**
  - At 220x55: character Edit gives body 107 · Try 60 with the rail closed; dictionary Entries gives body 107 · Try 60.
  - Character Card outside a work session gives 39·42·129 with the rail open and no column. With a session already active at 220 (rail closed by the column), Card → Edit → Card moves no pane. The rail moves only at session start and end.
  - 199↔200 and view ↔ column switches keep the preview's identity and its turns.
  - 200 → 195 while editing hides the column and brings the rail back.
  - Exactly one preference write per Hide or Keep.

#### B9 · Entries and the new draft domains (D12)

- **Scope.**
  - §3.9: the split at `w` ≥ 100 and drill-in below; filter, count, columns; no 12-row cap; `EntriesAnchor`.
  - **`entry_form_dirty` and `lore_settings_dirty` in the aggregate snapshot**, with a `LoreSettingsEdited` message, and the dated **ADR-046 amendment** (G4).
  - The inline entry guard (Update / Discard / Stay).
  - The anchored Save options; Export to More actions.
  - Inline regex validation.
  - The New lore book / chat dictionary draft (Options, `Create`).
  - The `ENTRIES` context (`n`, `t`, `space`, `del`, `/`); rung 4 returns to the same row.
  - **Ctrl+S** in the entry editor (`Update entry` / `Add entry`; dictionaries `Update rule` / `Add rule`) and in lore and dictionary Options (`Save options`; `Create …` on a draft) (R41).
- **Acceptance.**
  - A 250-entry book shows **23** entry rows at 120x36 (drill-in), **33** at 160x45 (split) and **43** at 220x55.
  - `/ bellw ⏎` from the entries list reaches #200 in ≤6 keys (today 99 wheel notches).
  - After Update, Delete or Move, the selection stays on the entry or its neighbour.
  - Crossing `w` 99/100 never loses a draft.
  - A dirty Options or entry draft triggers the aggregate dialog on a kind switch, an item switch and leaving Roleplay.
  - New lore book never saves "Untitled".
  - Ctrl+S in the entry editor and in Options presses the visible commit button; from the entries list or its filter it is a no-op.
- **Re-pin:** `test_personas_dictionaries.py` (79; tab ids kept, `.active` works), `test_personas_lore.py` (46).
- **Gates:** the `PersonasDictionaryDetailWidget` (1,350 B) and `PersonasLoreDetailWidget` (803 B) segments do not grow; ADR-011 capture.
- **Dependencies:** A 33781, 33782, 33783, 33785, 33786 (hard); B5b; B7 for the 160 split.

#### B10 · Landing, Start row, kind empty states (D11)

- **Scope.**
  - `PersonasLanding` and `PersonasKindEmptyState` in `PW/personas_work_pane.py` (taking over `#personas-characters-empty`).
  - Retire F-031's auto-select (`PS:1878-1924`).
  - Start is live at compose; one worker fills the rest.
  - The Start rail row.
  - The **Continue a character chat** block (Q7), shown, with the G5 amendment (ADR-120, mirrored in ADR-004) in the same PR.
  - The list's one-line empty message (R25).
- **Acceptance.**
  - The first frame has live Start buttons.
  - Arrival focus is on the list, and `Import…` is the first F6 stop.
  - Start unloads through the guard and ends the session.
  - Continue lists at most 5 character-bound local conversations through the shared projection and the canonical opener. Unresolved rows are shown but not activatable, the block is absent in server runtime, and Enter on a row passes the leave guard.
  - The 120x36 landing uses the 2 × 2 Start grid.
- **Re-pin:** `test_personas_workbench.py:12315-12471`; `#personas-characters-empty` (11 refs in 2 files). Run a census of the implicit-fixture tests first.
- **Gates:** one worker; pre-import +0; ADR-011 capture.

#### B11 · Editors

- **Scope.** §3.11: growth caps as `$ds-roleplay-*` tokens, sections, the Jump row, the Advanced split, the commit-bar restyle with the same ids, `Saved 14:02` and Done, scroll home and focus Name, ✨ removed, field-label errors with a test (closes TASK-1651), Generate after its field in Tab order, and **Ctrl+S in persona Look and Tool policy**, which completes R41.
- **Acceptance.**
  - In an Edit session at 120x36: ≈4 fields visible.
  - Focus never changes geometry.
  - Ctrl+S works in every Roleplay editor (the §4.6 table), always pressing the visible commit button.
  - Generate-then-Accept is unchanged.
- **Fallback.** If B11 lands before B7, its acceptance is measured on the interim layout ("fields grow to their caps; the editor scroller shows ≥3 fields at 160x45").
- **Dependencies:** A 33784 and 33789 (hard); B5b.

#### B12 · User Guide consolidation

§5.8. Gate: `scripts/check_guide_claim_strings.py` over the four pages, reading every candidate.

### 5.3 Interim affordances (each removed by a named slice)

| Added in | Interim | Removed in |
|---|---|---|
| B1 | Subtitle `› <item>`; `#personas-purpose` beside the new header | B5b; B6 |
| B2 | `Import…` in the items toolbar; Duplicate in the Inspector stack | B6; B5b |
| B2 | The New chooser's Lore book / Chat dictionary keep today's immediate create (default name "Untitled lore book" / "Untitled chat dictionary" from B6) | B9 |
| B4 | `#personas-preview-toggle` relabelled; `#personas-try-toggle` in the mode-strip row | B5b (hidden, id kept; deleted) |
| B5a | The Chats view reached by the card's Conversations button | B5b |
| B5b | `Buddy ▸` chooser at the right end of `#personas-mode-strip`; list 1fr / work 3fr | B6; B7 |
| B6 | Rail, list and work on fr widths | B7 |

Interim geometry per slice (each slice's geometry test asserts these):

| After | First list item row | Characters at 120x36 | Work pane at 120 / 160 / 220 |
|---|---|---|---|
| today | 23 | 5 | 58 / 78 / 108 (+ Inspector 30 / 39 / 54) |
| B1 | 19 | ≥7 | unchanged |
| B2 | 13 at 120, 12 at ≥160 | ≥19 | unchanged |
| B5b | as B2 | as B2 | ≥86 / ≥115 / ≥160 |
| B6 | 11 at 120, 10 at ≥160 | ≥21 | ≥58 / ≥78 / ≥108 |
| B7 | **7** | **28** | browse 50 / 73 / 129; session 110 / 108 / 129 |

### 5.4 Definition of done (every slice)

1. **Backlog task.** A task with outcome ACs (repeat `--ac`). Its id is assigned against `origin/dev`, all remotes and all worktrees, leapfrogging A's block TASK-33781..TASK-33793 (PR #2957) (`backlog/docs/lessons-backlog-hygiene.md`). Its Implementation Notes record the live verification; the User Guide does not.
2. **Tests.** Targeted tests plus a `--collect-only` sweep, reading a nonzero count ("no tests ran" fails). Each new geometry test fails on the pre-slice code (paired arms).
3. **Preflight.** `./scripts/preflight.sh` passes.
4. **Budget ledger** (§5.10) filled in the PR body:
   - the `PS` row lowered to the measurement (slack 50, `Tests/Architecture/test_module_size_ratchet.py:129`);
   - code moved out of a governed module gets a **recipient-ceiling row** in `test_module_size_ratchet` plus a **re-export identity test** for names tests patch (#2862's convention, `Tests/Architecture/test_size_repair_helper_bindings.py`).
5. **ADR-011 before/after capture** (heartbeat, worker backlog, timer registry, mount churn) for **every slice that retires a legacy path or adds a worker or timer**: B2, B3, B4, B5a, B5b, B6, B7, B9, B10.
6. **Screenshots.** Before/after PNGs at 120x36, 160x45 and 220x55 (plus 80x24 and 60x24 for B7) from the harness (§5.7.5). The `.txt` captures are committed beside the slice's plan. Owner screenshot approval comes before merge (ADR-007).
7. **Docs and merge.** The guide delta for the changed behaviour. Qodo threads resolved. Merge by **rebase**, re-syncing only the PR about to merge (CLAUDE.md merge rules).
8. **Lessons.** A lessons entry only if the slice produced a real incident.

### 5.5 Ordering against sub-project A and the Console

**H** = must be on dev before the slice merges. **S** = should land first; otherwise the slice absorbs or compensates.

| A task (RP) | B1 | B2 | B4 | B5b | B7 | B9 | B11 |
|---|---|---|---|---|---|---|---|
| 33781 (RP-001, leaked slots) | | | S | **H** | **H** | **H** | S |
| 33782 (RP-014, entry anchor) | | | | | | **H** | |
| 33783 (RP-014, confirm delete/detach) | | | | | | **H** | |
| 33784 (RP-002, 0-1-line fields) | | | | S | S | | **H** |
| 33785 (RP-005, Options scroll) | | | | S | | **H** | |
| 33786 (RP-003, entry form) | | | | | | **H** | |
| 33787 (RP-009, search box) | | S, absorbed | | | | | |
| 33788 (RP-022, no-match copy) | | S, absorbed | | | | | |
| 33789 (RP-073, persona sections) | | | | **H** | | | **H** |
| 33790 (RP-075, name crash) | S | | | S | | | |
| 33791 (RP-078/079, stale persona) | | | | **H** | | | |

- **RP-001** is hard for B5b, B7 and B9, because their geometry tests assert "every hidden slot has height 0". A leaked band would make the tests and the screenshots lie.
- **Console.** TASK-33621.2 (RP-072; RP-074 is its AC#2) gates only **B5c** (R29). B depends on neither RP-084 (TASK-33792) nor RP-097 (TASK-33793). The recovery card (RP-074) is Console's to ship first.

### 5.6 Other in-flight work

| Item | Effect on B | Plan |
|---|---|---|
| TASK-33281 PERF-22 (reusable routes) | Memory moves from `save_state` to the instance; `ScreenResume` replaces arrival | Designed for both: no per-visit config reads, no module-level state; whichever lands second adapts |
| TASK-22988, TASK-31243 (In Progress; chat browse and resume) | Their UI moves into the Chats view (B5a) | Agree the panel API before B5a |
| TASK-26983, TASK-27000 (ruff formatter batches) | Whole-file churn on every B file | Run before B1 or after B12, never between |
| TASK-33276 PERF-17 (pre-import paydown) | Frees census headroom | Not a gate: Q10 ruled per-slice measurement. Any headroom it frees is used, never waited for |
| TASK-32809.3 (`PS` decomposition maps) | Overlaps the extraction ledger | Align §5.11's ledger; B never changes the behaviour sections `PS:4634-15880` |
| Library width defaults (28→33, 40→50) | **Already on dev** (`ITEMS_TARGET_WIDTH = 50` in `ARS`). The main checkout's uncommitted diff targets an older base | No collision row. B7's golden test covers the rail-policy coupling. |

### 5.7 Testing

#### 5.7.1 Layers

| Layer | What | Examples |
|---|---|---|
| Pure unit (no Textual) | geometry, sessions, state, projection | the §2.4 table × postures, hysteresis and priority; the session reducer; the normaliser bypass pin; the golden Library-policy coupling; `fit_rail_row_label`; `fit_view_strip` at 46/69/74 for all four kinds, the character strip at 34 and step 5 for the dictionary strip at 34; header and rail state; the prompt builder; memory transitions; `escape_rung`; contexts × keys; the config adapter and env overrides |
| Mounted Pilot, real CSS | behaviour per slice | `StyledRoleplayTestApp` loads the consolidated bundle (`Tests/UI/consolidated_css.py:122`); never a lightweight host (`lessons-testing-evidence.md:670`) |
| Geometry matrix | containment and rows | Required at 120x36, 160x45 and 220x55; degrade checks at 80x24 and 60x24 |
| Journeys | the four jobs (§5.7.3) | keyboard only, and mouse |
| Provider end to end | D13 | `Tests/UI/test_roleplay_chat_now_end_to_end.py` (B5c) |
| Performance and ratchets | §5.10 | every PR |
| Docs | claim strings | B12 and every guide delta |
| Live harness | screenshot approval | §5.7.5 |

#### 5.7.2 Geometry assertions (`Tests/UI/roleplay_frame_harness.py`)

1. **Rows:** the first content row and the first list item row per §5.3.
2. **Work pane:** width per slice and posture, with the primary in work rows 1-2.
3. **Containment:** every visible descendant lies inside its pane's `scrollable_content_region` (`lessons-testing-evidence.md:120`, `lessons-live-verification.md:278`), and no docked sibling covers a focused target (`:608`).
4. **Hidden slots have height 0** (the RP-001 guard).
5. **No layout shift:** region equality before and after focus and hover.
6. **No geometry change across views** of one loaded item once a session is active, none from read views while inactive (R9), and none across Save.
7. **Stale inline widths:** moving between fixed, measured and fixed states leaves no stale inline widths (`lessons-testing-evidence.md:670`).
8. **Thresholds:** both sides of every threshold, with hysteresis.
9. **Hidden focus:** a focused widget hidden by any transition leaves focus on a named target, never `None` (I1).

#### 5.7.3 Journeys (each passes by the slice that completes it)

| Job | Journey | Slices |
|---|---|---|
| J1 | Import a PNG card → it is selected and loaded → Chat now → the provider received one request | B2, B5b, B5c |
| J1 fast | `/ isolde ⏎ ⏎ o` (5 keys) | B2, B3, B5c |
| J2 | `e` at 120x36 → session gives 110 columns → edit Description → `esc` → `t` (no pane moves; ≥20 transcript rows) → `esc` → `e` → Ctrl+S → `Saved` (still 110) → Done | B4, B5b, B7, B11 |
| J3 | 250-entry book → `/ grand ⏎ ⏎ e / bellw ⏎` → #200 → edit → Update (or Ctrl+S) → the selection stays → `t` → Run → summary visible | B4, B9 |
| J3 | Dictionary with an invalid regex → `warning` on the row, `Info !` on the chip, inline under Pattern, in Needs attention | B5b, B6, B9, B10 |
| J4 | Persona → Profile → Look → Tool policy → Test chat; then the You row → Settings › Console Behavior → back, with state restored | B5b, B6 |
| All | Ctrl+4 → focus on the list → 60 Tabs stay inside → 15 Escapes never leave → F1 lists what the footer shows | B3 onwards |

#### 5.7.4 Gates every PR runs

- **The slice's targeted files.**
- **Roleplay UI suites:** `Tests/UI/test_personas_workbench.py`, `test_personas_dictionaries.py`, `test_personas_lore.py`, `test_personas_deferred_center_views.py`, `test_workbench_pane_focus.py`, `test_destination_visual_parity_correction.py`, `test_screen_footer_hints.py`, `test_master_shell_navigation.py`, `test_theme_contrast.py`.
- **Performance:** `Tests/Performance/test_textual_css_fastpath.py`, `test_boot_css_byte_budget.py`, `test_ui_ready_module_census.py`, `test_screen_preimport_payload_budget.py`, `test_ui_latency_guardrails.py`, `test_screen_leaks.py`.
- **Architecture and packaging:** `Tests/Architecture/test_module_size_ratchet.py`, `Tests/Packaging/test_config_import_closure.py`.
- **`./scripts/preflight.sh`.**

#### 5.7.5 Live verification (the isolated harness)

- **Make the harness durable (K14).** Before B1, copy it out of scratch (`/private/tmp/…/rp-review`, where the macOS cleaner removes it) into the review's QA folder, `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness/`, with `R` derived from the script's own directory: `HARNESS.md`, `launch.sh`, `mkprofile.sh`, `seed.py`, `seedrun.sh`, `gapcheck/{scale_seed,persona_more,lore_fix,mock_llm_body}.py`, `mock_llm.py`, the mock `.toml` files, `where.py`, `click.sh`, `shot.sh` and `shot_png.py`.
- **Recipe:**
  ```bash
  R=<harness dir>; WT=<slice worktree under .worktrees/>; PY=<repo>/.venv/bin/python
  $PY $R/mock_llm.py 18765 &
  APP_WT=$WT $R/launch.sh rbB5b 120 36 golden      # also 160 45, 220 55; `empty`; the volume copy
  sleep 18                                         # cold start
  $PY $R/where.py rbB5b "⌃4 Roleplay" --click      # Ctrl+digit cannot be sent via tmux
  tmux -L rbB5b send-keys ...                      # the slice's journeys
  PNG=1 $R/shot.sh rbB5b $OUT/<state>-120x36
  tmux -L rbB5b resize-window -x 160 -y 45; sleep 3
  REUSE=1 APP_WT=$WT $R/launch.sh rbB5b 160 45 golden   # restart: preferred state restored (B7)
  tmux -L rbB5b send-keys C-q; sleep 2; tmux -L rbB5b kill-server
  ```
- **Rules:**
  - Never redirect the app's stderr.
  - Use a unique socket name per slice.
  - Capture three lifecycle entries per size: untouched arrival, route activation, resize restoration.
  - Build the volume profile on a **copy** of `golden`: 348 characters, 114 personas, 43 lore books (one with 250 entries), 28 dictionaries (one with 150 rules).
  - Launch only through `launch.sh` with `APP_WT` / `PYTHONPATH=$WT`, because the venv's editable install may point at another worktree (K16).

### 5.8 User Guide

- **Pages:**
  - `Docs/User_Guide/roleplay-chat-dictionaries.md` (layout `:59-75`, collapse `:246-248`, Inspector matrix `:118-124`, readiness `:131-143`, keys `:194-214`);
  - `…/characters-and-personas.md`;
  - `…/chat-dictionaries.md` (`:37-41`);
  - `…/lore-books.md` (`:31-36`).
- **Per slice**, each slice rewrites the sections whose behaviour it changed:

  | Slice | Guide delta |
  |---|---|
  | B1 | header |
  | B2 | list, marks, Import, New |
  | B3 | keyboard, generated from the projection's F1 groups so the guide and F1 match; where Ctrl+S works |
  | B4 | Test chat and Try it |
  | B5a | Chats |
  | B5b | views, More actions, Chat now |
  | B5c | `o` |
  | B6 | rail, You, Buddy, memory |
  | B7 | layout, grips, work sessions, persistence; replaces "collapse lasts only for the current visit" |
  | B8 | the Try column |
  | B9 | entries and drafts; Ctrl+S in entries and Options |
  | B10 | landing, including Continue |
  | B11 | the editor; Ctrl+S in Look and Tool policy |

- **B12 consolidation.**
  - Retitle the guide to **"Roleplay"** and **keep the filenames** (5 inbound links: `Docs/User_Guide/index.md`, `settings.md`, `buddy.md`, `console/context-and-rag.md`, `openai-compatible-tts.md`).
  - Model the layout section on `Docs/User_Guide/library.md:168-215`.
  - Fix the drift: the palette string, the retired Status row, "Attach to Console/Start Chat", Ctrl+1-4 for modes, the persona tooltip.
  - **Delete** the legacy "Verified against" stamps on rewritten pages (`roleplay-chat-dictionaries.md:256`, `characters-and-personas.md:365`, `chat-dictionaries.md:252`, `lore-books.md:228`); the convention now forbids them.
  - Run `scripts/check_guide_claim_strings.py`.
- **Not in B.** The Settings › Roleplay contract copy and the nav tooltip (`UI/Navigation/shell_destinations.py:85`, RP-011/RP-057) belong to sub-project D; the guide only links to them.

### 5.9 Governance

| # | Document (anchor) | Change | Slice |
|---|---|---|---|
| G1 | **New ADR-NNN**, amending ADR-086 and ADR-084 (number assigned at merge; dev holds duplicates and #2862 claims 182-203) | The adaptive pane shell is a shared destination primitive (Library, Roleplay). It covers the resolver aliases (byte-identical), the grip and shell, the normaliser and the stage bar; one collapse grammar; the slice order and the fold-before-rail arithmetic; Roleplay never imports `Widgets/Library/`; shared focus contracts (`TAB_REGION`, `arrival_focus_target()`, grip evacuation, dynamic F6); the census rule (§5.10); one round border per pane (Q1). **Not authorised:** Watchlists convergence, an app-wide workbench, a third shell pane. | B0 |
| G1a | ADR-NNN: the grip exception to design-language "never a sliver" (`backlog/docs/design-language.md:132-134`) | Cite **ADR-084:15-16, 73-74** (accepted 5-column grips) and the 2026-09-09 owner ruling, **not** ADR-148 (Proposed). The zero width holds below 64 only once the single-stage follow-up lands; until then the grips stay at all widths (R38). | B0 |
| G1b | ADR-NNN: the rail-width exception to design-language §2.8 (`$ds-sidebar-min-width: 35`) | Adaptive rails follow the 29-39 policy with a 24 floor; Roleplay **deliberately shares** the Library policy (recorded coupling, golden-pinned) | B0 |
| G1c | ADR-NNN: destination-owned non-geometry preferences | `last_kind` and per-kind `sort` live in `[roleplay.*]` outside the shared normaliser (a carve-out from ADR-084:79-80) | B7 |
| G1d | ADR-NNN: custom widths | **Not read in B.** No hidden config; a Settings slice may add them later (ADR-086:29-30; Q9) | B7 |
| G1e | ADR-NNN / DESIGN.md §7 | **Home rule:** Roleplay region widgets live in `Widgets/Persona_Widgets/` (ADR-004:55 precedence, existing practice) and controllers in `UI/Persona_Modules/`. Record which rule governs, because DESIGN.md:381-386 and ADR-004:55 conflict. | B5b |
| G2 | ADR-086 header + decisions README | "Amended by ADR-NNN" | B0 |
| G3 | ADR-084 | One line pointing to ADR-NNN plus G1c | B0 |
| G4 | **ADR-046 (dated amendment)**, cross-referenced from ADR-120:123-127 | The aggregate domains gain `entry_form_dirty` and `lore_settings_dirty`. One predicate drives `_run_guarded`, the header chip, the title-row word and the commit bar. The dialog's third choice stays **"Stay"** (ADR-046:110 and ADR-120:126 unchanged). | B1 (predicate), B5b (guard), B9 (domains) |
| G5 | **ADR-120** surface-role amendment, mirrored in ADR-004 (`backlog/decisions/004-personas-destination-native-workbench.md`) | **Approved (Q7):** bounded Roleplay landing recents (5 rows, character-bound only, shared projection, Data-Profile fence, the canonical opener with typed results, unresolved rows non-activatable, local only). Restate that Roleplay never deletes conversations (R31). | B10 |
| G6 | ADR-115 | Eager-set note: "card, rail, items list, work header, view strip, More actions region, Try column (preview + Try-it), Chats panel, Info, landing and kind empty states (≈20 widgets), attachments". The four demand bodies, the boundary and the verification contract are unchanged. Look and Tool policy are sections of the one persona editor. | B5b |
| G7 | ADR-152:37, :40, :63 | (a) "Letters act only from rail, list, grip and view-strip focus; never from buttons, text fields, form controls or reading bodies" (D8 = B, RP-033). (b) Order c p l d with **`MODE_HOTKEYS` reordered in lockstep**; footer text `c/p/l/d [ ] kind`. Mirror in `PS:436-445`. | (a) B3; (b) B6 |
| G8 | ADR-015:34 | The mode-descriptor and count statics are superseded by the rail's glosses and counts | B6 |
| G9 | **ADR-031 (`backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`), dated amendment** | (a) Rule 2 gains one exception (owner ruling Q5, D9): an **editor-scoped `ctrl+s`** may be bound screen-wide if it runs exactly the path of the visible Save of the active editor, only while that Save is offered and enabled, and is advertised only there (the task-1340 "advertised ⊆ bound" pattern). Roleplay additionally requires the press to come from inside the focused editor region and shows the state chip `ctrl+s: nothing to save here` elsewhere (R41), in every Roleplay editor. The Library's skill-editor Ctrl+S (`LS:1047, 25834-25848`, gated by `check_action` to `_library_skill_save_available()`, `LS:25365`) meets the base exception. (b) Hiding "Add to Console message", `u` and "Send to Console draft" is **mandated by rule 4** (inert advertised action), not an owner question. (c) `ctrl+enter` is retired at screen level. | (a) B3 records the amendment and binds character and persona Edit; B9 the entry editor and Options; B11 persona Look and Tool policy. (b) B5b. (c) B3 |
| G10 | ADR-097 ledger | No blanket re-pin. Each slice measures, defers or sheds first, then records its +1 with owner sign-off in the same commit (§5.10, Q10). Lower `MAX_ANCESTOR_SCOPED_BARE_TYPE_RULES` to the measured count after each re-key (up to 12 rules, 274 → ≈262). | B0-B10 |
| G11 | DESIGN.md:107 | The local mode bar may be a rail section (kinds) plus a work-pane view strip; the optional inspector may fold into the work pane (Roleplay, ADR-NNN) | B6 |
| G12 | **DESIGN.md:328-330 + front matter :81-85** | The inline one-row DestinationHeader variant (title, subtitle, authority chip, optional unsaved and blocked chips; used by Console, Lab and Roleplay). Readiness may sit beside the destination's primary action in the work pane. | B1 |
| G13 | DESIGN.md:236, :240 | **Dropped.** Q1 ruled bordered panes, so no exception is needed | — |
| G14 | DESIGN.md:301-304 | The fold-cue variant: a zero-row border label on bordered panes (R34) | B7 |
| G15 | design-language.md:135-136 | Runtime geometry promotes to the config-safe Python leaf, not `_variables.tcss` | B0 |
| G16 | `css/patterns.json` + `backlog/docs/component-patterns.md` + `Widgets/pattern_gallery.py` | The `panes` family (`test_component_pattern_governance.py:15-17`) | B0 |
| G17 | `css/core/_variables.tcss` | `$ds-roleplay-*` tokens for field caps and any feature geometry; `_roleplay.tcss` has zero numeric dimension literals (ADR-161 decision 4) | B1/B11 |
| G18 | `css/build_css.py` `ScreenOwnedSplit` | Narrow prefixes plus `pinned={"personas-library-rows"}` (R18) | B1 |
| G19 | DoD | ADR-011 capture in every slice that retires a legacy path or adds a worker or timer (§5.4) | all |
| G20 | `Widgets/glyph_fallback.py` legend (TASK-32235/32309) | `✓` = marked in Roleplay (it also means "done" in the house map); the new map entries `› ‹ … · —` and `⇥ → ← ↑ ↓ ×` (§4.12); a named active-marker constant | B1/B2 |

**Rules confirmed compliant:**

| Rule | Status |
|---|---|
| ADR-004 (`backlog/decisions/004-personas-destination-native-workbench.md`) | Reused `#ccp-*` ids and pinned ids are moved, not renamed. The exceptions are `#personas-workbench` and `#personas-inspector-actions`: containers that are replaced, not moved. |
| ADR-007 | Screenshot-approved slices. |
| ADR-037 / ADR-149 | Personas are assistant-side; the You row is never a persona or profile editor; persona Chat now stays Roleplay-originated. |
| ADR-115 | The demand-mount boundary is unchanged. |
| ADR-120 | The Chats view is the complete per-character browse, and Roleplay never deletes conversations. The landing recents are authorised by the G5 amendment (Q7). |
| ADR-031 (`backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md`) | Amended for the editor-scoped Ctrl+S (G9a); rule 4 drives the hidden hand-offs (G9b). |
| ADR-150 / ADR-161 | Tokens and catalog classes only. |

### 5.10 Budget ledger (fill one row per guard in each PR)

| Guard (file:line) | Today | B's expected effect |
|---|---|---|
| Broad selectors, `Tests/Performance/test_textual_css_fastpath.py:326` | 274 / 274 | No new bare-type subjects anywhere (lazy sheet or `BUNDLED_CSS`). Re-keys in B2 (≈6), B4 (1), B5a (2) and B5b (3 deleted): up to 12, **≈262**, with the ratchet lowered in each PR |
| Boot parsed CSS, `test_boot_css_byte_budget.py:117` | ≈584 KB / 608,090 B | B0 +≈1.1 KB; B1 ≤0; B2 ≤0; B5b −1,898 B plus the new widgets' boot segments (ledgered); B7 − (personas items in `AT:636-757`, re-estimated). Net fall. |
| `PersonasScreen` scoped segment, `boot_css_bytes.json:145` | 2,996 B | Falls (B6, B7) |
| UI-ready modules, `test_ui_ready_module_census.py:162` | 1,032 / 1,033 | +0. `personas_state.py` gains dataclasses but **imports nothing new** (guard test) |
| **Pre-import pass modules**, `test_screen_preimport_payload_budget.py:109` | **557 / 557** on dev; 483 / 500 if #2862 lands first | **+3** (`adaptive_pane_shell` B0, `roleplay_frame_state` B1, `roleplay_frame_controller` B5b); **−1** available after #2862 (fold `library_emergency_return.py` into the shared module: one import line at `LS:488`). **Net +2**, each measured first (Q10). |
| Pre-import pass LOC, `:113` | 411,886 / 425,347 on dev; **378,681 / 378,740 (59 LOC headroom)** if #2862 lands first | ≈+3-5k on the Roleplay route (moved code nets zero inside `PS`). If #2862 lands first: a per-slice LOC re-pin or an offsetting import diet (Q10) |
| Largest route LOC, `:117` | Library 125,111 / 135,111 | Library +≈0.5k (B0) |
| `PS` lines, `Tests/Architecture/test_module_size_ratchet.py:97` | 16,533 / 16,436 (**red**); 16,419 / 16,419 after #2862 | Falls every slice; about 1,000 lines leave (§5.11); ≈15.4-15.6k after B11 (estimate). New recipient-ceiling rows for `roleplay_frame_state.py`, `roleplay_frame_controller.py`, `personas_work_pane.py` and `personas_library_pane.py` |
| CSS sources after the tour, `Tests/Performance/test_ui_latency_guardrails.py:52` | < 56 | +1 (the B1 lazy sheet). New widgets declare `BUNDLED_CSS`, never `DEFAULT_CSS` |
| Destination tour switch time, `:57` | 10 s | Unchanged |
| Ctrl+4 retention, `Tests/Performance/test_screen_leaks.py` | 0 | 0 |
| ADR-115, `Tests/UI/test_personas_deferred_center_views.py` | 4 bodies absent at first paint | Unchanged |
| PERF-22 arrival | header 0.35 s, list ≈1 s at 348 | ≤ today; keystroke → rows ≤250 ms at 348 and 5,000; 0 per-row mounts |

### 5.11 Collisions and code placement

**PR #2862** (`codex/personal-context-memory-dev`; conflicting; 364 files):

| #2862 touches | B slices that meet it | Strategy |
|---|---|---|
| `PS` (−117 lines: `_drain_*` at `:831-951`, imports `:365-372`) | every slice except B0 and B8 | B never edits `PS:360-375` or `:820-960` |
| `test_module_size_ratchet.py:97` (+ recipient ceilings) | every `PS` slice | **Certain conflict.** Re-measure `PS` after the rebase; never pick a side |
| `PM/personas_preview_coordinator.py` | none | B moves the preview *widget* at compose time only |
| `LS`, `UI/Library_Modules/screen_helpers.py`, `Library/library_shell_state.py` | none | B never edits them. Library adopts `TAB_REGION`, the rail rows and the neutral grips later. The `library_emergency_return` shed (one import line, `LS:488`) waits for #2862. |
| `css/build_css.py`, `AT:1354, 2536`, generated sheets, `preimport_payload.json`, `ui_ready_modules.txt` | B0, B1, every CSS slice | Disjoint hunks. **Never hand-merge generated output:** rebase, build, `scripts/update_boot_budget_snapshots.py`, `./scripts/preflight.sh` |
| The pre-import budget constants (557→500 modules; 425,347→378,740 LOC) | every slice that adds code | Q10, both merge orders |
| `Tests/UI/app_factory.py` (+50/−6) and `Tests/UI/test_app_factory_cleanup.py` | B2 (the harness) | Build `StyledRoleplayTestApp` on `_build_test_app`, with no cleanup of its own |
| The re-export identity convention (`Tests/Architecture/test_size_repair_helper_bindings.py`) | every extraction | The §5.4 DoD item |

- **Sequencing.** Do not wait for #2862: whichever lands first, the other rebases. Never rebase onto #2862's unmerged branch.
- **Ratchet.** If B1 merges first, it brings `PS` to ≤16,436.
- **Draft PR #2885 (Buddy workbench).** B edits none of its files. The rail Buddy row's "Manage…" opens whatever host exists; "Buddy workbench…" appears in persona Look unchanged.
- **Draft PR #2563.** A mode-only `PS` change; watch it on rebase.
- **`PS` churn** (117 commits since 2026-08-15). One `PS` slice in flight at a time; extract first; rebase before review; a fresh `.worktrees/` worktree per slice, never `/tmp`.

**Module map** (net +3 new modules, +2 after the post-#2862 shed):

| Module | New or reused | Holds | Slice |
|---|---|---|---|
| `Widgets/adaptive_pane_shell.py` | **new (shared)** | shell, grip, messages, `StageReturnBar`, rail-row primitives, compact search input | B0 |
| `PM/roleplay_frame_state.py` | **new (pure)** | header and rail state, list prompt builder and density, profiles, the **work-session reducer**, the keyboard projection (contexts, armed keys, Escape rungs, footer, F1, F6 targets, `LEAVES_SCREEN_IDS`), `fit_view_strip` | B1, extended by B2, B3, B5b, B6, B7 |
| `PM/roleplay_frame_controller.py` | **new** | the view controller and the layout controller (two classes) | B5b, B7 |
| `PW/personas_work_pane.py` | **`git mv`** of `personas_inspector_pane.py` | `PersonasWorkHeader`, `PersonasInfoPanel`, `PersonasLanding`, `PersonasKindEmptyState` | B5b, B10 |
| `PW/personas_library_pane.py` | reused in place | `PersonasLibraryPane`, `RoleplayItemList`, `RoleplayRail` | B2, B6 |
| `PW/personas_conversation_transcript_widget.py` | reused | `PersonasCharacterChatsPanel` | B5a |
| `PW/personas_state.py` | reused (UI-ready-resident; imports nothing new) | `ModeMemory`, `ItemMemory`, `RoleplayMemory` | B6 |

**`PS` extraction ledger:**

| `PS` lines (approx. size) | What | Destination | Slice |
|---|---|---|---|
| `1547-1554`, `4551-4620` (≈80) | header compose and text | `PM/roleplay_frame_state.py` | B1 |
| `486`, `3690`, `3828-3890`, `4273-4360` (≈160) | paging, list population, counts | `PW/personas_library_pane.py`, `PM/roleplay_frame_state.py` | B2 |
| `1083-1121`, `16318-16351`, `16455-16526` (≈145) | BINDINGS, Escape, footer | `PM/roleplay_frame_state.py` (thin shims stay) | B3 |
| `1717-1738` + 40 Inspector call sites, `4459-4549`, `15909-15975` (≈180) | Inspector, mode visibility, `_show_center` | `PW/personas_work_pane.py`, `PM/roleplay_frame_controller.py` | B5a, B5b |
| `1558-1586`, `1748-1876` (≈157) | purpose and strip compose; save/restore into memory | `PW/personas_library_pane.py` (rail), `PW/personas_state.py` | B6 |
| `607-609`, `1122-1155`, `1160-1325`, `1590-1603`, `2308-2425`, `4427-4457` (≈350) | constants, F6 tuple, `BUNDLED_CSS`, handles, responsive code, rail toggles | `PM/roleplay_frame_controller.py`, `PM/roleplay_frame_state.py`, `css/features/_roleplay.tcss` | B7 |
| `1878-1924` (≈47) | F-031 auto-select | `PW/personas_work_pane.py` | B10 |

`PS`'s behaviour sections (`:4634-15880`) and the editors' internals stay where they are.

### 5.12 Risks

| # | Risk | Likelihood / impact | Mitigation |
|---|---|---|---|
| K1 | TASK-33621.2 slips | medium / **low now** | Only B5c waits (R29) |
| K2 | A tasks not landed, so geometry is distorted (RP-001) | medium / high | §5.5 hard prerequisites; hidden slots asserted at height 0; B2 absorbs 33787/33788 |
| K3 | `PS` conflicts (#2862; a hot file) | high / medium | One `PS` slice at a time; disjoint hunks; re-measure after each rebase; rebase-only merges |
| K4 | Pre-import census (557/557, or 59 LOC headroom after #2862) | certain / medium | +3/−1 module plan; measure, defer or shed first; Q10 |
| K5 | Test churn: 325 ids in 51 files; 91 row ids + 46 typed `ListView` refs; 101 Inspector refs; 358 workbench tests | certain / medium | Move, never rename; harness helpers; dry-run censuses before B2, B3 and B10; re-pins in the same PR |
| K6 | ADR-011 regression (≈+25 widgets; window and counts workers) | low / high | Before/after capture in every qualifying slice; zero-mount transitions |
| K7 | ADR-115 broken by moving widgets | low / high | Compose once in the final container (B4); widget-identity tests |
| K8 | Draft loss in new transitions (sessions, views, the Try column, drill-in) | low / high | Display-only toggles; the parametrised trigger table; the new draft domains (B9); Ctrl+S on the visible commit button only |
| K9 | Library regressions from B0 | low / high | Golden-identical resolver; same-object aliases; the visit-order live check |
| K10 | Interim states confuse users between releases | medium / low | Screenshot approval; the §5.3 register; a guide delta per slice |
| K11 | `OptionList` specifics (private API, global cursor rule, markup injection, O(n) marks) | medium / medium | Public API only; id-scoped override; markup-off prompts; the 5,000-row gate |
| K12 | The footer or F1 lies as contexts are added | medium / medium | One projection; a contexts × keys test in every slice that adds a context |
| K13 | `[roleplay]` written but never loaded | medium / medium | Projection in `load_settings()`; save → reload → read; restart in the harness |
| K14 | The live harness lives in scratch | high / medium | Copy it into `harness/` before B1 |
| K15 | Pilot-only evidence misses terminal behaviour | medium / medium | Harness at three sizes and three lifecycle entries; check `pilot.click` misses |
| K16 | Wrong code under test (a foreign editable install) | medium / high | `launch.sh` with `APP_WT` / `PYTHONPATH=$WT` |
| K17 | Owner questions open when a slice is ready | closed | All twelve were ruled on 2026-10-02 (§8) |
| K18 | Server (read-only) runtime regresses unseen | medium / medium | Stub-service Pilot tests per slice for `Server: … · read-only` and disabled rows that say why |
| K19 | Formatter batches land mid-programme | low / medium | Before B1 or after B12 |
| K20 | ADR number collision | high / low | Provisional number; re-verify at merge |
| K21 | Work sessions feel "sticky" (panes stay closed after Done) | medium / low | `esc to list` is always advertised; the crumb ends the session; a manual grip reopen wins; the harness journeys check it at 120 and 160 |
| K22 | Ctrl+S presses the wrong commit button (persona Look holds up to three) or a hidden one | low / high | One `editor_save` resolution in the projection; the contexts × editors test asserts the pressed id per row; `editor_save`'s ancestor-visibility check (§4.6), plus focus-scoped section saves, so a hidden button or a section without focus is never saved |

### 5.13 From spec to plan

1. **Plans.** The next step is the `superpowers:writing-plans` skill, producing **one implementation plan per slice** under `Docs/superpowers/plans/`, starting with **B0** (B1 may be planned alongside it) and then following §5.1's order. Each plan carries its slice card (§5.2), the definition of done (§5.4), its prerequisites (§5.5), its ledger rows (§5.10) and its guide delta (§5.8). A plan that changes a mockup regenerates it from §0.6's rulings at true width (R40).
2. **Backlog.** When planning starts (not with this spec), file the parent task "Roleplay on the Library frame (sub-project B)" with subtasks **B0, B1, B2, B3, B4, B5a, B5b, B5c, B6, B7, B8, B9, B10, B11 and B12** in dependency order. Ids are assigned against `origin/dev`, all remotes and all worktrees, leapfrogging TASK-33781..TASK-33793 (`backlog/docs/lessons-backlog-hygiene.md`). File these follow-ups beside it:
   - the true single stage below 64 columns (R38, §2.8);
   - the Library `/` convergence on the pane-scoped rule (R26, `LS:7211`);
   - the shell nav fragment `‹ le` (RP-024, `UI/Navigation/main_navigation.py`);
   - the content batch (§0.5);
   - Library adoption of `TAB_REGION` (§4.3);
   - persona Duplicate as a new capability (R14).
3. **What plans cite.** This spec, and the review report and findings that land with PR #2957 (header links).
4. **Nothing is filed by this spec.**

---

## 6. Coverage: every finding mapped to an element or a deferral

**Summary:**
- Of the 97 findings, **59 are covered** by B, **6 partly** (the rest is another sub-project's), and **32 are deferred** with an owner.
- Of the 43 findings scoped `frame` or `both`, 41 are covered. The other 2, RP-011 and RP-012, are partly covered: their remainders belong to sub-projects D and C.
- Findings that span sub-projects are owned as follows:
  - RP-081 content → B4;
  - RP-041 Roleplay strings → B6; palette and Console strings → the content batch;
  - RP-062 placeholder and inactive-chip contrast → B2/B5b;
  - RP-071 ✨ → B11;
  - RP-024 nav fragment → a shell task filed with the parent task (§5.13);
  - RP-044 field labels → B11;
  - RP-094 ≥32-column names → B2.

**Legend:** "A", "C" and "D" are the sub-projects; "Console" is TASK-33621.2 / 33792 / 33793; the "content batch" is the ownerless set listed in §0.5, filed as one task.

| RP | P | Scope | Design element (where) | Slice / status |
|---|---|---|---|---|
| 001 | P0 | content | Hidden slots asserted at height 0 in every geometry test (§5.7.2) | **deferred: A** (TASK-33781); hard prerequisite for B5b, B7, B9 |
| 002 | P0 | content | Session posture gives the rows (§2.3, 25/34/44 scroller rows); growth caps (§3.11) | **deferred: A** (33784) for floors; B7, B11 provide rows and caps |
| 003 | P0 | content | Full-height entry editor column re-hosts A's labelled form (§3.9) | **deferred: A** (33786); re-host in B9 |
| 004 | P0 | content | Honest copy from `roleplay_character_world_info_applied` (§3.13) | **deferred: C** (D6); copy only in B5b/B10 |
| 005 | P0 | both | Anchored `Save options` commit bar; Export → More actions (§3.5.3-4) | covered (B9, B5b); scroll fix **A** (33785) |
| 006 | P0 | both | Try it as its own view (§3.8); entries split/drill-in, 23/33/43 rows (§3.9); XL column (§2.7) | covered (B4, B9, B8) |
| 007 | P0 | frame | One-row header (§1.3); one framing layer, 5+2 chrome (§1.1, §2.10) | covered (B1, B7) |
| 008 | P0 | frame | One-row toolbar (§1.5.1); verbs to rail/More actions/marks bar; first item row 7 | covered (B2, B6, B7) |
| 009 | P0 | both | One-row compact `#personas-filter` (§1.5.1) | covered structurally (B2); fix **A** (33787), absorbed if open |
| 010 | P0 | both | Test chat as a full-height view; session gives 110/108 cols (§3.8, §2.3); XL column | covered (B4, B7, B8) |
| 011 | P1 | both | Persona gloss `reusable AI roles`; You row (§1.4.1, §1.4.5); Profile card line | partly (B6, B5b); nav tooltip + Settings copy **deferred: D** |
| 012 | P1 | both | Honest World / Used by copy (§3.13); Used by slot | partly (B5b, B9); data **deferred: C** |
| 013 | P1 | content | — | **deferred: C** |
| 014 | P0 | content | `EntriesAnchor` hook + item memory (§3.9.4); delete modal (§3.6) | **deferred: A** (33782/33783); hook in B9 |
| 015 | P1 | content | `Info (1 warning)` chip (§3.3); inline under Pattern (§3.5.4); row `warning` word + rail ` !` (§1.5.2, B6 aggregates); Needs attention (§3.10) | covered (B5b, B9, B6, B10); block-vs-warn stays warn (D9) |
| 016 | P1 | content | Session width (§2.3); sections + Jump row + caps (§3.11) | covered (B7, B11); sections instead of sub-modes |
| 017 | P1 | both | Roleplay profiles + work sessions (§2.2-2.4); XL column | covered (B7, B8) |
| 018 | P1 | both | Inspector fold (§3.7); Buddy rail row (§1.4.1) | covered (B5b, B6) |
| 019 | P1 | both | Primary in work rows 1-2 (R21); 12×6 thumbnail below `w` 84; `▾ more` border cue | covered (B5b, B7) |
| 020 | P1 | both | One action bar + More actions (§3.6); collection verbs in the rail | covered (B5b, B6) |
| 021 | P1 | both | `esc clear search` clears box + filter (§4.5); memory restores search into the box and applies the kind with no selection (§4.11); rail `3 of 28`; landing on return (§3.10) | covered (B3, B6, B10) |
| 022 | P1 | both | Search-aware empty states, `Also in:`, `3 of 28` (§1.5.4) | covered (B2 extends); fix **A** (33788), absorbed if open |
| 023 | P1 | content | Example dialogue slot in PROMPTING (§3.11) | **deferred: content batch** (slot in B11) |
| 024 | P1 | frame | Arrival on the list + `TAB_REGION` (§4.3); landing first stop `Import…` | covered (B0, B3, B10); `‹ le` nav fragment **deferred: shell** (task filed with the parent, §5.13) |
| 025 | P1 | both | Dynamic F6 with grip substitution; per-view work targets incl. entry editor and Try it (§4.4) | covered (B5b, B7, B9) |
| 026 | P1 | both | Escape rung 3 leaves the field; typing footer (§4.5, §4.8) | covered (B3) |
| 027 | P1 | content | No focusable scrollers; hidden tab rows; Generate after its field; focused control scrolled into view (§3.1, §3.5.3, §3.11, §4.3) | covered (B3, B5b, B11) |
| 028 | P2 | both | Rail counts `(…)`/`(28)`/`(3 · 2 on)`; list title counts (§1.4.2, §1.5.1) | covered (B2, B6) |
| 029 | P2 | frame | Header names the kind (§1.3); rail rows with counts + glosses (§1.4.1) | covered (B1, B6) |
| 030 | P2 | both | Lossless views + item memory; per-kind `ModeMemory` (§4.11) | covered (B5b, B6) |
| 031 | P2 | frame | `[roleplay.reader]` preferred state (§2.11) | covered (B7) |
| 032 | P2 | frame | One grip grammar: 5-cell labelled grips (§1.2, §2.1) | covered (B0, B7) |
| 033 | P2 | both | Letters scoped by `check_action` (§4.6); space/m/s on the list widget; toasts with undo | covered (B2, B3, B6) |
| 034 | P2 | frame | Focus-aware footer projection; `c/p/l/d [ ] kind` in rail order survives at 120 (§4.8) | covered (B3, B6) |
| 035 | P2 | both | `/` (pane-scoped), `o` (B5c), terminal-safe Try-it keys, space on lore, `ctrl+enter` retired (§4.6) | covered (B3, B4, B5c, B9); Ctrl+S in every editor (R41: B3, B9, B11) |
| 036 | P2 | frame | Contextual F1 from the projection (§4.9) | covered (B3, B6) |
| 037 | P2 | content | One guarded-transition dialog: Save and continue / Discard and continue / Stay (§3.12) | covered (B5b) |
| 038 | P2 | both | Three unsaved signals from one predicate (R24); `Saved 14:02` + Done (§3.11) | covered (B1, B5b, B11) |
| 039 | P2 | both | Marks by id, marks bar, gutter-click marking, one delete modal naming ≤5; Delete only on the open item (§1.5.5, §3.6) | covered (B2, B5b) |
| 040 | P2 | content | Entry draft + inline guard (§3.9.5); New lore book/dictionary as a draft (§3.9.6) | partly (B9); greeting-box overwrite → **content batch** |
| 041 | P2 | both | Canonical nouns in rail/list/strip; Roleplay-owned strings (default names, toasts, pickers, tooltips) in B6 | covered for Roleplay (B6); palette aliases + Console Inspector headings → **content batch / Console** |
| 042 | P2 | frame | Rail titled `Roleplay`, list titled by kind (§1.1, §1.4) | covered (B2, B6) |
| 043 | P2 | both | One `Import…` picker; New chooser (from a portrait = Actor Pack); Actor Pack as an export format (§1.4.6, §3.6) | covered (B2, B5b, B6) |
| 044 | P2 | content | Info shows findings only; count on the chip; field-id → label mapping with a test (closes TASK-1651) (§3.5, §3.11) | covered (B5b, B11) |
| 045 | P2 | content | No test chat without a loaded item (§3.8.1) | partly (B4); speaker labels → **content batch** (D for persona) |
| 046 | P2 | content | — | **deferred: content batch** |
| 047 | P1 | content | Add to Console message, `u`, `ctrl+enter`, Send to Console draft hidden behind one constant (ADR-031 rule 4) | covered as a hide (B3, B5b); payload → **Console** |
| 048 | P2 | content | — | **deferred: content batch** |
| 049 | P2 | both | Chats view: full-height list, rows with message count and age, split at `w` ≥ 100 (§3.5.1) | covered (B5a); Library's character-blind rows are Library's |
| 050 | P2 | content | One-row transcript toolbar (§3.5.1) | covered (B5a) |
| 051 | P2 | content | — | **deferred: content batch** |
| 052 | P2 | both | Work-pane title row names the loaded item (§3.2); interim header subtitle (B1) | covered (B1, B5b) |
| 053 | P2 | content | Card order = editor sections (§3.5.1); editor sections (§3.11) | partly (B5b, B11); field glosses → **content / D** |
| 054 | P2 | content | One scroll owner per body; detail stack becomes `Vertical` (§3.1) | covered (B5b) |
| 055 | P2 | both | Landing + Start row + kind empty states (single owner) + rail glosses (§3.10, §1.4) | covered (B6, B10) |
| 056 | P1 | both | Entry filter, position count, no cap (§3.9.3); `/` reach (§1.6) | covered for B's half (B9); entry-level cross-search → **C / Q8** |
| 057 | P2 | content | — | **deferred: content batch** (palette); nav tooltip **D** |
| 058 | P2 | content | Import picker title lists formats (§1.4.6) | **deferred: content batch** (error copy) |
| 059 | P2 | content | — | **deferred: content batch** |
| 060 | P2 | content | A guide delta per slice; B12 consolidation (§5.8) | covered (B12 + every slice) |
| 061 | P2 | shell | — | **deferred: shell** (B depends on it) |
| 062 | P2 | both | Focus-dependent cursor ≥4.5:1, focus edge; Roleplay placeholder and inactive-chip contrast to AA; `test_theme_contrast` extended (§1.5.3, §4.12) | covered for Roleplay (B2, B5b); app-wide token lift → **design system (D10)** |
| 063 | P2 | frame | Active vs focus cues differ on rail and strip (§1.4.3, §3.3) | covered (B3, B6) |
| 064 | P2 | frame | Focus evacuation + restore; named targets on every hide (§4.11) | covered (B7) |
| 065 | P2 | app-wide | Mitigation: no button arrival/F6 target; Enter chip names the action (§4.7) | **deferred: design system (D10)**; mitigated in B3 |
| 066 | P2 | frame | Tiers with labelled grips; event-keyed XS stages; list-first degrade below 64 (§2.8) | covered (B7); true single stage → follow-up task |
| 067 | P2 | both | No `Ready` badge; readiness word = model or reason, never dropped while disabled; blocked-destination chip (§1.3, §3.2) | covered (B1, B5b) |
| 068 | P3 | content | 1-line density, no blank separators (§1.5.2) | covered (B2) |
| 069 | P3 | content | — | **deferred: content batch** |
| 070 | P3 | content | Plurals and relative ages (§1.5.2) | covered (B2) |
| 071 | P3 | both | Every glyph via `resolve_glyph`; new map entries; ✨ removed from Generate; ASCII render test (§4.12, §3.11) | covered (B1, B2, B6, B7, B11) |
| 072 | P0 | external | Gates Chat-now promotion only (R29) | **deferred: Console** (TASK-33621.2) |
| 073 | P0 | content | Full-height Tool policy section home (§3.5.2) | **deferred: A** (33789); home in B5b |
| 074 | P1 | external | — | **deferred: Console** (TASK-33621.2 AC#2) |
| 075 | P1 | content | Markup-off prompts and toasts (R33) | **deferred: A** (33790); hardened in B2/B6 |
| 076 | P1 | content | Session width; persona field caps (§3.11) | covered (B7, B11); semantics **D** |
| 077 | P1 | content | Persona Test chat header says `Uses the saved persona`; row scent shows the System prompt | **deferred: D** |
| 078 | P1 | content | Generation fence (§4.11) | **deferred: A** (33791); hardened in B5b |
| 079 | P1 | content | Generation fence; Save returns the saved record (§4.11) | **deferred: A** (33791); hardened in B5b |
| 080 | P1 | content | Virtual list, no page boundary (§1.5.6) | covered (B2) |
| 081 | P1 | both | Summary-first full-height Try it; regrouped records; `budget full: next entry needs N tokens`; 2-line collapse (§3.8.2) | covered (B4, B8) |
| 082 | P2 | content | — | **deferred: D** (D15) |
| 083 | P2 | content | — | **deferred: C** |
| 084 | P2 | external | — | **deferred: Console** (TASK-33792) |
| 085 | P2 | content | Scroll home + focus Name on open; restore on re-entry (§3.11) | covered (B11) |
| 086 | P2 | content | No on/off word on persona rows (§1.5.2) | **deferred: D** |
| 087 | P2 | content | Look / Tool policy as sections of one editor (containers) | **deferred: D** (containers in B5b) |
| 088 | P2 | content | — | **deferred: D** |
| 089 | P2 | content | Look as a section container | **deferred: D** (container in B5b) |
| 090 | P2 | content | Portrait link visible on Profile; row scent `portrait: <character>` | partly (B5b, B2); semantics **D** |
| 091 | P2 | content | Entry columns with on/off words (§3.9.3) | covered (B9) |
| 092 | P2 | both | Virtual list removes the pager; fallback one-row pager in the title (§1.5.6) | covered (B2) |
| 093 | P2 | content | Personas paged until exhausted (§1.5.6) | covered (B2) |
| 094 | P2 | content | Middle ellipsis + tooltip; names get 36 cells at 160x45 (meta on line 2) (§1.5.2); title-row ellipsis | covered (B2, B5b) |
| 095 | P2 | content | TagFilterPicker Enter/Down; picker contract (§1.5.1, §4.6) | covered (B2) |
| 096 | P2 | content | — | **deferred: C** |
| 097 | P3 | external | — | **deferred: Console** (TASK-33793) |

---

## 7. Projected NN/g scorecard and journeys

### 7.1 Scorecard (compliance 0-4; higher is better)

This assumes sub-project A has landed and B5c's gate has passed. The C/D residue visible on B's surfaces counts against the score.

| H | Today | Library | Projected | Why (design elements) | What holds it back |
|---|---|---|---|---|---|
| H1 Visibility of system status | 1 | 3 | **3** | `(…)` counts and `3 of 28`; unsaved in three places from one predicate; a readiness word that is never hidden while disabled; the blocked-destination chip; `#200 of 250`; the summary-first Try it; honest "not applied in chats" copy | A chat's contents are still undisclosed (RP-082, D) |
| H2 Match with the real world | 1 | 3 | **3** | The canonical vocabulary (Lore books, Chat dictionaries, Personas = reusable AI roles, You); glosses; "Options" vs "Settings ›" disambiguated; Run preview | Raw strategy values and persona fields that are not sent (content, D) |
| H3 User control and freedom | 2 | 3 | **3** | Escape leaves a field and never the screen; one Save / Discard / **Stay** dialog everywhere; lossless views; per-kind memory; space undo toast; new entry and Options drafts guarded | Deletes are confirmed but have no undo |
| H4 Consistency and standards | 1 | 2 | **3** | One grip grammar; panes move only at session boundaries; pane-scoped `/` with one meaning; one action-bar slot; one dialog; one Import label; kind vs view vocabulary; Ctrl+S in every editor (R41) | Library still differs on `/` until its follow-up |
| H5 Error prevention | 1 | 3 | **3** | Letters only on navigation surfaces; `o` only on the loaded row; `del` only on items and entries, behind a modal naming its targets; no bulk retargeting; the generation fence; draft domains for entries and Options; New-book draft | Invalid regex still warns rather than blocks (D9) |
| H6 Recognition rather than recall | 1 | 3 | **3** (4 once B6's aggregates land) | Rail counts and glosses; the loaded item named; a strip that never hides a chip at ≥92 or on the XS work stage; labelled entry columns; warnings that never drop; the focus-aware footer | Info has no letter (`<` wraps to it) |
| H7 Flexibility and efficiency | 1 | 3 | **3** | Virtual list; `/ isolde ⏎ ⏎ o`; the entry filter (`/ bellw ⏎`); `e`/`t` sessions; marks that survive search; dynamic F6 | Reorder stays one step per press (C) |
| H8 Aesthetic and minimalist design | 1 | 2 | **3** | Chrome 17 → 7 rows; the Inspector gone; a one-row toolbar; 1-line rows; Create & import in 2 rows; no repeated item name in the header | 10 columns of grips at every width ≥64 (a recorded trade) |
| H9 Recognise, diagnose, recover from errors | 1 | 2 | **2** | Search-aware empty states with `Also in:`; inline regex errors; field-label editor errors; disabled reasons in words; failed counts retry | Import errors (RP-058) and provider errors (RP-059) are content work |
| H10 Help and documentation | 1 | 3 | **3** | F1 from the projection; glossed rail; a landing and empty states that teach; a guide delta per slice | Nothing yet says what a chat will contain (D) |
| **Total** | **11** | **27** | **≈29** (30 with B6's aggregates) | | |

### 7.2 The four jobs on this design

A user-advocate walkthrough of the four jobs on this design. Counts are actions; **W** = wheel notches; **flips** = rail or list closing or reopening during the task.

| Job | Today (report C.2/C.8/C.9, 160x45) | 120x36 | 160x45 | 220x55 |
|---|---|---|---|---|
| J1: import one card → Chat now | 7 incl. a mouse click; the first send refused (Console) | 8 / 0 W / 0 flips (`i`, picker ×5, `o`); readiness visible | same | same; the rail stays (Card is not eligible for the column) |
| J1: find an existing card → chat | 4 keys + a click | `/ isolde ⏎ ⏎ o` = 5 keys | same | same |
| J2: greeting + scenario + save | ≈14 + 10 W + 2 silent corruptions | ≈14 / 0 W; `e` opens a 110-column session | ≈14 / 0 W; 108 columns | ≈14 / 0 W |
| J2: + test the draft | unusable at 120; one exchange at 160 | +5 (`esc`, `t`, type, Enter, `esc`); **0 flips**; a 106-column chat | +5; 0 flips | +4 (`esc`, then `t` focuses the column beside Edit); 0 flips |
| J3a: lore book end to end (create, entries, Try it, link, copy to a character) | ≈26 + ≈30 W, 1 dead end | ≈32 / 0 W; one flip at session start | ≈30 / 0 W; one flip | ≈30 / 0 W; one flip if the column is on |
| J3b: 150-rule dictionary, find and fix the invalid rule, Try it | 70 W, then a dead end ("Validation: OK") | `d / grand ⏎ ⏎ e / unclosed ⏎ ⏎` + fix + Update + `t` ≈ 19 / 0 W | same, split view | same |
| J4: persona edit → test → chat | 4 + a dead end (1.5 fields visible) | ≈15 / 0 W / 0 flips after `e` | same | same |
| J4: see "you" / change "you" | dead end | 0 actions (the rail You row) / ≈7 + the Settings form, state restored | same | +1 when a session closed the rail |

---

## 8. Owner rulings (2026-10-02)

The owner approved all five design sections as presented and answered the twelve open questions in the brainstorming session of 2026-10-01/02. Every answer is option (a). The rulings are final and are applied throughout this spec; the last column says where.

| # | Question | Ruling | Rationale and accepted cost | Applied in |
|---|---|---|---|---|
| Q1 | **Pane borders** | **(a) Bordered:** one `round` border per pane, with zero-row border titles (`Characters 28`, `4 of 28`) and the zero-row fold cue `▾ more`. | No DESIGN.md:236 exception is needed, so G13 is dropped. Borderless tonal panes (4 + 1 chrome rows, 6 more columns) were considered and rejected: they lose the border titles and the zero-row fold cue, and change a design rule. Both options put the first list item on row 7. | §1.1, R19, R34, G1, G13, G14, B0 tests, every mockup |
| Q2 | **Try column at ≥200** (D2): where may it appear, and is it on by default? | **(a) Authoring views only** (Edit, Look, Entries), as a per-kind preference. Defaults: characters **on**, lore **on**, dictionaries **on**, personas **off**. | **Rail cost accepted:** starting an authoring session with the column on closes the rail at 200-222 (Edit/Look) and 200-233 (Entries); at 223+ / 234+ the rail stays. Personas default off because the persona preview reads the **saved** record, so a side-by-side column would look live but test stale text until sub-project D. Rejected: (b) off for every kind (opt-in through `Keep beside ›`); (c) also beside Card and Profile, which would hide the rail whenever an item is open at 200-237 and push Roleplay toward a live chat surface (PRODUCT.md:33). | D2 (§0.3), R20, §2.7, §2.11, B8, MK4 |
| Q3 | **When does the landing show** (D11)? | **(a) Only when there is nothing to restore.** Restore always wins; the Start rail row brings the landing back. | Showing it on every arrival would fight restore, which the owner values (report §6). | R6, §3.10.1, B10 |
| Q4 | **`o` as Chat now** | **(a) Accepted** as the single-letter Chat now, armed only on the loaded row of the list, on the view strip, and on conversation rows (where it is Continue in Console). It ships in **B5c** (D13-gated). | A 2-key J1 path. A stray `o` opens a Console chat but sends nothing, and Ctrl+4 returns with state restored. Rejected: (b) Enter on the focused button only. | R28, §4.6, §4.12, B5c |
| Q5 | **Ctrl+S** (D9; ADR-031 rule 2 bans it) | **(a) Amend ADR-031** to allow an editor-scoped Ctrl+S, and **extend it to every Roleplay editor**: character Edit; persona Edit, Look and Tool policy; lore and dictionary Options; the entry editor. Ctrl+S always presses the visible commit button of the focused editor; elsewhere it is a no-op with a footer-truthful hint. | Today Save works by key in two editors and not in the others (H4). B3 records the amendment and brings the two existing editors under the focus rule; B9 binds the entry editor and Options; B11 binds persona Look and Tool policy. Rejected: (b) keep it in two editors; (c) remove it. | R41, §0.4, §3.1, §3.11, §4.2, §4.6, §4.8 (S2, T1, T5, L6), §4.9, §4.13, G9, B3, B5b, B9, B11, K22, MK5 |
| Q6 | **Selection per kind across app restarts?** | **(a) Session only;** never persisted. | Positions go stale; the landing covers resuming across sessions. Rejected: (b) persist `selected_id`. | §2.11, §4.11, §4.13 |
| Q7 | **The landing's "Continue a character chat"** block. ADR-120 gives bounded cross-character recents to Console Context and gives Roleplay only the complete browse for one resolved character. | **(a) Allowed:** ADR-120 is amended, mirrored in ADR-004, for a bounded Roleplay block: 5 rows, character-bound only, the shared projection, the Data-Profile fence, the canonical opener with typed results, unresolved rows non-activatable, local only. The block is **shown** once B10 lands with the amendment (G5). | J1 ("chat fast") is one of the four equal jobs, and the landing is its surface. Rejected: (b) scope it to the last-loaded character, which is usually empty on arrival because restore wins whenever a character was loaded; (c) no block. | D11 (§0.3), §3.10.2, G5, B10, MK7, MK8 |
| Q8 | **Is cross-kind search a must for J3?** | **(a) Per-kind search plus `Also in:`** on a miss. | One "Search all Roleplay" box over entry text too would depend on sub-project C's entry search and give `/` a second meaning. | §1.4.7, §1.5.4, §1.6 |
| Q9 | **Pane widths in Settings › Appearance?** | **(a) Not in B;** no custom-width keys. | The automatic profiles are tuned for the design sizes; hidden config is not justified (ADR-086:29-30). | §2.11, G1d |
| Q10 | **Pre-import census** (ADR-097). Dev: 557/557 modules. If #2862 lands first: 483/500 modules but **59 LOC** of headroom. B's plan: +3 modules (−1 shed available after #2862) and ≈+3-5k LOC on the Roleplay route. | **(a) Per slice:** measure, defer or shed first, then an owner-signed ledger row for the measured +1 module or LOC delta in that slice's commit. **No blanket re-pin.** | It follows ADR-097's mandated order (defer → shed → exception; ADR-097:45-58). Rejected: (b) hold B until TASK-33276 (PERF-17) frees headroom; (c) lazy-import B's modules, which moves the cost onto the first Ctrl+4 and works against PERF-22. | §5.4, §5.6, §5.10, G10, B0, B1, B5b |
| Q11 | **D13 scope:** does TASK-33621.2 gate *moving* Chat now into the title row, or only *promoting* it? | **(a) Only promoting** (B5c: `o`, the landing hero, the end-to-end test) (B builds no landing hero; §3.10.2). B5b moves the button unchanged. | The button, its gate and its handler do not change in B5b, so nothing new can fail. Gating both would hold the rail and frame (B6, B7), the largest measured wins, behind a Console bug. | R29, §3.6, §5.1, §5.5, B5b, B5c |
| Q12 | **The browse-width trade at the design centre.** Read-only views (Card, Profile, Chats) get 50 columns at 120 and 73 at 160, against today's 58 and 78; editing gets 110 and 108 (today 58 and 78). | **(a) Accepted.** | The rail with the four jobs and counts, a list showing 28 / 18 characters instead of 5 / 9, and a folded Inspector are worth 5-8 columns of a read-only card; 69 content columns is a comfortable reading measure. Rejected: (b) a BROWSE floor of 58, which closes the rail at 120 (against C.5); (c) a Roleplay-owned narrower rail policy (+2 columns at 160), which breaks the shared "Nav" width. | §2.5, B7 acceptance, MK1, MK2 |

**Not owner questions:**
- **D3's "user description" ruling** belongs to sub-project D.
- **Hiding "Add to Console message"** (with `u`, the screen `ctrl+enter` and "Send to Console draft") is mandated by ADR-031 rule 4, an inert advertised action (G9b).

---

## Appendix A. Evidence still to capture at implementation (not owner calls)

- **Strip widths with real CSS.** §3.3's measurements assume `padding: 0 1` (`PS:1171-1191`) and a 1-cell tight variant. Pin them with Pilot (all four kinds at content 46, 69 and 74; the character strip at 34; step 5 for the dictionary strip at 34) before B5b's screenshots.
- **Footer lines** (§4.8) are simulated from `AppFooterStatus`'s rules. Verify them live before pinning.
- **The D14 dry-run census** of the `ListView`-API test migration on a scratch branch decides A versus B before B2 merges.
- **The census of tests relying on nav-bar arrival focus** (before B3) and on F-031's auto-select (before B10).
- **Boot-byte deltas** for removing only the `#personas-*` items from `AT:636-700`'s grouped rules are re-estimated in B7.
- **Index plans** for any aggregate query in B6 are captured with `sqlite_stat1` absent and recorded in `scripts/index_plan_pin_census.tsv`.

## Appendix B. Review issue → resolution (traceability)

Every issue the four-lens review raised against the integrated draft, with where this spec resolves it. The owner rulings of §8 are applied.

| Review (severity) | Issue | Resolution here |
|---|---|---|
| consistency, governance, feasibility, user-advocate (**blocker**) | The posture was picked per view: rail/list flaps on Edit ↔ Test chat at 120-174; Test chat starved to 46; +TRY closed the rail on Card at 200+ and flapped it | **Work sessions** (§2.3, R9); Try column limited to authoring views (§2.7, R20); Q2 ruled (a): authoring views only, per-kind defaults |
| consistency (**blocker**) | Save dropped the EDIT posture | Save does not change the session phase; a Pilot test asserts editor width 110/108 after Save (§2.3, B7) |
| governance (**blocker**) | Entry form and lore Settings drafts sat outside the ADR-046 aggregate | New domains `entry_form_dirty` and `lore_settings_dirty`, a `LoreSettingsEdited` message and a dated ADR-046 amendment (R30, §3.12, G4, B9) |
| feasibility (**blocker**) | D14-A assumed the local list is in memory | Database windows with SQL search, sort and tag; an end-to-end performance acceptance; a cross-kind count query (§1.5.6) |
| consistency (major) | R2 reordered `MODE_CHIP_ORDER` only | Both tuples reordered in lockstep, with a pin (R2, G7) |
| consistency, feasibility (major) | Width claims wrong; browse narrower than today | Re-derived from the resolver (§2.4); the trade stated (§2.5); Q12; the portrait rule fixed (§3.4) |
| consistency, feasibility, user-advocate (major) | No view-strip fitting | `fit_view_strip` with measured results; no hidden chip at ≥92 or on the XS work stage (R22, §3.3) |
| consistency (major) | "mode" meant two things | "kind" vs "view" (R23) |
| consistency, governance (major) | Five unsaved signals; ● overloaded; the chip used a narrow predicate | Three signals, one aggregate predicate, no ● (R24, R3) |
| consistency (major) | Two empty states | The work pane owns the copy; the list has one line (R25) |
| consistency, user-advocate, governance | `/` defined two ways | Pane-scoped `/`; Ctrl+F always the items search; Library follow-up (R26) |
| consistency (major) | Mockups showed the rejected IA | All mockups regenerated (§1.7, §3.14, §4.14; R40) |
| consistency, user-advocate (major) | B5 gated wholly on D13 | Split into B5a/B5b (ungated) and B5c (gated); Q11 (R29) |
| consistency, governance (major) | Stay relabelled "Keep editing" | "Stay" kept everywhere (§3.12, G4) |
| consistency (major) | Import label drift; RP-041 leftovers | One `Import…` (R27); hand-off verbs (R28); RP-041 Roleplay strings to B6, the rest to the content batch |
| consistency (major) | RP-081 content half unowned | In B4's scope (§3.8.2) |
| governance (major) | Landing cross-character recents need ADR-120 | G5 amendment approved (Q7); the block is shown from B10 |
| governance (major) | `del` armed on conversations/versions | `del`/`space`/`n` only on entries (R31) |
| governance (major) | DESIGN.md:330 header contract unamended | G12 (B1); blocked-destination chip |
| governance (major) | Q10 inverted ADR-097's order | Q10 rewritten (measure → defer/shed → per-slice exception); module plan +3/−1 |
| governance (major) | Type-scoped `BUNDLED_CSS` counted as broad; ledger incomplete | Class/id subjects only; re-keys in B2/B4/B5a/B5b with the ratchet lowered (§2.12, §5.10) |
| governance (major) | ADR-011 gate applied to two slices only | Every qualifying slice (§5.4 item 5, G19) |
| feasibility (major) | Row scent needs aggregates | v1 scent from page fields (B2); aggregates with invalidation in B6 (§1.5.2) |
| feasibility (major) | `ListView` test migration undercounted | Its own line item; harness helpers widened; dry-run census before B2 decides D14 (§1.5.6, B2) |
| feasibility (major) | #2862 LOC headroom and conventions | Q10 for both merge orders; recipient-ceiling rows and re-export identity in the DoD (§5.4, §5.11) |
| feasibility (major) | No shell single-stage mode below 64 | No new machinery; list-first degrade; follow-up task (R38, §2.8) |
| feasibility (major) | Inspector fold undercounted | Method → owner table; B5a/B5b split (§3.7) |
| user-advocate (major) | Readiness hidden at 160x45 | Wrap before drop; never drop while disabled; header blocked chip (§3.2, §1.3) |
| user-advocate (major) | Work-pane header drawn three ways | R21 |
| minors (all lenses) | Try column grip; `LibraryEmergencyReturn` reuse; key names; R18 pinned; rail-fit numbers; slice-plan contradictions; XS focus coupling; landing kind marker; stale numbers; the rename; fold cue; mouse marking and marks lifetime; Library rail coupling; delete form; toast markup; coverage holes; ADR-115 eager set; the run key; ctrl+s advertising; component-pattern registration; `AT` grouped selectors; ADR outline gaps; the home rule; glyph semantics; footer API; glyph map; return-focus ids; the TextArea Enter; citations; Start-row semantics; Create & import rows; header repetition; warning visibility; `<` `>` wrap; the New-book flow; w<60 rules; "Settings" collision | All applied: §1.2 (no grip); §2.1, B0 (`StageReturnBar`); §2.11 (`create_import`, sort); R18; B0 (24/31/35); §5.1, §5.2 (diagram, B11 fallback, census rows); R37; §1.4.3; §2.4; §3.7 + R39; R34; §1.5.5 + R16; §2.1 (recorded coupling); R32; R33; §6; G6; §3.8.2; G9; G16; §2.12 item 4; G1a-G1e; G1e; R3, G20; R35; §4.12; R4 + B3 mount tests; §3.8.2; §5.7.4 paths; R6 + §1.4.1; R5; R12; §1.5.2; R1; §3.9.6; §3.4; R36 |
