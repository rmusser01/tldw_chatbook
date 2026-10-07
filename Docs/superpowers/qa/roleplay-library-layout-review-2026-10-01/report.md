# Roleplay screen vs the Library frame: NN/g + HCI review

Code: `origin/dev @ 84247cb843` (worktree `.worktrees/roleplay-library-ux`). Code refs are `path:line` relative to that worktree; most are under `tldw_chatbook/`, so `personas_screen.py` means `tldw_chatbook/UI/Screens/personas_screen.py` and `personas_*_widget.py` / `personas_*_pane.py` live in `tldw_chatbook/Widgets/Persona_Widgets/`.
Captures: `captures/...` means `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/6e3b0191-cd3d-4e9c-9b08-af582f81f1fc/scratchpad/rp-review/captures/` (each has `.txt`, `.ansi`, most have `.png`), and `rp-review/...` (for example `rp-review/gapcheck/bodies.jsonl`, the logged prompt bodies) and `runs/...` are siblings in that same scratch folder. It is scratch space: copy any capture you want to keep into this QA folder's `evidence/`.
Inputs: the mapping docs in this folder (`map-roleplay.md`, `map-library.md`, `prior-art.md`, `capture-report.md`), `rp-review/metrics.md`, seven review lenses (two NN/g heuristic passes, layout, live task flows, keyboard/accessibility, information architecture, Library-pattern fit), and a **gap round** of three more lenses aimed at what the first round had not covered (Appendix C.7-C.9 and section 11): the end-to-end hand-off to Console with logged prompt bodies, the Personas editor, and realistic data volume. Every finding was checked by an independent verifier; refuted claims were removed and partial ones corrected. Machine-readable list: [`findings.json`](findings.json).

**How to read this**

- **Severity** uses the NN/g 0-4 scale: 1 = cosmetic, 2 = minor, 3 = major (fix soon), 4 = usability catastrophe (fix before release).
- **Priority**: **P0** = severity 4, or severity 3 where a core job's main step fails at the target sizes every time it is done, with no workaround short of blind keyboarding or heavy scrolling. **P1** = other severity 3. **P2** = severity 2. **P3** = severity 1.
- **Scope**: *frame* = fixed by adopting the Library's layout frame; *content* = needs its own fix whatever the frame; *both* = the frame helps but does not finish the job; *external* = a defect outside Roleplay (in the Console) that the redesign depends on but cannot fix itself. Two findings are shell or design-system wide and say so.
- **Terms.** *Chrome*: rows and columns spent on headers, borders, toolbars and hints rather than your data. *Rail*: the left navigation column. *Work pane*: the big centre area where the open item is shown or edited. *Inspector*: Roleplay's right-hand column of actions. *Grip*: a slim strip a collapsed pane shrinks to. *Focus*: which control the keyboard is currently talking to. *Footer hints*: the key reminders on the bottom row. *Hand-off*: anything that sends you from Roleplay into a Console chat.
- The four jobs (all equal, per your brief): **J1** import cards & chat fast; **J2** author characters in depth; **J3** build world info (lore books + chat dictionaries); **J4** manage personas / user profiles. Design sizes: medium 120x36-160x45 and large 200+ columns.

---

## 1. Summary

**Headline: Roleplay scores 11/40 on the ten NN/g heuristics (12 before the gap round; efficiency fell at realistic volume); the Library scores 27/40. Moving Roleplay onto the Library's frame would roughly double what you see at medium sizes, but most of the serious problems are inside the panes, in world-info semantics and in what actually reaches the model, and no frame change fixes those. Two things gate the redesign: on a default install the first message of every Chat-now chat is refused by a Console bug ([RP-072](#rp-072), TASK-33621.2), and lore or dictionaries attached to a character never reach the model ([RP-004](#rp-004), now confirmed live).**

97 findings: **13 P0, 26 P1, 53 P2, 5 P3**. 71 came from the first round (152 verified lens findings plus verifier notes, merged). The gap round's 35 verified findings added 26 new ones (RP-072 to RP-097) and merged 9 into existing ones, raising RP-014 to P0, RP-047 and RP-056 to P1 and RP-067 to P2. 1 claim was refuted in the first round, none in the gap round. Four findings are Console defects that Roleplay depends on (RP-072, RP-074, RP-084, RP-097); two are shell or design-system issues (RP-061, RP-065).

The most important things, in plain words:

1. **The payoff of every job is broken on a default install.** Chat now opens the Console chat, but your first message is refused before it reaches the model. The screen still says "Ready", nothing explains why, and the recovery actions the Console has already built never appear, so that chat can never send ([RP-072](#rp-072), [RP-074](#rp-074), [RP-067](#rp-067)). This is a Console bug already filed as P0 (TASK-33621.2; its fix, PR #2882, is not on dev). No Roleplay change fixes it: do not ship a redesigned Chat now until it lands (decision D13), and ship the small recovery-card fix first.
2. **What leaves Roleplay is not what the screen promises.** Lore and dictionaries attached to a character never reach the model; lore attached to the conversation does ([RP-004](#rp-004), confirmed with logged prompts). Both "Send to Console draft" buttons stage context that the Console then throws away, while still showing it as staged ([RP-047](#rp-047)). A persona chat sends only its System prompt while the test chat uses its Description ([RP-077](#rp-077)), and a persona chat silently runs as a 53-tool agent ([RP-082](#rp-082)).
3. **The screen spends its space on itself.** 38-47% of the height at the medium sizes (31% at 220 columns) goes to a boxed title, a purpose line, a mode strip, borders and padding (the Library spends 16-20%), and the Characters list then stacks seven buttons one per row. At 120x36 you see 5 of 28 characters and 2.5 editor fields. The Library frame fixes this directly ([RP-007](#rp-007), [RP-008](#rp-008), [RP-017](#rp-017)).
4. **Layout bugs hide content whatever the frame.** Once you open a lore book or dictionary, an invisible empty box keeps taking space, so the character editor later shrinks to a 2-row peephole ([RP-001](#rp-001)). Long character fields show 0-1 lines, so edits were saved corrupted in testing ([RP-002](#rp-002)). Dictionary and lore entry forms push most controls off-screen and label none ([RP-003](#rp-003)). In the persona editor three whole sections (tool policy, reactions, the Actor Pack portrait) draw as a title only while Tab still walks into them blind ([RP-073](#rp-073)), and each long persona field fills the whole screen ([RP-076](#rp-076)). These are mostly small CSS fixes; land them first, because they also distort every measurement of the new layout.
5. **At realistic volume, editing world info becomes dangerous.** After any entry action the lore or dictionary table jumps back to row 1, so the next Delete or Move hits entry #1, unseen and with no undo ([RP-014](#rp-014), now P0). Entries cannot be filtered and the table stops at 12 rows: entry #200 of 250 took 99 wheel notches ([RP-056](#rp-056)). With 348 characters, paging skips about 41 names after a scroll ([RP-080](#rp-080)), marks vanish at each page change ([RP-039](#rp-039)), and personas beyond the newest 100 disappear ([RP-093](#rp-093)). Speed is fine: the list loads in about 1 s and search takes 0.34 s.
6. **The persona editor (J4) is the least finished surface.** Besides the hidden sections: a name over 200 characters crashes the whole app and loses every draft ([RP-075](#rp-075)); the Inspector's tool-policy line can show another persona's rules ([RP-078](#rp-078)); after Save then Cancel the card shows the wrong persona ([RP-079](#rp-079)); one editor has four Save buttons with different meanings ([RP-087](#rp-087)). The "user profiles" half of J4 has no home anywhere: Settings sends you back to Roleplay, which has none ([RP-011](#rp-011), decision D3).
7. **The "does it work?" loop is crushed.** The test chat's input and buttons clip away (unusable at 120x36, [RP-010](#rp-010)); dictionary and lore editors show 0-5 entries next to a permanent Try-it box ([RP-006](#rp-006)); and with a realistic sample Try it's own summary is clipped off at 160x45 ([RP-081](#rp-081)). Where these live is decision D2.
8. **Words contradict behaviour.** Personas mode says "who you play", although you ruled personas are the AI's side and the app behaves that way ([RP-011](#rp-011)). Lore has four names ([RP-041](#rp-041)); the Roleplay rail is called "Library" ([RP-042](#rp-042)); a persona's "Enabled" switch does not stop it chatting ([RP-086](#rp-086)).
9. **The keyboard model leaks out of the screen.** Arrival focus sits on the nav bar's Home (Enter leaves Roleplay), Tab walks 14 nav buttons, Escape does not leave text fields, F6 never reaches the lore/dictionary editors, and the footer advertises keys that would be typed as text ([RP-024](#rp-024)-[RP-026](#rp-026), [RP-034](#rp-034)). The Library has already solved most of these.
10. **Context gets thrown away.** Switching modes drops your selection and search ([RP-030](#rp-030)); returning to Roleplay can show an empty list ([RP-021](#rp-021)); a search typo says "No characters yet" ([RP-022](#rp-022)). The Library frame does not provide per-mode memory; it must be added.

**Fix first (small, high value, independent of the frame):** the Console recovery card ([RP-074](#rp-074)), the persona-name crash ([RP-075](#rp-075), one line), keeping the entry selection across entry actions ([RP-014](#rp-014)), the three hidden persona sections ([RP-073](#rp-073)), the leaked slots ([RP-001](#rp-001)), the 0-1-line fields ([RP-002](#rp-002)), the unscrollable Settings tabs ([RP-005](#rp-005)), the search box ([RP-009](#rp-009)), and the stale persona displays ([RP-078](#rp-078), [RP-079](#rp-079)). All are effort S.

**What to keep:** the dictionary and lore Try-it previews (better than SillyTavern's), the Save/Discard/Stay leave guard (better than the Library's; it covers the persona editor too), Generate-then-Accept for AI text, the single Escape owner, mode/selection restore when an item is selected, fast Ctrl+F > Enter > Enter (it still takes 0.34 s across 348 characters), and the c/p/d/l mode keys (section 6).

**Your decisions** are in section 9. The ones that block design work: Inspector placement (D1), test chat / Try-it placement (D2), shell reuse and the ADR-086 amendment (D4), character attachments (D6, now unblocked because [RP-004](#rp-004) is confirmed), gating Chat now on the Console fix (D13), paging versus one long list (D14) and what a persona sends to the model (D15). Persona semantics (D3) is already ruled; the gap round turns its optional "You" row into a recommendation and adds one question about a user description.

---

## 2. Scorecard (NN/g heuristics)

This is a **compliance score: 0-4, higher is better** (4 = excellent compliance, 0 = no compliance). It is indicative, a summary of the evidence below rather than a measurement. The Library column scores the Library as the template, including its own defects (section 10).

| # | Heuristic (one-line meaning) | Roleplay | Library | Why |
|---|---|---|---|---|
| H1 | **Visibility of system status**: the screen tells you what is going on, promptly and truthfully | 1 | 3 | RP: counts read 0 beside full lists, typed search text is invisible, Save is off-screen, an invalid regex shows "Validation: OK", attachments are invisible from the lore side; gap round: "Ready" before refused sends, a tool-policy line describing another persona, a "Loading…" that never ends, a staged source the Console discarded. Lib: "(…)" while loading, "Saved" / "Modified now · v2", "1-4 of 4"; but on reader routes its rail closes below ~126 cols and the header stops saying where you are. |
| H2 | **Match between system and the real world**: uses your words and concepts, not the system's | 1 | 3 | RP: persona copy contradicts behaviour, raw ids ("character_lore_first", "session_scoped"), unexplained "Actor Pack"/"Buddy", four names for lore; persona Description and traits look like they shape the chat but are not sent, and "Enabled" does not disable. Lib: plain nouns with short glosses; one catch-all "Details" bucket. |
| H3 | **User control and freedom**: easy exits and undo | 2 | 3 | RP: safe single Escape, Save/Discard/Stay leave guard (persona editor too), AI text needs Accept; but Escape does not leave fields, entry delete/detach and rule delete have no undo, mode switches and page changes discard state. Lib: Escape leaves a field, delete has inline confirm; but it loses the open item across destinations and Prompts hard-vetoes Back. |
| H4 | **Consistency and standards**: the same thing looks and works the same way | 1 | 2 | RP: rail called "Library", Ctrl+F not "/", banned Ctrl+S, each mode saves differently (and the persona editor four ways), the rail changes shape by mode, hand-off verbs overloaded. Lib: "/" search, uniform rails; but two collapse grammars, inconsistent unsaved/delete handling, Ctrl+S too. |
| H5 | **Error prevention**: design out slips before they happen | 1 | 3 | RP: blind edits saved corrupted, instant entry delete and detach, invalid regex saved silently, stray single-letter keys switch modes, Delete silently retargets to marked rows; gap round: after an entry action the next Delete hits entry #1, hidden persona fields get saved blind, an over-long name crashes the app. Lib: Delete behind "More actions" with danger styling; state-machine action bar. |
| H6 | **Recognition rather than recall**: show options and labels instead of making you remember them | 1 | 3 | RP: unlabelled entry fields, counts/glosses for 3 of 4 modes hidden, "m" (mark) undocumented, the open item is not named, end-truncated duplicate names, an invisible persona-to-character link. Lib: rail counts + glosses, labelled fields with help lines, focus-aware footer. |
| H7 | **Flexibility and efficiency of use**: shortcuts for experts without burdening novices | **1** (was 2) | 3 | RP: c/p/d/l, find a card in 4 keys (0.34 s across 348), bulk marks, import remembers its folder; but no key for Chat now, palette commands only navigate, F6 misses editors, and at realistic volume entries have no filter (99 wheel notches to entry #200), reorder moves one step per ~1.25 s, no key turns a page, marks last one page, the tag picker's Enter does nothing and personas past 100 cannot be found. Lib: "/" plus per-list filters, single-letter verbs, work-first override when editing. |
| H8 | **Aesthetic and minimalist design**: every element earns its space | 1 | 2 | RP: 38-47% chrome, 9-row toolbar, 26 buttons for one selection, Buddy controls in every mode, blank bands, three framing layers, full-screen persona fields and a fixed 33-row visuals block. Lib: one header row, slim grips; but items panes spend 8-19 rows before item 1, three border layers, uncapped reader prose. |
| H9 | **Help users recognize, diagnose and recover from errors**: plain messages with a remedy | 1 | 2 | RP: widget ids in errors, raw parser text on import, generic provider error with an env-var remedy, invisible regex warning, "No characters yet" on a typo; gap round: a Python traceback on an over-long persona name, and a refused chat with no reason or recovery (Console). Lib: filter-aware empty copy and plain toasts; its unsaved veto offers no in-place Save; error copy was less tested. |
| H10 | **Help and documentation**: findable, task-focused help | 1 | 3 | RP: F1 is a raw dump titled "PersonasScreen Shortcuts", the User Guide contradicts the screen, empty states teach little (the one-line mode descriptors are good); Settings sends "user profiles" back to Roleplay, and nothing says what a chat will contain. Lib: F1 built from the live footer, a "Get started" landing. |
| | **Total (/40)** | **11** (was 12) | **27** | |

**Change in the gap round.** Roleplay's H7 drops from 2 to 1. The accelerators that earned the 2 (the 4-key find, bulk marks, c/p/d/l) still work for characters, but the realistic-volume check showed that world info, one of the four equal jobs, has none: no entry filter, a 12-row table, one-step reorder, and paging and marks that break at page boundaries ([RP-056](#rp-056), [RP-096](#rp-096), [RP-092](#rp-092), [RP-039](#rp-039), [RP-095](#rp-095), [RP-093](#rp-093)). The other nine scores stand: the new evidence deepens failures already scored 1 (H1, H2, H5, H9) rather than changing them, and H3 keeps its 2 because the leave guard and the safe Escape also hold in the persona editor. Library scores are unchanged; its pager and filters behaved well at 36 conversations, and it was not tested at larger volume.

---

## 3. Measured space efficiency

Measured from live captures with the same seeded profile (28 characters; Roleplay with a character selected vs Library with a conversation open). Sources: `rp-review/metrics.md` and the layout lens (Appendix C.1). "Content share" = the screen height left after chrome rows.

| Measure | 120x36 Roleplay | 120x36 Library | 160x45 Roleplay | 160x45 Library | 220x55 Roleplay | 220x55 Library |
|---|---|---|---|---|---|---|
| Chrome rows (above + below content) | 13 + 4 = **17** | 5 + 2 = 7 | 13 + 4 = **17** | 6 + 3 = 9 | 13 + 4 = **17** | 6 + 3 = 9 |
| Content share of height | **53%** | 81% | **62%** | 80% | **69%** | 84% |
| Rows available to the main list | 10 | 21 | 19 | 30 | 29 | 40 |
| List items visible | **5 of 28** | 6 of 6 (3 rows each) | **9 of 28** | 6 of 6 | **14 of 28** | 6 of 6 |
| Columns given to the work pane | 58 of 120 (48%) | 50 (42%) | 78 of 160 (49%) | ~62 (39%) | 108 of 220 (49%) | ~117 (53%) |
| Columns still used by collapsed side panes | 13 + 11 | 5 + 5 | 13 + 11 | 5 + 5 | 13 + 11 | 5 + 5 |
| Editor fields visible, clean session | **2.5** (text rows 1, 1, 2) | prompt 4 / note 3 (note body 4 rows) | **5** (text rows 1, 1, 2, 1, 1) | prompt 4 / note 3 (note body 16 rows) | not measured; heights are fixed, Scenario/Post-history/Creator notes still 0 rows | - |
| Persona editor fields visible (gap round, [RP-076](#rp-076)) | **1.5** (Name + a ~14-row Description box) | - | **1.5** (Name + a 23-row box) | - | **1.5** (Name + a ~33-row box) | - |
| Editor after one Lore visit ([RP-001](#rp-001)) | **2-5-row window** (persona editor: 6 rows, 0 text lines) | - | **5-9-row window** | - | - | - |
| Inspector area that is blank | 58% | (no inspector) | 71% | (no inspector) | **85%** | (no inspector) |
| Border-glyph share of all cells | 23.6% | 15.0% | 19.0% | 16.4% | 15.4% | 12.9% |

At 80x24 (degrade-gracefully tier) Roleplay spends 58% of the height on chrome and shows **1** character; the Library spends 21%.

**Where the Library is not better (do not copy blindly):** with its default settings its reader/editor is *narrower* than Roleplay's centre at 120 and 160 columns (50/62 vs 58/78); its items lists use 3 rows per item; its items panes spend 8 (Conversations, Prompts) to ~19 (Notes) rows before the first item; it has more blank cells overall (64-77% vs 36-56%); and its reader prose is uncapped. Its real advantages are structural: one header row, one-line rail rows with counts, bounded side-pane widths, an empty pane giving its width away, 5-column grips and one scroller per pane.

**What fixing the frame buys** (arithmetic from counted rows):

| Change | 160x45 list rows | characters visible (2-line / 1-line rows) | 120x36 list rows | characters visible (2-line / 1-line rows) |
|---|---|---|---|---|
| Today | 19 | 9 | 10 | 5 |
| Toolbar 7 rows -> 2 ([RP-008](#rp-008)) | 24 | 12 / 24 | 15 | 7 / 15 |
| + chrome 13+4 -> 6+2 ([RP-007](#rp-007)) | 33 | 16 / all 28 | 24 | 12 / 24 |

For editors, the Library's width resolver with Roleplay-specific settings and a work-first override would give the editor **106 columns at 120** and **90-146 at 160** (today 58 and 78), but only if the fields can grow ([RP-002](#rp-002), [RP-016](#rp-016)). With the Library's *default* settings the editor would get 50 and 62, worse than today ([RP-017](#rp-017), Appendix C.4).

**Caveat:** earlier readings of "wasted rows in Dictionaries/Lore" (`metrics.md` section C; capture-report #8, #13) are partly caused by [RP-001](#rp-001), the leaked empty slot. The gap round re-measured one cleanly: a fresh 160x45 visit to a 250-entry lore book shows 10 entry rows; the 3-row window seen later in the same session needed the leaked band. Fix RP-001 before re-measuring the rest.

**At realistic volume (gap round, Appendix C.9).** A scaled copy of the profile held 348 characters, 64 (then 114) personas, 43 lore books (one with 250 entries), 28 dictionaries (one with 150 rules) and 30 chats for one character. Speed holds; navigation does not.

| Measure (160x45 unless noted) | Result |
|---|---|
| Cold first visit to Roleplay (348 characters) | header 0.35 s, list 0.99-1.11 s; no event-loop stalls during Roleplay |
| Search across 348 (Ctrl+F) / page change / mode switch | 0.34 s / 0.05-0.30 s / 0.05-0.30 s |
| Characters visible per page of 50 | 8.5 at 160x45; 3.5 at 120x36, under up to 3 rows of page bar ([RP-092](#rp-092)) |
| Lore entry rows visible (250-entry book) | 4 at 120x36; 10 at 160x45 on a clean visit (3 with the RP-001 band); 10 at 220x55 (12-row cap) |
| Reach entry #200 with the wheel | 99 notches; 115 after a save, because the table returns to row 1 ([RP-056](#rp-056), [RP-014](#rp-014)) |
| Reach dictionary rule #150 | 70 notches |
| Move one entry one place | ~1-1.25 s, rewriting all 250 rows ([RP-096](#rp-096)) |
| Try-it summary with a 3-line sample | clipped off at 160x45; visible at 220x55 ([RP-081](#rp-081)) |

---

## 4. Themes

**T1. The frame spends the screen on itself.** Stacked header bands, a one-per-row toolbar, three framing layers, wide collapsed handles, a fixed 2:4:2 width split and a fourth "Inspector" column all take space from the item you are working on. *For the redesign:* this is exactly what the Library frame fixes. Take its one-row header (as an inline DestinationHeader), counted rail, bounded side panes, grips and empty-pane rule, but with Roleplay-specific width settings, because the Library defaults starve editors at medium sizes. Findings: RP-007, RP-008, RP-017, RP-018, RP-019, RP-020, RP-029, RP-032, RP-066, RP-068, RP-092.

**T2. Container bugs inside the panes hide content.** Leaked empty slots, fixed 0-1-row text boxes, persona fields that fill the screen, whole persona sections collapsed to a title, form rows whose first box eats the width, tabs that cannot scroll, a test chat and a Try-it result capped at 60% with no scroll, a fixed 33-row visuals block, nested scrollers, an editor that reopens scrolled to the bottom and a collapsing search box. *For the redesign:* the frame does not touch these. They are mostly small CSS/container fixes; land them first and protect them with geometry tests ("every control's region is inside the pane at 120x36 and 160x45"), or they will be copied into the new frame and keep corrupting measurements. Findings: RP-001, RP-002, RP-003, RP-005, RP-006, RP-009, RP-010, RP-027, RP-054, RP-073, RP-076, RP-081, RP-085, RP-089.

**T3. World info is disconnected from where it is used.** Character-attached lore does not reach chats (now confirmed); attachments are silent copies with no "used by" or "out of date"; card-embedded lorebooks are invisible; Detach can destroy the only copy; the route that does work (attach to the conversation) lists bare chat titles. *For the redesign:* this needs the semantics decision (D6) before the new Info/"Used by" surfaces are designed. Building a "Refresh copy" button on top of an inert feature would be wasted work. Findings: RP-004, RP-012, RP-013, RP-014, RP-083.

**T4. Words do not match behaviour.** Persona copy contradicts your ruling, one object has four names, the rail is called "Library", "Actor Pack" and "Buddy" go unexplained, palette commands promise actions they do not perform, the guide describes a different screen, validation says OK when it is not, a persona's "Enabled" does not disable it, and its link to a character is invisible. *For the redesign:* the new rail rows carry permanent one-line glosses, so a canonical vocabulary must be fixed first (Appendix C.5 proposes one). Empty states that explain nothing belong here too. Findings: RP-011, RP-041, RP-042, RP-043, RP-044, RP-047, RP-053, RP-055, RP-057, RP-060, RP-067, RP-069, RP-070, RP-086, RP-089, RP-090.

**T5. State and context get thrown away or go stale.** Mode switches reset selection, search, sort and marks; returning can render an empty list or a hidden filter; pane collapse is forgotten; collapsing a pane drops keyboard focus; the test chat keeps a stale greeting; persona displays are not refreshed (another persona's tool policy, the previous persona's card, a "Loading…" that never resolves). *For the redesign:* the frame brings persisted pane preferences, but not per-mode memory (the Library loses the open item too). Add per-mode memory deliberately, and render the work pane from the loaded item (ADR-084). Findings: RP-021, RP-030, RP-031, RP-046, RP-064, RP-078, RP-079, RP-088.

**T6. The keyboard model leaks.** Focus starts on the nav bar, Tab leaves the screen, Escape does not leave fields, F6 lands on secondary controls, single letters fire from buttons, the footer ignores focus, F1 is a raw dump, tab stops are invisible, list/chip/button focus is hard to see, and the tag picker ignores the Enter its hint promises. *For the redesign:* most fixes are Library patterns to copy (Tab region, "leave field" Escape, focus-aware footer, contextual F1, dynamic F6 targets). Lift them into the shared frame so every destination gains. Findings: RP-024, RP-025, RP-026, RP-027, RP-033, RP-034, RP-035, RP-036, RP-062, RP-063, RP-065, RP-071, RP-095.

**T7. Errors and edits fail quietly.** Invalid regex saved silently, instant deletes, mode-dependent save rules (and four commit models inside the persona editor), unsaved state far from the editor, Delete retargeting to marked rows, raw import errors, an env-var remedy for a missing API key, "No characters yet" on a typo, counts reading 0, and one unescaped error toast that crashes the app. *For the redesign:* adopt the Library's state-machine action bar (shows only valid verbs; Save/Discard when dirty) and its filter-aware copy, but keep Roleplay's stronger Save/Discard/Stay guard. Findings: RP-015, RP-022, RP-028, RP-037, RP-038, RP-039, RP-040, RP-058, RP-059, RP-061, RP-075, RP-087.

**T8. Authoring and browsing depth is incomplete.** Example dialogue cannot be seen or edited, alternate greetings cannot be chosen for Chat now, the editor is one long scroll, card and editor disagree on field order, the test chat misstates what is in play, and a character's chat history and search are thin. *For the redesign:* work-pane modes (Basics / Prompting / Greetings / Voice & look / Info) give these fields a home; the rest is content work. Findings: RP-016, RP-023, RP-045, RP-048, RP-049, RP-050, RP-051, RP-052, RP-056.

**T9. The hand-off to Console, the payoff of every job, is broken or opaque (gap round).** The first Chat-now message is refused on a default install and the recovery card never appears; character attachments and both "Send to Console draft" payloads are dropped before the model; a persona sends different fields in the test chat and in Console, and runs as a 53-tool agent without saying so; "Ready" is shown throughout; the Console rail names the wrong character; and its log points at a harmless error instead of the real one. *For the redesign:* most of this is not frame work, and four items are Console defects. Treat TASK-33621.2 as a hard dependency of the new Chat now (D13), and add one end-to-end test per job that asserts what the provider actually received. Findings: RP-004, RP-047, RP-067, RP-072, RP-074, RP-077, RP-082, RP-083, RP-084, RP-097.

**T10. Realistic volume breaks world-info editing and browsing (gap round).** At 250 entries, 150 rules and 348 characters, every entry action returns the table to row 1 (so the next Delete hits entry #1), entries cannot be filtered, the table stops at 12 rows, reordering is one slow step at a time, columns hide content and on/off, pages open at the wrong scroll offset, marks and filter counts break at page boundaries, end-truncated names make duplicates identical, the tag picker needs the mouse, and personas past the newest 100 vanish. *For the redesign:* decide paging versus one virtual list (D14), and give the entries list a filter, full height and selection anchored to the entry (D12). Findings: RP-014, RP-022, RP-030, RP-039, RP-049, RP-056, RP-080, RP-081, RP-091, RP-092, RP-093, RP-094, RP-095, RP-096.

**T11. The persona editor (J4) is the least finished surface (gap round).** It was never opened in the first round. It has three sections drawn as a title only, full-screen text boxes, a crash on a long name, a tool-policy summary that can describe another persona, a stale card after Save, an editor that reopens scrolled to the bottom, inert Enabled/Mode switches, four commit models, a loading message that never resolves, a jargon-heavy fixed-height visuals block and an invisible character link; and the "user profiles" half of J4 has no surface at all. *For the redesign:* the proposed persona work-pane modes (Profile, Edit, Look, Tool policy, Test chat, Info; Appendix C.8) give each subsystem its own pane and one commit bar, and decision D3 settles the "You" row. Findings: RP-011, RP-073, RP-075, RP-076, RP-077, RP-078, RP-079, RP-085, RP-086, RP-087, RP-088, RP-089, RP-090.

---

## 5. Top issues (P0 and P1)

Each entry gives: what is wrong, who it hurts and when, the evidence, the recommendation and the scope. Full evidence lists are in Appendix A and `findings.json`. P0 is ranked: the Console blocker first (it breaks every job's payoff), then the other severity-4 findings, then severity 3. P1 is in id order; the gap round's additions are RP-074 to RP-081.

### P0 (13 findings)

<a id="rp-072"></a>
#### RP-072 · On a default install, the first message in every Chat-now chat is refused before it reaches the model, with no reason and no way out (Console defect TASK-33621.2: a blocking dependency, not redesign work)

**P0 · severity 4 · scope: external (Console; TASK-33621.2) - blocks the redesign's primary action · kind: bug · effort M** · sizes: all (verified at 160x45 and 120x36)  
*Principle:* H1 Visibility of system status; H9 Help users recognize, diagnose and recover from errors; completion of J1  
*Sources:* G1-01 · *Verdict:* CONFIRMED (live, with full traceback; capture-off control completed)

- **What's wrong:** Chat now opens a Console chat with the character's greeting. When you type and press Enter, the message is refused about 4 ms after it is saved, before any request is sent. The screen still says 'Ready', the message row shows only a small 'blocked', and that chat refuses every later send. The cause is a Console bug: the character's or persona's system prompt has no saved owner, so building the trace record for the request fails. It happens because 'exchange capture' is on by default; with it off, the same sends complete.
- **Who it hurts, and when:** Job 1 cannot finish: import a card, Chat now, type, and nothing comes back, with no reason, no recovery, and a chat that can never send again. All four jobs end in a chat, so every Roleplay job's payoff is broken on a default install. *When:* Every first send of every Chat-now session on the default config: 4 of 4 in the gap round, and 4 of 4 in the earlier rv-flows, rv-flows-2 and rv-ia runs.
- **Evidence:**
  - `rv-gap1 160x45, golden profile, body-logging mock on :18777` (live; G1-01): Isolde, Vex and Cozy Storyteller: Chat now, type, Enter. Each logged 'controller_submit status=blocked' ~4 ms after 'durable_commit status=succeeded'; bodies.jsonl gained 0 lines. A plain send from a new Console tab in the same run reached the mock (20:28:15, tools=53)
  - `rp-review/gapcheck/trace.log 20:35:48, 20:36:46, 20:37:30, 20:48:09 (out-of-tree sitecustomize instrumentation; no product code changed)` (live; G1-01): Every refusal is 'TraceProvenanceAlignmentError(trace provenance category mismatch: system)' on [system 'You are Captain Isolde Varga…', user …], then SUBMIT RESULT BLOCKED with 'Trace provenance could not be saved. Retry, Send without capture, or Cancel.'
  - `tldw_chatbook/Chat/console_chat_controller.py:11955-11968; Chat/console_prepared_request.py:929-940; Chat/console_trace_provenance.py:841-857, 989-994` (code; G1-01): An unsaved provider row gets ACTIVE_REQUEST provenance; leading system rows fill the 'system' category, which accepts only saved or RENDERED_SYSTEM descriptors, so building the request raises
  - `console_chat_controller.py:12336-12354, 12036-12074` (code; G1-01): A bare 'except Exception' turns the error into a TRACE_PROVENANCE pause and sets run state BLOCKED; nothing is logged
  - ...and 6 more in [Appendix A](#a2-rp-072).
- **Recommendation:** Make TASK-33621.2 a hard dependency of the redesign's primary action (decision D13); do not ship or announce the new Chat now until it is fixed. The fix (PR #2882 / TASK-32951, not on dev at 84247cb843): in _build_durable_trace_request (console_chat_controller.py:11955-11968) give leading unsaved system rows RENDERED_SYSTEM provenance. Log exception_type at :12353 (today the refusal logs nothing). Ship the recovery-card fix (RP-074) first: it turns this into one extra click. Add Roleplay end-to-end tests with no stubbed builder (a character with a greeting: Chat now, send, assert the gateway received exactly one request; the same for a persona), and re-run the body-logging gap check after the fix.
- **Library frame:** The Library's 'Use in Console' and 'Resume conversation' land in generic sessions with the system prompt off, which do not trip this path (the gap round's baseline generic send completed). The frame changes nothing here: a redesigned Roleplay whose primary action is Chat now inherits the defect unchanged.
- **Verifier corrections applied:** Scope: neither Roleplay frame nor Roleplay content - an external Console dependency (TASK-33621.2, P0, which already names 'Start new chat' character chats); no Roleplay redesign work fixes it, so it is tracked as a blocking dependency. The capture-off control's ~1.2 s first reply holds for Isolde and Vex; the persona took ~12 s ('Connecting tools… · 11s'). This explains the 'cross-surface note' blocks in Appendix C.2; the capture_provider_failure ERROR logged beside them is a separate, harmless bug (RP-097).

<a id="rp-004"></a>
#### RP-004 · Lore books and dictionaries attached to a character are never applied in Console chats (confirmed live with logged prompts)

**P0 · severity 4 · scope: content · kind: bug · effort M** · sizes: n/a (behaviour)  
*Principle:* H1 Visibility of system status; H2 Match with the real world (an attachment shown as active should act)  
*Sources:* missed:nng-a#1, G1 (gap1 section 1) · *Verdict:* CONFIRMED live (gap round G1; originally verifier-sourced from code)

- **What's wrong:** A character card shows 'World Books (1)' and offers 'Attach world book...', so users expect that lore to shape chats. Logged prompts show it never does: Isolde's attached Kepler-9 entries are absent from what is sent, and Vex's attached dictionary does not rewrite 'police' as it should. Every live Console send path passes no character to the lore/dictionary step. Lore attached to the conversation itself does work.
- **Who it hurts, and when:** World-builders (job 3) attach lore to a character, test it, then chat and see no effect, with nothing on screen explaining why. The working route (attach the book to the conversation) is buried: Lore > Attachments > Attach to conversation (RP-083), or below the fold in Console Inspect. This also makes the 'copy is out of date' problem (RP-012) moot until it is fixed. *When:* Every chat with a character that has attached lore or dictionaries.
- **Evidence:**
  - `tldw_chatbook/Chat/console_chat_controller.py:1284-1297` (code; missed:nng-a#1): capture_prompt_transform_inputs passes char_data=None
  - `tldw_chatbook/UI/Screens/chat_screen.py:14400-14423` (code; missed:nng-a#1): 'Conversation-only: char_data is None (native sessions carry no character card)'
  - `tldw_chatbook/Chat/console_runtime.py:567-574, 630` (code; missed:nng-a#1): Same char_data=None on the other live send paths
  - `tldw_chatbook/Character_Chat/world_book_manager.py:69` (code; missed:nng-a#1): resolve_character_world_books is reachable only through the retired chat_events.py path
  - ...and 8 more in [Appendix A](#a2-rp-004).
- **Recommendation:** Confirmed: rule on D6 now. Interim (D6 option C): have Chat now link the character's attachments onto the new conversation, which reuses the one path proven to work end to end. Target (option B): pass the session's character id into capture_prompt_transform_inputs (console_chat_controller.py:1273-1297) and the two runtime appliers (Chat/console_runtime.py:567-574, 630); the session already knows its character. Until either ships, the 'Used by' view must say 'Not applied in chats' for character copies, and no 'Refresh copy' UI should be built on the inert feature. Add a send-path test against the live Console path - the existing test (Tests/Character_Chat/test_character_world_book_send_path.py) mirrors the retired chat_events.py path.
- **Library frame:** No Library analog. The frame does not touch it.
- **Verifier corrections applied:** Gap round: confirmed live with a body-logging mock, but only with [console] exchange_capture = false, because on the default config every character send is refused first (RP-072). Path corrected: console_runtime.py is under tldw_chatbook/Chat/, not UI/Console_Modules/. The Console Inspect rail can attach books and dictionaries to the active conversation (UI/Console_Modules/retrieval.py:584-624), so a workaround exists, below Inspect's fold.

<a id="rp-014"></a>
#### RP-014 · After any lore or dictionary entry action the table jumps back to row 1, so the next Delete, Move or Update silently hits entry #1; Delete and Detach have no confirmation or undo

**P0 · severity 4 · scope: content · kind: bug · effort S** · sizes: all; worst at 120x36 and 160x45, where row 1 is usually off-screen  
*Principle:* H5 Error prevention; H1 Visibility of system status; H3 User control and freedom; keep the selection stable across a mutation  
*Sources:* NA-06, missed:flows#1, G3-01 · *Verdict:* CONFIRMED (reproduced live in the gap round; raised from P1/3 to P0/4)

- **What's wrong:** Every entry action (Add, Update, Delete, Move) rebuilds the entry table, which puts the cursor back on row 1 and loads row 1 into the form. The next press of Delete, Move or Update therefore acts on entry #1, which may be hundreds of rows away and off-screen; the only cue is the form text changing. Delete itself asks nothing and shows no toast, and Detach on a character is just as instant (for a lorebook that came inside an imported card it destroys the only copy).
- **Who it hurts, and when:** World-info builders (job 3) lose or reorder an entry they never touched, invisibly: in a 250-entry book, deleting #199 and pressing Delete again deleted #1, 198 rows away. Lore has no version history, so the entry is gone; dictionary recovery means reverting the whole dictionary. Moving an entry more than one place is impossible without re-finding it after every press, and each extra press reorders entry #1. *When:* Every second entry action in a row (delete two entries, move one entry more than one place, Update then Delete); every Delete or Detach.
- **Evidence:**
  - `personas_lore_detail.py:174-199, 612-619; personas_dictionary_detail.py:242-266` (code; NA-06): Add | Update | Delete | Move up | Move down share one style; Delete posts directly
  - `personas_screen.py:6510-6518 (dictionary), 6673-6683 (lore)` (code; NA-06): Handlers call delete_entry / delete_world_book_entry with no dialog; no success toast
  - `captures/review-rv-nng-a-lore-entry-form-before-delete-160x45 row 27; review-rv-nng-a-lore-entry-deleted-no-confirm-160x45 rows 24-26` (live; NA-06): 'Ballast' gone at once; rail '6 entries' -> '5 entries - on'; form loads the next entry
  - `personas_screen.py:6321-6352 -> world_book_manager.py:864-893` (code; missed:flows#1): Detach from a character is immediate with no confirmation or undo
  - ...and 8 more in [Appendix A](#a2-rp-014).
- **Recommendation:** (1) Keep the selection: in update_entries() capture the selected entry and the scroll offset before table.clear() and restore both (lore by its stable DB id; dictionaries by content or index, because their entry ids are positional, personas_screen.py:5800). After Delete select the neighbour at the same index, never row 0; after Add select and scroll to the new row. Disable Delete/Update/Move while the reload runs and drop presses that arrive during it. (2) Say what happened: 'Deleted "Foxglove Vale Pact" · Undo'. (3) Move Delete away from Add/Update (right-aligned, error variant) behind an inline confirmation or an 8-second Undo; add version history or soft delete for lore entries. (4) Confirm Detach, with stronger copy when the attachment came from a card ('This is the only copy of Glass Coast Lore'). Regression test: 250 entries, select #200, Delete twice, assert #1 still exists.
- **Library frame:** Library Prompts keeps Delete under 'More actions' with danger styling; Notes uses 'Danger > Delete' with an inline confirmation. The Library has no in-pane entry tables, but Roleplay's own rail already re-anchors the selected book by id after a reload (mark_active_row, personas_screen.py:6608); the entry table needs the same. The frame does not reach these forms.
- **Verifier corrections applied:** Verifier: worse than stated - the form auto-loads the next entry after a delete. DESIGN.md:127 ('Confirm irreversible actions') is the governing rule (ADR-031:10 covers keys, not buttons). Gap round (G3-01): the earlier 'form then loads the next entry' was this row-0 reset. It does not depend on volume (reproduced on entry #3); volume only hides it because row 1 is off-screen. Raised to P0/severity 4 because an action lands on an entry the user never selected or saw. The same reset explains RP-040's 'after Add the form jumps to the first entry'. Dictionaries share the clear-and-reload path (code-verified, not driven live). Also guard a second reorder press that lands while the ~1 s per-entry write loop is still running (RP-096).

<a id="rp-001"></a>
#### RP-001 · Once a dictionary, lore book or editor has been opened, its empty slot stays on screen and squeezes every later view (the character editor shrinks to a 2-5-row window)

**P0 · severity 4 · scope: content · kind: bug · effort S** · sizes: all; worst at 120x36 (2-row editor window), 160x45 (5-9-row window)  
*Principle:* H8 Aesthetic and minimalist design; H1 Visibility of system status; stable layout (DESIGN.md:125)  
*Sources:* LY-01 · *Verdict:* CONFIRMED (reproduced live with a smaller trigger than first reported)

- **What's wrong:** The screen keeps an invisible, empty box for every heavy view you have opened once. Those empty boxes keep taking an equal share of the height, so after a normal tour of the four jobs the character editor becomes a 2-row peephole with blank space below it, and lore/dictionary tables show 0-1 rows under a blank band.
- **Who it hurts, and when:** Authors of characters (job 2) and world info (job 3) work through a 2-5-row window while most of the pane is blank. The blank space reads as a rendering fault, so nobody thinks to scroll the tiny inner region. It lasts until the screen is rebuilt. *When:* Every session that opens any two of: character editor, persona editor, dictionary detail, lore detail. One Lore visit is enough.
- **Evidence:**
  - `tldw_chatbook/UI/Screens/personas_screen.py:2030` (code; LY-01): _ensure_center_view sets slot.display = True after first mount; this is the only write - nothing hides a slot again
  - `tldw_chatbook/UI/Screens/personas_screen.py:15926-15932, 643-654, 665-670` (code; LY-01): _show_center toggles only the _CENTER_VIEW_IDS roots, never the four _DEMAND_CENTER_VIEW_SLOTS
  - `tldw_chatbook/UI/Screens/personas_screen.py:1634-1686; Textual 8.2.8 containers.py:124-131` (code; LY-01): Slots are bare Verticals (height: 1fr by default) inside VerticalScroll#personas-detail-stack, so each opened slot keeps an equal fr share
  - `captures/review-rv-layout-2-editor-porthole-after-dict-lore-120x36` (capture; LY-01): Editor is a 2-row window (rows 15-16); Save/Cancel at row 18; rows 21-32 blank
  - ...and 5 more in [Appendix A](#a2-rp-001).
- **Recommendation:** In _show_center, set each _DEMAND_CENTER_VIEW_SLOTS slot's display to match its root, or replace the mutually exclusive children with a ContentSwitcher so exactly one is laid out. Keep the ADR-115 first-use mounting. Add a regression test: mount all four demand views, open the character editor at 120x36 and 160x45, and assert the editor fills the stack (body height >= stack height - 2); same for Lore/Dictionary detail after an editor visit. Fix this BEFORE re-measuring any Dictionaries/Lore layout: several earlier readings (metrics.md section C, capture-report #8 and #13) are inflated by it.
- **Library frame:** Library's work pane is one canvas per route that switches by mode (LibraryPromptWorkPane, Widgets/Library/library_prompt_work_pane.py:13-58), so no hidden siblings compete for height. Adopting the Library frame does NOT fix this if the stack of hidden siblings is carried over.
- **Verifier corrections applied:** Verifier found a single Lore visit is enough (no Dictionaries visit needed). The same leak also splits the dictionary Settings tab and the conversation-transcript view (verifier missed note).

<a id="rp-002"></a>
#### RP-002 · The character editor's long-text fields show 0-1 lines of text, so clicks land mid-word and edits are saved corrupted

**P0 · severity 4 · scope: content · kind: bug · effort S** · sizes: all, including 220x55 (heights are fixed cell counts)  
*Principle:* H1 Visibility of system status; H5 Error prevention; HCI: WYSIWYG direct manipulation  
*Sources:* TF-01, NA-09, LY-09, NB-03 · *Verdict:* CONFIRMED (all four sources)

- **What's wrong:** The boxes for a character's greeting, personality and system prompt show one line of text; Scenario, Post-history instructions and Creator notes show none at all, even when you click into them. A click drops the cursor in the middle of a word you cannot see, and what you type is saved spliced into the old text.
- **Who it hurts, and when:** In-depth authors (job 2) cannot read or check existing text. In a live run two edits were saved corrupted without warning ('The pa Rain hammers the portcullis.rty stands at the gates'). Imported SillyTavern cards rely on Scenario and Post-history, which are invisible. *When:* Every edit of First message, Personality, System prompt, Scenario, Post-history instructions or Creator notes.
- **Evidence:**
  - `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py:98-123` (code; TF-01/NA-09/LY-09): First message, Personality, System prompt height 3 (1 text row); Description 4 (2 rows); Scenario, Post-history, Creator notes height 2 (0 rows - the border uses both)
  - `widget_defaults_self.tcss:2225-2248` (code; TF-01): Bundled CSS mirrors the same heights; no override
  - `rv-flows-2 160x45: Vex > Edit > click First message (90,23) / Scenario border (90,25), type, Save` (live; TF-01): Card afterwards: 'Scenario: ... The pa Rain hammers the portcullis.rty stands...' and 'The gates o The torches gutter.f Emberfall groan open'
  - `captures/review-rv-flows-2-j3-card-after-save-160x45 rows 21-25; review-rv-flows-2-j3-firstmsg-1row-editing-160x45 row 23` (capture; TF-01): Corrupted saves; the 1-row box shows the garbled wrapped tail while editing
  - ...and 2 more in [Appendix A](#a2-rp-002).
- **Recommendation:** Replace the fixed heights with height: auto; min-height: 5; max-height ~40% (at least 3 visible text rows) for First message, Description, Personality, System prompt, Scenario, Post-history and Creator notes; let the focused field grow. When a click lands in empty space, put the cursor at the end of the text. Add an 'Expand' (full-pane) affordance for one field. Add a test that every editor TextArea has a content region >= 2 rows at 120x36, blurred and focused.
- **Library frame:** The Library prompt editor gives each long field a label, a help line and a 4-6-row box that wraps the full text (captures/library-prompts-item-open-160x45 rows 20-38); its Notes body gets 16 rows at 160x45. Adopting the frame alone does not fix this - the editor's own CSS must change.
- **Verifier corrections applied:** Description (height 4) shows 2 rows; Personality and System prompt (height 3) show 1 row each - the problem is wider than the Advanced fields. NB-03's search half is in RP-009.

<a id="rp-003"></a>
#### RP-003 · Dictionary and lore entry forms: most controls are pushed off-screen, and the rest have no labels or show raw internal values

**P0 · severity 4 · scope: content · kind: bug · effort M** · sizes: all, verified at 160x45 and 220x55 (dictionary options unreachable by mouse at every size)  
*Principle:* H6 Recognition rather than recall (persistent labels); H2 Match with the real world; H1 Visibility  
*Sources:* NB-01, NA-16, IA-14, TF-11 · *Verdict:* CONFIRMED (NB-01 reproduced live; NA-16, IA-14 confirmed) + verifier-sourced additions

- **What's wrong:** When you edit a dictionary rule or a lore entry, the first text box takes the full width of the row, which pushes regex, probability, group, priority, position, case and enabled controls off the right edge. What remains has no labels: a box showing '4' or '500' gives no clue it is the scan depth or token budget, and the strategy menu shows internal ids like 'character_lore_first'.
- **Who it hurts, and when:** World-info builders (job 3, one of the four equal jobs) cannot tell the pattern box from the replacement box and cannot see or set most per-entry options without tabbing blind. The User Guide is the only place the controls are named. *When:* Every dictionary or lore edit.
- **Evidence:**
  - `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py:206-241` (code; NB-01): One Horizontal: Pattern Input (default width 100%) then 3 Switches (tooltip-only) and 4 Inputs; replacement TextArea unlabelled
  - `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py:131-173, 199-212` (code; NB-01): Keys Input first, Position/Priority/Enabled clipped; Settings Name/Scan depth ('3')/Token budget ('500') placeholder-only
  - `personas_dictionary_detail.py:29, 265-276; persona_profile_editor_widget.py:143-150` (code; NA-16/IA-14): Strategy Select shows raw 'sorted_evenly', 'character_lore_first', 'global_lore_first'; persona Mode shows 'session_scoped'/'persistent_scoped' (nothing reads it per characters-and-personas.md:268-270)
  - `captures/review-rv-nng-b-dict-entry-form-160x45 rows 26-28; -220x55 rows 30-37` (capture; NB-01): Only two boxes ('car', 'carriage'), full width, no labels
  - ...and 5 more in [Appendix A](#a2-rp-003).
- **Recommendation:** Rebuild both entry forms as vertical labelled forms: a visible label above every input (Pattern, Replacement, Match as regex, Probability %, Group, Max replacements, Priority, Case-sensitive, Enabled; Keys, Secondary keys, Content, Position, Priority, Selective, Case-sensitive, Regex, Enabled) and switches labelled on the same row; same for both Settings tabs (Name, Description, Strategy, Token budget, Scan depth, Recursive scanning, Enabled). Put rarely used options behind 'Advanced'. Give options plain words ('Spread evenly / Character entries first / Global entries first'). Hide persona 'Mode' until something reads it. Add a layout test asserting every form control's region lies inside the pane at 120x36 and 160x45. Pairs with the pane-height fix in RP-006.
- **Library frame:** Every Library editor field has a persistent label plus a one-line help text ('Instructions - Instructions the model always follows.', captures/library-prompts-item-open-160x45 rows 20-35). The frame does not reach inside these forms; copy the labelled-field idiom.
- **Verifier corrections applied:** Verifier notes: the dictionary case is worst - seven non-Pattern options are invisible at every size including 220x55; in lore only the first row overflows (Case/Selective/Regex have labels but their switches are pushed off). Placeholder-as-label also affects Name inputs and the primary Entries row (Probability '100', Max '1', Priority '0'); Description TextAreas have no label at all. Persona Mode and Enabled are covered in RP-086 (gap round), where 'hide Mode' is scoped to local mode.

<a id="rp-073"></a>
#### RP-073 · Three persona-editor sections draw as a title only at every size (tool policy, shared reactions, New Actor Pack's portrait picker), yet Tab still enters them blind and saves what it finds

**P0 · severity 4 · scope: content · kind: bug · effort S** · sizes: 120x36, 160x45 and 220x55 (policy verified at all three; pack portrait at 120x36; same CSS cause everywhere)  
*Principle:* H1 Visibility of system status; H6 Recognition rather than recall; H5 Error prevention (an invisible pre-selected value is committed); WCAG 2.4.7 Focus visible  
*Sources:* G2-02 · *Verdict:* CONFIRMED

- **What's wrong:** Scrolled to the bottom, the persona editor shows the heading 'Tool policy rules (narrowing-only)' and then Save: the rule list, its fields, its warning and its status line are not drawn. The shared-reactions browser and New Actor Pack's 'Required portrait' picker collapse the same way. Keyboard focus still walks into these hidden controls (8 invisible Tab stops at 220x55), so values can be typed and saved without ever being seen, and New Actor Pack pre-selects the first eligible character's portrait in its invisible picker.
- **Who it hurts, and when:** J4 users cannot see or use tool policy, shared reaction art or the portrait choice. A blind 'allow' rule silently un-advertises every other tool of that kind (including spawn_subagent), and the warning that says so is in the hidden block. An Actor Pack persona is silently bound to a portrait the user never chose. *When:* Every persona edit (policy, reactions) and every New Actor Pack.
- **Evidence:**
  - `captures/review-rv-gap2-persona-editor-bottom-policy-clipped-160x45 row 39; review-rv-gap2-2-persona-editor-bottom-policy-clipped-120x36 row 30; review-rv-gap2-3-persona-editor-bottom-policy-clipped-220x55 row 49` (capture; G2-02): Heading 'Tool policy rules (narrowing-only)', then Save/Cancel two rows below; rule list, Kind, Name, Allowed, Require confirmation, Max calls/turn, New/Save/Delete, warning and status are absent at all three sizes
  - `rv-gap2 160x45 and rv-gap2-3 220x55, Tab-by-Tab ANSI diffs` (live; G2-02): 220x55: 22 Tabs from Name to Save, Tabs 14-21 change nothing on screen; 160x45: Tabs 8-15 of 16 are invisible
  - `rv-gap2 160x45: blind keys 'skill' Tab 'web_search' Tab x5 Enter` (live; G2-02): A rule was created and persisted without ever being visible; toast 'Tool policy rules saved.'; Inspector 'Tool policy: 1 rule(s) / skill: web_search → allow' (captures/review-rv-gap2-policy-rule-saved-blind-while-form-unsaved-160x45 rows 19-20)
  - `captures/review-rv-gap2-2-new-actor-pack-portrait-select-hidden-120x36 rows 15-17; review-rv-gap2-2-actor-pack-portrait-dropdown-blind-120x36.png` (capture; G2-02): Label 'Required portrait Character' with no Select drawn; Shift+Tab, Enter opens its overlay with 'Samira “Sammy” Vadem' pre-selected among 3 eligible, plus the blank prompt shown as an option
  - ...and 4 more in [Appendix A](#a2-rp-073).
- **Recommendation:** Now: id-scoped rules giving #personas-editor-pack-portrait, #personas-editor-shared-visual-identity-host and #personas-policy-rules-editor height:auto (inner list min-height 3), without new broad selectors (boot budget 274/274), and a geometry test asserting each section's region height >= its content height at 120x36. Redesign: Tool policy and Reactions become their own work-pane modes; Kind becomes a Select ('MCP tool' / 'Skill') and Name a picker over the live catalog; the deny-by-default warning appears in the persona's Profile/Info summary; the portrait picker comes first in New Actor Pack, with a thumbnail and no pre-selection.
- **Library frame:** Not a frame issue: it is a container/CSS bug inside the pane. The Library's mode strip (Prompts Basic/Advanced/Info, library_prompts_canvas.py:1319-1345) gives each section the whole pane; the proposed persona modes (Tool policy, Look; Appendix C.8) would do the same if each mode's body is sized on its own.
- **Verifier corrections applied:** Only the 120x36 capture shows the pack-portrait clipping; the 160x45 and 220x55 evidence covers the policy section, though the same CSS cause makes all-size clipping near-certain. A persona-editor relative of RP-003 (different cause: 1fr containers collapsed inside a scroll, not a row pushed off the right edge).

<a id="rp-005"></a>
#### RP-005 · Lore and dictionary Settings tabs cannot scroll: 'Save settings', Export and Enabled are unreachable by mouse at the target sizes

**P0 · severity 3 · scope: both · kind: bug · effort S** · sizes: Dictionaries: 120x36, 160x45 (visible at 220x55). Lore: 120x36, 160x45 and 220x55  
*Principle:* H1 Visibility of system status; H3 User control and freedom  
*Sources:* NA-07, TF-10, IA-08 · *Verdict:* CONFIRMED (reproduced live by three verifiers) + verifier-sourced severity notes

- **What's wrong:** The Settings tab of a lore book or dictionary is clipped and the mouse wheel does nothing, so its Save button, Export buttons and Enabled switch are never on screen. Creating a new dictionary drops you here to rename it - and the rename can only be saved by pressing Tab five times blind and then Enter.
- **Who it hurts, and when:** Job 3 cannot save scan depth, token budget, strategy, recursion or a rename by mouse at the owner's medium sizes; keyboard focus disappears off-screen. The User Guide's own step 2 ('Type a real name and click Save settings') cannot be done. *When:* Every lore/dictionary settings edit, and every New (which lands here to rename).
- **Evidence:**
  - `personas_dictionary_detail.py:269-305; personas_lore_detail.py:201-232` (code; NA-07): Settings TabPanes are plain containers; only Entries is wrapped in VerticalScroll
  - `personas_lore_detail.py:91-94; personas_dictionary_detail.py:138-141; personas_screen.py:1634` (code; NA-07 verification): Detail widgets are height 1fr inside VerticalScroll#personas-detail-stack, so the outer scroll never overflows and the tab content is clipped
  - `captures/review-vf-nng-a-lore-settings-wheel-no-save-160x45; review-rv-nng-a-3-lore-settings-no-scroll-save-hidden-160x45` (live; NA-07): Only Name, Description and a bare '4'; 10 wheel events do nothing; no 'Save settings' on screen
  - `captures/review-rv-nng-a-3-lore-settings-focus-offscreen-160x45; review-rv-nng-a-2-dict-settings-clipped-focus-lost-120x36` (live; NA-07): Tab moves focus to invisible controls
  - ...and 5 more in [Appendix A](#a2-rp-005).
- **Recommendation:** Wrap every detail TabPane body in VerticalScroll. Anchor 'Save settings' and Export in a fixed bar outside the scroll. Move Try-it out of this column (RP-006). Add a test that Save settings lies inside the visible region at 120x36 and 160x45.
- **Library frame:** In the Library the editor is the work pane and gets the full height; actions are anchored (captures/library-prompts-item-open-160x45 row 43). Caveat: the Library notes editor also hides Save at 120x36 (capture-report 6.16), so copy the principle, not that screen.
- **Verifier corrections applied:** NA-07's sizes were wrong: Lore is broken at 220x55 too. TF-10's 'New persists immediately' half is in RP-040; IA-08's scattered-actions half is in RP-020. One verifier argued severity 4; kept at 3 because a blind-keyboard workaround exists, but ranked P0.

<a id="rp-006"></a>
#### RP-006 · Dictionary and lore detail starves the entries: 0-5 entry rows at the target sizes while a permanent Try-it block and blank bands take the room

**P0 · severity 3 · scope: both · kind: usability · effort L** · sizes: 120x36 worst (0 entries visible); 160x45 (3-5 rows); 220x55 wasteful  
*Principle:* H8 Aesthetic and minimalist design; master-detail (list -> editor) ecosystem convention  
*Sources:* NB-06, TF-11, LF-20, LY-15, LF-08 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

- **What's wrong:** The rules and facts you are building are the main content, but you see between zero and five of them. The entry table and its edit form share one small scrolling area, a 'Try it' preview box is always stacked below (even with nothing selected), and blank bands sit above or below the tabs.
- **Who it hurts, and when:** Job 3 means blind scrolling through a 4-5-row keyhole (about 14 wheel notches to add one lore entry), with entry content cut to 25-45 characters. Selecting a row pre-loads its values, so users risk overwriting an entry. *When:* Every dictionary or lore visit.
- **Evidence:**
  - `personas_screen.py:1679-1715, 4490-4498` (code; NB-06/LY-15): Both Try-it widgets are siblings below the detail stack; display set by mode only, so shown with nothing selected
  - `personas_lore_tryit.py:39-43, 75-84; personas_dictionary_tryit.py:76-79` (code; LY-15): max-height 60%; a permanently disabled 'Include recent turns (soon)' switch takes 3 rows
  - `personas_dictionary_tryit.py:114-120` (code; missed:layout#1): Five empty Statics reserve blank rows before any run
  - `personas_lore_detail.py:98-101, 136-174; personas_dictionary_detail.py:142-146, 202-204` (code; LF-20): Entries tab = one VerticalScroll holding a DataTable (min 6 / max 12) followed by the entry form
  - ...and 7 more in [Appendix A](#a2-rp-006).
- **Recommendation:** Inside the work pane: an entries list (key, first line of content, on/off, priority) beside or above a full-height labelled entry editor, with an explicit '+ New entry' that clears the form. Keep books in the Items list (do not make each book a rail row - that does not scale to dozens of imported books). Make Try-it a 'Try it' tab/mode (or a right column when the work pane is >= ~100 cols), hidden until something is selected. Delete the disabled 'Include recent turns (soon)' switch. Fix RP-001 first and re-measure, because part of the blank band is that bug.
- **Library frame:** Library readers keep a list beside a full-height work pane, and an empty pane donates its width (adaptive_reader_state.py:519-528); secondary tools are modes (Basic/Advanced/Info), not permanent stacks. SillyTavern/NovelAI lorebooks use entry list -> entry editor, which maps onto the Library's items -> work split. The frame fixes the width; Try-it placement and the entry list/editor split are content decisions.
- **Verifier corrections applied:** Part of the blank band is RP-001 (a clean first visit to Lore at 160x45 shows 10 entry rows). Independently, the detail TabbedContent is clamped to a few rows with blank space above or below depending on path - fix the pane's height allocation, not only first mount. LF-20's 'books as rail sub-rows' was rejected by its verifier (does not scale). TF-11's off-screen-controls half is in RP-003. Gap round (G3) suggested severity 4 at realistic volume; not adopted: a clean 160x45 Lore visit shows 10 entry rows (the 3-row window is RP-001), and the volume cost is captured by RP-056 (no entry filter, 12-row cap) and RP-014 (row-0 reset). Try it's own clipping at volume is RP-081.

<a id="rp-007"></a>
#### RP-007 · 13 rows of stacked chrome above the panes and 4 below take 38-47% of the screen height at the medium sizes (31% at 220 cols); the Library spends 16-20%

**P0 · severity 3 · scope: frame · kind: usability · effort M** · sizes: all; 47% at 120x36, 38% at 160x45, 31% at 220x55, 58% at 80x24  
*Principle:* H8 Aesthetic and minimalist design (chrome-to-content ratio); PRODUCT.md:31 density  
*Sources:* LY-02, NB-07, LF-01, LY-07, IA-20 · *Verdict:* CONFIRMED / PARTIAL (governance corrections applied)

- **What's wrong:** Before any content you get a 5-row boxed title ('Roleplay / Author the pieces that shape a chat / Ready'), a line naming the mode again, a 'Modes:' chip row, a workbench border, a blank padding row and the pane borders. That stack repeats the same fact - you are in Roleplay > Characters - three times and costs every list and editor 8-9 rows.
- **Who it hurts, and when:** All four jobs lose rows on every visit: at 120x36 the list shows 5 characters and the editor 2.5 fields. The eye's first stop is a fixed tagline instead of content. *When:* Every visit, every mode.
- **Evidence:**
  - `captures/roleplay-character-selected-160x45 rows 1-13, 42-45` (capture; LY-02/LF-01): Nav 3 + header box 5 + purpose 1 + Modes 1 + workbench border 1 + padding 1 + pane border 1; 4 rows below
  - `captures/library-conversations-item-open-160x45 rows 1-5` (capture; LY-02): Library: nav 3 + 'Library | Local' + frame; content starts row 6
  - `personas_screen.py:1547-1586; css/components/_workbench.tcss:22-30` (code; LY-02/LF-01): DestinationHeader (bordered, auto height), Static#personas-purpose and DestinationModeStrip are three stacked regions
  - `css/components/_agentic_terminal.tcss:639-660, 709-714; css/layout/_destination.tcss:20-28` (code; LY-07): Workbench border + padding 1; panes round border + padding; Inspector square border + padding 1 (contradicts DESIGN.md:236 'round borders for panels')
  - ...and 4 more in [Appendix A](#a2-rp-007).
- **Recommendation:** One header row via an inline DestinationHeader (pattern: .lab-header-inline / #console-workbench-header.console-header-inline), e.g. 'Roleplay - Characters (28) - Ready'. Move the four modes into the rail (RP-029). Drop #personas-workbench's padding and outer border with a Roleplay-scoped override (the rule is shared by 10 destinations). One framing layer; consistent round borders. Target <= 6 rows above the first content row and <= 2 below. Use an id-scoped rule rather than a new broad selector (boot budget 274/274). Re-pin tests: test_personas_workbench.py:563, test_destination_shells.py:1306, test_destination_visual_parity_correction.py:997-1007.
- **Library frame:** Library uses one header row ('Library | Local', library_screen.py:15017-15021) and puts destinations in the rail. Adopting the frame fixes this directly. Library gets there by dropping DestinationHeader, which departs from ADR-015:30 and DESIGN.md:107; Console and Lab already ship a one-row inline DestinationHeader that stays inside the rules.
- **Verifier corrections applied:** By design per DESIGN.md:107 and ADR-015:34, so an inline DestinationHeader (or an ADR amendment) is required. The workbench border/padding is a shared rule: override for Roleplay only. LY-07's frame rows overlap LY-02 - counted once. The header's Ready/Blocked badge is split out as RP-067. Test pin is test_personas_workbench.py:563, not :548.

<a id="rp-008"></a>
#### RP-008 · The Characters rail stacks seven one-per-row buttons above the list, so the first character sits on row 23 at every size (5 of 28 visible at 120x36)

**P0 · severity 3 · scope: frame · kind: usability · effort M** · sizes: all, including 220x55; 80x24 shows 1 character  
*Principle:* H8 Aesthetic and minimalist design; Hick's law (five similar verbs); Gestalt similarity (filters styled as actions)  
*Sources:* LY-03, NB-08, LF-03, LY-21 · *Verdict:* CONFIRMED / PARTIAL (scope corrections applied)

- **What's wrong:** New, New Actor Pack, Import, Import Actor Pack, Duplicate, Sort and Tag each take a full row above the character list. The rule that decides whether they fit on one row is all-or-nothing, so they stack even at 220 columns. At 80 columns the labels read 'New / New / Import / Import / Duplic'.
- **Who it hurts, and when:** Finding and picking a card - the first step of 'import cards and chat fast' - gets 10 rows at 120x36 and 19 at 160x45. Two near-identical verb pairs add a decision to every create or import. *When:* Every Characters visit; Personas at <=160; every mode at 120x36.
- **Evidence:**
  - `personas_library_pane.py:200-242, 147-170, 265-311` (code; LY-03): Characters needs 64 cols for one row (49 label chars + 5x3 chrome); rail content is ~24/35/50 at 120/160/220; switch is all-or-nothing
  - `captures/roleplay-character-selected-220x55 rows 14-23` (capture; LY-03): 9 control rows; first item row 23 at 220 cols
  - `captures/roleplay-characters-arrival-80x24 rows 13-21` (capture; NB-08): 'New / New / Import / Import / Duplic / Sort: / Tag:'; one list row
  - `captures/roleplay-character-selected-160x45.png` (capture; LY-03): Bold centred verbs are the heaviest ink in the rail; Sort/Tag look like actions
  - ...and 3 more in [Appendix A](#a2-rp-008).
- **Recommendation:** Move creation and import into rail sections with full nouns (New character, New persona, New lore book, New chat dictionary, Import...), or collapse to one 'New...' and one 'Import...' chooser. Move Duplicate to the item's actions. Put Sort and Tag on one subdued row styled differently from verbs. Replace the all-or-nothing stacking with wrapping onto <= 2 rows. Target <= 3 control rows before the first item and the same header shape in every mode. Keep F-030's 'every action reachable' guarantee. Remove the permanently empty #personas-library-count row.
- **Library frame:** Library puts creation and import in rail sections ('Create', 'Import / Export', one primary 'Import...'). Copy the rail grammar. Do NOT copy the Library's own items-pane toolbars, which spend 8 (Prompts, Conversations) to ~19 (Notes) rows before item 1 - copying them saves only ~1 row.
- **Verifier corrections applied:** Dictionaries and Lore fit on one row at 160x45 but stack at 120x36; Personas stacks at 160 too, so the rail changes shape by mode and width (LY-21). The Library's list block is not the compact model (LF-03 correction). The invisible-search half of LF-03 is in RP-009.

<a id="rp-009"></a>
#### RP-009 · The search box hides what you type once it has focus, and the whole list jumps down a row

**P0 · severity 3 · scope: both · kind: bug · effort S** · sizes: Characters and Personas at every measured width; all four modes at 120x36  
*Principle:* H1 Visibility of system status; stable layout (DESIGN.md:117, :125 - focus never moves layout)  
*Sources:* NA-03, NB-03, LY-10, KA-09, TF-12, LF-03 · *Verdict:* CONFIRMED (reproduced live by several verifiers)

- **What's wrong:** When you click or Ctrl+F into the Characters search box, it collapses to an empty two-line frame. The list filters as you type, but you cannot see, check or correct your query. Focusing it also pushes every row below down by one.
- **Who it hurts, and when:** Finding a card by name, the core of job 1, is done blind. Typos and leftover text are invisible, and with RP-022 a typo reads as an empty library. *When:* Every search (and every F6 into the rail, which lands on the search box).
- **Evidence:**
  - `personas_library_pane.py:157-164; css/components/_forms.tcss:105-111 (trap documented at :113-130)` (code; NA-03/LY-10): Stacked mode forces the search to height:1, border:none; app-level Input:focus re-adds a 2-row border, leaving 0 text rows
  - `captures/roleplay-characters-search-nomatch-nothing-selected-160x45 rows 15-16` (capture; NA-03): '┌──┐ / └──┘' with no text after typing 'zzqx'
  - `captures/roleplay-characters-search-typed-invisible-80x24 rows 8, 13-14` (capture; NA-03): Query invisible; purpose line '· 1'; the result row pushed out of the pane
  - `captures/review-rv-layout-search-focused-typed-160x45` (live; LY-10): Typed 'Wren': empty box; list filtered; first row moved y23 -> y24
  - ...and 3 more in [Appendix A](#a2-rp-009).
- **Recommendation:** Use a fixed 3-row search with a clear button (Library's LibraryRailSearchInput) that never resizes on focus. Remove the height:1/border:none stacked-mode override. Make the rail's F6 target the list's selected row, not the search box. Add a test that the focused search has >= 1 content row and siblings do not move.
- **Library frame:** Library rail search is a stable 3-row box with an 'x' clear and the '/' key (library_rail.py:1099-1126). Adopting it fixes this.
- **Verifier corrections applied:** Affected modes depend on width: Characters and Personas everywhere; Dictionaries/Lore stack (and lose the box) only at 120x36.

<a id="rp-010"></a>
#### RP-010 · The test chat's input and buttons clip away: unusable at 120x36, broken after one reply at 160x45

**P0 · severity 3 · scope: both · kind: bug · effort M** · sizes: 120x36 unusable from the first paint; 160x45 buttons lost after the first reply  
*Principle:* H3 User control and freedom; H1 Visibility; HCI: controls must not depend on content length  
*Sources:* LY-14, TF-04, LF-08 · *Verdict:* CONFIRMED

- **What's wrong:** The test chat opens above the card in a panel capped at 60% of the pane that cannot scroll. Its input and its Test Reply / Reset / Send / Configure buttons come last, so they are cut off first. At 120x36 you see only the input's top edge; after one reply the input vanishes. The alternate-greeting picker paints as a blank grey box.
- **Who it hurts, and when:** The 'does my character work?' loop before Chat now (jobs 1 and 2) needs blind typing and blind Tab presses at a design-target size. Reset, Configure and the hand-off disappear after one message. *When:* Every test chat.
- **Evidence:**
  - `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py:43-50, 63-79, 112-170` (code; LY-14/TF-04): Non-scrolling Vertical, height auto, max-height 60%; transcript max 10; input and button row come after it
  - `personas_screen.py:1613-1620` (code; LY-14): The preview is the first child of the work area, above the detail stack
  - `captures/roleplay-testchat-open-clipped-120x36 rows 14-27` (capture; LY-14): Provider line, greeting, 'Greeting:' + blank rows, only the input's top border; no buttons
  - `captures/review-rv-flows-3-j4-testchat-open-120x36; review-rv-flows-3-j4-testchat-reply-120x36` (live; TF-04): After one reply the input disappears; second message typed blind; reply needed 3 wheel notches
  - ...and 3 more in [Appendix A](#a2-rp-010).
- **Recommendation:** Make the test chat a work-pane mode ('Card | Edit | Test chat', see decision D2) using the full pane height, with the input and buttons docked at the bottom and the transcript the only part that grows and auto-scrolls. Replace the greeting Select with a one-row 'Greeting 1/3 < >' cycler. Stop pinning the 'Try a test chat' toggle as the first row of every character view, including the editor.
- **Library frame:** Library has no test chat; its closest analog (the conversation reader) is a full work pane with actions anchored. A Library-style work-pane mode is the natural home.
- **Verifier corrections applied:** At 120x36 on first paint the input's top border and placeholder row are visible (it can be typed into); the bottom border and the whole button row are clipped. The greeting row paints as a ~3-4-row blank box, not height 1 as the CSS suggests. LF-08's Try-it half is in RP-006.


### P1 (26 findings)

<a id="rp-011"></a>
#### RP-011 · Personas mode says 'who you play', but a persona is the AI's side, and the 'you' half of J4 has no home: Settings sends 'user profiles' back to Roleplay, which has none

**P1 · severity 3 · scope: both · kind: usability · effort S** · sizes: all  
*Principle:* H2 Match between system and the real world; H4 Consistency and standards; H10 Help (circular pointer); information scent  
*Sources:* NA-01, TF-17, IA-01, G2-13 · *Verdict:* CONFIRMED / PARTIAL (copy defect per the existing ruling) + gap-round trace of the 'user profiles' half

- **What's wrong:** The purpose line and chip tooltip say personas are 'who you play'. But a persona's main field is a system prompt, Chat now tells the AI 'Respond as <name>', and Console shows the persona in the assistant slot. You already ruled (ADR-037/149, TASK-617) that a persona is assistant-side, so the screen copy is the defect. The other half of job 4, 'user profiles', has no surface anywhere in Roleplay. Your {{user}} name lives in Settings > Console Behavior (global) and Console > Chat settings > Conversation identity (per chat), while Settings > Roleplay says user profiles are Roleplay's business ('Writes allowed: No - change this in Roleplay instead').
- **Who it hurts, and when:** Job 4 (manage personas) sits on a contradictory label. A user who writes a persona expecting it to describe themselves finds the AI speaking as it. The new rail's persona gloss cannot be written until the copy matches the ruling. Someone who wants characters to call them by name follows the app's own copy in a circle. *When:* Every Personas visit; every persona Chat now and test chat; every user who wants to set or check how characters address them.
- **Evidence:**
  - `personas_screen.py:449-453` (code; NA-01): _MODE_DESCRIPTORS['personas'] = 'Personas - who you play in the chat.' (purpose line and chip tooltip)
  - `personas_screen.py:7378-7382` (code; IA-01): Chat now on a persona sends suggested_prompt 'Respond as {name}.'
  - `backlog/decisions/037-...md:12-16; backlog/decisions/149-console-persona-session-identity.md:10-20` (doc; NA-01): Persona = assistant-side profile; persona name becomes the assistant's speaker label
  - `persona_profile_editor_widget.py:131-134, 159` (code; NA-01): Core field 'System prompt'; 'Persona Visual operational states' (idle, thinking, speaking...)
  - ...and 9 more in [Appendix A](#a2-rp-011).
- **Recommendation:** Change the descriptor to e.g. 'Personas - reusable AI roles' (or 'assistant voices the AI speaks as'); gloss the rail row the same way; add a line to the persona card: 'Chat now: the AI answers as <name>'. Fix the nav purpose/tooltip (shell_destinations.py:84-86), which still says 'user profiles'. Close TASK-19577's persona criterion as superseded and annotate the north-star spec line. Label persona test-chat replies with the persona's name (RP-045). Gap round: promote D3 to B + C. Add a 'You' rail row ('your name in chats: <effective name> · set in Settings > Console Behavior'); Change… deep-links via NavigateToScreen('settings', {'category': CONSOLE_BEHAVIOR}) as personas_preview_controller.py:390-395 already does for providers; add a line on per-chat names. Use the same name in the test chat and transcript (RP-045, RP-051). Rewrite the Settings > Roleplay contract copy (settings_screen.py:1665-1683) to drop 'user profiles' (TASK-496). Keep ADR-037/046: the row shows and links the display name; it is never a persona and never a user-profile editor.
- **Library frame:** Not a frame problem. A Library-style rail row carries a permanent gloss, which would repeat whatever wording ships - so the copy must be right first.
- **Verifier corrections applied:** Verifiers: the F-034 wording was chosen deliberately and TASK-19577 endorses it, but the owner's ruling (TASK-617:19; ADR-037, ADR-149) is assistant-side, so this is a copy defect, not an open question. Nav copy in shell_destinations.py is part of the same defect (missed ia#4), not independent evidence. Gap round (G2-13) merged here and tied to D3 rather than filed as a separate P1: its new value is the circular Settings > Roleplay copy and the missing display-name route. The owner's 'user profiles' may itself reflect the persona-means-user confusion this finding describes. Settings > My Profile (ADR-102) is a separate domain from {{user}}.

<a id="rp-012"></a>
#### RP-012 · Character attachments are silent frozen copies, and the lore/dictionary side says 'Not attached' even when a character holds one

**P1 · severity 3 · scope: both · kind: usability · effort M** · sizes: all  
*Principle:* H1 Visibility of system status; H6 Recognition rather than recall; PRODUCT.md:25 consequences without guesswork  
*Sources:* NA-02, TF-02, IA-05, NB-04, TF-09 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

- **What's wrong:** Attaching a lore book or dictionary to a character stores a copy. Later edits to the original never reach the character, and the only warning is a hover tooltip. From the other side, the lore book's 'Attachments' tab lists conversations only, so it reads 'Not attached to any conversation yet.' while a character carries a copy of it.
- **Who it hurts, and when:** World-builders (job 3) edit a lore book expecting attached characters to change, and cannot answer 'who uses this?' from where they are working. The word 'Attach' means 'copy' on a character but 'link' on a conversation. *When:* Every world-info edit after an attach; every 'who uses this?' question.
- **Evidence:**
  - `tldw_chatbook/Character_Chat/world_book_manager.py:823-862` (code; NA-02/TF-02): attach_world_book_to_character embeds an export snapshot; idempotent by name, so re-attach never refreshes
  - `Character_Chat/local_chat_dictionary_service.py:1134-1168, 1206-1232` (code; NA-02 verification): Dictionaries attached to characters are snapshots too
  - `personas_screen.py:6273-6290` (code; NA-02): The attach picker hides names already attached, so it cannot refresh a stale copy
  - `personas_character_world_books.py:86-93; personas_character_dictionaries.py:86-94` (code; IA-05): Only disclosure is a hover tooltip; task-2231 folded away the visible '(copied into this character)' label the guide still promises (lore-books.md:128-131)
  - ...and 5 more in [Appendix A](#a2-rp-012).
- **Recommendation:** First rule on D6 (snapshot vs live link, and whether character attachments apply in Console - RP-004). Then: rename the tab 'Used by' and list characters ('Captain Isolde Varga - copy from <date> - 6 entries (source now 7) - Update copy - Detach') and conversations ('linked'); show a one-line summary near the character header ('Lore: Station Kepler-9 Ops Manual - Dictionaries: none - Manage...'); replace the tooltip with visible text ('Copies - editing the original doesn't update them'); name the verbs 'Copy into character...' vs 'Link to chat...'; add usage to list rows ('in 1 character, 0 chats'). Cover dictionaries the same way.
- **Library frame:** Library has no snapshot concept, but its Info work-pane mode is the right home for a 'Used by' list. The frame provides the slot; the data and copy must be built.
- **Verifier corrections applied:** Snapshot semantics are deliberate and documented (P2f spec :9, :28; lore-books.md:128-133) - a harmful design decision, not a bug. Dictionaries attached to characters are snapshots too ('live vs copy' contrast dropped). Deleting the source does not break characters (they hold copies). 'Chats keep injecting the old copy' was false: see RP-004 - character copies appear not to be applied at all.

<a id="rp-013"></a>
#### RP-013 · A lorebook embedded in an imported card is invisible in Lore mode and cannot be viewed, edited or tested

**P1 · severity 3 · scope: content · kind: gap · effort M** · sizes: all  
*Principle:* H6 Recognition rather than recall; H4 Consistency (one kind of object, one home)  
*Sources:* TF-03 · *Verdict:* CONFIRMED

- **What's wrong:** Importing a card that carries its own lorebook shows a toast 'Lorebook ... attached (2 entries)', but Lore mode then says 'No lore books yet'. The lore exists only as a read-only row at the very bottom of the character card, with no way to read its entries, test it or edit it.
- **Who it hurts, and when:** After 'import cards & chat fast', users cannot inspect or tune the lore that came with the card. The toast promises something the Lore mode denies exists. *When:* Every card import with an embedded character_book (common for community cards).
- **Evidence:**
  - `tldw_chatbook/Character_Chat/Character_Chat_Lib.py:1788-1822` (code; TF-03): Embedded character_book is converted into a snapshot on the character; no standalone world book is created
  - `rv-flows 160x45 empty profile: Import ilsa_with_book.v2.json, then l` (live; TF-03): Toast 'Lorebook 'Glass Coast Lore' attached (2 entries)'; Lore shows 'No lore books yet' and '· 0' (captures/review-rv-flows-j2-lore-mode-after-import-160x45 rows 9-21)
  - `captures/review-rv-flows-j2-worldbooks-expanded-160x45 rows 38-41` (capture; TF-03): Only a read-only 'Glass Coast Lore 2' row below Edit/Conversations
- **Recommendation:** List embedded books in Lore mode (a 'From cards' group or rows tagged 'embedded in Ilsa'); offer 'Open as editable lore book' (promote to standalone and re-link); make the import toast link to the book; show it in the character's World/Used-by summary (RP-012).
- **Library frame:** Library imports land items in their destination list with updated counts. A Lore rail row with a count helps only if embedded books are listed there.

<a id="rp-015"></a>
#### RP-015 · An invalid regex is saved silently; its warning list renders with zero rows while the Inspector says 'Validation: OK'

**P1 · severity 3 · scope: content · kind: bug · effort S** · sizes: all  
*Principle:* H9 Help users recognize, diagnose, and recover from errors  
*Sources:* NB-02 · *Verdict:* CONFIRMED (reproduced live)

- **What's wrong:** If you type a broken regular expression and press Add, the rule is saved and quietly treated as plain text that will never match. The warning exists in code but is drawn in a box with no visible rows, and the Inspector still says 'Validation: OK'. The only trace is in a log file.
- **Who it hurts, and when:** Users believe a rule works when it never will (job 3), and the 'OK' actively misleads. *When:* Whenever a regex is mistyped.
- **Evidence:**
  - `rv-nng-b 160x45: Fantasy Anachronism Swaps > pattern '(unclosed' > regex on > Add` (live; NB-02): No message; rail '8 entries' -> '9 entries'; app log: 'Invalid regex ... Treating as literal string'
  - `captures/review-rv-nng-b-dict-validation-panel-220x55.png` (capture; NB-02): Advisory list is a 2-row bordered strip with zero content rows; Inspector 'Validation: OK'
  - `personas_dictionary_detail.py:159-162, 268, 524-555, 610-680` (code; NB-02): OptionList height auto max 5 with border; options rendered '[code] pattern - message'; no compile check in form_payload
  - `personas_inspector_pane.py:527-533; personas_screen.py:5337, 5428` (code; NB-02): The Inspector summary is never fed dictionary findings
- **Recommendation:** Compile regex patterns in form_payload() and show an inline warning under Pattern ('This regex doesn't compile: missing ) - fix it or turn off "Match as regex"'). Give the advisory list >= 3 content rows and drop the '[invalid_regex]' code prefix. Feed findings to the summary ('Validation: 1 warning - jump'). After Add, select and scroll to the new row. Whether to block the save is decision D9 (today's design is warn-not-block).
- **Library frame:** Library has no rule editor; its Notes put errors next to the work (inline conflict region). Copy that.
- **Verifier corrections applied:** Saving without blocking is deliberate (personas_dictionary_validation.py:1 'Warn-not-block validation (Roleplay P1c)'); the defect is that the warning is invisible and the Inspector says OK.

<a id="rp-016"></a>
#### RP-016 · The character editor is one long scroll of small fixed-height fields with no sections, so 2.5 fields fit at 120x36 and a wider canvas would not help

**P1 · severity 3 · scope: content · kind: usability · effort M** · sizes: 120x36 worst (2.5 fields), 160x45 (5 fields); still fixed heights at 220x55  
*Principle:* Progressive disclosure; H8 Aesthetic and minimalist design; form scanning  
*Sources:* LY-11, LF-06, missed:nng-a#3 · *Verdict:* CONFIRMED + verifier-sourced evidence

- **What's wrong:** All editor fields (basics, prompts, greetings, voice, avatar, expressions, visual identity) sit in one scrolling column; only 'Advanced' folds away. Each field has a fixed small height, so a 1,200-character description shows 2 lines (about 11%). With the character's World Books section open below Save, the editable area shrinks to about 8 rows.
- **Who it hurts, and when:** Writing and revising long descriptions and system prompts - the heart of job 2 - means constant scrolling while space elsewhere goes unused. *When:* Every edit session.
- **Evidence:**
  - `personas_character_editor_widget.py:98-112, 400-711` (code; LY-11/LF-06): Fixed heights (3/4/3/3); one VerticalScroll; only Advanced collapses
  - `captures/roleplay-character-editor-160x45 rows 16-37; roleplay-character-editor-120x36 rows 16-29` (capture; LY-11): 5 fields at 160x45; 2.5 at 120x36
  - `captures/roleplay-character-editor-scrolled-160x45.png` (capture; LY-11): ~7 empty rows below 'Avatar: none' while text fields stay one line tall
  - `captures/review-rv-nng-a-editor-unsaved-160x45 rows 15-25` (capture; missed:nng-a#3): With World Books expanded below Save/Cancel, only Name and a 1-row First message are visible
  - ...and 1 more in [Appendix A](#a2-rp-016).
- **Recommendation:** Split the editor into work-pane modes (e.g. Basics | Prompting | Greetings | Voice & look | Info; persona: Profile | Visuals | Policy). In a mode, give the focused long field height 1fr (min 3 rows). Keep Save/Cancel anchored outside the scroll (already good). Shrink the Generate-whole-character / concept rows to one compact row. Move character attachments out of the editor column.
- **Library frame:** Library rule: the primary content box takes the pane's remaining height with its actions beneath (_library.tcss:1293-1312); editors are split into work-pane modes (Prompts Basic/Advanced/Info; Notes Edit/Preview/Info). Adopt both. Do not copy the Notes body, which stays at 4 rows at 120x36.

<a id="rp-017"></a>
#### RP-017 · Width goes to the side panes, not to what is open: a fixed 2:4:2 split, no collapse priority, no 'work-first' while editing, and blank panes at large sizes

**P1 · severity 3 · scope: both · kind: usability · effort L** · sizes: 120x36-160x45 (editor ~half the width); 200+ (blank Inspector, 100-185-char lines)  
*Principle:* HCI: allocate space to the primary task; H7 Flexibility and efficiency; readable line length  
*Sources:* LF-04, LF-05, LY-06, LY-17 · *Verdict:* CONFIRMED / PARTIAL (LY-06 rated 2 on its own)

- **What's wrong:** The list, centre and Inspector always share the width 2:4:2, so both side panes grow with the window. Opening the editor never reclaims space. At 120 columns the editor gets 58 columns next to 58 columns of side panes; at 220 columns the Inspector is 54 columns wide and 85% empty, and description lines run about 100 characters (185 with rails collapsed).
- **Who it hurts, and when:** Editors, entry tables and the test chat get about half the width at the medium sizes; the extra width at large sizes is wasted on empty rails. *When:* Constant.
- **Evidence:**
  - `css/components/_agentic_terminal.tcss:709-734` (code; LF-04/LY-06): Inspector 2fr (min 30), Library rail 2fr (min 24), work 4fr (min 40); no max-width
  - `personas_screen.py:2316-2393` (code; LF-04): Only compact (<=90) and narrow (<=60) breakpoints; all three panes open from 91 cols
  - `personas_screen.py:15909-15975` (code; LF-05): _show_center never changes pane allocation when an editor opens
  - `captures/roleplay-character-editor-120x36.png` (capture; LF-05): Editor 58 cols beside a 28-col list and a 30-col Inspector with Export/Delete/Buddy
  - ...and 5 more in [Appendix A](#a2-rp-017).
- **Recommendation:** Reuse the pure resolver (decision D4) with Roleplay profiles: card/list work_min 56 with list 32-50; editor work_min 72; entries work_min 64; bounded rail 29-39 cols, never fr-scaled. While an editor is open, apply a transient work-first override (close the nav rail, and the items list below ~170 cols; a manual reopen wins; never persisted; display-only, so ADR-115 bodies and ADR-046 drafts survive). With nothing selected, let the list widen. Cap prose at ~90 cells. At >= 200 cols, spend spare width on a second work column (editor | test chat). Pin widths at 120/160/200 in a pure test.
- **Library frame:** The Library's pure layout resolver closes side panes in a fixed order, keeps the work pane above a minimum, and lets an empty work pane donate width to the list; Library Notes adds a temporary 'work-first' override during editing. BUT the Library's default profiles give the work pane only 50/45/62 cols at 120/140/160 - narrower than Roleplay today. Copy the mechanism with Roleplay-specific profiles, not the Library defaults.
- **Verifier corrections applied:** At 120 cols the authoring profile also closes the items list (so '106 cols for the editor' means the list is closed too). LY-06 alone is an aesthetic cost (severity 2); merged here because the root cause is the same fr split.

<a id="rp-018"></a>
#### RP-018 · The Inspector column mixes per-item actions, app-wide Buddy controls and wrong-mode advice in ten equal one-row buttons

**P1 · severity 3 · scope: both · kind: usability · effort L** · sizes: all; width cost at 120-160, blank column at 200+  
*Principle:* H8 Aesthetic and minimalist design; Hick's law; Gestalt proximity; visibility of the primary action  
*Sources:* LF-07, NB-10, IA-10, LY-12, KA-14, NA-22 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

- **What's wrong:** The right column stacks Chat now, Send to Console draft, three Export buttons, Delete and four 'Buddy' buttons, all the same size and weight. The Buddy buttons control an app-wide companion, not the selected item, yet they appear in every mode - even under a lore book or with nothing selected. In Dictionaries it says 'Pick a character or persona to start chatting'. Disabled buttons explain themselves only in hover tooltips that the keyboard cannot reach.
- **Who it hurts, and when:** The real next step (Chat now) is diluted by 12 other equal targets; a third of the column is unrelated to the selection, and the column takes 30-39 columns from the work pane at medium sizes. *When:* Every selection.
- **Evidence:**
  - `personas_inspector_pane.py:212-363` (code; LF-07): Selected/Type/avatar, Validation, conversations, Readiness, Chat now/Send/Export x3/Delete, 4 Buddy buttons
  - `personas_inspector_pane.py:1145-1159` (code; NB-10/IA-10): All four Buddy buttons forced display=True in every state
  - `personas_inspector_pane.py:35, 1005-1011` (code; NB-10): 'Pick a character or persona to start chatting.' in every mode; dictionary/lore get 'Console chat is for characters and personas.'
  - `personas_inspector_pane.py:123-134` (code; LY-12): Every button width 100%, height 1, border none, no gaps
  - ...and 5 more in [Appendix A](#a2-rp-018).
- **Recommendation:** Per decision D1: fold actions into the work pane header (Chat now primary plus a readiness word; 'More actions' for Send to Console draft, Export..., Duplicate, Delete); validation and 'what is in play' into an Info mode; the character's chats into a Chats mode; Buddy into its own rail row. Until then: hide Buddy actions that do not apply, make guidance mode-aware, and show disabled reasons as inline text. Keep widget ids (#personas-start-chat, #personas-attach-to-console, #personas-conversations-list).
- **Library frame:** Library has no right inspector. Per-item verbs live in the work pane (primary action plus a 'More actions' menu that shows only valid verbs), metadata in an Info mode, destination-level things in rail sections. Adopting that removes the column; Roleplay additionally needs homes for readiness, the ADR-120 chat browse and Buddy.
- **Verifier corrections applied:** The Buddy controls are deliberately global (independent Buddy, ADR-139; commits 231a8533ea/45fced8834) - noise in the item Inspector, not a bug. DESIGN.md:363 accepts tooltips as a channel, but DESIGN.md:287-288 wants a text annotation and keyboard users never see tooltips on disabled buttons. The tonal disabled style (_buttons.tcss:48-60) is deliberate. NA-22's toolbar-verb half is in RP-043.

<a id="rp-019"></a>
#### RP-019 · At 120x36, 'Chat now' falls below the Inspector's fold whenever the character has a portrait, with no 'more below' cue

**P1 · severity 3 · scope: both · kind: usability · effort S** · sizes: 120x36 (hidden); 160x45 (Buddy hidden); <=100x30 hidden for every character  
*Principle:* H1 Visibility; F-pattern (primary action in the least-scanned spot); Fitts's law  
*Sources:* LY-13 · *Verdict:* CONFIRMED (reproduced live)

- **What's wrong:** A 10-row portrait box sits above the Inspector's actions, so for an imported PNG card the Chat now button is below the visible area at 120x36. A single scrollbar glyph is the only hint; the design system's required 'more below' row is missing.
- **Who it hurts, and when:** The 'import cards & chat fast' job hides its own final step at a target size, exactly for the cards most likely to be imported. *When:* Every character with a portrait (typical for imported PNG cards).
- **Evidence:**
  - `personas_inspector_pane.py:78-87, 212-300` (code; LY-13): Avatar box (max-height 10), validation, conversations, readiness, then actions, in one VerticalScroll
  - `captures/review-rv-layout-2-portrait-cta-below-fold-120x36` (live; LY-13): Portrait rows 18-28, Validation 29, Conversations 30-32; Readiness and Chat now below; one '▂' glyph as the only hint
  - `captures/review-rv-layout-portrait-inspector-160x45` (live; LY-13): Chat now row 35; Buddy below the fold
  - `DESIGN.md:301-304` (doc; LY-13): Scrolling inspectors must reserve a '▼ more' fold row
- **Recommendation:** Move Chat now and Send to Console draft into the work-pane header (RP-018), or pin the Inspector's action block above its scroll body. Cap the portrait at ~6 rows on terminals <= 36 rows and collapse it when it cannot render (TASK-1532). Add the DESIGN.md:301-304 fold row.
- **Library frame:** Library puts an item's primary action at the top of the work pane ('Use in Console', 'Resume conversation') and shows '▾ scroll for more' in rails.
- **Verifier corrections applied:** At 160x45 Delete is visible (row 40); only the Buddy buttons are hidden.

<a id="rp-020"></a>
#### RP-020 · Per-item actions live in three different places and move by mode; dictionary and lore Export sit at the bottom of an unscrollable tab

**P1 · severity 3 · scope: both · kind: usability · effort M** · sizes: all; dictionary/lore Export unreachable at <=160x45  
*Principle:* H4 Consistency and standards; H6 Recognition (where is the action?)  
*Sources:* IA-08, LF-17 · *Verdict:* CONFIRMED (reproduced live)

- **What's wrong:** For one character, Duplicate is in the left rail toolbar, Chat now / Export / Delete are in the right column, and Edit is at the bottom of the card. For a dictionary or lore book, Export is at the bottom of the Settings tab - which cannot scroll (RP-005). For a persona, Edit can be below the fold although the card is mostly empty.
- **Who it hurts, and when:** 'How do I export, duplicate or delete this?' has a different answer in each mode, and for world info the answer is often 'you can't see it'. *When:* Every export, duplicate, edit or delete.
- **Evidence:**
  - `personas_library_pane.py:291-296` (code; IA-08): Duplicate (an item action) in the rail's collection toolbar
  - `personas_inspector_pane.py:51-52; personas_dictionary_detail.py:288-300; personas_lore_detail.py:222-226` (code; IA-08): Inspector export only for characters/personas; dictionary/lore export at the bottom of Settings
  - `personas_character_card_widget.py:124-139; personas_screen.py:7077-7090` (code; LF-17): Edit and 'Conversations (N)' at the card bottom
  - `captures/roleplay-character-selected-160x45 rows 20, 27-32, 41` (capture; IA-08): Three regions for one character's actions
  - ...and 2 more in [Appendix A](#a2-rp-020).
- **Recommendation:** One action bar per kind in the work pane: primary (Chat now for characters/personas; Run preview for lore/dictionaries), then Edit/Save, then 'More actions': Add to Console message, Export (JSON / PNG card / Markdown / Actor Pack), Duplicate, Delete (danger group). The rail keeps only collection verbs (New, Import). Remove the card-bottom Edit and Conversations buttons.
- **Library frame:** Library puts per-item verbs in the work pane: a header action ('Use in Console', hidden while dirty) and a state-machine action bar ('More actions' when clean -> Save changes / Discard when dirty -> Save / Cancel when new). Copy it. Do not copy the Library's inconsistent delete confirmation (inline vs modal).
- **Verifier corrections applied:** IA-08 verifier: 'Save settings' is also clipped (moved to RP-005). Library's 'Use in Console' sits mid-pane and 'More actions' at the pane bottom, so 'one bar at the top' is a design choice, not Library precedent.

<a id="rp-021"></a>
#### RP-021 · Returning to Roleplay restores a broken state: an empty list and '· 0' when nothing was selected, or a hidden leftover search filter

**P1 · severity 3 · scope: both · kind: bug · effort S** · sizes: all  
*Principle:* H1 Visibility of system status; H9 Help users recover (it looks like data loss)  
*Sources:* TF-07, TF-12 · *Verdict:* CONFIRMED (both halves reproduced live by the verifier)

- **What's wrong:** If you leave Roleplay while in Personas, Dictionaries or Lore with nothing selected, coming back shows an empty list and a count of 0, as if your data were gone; clicking the active mode does nothing. If you leave after searching, the list comes back filtered to the old query but the search box looks empty, so 27 of 28 characters seem to have vanished.
- **Who it hurts, and when:** Personas, lore books or dictionaries appear deleted after a routine trip to Console or Library. Recovery means switching to another mode and back, which nobody would guess. *When:* Every leave/return while a non-Characters mode has no selection (e.g. right after switching modes); every return after searching.
- **Evidence:**
  - `personas_screen.py:1808-1817, 1819-1823` (code; TF-07): No selected id -> _pending_restore = None -> _apply_pending_restore returns before _apply_mode(saved_mode)
  - `personas_screen.py:1797-1806, 4476, 13935` (code; TF-12): state.search_query is restored, but the search Input value is only ever cleared
  - `captures/review-rv-flows-j8-return-personas-count0-220x55.png; review-rv-flows-j8-return-lore-empty-220x55 rows 9-26` (live; TF-07): Empty list, '· 0', Characters-only test-chat toggle in Lore; active chip dead
  - `captures/review-vf-flows-tf07-return-personas-empty-160x45` (live; TF-07 verification): Personas '· 4' -> Library -> Roleplay: '· 0' and an empty list
  - ...and 1 more in [Appendix A](#a2-rp-021).
- **Recommendation:** In restore, always run _apply_mode(saved_mode) (or the rail-section equivalent) whether or not a selection was saved, then re-select if one was. Restore the search Input's value from state.search_query (or clear both together). Make a click on the already-active mode re-render its list. TASK-488, marked 'possibly stale' in prior-art, is still live for the no-selection case.
- **Library frame:** Library re-renders the remembered destination's list on return. A rail-section model that always re-applies the active section on mount fixes the first half, provided restore does not depend on having a selection.

<a id="rp-022"></a>
#### RP-022 · A search with no matches says 'No characters yet - use New or Import', and filtered counts never say 'of 28'

**P1 · severity 3 · scope: both · kind: bug · effort S** · sizes: all  
*Principle:* H9 Help users recognize and recover from errors; H5 Error prevention; H1  
*Sources:* NA-04, NB-16, IA-07, LF-14, TF-12, G3-11 · *Verdict:* CONFIRMED

- **What's wrong:** Type a name that matches nothing and the list says your library is empty and tells you to create or import. The count line says '0', not '0 of 28'. Combined with the invisible search text (RP-009), a typo looks exactly like data loss.
- **Who it hurts, and when:** Users conclude their characters are gone, or re-import cards and create duplicates. *When:* Every failed search.
- **Evidence:**
  - `personas_library_pane.py:431-442` (code; NA-04/NB-16): Empty branch always renders 'No {noun} yet - {hint} to add one.' and ignores `filtered`
  - `personas_screen.py:4329-4335, 4413 vs 3690, 4021` (code; LF-14): Dictionaries/Lore pass filtered; Characters/Personas never do
  - `captures/roleplay-characters-search-nomatch-nothing-selected-160x45 rows 9, 23-24` (capture; NA-04): 28 characters exist: '· 0' and 'No characters yet - use New or I...'
  - `captures/roleplay-characters-search-nomatch-120x36 rows 9, 22-24, 29` (capture; NB-16): Three conflicting messages: '· 0', 'No characters yet', 'Pick a character from the list'
  - ...and 5 more in [Appendix A](#a2-rp-022).
- **Recommendation:** When filtered and empty: 'No characters match "zzqx" (28 total) - Esc or [Clear search]'. Show 'No characters yet - New or Import' only when the unfiltered total is 0. Pass `filtered` from the Characters and Personas renders too, so the 'N of M' count appears. Hide the 'Pick a character' centre guidance while the filter has zero results. Same rule in all four modes. At volume: page bar '1-50 of 85 matches'; tag filter '20 of 348 · tag fantasy ×'.
- **Library frame:** Library keeps totals in the rail ('Conversations (6)'), shows '1-4 of 4' under lists, and uses filter-aware copy ('No prompts match your filter.'). The copy branch still needs a content fix.
- **Verifier corrections applied:** Severity 3 holds mainly because the query is also invisible (RP-009); on its own this would be 2. The 'N of M' count already exists for Dictionaries/Lore (filtered=bool(needle)); Characters/Personas never pass `filtered` (LF-14 correction). The '· 0 while loading' half is RP-028. Gap round (G3-11): the same count defect at 348 cards, now also in the page-bar label and for tag filters; folded in as volume evidence.

<a id="rp-023"></a>
#### RP-023 · Example dialogue (mes_example) is stored and sent to the model, but cannot be seen or edited anywhere

**P1 · severity 3 · scope: content · kind: gap · effort S** · sizes: all  
*Principle:* H2 Match with the real world (card-format vocabulary); completeness of the authoring model  
*Sources:* TF-15 · *Verdict:* CONFIRMED

- **What's wrong:** Character cards in the common V2 format carry example dialogue that teaches the model how the character talks. Roleplay imports it and sends it to the model, but neither the card view nor the editor shows it.
- **Who it hurts, and when:** Authors cannot see or tune one of the strongest voice-shaping fields (job 2); imported behaviour is opaque (job 1). *When:* Every imported community card (most carry example dialogue); every in-depth authoring session.
- **Evidence:**
  - `personas_character_editor_widget.py; personas_character_card_widget.py; personas_screen.py:8103; personas_preview_controller.py:96` (code; TF-15): grep message_example/mes_example: only the duplicate path and the preview system prompt; no card row or editor field anywhere in the app
  - `rv-flows 160x45 empty profile: Import ser_corwin.v2.json` (live; TF-15): Card shows Description, Personality, Scenario, First message, Creator, Version, Tags, Alternate greetings - no example dialogue (captures/review-rv-flows-j2-after-json-import-160x45 rows 15-35)
- **Recommendation:** Add a labelled, auto-height 'Example dialogue' field (help text: 'Shows the model how {{char}} talks; use <START> to separate examples') to the editor's Prompting/Advanced section and a collapsed row on the card.
- **Library frame:** Library Prompts shows every field across Basic/Advanced/Info; no field is stored but invisible.

<a id="rp-024"></a>
#### RP-024 · Keyboard focus escapes the screen: arrival focus sits on the nav bar's '⌃1 Home' (Enter leaves Roleplay), and Tab walks 14 nav buttons on every cycle

**P1 · severity 3 · scope: frame · kind: bug · effort S** · sizes: all  
*Principle:* H3 User control and freedom; H4 Consistency; WCAG 2.4.3 Focus Order  
*Sources:* KA-01, KA-02, missed:a11y#4 · *Verdict:* CONFIRMED (both reproduced live)

- **What's wrong:** When Roleplay opens, keyboard focus is actually on the 'Home' tab in the top navigation bar, not in the list - so pressing Enter takes you to Home. Pressing Tab repeatedly leaves the screen and walks through all 14 navigation buttons before returning, and the nav strip then stays scrolled, hiding Home and Console.
- **Who it hurts, and when:** Keyboard users (the TUI's main audience, DESIGN.md:117) start every visit outside the screen; a stray Enter while hunting for Chat now or Save navigates away from the work. *When:* Every visit (arrival); every Tab cycle.
- **Evidence:**
  - `personas_screen.py:1083-1084` (code; KA-01): No tab/shift+tab override, so Textual's Screen default walks the whole app focus chain, nav bar included
  - `library_screen.py:1010-1021, 8975-9008` (code; KA-01): Library re-binds tab/shift+tab and confines focus to '#screen-content, #screen-content *'
  - `rp-review/a11y/steps/rv-a11y-tab2/4..20.ansi` (live; KA-01): After 'Show Buddy', Tab #5 focused '⌃1 Home', then 14 nav stops; Library: 45 Tabs, 0 nav stops
  - `personas_screen.py:1878-1924; UI/Navigation/main_navigation.py:745-760` (code; KA-02): _auto_select_first_library_row never moves focus; App AUTO_FOCUS='*' lands on the first nav button
  - ...and 2 more in [Appendix A](#a2-rp-024).
- **Recommendation:** Promote Library's Tab-region rule into the shared destination frame (e.g. a BaseAppScreen flag TAB_REGION='#screen-content') and enable it for Roleplay. On mount/resume, focus the list's selected row (or the last-focused widget from save_state); set AUTO_FOCUS so the nav bar is never the default. Test that app.focused is inside #screen-content after arrival. File the scrolled nav-strip fragment ('‹ le') against main_navigation.py.
- **Library frame:** Library confines Tab to '#screen-content' (task-32052) - adopt it, lifted into the shared frame. Library has the same arrival-focus defect - do not copy that; fix it once in the frame.

<a id="rp-025"></a>
#### RP-025 · F6 (next pane) lands on secondary controls and never reaches the dictionary or lore editors

**P1 · severity 3 · scope: both · kind: bug · effort M** · sizes: all  
*Principle:* H7 Flexibility and efficiency; WCAG 2.1.1 Keyboard; DESIGN.md:117 keyboard-first  
*Sources:* KA-03 · *Verdict:* CONFIRMED

- **What's wrong:** F6 is the advertised way to jump between panes. In Dictionaries and Lore it only alternates between the search box and the Inspector's collapse button - it never reaches the entries table, the entry form or Try it. In Characters it lands on 'Try a test chat' instead of the card or Edit, and in the Inspector on an empty list or on the collapse button, never on Chat now.
- **Who it hurts, and when:** Keyboard users cannot reach two of the four core editors (lore books, dictionaries) with the pane key; a reflexive Enter on the Inspector's F6 target collapses the Inspector. The User Guide promises F6 reaches the centre. *When:* Every F6 press.
- **Evidence:**
  - `personas_screen.py:1122-1155, 4490, 1613; Widgets/workbench_focus.py:66-79` (code; KA-03): Work-area targets are all inside the preview pane, which is hidden in Dictionaries/Lore; the non-focusable work area is then dropped from the cycle
  - `rp-review/a11y/steps/rv-a11y-dictf6 0-4` (live; KA-03): F6 alternates search <-> Inspector ' > ' only
  - `rv-a11y f6-step2, persona-f6, tab3` (live; KA-03): Work F6 -> '▸ Try a test chat'; Inspector F6 -> empty conversations list or collapse button; opening the editor took F6 + 5 Tabs (2 invisible)
  - `Docs/User_Guide/roleplay-chat-dictionaries.md:59-75` (doc; KA-03): The guide says F6 cycles through the centre
- **Recommendation:** Replace the static target tuple with per-mode targets: rail -> the list's selected row; work -> the editor's first field when editing, Edit on a card, the Entries table for lore/dictionaries; Try it -> its own stop; Inspector (or its successor) -> the first enabled primary action, never the collapse button; collapsed panes -> their grip. Extend Tests/UI/test_workbench_pane_focus.py to all four modes.
- **Library frame:** Library Artifacts computes F6 targets per state and substitutes a pane's grip when it is closed (library_artifacts_reader_shell.py:43-72). Copy that method; Library's rail target (the search box) is not the improvement.

<a id="rp-026"></a>
#### RP-026 · Escape does not leave text fields, so the next advertised mode key is typed into the field

**P1 · severity 3 · scope: both · kind: gap · effort S** · sizes: all  
*Principle:* H3 User control and freedom (clearly marked exit); H4 Consistency with Library; WCAG 2.1.2 (soft trap)  
*Sources:* KA-04, NA-11, missed:a11y#1 · *Verdict:* CONFIRMED (reproduced live)

- **What's wrong:** After typing in the test-chat box, a Try-it sample or an entry field, pressing Escape does nothing, so pressing 'p' to go to Personas types a 'p'. The User Guide tells users to press Escape first - which only works in the search box.
- **Who it hurts, and when:** Keyboard users get 'why did my key type a letter?' moments in the test-chat and Try-it workflows, and must know to Tab or F6 out instead. *When:* Every time a user types and then wants a hotkey.
- **Evidence:**
  - `personas_screen.py:16318-16351` (code; KA-04/NA-11): action_personas_escape handles only pending resume, editor cancel, transcript close and focus id 'personas-library-search'
  - `captures/review-rv-a11y-esc-in-testchat-then-p-160x45; review-rv-a11y-tryit-esc-then-c-160x45` (live; KA-04): 'p' inserted into the test input; 'c' into the Try-it sample; footer still advertises c/p/d/l
  - `captures/review-rv-nng-a-2-escape-does-not-leave-tryit-120x36` (live; NA-11): Box reads 'my colourp'; mode unchanged
  - `vf-a11y 160x45` (live; KA-04 verification): 'hi', Escape, 'p' -> 'hip'
  - ...and 2 more in [Appendix A](#a2-rp-026).
- **Recommendation:** Add a lowest-priority 'leave field' branch to the one Escape handler: from any Input/TextArea in the work pane or Inspector, move focus to that pane's first non-text control (Run preview, Test Reply, the list). Advertise it as 'esc leave field'. In editors keep Escape = cancel-with-guard, and say so in the footer.
- **Library frame:** Library's last-resort Escape ('library_blur_text_field', 'Back to canvas') hands focus to the first non-text control so printable keys work on the next press; its footer says 'esc leave field'. Adopt that, but keep Roleplay's single Escape owner rather than Library's 15-18-binding ladder.
- **Verifier corrections applied:** NA-11's footer half is in RP-034.

<a id="rp-027"></a>
#### RP-027 · Tab stops are invisible or off-screen: nested scroll containers, a blank greeting picker, entry-form controls scrolled out of view, and a Generate button before every field

**P1 · severity 3 · scope: content · kind: bug · effort M** · sizes: all; worse at 120x36  
*Principle:* WCAG 2.4.7 Focus Visible and 2.4.11 Focus Not Obscured; H1 Visibility  
*Sources:* KA-07, TF-20 · *Verdict:* PARTIAL (corrections applied)

- **What's wrong:** Pressing Tab often changes nothing on screen - up to six times in a row in the dictionary form - because focus has gone to an invisible scroll container, a blank greeting picker, or a form control scrolled out of view. In the character editor, each field's 'Generate' button takes a Tab stop before the field itself.
- **Who it hurts, and when:** Keyboard authors lose their place and cannot tell whether the next key will edit a hidden field. *When:* Every keyboard pass through the card, test chat, editor or dictionary form.
- **Evidence:**
  - `personas_screen.py:1634; personas_character_card_widget.py:100; personas_preview_pane.py:124, 131` (code; KA-07): VerticalScroll#personas-detail-stack wraps VerticalScroll#personas-character-card-body; preview transcript scroll; greeting Select
  - `rp-review/a11y/steps tab1 #2-#4, tab3 #1-#2, tabchat #1-#3` (live; KA-07): Two invisible stops, then the card scrolls with no cue
  - `rp-review/a11y/steps dicttab #7-#12; captures/review-rv-a11y-dict-entry-form-focus-offscreen-160x45.png` (live; KA-07): Six consecutive Tabs with no visible change; no focus indicator anywhere
  - `rv-flows 160x45 J1: Name > Tab` (live; TF-20): First Tab moves to the First-message 'Generate' button with no visible change
- **Recommendation:** Set can_focus=False on #personas-detail-stack and #personas-character-card-body (one scroll owner per pane). Fix the greeting Select's rendering. Scroll the focused descendant into view in entry forms (scroll_visible on DescendantFocus) and give the forms enough height (RP-006). Put Generate after its field in tab order, or give it a visible focus style.
- **Library frame:** No Library pattern directly addresses these stops; Library work panes keep one scroll owner per pane. Fix locally.
- **Verifier corrections applied:** The Inspector VerticalScroll does show focus (border turns blue) - dropped. The greeting row is already hidden when there are no alternates; the bug is that it paints blank when alternates exist. TF-20's post-save half is in RP-038.

<a id="rp-047"></a>
#### RP-047 · Both 'Send to Console draft' hand-offs are inert: the card or test-chat transcript never reaches the model, yet stays shown as staged and follows you into later chats

**P1 · severity 3 · scope: content · kind: bug · effort M** · sizes: all  
*Principle:* H1 Visibility of system status; H2 Match (say what will happen); H4 Consistency (one label, several meanings); H5 Error prevention  
*Sources:* TF-06, IA-12, G1-04 · *Verdict:* CONFIRMED (gap round, logged prompt bodies; raised from P2/2 to P1/3)

- **What's wrong:** Roleplay's secondary hand-off stages the character card (from the Inspector, also bound to ctrl+enter, the footer's only hand-off key) or the test-chat transcript as a 'source' for your next Console message. At send time the Console drops it, because it is not a Library note, media item or conversation, and logs 'excluded; reason=not_canonical_local_evidence'. The model sees only 'Use Captain Isolde Varga to guide the next response.' or 'Continue this conversation in character.' The screen keeps saying one source is staged, and that phantom source follows you into unrelated chats until you find Un-stage.
- **Who it hurts, and when:** A user who adds Isolde as context, or continues a test chat in Console, gets a generic assistant that has never seen the card or the transcript, while the UI claims the context is attached. The leftover source then appears in later chats. Starting a chat and adding context also look and toast alike. (The leak does not block sends: the blocks seen earlier were RP-072.) *When:* Every use of either 'Send to Console draft'.
- **Evidence:**
  - `personas_preview_controller.py:539-552` (code; TF-06): Test-chat hand-off stages only the transcript as 'Personas preview conversation' with 'Continue this conversation in character.'
  - `captures/review-rv-flows-2-j4-handoff-result-160x45 rows 4, 38-42; review-rv-flows-2-j7-persona-chat-console-160x45 rows 39-43` (live; TF-06): 'No current character', 'Assistant: General'; leftover source in the later Cozy Storyteller session
  - `personas_inspector_pane.py:287, 294, 1037-1094; personas_screen.py:7378-7387, 1655-1662` (code; IA-12): No tooltip on enabled Chat now/Send; toasts 'Chat staged in Console.' vs 'Staged in Console.'; 'Resume chat', 'Send transcript to Console draft'
  - `library_conversation_reader.py:288; library_media_viewer.py:435` (code; IA-12): Library: 'Resume conversation', 'Use in Console'
  - ...and 7 more in [Appendix A](#a2-rp-047).
- **Recommendation:** Either (a) make Roleplay hand-offs real context: inject the card or transcript as a rendered context block with RENDERED_SYSTEM provenance; or (b) re-route them: the test chat's 'Continue in Console' starts a character-bound session seeded with the preview turns, and the Inspector secondary becomes 'Add card to next message', inlining the card text into the draft. Either way, clear staged sources after the send that admitted them, and label an excluded source 'Not used: <reason>' in the staged strip. Until fixed, hide both buttons and the ctrl+enter binding. Keep 'Chat now' (sub-line 'Opens a new Console chat as <name>'), call the transcript action 'Resume conversation', and give each hand-off a distinct toast. Test: the card text reaches the provider request.
- **Library frame:** The Library's 'Use in Console' stages canonical media and note evidence, which is what the Console accepts, so the Library hand-off works. Roleplay's payload types were never made canonical. Copy the Library's naming ('Use in Console', 'Resume conversation'), but the payload needs its own fix; a single verb would also erase Roleplay's useful start-vs-stage distinction.
- **Verifier corrections applied:** The Inspector's 'Send to Console draft' also binds no character (it stages the card as a source) and the shared label was deliberate (task-2232). Staged-source persistence is Console staging scope. Gap round (G1-04): raised to severity 3 because a visible action is silently a no-op while the UI claims a source is staged. The leak does not block (a persona send with the leaked source completed with capture off); the earlier rv-flows-2 block was RP-072.

<a id="rp-056"></a>
#### RP-056 · Lore and dictionary entries cannot be found at realistic volume: search matches book names only, entry tables have no filter and stop at 12 rows (entry #200 of 250 took 99 wheel notches)

**P1 · severity 3 · scope: both · kind: gap · effort M** · sizes: all: 4 entry rows at 120x36; 10 at 160x45 on a clean visit (3 with the RP-001 band); 10 at 220x55 (the cap does not grow with height)  
*Principle:* H7 Flexibility and efficiency of use; H6 Recognition rather than recall; scroll cost  
*Sources:* NB-14, LF-21, G3-02 · *Verdict:* CONFIRMED at realistic volume (gap round G3-02; raised from P2/2 to P1/3)

- **What's wrong:** In Lore and Dictionaries the rail search matches book names only, and the entry table has no filter of its own and never shows more than 12 rows. To edit entry #200 of a 250-entry book you scroll a keyhole: 99 wheel notches, and because every save puts the table back on row 1 (RP-014), 115 notches to find it again. 'Which lore book defines Emberfall?' means opening every book.
- **Who it hurts, and when:** Imported community lorebooks (100-300 entries) are effectively uneditable by mouse: each fix costs a long blind scroll and the position is lost after every save. The 4-keystroke find that works for characters does not exist for world info. *When:* Every edit of a specific entry in a book above ~20 entries; every 'which book defines X?' question.
- **Evidence:**
  - `personas_screen.py:4308-4312, 4390-4394` (code; NB-14): Dictionary and lore filters match record name only
  - `personas_dictionary_detail.py:142-148, 203` (code; NB-14): Entry table min 6/max 12 rows, no filter input
  - `personas_screen.py:4466-4477; Widgets/Library/library_rail.py:1099-1126` (code; LF-21): Search cleared per mode; Library two-level search
  - `rv-gap3 160x45, Lore > Grand Codex (250 entries), find #200 'Saltmarsh Bellwarden' by wheel` (live; G3-02): 99 notches through a 3-row window (RP-001 band present); after Update, 16 notches up plus 99 down to re-find it; 7 Tabs through 6 off-screen controls to reach Content (RP-003)
  - ...and 4 more in [Appendix A](#a2-rp-056).
- **Recommendation:** Add a 'Filter entries…' box above each entry table matching keys, secondary keys and content (lore) or pattern and replacement (dictionaries), with a live 'showing 3 of 250' count. Remove the 12-row cap so the entries list takes the work pane's remaining height (at least 15 rows at 160x45). Keep the cursor anchored by entry across saves (RP-014). Extend rail search to entry keys and content, with the hit reason under the row ('matches key "Emberfall"'). Cross-mode search is optional.
- **Library frame:** Every Library list has its own filter box ('Filter conversations… (Enter)') above the items, plus the rail search. That two-level pattern is what entry tables need; the frame supplies height but not the filter or a taller table.
- **Verifier corrections applied:** LF-21 over-counted (Kepler-9 lives in two places, and character search already matches descriptions); cross-mode search is a nice-to-have. Gap round (G3-02): a clean first Lore visit at 160x45 shows 10 entry rows; the 3-row window at volume exists only with the RP-001 band. DataTable has PageDown/End/Ctrl+End (textual _data_table.py:279-284), so 99 notches is the mouse-only worst case: the cost comes from distance, not window size. Raised to severity 3 because at realistic volume entries cannot be found, and their position cannot be kept.

<a id="rp-074"></a>
#### RP-074 · After a refused send, the Console's recovery card never appears, so 'Send without capture', 'Retry' and 'Cancel' are unreachable and the chat is a dead end (TASK-33621.2 AC#2/#3)

**P1 · severity 3 · scope: external (Console; TASK-33621.2 AC#2/#3) · kind: bug · effort S** · sizes: all  
*Principle:* H9 Help users recognize, diagnose and recover from errors; H3 User control and freedom  
*Sources:* G1-02 · *Verdict:* PARTIAL (rated 3, not 4: same incident as RP-072)

- **What's wrong:** The Console already has a 'Trace capture blocked' card with Retry, Send without capture and Cancel, and the controller supports all three for this kind of pause. But the card is filtered to appear only for two other pause kinds, so after the RP-072 refusal nothing appears. The controller's own message ('Retry, Send without capture, or Cancel') promises actions nobody can see, and Inspect says 'Run: Ready'.
- **Who it hurts, and when:** A recoverable error becomes a dead end: the one working escape (Send without capture) is built but invisible, and the composer refuses a retry. While RP-072 is open, this S-sized fix turns the catastrophe into one extra click. *When:* Every RP-072 refusal, and any later trace-provenance failure.
- **Evidence:**
  - `tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py:43-66 (filter at 55-62), 179-205` (code; G1-02): trace_call_recovery_state returns None unless pause_kind is TRACE_CALL or TEMPORARY_CAPTURE, so the always-mounted card stays hidden (display = state is not None, :205)
  - `tldw_chatbook/Chat/console_chat_controller.py:12042-12060, 8250-8864; Chat/console_turn_preparation.py:78-82` (code; G1-02): The controller writes 'Trace provenance could not be saved. Retry, Send without capture, or Cancel.' and supports all three actions for TRACE_PROVENANCE
  - `rv-gap1 160x45: Inspect rail opened and scrolled x12; status strip toggled; grep for trace/retry/without capture` (live; G1-02): No match anywhere; Inspect says 'Run: Ready' (captures/review-rv-gap1-isolde-blocked-inspect-160x45 rows 13, 19-24); the only cues are the gutter 'blocked' and '⛔ Blocked' in the Chats list
  - `rv-gap1 persona session (trace.log 20:38:27); vf-gap1` (live; G1-02): A composer retry is refused ('Another send is still preparing…', controller_submit status=refused) and parked in the unsent-turn strip
- **Recommendation:** Add TRACE_PROVENANCE to the set at provider_continuation_recovery.py:59-62; map Retry to retry_library_preparation, Send without capture to send_without_capture and Cancel to cancel_library_preparation; Problem line 'Chat history could not be recorded for tracing'. Make Inspect's 'Run:' line and the header chip say 'Blocked: <reason>'. Ship it ahead of the RP-072 fix.
- **Library frame:** Not a Library concern. The Console's own 'Trace capture blocked' card (Owner / Problem / Impact / actions) is a good pattern; it is just not wired for this pause kind.
- **Verifier corrections applied:** Same incident as RP-072, and the recovery half is already TASK-33621.2 AC#2/#3, so a second independent severity 4 would double-count one failure. Rated 3 on its own: once RP-072 is fixed it bites only on residual provenance failures. A Console dependency, not Roleplay content.

<a id="rp-075"></a>
#### RP-075 · Saving a persona whose name is over 200 characters crashes the whole app, losing every unsaved draft

**P1 · severity 3 · scope: content · kind: bug · effort S** · sizes: all (reproduced at 160x45 and 120x36)  
*Principle:* H9 Help users recognize, diagnose and recover from errors; H5 Error prevention (unconstrained input); stability  
*Sources:* G2-01 · *Verdict:* PARTIAL (reproduced live; rated 3 for the rare trigger)

- **What's wrong:** The persona Name box accepts any length, but the data model allows 200 characters. Save raises a validation error whose text contains square brackets; the screen passes that text to a toast that reads brackets as styling markup, the markup parser fails, and the app exits to a Python traceback.
- **Who it hurts, and when:** One over-long paste into Name (for example, a description pasted into the wrong box) kills the app. Every open draft (character, persona, visual) is lost, and the only message the user sees is a traceback. *When:* Rare trigger, deterministic: any persona name over 200 characters (or a tool-rule name over 512) on Save.
- **Evidence:**
  - `rv-gap2 160x45: Personas > Cozy Storyteller > Edit > Name = 204 chars > Save` (live; G2-01): The app exited to a Textual traceback ('MarkupError: Expected markup value …') and the tmux server died; the unsaved System prompt edit was lost (captures/review-rv-gap2-persona-name-too-long-save-160x45 rows 1-46)
  - `runs/rv-gap2/data/rp_review/tldw_cli_app.log:394, 409-410` (capture; G2-01): Pydantic string_too_long, then 'event=unhandled_exception … exception_type=MarkupError widget_type=PersonasScreen', repeated in MainNavigationBar._update_overflow_hints
  - `vf-gap2-2 120x36: New persona with a 205-char name > Save` (live; G2-01 verification): App exited; tldw_cli_app.log:392-408 shows LocalPersonaProfileCreate string_too_long, then the same MarkupError
  - `persona_profile_editor_widget.py:127, 544-568; tldw_api/character_persona_schemas.py:562, 581, 626` (code; G2-01): The Name Input has no max_length and validate() checks only for blank; the schemas cap name at 200 and rule_name at 512
  - ...and 1 more in [Appendix A](#a2-rp-075).
- **Recommendation:** (1) In PersonasScreen._notify pass markup=False (or escape the message): one line that protects every caller. (2) Give the Name input max_length=200 with a live inline check ('Name: 200 characters max (204)'). (3) Map pydantic errors to field-level copy instead of interpolating str(exc). (4) Regression test: a ValidationError from the persona save path renders a toast without raising. Fix first: it is S-sized and kills the app.
- **Library frame:** Not a frame issue. House precedent for the fix: UI/Evals/notify_mixin.py:40-48 (task-1476) and evals_screen.py:1228 pass markup=False for exception text. Check the Library's own notify calls for the same default before sharing a wrapper.
- **Verifier corrections applied:** The crash also hits the create path (LocalPersonaProfileCreate, personas_screen.py:15671), and the ValidationError comes from the screen's own model construction (:15697), not the service. Frequency narrowed: only bracket text shaped like markup attributes ('[type=…, input_value=…]') or '[/]' raises; '[Errno 2] …' and similar render fine, so the trigger is pydantic validation text, not every bracketed error. Severity 3: catastrophic impact, rare trigger.

<a id="rp-076"></a>
#### RP-076 · Each long persona field is a full-screen box, so one field is visible at a time and Mode/Enabled sit 3-4 screens down (the inverse of RP-002)

**P1 · severity 3 · scope: content · kind: usability · effort S** · sizes: all: one field per screen at 120x36, 160x45 and 220x55  
*Principle:* H8 Aesthetic and minimalist design; progressive disclosure; form scanning (related fields seen together)  
*Sources:* G2-03 · *Verdict:* CONFIRMED

- **What's wrong:** The persona editor's Description, System prompt and Personality traits boxes have no height rule, so each takes the full height of the editor. At 160x45 a one-line description sits in a 23-row box; you see Name and Description and nothing else, and the editor is about 4-6.5 screens tall. Text stays readable and clicks land correctly (unlike RP-002), but related fields are never visible together.
- **Who it hurts, and when:** An author cannot see the system prompt next to the description or traits, scrolls through screens of empty box to reach Mode/Enabled, and the full-height boxes squeeze the sections below them to nothing (RP-073). *When:* Every persona edit.
- **Evidence:**
  - `captures/review-rv-gap2-2-persona-editor-open-120x36 rows 16-30` (capture; G2-03): 15-row window: Name, then the top of a ~14-row Description box holding a 2-line description
  - `captures/review-rv-gap2-persona-editor-open-160x45 rows 16-39` (capture; G2-03): Name, then a 23-row Description box with 1 line of text; System prompt, traits, Mode and Enabled below the fold
  - `captures/review-rv-gap2-3-persona-editor-open-220x55 rows 16-49` (capture; G2-03): A ~33-row Description box holding 1 line
  - `rv-gap2 160x45 / rv-gap2-2 120x36, wheel over the editor` (live; G2-03): Mode first appears after ~31 wheel steps at 160x45 and ~18 at 120x36; scroll extent ≈125 rows at 160x45 (~5 screens), ≈97 at 120x36 (~6.5), ≈150 at 220x55 (~4.4)
  - ...and 2 more in [Appendix A](#a2-rp-076).
- **Recommendation:** In the persona Edit mode make System prompt the single fill field (height 1fr, min 6); Description and Personality traits height:auto, min 3 / max 8 rows, growing with content; move Mode/Enabled to Profile/Info (RP-086). Test: at 120x36, Name, Description and the top of System prompt are visible together.
- **Library frame:** The Library gives exactly one primary field the remaining height and keeps the others compact (css/features/_library.tcss:1293-1312). Copy that rule; the frame alone does not change field heights.

<a id="rp-077"></a>
#### RP-077 · A persona's Description and Personality traits never reach the model in Console, but the test chat sends the Description, so the preview and the real chat disagree

**P1 · severity 3 · scope: content · kind: bug · effort M** · sizes: all (behaviour)  
*Principle:* H1 Visibility of system status (truthful feedback); H2 Match (fields imply effect); preview/production consistency  
*Sources:* G2-04 · *Verdict:* CONFIRMED

- **What's wrong:** A Console persona chat sends only the persona's System prompt. The Roleplay test chat builds its prompt from the persona's Description instead, and from the saved record rather than your unsaved edits. Personality traits are sent nowhere. A persona written the way the editor invites (description plus traits, no system prompt) tests well in Roleplay, then chats in Console as the default assistant wearing the persona's name.
- **Who it hurts, and when:** J4 authors cannot trust the test chat for personas, traits are dead data, and persona edits cannot be tested before saving. *When:* Every persona whose behaviour lives in Description or Personality traits; every persona test chat.
- **Evidence:**
  - `tldw_chatbook/UI/Console_Modules/session.py:584-596, 615-645` (code; G2-04): _persona_session_prompt_seed takes only profile['system_prompt'] ('no multi-field join'); characters join description, personality, scenario and more
  - `UI/Persona_Modules/personas_preview_controller.py:49-101, 463-494` (code; G2-04): The test chat runs persona records through compose_character_card_text with description and reads 'personality' (a key personas lack); personas use the saved record, characters the draft; the docstring claims the preview 'shows exactly the system prompt a real Console session would send'
  - `rv-gap2-3 220x55: Isolde's Bridge Officer (Description only) > Try a test chat (captures/review-rv-gap2-3-linked-persona-testchat-uses-description-220x55 rows 16-18)` (live; G2-04): Reply 'Staying in character as A persona bound to the Captain Isolde Varga card…': the Description is used as the system prompt
  - `rv-gap2-4 160x45: same persona > Chat now > send (captures/review-rv-gap2-4-linked-persona-chat-now-console-160x45 row 41; review-rv-gap2-4-linked-persona-console-reply-160x45 row 35)` (live; G2-04): 'System Prompt: off · Persona: Isolde's Bridge Offi…'; reply 'Staying in character as You are a capable assistant with optional tools…'
  - ...and 1 more in [Appendix A](#a2-rp-077).
- **Recommendation:** Decide what a persona sends (decision D15): either (a) compose System prompt + Description + Personality traits in one function used by both the preview and _persona_session_prompt_seed, or (b) label Description and traits 'Notes - not sent to the model'. Either way make the persona preview use that same seed function and the editor draft; warn on Save when System prompt is empty ('No system prompt: chats will use the default assistant prompt'); add a preview-vs-Console parity test.
- **Library frame:** Library editors put a one-line help gloss under each field (for example 'Instructions - Instructions the model always follows.'). The frame does not fix this; it needs one shared persona composer.
- **Verifier corrections applied:** Console's system-prompt-only behaviour may be deliberate tldw_server parity (TASK-617). Whichever way it is decided, the preview/Console mismatch is the defect. Extends RP-045 (test chat truthfulness).

<a id="rp-078"></a>
#### RP-078 · The Inspector's 'Tool policy' line shows the last-edited persona's rules on other personas, and 'no rules' for personas that have rules

**P1 · severity 3 · scope: content · kind: bug · effort S** · sizes: all  
*Principle:* H1 Visibility of system status (false status on a permissions setting)  
*Sources:* G2-05 · *Verdict:* CONFIRMED

- **What's wrong:** Selecting a persona never refreshes the Inspector's tool-policy line. It keeps whatever the last opened editor or rule save wrote, so persona B can show persona A's rules, and a persona that has rules shows 'no rules' when first selected.
- **Who it hurts, and when:** Users judge a persona's tool permissions from a line that may describe a different persona, and may hand a persona to Console believing it is restricted when it is not, or the reverse. *When:* Every persona selection after any Edit open or rule save in the session; the first selection of any persona that has rules.
- **Evidence:**
  - `rv-gap2-4 160x45: Terse Code Reviewer > Edit > (blind) add 'skill: summarize → allow' > Cancel > select Cozy Storyteller, then Isolde's Bridge Officer` (live; G2-05): Both show 'Tool policy: 1 rule(s) / skill: summarize → allow' although neither has rules (captures/review-rv-gap2-4-stale-policy-summary-other-persona-160x45 rows 19-20)
  - `personas_screen.py:5383-5440 vs 8374, 15600, 15784; personas_inspector_pane.py:249-253, 403-431` (code; G2-05): _select_profile never calls show_policy_rules; only Edit-open, rule save and after-save do; clear_selection does not reset it; the compose-time text is 'Tool policy: no rules'
  - `personas_inspector_pane.py:542-567, 984-987` (code; G2-05): Visible for every persona; the copy flips between 'no rules' and 'no rules (default posture)', and neither mentions the deny-by-default effect of allow rules
  - `runs/rv-gap2-4 personas.json` (data; G2-05 verification): Cozy Storyteller has rules=None; Terse Code Reviewer has [skill: summarize]
- **Recommendation:** Call show_policy_rules(record.get('policy_rules') or []) in _select_profile and clear it in clear_selection. Render it in the persona Profile/Info mode, with the deny-by-default sentence when any allow rule exists. One copy: 'Tool policy: none (all tools allowed by default)'. Test: select persona B after editing persona A and assert B's rules.
- **Library frame:** The Library has no inspector: metadata is rendered in an Info mode from the loaded item, with selected and loaded identity fenced by a generation (ADR-084). Moving the summary into the persona Profile/Info mode fixes this structurally.
- **Verifier corrections applied:** A content fix (one line in _select_profile and clear_selection); moving the summary to an Info mode is optional. The stale value sticks after any Edit open or rule save, even one that sets 'no rules'; merely having a persona with rules does not leak, but that persona reads 'no rules' on first selection.

<a id="rp-079"></a>
#### RP-079 · After a persona Save then Cancel, the card still shows the previous persona (or the pre-edit text) while the Inspector names the saved one; Edit then fails with 'Selection out of sync'

**P1 · severity 3 · scope: content · kind: bug · effort S** · sizes: all (reproduced at 120x36 and 160x45)  
*Principle:* H1 Visibility of system status; H4 Consistency (one selection, shown everywhere); H5 Error prevention (actions target an unseen item)  
*Sources:* G2-06 · *Verdict:* CONFIRMED (wider than first reported)

- **What's wrong:** Save keeps the persona editor open; Cancel then re-shows the card, but the card is never re-rendered after a save. After creating a persona, the card shows the persona that was selected before while the Inspector names the new one, and Edit refuses with 'Selection out of sync; reselect the persona.' After editing an existing persona, the card shows the old text.
- **Who it hurts, and when:** Right after the main J4 create step, the screen shows one persona while Chat now, Export and Delete act on another (the Delete dialog names the right one, which mitigates but does not remove the risk). The user is told to 'reselect' with no reason. *When:* Most persona edit sessions (Save keeps the editor open, so Cancel is the normal exit after every save); every create followed by Cancel.
- **Evidence:**
  - `rv-gap2-2 120x36: select Default Narrator > Ctrl+N > 'Repro B' > Save > Cancel` (live; G2-06): Card 'Name: Default Narrator…' while the Inspector reads 'Selected: Repro B'; Edit gives 'Selection out of sync; reselect the persona.' (captures/review-rv-gap2-2-stale-card-after-create-save-cancel-120x36 rows 15-21)
  - `rv-gap2-2 120x36: New Actor Pack 'Wren Pack Persona' > Save > Cancel` (live; G2-06): Card 'Cozy Storyteller', Inspector 'Selected: Wren Pack Persona', Edit refused (captures/review-rv-gap2-2-edit-selection-out-of-sync-120x36 rows 15-31)
  - `vf-gap2 160x45: Default Narrator > Ctrl+N > 'Verify B' > Save > Cancel; then rename Cozy > Save > Cancel` (live; G2-06 verification): Same split (captures/review-vf-gap2-stale-card-edit-out-of-sync-160x45); the edit case leaves the pre-edit card (captures/review-vf-gap2-stale-card-after-edit-save-cancel-160x45 rows 15-20)
  - `personas_screen.py:5408, 8333-8341, 15733-15810, 15866-15882` (code; G2-06): show_persona is called only in _select_profile; _after_profile_save selects the saved id but never re-renders the card; the cancel path re-shows the stale card; Edit rejects the mismatched id
- **Recommendation:** In _after_profile_save call card.show_persona(saved); in _finish_cancel_profile_edit re-render the card from the selected record (or call _select_profile(selected_id)). Tests: create -> save -> cancel and edit -> save -> cancel, then card name == Inspector name.
- **Library frame:** ADR-084 (Library Media) separates selected and loaded identity and fences them with a generation; the work pane always renders the loaded item. Adopt that, with a pinned title row naming the open item (RP-052).
- **Verifier corrections applied:** Wider than stated: editing an EXISTING persona, Save, then Cancel also shows the pre-edit card (live: renamed Cozy to 'Cozy Storyteller Renamed'; the card still read 'Cozy Storyteller' while the rail and Inspector read 'Renamed'). Edit still works in that case because the ids match.

<a id="rp-080"></a>
#### RP-080 · Changing page keeps the old scroll offset: after scrolling to the end of one page, the next page opens at its bottom and about 41 characters are never shown

**P1 · severity 3 · scope: content · kind: bug · effort S** · sizes: all where a page holds more items than the viewport (every target size at 50 per page)  
*Principle:* H1 Visibility of system status; predictable navigation (a new page starts at its top)  
*Sources:* G3-04 · *Verdict:* CONFIRMED

- **What's wrong:** With more than 50 characters the list is paged. Scroll to the end of a page and press '>': the next page opens scrolled to its bottom, so most of it is skipped without any sign. The first round never saw this because its 28 characters fit on one page.
- **Who it hurts, and when:** Users browsing hundreds of cards skip whole runs of names without knowing; with 2-row items and an ~8-item viewport about 80% of each page goes unseen. *When:* Every page change after scrolling - the natural flow, because '>' sits under the list.
- **Evidence:**
  - `rv-gap3 160x45 (348 characters): page 4, wheel to the end ('Niall 3' last), click '>'` (live; G3-04): Page 5 (201-250) opened at items 42-50; 'Niall 4' through 'Quilla Calloway' were never shown (captures/review-rv-gap3-page5-lands-at-bottom-160x45 rows 23-40: thumb low, '201-250 of 348')
  - `captures/review-rv-gap3-page2-160x45 row 23` (capture; G3-04): Page 2 opens on an orphan description line whose name row is scrolled off above
  - `vf-gap3 160x45: end of page 1, '>'` (live; G3-04 verification): Page 2 opened at its bottom (captures/review-vf-gap3-page2-after-scroll-160x45); page 3 too; going back kept the bottom offset
  - `personas_library_pane.py:419-482, 525-533; personas_screen.py:4141-4152; textual _list_view.py:261-270` (code; G3-04): update_rows clears and extends with no scroll reset; ListView.clear() only sets index=None; only mark_active_row moves the viewport
  - ...and 1 more in [Appendix A](#a2-rp-080).
- **Recommendation:** On every page change call list_view.scroll_home(animate=False) and set index 0 unless the selected row is on the new page. Better: one virtual list without page boundaries (D14).
- **Library frame:** Library Conversations' 'Next' lands at the top of the next page (live: scrolled to the bottom of page 1, Next shows 'Aetheria session 10', the first item of page 2). Copy the scroll reset, or remove paging (D14).

<a id="rp-081"></a>
#### RP-081 · Try it at realistic volume: at 160x45 the 'what fired / what was dropped' summary and per-rule lines are clipped off, and the summary calls budget-dropped lore 'near-miss'

**P1 · severity 3 · scope: both · kind: usability · effort M** · sizes: 160x45 (summary invisible), 120x36 (worse), 220x55 (summary visible, 1-7 detail rows)  
*Principle:* H1 Visibility of system status; H2 Match between system and the real world (why lore was dropped)  
*Sources:* G3-09 · *Verdict:* PARTIAL (vocabulary claim narrowed)

- **What's wrong:** Try it is the best feature on the screen, but its result box is capped at 60% height and does not scroll. With a 3-line sample against a big dictionary or lore book, the processed text fills the box and the summary ('24 fired · 2 skipped') and the per-rule lines are cut off; the wheel does nothing. At 220x55 the lore summary is visible but reads '5 fired · 64 near-miss · over budget', although those 64 entries matched their keys and were dropped by the token budget.
- **Who it hurts, and when:** Try it is the only way to verify world info (J3). At the medium target size it hides the answer to 'what fired, what was dropped and why', and 'near-miss' points users at their keys when the real fix is the budget or entry priority. *When:* Every preview with a realistic sample (3+ lines), or a book or dictionary that fires more than a few entries.
- **Evidence:**
  - `captures/review-rv-gap3-dict-tryit-many-160x45 rows 26-41; review-rv-gap3-dict-tryit-many-scrolled-160x45` (capture; G3-09): 3-line sample vs the 150-rule dictionary: summary and per-rule lines fall below the widget; 40 wheel notches over the output moved nothing
  - `captures/review-rv-gap3-dict-tryit-many-220x55 rows 44-51` (capture; G3-09): Only at 220x55: '24 fired · 2 skipped · 24/1000 tokens', then a 7-row keyhole of 26 lines
  - `captures/review-rv-gap3-lore-tryit-many-160x45 rows 33-41; review-rv-gap3-lore-tryit-many-220x55 rows 51-52` (capture; G3-09): 160x45: the injection text alone fills the space; 220x55: '5 fired · 64 near-miss · 919/1000 tokens · over budget' with 1 detail line
  - `personas_dictionary_tryit.py:74-93; personas_lore_tryit.py:39-62, 148-158, 183-189` (code; G3-09): Both widgets are Vertical (overflow hidden) with max-height 60%; the lore summary counts every non-fired record as 'near-miss'
  - ...and 1 more in [Appendix A](#a2-rp-081).
- **Recommendation:** Make Try it its own work-pane mode with its own scroll and the summary first: '5 injected · 64 dropped by budget (919/1000 tokens) · 3 disabled'. Collapse long injected text to 2 lines per entry, expandable. Group non-fired records ('Dropped by token budget (64)', sorted by priority with the cut-off shown; 'Disabled (n)'; 'Secondary key missing (n)'). Replace 'over budget' with 'budget full: next entry needs N tokens'. Keep 220x55's behaviour as the minimum.
- **Library frame:** No Library equivalent. Making Try it a full-height work-pane mode (D2) is the frame-level fix; the vocabulary fix is content.
- **Verifier corrections applied:** Only the summary count lumps budget, disabled and secondary-key cases together as 'near-miss'; each non-fired row already prints its true reason ('dropped by token budget', 'disabled (key … matched)', 'secondary key not found'; personas_lore_tryit.py:183-189), but those rows are invisible at 160x45 because of the clip. So the fix is mainly the frame/scroll plus splitting the summary count. Overlaps RP-006 (the permanent Try-it block) and mirrors RP-010 (the test chat's 60% cap).


---

## 6. What works: preserve in the redesign

Verified strengths, merged across lenses. Several are better than the Library.

**Protects your work**
- **AI generation never overwrites without consent.** A "Generate" result appears in a preview and applies only on Accept (`personas_character_editor_widget.py:443-462, 1115-1136, 1255-1264`). Keep this in any re-hosted editor.
- **The leave-screen guard offers a real choice:** "Finish Roleplay drafts before leaving: Save and continue / Discard and continue / Stay" (`captures/review-rv-flows-2-j8-leave-with-unsaved-160x45`; the persona editor gets it too, `review-rv-gap2-persona-leave-guard-160x45`). This is ADR-046's aggregate draft veto and is better than the Library's Prompts/Skills hard-veto toast. Extend it to in-screen switches ([RP-037](#rp-037)).
- **Escape is safe.** One handler (`personas_screen.py:16318-16351`) tries cancel-resume, then editor cancel through the guard, then transcript back, then search to list, and never leaves the screen. Keep Roleplay's single owner rather than the Library's 15-18 Escape bindings.
- **Whole-item delete always confirms**, and dictionaries have version history with a confirmed revert (`personas_screen.py:14746-14760, 6390`). Extend both to entries and lore ([RP-014](#rp-014)).
- **Save/Cancel stay anchored outside the editor scroll** (`personas_character_editor_widget.py:700-711`), unlike the Library notes editor, which hides Save at 120x36.
- **Lore "Add" never overwrites the loaded entry**: it creates a new one (but afterwards the table jumps to row 1, [RP-014](#rp-014)).

**Answers "what does this do?"**
- **The dictionary and lore Try-it previews are the best feature on the screen:** struck-through originals, underlined replacements, "5 fired · 0 skipped · 6/1000 tokens", per-rule lines, injections grouped by position and near-misses (`captures/roleplay-dictionaries-tryit-160x45`, `review-rv-flows-2-j6-lore-tryit-result-160x45`). Give them more room as a mode; do not cut them.
- **The character test chat is draft-aware and honest:** it uses unsaved fields, says "(nothing saved)", and shows the provider and status (`personas_preview_controller.py:456-494, 747-858`). For personas it is not: it uses the saved record and a different prompt from Console ([RP-077](#rp-077)). Make the persona preview match before relying on it.
- **Conversation-level lore works end to end.** Attaching a lore book to the conversation put both Kepler-9 entries into the logged prompt (gap round). It is the proven mechanism to build on for D6.

**Fast for experts**
- **Finding a card takes 4 keystrokes** (Ctrl+F, name, Enter, Enter), and mode switches react in ~0.1 s. At 348 characters the search still takes 0.34 s. It does not extend to lore/dictionary entries or to personas past the newest 100 ([RP-056](#rp-056), [RP-093](#rp-093)).
- **Performance holds at volume:** with 348 characters, a cold first visit lists them in about 1 s, page changes take 0.05-0.30 s, and no event-loop stalls were logged during Roleplay (Appendix C.9).
- **The c/p/d/l and [ ] mode keys** (ADR-152) have tooltips naming the key. Keep them as rail-row hints ([RP-033](#rp-033) only narrows where they fire).
- **Bulk mark/delete/export exists** (m, ● marks, "N marked"). Make it visible rather than removing it, and keep marks across pages and searches ([RP-039](#rp-039)).
- **Import is quick:** the picker remembers its folder, the new card is auto-selected and ready to chat, the toast reports an embedded lorebook, and re-importing explains itself ("Character already existed; selected it...").
- **Chat now is a one-click hand-off bound to the character** (Console tab "Chat with Ilsa...", greeting rendered, "System Prompt: set"). The hand-off itself works; the first send in Console currently does not ([RP-072](#rp-072), a Console defect).

**Says things in words**
- **Plain-language mode descriptors:** "who the AI plays", "text find/replace rules", "world facts injected on keywords" (`personas_screen.py:448-457`). These are ready-made rail glosses (the persona one changes per [RP-011](#rp-011)).
- **State is carried by words, not colour alone:** "8 entries · on/off", readiness sentences, "Editing Ada - unsaved", the ● mark plus "N marked".
- **Disabled actions explain why** (F-037; "Save or discard your edits to chat in Console."). Keep the reasons, but show them as text ([RP-018](#rp-018)).
- **Consequence-explicit copy:** "Try a test chat (nothing saved)", "Exports read the last saved state."
- **Default Assistant is a worked example** with breadcrumb paths ("editable from Roleplay > Characters > Default Assistant"). Update the breadcrumb to the new names.

**Good bones**
- **Footer hints are rebuilt from live state** ("space toggle" only in Dictionaries, "ctrl+s save | esc back" only while editing). Extend the mechanism with focus ([RP-034](#rp-034)); do not replace it.
- **Mode and selection survive leaving and returning** when an item is selected (also at 348 characters: page 5 and the selection came back); the Library loses the open item. This is the base for per-mode memory ([RP-030](#rp-030)).
- **Kind-aware hiding of inapplicable actions** (conversations only for characters; Import/Sort/Tag only where they work).
- **The character editor has persistent labels**, per-field Generate, and a red outline on the invalid field. This is the labelled-form idiom the lore/dictionary forms should adopt.
- **Rows that scan:** character rows are 2 lines with no blank separator (denser than the Library's 3), and dictionary/lore rows carry "10 entries · on".
- **Attachments show as one-row headers with counts** ("▸ World Books (1)"), and the conversation list is capped at 10 rows.
- **Architecture that makes migration cheap:** a typed message seam between widgets and screen (`personas_messages.py`, `personas_pane_messages.py`); the ADR-115 demand-mount boundary for the four heavy views; dictionary/lore tabs that map onto work-pane modes; stable pane geometry across modes; toolbar buttons that never clip (F-030).
- **Text inputs have a strong focus style** (accent box). Use it as the reference when fixing list and button focus.
- **At narrow widths, selecting a row switches to the work pane**, matching the Library's single-stage behaviour.

---

## 7. Library pattern transfer

| Library pattern (evidence) | Fits Roleplay? | What it fixes | Caveats / adaptations | Reuse path |
|---|---|---|---|---|
| One-row header "Library \| Local" (`library_screen.py:15017-15021`) | **Yes** | [RP-007](#rp-007), [RP-067](#rp-067) | Library drops DestinationHeader, which ADR-015:30 and DESIGN.md:107 require. Use the one-row inline DestinationHeader that Console/Lab ship. Keep the unsaved signal ([RP-038](#rp-038)). | Generalise `.lab-header-inline`, or an id-scoped rule (boot broad-selector budget is 274/274). No Library code needed. |
| Navigation rail: sections, one-line rows, counts, glosses, ▸ selected (`library_rail.py:579`; `library_shell_state.py:366-420`) | **Yes, adapted** | [RP-029](#rp-029), [RP-028](#rp-028), [RP-008](#rp-008), [RP-043](#rp-043), Buddy's home for [RP-018](#rp-018) | Four browse rows earn 29-39 cols only with Create, Import/Export and Buddy rows too. Must stay open at 120 cols, unlike Library reader routes. Persona gloss per your ruling. Keep c/p/d/l and [ ] (ADR-152). | Build on the shared `DestinationRailSectionHeader` (`destination_rail.py:163`, Home already does) plus a row dataclass like `LibraryRailRow`. Do not subclass `LibraryRail`; it is coupled to Library state. |
| Short-label fallback, gloss dropped first, count never clipped (`library_rail.py:1000-1063`) | **Yes** | Truncated duplicate labels ([RP-008](#rp-008)) | Full nouns ("New lore book") make most fallbacks unnecessary. | Extract `_row_label` into a pure helper. |
| Rail search: stable 3-row box, "x" clear, "/" to focus (`library_rail.py:1099-1126`) | **Yes** | [RP-009](#rp-009), [RP-035](#rp-035) | "/" is printable, so gate it the same way as c/p/d/l. | `SelectAllOnFocusingClickInput` / `LibraryRailSearchInput` reusable as-is. |
| Create and Import / Export rail sections | **Yes** | [RP-008](#rp-008), [RP-043](#rp-043) | Name every noun, or show only the active mode's rows. | Rows in the Roleplay rail. |
| 5-cell grips spelling the pane name (`library_adaptive_reader_shell.py:36, 73-230`) | **Yes, with a decision** | [RP-032](#rp-032) | `design-language.md:132-134` says "never a sliver" (needs a recorded exception). Grips reserve 10 cols even when panes are open. The Library also runs a second, 3-cell grammar; pick one. | Python imports only Utils. CSS is library-prefixed and lazily loaded: rename to neutral tokens. |
| Width resolver: fixed collapse order, work-pane minimum, hysteresis (ADR-086; `adaptive_reader_state.py:217`) | **Yes, with Roleplay settings** | [RP-017](#rp-017), [RP-066](#rp-066) | **Default profiles starve editors** (work pane 50/45/62 cols at 120/140/160). Models only 2 optional panes. ADR-086 says "Library-local", so it needs an amendment (D4). | Pure, config-safe Utils leaf plus shell Python. Template: `LibraryArtifactsController` (~90 lines). Not the 35.9k-line `library_screen.py` glue. |
| Empty work pane donates its width (`adaptive_reader_state.py:519-528`) | **Yes** | [RP-017](#rp-017), [RP-055](#rp-055) | - | Resolver flag `reader_has_item`. |
| Work-first override while editing (`library_notes_controller.py:2446-2473`) | **Yes** | [RP-017](#rp-017), [RP-016](#rp-016) | Library Notes closes only the nav rail. Roleplay should also close the list below ~170 cols. Transient, never persisted; a manual reopen wins. | Copy the contract into a Roleplay controller. |
| Work-pane modes + header CTA + state-machine action bar (`library_prompts_canvas.py:1294-1375, 1628-1720`) | **Yes** | [RP-016](#rp-016), [RP-018](#rp-018), [RP-019](#rp-019), [RP-020](#rp-020), [RP-038](#rp-038), [RP-040](#rp-040), [RP-010](#rp-010) | Keep ADR-046's Save/Discard/Stay, not the Library's veto toast. | Content work inside existing widgets (the typed message seam helps). |
| Info mode for metadata and relationships | **Yes** | "Used by" ([RP-012](#rp-012)), validation ([RP-044](#rp-044)) | Needs D6 first. | A work-pane mode. |
| Labelled fields with a one-line help text (Prompts editor) | **Yes** | [RP-003](#rp-003), [RP-053](#rp-053) | - | Copy the idiom. |
| Count "(…)" while loading; filter-aware empty copy; "1-4 of 4" | **Yes** | [RP-028](#rp-028), [RP-022](#rp-022) | - | Copy and state change. |
| Focus-aware footer: "typing in field · esc leave field · after esc:" (`library_screen.py:4741-4835`) | **Yes** | [RP-034](#rp-034), [RP-033](#rp-033) | - | `register_footer_shortcuts` is shared; copy the rule. |
| Contextual F1 built from the footer set (`library_screen.py:25683-25770`) | **Yes** | [RP-036](#rp-036) | - | Copy the method. |
| Tab confined to `#screen-content` (`library_screen.py:1010-1021, 8975-9008`) | **Yes, lift into the shared frame** | [RP-024](#rp-024) | Library-local code today. | Opt-in flag on BaseAppScreen. |
| Escape "leave field" (`library_blur_text_field`, `library_screen.py:1128, 27190-27212`) | **Yes** | [RP-026](#rp-026) | Add it as a branch of Roleplay's single Escape owner. | One branch. |
| Focus evacuation on collapse/expand (`library_adaptive_reader_shell.py:317-326, 352-447`) | **Yes** | [RP-064](#rp-064) | - | Shell `sync_layout` contract. |
| Dynamic F6 targets with grip substitution (`library_artifacts_reader_shell.py:43-72`) | **Yes** | [RP-025](#rp-025) | Other Library destinations still use static tuples. | Copy the Artifacts method. |
| Persisted preferred pane state, transient effective state (`config.py:2370-2430`; ADR-086) | **Yes** | [RP-031](#rp-031) | Library's ordinary-route Collapse is not persisted; follow the adaptive contract. | New `[roleplay.reader]` section via the existing normaliser (D5). |
| Delete inside "More actions" with danger style; inline confirmation | **Yes** | [RP-014](#rp-014), [RP-039](#rp-039) | Library mixes inline and modal confirmations; pick one. | Content. |
| Below 64 cols: single stage, "‹ Library" return, "esc back" chip (`library_screen.py:5790-5870`) | **Yes (small sizes)** | [RP-066](#rp-066) | Off the design centre. Library's own 60-col reader clips text. | `LibraryEmergencyReturn` (37 lines) reusable. |
| Landing hub (Continue / Needs attention / Quick actions, capped at 96) | **Partial** | [RP-055](#rp-055) | Must be Roleplay-native (recent character chats, Import card). Keep a non-void first paint with a live primary action (F-031). Decide whether it shows on every visit. | Build new. |
| Width preferences in Settings (`settings_appearance_defaults.py:66-82`) | Optional | - | Do not share `[library.reader]` (ADR-086 scopes it to Library). | Later, if asked. |
| Per-list filter box above the items ("Filter conversations… (Enter)") | **Yes** | [RP-056](#rp-056) | Entry tables need it inside the work pane, with a "showing 3 of 250" count. | Content. |
| Pager that resets scroll on Next and says "Page 1 of 2" (Conversations, live at 36 items) | **Yes, if paging stays** | [RP-080](#rp-080), [RP-092](#rp-092) | Do not copy its 3-row pager stack or its missing paging keys (section 10). A virtual list may remove paging (D14). | Copy the wording and the scroll reset. |
| Conversation row meta ("3 messages · 16m · Default · Active") | **Yes, plus the character's name** | [RP-049](#rp-049), [RP-083](#rp-083) | Library rows do not name the character (section 10, #15). | Copy the row format. |
| Selected vs loaded identity fenced by a generation; the work pane renders the loaded item (ADR-084, Library Media) | **Yes** | [RP-078](#rp-078), [RP-079](#rp-079) | - | Copy the contract into the persona and character work panes. |
| Canonical hand-off payloads ("Use in Console" stages media/notes the Console accepts) | **Partly** | [RP-047](#rp-047) | Roleplay's card and transcript payloads are not canonical, so the Console drops them; copy the naming, fix the payload. | Content (Console contract). |

---

## 8. What the Library frame will NOT fix

These findings survive a perfect frame swap and need their own work. Several are cheap and should go first.

- **Container bugs inside panes** (fix first, with geometry tests): leaked slots [RP-001](#rp-001); 0-1-row editor fields [RP-002](#rp-002); off-screen, unlabelled entry-form controls [RP-003](#rp-003); unscrollable Settings tabs [RP-005](#rp-005) (the frame helps only by anchoring actions); invisible tab stops [RP-027](#rp-027); nested scrollers [RP-054](#rp-054).
- **World-info semantics:** character attachments not applied in Console [RP-004](#rp-004) (confirmed live); silent copies with no "Used by" [RP-012](#rp-012) (the frame gives the slot, not the data); card-embedded lore invisible [RP-013](#rp-013); the row-1 reset plus instant delete and detach [RP-014](#rp-014); a bare-title picker for the route that works [RP-083](#rp-083).
- **Authoring completeness:** editor structure and modes [RP-016](#rp-016) (the frame supplies the mode grammar; the field split is content); example dialogue [RP-023](#rp-023); alternate greeting for Chat now [RP-048](#rp-048); card/editor order and glosses [RP-053](#rp-053); stale test-chat greeting [RP-046](#rp-046); test chat truthfulness [RP-045](#rp-045).
- **Copy and error handling:** persona copy [RP-011](#rp-011); invalid-regex warning [RP-015](#rp-015); no-match copy [RP-022](#rp-022); unsaved-dialog copy [RP-037](#rp-037); save rules per mode [RP-040](#rp-040); validation copy and widget ids [RP-044](#rp-044); hand-off verbs [RP-047](#rp-047); import errors [RP-058](#rp-058); provider errors [RP-059](#rp-059); User Guide [RP-060](#rp-060); palette commands [RP-057](#rp-057); raw markdown [RP-069](#rp-069); copy slips [RP-070](#rp-070).
- **Per-mode memory** [RP-030](#rp-030): the Library also forgets the open item, so add it deliberately. Restore bugs [RP-021](#rp-021) are Roleplay code paths.
- **Transcript view:** button block [RP-050](#rp-050); user turns labelled with the character's name [RP-051](#rp-051).
- **Outside Roleplay:** the nav-bar rollback after "Stay" [RP-061](#rp-061) (shell); the button focus style [RP-065](#rp-065) (design system); the arrival-focus defect [RP-024](#rp-024), which the Library shares (fix it once in the shared frame).
- **Console dependencies (gap round):** the refused first send after Chat now [RP-072](#rp-072), which blocks the redesign's primary action; the hidden recovery card [RP-074](#rp-074); the wrong character in the Console rail [RP-084](#rp-084); a misleading log error [RP-097](#rp-097).
- **What reaches the model (gap round):** inert "Send to Console draft" payloads [RP-047](#rp-047); persona fields that the test chat and Console send differently [RP-077](#rp-077); undisclosed chat contents [RP-082](#rp-082). (Character attachments, [RP-004](#rp-004), are listed under world-info semantics.)
- **The persona editor (gap round):** hidden sections [RP-073](#rp-073); full-screen fields [RP-076](#rp-076) (the frame supplies modes, not field heights); the crash [RP-075](#rp-075); stale tool policy and card [RP-078](#rp-078), [RP-079](#rp-079); scroll on reopen [RP-085](#rp-085); inert Enabled/Mode [RP-086](#rp-086); four commit models [RP-087](#rp-087); endless "Loading" [RP-088](#rp-088); the visuals block [RP-089](#rp-089); the invisible character link [RP-090](#rp-090).
- **Volume (gap round):** the row-1 reset [RP-014](#rp-014); the entry filter and 12-row cap [RP-056](#rp-056) (the frame gives height, not the filter); paging scroll [RP-080](#rp-080); Try-it clipping [RP-081](#rp-081) (fixed only if Try it becomes a full-height mode, D2); table columns [RP-091](#rp-091); the persona 100-cap [RP-093](#rp-093); end-truncated names [RP-094](#rp-094); the tag picker [RP-095](#rp-095); reordering [RP-096](#rp-096). The page bar [RP-092](#rp-092) is partly frame work.

---

## 9. Decisions you need to make

### D1. Where do the Inspector's contents go?
The Inspector holds Chat now / Send to Console draft, readiness, validation, the character's chats (ADR-120), exports, Delete and four Buddy buttons. The Library has no right-hand inspector.
- **A. Fold it into the work pane.** Chat now plus a readiness word in the work-pane header; "More actions" for Send, Export, Duplicate and Delete; validation and "what's in play" in an Info mode; chats in a Chats mode; Buddy as a rail row.
- **B. Stack it under the work pane below 120 cols** (ADR-148 precedent, still "Proposed").
- **C. Keep it as a fourth pane.**

**Recommendation: A.** It frees 30-39 columns at 120-160, puts Chat now next to the content (fixes [RP-019](#rp-019)), and ends the mode-blind noise ([RP-018](#rp-018)). B costs rows, which is Roleplay's scarcer dimension: 17 of 36 rows are already chrome at 120x36. C does not fit the resolver (it models two optional panes) and would be collapsed by default below ~200 columns, the same hidden-by-default failure ADR-148 rejected. Cost of A: re-pin about 60 Inspector tests. Keep the widget ids (`#personas-start-chat`, `#personas-attach-to-console`, `#personas-conversations-list`).

### D2. Where do the test chat and Try-it live?
Today the test chat sits on top of the card (max 60%, controls clipped) and Try-it is a fixed block under the entries. The 2026-07-13 north-star spec put Try-it in a right-hand pane.
- **A. A right-hand "Try" pane** (the resolver's third optional pane), falling back to a mode at small widths.
- **B. A "Try" work-pane mode** ("Card | Edit | Test chat"; "Entries | Settings | Try it") at every size.
- **C. A bounded bottom band** with its input pinned.

**Recommendation: B at the medium sizes, plus A as an extra column only at 200+ columns.** B uses the full pane height with the input docked at the bottom, fixes [RP-010](#rp-010) and [RP-006](#rp-006), and needs no resolver change. Side-by-side edit-and-test is valuable, and the large sizes have spare width for it ([RP-017](#rp-017)). C is today's failure shape. Constraint: compose the preview once in its final container (ADR-115: Textual cannot move a widget without remounting it) and keep the saved turns. **The gap round confirms B for Try it:** with a realistic 3-line sample, a 60%-capped box that cannot scroll hides the result summary at 160x45 ([RP-081](#rp-081)). In the Try mode put the summary first and give the output its own scroll.

### D3. Persona wording: you already ruled personas are assistant-side. How far should the correction go?
You ruled (TASK-617; ADR-037; ADR-149) that a persona is an assistant-side profile. The screen still says "Personas - who you play in the chat", the nav tooltip says "user profiles", and TASK-19577 plus one north-star line still encode the old reading.
- **A.** Fix the screen copy only.
- **B.** Fix the screen copy, nav tooltip and User Guide; close TASK-19577's persona criterion as superseded; annotate the north-star line.
- **C.** B, plus a small "You" row in the rail that shows or deep-links to the Settings display name used for {{user}}.

**Recommendation (updated after the gap round): B + C**, plus making the test chat use your Settings display name instead of the hard-coded "User" ([RP-045](#rp-045)). Suggested gloss: "Personas - reusable AI roles". Why: the contradicting artifacts keep resurfacing (TASK-19577 cites the nav copy as evidence), and the gap round found that the "user profiles" half of J4 has no surface at all: your {{user}} name lives in Settings › Console Behavior and in a collapsed per-chat Console section, and Settings › Roleplay tells you to change user profiles in Roleplay, which cannot ([RP-011](#rp-011)). So C is no longer optional. Keep it narrow, so it respects ADR-037/046: the "You" row shows the effective name and where it comes from (global or per chat), deep-links to Settings › Console Behavior, and is never a persona or a user-profile editor. Also fix the Settings › Roleplay contract copy and the nav tooltip (TASK-496, [RP-057](#rp-057)).

**One more ruling needed:** should Chatbook offer a *user description* for roleplay (what SillyTavern calls a persona description)? Today none exists, and {{persona}} deliberately resolves to the AI's name (task-442). If yes, it belongs in the "You" row as a new, explicitly user-side field, not in Personas. If no, gloss the row "your name in chats" and describe J4 as "Personas (AI roles) + You (your chat name)".

### D4. Reuse the Library shell, or copy it? (ADR-086 amendment)
- **A. Amend ADR-086, or write a new ADR, to promote the pure width resolver, the grip widget and the persistence normaliser to shared code** with neutral CSS names and Roleplay-specific width settings.
- **B. Copy the shell into Roleplay locally.**
- **C. Keep Roleplay's own fraction layout** and only fix chrome and toolbar.

**Recommendation: A.** `design-language.md:135-136` says shared geometry "promotes on the second consumer", and Roleplay is that consumer. The resolver is already a pure, config-safe leaf. B duplicates the copy-paste debt the Library already carries (map-library section 7.5). C leaves [RP-017](#rp-017) unfixed. In the same ADR, record: the grip exception to "never a sliver"; that `library_screen.py` glue is not reused (template: `LibraryArtifactsController`); and the screenshot-approved slice order (header + rail, then resolver + grips + persistence, then work-pane modes + Inspector fold, then the Dictionaries/Lore restructure). Watch the boot broad-selector budget (274/274) and the in-flight PR #2862, which touches the same files.

### D5. Should pane layout and per-mode position be remembered?
- **A.** Keep per-visit behaviour (today; documented in the guide).
- **B.** Add a `[roleplay.reader]` config section that stores only your requested choices (nav open, per-mode list open and width), never the automatic responsive state (ADR-086 split), plus per-mode memory of selection, search, sort and scroll.
- **C.** Share `[library.reader]`.

**Recommendation: B.** It follows the Library's proven contract. The route is rebuilt every visit, so preferences must come from config. ADR-086 scopes `[library.reader]` to Library. Per-mode memory fixes [RP-030](#rp-030), which the frame does not provide. Note: PERF-22 (TASK-33281) may make the route reusable, which would change where this state lives. **Gap round addition:** per selected lore book or dictionary, also store the active tab, the selected entry, the entries anchor and the entry filter, and restore them after every entry action. That last part is [RP-014](#rp-014)'s fix and must land first; without it the entry is lost on every save, mode switch or not.

### D6. How should lore books and dictionaries attached to a character behave?
Today attaching makes a silent copy that never refreshes ([RP-012](#rp-012)) and is never applied in Console chats: the gap round **confirmed this live** with logged prompts, while lore attached to the conversation does reach the model ([RP-004](#rp-004)). The P2f spec chose copies for portability.
- **A. Keep copies**, apply them in character-bound Console chats, and add "Update copy" plus "Used by".
- **B. Switch to live links** (as conversations already work), produce portable copies only at export, and keep card-embedded lore as an editable book.
- **C. Make Chat now link the character's attachments onto the new conversation.**

**Recommendation (updated): rule now. Ship C as the interim fix and adopt B as the target.** C reuses the one mechanism shown to work end to end and needs no new send-path code. B needs the session's character id passed into `capture_prompt_transform_inputs` (`console_chat_controller.py:1273-1297`) and the two runtime appliers (`Chat/console_runtime.py:567-574`, `:630`); the session already knows its character, so this is feasible. Live links remove the whole "which copy is current?" class, match how conversation attachments and other tools (SillyTavern) work, and portability can still be produced at export time. Until B or C ships, the "Used by" view must say "Not applied in chats" for character copies. Choose A instead only if self-contained characters are a hard requirement. High confidence that something must change; medium confidence on B versus A, which weighs your portability goal against stale-copy confusion.

### D7. Header: inline DestinationHeader, or drop it like the Library?
- **A.** A one-row inline DestinationHeader (as Console and Lab ship), which stays within ADR-015 and DESIGN.md:107.
- **B.** Drop it like the Library (amend ADR-015 and DESIGN.md:107).

**Recommendation: A.** Same row budget, no governance change, and it keeps a slot for the unsaved-changes signal ([RP-038](#rp-038)).

### D8. Where should c/p/d/l and [ ] work?
- **A.** Screen-wide from any non-text focus (today).
- **B.** Only from the list, rail and mode controls, with a focus-aware footer.
- **C.** Only from rail headers.

**Recommendation: B.** It keeps the fast keys (ADR-152) while stopping misfires from card text and buttons ([RP-033](#rp-033)). Check whether ADR-152's wording ("list/button focus") needs a one-line clarification. C changes the ADR-152 contract.

### D9. Two small rule calls
- **Ctrl+S.** ADR-031 rule 2 bans it, yet Roleplay and the Library both bind it. Either amend ADR-031 to allow editor-scoped Ctrl+S as an alias, or remove it in favour of a visible Save plus a status line. *Recommendation:* amend. Both destinations rely on it, and Textual's raw mode makes it safe in-app. Keep the visible Save.
- **Invalid regex: warn or block?** The current design is "warn-not-block" (P1c). *Recommendation:* keep warning but make it visible (inline under Pattern, plus "Validation: 1 warning", [RP-015](#rp-015)). Block only if users keep shipping broken rules.

### D10. Button focus style (app-wide)
DESIGN.md:226 asks for a heavy accent outline; the shipped `_buttons.tcss` deliberately uses fill plus underline (TASK-33003.6), giving a 1.06:1 change on the primary button ([RP-065](#rp-065)). *Recommendation:* a design-system ruling. Either amend DESIGN.md or add a 3:1 focus change. Outside the Roleplay redesign's scope.

### D11. Arrival: auto-select the first character, or a landing view?
- **A.** Keep F-031's auto-select (opens "Ada", alphabetically first).
- **B.** A Roleplay landing in the work pane: recent character chats to resume, Import card, New character, needs attention.

**Recommendation: B, if it keeps a non-void first paint with a live primary action** (the reason F-031 exists). It serves J1 better than an arbitrary character ([RP-055](#rp-055)). Decide whether it shows every visit; the Library shows its landing only once per launch.

### D12. Dictionary/lore editors: where does the entry list go?
- **A.** Entries list and entry editor side by side (or stacked) inside the work pane; books stay in the items list.
- **B.** Each book as a rail sub-row, entries as the items list.
- **C.** Drill-in (book, then entries, with a back control).

**Recommendation: A, confirmed by the volume check.** B looks neat with 3 books but with 43 books it turns the rail into a 50-plus-row scroller and buries Cast and World info. C costs a navigation step per book. What the entries list needs: a "Filter entries…" box over keys and content ([RP-056](#rp-056)); no 12-row cap (at least 15-20 rows at 160x45); fixed-width columns (on/off word, first key, content filling the rest, then order; [RP-091](#rp-091)); selection anchored to the entry across every action ([RP-014](#rp-014)); a position count ("#200 of 250 · showing 3 of 250"); and "+ New entry" that clears the form. Placement: list and editor side by side when the work pane has 100+ columns (220x55, or 160 with work-first); narrower, fall back to C (Enter opens the editor full-pane, Escape returns to the same row). Try it is its own mode, not a share of the Entries tab.

### D13. Gate the redesigned Chat now on the Console send fix?
On a default install the first message of every Chat-now chat is refused before it reaches the model, with no recovery ([RP-072](#rp-072), [RP-074](#rp-074)). The fix is Console work (TASK-33621.2; PR #2882 / TASK-32951, not on dev).
- **A.** Make TASK-33621.2 a hard dependency of the new Chat now, verified by a Roleplay end-to-end test (a character with a greeting and a persona, real send path, no stubbed builder, asserting the provider received exactly one request).
- **B.** Ship the redesign regardless and document the known issue.
- **C.** Work around it from Roleplay, for example by starting Chat-now sessions with exchange capture off.

**Recommendation: A, and ship the recovery card ([RP-074](#rp-074)) immediately.** B ships a primary action that fails every time. C changes trace-capture behaviour, a Console and privacy decision, from the wrong layer. RP-074 is effort S and turns the dead end into one extra click (Send without capture) while A waits.

### D14. Long lists: paging, or one virtual list?
With 348 characters the list pages at 50. The page boundary causes [RP-080](#rp-080) (the next page opens scrolled to its bottom), [RP-039](#rp-039) (marks lost) and [RP-092](#rp-092) (a wrapping pager with no keys), and 50 two-line rows still need 3-4 screens per page.
- **A.** One line-based virtual list for all items (Textual OptionList or the Line API rather than a widget per row), loading in chunks of about 200 if needed, with the count in the items header ("Characters 348", "85 of 348 match").
- **B.** Keep paging but fix it: a one-row, non-wrapping pager in the items header ("‹ 2/7 ›"), scroll to the top on every page change, PgUp/PgDn at the list edges, marks stored by id, and a cue when the selection is on another page.
- **C.** Bigger pages.

**Recommendation: A.** It removes three defects instead of patching them, and a line-based list also addresses PERF-22's "remounts every row". B is the fallback if A cannot land with the frame. Either way, fix the persona 100-cap first ([RP-093](#rp-093)): it truncates data before any list sees it. Also default to one-line rows once a list passes about 40 items or the column is under 40 characters, with middle-ellipsis names ([RP-094](#rp-094)).

### D15. What does a persona send to the model, and should persona chats run as a tool agent?
Today a Console persona chat sends only the System prompt; the test chat sends the Description; Personality traits go nowhere ([RP-077](#rp-077)). Persona Chat now runs the Console agent with 53 tools, consistent with ADR-037/149 but never disclosed ([RP-082](#rp-082)).
- **Fields A.** One shared composer: System prompt + Description + Personality traits, used by both the test chat and Console.
- **Fields B.** Keep System-prompt-only (possibly tldw_server parity, TASK-617) and label Description and traits "Notes - not sent to the model"; the test chat sends the same thing.
- **Tools (i).** Keep the agent, and say so under Chat now ("Console agent · 53 tools · Tool policy: none").
- **Tools (ii).** Start persona sessions with tools off unless the persona's policy allows some.

**Recommendation: check TASK-617 parity first; if tldw_server sends only the system prompt, choose Fields B, otherwise Fields A; and Tools (i).** Either way the defect is that the preview and Console disagree, so both must use one seed function. Medium confidence: this is a product call on what a persona is for.

---

## 10. Library defects not to copy

1. **Arrival focus on the nav bar's Home** (shared defect; fix once in the frame, [RP-024](#rp-024)).
2. **Weak button focus** (underline plus a small shade change, [RP-065](#rp-065)).
3. **Default width profiles that starve editors** at 120-160 cols (work pane 50/45/62 cols). The Notes editor at 120x36 shows a 4-row body with its toolbar clipped and Save hidden.
4. **The rail vanishes below ~126 cols on reader routes**, and the header then does not say where you are.
5. **Two collapse grammars:** a 3-cell, non-persisted handle and persisted 5-cell grips (the guide claims five cells for both). Grips reserve 10 columns even when both panes are open.
6. **Escape is overloaded:** 15-18 bindings resolved by declaration order, and the roster keeps going stale.
7. **Per-destination glue is copy-pasted** (pane-toggle if-ladder, static F6 tuples, emergency geometry with hard-coded ids) in a 35.9k-line screen that is over its size ratchet.
8. **Three border layers** (grid, rail, canvas), about 8 columns at 120 and wider.
9. **Inconsistent unsaved-work handling:** Notes autosave, while Prompts/Skills hard-veto Back with a toast (no in-place Save); delete confirmation is inline in one place and modal in another.
10. **It forgets the open item across destinations**, and the landing appears only on the first visit per launch.
11. **Heavy items-pane chrome:** Conversations and Prompts spend 8 rows before the first item; Notes ~17-19 at 120x36 (a 4-row explanatory paragraph, filter, New/Sort/Select, "Add from files.../Export", a 3-row "Folders & placement" block).
12. **3 rows per list item** (title, meta, blank separator).
13. **Ctrl+S bound for skill save** (an ADR-031 rule 2 violation).
14. **A catch-all "Details" rail bucket**, and gloss-fitting heuristics that hide some glosses at default width.
15. **Conversation rows do not name the character** ("2 messages · 1m · Default · Active").
16. **Uncapped reader prose** (~108 chars per line at 220 cols); only the landing is capped.
17. **Small-size quirks:** at 80x24 the list collapses into one wrapped run-on line; at 60x24 the reader clips text ("Resume conversatio"); a stray "S" glyph under the Notes grip.
18. **Open shell debt:** glyph legend overload (TASK-32235 / 32309), `apply_pane_width` is Notes-only (TASK-32576), recompose instead of targeted updates (TASK-281), UI test debt (TASK-31249), landing footer key story (TASK-2520).
19. **A 3-row pager with no paging keys** (Conversations: "1-20 of 36 · Page 1 of 2", a reason line, Previous/Next). Copy its wording and its scroll reset, not the stack ([RP-092](#rp-092)).
20. **Unchecked: toast markup.** Roleplay's crash on an over-long persona name ([RP-075](#rp-075)) comes from passing exception text to a toast with markup on. Check the Library's notify calls for the same default before sharing any wrapper. (The Library's multi-select across pages was not tested either.)

---

## 11. Coverage

After the first round, an audit mapped what the 71 findings had and had not examined. That matrix (11.2, kept as audited) prompted the gap round. 11.1 says what the gap round closed; 11.3 lists what is still not covered.

### 11.1 What the gap round closed

| Gap in the 11.2 matrix | Now | Where |
|---|---|---|
| J1: the first Console send after Chat now never succeeded (note A) | Root-caused. A Console trace-provenance bug refuses every Chat-now first send on the default config (4 of 4; 0 lines reached a body-logging mock). With `[console] exchange_capture = false` all three sends completed. The `capture_provider_failure` error in note A is a separate, harmless bug, and the leaked staged source did not block. | [RP-072](#rp-072), [RP-074](#rp-074), [RP-097](#rp-097), [RP-047](#rp-047), C.7 |
| RP-004 was code-reading only | Confirmed live with logged prompt bodies; conversation-level attach works | [RP-004](#rp-004), C.7 |
| The persona editor was never opened | Opened, edited, saved and created (plain and Actor Pack) at 120x36, 160x45 and 220x55, with Tab-by-Tab walks; unsaved state and the leave guard checked | [RP-073](#rp-073), [RP-075](#rp-075)-[RP-079](#rp-079), [RP-085](#rp-085)-[RP-090](#rp-090), C.8 |
| Personas at 120x36 (1 capture) and 220x55 (no editor) | +10 capture sets at 120x36, +4 at 220x55 | C.8 |
| The "user profiles" half of J4 | Traced end to end: no surface in Roleplay, and Settings points back to Roleplay | [RP-011](#rp-011), D3 |
| Volume (page bar never rendered; books of ≤15 / ≤10 entries) | 348 characters, 64 and 114 personas, 43 lore books (one with 250 entries), 28 dictionaries (one with 150 rules), 30 chats for one character | [RP-014](#rp-014), [RP-056](#rp-056), [RP-080](#rp-080), [RP-081](#rp-081), [RP-091](#rp-091)-[RP-096](#rp-096), C.9 |
| Search no-match in Personas | Tested: "zzqx" gives "No personas yet" with the count unchanged | [RP-022](#rp-022) |
| Capture-report #13 not re-measured in a clean session | A clean 160x45 Lore visit shows 10 entry rows; the 3-row window needs the RP-001 band | [RP-056](#rp-056) |
| Open dependency: D6 waits on RP-004 | Unblocked; D6 updated | D6 |
| Open dependency: no decision on volume behaviour | Added | D14 |
| Open dependency: the last step of J1 never observed | Observed: it fails on a default install and works with capture off | D13 |

**Heuristic tags after the gap round** (97 findings; counted from each finding's principle line): H1 45 · H2 23 · H3 17 · H4 18 · H5 10 · H6 16 · H7 10 · H8 14 · H9 10 · H10 5 · WCAG 11. H5 doubled and H10 grew, but H10 remains the thinnest.

**Capture sets added by the gap round** (124 unique, classified by name): Characters 8 at 120x36 and 42 at 160x45 (18 of them Chat-now and hand-off states, the rest volume states); Personas 10 / 19 / 4 at 120x36 / 160x45 / 220x55; Dictionaries 7 at 160x45 and 2 at 220x55; Lore 1 / 20 / 3; Library 5 at 160x45; Settings 2 at 220x55; one plain Console send.

### 11.2 Coverage matrix before the gap round (as audited; report.md and findings.json, 71 findings)

Legend: ✓ = covered with live or capture evidence · ~ = partial (code-only, one size only, or incidental) · ✗ = not examined

#### 1. Mode × core job

| Mode | J1 import & chat fast | J2 in-depth authoring | J3 world info | J4 personas / user profiles |
|---|---|---|---|---|
| Characters | ✓ import of PNG, V2 JSON and a card with embedded lore (flows J2, RP-013/058). Chat now opens Console (✓). **✗ the first Console send after Chat now never succeeded in any run** (see note A) | ✓ flows J1/J3, editor captured at 120/160/220, RP-002/016/023/053/054 | ✓ attach sections (RP-012/014, J5/J6). **~ whether attached lore and dictionaries actually reach the chat (RP-004, P0) is code-reading only, "confirm live"** | n/a |
| Personas | ~ Chat now opens Console. All 3 persona sends that were attempted were blocked; never root-caused, filed as "cross-surface" | **✗ the persona editor was never opened in any capture or journey** (create, edit, Save, Required-portrait select, visual states, policy rules, Mode/Enabled). RP-003 and RP-016 cite it from code only | n/a | ✓ copy and semantics (RP-011, D3, ADR-037/046/149). ✗ persona create/edit lifecycle. ~ the "user profiles" half (the {{user}} identity) appears only as the optional D3-C "You" row |
| Dictionaries | ✓ import errors (RP-058) | n/a | ✓ J5 end to end, RP-003/005/006/015/040 (seeded books have ≤15 entries) | n/a |
| Lore | ✓ import errors and wrong shape (RP-058, RP-013) | n/a | ✓ J6 end to end, RP-003/005/006/014 (seeded books have ≤10 entries) | n/a |

#### 2. Mode × size (unique capture sets, Roleplay modes classified by name)

| | 60/80 cols | 120x36 | 160x45 | 220x55 |
|---|---|---|---|---|
| Characters | 7 | 17 | 81 | 8 (no test chat at 220) |
| Personas | 1 | **1** (selected only) | 16 | 3 (no editor at any size) |
| Dictionaries | 2 | 6 | 35 | 5 |
| Lore | 3 | 5 | 29 | 4 |
| Library | 5 | 8 | 24 | **3** (landing, conversation open, empty landing; no editor or list-edit state at 220) |

Most of the evidence is at 160x45. 120x36 is moderate. 220x55 (the owner's LARGE target) is light, and the Library side at 220 is computed by the resolver rather than observed.

#### 3. State × mode

| State | Characters | Personas | Dictionaries | Lore |
|---|---|---|---|---|
| Arrival | ✓ all sizes (F-031 auto-select; D11) | ✓ | ✓ | ✓ |
| Empty | ✓ (3 built-ins remain) | ✓ | ✓ | ✓ |
| Loading | ~ count "· 0" only | ~ | ✓ RP-028 | ✓ RP-028 |
| Selected | ✓ | ✓ | ✓ | ✓ |
| Editing | ✓ 120/160/220 | **✗** | ✓ | ✓ |
| Unsaved | ✓ RP-037/038, leave guard | **✗** | ~ RP-040 | ~ RP-040 |
| Error / blocked | ✓ provider down (RP-059), bad PNG, empty name | ~ provider blocked; the Console block was not analysed | ✓ invalid regex (RP-015) | ✓ bad and wrong-shape JSON |
| Long content | ✓ 1.2k description. **✗ volume: 28 cards is under the 50-row page size, so the page bar never rendered** | ✗ | **✗** ≤15 entries | **✗** ≤10 entries; no large books |
| Search no-match | ✓ RP-022 | ~ asserted "all" | ~ | ~ |
| Test chat / Try it | ✓ 120/160 (RP-010/045/046); ✗ 220 | ✓ RP-045 | ✓ all 4 sizes | ✓ 120/160 |
| Hand-off result in Console | **✗ 0 of 4 Chat-now sends completed** | **✗** | n/a | n/a |

#### 4. Channel × mode

| | Characters | Personas | Dictionaries | Lore | Library |
|---|---|---|---|---|---|
| Mouse (click, wheel) | ✓ | ~ | ✓ | ✓ | ✓ 120/160 |
| Keyboard | ✓ a11y lens C.3 | ~ mode switch only | ✓ entry form and Space toggle | ~ settings focus | ✓ F1, grips, rail |

#### 5. Heuristics (findings tagged per heuristic)

H1 26 · H2 15 · H3 14 · H4 14 · H5 5 · H6 11 · H7 6 · H8 9 · H9 6 · H10 3 · HCI layout ~15 (6 P0) · WCAG 8. All ten heuristics are scored with evidence on both sides. H10 and H5 are thin but adequate for the current scope.

#### 6. The 17 capture-report observations

| # | Disposition |
|---|---|
| 1 | Adopted as RP-002 |
| 2 | Adopted as RP-009 |
| 3 | Adopted as RP-022 |
| 4 | Adopted as RP-028 |
| 5 | Adopted as RP-033 |
| 6 | Adopted as RP-010; the greeting picker is folded in |
| 7 | Adopted as RP-021 and D11 |
| 8 | Adopted as RP-006, re-attributed in part to RP-001 |
| 9 | Adopted as RP-008 and RP-043 |
| 10 | Adopted as RP-018 |
| 11 | Adopted as RP-032 |
| 12 | Adopted as RP-054 |
| 13 | Re-attributed to RP-001 (C.1 caveat), but not re-measured in a clean session |
| 14 | Partly adopted: "(soon)" switch (RP-006/055), raw markdown (RP-069). Not dispositioned: the 80x24 lore list opening scrolled past its first row, and the 120-col "Search this" truncation (small-size and cosmetic, not material) |
| 15 | Adopted as RP-060 |
| 16 | Adopted in §10 |
| 17 | Splash-screen log error not dispositioned (out of scope, not material) |

All material observations are accounted for.

#### 7. Library-side comparison claims

- Evidenced at 120 and 160 by captures, one analog flow (prompt edit in 5 actions) and Library keyboard captures.
- Resolver widths are pure-function arithmetic, not observed.
- Library at 220 has no edit-state capture.
- Library error copy is admitted to be "less tested".
- The scorecard is labelled indicative.

Adequate for a frame decision, because the redesign borrows structure and §10 lists the Library defects not to copy.

#### 8. Owner decisions

| Owner decision | Coverage |
|---|---|
| (1) Own destination on the Library frame | Covered by D4, D7 and §7 |
| (2) Four equal jobs | The C.5 rail gives each job a row, but J4 evidence is thin and "user profiles" is left as optional polish |
| (3) Target sizes | Metrics at 120, 160 and 220; small sizes rated lower. 220 is light |
| (4) Discover pain points | Done: 71 findings |

Open dependencies:

- D6 depends on RP-004, which has not been confirmed.
- No decision covers volume behaviour (paging versus a virtual list, filtering entries).
- The terminal step of J1, the first send in Console, has never been observed working.

**Note A.** Every Console send from a Chat-now session failed with `controller_submit status=blocked`; the only completed send was the generic preview hand-off (rv-flows-2 log line 438, with tools=53 in the mock log).

- rv-flows log, lines 404-414 and 437-443: blocked after `capture_provider_failure`.
- rv-ia log, lines 419-430: blocked after `capture_provider_failure`.
- rv-flows-2 log, lines 490-501: blocked after "RAG evidence excluded; reason=not_canonical_local_evidence". This came from the staged test-chat source that RP-047, a P2, describes only as "appears in a later chat".

The mock logs only metadata (`mock_llm.py`), so RP-004 could not be confirmed live.

### 11.3 Still not covered

- **Library at 220x55 in an edit or list-edit state.** Its 220 figures are still the width resolver's arithmetic.
- **A character test chat at 220x55.** The gap round ran a persona test chat at 220x55, not a character one.
- **The Library at larger volume.** It was tested at 36 conversations only, so its multi-select across pages (the comparison in [RP-039](#rp-039)) is untested.
- **Keyboard-only journeys at volume.** No full F6/Tab journey through a 250-entry book; Lore keyboard coverage is still incidental (Tab to Delete/Move in the gap round, settings focus before).
- **A real model.** The mock echoes prompts, so how a model weighs a 39-character persona prompt inside an 8,000-character agent prompt ([RP-082](#rp-082)) is inferred. The ~12 s first persona reply is unexplained: either a personal-context bootstrap timing out under the isolated HOME, or the tool-agent shape.
- **Real imported lorebooks.** The 250-entry book and the 150-rule dictionary are synthetic.
- **Small sizes (80x24 and below)** were not revisited, by design.
- **Dictionary entry actions at volume** were verified from code (the same clear-and-reload path as lore, [RP-014](#rp-014)), not driven live.

---

## Appendix A. All findings by priority

A.1 is the index; A.2 lists every evidence reference, the recommendation and the verifier's corrections. Sources are the lens finding ids (NA/NB = NN/g heuristic passes, LY = layout, TF = task flows, KA = keyboard/accessibility, IA = information architecture, LF = Library fit; `missed:` = a verifier note promoted to evidence or a finding; G1/G2/G3 = the gap-round lenses: end-to-end hand-off, Personas follow-up, realistic volume; `verification` = the independent verifier's own reproduction).

### A.1 Index

| ID | Pri | Sev | Title | Heuristic / principle | Area | Kind | Scope | Sizes | Effort | Sources | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [RP-072](#rp-072) | P0 | 4 | On a default install, the first message in every Chat-now chat is refused before it reaches the model, with no reason and no way out (Console defect TASK-33621.2: a blocking dependency, not redesign work) | H1 Visibility of system status; H9 Help users recognize, diagnose and recover from errors; completion of J1 | Chat now -> first Console send in a character- or persona-bound session | bug | external (Console; TASK-33621.2) - blocks the redesign's primary action | all (verified at 160x45 and 120x36) | M | G1-01 | CONFIRMED (live, with full traceback; capture-off control completed) |
| [RP-004](#rp-004) | P0 | 4 | Lore books and dictionaries attached to a character are never applied in Console chats (confirmed live with logged prompts) | H1 Visibility of system status; H2 Match with the real world (an attachment shown as active should act) | Character attachments (World Books / Dictionaries sections) -> Console send path | bug | content | n/a (behaviour) | M | missed:nng-a#1, G1 (gap1 section 1) | CONFIRMED live (gap round G1; originally verifier-sourced from code) |
| [RP-014](#rp-014) | P0 | 4 | After any lore or dictionary entry action the table jumps back to row 1, so the next Delete, Move or Update silently hits entry #1; Delete and Detach have no confirmation or undo | H5 Error prevention; H1 Visibility of system status; H3 User control and freedom; keep the selection stable across a mutation | Lore / Dictionary Entries tab (entry table + Add/Update/Delete/Move); character World Books / Dictionaries sections | bug | content | all; worst at 120x36 and 160x45, where row 1 is usually off-screen | S | NA-06, missed:flows#1, G3-01 | CONFIRMED (reproduced live in the gap round; raised from P1/3 to P0/4) |
| [RP-001](#rp-001) | P0 | 4 | Once a dictionary, lore book or editor has been opened, its empty slot stays on screen and squeezes every later view (the character editor shrinks to a 2-5-row window) | H8 Aesthetic and minimalist design; H1 Visibility of system status; stable layout (DESIGN.md:125) | Work pane: #personas-detail-stack and its four demand-mounted slots | bug | content | all; worst at 120x36 (2-row editor window), 160x45 (5-9-row window) | S | LY-01 | CONFIRMED (reproduced live with a smaller trigger than first reported) |
| [RP-002](#rp-002) | P0 | 4 | The character editor's long-text fields show 0-1 lines of text, so clicks land mid-word and edits are saved corrupted | H1 Visibility of system status; H5 Error prevention; HCI: WYSIWYG direct manipulation | Character editor (Characters > Edit), main and Advanced fields | bug | content | all, including 220x55 (heights are fixed cell counts) | S | TF-01, NA-09, LY-09, NB-03 | CONFIRMED (all four sources) |
| [RP-003](#rp-003) | P0 | 4 | Dictionary and lore entry forms: most controls are pushed off-screen, and the rest have no labels or show raw internal values | H6 Recognition rather than recall (persistent labels); H2 Match with the real world; H1 Visibility | Dictionaries / Lore detail: Entries and Settings forms; persona Mode | bug | content | all, verified at 160x45 and 220x55 (dictionary options unreachable by mouse at every size) | M | NB-01, NA-16, IA-14, TF-11 | CONFIRMED (NB-01 reproduced live; NA-16, IA-14 confirmed) + verifier-sourced additions |
| [RP-073](#rp-073) | P0 | 4 | Three persona-editor sections draw as a title only at every size (tool policy, shared reactions, New Actor Pack's portrait picker), yet Tab still enters them blind and saves what it finds | H1 Visibility of system status; H6 Recognition rather than recall; H5 Error prevention (an invisible pre-selected value is committed); WCAG 2.4.7 Focus visible | Persona editor: #personas-policy-rules-editor, #personas-editor-shared-visual-identity-host, #personas-editor-pack-portrait | bug | content | 120x36, 160x45 and 220x55 (policy verified at all three; pack portrait at 120x36; same CSS cause everywhere) | S | G2-02 | CONFIRMED |
| [RP-005](#rp-005) | P0 | 3 | Lore and dictionary Settings tabs cannot scroll: 'Save settings', Export and Enabled are unreachable by mouse at the target sizes | H1 Visibility of system status; H3 User control and freedom | Dictionary / Lore detail > Settings tab | bug | both | Dictionaries: 120x36, 160x45 (visible at 220x55). Lore: 120x36, 160x45 and 220x55 | S | NA-07, TF-10, IA-08 | CONFIRMED (reproduced live by three verifiers) + verifier-sourced severity notes |
| [RP-006](#rp-006) | P0 | 3 | Dictionary and lore detail starves the entries: 0-5 entry rows at the target sizes while a permanent Try-it block and blank bands take the room | H8 Aesthetic and minimalist design; master-detail (list -> editor) ecosystem convention | Dictionaries / Lore centre pane (Entries tab + Try-it) | usability | both | 120x36 worst (0 entries visible); 160x45 (3-5 rows); 220x55 wasteful | L | NB-06, TF-11, LF-20, LY-15, LF-08 | CONFIRMED / PARTIAL (corrections applied) |
| [RP-007](#rp-007) | P0 | 3 | 13 rows of stacked chrome above the panes and 4 below take 38-47% of the screen height at the medium sizes (31% at 220 cols); the Library spends 16-20% | H8 Aesthetic and minimalist design (chrome-to-content ratio); PRODUCT.md:31 density | Screen frame: DestinationHeader box, purpose line, 'Modes:' strip, workbench frame, pane borders | usability | frame | all; 47% at 120x36, 38% at 160x45, 31% at 220x55, 58% at 80x24 | M | LY-02, NB-07, LF-01, LY-07, IA-20 | CONFIRMED / PARTIAL (governance corrections applied) |
| [RP-008](#rp-008) | P0 | 3 | The Characters rail stacks seven one-per-row buttons above the list, so the first character sits on row 23 at every size (5 of 28 visible at 120x36) | H8 Aesthetic and minimalist design; Hick's law (five similar verbs); Gestalt similarity (filters styled as actions) | Left rail toolbar and filter bar (all modes; worst in Characters) | usability | frame | all, including 220x55; 80x24 shows 1 character | M | LY-03, NB-08, LF-03, LY-21 | CONFIRMED / PARTIAL (scope corrections applied) |
| [RP-009](#rp-009) | P0 | 3 | The search box hides what you type once it has focus, and the whole list jumps down a row | H1 Visibility of system status; stable layout (DESIGN.md:117, :125 - focus never moves layout) | Left rail search | bug | both | Characters and Personas at every measured width; all four modes at 120x36 | S | NA-03, NB-03, LY-10, KA-09, TF-12, LF-03 | CONFIRMED (reproduced live by several verifiers) |
| [RP-010](#rp-010) | P0 | 3 | The test chat's input and buttons clip away: unusable at 120x36, broken after one reply at 160x45 | H3 User control and freedom; H1 Visibility; HCI: controls must not depend on content length | Characters > 'Try a test chat' pane | bug | both | 120x36 unusable from the first paint; 160x45 buttons lost after the first reply | M | LY-14, TF-04, LF-08 | CONFIRMED |
| [RP-011](#rp-011) | P1 | 3 | Personas mode says 'who you play', but a persona is the AI's side, and the 'you' half of J4 has no home: Settings sends 'user profiles' back to Roleplay, which has none | H2 Match between system and the real world; H4 Consistency and standards; H10 Help (circular pointer); information scent | Personas mode descriptor, chip tooltip, nav tooltip, persona card; Settings > Roleplay contract copy; the {{user}} display name | usability | both | all | S | NA-01, TF-17, IA-01, G2-13 | CONFIRMED / PARTIAL (copy defect per the existing ruling) + gap-round trace of the 'user profiles' half |
| [RP-012](#rp-012) | P1 | 3 | Character attachments are silent frozen copies, and the lore/dictionary side says 'Not attached' even when a character holds one | H1 Visibility of system status; H6 Recognition rather than recall; PRODUCT.md:25 consequences without guesswork | Character World Books / Dictionaries sections; Lore and Dictionary 'Attachments' tabs | usability | both | all | M | NA-02, TF-02, IA-05, NB-04, TF-09 | CONFIRMED / PARTIAL (corrections applied) |
| [RP-013](#rp-013) | P1 | 3 | A lorebook embedded in an imported card is invisible in Lore mode and cannot be viewed, edited or tested | H6 Recognition rather than recall; H4 Consistency (one kind of object, one home) | Import > Lore mode | gap | content | all | M | TF-03 | CONFIRMED |
| [RP-015](#rp-015) | P1 | 3 | An invalid regex is saved silently; its warning list renders with zero rows while the Inspector says 'Validation: OK' | H9 Help users recognize, diagnose, and recover from errors | Dictionaries > Entries > Add/Update, advisory list, Inspector validation line | bug | content | all | S | NB-02 | CONFIRMED (reproduced live) |
| [RP-016](#rp-016) | P1 | 3 | The character editor is one long scroll of small fixed-height fields with no sections, so 2.5 fields fit at 120x36 and a wider canvas would not help | Progressive disclosure; H8 Aesthetic and minimalist design; form scanning | Character editor (and persona editor) | usability | content | 120x36 worst (2.5 fields), 160x45 (5 fields); still fixed heights at 220x55 | M | LY-11, LF-06, missed:nng-a#3 | CONFIRMED + verifier-sourced evidence |
| [RP-017](#rp-017) | P1 | 3 | Width goes to the side panes, not to what is open: a fixed 2:4:2 split, no collapse priority, no 'work-first' while editing, and blank panes at large sizes | HCI: allocate space to the primary task; H7 Flexibility and efficiency; readable line length | Pane width model | usability | both | 120x36-160x45 (editor ~half the width); 200+ (blank Inspector, 100-185-char lines) | L | LF-04, LF-05, LY-06, LY-17 | CONFIRMED / PARTIAL (LY-06 rated 2 on its own) |
| [RP-018](#rp-018) | P1 | 3 | The Inspector column mixes per-item actions, app-wide Buddy controls and wrong-mode advice in ten equal one-row buttons | H8 Aesthetic and minimalist design; Hick's law; Gestalt proximity; visibility of the primary action | Inspector (right column) | usability | both | all; width cost at 120-160, blank column at 200+ | L | LF-07, NB-10, IA-10, LY-12, KA-14, NA-22 | CONFIRMED / PARTIAL (corrections applied) |
| [RP-019](#rp-019) | P1 | 3 | At 120x36, 'Chat now' falls below the Inspector's fold whenever the character has a portrait, with no 'more below' cue | H1 Visibility; F-pattern (primary action in the least-scanned spot); Fitts's law | Inspector primary call to action | usability | both | 120x36 (hidden); 160x45 (Buddy hidden); <=100x30 hidden for every character | S | LY-13 | CONFIRMED (reproduced live) |
| [RP-020](#rp-020) | P1 | 3 | Per-item actions live in three different places and move by mode; dictionary and lore Export sit at the bottom of an unscrollable tab | H4 Consistency and standards; H6 Recognition (where is the action?) | Duplicate (rail), Chat/Export/Delete (Inspector), Edit/Conversations (card bottom), dictionary/lore Export (Settings tab) | usability | both | all; dictionary/lore Export unreachable at <=160x45 | M | IA-08, LF-17 | CONFIRMED (reproduced live) |
| [RP-021](#rp-021) | P1 | 3 | Returning to Roleplay restores a broken state: an empty list and '· 0' when nothing was selected, or a hidden leftover search filter | H1 Visibility of system status; H9 Help users recover (it looks like data loss) | Screen restore on leave/return | bug | both | all | S | TF-07, TF-12 | CONFIRMED (both halves reproduced live by the verifier) |
| [RP-022](#rp-022) | P1 | 3 | A search with no matches says 'No characters yet - use New or Import', and filtered counts never say 'of 28' | H9 Help users recognize and recover from errors; H5 Error prevention; H1 | Rail list empty/no-match state and count line | bug | both | all | S | NA-04, NB-16, IA-07, LF-14, TF-12, G3-11 | CONFIRMED |
| [RP-023](#rp-023) | P1 | 3 | Example dialogue (mes_example) is stored and sent to the model, but cannot be seen or edited anywhere | H2 Match with the real world (card-format vocabulary); completeness of the authoring model | Character card and editor | gap | content | all | S | TF-15 | CONFIRMED |
| [RP-024](#rp-024) | P1 | 3 | Keyboard focus escapes the screen: arrival focus sits on the nav bar's '⌃1 Home' (Enter leaves Roleplay), and Tab walks 14 nav buttons on every cycle | H3 User control and freedom; H4 Consistency; WCAG 2.4.3 Focus Order | Frame: arrival focus and Tab traversal | bug | frame | all | S | KA-01, KA-02, missed:a11y#4 | CONFIRMED (both reproduced live) |
| [RP-025](#rp-025) | P1 | 3 | F6 (next pane) lands on secondary controls and never reaches the dictionary or lore editors | H7 Flexibility and efficiency; WCAG 2.1.1 Keyboard; DESIGN.md:117 keyboard-first | F6 pane cycle | bug | both | all | M | KA-03 | CONFIRMED |
| [RP-026](#rp-026) | P1 | 3 | Escape does not leave text fields, so the next advertised mode key is typed into the field | H3 User control and freedom (clearly marked exit); H4 Consistency with Library; WCAG 2.1.2 (soft trap) | Escape semantics | gap | both | all | S | KA-04, NA-11, missed:a11y#1 | CONFIRMED (reproduced live) |
| [RP-027](#rp-027) | P1 | 3 | Tab stops are invisible or off-screen: nested scroll containers, a blank greeting picker, entry-form controls scrolled out of view, and a Generate button before every field | WCAG 2.4.7 Focus Visible and 2.4.11 Focus Not Obscured; H1 Visibility | Work pane content (card, test chat, dictionary/lore forms, character editor) | bug | content | all; worse at 120x36 | M | KA-07, TF-20 | PARTIAL (corrections applied) |
| [RP-047](#rp-047) | P1 | 3 | Both 'Send to Console draft' hand-offs are inert: the card or test-chat transcript never reaches the model, yet stays shown as staged and follows you into later chats | H1 Visibility of system status; H2 Match (say what will happen); H4 Consistency (one label, several meanings); H5 Error prevention | Inspector 'Send to Console draft' (ctrl+enter) and test-chat 'Send to Console draft' -> Console staged sources; hand-off verbs | bug | content | all | M | TF-06, IA-12, G1-04 | CONFIRMED (gap round, logged prompt bodies; raised from P2/2 to P1/3) |
| [RP-056](#rp-056) | P1 | 3 | Lore and dictionary entries cannot be found at realistic volume: search matches book names only, entry tables have no filter and stop at 12 rows (entry #200 of 250 took 99 wheel notches) | H7 Flexibility and efficiency of use; H6 Recognition rather than recall; scroll cost | Rail search; Lore and Dictionaries > Entries tab (no entry filter; table capped at 12 rows) | gap | both | all: 4 entry rows at 120x36; 10 at 160x45 on a clean visit (3 with the RP-001 band); 10 at 220x55 (the cap does not grow with height) | M | NB-14, LF-21, G3-02 | CONFIRMED at realistic volume (gap round G3-02; raised from P2/2 to P1/3) |
| [RP-074](#rp-074) | P1 | 3 | After a refused send, the Console's recovery card never appears, so 'Send without capture', 'Retry' and 'Cancel' are unreachable and the chat is a dead end (TASK-33621.2 AC#2/#3) | H9 Help users recognize, diagnose and recover from errors; H3 User control and freedom | Console transcript recovery card after a refused Chat-now send | bug | external (Console; TASK-33621.2 AC#2/#3) | all | S | G1-02 | PARTIAL (rated 3, not 4: same incident as RP-072) |
| [RP-075](#rp-075) | P1 | 3 | Saving a persona whose name is over 200 characters crashes the whole app, losing every unsaved draft | H9 Help users recognize, diagnose and recover from errors; H5 Error prevention (unconstrained input); stability | Persona editor Save (create and edit) -> error toast (PersonasScreen._notify) | bug | content | all (reproduced at 160x45 and 120x36) | S | G2-01 | PARTIAL (reproduced live; rated 3 for the rare trigger) |
| [RP-076](#rp-076) | P1 | 3 | Each long persona field is a full-screen box, so one field is visible at a time and Mode/Enabled sit 3-4 screens down (the inverse of RP-002) | H8 Aesthetic and minimalist design; progressive disclosure; form scanning (related fields seen together) | Persona editor: Description, System prompt, Personality traits | usability | content | all: one field per screen at 120x36, 160x45 and 220x55 | S | G2-03 | CONFIRMED |
| [RP-077](#rp-077) | P1 | 3 | A persona's Description and Personality traits never reach the model in Console, but the test chat sends the Description, so the preview and the real chat disagree | H1 Visibility of system status (truthful feedback); H2 Match (fields imply effect); preview/production consistency | Persona prompt composition: test chat vs Console persona session vs Chat now | bug | content | all (behaviour) | M | G2-04 | CONFIRMED |
| [RP-078](#rp-078) | P1 | 3 | The Inspector's 'Tool policy' line shows the last-edited persona's rules on other personas, and 'no rules' for personas that have rules | H1 Visibility of system status (false status on a permissions setting) | Inspector: #personas-policy-rules-summary on persona selection | bug | content | all | S | G2-05 | CONFIRMED |
| [RP-079](#rp-079) | P1 | 3 | After a persona Save then Cancel, the card still shows the previous persona (or the pre-edit text) while the Inspector names the saved one; Edit then fails with 'Selection out of sync' | H1 Visibility of system status; H4 Consistency (one selection, shown everywhere); H5 Error prevention (actions target an unseen item) | Persona create/edit -> Save -> Cancel; persona card vs Inspector | bug | content | all (reproduced at 120x36 and 160x45) | S | G2-06 | CONFIRMED (wider than first reported) |
| [RP-080](#rp-080) | P1 | 3 | Changing page keeps the old scroll offset: after scrolling to the end of one page, the next page opens at its bottom and about 41 characters are never shown | H1 Visibility of system status; predictable navigation (a new page starts at its top) | Characters/Personas list paging (page bar) | bug | content | all where a page holds more items than the viewport (every target size at 50 per page) | S | G3-04 | CONFIRMED |
| [RP-081](#rp-081) | P1 | 3 | Try it at realistic volume: at 160x45 the 'what fired / what was dropped' summary and per-rule lines are clipped off, and the summary calls budget-dropped lore 'near-miss' | H1 Visibility of system status; H2 Match between system and the real world (why lore was dropped) | Dictionary and Lore Try-it output (summary + fired/skipped lists) | usability | both | 160x45 (summary invisible), 120x36 (worse), 220x55 (summary visible, 1-7 detail rows) | M | G3-09 | PARTIAL (vocabulary claim narrowed) |
| [RP-028](#rp-028) | P2 | 2 | Counts read '· 0' next to a populated list (Dictionaries and Lore until something is selected), with no loading marker | H1 Visibility of system status | Purpose/count line | bug | both | all | S | NA-10, TF-19, LF-14, IA-07 | CONFIRMED |
| [RP-029](#rp-029) | P2 | 2 | The four jobs are flat mode chips: only the active mode shows a count and a plain-language gloss | H6 Recognition rather than recall; H1 Visibility; information scent | Mode navigation (chip strip) | gap | frame | all | M | LY-05, LF-02, IA-06 | CONFIRMED / PARTIAL (governance correction) |
| [RP-030](#rp-030) | P2 | 2 | Switching modes throws away selection, search, sort, tag, page, marks and the test-chat transcript | H3 User control and freedom; H6 Recognition rather than recall; preserve context across navigation | Mode switching | usability | both | all | M | NA-20, NB-05, TF-08, LF-09, G3-12 | PARTIAL (by design but harmful; rated 2) |
| [RP-031](#rp-031) | P2 | 2 | Collapsed panes reopen on every visit: there is no remembered (preferred) layout | H3 User control and freedom; consistency with Library | Pane persistence | gap | frame | all | M | LF-10, NA-20, NB-05 | CONFIRMED (documented behaviour; a gap, not a rule) |
| [RP-032](#rp-032) | P2 | 2 | Collapsing a pane means hitting a 3x1 '<' button, and the collapsed handles still cost 13 + 11 columns (26 of 120) | Fitts's law; space efficiency; H6 (glyph-only control) | Pane collapse affordances | usability | frame | all | S | LF-11, LY-19, KA-21 | CONFIRMED |
| [RP-033](#rp-033) | P2 | 2 | Single-letter keys act from any non-text focus: stray typing switches modes, and Space on a button silently disables a dictionary | H5 Error prevention (mode errors); H3 User control | Screen keys c/p/d/l, [ ], and list keys Space/m/s | usability | both | all | S | NA-05, TF-13, KA-05, NA-15 | PARTIAL (by design but harmful; rated 2) |
| [RP-034](#rp-034) | P2 | 2 | Footer hints ignore focus: they advertise c/p/d/l, [ ] and s while you type, drop the context keys at 120 cols, and never mention 'm' (mark) | ADR-031 rule 4 (advertised == working in context); H6 Recognition rather than recall | Footer hints | usability | frame | all; most visible at 120x36 and below | S | NB-11, KA-10, LF-12, NA-11 | CONFIRMED (reproduced live) |
| [RP-035](#rp-035) | P2 | 2 | The key set breaks house conventions: no '/' for search, a banned Ctrl+S, Ctrl+Enter that many terminals cannot send, and no key at all for Chat now | H4 Consistency and standards; H7 Flexibility and efficiency (accelerators) | Keybindings | usability | both | all | S | NA-19, LF-13, KA-13, NB-13, KA-19 | CONFIRMED / PARTIAL (corrections applied) |
| [RP-036](#rp-036) | P2 | 2 | F1 help is the generic dump: titled 'PersonasScreen Shortcuts', raw key specs, keys that do nothing in context, and no m/s/Space | H10 Help and documentation; H2 (internal class name shown) | F1 contextual help | gap | frame | all; most needed at <=100 cols where the footer truncates | S | NB-21, KA-12 | CONFIRMED |
| [RP-037](#rp-037) | P2 | 2 | The in-screen unsaved-changes dialog talks about closing a 'tab' and offers no Save | H3 User control and freedom; H2 Match (there are no tabs); H4 (two different guards for the same drafts) | Unsaved-changes guard on mode/selection switch and Cancel | usability | content | all | S | NA-12, NB-20, KA-17 | CONFIRMED / PARTIAL (ADR scope corrected) |
| [RP-038](#rp-038) | P2 | 2 | Save state is hard to read: the unsaved signal sits far from the editor (the rail badge never renders), and after Save the editor looks unchanged | H1 Visibility of system status | Character/persona editor status | usability | both | all; no signal at all at <=24 rows | S | NA-13, TF-20 | PARTIAL (corrections applied) |
| [RP-039](#rp-039) | P2 | 2 | Inspector 'Delete' looks like an export button and silently switches to the marked rows, and marks vanish on any page change or search, so bulk actions reach one page at most | H5 Error prevention; H6 (the target of an action must be visible); H3 User control (a selection survives filtering) | Inspector Delete/Export; bulk mark (m, ●, 'N marked'); confirm dialog | usability | both | all | S | NA-14, NB-12, G3-05 | CONFIRMED + gap-round volume evidence (mark pruning itself is by design) |
| [RP-040](#rp-040) | P2 | 2 | Create and save rules differ by mode: New dictionary/lore saves 'Untitled' at once, entries save instantly, settings need Save, a greeting box is overwritten by browsing | H4 Consistency and standards; H3 User control (no cancel) | Create/edit flows across the four modes | usability | content | all | M | NA-24, TF-10, missed:nng-b#3 | CONFIRMED / PARTIAL (create-then-rename is documented) |
| [RP-041](#rp-041) | P2 | 2 | One concept, several names: Lore / lore book / world book / World Books; 'Dictionaries' here but 'Chat Dictionaries' in Console | H4 Consistency and standards; H2 Match (ecosystem vocabulary) | Terminology across Roleplay, Console and the guide | usability | both | all | S | NA-17, IA-04 | CONFIRMED |
| [RP-042](#rp-042) | P2 | 2 | The Roleplay list rail is titled 'Library', next to the separate ⌃3 Library destination and an 'Open in Library' action | H4 Consistency and standards; wayfinding | Rail title, collapsed handle, transcript action | usability | frame | all | S | NA-18, IA-03, LF-16 | CONFIRMED |
| [RP-043](#rp-043) | P2 | 2 | Product jargon and look-alike verbs: New / New Actor Pack, Import / Import Actor Pack, four 'Buddy' buttons, 'Type: lore', explained only in tooltips | H2 Match between system and the real world; H8; information scent | Rail toolbar; Inspector labels | usability | both | all; worst at 80x24 | M | NA-22, IA-09, NB-24 | CONFIRMED |
| [RP-044](#rp-044) | P2 | 2 | Validation copy: 'Validation: OK' is a constant for every saved item, and editor errors show internal widget ids ('personas-char-editor-name: required') | H9 Error messages in plain language; H1 truthful status; H8 | Inspector summary lines; editor validation footer | bug | content | all | S | NB-17, IA-11 | CONFIRMED |
| [RP-045](#rp-045) | P2 | 2 | The test chat misrepresents what is in play: it runs with nothing selected, labels replies 'character:', and always calls you 'User' | H1 Visibility of system status; H2 Match | Test chat (preview pane) | usability | content | all | S | NA-08, NB-23, IA-02, missed:ia#2 | PARTIAL (corrections applied) + verifier-sourced display-name bug |
| [RP-046](#rp-046) | P2 | 2 | After saving a new first message, the test chat keeps greeting with the old one, even after Reset | H1 Visibility of system status (feedback must reflect the current model) | Test chat <-> character editor | bug | content | all | S | TF-05 | PARTIAL (root cause corrected) |
| [RP-048](#rp-048) | P2 | 2 | Chat now always opens with the primary greeting; alternate greetings cannot be chosen | H7 Flexibility and efficiency; H3 User control | Chat now | gap | content | all | S | TF-16 | CONFIRMED |
| [RP-049](#rp-049) | P2 | 2 | A character's chat history is a 10-row, title-only list squeezed into the Inspector, and 'Conversations (N)' does nothing visible when the Inspector is collapsed | Information scent; H7 (resume fast); H1 | Character conversations (Inspector list, card button) | gap | both | all | M | NA-23, IA-16, G3-10 | CONFIRMED |
| [RP-050](#rp-050) | P2 | 2 | The conversation view opens with a 9-row block of 3-row buttons, and the transcript gets what is left | Visual hierarchy (content first); space efficiency | Character conversation (transcript) view | usability | content | all | S | LY-16 | PARTIAL (numbers corrected) |
| [RP-051](#rp-051) | P2 | 2 | The conversation transcript labels the user's own turns with the character's name (verifier-sourced) | H2 Match with the real world; H1 | Character conversation transcript | bug | content | all | S | missed:layout#4 | VERIFIER-SOURCED (live capture) |
| [RP-052](#rp-052) | P2 | 2 | The work pane never names the open item, and the visual weight sits on buttons rather than content | H1 Visibility (where am I / what is this); visual hierarchy; Gestalt similarity | Card, editor and detail headers; card labels | usability | both | all; worst at 120x36 with rails collapsed | S | LY-18, IA-13 | CONFIRMED / PARTIAL (Library comparison corrected) |
| [RP-053](#rp-053) | P2 | 2 | Card and editor list fields in different orders, card-spec fields have no explanations, and persona rows/cards are thin | H4 Consistency; H2 Match (jargon); H6 Recognition | Character card and editor; persona card and rows; Voice & Speech | usability | content | all | M | IA-15 | CONFIRMED |
| [RP-054](#rp-054) | P2 | 2 | The card and the editor scroll inside a scrolling stack (two scrollbars side by side), and the character's attachments sit below the outer fold | Consistent, predictable scrolling; H1 Visibility | Work pane (card, editor) | usability | content | all | M | LY-08, missed:fit#4 | CONFIRMED |
| [RP-055](#rp-055) | P2 | 2 | Empty and nothing-selected states teach nothing: one truncated rail line, a blank centre, character-only guidance, and inert controls | H10 Help and documentation (in-context onboarding); H8 (inert controls) | Empty Personas/Dictionaries/Lore; no-selection centre; arrival | gap | both | all | M | NB-09, TF-18, IA-19, LF-15 | PARTIAL (corrections applied) |
| [RP-057](#rp-057) | P2 | 2 | Palette commands promise actions they do not perform, and the nav tooltip describes things that do not exist | Information scent; H2 Match; ADR-011 (secondary routes must be truthful) | Command palette; nav purpose/tooltip | usability | content | all | S | NB-15, IA-17, TF-18 | CONFIRMED |
| [RP-058](#rp-058) | P2 | 2 | Import errors show raw parser text with no file name or accepted formats, and a file with no entries imports as a success | H9 Help users recognize, diagnose, and recover from errors | Character / lore / dictionary import | usability | content | all | S | NB-18 | CONFIRMED |
| [RP-059](#rp-059) | P2 | 2 | Provider failures: the test chat gives a generic error after ~50 s, and 'Chat now blocked' tells users to edit environment variables, with no button | H9 Help users recover from errors | Test chat + Inspector readiness | usability | content | all | M | NB-19 | CONFIRMED |
| [RP-060](#rp-060) | P2 | 2 | The Roleplay User Guide contradicts the live screen on its title, buttons, readiness messages and keys | H10 Help and documentation (accuracy) | Docs/User_Guide/roleplay-chat-dictionaries*.md | bug | content | n/a | M | NB-22, IA-18, missed:a11y#1 | CONFIRMED |
| [RP-061](#rp-061) | P2 | 2 | After choosing 'Stay' on the leave guard, the nav bar highlights the destination you did not go to, and clicking it does nothing (shell) | H1 Visibility of system status; H3 User control | Shell navigation <-> unsaved-changes veto | bug | shell (app_navigation.py; affects every guarded screen) | all | S | TF-14, missed:flows#3 | CONFIRMED (reproduced live); flows#3 note unconfirmed |
| [RP-062](#rp-062) | P2 | 2 | The list's focus state is invisible and its cursor uses a border colour (3.83:1); after a mode switch no cursor row is painted at all | WCAG 2.4.7 Focus Visible, 1.4.3 Contrast; H1 | Rail list | usability | both | all | M | KA-06, KA-20, missed:a11y#2 | PARTIAL (root cause corrected) |
| [RP-063](#rp-063) | P2 | 2 | A focused mode chip looks exactly like the active one, so two modes appear selected while tabbing | WCAG 2.4.7 Focus Visible; H1 | Mode chips | usability | frame | all | S | KA-11 | CONFIRMED |
| [RP-064](#rp-064) | P2 | 2 | Collapsing or expanding a pane (or selecting a row at narrow width) drops keyboard focus to nothing | WCAG 2.4.3 Focus Order; H3 | Rails | bug | frame | all; worst at <=60 cols | S | KA-08 | PARTIAL (rated 2: keyboard-only, recoverable with F6) |
| [RP-065](#rp-065) | P2 | 2 | Button focus is a thin underline plus a 1.06-1.64:1 background shift - not the heavy accent outline DESIGN.md asks for (app-wide) | WCAG 2.4.7 / 2.4.13 Focus Appearance; DESIGN.md:226, :347 | Buttons app-wide (Inspector, toolbar, editor) | usability | app-wide (design system) | all | S | KA-15 | PARTIAL (app-wide, deliberate; needs a ruling) |
| [RP-066](#rp-066) | P2 | 2 | At 80x24 and below, Roleplay keeps three cramped panes (1 list row) and at <=60 cols shows unlabelled 'L'/'In' slivers with no Escape way back | Graceful degradation; H3 (clearly marked exit); H6 | Responsive behaviour (small sizes) | usability | frame | 80x24, <=63 cols (graceful-degradation tier) | L | LY-20, KA-18, LF-18 | CONFIRMED (KA-18/LF-18 rated 1 alone) |
| [RP-067](#rp-067) | P2 | 2 | 'Ready' (header badge and Inspector line) reads provider settings only: it names no object, and said 'Ready' before all four Chat-now sends that were refused | H1 Visibility of system status (say what is ready); H2 Match ('Ready' should mean a reply will come) | Destination header badge; Inspector readiness line (characters and personas) | usability | both | all | S | NA-21, NB-07, IA-20, G1-03 | PARTIAL (restated; raised from 1 to 2 with gap-round evidence) |
| [RP-082](#rp-082) | P2 | 2 | Chat now does not say what the chat will contain: a persona chat becomes a 53-tool agent with an ~8,000-character prompt, and character chats carry instructions the user never sees | H1 Visibility of system status (what is in play); H2 Match with the real world; H10 Help and documentation | Chat now (persona vs character) -> Console request shape | usability | content | all | M | G1-05 | PARTIAL (by design but opaque) |
| [RP-083](#rp-083) | P2 | 2 | Roleplay's route for putting lore into a chat (Lore > Attachments > Attach to conversation…) lists bare chat titles and reports success with a raw conversation UUID | H6 Recognition rather than recall; H2 Match with the real world; H7 Flexibility and efficiency | Lore / Dictionary detail > Attachments > 'Attach to conversation…' | usability | content | all | S | G1-06 | PARTIAL ('only path' claim dropped) |
| [RP-084](#rp-084) | P2 | 2 | In persona and generic Console chats, the Console rail's Character header names an unrelated character ('Character · Dungeon Ma…') | H1 Visibility of system status; H2 Match with the real world | Console left rail, Character section header (where Chat now lands) | bug | external (Console left rail) | 160x45 and wider (collapsed to '▼ Character · Model · +2' at 120x36) | S | G1-07 | CONFIRMED |
| [RP-085](#rp-085) | P2 | 2 | New persona and Edit open at the previous session's scroll position, so the focused Name field can be off-screen and typing goes in blind | H1 Visibility of system status; WCAG 2.4.7 / 2.4.11 Focus visible / not obscured | Persona editor load (demand-mounted, reused instance) | bug | content | all; most visible at 120x36 | S | G2-07 | CONFIRMED |
| [RP-086](#rp-086) | P2 | 2 | A persona's 'Enabled' and 'Mode' look like behaviour switches but change nothing visible: a disabled persona still chats, and nothing local reads Mode | H2 Match between system and the real world; H1 Visibility (state not shown); H8 (inert controls) | Persona editor Mode/Enabled; persona list rows, card and Inspector | usability | content | all | S | G2-08 | CONFIRMED |
| [RP-087](#rp-087) | P2 | 2 | Four Save buttons with different scopes in one persona editor, and the tool-rule Save commits at once while its status says the persona Save will | H4 Consistency and standards; H5 Error prevention; H1 (contradictory save state) | Persona editor commit model (form / Buddy pack / reactions / policy rules) | usability | content | all | M | G2-09 | CONFIRMED |
| [RP-088](#rp-088) | P2 | 2 | 'Loading Shared Visual Identity reactions…' never resolves when an existing persona is opened with Edit | H1 Visibility of system status (a progress message that never ends) | Persona editor: shared visual identity host | bug | content | all | S | G2-10 | CONFIRMED (the failing fence is not pinned; medium confidence) |
| [RP-089](#rp-089) | P2 | 2 | The persona's look is a fixed 33-37-row block of implementation names and 3-row buttons whose labels truncate at 120x36 | H2 Match between system and the real world; H8 Aesthetic and minimalist design; H4 (two titles for one thing) | Persona editor: 'Persona Visual operational states' / 'Persona Visual' pack; Actor Pack titles | usability | content | all; truncation at 120x36 | M | G2-11 | CONFIRMED |
| [RP-090](#rp-090) | P2 | 2 | The link from a persona to the character whose portrait it uses is invisible and cannot be changed after creation | H1 Visibility of system status; H6 Recognition rather than recall; H3 User control and freedom | Persona list, card and editor; character_card_id | gap | content | all | M | G2-12 | CONFIRMED |
| [RP-091](#rp-091) | P2 | 2 | Lore entry tables size columns to the widest keys, so content shows 4-24 characters and on/off sits off-screen at every size; disabled entries are shown only by dim text | WCAG 1.4.1 Use of color; H1 Visibility of system status; H8 (show the content, not the chrome) | Lore entries table (keys, content, position, priority, enabled) | usability | content | 120x36 (content ~4 chars), 160x45 (~23-24), 220x55 (~54); the enabled column is off-screen at all three | S | G3-03 | CONFIRMED |
| [RP-092](#rp-092) | P2 | 2 | The page bar wraps and shifts, takes up to 3 rows under a 7-row list at 120x36, and has no page number or paging keys | H8 Aesthetic and minimalist design; stable layout; H7 Flexibility and efficiency (keyboard); H1 ('page 2 of 7') | Characters/Personas page bar and count line | usability | both | 120x36 (2-row bar + a blank count line on every page), 160x45 (1 row on page 1, 2 from page 2), 220x55 (1 row) | M | G3-06 | CONFIRMED |
| [RP-093](#rp-093) | P2 | 2 | Roleplay loads only the newest 100 personas: with 114 it says 100, and the oldest (including 'Default Narrator') cannot be found even by search | H1 Visibility of system status; integrity of browse (no silent truncation) | Personas mode list (persona load path) | bug | content | all | S | G3-07 | PARTIAL (rated 2: rare at the target user's scale) |
| [RP-094](#rp-094) | P2 | 2 | Long names are cut at the end, so near-duplicate cards, books and chats look identical ('Highspire Gazetteer (imported fr…' twice), with no tooltip | H6 Recognition rather than recall; information scent; telling list items apart | Items list rows (all modes); Inspector conversation list | usability | content | 120x36 (~21-char names), 160x45 (~32), 220x55 (~48) | S | G3-08 | CONFIRMED |
| [RP-095](#rp-095) | P2 | 2 | Tag filter picker: Enter and Down in its filter box do nothing, although the hint says 'Enter to pick' | H7 Flexibility and efficiency of use; H4 Consistency (the hint promises Enter) | Characters > Tag: filter picker (TagFilterPicker) | bug | content | all | S | G3-13 | CONFIRMED |
| [RP-096](#rp-096) | P2 | 2 | Reordering lore entries is one step per press at ~1-1.25 s each, with no 'move to position' or editable order, and the rewrite is not atomic | H7 Flexibility and efficiency of use; direct manipulation of order | Lore and Dictionary entries: Move up / Move down | gap | content | all | M | G3-14 | CONFIRMED |
| [RP-068](#rp-068) | P3 | 1 | Each character takes 2 list rows; at medium sizes the second row is a 21-32-character description fragment | Information scent; list density | Character list rows | usability | content | 120x36, 160x45 | S | LY-04 | PARTIAL (deliberate trade-off; severity 1) |
| [RP-069](#rp-069) | P3 | 1 | The card shows raw markdown asterisks while the test chat renders them | H2 Match; H4 Consistency | Character card view | usability | content | all | S | NA-25 | CONFIRMED |
| [RP-070](#rp-070) | P3 | 1 | Copy slips: '1 entries', and a raw UTC date on new character rows | H4 Consistency; H2 | List row meta | bug | content | all | S | TF-21 | CONFIRMED |
| [RP-071](#rp-071) | P3 | 1 | Roleplay ignores the ASCII glyph fallback (hard-coded ▸/▾, the ● mark and a ✨ emoji) | ADR-034 (glyph ownership); robustness | Glyphs | gap | both | all | S | KA-16 | PARTIAL (severity 1) |
| [RP-097](#rp-097) | P3 | 1 | The 'Console RAG capture unavailable; reason=capture_provider_failure' ERROR on every send is a harmless argument-mismatch bug, not why sends were blocked | H9 Help users diagnose errors (diagnostics must name the real cause) | Console runtime capture-provider wiring (F3 Logs; turn trace) | bug | external (Console; ledger G2-37, out of scope in TASK-33621.20) | n/a (log) | S | G1-08 | CONFIRMED (already in the Console ledger) |

### A.2 Details


#### P0

<a id="a2-rp-072"></a>
##### RP-072 · On a default install, the first message in every Chat-now chat is refused before it reaches the model, with no reason and no way out (Console defect TASK-33621.2: a blocking dependency, not redesign work)

P0 · severity 4 · bug · scope external (Console; TASK-33621.2) - blocks the redesign's primary action · effort M · confidence high  
*Principle:* H1 Visibility of system status; H9 Help users recognize, diagnose and recover from errors; completion of J1 · *Area:* Chat now -> first Console send in a character- or persona-bound session · *Sizes:* all (verified at 160x45 and 120x36) · *Frequency:* Every first send of every Chat-now session on the default config: 4 of 4 in the gap round, and 4 of 4 in the earlier rv-flows, rv-flows-2 and rv-ia runs  
*Sources:* G1-01 · *Verdict:* CONFIRMED (live, with full traceback; capture-off control completed)

Evidence:
- `rv-gap1 160x45, golden profile, body-logging mock on :18777` (live; G1-01): Isolde, Vex and Cozy Storyteller: Chat now, type, Enter. Each logged 'controller_submit status=blocked' ~4 ms after 'durable_commit status=succeeded'; bodies.jsonl gained 0 lines. A plain send from a new Console tab in the same run reached the mock (20:28:15, tools=53)
- `rp-review/gapcheck/trace.log 20:35:48, 20:36:46, 20:37:30, 20:48:09 (out-of-tree sitecustomize instrumentation; no product code changed)` (live; G1-01): Every refusal is 'TraceProvenanceAlignmentError(trace provenance category mismatch: system)' on [system 'You are Captain Isolde Varga…', user …], then SUBMIT RESULT BLOCKED with 'Trace provenance could not be saved. Retry, Send without capture, or Cancel.'
- `tldw_chatbook/Chat/console_chat_controller.py:11955-11968; Chat/console_prepared_request.py:929-940; Chat/console_trace_provenance.py:841-857, 989-994` (code; G1-01): An unsaved provider row gets ACTIVE_REQUEST provenance; leading system rows fill the 'system' category, which accepts only saved or RENDERED_SYSTEM descriptors, so building the request raises
- `console_chat_controller.py:12336-12354, 12036-12074` (code; G1-01): A bare 'except Exception' turns the error into a TRACE_PROVENANCE pause and sets run state BLOCKED; nothing is logged
- `tldw_chatbook/config.py:3829-3831, 8954` (code; G1-01): The shipped template writes [console] exchange_capture = true and the reader defaults to True, so fresh installs take this path; it does not depend on embeddings or RAG config
- `rv-gap1-2 160x45, same profile plus [console] exchange_capture = false` (live; G1-01): Control: the same 3 Chat-now sends completed (trace.log 20:39:27, 20:40:27, 20:41:14); capture is the only variable
- `captures/review-rv-gap1-vex-chatnow-after-160x45 rows 36-41; review-rv-gap1-isolde-chatnow-after-120x36 rows 32-34; review-rv-gap1-persona-cozy-after-160x45 rows 37-41` (capture; G1-01): The user row stays with a lowercase 'blocked' in the right gutter; the header still says 'Ready'; no reason, no action
- `captures/review-rv-gap1-persona-second-send-160x45 row 40; console_chat_controller.py:10774` (live; G1-01): A second send is refused ('Another send is still preparing for this conversation.') and parked in 'Queue 0/10 · Unsent turn needs attention [Rest][Discard]'. In 1 of 4 runs the user row never rendered and the composer was cleared, though the DB holds the row (review-rv-gap1-isolde-blocked-userrow-missing-run2-160x45 rows 30-40)
- `vf-gap1 160x45, golden, shared mock: Roleplay > Isolde > Chat now > send` (live; G1-01 verification): Independent repro: app log 412-413 'durable_commit succeeded' then 'controller_submit status=blocked' 4 ms later with no error line; mock request log unchanged; header 'Ready', Inspect 'Run: Ready'
- `backlog/tasks/task-33621.2 (To Do, P0)` (doc; G1-01): Already filed for chats with a system prompt; candidate fix PR #2882 / TASK-32951 is not in 84247cb843

Recommendation: see [section 5](#rp-072).

*Verifier corrections applied:* Scope: neither Roleplay frame nor Roleplay content - an external Console dependency (TASK-33621.2, P0, which already names 'Start new chat' character chats); no Roleplay redesign work fixes it, so it is tracked as a blocking dependency. The capture-off control's ~1.2 s first reply holds for Isolde and Vex; the persona took ~12 s ('Connecting tools… · 11s'). This explains the 'cross-surface note' blocks in Appendix C.2; the capture_provider_failure ERROR logged beside them is a separate, harmless bug (RP-097).

<a id="a2-rp-004"></a>
##### RP-004 · Lore books and dictionaries attached to a character are never applied in Console chats (confirmed live with logged prompts)

P0 · severity 4 · bug · scope content · effort M · confidence high (confirmed live with logged prompt bodies in the gap round; first found by code reading)  
*Principle:* H1 Visibility of system status; H2 Match with the real world (an attachment shown as active should act) · *Area:* Character attachments (World Books / Dictionaries sections) -> Console send path · *Sizes:* n/a (behaviour) · *Frequency:* Every chat with a character that has attached lore or dictionaries  
*Sources:* missed:nng-a#1, G1 (gap1 section 1) · *Verdict:* CONFIRMED live (gap round G1; originally verifier-sourced from code) · **verifier-sourced**

Evidence:
- `tldw_chatbook/Chat/console_chat_controller.py:1284-1297` (code; missed:nng-a#1): capture_prompt_transform_inputs passes char_data=None
- `tldw_chatbook/UI/Screens/chat_screen.py:14400-14423` (code; missed:nng-a#1): 'Conversation-only: char_data is None (native sessions carry no character card)'
- `tldw_chatbook/Chat/console_runtime.py:567-574, 630` (code; missed:nng-a#1): Same char_data=None on the other live send paths
- `tldw_chatbook/Character_Chat/world_book_manager.py:69` (code; missed:nng-a#1): resolve_character_world_books is reachable only through the retired chat_events.py path
- `personas_screen.py:6952; chat_screen.py:14536` (code; missed:nng-a#1): The only associate_world_book_with_conversation callers - Chat now does not copy character attachments onto the conversation
- `console_chat_controller.py:1284-1297; chat_screen.py:14421-14423; console_runtime.py:567-574, 630` (code; NA-02/NA-08 verification): The NA-02 and NA-08 verification checks traced the same char_data=None paths
- `captures/review-rv-nng-a-isolde-worldbook-stale-snapshot-160x45 rows 36-39` (capture; NA-02): The card presents '▾ World Books (1) ... Attach world book...' as active
- `rp-review/gapcheck/bodies.jsonl 20:39:27 (rv-gap1-2, capture off; Isolde, Kepler-9 attached)` (live; G1): The system prompt holds only card text; the Kepler-9 entry ('an orbital shipyard station…') and the Hadley Gap entry ('survey ship *Tamsin*…') are absent; instrumentation logged WORLD_INFO changed=False
- `bodies.jsonl 20:40:27 (Vex; Aetheria Atlas and Fantasy Anachronism Swaps attached)` (live; G1): User turn sent verbatim 'Call the police at the gates of Emberfall, OK?' - no 'city watch', no lore text; CHAT_DICT changed=False
- `bodies.jsonl 20:42:34 (control)` (live; G1): After Lore > Kepler-9 > Attachments > Attach to conversation > the Isolde chat, the user turn is prefixed with both Kepler-9 entries: the conversation-level path works
- `bodies.jsonl 20:43:57` (live; G1): The test chat also ignores character attachments (no rewrites, no lore)
- `tldw_chatbook/Character_Chat/Chat_Dictionary_Lib.py:1433-1434` (code; G1): Docstring: char_data is 'always None for the native Console today'

Recommendation: see [section 5](#rp-004).

*Verifier corrections applied:* Gap round: confirmed live with a body-logging mock, but only with [console] exchange_capture = false, because on the default config every character send is refused first (RP-072). Path corrected: console_runtime.py is under tldw_chatbook/Chat/, not UI/Console_Modules/. The Console Inspect rail can attach books and dictionaries to the active conversation (UI/Console_Modules/retrieval.py:584-624), so a workaround exists, below Inspect's fold.

<a id="a2-rp-014"></a>
##### RP-014 · After any lore or dictionary entry action the table jumps back to row 1, so the next Delete, Move or Update silently hits entry #1; Delete and Detach have no confirmation or undo

P0 · severity 4 · bug · scope content · effort S · confidence high  
*Principle:* H5 Error prevention; H1 Visibility of system status; H3 User control and freedom; keep the selection stable across a mutation · *Area:* Lore / Dictionary Entries tab (entry table + Add/Update/Delete/Move); character World Books / Dictionaries sections · *Sizes:* all; worst at 120x36 and 160x45, where row 1 is usually off-screen · *Frequency:* Every second entry action in a row (delete two entries, move one entry more than one place, Update then Delete); every Delete or Detach  
*Sources:* NA-06, missed:flows#1, G3-01 · *Verdict:* CONFIRMED (reproduced live in the gap round; raised from P1/3 to P0/4)

Evidence:
- `personas_lore_detail.py:174-199, 612-619; personas_dictionary_detail.py:242-266` (code; NA-06): Add | Update | Delete | Move up | Move down share one style; Delete posts directly
- `personas_screen.py:6510-6518 (dictionary), 6673-6683 (lore)` (code; NA-06): Handlers call delete_entry / delete_world_book_entry with no dialog; no success toast
- `captures/review-rv-nng-a-lore-entry-form-before-delete-160x45 row 27; review-rv-nng-a-lore-entry-deleted-no-confirm-160x45 rows 24-26` (live; NA-06): 'Ballast' gone at once; rail '6 entries' -> '5 entries - on'; form loads the next entry
- `personas_screen.py:6321-6352 -> world_book_manager.py:864-893` (code; missed:flows#1): Detach from a character is immediate with no confirmation or undo
- `DESIGN.md:127` (doc; NA-06): 'Confirm irreversible actions'
- `rv-gap3 160x45, Lore > Grand Codex (250 entries): select #199 'Foxglove Vale Pact', Tab to Delete, Enter, Enter` (live; G3-01): First Enter deleted #199; second Enter deleted entry #1 'Brackenfell', 198 rows away (DB 250 -> 248; order 0 became 'Thornhold'); no dialog, no toast
- `captures/review-rv-gap3-lore-after-delete2-160x45 rows 20-25` (capture; G3-01): Rail '248 entries · on'; the form shows entry #1 ('The Thornhold has stood…'), not the area being edited
- `rv-gap3 160x45: select #200 'Saltmarsh Bellwarden', Move down, Move down` (live; G3-01): First press moved #200 to #201; second press moved entry #1 'Thornhold' to position 2 (DB before/after)
- `captures/review-rv-gap3-lore-after-update-200-160x45 rows 24-25; review-rv-gap3-lore-after-add-160x45 rows 21-25` (capture; G3-01): After Update on #200, and after Add (new entry lands at #249), the form reloads entry #1; nothing confirms what was saved or where
- `personas_lore_detail.py:309-318, 405-414, 572-581; textual _data_table.py:1582-1611 (Textual 8.2.8)` (code; G3-01): update_entries() calls table.clear() (cursor to (0,0), scroll to 0) and re-adds every row; selected_entry_id is simply the cursor row; RowHighlighted refills the form
- `personas_screen.py:6578-6633; 5799-5835; personas_dictionary_detail.py:390-399` (code; G3-01): Every lore op reloads entries without restoring the selected entry (the rail is restored with mark_active_row, :6608) and sets status '' on success; dictionaries use the same clear-and-reload path
- `vf-gap3 160x45 (scaled copy): select #3 Vellmoor, Move down twice` (live; G3-01 verification): First press moved Vellmoor to #4; second swapped Thornhold (#1) with Saltmarsh Abbey, checked in the DB (captures/review-vf-gap3-lore-movedown1-160x45, -movedown2-160x45)

Recommendation: see [section 5](#rp-014).

*Verifier corrections applied:* Verifier: worse than stated - the form auto-loads the next entry after a delete. DESIGN.md:127 ('Confirm irreversible actions') is the governing rule (ADR-031:10 covers keys, not buttons). Gap round (G3-01): the earlier 'form then loads the next entry' was this row-0 reset. It does not depend on volume (reproduced on entry #3); volume only hides it because row 1 is off-screen. Raised to P0/severity 4 because an action lands on an entry the user never selected or saw. The same reset explains RP-040's 'after Add the form jumps to the first entry'. Dictionaries share the clear-and-reload path (code-verified, not driven live). Also guard a second reorder press that lands while the ~1 s per-entry write loop is still running (RP-096).

<a id="a2-rp-001"></a>
##### RP-001 · Once a dictionary, lore book or editor has been opened, its empty slot stays on screen and squeezes every later view (the character editor shrinks to a 2-5-row window)

P0 · severity 4 · bug · scope content · effort S · confidence high  
*Principle:* H8 Aesthetic and minimalist design; H1 Visibility of system status; stable layout (DESIGN.md:125) · *Area:* Work pane: #personas-detail-stack and its four demand-mounted slots · *Sizes:* all; worst at 120x36 (2-row editor window), 160x45 (5-9-row window) · *Frequency:* Every session that opens any two of: character editor, persona editor, dictionary detail, lore detail. One Lore visit is enough.  
*Sources:* LY-01 · *Verdict:* CONFIRMED (reproduced live with a smaller trigger than first reported)

Evidence:
- `tldw_chatbook/UI/Screens/personas_screen.py:2030` (code; LY-01): _ensure_center_view sets slot.display = True after first mount; this is the only write - nothing hides a slot again
- `tldw_chatbook/UI/Screens/personas_screen.py:15926-15932, 643-654, 665-670` (code; LY-01): _show_center toggles only the _CENTER_VIEW_IDS roots, never the four _DEMAND_CENTER_VIEW_SLOTS
- `tldw_chatbook/UI/Screens/personas_screen.py:1634-1686; Textual 8.2.8 containers.py:124-131` (code; LY-01): Slots are bare Verticals (height: 1fr by default) inside VerticalScroll#personas-detail-stack, so each opened slot keeps an equal fr share
- `captures/review-rv-layout-2-editor-porthole-after-dict-lore-120x36` (capture; LY-01): Editor is a 2-row window (rows 15-16); Save/Cancel at row 18; rows 21-32 blank
- `captures/review-rv-layout-editor-collapsed-after-dict-lore-160x45` (capture; LY-01): 5-row editor window with 17 blank rows below
- `rv-layout-3 160x45: Lore > Aetheria Atlas > Characters > Captain Isolde Varga > Edit` (live; LY-01): Editor body rows 16-24 then 14 blank rows; same path without the Lore visit fills the pane (roleplay-character-editor-160x45)
- `captures/review-vf-layout-editor-after-lore-only-120x36` (live; LY-01 verification): Lore visit only: editor body rows 16-20, Save/Cancel row 22, rows 25-32 blank
- `captures/review-rv-layout-2-lore-after-dict-blank-band-120x36 rows 14-21; roleplay-lore-selected-160x45 rows 14-22` (capture; LY-01): Blank band above the Lore tabs and 0-1 entry rows; the 160x45 driver had opened the editor first
- `captures/review-rv-layout-dict-settings-clipped-160x45 rows 23-30; review-rv-layout-3-transcript-view-160x45` (capture; layout missed #5): The same leak splits the dictionary Settings tab and the transcript view

Recommendation: see [section 5](#rp-001).

*Verifier corrections applied:* Verifier found a single Lore visit is enough (no Dictionaries visit needed). The same leak also splits the dictionary Settings tab and the conversation-transcript view (verifier missed note).

<a id="a2-rp-002"></a>
##### RP-002 · The character editor's long-text fields show 0-1 lines of text, so clicks land mid-word and edits are saved corrupted

P0 · severity 4 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; H5 Error prevention; HCI: WYSIWYG direct manipulation · *Area:* Character editor (Characters > Edit), main and Advanced fields · *Sizes:* all, including 220x55 (heights are fixed cell counts) · *Frequency:* Every edit of First message, Personality, System prompt, Scenario, Post-history instructions or Creator notes  
*Sources:* TF-01, NA-09, LY-09, NB-03 · *Verdict:* CONFIRMED (all four sources)

Evidence:
- `tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py:98-123` (code; TF-01/NA-09/LY-09): First message, Personality, System prompt height 3 (1 text row); Description 4 (2 rows); Scenario, Post-history, Creator notes height 2 (0 rows - the border uses both)
- `widget_defaults_self.tcss:2225-2248` (code; TF-01): Bundled CSS mirrors the same heights; no override
- `rv-flows-2 160x45: Vex > Edit > click First message (90,23) / Scenario border (90,25), type, Save` (live; TF-01): Card afterwards: 'Scenario: ... The pa Rain hammers the portcullis.rty stands...' and 'The gates o The torches gutter.f Emberfall groan open'
- `captures/review-rv-flows-2-j3-card-after-save-160x45 rows 21-25; review-rv-flows-2-j3-firstmsg-1row-editing-160x45 row 23` (capture; TF-01): Corrupted saves; the 1-row box shows the garbled wrapped tail while editing
- `captures/review-rv-flows-j3-advanced-220x55 rows 32-40` (capture; TF-01): Scenario, Post-history, Creator notes still border-only boxes at 220x55
- `captures/roleplay-character-editor-scenario-zero-height-160x45 rows 18-20; roleplay-character-editor-advanced-160x45` (capture; NA-09/LY-09): Focused Scenario is an empty frame although Isolde has a scenario (roleplay-character-selected-220x55)

Recommendation: see [section 5](#rp-002).

*Verifier corrections applied:* Description (height 4) shows 2 rows; Personality and System prompt (height 3) show 1 row each - the problem is wider than the Advanced fields. NB-03's search half is in RP-009.

<a id="a2-rp-003"></a>
##### RP-003 · Dictionary and lore entry forms: most controls are pushed off-screen, and the rest have no labels or show raw internal values

P0 · severity 4 · bug · scope content · effort M · confidence high  
*Principle:* H6 Recognition rather than recall (persistent labels); H2 Match with the real world; H1 Visibility · *Area:* Dictionaries / Lore detail: Entries and Settings forms; persona Mode · *Sizes:* all, verified at 160x45 and 220x55 (dictionary options unreachable by mouse at every size) · *Frequency:* Every dictionary or lore edit  
*Sources:* NB-01, NA-16, IA-14, TF-11 · *Verdict:* CONFIRMED (NB-01 reproduced live; NA-16, IA-14 confirmed) + verifier-sourced additions

Evidence:
- `tldw_chatbook/Widgets/Persona_Widgets/personas_dictionary_detail.py:206-241` (code; NB-01): One Horizontal: Pattern Input (default width 100%) then 3 Switches (tooltip-only) and 4 Inputs; replacement TextArea unlabelled
- `tldw_chatbook/Widgets/Persona_Widgets/personas_lore_detail.py:131-173, 199-212` (code; NB-01): Keys Input first, Position/Priority/Enabled clipped; Settings Name/Scan depth ('3')/Token budget ('500') placeholder-only
- `personas_dictionary_detail.py:29, 265-276; persona_profile_editor_widget.py:143-150` (code; NA-16/IA-14): Strategy Select shows raw 'sorted_evenly', 'character_lore_first', 'global_lore_first'; persona Mode shows 'session_scoped'/'persistent_scoped' (nothing reads it per characters-and-personas.md:268-270)
- `captures/review-rv-nng-b-dict-entry-form-160x45 rows 26-28; -220x55 rows 30-37` (capture; NB-01): Only two boxes ('car', 'carriage'), full width, no labels
- `captures/review-rv-nng-b-3-lore-entry-form-220x55` (capture; NB-01): 'Case-sensitive' and 'Regex' labels with no switch; Position/Priority/Enabled absent; column header cut to 'pr'
- `captures/review-rv-nng-b-2-lore-settings-unlabeled-160x45 rows 16-26; review-rv-ia-lore-settings-unlabelled-160x45` (capture; NB-01/IA-14): A bare '4' (scan depth) with no label
- `vf-nng-b 160x45` (live; NB-01 verification): Clicking an entry changed nothing visible; next Tab moved focus to the off-screen regex switch
- `captures/review-vf-nng-a-dict-entry-form-tab2-160x45` (capture; nng-a missed #2): Second Tab from Pattern lands on an invisible off-screen control
- `Docs/User_Guide/roleplay-chat-dictionaries/chat-dictionaries.md:55-66` (doc; NB-01): The guide names every control; it is the only place those names appear

Recommendation: see [section 5](#rp-003).

*Verifier corrections applied:* Verifier notes: the dictionary case is worst - seven non-Pattern options are invisible at every size including 220x55; in lore only the first row overflows (Case/Selective/Regex have labels but their switches are pushed off). Placeholder-as-label also affects Name inputs and the primary Entries row (Probability '100', Max '1', Priority '0'); Description TextAreas have no label at all. Persona Mode and Enabled are covered in RP-086 (gap round), where 'hide Mode' is scoped to local mode.

<a id="a2-rp-073"></a>
##### RP-073 · Three persona-editor sections draw as a title only at every size (tool policy, shared reactions, New Actor Pack's portrait picker), yet Tab still enters them blind and saves what it finds

P0 · severity 4 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; H6 Recognition rather than recall; H5 Error prevention (an invisible pre-selected value is committed); WCAG 2.4.7 Focus visible · *Area:* Persona editor: #personas-policy-rules-editor, #personas-editor-shared-visual-identity-host, #personas-editor-pack-portrait · *Sizes:* 120x36, 160x45 and 220x55 (policy verified at all three; pack portrait at 120x36; same CSS cause everywhere) · *Frequency:* Every persona edit (policy, reactions) and every New Actor Pack  
*Sources:* G2-02 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap2-persona-editor-bottom-policy-clipped-160x45 row 39; review-rv-gap2-2-persona-editor-bottom-policy-clipped-120x36 row 30; review-rv-gap2-3-persona-editor-bottom-policy-clipped-220x55 row 49` (capture; G2-02): Heading 'Tool policy rules (narrowing-only)', then Save/Cancel two rows below; rule list, Kind, Name, Allowed, Require confirmation, Max calls/turn, New/Save/Delete, warning and status are absent at all three sizes
- `rv-gap2 160x45 and rv-gap2-3 220x55, Tab-by-Tab ANSI diffs` (live; G2-02): 220x55: 22 Tabs from Name to Save, Tabs 14-21 change nothing on screen; 160x45: Tabs 8-15 of 16 are invisible
- `rv-gap2 160x45: blind keys 'skill' Tab 'web_search' Tab x5 Enter` (live; G2-02): A rule was created and persisted without ever being visible; toast 'Tool policy rules saved.'; Inspector 'Tool policy: 1 rule(s) / skill: web_search → allow' (captures/review-rv-gap2-policy-rule-saved-blind-while-form-unsaved-160x45 rows 19-20)
- `captures/review-rv-gap2-2-new-actor-pack-portrait-select-hidden-120x36 rows 15-17; review-rv-gap2-2-actor-pack-portrait-dropdown-blind-120x36.png` (capture; G2-02): Label 'Required portrait Character' with no Select drawn; Shift+Tab, Enter opens its overlay with 'Samira “Sammy” Vadem' pre-selected among 3 eligible, plus the blank prompt shown as an option
- `rv-gap2-4 160x45 and rv-gap2-2 120x36 after a persona Save` (live; G2-02): The Shared Visual Identity browser (filter, list, preview, Replace/Generate/Generate All/Clear/Save/Cancel) draws as one row, 'Shared Visual Identity reactions'
- `persona_profile_editor_widget.py:109-175; widget_defaults_self.tcss:2120-2150; textual containers.py:22-27, 124-130` (code; G2-02): The three containers sit in VerticalScroll#personas-editor-body with Textual's default height:1fr and overflow hidden; no rule gives them height:auto, and the 1fr TextAreas take the space (RP-076)
- `personas_policy_rules_editor.py:33-37, 52, 153-181; persona_profile_editor_widget.py:323, 333-335, 561-566` (code; G2-02): The hidden block holds the deny-by-default warning and every validation message; begin_actor_pack_creation pre-selects portrait_options[0]; the only visible portrait error is footer text 'personas-editor-character-portrait: portrait Character required'
- `runs/rv-gap2 personas.json` (data; G2-02 verification): Cozy Storyteller holds the blind-created rule skill:web_search allow

Recommendation: see [section 5](#rp-073).

*Verifier corrections applied:* Only the 120x36 capture shows the pack-portrait clipping; the 160x45 and 220x55 evidence covers the policy section, though the same CSS cause makes all-size clipping near-certain. A persona-editor relative of RP-003 (different cause: 1fr containers collapsed inside a scroll, not a row pushed off the right edge).

<a id="a2-rp-005"></a>
##### RP-005 · Lore and dictionary Settings tabs cannot scroll: 'Save settings', Export and Enabled are unreachable by mouse at the target sizes

P0 · severity 3 · bug · scope both · effort S · confidence high  
*Principle:* H1 Visibility of system status; H3 User control and freedom · *Area:* Dictionary / Lore detail > Settings tab · *Sizes:* Dictionaries: 120x36, 160x45 (visible at 220x55). Lore: 120x36, 160x45 and 220x55 · *Frequency:* Every lore/dictionary settings edit, and every New (which lands here to rename)  
*Sources:* NA-07, TF-10, IA-08 · *Verdict:* CONFIRMED (reproduced live by three verifiers) + verifier-sourced severity notes

Evidence:
- `personas_dictionary_detail.py:269-305; personas_lore_detail.py:201-232` (code; NA-07): Settings TabPanes are plain containers; only Entries is wrapped in VerticalScroll
- `personas_lore_detail.py:91-94; personas_dictionary_detail.py:138-141; personas_screen.py:1634` (code; NA-07 verification): Detail widgets are height 1fr inside VerticalScroll#personas-detail-stack, so the outer scroll never overflows and the tab content is clipped
- `captures/review-vf-nng-a-lore-settings-wheel-no-save-160x45; review-rv-nng-a-3-lore-settings-no-scroll-save-hidden-160x45` (live; NA-07): Only Name, Description and a bare '4'; 10 wheel events do nothing; no 'Save settings' on screen
- `captures/review-rv-nng-a-3-lore-settings-focus-offscreen-160x45; review-rv-nng-a-2-dict-settings-clipped-focus-lost-120x36` (live; NA-07): Tab moves focus to invisible controls
- `captures/review-vf-nng-b-dict-settings-clipped-160x45` (live; missed:nng-b#1): Strategy, Token budget, Enabled, Save settings, Export JSON/Markdown never appear; 8 blank rows below
- `captures/review-vf-ia-lore-settings-wheel-noscroll-160x45; review-rv-ia-dict-settings-scrolled-160x45` (live; missed:ia#1): Lore shows only Name under a 7-row band; dictionary at 220x55 is fine (review-vf-ia-2-dict-settings-visible-220x55)
- `captures/review-rv-nng-b-lore-settings-tab-220x55` (capture; NA-07 correction): Lore Settings clipped at 220x55 as well
- `rv-flows-2 160x45: Dictionaries > Ctrl+N > type 'Pirate Speak' > wheel x5` (live; TF-10): 'Save settings' never visible; saved with a blind Tab x5 + Enter (captures/review-rv-flows-2-j5-new-dictionary-160x45)
- `personas_screen.py:8044-8048, 8242-8248` (code; NA-07): New dictionary / New lore deliberately land in Settings 'to rename immediately'

Recommendation: see [section 5](#rp-005).

*Verifier corrections applied:* NA-07's sizes were wrong: Lore is broken at 220x55 too. TF-10's 'New persists immediately' half is in RP-040; IA-08's scattered-actions half is in RP-020. One verifier argued severity 4; kept at 3 because a blind-keyboard workaround exists, but ranked P0.

<a id="a2-rp-006"></a>
##### RP-006 · Dictionary and lore detail starves the entries: 0-5 entry rows at the target sizes while a permanent Try-it block and blank bands take the room

P0 · severity 3 · usability · scope both · effort L · confidence high  
*Principle:* H8 Aesthetic and minimalist design; master-detail (list -> editor) ecosystem convention · *Area:* Dictionaries / Lore centre pane (Entries tab + Try-it) · *Sizes:* 120x36 worst (0 entries visible); 160x45 (3-5 rows); 220x55 wasteful · *Frequency:* Every dictionary or lore visit  
*Sources:* NB-06, TF-11, LF-20, LY-15, LF-08 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

Evidence:
- `personas_screen.py:1679-1715, 4490-4498` (code; NB-06/LY-15): Both Try-it widgets are siblings below the detail stack; display set by mode only, so shown with nothing selected
- `personas_lore_tryit.py:39-43, 75-84; personas_dictionary_tryit.py:76-79` (code; LY-15): max-height 60%; a permanently disabled 'Include recent turns (soon)' switch takes 3 rows
- `personas_dictionary_tryit.py:114-120` (code; missed:layout#1): Five empty Statics reserve blank rows before any run
- `personas_lore_detail.py:98-101, 136-174; personas_dictionary_detail.py:142-146, 202-204` (code; LF-20): Entries tab = one VerticalScroll holding a DataTable (min 6 / max 12) followed by the entry form
- `captures/roleplay-dictionaries-tryit-120x36; roleplay-lore-selected-120x36` (capture; NB-06): 0 entry rows at 120x36
- `captures/review-rv-flows-2-j6-kepler-selected-160x45 rows 14-27; review-rv-flows-2-j6-entries-scrolled-160x45` (capture; TF-11): 7-row blank band; 3 of 6 entries; content cut ~45 chars; ~14 wheel notches through a 5-row window
- `captures/review-rv-flows-3-j6-lore-120x36` (capture; TF-11): 4-row window; content cut ~25 chars
- `vf-nng-b 160x45 and 120x36` (live; missed:nng-b#2): 4-row peephole with 8 blank rows beneath; Dictionaries at 120x36 showed the column header and zero rows
- `captures/review-rv-nng-b-dict-entry-form-160x45; review-rv-nng-b-lore-settings-tab-220x55` (capture; missed:nng-a#4): ~8-row blank band at 160x45, ~15 at 220x55 - more height makes it worse
- `rv-layout 160x45 fresh profile, Lore > Aetheria Atlas` (live; LY-15): Clean first visit: tabs row 14, entries 17-27, Try-it 28-37 - entry form only reachable by scrolling a 14-row region
- `captures/review-rv-gap3-lore-grand-clean-120x36 rows 17-21; review-rv-gap3-lore-grand-clean-220x55 rows 17-27` (capture; G3-02): 250-entry book, clean visits: 4 entry rows at 120x36 and 10 at 220x55; the 12-row cap does not grow with the terminal

Recommendation: see [section 5](#rp-006).

*Verifier corrections applied:* Part of the blank band is RP-001 (a clean first visit to Lore at 160x45 shows 10 entry rows). Independently, the detail TabbedContent is clamped to a few rows with blank space above or below depending on path - fix the pane's height allocation, not only first mount. LF-20's 'books as rail sub-rows' was rejected by its verifier (does not scale). TF-11's off-screen-controls half is in RP-003. Gap round (G3) suggested severity 4 at realistic volume; not adopted: a clean 160x45 Lore visit shows 10 entry rows (the 3-row window is RP-001), and the volume cost is captured by RP-056 (no entry filter, 12-row cap) and RP-014 (row-0 reset). Try it's own clipping at volume is RP-081.

<a id="a2-rp-007"></a>
##### RP-007 · 13 rows of stacked chrome above the panes and 4 below take 38-47% of the screen height at the medium sizes (31% at 220 cols); the Library spends 16-20%

P0 · severity 3 · usability · scope frame · effort M · confidence high  
*Principle:* H8 Aesthetic and minimalist design (chrome-to-content ratio); PRODUCT.md:31 density · *Area:* Screen frame: DestinationHeader box, purpose line, 'Modes:' strip, workbench frame, pane borders · *Sizes:* all; 47% at 120x36, 38% at 160x45, 31% at 220x55, 58% at 80x24 · *Frequency:* Every visit, every mode  
*Sources:* LY-02, NB-07, LF-01, LY-07, IA-20 · *Verdict:* CONFIRMED / PARTIAL (governance corrections applied)

Evidence:
- `captures/roleplay-character-selected-160x45 rows 1-13, 42-45` (capture; LY-02/LF-01): Nav 3 + header box 5 + purpose 1 + Modes 1 + workbench border 1 + padding 1 + pane border 1; 4 rows below
- `captures/library-conversations-item-open-160x45 rows 1-5` (capture; LY-02): Library: nav 3 + 'Library | Local' + frame; content starts row 6
- `personas_screen.py:1547-1586; css/components/_workbench.tcss:22-30` (code; LY-02/LF-01): DestinationHeader (bordered, auto height), Static#personas-purpose and DestinationModeStrip are three stacked regions
- `css/components/_agentic_terminal.tcss:639-660, 709-714; css/layout/_destination.tcss:20-28` (code; LY-07): Workbench border + padding 1; panes round border + padding; Inspector square border + padding 1 (contradicts DESIGN.md:236 'round borders for panels')
- `css/components/_agentic_terminal.tcss:353-370` (code; missed:fit#1): A one-row rule exists for '#personas-title' but no widget has that id (live header is #personas-header) - drift, not design
- `css/features/_lab.tcss:214-262` (code; LF-01): Console and Lab collapse DestinationHeader into one inline row
- `rp-review/metrics.md section A; layout lens table` (doc; LY-02): Vertical chrome share 47/38/31/58% vs Library 19/20/16/21% at 120/160/220/80
- `captures/roleplay-characters-arrival-160x45 rows 4-10` (capture; IA-20): Mode named twice; 7 rows of chrome copy

Recommendation: see [section 5](#rp-007).

*Verifier corrections applied:* By design per DESIGN.md:107 and ADR-015:34, so an inline DestinationHeader (or an ADR amendment) is required. The workbench border/padding is a shared rule: override for Roleplay only. LY-07's frame rows overlap LY-02 - counted once. The header's Ready/Blocked badge is split out as RP-067. Test pin is test_personas_workbench.py:563, not :548.

<a id="a2-rp-008"></a>
##### RP-008 · The Characters rail stacks seven one-per-row buttons above the list, so the first character sits on row 23 at every size (5 of 28 visible at 120x36)

P0 · severity 3 · usability · scope frame · effort M · confidence high  
*Principle:* H8 Aesthetic and minimalist design; Hick's law (five similar verbs); Gestalt similarity (filters styled as actions) · *Area:* Left rail toolbar and filter bar (all modes; worst in Characters) · *Sizes:* all, including 220x55; 80x24 shows 1 character · *Frequency:* Every Characters visit; Personas at <=160; every mode at 120x36  
*Sources:* LY-03, NB-08, LF-03, LY-21 · *Verdict:* CONFIRMED / PARTIAL (scope corrections applied)

Evidence:
- `personas_library_pane.py:200-242, 147-170, 265-311` (code; LY-03): Characters needs 64 cols for one row (49 label chars + 5x3 chrome); rail content is ~24/35/50 at 120/160/220; switch is all-or-nothing
- `captures/roleplay-character-selected-220x55 rows 14-23` (capture; LY-03): 9 control rows; first item row 23 at 220 cols
- `captures/roleplay-characters-arrival-80x24 rows 13-21` (capture; NB-08): 'New / New / Import / Import / Duplic / Sort: / Tag:'; one list row
- `captures/roleplay-character-selected-160x45.png` (capture; LY-03): Bold centred verbs are the heaviest ink in the rail; Sort/Tag look like actions
- `rp-review/metrics.md` (doc; LY-03): Rows for the list: 10/19/29 (Roleplay) vs 21/30/40 (Library) at 120/160/220
- `captures/roleplay-character-selected-160x45 rows 15-23 vs roleplay-lore-selected-160x45 rows 15-20` (capture; LY-21): First list row jumps between y23 and y20 by mode
- `personas_library_pane.py:330` (code; missed:layout#2): #personas-library-count is a permanent Static that leaves the rail's last row blank

Recommendation: see [section 5](#rp-008).

*Verifier corrections applied:* Dictionaries and Lore fit on one row at 160x45 but stack at 120x36; Personas stacks at 160 too, so the rail changes shape by mode and width (LY-21). The Library's list block is not the compact model (LF-03 correction). The invisible-search half of LF-03 is in RP-009.

<a id="a2-rp-009"></a>
##### RP-009 · The search box hides what you type once it has focus, and the whole list jumps down a row

P0 · severity 3 · bug · scope both · effort S · confidence high  
*Principle:* H1 Visibility of system status; stable layout (DESIGN.md:117, :125 - focus never moves layout) · *Area:* Left rail search · *Sizes:* Characters and Personas at every measured width; all four modes at 120x36 · *Frequency:* Every search (and every F6 into the rail, which lands on the search box)  
*Sources:* NA-03, NB-03, LY-10, KA-09, TF-12, LF-03 · *Verdict:* CONFIRMED (reproduced live by several verifiers)

Evidence:
- `personas_library_pane.py:157-164; css/components/_forms.tcss:105-111 (trap documented at :113-130)` (code; NA-03/LY-10): Stacked mode forces the search to height:1, border:none; app-level Input:focus re-adds a 2-row border, leaving 0 text rows
- `captures/roleplay-characters-search-nomatch-nothing-selected-160x45 rows 15-16` (capture; NA-03): '┌──┐ / └──┘' with no text after typing 'zzqx'
- `captures/roleplay-characters-search-typed-invisible-80x24 rows 8, 13-14` (capture; NA-03): Query invisible; purpose line '· 1'; the result row pushed out of the pane
- `captures/review-rv-layout-search-focused-typed-160x45` (live; LY-10): Typed 'Wren': empty box; list filtered; first row moved y23 -> y24
- `captures/review-rv-a11y-search-typed-invisible-160x45; ansidiff arrival -> f6-step1` (live; KA-09): Rows 15-40 all shift on focus and back on blur
- `captures/review-vf-fit-search-typed-120x36 rows 15-16` (live; LF-03 verification): Reproduced at 120x36
- `rv-a11y f6-step1` (live; missed:a11y#3): Every F6 into the rail in Characters/Personas lands on the search box and triggers the full-rail reflow

Recommendation: see [section 5](#rp-009).

*Verifier corrections applied:* Affected modes depend on width: Characters and Personas everywhere; Dictionaries/Lore stack (and lose the box) only at 120x36.

<a id="a2-rp-010"></a>
##### RP-010 · The test chat's input and buttons clip away: unusable at 120x36, broken after one reply at 160x45

P0 · severity 3 · bug · scope both · effort M · confidence high  
*Principle:* H3 User control and freedom; H1 Visibility; HCI: controls must not depend on content length · *Area:* Characters > 'Try a test chat' pane · *Sizes:* 120x36 unusable from the first paint; 160x45 buttons lost after the first reply · *Frequency:* Every test chat  
*Sources:* LY-14, TF-04, LF-08 · *Verdict:* CONFIRMED

Evidence:
- `tldw_chatbook/Widgets/Persona_Widgets/personas_preview_pane.py:43-50, 63-79, 112-170` (code; LY-14/TF-04): Non-scrolling Vertical, height auto, max-height 60%; transcript max 10; input and button row come after it
- `personas_screen.py:1613-1620` (code; LY-14): The preview is the first child of the work area, above the detail stack
- `captures/roleplay-testchat-open-clipped-120x36 rows 14-27` (capture; LY-14): Provider line, greeting, 'Greeting:' + blank rows, only the input's top border; no buttons
- `captures/review-rv-flows-3-j4-testchat-open-120x36; review-rv-flows-3-j4-testchat-reply-120x36` (live; TF-04): After one reply the input disappears; second message typed blind; reply needed 3 wheel notches
- `captures/review-rv-flows-2-j4-testchat-reply-160x45 rows 14-30; roleplay-testchat-reply-160x45` (live; TF-04): Button row gone after one exchange; wheel does nothing; 'Send to Console draft' reached by blind Tab x3
- `captures/review-rv-flows-2-j4-greeting-select-click-160x45.png` (capture; TF-04): Blank grey 'Greeting:' box with no value or arrow; does not open (Vex has 2 alternates)
- `captures/roleplay-character-editor-160x45 row 14; roleplay-personas-selected-160x45 row 14` (capture; missed:layout#3/fit#5): The toggle takes the top row of the centre in every character state, including the editor

Recommendation: see [section 5](#rp-010).

*Verifier corrections applied:* At 120x36 on first paint the input's top border and placeholder row are visible (it can be typed into); the bottom border and the whole button row are clipped. The greeting row paints as a ~3-4-row blank box, not height 1 as the CSS suggests. LF-08's Try-it half is in RP-006.


#### P1

<a id="a2-rp-011"></a>
##### RP-011 · Personas mode says 'who you play', but a persona is the AI's side, and the 'you' half of J4 has no home: Settings sends 'user profiles' back to Roleplay, which has none

P1 · severity 3 · usability · scope both · effort S · confidence high  
*Principle:* H2 Match between system and the real world; H4 Consistency and standards; H10 Help (circular pointer); information scent · *Area:* Personas mode descriptor, chip tooltip, nav tooltip, persona card; Settings > Roleplay contract copy; the {{user}} display name · *Sizes:* all · *Frequency:* Every Personas visit; every persona Chat now and test chat; every user who wants to set or check how characters address them  
*Sources:* NA-01, TF-17, IA-01, G2-13 · *Verdict:* CONFIRMED / PARTIAL (copy defect per the existing ruling) + gap-round trace of the 'user profiles' half

Evidence:
- `personas_screen.py:449-453` (code; NA-01): _MODE_DESCRIPTORS['personas'] = 'Personas - who you play in the chat.' (purpose line and chip tooltip)
- `personas_screen.py:7378-7382` (code; IA-01): Chat now on a persona sends suggested_prompt 'Respond as {name}.'
- `backlog/decisions/037-...md:12-16; backlog/decisions/149-console-persona-session-identity.md:10-20` (doc; NA-01): Persona = assistant-side profile; persona name becomes the assistant's speaker label
- `persona_profile_editor_widget.py:131-134, 159` (code; NA-01): Core field 'System prompt'; 'Persona Visual operational states' (idle, thinking, speaking...)
- `captures/review-rv-nng-a-persona-testchat-speaker-160x45 rows 9, 15-18; review-rv-nng-a-persona-chat-now-console-160x45 rows 7, 41` (live; NA-01): Purpose line 'who you play'; reply follows the persona's system prompt; Console 'Chat with Default Narrator' / 'Persona: Default Narrator'
- `captures/review-rv-flows-2-j7-persona-selected-160x45 rows 9, 16-20` (capture; TF-17): 'who you play' beside an assistant system prompt
- `tldw_chatbook/UI/Navigation/shell_destinations.py:84-86` (code; missed:ia#4): Nav purpose/tooltip: 'user profiles', 'Manage behavior profiles and user profile context.'
- `Chat/console_roleplay_identity.py:94-99, 121-139, 155-200` (code; G2-13): {{user}} in character and persona prompts resolves per-chat override -> global chat_defaults.user_display_name -> 'User'
- `Widgets/Console/console_settings_modal.py:2095-2115; UI/Screens/settings_screen.py:19313-19330` (code; G2-13): Per-chat: Console > Chat settings > collapsed 'Conversation identity' > 'Your name in this chat'; global: Settings > Console Behavior > 'Default chat display name'
- `config.py:9869-9876; Character_Chat/Character_Chat_Lib.py:572-582` (code; G2-13): [general] users_name is the data-profile folder, not a chat name (the Console still labelled the human 'User'); no user description exists, and {{persona}} deliberately resolves to the AI's name (task-442)
- `rv-gap2-3 220x55: Settings > Domain Defaults > Roleplay (captures/review-rv-gap2-3-settings-roleplay-user-profiles-copy-220x55 rows 16-26; settings_screen.py:1665-1683)` (live; G2-13): 'Source 1: Your saved characters and user profiles' · 'Picking a character or user profile, and sending it to Console' · 'Writes allowed: No - change this in Roleplay instead'
- `captures/review-rv-gap2-3-settings-my-profile-220x55` (live; G2-13): A third 'you' surface (Settings > My Profile, the Personal Context Profile, ADR-102), linked from neither Roleplay nor Console identity
- `grep user_display_name|users_name over UI/Screens/personas_screen.py, UI/Persona_Modules/, Widgets/Persona_Widgets/` (code; G2-13): Roleplay never reads or links the display name

Recommendation: see [section 5](#rp-011).

*Verifier corrections applied:* Verifiers: the F-034 wording was chosen deliberately and TASK-19577 endorses it, but the owner's ruling (TASK-617:19; ADR-037, ADR-149) is assistant-side, so this is a copy defect, not an open question. Nav copy in shell_destinations.py is part of the same defect (missed ia#4), not independent evidence. Gap round (G2-13) merged here and tied to D3 rather than filed as a separate P1: its new value is the circular Settings > Roleplay copy and the missing display-name route. The owner's 'user profiles' may itself reflect the persona-means-user confusion this finding describes. Settings > My Profile (ADR-102) is a separate domain from {{user}}.

<a id="a2-rp-012"></a>
##### RP-012 · Character attachments are silent frozen copies, and the lore/dictionary side says 'Not attached' even when a character holds one

P1 · severity 3 · usability · scope both · effort M · confidence high  
*Principle:* H1 Visibility of system status; H6 Recognition rather than recall; PRODUCT.md:25 consequences without guesswork · *Area:* Character World Books / Dictionaries sections; Lore and Dictionary 'Attachments' tabs · *Sizes:* all · *Frequency:* Every world-info edit after an attach; every 'who uses this?' question  
*Sources:* NA-02, TF-02, IA-05, NB-04, TF-09 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

Evidence:
- `tldw_chatbook/Character_Chat/world_book_manager.py:823-862` (code; NA-02/TF-02): attach_world_book_to_character embeds an export snapshot; idempotent by name, so re-attach never refreshes
- `Character_Chat/local_chat_dictionary_service.py:1134-1168, 1206-1232` (code; NA-02 verification): Dictionaries attached to characters are snapshots too
- `personas_screen.py:6273-6290` (code; NA-02): The attach picker hides names already attached, so it cannot refresh a stale copy
- `personas_character_world_books.py:86-93; personas_character_dictionaries.py:86-94` (code; IA-05): Only disclosure is a hover tooltip; task-2231 folded away the visible '(copied into this character)' label the guide still promises (lore-books.md:128-131)
- `personas_lore_detail.py:226-245; personas_dictionary_detail.py:324-347; personas_screen.py:5868-5888` (code; NB-04/TF-09): Attachments tabs list conversations only
- `captures/review-rv-nng-a-isolde-worldbook-stale-snapshot-160x45 rows 36-39` (live; NA-02): After deleting an entry (source '5 entries'), Isolde still lists 'Station Kepler-9 Ops Manual 6'
- `captures/review-rv-flows-2-j6-lore-tryit-result-160x45; review-rv-flows-2-j6-isolde-worldbook-snapshot-160x45` (live; TF-02): Source has 7 entries and Try it fires the new one; Isolde's copy shows 6
- `captures/review-rv-nng-b-dict-attachments-tab-160x45 rows 22-23; review-rv-ia-lore-attachments-kepler-160x45; review-rv-ia-vex-attachments-expanded-160x45 rows 34-37` (capture; NB-04/IA-05): 'Not attached to any conversation yet.' although Vex/Isolde hold copies
- `captures/roleplay-character-selected-scrolled-160x45 rows 39-41` (capture; NB-04): Character-side attachments are two collapsed rows below Edit at the bottom of a long scroll

Recommendation: see [section 5](#rp-012).

*Verifier corrections applied:* Snapshot semantics are deliberate and documented (P2f spec :9, :28; lore-books.md:128-133) - a harmful design decision, not a bug. Dictionaries attached to characters are snapshots too ('live vs copy' contrast dropped). Deleting the source does not break characters (they hold copies). 'Chats keep injecting the old copy' was false: see RP-004 - character copies appear not to be applied at all.

<a id="a2-rp-013"></a>
##### RP-013 · A lorebook embedded in an imported card is invisible in Lore mode and cannot be viewed, edited or tested

P1 · severity 3 · gap · scope content · effort M · confidence high  
*Principle:* H6 Recognition rather than recall; H4 Consistency (one kind of object, one home) · *Area:* Import > Lore mode · *Sizes:* all · *Frequency:* Every card import with an embedded character_book (common for community cards)  
*Sources:* TF-03 · *Verdict:* CONFIRMED

Evidence:
- `tldw_chatbook/Character_Chat/Character_Chat_Lib.py:1788-1822` (code; TF-03): Embedded character_book is converted into a snapshot on the character; no standalone world book is created
- `rv-flows 160x45 empty profile: Import ilsa_with_book.v2.json, then l` (live; TF-03): Toast 'Lorebook 'Glass Coast Lore' attached (2 entries)'; Lore shows 'No lore books yet' and '· 0' (captures/review-rv-flows-j2-lore-mode-after-import-160x45 rows 9-21)
- `captures/review-rv-flows-j2-worldbooks-expanded-160x45 rows 38-41` (capture; TF-03): Only a read-only 'Glass Coast Lore 2' row below Edit/Conversations

Recommendation: see [section 5](#rp-013).

<a id="a2-rp-015"></a>
##### RP-015 · An invalid regex is saved silently; its warning list renders with zero rows while the Inspector says 'Validation: OK'

P1 · severity 3 · bug · scope content · effort S · confidence high  
*Principle:* H9 Help users recognize, diagnose, and recover from errors · *Area:* Dictionaries > Entries > Add/Update, advisory list, Inspector validation line · *Sizes:* all · *Frequency:* Whenever a regex is mistyped  
*Sources:* NB-02 · *Verdict:* CONFIRMED (reproduced live)

Evidence:
- `rv-nng-b 160x45: Fantasy Anachronism Swaps > pattern '(unclosed' > regex on > Add` (live; NB-02): No message; rail '8 entries' -> '9 entries'; app log: 'Invalid regex ... Treating as literal string'
- `captures/review-rv-nng-b-dict-validation-panel-220x55.png` (capture; NB-02): Advisory list is a 2-row bordered strip with zero content rows; Inspector 'Validation: OK'
- `personas_dictionary_detail.py:159-162, 268, 524-555, 610-680` (code; NB-02): OptionList height auto max 5 with border; options rendered '[code] pattern - message'; no compile check in form_payload
- `personas_inspector_pane.py:527-533; personas_screen.py:5337, 5428` (code; NB-02): The Inspector summary is never fed dictionary findings

Recommendation: see [section 5](#rp-015).

*Verifier corrections applied:* Saving without blocking is deliberate (personas_dictionary_validation.py:1 'Warn-not-block validation (Roleplay P1c)'); the defect is that the warning is invisible and the Inspector says OK.

<a id="a2-rp-016"></a>
##### RP-016 · The character editor is one long scroll of small fixed-height fields with no sections, so 2.5 fields fit at 120x36 and a wider canvas would not help

P1 · severity 3 · usability · scope content · effort M · confidence high  
*Principle:* Progressive disclosure; H8 Aesthetic and minimalist design; form scanning · *Area:* Character editor (and persona editor) · *Sizes:* 120x36 worst (2.5 fields), 160x45 (5 fields); still fixed heights at 220x55 · *Frequency:* Every edit session  
*Sources:* LY-11, LF-06, missed:nng-a#3 · *Verdict:* CONFIRMED + verifier-sourced evidence

Evidence:
- `personas_character_editor_widget.py:98-112, 400-711` (code; LY-11/LF-06): Fixed heights (3/4/3/3); one VerticalScroll; only Advanced collapses
- `captures/roleplay-character-editor-160x45 rows 16-37; roleplay-character-editor-120x36 rows 16-29` (capture; LY-11): 5 fields at 160x45; 2.5 at 120x36
- `captures/roleplay-character-editor-scrolled-160x45.png` (capture; LY-11): ~7 empty rows below 'Avatar: none' while text fields stay one line tall
- `captures/review-rv-nng-a-editor-unsaved-160x45 rows 15-25` (capture; missed:nng-a#3): With World Books expanded below Save/Cancel, only Name and a 1-row First message are visible
- `css/features/_library.tcss:1293-1312; Widgets/Library/library_prompts_canvas.py:1319-1345` (code; LF-06): Library fill-height primary field; Basic/Advanced/Info mode strip

Recommendation: see [section 5](#rp-016).

<a id="a2-rp-017"></a>
##### RP-017 · Width goes to the side panes, not to what is open: a fixed 2:4:2 split, no collapse priority, no 'work-first' while editing, and blank panes at large sizes

P1 · severity 3 · usability · scope both · effort L · confidence high  
*Principle:* HCI: allocate space to the primary task; H7 Flexibility and efficiency; readable line length · *Area:* Pane width model · *Sizes:* 120x36-160x45 (editor ~half the width); 200+ (blank Inspector, 100-185-char lines) · *Frequency:* Constant  
*Sources:* LF-04, LF-05, LY-06, LY-17 · *Verdict:* CONFIRMED / PARTIAL (LY-06 rated 2 on its own)

Evidence:
- `css/components/_agentic_terminal.tcss:709-734` (code; LF-04/LY-06): Inspector 2fr (min 30), Library rail 2fr (min 24), work 4fr (min 40); no max-width
- `personas_screen.py:2316-2393` (code; LF-04): Only compact (<=90) and narrow (<=60) breakpoints; all three panes open from 91 cols
- `personas_screen.py:15909-15975` (code; LF-05): _show_center never changes pane allocation when an editor opens
- `captures/roleplay-character-editor-120x36.png` (capture; LF-05): Editor 58 cols beside a 28-col list and a 30-col Inspector with Export/Delete/Buddy
- `captures/roleplay-character-selected-220x55 row 13; review-rv-layout-rails-collapsed-220x55` (capture; LY-06): Panes 54|108|54; Inspector 85% blank (21 of 38 rows inked); prose ~110 cells, ~185 collapsed
- `captures/roleplay-characters-nothing-selected-160x45` (capture; LY-17): Nothing selected: centre 97% blank, Inspector 89%
- `pure resolver run (scratch copy of Utils/adaptive_reader_state.py + library_rail_width.py)` (live; LF-04): Library default work width 50/45/62 at 120/140/160 vs Roleplay today 58/78; authoring profile 106 at 120 and 90-146 at 160
- `tldw_chatbook/UI/Library_Modules/library_notes_controller.py:2446-2473; Docs/User_Guide/library.md:313-318` (code; LF-05): Library's transient work-first override; manual reopen wins
- `tldw_chatbook/Utils/adaptive_reader_state.py:519-528` (code; LY-17): Empty work pane donates its width

Recommendation: see [section 5](#rp-017).

*Verifier corrections applied:* At 120 cols the authoring profile also closes the items list (so '106 cols for the editor' means the list is closed too). LY-06 alone is an aesthetic cost (severity 2); merged here because the root cause is the same fr split.

<a id="a2-rp-018"></a>
##### RP-018 · The Inspector column mixes per-item actions, app-wide Buddy controls and wrong-mode advice in ten equal one-row buttons

P1 · severity 3 · usability · scope both · effort L · confidence high  
*Principle:* H8 Aesthetic and minimalist design; Hick's law; Gestalt proximity; visibility of the primary action · *Area:* Inspector (right column) · *Sizes:* all; width cost at 120-160, blank column at 200+ · *Frequency:* Every selection  
*Sources:* LF-07, NB-10, IA-10, LY-12, KA-14, NA-22 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

Evidence:
- `personas_inspector_pane.py:212-363` (code; LF-07): Selected/Type/avatar, Validation, conversations, Readiness, Chat now/Send/Export x3/Delete, 4 Buddy buttons
- `personas_inspector_pane.py:1145-1159` (code; NB-10/IA-10): All four Buddy buttons forced display=True in every state
- `personas_inspector_pane.py:35, 1005-1011` (code; NB-10): 'Pick a character or persona to start chatting.' in every mode; dictionary/lore get 'Console chat is for characters and personas.'
- `personas_inspector_pane.py:123-134` (code; LY-12): Every button width 100%, height 1, border none, no gaps
- `personas_inspector_pane.py:1080-1143` (code; KA-14): Disabled reasons are tooltip-only
- `captures/roleplay-dictionaries-selected-160x45 rows 15-27; roleplay-dictionaries-nothing-selected-160x45 rows 16-23` (capture; NB-10): For a dictionary, 1 of 13 rows (Delete) is relevant
- `captures/roleplay-character-selected-160x45 rows 27-36` (capture; LY-12): 10 consecutive centred buttons; Delete between Export Actor Pack and Manage Buddy; 26 buttons on screen (13 item actions) vs 6 in the Library prompt view
- `captures/review-rv-a11y-arrival-focus-160x45.ansi rows 27-34` (capture; KA-14): Disabled vs enabled background differ 1.17:1; no reason text; Tab skips disabled buttons
- `backlog/decisions/148-mcp-hub-rail-ia-and-responsive-triad.md:28-36, 57-60; Utils/adaptive_reader_state.py:39` (doc; LF-07): ADR-148 (Proposed) stacks the inspector below 120 cols and rejects toggle-hiding; the resolver models only two optional panes

Recommendation: see [section 5](#rp-018).

*Verifier corrections applied:* The Buddy controls are deliberately global (independent Buddy, ADR-139; commits 231a8533ea/45fced8834) - noise in the item Inspector, not a bug. DESIGN.md:363 accepts tooltips as a channel, but DESIGN.md:287-288 wants a text annotation and keyboard users never see tooltips on disabled buttons. The tonal disabled style (_buttons.tcss:48-60) is deliberate. NA-22's toolbar-verb half is in RP-043.

<a id="a2-rp-019"></a>
##### RP-019 · At 120x36, 'Chat now' falls below the Inspector's fold whenever the character has a portrait, with no 'more below' cue

P1 · severity 3 · usability · scope both · effort S · confidence high  
*Principle:* H1 Visibility; F-pattern (primary action in the least-scanned spot); Fitts's law · *Area:* Inspector primary call to action · *Sizes:* 120x36 (hidden); 160x45 (Buddy hidden); <=100x30 hidden for every character · *Frequency:* Every character with a portrait (typical for imported PNG cards)  
*Sources:* LY-13 · *Verdict:* CONFIRMED (reproduced live)

Evidence:
- `personas_inspector_pane.py:78-87, 212-300` (code; LY-13): Avatar box (max-height 10), validation, conversations, readiness, then actions, in one VerticalScroll
- `captures/review-rv-layout-2-portrait-cta-below-fold-120x36` (live; LY-13): Portrait rows 18-28, Validation 29, Conversations 30-32; Readiness and Chat now below; one '▂' glyph as the only hint
- `captures/review-rv-layout-portrait-inspector-160x45` (live; LY-13): Chat now row 35; Buddy below the fold
- `DESIGN.md:301-304` (doc; LY-13): Scrolling inspectors must reserve a '▼ more' fold row

Recommendation: see [section 5](#rp-019).

*Verifier corrections applied:* At 160x45 Delete is visible (row 40); only the Buddy buttons are hidden.

<a id="a2-rp-020"></a>
##### RP-020 · Per-item actions live in three different places and move by mode; dictionary and lore Export sit at the bottom of an unscrollable tab

P1 · severity 3 · usability · scope both · effort M · confidence high  
*Principle:* H4 Consistency and standards; H6 Recognition (where is the action?) · *Area:* Duplicate (rail), Chat/Export/Delete (Inspector), Edit/Conversations (card bottom), dictionary/lore Export (Settings tab) · *Sizes:* all; dictionary/lore Export unreachable at <=160x45 · *Frequency:* Every export, duplicate, edit or delete  
*Sources:* IA-08, LF-17 · *Verdict:* CONFIRMED (reproduced live)

Evidence:
- `personas_library_pane.py:291-296` (code; IA-08): Duplicate (an item action) in the rail's collection toolbar
- `personas_inspector_pane.py:51-52; personas_dictionary_detail.py:288-300; personas_lore_detail.py:222-226` (code; IA-08): Inspector export only for characters/personas; dictionary/lore export at the bottom of Settings
- `personas_character_card_widget.py:124-139; personas_screen.py:7077-7090` (code; LF-17): Edit and 'Conversations (N)' at the card bottom
- `captures/roleplay-character-selected-160x45 rows 20, 27-32, 41` (capture; IA-08): Three regions for one character's actions
- `captures/roleplay-personas-selected-160x45` (capture; LF-17): Persona Edit below the fold with rows 23-41 blank (partly RP-001)
- `Widgets/Library/library_prompts_canvas.py:1294-1375, 1628-1720` (code; LF-17): Library work-pane header and state-machine action bar

Recommendation: see [section 5](#rp-020).

*Verifier corrections applied:* IA-08 verifier: 'Save settings' is also clipped (moved to RP-005). Library's 'Use in Console' sits mid-pane and 'More actions' at the pane bottom, so 'one bar at the top' is a design choice, not Library precedent.

<a id="a2-rp-021"></a>
##### RP-021 · Returning to Roleplay restores a broken state: an empty list and '· 0' when nothing was selected, or a hidden leftover search filter

P1 · severity 3 · bug · scope both · effort S · confidence high  
*Principle:* H1 Visibility of system status; H9 Help users recover (it looks like data loss) · *Area:* Screen restore on leave/return · *Sizes:* all · *Frequency:* Every leave/return while a non-Characters mode has no selection (e.g. right after switching modes); every return after searching  
*Sources:* TF-07, TF-12 · *Verdict:* CONFIRMED (both halves reproduced live by the verifier)

Evidence:
- `personas_screen.py:1808-1817, 1819-1823` (code; TF-07): No selected id -> _pending_restore = None -> _apply_pending_restore returns before _apply_mode(saved_mode)
- `personas_screen.py:1797-1806, 4476, 13935` (code; TF-12): state.search_query is restored, but the search Input value is only ever cleared
- `captures/review-rv-flows-j8-return-personas-count0-220x55.png; review-rv-flows-j8-return-lore-empty-220x55 rows 9-26` (live; TF-07): Empty list, '· 0', Characters-only test-chat toggle in Lore; active chip dead
- `captures/review-vf-flows-tf07-return-personas-empty-160x45` (live; TF-07 verification): Personas '· 4' -> Library -> Roleplay: '· 0' and an empty list
- `captures/review-vf-flows-tf12-return-hidden-filter-160x45` (live; TF-12 verification): Box reads 'Search...', list shows only Chef Auguste, 'Sort: Relevance', '· 1'

Recommendation: see [section 5](#rp-021).

<a id="a2-rp-022"></a>
##### RP-022 · A search with no matches says 'No characters yet - use New or Import', and filtered counts never say 'of 28'

P1 · severity 3 · bug · scope both · effort S · confidence high  
*Principle:* H9 Help users recognize and recover from errors; H5 Error prevention; H1 · *Area:* Rail list empty/no-match state and count line · *Sizes:* all · *Frequency:* Every failed search  
*Sources:* NA-04, NB-16, IA-07, LF-14, TF-12, G3-11 · *Verdict:* CONFIRMED

Evidence:
- `personas_library_pane.py:431-442` (code; NA-04/NB-16): Empty branch always renders 'No {noun} yet - {hint} to add one.' and ignores `filtered`
- `personas_screen.py:4329-4335, 4413 vs 3690, 4021` (code; LF-14): Dictionaries/Lore pass filtered; Characters/Personas never do
- `captures/roleplay-characters-search-nomatch-nothing-selected-160x45 rows 9, 23-24` (capture; NA-04): 28 characters exist: '· 0' and 'No characters yet - use New or I...'
- `captures/roleplay-characters-search-nomatch-120x36 rows 9, 22-24, 29` (capture; NB-16): Three conflicting messages: '· 0', 'No characters yet', 'Pick a character from the list'
- `captures/roleplay-character-selected-80x24 rows 8, 13` (capture; IA-07): With search 'Isolde', purpose line silently '· 1'
- `library_prompts_canvas.py:62-63, 623; captures/library-prompts-item-open-160x45 row 29` (code; LF-14): Library: 'No prompts match your filter.'; '1-4 of 4'
- `captures/review-rv-gap3-search-clockwork-paged-160x45 rows 9, 40` (capture; G3-11): 348 characters: 'Characters — who the AI plays · 85' and '1-50 of 85 characters'; nothing says 'matching' or 'of 348'
- `captures/review-rv-gap3-tag-fantasy-applied-160x45 rows 9, 22` (capture; G3-11): '· 20' with 'Tag: fantasy'; no '20 of 348'
- `personas_screen.py:3690-3696; personas_library_pane.py:493-503` (code; G3-11): The page-bar label uses the filtered total; characters never pass filtered=True

Recommendation: see [section 5](#rp-022).

*Verifier corrections applied:* Severity 3 holds mainly because the query is also invisible (RP-009); on its own this would be 2. The 'N of M' count already exists for Dictionaries/Lore (filtered=bool(needle)); Characters/Personas never pass `filtered` (LF-14 correction). The '· 0 while loading' half is RP-028. Gap round (G3-11): the same count defect at 348 cards, now also in the page-bar label and for tag filters; folded in as volume evidence.

<a id="a2-rp-023"></a>
##### RP-023 · Example dialogue (mes_example) is stored and sent to the model, but cannot be seen or edited anywhere

P1 · severity 3 · gap · scope content · effort S · confidence high  
*Principle:* H2 Match with the real world (card-format vocabulary); completeness of the authoring model · *Area:* Character card and editor · *Sizes:* all · *Frequency:* Every imported community card (most carry example dialogue); every in-depth authoring session  
*Sources:* TF-15 · *Verdict:* CONFIRMED

Evidence:
- `personas_character_editor_widget.py; personas_character_card_widget.py; personas_screen.py:8103; personas_preview_controller.py:96` (code; TF-15): grep message_example/mes_example: only the duplicate path and the preview system prompt; no card row or editor field anywhere in the app
- `rv-flows 160x45 empty profile: Import ser_corwin.v2.json` (live; TF-15): Card shows Description, Personality, Scenario, First message, Creator, Version, Tags, Alternate greetings - no example dialogue (captures/review-rv-flows-j2-after-json-import-160x45 rows 15-35)

Recommendation: see [section 5](#rp-023).

<a id="a2-rp-024"></a>
##### RP-024 · Keyboard focus escapes the screen: arrival focus sits on the nav bar's '⌃1 Home' (Enter leaves Roleplay), and Tab walks 14 nav buttons on every cycle

P1 · severity 3 · bug · scope frame · effort S · confidence high  
*Principle:* H3 User control and freedom; H4 Consistency; WCAG 2.4.3 Focus Order · *Area:* Frame: arrival focus and Tab traversal · *Sizes:* all · *Frequency:* Every visit (arrival); every Tab cycle  
*Sources:* KA-01, KA-02, missed:a11y#4 · *Verdict:* CONFIRMED (both reproduced live)

Evidence:
- `personas_screen.py:1083-1084` (code; KA-01): No tab/shift+tab override, so Textual's Screen default walks the whole app focus chain, nav bar included
- `library_screen.py:1010-1021, 8975-9008` (code; KA-01): Library re-binds tab/shift+tab and confines focus to '#screen-content, #screen-content *'
- `rp-review/a11y/steps/rv-a11y-tab2/4..20.ansi` (live; KA-01): After 'Show Buddy', Tab #5 focused '⌃1 Home', then 14 nav stops; Library: 45 Tabs, 0 nav stops
- `personas_screen.py:1878-1924; UI/Navigation/main_navigation.py:745-760` (code; KA-02): _auto_select_first_library_row never moves focus; App AUTO_FOCUS='*' lands on the first nav button
- `captures/review-rv-a11y-arrival-focus-160x45 rows 1-2; verifier vf-a11y` (live; KA-02): '⌃1 Home' focus-styled; Enter immediately -> header 'Home | Ready - Local'
- `captures/review-rv-a11y-dict-mode-kbd-160x45 row 2` (capture; missed:a11y#4): Nav strip stays scrolled showing a clipped '‹ le'; Home and Console hidden at 160 cols

Recommendation: see [section 5](#rp-024).

<a id="a2-rp-025"></a>
##### RP-025 · F6 (next pane) lands on secondary controls and never reaches the dictionary or lore editors

P1 · severity 3 · bug · scope both · effort M · confidence high  
*Principle:* H7 Flexibility and efficiency; WCAG 2.1.1 Keyboard; DESIGN.md:117 keyboard-first · *Area:* F6 pane cycle · *Sizes:* all · *Frequency:* Every F6 press  
*Sources:* KA-03 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:1122-1155, 4490, 1613; Widgets/workbench_focus.py:66-79` (code; KA-03): Work-area targets are all inside the preview pane, which is hidden in Dictionaries/Lore; the non-focusable work area is then dropped from the cycle
- `rp-review/a11y/steps/rv-a11y-dictf6 0-4` (live; KA-03): F6 alternates search <-> Inspector ' > ' only
- `rv-a11y f6-step2, persona-f6, tab3` (live; KA-03): Work F6 -> '▸ Try a test chat'; Inspector F6 -> empty conversations list or collapse button; opening the editor took F6 + 5 Tabs (2 invisible)
- `Docs/User_Guide/roleplay-chat-dictionaries.md:59-75` (doc; KA-03): The guide says F6 cycles through the centre

Recommendation: see [section 5](#rp-025).

<a id="a2-rp-026"></a>
##### RP-026 · Escape does not leave text fields, so the next advertised mode key is typed into the field

P1 · severity 3 · gap · scope both · effort S · confidence high  
*Principle:* H3 User control and freedom (clearly marked exit); H4 Consistency with Library; WCAG 2.1.2 (soft trap) · *Area:* Escape semantics · *Sizes:* all · *Frequency:* Every time a user types and then wants a hotkey  
*Sources:* KA-04, NA-11, missed:a11y#1 · *Verdict:* CONFIRMED (reproduced live)

Evidence:
- `personas_screen.py:16318-16351` (code; KA-04/NA-11): action_personas_escape handles only pending resume, editor cancel, transcript close and focus id 'personas-library-search'
- `captures/review-rv-a11y-esc-in-testchat-then-p-160x45; review-rv-a11y-tryit-esc-then-c-160x45` (live; KA-04): 'p' inserted into the test input; 'c' into the Try-it sample; footer still advertises c/p/d/l
- `captures/review-rv-nng-a-2-escape-does-not-leave-tryit-120x36` (live; NA-11): Box reads 'my colourp'; mode unchanged
- `vf-a11y 160x45` (live; KA-04 verification): 'hi', Escape, 'p' -> 'hip'
- `Docs/User_Guide/roleplay-chat-dictionaries.md (Keyboard section)` (doc; missed:a11y#1): 'press Escape first if a field has focus' before c/p/d/l - fails outside search
- `library_screen.py:1128, 27190-27212` (code; KA-04): Library's library_blur_text_field

Recommendation: see [section 5](#rp-026).

*Verifier corrections applied:* NA-11's footer half is in RP-034.

<a id="a2-rp-027"></a>
##### RP-027 · Tab stops are invisible or off-screen: nested scroll containers, a blank greeting picker, entry-form controls scrolled out of view, and a Generate button before every field

P1 · severity 3 · bug · scope content · effort M · confidence high  
*Principle:* WCAG 2.4.7 Focus Visible and 2.4.11 Focus Not Obscured; H1 Visibility · *Area:* Work pane content (card, test chat, dictionary/lore forms, character editor) · *Sizes:* all; worse at 120x36 · *Frequency:* Every keyboard pass through the card, test chat, editor or dictionary form  
*Sources:* KA-07, TF-20 · *Verdict:* PARTIAL (corrections applied)

Evidence:
- `personas_screen.py:1634; personas_character_card_widget.py:100; personas_preview_pane.py:124, 131` (code; KA-07): VerticalScroll#personas-detail-stack wraps VerticalScroll#personas-character-card-body; preview transcript scroll; greeting Select
- `rp-review/a11y/steps tab1 #2-#4, tab3 #1-#2, tabchat #1-#3` (live; KA-07): Two invisible stops, then the card scrolls with no cue
- `rp-review/a11y/steps dicttab #7-#12; captures/review-rv-a11y-dict-entry-form-focus-offscreen-160x45.png` (live; KA-07): Six consecutive Tabs with no visible change; no focus indicator anywhere
- `rv-flows 160x45 J1: Name > Tab` (live; TF-20): First Tab moves to the First-message 'Generate' button with no visible change

Recommendation: see [section 5](#rp-027).

*Verifier corrections applied:* The Inspector VerticalScroll does show focus (border turns blue) - dropped. The greeting row is already hidden when there are no alternates; the bug is that it paints blank when alternates exist. TF-20's post-save half is in RP-038.

<a id="a2-rp-047"></a>
##### RP-047 · Both 'Send to Console draft' hand-offs are inert: the card or test-chat transcript never reaches the model, yet stays shown as staged and follows you into later chats

P1 · severity 3 · bug · scope content · effort M · confidence high  
*Principle:* H1 Visibility of system status; H2 Match (say what will happen); H4 Consistency (one label, several meanings); H5 Error prevention · *Area:* Inspector 'Send to Console draft' (ctrl+enter) and test-chat 'Send to Console draft' -> Console staged sources; hand-off verbs · *Sizes:* all · *Frequency:* Every use of either 'Send to Console draft'  
*Sources:* TF-06, IA-12, G1-04 · *Verdict:* CONFIRMED (gap round, logged prompt bodies; raised from P2/2 to P1/3)

Evidence:
- `personas_preview_controller.py:539-552` (code; TF-06): Test-chat hand-off stages only the transcript as 'Personas preview conversation' with 'Continue this conversation in character.'
- `captures/review-rv-flows-2-j4-handoff-result-160x45 rows 4, 38-42; review-rv-flows-2-j7-persona-chat-console-160x45 rows 39-43` (live; TF-06): 'No current character', 'Assistant: General'; leftover source in the later Cozy Storyteller session
- `personas_inspector_pane.py:287, 294, 1037-1094; personas_screen.py:7378-7387, 1655-1662` (code; IA-12): No tooltip on enabled Chat now/Send; toasts 'Chat staged in Console.' vs 'Staged in Console.'; 'Resume chat', 'Send transcript to Console draft'
- `library_conversation_reader.py:288; library_media_viewer.py:435` (code; IA-12): Library: 'Resume conversation', 'Use in Console'
- `rv-gap1-3 160x45, capture off: Vex test chat, one message, Tab x3 + Enter, then send the pre-filled draft` (live; G1-04): Body 20:45:34: generic agent prompt plus 'Continue this conversation in character.'; the test transcript ('Call the police…', 'Vex', 'Emberfall') is absent
- `rv-gap1-3: Characters > Isolde > Inspector 'Send to Console draft' > Enter` (live; G1-04): Body 20:46:57: no card text ('Meridian' absent); only 'Use Captain Isolde Varga to guide the next response.'
- `rp-review/gapcheck/rv-gap1-3-app.log lines 428, 463, 506` (live; G1-04): Every send logs 'Console RAG evidence excluded; count=1; reason=not_canonical_local_evidence'
- `personas_screen.py:7134-7185; Event_Handlers/Chat_Events/chat_rag_events.py:1686-1709; RAG_Search/local_citation_capture.py:30-41, 377-420` (code; G1-04): Hand-offs are a ChatHandoffPayload(source='personas', '{kind}-card' / 'preview-conversation'); references that do not normalize to media, note or conversation are dropped (exclusion dates from 33507a93db, 2026-07-26)
- `Chat/console_runtime.py:2139-2149, 2178-2195` (code; G1-04): Staged evidence is one runtime-wide slot, released only when captured context is non-empty, so an excluded source is never released
- `captures/review-rv-gap1-3-inspector-draft-sent-160x45 rows 14, 24-26, 38-41; review-rv-gap1-3-persona-send-with-leaked-source-160x45 rows 39-41` (capture; G1-04): 'Staged for next send · 1 source' / 'Sources: 1' after the send; the leaked preview source survived into a new Cozy Storyteller chat until Un-stage
- `captures/review-rv-gap1-isolde-selected-120x36 row 36` (capture; G1-04): The footer's only hand-off accelerator is 'ctrl+enter Send to Console draft'

Recommendation: see [section 5](#rp-047).

*Verifier corrections applied:* The Inspector's 'Send to Console draft' also binds no character (it stages the card as a source) and the shared label was deliberate (task-2232). Staged-source persistence is Console staging scope. Gap round (G1-04): raised to severity 3 because a visible action is silently a no-op while the UI claims a source is staged. The leak does not block (a persona send with the leaked source completed with capture off); the earlier rv-flows-2 block was RP-072.

<a id="a2-rp-056"></a>
##### RP-056 · Lore and dictionary entries cannot be found at realistic volume: search matches book names only, entry tables have no filter and stop at 12 rows (entry #200 of 250 took 99 wheel notches)

P1 · severity 3 · gap · scope both · effort M · confidence high  
*Principle:* H7 Flexibility and efficiency of use; H6 Recognition rather than recall; scroll cost · *Area:* Rail search; Lore and Dictionaries > Entries tab (no entry filter; table capped at 12 rows) · *Sizes:* all: 4 entry rows at 120x36; 10 at 160x45 on a clean visit (3 with the RP-001 band); 10 at 220x55 (the cap does not grow with height) · *Frequency:* Every edit of a specific entry in a book above ~20 entries; every 'which book defines X?' question  
*Sources:* NB-14, LF-21, G3-02 · *Verdict:* CONFIRMED at realistic volume (gap round G3-02; raised from P2/2 to P1/3)

Evidence:
- `personas_screen.py:4308-4312, 4390-4394` (code; NB-14): Dictionary and lore filters match record name only
- `personas_dictionary_detail.py:142-148, 203` (code; NB-14): Entry table min 6/max 12 rows, no filter input
- `personas_screen.py:4466-4477; Widgets/Library/library_rail.py:1099-1126` (code; LF-21): Search cleared per mode; Library two-level search
- `rv-gap3 160x45, Lore > Grand Codex (250 entries), find #200 'Saltmarsh Bellwarden' by wheel` (live; G3-02): 99 notches through a 3-row window (RP-001 band present); after Update, 16 notches up plus 99 down to re-find it; 7 Tabs through 6 off-screen controls to reach Content (RP-003)
- `captures/review-rv-gap3-lore-grand-selected-160x45 rows 24-27; review-rv-gap3-lore-200-found-by-wheel-160x45 rows 25-27` (capture; G3-02): Header plus 3 of 250 rows at 160x45 with the leaked band (rows 14-20)
- `captures/review-rv-gap3-lore-grand-clean-120x36 rows 17-21; review-rv-gap3-lore-grand-clean-220x55 rows 17-27` (capture; G3-02): Clean first visits: 4 rows at 120x36, 10 at 220x55
- `captures/review-rv-gap3-dict-grand-selected-160x45 rows 18-28; review-rv-gap3-dict-grand-form-validation-160x45 rows 17-24` (capture; G3-02): 150-rule dictionary: 11 rows visible, 70 notches to rule #150; the wheel scrolls without moving the cursor, so the form below still holds rule #1
- `personas_lore_detail.py:98-101; personas_dictionary_detail.py:145-148; css/widget_defaults_self.tcss:2552, 2775` (code; G3-02): Entry tables min-height 6 / max-height 12; no filter input; table and form share one VerticalScroll

Recommendation: see [section 5](#rp-056).

*Verifier corrections applied:* LF-21 over-counted (Kepler-9 lives in two places, and character search already matches descriptions); cross-mode search is a nice-to-have. Gap round (G3-02): a clean first Lore visit at 160x45 shows 10 entry rows; the 3-row window at volume exists only with the RP-001 band. DataTable has PageDown/End/Ctrl+End (textual _data_table.py:279-284), so 99 notches is the mouse-only worst case: the cost comes from distance, not window size. Raised to severity 3 because at realistic volume entries cannot be found, and their position cannot be kept.

<a id="a2-rp-074"></a>
##### RP-074 · After a refused send, the Console's recovery card never appears, so 'Send without capture', 'Retry' and 'Cancel' are unreachable and the chat is a dead end (TASK-33621.2 AC#2/#3)

P1 · severity 3 · bug · scope external (Console; TASK-33621.2 AC#2/#3) · effort S · confidence high  
*Principle:* H9 Help users recognize, diagnose and recover from errors; H3 User control and freedom · *Area:* Console transcript recovery card after a refused Chat-now send · *Sizes:* all · *Frequency:* Every RP-072 refusal, and any later trace-provenance failure  
*Sources:* G1-02 · *Verdict:* PARTIAL (rated 3, not 4: same incident as RP-072)

Evidence:
- `tldw_chatbook/UI/Console_Modules/provider_continuation_recovery.py:43-66 (filter at 55-62), 179-205` (code; G1-02): trace_call_recovery_state returns None unless pause_kind is TRACE_CALL or TEMPORARY_CAPTURE, so the always-mounted card stays hidden (display = state is not None, :205)
- `tldw_chatbook/Chat/console_chat_controller.py:12042-12060, 8250-8864; Chat/console_turn_preparation.py:78-82` (code; G1-02): The controller writes 'Trace provenance could not be saved. Retry, Send without capture, or Cancel.' and supports all three actions for TRACE_PROVENANCE
- `rv-gap1 160x45: Inspect rail opened and scrolled x12; status strip toggled; grep for trace/retry/without capture` (live; G1-02): No match anywhere; Inspect says 'Run: Ready' (captures/review-rv-gap1-isolde-blocked-inspect-160x45 rows 13, 19-24); the only cues are the gutter 'blocked' and '⛔ Blocked' in the Chats list
- `rv-gap1 persona session (trace.log 20:38:27); vf-gap1` (live; G1-02): A composer retry is refused ('Another send is still preparing…', controller_submit status=refused) and parked in the unsent-turn strip

Recommendation: see [section 5](#rp-074).

*Verifier corrections applied:* Same incident as RP-072, and the recovery half is already TASK-33621.2 AC#2/#3, so a second independent severity 4 would double-count one failure. Rated 3 on its own: once RP-072 is fixed it bites only on residual provenance failures. A Console dependency, not Roleplay content.

<a id="a2-rp-075"></a>
##### RP-075 · Saving a persona whose name is over 200 characters crashes the whole app, losing every unsaved draft

P1 · severity 3 · bug · scope content · effort S · confidence high  
*Principle:* H9 Help users recognize, diagnose and recover from errors; H5 Error prevention (unconstrained input); stability · *Area:* Persona editor Save (create and edit) -> error toast (PersonasScreen._notify) · *Sizes:* all (reproduced at 160x45 and 120x36) · *Frequency:* Rare trigger, deterministic: any persona name over 200 characters (or a tool-rule name over 512) on Save  
*Sources:* G2-01 · *Verdict:* PARTIAL (reproduced live; rated 3 for the rare trigger)

Evidence:
- `rv-gap2 160x45: Personas > Cozy Storyteller > Edit > Name = 204 chars > Save` (live; G2-01): The app exited to a Textual traceback ('MarkupError: Expected markup value …') and the tmux server died; the unsaved System prompt edit was lost (captures/review-rv-gap2-persona-name-too-long-save-160x45 rows 1-46)
- `runs/rv-gap2/data/rp_review/tldw_cli_app.log:394, 409-410` (capture; G2-01): Pydantic string_too_long, then 'event=unhandled_exception … exception_type=MarkupError widget_type=PersonasScreen', repeated in MainNavigationBar._update_overflow_hints
- `vf-gap2-2 120x36: New persona with a 205-char name > Save` (live; G2-01 verification): App exited; tldw_cli_app.log:392-408 shows LocalPersonaProfileCreate string_too_long, then the same MarkupError
- `persona_profile_editor_widget.py:127, 544-568; tldw_api/character_persona_schemas.py:562, 581, 626` (code; G2-01): The Name Input has no max_length and validate() checks only for blank; the schemas cap name at 200 and rule_name at 512
- `personas_screen.py:15596, 15711, 16249-16269` (code; G2-01): _notify(f"Save failed: {exc}") reaches app.notify with Textual's default markup=True; the policy-rule path does the same, and about 30 other _notify(f"…{exc}") calls share the wrapper

Recommendation: see [section 5](#rp-075).

*Verifier corrections applied:* The crash also hits the create path (LocalPersonaProfileCreate, personas_screen.py:15671), and the ValidationError comes from the screen's own model construction (:15697), not the service. Frequency narrowed: only bracket text shaped like markup attributes ('[type=…, input_value=…]') or '[/]' raises; '[Errno 2] …' and similar render fine, so the trigger is pydantic validation text, not every bracketed error. Severity 3: catastrophic impact, rare trigger.

<a id="a2-rp-076"></a>
##### RP-076 · Each long persona field is a full-screen box, so one field is visible at a time and Mode/Enabled sit 3-4 screens down (the inverse of RP-002)

P1 · severity 3 · usability · scope content · effort S · confidence high  
*Principle:* H8 Aesthetic and minimalist design; progressive disclosure; form scanning (related fields seen together) · *Area:* Persona editor: Description, System prompt, Personality traits · *Sizes:* all: one field per screen at 120x36, 160x45 and 220x55 · *Frequency:* Every persona edit  
*Sources:* G2-03 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap2-2-persona-editor-open-120x36 rows 16-30` (capture; G2-03): 15-row window: Name, then the top of a ~14-row Description box holding a 2-line description
- `captures/review-rv-gap2-persona-editor-open-160x45 rows 16-39` (capture; G2-03): Name, then a 23-row Description box with 1 line of text; System prompt, traits, Mode and Enabled below the fold
- `captures/review-rv-gap2-3-persona-editor-open-220x55 rows 16-49` (capture; G2-03): A ~33-row Description box holding 1 line
- `rv-gap2 160x45 / rv-gap2-2 120x36, wheel over the editor` (live; G2-03): Mode first appears after ~31 wheel steps at 160x45 and ~18 at 120x36; scroll extent ≈125 rows at 160x45 (~5 screens), ≈97 at 120x36 (~6.5), ≈150 at 220x55 (~4.4)
- `rv-gap2 160x45: click (51,24) in System prompt, type 'XX'` (live; G2-03): Caret landed correctly and the text stayed visible: no RP-002-style corruption
- `persona_profile_editor_widget.py:45-47, 128-136; textual widgets/_text_area.py:113-116` (code; G2-03): No height rule on the three TextAreas; Textual's default height:1fr resolves to the viewport inside the editor's VerticalScroll

Recommendation: see [section 5](#rp-076).

<a id="a2-rp-077"></a>
##### RP-077 · A persona's Description and Personality traits never reach the model in Console, but the test chat sends the Description, so the preview and the real chat disagree

P1 · severity 3 · bug · scope content · effort M · confidence high  
*Principle:* H1 Visibility of system status (truthful feedback); H2 Match (fields imply effect); preview/production consistency · *Area:* Persona prompt composition: test chat vs Console persona session vs Chat now · *Sizes:* all (behaviour) · *Frequency:* Every persona whose behaviour lives in Description or Personality traits; every persona test chat  
*Sources:* G2-04 · *Verdict:* CONFIRMED

Evidence:
- `tldw_chatbook/UI/Console_Modules/session.py:584-596, 615-645` (code; G2-04): _persona_session_prompt_seed takes only profile['system_prompt'] ('no multi-field join'); characters join description, personality, scenario and more
- `UI/Persona_Modules/personas_preview_controller.py:49-101, 463-494` (code; G2-04): The test chat runs persona records through compose_character_card_text with description and reads 'personality' (a key personas lack); personas use the saved record, characters the draft; the docstring claims the preview 'shows exactly the system prompt a real Console session would send'
- `rv-gap2-3 220x55: Isolde's Bridge Officer (Description only) > Try a test chat (captures/review-rv-gap2-3-linked-persona-testchat-uses-description-220x55 rows 16-18)` (live; G2-04): Reply 'Staying in character as A persona bound to the Captain Isolde Varga card…': the Description is used as the system prompt
- `rv-gap2-4 160x45: same persona > Chat now > send (captures/review-rv-gap2-4-linked-persona-chat-now-console-160x45 row 41; review-rv-gap2-4-linked-persona-console-reply-160x45 row 35)` (live; G2-04): 'System Prompt: off · Persona: Isolde's Bridge Offi…'; reply 'Staying in character as You are a capable assistant with optional tools…'
- `personas_screen.py:7341-7347; persona_profile_card_widget.py:57-61` (code; G2-04): The Chat now hand-off body lists Description and System prompt only; the card hides Personality traits

Recommendation: see [section 5](#rp-077).

*Verifier corrections applied:* Console's system-prompt-only behaviour may be deliberate tldw_server parity (TASK-617). Whichever way it is decided, the preview/Console mismatch is the defect. Extends RP-045 (test chat truthfulness).

<a id="a2-rp-078"></a>
##### RP-078 · The Inspector's 'Tool policy' line shows the last-edited persona's rules on other personas, and 'no rules' for personas that have rules

P1 · severity 3 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status (false status on a permissions setting) · *Area:* Inspector: #personas-policy-rules-summary on persona selection · *Sizes:* all · *Frequency:* Every persona selection after any Edit open or rule save in the session; the first selection of any persona that has rules  
*Sources:* G2-05 · *Verdict:* CONFIRMED

Evidence:
- `rv-gap2-4 160x45: Terse Code Reviewer > Edit > (blind) add 'skill: summarize → allow' > Cancel > select Cozy Storyteller, then Isolde's Bridge Officer` (live; G2-05): Both show 'Tool policy: 1 rule(s) / skill: summarize → allow' although neither has rules (captures/review-rv-gap2-4-stale-policy-summary-other-persona-160x45 rows 19-20)
- `personas_screen.py:5383-5440 vs 8374, 15600, 15784; personas_inspector_pane.py:249-253, 403-431` (code; G2-05): _select_profile never calls show_policy_rules; only Edit-open, rule save and after-save do; clear_selection does not reset it; the compose-time text is 'Tool policy: no rules'
- `personas_inspector_pane.py:542-567, 984-987` (code; G2-05): Visible for every persona; the copy flips between 'no rules' and 'no rules (default posture)', and neither mentions the deny-by-default effect of allow rules
- `runs/rv-gap2-4 personas.json` (data; G2-05 verification): Cozy Storyteller has rules=None; Terse Code Reviewer has [skill: summarize]

Recommendation: see [section 5](#rp-078).

*Verifier corrections applied:* A content fix (one line in _select_profile and clear_selection); moving the summary to an Info mode is optional. The stale value sticks after any Edit open or rule save, even one that sets 'no rules'; merely having a persona with rules does not leak, but that persona reads 'no rules' on first selection.

<a id="a2-rp-079"></a>
##### RP-079 · After a persona Save then Cancel, the card still shows the previous persona (or the pre-edit text) while the Inspector names the saved one; Edit then fails with 'Selection out of sync'

P1 · severity 3 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; H4 Consistency (one selection, shown everywhere); H5 Error prevention (actions target an unseen item) · *Area:* Persona create/edit -> Save -> Cancel; persona card vs Inspector · *Sizes:* all (reproduced at 120x36 and 160x45) · *Frequency:* Most persona edit sessions (Save keeps the editor open, so Cancel is the normal exit after every save); every create followed by Cancel  
*Sources:* G2-06 · *Verdict:* CONFIRMED (wider than first reported)

Evidence:
- `rv-gap2-2 120x36: select Default Narrator > Ctrl+N > 'Repro B' > Save > Cancel` (live; G2-06): Card 'Name: Default Narrator…' while the Inspector reads 'Selected: Repro B'; Edit gives 'Selection out of sync; reselect the persona.' (captures/review-rv-gap2-2-stale-card-after-create-save-cancel-120x36 rows 15-21)
- `rv-gap2-2 120x36: New Actor Pack 'Wren Pack Persona' > Save > Cancel` (live; G2-06): Card 'Cozy Storyteller', Inspector 'Selected: Wren Pack Persona', Edit refused (captures/review-rv-gap2-2-edit-selection-out-of-sync-120x36 rows 15-31)
- `vf-gap2 160x45: Default Narrator > Ctrl+N > 'Verify B' > Save > Cancel; then rename Cozy > Save > Cancel` (live; G2-06 verification): Same split (captures/review-vf-gap2-stale-card-edit-out-of-sync-160x45); the edit case leaves the pre-edit card (captures/review-vf-gap2-stale-card-after-edit-save-cancel-160x45 rows 15-20)
- `personas_screen.py:5408, 8333-8341, 15733-15810, 15866-15882` (code; G2-06): show_persona is called only in _select_profile; _after_profile_save selects the saved id but never re-renders the card; the cancel path re-shows the stale card; Edit rejects the mismatched id

Recommendation: see [section 5](#rp-079).

*Verifier corrections applied:* Wider than stated: editing an EXISTING persona, Save, then Cancel also shows the pre-edit card (live: renamed Cozy to 'Cozy Storyteller Renamed'; the card still read 'Cozy Storyteller' while the rail and Inspector read 'Renamed'). Edit still works in that case because the ids match.

<a id="a2-rp-080"></a>
##### RP-080 · Changing page keeps the old scroll offset: after scrolling to the end of one page, the next page opens at its bottom and about 41 characters are never shown

P1 · severity 3 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; predictable navigation (a new page starts at its top) · *Area:* Characters/Personas list paging (page bar) · *Sizes:* all where a page holds more items than the viewport (every target size at 50 per page) · *Frequency:* Every page change after scrolling - the natural flow, because '>' sits under the list  
*Sources:* G3-04 · *Verdict:* CONFIRMED

Evidence:
- `rv-gap3 160x45 (348 characters): page 4, wheel to the end ('Niall 3' last), click '>'` (live; G3-04): Page 5 (201-250) opened at items 42-50; 'Niall 4' through 'Quilla Calloway' were never shown (captures/review-rv-gap3-page5-lands-at-bottom-160x45 rows 23-40: thumb low, '201-250 of 348')
- `captures/review-rv-gap3-page2-160x45 row 23` (capture; G3-04): Page 2 opens on an orphan description line whose name row is scrolled off above
- `vf-gap3 160x45: end of page 1, '>'` (live; G3-04 verification): Page 2 opened at its bottom (captures/review-vf-gap3-page2-after-scroll-160x45); page 3 too; going back kept the bottom offset
- `personas_library_pane.py:419-482, 525-533; personas_screen.py:4141-4152; textual _list_view.py:261-270` (code; G3-04): update_rows clears and extends with no scroll reset; ListView.clear() only sets index=None; only mark_active_row moves the viewport
- `captures/review-vf-gap3-library-next-after-scroll-160x45` (live; G3-04 verification): Library: scrolled to the bottom of page 1, Next lands on the first item of page 2

Recommendation: see [section 5](#rp-080).

<a id="a2-rp-081"></a>
##### RP-081 · Try it at realistic volume: at 160x45 the 'what fired / what was dropped' summary and per-rule lines are clipped off, and the summary calls budget-dropped lore 'near-miss'

P1 · severity 3 · usability · scope both · effort M · confidence high  
*Principle:* H1 Visibility of system status; H2 Match between system and the real world (why lore was dropped) · *Area:* Dictionary and Lore Try-it output (summary + fired/skipped lists) · *Sizes:* 160x45 (summary invisible), 120x36 (worse), 220x55 (summary visible, 1-7 detail rows) · *Frequency:* Every preview with a realistic sample (3+ lines), or a book or dictionary that fires more than a few entries  
*Sources:* G3-09 · *Verdict:* PARTIAL (vocabulary claim narrowed)

Evidence:
- `captures/review-rv-gap3-dict-tryit-many-160x45 rows 26-41; review-rv-gap3-dict-tryit-many-scrolled-160x45` (capture; G3-09): 3-line sample vs the 150-rule dictionary: summary and per-rule lines fall below the widget; 40 wheel notches over the output moved nothing
- `captures/review-rv-gap3-dict-tryit-many-220x55 rows 44-51` (capture; G3-09): Only at 220x55: '24 fired · 2 skipped · 24/1000 tokens', then a 7-row keyhole of 26 lines
- `captures/review-rv-gap3-lore-tryit-many-160x45 rows 33-41; review-rv-gap3-lore-tryit-many-220x55 rows 51-52` (capture; G3-09): 160x45: the injection text alone fills the space; 220x55: '5 fired · 64 near-miss · 919/1000 tokens · over budget' with 1 detail line
- `personas_dictionary_tryit.py:74-93; personas_lore_tryit.py:39-62, 148-158, 183-189` (code; G3-09): Both widgets are Vertical (overflow hidden) with max-height 60%; the lore summary counts every non-fired record as 'near-miss'
- `Character_Chat/world_info_processor.py:430-449` (code; G3-09): Non-fired candidates carry 'skipped:disabled' / 'skipped:budget' ('dropped by token budget') reasons

Recommendation: see [section 5](#rp-081).

*Verifier corrections applied:* Only the summary count lumps budget, disabled and secondary-key cases together as 'near-miss'; each non-fired row already prints its true reason ('dropped by token budget', 'disabled (key … matched)', 'secondary key not found'; personas_lore_tryit.py:183-189), but those rows are invisible at 160x45 because of the clip. So the fix is mainly the frame/scroll plus splitting the summary count. Overlaps RP-006 (the permanent Try-it block) and mirrors RP-010 (the test chat's 60% cap).


#### P2

<a id="rp-028"></a>
<a id="a2-rp-028"></a>
##### RP-028 · Counts read '· 0' next to a populated list (Dictionaries and Lore until something is selected), with no loading marker

P2 · severity 2 · bug · scope both · effort S · confidence high  
*Principle:* H1 Visibility of system status · *Area:* Purpose/count line · *Sizes:* all · *Frequency:* First entry to Dictionaries/Lore on every Roleplay visit (the route is rebuilt each visit)  
*Sources:* NA-10, TF-19, LF-14, IA-07 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:4484, 4276-4335, 4361-4416, 4598-4621` (code; NA-10): _update_purpose_line runs before the dictionary/lore renders fill their caches; nothing refreshes it afterwards
- `captures/review-rv-nng-a-dict-count-zero-after-10s-160x45 rows 9, 20-25; review-rv-nng-a-lore-count-zero-after-10s-160x45` (live; NA-10): '· 0' ten seconds after entry with 3 items listed
- `captures/review-vf-flows-tf13-typed-mode-error-160x45; review-rv-a11y-dict-mode-kbd-160x45` (live; TF-19 verification): 'Dictionaries ... · 0' for >14 s
- `captures/roleplay-dictionaries-nothing-selected-160x45 row 9` (capture; LF-14): '· 0' while 3 dictionaries are listed

*Who it hurts:* Users read '0' next to three items and doubt the data or think an import failed; it also makes RP-021 look like data loss.

*Recommendation:* Show every mode's count on its rail row (Characters 28 - Personas 4 - Dictionaries 3 - Lore 3), '(…)' until loaded, refreshed from the same code path that renders rows; 'N of M' when filtered.

*Library frame:* Library rail counts show '(…)' while loading (count_loading, library_shell_state.py:395-400).

*Verifier corrections applied:* The '0' appears on the first visit to each mode per screen instance and stays until a selection (not 2-3 s). One TF-19 citation was wrong (review-rv-flows-2-j5-new-dictionary-160x45 row 9 reads '· 4').

<a id="rp-029"></a>
<a id="a2-rp-029"></a>
##### RP-029 · The four jobs are flat mode chips: only the active mode shows a count and a plain-language gloss

P2 · severity 2 · gap · scope frame · effort M · confidence high  
*Principle:* H6 Recognition rather than recall; H1 Visibility; information scent · *Area:* Mode navigation (chip strip) · *Sizes:* all · *Frequency:* Every visit  
*Sources:* LY-05, LF-02, IA-06 · *Verdict:* CONFIRMED / PARTIAL (governance correction)

Evidence:
- `personas_screen.py:1558-1586, 4598-4621` (code; LY-05/LF-02): Chips carry no counts; descriptor only in tooltip and purpose line; purpose line counts only the active mode
- `captures/roleplay-characters-arrival-160x45 rows 9-10` (capture; IA-06): 'Characters - who the AI plays · 28' / 'Modes: Characters Personas Dictionaries Lore'
- `captures/library-landing-160x45 rows 14-21` (capture; LF-02): Library rail rows with counts and glosses
- `Library/library_shell_state.py:366-420, 756-763; Widgets/destination_rail.py:163; Widgets/Home/home_rail.py:12, 146` (code; LF-02): Generic row dataclass; shared section header; Home precedent
- `Widgets/Persona_Widgets/personas_state.py:31; personas_screen.py:430-433` (code; IA-06): An unused 'Import / Export' mode label; comment says import/export should become an action

*Who it hurts:* Users cannot see whether any lore books or personas exist without switching modes (which drops their selection, RP-030); 'Dictionaries' and 'Lore' are explained only in tooltips until visited.

*Recommendation:* Build a Roleplay rail: Cast (Characters, Personas), World info (Lore books, Chat dictionaries), Create, Import / Export, Buddy - one-line rows with counts and short glosses (see Appendix C, IA sketch). Keep c/p/d/l and [ ] (ADR-152) as row hints. Keep the active mode named in the header or items title when the rail is closed. Build it from DestinationRailSectionHeader plus a LibraryRailRow-like dataclass; do not subclass LibraryRail.

*Library frame:* The Library rail shows every collection with a count and an optional gloss ('Media (2) - your files', 'Prompts (4) - reuse'); fitting rules drop the gloss first and never clip the count. The section header widget is already shared (destination_rail.py:163; Home uses it). Caveat: on reader routes Library closes its rail below ~126 cols and then loses 'where am I' - Roleplay must not.

*Verifier corrections applied:* DESIGN.md:107 and ADR-015:34 keep a local mode bar and Roleplay's descriptor/count statics, so moving modes into a rail needs a DESIGN.md/ADR-015 amendment alongside D4. Library glosses are optional (several rows have none).

<a id="rp-030"></a>
<a id="a2-rp-030"></a>
##### RP-030 · Switching modes throws away selection, search, sort, tag, page, marks and the test-chat transcript

P2 · severity 2 · usability · scope both · effort M · confidence high  
*Principle:* H3 User control and freedom; H6 Recognition rather than recall; preserve context across navigation · *Area:* Mode switching · *Sizes:* all · *Frequency:* Every mode switch  
*Sources:* NA-20, NB-05, TF-08, LF-09, G3-12 · *Verdict:* PARTIAL (by design but harmful; rated 2)

Evidence:
- `Widgets/Persona_Widgets/personas_state.py:54-70` (code; NB-05/TF-08): switch_mode clears selection and resets sort, tag and page
- `personas_screen.py:4459-4549` (code; NA-20): _apply_mode clears the search Input, marks, preview and inspector and calls _show_center(None)
- `captures/roleplay-characters-nothing-selected-160x45 rows 14-16` (capture; NB-05): After Characters -> Personas -> Characters: 'Pick a character from the list'
- `captures/review-rv-fit-roleplay-isolde-selected-160x45 -> review-rv-fit-roleplay-back-to-characters-160x45` (live; LF-09): Isolde selected; after Lore and back: 'Selected: none'
- `rv-flows-2 160x45 J5/J6` (live; TF-08): Attaching a dictionary to Wren needed a full re-find; going back dropped the dictionary; confirming Kepler-9 needed another re-find plus 14 wheel notches
- `captures/review-rv-fit-library-back-to-prompts-160x45` (live; LF-09): Library loses the open prompt too
- `captures/review-rv-gap3-chars-120x36 rows 16, 23, 30` (capture; G3-12): 348 cards, Lore -> Characters at 120x36: 'Selected: none', back to page 1 ('1-50 of 348'), Sort and Tag reset
- `captures/review-rv-gap3-return-after-library-160x45 rows 16, 37, 39` (capture; G3-12): Positive: leaving to Library and returning keeps page 5 and the selection 'Quilla Merrow 2'; mode switches bypass that path

*Who it hurts:* The four jobs are interlinked (characters use lore and dictionaries), but every hop discards where you were: 4-10 extra actions per hop to re-find the item. A stray mode key (RP-033) costs the user their place.

*Recommendation:* Keep a per-mode memory {selection, search, sort, tag, page, list scroll, active detail tab} in PersonasWorkbenchState; restore it on re-entry; persist it in save_state. Leaving a dirty editor still goes through _run_guarded and the ADR-046 veto. For lore and dictionaries also keep the selected entry, entries anchor, entry filter and active tab - useful only once RP-014 keeps the entry across saves.

*Library frame:* Library loses the open item across destinations too (live: Prompts > Notes > Prompts shows 'Select a prompt'), so the frame will not supply this; it must be added.

*Verifier corrections applied:* The reset is deliberate ('Each mode starts with a fresh library window', personas_state.py:59-60; mode switches deliberately do not auto-select, personas_screen.py:1878-1891). Selection and mode DO survive leaving and returning to Roleplay when something is selected. Rail-collapse persistence is RP-031. Gap round (G3-12) proposed raising this to 3; not adopted. Characters re-find cheaply by search (0.34 s across 348 cards), and the big-book cost comes from the missing entry filter (RP-056) and the row-0 reset (RP-014), which loses the entry even without a mode switch.

<a id="rp-031"></a>
<a id="a2-rp-031"></a>
##### RP-031 · Collapsed panes reopen on every visit: there is no remembered (preferred) layout

P2 · severity 2 · gap · scope frame · effort M · confidence high  
*Principle:* H3 User control and freedom; consistency with Library · *Area:* Pane persistence · *Sizes:* all · *Frequency:* Every visit after a collapse  
*Sources:* LF-10, NA-20, NB-05 · *Verdict:* CONFIRMED (documented behaviour; a gap, not a rule)

Evidence:
- `personas_screen.py:1460-1461, 4427-4457, 1748-1765` (code; LF-10): Collapse flags are instance attributes; save_state stores only the workbench dataclass and preview
- `tldw_chatbook/UI/Navigation/screen_registry.py:116-121` (code; LF-10): The personas route is not reusable - each visit builds a fresh screen
- `tldw_chatbook/config.py:2370-2430` (code; LF-10): Library's [library.reader] persistence
- `Docs/User_Guide/roleplay-chat-dictionaries.md:246-248` (doc; LF-10): 'Rail collapse lasts for the current visit'

*Who it hurts:* A user who collapses the Inspector to give the editor room has to redo it on every visit.

*Recommendation:* Add a [roleplay.reader] config section (name: owner's call, D5) storing only requested choices (nav open, per-mode items open/width, optional custom width), normalised through Utils/adaptive_reader_state.py and written off-thread. Never persist responsive or work-first overrides. Update the guide line that documents per-visit reset.

*Library frame:* Library persists requested pane state in [library.reader] / per-destination sections and keeps effective (responsive) state transient (ADR-086). Its ordinary-route Collapse is not persisted - follow the adaptive contract only.

<a id="rp-032"></a>
<a id="a2-rp-032"></a>
##### RP-032 · Collapsing a pane means hitting a 3x1 '<' button, and the collapsed handles still cost 13 + 11 columns (26 of 120)

P2 · severity 2 · usability · scope frame · effort S · confidence high  
*Principle:* Fitts's law; space efficiency; H6 (glyph-only control) · *Area:* Pane collapse affordances · *Sizes:* all · *Frequency:* Every collapse/expand  
*Sources:* LF-11, LY-19, KA-21 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:607-609, 1590-1603, 1725-1738` (code; LF-11): Handle widths 13 and 11
- `css/features/_console.tcss:702-709, 753-761` (code; LY-19): Collapse button 3 wide x 1 tall; right handle 3 rows tall
- `personas_inspector_pane.py:231` (code; KA-21): 'Collapse Inspector rail' exists only as a tooltip
- `captures/roleplay-rails-collapsed-160x45; verifier vf-fit 120x36` (capture; LF-11): Closed handles 14 + 12 cols; still 13 + 12 at 120x36
- `Widgets/Library/library_adaptive_reader_shell.py:36, 73-230; library_rail.py:538-541` (code; LF-11): Library 5-cell named grips vs its 3-cell ordinary handle

*Who it hurts:* Reclaiming width for writing takes two small precise clicks every visit and returns only part of the width; at 120 cols the closed handles still use 22% of the width. Keyboard users focusing the control see only '<'.

*Recommendation:* Replace '<'/'>' and the wide handles with 5-cell named grips for both side panes, one grammar for open and closed; rename the library-prefixed grip CSS to neutral tokens; record the design-language exception in the ADR-086 amendment (D4). Keep grips as F6 targets only when their pane is closed.

*Library frame:* Library adaptive routes use 5-cell, full-height grips that spell the pane's name vertically. Caveats: design-language.md:132-134 says 'never a sliver' (grips need a recorded exception); Library's grips reserve 10 cols even when panes are open; Library also runs a second, 3-cell non-persisted grammar - pick one.

<a id="rp-033"></a>
<a id="a2-rp-033"></a>
##### RP-033 · Single-letter keys act from any non-text focus: stray typing switches modes, and Space on a button silently disables a dictionary

P2 · severity 2 · usability · scope both · effort S · confidence high  
*Principle:* H5 Error prevention (mode errors); H3 User control · *Area:* Screen keys c/p/d/l, [ ], and list keys Space/m/s · *Sizes:* all · *Frequency:* Any typing after a click that missed a text field; Space with a rail button focused  
*Sources:* NA-05, TF-13, KA-05, NA-15 · *Verdict:* PARTIAL (by design but harmful; rated 2)

Evidence:
- `personas_screen.py:1083-1121, 437-443; personas_library_pane.py:80-86` (code; NA-05): Screen-level c/p/d/l and [ ]; pane-level space/m/s
- `personas_library_pane.py:706-719; textual/widgets/_button.py:297` (code; KA-05): Space bubbles from a focused Button (Button binds only Enter) to the pane's toggle
- `captures/review-rv-nng-a-typed-spell-in-list-160x45 row 9` (live; NA-05): Clicked a row, typed 'spell': sort cycled, Characters -> Personas -> Lore, selection lost
- `captures/review-rv-flows-mode-error-typing-220x55; review-vf-flows-tf13-typed-mode-error-160x45` (live; TF-13): Clicked card text, typed 'a pleasant day': ended in Dictionaries with '· 0'
- `captures/review-rv-nng-a-space-toggled-dictionary-off-160x45 row 23` (live; NA-15): '8 entries - off', no toast (handler personas_screen.py:7518-7520 only reports failure)
- `captures/review-rv-a11y-space-on-duplicate-toggles-dict-120x36 rows 16-22` (live; KA-05): Space with 'Duplicate' focused toggled '15 entries - on' -> 'off'

*Who it hurts:* Users who think they are typing lose their place, selection and search; in Dictionaries Space turns a dictionary off for every conversation that links it, with no toast.

*Recommendation:* Keep c/p/d/l and [ ] (ADR-152) but scope them to the list, rail and mode controls rather than the whole screen; move Space/m/s onto the ListView so they act only with list focus; give buttons an 'enter,space' press binding; show a toast with undo ('Profanity Softener turned off - Space again to undo'). Lossless mode switches (RP-030) make any remaining misfire cheap.

*Library frame:* Library also uses printable verbs but gates them to non-text navigator focus (check_action, library_screen.py:1030-1044) and its footer says 'typing in field / after esc:'.

*Verifier corrections applied:* Printable mode keys are deliberate (ADR-031 rule 3, ADR-152); unsaved edits are protected by the guard, so a stray key costs position, not data. Space affects only conversation-linked dictionaries - character copies have their own flag, so a 'used by Vex' toast would be wrong. Mechanism: the grey list cursor stays painted while focus is on a toolbar button, so Space acts on a row that looks current (RP-062).

<a id="rp-034"></a>
<a id="a2-rp-034"></a>
##### RP-034 · Footer hints ignore focus: they advertise c/p/d/l, [ ] and s while you type, drop the context keys at 120 cols, and never mention 'm' (mark)

P2 · severity 2 · usability · scope frame · effort S · confidence high  
*Principle:* ADR-031 rule 4 (advertised == working in context); H6 Recognition rather than recall · *Area:* Footer hints · *Sizes:* all; most visible at 120x36 and below · *Frequency:* Every typing session  
*Sources:* NB-11, KA-10, LF-12, NA-11 · *Verdict:* CONFIRMED (reproduced live)

Evidence:
- `personas_screen.py:16453-16505` (code; NB-11/KA-10/LF-12): _shortcut_context depends only on mode and edit state; no DescendantFocus handler; 'm' never projected
- `captures/review-rv-nng-b-search-typing-footer-160x45 row 45; review-rv-nng-b-editor-empty-name-save-160x45 row 45` (live; NB-11): 'c/p/d/l mode | [ ] mode | s sort' while typing in search / the Name field
- `captures/review-vf-fit-search-typed-120x36` (live; LF-12): Same at 120x36
- `captures/roleplay-characters-arrival-120x36 last row; roleplay-character-editor-120x36; roleplay-characters-arrival-80x24` (capture; KA-10): At 120 '[ ] mode' and 's sort' dropped while 'ctrl+enter Send to Console draft' keeps 34 cols; editor footer leads with ctrl+n/ctrl+f
- `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md:10, 17` (doc; KA-10): task-1340 refinement: advertised == working in the active context

*Who it hurts:* The main discoverability channel teaches keys that type letters, hides the bulk-mark key, and at the medium target drops mode/sort hints in favour of a 34-column Ctrl+Enter chip.

*Recommendation:* Re-register the footer context on focus changes. In a text field: 'typing in field - esc leave field - after esc: c p d l [ ] mode - s sort'. In an editor: save and cancel first. On the list: 'enter select - m mark - s sort'. Merge the two mode chips into one ('c p d l [ ] mode'), demote ctrl+n/ctrl+f, shorten the Console hint (RP-035).

*Library frame:* Library's footer is focus-aware: 'typing in field | esc leave field | after esc: / focus search' (library_screen.py:4741-4835). Copy the rule (status chip, esc chip, 'after esc:' bucket).

<a id="rp-035"></a>
<a id="a2-rp-035"></a>
##### RP-035 · The key set breaks house conventions: no '/' for search, a banned Ctrl+S, Ctrl+Enter that many terminals cannot send, and no key at all for Chat now

P2 · severity 2 · usability · scope both · effort S · confidence high  
*Principle:* H4 Consistency and standards; H7 Flexibility and efficiency (accelerators) · *Area:* Keybindings · *Sizes:* all · *Frequency:* Constant  
*Sources:* NA-19, LF-13, KA-13, NB-13, KA-19 · *Verdict:* CONFIRMED / PARTIAL (corrections applied)

Evidence:
- `personas_screen.py:1083-1121` (code; NA-19/LF-13): ctrl+f search, ctrl+s save, ctrl+enter Send to Console draft; no '/'; no key for Chat now/Edit/Import/Export
- `personas_screen.py:1122-1155` (code; NB-13): Inspector F6 targets: conversations list, collapse, attach-to-console - never Chat now
- `backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md:8-9` (doc; KA-19): Rule 2: never bind ctrl+s (Library violates it too, library_screen.py:1047)
- `schedules_workbench.py:431; trajectory_screen.py:209; Logs_Window.py:141; library_screen.py:8859-8900` (code; LF-13): '/' is the house search key
- `personas_dictionary_tryit.py:71-73, 113; personas_lore_tryit.py:36-38, 90` (code; KA-13): Try-it Ctrl+Enter = Run preview, tooltip-only
- `captures/review-rv-nng-b-keyboard-tab1-160x45.png` (live; NB-13): Ctrl+F, 'isolde', Enter, Enter, F6, F6, Tab to reach Chat now
- `personas_library_pane.py:706-719` (code; NB-13): Space toggling returns unless row.kind == 'dictionary'

*Who it hurts:* Keys learned in the Library ('/') do nothing here; the primary action of job 1 needs mouse or Tab-hunting (7 keys plus typing to reach Chat now); Ctrl+Enter is indistinguishable from Enter in terminals without extended key reporting; lore books cannot be toggled with Space like dictionaries.

*Recommendation:* Bind '/' to search (keep Ctrl+F as an alias). Give Chat now a primary, terminal-safe key from list/button focus (e.g. a single letter per ADR-031) and Send to Console draft a second one; keep Ctrl+Enter only as an unadvertised alias. Put Chat now first in the Inspector/work-header F6 target. Advertise 'enter run' / 'ctrl+enter run' when Try it has focus. Extend Space toggling to lore rows. Ctrl+S: decision D9.

*Library frame:* Library, Schedules, Trajectory and Logs use '/' for search; Library advertises single-letter verbs and names what Enter does. Library also binds ctrl+s (do not copy that).

*Verifier corrections applied:* The two Ctrl+Enter meanings never collide (Try it exists only in Dictionaries/Lore where Send is unavailable); the Try-it key is advertised in tooltips, not the footer. The tmux C-Enter probe shows a harness limitation; the terminal point stands as general knowledge.

<a id="rp-036"></a>
<a id="a2-rp-036"></a>
##### RP-036 · F1 help is the generic dump: titled 'PersonasScreen Shortcuts', raw key specs, keys that do nothing in context, and no m/s/Space

P2 · severity 2 · gap · scope frame · effort S · confidence high  
*Principle:* H10 Help and documentation; H2 (internal class name shown) · *Area:* F1 contextual help · *Sizes:* all; most needed at <=100 cols where the footer truncates · *Frequency:* Each help request  
*Sources:* NB-21, KA-12 · *Verdict:* CONFIRMED

Evidence:
- `app_lifecycle.py:1688-1702; app.py:5646-5658` (code; NB-21/KA-12): No Roleplay help action; generic fallback flattens BINDINGS and titles the panel with the class name
- `captures/review-rv-a11y-f1-help-roleplay-160x45.png/.txt; review-rv-nng-b-f1-help-160x45.png` (capture; KA-12): 'PersonasScreen Shortcuts' listing tab, ctrl+c/super+c, ctrl+s, esc, c/p/d/l, [ ]; no s, Space or m
- `captures/review-rv-a11y-f1-help-library-160x45` (capture; KA-12): 'Library Shortcuts - Landing'

*Who it hurts:* Help is the fallback when the footer truncates, and it shows developer text ('ctrl+c,super+c: Copy selected text', 'ctrl+s: Save' with no editor open) while hiding bulk marking and sort.

*Recommendation:* Add action_show_workbench_help to the Roleplay screen built from the focus-aware footer projection plus the list keys (m, s, Space) and a 3-line mode legend; title it 'Roleplay - keys (<mode>)'; end with a pointer to the User Guide.

*Library frame:* Library overrides F1 and builds it from the same per-state set as its footer ('Library Shortcuts - Landing'); it fixed this exact 'LibraryScreen Shortcuts' defect (library_screen.py:25683-25770).

<a id="rp-037"></a>
<a id="a2-rp-037"></a>
##### RP-037 · The in-screen unsaved-changes dialog talks about closing a 'tab' and offers no Save

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* H3 User control and freedom; H2 Match (there are no tabs); H4 (two different guards for the same drafts) · *Area:* Unsaved-changes guard on mode/selection switch and Cancel · *Sizes:* all · *Frequency:* Every guarded in-screen transition while editing  
*Sources:* NA-12, NB-20, KA-17 · *Verdict:* CONFIRMED / PARTIAL (ADR scope corrected)

Evidence:
- `tldw_chatbook/Widgets/confirmation_dialog.py:164-192` (code; NA-12/NB-20): 'The tab "..." has unsaved changes. Are you sure you want to close it?' - Close Without Saving / Keep Open
- `personas_screen.py:16179-16243` (code; NA-12): _confirm_discard_unsaved used for every guarded in-screen transition
- `captures/review-rv-nng-a-unsaved-dialog-on-mode-switch-160x45 rows 17-27; review-rv-nng-b-unsaved-guard-160x45; review-rv-a11y-unsaved-guard-dialog-160x45` (live; NA-12): Dialog on mode switch and on Escape; Keep Open focused by default (good)
- `UI/Navigation/character_conversation_navigation.py:233-283; backlog/decisions/046-...md:106-113` (code; NA-12): RoleplayDraftNavigationDialog (Save and continue / Discard and continue / Stay) exists, scoped to deep links

*Who it hurts:* To keep the work and still switch, the user must Keep Open, find Save (often scrolled away), save and redo the switch. 'Close the tab' misdescribes the action and invites the wrong choice.

*Recommendation:* Use RoleplayDraftNavigationDialog for mode and selection switches: 'Switch to Personas? "Captain Isolde Varga" has unsaved changes. [Save and switch] [Discard and switch] [Stay]'. For an explicit Cancel/Escape, keep two choices but fix the copy ('Discard changes to Ada?' Keep editing / Discard). Keep the safe default focus and Escape = keep.

*Library frame:* Library Prompts/Skills only veto with a toast ('Save or Discard changes first') - no in-place Save either; do not copy it. Roleplay's own leave-screen guard already offers Save and continue / Discard and continue / Stay.

*Verifier corrections applied:* ADR-046's Save/Discard/Stay governs deep links and coordinator unmounts, not an explicit Cancel - so this is inconsistency, not an ADR violation; for Escape=Cancel a keep-or-discard choice is conventional.

<a id="rp-038"></a>
<a id="a2-rp-038"></a>
##### RP-038 · Save state is hard to read: the unsaved signal sits far from the editor (the rail badge never renders), and after Save the editor looks unchanged

P2 · severity 2 · usability · scope both · effort S · confidence high  
*Principle:* H1 Visibility of system status · *Area:* Character/persona editor status · *Sizes:* all; no signal at all at <=24 rows · *Frequency:* Every edit session  
*Sources:* NA-13, TF-20 · *Verdict:* PARTIAL (corrections applied)

Evidence:
- `personas_screen.py:4551-4565; base_app_screen.py:53-66; css/components/_workbench.tcss:316-318` (code; NA-13): Header subtitle ' - unsaved', hidden at <=24 rows
- `personas_library_pane.py:545-557; css/components/_agentic_terminal.tcss:931-933; personas_screen.py CSS ~:1276` (code; NA-13): Rail .is-unsaved border is overridden by the screen's 'border: none' and never renders
- `captures/review-rv-nng-a-editor-unsaved-160x45 rows 6, 16, 28-31` (capture; NA-13): Header 'Editing Captain Isolde Varga - unsaved'; Inspector 'Validation: editing...'; rail row unchanged
- `captures/review-rv-flows-j1-after-save-160x45.png` (capture; TF-20): Toast 'Character saved.'; same Save/Cancel bar

*Who it hurts:* Authors lose track of whether a long edit is saved. The redesign's plan to drop the boxed header would delete the main remaining signal ('Editing <name> - unsaved').

*Recommendation:* Put a status line in the editor's anchored Save bar ('Unsaved changes' / 'Saved 15:23'); make Save primary only when dirty; after saving show 'Saved' with 'Done' replacing Cancel; fix the rail row's unsaved glyph (e.g. '✎ Captain Isolde Varga'); carry the unsaved state into the new one-row header.

*Library frame:* Notes shows 'Saved - Next: Keep editing...'; Prompts switches its bar to 'Save changes / Discard changes' when dirty and shows 'Modified now - v2' after saving.

*Verifier corrections applied:* At target sizes the unsaved state does show in the header ('Editing ... - unsaved') and the Inspector readiness line, so 'barely visible' overstated it. Staying in the editor after a save is deliberate (personas_screen.py:15512-15515).

<a id="rp-039"></a>
<a id="a2-rp-039"></a>
##### RP-039 · Inspector 'Delete' looks like an export button and silently switches to the marked rows, and marks vanish on any page change or search, so bulk actions reach one page at most

P2 · severity 2 · usability · scope both · effort S · confidence high  
*Principle:* H5 Error prevention; H6 (the target of an action must be visible); H3 User control (a selection survives filtering) · *Area:* Inspector Delete/Export; bulk mark (m, ●, 'N marked'); confirm dialog · *Sizes:* all · *Frequency:* Every delete; every bulk operation; any clean-up larger than one page (348 cards = 7 pages) or one search  
*Sources:* NA-14, NB-12, G3-05 · *Verdict:* CONFIRMED + gap-round volume evidence (mark pruning itself is by design)

Evidence:
- `personas_inspector_pane.py:332-337, 339-362` (code; NA-14): 'personas-destructive' class has no stylesheet; Delete sits directly above Manage Buddy
- `personas_inspector_pane.py:1095-1144; personas_screen.py:14798, 14995-15000` (code; NA-14/NB-12): Retargeting disclosed only by tooltip; dialog passes only a count
- `captures/review-rv-nng-a-marks-vs-selection-delete-160x45; review-rv-nng-a-bulk-delete-dialog-160x45` (live; NA-14): 'Selected: Terse Code Reviewer' -> 'Delete 2 personas?'
- `captures/review-rv-nng-b-marked-rows-160x45 rows 17-41; review-rv-nng-b-bulk-delete-confirm-160x45 rows 15-24` (live; NB-12): 'Selected: Ada' with ARIA-7 and Isolde marked
- `rv-gap3 160x45 (348 cards): page 2, mark 2 rows ('2 marked'), '>' to page 3, '<' back` (live; G3-05): '2 marked' gone on page 3; back on page 2 the ● marks are gone too, with no notice (captures/review-rv-gap3-page3-marks-gone-160x45 rows 39-41; review-rv-gap3-back-page2-marks-lost-160x45 rows 30-34)
- `rv-gap3-2 160x45: mark 2, Ctrl+F 'Corwin', clear the search` (live; G3-05): Marks gone (captures/review-rv-gap3-2-marks-lost-after-search-160x45)
- `personas_library_pane.py:483-488, 575-582` (code; G3-05): 'A mark never outlives its row': marks are pruned to the rendered rows on every update_rows

*Who it hurts:* With two rows marked and a third selected, the screen says 'Selected: Ada' and the button says 'Delete', but the dialog deletes two other cards ('Delete 2 characters? This cannot be undone here.' - no names). At volume, cleaning up duplicate imports means one bulk pass per page or per search, with marks lost silently whenever the view changes, so users may believe they marked 12 and delete the 3 on screen.

*Recommendation:* Give Delete the error variant and space it from other actions. While marks exist, show a bar above the list ('2 marked - Delete 2 - Export 2 - Clear (esc)') and relabel buttons ('Delete 2 marked'). List up to 5 names in the confirm dialog. Store marks as entity ids in screen state, kept across page, search, sort and tag changes and cleared only on mode switch or an explicit Clear; show '12 marked (9 not shown) · Delete 12 · Export 12 · Clear'; the confirm dialog must then name off-screen items (up to 5, plus 'and 7 more').

*Library frame:* Library Prompts keeps Delete inside 'More actions' with danger styling and exposes multi-select as a visible 'Select' control.

*Verifier corrections applied:* Marked rows do show a '●' glyph and an 'N marked' count, and the dialog confirms - hence severity 2. Gap round (G3-05): pruning marks to the rendered page is an explicit acceptance criterion of TASK-2091 / F-040 and a safety choice (bulk Delete can never hit unseen rows). If marks persist, the dialog must name the off-screen items so that safety is kept. Severity stays 2.

<a id="rp-040"></a>
<a id="a2-rp-040"></a>
##### RP-040 · Create and save rules differ by mode: New dictionary/lore saves 'Untitled' at once, entries save instantly, settings need Save, a greeting box is overwritten by browsing

P2 · severity 2 · usability · scope content · effort M · confidence medium  
*Principle:* H4 Consistency and standards; H3 User control (no cancel) · *Area:* Create/edit flows across the four modes · *Sizes:* all · *Frequency:* Every create; frequent edits  
*Sources:* NA-24, TF-10, missed:nng-b#3 · *Verdict:* CONFIRMED / PARTIAL (create-then-rename is documented)

Evidence:
- `personas_screen.py:8022-8051, 8220-8248` (code; NA-24/TF-10): New in Dictionaries/Lore writes 'Untitled dictionary' / 'Untitled world book' immediately and lands in Settings
- `personas_screen.py:6483-6518, 6640-6683; personas_dictionary_detail.py:285; personas_lore_detail.py:221` (code; NA-24): Entry Add/Update/Delete persist at once; Settings need 'Save settings'
- `personas_character_editor_widget.py:1827-1842` (code; NA-24): Highlighting a greeting row reloads the shared Add/Update box
- `rv-flows-2 160x45` (live; TF-10): List showed 'Untitled dictionary 0 entries - on' before any save
- `vf-nng-b` (live; missed:nng-b#3): After Add the form reloads the first entry ('speaking-stone'), not the new one

*Who it hurts:* Users cannot predict what is saved. A stray New or Ctrl+N leaves an 'Untitled dictionary' row that needs a confirmed delete; arrowing through alternate greetings overwrites the one being typed; after Add the form jumps to the first entry, giving no confirmation of what was saved.

*Recommendation:* Make New a draft with Create/Cancel for all four kinds (or keep create-then-rename but make Save reachable, RP-005). Entry edits either autosave with a visible 'Saved' status or are staged behind a dirty bar. After Add, select and scroll to the new row. Split 'New greeting' from 'Edit selected' and never reload the box over unsaved text.

*Library frame:* Library Prompts uses one state machine (new: 'Save prompt + Cancel'; dirty: 'Save changes + Discard changes'); Notes autosaves with visible status. Pick one model per record type and show it.

*Verifier corrections applied:* Persisting on New is a documented create-then-rename pattern (lore-books.md:138-143) - a lower-priority preference, not a missing cancel. The 'after Add the form reloads the first entry' half is the row-0 reset in RP-014 (gap round G3-01).

<a id="rp-041"></a>
<a id="a2-rp-041"></a>
##### RP-041 · One concept, several names: Lore / lore book / world book / World Books; 'Dictionaries' here but 'Chat Dictionaries' in Console

P2 · severity 2 · usability · scope both · effort S · confidence high  
*Principle:* H4 Consistency and standards; H2 Match (ecosystem vocabulary) · *Area:* Terminology across Roleplay, Console and the guide · *Sizes:* all · *Frequency:* Every world-info task; every Console-to-Roleplay round trip  
*Sources:* NA-17, IA-04 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:456, 4498, 6816, 8230, 8234, 8269` (code; NA-17/IA-04): 'Lore', 'Select a lore book', 'Export World Book', 'Untitled world book' next to 'A lore book with that name already exists.'
- `personas_character_world_books.py:93-124; personas_library_pane.py:348; world_book_picker.py:53, 67; personas_lore_tryit.py:90` (code; IA-04): 'World Books (N)', 'Attach world book...', 'Import a world book (JSON).', 'Search world books...', 'world book scan'
- `Widgets/Console/console_inspector_ownership.py:159-162; UI/Console_Modules/retrieval.py:570; app_command_providers.py:291` (code; IA-04): Console: 'Chat Dictionaries', 'World Books', 'No world books in play'; palette 'world books'
- `captures/review-rv-nng-a-isolde-worldbook-stale-snapshot-160x45 rows 37-39` (capture; NA-17): Chip 'Lore' and section 'World Books (1)' on one screen

*Who it hurts:* Users cannot be sure the 'World Books' on a character are the 'Lore' objects; 'Attach world book' in Console leads to a screen with no 'World books' mode. The new rail must pick one label per row.

*Recommendation:* Canonicalise 'Lore books' (short 'Lore'; gloss 'facts added on keywords - aka world info / lorebook') and 'Chat dictionaries' (short 'Dictionaries'; gloss 'find & replace rules (regex)') everywhere: rail, list header, card sections, pickers, Import/Export tooltips, default names ('Untitled lore book'), toasts, Console Inspector headings, palette aliases and the guide.

*Library frame:* Library uses one noun per destination with a deliberate short title (Conversations -> 'Chats').

<a id="rp-042"></a>
<a id="a2-rp-042"></a>
##### RP-042 · The Roleplay list rail is titled 'Library', next to the separate ⌃3 Library destination and an 'Open in Library' action

P2 · severity 2 · usability · scope frame · effort S · confidence high  
*Principle:* H4 Consistency and standards; wayfinding · *Area:* Rail title, collapsed handle, transcript action · *Sizes:* all · *Frequency:* Constant  
*Sources:* NA-18, IA-03, LF-16 · *Verdict:* CONFIRMED

Evidence:
- `personas_library_pane.py:250-264; personas_screen.py:1590-1597, 1673-1676` (code; NA-18/IA-03): Rail title and handle 'Library' ('Open Library rail'); 'Open in Library' goes to the ⌃3 destination
- `captures/roleplay-character-selected-160x45 rows 2, 14` (capture; NA-18): '⌃3 Library' and 'Library <' together
- `Docs/User_Guide/roleplay-chat-dictionaries.md:59-63` (doc; IA-03): Guide calls the left pane 'Library (left rail)'
- `Widgets/Library/library_adaptive_reader_shell.py:31-36; library_rail.py:1076` (code; LF-16): Library: 'Navigation' rail; grip names {'Library': 'Nav'}

*Who it hurts:* 'Library' means two places on one screen; rebuilding Roleplay on the Library frame while keeping the name would make the two destinations look alike and deepen the confusion.

*Recommendation:* Title the rail 'Roleplay' (or 'Navigation'), title the list pane with the mode noun ('Characters (28)'), reserve 'Library' for the destination. Keep DOM ids (#personas-library-*) unchanged per ADR-004.

*Library frame:* Library calls its own rail 'Navigation' and names grips by role ('Nav', then the list noun).

<a id="rp-043"></a>
<a id="a2-rp-043"></a>
##### RP-043 · Product jargon and look-alike verbs: New / New Actor Pack, Import / Import Actor Pack, four 'Buddy' buttons, 'Type: lore', explained only in tooltips

P2 · severity 2 · usability · scope both · effort M · confidence high  
*Principle:* H2 Match between system and the real world; H8; information scent · *Area:* Rail toolbar; Inspector labels · *Sizes:* all; worst at 80x24 · *Frequency:* Every create/import (job 1); first encounters  
*Sources:* NA-22, IA-09, NB-24 · *Verdict:* CONFIRMED

Evidence:
- `personas_library_pane.py:262-296, 341-367` (code; NA-22/IA-09): Five toolbar verbs; tooltips 'Create a local actor with a required portrait and portable identity.' / 'Review and activate a portable .tldw-actor-pack archive.'; Personas mode hides plain Import
- `backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md:20-23` (doc; IA-09): An Actor Pack is an envelope holding one Character or Persona
- `captures/roleplay-characters-arrival-80x24 rows 14-20` (capture; NA-22): 'New / New / Import / Import / Duplic / Sort: / Tag:'
- `personas_inspector_pane.py:233-243; persona_profile_editor_widget.py:111, 159` (code; NA-22): 'Type: character', 'Validation: OK', 'Required portrait Character', 'Persona Visual operational states'
- `Docs/User_Guide/buddy.md:215, 250` (doc; NB-24): 'Actor Pack' appears only in the Buddy guide

*Who it hurts:* Someone holding a SillyTavern PNG must choose between two Import buttons and decode 'actor'. Personas mode has no plain Import - only 'Import Actor Pack'. 'Actor Pack' is a file format (ADR-074) presented as a fifth kind of thing.

*Recommendation:* One 'Import...' that detects card PNG/JSON, .tldw-actor-pack, lorebook JSON and dictionary JSON/MD and shows a review step. Create rows: New character, New character from concept..., New character from portrait... (today's New Actor Pack), New persona, New lore book, New chat dictionary. Actor Pack becomes an export format under More actions. Drop 'Type:'; show validation only when it fails. Add an 'Actor packs & Buddy' section to the guide.

*Library frame:* Library has one primary 'Import...' and a Create section with explicit nouns ('New note', 'New prompt').

*Verifier corrections applied:* The persona look block's own jargon and layout are RP-089 (gap round).

<a id="rp-044"></a>
<a id="a2-rp-044"></a>
##### RP-044 · Validation copy: 'Validation: OK' is a constant for every saved item, and editor errors show internal widget ids ('personas-char-editor-name: required')

P2 · severity 2 · bug · scope content · effort S · confidence high  
*Principle:* H9 Error messages in plain language; H1 truthful status; H8 · *Area:* Inspector summary lines; editor validation footer · *Sizes:* all · *Frequency:* Every selection; each failed save  
*Sources:* NB-17, IA-11 · *Verdict:* CONFIRMED

Evidence:
- `personas_inspector_pane.py:399-400, 533; personas_screen.py:5337, 5428, 15497, 15781, 15847, 15875` (code; IA-11): Every Inspector show_validation call passes () -> 'Validation: OK'
- `personas_character_editor_widget.py:1494; persona_profile_editor_widget.py:601` (code; NB-17/IA-11): show_validation(f"{fid}: {msg}") - the widget id is the field name (TASK-1651, still To Do)
- `captures/review-rv-nng-b-editor-empty-name-save-160x45 rows 37-40` (live; NB-17): 'Validation errors: personas-char-editor-name: required'; Inspector 'Validation: editing...'; Name outlined red (good)

*Who it hurts:* The top Inspector lines carry no information (Selected/Type repeat the card; Validation is always OK), while real errors read like a crash dump, below the fold, far from the field.

*Recommendation:* Drop 'Selected:'/'Type:' (the work-pane title covers them, RP-052). Show validation only when there are findings ('2 problems - jump'). Map field ids to labels and render errors inline under the field (keep the red outline). Close TASK-1651 with a test that no '#personas-'/'personas-char-editor' string reaches user-facing text.

*Library frame:* Library editors use labelled fields and plain toasts; metadata lives in an Info mode.

<a id="rp-045"></a>
<a id="a2-rp-045"></a>
##### RP-045 · The test chat misrepresents what is in play: it runs with nothing selected, labels replies 'character:', and always calls you 'User'

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; H2 Match · *Area:* Test chat (preview pane) · *Sizes:* all · *Frequency:* Every test chat of a persona, or with nothing selected; every user who set a display name  
*Sources:* NA-08, NB-23, IA-02, missed:ia#2 · *Verdict:* PARTIAL (corrections applied) + verifier-sourced display-name bug

Evidence:
- `UI/Persona_Modules/personas_preview_controller.py:50-100` (code; NA-08): Preview prompt = card fields + greeting
- `personas_preview_controller.py:98, 182; personas_preview_pane.py:39-40, 110, 234` (code; missed:ia#2): user_name="User"; replace_placeholders(g, name, "User"); default labels 'character' / 'User'
- `personas_screen.py:5325, 15492, 4490` (code; NA-08): set_speakers only on character selection/save; preview display by mode only
- `captures/review-rv-nng-a-testchat-nothing-selected-160x45; review-rv-nng-b-personas-testchat-nothing-selected-send-160x45 rows 14-24` (live; NA-08/NB-23): Reply 'Staying in character as Stay in character..' with 'Selected: none'
- `captures/review-rv-nng-a-persona-testchat-speaker-160x45 row 16` (capture; NA-08): Persona reply labelled 'character:'
- `UI/Screens/settings_screen.py:19313-19330` (code; IA-02): The only human-identity control: Settings > Console 'Default chat display name'

*Who it hurts:* A test with nothing selected looks like a real one (and offers 'Send to Console draft'); persona replies are labelled 'character:'; {{user}} renders as the literal 'User' even after the user sets a display name, so the preview misstates how the character will address them.

*Recommendation:* Disable the test chat and its Send button until a character or persona is selected ('Select a character or persona to try it'). Label replies with the entity's name. Use the ADR-046 display name (Settings) for {{user}} and the human label. Add an 'In play' line: 'System prompt + greeting - attached lore: not used' (until RP-004 is decided). Do NOT add character-attached lore to the preview unless Console applies it too.

*Library frame:* Library's empty reader shows one instruction ('Select a note to edit it here.') instead of live controls.

*Verifier corrections applied:* NA-08's 'preview and Console diverge' premise was refuted: the native Console also ignores character-attached lore (RP-004), so the preview is truthful there. IA-02 reframed: ADR-046 deliberately puts the human display name in Settings (a Roleplay 'You' row is optional, decision D3); the concrete defect is the hard-coded 'User'. Gap round: for personas the preview and Console do diverge, on which fields are sent and on draft vs saved record (RP-077).

<a id="rp-046"></a>
<a id="a2-rp-046"></a>
##### RP-046 · After saving a new first message, the test chat keeps greeting with the old one, even after Reset

P2 · severity 2 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status (feedback must reflect the current model) · *Area:* Test chat <-> character editor · *Sizes:* all · *Frequency:* Every edit-then-test cycle on a character previewed earlier in the visit  
*Sources:* TF-05 · *Verdict:* PARTIAL (root cause corrected)

Evidence:
- `captures/review-rv-flows-2-j4-after-reset-stale-greeting-160x45.png` (live; TF-05): Test chat greets with the old first message while the card shows the new one
- `personas_screen.py:15466-15491, 15834-15853; personas_preview_controller.py:418-436` (code; TF-05 verification): Save updates card, cache and speaker label but never calls the preview reseed

*Who it hurts:* Authors cannot confirm a greeting edit; the test chat contradicts the card shown below it.

*Recommendation:* On a successful save, refresh the preview's greeting seed (and reseed when the transcript holds only the greeting); make Reset always use the latest saved greeting; with user turns present, show 'Card changed - Restart test'.

*Library frame:* No Library analog.

*Verifier corrections applied:* The cause is that the save path (personas_screen.py:15466-15491) and _finish_cancel_edit (15834-15853) never refresh the preview seed - not the preserve logic. Replies already use the edited prompt; only the greeting line is stale.

<a id="rp-048"></a>
<a id="a2-rp-048"></a>
##### RP-048 · Chat now always opens with the primary greeting; alternate greetings cannot be chosen

P2 · severity 2 · gap · scope content · effort S · confidence high  
*Principle:* H7 Flexibility and efficiency; H3 User control · *Area:* Chat now · *Sizes:* all · *Frequency:* Characters with alternate greetings (seeded Isolde 3, Vex 2; imported Ilsa 2)  
*Sources:* TF-16 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:7355-7398` (code; TF-16): start_chat stages the card with no greeting index; no Console module reads alternate_greetings
- `rv-flows 160x45: Ilsa > Chat now` (live; TF-16): Console opened with the primary greeting; no choice offered

*Who it hurts:* Authored alternate greetings cannot be used from Roleplay, undercutting 'import & chat fast' for community cards built around them.

*Recommendation:* Show 'Opening: 1 of 3 < >' with a preview line next to Chat now (default primary) and pass the index through the hand-off.

*Library frame:* No Library analog.

<a id="rp-049"></a>
<a id="a2-rp-049"></a>
##### RP-049 · A character's chat history is a 10-row, title-only list squeezed into the Inspector, and 'Conversations (N)' does nothing visible when the Inspector is collapsed

P2 · severity 2 · gap · scope both · effort M · confidence high  
*Principle:* Information scent; H7 (resume fast); H1 · *Area:* Character conversations (Inspector list, card button) · *Sizes:* all · *Frequency:* Every resume of a character chat  
*Sources:* NA-23, IA-16, G3-10 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:7062-7065, 7084-7100` (code; NA-23): ConversationsRequested only focuses the Inspector list; reveals it only at widths <= 60
- `captures/review-rv-nng-a-2-conversations-button-inspector-collapsed-120x36` (live; NA-23): Clicking 'Conversations (2)' with the Inspector collapsed changes nothing
- `personas_inspector_pane.py:84-87, 721-735, 772-773; Constants.py:73` (code; IA-16): max-height 10; rows are (id, title); pages of 20
- `backlog/decisions/120-character-conversation-navigation-and-local-semantic-search.md:30` (doc; IA-16): 'Roleplay complete browse'
- `captures/review-rv-gap3-vex-30-chats-160x45 rows 19-30` (capture; G3-10): 'Conversations (30)': 10 title-only rows, no date, message count or last-active time
- `captures/review-rv-gap3-vex-chats-scrolled-160x45 rows 19-27` (capture; G3-10): After 60 wheel notches the list stops at session 11 behind 'Load 20 older conversations'
- `captures/review-rv-gap3-library-conversations-36-160x45 rows 14-43` (capture; G3-10): Library: 'Aetheria session 30: Emberfall gates' / '3 messages · 16m · Default · Active', 'Page 1 of 2'

*Who it hurts:* 'Which Isolde chat was most recent?' cannot be answered (no dates, no message counts); the card button looks broken when the Inspector is collapsed.

*Recommendation:* Give the character work pane a 'Chats (N)' mode: list (title - N messages - last active) with reader and 'Resume conversation'; point the card button at it; an action that targets a collapsed pane reopens it. Coordinate with TASK-22988 and TASK-31243 (in progress). Add 'with <character>' to Library conversation rows. Use middle-ellipsis titles (RP-094).

*Library frame:* Library Conversations is a real list plus reader with 'Resume conversation' - the right model for ADR-120's 'Roleplay complete browse'. Its rows do not name the character; do not copy that gap. Library reopens a closed pane on demand.

*Verifier corrections applied:* TASK-22988/TASK-31243 own this surface; per-character search and a transcript preview with Resume already exist - the gap is metadata and placement. Gap round (G3-10, rated 1 on its own): the Inspector already has 'Search this character's chats' (personas_inspector_pane.py:261-266), so the oldest chat is reachable by typing, and newest-first order answers 'last played'. New evidence only; severity unchanged.

<a id="rp-050"></a>
<a id="a2-rp-050"></a>
##### RP-050 · The conversation view opens with a 9-row block of 3-row buttons, and the transcript gets what is left

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* Visual hierarchy (content first); space efficiency · *Area:* Character conversation (transcript) view · *Sizes:* all · *Frequency:* Every transcript view  
*Sources:* LY-16 · *Verdict:* PARTIAL (numbers corrected)

Evidence:
- `personas_screen.py:1285-1310, 1651-1678, 1689, 2316-2330` (code; LY-16): #personas-conversation-actions height 9 with 3-row buttons, before the transcript
- `captures/review-vf-layout-2-transcript-uncontaminated-120x36` (live; LY-16 verification): Actions rows 15-23; transcript from row 24

*Who it hurts:* The messages being browsed are pushed down; the most prominent thing on screen is a set of buttons (about half the pane at 120x36).

*Recommendation:* One 1-row toolbar ('< Back - Resume conversation - Send to Console draft - Open in Library') above a full-height transcript; keep the stacked layout only at <= 60 cols.

*Library frame:* Library's conversation reader puts title and one-row actions at the top and lets the transcript fill the pane.

*Verifier corrections applied:* The cited 160x45 capture was inflated by RP-001; uncontaminated at 120x36: actions rows 15-23, transcript 9 of 19 rows (captures/review-vf-layout-2-transcript-uncontaminated-120x36).

<a id="rp-051"></a>
<a id="a2-rp-051"></a>
##### RP-051 · The conversation transcript labels the user's own turns with the character's name (verifier-sourced)

P2 · severity 2 · bug · scope content · effort S · confidence medium (one live capture; code path not traced)  
*Principle:* H2 Match with the real world; H1 · *Area:* Character conversation transcript · *Sizes:* all · *Frequency:* Every transcript view (seen on one capture; cause not traced)  
*Sources:* missed:layout#4 · *Verdict:* VERIFIER-SOURCED (live capture) · **verifier-sourced**

Evidence:
- `captures/review-vf-layout-2-transcript-uncontaminated-120x36` (live; missed:layout#4): 'Captain Isolde Varga: [user] Permission to go ashore?'

*Who it hurts:* 'Captain Isolde Varga: [user] Permission to go ashore?' makes it unclear who said what in a past chat.

*Recommendation:* Label user turns with the user's display name (ADR-046), not the character's; add a regression test on the transcript widget.

*Library frame:* Library's conversation reader is the comparison surface; not checked for this.

<a id="rp-052"></a>
<a id="a2-rp-052"></a>
##### RP-052 · The work pane never names the open item, and the visual weight sits on buttons rather than content

P2 · severity 2 · usability · scope both · effort S · confidence medium  
*Principle:* H1 Visibility (where am I / what is this); visual hierarchy; Gestalt similarity · *Area:* Card, editor and detail headers; card labels · *Sizes:* all; worst at 120x36 with rails collapsed · *Frequency:* Every collapsed-rail session; every dictionary/lore visit  
*Sources:* LY-18, IA-13 · *Verdict:* CONFIRMED / PARTIAL (Library comparison corrected)

Evidence:
- `personas_character_card_widget.py:95; persona_profile_card_widget.py:69; personas_dictionary_detail.py:201; personas_lore_detail.py:135` (code; IA-13): Headings 'Character' / 'Persona Profile'; dictionary/lore details start straight with tabs
- `captures/roleplay-dictionaries-selected-120x36 rows 13-30; roleplay-rails-collapsed-160x45 rows 13-30` (capture; IA-13): Name appears only in the Inspector; with rails collapsed nothing names Isolde
- `captures/roleplay-character-selected-160x45.png` (capture; LY-18): Brightest ink: rail verbs and Inspector CTAs; item name in body weight
- `personas_character_card_widget.py:77-112; persona_profile_card_widget.py:26-33, 76-82` (code; LY-18): Single-Static 'Label: value' rows; persona card height 100% with 1fr body

*Who it hurts:* Collapsing the rails - the screen's own space-saving feature - loses 'what is this?'; dictionary and lore details have no title at all. Cards print 'Label: value' in one style, so finding 'Scenario' means reading every line; bold centred verbs outweigh the item name.

*Recommendation:* Pin one header row in the work pane, outside the scroll: '< Characters - Captain Isolde Varga - edited 2m - Local' (plus unsaved marker), tab strip beneath. Style card labels apart from values. Render rail verbs subdued. Size read-only card bodies to content so actions sit under the last field.

*Library frame:* Library work panes carry identity (Name field, '< Back to list', 'Loaded Sourdough troubleshooting') and label rows with captions. The Library reader's own title ('Conversation reader') is weak - do not copy that part.

*Verifier corrections applied:* LY-18 misread the Library rule: Library's 1fr-body + bottom action bar is the same shape as the persona card; the difference is that Library's body is an editable box. Recommend height:auto read-only bodies. Persona Edit below the fold at 160x45 is RP-001.

<a id="rp-053"></a>
<a id="a2-rp-053"></a>
##### RP-053 · Card and editor list fields in different orders, card-spec fields have no explanations, and persona rows/cards are thin

P2 · severity 2 · usability · scope content · effort M · confidence high  
*Principle:* H4 Consistency; H2 Match (jargon); H6 Recognition · *Area:* Character card and editor; persona card and rows; Voice & Speech · *Sizes:* all · *Frequency:* Every in-depth authoring session (job 2)  
*Sources:* IA-15 · *Verdict:* CONFIRMED

Evidence:
- `personas_character_card_widget.py:76-87 vs personas_character_editor_widget.py:467-602` (code; IA-15): Card: Name, Description, Personality, Scenario, First message...; editor: Name, First message, Description, Personality, System prompt, then Advanced
- `personas_character_tts_widget.py:260-267; personas_screen.py:2459-2471` (code; IA-15): Bare 'Create' voice button with no tooltip; persona rows get no meta
- `captures/roleplay-character-selected-scrolled-160x45 rows 34-39; roleplay-personas-selected-160x45 rows 20-23` (capture; IA-15): Bare Create/Edit voice buttons beside the card Edit; persona rows are bare names

*Who it hurts:* Moving between card and editor reorders the fields; 'Post-history instructions' and 'Creator notes' are unexplained (sent to the model or not?); Scenario hides under Advanced although core to roleplay; persona rows show only a name.

*Recommendation:* One field order for card and editor; move Scenario to Basics; add one-line glosses stating whether and when each field is sent to the model; rename the voice 'Create' button 'New voice profile...'; give persona rows a description snippet and on/off.

*Library frame:* Library editors use modes with a one-line gloss per field.

*Verifier corrections applied:* Dictionary/lore rows already carry meta; only persona rows lack it.

<a id="rp-054"></a>
<a id="a2-rp-054"></a>
##### RP-054 · The card and the editor scroll inside a scrolling stack (two scrollbars side by side), and the character's attachments sit below the outer fold

P2 · severity 2 · usability · scope content · effort M · confidence high  
*Principle:* Consistent, predictable scrolling; H1 Visibility · *Area:* Work pane (card, editor) · *Sizes:* all · *Frequency:* Every card/editor view  
*Sources:* LY-08, missed:fit#4 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:1634; personas_character_card_widget.py:40-47, 100-103; personas_character_editor_widget.py:406` (code; LY-08): Card (height 100%) has its own VerticalScroll inside VerticalScroll#personas-detail-stack; editor body likewise
- `captures/roleplay-character-selected-160x45 (cols 116/117); roleplay-character-editor-scrolled-160x45` (capture; LY-08): Two scrollbar thumbs in adjacent columns
- `captures/roleplay-character-editor-120x36 rows 19-32` (capture; missed:fit#4): Attachment sections below Save/Cancel in the outer scroll

*Who it hurts:* The same wheel gesture scrolls the description or the whole card depending on the pointer; a character's lore and dictionaries stay invisible until the outer scroll is discovered.

*Recommendation:* One scroll owner per pane: make #personas-detail-stack a plain container where each view owns its scroll, or drop the card's inner VerticalScroll. Move attachments into the card/editor as a section or a 'World' tab in the same scroll.

*Library frame:* Library work panes have one scroll owner (one thumb). Re-hosting helps only if the card and editor get a single scroller.

<a id="rp-055"></a>
<a id="a2-rp-055"></a>
##### RP-055 · Empty and nothing-selected states teach nothing: one truncated rail line, a blank centre, character-only guidance, and inert controls

P2 · severity 2 · gap · scope both · effort M · confidence high  
*Principle:* H10 Help and documentation (in-context onboarding); H8 (inert controls) · *Area:* Empty Personas/Dictionaries/Lore; no-selection centre; arrival · *Sizes:* all · *Frequency:* First use of each mode; every mode switch (never auto-selects)  
*Sources:* NB-09, TF-18, IA-19, LF-15 · *Verdict:* PARTIAL (corrections applied)

Evidence:
- `captures/roleplay-empty-dictionaries-160x45; roleplay-empty-lore-160x45; roleplay-empty-personas-160x45; review-rv-flows-j1-empty-p/d/l-160x45` (capture; NB-09/TF-18): 'No dictionaries yet - use New or...' truncated; blank centre; character guidance and 4 Buddy buttons in the Inspector
- `personas_library_pane.py:431-442; personas_screen.py:474-483` (code; NB-09): Single disabled ListItem; rich guidance only for Characters
- `personas_lore_tryit.py:80-84` (code; IA-19): Disabled 'Include recent turns (soon)' switch
- `personas_screen.py:1878-1924` (code; LF-15): First paint auto-selects the first row (F-031)
- `Docs/superpowers/specs/2026-07-13-roleplay-p0-reframe-northstar-design.md:18-48` (doc; NB-09): North-star promised starter templates instead of a blank void

*Who it hurts:* Newcomers get no explanation of what a dictionary or lore book does, how it reaches a chat, or an example; arrival opens an arbitrary character (alphabetically first) instead of recent chats or Import.

*Recommendation:* Give each mode a centre empty state: what it is, how it is used ('Lore books add world facts to a chat when a keyword appears. Copy one into a character or link it to a chat.'), and New / Import / Start from an example. With items but nothing selected: a one-line stub and a wider list. Consider a Roleplay landing (Continue recent character chats, Import card, New character, needs-attention) that keeps a non-void first paint with a live primary action (F-031).

*Library frame:* Library's empty landing shows 'Get started' with a 3-step strip and primary actions, capped at 96 cells; an empty work pane says 'Select a note to edit it here.' and donates width.

*Verifier corrections applied:* The purpose line does explain each mode in one line - what is missing is how it reaches a chat, an example and a call to action (severity 2, first-use). Arrival auto-select is a deliberate F-031 fix for a void first paint. LF-15's list-truncation claim was false. Palette half is RP-057.

<a id="rp-057"></a>
<a id="a2-rp-057"></a>
##### RP-057 · Palette commands promise actions they do not perform, and the nav tooltip describes things that do not exist

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* Information scent; H2 Match; ADR-011 (secondary routes must be truthful) · *Area:* Command palette; nav purpose/tooltip · *Sizes:* all · *Frequency:* Every palette or nav-hover discovery  
*Sources:* NB-15, IA-17, TF-18 · *Verdict:* CONFIRMED

Evidence:
- `tldw_chatbook/app_command_providers.py:707-786, 512-515, 576-582` (code; NB-15/IA-17): All character commands call _navigate_via_screen(TAB_PERSONAS); help 'Navigate to Character Chat tab'
- `UI/Navigation/shell_destinations.py:84-86; app_command_providers.py:282, 291` (code; IA-17): 'behavior profiles', 'user profile context'

*Who it hurts:* 'Create New Character' and 'Quick Actions: New Character Chat' only open the screen; there are no commands for import, chatting with a named character, lore books, dictionaries or personas; the nav tooltip mentions 'behavior profiles' and 'user profiles'.

*Recommendation:* Make the commands deep links (open the editor for New character; open Import...); add 'Roleplay: Import character card...', 'Roleplay: Chat with...' (fuzzy over names -> Chat now), 'Roleplay: New lore book / New chat dictionary', 'Roleplay: Go to <mode>'. Rewrite the nav tooltip: 'Characters, personas, lore books and chat dictionaries - author and test before chatting'.

*Library frame:* Not a frame issue.

<a id="rp-058"></a>
<a id="a2-rp-058"></a>
##### RP-058 · Import errors show raw parser text with no file name or accepted formats, and a file with no entries imports as a success

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* H9 Help users recognize, diagnose, and recover from errors · *Area:* Character / lore / dictionary import · *Sizes:* all · *Frequency:* Each failed or wrong-file import  
*Sources:* NB-18 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:14200-14234, 13849-13853, 14298-14367` (code; NB-18): f"Import failed: {exc}"; success regardless of entry count
- `captures/review-rv-nng-b-lore-import-bad-json-160x45; review-rv-nng-b-lore-import-wrong-shape-160x45; review-rv-nng-b-character-import-bad-png-160x45` (live; NB-18): 'not valid JSON (Expecting value: line 1 column 1 (char 0))'; 'Imported 'x'.' (0 entries, Validation: OK); 'did not contain a valid character card'

*Who it hurts:* Users importing community cards and lorebooks (job 1) cannot tell wrong file from wrong format from bug, and toasts vanish before they are read.

*Recommendation:* Format-aware copy: 'Couldn't import bad_lore.json: it isn't valid JSON (line 1). Roleplay imports SillyTavern / Character Card V2 world books (.json).' Warn on 0-entry imports ('Imported "x" but it has 0 entries - was this a world book? [Undo import]'). Keep the last import result visible in the pane.

*Library frame:* Library routes imports through an Import canvas where status persists in the work pane (copy not compared live).

<a id="rp-059"></a>
<a id="a2-rp-059"></a>
##### RP-059 · Provider failures: the test chat gives a generic error after ~50 s, and 'Chat now blocked' tells users to edit environment variables, with no button

P2 · severity 2 · usability · scope content · effort M · confidence high  
*Principle:* H9 Help users recover from errors · *Area:* Test chat + Inspector readiness · *Sizes:* all · *Frequency:* Whenever a provider is missing or down  
*Sources:* NB-19 · *Verdict:* CONFIRMED

Evidence:
- `personas_preview_controller.py:329-333, 390-395, 747, 820, 842; personas_inspector_pane.py:1019-1024` (code; NB-19): Generic error copy; non-stream retry; 'Console default if unavailable'; settings deep link not wired to the Inspector
- `captures/review-rv-nng-b-2-testchat-provider-down-160x45 rows 15-21; review-rv-nng-b-2-blocked-provider-120x36 rows 24-31` (live; NB-19): 'Provider error - try again or configure in Settings'; env-var/TOML remedy with no button

*Who it hurts:* A non-expert is told to 'Set ANTHROPIC_API_KEY or add api_key under [api_settings.anthropic]'; the test chat and Chat now can use different providers without saying so.

*Recommendation:* Add 'Open provider settings' under any blocked readiness line (reuse open_provider_settings). Classify preview failures (connection refused / auth failed / timed out) and fail fast on connection refused. Say when the test chat and Chat now use different providers.

*Library frame:* Not applicable to these Library routes; Console's own blocked state offers 'Enter continue setup' - the model to copy.

*Verifier corrections applied:* The two-provider point is established from code, not from the cited 120x36 capture; the ~50 s timing was not re-measured.

<a id="rp-060"></a>
<a id="a2-rp-060"></a>
##### RP-060 · The Roleplay User Guide contradicts the live screen on its title, buttons, readiness messages and keys

P2 · severity 2 · bug · scope content · effort M · confidence high  
*Principle:* H10 Help and documentation (accuracy) · *Area:* Docs/User_Guide/roleplay-chat-dictionaries*.md · *Sizes:* n/a · *Frequency:* Every guide consult  
*Sources:* NB-22, IA-18, missed:a11y#1 · *Verdict:* CONFIRMED

Evidence:
- `Docs/User_Guide/roleplay-chat-dictionaries.md:1, 23, 39, 47-48, 96-99, 118-143, 151-162, 175-178, 186-189, 204-214` (doc; NB-22/IA-18): Old title and palette string; retired Status row; 'Attach to Console'/'Start Chat'; 'Console ready/blocked'; Ctrl+1-4; key table omits m and s
- `Docs/User_Guide/roleplay-chat-dictionaries/characters-and-personas.md:6-8, 35, 45-56, 274-275, 306-308; chat-dictionaries.md:42-43; lore-books.md:128-131` (doc; NB-22/IA-18): 'Start Chat', 'Use for Buddy', 'Personas chip (Ctrl+2)', a no-longer-rendered '(copied into this character)' label
- `Docs/User_Guide/roleplay-chat-dictionaries.md (Keyboard)` (doc; missed:a11y#1): 'press Escape first if a field has focus'; Space 'in the Library list' - both fail live (RP-026, RP-033)

*Who it hurts:* Readers look for 'Attach to Console', 'Start Chat', a 'Status row' and Ctrl+1-4 mode keys that do not exist; the guide's Escape and Space advice fails in practice.

*Recommendation:* Rewrite the hub and child pages with each redesign slice using the canonical names (RP-041). Keep the guide's 'assistant profile' meaning of personas (it matches the owner's ruling - fix the screen, RP-011). Generate or test-pin readiness and empty-state strings. Do not add 'Verified against' stamps (dev convention).

*Library frame:* Docs/User_Guide/library.md:168-215 is the model for layout prose.

<a id="rp-061"></a>
<a id="a2-rp-061"></a>
##### RP-061 · After choosing 'Stay' on the leave guard, the nav bar highlights the destination you did not go to, and clicking it does nothing (shell)

P2 · severity 2 · bug · scope shell (app_navigation.py; affects every guarded screen) · effort S · confidence high  
*Principle:* H1 Visibility of system status; H3 User control · *Area:* Shell navigation <-> unsaved-changes veto · *Sizes:* all · *Frequency:* Every leave attempt where the user chooses Stay  
*Sources:* TF-14, missed:flows#3 · *Verdict:* CONFIRMED (reproduced live); flows#3 note unconfirmed

Evidence:
- `tldw_chatbook/app_navigation.py:294-327, 590-598; UI/Navigation/main_navigation.py:1175-1180` (code; TF-14): A veto returns False without rollback; clicks on the 'active' destination are swallowed
- `captures/review-rv-flows-2-j8-after-stay-160x45 rows 1-7; review-vf-flows-tf14-library-click-after-stay-160x45` (live; TF-14): Library highlighted above 'Editing Chef Auguste - unsaved'; further Library clicks produce no navigation

*Who it hurts:* A user who stays to save and then tries Library again finds the tab dead; the nav says they are somewhere they are not.

*Recommendation:* On a confirm_navigation veto, run the same restore_active / _notify_navigation_failure rollback as the failure path; add a regression test beside the task-2720 test. (Unconfirmed side note: the editor's first Cancel click after Stay did nothing once - re-check.)

*Library frame:* Shell-level; affects Library's guards too. Not fixed by any reframe.

<a id="rp-062"></a>
<a id="a2-rp-062"></a>
##### RP-062 · The list's focus state is invisible and its cursor uses a border colour (3.83:1); after a mode switch no cursor row is painted at all

P2 · severity 2 · usability · scope both · effort M · confidence high  
*Principle:* WCAG 2.4.7 Focus Visible, 1.4.3 Contrast; H1 · *Area:* Rail list · *Sizes:* all · *Frequency:* Constant while browsing  
*Sources:* KA-06, KA-20, missed:a11y#2 · *Verdict:* PARTIAL (root cause corrected)

Evidence:
- `css/components/_agentic_terminal.tcss:871-912; Tests/UI/test_personas_library_rail_focus_outline.py` (code; KA-06): #personas-library-rows:focus { outline: none }; cursor style without :focus
- `captures/review-rv-a11y-esc-from-search-list-focus-160x45; review-rv-a11y-dict-mode-kbd-160x45 rows 20-25` (live; KA-06): List focus changes background 1.10:1 only; after 'd' no row painted
- `vf-a11y Dictionaries, Duplicate focused` (live; missed:a11y#2): Row 20 still [bold #ededed on #777777] while the list is unfocused
- `captures/review-rv-a11y-arrival-focus-160x45.ansi rows 15, 20; review-rv-a11y-dict-selected-kbd-160x45.ansi row 22` (capture; KA-20): Cursor 3.83:1; placeholders 3.52:1; inactive tabs 3.81:1

*Who it hurts:* Keyboard users cannot tell whether the list has focus or which row Enter or Space will act on; two highlights (navy selected, grey cursor) show even while focus is on a toolbar button.

*Recommendation:* Give the cursor its own token with >= 4.5:1 text contrast and make it focus-dependent; add a non-colour cue for list focus ('▸' glyph plus bold/underline; avoid side-stripe accents per DESIGN.md:361); set ListView.index on mode switch; re-pin Tests/UI/test_personas_library_rail_focus_outline.py to assert a visible cue. Lift placeholder (3.52:1) and inactive-tab (3.81:1) tokens to AA.

*Library frame:* Library rows mark the selected item with '▸' plus underline.

*Verifier corrections applied:* The cursor is not invisible: the '-highlight' rule uses $ds-grid-line (= $tldw-boundary #777777) with no :focus condition, so it paints regardless of focus. Raising selector specificity changes nothing; the fix is a dedicated token. outline:none on the list was deliberate (task-445).

<a id="rp-063"></a>
<a id="a2-rp-063"></a>
##### RP-063 · A focused mode chip looks exactly like the active one, so two modes appear selected while tabbing

P2 · severity 2 · usability · scope frame · effort S · confidence high  
*Principle:* WCAG 2.4.7 Focus Visible; H1 · *Area:* Mode chips · *Sizes:* all · *Frequency:* Whenever chips are traversed  
*Sources:* KA-11 · *Verdict:* CONFIRMED

Evidence:
- `css/components/_agentic_terminal.tcss:857-870` (code; KA-11): Active chip and active:focus share $ds-focus-bg + bold underline
- `rp-review/a11y/steps/rv-a11y-tab2/19.ansi, 20.ansi row 10` (live; KA-11): Focus on the active chip: no change; focused inactive chip identical to the active one

*Who it hurts:* While tabbing users cannot tell which mode they are in versus which Enter will open.

*Recommendation:* In the new rail use '▸' plus bold for the current mode and a different cue for focus; never one token for both.

*Library frame:* Library has no chips; its rail marks the current destination with '▸' and uses a separate focus cue. Moving modes into a rail removes this.

<a id="rp-064"></a>
<a id="a2-rp-064"></a>
##### RP-064 · Collapsing or expanding a pane (or selecting a row at narrow width) drops keyboard focus to nothing

P2 · severity 2 · bug · scope frame · effort S · confidence high  
*Principle:* WCAG 2.4.3 Focus Order; H3 · *Area:* Rails · *Sizes:* all; worst at <=60 cols · *Frequency:* Every rail toggle  
*Sources:* KA-08 · *Verdict:* PARTIAL (rated 2: keyboard-only, recoverable with F6)

Evidence:
- `personas_screen.py:4427-4457, 2395-2425` (code; KA-08): Handlers hide the pane holding the focused button and never call .focus()
- `rp-review/a11y/steps collapsed-tab, reopen-tab, 2-narrow-after-select` (live; KA-08): Next Tab lands on '⌃1 Home' (focus was None)

*Who it hurts:* Collapsing rails to give the editor room - what the new frame encourages - throws keyboard users back to the top of the app; with RP-024 a stray Enter navigates away.

*Recommendation:* On collapse focus the grip; on expand restore the pane's last-focused descendant (default: the list's selected row); at <= 60 cols focus the work pane after a selection. Reuse the shell's focus-evacuation contract.

*Library frame:* Library moves focus to the open handle on collapse and to search on expand; its adaptive shell evacuates focus to the grip and restores the last-focused descendant (library_adaptive_reader_shell.py:317-326, 352-447).

<a id="rp-065"></a>
<a id="a2-rp-065"></a>
##### RP-065 · Button focus is a thin underline plus a 1.06-1.64:1 background shift - not the heavy accent outline DESIGN.md asks for (app-wide)

P2 · severity 2 · usability · scope app-wide (design system) · effort S · confidence high  
*Principle:* WCAG 2.4.7 / 2.4.13 Focus Appearance; DESIGN.md:226, :347 · *Area:* Buttons app-wide (Inspector, toolbar, editor) · *Sizes:* all · *Frequency:* Every button focus  
*Sources:* KA-15 · *Verdict:* PARTIAL (app-wide, deliberate; needs a ruling)

Evidence:
- `rp-review/a11y/steps/rv-a11y-tab1/13-15.ansi rows 27-28` (live; KA-15): 'Chat now' #17486e -> #194466 (1.06:1); 'Send to Console draft' 1.64:1; 'Duplicate'/'Edit' 1.37:1
- `css/components/_buttons.tcss:24-46` (code; KA-15 verification): 'outline: none' on Button:focus by design

*Who it hurts:* On a stack of ten buttons users must find a thin underline; the primary 'Chat now' focus is effectively invisible (1.06:1).

*Recommendation:* Design-system ruling (D10): the shipped _buttons.tcss deliberately uses 'outline: none' with fill + bold underline (TASK-33003.6), which conflicts with DESIGN.md:226. Either amend DESIGN.md or apply a >= 3:1 focus change (e.g. accent outline over the primary fill).

*Library frame:* Library buttons share this style - do not copy it.

*Verifier corrections applied:* Out of the Roleplay frame/content scope; the underline gives a minimal non-colour cue, so WCAG 2.4.7 is minimally met (2.4.13 is AAA).

<a id="rp-066"></a>
<a id="a2-rp-066"></a>
##### RP-066 · At 80x24 and below, Roleplay keeps three cramped panes (1 list row) and at <=60 cols shows unlabelled 'L'/'In' slivers with no Escape way back

P2 · severity 2 · usability · scope frame · effort L · confidence high  
*Principle:* Graceful degradation; H3 (clearly marked exit); H6 · *Area:* Responsive behaviour (small sizes) · *Sizes:* 80x24, <=63 cols (graceful-degradation tier) · *Frequency:* Small terminals only  
*Sources:* LY-20, KA-18, LF-18 · *Verdict:* CONFIRMED (KA-18/LF-18 rated 1 alone)

Evidence:
- `personas_screen.py:607, 2316-2318, 2383-2414; css/components/_agentic_terminal.tcss:736-756` (code; LY-20): Compact <= 90 keeps three panes (min 16/34/22); narrow <= 60 forces w-3 handles
- `captures/roleplay-character-selected-80x24 rows 11-22` (capture; LY-20): Boxes 16|40|22; one list row
- `captures/review-rv-a11y-2-narrow-arrival-60x24; roleplay-lore-narrow-rails-autocollapsed-60x24` (capture; KA-18): 'L'/'In' handles; footer 'ctrl+n new | F1 - F6...'; Escape inert
- `captures/library-narrow-after-esc-60x24` (capture; LF-18): Library: '< Library' return and 'esc back to Library'

*Who it hurts:* Outside the design centre, but at 80x24 none of the three panes is usable (1 list row, a 22-col Inspector, 34.5% of cells are borders).

*Recommendation:* Decide visibility with the resolver (D4): Inspector first, then the list to a grip, work >= 40 cols; below ~64 cols a single stage with '< Characters' (current mode) return, Escape back to the list and a leading footer chip. Reuse LibraryEmergencyReturn.

*Library frame:* Library sizes panes from content, collapses side panes in order, keeps the work pane >= 40-48 cols and below 64 cols shows a single pane with a labelled '< Library' return and an 'esc back' chip. Avoid its 80x24 run-on list and clipped 60-col reader text.

<a id="rp-067"></a>
<a id="a2-rp-067"></a>
##### RP-067 · 'Ready' (header badge and Inspector line) reads provider settings only: it names no object, and said 'Ready' before all four Chat-now sends that were refused

P2 · severity 2 · usability · scope both · effort S · confidence high  
*Principle:* H1 Visibility of system status (say what is ready); H2 Match ('Ready' should mean a reply will come) · *Area:* Destination header badge; Inspector readiness line (characters and personas) · *Sizes:* all · *Frequency:* Every visit with a character or persona selected  
*Sources:* NA-21, NB-07, IA-20, G1-03 · *Verdict:* PARTIAL (restated; raised from 1 to 2 with gap-round evidence)

Evidence:
- `personas_screen.py:4574-4597, 7238-7250` (code; NA-21): Badge from _provider_send_block_reason, which returns None unless a character/persona is selected
- `captures/review-rv-nng-b-2-blocked-provider-120x36 row 7; review-rv-nng-b-2-blocked-provider-dictionaries-header-160x45 rows 5-7` (live; NB-07): 'Blocked' in Characters, 'Ready' in Dictionaries with the same broken provider
- `captures/review-rv-gap1-isolde-selected-ready-160x45 row 7; review-rv-gap1-persona-cozy-ready-160x45 rows 7, 20; review-rv-gap1-isolde-selected-120x36 rows 7, 27` (capture; G1-03): Header 'Ready' and 'Ready to chat in Console.' for Isolde, Vex and Cozy Storyteller; all 4 resulting sends were refused (RP-072)
- `rv-gap1: after the refused Isolde send, return to Roleplay and reselect Isolde` (live; G1-03): Still 'Ready to chat in Console.'; Console header 'Ready' and Inspect 'Run: Ready' while blocked
- `personas_inspector_pane.py:1019-1025; UI/Persona_Modules/personas_preview_controller.py:345-375; personas_screen.py:7222-7250` (code; G1-03): Readiness reads chat_defaults credentials through get_provider_readiness; it never probes the send path or past outcomes

*Who it hurts:* On a default install the user is told 'Ready' 4 of 4 times and the send fails 4 of 4 times (RP-072), with nothing pointing at the cause, so later status claims lose credibility. In Dictionaries, Lore or with nothing selected the badge carries no information.

*Recommendation:* Remove the header badge. Under Chat now, say what it opens: 'Opens a new Console chat as <name> · OpenAI / gpt-4.1-mini', or the block reason with a settings link (RP-059). Optionally watch the run state of sessions Chat now created and, on a block, show 'Last chat with <name> was not sent: <reason> · Open chat'.

*Library frame:* Library's header ('Library | Local') makes no readiness claim. No claim beats a false one, but Roleplay needs an honest line next to Chat now, so the Library pattern only partly fits.

*Verifier corrections applied:* The original claim (red 'Blocked' in every mode) was false; the badge only reports 'blocked' with a character or persona selected - deliberately (precedence rule, Qodo #824-2). Gap round (G1-03): the provider-only check is deliberate (TASK-440: 'must stay cheap'; personas_preview_controller.py:351-371 calls it a config-side approximation), and no readiness check could predict RP-072's internal bug, so the false 'Ready' is mostly RP-072. Raised to 2 for the Roleplay-side residue (say what Chat now opens; report a refused chat). The Console's 'Run: Ready' while blocked is TASK-33621.2 AC#3. The 'dry-run the trace build for readiness' idea was dropped as disproportionate.

<a id="rp-082"></a>
<a id="a2-rp-082"></a>
##### RP-082 · Chat now does not say what the chat will contain: a persona chat becomes a 53-tool agent with an ~8,000-character prompt, and character chats carry instructions the user never sees

P2 · severity 2 · usability · scope content · effort M · confidence medium  
*Principle:* H1 Visibility of system status (what is in play); H2 Match with the real world; H10 Help and documentation · *Area:* Chat now (persona vs character) -> Console request shape · *Sizes:* all · *Frequency:* Every Chat now  
*Sources:* G1-05 · *Verdict:* PARTIAL (by design but opaque)

Evidence:
- `rp-review/gapcheck/bodies.jsonl 20:41:13 (rv-gap1-2, capture off, Cozy Storyteller Chat now)` (live; G1-05): tools=53; 8,085-char system prompt; the persona's prompt ('Tell gentle stories with a calm ending.') is 39 chars, followed by 'You are a capable assistant with optional tools… Keep replies concise.' plus run-log, Canvas and Mermaid guidance and a workspace note
- `bodies.jsonl 20:39:27 / 20:40:27 (Isolde / Vex Chat now)` (live; G1-05): tools=0; system prompt = system_prompt + Personality + Description + Scenario + Example dialogue + 'You already opened this conversation with…' + an injected 'Emote: <state>' instruction
- `captures/review-rv-gap1-persona-cozy-ready-160x45 rows 17-21` (capture; G1-05): The persona Inspector shows only 'Tool policy: no rules' and 'Ready to chat in Console.'
- `captures/review-rv-gap1-2-persona-captureoff-after-160x45 rows 40-41` (capture; G1-05): Status strip: 'Run: Agent running.' beside 'Agent blocked'

*Who it hurts:* A J4 author tuning a 'warm bedtime-story voice' gets a concise tool assistant with a one-line persona prefix and cannot see why; characters and personas behave differently with no explanation; the 'Emote:' instruction and example dialogue shape replies invisibly.

*Recommendation:* Under Chat now (the Inspector today, the work-pane header in the redesign) add an 'In this chat' summary. Character: 'System prompt + greeting + example dialogue · Lore: not applied (D6) · Tools: none'. Persona: 'Persona prompt + Console agent · 53 tools (Tool policy: none)'. Add 'Preview prompt' to show the exact system text, and rename 'Tool policy: no rules' to 'Tools: all allowed (53)'. Whether persona chats should run as a tool agent at all is part of D15. Pairs with RP-045, RP-023 and RP-077.

*Library frame:* The Library prompt editor shows the full instruction text with a helper line ('Instructions - Instructions the model always follows.'). Copy that transparency into Roleplay's Info view.

*Verifier corrections applied:* Persona-as-Console-agent is consistent with ADR-037 (a persona has its own tool policy) and ADR-149 (persona Chat now is a persona-bound native Console session), and the emote line belongs to the expression feature: this is a transparency gap, not a defect in what is sent. 'The instructions outweigh the persona' is an inference the mock cannot test. The persona's first reply took ~12 s ('Connecting tools… · 11s') against ~1 s for characters; the gap lens attributes most of it to a personal-context bootstrap timeout under the isolated HOME, so its cause is unresolved and it is not counted. The strip's 'Agent blocked' beside 'Run: Agent running.' also appears before any send and in character chats: a Console status-strip issue.

<a id="rp-083"></a>
<a id="a2-rp-083"></a>
##### RP-083 · Roleplay's route for putting lore into a chat (Lore > Attachments > Attach to conversation…) lists bare chat titles and reports success with a raw conversation UUID

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* H6 Recognition rather than recall; H2 Match with the real world; H7 Flexibility and efficiency · *Area:* Lore / Dictionary detail > Attachments > 'Attach to conversation…' · *Sizes:* all · *Frequency:* Every conversation-level attach from Roleplay  
*Sources:* G1-06 · *Verdict:* PARTIAL ('only path' claim dropped)

Evidence:
- `rv-gap1-2 160x45: Lore > Station Kepler-9 Ops Manual > Attachments > Attach to conversation… > 'Chat with Captain Isolde Varga' > Attach > Console > resend` (live; G1-06): Body 20:42:34: the user turn now starts with both Kepler-9 entries; WORLD_INFO changed=True
- `captures/review-rv-gap1-2-kepler-attach-picker-160x45 rows 11-21` (capture; G1-06): Nine bare titles; Attach and Cancel are text rows
- `captures/review-rv-gap1-2-kepler-attached-to-isolde-conv-160x45 rows 17-18` (capture; G1-06): Result table 'conversation | id' with '9e83affa-9e66-4946-b4d9-71b6cd1c455d'

*Who it hurts:* The user has to recall which identical-looking title is the chat they just opened, and gets a UUID as confirmation.

*Recommendation:* Picker rows 'title · character · last active · N messages', with the open Console chat first and marked 'open'. Replace the id column with 'Last active' plus 'Open'. Point Roleplay's attachment help at the Console Inspect attach ('World Books: Attach world book…'), or have Chat now carry the character's attachments (D6 option C).

*Library frame:* Library conversation rows carry '2 messages · 1m · Default · Active'; that row format fixes the recognition problem. The frame does not.

*Verifier corrections applied:* Not the only path: the Console Inspect rail already has 'World Books: Attach world book… / Detach world book…' and 'Chat Dictionaries: Attach dictionary…' for the active conversation (UI/Console_Modules/retrieval.py:584-624; chat_screen.py:2780-2806, 14478-14545), seen live on vf-gap1 though below Inspect's fold; its picker lists books, not conversations, so it avoids the recall problem. The suggestion to add 'Lore…' to the Console chat was dropped.

<a id="rp-084"></a>
<a id="a2-rp-084"></a>
##### RP-084 · In persona and generic Console chats, the Console rail's Character header names an unrelated character ('Character · Dungeon Ma…')

P2 · severity 2 · bug · scope external (Console left rail) · effort S · confidence high  
*Principle:* H1 Visibility of system status; H2 Match with the real world · *Area:* Console left rail, Character section header (where Chat now lands) · *Sizes:* 160x45 and wider (collapsed to '▼ Character · Model · +2' at 120x36) · *Frequency:* Every persona or generic Console chat once any character chat exists  
*Sources:* G1-07 · *Verdict:* CONFIRMED

Evidence:
- `tldw_chatbook/UI/Console_Modules/left_rail.py:546, 552-560, 2263` (code; G1-07): _character_context_title falls back to state.groups[0] and renders 'Character · <label> · <n> chats'; no test pins it
- `captures/review-rv-gap1-persona-cozy-after-160x45 rows 34-35` (capture; G1-07): 'Character · Dungeon Ma ▾' above 'No current character' while the strip says 'Persona: Cozy Storyteller'
- `captures/review-rv-gap1-3-persona-send-with-leaked-source-160x45 row 34; review-rv-gap1-3-inspector-draft-sent-160x45 row 33; review-rv-flows-2-j7-persona-chat-blocked-160x45 row 35` (capture; G1-07): 'Character · Detective ▾' in persona and generic chats

*Who it hurts:* Right after a persona Chat now, the Console names a character the user never chose. It looks as if the wrong identity is in play, and it muddies diagnosis when a send fails.

*Recommendation:* With no current group, title the section 'Character · none'; in persona sessions title it 'Persona · <name>' or hide the Character section; never fall back to groups[0]. Test: persona Chat now, then assert the header contains no character label.

*Library frame:* Library rail rows only describe the open destination; there is no fallback-to-first pattern.

*Verifier corrections applied:* Console left-rail scope, outside the Roleplay redesign. Moderated by the body row ('No current character') and the status strip, which name the persona correctly; severity 2 is the ceiling.

<a id="rp-085"></a>
<a id="a2-rp-085"></a>
##### RP-085 · New persona and Edit open at the previous session's scroll position, so the focused Name field can be off-screen and typing goes in blind

P2 · severity 2 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; WCAG 2.4.7 / 2.4.11 Focus visible / not obscured · *Area:* Persona editor load (demand-mounted, reused instance) · *Sizes:* all; most visible at 120x36 · *Frequency:* Every New/Edit after an editor session that was scrolled  
*Sources:* G2-07 · *Verdict:* CONFIRMED

Evidence:
- `rv-gap2-2 120x36: scroll Cozy's editor to the bottom > Cancel > Ctrl+N` (live; G2-07): The form opened on the visual-pack buttons; typing 'Z' made the header 'New persona - unsaved' and put 'Z' in the off-screen Name (captures/review-rv-gap2-2-new-persona-opens-scrolled-to-bottom-120x36 rows 16-31)
- `vf-gap2-2 120x36: Cozy > Edit > wheel to the bottom > Cancel > Ctrl+N` (live; G2-07 verification): Reproduced (captures/review-vf-gap2-2-new-persona-after-scrolled-edit-120x36)
- `persona_profile_editor_widget.py:219-338; personas_screen.py:16387-16402` (code; G2-07): load_persona / new_persona / begin_actor_pack_creation never reset the body's scroll; _focus_editor_name focuses Name without bringing it into view; the editor is reused (ADR-115)

*Who it hurts:* The user lands mid-form for a new persona; keystrokes go into a field they cannot see, and from a non-text focus the same letters would switch modes (RP-033).

*Recommendation:* Call query_one('#personas-editor-body').scroll_home(animate=False) in load_persona, new_persona and begin_actor_pack_creation, then call_after_refresh(name.focus). Apply the same to the character editor if it shares the pattern.

*Library frame:* Not a frame issue. The Library opens editors on their primary field, but this reuse-and-scroll case is not verified there, so do not assume the shell provides it.

*Verifier corrections applied:* Textual's focus() does schedule a scroll_to_center (textual/screen.py:1138-1145); it evidently fails on the reused, just-re-displayed editor. An explicit scroll_home before focusing is still the fix.

<a id="rp-086"></a>
<a id="a2-rp-086"></a>
##### RP-086 · A persona's 'Enabled' and 'Mode' look like behaviour switches but change nothing visible: a disabled persona still chats, and nothing local reads Mode

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* H2 Match between system and the real world; H1 Visibility (state not shown); H8 (inert controls) · *Area:* Persona editor Mode/Enabled; persona list rows, card and Inspector · *Sizes:* all · *Frequency:* Every Enabled toggle; every persona edit (Mode)  
*Sources:* G2-08 · *Verdict:* CONFIRMED

Evidence:
- `rv-gap2-2 120x36: Cozy Storyteller > Edit > Enabled off > Save > Cancel` (live; G2-08): Row, card and Inspector unchanged: 'Ready to chat in Console.' (captures/review-rv-gap2-2-disabled-persona-no-visible-state-120x36 rows 15-26)
- `the same disabled persona > Chat now` (live; G2-08): Console opened 'Chat with Cozy Storyteller' (captures/review-rv-gap2-2-disabled-persona-chat-now-console-120x36 rows 7-8)
- `personas_screen.py:7330-7398; UI/Navigation/buddy_management.py:467; Chat/console_persona_assignment.py:46-56` (code; G2-08): Chat now has no is_active check; Enabled gates only Buddy (active_only list; 'Selected Persona is unavailable or inactive.')
- `personas_screen.py:2459-2475 vs 4268` (code; G2-08): Persona rows get no meta line; dictionary rows show on/off
- `ccp_persona_handler.py:84, 245; personas_screen.py:15674, 15683; Actor_Packs/contracts.py:571-575; persona_profile_editor_widget.py:143-150` (code; G2-08): Mode is defaulted and validated but never read at runtime locally; the editor offers the raw literals

*Who it hurts:* Turning a persona off is expected to take it out of use, but it only hides it from Buddy; users cannot tell which personas are off; Mode invites a decision with no local consequence.

*Recommendation:* Rename Enabled 'Available to Buddy' with a help line, or make it gate Chat now ('Cozy Storyteller is off. Turn it on to chat.'). Show 'off' on the row and in the Profile. Hide Mode in local mode until something reads it, or wire it with plain labels ('This chat only' / 'Remembers across chats'). Supersedes the persona-Mode line in RP-003.

*Library frame:* Roleplay's own dictionary rows already show on/off (personas_screen.py:4268); the Library frame does not address it.

*Verifier corrections applied:* Mode is a tldw_server-parity field and is sent on the server runtime path, so it is inert only for local personas; 'hide Mode' applies to local mode.

<a id="rp-087"></a>
<a id="a2-rp-087"></a>
##### RP-087 · Four Save buttons with different scopes in one persona editor, and the tool-rule Save commits at once while its status says the persona Save will

P2 · severity 2 · usability · scope content · effort M · confidence high  
*Principle:* H4 Consistency and standards; H5 Error prevention; H1 (contradictory save state) · *Area:* Persona editor commit model (form / Buddy pack / reactions / policy rules) · *Sizes:* all · *Frequency:* Every persona session that touches visuals or policy  
*Sources:* G2-09 · *Verdict:* CONFIRMED

Evidence:
- `persona_profile_editor_widget.py:174; personas_persona_visual_pack_widget.py:291-300; personas_visual_identity_pack_widget.py:167-178; personas_policy_rules_editor.py:97-99` (code; G2-09): Four commit controls: persona Save, Save Pack / Cancel Draft, reactions Save / Cancel, rule Save / Delete
- `personas_policy_rules_editor.py:243-261 vs personas_screen.py:15550-15601` (code; G2-09): Rule status 'Saved rule … - persona save persists it.' while the screen persists immediately ('immediate local-record update'); rule Delete is instant
- `captures/review-rv-gap2-policy-rule-saved-blind-while-form-unsaved-160x45 rows 6, 19-20` (capture; G2-09): Toast 'Tool policy rules saved.' while the header reads 'Editing Cozy Storyteller - unsaved'
- `personas_screen.py:15624-15629` (code; G2-09): Persona Save is refused while a visual draft is unsaved: 'Save or Cancel Persona Visual changes before saving the Persona.'

*Who it hurts:* Users cannot tell what 'saved' means: rules commit instantly, the Buddy pack needs its own button, and the main Save may be blocked by a visual draft they scrolled past.

*Recommendation:* One commit model per work-pane mode, each with an anchored, named bar: Edit 'Save persona'; Reactions 'Save reactions'; Buddy states 'Publish Buddy pack'; Tool policy auto-saves each rule with 'Saved to Cozy Storyteller · Undo'. Fix the rule status copy; confirm or undo rule Delete. The in-editor version of RP-040 and RP-038.

*Library frame:* Library Prompts has one action bar per editor that switches to 'Save changes / Discard changes' when dirty and shows 'Modified now · v2' after saving. Copy one scoped bar per work-pane mode, not one per widget.

<a id="rp-088"></a>
<a id="a2-rp-088"></a>
##### RP-088 · 'Loading Shared Visual Identity reactions…' never resolves when an existing persona is opened with Edit

P2 · severity 2 · bug · scope content · effort S · confidence medium  
*Principle:* H1 Visibility of system status (a progress message that never ends) · *Area:* Persona editor: shared visual identity host · *Sizes:* all · *Frequency:* Every Edit of an existing local persona (4 of 4 sessions)  
*Sources:* G2-10 · *Verdict:* CONFIRMED (the failing fence is not pinned; medium confidence)

Evidence:
- `rv-gap2-4 160x45 clean session: Default Narrator > Edit > wait 15 s > scroll` (live; G2-10): Still 'Loading Shared Visual Identity reactions…' (captures/review-rv-gap2-4-shared-visual-loading-forever-160x45 row 23); the same for Cozy on rv-gap2, rv-gap2-2 and rv-gap2-3
- `vf-gap2 160x45 clean session: Cozy > Edit > wait ~17 s` (live; G2-10 verification): Still loading; after Save it became the single row 'Shared Visual Identity reactions'
- `persona_profile_editor_widget.py:154-157, 272-276; personas_screen.py:8360-8368, 8490-8496` (code; G2-10): load_persona sets the Loading copy; the configure worker returns silently when its first freshness fence fails; nothing clears it

*Who it hurts:* The user waits on a loading message that never ends, and cannot tell whether reactions exist, failed or are unsupported.

*Recommendation:* Make the configure worker always end in a terminal state (the browser, 'No reactions yet: Add…', or 'Unavailable: <reason>' with Retry); log fence failures at debug; test that Edit on a saved local persona leaves no 'Loading' copy once the worker drains.

*Library frame:* The Library shows '(…)' while counts load and always resolves it; the frame does not fix a worker that never reports.

<a id="rp-089"></a>
<a id="a2-rp-089"></a>
##### RP-089 · The persona's look is a fixed 33-37-row block of implementation names and 3-row buttons whose labels truncate at 120x36

P2 · severity 2 · usability · scope content · effort M · confidence high  
*Principle:* H2 Match between system and the real world; H8 Aesthetic and minimalist design; H4 (two titles for one thing) · *Area:* Persona editor: 'Persona Visual operational states' / 'Persona Visual' pack; Actor Pack titles · *Sizes:* all; truncation at 120x36 · *Frequency:* Every persona edit (the block sits between the fields and the policy)  
*Sources:* G2-11 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap2-4-shared-visual-loading-forever-160x45 rows 23-41` (capture; G2-11): Two titles; 'Local Persona - active visuals stay unchanged until Save Pack.'; 'Ready - 0 assets • incomplete'; 'Preview unavailable.' beside 'Idle  empty'; states include 'Wake Armed' and 'Tool Running'
- `captures/review-rv-gap2-2-persona-editor-bottom-policy-clipped-120x36 rows 16-28` (capture; G2-11): Buttons truncate to 'Create', 'From Buddy', 'Export saved'; a nested scrollbar inside the editor's
- `widget_defaults_self.tcss:2853-2930; personas_persona_visual_pack_widget.py:270-323` (code; G2-11): height 33 (37 when narrow); the actions grid is 12 rows of 3-row buttons, 10 buttons
- `captures/review-rv-gap2-2-new-actor-pack-portrait-select-hidden-120x36 rows 6, 15` (capture; G2-11): Header 'New persona' vs pane 'New Persona Actor Pack'; nothing explains what an Actor Pack is

*Who it hurts:* J4 users meet the persona's appearance as Buddy-runtime jargon with dead buttons, and cannot tell 'empty' from 'broken' without the Buddy guide.

*Recommendation:* One 'Look' mode with two labelled tabs: 'Reactions (expressions in chat)' and 'Buddy states (desktop companion)'; one-line glosses; 'No image yet' instead of 'Preview unavailable.'; 'Ready - 0 of 9 states set'. Show only the actions valid now and move Create character…, Buddy archive, Petdex and Export to More actions. height:auto with no fixed 33/37 and 1-row buttons. Header 'New persona (Actor Pack)'. Keep state names consistent with the Buddy guide (ADR-122/139/144). Extends RP-043.

*Library frame:* The Library uses plain nouns with short glosses; the frame does not rename these.

*Verifier corrections applied:* State names such as 'Wake Armed' and 'Tool Running' come from the Buddy runtime contract (ADR-122/139/144 family); renaming them in the editor must stay consistent with the Buddy guide and modal.

<a id="rp-090"></a>
<a id="a2-rp-090"></a>
##### RP-090 · The link from a persona to the character whose portrait it uses is invisible and cannot be changed after creation

P2 · severity 2 · gap · scope content · effort M · confidence high  
*Principle:* H1 Visibility of system status; H6 Recognition rather than recall; H3 User control and freedom · *Area:* Persona list, card and editor; character_card_id · *Sizes:* all · *Frequency:* Every linked persona (all Actor Pack personas)  
*Sources:* G2-12 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap2-4-linked-persona-card-160x45 rows 15-19; review-rv-gap2-3-linked-persona-editor-no-link-220x55 rows 16-24` (capture; G2-12): Card and editor show only Name and Description
- `persona_profile_editor_widget.py:455-460; personas_screen.py:15671-15705` (code; G2-12): character_card_id is collected only in Actor Pack mode; plain create and update payloads never send it
- `Character_Chat/persona_visual_identity.py:418-440; Persona_Buddy/controller.py:270-290` (code; G2-12): The only consumers load the linked character's portrait
- `rv-gap2-4 160x45: linked persona > Chat now (captures/review-rv-gap2-4-linked-persona-chat-now-console-160x45 rows 30-31, 41)` (live; G2-12): Console binds the persona; the Character section reads 'No current character'

*Who it hurts:* Users cannot see or manage which character a persona borrows its face from; deleting that character or changing its portrait silently changes the persona; 'linked' suggests the persona speaks as or with the character, which it does not.

*Recommendation:* Show 'Portrait from: Captain Isolde Varga' (or 'No portrait character') in the persona Profile/Info, with Change…/Unlink and the gloss 'supplies the picture only'; list linked personas on the character's Info; allow setting the link outside New Actor Pack.

*Library frame:* No Library equivalent. The proposed 'Used by' / 'World' cross-object pattern (Appendix C.5) is the model.

*Verifier corrections applied:* Renaming the source character does not change the persona's portrait; deleting it, or changing its portrait, would.

<a id="rp-091"></a>
<a id="a2-rp-091"></a>
##### RP-091 · Lore entry tables size columns to the widest keys, so content shows 4-24 characters and on/off sits off-screen at every size; disabled entries are shown only by dim text

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* WCAG 1.4.1 Use of color; H1 Visibility of system status; H8 (show the content, not the chrome) · *Area:* Lore entries table (keys, content, position, priority, enabled) · *Sizes:* 120x36 (content ~4 chars), 160x45 (~23-24), 220x55 (~54); the enabled column is off-screen at all three · *Frequency:* Every lore book with multi-key entries (typical of imported lorebooks)  
*Sources:* G3-03 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap3-lore-grand-clean-120x36 rows 17-21` (capture; G3-03): The keys column is ~46 wide; content reads 'cont' / 'The'
- `captures/review-rv-gap3-lore-grand-clean-220x55 rows 21, 28` (capture; G3-03): Disabled 'Cinderreach, codex-005' is only greyed; a horizontal scrollbar shows position, priority and enabled off-screen at 220 columns
- `personas_lore_detail.py:257, 316-331` (code; G3-03): Auto-width columns keys, content, position, priority, enabled; keys joined with ', '; content cut at 60; disabled rows only 'dim', with 'no' in the last column

*Who it hurts:* Users cannot see what an entry says or whether it is on without opening it; at 250 entries that means opening them one by one. Colour-only state fails low-contrast and colour-blind users.

*Recommendation:* Columns in this order: on/off word (3 cols), first key (20-24 cols, '+2' when there are more), content (fills the rest, first line), then priority/order. Fix the keys width rather than auto-sizing to the widest cell. Mark disabled rows with an 'off' word, as dictionary rows already do (personas_dictionary_detail.py:403-404). Show all keys in the editor and a tooltip.

*Library frame:* Library rows state status in words on line 2 ('3 messages · 16m · Default · Active'). The frame does not touch table columns.

*Verifier corrections applied:* At 220x55 the viewport, not the code, limits content to ~54 characters (the code cap is 60). Dictionary rows already append a visible 'off' word, so lore is the inconsistent one.

<a id="rp-092"></a>
<a id="a2-rp-092"></a>
##### RP-092 · The page bar wraps and shifts, takes up to 3 rows under a 7-row list at 120x36, and has no page number or paging keys

P2 · severity 2 · usability · scope both · effort M · confidence high  
*Principle:* H8 Aesthetic and minimalist design; stable layout; H7 Flexibility and efficiency (keyboard); H1 ('page 2 of 7') · *Area:* Characters/Personas page bar and count line · *Sizes:* 120x36 (2-row bar + a blank count line on every page), 160x45 (1 row on page 1, 2 from page 2), 220x55 (1 row) · *Frequency:* Every visit with more than 50 characters or personas  
*Sources:* G3-06 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap3-chars-arrival-160x45 rows 23-41` (capture; G3-06): 17 list rows (8.5 of the page's 50 cards; 348 total); pager on row 40; a permanently reserved blank count line on row 41
- `captures/review-rv-gap3-page2-160x45 rows 38-41` (capture; G3-06): Page 2 wraps to two rows; '>' jumps from row 40 to 39 and the list loses a row
- `captures/review-rv-gap3-chars-120x36 rows 23-32` (capture; G3-06): 7 list rows, a 2-row pager and a blank count line
- `rv-gap3 160x45: Ctrl+F, Enter, End, Down, PageDown, Tab` (live; G3-06): End and PageDown scroll only inside the page; Tab reaches '>' (2 Tabs from page 2); no paging key; the footer says nothing about paging
- `personas_library_pane.py:80-85, 115-124, 307-322, 330, 493-503, 699, 704` (code; G3-06): BINDINGS only space/m/s; two 5-column buttons around a wrapping 1fr label; the count Static is always mounted

*Who it hurts:* At the medium sizes paging chrome takes 12-30% of the list; the pager moves under the pointer, gives no sense of position, and keyboard users must Tab out of the list to turn a page.

*Recommendation:* Preferred: one virtual list instead of pages (D14). If paging stays: one non-wrapping row in the items header ('Characters 348 · ‹ 2/7 ›'); drop the reserved blank count line; PgDn on the last row and PgUp on the first turn pages; 'pgup/pgdn page' in the footer while the list has focus.

*Library frame:* Library's pager has better copy ('1-20 of 36 · Page 1 of 2', disabled reasons) and resets scroll, but it is also 3 rows with no paging keys. Copy its wording and scroll reset, not its 3-row stack.

*Verifier corrections applied:* The stable-layout argument stands, but DESIGN.md:125 ('focus never causes layout movement') was the wrong citation: the wrap is a content-length change. The bigger chrome cost in the same rail is the toolbar stack above the list (RP-008).

<a id="rp-093"></a>
<a id="a2-rp-093"></a>
##### RP-093 · Roleplay loads only the newest 100 personas: with 114 it says 100, and the oldest (including 'Default Narrator') cannot be found even by search

P2 · severity 2 · bug · scope content · effort S · confidence high  
*Principle:* H1 Visibility of system status; integrity of browse (no silent truncation) · *Area:* Personas mode list (persona load path) · *Sizes:* all · *Frequency:* Every visit once a profile holds more than 100 personas (uncommon, but permanent once reached)  
*Sources:* G3-07 · *Verdict:* PARTIAL (rated 2: rare at the target user's scale)

Evidence:
- `rv-gap3-2 (scaled copy + 50 personas)` (live; G3-07): list_persona_profiles(limit=1000) = 114; the default call returns 100, newest first
- `captures/review-rv-gap3-2-personas-114-cap-160x45 rows 9, 40` (capture; G3-07): 'Personas … · 100' and '1-50 of 100 personas' with 114 present; 'Baroque Chronicler 09', 'Cozy Storyteller' and 'Default Narrator' missing
- `captures/review-rv-gap3-2-personas-search-default-missing-160x45 rows 21-26` (capture; G3-07): Search 'Narrator' lists 6; 'Default Narrator' is absent
- `Character_Chat/character_persona_scope_service.py:629-652; local_character_persona_service.py:1139-1170; UI/CCP_Modules/ccp_persona_handler.py:140; personas_screen.py:3943-3995` (code; G3-07): limit=100 by default, newest first; the handler passes no limit; search and paging work in memory over those 100

*Who it hurts:* Once a user creates their 101st persona, their longest-standing personas vanish from Roleplay with no message; the count looks complete, so it reads as data loss.

*Recommendation:* Page through list_persona_profiles until it is exhausted (UI/Navigation/buddy_management.py:463-475 already does) or page from the data layer as characters do, and search the full set. Until then show '100+ (showing newest 100)'. Test: with 101 personas the oldest is findable.

*Library frame:* Library rail counts match the true totals ('Conversations (36)'). This is a data-layer fix.

*Verifier corrections applied:* A real bug, but more than 100 personas is rare for the owner's target user, so severity 2. Other surfaces already page past 100 (buddy_management.py:463-475): Roleplay is the inconsistent path.

<a id="rp-094"></a>
<a id="a2-rp-094"></a>
##### RP-094 · Long names are cut at the end, so near-duplicate cards, books and chats look identical ('Highspire Gazetteer (imported fr…' twice), with no tooltip

P2 · severity 2 · usability · scope content · effort S · confidence high  
*Principle:* H6 Recognition rather than recall; information scent; telling list items apart · *Area:* Items list rows (all modes); Inspector conversation list · *Sizes:* 120x36 (~21-char names), 160x45 (~32), 220x55 (~48) · *Frequency:* Common at volume: imported packs and duplicates differ only by a suffix ('v2', '2', 'pack 22', 'copy')  
*Sources:* G3-08 · *Verdict:* CONFIRMED

Evidence:
- `captures/review-rv-gap3-lore-gazetteer-truncation-160x45 rows 26-29` (capture; G3-08): Two rows read 'Highspire Gazetteer (imported fr…'; only '6 entries' / '1 entries' differ
- `rv-gap3 160x45, Characters page 1 after End` (live; G3-08): 'Cassian Oakhaven of the Drowned…' twice ('…Coast' and '…Coast 2')
- `captures/review-rv-gap3-vex-30-chats-160x45 rows 21-30` (capture; G3-08): 30 'Aetheria session NN: …' chats truncate inside the shared prefix
- `personas_library_pane.py:96-105, 455-481` (code; G3-08): Row Statics are nowrap + ellipsis; no tooltip on ListItem

*Who it hurts:* With hundreds of imported cards or books, users open the wrong item, or both, and risk editing or deleting the wrong duplicate.

*Recommendation:* Middle ellipsis that keeps the last 6-10 characters ('Highspire Gazetteer…pack 22)'); a tooltip with the full name; when two visible rows share a truncated label, add a distinguishing line-2 field (source, created date). Size the redesign's items column so names get at least 32 columns at 160x45.

*Library frame:* The Library's wider items column shows full titles at 160x45; at 120 columns the same tail cut returns.

*Verifier corrections applied:* The Gazetteer names were seeded by the reviewer: the duplicate-suffix pattern is realistic, the long 'imported from …' prefix is synthetic.

<a id="rp-095"></a>
<a id="a2-rp-095"></a>
##### RP-095 · Tag filter picker: Enter and Down in its filter box do nothing, although the hint says 'Enter to pick'

P2 · severity 2 · bug · scope content · effort S · confidence high  
*Principle:* H7 Flexibility and efficiency of use; H4 Consistency (the hint promises Enter) · *Area:* Characters > Tag: filter picker (TagFilterPicker) · *Sizes:* all · *Frequency:* Every keyboard tag filter  
*Sources:* G3-13 · *Verdict:* CONFIRMED

Evidence:
- `rv-gap3 160x45: 'Tag: All', type 'fantasy', Enter; then 'fanta', Down, Down, Enter` (live; G3-13): The modal stayed open both times; a mouse click on 'fantasy' worked ('· 20', 'Tag: fantasy'; captures/review-rv-gap3-tag-fantasy-applied-160x45)
- `captures/review-rv-gap3-tag-picker-typed-160x45` (capture; G3-13): After filtering, no option is highlighted; footer 'Enter to pick - Esc to cancel'
- `vf-gap3 160x45` (live; G3-13 verification): Reproduced (captures/review-vf-gap3-tagpicker-enter-160x45)
- `Widgets/Persona_Widgets/tag_filter_picker.py:58-63, 87-115` (code; G3-13): Focus starts in the Input; no Input.Submitted handler; Down is not forwarded; only ListView.Selected dismisses

*Who it hurts:* At volume (about 55 tags in the scaled copy) users must filter, but the advertised Enter does nothing, so they fall back to the mouse or discover Tab.

*Recommendation:* On Input.Submitted pick the single remaining tag (or the highlighted row); forward Down/Up from the input to the list; highlight the first match after each filter; show per-tag counts ('fantasy (20)').

*Library frame:* Not applicable to the frame.

*Verifier corrections applied:* A keyboard workaround exists (Tab into the list, arrows, Enter), so this is a misleading hint and an efficiency bug, not a blocker.

<a id="rp-096"></a>
<a id="a2-rp-096"></a>
##### RP-096 · Reordering lore entries is one step per press at ~1-1.25 s each, with no 'move to position' or editable order, and the rewrite is not atomic

P2 · severity 2 · gap · scope content · effort M · confidence high  
*Principle:* H7 Flexibility and efficiency of use; direct manipulation of order · *Area:* Lore and Dictionary entries: Move up / Move down · *Sizes:* all · *Frequency:* Whenever insertion order matters (lore injection order)  
*Sources:* G3-14 · *Verdict:* CONFIRMED

Evidence:
- `rv-gap3 160x45, Move down on #200` (live; G3-14): 1.25 s from Enter to the visible table reset for one position (the verifier measured ~1.0 s)
- `personas_lore_detail.py:466-476, 621-641; personas_screen.py:4354-4359, 6686-6703` (code; G3-14): _post_reorder swaps with the neighbour only; the handler calls update_world_book_entry once per entry, each in its own thread and with no transaction, then re-renders the rail by fetching every book's entries just to count them; no editable order field

*Who it hurts:* Moving an entry from #200 to the top is effectively impossible (and with RP-014 each extra press reorders entry #1 instead), so users leave the order wrong, which silently changes which lore wins the token budget (RP-081).

*Recommendation:* Expose insertion order as an editable number in the entry editor and as a sortable column; add 'Move to top / bottom / position…'; write a reorder in one transaction; keep the moved entry selected (RP-014).

*Library frame:* The Library has no ordered lists. Ecosystem convention (prior-art.md section 6): SillyTavern exposes an editable per-entry 'Order' number.

*Verifier corrections applied:* Also a correctness issue: the per-entry loop is not one transaction, so a failure or a second press mid-loop can leave a half-renumbered book.


#### P3

<a id="rp-068"></a>
<a id="a2-rp-068"></a>
##### RP-068 · Each character takes 2 list rows; at medium sizes the second row is a 21-32-character description fragment

P3 · severity 1 · usability · scope content · effort S · confidence high  
*Principle:* Information scent; list density · *Area:* Character list rows · *Sizes:* 120x36, 160x45 · *Frequency:* Constant  
*Sources:* LY-04 · *Verdict:* PARTIAL (deliberate trade-off; severity 1)

Evidence:
- `personas_screen.py:610-638; personas_library_pane.py:458-475` (code; LY-04): 72-char snippet replaced an identical last-modified date
- `captures/roleplay-character-selected-120x36 rows 23-32` (capture; LY-04): Snippets cut to ~21 chars

*Who it hurts:* Minor; the main list-capacity loss is RP-007/RP-008. With those fixed, 2-line rows still show ~16 characters at 160x45.

*Recommendation:* Optional: a 1-line row when the rail content is < 40 cols, with the snippet in the work pane or a tooltip. Keep the snippet (it beats counts for recognition).

*Library frame:* Library rows are less dense (3 rows per item); do not copy the separator.

<a id="rp-069"></a>
<a id="a2-rp-069"></a>
##### RP-069 · The card shows raw markdown asterisks while the test chat renders them

P3 · severity 1 · usability · scope content · effort S · confidence high  
*Principle:* H2 Match; H4 Consistency · *Area:* Character card view · *Sizes:* all · *Frequency:* Cards with emphasis  
*Sources:* NA-25 · *Verdict:* CONFIRMED

Evidence:
- `personas_character_card_widget.py:94-140` (code; NA-25): Plain Static, markup=False
- `captures/roleplay-rails-collapsed-120x36 row 18` (capture; NA-25): '*Meridian's Wake*' with literal asterisks

*Who it hurts:* Cosmetic; authors cannot preview emphasis.

*Recommendation:* Render card text as escaped Markdown in view mode; keep raw text in the editor.

*Library frame:* Library Notes has Edit / Preview / Info modes.

<a id="rp-070"></a>
<a id="a2-rp-070"></a>
##### RP-070 · Copy slips: '1 entries', and a raw UTC date on new character rows

P3 · severity 1 · bug · scope content · effort S · confidence high  
*Principle:* H4 Consistency; H2 · *Area:* List row meta · *Sizes:* all · *Frequency:* 1-entry books; characters without a description  
*Sources:* TF-21 · *Verdict:* CONFIRMED

Evidence:
- `personas_screen.py:4273, 4350, 637-638` (code; TF-21): f"{count} entries"; last_modified[:10]
- `captures/review-rv-flows-j1-after-save-160x45` (capture; TF-21): '2026-10-02' on 2026-10-01 local

*Who it hurts:* Minor polish.

*Recommendation:* Pluralise ('1 entry'); show local, relative dates ('edited 2 min ago').

*Library frame:* Library uses relative local times ('Modified now').

<a id="rp-071"></a>
<a id="a2-rp-071"></a>
##### RP-071 · Roleplay ignores the ASCII glyph fallback (hard-coded ▸/▾, the ● mark and a ✨ emoji)

P3 · severity 1 · gap · scope both · effort S · confidence high  
*Principle:* ADR-034 (glyph ownership); robustness · *Area:* Glyphs · *Sizes:* all · *Frequency:* ASCII-mode users  
*Sources:* KA-16 · *Verdict:* PARTIAL (severity 1)

Evidence:
- `personas_preview_pane.py:114, 183, 442; personas_character_dictionaries.py:124; personas_character_world_books.py:123; personas_character_editor_widget.py:517, 614, 1615; personas_library_pane.py:457, 588` (code; KA-16): Literal glyphs; zero resolve_glyph calls in Roleplay code
- `Widgets/glyph_fallback.py:30-60` (code; KA-16): ▸ -> '>', ▾ -> 'v', ● -> '[*]'

*Who it hurts:* Opt-in niche; the double-width ✨ is the glyph most likely to misalign.

*Recommendation:* Route markers through resolve_glyph (GLYPH_EXPANDED/COLLAPSED from destination_rail); drop ✨ from labels; add an ASCII-mode render test.

*Library frame:* Library and the shared rail honour resolve_glyph.

<a id="rp-097"></a>
<a id="a2-rp-097"></a>
##### RP-097 · The 'Console RAG capture unavailable; reason=capture_provider_failure' ERROR on every send is a harmless argument-mismatch bug, not why sends were blocked

P3 · severity 1 · bug · scope external (Console; ledger G2-37, out of scope in TASK-33621.20) · effort S · confidence high  
*Principle:* H9 Help users diagnose errors (diagnostics must name the real cause) · *Area:* Console runtime capture-provider wiring (F3 Logs; turn trace) · *Sizes:* n/a (log) · *Frequency:* Every send with nothing staged  
*Sources:* G1-08 · *Verdict:* CONFIRMED (already in the Console ledger)

Evidence:
- `tldw_chatbook/Chat/console_runtime.py:2503-2516, 3670-3677` (code; G1-08): rag_capture_provider is _capture_frozen_console_staged_rag(draft, turn_context, launch), with launch required
- `Chat/console_chat_controller.py:23826-23863` (code; G1-08): accepts_context is True and there is no origin parameter, so it calls provider(draft, turn_context); the TypeError is swallowed and logged without its type, and a 'Retrieval failed' trace event is written (:23859)
- `rp-review/gapcheck/trace.log 'RAG PROVIDER=… sig=(draft, turn_context, launch)'; a standalone replica of the dispatch` (live; G1-08): The replica raised 'missing 1 required positional argument: launch'
- `rp-review/gapcheck/rv-gap1-run1-app.log lines 390 and 444` (live; G1-08): The same ERROR on a send that completed and on a blocked Isolde send

*Who it hurts:* Anyone reading F3 Logs, including this review's first round, is pointed at RAG capture while the real refusal (RP-072) logs nothing; it also writes a false 'Retrieval failed' event into the turn's trace.

*Recommendation:* Give launch a default of None, or have the controller pass the pending launch; log exception_type at console_chat_controller.py:23857; test _capture_rag_context with the production-wired provider.

*Library frame:* Shared Console path; not a Library concern.

*Verifier corrections applied:* Not new: it is the Console ledger's G2-37, named out of scope in TASK-33621.20; the useful addition is the root cause (the missing 'launch' argument). Benign apart from diagnostics: with launch=None the capture would return empty anyway. Outside the Roleplay redesign.


---

## Appendix B. Refuted findings

| ID | Claim | Why it was refuted |
|---|---|---|
| LF-19 | The F6 pane cycle uses static targets that always include both handles; Library Artifacts computes targets dynamically | The static tuple (personas_screen.py:1122-1155) is filtered at runtime by _available_targets/_is_available (Widgets/workbench_focus.py:55-117), which drops any hidden pane or handle. Handles are hidden while their pane is open and vice versa, so the tuple already behaves as 'pane or grip' - the same as Library Artifacts' dynamic targets. F6 never stops on a hidden handle; the claimed impact does not occur. (The separate, real F6 problem - wrong targets in Dictionaries/Lore - is RP-025.) |

Other claims were not refuted outright but were corrected inside merged findings (for example: 'preview and Console diverge on lore' (NA-08), 'red Blocked in every mode' (NA-21), 'dictionaries are live links' (NA-02), 'Library rows all carry glosses' (IA-06), 'rail sub-rows per lore book' (LF-20)). Those corrections are listed per finding in Appendix A.

The gap round refuted nothing outright (24 of its 35 findings confirmed, 11 partly confirmed and corrected). Its corrections include: 'the only way to put lore into a chat' (dropped, RP-083: Console Inspect can attach too), 'raise RP-030 to severity 3' (not adopted), 'RP-006 severity 4 at volume' (not adopted), the persona-cap severity (lowered to 2, RP-093), the 'dry-run the send for readiness' idea (dropped, RP-067), the 'near-miss' vocabulary claim (narrowed to the summary count, RP-081), the recovery card as a second severity 4 (rated 3, RP-074) and the chat-history scroll cost (the Inspector already has search, RP-049). Each is listed per finding in Appendix A.

---

## Appendix C. Lens appendices (lightly edited; finding ids mapped to RP ids)

### C.1 Layout metrics (layout lens)

Roleplay = `roleplay-character-selected-<size>` (Isolde selected, rails open). Library = `library-conversations-item-open-<size>`. Border and blank shares come from `rp-review/metrics.py`; the other figures were hand-counted from the captures.

| Measure | 120x36 RP | 120x36 Lib | 160x45 RP | 160x45 Lib | 220x55 RP | 220x55 Lib | 80x24 RP | 80x24 Lib |
|---|---|---|---|---|---|---|---|---|
| Chrome rows above first content row | 13 | 5 | 13 | 6 | 13 | 6 | 11 | 4 |
| Chrome rows below content | 4 | 2 | 4 | 3 | 4 | 3 | 3 | 1 |
| Vertical chrome share | **47%** | 19% | **38%** | 20% | **31%** | 16% | **58%** | 21% |
| List-pane control rows before item 1 | 9 | 8 | 9 | 8 | 9 | 8 | 7 | (list is a grip) |
| First list-item row | 23 | 14 | 23 | 14 | 23 | 14 | 21 | - |
| Rows available to the list | 10 | 21 | 19 | 30 | 29 | 40 | 1 | 0 |
| Items visible | 5 of 28 (2 rows each) | 6 of 6 (3 rows each) | 9 of 28 | 6 of 6 | 14 of 28 | 6 of 6 | 1 | 0 |
| Column split | 28 \| **58** \| 30 | grip 5, list 57, grip 5, reader **50** | 39 \| **78** \| 39 | nav 34, 5, list ~51, 5, reader **~62** | 54 \| **108** \| 54 | nav 39, 5, list 50, 5, reader **~117** | 16 \| 40 \| 22 | grips 5+5, reader ~70 |
| Work-pane share of width | 48% | 42% | 49% | 39% | 49% | 53% | 50% | ~88% |
| Inspector blank share | 58% | - | 71% | - | **85%** | - | - | - |
| Collapsed side panes (work width after collapse) | 13+11 (→92) | 5+5 | 13+11 (→132) | 5+5 | 13+11 (→192) | 5+5 | 3+3 | 5+5 |
| Border-glyph share | 23.6% | 15.0% | 19.0% | 16.4% | 15.4% | 12.9% | **34.5%** | 12.1% |
| Blank share | 44.9% | 64.2% | 46.5% | 66.7% | 55.8% | 76.9% | 36.3% | 62.8% |
| Description line length, rails open / collapsed (chars) | ~50 / ~90 | reader ~50 | ~70 / ~124 | reader ~60 | ~100 / **~185** | reader ~108 | ~32 | ~62 |
| Editor fields visible, clean session | 2.5 | prompt 4 / note 3 (body 4 rows) | 5 (1,1,2,1,1 text lines) | prompt 4 / note 3 (body 16 rows) | - | - | - | - |
| Editor fields after a Dictionaries + Lore visit ([RP-001](#rp-001)) | **0.5 (2-row window)** | - | **1 (5-row window)** | - | - | - | - | - |
| Buttons on screen with one item selected | - | - | **26** (13 item actions) | 6 item actions (prompt open) | - | - | - | - |

**Caveat ([RP-001](#rp-001)).** Several earlier "wasted space in Dictionaries/Lore" readings (`metrics.md` section C; capture-report #8, #13: the blank band above the tabs, "Lore shows 0-1 entries", the persona Edit below the fold) are mostly the leaked-slot bug, because the capture driver opened the editor before visiting Dictionaries/Lore. A clean first visit to Lore at 160x45 shows the tabs at row 14, 10 entry rows (17-27), Try-it (28-37) and 4 blank rows. Fix [RP-001](#rp-001) before re-measuring.

**Shortest reproduction of [RP-001](#rp-001)** (160x45, fresh golden profile): ⌃4 Roleplay, click the `Lore` chip, click `Aetheria Atlas`, click the `Characters` chip, click `Captain Isolde Varga`, click `Edit`. The editor body occupies ~9 rows with 14 blank rows below. At 120x36 a single Lore visit gives a 5-row window; after Dictionaries + Lore, a 2-row window.

New captures from this lens (in `captures/`): `review-rv-layout-search-focused-typed-160x45`, `review-rv-layout-2-portrait-cta-below-fold-120x36`, `review-rv-layout-portrait-inspector-160x45`, `review-rv-layout-2-editor-porthole-after-dict-lore-120x36`, `review-rv-layout-editor-collapsed-after-dict-lore-160x45`, `review-rv-layout-2-lore-after-dict-blank-band-120x36`, `review-rv-layout-dict-selected-160x45`, `review-rv-layout-dict-settings-clipped-160x45`, `review-rv-layout-dict-entries-scrolled-160x45`, `review-rv-layout-3-transcript-view-160x45`, `review-rv-layout-rails-collapsed-220x55`.

### C.2 Live task flows (flows lens)

Isolated harness (`launch.sh`); sockets `rv-flows` (empty profile at 160x45, then golden at 220x55), `rv-flows-2` (golden, 160x45) and `rv-flows-3` (golden, 120x36, resized to 220x55). One scratch fixture was added: `scratchpad/rv-flows-fixtures/ilsa_with_book.v2.json`, a V2 card with a 2-entry embedded lorebook. Mock replies took 1-4 s. Legend: **A** = action (click, key or typed string); **W** = wheel notches; **DE** = dead end; **?** = moment of uncertainty; **✗** = data or flow error.

**J1. First visit, empty profile, 160x45**

| # | Action | Result | Notes |
|---|---|---|---|
| 1 | click ⌃4 Roleplay | Characters; Default Assistant auto-selected (worked-example card) | ? "Author the pieces that shape a chat"; Personas, Dictionaries and Lore explained only in tooltips |
| 2-4 | p, d, l | Blank centre in each; rail copy cut off; Inspector "Pick a character or persona..." with 4 Buddy buttons in every mode | RP-055, RP-018 |
| 5 | c, click **New** (row 3 of a 7-row stack) | Editor opens, Name focused | Four creation verbs (RP-008, RP-043) |
| 6 | type a name | ok | Footer still advertises c/p/d/l and s while typing (RP-034) |
| 7-8 | Tab, Tab | 1st Tab lands on an invisible "Generate"; 2nd reaches First message | ? RP-027 |
| 9 | type greeting | 1-row box | RP-002 |
| 10 | click Save | Toast "Character saved."; still in editor; row date shows UTC "2026-10-02" | ? RP-038, RP-070 |
| 11 | Esc | Card | **7 A, 0 DE, 2 ?** |

**J2. Import cards, inspect, Chat now (empty profile, 160x45)**

| # | Action | Result | Notes |
|---|---|---|---|
| 1-6 | Import, Ctrl+L, type path, Enter, click PNG, Import | Wren imported, selected, shown | 6 A, ~3 s |
| 7-9 | Import, click JSON, Enter | Corwin card; alternate greetings shown; **example dialogue nowhere** | Picker remembered the folder. RP-023 |
| 10-14 | Import the card with an embedded lorebook | Toast "Lorebook 'Glass Coast Lore' attached (2 entries)" | |
| 15-17 | W x5, click "▸ World Books (1)" (below Edit) | Read-only row "Glass Coast Lore 2" | ? hard to find |
| 18 | l | "No lore books yet" | **DE** RP-013 |
| 19 | c | Selection lost: "Pick a character..." | RP-030 |
| 20-21 | click Ilsa, Chat now | Console "Chat with Ilsa..." with greeting, ~2 s | Primary greeting only (RP-048). **~21 A + 5 W, 1 DE** |

**J3. Find Vex, edit greeting and scenario, save, verify (golden)**

| # | Action | Result | Notes |
|---|---|---|---|
| 1-4 | Ctrl+F, "vex", Enter, Enter | Vex selected (4 keys, ~3 s) | Typed text invisible (RP-009) |
| 5 | click Edit | Editor | |
| 6-7 | click First message, type | Inserted at the clicked column, mid-word; 1-row box | **✗** RP-002 |
| 8-10 | W x6, "Advanced ▸", W x4 | Scenario box shows 0 text rows (existing text invisible) | ? |
| 11-12 | click Scenario, type blind | Inserted mid-word | **✗** |
| 13-14 | Save, Esc | Card: "The pa Rain hammers the portcullis.rty ..." and "The gates o The torches gutter.f ..." | **~14 A + 10 W, 2 silent corruptions** |

At 220x55, First message is still 1 row and Scenario, Post-history and Creator notes are still 0 rows. **Library analog (prompt edit, 160x45):** Prompts (4), open the item, click Instructions, type, "Save changes" → "Modified now · v2": **5 A, 0 W, all text visible**, and the footer switches to "typing in field | esc save or discard first".

**J4. Test chat and hand-off**

| Size | Steps | Result |
|---|---|---|
| 160x45 | open "▸ Try a test chat" | Greeting shows the **pre-edit** first message; Reset doesn't fix it (RP-046). The "Greeting:" picker is a blank 3-row box and does not open (RP-010) |
| 160x45 | click input, type, Enter | Reply in ~2 s; **button row disappears**; wheel does nothing |
| 160x45 | click input, Tab x3, Enter (blind) | "Preview conversation staged in Console"; the Console reply came from the **generic tool assistant**, not Vex: the gap round showed the staged transcript is dropped at send (RP-047); the staged source later leaked into a persona chat, but did not block it |
| 120x36 | open | Input clipped to its top border; buttons never visible |
| 120x36 | send 1 | After the reply **the input vanishes**; 2nd message typed blind; reply needed 3 W (no auto-scroll) |
| Verdict | | 120x36: effectively unusable. 160x45: one exchange, then keyboard workarounds |

**J5. New dictionary with 2 entries, Try it, attach (golden, 160x45)**

| # | Action | Result | Notes |
|---|---|---|---|
| 1 | Dictionaries chip | List of 3; purpose "· 0" | RP-028 |
| 2 | click New (hit Import in the tight row; Esc) | | ? |
| 3 | Ctrl+N | "Untitled dictionary" **persisted at once**; Settings tab; fields unlabelled; Save clipped | RP-005, RP-040 |
| 4-10 | type name, Tab x5, Enter (blind) | Saved as "Pirate Speak" | **✗ blind save** |
| 11-17 | Entries tab, Pattern, type, unlabelled replacement, type, W x3, Add | "1 entries" | Other controls off-screen (RP-003); "1 entries" (RP-070) |
| 18-22 | clear and retype both fields, Add | 2 entries | Form not cleared after Add (RP-040) |
| 23-25 | Try it: click, type, Run | Correct diff and fired rules | ✓ |
| 26 | Attachments tab | Conversation-only | **DE** RP-012 |
| 27-36 | c, Ctrl+F wren, Enter, Enter, W x8, "▸ Dictionaries (0)", "Attach dictionary...", pick, Attach | Attached | Selection memory lost (RP-030) |
| 37 | back to Dictionaries > Pirate Speak > Attachments | "Not attached to any conversation yet." | **~37 A + 11 W, 2 DE, 1 blind save** |

**J6. Kepler-9: add entry, Try it, confirm on Isolde (golden, 160x45; checked at 120x36)**

| # | Action | Result | Notes |
|---|---|---|---|
| 1-2 | Lore chip, click Kepler-9 | 7 blank rows above tabs; table shows 3 of 6; content truncated | RP-006 (part RP-001) |
| 3-18 | W through a 5-row window to Keys (pre-filled), replace, W to Content, replace, W, Add | 7 entries | ~14 W; ? Add vs Update |
| 19-21 | Try it: type, Run | Fires the new entry | ✓ |
| 22 | Attachments tab | Conversation-only | **DE** |
| 23-26 | c, click Isolde, W x14, "▾ World Books (1)" | "Station Kepler-9 Ops Manual **6**" | **✗ stale copy** (RP-012). **~26 A + ~30 W** |

At 120x36 the entry window is 4 rows and content is cut at ~25 characters.

**J7. Personas**

| # | Action | Result | Notes |
|---|---|---|---|
| 1 | Personas chip | "Personas - who you play in the chat · 4"; blank centre | RP-011 |
| 2 | click Cozy Storyteller | Card: Description plus **System prompt** (assistant-side); Edit at the bottom | |
| 3 | Chat now | Console "Chat with Cozy Storyteller"; "Persona: Cozy Storyteller" in the assistant slot | ✓ hand-off |
| 4 | send | Blocked: the Console refuses the first send ([RP-072](#rp-072), found in the gap round) | |
| - | look for a "you" / user-profile setting | None in Roleplay (display name lives in Settings by design, ADR-046) | D3 |

**J8. Modes and leave/return**

| Probe | Result |
|---|---|
| Mode switch | Selection, search, sort and tag cleared every time (RP-030); switching takes ~0.1 s |
| Leave/return with a selection | Mode and selection restored ✓ |
| Leave/return with search active | List still filtered (1 of 28, "Sort: Relevance"); box shows the placeholder (RP-021) |
| Leave/return in Personas/Lore with no selection | Empty list, "· 0", wrong panes, dead active chip (RP-021) |
| Leave with unsaved editor | Save/Discard/Stay guard ✓; after Stay the nav shows Library and Library clicks do nothing (RP-061) |
| Mis-click, then type | Mode keys fire (RP-033) |
| Library comparison | Library also drops the open prompt on Prompts > Notes > Prompts (do not copy) |

**Cross-surface note (Console, not Roleplay); resolved in the gap round.** On `rv-flows` (golden copy, 220x55) the first send in both a persona and a character Chat-now session ended `controller_submit status=blocked`, after `ERROR ... Console RAG capture unavailable; reason=capture_provider_failure`; the chat list showed "⛔ Blocked" with no visible reason. The preview hand-off chat on `rv-flows-2` did reply. The gap round (C.7) found the real cause: a trace-provenance error on the character's or persona's system prompt that refuses every Chat-now first send on the default config ([RP-072](#rp-072), TASK-33621.2). The `capture_provider_failure` ERROR is a separate, harmless bug that also appears on successful sends ([RP-097](#rp-097)), and the leaked staged source does not block ([RP-047](#rp-047)). One 220x55 editor-height oddity after a resize did not reproduce on a fresh launch and was dropped.

**Journey effort summary (160x45 unless noted)**

| Journey | Actions | Wheel | Dead ends | Errors / blind steps |
|---|---|---|---|---|
| J1 create first character | 7 | 0 | 0 | 0 (2 ?) |
| J2 import x3 + inspect + Chat now | ~21 | 5 | 1 | 0 |
| J3 find + edit greeting + scenario | ~14 | 10 | 0 | **2 silent corruptions** |
| Library prompt edit (analog) | 5 | 0 | 0 | 0 |
| J4 test chat + hand-off | ~8 | 1 | 0 | blind Tab hand-off; wrong assistant |
| J5 dictionary end-to-end | ~37 | 11 | 2 | 1 blind save |
| J6 lore entry + verify | ~26 | ~30 | 1 | stale copy |
| J7 persona chat | 4 | 0 | 1 | Console block (external, RP-072) |

### C.3 Keyboard-only journeys (accessibility lens)

Focus was read from the rendered output; a Tab or F6 press that changed nothing on screen is counted as an invisible stop. Per-keystroke snapshots are in `rp-review/a11y/steps/<socket>-<label>/<n>.ansi`. Helpers (scratch only): `rp-review/a11y/ansidiff.py` (style-aware cell diff), `rowstyles.py`, `stepkeys.sh`. Contrast ratios use the WCAG relative-luminance formula on the captured hex colours.

| Journey (160x45, golden) | Result | Cost |
|---|---|---|
| Select a character | Works | Arrival focus is on the nav bar's Home (RP-024), so F6 or Ctrl+F, Esc, Down, Enter |
| Open the editor | Works | No hotkey: F6 lands on the test-chat toggle, then 5 Tabs, 2 of them invisible (RP-025, RP-027) |
| Change a field and save | Works | Ctrl+S (ADR-031 rule 2 conflict, D9) |
| Leave the editor | Works | Escape raises the guard: Keep Open / Close Without Saving, no Save (RP-037) |
| Run a test chat | Works | F6, Enter, 3 Tabs (2 invisible), type, Enter |
| Switch mode | Only from non-text focus | Escape does not leave the test input or Try-it fields (RP-026) |
| Collapse / expand rails | Works | Focus dropped to nothing each time (RP-064) |
| Reach the dictionary editor or Try it with F6 | **Impossible** | Tab works, but 6 consecutive stops are off-screen (RP-025, RP-027) |

Shared defects to fix in the frame rather than copy: the Library also puts arrival focus on the nav bar and has weak button focus. Library fixes to copy: Tab region, Escape leaves text fields, focus-aware footer, contextual F1, focus placement on collapse/expand, narrow-stage return. Side effects stayed inside the disposable run copy (`runs/rv-a11y`): Isolde was saved as "Captain Isolde Varga II", and one accidental Send-to-Console happened while focus was not visible (itself evidence for RP-024 and RP-065).

### C.4 Library-fit analysis (fit lens)

**Resolver numbers** (pure `resolve_adaptive_reader_layout`, W ≈ cols - 4; each cell is Nav / Items / Work in columns):

| cols | Library default (all open) | RP card profile (work_min 56) | RP edit, work-first (Nav closed, work_min 72) | RP edit, Nav and Items closed | Roleplay today (work) |
|---|---|---|---|---|---|
| 100 | 0/42/44 | 0/0/86 | 0/0/86 | 0/0/86 | 42 |
| 120 | 0/56/**50** | 0/50/56 | 0/0/**106** | 0/0/106 | **58** |
| 140 | 31/50/**45** | 30/40/56 | 0/54/72 | 0/0/126 | - |
| 160 | 34/50/**62** | 34/50/62 | 0/56/**90** | 0/0/146 | **78** (+39-col Inspector) |
| 200 | 39/50/97 | 39/50/97 | 0/56/130 | 0/0/186 | - |
| 220 | 39/50/117 | 39/50/117 | 0/56/150 | 0/0/206 | 106 |

A fourth 30-column Inspector or Try pane beside Nav + Items 50 + three 5-cell grips would leave the work pane -8 cols at 120, 27 at 160, 62 at 200 and 82 at 220. **Conclusion:** the Library frame helps authoring only with Roleplay profiles plus the work-first override. Copied with Library defaults, it is worse than today at 120-160 columns. (At 120, the authoring profile also closes the items list.)

**Inspector options (input to D1).**
1. *Fold into the work pane.* For: the Library grammar; frees 30-39 cols at 120-160; primary action next to content; ends Buddy/guidance noise. Against: readiness visible only in the work header; chats become a mode, so card and chats are not seen together; breaks pins in `test_personas_inspector_pane.py` (60 tests) and `test_destination_visual_parity_correction.py:997-1007` (which already exempts Watchlists, a precedent).
2. *Stack below the canvas under 120 cols* (ADR-148, Proposed). For: always-visible actions; ADR-148:57-60 rejected toggle-hiding. Against: rows are Roleplay's scarce dimension; ADR-148 is MCP-scoped.
3. *Keep as a fourth pane.* For: least churn. Against: two-pane resolver; collapsed by default below ~200 cols; mostly blank at 220 today.
Constraints for all: ADR-120 puts "complete browse" in Roleplay; ADR-004 forbids a Conversations view for Personas; keep `#personas-start-chat`, `#personas-attach-to-console`, `#personas-conversations-list`.

**Test chat / Try-it options (input to D2).** (1) Right "Try" pane: north-star and side-by-side edit/test, but competes with Items and needs a fallback at ≤140 cols. (2) "Try" work-pane mode: full width and height, no resolver change, matches Library Notes Edit/Preview, but loses side-by-side editing (ADR-115 cached bodies keep drafts). (3) Bounded bottom band: visible while editing, but costs rows, the shape of today's failure. Constraints: ADR-115:24-28 keeps the preview eager; compose it once in its final container (no reparenting); `save_state` already persists preview turns.

**Prerequisites for reuse.**
- An ADR-086 amendment or new ADR before using the shell outside Library, citing `design-language.md:135-136` and recording the grip exception to `:132-134`.
- Header via inline DestinationHeader (ADR-015 holds), or amend ADR-015 and DESIGN.md:107 explicitly.
- CSS: shell rules are `library-`-prefixed and lazily loaded (`build_css.py:447-455`); the boot broad-selector budget is full (274/274); Roleplay's scoped CSS budget is 2,996 bytes (`boot_css_bytes.json:145`).
- ADR-115: keep the four demand-mounted slots as work-pane hosts composed once; pane collapse is display-only (`LibraryAdaptiveReaderShell.sync_layout`, `library_adaptive_reader_shell.py:342-455`).
- ADR-046: every rail click, items click, mode key and work-first restore that changes selection must go through `_run_guarded` (`personas_screen.py:16179`).
- ADR-004 / test pins: 325 `#personas-*`/`#ccp-*` ids pinned across 51 test files - move widgets, do not rename them. Tests to re-pin: `test_personas_workbench.py:510, 532, 563, 586-652`, `test_personas_library_toolbar_layout.py`, `test_personas_center_canvas_layout.py`, `test_destination_visual_parity_correction.py:997-1007`, `test_destination_shells.py:1306`, `test_workbench_pane_focus.py:106-126`.
- Collisions: PR #2862 (conflicting; touches the same screen and the preview coordinator) and heavy churn on `personas_screen.py` argue for screenshot-approved slices (ADR-007): header + rail, then resolver + grips + persistence, then work-pane modes and the Inspector fold, then the Dictionaries/Lore restructure.

### C.5 Proposed rail and content architecture (IA lens; a proposal, not a decision)

Edited to follow your existing persona ruling (TASK-617, ADR-037/149): personas are assistant-side, so the "who you play" alternative is not presented.

```
Roleplay | Local · Console: ready                      <- one header line
┌ Roleplay ─────────────────────────────┐            <- NOT "Library" (RP-042)
│ [ Import… ]                           │  one picker: card .png/.json, .tldw-actor-pack,
│ [ Search Roleplay…               ][x] │  lorebook JSON, dictionary JSON/MD
│ ───────────────────────────────────── │
│ Cast                              ▾   │
│ ▸ Characters (28) — who the AI plays   c
│   Personas (4)    — reusable AI roles  p
│   You             — your name for {{user}}   (optional; D3 option C)
│ World info                        ▾   │
│   Lore books (3 · 2 on) — facts on keywords      l
│   Chat dictionaries (3 · 2 on) — find/replace    d
│ Continue                          ▾   │   (optional; recent character chats, ADR-120)
│   Shore leave on Kepler-9 · Isolde    │
│ Create                            ▾   │
│   New character / from concept… / from portrait…
│   New persona · New lore book · New chat dictionary
│ Import / Export                   ▾   │
│   Import… · Export selected…          │   (JSON · PNG card · Markdown · Actor Pack)
│ Buddy                             ▸   │   (app-wide companion, out of the item Inspector)
└───────────────────────────────────────┘
```

Fitting rules: the gloss drops first and the title falls back to its short form; the count is never clipped and shows "(…)" while loading; the rail stays open down to 120 columns (drop the Inspector first); no catch-all "Details" bucket; mode keys `c p l d` shown as dim hints, with `[` / `]` following rail order (ADR-152).

**Items column:** header `Characters (28)` or `3 of 28`, with filter, sort and tag. Row line 2 (scent): character = description snippet · `2 chats`; persona = snippet · on/off; lore book = `10 entries · on · in 1 character, 0 chats`; chat dictionary = `8 entries (1 regex) · on · in Dungeon Master Vex`.

**Work pane:** a pinned title row (`‹ Characters · Captain Isolde Varga · edited 2m · Local`), a mode strip, and one action bar (primary, Edit/Save, `More actions ▸`).

| Kind | Modes | Primary | More actions |
|---|---|---|---|
| Character | Card · Edit · Chats (2) · World (2) · Test chat · Info | Chat now | Add to Console message, Export…, Duplicate, Delete |
| Persona | Profile · Edit · Visual states · Tool policy · Test chat · Info | Chat now | Add to Console message, Export…, Duplicate, Delete |
| Lore book | Entries · Settings · Used by · Try it · Info | Run preview | Export…, Duplicate, Delete |
| Chat dictionary | Entries · Settings · Used by · Try it · History · Info | Run preview | Export…, Duplicate, Delete |

The Persona row is superseded by the gap round's proposal (C.8: Profile · Edit · Look · Tool policy · Test chat · Info), which matches what the persona editor actually contains. "World" is Roleplay's own "what's in play" (character attachments plus conversation links); "Used by" lists characters holding a copy and linked chats (shape depends on D6); "Chats" is ADR-120's complete browse. Info carries validation (only when there are findings), source and timestamps.

**Canonical vocabulary**

| Object | Rail / title | Short | Gloss | Retire |
|---|---|---|---|---|
| character | Characters | Characters | who the AI plays | "Character Tab", "Character/Persona Management", "behavior profiles" |
| persona | Personas | Personas | reusable AI roles | "who you play in the chat", "Persona Profile" heading, "user profile context" |
| user identity | You (optional) | You | your name for {{user}} | the hard-coded "User" in previews |
| lore book | Lore books | Lore | facts added on keywords (aka world info / lorebook) | "World Books", "world book", "Type: lore", "Untitled world book" |
| chat dictionary | Chat dictionaries | Dictionaries | find/replace rules (regex) | the Console/Roleplay split in naming |
| actor pack | Actor Pack (an export format) | - | portable character or persona plus art | "New/Import Actor Pack" as peer nouns |
| attach to a character | "Copy into character…" (or "Link…" if D6 picks live links) | - | state copy vs link plainly | "Attach" with hidden copy semantics |
| attach to a chat | "Link to chat…" | - | live | - |
| hand-offs | Chat now / Add to Console message / Continue in Console / Resume conversation | - | - | "Send to Console draft" (three meanings), "Resume chat" |
| left pane | Roleplay | - | - | "Library" |

**Reasoning.** The four jobs are equal, so each gets an always-visible row with a count instead of tabs where only the active one has a gloss. "Cast" vs "World info" mirrors the two questions every roleplay tool asks (who is in the scene; what does the scene know or rewrite), and grouping lore books with dictionaries puts the two attachable objects side by side so "Used by" can share one model. Collection verbs (Create, Import/Export) live in the rail; item verbs live in the work pane's action bar. "Continue" serves J1 and ADR-120 but must stay character-scoped (ADR-004 makes the Library the global archive). Where the Library frame does not fit: no inspector (Roleplay's actions, readiness and chats move into the action bar and Chats/Info modes, ADR-148 precedent); the rail must not collapse at 120 cols; Library's character-blind conversation rows and its "Details" bucket should not be copied. Constraints: c/p/l/d and `[` / `]` (ADR-152) with truthful hints (ADR-031); the ADR-086 amendment; the reused `#ccp-*` ids (ADR-004); the ADR-115 demand-mount boundary; dictionaries and lore stay in Roleplay, not Settings (ADR-007).

### C.6 Method and hygiene notes

- **Two NN/g heuristic passes** (H1-H5 and H6-H10), each driven live on isolated profile copies (sockets `rv-nng-a*`, `rv-nng-b*`, 120x36 / 160x45 / 220x55), plus a blocked-provider config (`chat_defaults` Anthropic with no key; `character_defaults` OpenAI at dead port 18799).
- **Layout lens:** counted rows/columns from text captures plus `metrics.py` cell classification.
- **Flows lens:** eight end-to-end journeys (C.2).
- **Accessibility lens:** per-keystroke ANSI diffs (C.3).
- **IA lens:** taxonomy and vocabulary (C.5).
- **Fit lens:** pure-resolver runs and code reading of reuse paths (C.4).
- **Verification:** every finding was re-checked by an independent verifier (code re-read, capture re-read and, where marked, live reproduction on fresh `vf-*` sockets). 1 claim refuted (Appendix B); partial findings carry their corrections in Appendix A and `findings.json`.
- **Gap round:** three more lenses aimed at the coverage gaps in section 11: G1, the end-to-end hand-off, with a body-logging mock on :18777 and out-of-tree instrumentation (C.7); G2, the Personas editor at 120x36, 160x45 and 220x55 (C.8); G3, realistic volume on a scaled copy of the profile (C.9). Each of the 35 findings was re-verified independently (code re-read and live reproduction on fresh `vf-gap*` sockets): 24 confirmed, 11 partly confirmed and corrected, none refuted. 26 became new findings (RP-072 to RP-097) and 9 were merged into existing ones.
- **Isolation:** all runs used copies of the seeded golden/empty profiles (the gap round also used a scaled copy built with the app's own APIs); the real-profile snapshot diff stayed clean (`isolation-before.txt`); app logs had 0 references to `~/.config` or `~/.local`; all tmux servers were killed; nothing in the worktree was written except this report and `findings.json`. Mutations stayed in disposable `runs/` copies (one lore entry deleted, one dictionary toggled, one dictionary entry and one lore book imported, Isolde renamed in one run; in the gap round, policy rules added, personas created, disabled and renamed, and lore entries deleted and moved, all in copies).

### C.7 Gap round: end-to-end hand-off (G1)

Lightly edited; gap ids are mapped to RP ids. `R` = the `rp-review` scratch folder (see the header).

**Setup**
- **Body-logging mock:** `R/gapcheck/mock_llm_body.py` on :18777 (stopped after the run); request bodies in `R/gapcheck/bodies.jsonl` (9 lines).
- **Configs:** default-config runs used `R/gapcheck/mock_body.toml`; the capture-off control used `mock_body_captureoff.toml`, which adds `[console] exchange_capture = false`.
- **Instrumentation:** an out-of-tree `sitecustomize.py` (`R/gapcheck/inject`), loaded through the documented `APP_WT` override, wraps controller methods and logs to `R/gapcheck/trace.log`. The worktree was never edited.
- **Logs and cleanup:** app-log copies in `R/gapcheck/rv-gap1-*-app.log`. All tmux servers killed; the shared mock on :18765 still answers; the real-profile snapshot was untouched and the run logs contain 0 references to real profile paths.

**1. RP-004 confirmed live, with prompt bodies.**
- *Isolde (Kepler-9 attached), body 20:39:27.* The system prompt holds only the card text (its "orbital shipyards of Kepler-9" and "Hadley Gap" come from the description). The Kepler-9 entry ("Kepler-9 is an orbital shipyard station…") and the Hadley Gap entry ("survey ship *Tamsin*…") are absent. The user turn is sent verbatim.
- *Vex (Aetheria Atlas and the Fantasy Anachronism Swaps dictionary attached), body 20:40:27.* The lore text "volcanic fortress-city" is absent; the user turn goes out verbatim as `Call the police at the gates of Emberfall, OK?`, with no "city watch" and no "Aye".
- *Transform layer:* every character send logged `WORLD_INFO … changed=False` and `CHAT_DICT … changed=False`.
- *Control, body 20:42:34:* after Lore > Kepler-9 > Attachments > Attach to conversation… > the Isolde chat, the user turn became "Kepler-9 is an orbital shipyard station with 40,000 residents… / A navigation hazard where the survey ship *Tamsin* vanished six years ago. / Any news from Kepler-9 about the Hadley Gap?". The conversation-level path works; the character-level path is inert.
- *Test chat:* it also ignores character attachments (body 20:43:57: no rewrites, no lore).
- *Code:* `Chat/console_chat_controller.py:1273-1297` and `Chat/console_runtime.py:567-574, 630` pass `None` for the character; `Character_Chat/Chat_Dictionary_Lib.py:1433-1434` says char_data is "always `None` for the native Console today".
- *Caveat:* observable only with capture off, because on the default config every one of these sends is refused first (RP-072).

**2. J1's chat step does not work on a default profile: a product defect, not a harness artifact (RP-072).**
- *What fails:* 0 of 4 character and persona Chat-now sends reached the provider (Isolde, Vex and Cozy at 160x45; Isolde at 120x36). A plain Console send in the same run reached the mock (body 20:28:15).
- *Cause:* `TraceProvenanceAlignmentError('trace provenance category mismatch: system')` (trace.log). (1) The session's leading system prompt has no saved owner, so `_build_durable_trace_request` gives it an ACTIVE_REQUEST descriptor (`console_chat_controller.py:11955-11968`). (2) The `system` category accepts only saved or RENDERED_SYSTEM descriptors (`console_prepared_request.py:929-940`, `console_trace_provenance.py:841-857, 989-994`), so building the request raises. (3) A bare `except` (`console_chat_controller.py:12353`) turns that into a silent TRACE_PROVENANCE pause.
- *Who hits it:* the shipped template sets `exchange_capture = true` (`config.py:3831`; default True at `:8954`). It does not depend on embeddings or RAG config, so every fresh install hits it.
- *Workaround:* with `[console] exchange_capture = false` all three Chat-now sends completed (first reply ~1.2 s for characters, ~12 s for the persona).
- *Earlier logs:* the `capture_provider_failure` ERROR the first round saw is a separate, harmless arity bug that also appears on sends that complete (RP-097). The rv-flows, rv-ia and rv-flows-2 blocks are fully explained by RP-072; the leaked source in rv-flows-2 was not the cause.
- *Already filed:* TASK-33621.2 (To Do, P0). Its fix, PR #2882 / TASK-32951, is not on dev. TASK-33621.2 recorded "no error line"; the exact exception is new evidence for it.

**3. RP-047 raised to P1 / severity 3, but not because it blocks.** With capture off, a persona send completed with the leaked "Personas preview conversation" source staged; the source was excluded (`not_canonical_local_evidence`, log line 428), stayed staged after the send ("Sources: 1"), and Un-stage cleared it in one click. The new evidence is that both "Send to Console draft" hand-offs (test chat and Inspector, ctrl+enter) are inert: the transcript and the card never reach the model (bodies 20:45:34 and 20:46:57) while the UI says one source is staged. Silently doing nothing while claiming staged context is a major H1/H2 failure, hence P1; Chat now is the primary path and the leak is recoverable, hence not P0.

**4. Findings.** RP-072 (P0) and RP-074 (P1: the lens proposed P0, the verifier rated it 3 because it is the same incident as RP-072) are Console mechanisms that block J1's payoff. The readiness wording merged into RP-067 (P2) and the inert hand-offs into RP-047 (P1). New P2s: RP-082 (what a chat contains), RP-083 (the attach-to-conversation picker), RP-084 (the Console rail's wrong character). New P3: RP-097. The ~10-12 s wait before persona replies ("personal context bootstrap exceeded its 10.0s budget", which talks to the OS credential store) is probably an artifact of the isolated HOME and was not counted.

**5. Implications for D6, the Chat now action and readiness wording.**
- *D6:* rule now. Option C (Chat now links the character's attachments onto the new conversation) reuses the one mechanism shown to work end to end and is the lowest-risk interim fix. Option B needs the session's character id passed into `capture_prompt_transform_inputs` and the two runtime appliers; the session already knows its character ("Character: Captain Isolde Varga" in the status strip). The "Used by" view should list only conversations where the book is actually applied, and say "Not applied in chats" for character copies until B or C ships.
- *Chat now:* do not ship the redesign's Chat now as the hero action until TASK-33621.2 is fixed (D13); ship RP-074's recovery fix earlier; gate the redesign on a Roleplay-to-provider end-to-end test for a character with a greeting and for a persona.
- *Readiness wording:* drop "Ready to chat in Console." and the header "Ready". Under Chat now show "Opens a new Console chat as <name> · <provider/model>", plus an "In this chat" summary (RP-082). If a Chat-now session was refused, the Inspector may say so, with the reason and Open chat (the verifier kept this optional and dropped a proposed dry-run of the send path). Hide "Send to Console draft" and its ctrl+enter binding until RP-047 is fixed.

**Readiness shown vs outcome**

| Entity, size, config | Roleplay header | Inspector | Console after send | Outcome |
|---|---|---|---|---|
| Isolde 160x45, default | Ready | Ready to chat in Console. | header Ready, Inspect "Run: Ready", gutter "blocked" | refused |
| Vex 160x45, default | Ready | Ready to chat in Console. | same | refused |
| Cozy 160x45, default | Ready | Ready to chat in Console. | same, plus a later send parked as "Unsent turn" | refused |
| Isolde 120x36, default | Ready | Ready to chat in Console. | same | refused |
| All three 160x45, capture off | Ready | Ready to chat in Console. | reply | completed |

### C.8 Gap round: Personas follow-up, J4 (G2)

Lightly edited; gap ids are mapped to RP ids.

| Socket | Size | Use |
|---|---|---|
| rv-gap2 | 160x45 | Crashed on RP-075; its log is kept |
| rv-gap2-4 | 160x45 | Clean relaunch |
| rv-gap2-2 | 120x36 | |
| rv-gap2-5 | 120x36 | Clean session plus one Lore visit, for the RP-001 run |
| rv-gap2-3 | 220x55 | |

All runs used copies of the golden profile; all tmux servers were killed; the real-profile snapshot diff was empty and the app logs hold 0 `~/.config` / `~/.local` references; nothing was written in the worktree. New captures are `captures/review-rv-gap2*-*` (about 30 states). Mutations (disposable copies only): policy rules on Cozy Storyteller and Terse Code Reviewer; Default Narrator re-saved; Zed Gap Persona, Wren Pack Persona (Actor Pack, Wren portrait) and Repro B (then deleted) created; Cozy disabled.

**Do RP-001, RP-002, RP-003 and RP-016 apply to the persona editor?**
- **RP-001 (leaked slots): yes.** Clean 120x36: the editor window is rows 16-30 (15 rows). After one Lore visit (`captures/review-rv-gap2-5-persona-editor-after-lore-leak-120x36`) it shrinks to rows 16-21 (6 rows: Name, the Description label and its top border), 0 Description text lines are visible, Save/Cancel sits at row 23 and rows 24-32 are blank. The persona card is not squeezed.
- **RP-002 (0-1-line fields): no, the inverse ([RP-076](#rp-076)).** Each TextArea fills the viewport: ~14 rows at 120x36, 23 at 160x45, ~33 at 220x55. Text is visible, caret placement is correct, no corruption was seen.
- **RP-003 (off-screen, unlabelled or raw controls): yes, and worse.** Mode shows raw `session_scoped` / `persistent_scoped`; three sections draw as a title only at all three sizes ([RP-073](#rp-073)); the policy "Kind" is a free-text input with the placeholder "mcp_tool or skill"; error copy shows widget ids ("personas-editor-name: required", RP-044).
- **RP-016 (one long unsectioned scroll): yes in structure,** with different heights: one VerticalScroll whose only grouping is the bordered visuals box. Scroll extent ≈97 rows at 120x36 (~6.5 screens; Mode after ~36 rows), ≈125 at 160x45 (~5; Mode after ~62 rows), ≈150 at 220x55 (~4.4). Visible at once at every size: Name plus part of one long field. Tab order from Name to Save is 22 stops, 8 of them invisible.

**Other examination items**
- **Unsaved work:** Escape, a mode chip or a different list row while dirty opens the same "The tab … has unsaved changes / Keep Open / Close Without Saving" dialog as the character editor (RP-037; `captures/review-rv-gap2-persona-escape-unsaved-dialog-160x45`). Leaving through the nav bar opens the good aggregate guard. Signals: header "Editing Cozy Storyteller - unsaved", Inspector "Validation: editing…" and "Save or discard your edits to chat in Console."; the rail row is unmarked; after Save the toast "Persona saved." appears and the Save/Cancel bar looks unchanged (RP-038 applies as is).
- **Keyboard:** Ctrl+S saves the persona (the ADR-031 conflict, RP-035/D9). F6 behaves as in RP-025. Persona-specific invisible stops are in RP-073 and RP-085.
- **Search and empty state:** "zzqx" in Personas gives "No personas yet - use…" with the count still "· 6" (RP-022). The empty Personas state (RP-055) teaches the wrong concept ("who you play") and does not say that only System prompt reaches the model (RP-077).
- **Creating a persona:** New or Ctrl+N needs only a Name. New Actor Pack needs a Name plus a portrait character: satisfiable with the seeded data (3 eligible), but the picker is invisible and pre-selects Samira (RP-073). No dead end or blocked save was seen; the one blocking rule is visual-draft-unsaved (`personas_screen.py:15624-15629`).
- **Chat now when a character and a persona both apply:** only the selected persona is bound; the linked character is ignored ("No current character"). The system prompt is the persona's System prompt, or the generic agent prompt if that is empty (RP-077, RP-090). Enabled off does not block Chat now (RP-086).

**Does the C.5 persona work-pane plan match the editor?** C.5 proposed Profile · Edit · Visual states · Tool policy · Test chat · Info, plus More actions with Duplicate. The editor actually holds: (1) Name, Description, System prompt, Personality traits (only System prompt reaches Console); (2) Mode (inert locally) and Enabled (Buddy-only); (3) the Actor-Pack portrait picker and portable UUID (create only); (4) the Shared Visual Identity *reactions* browser with its own Save/Cancel; (5) the Buddy *operational states* pack with Save Pack, Import, Create character…, From Buddy archive…, Petdex… and Export saved pack…; (6) tool-policy rules, persisted at once; (7) the hidden portrait link (`character_card_id`); and, in the Inspector, (8) the tool-policy summary, Export JSON, Export Actor Pack and four Buddy buttons. Mismatches: "Visual states" merges two subsystems with separate saves; no mode holds the character link, the Actor-Pack identity or the meaning of Enabled; Duplicate does not exist for personas and there is no plain Import; the persona test chat diverges from Console and ignores drafts; nothing local reads Mode.

Proposed persona modes:

| Mode | Contents |
|---|---|
| **Profile** (read) | What is sent to the model (System prompt, plus whatever D15 decides); "Portrait from: <character>"; "Available to Buddy: on/off"; the tool-policy summary with the deny-by-default sentence; primary Chat now |
| **Edit** | Name; System prompt as the fill field; Description and Personality traits as auto-height fields with "sent / not sent" glosses; one "Save persona" bar |
| **Look** | Tabs Reactions \| Buddy states, each with its own named, anchored bar ("Save reactions", "Publish Buddy pack") and no fixed 33/37-row block |
| **Tool policy** | Full-height rule list and form; Kind as a Select; Name picked from the catalog; auto-save with Undo; delete with confirmation |
| **Test chat** | Built by the same composer as Console's persona seed, draft-aware, with speaker labels from names |
| **Info** | Source (local/server), version, portable UUID, validation only when failing |

Drop Mode (locally) until it is wired. More actions: Add to Console message, Export JSON / Actor Pack / Buddy pack, Create character from Buddy…, Import Buddy archive / Petdex…, Delete. Duplicate is new capability: add it deliberately, with a backlog task, not as a by-product of the reframe.

**D3 and the scope of J4 "user profiles":** see D3 (now B + C, plus one ruling on a user description) and RP-011. The three places that hold "you" today: Settings › Console Behavior "Default chat display name"; Console › Chat settings › "Conversation identity" (per chat); Settings › My Profile (the Personal Context Profile, ADR-102, a separate domain from roleplay {{user}}).

**Suggested order:** RP-075 and RP-073 land before or with the RP-001..RP-005 batch (both small); RP-077, RP-078 and RP-079 are S-M content fixes; the "You" row waits for the D3 ruling.

### C.9 Gap round: realistic volume (G3)

Lightly edited; gap ids are mapped to RP ids.

**Scaled copy.** `runs/rv-gap3` was built from golden with `rp-review/gapcheck/scale_seed.py` and `lore_fix.py`, using the app's own APIs: 348 characters (320 added, names with shared prefixes such as Captain/Lady/Sir/Agent, about a third tagged, a few long descriptions; Ser Corwin falls on page 6); 64 personas; 43 lore books ("Grand Codex of the Shattered Realms" has 250 multi-key entries of 300-1500 characters, 28 disabled; entry #200 is "Saltmarsh Bellwarden"; attached to Vex); 28 dictionaries ("Grand Dialect Pack" has 150 rules, 10 regex, one invalid); 30 Vex chats. `runs/rv-gap3-2` is a copy with 114 personas, used for the persona cap. Captures are `captures/review-rv-gap3-*` and `review-rv-gap3-2-*`. Golden and empty were never touched; the isolation diff reads "untouched"; run logs hold 0 references to the real profile; tmux servers were killed.

**Severity changes the lens proposed, and what was adopted**

| Finding | Proposal | Adopted? |
|---|---|---|
| RP-056 | P2/2 -> P1/3 (entry #200 is 99 notches away, 115 after every save) | Yes |
| RP-014 | P1/3 -> P0/4 (the deleted entry can be #1, 198 rows from the one being edited) | Yes (merged with the row-1 reset) |
| RP-006 | Stay P0, severity 3 -> 4 (3-4 of 250 entries visible) | No: a clean 160x45 visit shows 10 rows; the 3-row window is RP-001 |
| RP-030 | P2/2 -> P1/3 (entry position lost on mode switch) | No: the entry is lost on every save anyway (RP-014); characters re-find in 0.34 s |
| RP-039 | Stay 2; scope widens to marks lost at page boundaries | Yes |
| RP-022, RP-049 | New evidence only | Yes (RP-049's 60-notch cost dropped: the Inspector has chat search) |

**The "4-keystroke find" claim** (section 6) holds for characters at 0.34 s across all 348. It does not hold for lore or dictionary entries (no entry search), or for personas beyond the newest 100, which cannot be found at all.

**Measured timings** (wall clock, `waitfor.py` polling every 40 ms; 160x45 unless noted)

| Action | Time | Notes |
|---|---|---|
| Palette → Roleplay, cold first visit (348 characters) | header 0.35 s, list 0.99-1.11 s (3 trials) | Every arrival also runs `fetch_all_characters()` over all 348 cards (`personas_screen.py:2068`), on top of the 50-row page; matches PERF-22's 0.7-1.0 s |
| Return from Library to Roleplay | 0.64 s to the list | Restores page 5 and the selection |
| Page change (next/prev) | 0.05-0.30 s to the label | No stall at or above 250 ms |
| Ctrl+F search across 348 (FTS) | 0.34 s including the 0.2 s debounce | Resets to page 1 |
| Mode switch → Personas (64) / Dictionaries (28) / Lore (43) / back to Characters | 0.10 / 0.05 / under 0.10 / 0.15-0.30 s | Dictionaries and Lore still show "· 0" until something is selected (RP-028) |
| Select the 250-entry book / the 150-rule dictionary | 0.05 s / 0.10 s | |
| Lore Move down, one step | 1.25 s | Rewrites 250 rows one at a time (RP-096) |
| Wheel to entry #200 / re-find after a save / dictionary rule #150 | 99 / 115 / 70 notches | 3-row window at 160x45 with the RP-001 band |
| Inspector chats: wheel to the oldest | more than 60 notches, then a click | Stops at session 11 behind "Load 20 older conversations" (the Inspector's chat search is the faster route) |
| event_loop_stall in tldw_cli_app.log | none during Roleplay | Only at boot: 358 ms and 356 ms in storage_admission |

**Implications for the redesign**
- **a. Paging or a virtual list (D14): a virtual list.** Today the page boundary breaks things: marks die at it (RP-039), the scroll offset leaks onto the next page and skips about 41 items (RP-080), turning a page needs Tab, and the pager costs 1-3 rows, 30% of the list at 120x36 (RP-092). With the frame fixed, section 3 projects 33 list rows at 160x45 and 24 at 120x36, which is 16 or 12 two-line items, so each 50-item page still needs 3-4 screens and paging only adds a second navigation axis. Build one line-based virtual list (OptionList / Line API rather than a ListView with a widget per row, which also addresses PERF-22's "remounts every row"), loading in chunks of about 200 if needed, with the count in the items header. Fallback if paging stays: a one-row, non-wrapping pager in the items header ("‹ 2/7 ›"), scroll to the top on every page change, PgUp/PgDn at the list edges, marks stored by id, and a cue when the selected item is on another page. Fix the persona cap (RP-093) first.
- **b. Row density.** Today 8.5 cards are in view at 160x45 and 3.5 at 120x36. Default to one-line rows once a list passes about 40 items or the column is narrower than 40 characters, showing line 2 only for the highlighted row or behind a density toggle (personas are already one line). Truncate in the middle, with a tooltip (RP-094). Line 2 should help tell items apart: in the seed, identical description openings made the 32-character snippet useless, so prefer tags, chat count or last used for characters, and "N entries · N off · on" for books. Keep lore and dictionary books in the items list: at 43 books the list is fine with search and one-line rows.
- **c. Entry filter and placement (D12): option A, confirmed.** Reject B: 43 books as rail sub-rows would make the rail a 50-plus-row scroller and bury Cast and World info. The entries list needs a filter over keys and content, at least 15-20 rows at 160x45 with no 12-row cap, fixed-width columns (RP-091), selection anchored to the entry across every action (RP-014), a position count ("#200 of 250 · showing 3 of 250"), and "+ New entry" that clears the form and selects the new row after Add. Side by side with the editor when the work pane has 100+ columns; narrower, Enter opens the editor full-pane and Escape returns to the same row. Try it becomes its own mode with a scroll and the summary first (RP-081).
- **d. What per-mode memory must store (D5).** Per mode: selected id, list anchor (the top visible item's id; the page offset if paging stays), search, sort, tag, and marked ids kept across pages and searches. Per selected book or dictionary: active tab, selected entry, entries anchor, entry filter text, and the Try-it sample (already kept). Restore on mode re-entry (RP-030), after every entry action (RP-014), and across leave and return, which already works for page and selection. Persisting sort and tag across app restarts is optional; selection and anchor matter more.
