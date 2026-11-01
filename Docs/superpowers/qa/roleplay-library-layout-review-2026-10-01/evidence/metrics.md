# Roleplay vs Library — layout metrics (live captures, origin/dev @ 84247cb843)

Captures: `rp-review/captures/<screen>-<state>-<cols>x<rows>.txt` (plain `tmux capture-pane -p`).
Profile: seeded *golden* profile (28 characters incl. 3 built-ins, 4 personas, 3 dictionaries, 3 lore books,
6 conversations, 5 notes, 4 prompts, 2 media) — see HARNESS.md.

## A. Structural budget (hand-counted from the captures; row/col numbers are 1-based)

| Measure | Roleplay 160x45 (character selected) | Library 160x45 (Conversations, item open) | Roleplay 120x36 | Library 120x36 |
|---|---|---|---|---|
| Rows of screen chrome above the first pane-interior row | **13** = nav tabs 3 + title box 5 (border, "Roleplay", subtitle, " Ready", border) + purpose/count line 1 + "Modes:" chips 1 + outer frame 1 + blank spacer 1 + pane top border 1 | **5** = nav 3 + "Library \| Local" 1 + outer frame 1 (list/reader are borderless and start on row 6; the Nav box adds its own border row) | **13** (identical stack) | **5** |
| In-list chrome rows before the first list item | **9** (rail header "Library <", Search…, New, New Actor Pack, Import, Import Actor Pack, Duplicate, Sort: Name, Tag: All — one button per row) → first item row 23. Personas 6 → row 20; Dictionaries/Lore 6 (3-row search box + 1 button row + blank) → row 20 | **8** (list title, Active/Archived/All, Export…, Select, 3-row filter box, blank) → first item row 14 | **9** → first item row 23 | **8** → first item row 14 |
| Rows of chrome below the content | **4** (pane bottom border, blank spacer, outer frame, hint bar) | **2–3** (outer frame, hint bar; +1 where the Nav box closes) | **4** | **2–3** |
| Rows the list can use | 19 (rows 23–41) | 30 (rows 14–43) | 10 (rows 23–32) | 21 (rows 14–34) |
| List items visible | **9** characters (2 rows each; 28 total) | **6 of 6** conversations (3 rows each: title, meta, blank) — room for ~10 | **5** (fifth only its name row) | **6 of 6** |
| Column budget | frame 2 + **Library rail 39** + **centre 78** + **Inspector 39** + frame 2 → primary (centre) = 49% of width, rails = 49% | frame 2 + **Nav 34** + grip 5 + **list ~51** + grip 5 + **reader ~62** + frame 1 (3 panes + 2 grips) | frame 2 + rail 28 + centre 58 + Inspector 30 + frame 2 → centre 48% | Nav collapsed to a 5-col grip; grip + **list 57** + grip 5 + **reader 50** |
| Rails when collapsed | Library handle 14 cols + Inspector handle 12 cols (26 cols ≈ 16% of 160) | grips are 5–7 cols each, carrying a vertical name ("Nav", "Items", "Notes", "Prompts") + "<---"/"--->" arrows | 14 + 12 = 26 cols (22% of 120) | 5–7 cols |
| Width behaviour | rails scale with the window: 28 → 39 → 54 cols at 120 → 160 → 220 (220x55: rail 54 + centre 106 + Inspector 54; the Inspector shows ~25 rows of blank) | Nav rail fixed ~29–39 cols; at ≤120 cols the Library auto-collapses one pane to a grip so at most two panes are open; at 80x24 and 60x24 Nav+Items both become grips and the reader takes the width | — | — |

Editor fields visible (label + at least one input row on screen):

| Editor | 160x45 | 120x36 | Notes |
|---|---|---|---|
| Roleplay character editor | **5** (Name, First message, Description, Personality, System prompt); inputs show 1, 1, 2, 1, 1 content rows | **2½** (Name, First message, Description label + 2 rows) | Scenario / Post-history / Creator notes / Tags / Alternate greetings are behind "Advanced ▸"; once expanded, Scenario, Post-history instructions and Creator notes render with **0 content rows** (`height: 2` minus the 2-row field border, `personas_character_editor_widget.py:119-123`) — the Scenario text is invisible even when focused (capture `roleplay-character-editor-scenario-zero-height-160x45`) |
| Library note editor | 3 (Title, Keywords, Body — Body box 16 rows) | 3 (Body box 4 rows) | Nav auto-collapses to a grip when the editor opens; toolbar "Edit Preview Info Save Use in…" is clipped on the right at both sizes (Save hidden at 120x36) |
| Library prompt editor | 4 (Name, Description, Instructions 4 rows, Message template 4 rows) | 4 | — |

## B. Automated cell classification (all captures at 160x45 and 120x36)

Method (`rp-review/metrics.py`, reproducible): every cell of the W×H screen is put in exactly one class.
* **border** — frame/box glyphs `─│┌┐└┘╭╮╰╯├┤┬┴┼▊▎▔▁┃━┏┓┗┛▕▏` and scrollbar blocks `▂▃▄▅▆▇█`.
* **data** — glyphs of a text segment that is seeded user data. A segment is a run of text between border glyphs / 2+ spaces, further split on " · "; an optional "Label: " prefix is dropped; a multi-word segment counts when it is a (case-sensitive) substring of the corpus of every user-data string in the golden profile (read from a copy of its DBs); a single word counts only when it is an item name or a dictionary/lore key; trailing "…" is allowed (truncated names still count). Rows above the pane interior (nav/header) are never data.
* **label** — every other non-space glyph: nav tabs, headings, buttons, hints, field labels, counts, list metadata ("2 messages · 14m"), grip letters.
* **blank** — spaces.

Columns: `rows above pane interior` = rows from the top through the pane-top border (for borderless Library layouts, through the outer frame); `1st list-item row` = first row on which an item name of the active list appears, counted only inside the one pane holding most item names; `1st data row` = first row with any data cell; `bottom chrome rows` = rows from the pane bottom border to the screen bottom; `column budget` = widths of the bordered boxes opened on the pane-top row and the gaps around them (Library grips/list/reader are borderless, so they show as one large gap — see section A for their split); `items visible` = distinct item names visible in that pane; `editor fields visible` = field labels on screen (editor captures only); `data %` = data cells / all cells; `data share of ink` = data / (data + label + border).

Caveats: data % rewards long seeded text (Isolde's 1.2k-char description inflates the Roleplay character-detail rows), so compare like with like (list-only states, editor states). The "items visible" figure for `roleplay-rails-collapsed-*` counts the name in the detail header (the list is hidden), so read it as 0.

| capture | rows above pane interior | 1st list-item row | 1st data row | bottom chrome rows | column budget (pane-top row) | items visible | editor fields visible | data % | label % | border % | blank % | data share of ink % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| library-conversations-item-open-160x45 | 6 | 14 | 17 | 3 | boxes 34 / gaps 2+124 | 6 |  | 2.2 | 14.7 | 16.4 | 66.7 | 7 |
| library-empty-landing-160x45 | 6 |  |  | 3 | boxes 34+122 / gaps 2+0+2 | 0 |  | 0.0 | 5.4 | 14.3 | 80.3 | 0 |
| library-landing-160x45 | 6 | 11 | 11 | 3 | boxes 34+122 / gaps 2+0+2 | 3 |  | 0.9 | 9.3 | 17.7 | 72.1 | 3 |
| library-media-list-160x45 | 6 | 16 | 18 | 3 | boxes 34+66 / gaps 2+5+53 | 2 |  | 0.5 | 10.2 | 17.4 | 71.8 | 2 |
| library-notes-editor-160x45 | 7 | 28 | 8 | 3 | boxes 64 / gaps 7+89 | 5 | 3 | 3.2 | 10.2 | 19.6 | 67.0 | 10 |
| library-notes-list-160x45 | 7 | 30 | 30 | 3 | boxes 34+64 / gaps 2+5+55 | 5 |  | 1.3 | 13.4 | 17.2 | 68.1 | 4 |
| library-prompts-item-open-160x45 | 6 | 15 | 9 | 3 | boxes 34+50 / gaps 2+5+69 | 4 | 4 | 3.0 | 12.1 | 24.1 | 60.8 | 8 |
| library-prompts-list-160x45 | 6 | 15 | 15 | 3 | boxes 34+64 / gaps 2+5+55 | 4 |  | 1.1 | 10.3 | 17.8 | 70.9 | 4 |
| roleplay-character-editor-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 | 5 | 9.9 | 11.9 | 29.0 | 49.2 | 20 |
| roleplay-character-editor-advanced-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 | 6 | 6.3 | 14.6 | 27.9 | 51.2 | 13 |
| roleplay-character-editor-scenario-zero-height-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 | 6 | 6.3 | 14.6 | 27.9 | 51.2 | 13 |
| roleplay-character-editor-scrolled-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 | 4 | 7.7 | 11.9 | 23.0 | 57.4 | 18 |
| roleplay-character-selected-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 |  | 24.2 | 10.3 | 19.0 | 46.5 | 45 |
| roleplay-character-selected-scrolled-160x45 | 13 | 23 | 15 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 |  | 11.3 | 13.6 | 19.0 | 56.0 | 26 |
| roleplay-characters-arrival-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 |  | 8.6 | 12.1 | 19.0 | 60.2 | 22 |
| roleplay-characters-nothing-selected-160x45 | 13 | 23 | 23 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 |  | 5.2 | 7.9 | 19.0 | 67.9 | 16 |
| roleplay-characters-search-nomatch-nothing-selected-160x45 | 13 |  |  | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 0 |  | 0.0 | 8.1 | 20.0 | 72.0 | 0 |
| roleplay-dictionaries-nothing-selected-160x45 | 13 | 20 | 20 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 3 |  | 0.9 | 8.0 | 22.1 | 69.0 | 3 |
| roleplay-dictionaries-selected-160x45 | 13 | 20 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 3 |  | 1.2 | 11.0 | 22.1 | 65.6 | 4 |
| roleplay-dictionaries-tryit-160x45 | 13 | 20 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 3 |  | 1.2 | 14.7 | 22.1 | 62.0 | 3 |
| roleplay-empty-characters-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 3 |  | 16.5 | 10.7 | 19.0 | 53.8 | 36 |
| roleplay-empty-dictionaries-160x45 | 13 |  |  | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 0 |  | 0.0 | 7.9 | 22.1 | 70.0 | 0 |
| roleplay-empty-lore-160x45 | 13 |  |  | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 0 |  | 0.0 | 7.9 | 22.4 | 69.7 | 0 |
| roleplay-empty-personas-160x45 | 13 |  |  | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 0 |  | 0.0 | 7.4 | 19.0 | 73.7 | 0 |
| roleplay-lore-selected-160x45 | 13 | 20 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 3 |  | 1.5 | 9.2 | 22.4 | 66.9 | 4 |
| roleplay-lore-tryit-160x45 | 13 | 20 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 3 |  | 2.6 | 10.3 | 22.4 | 64.7 | 7 |
| roleplay-personas-nothing-selected-160x45 | 13 | 20 | 20 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 4 |  | 1.1 | 7.0 | 19.0 | 73.0 | 4 |
| roleplay-personas-selected-160x45 | 13 | 20 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 4 |  | 2.8 | 9.1 | 19.0 | 69.1 | 9 |
| roleplay-personas-selected-after-scroll-160x45 | 13 | 20 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 4 |  | 2.8 | 9.2 | 19.0 | 69.0 | 9 |
| roleplay-rails-collapsed-160x45 | 13 | 26 | 16 | 4 | boxes 132 / gaps 15+13 | 1 |  | 9.4 | 8.9 | 16.8 | 65.0 | 27 |
| roleplay-testchat-open-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 |  | 13.8 | 12.3 | 21.1 | 52.7 | 29 |
| roleplay-testchat-reply-160x45 | 13 | 23 | 16 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 9 |  | 13.0 | 14.9 | 20.1 | 52.0 | 27 |
| roleplay-typed-text-switched-mode-160x45 | 13 | 21 | 20 | 4 | boxes 39+78+39 / gaps 2+0+0+2 | 1 |  | 1.1 | 7.0 | 19.0 | 73.0 | 4 |
| library-conversations-item-open-120x36 | 5 | 14 | 17 | 2 | boxes 120 / gaps 0+0 | 6 |  | 3.7 | 17.1 | 15.0 | 64.2 | 10 |
| library-conversations-nav-expanded-120x36 | 6 | 15 | 15 | 3 | boxes 29 / gaps 2+89 | 2 |  | 2.1 | 16.6 | 19.7 | 61.6 | 5 |
| library-landing-120x36 | 6 | 11 | 11 | 3 | boxes 29+87 / gaps 2+0+2 | 3 |  | 1.5 | 12.6 | 21.4 | 64.5 | 4 |
| library-media-list-120x36 | 6 | 16 | 18 | 3 | boxes 60 / gaps 7+53 | 2 |  | 0.9 | 9.5 | 16.0 | 73.6 | 3 |
| library-notes-editor-120x36 | 7 | 29 | 9 | 3 | boxes 58 / gaps 7+55 | 5 | 3 | 4.0 | 16.3 | 23.4 | 56.3 | 9 |
| library-notes-list-120x36 | 7 | 31 | 31 | 3 | boxes 58 / gaps 7+55 | 3 |  | 1.2 | 14.6 | 16.4 | 67.8 | 4 |
| library-prompts-item-open-120x36 | 6 | 15 | 9 | 3 | boxes 56 / gaps 7+57 | 3 | 4 | 3.1 | 11.6 | 23.5 | 61.8 | 8 |
| roleplay-character-editor-120x36 | 13 | 23 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 | 3 | 7.3 | 15.6 | 29.7 | 47.5 | 14 |
| roleplay-character-editor-scrolled-120x36 | 13 | 23 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 | 4 | 6.6 | 15.2 | 28.4 | 49.7 | 13 |
| roleplay-character-selected-120x36 | 13 | 23 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 |  | 17.4 | 14.1 | 23.6 | 44.9 | 32 |
| roleplay-characters-arrival-120x36 | 13 | 23 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 |  | 8.4 | 16.8 | 23.6 | 51.2 | 17 |
| roleplay-characters-nothing-selected-120x36 | 13 | 23 | 23 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 |  | 3.1 | 12.4 | 23.5 | 61.0 | 8 |
| roleplay-characters-search-nomatch-120x36 | 13 |  |  | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 0 |  | 0.0 | 12.6 | 24.6 | 62.8 | 0 |
| roleplay-dictionaries-selected-120x36 | 13 | 20 | 17 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 3 |  | 1.7 | 15.5 | 26.1 | 56.8 | 4 |
| roleplay-dictionaries-tryit-120x36 | 13 | 20 | 17 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 3 |  | 1.7 | 18.2 | 26.1 | 54.1 | 4 |
| roleplay-empty-characters-120x36 | 13 | 23 | 23 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 3 |  | 2.1 | 12.5 | 23.5 | 61.9 | 6 |
| roleplay-lore-selected-120x36 | 13 | 20 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 3 |  | 1.4 | 14.1 | 26.6 | 57.8 | 3 |
| roleplay-personas-selected-120x36 | 13 | 20 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 4 |  | 4.3 | 14.1 | 23.5 | 58.1 | 10 |
| roleplay-rails-collapsed-120x36 | 13 | 16 | 16 | 4 | boxes 92 / gaps 15+13 | 1 |  | 22.3 | 8.5 | 20.7 | 48.5 | 43 |
| roleplay-testchat-open-clipped-120x36 | 13 | 23 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 |  | 11.3 | 14.9 | 24.8 | 49.0 | 22 |
| roleplay-testchat-reply-120x36 | 13 | 23 | 16 | 4 | boxes 28+58+30 / gaps 2+0+0+2 | 5 |  | 12.3 | 19.3 | 23.6 | 44.8 | 22 |

## C. Readings that matter for the redesign

* **Vertical chrome.** Roleplay spends 13 rows above any pane content plus 4 below (17 of 45 rows = 38% at 160x45; 17 of 36 = 47% at 120x36). Library spends 5–6 above and 2 below (7–8 rows: 16–17% at 160x45, 19–22% at 120x36).
* **List capacity.** Roleplay's Characters rail stacks seven one-per-row buttons above the list, so the list starts on row 23 and shows 9 characters at 160x45, 5 at 120x36 and **1** at 80x24 (`roleplay-characters-arrival-80x24`). Library's conversations list starts on row 14 and shows the whole set (6) at both sizes.
* **Horizontal split.** Roleplay keeps three bordered panes whose rails grow with the window (rails ≈ 49% of width at every size; the Inspector is mostly blank at 220x55). Library keeps a fixed-width Nav, collapses panes to 5–7-col named grips as width shrinks, and gives the remaining width to the reader/editor.
* **Detail density in Dictionaries/Lore.** A blank band sits above the Entries/Settings tabs (≈7 rows at 160x45, 5 at 120x36, 12 at 220x55) while the Try-it preview squeezes the entry table: at 160x45 Lore shows 1 entry row of 10 before running the preview and 0 after; at 120x36 Dictionaries and Lore show **0** entry rows (`roleplay-dictionaries-tryit-120x36`, `roleplay-lore-selected-120x36`).
