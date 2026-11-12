Note: claude-sonnet-5[1m] (the safety classifier) was unavailable (timed out) when reviewing this subagent's work. Please carefully verify the subagent's actions and output before acting on them.

## Live-capture harness: Roleplay vs Library (origin/dev @ 84247cb843)

The harness works and both screens are captured at every required size. The real profile was not touched: the before/after diff of `~/.config/tldw_cli` and `~/.local/share/tldw_cli` is empty. 80 capture sets are in scratch, and 50 plain-text captures plus `metrics.md` and `HARNESS.md` are in the worktree evidence directory.

### 1. Harness paths and how to launch

Root: `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/6e3b0191-cd3d-4e9c-9b08-af582f81f1fc/scratchpad/rp-review` (written as `$R` below)

```
$PY $R/mock_llm.py 18765 &               # optional: local mock LLM, needed only for test-chat replies
$R/launch.sh <socket> <cols> <rows> [golden|empty]
sleep 18                                  # cold start; don't navigate before the nav bar and footer render
$R/shot.sh <socket> $R/captures/<name>-<cols>x<rows>     # writes .txt + .ansi; PNG=1 also writes .png
tmux -L <socket> send-keys C-q; tmux -L <socket> kill-server
```

`$PY` is `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.

| File | What it does |
|---|---|
| `launch.sh` | Copies the chosen profile to `runs/<socket>/` and regenerates its config. Starts the app in tmux with a throwaway HOME, XDG dirs, `TLDW_CONFIG_PATH` and `[paths].data_dir`, and cwd `runs/<socket>/cwd`. stderr is never redirected. Env options: `REUSE=1` (keep the run dir), `APP_WT=` (other worktree), `EXTRA_TOML=` (extra config). |
| `mkprofile.sh` | Writes the isolated config: `users_name="rp_review"`, `[paths] data_dir`, all 14 `[database] *_db_path` keys plus `USER_DB_BASE_DIR`, `[first_run]` both flags true, `[model_catalog] auto_refresh_enabled=false`. All keys checked against `config.py` and `Backup_Recovery/profile_paths.py:DATABASE_PATHS`. |
| `golden/`, `empty/` | Master profiles, left intact. `golden.preseed.bak/` is the booted profile before seeding. |
| `seed.py`, `seedrun.sh` | Seeding script and its isolated runner. |
| `fixtures/` | The two card files that were imported. |
| `mock_llm.py`, `mock_llm.toml` | Mock server and the config fragment pointing at it. |
| `where.py`, `click.sh`, `drive_lib.sh`, `drive_roleplay.sh`, `drive_library.sh` | Find text on screen and click it; scripted capture passes. |
| `metrics.py`, `metrics.md`, `HARNESS.md` | Metrics script, results, and full instructions. |

- **Navigation:**
  - The palette queries "Switch to Roleplay" and "Switch to Library" each leave one match.
  - Roleplay modes: click a chip, or press Escape and then `c`, `p`, `d`, `l`, `[` or `]`.
  - Library destinations: use the rail entries with `--click L`. At 120 cols and below, expand the collapsed Nav grip first with `"--->"`.
  - Click coordinates for both screens are listed in `HARNESS.md`.
- **Mock LLM (part E):** done. Roleplay's Inspector reads "Ready to chat in Console", and the test chat got real replies (requests are logged in `mock_llm.requests.log`). The mock is stopped now, so restart it before launching if you need replies.

### 2. Isolation check

```
diff <(tail -n +2 isolation-before.txt) <(tail -n +2 isolation-after.txt) && echo "NO DIFF"
→ NO DIFF        (checked after the first boot, mid-run, and at the end)
```

- None of the 8 run logs mentions `/Users/macbook-dev/.config` or `/Users/macbook-dev/.local` (0 hits each).
- `git status` in the worktree shows only `?? Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/`.
- No tmux servers are left running.

### 3. What was seeded (all through the app's own APIs)

- **Characters: 28.** 23 written with `add_character_card`, with:
  - very long and very short names;
  - a 1.2k-character description (Captain Isolde Varga);
  - tags, alternate greetings, system prompts, creator and version fields.
  
  Two cards came in through the real import path (`import_and_save_character_from_file`): "Ser Corwin of the Ashen Vale" (V2 JSON) and "Wren Halloway" (PNG card). The other 3 are built-ins the app adds itself (Default Assistant, Samira, pixel-migu). There were no card files under `Tests/` to import.
- **Personas: 4**, created with `build_persona_service().create_persona_profile`. One ("Isolde's Bridge Officer (linked)") is linked to a character.
- **Dictionaries: 3**, with 15, 8 (one regex) and 6 entries; the last is disabled. "Fantasy Anachronism Swaps" is attached to Dungeon Master Vex.
- **Lore books: 3**, with 10, 6 and 4 entries; one is disabled. Kepler-9 is attached to Isolde and Aetheria Atlas to Vex.
- **Character chats: 3 sessions**, 11 messages in total (two for Isolde, one for Rosa).
- **Library:** 3 plain conversations, 5 notes, 4 prompts, 2 media documents.
- **The `empty/` profile** only has the 3 built-in characters, so "empty" Characters mode is never truly empty. Personas, Dictionaries and Lore are truly empty there.

### 4. Capture index

All captures are in `$R/captures`. The 50 starred-in-`curated.txt` ones are copied to `…/roleplay-library-layout-review-2026-10-01/evidence/`.

**Roleplay, 160x45 (full set)**
- `characters-arrival`: the first character (Ada) is selected automatically on arrival.
- `characters-nothing-selected`: reached by switching to another mode and back.
- `character-selected`, `-scrolled`: Isolde's long card.
- `character-editor`, `-scrolled`, `-advanced`, `-scenario-zero-height`: the editor, including the Advanced section.
- `testchat-open`, `testchat-reply`: test chat before and after a mock reply.
- `rails-collapsed`
- `personas-nothing-selected`, `personas-selected`, `-after-scroll`
- `dictionaries-nothing-selected`, `-selected`, `-tryit`
- `lore-selected`, `lore-tryit`
- `characters-search-nomatch-nothing-selected`
- `typed-text-switched-mode`: typed text triggered mode hotkeys.
- `empty-characters`, `empty-personas`, `empty-dictionaries`, `empty-lore`

**Roleplay, 120x36 (full set):** `characters-arrival`, `characters-nothing-selected`, `character-selected`, `character-editor`, `-scrolled`, `testchat-open-clipped`, `testchat-reply`, `rails-collapsed`, `personas-selected`, `dictionaries-selected`, `-tryit`, `lore-selected`, `characters-search-nomatch`, `empty-characters`.

**Roleplay, 80x24 (key states):** `characters-arrival`, `character-selected`, `characters-nothing-selected`, `characters-search-typed-invisible`, `personas-selected`, `dictionaries-count-zero-on-entry`, `dictionaries-tryit`, `lore-list-first-row-clipped`, `lore-selected`, `empty-characters`.

**Roleplay, 220x55 (key states):** `characters-arrival`, `character-selected`, `characters-nothing-selected`, `personas-selected`, `dictionaries-tryit`, `lore-selected`, `empty-characters`.

**Roleplay, 60x24:** `lore-narrow-rails-autocollapsed`. Both rails collapse automatically and their labels clip to "L" and "In".

**Library:**
- 160x45: `landing`, `conversations-item-open` (three panes), `notes-list`, `notes-editor`, `prompts-list`, `prompts-item-open`, `media-list`, `empty-landing`.
- 120x36: `landing`, `conversations-item-open`, `conversations-nav-expanded`, `notes-list`, `notes-editor`, `prompts-item-open`, `media-list`.
- 80x24: `landing`, `conversations-item-open`.
- 220x55: `landing`, `conversations-item-open`, `empty-landing`.
- 60x24: `conversations-item-open` (single stage), `narrow-after-esc`.

**States not reached as specified:**
- Characters mode has no "nothing selected" state on arrival because the first item is auto-selected. I captured it by switching modes and back.
- The test chat at 120x36 could not be operated normally; it needed a workaround (bug 6).

### 5. Metrics

Hand-counted from the captures. The full automated table (39 Roleplay and 15 Library rows) and its method are in `metrics.md`.

| Measure | Roleplay 160x45 | Library 160x45 (conversation open) | Roleplay 120x36 | Library 120x36 |
|---|---|---|---|---|
| Chrome rows above the first pane row | **13**: nav 3, title box 5, purpose line 1, Modes 1, frame 1, spacer 1, pane border 1 | **5–6**: nav 3, "Library \| Local" 1, frame 1 (+1 for the Nav box border) | **13** | **5–6** |
| List chrome before the first item | **9 rows** (one button per row), so the first item is on row 23. Personas, Dictionaries and Lore: 6 rows | **8 rows**, first item on row 14 | 9 | 8 |
| Chrome rows below the content | **4** | **2–3** | 4 | 2–3 |
| List items visible | **9 of 28** | **6 of 6** (room for about 10) | **5** (1 at 80x24) | 6 of 6 |
| Columns | frame 2 + rail 39 + **centre 78** + Inspector 39 + frame 2: centre gets 49%, rails get 49% | Nav 34 + grip 5 + list about 50 + grip 5 + **reader 62** | 28 + **58** + 30 | Nav collapses to a grip; list 57 + reader 50 |
| Collapsed rails | 14 + 12 cols | grips are 5–7 cols, with a vertical name | 26 cols | 5–7 cols |
| Editor fields visible | **5** (inputs show 1, 1, 2, 1, 1 rows of text) | Note editor 3 (16-row body); prompt editor 4 | **2½** | 3 and 4 |
| Vertical chrome as a share of height | 17/45 = **38%** | 7–8/45 = 16–17% | 17/36 = **47%** | 19–22% |

From the automated table (key states; percentages are shares of all screen cells):

| Capture | Data % | Label % | Border % | Blank % |
|---|---|---|---|---|
| roleplay-character-selected 160 | 24.2 | 10.3 | 19.0 | 46.5 |
| roleplay-characters-arrival 160 | 8.6 | 12.1 | 19.0 | 60.2 |
| roleplay-character-editor 160 | 9.9 | 11.9 | 29.0 | 49.2 |
| roleplay-dictionaries-tryit 120 | 1.7 | 18.2 | 26.1 | 54.1 |
| roleplay-lore-selected 120 | 1.4 | 14.1 | 26.6 | 57.8 |
| library-conversations-item-open 160 | 2.2 | 14.7 | 16.4 | 66.7 |
| library-prompts-item-open 160 | 3.0 | 12.1 | 24.1 | 60.8 |
| library-notes-editor 120 | 4.0 | 16.3 | 23.4 | 56.3 |

- **Method:** `metrics.py` puts each cell in one class. "Data" means seeded user text, matched against a copy of the golden databases. "Label" is all other text, "border" is box and scrollbar glyphs, and "blank" is spaces.
- **Caveat:** the data share rewards long seeded text. Isolde's long description inflates Roleplay's detail views, so compare like states with like.

### 6. Bugs and oddities seen while driving

1. **Advanced editor fields are invisible.** Scenario, Post-history instructions and Creator notes have no visible text rows. Isolde's scenario cannot be seen even with the field focused. The CSS sets `height: 2`, which the 2-row border uses up (`tldw_chatbook/Widgets/Persona_Widgets/personas_character_editor_widget.py:119-123`). Capture: `roleplay-character-editor-scenario-zero-height-160x45`.
2. **The Characters search box hides what you type.** When focused it becomes a 2-row border with no text row, at every size. The rail CSS sets `height:1; border:none` (`personas_library_pane.py:159-164`), and the focus border appears to override it. Captures: `roleplay-characters-search-nomatch-*`, `-search-typed-invisible-80x24`.
3. **A search with no matches says the library is empty.**
   - The list reads "No characters yet - use New or Import…" even though 28 exist. The empty-state code ignores `filtered` (`personas_library_pane.py:431-436`).
   - The purpose line shows "· 0" rather than "0 of 28".
4. **Counts read "· 0" for 2–3 seconds** after entering Dictionaries or Lore, with no loading indicator, before changing to 3. Captures: `roleplay-dictionaries-nothing-selected-160x45`, `-count-zero-on-entry-80x24`.
5. **Single-letter hotkeys fire when a click misses a text field.** `c p d l [ ] s` and Space (which toggles a dictionary on or off) all act as hotkeys. Typing a sentence switched modes several times in two separate runs. Capture: `roleplay-typed-text-switched-mode-160x45`.
6. **The test chat is clipped and cannot be scrolled at 120x36.**
   - Only the input's top border shows, and the Test Reply, Reset, Send and Configure buttons are hidden. After a reply the input disappears completely.
   - At 160x45 the button row is clipped after one exchange.
   - The panel always shows an empty "Greeting:" block of about 3 rows.
   
   Captures: `roleplay-testchat-open-clipped-120x36`, `-reply-120x36`, `-reply-160x45`.
7. **Selection behaves inconsistently.** The first character is auto-selected on arrival, but returning from another mode shows nothing selected.
8. **Dictionaries and Lore waste rows at the top while the entry table starves.** A blank band of 7 rows (160x45), 5 (120x36) or 12 (220x55) sits above the tabs.
   - Lore shows 1 of 10 entries at 160x45, then none after running the preview.
   - Lore and Dictionaries show no entries at 120x36; Lore shows none at 80x24.
   - At 80x24 the preview results are below the fold.
9. **The Characters rail stacks 7 buttons one per row:** New, New Actor Pack, Import, Import Actor Pack, Duplicate, Sort, Tag. At 80x24 the labels truncate to "Duplic", "Sort:", "Tag:" and two identical "New" and "Import" rows, and only one character is visible.
10. **The Inspector shows irrelevant content.**
    - The four Buddy buttons appear in every mode, including Dictionaries and Lore and with nothing selected.
    - It says "Pick a character or persona to start chatting." in Dictionaries and Lore.
    - At 220x55 it is mostly blank, because rails grow with the window (39 cols at 160, 54 at 220).
11. **Collapsed Roleplay rails are wide:** 14 and 12 cols, against 5–7 for Library grips.
12. **The character detail has two scrollbars side by side,** and the Dictionaries (0) and World Books (1) sections are only reachable by scrolling the outer pane.
13. **The persona "Edit" button is below the fold** at 160x45 and 120x36 even though there is empty space above it. Compare `roleplay-personas-selected-160x45` with `-after-scroll`.
14. **Smaller rendering issues:**
    - The Lore list at 80x24 opens scrolled past the first item's name.
    - There is a "(soon)" placeholder control: "Include recent turns (soon)".
    - The Default Assistant's first message shows raw markdown (`**…**`), while the test chat strips `*`.
    - At 120 cols the Inspector's search placeholder truncates to "Search this".
15. **The Roleplay user guide is out of date** (`Docs/User_Guide/roleplay-chat-dictionaries.md`):
    - Lines 23 and 39 name the screen "Roleplay & Chat Dictionaries"; the live title is "Roleplay" and the palette entry is "Tab Navigation: Switch to Roleplay".
    - Lines 6 and 55 describe personas as "assistant profiles"; the live text says "who you play in the chat".
    - Line 118 lists Inspector actions Attach to Console and Start Chat; the live buttons are Chat now, Send to Console draft, Export Actor Pack and four Buddy buttons.
16. **Library problems.** Library is the model for the redesign, but it has issues too:
    - The note editor toolbar is clipped on the right, and Save is hidden at 120x36.
    - A stray "S" glyph appears under the Notes grip.
    - At 120x36 the Notes list shows 3 of 5 notes under about 20 rows of chrome.
    - At 80x24 the conversation reader's messages are below the fold.
    - At 60x24 the reader text is clipped on the right ("Resume conversatio").
    - The landing view only appears on the first visit per launch; after that the Library reopens the last destination.
17. **The app log contains a splash-screen error:** "Error updating animation frame: closing tag '[/bold]' does not match any open tag". It is logged at `tldw_chatbook/Widgets/splash_screen.py:477` and appears in `runs/rp2`.