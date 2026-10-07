# rp-review live-capture harness (Roleplay vs Library, tldw_chatbook)

`R=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/6e3b0191-cd3d-4e9c-9b08-af582f81f1fc/scratchpad/rp-review`
`PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`
Code under test: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/roleplay-library-ux` (origin/dev @ 84247cb843), via `PYTHONPATH`. Override with `APP_WT=<worktree>`.

## Hard rules

* Launch the app **only** through `launch.sh`. It gives every run a disposable HOME / XDG_CONFIG_HOME / XDG_DATA_HOME / XDG_CACHE_HOME / `TLDW_CONFIG_PATH` / `[paths].data_dir` / every `[database]` path, all under `$R/runs/<socket>/`. Never run `python -c "import tldw_chatbook.config ..."` outside `seedrun.sh` — config readers can write the real `~/.config/tldw_cli`.
* Never redirect the app's stderr (Textual paints the UI on stderr; `2>log` blanks the pane).
* Use a socket name unique to you; `tmux -L <socket> kill-server` when done, even on failure.
* Never modify `$R/golden` or `$R/empty` (they are the masters; runs work on copies).
* Isolation check: `$R/isolation-before.txt` is the baseline `ls -la`/`stat` of the real profile; re-run the same commands and diff (see the end of this file).

## Files

| Path | What |
|---|---|
| `launch.sh <socket> <cols> <rows> [golden\|empty]` | copy profile → `runs/<socket>/`, regenerate its config, start `tmux -L <socket>` with the app (cwd = `runs/<socket>/cwd`, where trace exports land). `REUSE=1` keeps an existing run dir (persistence across restarts). `INPLACE=1` boots the master profile itself (maintenance only). `EXTRA_TOML=<file>` appends TOML (default: `mock_llm.toml`). |
| `mkprofile.sh <dir>` | writes the isolated config: `[general] users_name="rp_review"`, `[paths] data_dir`, every `[database] *_db_path` (14 keys from `Backup_Recovery/profile_paths.py:DATABASE_PATHS`) + `USER_DB_BASE_DIR`, `[first_run] setup_started/setup_completed = true` (no wizard), `[model_catalog] auto_refresh_enabled = false` (no consent modal) |
| `golden/` | seeded master profile (`seed.py`); `golden.preseed.bak/` = same profile after first boot, before seeding (re-seed from it) |
| `empty/` | same isolation config, never seeded (only the 3 built-in characters the app seeds itself: Default Assistant, Samira, pixel-migu) |
| `seed.py`, `seedrun.sh <profile> <script>` | seeding via the app's own APIs, run with the same isolated env (script refuses to run otherwise) |
| `fixtures/ser_corwin.v2.json`, `fixtures/wren_halloway.card.png` | Character Card V2 JSON + PNG (`chara` tEXt) imported through `Character_Chat_Lib.import_and_save_character_from_file` |
| `mock_llm.py [port]`, `mock_llm.toml` | stdlib OpenAI-compatible mock (GET /v1/models, POST /v1/chat/completions, SSE when `stream:true`); requests logged to `mock_llm.requests.log` |
| `shot.sh <socket> <out-no-ext>` | `.txt` (capture-pane -p) + `.ansi` (-e -J); `PNG=1` also renders `.png` via `shot_png.py` (Rich → HTML → headless Chromium) |
| `where.py <socket> "<text>" [--click [n\|L\|B]]` | find text on screen (1-based row/col, character columns); `--click` clicks the nth / Leftmost / Bottom-most match's centre |
| `click.sh <socket> <col> <row>` | SGR left click |
| `drive_lib.sh`, `drive_roleplay.sh <socket> <size> [full\|key]`, `drive_library.sh <socket> <size> [full\|key]` | scripted capture passes (see "Driving notes" before trusting them) |
| `metrics.py <captures-dir> <size>...` | mechanical layout metrics (method in `metrics.md`); corpus read from `corpus_tmp/` (a copy of the golden DBs) |
| `captures/` | all captures `<screen>-<state>-<cols>x<rows>.{txt,ansi[,png]}` |

## Launch

```bash
# optional, for test-chat replies (port 18765 is what mock_llm.toml points at):
$PY $R/mock_llm.py 18765 &          # or run_in_background
$R/launch.sh rpX 160 45 golden      # or empty
sleep 18                            # COLD START ~10-15 s; do not touch the app before the nav bar + footer render.
```
Navigating during cold start is a trap: the delayed initial-tab switch yanks the screen back to Console and DB handles are not assigned yet. Boot lands on Console. With the mock config the Console header reads "Ready" and Roleplay's Inspector reads "Ready to chat in Console." (provider "OpenAI / gpt-4.1-mini" → mock).
App log: `$R/runs/<socket>/data/rp_review/tldw_cli_app.log`.
Quit: `tmux -L rpX send-keys C-q; sleep 2; tmux -L rpX kill-server`.

## Drive

* Keys: `tmux -L S send-keys Escape|Enter|Tab|Down|C-p|C-f|C-n|C-s|C-q`; text: `send-keys -l "literal"`. **Ctrl+digit hotkeys cannot be sent through tmux.**
* Click (1-based CHARACTER columns, not bytes): `send-keys -l $'\x1b[<0;COL;ROWM'` then `$'\x1b[<0;COL;ROWm'` (= `click.sh`). Wheel: `$'\x1b[<65;COL;ROWM'` down / `64` up.
* Capture: `capture-pane -p` (plain), `capture-pane -e -p -J` (ANSI). Resize: `tmux -L S resize-window -x C -y R` (sleep 3 before capturing).
* Command palette: `C-p`, type until ONE match remains, `Down`, `Enter`. Known-good single-match queries: **"Switch to Roleplay"** → "Tab Navigation: Switch to Roleplay"; **"Switch to Library"** → "Tab Navigation: Switch to Library". (The user guide's "Switch to Roleplay & Chat Dictionaries" label is stale.)
* Nav bar: `where.py S "⌃4 Roleplay" --click` / `"⌃3 Library"` (row 2; visible at every width ≥ 60).

### Roleplay (all text anchors; work at every size unless noted)

| Goal | How |
|---|---|
| Switch mode | click the chip: `"Characters  "`, `"Personas  "`, `"Dictionaries  "`, `"    Lore"` (two trailing / four leading spaces avoid matching other text) — or press **Escape** (leave any text box) then `c` / `p` / `d` / `l`; `[` / `]` cycle. **Hazard:** if focus is not in a text box, every typed `c p d l [ ] s` (and Space in Dictionaries) is a hotkey — typing a sentence into a mis-clicked field switches modes several times (`captures/roleplay-typed-text-switched-mode-160x45`). Always verify the field shows a focused `┌` border before `send-keys -l`. |
| Select an item | click its name: `"Captain Isolde Varga"`, `"Default Narrator"`, `"Fantasy Anachron"`, `"Aetheria Atlas"`. At 80x24 the rail shows ~1 item: use `C-f`, type (text is invisible — zero-height input), `Enter` (to list), `Enter` (select). Truncated names: `"Default …"`, `"Fantasy …"`; in Lore at 80x24 the first row's name is scrolled off — click `"10 entri…"`. |
| Arrival state | first character (Ada) is auto-selected on arrival; **nothing-selected** is reached by switching to another mode and back (`Personas  ` → `Characters  `). |
| Character editor | `where.py S "  Edit  " --click B`; scroll with wheel over the centre pane; `"Advanced ▸"` expands the long tail; leave with `"Cancel"` or Escape. |
| Test chat | `"Try a test chat"` toggles (click again if still `▸` after 3 s). At 160x45: click `"Test message..."`, type, click `"Test Reply"` (or Enter). At 120x36 the input is clipped to its top border: click that border row (`▊▔▔…` under "Greeting:"), confirm it turns `┌`, type blind, Enter. |
| Collapse rails | click the `<` on the "Library" rail header row and the `>` on the "Inspector" header row (find with `where.py` restricted to rows > 3; `drive_roleplay.sh` shows the awk). Restore: click the `" Library "` handle (leftmost) and `" Inspector "` handle. |
| Try it (Dictionaries/Lore) | row r of `"Try it — substitution"` / `"Try it — injection"`; click (col+10, r+2); confirm `┌`; clear old text with `Down×6 End` + `send-keys -N 700 BSpace` (Ctrl+A is NOT select-all there); type; click `"Run preview"`. Draft text persists across selections. |
| No-match search | `C-f`, type `zzqx` → shows the wrong empty-state copy ("No characters yet - use New or Import…") and count "· 0". |

Click coordinates at 160x45 (golden, arrival layout): nav Roleplay (44,2), Library (30,2); chips Characters (18,10) Personas (31,10) Dictionaries (45,10) Lore (54,10); rail collapse `<` (38,14); Inspector collapse `>` (155,15); collapsed handles: Library (8,27), Inspector (154,14); test-chat toggle (73,14); first character row (Ada) (16,23); Captain Isolde (16,29); Edit (≈49, row 39–41 depending on detail scroll; use `--click B`). At 120x36: chips identical rows/cols (row 10); rail `<` (34,14); list rows 23–32.

### Library

| Goal | How |
|---|---|
| Landing | only on the first visit of a launch; the Library remembers its last destination for later visits. |
| Destination | rail entries `"Conversations (6)"`, `"Notes (5)"`, `"Prompts (4)"`, `"Media (2)"` with `--click L` (leftmost; the landing body repeats some labels). At ≤120 cols opening an item collapses Nav to a grip: click the leftmost `"--->"` (or `"‹"`) first (`drive_library.sh: navto`). At 60x24 the rail says `"Chats (6)"`. |
| Open an item | click its title with `--click B`: `"Aetheria session 12 prep"` (160x45) / `"Reading list"` (120x36, the 5th note is below the fold), `"Roleplay scene setter"`; a conversation auto-loads the first one into the reader. |
| Back | Escape (editor → list → rail search). |

## Driving notes (read before reusing the drivers)

* `drive_*.sh` are text-anchor based and were corrected after a first pass; verify each capture (the 120x36 test-chat + dictionaries steps needed manual handling, documented above). Runs are cheap — relaunch rather than repair a confused state.
* Library at 120x36 run twice in one launch: the second "landing" is the remembered destination, not the landing.
* After any typing step, check the screen's mode line (row 9 at most sizes) did not change.

## Isolation check

```bash
snap(){ echo "== $(date)"; ls -la ~/.config/tldw_cli ~/.local/share/tldw_cli; echo "== stat"; stat -f '%N %m %Sm %z' ~/.config/tldw_cli/config.toml ~/.local/share/tldw_cli/default_user ~/.config/tldw_cli ~/.local/share/tldw_cli; echo "== default_user listing"; ls -la ~/.local/share/tldw_cli/default_user; }
snap > /tmp/x.txt 2>&1; diff <(tail -n +2 $R/isolation-before.txt) <(tail -n +2 /tmp/x.txt) && echo untouched
grep -c '/Users/macbook-dev/.config\|/Users/macbook-dev/.local' $R/runs/*/data/rp_review/tldw_cli_app.log   # expect 0
```

## Mock LLM

`mock_llm.py` replies `*(mock reply)* You said: "<last user msg>". Staying in character as <first line of system prompt> …`. The Roleplay test chat used non-streaming (`stream=False`) calls; Console probes use `stream=True`. It is not running after this session — start it before launching if you need replies (otherwise Readiness still says Ready but sends fail).
