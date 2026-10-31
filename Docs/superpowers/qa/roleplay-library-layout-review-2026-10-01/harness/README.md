# rp-review live-capture harness

Drives the real tldw_chatbook TUI in tmux against disposable, seeded profiles. It produced the
captures, metrics and gap checks of the 2026-10-01 Roleplay vs Library review, and is the live
check for the Roleplay-on-Library-frame slices (B0 onward), including B7's "restart: preferred
state restored" step.

Only the **scripts** live here. Profiles, runs, captures, mock logs and bytecode live under
`HARNESS_STATE`, which is never inside a tracked part of a checkout.

## Isolation guarantees

- **No real profile.** Every app process runs with a cleared environment (`env -i` plus an
  allowlist). Its HOME, XDG config/data/cache/state dirs and `TLDW_CONFIG_PATH` all point into
  the run directory. Its config sets `[paths].data_dir`, all 14 `[database] *_db_path` keys
  (`Backup_Recovery/profile_paths.py:DATABASE_PATHS`) and `USER_DB_BASE_DIR` inside the run.
  The user folder is `rp_review`.
- **No inherited provider keys.** `OPENAI_API_KEY` and friends are not passed through. A run
  without the mock fragment reads "Missing API key" (verified).
- **No OS keychain.** The login keychain is per-user state outside HOME, so the app gets
  `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`, the same approach as
  `Backup_Recovery/isolated_restore.py`. Keyring-backed features (personal context and profile
  tools, web auth tokens, skill trust) run in their locked, fail-closed state.
  `RP_KEYRING_BACKEND=""` restores the OS keychain; avoid it.
  - Measured without the null backend: the personal-context bootstrap parked on its keychain call, the
    first Console send waited out its 10 s budget, and Ctrl+Q took the full 120 s shutdown grace.
  - Measured with it: the reply arrived in 4.7 s and the app exited in 2 s.
- **No writes into the code checkout.** Bytecode is redirected with
  `PYTHONPYCACHEPREFIX=$HARNESS_STATE/pycache`. The app's cwd is `runs/<socket>/cwd`, so trace
  exports land there.
- **Guarded seeding.** Every script that opens app databases (`seed.py`, `scale_seed.py`,
  `persona_more.py`, `lore_fix.py`) calls `harness_guard.py` before importing `tldw_chatbook`,
  and again after. The guard refuses, with a message starting `REFUSING:`, when:
  - `HARNESS_STATE` is unset, relative, too broad, or overlaps `~/.config/tldw_cli` or
    `~/.local/share/tldw_cli`. The real home comes from the password database, never from `$HOME`.
  - `TLDW_CONFIG_PATH`, `HOME` or `XDG_*` resolve outside `HARNESS_STATE` or into the real profile.
  - The target is a protected master.
  - The app's own data dir or a DB path resolves outside the profile.
  - `tldw_chatbook` was imported from outside `APP_WT`.
- **No committable state.** `HARNESS_STATE` is refused when it sits inside a git work tree and is
  not git-ignored, for example under `Docs/`.
- **Masters are never launched directly.** `launch.sh` copies a master into `runs/<socket>/`.
  `INPLACE=1` boots a master itself and is used only by `make_profiles.sh`.
- **Proof.** `launch.sh --self-test` diffs a byte/mtime manifest of the real profile before and
  after two app boots (`isolation_snap.py`). The manifest also lists the metadata, never the
  secrets, of every `tldw_chatbook*` login-keychain item.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `HARNESS_STATE` | `<main checkout>/.worktrees/.rp-harness-state` | Root for profiles, runs, captures, mock logs and pycache. `<main checkout>` is the parent of `git rev-parse --git-common-dir`. `.worktrees/` is git-ignored and outlives slice worktrees. |
| `APP_WT` | toplevel of the checkout that holds this harness | The checkout whose `tldw_chatbook` runs. It goes on `PYTHONPATH` for both `launch.sh` and `seedrun.sh`. |
| `PY` | `<main checkout>/.venv/bin/python`, else `$APP_WT/.venv/bin/python` | Interpreter. Python 3.11 or later is needed for `tomllib`. |
| `EXTRA_PYTHONPATH` | (none) | Appended after `APP_WT`, for example a directory holding an instrumentation `sitecustomize.py`. |
| `REUSE=1` | off | `launch.sh` keeps `runs/<socket>/`. An existing `config.toml` is kept, and only its path keys are rewritten. |
| `INPLACE=1` | off | `launch.sh` boots the master itself (maintenance only). |
| `EXTRA_TOML` | `mock_llm.toml` | Fragment appended to a **fresh** run config. Set it to `""` for none. It is ignored on REUSE. |
| `PNG=1` | off | `shot.sh` also renders a PNG (needs playwright and Chromium). |
| `CAPTURES` | `$HARNESS_STATE/captures` | Output directory for the `drive_*.sh` drivers. |
| `CORPUS_DIR` | `$HARNESS_STATE/corpus_tmp` | `metrics.py` corpus, a copy of the golden DBs. |
| `BOOT_TIMEOUT` | `240` | Seconds to wait for the nav bar (`make_profiles.sh`, self-test). |
| `KEEP=1` | off | Self-test keeps its temporary state dir on PASS. |
| `RP_KEYRING_BACKEND` | `keyring.backends.null.Keyring` | Value given to the app's `PYTHON_KEYRING_BACKEND`. `""` means the OS keychain (not isolated). |

Requirements: bash (macOS `/bin/bash` 3.2 is fine), tmux 3.0 or later (argv commands,
`-c`), git 2.31 or later, and the app's venv.

## Quick start

```bash
H=Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness   # from a checkout
export APP_WT=/path/to/the/worktree/under/test     # optional; default = this checkout
PY=/path/to/main/checkout/.venv/bin/python          # the scripts find it themselves; this is for the lines below

$H/launch.sh --self-test          # ~1 min; must print SELF-TEST: PASS
$H/make_profiles.sh               # empty/ + golden/ (+ golden.preseed.bak/); add --volume for volume/
$PY $H/mock_llm.py 18765 &        # optional: test-chat / Console replies (port of mock_llm.toml)
$H/launch.sh rpX 160 45 golden    # or empty | volume
$PY $H/waitfor.py rpX "4 Roleplay" 240 && $PY $H/waitfor.py rpX "Ctrl+Q" 30
$H/drive_roleplay.sh rpX 160x45 key    # captures -> $HARNESS_STATE/captures/
tmux -L rpX send-keys C-q; sleep 3; tmux -L rpX kill-server
```

To restart and check persisted state (B7), use `REUSE=1 $H/launch.sh rpX 160 45 golden`. The run
dir and its `config.toml` are kept, including sections such as `[roleplay.reader]`; only the path
keys are rewritten.

The app log is `$HARNESS_STATE/runs/<socket>/data/rp_review/tldw_cli_app.log`.

## Files

| Path | What |
|---|---|
| `env.sh` | Sourced by every shell script. It resolves and validates `HARNESS_STATE`, `APP_WT` and `PY`, and defines `rp_app_env` (the `env -i` allowlist) and `rp_under_state`. |
| `launch.sh <socket> <cols> <rows> [master]` | Copies the master to `runs/<socket>/`, writes a fresh config, and starts `tmux -f /dev/null -L <socket>` with the app as argv (no shell, no stderr redirect). Supports `REUSE=1`, `INPLACE=1` and `EXTRA_TOML`. |
| `launch.sh --self-test` (`launch_selftest.sh`) | Isolation and restart self-test. See below. |
| `make_profiles.sh [--volume] [--force]` | Builds the masters from scratch: empty skeleton → golden first boot (INPLACE, nav bar, Ctrl+Q) → `golden.preseed.bak/` snapshot → `seed.py` → golden. `--volume` then copies golden → fresh config → `scale_seed.py` + `lore_fix.py` → `volume/` (348 characters, 43 lore books, 28 dictionaries; counts verified). |
| `mkprofile.sh <dir>` | Skeleton plus a **fresh** config (`profile_config.py init`). It overwrites. |
| `profile_config.py init/repath/set/get` | Config writer. `repath` rewrites only the path keys, and `set` sets one key. Each edit is verified by re-parsing: the result must equal the old content plus exactly the requested keys. Writes are atomic with mode 0600. |
| `seedrun.sh <profile> <script.py> [args]` | Runs a script with the same isolated env and code as `launch.sh`. |
| `harness_guard.py` | The REFUSING guard (see Isolation guarantees). Its CLI is `state` and `real-profiles`. |
| `seed.py` | Golden data through the app's own APIs. It covers 23 characters plus a Character Card V2 JSON and a PNG (`chara` tEXt) import, generated synthetically into `$HARNESS_STATE/fixtures/`. It also adds 4 personas, 3 dictionaries (one attached to Vex, one regex), 3 lore books (attached to Isolde and Vex), 3 character chats, 3 conversations, 5 notes, 4 prompts and 2 media items. Masters: golden only (runs allowed). |
| `scale_seed.py`, `lore_fix.py`, `persona_more.py` | Gap-round volume helpers: 320 characters, 60 personas, 40 lore books (a 250-entry codex), 25 dictionaries (one with 150 rules incl. 10 regex) and 30 Vex chats. `lore_fix.py` tops the lore books up to 43. `persona_more.py` adds 50 personas to probe the 100-row list cap. Allowed on runs and the `volume` master only. |
| `mock_llm.py [port] [--bodies]`, `mock_llm.toml` | Stdlib OpenAI-compatible mock (`GET /v1/models`, `POST /v1/chat/completions`, SSE when `stream:true`), bound to 127.0.0.1. Logs go to `$HARNESS_STATE/mock/`, and `--bodies` also records full request bodies. |
| `shot.sh <socket> <out>` | Writes `.txt` (`capture-pane -p`) and `.ansi` (`-e -J`). `PNG=1` adds `.png` via `shot_png.py`. Relative `<out>` goes under `$HARNESS_STATE/captures/`. |
| `where.py <socket> "<text>" [--click [n\|L\|B]]` | Finds text on screen (1-based row and column, character columns). `--click` clicks the centre of the nth, leftmost or bottom-most match. |
| `click.sh <socket> <col> <row>`, `ia_click.py <socket> <text> [minrow] [mincol]` | SGR left click, and click-first-match restricted to a region. |
| `waitfor.py <socket> <text> [timeout] [--absent]` | Polls the pane every 40 ms and prints the elapsed time. Use this instead of fixed sleeps. |
| `drive_lib.sh`, `drive_roleplay.sh <socket> <size> [full\|key]`, `drive_library.sh …` | Scripted capture passes, anchored on text. See Driving notes. |
| `metrics.py <captures-dir> <size>...` | Mechanical layout metrics (method in the review's `evidence/metrics.md`), computed against a corpus copied from golden. |
| `isolation_snap.py` | Manifest of the real profile (type, size, mtime and SHA-256 up to depth 2) plus `tldw_chatbook*` keychain item metadata. It is read-only. |

## Self-test

`launch.sh --self-test` creates `$HARNESS_STATE/selftest.XXXXXX`, then:

1. builds an `empty` master and starts `mock_llm.py` on a random port;
2. boot 1: a fresh run on a unique socket → nav bar → Ctrl+Q;
3. writes `[roleplay.reader] harness_selftest_sentinel = "<token>"` and moves `[paths].data_dir`
   elsewhere;
4. boot 2: `REUSE=1` → asserts that:
   - the sentinel survived `launch.sh`, the app's boot and its exit;
   - the data dir was rewritten back into the run;
   - the nav bar rendered (one pane captured);
   - readiness reads `Ready`;
   - a Console send is answered by the mock;
5. kills the tmux server, then asserts that:
   - the real profile is byte- and mtime-unchanged, and no `tldw_chatbook` keychain item
     appeared or changed;
   - the app log never names the real profile;
   - `tldw_chatbook` came from `APP_WT`.

It prints a `PASS`/`FAIL` line per check and ends with `SELF-TEST: PASS` or `SELF-TEST: FAIL`. The
exit status is 0 only on PASS. On PASS the temporary directory is removed (keep it with `KEEP=1`).

## Driving the app

- **Keys:** `tmux -L S send-keys Escape|Enter|Tab|Down|C-p|C-f|C-n|C-s|C-q`. **Text:**
  `send-keys -l "literal"`.
- **Click:** `send-keys -l $'\x1b[<0;COL;ROWM'`, then `$'\x1b[<0;COL;ROWm'`; `click.sh` does
  both. COL and ROW are 1-based **character** columns, not bytes.
- **Wheel:** `$'\x1b[<65;COL;ROWM'` scrolls down and `64` scrolls up (`drive_lib.sh: wheel`).
- **Capture:** `capture-pane -p` gives plain text and `capture-pane -e -p -J` gives ANSI.
- **Resize:** `tmux -L S resize-window -x C -y R`, then wait about 3 s before capturing.
- **Command palette:** press `C-p`, type until one match remains, then `Down`, `Enter`.
  Known-good single-match queries are "Switch to Roleplay" and "Switch to Library".
- **Nav bar:** use `where.py S "⌃4 Roleplay" --click` or `"⌃3 Library"` (row 2, visible at every
  width ≥ 60).

### Roleplay (golden data; text anchors work at every size unless noted)

| Goal | How |
|---|---|
| Switch mode | Click the chip `"Characters  "`, `"Personas  "`, `"Dictionaries  "` or `"    Lore"`. Or press **Escape** (to leave any text box), then `c`, `p`, `d` or `l`; `[` and `]` cycle. **Hazard:** when focus is not in a text box, every typed `c p d l [ ] s` (and Space in Dictionaries) is a hotkey. Check that the field shows a focused `┌` border before `send-keys -l`. |
| Select an item | Click its name: `"Captain Isolde Varga"`, `"Default Narrator"`, `"Fantasy Anachron"` or `"Aetheria Atlas"`. At 80x24 the rail shows about one item. Press `C-f`, type (the text is invisible because the input has zero height), then `Enter`, `Enter`. |
| Arrival state | The first character is auto-selected. To reach nothing-selected, switch mode and back. |
| Character editor | `where.py S "  Edit  " --click B`. Scroll with the wheel over the centre pane. Leave with `"Cancel"` or Escape. |
| Test chat | `"Try a test chat"` toggles it; click again if it still shows `▸` after 3 s. At 160x45, click `"Test message..."`, type, then Enter. At 120x36 the input is clipped to its top border: click that border row, confirm it turns `┌`, then type blind. |
| Collapse rails | Click the `<` on the "Library" rail header and the `>` on the "Inspector" header (rows > 3; see `drive_roleplay.sh`). To restore, click the `" Library "` and `" Inspector "` handles. |
| Try it (Dictionaries/Lore) | Find the row of `"Try it — substitution"` or `"Try it — injection"` and click (col+10, row+2). Confirm the `┌` border. Clear the field with `Down×6 End` plus `send-keys -N 700 BSpace`; Ctrl+A is not select-all there. |

### Library

| Goal | How |
|---|---|
| Landing | Shown only on the first visit of a launch. Later visits open the remembered destination. |
| Destination | Rail entries `"Conversations (6)"`, `"Notes (5)"`, `"Prompts (4)"`, `"Media (2)"` with `--click L`. At ≤120 columns, opening an item collapses Nav to a grip: click `"--->"` or `"‹"` first (`drive_library.sh: navto`). At 60x24 the rail says `"Chats (6)"`. |
| Open an item | Click its title with `--click B`. |
| Back | Escape (editor → list → rail search). |

Click coordinates at 160x45 (golden, arrival layout):

- Nav: Roleplay (44,2), Library (30,2).
- Chips: Characters (18,10), Personas (31,10), Dictionaries (45,10), Lore (54,10).
- Rail collapse `<` (38,14).

Prefer text anchors: coordinates drift with every layout change.

### Driving notes

- `drive_*.sh` anchor on text. Verify each capture: the 120x36 test-chat and dictionaries steps
  needed manual handling. Runs are cheap, so relaunch rather than repair a confused state.
- After any typing step, check that the screen's mode line (row 9 at most sizes) did not change.

## Traps

- **Cold start.** Boot takes 10-15 s, and the very first boot of a fresh `HARNESS_STATE` takes
  longer while bytecode compiles into `pycache/`. Wait for the nav bar (`waitfor.py S "4 Roleplay"`)
  and the footer (`"Ctrl+Q"`). Navigating earlier is a trap: the delayed initial-tab switch yanks
  the screen back to Console, and DB handles are not assigned yet.
- **Ctrl+digit hotkeys cannot be sent through tmux.** Click the nav bar or use the palette.
- **SGR clicks use character columns**, 1-based, not byte offsets. Multi-byte glyphs (`⌃`, `▸`,
  `…`) count as one column. `where.py` already reports character columns.
- **Textual draws the UI on stderr.** Never redirect the app's stderr: `2>log` blanks the pane.
  `launch.sh` passes the command to tmux as argv, with no shell.
- **Trace exports land in the app's cwd**, which is `runs/<socket>/cwd`. Never launch the app from
  a checkout.
- **Quit before kill.** Ctrl+Q exits in about 2 s with the null keyring, but up to the 120 s
  `shutdown_grace_seconds` if a worker is parked (for example on the OS keychain). Wait for the
  process before `kill-server`, because the app writes its config on the way out.
- **REUSE is the only restart.** A plain relaunch on the same socket wipes `runs/<socket>/` and
  writes a fresh config.
- **Path spelling.** `HARNESS_STATE` is resolved (`/tmp` becomes `/private/tmp` on macOS). Pass
  profile paths as `$HARNESS_STATE/...`, or they are refused as outside the state.
- **macOS purges `/private/tmp`.** Keep long-lived state at the default under `.worktrees/`.
- **Never import `tldw_chatbook.config` outside `seedrun.sh`.** Config readers create and write
  the real `~/.config/tldw_cli`.
- **The `scale_seed.py` lore step stops** at its first duplicate generated book name (a
  `ConflictError` under `ERRORS`). `lore_fix.py` tops up to 43 books. That reproduces the gap-round
  dataset exactly (same 43 book names), so `make_profiles.sh --volume` checks counts instead of the
  error list.
- **Concurrent real-app use** (another session running the app on the real profile) makes the
  self-test's isolation check FAIL spuriously. Read the diff before blaming the harness.

## Manual isolation check

```bash
$PY $H/isolation_snap.py > "$HARNESS_STATE/real.before"   # before a session
$PY $H/isolation_snap.py | diff "$HARNESS_STATE/real.before" - && echo untouched
grep -c "$HOME/.config/tldw_cli\|$HOME/.local/share/tldw_cli" "$HARNESS_STATE"/runs/*/data/rp_review/tldw_cli_app.log  # expect 0
```

## Mock LLM

`mock_llm.py` replies `*(mock reply)* You said: "<last user msg>". Staying in character as <first
line of system prompt> …`. The Roleplay test chat makes non-streaming calls; Console sends use
`stream=False` or `True` depending on settings.

Start the mock before launching if you need replies. Otherwise readiness still says Ready, but
sends fail.

The fragment's `api_key` is the placeholder `rp-harness-mock-placeholder-not-a-secret`. It is not
in `config.py`'s `PROVIDER_API_KEY_PLACEHOLDERS`, so `resolve_provider_api_key()` accepts it. Never
put a real key in a harness fragment.

To use another port, or extra settings (for example `[console] exchange_capture = false`), copy
`mock_llm.toml` into `$HARNESS_STATE`, edit it, and pass `EXTRA_TOML=<copy>`.

## Not ported from the scratch harness

The following scratch files were not ported:

| Scratch file | Why |
|---|---|
| `ia_wait.py` | A fixed sleep; replaced by `waitfor.py`. |
| `seed_fix_persona.py` | A one-off repair; `seed.py` now creates "Terse Code Reviewer" itself. |
| `gapcheck/mock_llm_body.py` | Merged into `mock_llm.py --bodies`. |
| `mock_body*.toml` | Port and option variants of `mock_llm.toml`; see Mock LLM. |
| `gapcheck/inject/sitecustomize.py` | One-off Console-controller instrumentation for gap 1; load such files with `EXTRA_PYTHONPATH`. |
| `gap2/nav.sh`, `gap2/wheel.sh` | Covered by `drive_lib.sh` (`pal`, `wheel`). |
