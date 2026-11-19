# Phase 7 (TASK-33007) full-screen captures (after the post-rebase fix, 2026-10-13)

The post-rebase fix and its round 1 (`f33ed7b121..5068e68fe6`) changed no screen that an
earlier capture shows. They changed what Settings keeps when it is left (unsaved
Prompt-cache snapshot preferences) and what a leave or a quit waits for (a snapshot Save
still writing). So nothing was retaken. Three new captures were taken on 2026-10-13 at
`5068e68fe6`: `13a`, `13b` and `13c` (see "Snapshot preferences session" below).

Before that, the captures of the screens capture fix 3 changed were taken again on 2026-10-13, at
`a036d5db41` (branch `feat/model-config-p7-settings-pm`, after capture fix 3 and its
five rounds): `02a`, `02`, `08`, `10a`, `10b`, and the new `10d`, `10e`, `11`, `12a`,
`12b` and `12c`. `10b` came out byte-identical to the earlier one. Every other capture
is from the set taken earlier on 2026-10-13 at `940965848f` (after capture fix 2),
which capture fix 3 did not change.

Each moment was captured at 211x44, then the tmux window was resized to 235x52,
captured again, and resized back. A resize moves the card's scroll position, so `06c`
and `06d` were scrolled again at 235x52 before that capture. `.txt` is
`tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

## Procedure

- **Isolation.** The app ran from this worktree in its own tmux server
  (`-L cap3057v`) under `env -i`. `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` pointed into a scratch directory. It used the
  null keyring, `PYTHONPATH` set to this worktree, `TERM=xterm-256color` and
  `COLORTERM=truecolor`. A `sandbox-exec` rule denied every read and write under
  `~/.config/tldw_cli` and `~/.local/share/tldw_cli`.
- **Profiles.** Four profiles were written from the shipped template
  (`config.CONFIG_TOML_CONTENT`, read by a process whose own `HOME`/`XDG_*` were a
  second scratch directory). All four set:
  - `users_name = "verify_cap3057v"`;
  - the splash off, `[model_catalog] auto_refresh_enabled = false`,
    `[first_run] setup_completed = true` and
    `[console.onboarding] first_send_completed = true`;
  - a made-up `api_key` in `[api_settings.openai]`, `[api_settings.anthropic]` and
    `[api_settings.azure]`.

  They differ in `[chat_defaults]`:
  - OpenAI / `gpt-5.6-terra` (the template's pair);
  - `anthropic` / `claude-sonnet-4-5`;
  - `azure` / `my-gpt-deployment`, with no base URL;
  - `llama_cpp` / `qwen3-8b-q4`, with `api_url = "http://127.0.0.1:18791"` and
    `[providers] Llama_cpp = ["qwen3-8b-q4", "mistral-7b-instruct-q5"]`.

  The `a036d5db41` run built the OpenAI and llama.cpp profiles the same way, with
  `users_name = "verify_cap3057v3"`, and a fifth: `Groq` /
  `llama-3.3-70b-versatile` with no Groq key anywhere (`12a`-`12c`).

  The `5068e68fe6` run built the OpenAI profile the same way, with
  `users_name = "verify_cap3057v6"` (`13a`-`13c`). `TLDW_CONFIG_PATH` pointed at that
  profile file. Its `[llamacpp_snapshots]` started at the template's `enabled = false`,
  `keep_count = 10`.
- **Loopback server (10a-10e only).** A stub on 127.0.0.1:18791 answered every GET with
  a three-model `/v1/models` listing: `qwen3-8b-q4`,
  `gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf` and `llama-4-scout-17b-16e.Q4_K_M.gguf`. Its
  log shows one request, from Discover. For `10e` the stub was stopped, so the second
  Discover was refused on loopback. Nothing left the machine.
- **Opening the card.** Wait for the nav bar, then press **F4**, **F6**, **Down**,
  **Enter**, **Esc** (Providers & Models). Keys came from `tmux send-keys`; clicks and
  wheel steps were SGR mouse sequences. Before the next capture, a toast was dismissed
  with a click or left to time out.
- **No provider contacted.** No Test and no send. In the `940965848f` run nothing was
  saved: after each session the only config key that differed from its seeded profile
  was `console.rail_state`. In the `a036d5db41` run the OpenAI session saved nothing
  (the same single difference). The llama.cpp session's **Save selected** added one id
  to `[providers] Llama_cpp`. The Groq session saved OpenAI as the default, then,
  on the routed page, a made-up Groq key with the linked pair (Groq became the default
  again); both writes went to its scratch profile. In the `5068e68fe6` run the only
  writes were snapshot preference Saves (enabled `true` with keep count 25, then keep
  count 30, 35 and 36), all to its scratch profile. No key
  appears in any capture. Afterwards the tmux server was killed and the scratch
  directory deleted. The driver is not committed.

## Captures

### OpenAI session 1, in this order

- `01-connect-openai-saved-key`: the card at rest.
  - Header: "State: No changes · edits save with s".
  - The Inspector opens at its top ("Scope Inspector").
  - The Focused field guide reads "Save: edits save with s · r reverts".
- `02a-default-model-picker-unfiltered` (`a036d5db41`): **F6**, **Tab** x4, nothing
  typed. The picker opens on focus: "Catalog refresh is off. Showing 14 configured
  models.", **Saved fallback**, `gpt-5.6-terra  ● CURRENT`. **Custom ID** draws an edge
  on both sides and reads on one row ("▊ Custom ID  ▎"); the picker is one row shorter
  than at `940965848f`, where the label wrapped over two.
- `02-default-model-picker-filtered` (`a036d5db41`): `mini` typed. Four matches under
  **Saved fallback**, "4 found · Enter picks · Esc cancels", and **Custom ID** on one
  row.
- `03-applies-to-staged-model`: **Enter**, **Esc**.
  - Model `o4-mini-2025-04-16` reads **edited \***; the header reads "Draft — save with
    s · 1 unsaved".
  - At 211x44 the Inspector still shows Applies to and "Next new chat will use", with
    "Unsaved: Model. Edits apply only after save (s)."
- `07-inspector-save-reach-staged`: nine wheel steps up over the Inspector. It was
  already at its top, so the capture equals `03`.
- `04-model-defaults-temperature-edited`:
  - **r**, then **Discard changes**: one report, the toast "Settings category changes
    reverted.".
  - The toast was dismissed, Temperature clicked and `0.4` typed: Temperature 0.4
    **edited \*** with its whole help.
  - The Inspector stays at its top, with "Unsaved: Temperature.".
- `05a-advanced-closed`: **Esc**, **r**, **Discard changes**, the toast dismissed.
  - Advanced shows six one-row closed titles.
  - Context window reads "unknown, 32,000 assumed · enter the model's documented
    limit".
  - The Inspector starts at its heading.
- `05b-advanced-context-window-open`: a click on the Context window title. One row:
  field, **not set**, "unknown, 32,000 assumed · budgets use it until set".
- `06a-console-behavior-replay-selects`: Console Behavior, then a click on "Reasoning
  replay override". With no local model selected, the disclosure shows only "Select a
  local model in Console to set a remembered override." (no dimmed controls).
  Checkboxes draw `▐✓▌` when on and `▐ ▌` when off.
- `06b-console-behavior-rail-layout-scope`: wheel steps down until Rail presentation
  sits in the top half. "▐ ▌ Stack collapsed rail labels" over "Saved style:
  Horizontal".
- `06d-console-behavior-approval-summary-model`: Permission summaries reads "applies
  immediately · no Save needed". Its rows are Summary mode Off, Provider as a Select
  ("not set ▼") taking the rest of its row, and Model "empty = the provider's default
  model".
- `06c-console-behavior-global-fallbacks`: scrolled until Global fallback defaults sits
  in the top half.
  - Blank rows read **provider** "the provider decides"; the reasoning Selects show
    "default".
  - "Summary prompt" shows "▊ Edit in Internal Prompts… ▎" with an edge.
  - At 235x52 the Inspector's Override rules read "Scope: Global fallback defaults and
    paste collapse".

### Anthropic and Azure sessions

- `01b-connect-anthropic-sign-in-with`: **Sign in with** is one row ("API key",
  **built-in**, "bills API credits through your key"). Temperature's placeholder is
  "0.0 - 1.0".
- `01c-connect-azure-saved-key`: Endpoint "Enter your resource host", **not set**,
  "required: your resource host"; Key check "Not ready · no endpoint".

### OpenAI session 2

- `03b-applies-to-chat-with-work`:
  - In the Console: **Ctrl+O**, Temperature 0.6 → 0.9, Apply to this chat
    (`\e[13;5u`). After the edit, the line above the footer read "Save as model
    default: future chats on this OpenAI model use these values.".
  - Then in the card: **F6**, **Tab** x4, `o3-mini`, **Enter**, **Esc**. Applies to
    reads "… open chat “Chat 1” keeps OpenAI · gpt-5.6-terra."

### OpenAI session 3

- `09a-provider-filter-typed`: **F6**, `llama` typed.
  - The list shows Cloud, Local, then "Custom & legacy aliases", ending on its row
    "Enter provider ID", whole at both sizes.
  - The Inspector reads "choose one from the list, or its last row, Enter provider ID,
    for any other".
- `09b-provider-chosen-llama-cpp`: **Enter**.
  - Provider llama.cpp **edited \***; the header reads "2 unsaved"; the Inspector reads
    "Unsaved: Provider, Model.".
  - Endpoint `http://localhost:8080` **saved in config**, "the server's base URL".
  - Key check "Not ready · no model".
- `09c-provider-after-revert`: **Esc**, **r**, **Discard changes**, the toast
  dismissed. The saved OpenAI pair is back, and the Inspector is at its top.
- `08-chat-settings-context` (`a036d5db41`, OpenAI session 1, after `02`): the Console,
  then **Ctrl+O**.
  - The tab strip reads "● Model and generation  Context and memory", with no
    "Selected".
  - The three reasoning rows read "Support not verified for this model. The provider
    decides · …", a capital after the full stop.
  - The box is 176 columns at both sizes.
  - Footer: "Esc close  Matches saved defaults   Default for new chats (Ctrl+N)   Apply
    to this chat (Ctrl+Enter)".
- `08b-chat-settings-context-view`: a click on **Context and memory**.
  - At 211x44 the box spans rows 4-41, so the nav bar stays visible. The footer reads
    "Esc close  Apply to this chat (Ctrl+Enter)", with no Cancel.
  - At 235x52, "Current memory" shows one sentence: "No conversation memory. Transcript
    remains authoritative.".
- `08c-advanced-context-window`: **Esc**, then **F4** (Settings reopens on Providers &
  Models). The closed title reads "Context window · unknown, 32,000 assumed · enter
  the model's documented limit". The Inspector is at its top.

### llama.cpp session (ruling d-N1; TASK-33007.3 AC#11, TASK-33007.6 AC#9 and AC#12)

`10a`, `10b`, `10d` and `10e` are from the `a036d5db41` run, `10c` from `940965848f`.

- `10b-saved-model-list-open-after-discovery`: a click on the Saved model list title,
  then on **Discover models**.
  - Discovery reported once: the toast "Discovered 3 models from llama.cpp.". There was
    no card line and no second toast; the toast had gone by the capture.
  - Title: "2 saved in config · 3 discovered, 2 not saved".
  - Each row says in words whether it is selected and saved. Its off box paints the "X"
    in the box's own colour (119,119,119 on 119,119,119), so it shows empty in the
    `.ansi.txt`.
  - Under the list: "Saved, not served now: mistral-7b-instruct-q5".
  - Discover models, Save selected and Clear each draw an edge.
- `10a-default-model-picker-served-now`: **F6** until Provider has focus, then **Tab**
  x4 (keyboard only).
  - **Served now** lists `qwen3-8b-q4  ● CURRENT` and the two ids only the server holds.
  - **Saved fallback** lists `mistral-7b-instruct-q5`.
  - Status: "4 models available. Type to filter.". **Custom ID** reads on one row.
- `10d-save-selected-receipt`: **Esc** twice, a click on the
  `gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf` row (it reads **selected**), then a click on
  **Save selected**.
  - One report, on the line under the title: "Saved 1 discovered model to llama.cpp."
    (display name, singular). No toast.
  - Title: "3 saved in config · 3 discovered, 1 not saved · 1 selected"; the row now
    reads "saved".
  - Not captured: a second **Save selected** read "The selected models are already
    saved for llama.cpp." on the same line, again with no toast.
- `10e-failed-discovery-one-report`: the stub stopped, then a click on **Discover
  models**. One report, on the line: "Model discovery request failed. Check the
  endpoint URL, server availability, and credentials." No toast appeared in the six
  seconds after the click.
- `10c-catalog-refresh-open`: a revert, the Saved model list closed, then a click on
  the Catalog refresh title.
  - "applies immediately · no Save needed" and "▐ ▌ Refresh on startup Off".
  - Each provider row reads "▐✓▌ <provider>: refresh On ▐ ▌ save to config Off".
  - All 30 rows fit at 235x52; the last four are below the fold at 211x44.

### OpenAI session 1 at `a036d5db41`, after `08` (ruling C-N1)

- `11-quit-from-console-kept-settings-draft`: **Esc**, **F4**, Temperature clicked and
  `0.4` typed, **Esc** (the header read "1 unsaved", the rail "Providers & Models
  \*"), a click on Console in the nav bar, then **Ctrl+Q**. The Console asks "Quit and
  discard unsaved Settings changes?": "These Settings have unsaved changes: Providers &
  Models." with **Keep editing** and **Discard and quit**.
  - Not captured: **Keep editing** went to Settings with the draft (Temperature 0.4
    **edited \***, "1 unsaved"). **Ctrl+Q** there asked once. From the Console with
    Chat settings open, **Ctrl+Q** asked the same and **Keep editing** stayed in Chat
    settings. Then **Esc**, **Ctrl+Q**, **Discard and quit** quit the app, with the
    draft unsaved.

### Groq session at `a036d5db41` (a routed page that is not the default provider's)

Before the captures:
- The chat was renamed "Groq chat" (a click on its active tab), so it keeps its pair.
  (**Apply to this chat** is refused while Groq has no key.)
- Settings saved OpenAI · `gpt-5.6-terra` as the default; Applies to read "open chat
  “Groq chat” keeps Groq · llama-3.3-70b-versatile.".
- Temperature `0.4` was staged and Settings left.
- In the Console: **Ctrl+O**. Chat settings showed "Not ready · no key" and
  **Configure credential…**, which was clicked.

- `12a-routed-provider-conflict-box`: Settings opens on the OpenAI draft with the
  conflict box under Applies to. It holds "Unsaved changes to Temperature belong to
  another provider. …", then **Review existing changes**, **Discard changes and
  configure Groq** and **Return to Chat settings**. Its last row is followed directly by
  "▼ Model defaults" at both sizes.
- `12b-routed-provider-needs-setup-arrival`: a click on **Discard changes and configure
  Groq**, the "Settings category changes reverted." toast left to time out.
  - Provider Groq and Model `llama-3.3-70b-versatile` read **not the default**.
  - API key has focus: "missing", "paste one, or set GROQ_API_KEY"; Key check "Not
    ready · no key".
  - No return box on arrival: on this branch it shows only after a save.
- `12c-routed-provider-return-box-after-save`: a made-up key typed, **Esc**, **s**, the
  save toast left to time out.
  - Under Applies to: "Credential saved. Return to Chat settings to check readiness;
    provider acceptance is not yet verified.", then **Return to Chat settings** and
    **Stay in Settings**.
  - "▼ Model defaults" follows on the next row at both sizes. Save wrote the linked
    pair, so both rows now read **new-chat default**.

### Snapshot preferences session at `5068e68fe6` (ruling cf3r1-N1 (b))

The OpenAI profile. Settings ▸ Providers & Models was opened as above. Then a click on the
**Prompt-cache snapshots** title, a click on **Enable snapshots**, and the keep count
changed from `10` to `25`. The header read "2 unsaved", and the Inspector read "Unsaved:
Prompt-cache snapshots, Snapshot keep count.".

- `13a-snapshot-edit-kept-after-leaving`: a click on Console in the nav bar, then **F4**.
  - The edit is kept and marked unsaved. The header reads "Draft — save with s · 2
    unsaved", and the rail reads "Providers & Models \*".
  - The closed title reads "Prompt-cache snapshots · llama.cpp only · On · keep 25".
  - The Inspector reads "Unsaved: Prompt-cache snapshots, Snapshot keep count.".
  - Not captured: opening the title showed **Enable snapshots** checked and `25` in the
    field.
- `13b-quit-from-console-names-snapshot-edit`: a click on Console, then **Ctrl+Q**. The
  Console asks "Quit and discard unsaved Settings changes?", with "These Settings have
  unsaved changes: Providers & Models.".
  - Not captured: **Keep editing** went to Settings with the edit ("2 unsaved", "keep
    25").
- `13c-reverted-snapshot-edit-back-after-leaving` (a defect; see "New on the screens this
  wave changed").
  - Before it, in the same session, after **Keep editing**:
    - **s** and a click on Console: "Snapshot preferences saved.", and the config held
      keep 25. **F4** read "2 unsaved · keep 25" again.
    - **r**, **Discard changes** ("No changes"), keep `30` typed, **s**, and a click on
      Console 0.4 s later: "Snapshot preferences saved.", and the config held 30.
      **F4** read "2 unsaved", "On · keep 25" again. **s** there was refused with the
      toast "Snapshot preferences changed elsewhere; Revert before Save.".
    - **r**, **Discard changes**: "No changes", "On · keep 30".
  - The capture: a click on Console, then **F4**. Settings again reads "2 unsaved",
    "On · keep 25" and "Unsaved: Prompt-cache snapshots, Snapshot keep count.", while
    the config holds keep 30. It is byte-identical to `13a` at both sizes.
- Not captured, in a fresh session (nothing kept before):
  - Keep `35` typed, then **s** and a click on Console sent together. The toast read
    "Snapshot preferences saved.", the config held 35, and **F4** showed "No changes ·
    keep 35".
  - Keep `36` typed, then **s** and **Ctrl+Q** 0.1 s apart. The app quit with no prompt,
    and the config held 36.
  - When **s** and **Ctrl+Q** were sent in one write, the quit prompt opened first and
    the **s** never reached Settings. That is the order the two keys were read in, not
    the wait this wave added.

## Verification of findings round 4 (`.superpowers/sdd/p7-review-fix/capture-findings-4.md`)

1. **Leaving Settings keeps unsaved Prompt-cache snapshot preferences** (ruling cf3r1-N1
   (b)): fixed for one leave. The edit comes back marked unsaved (`13a`). **Ctrl+Q** from
   the Console names Providers & Models (`13b`), and **Keep editing** returns to it. A
   Save followed at once by a leave or a quit settles first (the fresh-session checks
   above). The kept pair is not dropped once it stops being unsaved, though: see `13c`
   below.
2. **A Save selected that raises after Discover or Clear took the line reports its
   failure**: this wave did not change it (fixed at `f33ed7b121`). It cannot be shown
   live without making the save raise. Its pinned node,
   `Tests/UI/test_settings_save_selected_not_cancelled.py::test_save_selected_raised_after_another_action_names_a_true_next_step`,
   passed 5/5 through the scratch runner at `5068e68fe6`.

## New on the screens this wave changed

- **A kept snapshot edit outlives its Save or Revert** (`13c`). Once a leave has kept the
  pair, every later return to Settings restores it, even after it was saved or reverted.
  The screen then shows an edit nobody has pending, against a stale baseline, and Save
  refuses it as "changed elsewhere".
  - Cause: `BaseAppScreen.save_state` returns `self.state_data`, and `restore_state` set
    that to the restored snapshot. So `SettingsScreen.save_state` starts from a dict that
    still holds `snapshot_preferences`. `keep_snapshot_preferences` writes that key only
    when the pair is unsaved, and never removes it.
- Nothing else. `11` would read the same: a kept snapshot edit is named as a kept
  Temperature edit is.
