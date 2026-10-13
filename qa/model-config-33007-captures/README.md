# Phase 7 (TASK-33007) full-screen captures (after capture fix 2, 2026-10-13)

Every capture in this directory was taken again on 2026-10-13, at `940965848f`
(branch `feat/model-config-p7-settings-pm`, after capture fix 2 and its two rounds).
It replaces the earlier 2026-10-13 set taken at `97e4af5beb`. `06d` now has a 235x52
capture too. Each moment was captured at 211x44, then the tmux window was resized to
235x52, captured again, and resized back. A resize moves the card's scroll position, so
`06c` and `06d` were scrolled again at 235x52 before that capture. `.txt` is
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
- **Loopback server (10a-10c only).** A stub on 127.0.0.1:18791 answered every GET with
  a three-model `/v1/models` listing: `qwen3-8b-q4`,
  `gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf` and `llama-4-scout-17b-16e.Q4_K_M.gguf`. Its
  log shows one request, from Discover. Nothing left the machine.
- **Opening the card.** Wait for the nav bar, then press **F4**, **F6**, **Down**,
  **Enter**, **Esc** (Providers & Models). Keys came from `tmux send-keys`; clicks and
  wheel steps were SGR mouse sequences. Before the next capture, a toast was dismissed
  with a click or left to time out.
- **Nothing saved, no provider contacted.** No Test, no send, no Save. After each
  session, the only config key that differed from its seeded profile was
  `console.rail_state`. No key appears in any capture. Afterwards the tmux server was
  killed and the scratch directory deleted. The driver is not committed.

## Captures

### OpenAI session 1, in this order

- `01-connect-openai-saved-key`: the card at rest.
  - Header: "State: No changes · edits save with s".
  - The Inspector opens at its top ("Scope Inspector").
  - The Focused field guide reads "Save: edits save with s · r reverts".
- `02a-default-model-picker-unfiltered`: **F6**, **Tab** x4, nothing typed. The picker
  opens on focus: "Catalog refresh is off. Showing 14 configured models.", **Saved
  fallback**, `gpt-5.6-terra  ● CURRENT`. **Custom ID** now draws an edge on both
  sides, and its label wraps over two rows ("Custom" / "ID").
- `02-default-model-picker-filtered`: `mini` typed. Four matches under **Saved
  fallback**, and "4 found · Enter picks · Esc cancels".
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
- `08-chat-settings-context`: the Console, then **Ctrl+O**.
  - The tab strip reads "● Model and generation  Context and memory", with no
    "Selected".
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
  - Status: "4 models available. Type to filter.".
- `10c-catalog-refresh-open`: a revert, the Saved model list closed, then a click on
  the Catalog refresh title.
  - "applies immediately · no Save needed" and "▐ ▌ Refresh on startup Off".
  - Each provider row reads "▐✓▌ <provider>: refresh On ▐ ▌ save to config Off".
  - All 30 rows fit at 235x52; the last four are below the fold at 211x44.

## Verification of capture findings round 2 (`.superpowers/sdd/p7-review-fix/capture-findings-2.md`)

Every finding is fixed (the numbers are the findings file's). These were checked live
rather than captured:

- **6:** discovery reported once, by the toast.
- **17:** the line above Chat settings' footer.
- **22:** Ctrl+Q with a staged Settings change asks "Quit and discard unsaved Settings
  changes?" with **Keep editing** / **Discard and quit**.
- **24:** Tab walks the Model input, then Custom ID (the list closes), then the Model
  defaults title, then Temperature. The results list is never a stop, and Down/Enter
  still picks from it.
- **25:** Anthropic's Alt+M refuses Temperature 1.5 with "Temperature must be a finite
  number from 0 to 1.".
- **26:** the spec amendment lines.

## New on the screens this wave changed

1. **Custom ID wraps** (`02a`, `02`, `10a`). The edge finding 10 added
   (`163446c1f9`: `border-left`/`border-right: tall` on `#settings-model-picker Button`)
   leaves `ModelSearchPicker #model-search-picker-custom`'s fixed `width: 12` too narrow
   for the label. "Custom ID" now prints over two rows ("Custom" / "ID"), and the picker
   grows by a row.
2. **A lowercase sentence after a full stop** (`08`). Chat settings' reasoning rows read
   "Support not verified for this model. the provider decides · …". Finding 12's new
   `BLANK_FIELD_HELP` ("the provider decides") follows the support note's full stop.
   The old "blank = provider default" there read as a formula, not a sentence.
