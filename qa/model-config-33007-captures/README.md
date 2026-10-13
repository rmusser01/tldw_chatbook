# Phase 7 (TASK-33007) full-screen captures (after the capture fixes, 2026-10-13)

Every capture in this directory was taken again on 2026-10-13, at `97e4af5beb`
(branch `feat/model-config-p7-settings-pm`, after the capture fix wave and its three
rounds). It replaces the 2026-10-12 set taken at `3b75942fc5`, and adds `02a` (the model
picker opened without typing) and `06d` (Console Behavior's approval-summary Model row,
211x44 only). Each moment was captured at 211x44, then the tmux window was resized to
235x52, captured again, and resized back. `.txt` is `tmux capture-pane -p`; `.ansi.txt`
is the same moment with `-e` colour escapes.

## Procedure

- **Isolation.** The app ran from this worktree in its own tmux server
  (`-L cap3057c`) under `env -i`, with `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` in a scratch directory, the null keyring,
  `PYTHONPATH` set to this worktree, `TERM=xterm-256color` and `COLORTERM=truecolor`.
  A `sandbox-exec` rule denied every read and write under `~/.config/tldw_cli` and
  `~/.local/share/tldw_cli`.
- **Profiles.** Four profiles were written from the shipped template (taken from a
  process whose own `HOME`/`XDG_*` were a second scratch directory). All four set
  `users_name = "verify_cap3057c"`, the splash off, `[model_catalog]
  auto_refresh_enabled = false`, `[first_run] setup_completed = true`,
  `[console.onboarding] first_send_completed = true`, and a made-up `api_key` in
  `[api_settings.openai]`, `[api_settings.anthropic]` and `[api_settings.azure]`. They
  differ in `[chat_defaults]`: OpenAI / `gpt-5.6-terra` (the template's pair),
  `anthropic` / `claude-sonnet-4-5`, `azure` / `my-gpt-deployment` (no base URL), and
  `llama_cpp` / `qwen3-8b-q4` with `api_url = "http://127.0.0.1:18791"` and
  `[providers] Llama_cpp = ["qwen3-8b-q4", "mistral-7b-instruct-q5"]`.
- **Loopback server (10a-10c only).** A stub on 127.0.0.1:18791 answered every GET with
  a three-model `/v1/models` listing: `qwen3-8b-q4`,
  `gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf`, `llama-4-scout-17b-16e.Q4_K_M.gguf`. It was
  asked once, by Discover. Nothing left the machine.
- **Opening the card.** Wait for the nav bar, then **F4**, **F6**, **Down**,
  **Enter**, **Esc** (Providers & Models). Keys came from `tmux send-keys`; clicks and
  wheel steps were SGR mouse sequences. A toast was dismissed with a click, or left to
  time out, before the next capture.
- **Nothing saved, no provider contacted.** No Test, no send, no Save. After each
  session the only config key that differed from its seeded profile was
  `console.rail_state`. No key appears in any capture. The tmux server was killed and
  the scratch directory deleted afterwards; the driver is not committed.

## Captures

OpenAI session 1, in this order:

- `01-connect-openai-saved-key`: the card at rest. Header "State: No changes · edits save with s". API key **saved in config**, "hidden · ctrl+l clears the saved key"; Env var `OPENAI_API_KEY` **not set**, "safer: a variable's name, never the key"; Key check "Ready · not tested". Two spaces now separate each Source word from its help.
- `02a-default-model-picker-unfiltered`: **F6**, **Tab** x4, nothing typed. The picker opens on focus: "Catalog refresh is off. Showing 14 configured models.", **Saved fallback**, `gpt-5.6-terra  ● CURRENT` highlighted.
- `02-default-model-picker-filtered`: `mini` typed. Four matches under **Saved fallback**, the first highlighted, "4 found · Enter picks · Esc cancels".
- `03-applies-to-staged-model`: **Enter**, **Esc**. Model `o4-mini-2025-04-16` **edited \***, "Draft — save with s · 1 unsaved", Applies to "… is unused and will use OpenAI · o4-mini-2025-04-16." The Inspector shows "Next new chat will use" (still the saved pair) and "Edits apply only after save (s)." on one line.
- `07-inspector-save-reach-staged`: nine wheel steps up over the Inspector. Applies to, Next new chat will use ("T 0.6 · max 4096 · stream Off") and "Edits apply only after save (s).".
- `04-model-defaults-temperature-edited`: **r**, **Discard changes** (one report: the toast "Settings category changes reverted."; no card line), a click on Temperature, `0.4` typed. Temperature 0.4 **edited \*** with its whole help at both sizes.
- `05a-advanced-closed`: **Esc**, **r**, **Discard changes**, the toast dismissed. Advanced with its six one-row closed titles; Context window reads "unknown, 32,000 assumed · enter the model's documented limit". The line under the panes is empty.
- `05b-advanced-context-window-open`: a click on the Context window title. One row: field, **not set**, "unknown, 32,000 assumed · budgets use it until set".
- `06a-console-behavior-replay-selects`: Console Behavior, then a click on "Reasoning replay override". **Default replay** and **This model's replay** are one-row Selects in one control column.
- `06b-console-behavior-rail-layout-scope`: wheel steps down until Rail presentation sits in the top half. **Trace viewer** and **Rail layout scope** in the same column; Status row is now a Select ("Above composer ▼").
- `06c-console-behavior-global-fallbacks`: wheel steps down until Global fallback defaults sits in the top half (scrolled again after the resize). Temperature 0.6 with its whole help, Streaming "On · built-in · used unless the provider sets its own", "▶ Sampling · Top P 0.95 · Min P 0.05 · Top K 50", the summary prompt button "Edit in Internal Prompts…", Background effects.
- `06d-console-behavior-approval-summary-model` (211x44 only): wheel steps up to Permission summaries. Summary mode Off; Provider "e.g. OpenAI"; Model "empty = the provider's default model", whole.

Anthropic and Azure sessions:

- `01b-connect-anthropic-sign-in-with`: **Sign in with** is one row ("API key", **built-in**, "bills API credits through your key"); Endpoint `https://api.anthropic.com` **built-in**; Temperature placeholder "0.0 - 1.0".
- `01c-connect-azure-saved-key`: Endpoint placeholder "Enter your resource host", **not set**, "required: your resource host"; Key check "Not ready · no URL".

OpenAI session 2:

- `03b-applies-to-chat-with-work`: in the Console, **Ctrl+O**, Temperature 0.6 → 0.9, Apply to this chat (`\e[13;5u`); then the card, **F6**, **Tab** x4, `o3-mini`, **Enter**, **Esc**. Applies to "… open chat “Chat 1” keeps OpenAI · gpt-5.6-terra."

OpenAI session 3 (09, then 08; between them F2 opened the Lab tab by mistake and a click on Console left it, nothing changed):

- `09a-provider-filter-typed`: **F6**, `llama` typed. Cloud, Local, then legacy aliases, "6 found · Enter picks · Esc cancels", llama.cpp highlighted.
- `09b-provider-chosen-llama-cpp`: **Enter**. Provider llama.cpp **edited \***, "2 unsaved"; API key "No credential required", **not required**, "this provider needs no key", no Clear; Env var "optional · a variable's name, not a key"; Endpoint `http://localhost:8080` **saved in config** "required: the server's base URL"; Key check "Not ready · no model"; Applies to "… will use llama.cpp · no model."
- `09c-provider-after-revert`: **Esc**, **r**, **Discard changes**, the toast dismissed. The saved OpenAI pair again, no list open.
- `08-chat-settings-context`: the Console, **Ctrl+O**. The box is 176 columns at both sizes; the Reasoning rows' help is whole; footer "Esc close  Matches saved defaults   Default for new chats (Ctrl+N)   Apply to this chat (Ctrl+Enter)"; Request estimate "0 / 32,000 tokens (assumed; window unknown)".
- `08b-chat-settings-context-view`: a click on **Context and memory**. Model window "unknown, 32,000 assumed", the note "Enter the model's documented limit in Settings ▸ Providers & Models (F4)", "Request overhead unknown".
- `08c-advanced-context-window`: **Esc**, **F4** (Settings reopens on Providers & Models). The closed title reads "Context window · unknown, 32,000 assumed · enter the model's documented limit"; the line under the panes is empty.

llama.cpp session (ruling d-N1; TASK-33007.3 AC#11, TASK-33007.6 AC#9 and AC#12):

- `10b-saved-model-list-open-after-discovery`: a click on the Saved model list title ("No models discovered yet. Discover models lists what the configured endpoint serves."), then on **Discover models**. "Discovered 3 models from llama.cpp."; title "2 saved in config · 3 discovered, 2 not saved"; each row says in words whether it is selected and saved.
- `10a-default-model-picker-served-now`: **F6** until Provider has focus, **Tab** x4 (keyboard only). **Served now** lists `qwen3-8b-q4  ● CURRENT` (highlighted) and the two ids only the server holds; **Saved fallback** lists `mistral-7b-instruct-q5`; "4 models available. Type to filter."
- `10c-catalog-refresh-open`: **Esc**, the Saved model list closed, a click on the Catalog refresh title. "Refresh on startup Off", "Refresh after (hours) │ 24", one row per provider and no "Saved." line; all 30 fit at 235x52, the last four are below the fold at 211x44.

## What looks wrong at full screen (2026-10-13)

1. The Inspector keeps its scroll position, so a section's tail shows with no heading: "there with Alt+M" and "Model defaults: chats that switch to this model pick them up" under "Local-only…" (`05a`, `04` at 235x52), "Focused setting: None — Tab to a setting" without "Focused field guide" (`05a`, `08c`, `09c` at 211x44). Only "▼ more" is shown; nothing says there is more above.
2. At 211x44 a focused field scrolls "Applies to" and "Next new chat will use" out of the Inspector (`03`, `04`, `03b`, `09b`); they come back only by wheeling up (`07`).
3. The Inspector says "Save: staged - press s to save, r to revert" when nothing is staged (`01`, `05a`, `09c`, `10a`-`10c`).
4. A checkbox always draws `▐X▌`, so the mark contradicts the words beside it: "▐X▌ not selected · qwen3-8b-q4" (`10b`), "▐X▌ Refresh on startup Off" and "▐X▌ save to config Off" on every provider row (`10c`), "▐X▌ Stack collapsed rail labels" over "Saved style: Horizontal" (`06b`).
5. The Saved model list title says "2 saved in config", but the open list shows only the saved model the server serves; `mistral-7b-instruct-q5` appears only in the picker (`10b`, `10a`).
6. Discovery reports twice: the card line "Discovered 3 models from llama.cpp." and the toast "Provider model discovery finished." (seen live; the toast had gone by `10b`).
7. llama.cpp: Endpoint holds a URL and still says "required: the server's base URL", which reads as an error; choosing one provider counts as "2 unsaved" (`09b`).
8. The Inspector's validation says "use Manual for custom aliases"; the picker calls that action "Enter provider ID", under "Other" (`09b`). In `09a` the list ends on the "Other" heading with its only row below the fold at both sizes.
9. Azure: Key check says "no URL" where the Endpoint row says "your resource host" (`01c`).
10. "Custom ID", "Discover models" and "Edit in Internal Prompts…" are flat blocks one shade darker than the card (RGB 30,30,30 on 36,47,56), with no edge (`02a`, `10b`, `06c`).
11. Console Behavior's Inspector prints raw config section names: "Scope: [chat_defaults] response fallbacks and [console] paste display settings" (`06c` at 235x52).
12. The fallback rows and the Chat settings box show "default" in the Select while the help says "blank = provider default" (`06c`, `08`). Catalog refresh and Permission summaries say "applies immediately - no Save needed" with an ASCII hyphen; the closed titles say "applies immediately" (`10c`, `06d`).
13. Reasoning replay override says "Select a local model in Console to set a remembered override" over a dimmed "This model's replay" and "Native tool support" that say "Remembered for this normalized endpoint and model" with no model named (`06a`).
14. Approval summary: Provider is free text ("e.g. OpenAI") with no picker and no hint which ids are valid (`06d`).
15. Chat settings: the tab strip prints the word "Selected" ("Model and generation · Selected  Context and memory", `08`, `08b`). Once a field is edited, an unlabelled right-aligned line "Used by future conversations for OpenAI." appears above the buttons (seen live, not captured).
16. Chat settings ▸ Context and memory: "No conversation memory. Transcript remains authoritative." is followed by "No conversation memory is active on this branch."; "Reset overrides" and "Reset all conversation memory…" share one shaded bar with no gap; the footer has both "Esc close" and "Cancel"; the box grows to 42 rows and covers the nav bar at 211x44 (`08b`; the buttons are below the fold at 211x44).
17. Ctrl+Q quits at once with a staged Settings change ("1 unsaved"): no prompt, the draft is lost (seen live in two sessions, not captured; may predate Phase 7).
18. After a click, the line under the panes shows the hovered control's description ("Query the configured OpenAI-compatible provider endpoint for available models." in `10b`, under a llama.cpp card): a hover leftover, not Home's text, which no longer appears.
