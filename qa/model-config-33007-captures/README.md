# Phase 7 (TASK-33007) full-screen captures (after the review fixes, 2026-10-12)

Every capture in this directory was taken again on 2026-10-12, at `3b75942fc5`
(branch `feat/model-config-p7-settings-pm`, after the #3057 review fixes and fix
rounds 1-7). This is the controller's retake under ledger ruling C13, and it adds the
three surfaces ruling d-N1 asked for (`10a`-`10c`). Each moment was captured at 211x44,
then the tmux window was resized to 235x52, captured again, and resized back. `.txt` is
`tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

## Procedure

- **Isolation.** The app ran from this worktree in its own tmux server
  (`-L cap3057r`) under `env -i`, with `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` in a scratch directory, the null keyring,
  `PYTHONPATH` set to this worktree, `TERM=xterm-256color` and `COLORTERM=truecolor`.
  A `sandbox-exec` rule denied every read and write under `~/.config/tldw_cli` and
  `~/.local/share/tldw_cli`.
- **Profiles.** Four profiles were written from the shipped template (taken from a
  process whose own `HOME`/`XDG_*` were a second scratch directory). All four set
  `users_name = "verify_cap3057r"`, the splash off, `[model_catalog]
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
  asked twice, both by Discover. Nothing left the machine.
- **Opening the card.** Wait for the nav bar, then **F4**, **F6**, **Down**,
  **Enter**, **Esc** (Providers & Models). Keys came from `tmux send-keys`; clicks and
  wheel steps were SGR mouse sequences. A toast left by a revert was dismissed with a
  click before the next capture.
- **Nothing saved, no provider contacted.** No Test, no send, no Save. After each
  session the only config key that differed from its seeded profile was
  `console.rail_state`. No key appears in any capture. The tmux server was killed and
  the scratch directory deleted afterwards; the driver is not committed.

## Captures

OpenAI session 1 (01-07), in this order:

- `01-connect-openai-saved-key`: the card at rest. API key **saved in config**, "hidden · ctrl+l clears the saved key"; Env var `OPENAI_API_KEY` **not set**, "safer: a variable's name, never the key"; Key check "Ready · not tested".
- `02-default-model-picker-filtered`: **F6**, **Tab** x4, `mini` typed. Four matches under **Saved fallback**, the first highlighted, "4 found · Enter picks · Esc cancels".
- `03-applies-to-staged-model`: **Enter**, **Esc**. Model `o4-mini-2025-04-16` **edited \***, "1 unsaved", Applies to "… is unused and will use OpenAI · o4-mini-2025-04-16." on one row.
- `07-inspector-save-reach-staged`: nine wheel steps up over the Inspector. Applies to, Next new chat will use (still the saved pair, "T 0.6 · max 4096 · stream Off") and "Unsaved edits apply only after save (s)."
- `04-model-defaults-temperature-edited`: **r**, **Discard changes**, a click on Temperature, `0.4` typed. Temperature 0.4 **edited \*** with its whole help at both sizes; the card's line still reads "Provider settings reverted to last loaded values."
- `05a-advanced-closed`: **Esc**, **r**, **Discard changes**. Advanced with its six one-row closed titles; Context window reads "unknown, 32,000 assumed · enter the model's documented limit".
- `05b-advanced-context-window-open`: a click on the Context window title. One row: field, **not set**, "unknown, 32,000 assumed · budgets use it until set".
- `06a-console-behavior-replay-selects`: Console Behavior, then a click on "Reasoning replay override". **Default replay** and **This model's replay** are one-row Selects in one control column.
- `06b-console-behavior-rail-layout-scope`: wheel steps down until Rail presentation sits in the top half. **Trace viewer** and **Rail layout scope** in the same column.
- `06c-console-behavior-global-fallbacks`: wheel steps down until Global fallback defaults sits in the top half (scrolled again after the resize). Chat display name, the fallback rows with their Source words, "▶ Sampling · Top P 0.95 · Min P 0.05 · Top K 50", Background effects.

Anthropic and Azure sessions:

- `01b-connect-anthropic-sign-in-with`: **Sign in with** is one row ("API key", **built-in**, "bills API credits through your key"); Endpoint `https://api.anthropic.com` **built-in**.
- `01c-connect-azure-saved-key`: Endpoint placeholder "Enter your resource host", **not set**, "required: your resource host"; Key check "Not ready · no URL".

OpenAI session 2:

- `03b-applies-to-chat-with-work`: in the Console, **Ctrl+O**, Temperature 0.6 → 0.9, Apply to this chat (`\e[13;5u`); then the card, **F6**, **Tab** x4, `o3-mini`, **Enter**, **Esc**. Applies to "… open chat “Chat 1” keeps OpenAI · gpt-5.6-terra."

OpenAI session 3 (09, then 08):

- `09a-provider-filter-typed`: **F6**, `llama` typed. Cloud, Local, then legacy aliases, "6 found · Enter picks · Esc cancels", llama.cpp highlighted.
- `09b-provider-chosen-llama-cpp`: **Enter**. Provider llama.cpp **edited \***; API key **not required**; Endpoint `http://localhost:8080` "required: the server's base URL"; Key check "Not ready · no model"; Applies to "… will use llama.cpp · no model."
- `09c-provider-after-revert`: **Esc**, **r**, **Discard changes**. The saved OpenAI pair again, no list open.
- `08-chat-settings-context`: the Console, **Ctrl+O**. Model "Ready · not tested · context unknown"; Request estimate "0 / 32,000 tokens (assumed; window unknown)".
- `08b-chat-settings-context-view`: a click on **Context and memory**. Model window "unknown, 32,000 assumed" and the note pointing to F4 Settings.
- `08c-advanced-context-window`: **Esc**, **F4** (Settings reopens on Providers & Models). The closed title reads "Context window · unknown, 32,000 assumed · enter the model's documented limit".

llama.cpp session (ruling d-N1; TASK-33007.3 AC#11, TASK-33007.6 AC#9 and AC#12):

- `10b-saved-model-list-open-after-discovery`: a click on the Saved model list title, then on **Discover models**. Title "2 saved in config · 3 discovered, 2 not saved"; each row says in words whether it is selected and saved.
- `10a-default-model-picker-served-now`: **F6** until Provider has focus, **Tab** x4 (keyboard only). The open picker lists the two ids only the server holds under **Served now**, then **Saved fallback** with `qwen3-8b-q4  ● CURRENT` highlighted; "4 models available. Type to filter."
- `10c-catalog-refresh-open`: **Esc**, the Saved model list closed, a click on the Catalog refresh title. "Refresh on startup Off", "Refresh after (hours) │ 24", one row per provider; all 30 fit at 235x52, the last four are below the fold at 211x44.

## What looks wrong at full screen (2026-10-12)

1. The Inspector shows an orphan "(s)." line under "Local-only: saves write your config file." at 211x44 whenever a change is staged (`03`, `04`, `03b`, `09b`): "Unsaved edits apply only after save (s)." wraps there and the Inspector's scroll leaves only its tail in view.
2. A 16-cell Source word touches the help with one space, so "new-chat default 3 of 60 configured" and "Console Behavior inherits 0.6" read as one phrase (every Providers & Models capture).
3. The header reads "State: Draft — save with s" at rest while the Inspector says "No unsaved changes" (`01`, `05a`, `09c`).
4. "Provider settings reverted to last loaded values." stays on the card after the next edit, beside "1 unsaved" (`04`); one revert also says it twice, in different words (the card line and the "Settings category changes reverted." toast).
5. Catalog refresh says "Saved. Applies at next startup." with nothing changed, under a hint that says "applies immediately - no Save needed" (`10c`).
6. Discovery reports "Discovered 3 model(s) from Llama_cpp.": the config key, not "llama.cpp" (`10b`). Before discovery the disclosure says the same thing twice ("Discover models from configured endpoint" / "No discovered models yet. Use Discover models after endpoint is configured.").
7. `qwen3-8b-q4` is served and is the default, but it is listed only under Saved fallback, so Served now looks as if the server lacks the current model (`10a`).
8. For llama.cpp, which needs no key, the API key row still shows **Clear** and the Env var row still says "safer: a variable's name, never the key" (`09b`, `10a`).
9. Anthropic's Temperature placeholder reads "0.0 - 2.0"; Anthropic accepts 0 to 1 (`01b`).
10. Console Behavior: "Above composer", "Disabled" (Chat images) and "Edit summary prompt…" are buttons drawn as bold text on the card's own background, with no edge (`06b`, `06c`). Temperature's help is cut at 211 ("higher makes th…", `06c`); Streaming reads "On" beside "not set" (`06c`).
11. Chat settings (Ctrl+O): the modal stays about 150 columns wide at 235x52, so the Reasoning rows' help is cut at both sizes (`08`); its footer runs "Esc close Matches saved defaults" together. The Context view writes "F4 Settings > Providers & Models" where the Model view writes "Settings ▸ Providers & Models", and reads "Request overhead unknown tokens" (`08b`).
12. The Inspector's Owns list prints raw config keys split across two lines ("api_settings.<provider>." / "endpoint") at both sizes (`10b`).
13. The status line under the panes reads Home's description, "Open dashboard, notifications, and active work.", in Settings (`05a`, `08c`).
14. Seen live, not captured: the OpenAI picker opened with no filter says "Live catalog unavailable. Showing 14 configured models.", which reads as a fault when catalog refresh is simply off; Console Behavior's approval-summary Model placeholder is cut to "empty = the provider's defaul" at 211x44.
