# Phase 8 (TASK-33008) full-screen captures

Live captures of the surfaces Phase 8 changed, taken 2026-10-12 from this branch at
`00c0927db6`, keyboard only. Each moment was captured at 211x44, the tmux window
was resized to 235x52 and captured again, then resized back. `.txt` is
`tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes
(focus shows there: a focused button is bold and underlined on the accent colour).

## Key counts

- **Parent AC#1, local server: 2 key presses**, measured at 211x44. From the card
  with "Use detected llama.cpp (127.0.0.1:8080)" focused (`02a`): **Enter** opens
  Switch model on llama.cpp (`02b`), **Enter** applies the highlighted model. The
  card is gone and the composer has focus (`02c`; a typed character landed in the
  composer and was deleted). Saved: `chat_defaults` llama_cpp · qwen3-8b-instruct-q4_k_m
  and `api_settings.llama_cpp.api_url = "http://127.0.0.1:8080"`.
- **Parent AC#2, cloud key: 6 key actions** (budget 12), measured at 211x44, the
  key check and the return included. From the card (`03a`): **Enter** on Set up
  provider (Settings opens on OpenAI with the API key field focused, `03b`), one
  **paste** of the key (`03c`), **Esc**, **s** (saved, focus on Return to Console,
  `03d`), **t** (key check: "Ready · verified", focus stays on Return to Console,
  `03e`), **Enter** (back in "Chat 1", card gone, composer focused, `03f`). No key
  navigated back to Console by hand.

## Captures

| File prefix | What it shows |
|---|---|
| `01a-card-at-rest-set-up-provider-focused` | Get started card, no provider, no server found: plain dim backdrop, Tab stop 1 (Set up provider). |
| `01b-card-tab-write-a-note-focused` | Same card after **Tab**: Tab stop 2 (Write a note in Library); one more **Tab** wraps to Set up provider. |
| `02a-card-detected-local-server-focused` | Card with a loopback server found: "Use detected llama.cpp (127.0.0.1:8080)" added and focused. |
| `02b-switch-model-from-card-filtered-to-llama-cpp` | AC#1 key 1: Switch model with Find "llama.cpp", the server's three models, the first highlighted, "Applies to: this chat · also the default for new chats". |
| `02c-console-ready-after-enter-enter` | AC#1 key 2: card gone, "Setup complete — llama.cpp · qwen3-8b-instruct-q4_k_m", composer focused. |
| `03a-cloud-card-set-up-provider` | AC#2 start: card on OpenAI with no key ("Not ready · no key"). |
| `03b-cloud-settings-api-key-focused-return-to-console` | AC#2 key 1: Settings ▸ Providers & Models on OpenAI, API key field focused, Return to Console shown on arrival. |
| `03c-cloud-key-pasted-return-without-saving` | AC#2 key 2: key pasted (masked, "edited *"); Return to Console replaced by Return without saving. |
| `03d-cloud-saved-focus-on-return-to-console` | AC#2 keys 3-4 (**Esc**, **s**): "Credential saved. Return to Console to continue in “Chat 1”…", focus on Return to Console. |
| `03e-cloud-key-check-verified` | AC#2 key 5 (**t**): Key check "Ready · verified", "key accepted (3 models listed)" in the inspector, focus still on Return to Console. |
| `03f-cloud-back-in-console-composer-ready` | AC#2 key 6 (**Enter**): back in "Chat 1", "Ready · verified" badge, composer focused. |
| `04a-return-without-saving-focused-key-unsaved` | A key pasted and not saved; Return without saving focused (6 **Tab**s from the API key field). |
| `04b-return-without-saving-confirmation` | **Enter** on it: the discard confirmation, Keep editing focused by default. |
| `04c-discard-returns-to-card` | **Tab**, **Enter** on Discard changes: back on the Get started card, Set up provider focused, nothing saved. |
| `05a-switch-model-needs-setup-row-deepseek` | Configured Console (OpenAI ready), **Alt+M**, "deepseek" typed, **Down** ×3 from Find's best match (Command Code's row) to DeepSeek's own NEEDS SETUP row. |
| `05b-needs-setup-route-settings-deepseek` | **Enter** on that row: Settings on DeepSeek, API key focused, model deepseek-v4-flash, Return to Console shown. |
| `05c-reopened-switch-model-after-save-deepseek` | Paste, **Esc**, **s**, **Enter** on Return to Console: Switch model reopened with Find "DeepSeek", DeepSeek's rows "Ready · not tested", deepseek-v4-flash highlighted and ● CURRENT (the unused chat followed the save). |
| `06a-wizard-model-step-authentication-failed` | First-run wizard, llama.cpp, Model step: the authentication-failed status as plain text over the model-ID field. |

## Procedure

- **Isolation.** The app ran from this worktree (`PYTHONPATH` = the worktree) in its
  own tmux server (`-L cap33008`) under `env -i`, with `HOME`, `XDG_CONFIG_HOME`,
  `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` in a fresh scratch
  directory, the null keyring backend, `TERM=xterm-256color`, `COLORTERM=truecolor`,
  and a `sandbox-exec` profile denying writes to the real profile directories.
- **Profiles.** Each one is the shipped template (`CONFIG_TOML_CONTENT`, dumped by a
  process whose own `HOME`/`XDG_*` were scratch) with `users_name =
  "verify_cap33008_<profile>"`, the splash off, `[model_catalog]
  auto_refresh_enabled = false` and `[first_run] setup_completed = true`. No provider
  key, except: `05` adds a made-up OpenAI key and `[console.onboarding]
  first_send_completed = true`; `06` keeps the template's first-run state, so the
  wizard opens.
- **Stubs.** A loopback HTTP server answering `GET /v1/models`: on 127.0.0.1:8080
  with three model ids for `02`; on 127.0.0.1:8080 answering 401 for `06` (no key was
  sent); on 127.0.0.1:18431 for `03`-`05`, where `[api_settings.openai] api_base_url`
  points at it so the key check (`t`) lists models without contacting OpenAI.
  That is why the Endpoint row reads `http://127.0.0.1:18431/v1` in `03` and `04`.
- **Input.** `tmux send-keys`; the key in `03c`, `04a` and `05c` was pasted with
  `tmux paste-buffer -p` (bracketed paste, one action). Keys are made-up
  placeholders and never appear in a capture.
- **No key check or send reached a real provider.** No message was sent. The only
  key check went to the 18431 stub (one authenticated `GET /v1/models`).
