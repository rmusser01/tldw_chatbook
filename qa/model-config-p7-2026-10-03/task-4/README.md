# TASK-33007.4: what a Providers & Models save reaches (2026-10-04)

Keyboard-and-click captures of Settings ▸ Providers & Models at 211x44 with the real
stylesheet, while the open Console chat holds work (one sent message and its reply).

## Procedure

- The real app ran in an isolated tmux server at 211x44 under `env -i`. `HOME`,
  `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` pointed into a scratch
  directory, and the keyring backend was the null one.
- The scratch profile (`users_name = "verify_p7t4_live"`) was written from the shipped
  template by a process whose own `HOME`/`XDG_*` pointed at a second scratch directory.
  It set `[first_run] setup_completed = true`, the splash off, `[model_catalog]
  auto_refresh_enabled = false`, `[console.onboarding] first_send_completed = true`,
  `[chat_defaults] provider = "llama_cpp"`, `model = "qwen3-8b-q4"`,
  `[api_settings.llama_cpp] api_url = "http://127.0.0.1:18767"` and `[providers]
  Llama_cpp = ["qwen3-8b-q4", "mistral-7b-instruct-q5"]`.
- A loopback HTTP server on 127.0.0.1:18767 answered `/v1/models` with those two ids
  and `/v1/chat/completions` with a short streamed reply. Nothing left the machine. The
  driver and the server are not committed.
- Steps, in order:
  1. In Console, type "Keep my settings please" and press **Enter**. The reply arrives,
     and the chat is auto-titled "Keep my settings please".
  2. **F4**, **F6**, **Down**, **Enter**, **Esc** (Providers & Models). Capture `01`.
  3. **F6** (Provider), **Tab** ×4 (Model), type `mistral`, **Down**, **Enter**,
     **Esc**. Capture `02`.
  4. **F6** (rail), **Up**, **Enter** (Overview), **Down**, **Enter** (back to Providers
     & Models, which puts the inspector at its top again). Capture `03`.
  5. **s**. Capture `04` after the toast cleared. The scratch `config.toml` then held
     `[chat_defaults] model = "mistral-7b-instruct-q5"`.
  6. **F6**, **Tab** ×4 (Model), then a mouse click on the inspector's "▶ config key"
     title. Capture `05`.
- The app was quit, the tmux server killed, and the scratch directory (including the
  `verify_p7t4_live` data dir) deleted.

## Captures

Each capture is plain text (`.txt`, `tmux capture-pane -p`) and with colour escapes
(`.ansi.txt`, `-e`).

- `01-card-and-inspector-chat-with-work` (AC#8): under Default model, the **Applies
  to** row reads "new chats (Ctrl+T, temporary, workspace). Open chat “Keep my
  settings please” keeps llama.cpp · qwen3-8b-q4." The title is long, so the row
  wraps to a second line rather than cutting off the pair. The inspector leads with
  **Applies to** (four rows), then **Next new chat will use** ("llama.cpp ·
  qwen3-8b-q4", "T 0.6 · max 4096 · stream On"), then **Focused field guide** with no
  "Saved as" row and the closed one-row "▶ config key" under it. The card's old
  catalog, credential-policy, manual-entry, sampling-route and endpoint-key rows are
  gone.
- `02-model-staged-guide-and-closed-config-key`: `mistral-7b-instruct-q5` is staged
  (Source word **edited \***, "1 unsaved"). The chat holds work, so the Applies-to row
  still names the chat's own pair. The inspector has scrolled to the guide for Model:
  Purpose, Save and Validation, then "▶ config key" closed. The Provider Source word
  also reads **edited \***; that behaviour predates this task (TASK-33007.3 noted it).
- `03-unsaved-edit-next-new-chat-note` (AC#3): with the edit still unsaved, Next new
  chat still names the saved `llama.cpp · qwen3-8b-q4` and adds "Unsaved edits apply
  only after save (s)."
- `04-saved-next-new-chat-open-chat-keeps`: after **s**, Next new chat reads
  `llama.cpp · mistral-7b-instruct-q5` and the note is gone. The open chat, which holds
  work, still keeps `llama.cpp · qwen3-8b-q4`.
- `05-config-key-opened-keeps-the-model-key` (AC#4, AC#5): clicking the title opens
  the disclosure. It reads "Saved as: chat_defaults.model", which is the Model field's
  key: the guide keeps the field the user was on. It also holds the endpoint key
  (`api_settings.llama_cpp.api_url`) and the provider catalog. Opening the disclosure
  scrolls the inspector so the disclosure sits at the top.
