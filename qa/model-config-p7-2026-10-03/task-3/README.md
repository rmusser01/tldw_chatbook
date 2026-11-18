# TASK-33007.3: the Default model is a searchable picker with discovered models merged in (2026-10-03)

Keyboard-only captures of Settings ▸ Providers & Models at 211x44 with the real
stylesheet, for a local provider (llama.cpp) whose endpoint lists three models.

## Procedure

- The real app ran in an isolated tmux server at 211x44 under `env -i`. `HOME`,
  `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` pointed into a scratch
  directory, and the keyring backend was the null one.
- The scratch profile (`users_name = "verify_p7t3_live"`) was written from the shipped
  template by a process whose own `HOME`/`XDG_*`/`TLDW_CONFIG_PATH` pointed at a second
  scratch directory. It set `[first_run] setup_completed = true`, the splash off,
  `[model_catalog] auto_refresh_enabled = false`, `[console.onboarding]
  first_send_completed = true`, `[chat_defaults] provider = "llama_cpp"`, `model =
  "qwen3-8b-q4"`, `[api_settings.llama_cpp] api_url = "http://127.0.0.1:18766"` and
  `[providers] Llama_cpp = ["qwen3-8b-q4", "mistral-7b-instruct-q5"]`.
- A loopback HTTP server on 127.0.0.1:18766 answered every GET with a three-model
  `/v1/models` listing: `qwen3-8b-q4`, `gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf` and
  `llama-4-scout-17b-16e.Q4_K_M.gguf`. Nothing left the machine. The driver and the
  server are not committed.
- Keys, in order:
  1. Wait for the nav bar, then **F4**, **F6**, **Down**, **Enter** (Providers &
     Models). Capture `01`.
  2. **F6** (the Provider control), **Tab** ×4 (API key, Env var, Endpoint, Model).
     Capture `02`.
  3. **Tab** ×4 (Custom ID, the list, Context window, Discover models), **Enter**
     (Discover), **Shift+Tab** ×2 (back to Model). Capture `03` after the toast.
  4. Type `gemma`. Capture `04`.
  5. **Down**, **Enter**, **Esc**. Capture `05`.
  6. **s**. Capture `06`. The scratch `config.toml` then held
     `[chat_defaults] model = "gemma-4-26B-A4B-it-ultra.Q4_K_M.gguf"` and an unchanged
     `Llama_cpp = [ "qwen3-8b-q4", "mistral-7b-instruct-q5",]`.
- The tmux server and the loopback server were stopped, and the scratch directory
  (including the `verify_p7t3_live` data dir) was deleted.

## Captures

Each capture is plain text (`.txt`, `tmux capture-pane -p`) and with colour escapes
(`.ansi.txt`, `-e`).

- `01-default-model-at-rest`: under "Default model for new chats", **Model** is one
  row: the field in the Connect control column (`qwen3-8b-q4`), the Source word
  **new-chat default** lined up with Connect's, and the help "type to search · Custom
  ID for others".
- `02-picker-open-before-discover`: Model focused (thick edge). The open list spans
  the row; **Custom ID** and the status line ("2 models available. Type to filter.")
  sit under the field; the Source word and help step aside. The list holds the
  saved ids under **Saved fallback**, with `qwen3-8b-q4  ● CURRENT` highlighted (the
  `.ansi.txt` shows its highlight). The inspector names the focused setting, Model.
- `03-picker-open-with-discovered` (AC#11): after Discover, the same list adds
  **Served now** with the two listed ids no list held; `qwen3-8b-q4` (saved and
  listed) stays under Saved fallback, still marked and highlighted.
- `04-prefix-search-finds-discovered`: `gemma` typed; the list holds only the
  discovered gguf id, as a visible row (no ghost text).
- `05-discovered-chosen-staged`: chosen and Esc. The field shows the gguf id, its
  Source word reads **edited \***, the State badge reads "1 unsaved", the category
  carries `*`, focus is released, and the footer reads "s save category" (no Esc
  prefix).
- `06-saved-default-model`: after **s**, the save result names its scope and the
  Source word reads **new-chat default** again.
