# TASK-33007.2: Connect as one-row provider, key, endpoint and key-check rows (2026-10-03)

Keyboard-only captures of Settings ▸ Providers & Models at 211x44 with the real
stylesheet, for one cloud provider (Anthropic, key saved in config) and one local
provider (Ollama, server not running).

## Procedure

- The real app ran in an isolated tmux server at 211x44 under `env -i`. `HOME`,
  `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` pointed into a scratch
  directory, and the keyring backend was the null one.
- The scratch profile (`users_name = "verify_p7t2_head"`) set `[first_run]
  setup_completed = true`, the splash off, `[model_catalog] auto_refresh_enabled =
  false`, `[console.onboarding] first_send_completed = true`, `[chat_defaults]
  provider = "anthropic"`, `model = "claude-sonnet-4-5"`, and a made-up
  `[api_settings.anthropic] api_key`. Everything else came from the shipped template.
  No key check was run against Anthropic, so nothing left the machine.
- Keys, in order:
  1. Wait for the nav bar, then **F4** (Settings), **F6** (category rail), **Down**,
     **Enter** (Providers & Models). Capture `01`.
  2. **F6** moves to the detail pane, which lands on the Provider control. Capture `02`.
  3. Type `olla`. Capture `03`: the list is open under the control.
  4. **Down**, **Enter** (Ollama). Capture `04`.
  5. **Esc** (leave the field), **t** (the key check). Capture `05`.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix `15c6cb224a6a51c7`) and the
  `~/.local/share/tldw_cli` name listing (`db7e7faf5bff92d2`) were the same before and
  after. The driver is not committed.

## Captures

Each capture is plain text (`.txt`, `tmux capture-pane -p`) and with colour escapes
(`.ansi.txt`, `-e`).

- `01-cloud-connect-rest`: Connect for Anthropic. Provider, API key (**saved in
  config**, with **Clear**), Env var (**not set**), Endpoint (**built-in**), each with
  a Source word and help, then **Key check** "Ready · not tested" and **Test (t)**.
  Default model for new chats follows. The inspector shows the **Key** block ("Configuration
  check has not run.").
- `02-cloud-provider-focused`: the Provider control focused (thick edge), one row; the
  inspector's focused-field guide names Provider.
- `03-provider-list-filtered`: `olla` typed. The list under the control shows Ollama
  Cloud, Ollama and Ollama (legacy alias) in their groups, every first character
  painted; the help reads "3 found · Enter picks · Esc cancels".
- `04-local-connect-chosen`: Ollama chosen. The control names it, its Source word reads
  **edited \***, the API key row reads **not required**, Endpoint **config** with "required:
  the server's base URL", and the model moved to Ollama's own.
- `05-local-key-check-after-t`: after **t**, the Key check row reads "Not ready ·
  refused :11434" and the card did not move; the labelled result rows are in the
  inspector's Key block, and the toast says the listing failed.
