# TASK-33006.3 Chat settings one-row disclosures captures (2026-10-02)

Chat settings (Ctrl+O) after TASK-33006.3, captured at 211x44 and 235x52 from the real app in this worktree. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold underlined text on the focus fill.

## Setup

- The app ran in an isolated tmux server (`-L cap33006t3`) under `env -i`, with `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` in a scratch directory and the null keyring backend.
- The scratch profile (`verify_cap33006_3`) set `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false` and the splash off.
- Run 1 (captures 01-05): the chat default was Anthropic `claude-sonnet-4-5`, with no key in the config and a dummy `ANTHROPIC_API_KEY` in the environment. The dummy key appears in no capture.
- Run 2 (capture 06): the chat default was OpenAI `gpt-5`, with no key anywhere, so the chat is Not ready.
- No provider was contacted (no Test connection, no send). The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7) and the listing of `~/.config/tldw_cli` + `~/.local/share/tldw_cli` were the same before and after. tmux was killed and the scratch profile deleted. The driver is not committed.

## Captures

1. `01-anthropic-model-view-211x44`: Ctrl+O. Each closed disclosure is one row whose title carries its value: "▶ Connection · api.anthropic.com · key from env ANTHROPIC_API_KEY · change it in Settings ▸ Providers & Models", "▶ Request estimate · 0 / 200,000 tokens (estimated; model unverified)" and "▶ Your name in this chat · User (global default)". The Sampling line wraps by TASK-33006.2's ruling. The footer shows without scrolling.
2. `02-connection-opened-from-its-title-211x44`: Tab ×6 from Temperature lands on the Connection title, then Enter. Connection opens in place with the same title, and no Endpoint label shows: Anthropic takes no server address and this profile has no saved endpoints, so the whole Endpoint row is hidden.
3. `03-connection-contents-scrolled-211x44`: the mouse wheel scrolls the rest of Connection into view: the probe copy ("No non-billable live connection check is available for this provider."), the readiness detail, and the generation test's unavailable copy (Anthropic has neither probe; TASK-30014's tests cover a provider that has both).
4. `04-anthropic-model-view-235x52`: Esc (no edits, so it closes), the terminal resized to 235x52, Ctrl+O again. The frame stays 150x22 and every closed disclosure is still one row.
5. `05-name-title-follows-typing-235x52`: Tab ×8 to the name title, Enter, Tab into the field, type "Ada". The title reads "▼ Your name in this chat · Ada" while typing.
6. `06-not-ready-openai-opens-on-configure-credential-211x44`: run 2, Ctrl+O. Chat settings opens with Connection already open, its title reading "Connection · api.openai.com · key missing · change it in Settings ▸ Providers & Models", and focus on **Configure credential…** (bold underlined on the focus fill in the `.ansi.txt`); no tuning field has focus.

The header still reads "Conversation settings" and the footer keeps its old labels: TASK-33006.5 renames them. Connection still holds the Provider and Model pickers until TASK-33006.4.
