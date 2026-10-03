# TASK-33006.2 Chat settings hidden-fields captures (2026-10-02)

Chat settings (Ctrl+O) on an Anthropic chat after TASK-33006.2, captured at 211x44 and 235x52 from the real app in this worktree. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold text on the focus fill.

## Setup

- The app ran in an isolated tmux server (`-L cap33006t2`) under `env -i`, with `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` in a scratch directory and the null keyring backend.
- The scratch profile (`verify_cap33006_2`) set `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false` and the splash off. The chat default was Anthropic `claude-sonnet-4-5` with a dummy API key; the shipped `chat_defaults` still carry Min P 0.05.
- No provider was contacted (no Test connection, no send). The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7, mtime Sep 26) and the `~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the same before and after. tmux was killed and the scratch profile deleted. The driver is not committed.

## Captures

1. `01-anthropic-model-view-211x44`: Ctrl+O on the Anthropic chat. The CORE rows are Temperature, Max tokens, Streaming, Thinking and Thinking budget; Reasoning effort, Reasoning summary and Verbosity are not rendered. The Sampling title reads "Sampling · hidden for Anthropic: Min P, Seed, Presence penalty, Frequency penalty, Reasoning effort, Reasoning summary, Verbosity (this provider does not accept them)" and wraps onto a second row inside the frame; the footer still shows without scrolling. Thinking and Thinking budget, whose support for this model is unknown, stay visible and their help lines start with "Support not verified for this model." (no separate support note).
2. `02-sampling-opened-from-its-title-211x44`: Tab ×5 from Temperature (Max tokens, Streaming, Thinking, Thinking budget, then the Sampling title: no hidden row takes focus), Enter. Sampling opens with Top P and Top K only.
3. `03-tab-reaches-top-k-211x44`: Tab ×2 more. Focus goes Top P, then Top K (bold on the focus fill in the `.ansi.txt`); Min P is skipped.
4. `04-anthropic-model-view-235x52`: Esc (no edits, so it closes), the terminal resized to 235x52, Ctrl+O again. The frame stays 150x22 and the same line wraps the same way.

The header still reads "Conversation settings" and the footer keeps its old labels: TASK-33006.5 renames them.
