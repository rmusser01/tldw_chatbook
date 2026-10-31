# TASK-33006.3 Chat settings one-row disclosures captures

Chat settings (Ctrl+O), captured from the real app at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold underlined text on the focus fill.

These were retaken on 2026-10-03 in the Phase 6 final fix wave, at commit 5ef85f878d (`git status -- tldw_chatbook` clean). That commit implements the owner ruling of 2026-10-02: every closed disclosure title stays one row. The earlier sets showed the Sampling title wrapping to two rows.

## Setup

- **Isolation.** Each run used its own tmux server (`-L p6finalt3a`, then `-L p6finalt3b`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** `[chat_defaults]` sets the provider, the model, Temperature 0.7 and Max tokens 2048. The profile also sets `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true` and `[model_catalog] auto_refresh_enabled = false`, and turns the splash screen off.
- **Run 1 (captures 01-05).** The chat default was Anthropic `claude-sonnet-4-5`. The config held no key, and a dummy `ANTHROPIC_API_KEY` was in the environment; the key appears in no capture.
- **Run 2 (capture 06).** The chat default was OpenAI `gpt-5`, with no key anywhere, so the chat is Not ready.
- **No provider contacted.** Neither run used Test connection or sent anything.
- **Real profile unchanged.** Before and after, the real `~/.config/tldw_cli/config.toml` had sha256 prefix 15c6cb224a6a51c7, and `ls ~/.local/share/tldw_cli | shasum -a 256` gave db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch homes deleted. The driver is not committed.

## Captures

1. `01-anthropic-model-view-211x44`: Ctrl+O. Every closed disclosure is one row whose title carries its value:
   - "▶ Sampling · Anthropic does not accept 7 fields (open to list them)"
   - "▶ Connection · api.anthropic.com · key from env ANTHROPIC_API_KEY · change it in Settings ▸ Providers & Models"
   - "▶ Request estimate · 0 / 200,000 tokens (estimated; model unverified)"
   - "▶ Your name in this chat · User (global default)"

   The footer shows without scrolling.
2. `02-connection-opened-from-its-title-211x44`: Tab ×6 from Temperature lands on the Connection title, then Enter. Connection opens in place with the same title. No Endpoint label shows: Anthropic takes no server address and this profile has no saved endpoints, so the whole Endpoint row is hidden. The body scrolls to keep the open disclosure in view.
3. `03-connection-contents-scrolled-211x44`: five mouse-wheel steps over the Connection contents scroll the rest into view. Anthropic has neither probe, so it shows:
   - the probe copy: "No non-billable live connection check is available for this provider.";
   - the readiness detail;
   - "Generation test unavailable for this provider."

   TASK-30014's tests cover a provider that has both probes.
4. `04-anthropic-model-view-235x52`: Esc (no edits, so it closes), the terminal resized to 235x52, then Ctrl+O. The frame stays 150x22, and all four closed titles are one row each, as in 01.
5. `05-name-title-follows-typing-235x52`: Tab ×8 to the name title, Enter, Tab into the field, then type "Ada". The title reads "▼ Your name in this chat · Ada" while typing, and the footer counts one unsaved edit.
6. `06-not-ready-openai-opens-on-configure-credential-211x44`: run 2, Ctrl+O. Chat settings opens with Connection already open, and focus is on **Configure credential…** (bold underlined on the focus fill in the `.ansi.txt`); no tuning field has focus. The Connection title reads "Connection · api.openai.com · key missing · change it in Settings ▸ Providers & Models". The closed Sampling title is the named form ("Sampling · hidden for OpenAI: Min P, Top K, Thinking, Thinking budget (this provider does not accept them)"), because the names fit one row.
