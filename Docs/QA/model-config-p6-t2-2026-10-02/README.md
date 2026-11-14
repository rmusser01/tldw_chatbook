# TASK-33006.2 Chat settings hidden-fields captures

Chat settings (Ctrl+O) on an Anthropic chat, captured from the real app at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold text on the focus fill.

These were retaken on 2026-10-03 in the Phase 6 final fix wave, at commit 5ef85f878d (`git status -- tldw_chatbook` clean). That commit implements the owner ruling of 2026-10-02: every closed disclosure title stays one row. The first set, taken at TASK-33006.2's own commit, showed the Sampling title wrapping to two rows. The ruling replaced that design.

## Setup

- **Isolation.** The app ran in its own tmux server (`-L p6finalt2`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory. The keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6final_t2`. `[chat_defaults]` is Anthropic `claude-sonnet-4-5`, Temperature 0.7 and Max tokens 2048. It also sets `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`, and the splash screen off.
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. No provider was contacted (no Test connection, no send), and the key appears in no capture.
- **Input.** Keys only, sent with `tmux send-keys`.
- **Real profile unchanged.** `~/.config/tldw_cli/config.toml` had sha256 prefix 15c6cb224a6a51c7, and `ls ~/.local/share/tldw_cli | shasum -a 256` gave db7e7faf5bff92d2. Both were the same before and after.
- **Cleanup.** tmux was killed and the scratch profile deleted. The driver is not committed.

## Captures

1. `01-anthropic-model-view-211x44` (Ctrl+O).
   - The CORE rows are Temperature, Max tokens, Streaming, Thinking and Thinking budget. Reasoning effort, Reasoning summary and Verbosity are not rendered.
   - The closed Sampling title is one row: "▶ Sampling · Anthropic does not accept 7 fields (open to list them)". Naming all seven would not fit the 141 cells a closed title holds, so the title counts them.
   - Thinking and Thinking budget stay visible because their support for this model is unknown. Their help lines start with "Support not verified for this model.".
   - The blank Thinking Select shows "default", and both blank rows read "provider" in the Source column.
   - The footer shows without scrolling: Esc close, **Matches saved defaults** (dimmed), Default for new chats (Ctrl+N), Apply to this chat (Ctrl+Enter). This untouched chat already holds the saved defaults, so Save as model default is not offered.
2. `02-sampling-opened-from-its-title-211x44` (Tab ×5 from Temperature, then Enter).
   - The five Tabs go through Max tokens, Streaming, Thinking and Thinking budget to the Sampling title. No hidden row takes focus.
   - Sampling opens with the list "Anthropic does not accept: Min P, Seed, Presence penalty, Frequency penalty, Reasoning effort, Reasoning summary, Verbosity.", then Top P and Top K only.
3. `03-tab-reaches-top-k-211x44` (Tab ×2). Focus goes to Top P, then Top K (█); Min P is skipped.
4. `04-anthropic-model-view-235x52`. Esc closed the modal (no edits), the terminal was resized to 235x52, and Ctrl+O reopened it. The frame stays 150x22, and the closed Sampling title is the same single row.
