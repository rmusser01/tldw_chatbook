# TASK-33006.5 Chat settings name, scope, footer and Use saved defaults captures

Chat settings (Ctrl+O) after TASK-33006.5, captured from the real app in this worktree at 211x44 and 235x52 on 2026-10-03. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

## Setup

- **Isolation.** The app ran in its own tmux server (`-L p6t5capbd0be72`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6t5_bd0be72`; `[chat_defaults]` Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048; `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`; splash off. The app filled in the rest of the shipped template on start (it sets Anthropic's own Streaming to false, so Streaming reads Off throughout).
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. No request was sent, and the key appears in no capture.
- **Input.** Keys only, sent with `tmux send-keys`: Ctrl+O, Ctrl+T, Ctrl+K, Enter, End, Backspace, Tab, Shift+Tab, Esc, Ctrl+N, Ctrl+Q and typed text. Ctrl+Enter was sent as its CSI u sequence (`\e[13;5u`), the form a terminal with the kitty keyboard protocol sends; tmux has no key name for it. Two buttons were pressed with an SGR mouse click (`\e[<0;COL;ROWM`/`m`): Save as model default (between 02's setup steps) and Use saved defaults (before 03).
- **Real profile unchanged.** Before and after: `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7 (mtime Sep 26), and `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch profile deleted. The driver is not committed.

## Captures

1. **`01-untouched-chat-matches-saved-defaults-211x44`** (Ctrl+O on the first chat). The title reads "Chat settings · Chat 1"; the scope line reads "Applies to this chat only · saved with the conversation · defaults live in Settings ▸ Providers & Models (F4)". The footer reads "Esc close", then **Matches saved defaults** (dimmed: this chat already holds the saved defaults), **Save as model default**, **Default for new chats (Ctrl+N)** and **Apply to this chat (Ctrl+Enter)**, on one row. There is no Cancel in this view.
2. **`02-chat-with-work-keeps-its-values-211x44`**. Steps before it:
   - in Chat 1, Temperature 0.9, then Ctrl+Enter ("This chat updated"): the chat now holds work;
   - Ctrl+T, Ctrl+O in Chat 2, Temperature 0.3 and Max tokens 4096, then a click on **Save as model default** ("Model profile default saved: anthropic/claude-sonnet-4-5"; the scratch `config.toml` gained the profile);
   - Ctrl+K, Enter back to Chat 1, Ctrl+O.

   Chat 1 kept its own values: Temperature 0.9 and Max tokens 2048 read "this chat", and **Use saved defaults** is offered.
3. **`03-use-saved-defaults-staged-211x44`** (click on Use saved defaults). Temperature 0.3 and Max tokens 4096 read "edited *", the title counts "2 unsaved edits", the Esc hint asks about 2 unsaved, the pair is still claude-sonnet-4-5 · Anthropic ("this chat"), and the button now reads **Matches saved defaults**. Focus moved to Apply to this chat (underlined in the `.ansi.txt`). Nothing was applied yet.
4. **`04-use-saved-defaults-staged-235x52`**. The same draft after resizing to 235x52; the frame stays 150 wide.
5. **`05-applied-235x52`** (Ctrl+Enter). "This chat updated". The scratch `config.toml` hash was the same before and after this Apply (a94c22ea7dbd9a6e).
6. **`06-reopened-applied-values-235x52`** (Ctrl+O on Chat 1). Temperature 0.3 and Max tokens 4096 are the chat's values now, read as "model default" because they equal the saved chain; the button reads **Matches saved defaults**.
7. **`07-context-view-keeps-cancel-235x52`** (Shift+Tab twice, Enter on Context and memory). The Context view keeps its own scope line ("Use: this conversation only. Defaults: F4 Settings > Console Behavior.") and its footer is "Esc close", **Cancel**, **Apply to this chat (Ctrl+Enter)**.
8. **`08-ctrl-n-default-for-new-chats-235x52`** (Esc, Ctrl+O, Ctrl+N). Ctrl+N ran Default for new chats: "Eligible new-chat default saved". Ctrl+Q then quit.

The closed Sampling title still wraps to two rows for Anthropic in 01-04; that is TASK-33006.2's line and the owner ruling recorded in the Phase 6 plan, not this task's.
