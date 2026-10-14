# TASK-33006.5 Chat settings name, scope, footer and Use saved defaults captures

Chat settings (Ctrl+O), captured from the real app in this worktree at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

These were retaken on 2026-10-03 in the Phase 6 final fix wave, at commit 5ef85f878d (`git status -- tldw_chatbook` clean). The retake covers three fixes:

- Each closed disclosure title stays one row (owner ruling of 2026-10-02).
- **Save as model default** shows only while the draft differs from the saved defaults, the same test that dims **Use saved defaults** (final review I5).
- A blank choice row reads "default" with the Source word "provider" (I6).

## Setup

- **Isolation.** The app ran in its own tmux server (`-L p6finalt5`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6final_t5`. `[chat_defaults]` sets Anthropic `claude-sonnet-4-5`, Temperature 0.7 and Max tokens 2048. The profile also sets `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true` and `[model_catalog] auto_refresh_enabled = false`, and turns the splash screen off. The app filled in the rest of the shipped template on start; that template sets Anthropic's own Streaming to false, so Streaming reads Off throughout.
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. No request was sent, and the key appears in no capture.
- **Input.** Keys were sent with `tmux send-keys`: Ctrl+O, Ctrl+T, Ctrl+K, Enter, End, Backspace, Tab, Shift+Tab, Esc, Ctrl+N, Ctrl+Q and typed text. Ctrl+Enter was sent as its CSI u sequence (`\e[13;5u`). Two buttons were pressed with an SGR mouse click (`\e[<0;COL;ROWM`/`m`): Save as model default (between 02's setup steps) and Use saved defaults (before 03).
- **Real profile unchanged.** Before and after, `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7, and `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch home deleted. The driver is not committed.

## Captures

1. **`01-untouched-chat-matches-saved-defaults-211x44`** (Ctrl+O on the first chat).
   - The title reads "Chat settings · Chat 1".
   - The scope line reads "Applies to this chat only · saved with the conversation · defaults live in Settings ▸ Providers & Models (F4)".
   - The footer reads "Esc close", then **Matches saved defaults** (dimmed, because this chat already holds the saved defaults), **Default for new chats (Ctrl+N)** and **Apply to this chat (Ctrl+Enter)**, on one row.
   - **Save as model default** is not offered: saving would change nothing.
   - There is no Cancel in this view, and the closed Sampling title is one row.
2. **`02-chat-with-work-keeps-its-values-211x44`**. Steps before it:
   - In Chat 1, set Temperature to 0.9, then Ctrl+Enter ("This chat updated"). The chat now holds work.
   - Ctrl+T, then Ctrl+O in Chat 2. Set Temperature to 0.3 and Max tokens to 4096, then click **Save as model default**, which those edits brought into the footer. The scratch `config.toml` gained the model profile.
   - Ctrl+K, then Enter back to Chat 1, then Ctrl+O.

   Chat 1 kept its own values: Temperature 0.9 and Max tokens 2048 read "this chat". Its draft differs from the new saved defaults, so **Use saved defaults** and **Save as model default** are both offered, with the line "Used by future conversations for Anthropic." beside them.
3. **`03-use-saved-defaults-staged-211x44`** (click on Use saved defaults).
   - Temperature 0.3 and Max tokens 4096 read "edited *". The title counts "2 unsaved edits", and the Esc hint asks about 2 unsaved.
   - The pair is still claude-sonnet-4-5 · Anthropic ("this chat").
   - The button now reads **Matches saved defaults**, and Save as model default is gone with it.
   - Focus moved to Apply to this chat (underlined in the `.ansi.txt`). Nothing was applied yet.
4. **`04-use-saved-defaults-staged-235x52`**. The same draft after resizing to 235x52; the frame stays 150 wide.
5. **`05-applied-235x52`** (Ctrl+Enter). "This chat updated". This run did not hash the scratch `config.toml` around Apply; `Tests/UI/test_console_saved_defaults_flow.py` pins that Apply leaves it byte-identical.
6. **`06-reopened-applied-values-235x52`** (Ctrl+O on Chat 1). Temperature 0.3 and Max tokens 4096 are now the chat's values. They read "model default" because they equal the saved chain, and the button reads **Matches saved defaults**.
7. **`07-context-view-keeps-cancel-235x52`** (Shift+Tab twice, Enter on Context and memory). The Context view keeps its own scope line ("Use: this conversation only. Defaults: F4 Settings > Console Behavior."). Its footer is "Esc close", **Cancel**, **Apply to this chat (Ctrl+Enter)**.
8. **`08-ctrl-n-default-for-new-chats-235x52`** (Esc, Ctrl+O, Ctrl+N). Ctrl+N ran Default for new chats: "Eligible new-chat default saved". Ctrl+Q then quit.
