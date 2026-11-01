# TASK-33006.6 Context and memory labels and the defaults line

Chat settings (Ctrl+O), captured from the real app in this worktree at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

These were retaken on 2026-10-03 in the Phase 6 final fix wave, at commit 5ef85f878d (`git status -- tldw_chatbook` clean). Each closed disclosure title is now one row (owner ruling of 2026-10-02).

Since the final review's I5 fix, **Save as model default** and the defaults line beside it show only while the draft differs from the saved defaults. An untouched chat shows neither. So before capture 01 this chat applied Temperature 0.9 (Ctrl+O, Temperature 0.9, Ctrl+Enter). Its value then differs from the saved 0.7, the save is offered again, and the line has something to stand beside.

## Setup

- **Isolation.** The app ran in its own tmux server (`-L p6finalt6`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6final_t6`. `[chat_defaults]` sets Anthropic `claude-sonnet-4-5`, Temperature 0.7 and Max tokens 2048. The profile also sets `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true` and `[model_catalog] auto_refresh_enabled = false`, and turns the splash screen off.
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. No request was sent, and the key appears in no capture.
- **Input.** Keys only: Ctrl+O, End, Backspace, typed text, Ctrl+Enter (as its CSI u sequence), Shift+Tab, Enter, Esc and Ctrl+Q, sent with `tmux send-keys`. Captures 03 and 04 follow a `tmux resize-window` to 235x52.
- **Real profile unchanged.** Before and after, `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7, and `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch home deleted. The driver is not committed.

## Captures

1. **`01-model-view-defaults-line-beside-save-211x44`** (Ctrl+O on the chat that applied 0.9). The Model view offers **Use saved defaults** and **Save as model default**, and the line above the footer reads "Used by future conversations for Anthropic." beside them.
2. **`02-context-view-labels-clear-no-defaults-line-211x44`** (Shift+Tab twice, then Enter on Context and memory). This is AC#4's capture. Every label ends at least one blank cell before its control, for example "Conversation max tokens │ if Custom" (it read "Conversation max tokens│ if Custom" before). The row above the footer is blank. The footer offers only Esc close, Cancel and Apply to this chat, so the defaults line is gone.
3. **`03-context-view-labels-clear-no-defaults-line-235x52`** (resized). The same at 235x52.
4. **`04-model-view-after-round-trip-line-returns-235x52`**. The steps: Esc, Ctrl+O, Shift+Tab twice, Enter on Context and memory, Shift+Tab twice, Enter on Model and generation. Back in the Model view, the line returns beside **Save as model default**. The switch moves focus to Temperature (TASK-33006.7), and the Model tab paints "Model and generation · Selected".
