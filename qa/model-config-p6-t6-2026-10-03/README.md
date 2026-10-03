# TASK-33006.6 Context and memory labels and the defaults line

Chat settings (Ctrl+O) after TASK-33006.6, captured from the real app in this worktree at 211x44 and 235x52 on 2026-10-03. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

## Setup

- **Isolation.** The app ran in its own tmux server (`-L p6t6cap2aa1d6e`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6t6_2aa1d6e`; `[chat_defaults]` Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048; `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`; splash off.
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. No request was sent, and the key appears in no capture.
- **Input.** Keys only for every capture: Ctrl+O, Shift+Tab, Enter, Esc and Ctrl+Q, sent with `tmux send-keys`; 03 and 04 follow a `tmux resize-window` to 235x52.
- **Real profile unchanged.** Before and after: `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7 (mtime Sep 26), and `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch home, with its profile, deleted. The driver is not committed.

## Captures

1. **`01-model-view-defaults-line-beside-save-211x44`** (Ctrl+O on Chat 1). The Model view offers **Save as model default**, and the line above the footer reads "Used by future conversations for Anthropic." beside it.
2. **`02-context-view-labels-clear-no-defaults-line-211x44`** (Shift+Tab twice, Enter on Context and memory). AC#4's capture. Every label ends at least one blank cell before its control: "Conversation max tokens │ if Custom" (it read "Conversation max tokens│ if Custom" before). The row above the footer is blank: the footer offers only Esc close, Cancel and Apply to this chat, so the defaults line is gone.
3. **`03-context-view-labels-clear-no-defaults-line-235x52`** (resized). The same at 235x52; compare `qa/model-config-p6-t5-2026-10-03/07-context-view-keeps-cancel-235x52.txt`, which shows both defects.
4. **`04-model-view-after-round-trip-line-returns-235x52`** (Esc, Ctrl+O, Shift+Tab twice, Enter on Context and memory, Shift+Tab twice, Enter on Model and generation). Back in the Model view, the line returns beside **Save as model default**.

Not this task's: in 01 and 04 the closed Sampling title wraps to two rows for Anthropic (TASK-33006.2's line, under the owner ruling recorded in the Phase 6 plan). In 04 the focused Model tab paints "Model and generation" without "· Selected"; a mounted probe shows the same paint at base 2aa1d6e1c6 after a keyboard round trip, while the button's label is "Model and generation · Selected".
