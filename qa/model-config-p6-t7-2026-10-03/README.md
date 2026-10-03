# TASK-33006.7 A view switch opens the new view at its top

Chat settings (Ctrl+O) after TASK-33006.7, captured from the real app in this worktree at 211x44 and 235x52 on 2026-10-03. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

## Setup

- **Isolation.** The app ran in its own tmux server (`-L p6t7capd3c90a8`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6t7_d3c90a8`; `[chat_defaults]` Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048; `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`; splash off.
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. No request was sent, and the key appears in no capture.
- **Input.** Ctrl+O, Tab and Enter with `tmux send-keys`. The Model view was scrolled by tabbing through the opened Sampling and Connection disclosures (focus scrolls the body). 01 → 02 switches with Enter on the focused Context and memory tab. 03 → 04 and 05 → 06 switch with an SGR mouse click on the tab: the driver could not read from the capture which tab had keyboard focus there, and a click leaves the body's scroll as it is. 05 and 06 follow a `tmux resize-window` to 235x52.
- **Real profile unchanged.** Before and after: `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7 (mtime Sep 26), and `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch home, with its profile, deleted. The driver is not committed.

## Captures

1. **`01-model-view-scrolled-211x44`**. Sampling and Connection open, the Model view scrolled to its end: Top K is the first body row, and the Model row and the core fields sit above the fold. The Context and memory tab has focus.
2. **`02-context-view-opens-at-model-capacity-211x44`** (Enter). AC#4's capture. Context and memory opens at its top: **Model capacity** with Model window, Max tokens, Safety margin and Safe input ceiling is the first section, and Budget strategy has focus (█) on screen. Before the fix it opened at the Model view's offset with Model capacity above the fold (the task's live repro and the mounted test's RED run).
3. **`03-context-view-scrolled-211x44`**. The Context view scrolled down: Safe input ceiling is its first body row.
4. **`04-model-view-opens-at-model-row-211x44`** (click Model and generation). The Model view opens at its top, on the Model row, with Temperature focused (█).
5. **`05-model-view-scrolled-235x52`**. The Model view scrolled at 235x52 (Top K first, Model row hidden).
6. **`06-context-view-opens-at-model-capacity-235x52`** (click Context and memory). Model capacity first, Budget strategy focused. At this size the Context view fits, so it cannot be scrolled before a switch back.

Not this task's: the closed and open Sampling title wraps to two rows for Anthropic in 01, 04 and 05. That is TASK-33006.2's title as committed at this task's base, and it predates the owner ruling of 2026-10-02 that every closed disclosure title stays one row. This task relies only on which section each view opens at and where focus lands.
