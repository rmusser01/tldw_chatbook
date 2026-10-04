# TASK-33006.7 A view switch opens the new view at its top

Chat settings (Ctrl+O), captured from the real app at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

These were retaken on 2026-10-03 in the Phase 6 final fix wave, at commit 5ef85f878d (`git status -- tldw_chatbook` clean). That commit makes every closed disclosure title one row (owner ruling of 2026-10-02) and adds a list of the hidden fields to the opened Sampling disclosure.

## Setup

- **Isolation.** Each session ran in its own tmux server (`-L p6finalt7a`, `-L p6finalt7b`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** `[chat_defaults]` is Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048. The profile also sets `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true` and `[model_catalog] auto_refresh_enabled = false`, and the splash is off.
- **Two sessions.**
  - 01-06: a dummy `ANTHROPIC_API_KEY` was in the environment, so the chat is ready. No request was sent, and the key appears in no capture.
  - 07-10: no key anywhere, so the chat is not ready (key missing).
- **Input.** Ctrl+O was sent with `tmux send-keys`. Everything else is SGR mouse input: a press and release on a disclosure title or a view tab, and 30 wheel-down events on a body row to scroll a view to its end. A click on a view tab gives that tab focus before the switch, as Enter on a focused tab does. 05, 06 and 10 follow a `tmux resize-window` to 235x52.
- **Real profile unchanged.** Before and after:
  - `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7;
  - `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2.
- **Cleanup.** tmux was killed and the scratch homes deleted. The driver is not committed.

## Captures

Focus on an Input or a Select shows in the `.txt` as the █ cursor. Focus on a Button shows only in the `.ansi.txt`: its label is bold underline (`ESC[1;4m`) on a blue background (`ESC[48;2;25;68;102m`).

1. **`01-model-view-scrolled-211x44`**. Sampling and Connection were opened with a click on their titles, then the Model view was scrolled to its end. Top P is the first body row, and the Model row and the core fields sit above the fold.
2. **`02-context-view-opens-at-model-capacity-211x44`** (click Context and memory). This is AC#4's capture. Context and memory opens at its top: the first section is **Model capacity**, with Model window, Max tokens, Safety margin and Safe input ceiling. Budget strategy has focus (█).
3. **`03-context-view-scrolled-211x44`**. The Context view scrolled down: Safe input ceiling is its first body row. Budget strategy keeps focus (█), because the wheel does not move it.
4. **`04-model-view-opens-at-model-row-211x44`** (click Model and generation).
   - The Model view opens at its top, on the Model row, with Temperature focused (█).
   - Sampling and Connection are still open from 01, so both titles show ▼.
   - Under Sampling's one-row title, the open disclosure lists "Anthropic does not accept: Min P, Seed, Presence penalty, Frequency penalty, Reasoning effort, Reasoning summary, Verbosity.".
5. **`05-model-view-scrolled-235x52`**. The Model view scrolled at 235x52 (Top P first, Model row hidden).
6. **`06-context-view-opens-at-model-capacity-235x52`** (click Context and memory). Model capacity is first, and Budget strategy is focused (█).
7. **`07-missing-key-chat-opens-on-its-fix-211x44`** (Ctrl+O, no key). Opening a not-ready chat:
   - the Model row is first, and the tuning rows follow;
   - the closed Sampling title is one row;
   - Connection is open, and Configure credential… has focus (`.ansi.txt` only).
8. **`08-missing-key-context-view-scrolled-211x44`** (click Context and memory, then scroll). The Context view scrolled down.
9. **`09-missing-key-switch-lands-as-open-211x44`** (click Model and generation). The switch lands as opening does: 09 is byte-identical to 07 in both the `.txt` and the `.ansi.txt`, including the focus escape on Configure credential….
10. **`10-missing-key-switch-lands-as-open-235x52`** (resize, click Context and memory, click Model and generation). The same at 235x52: the Model row is first, and Configure credential… is focused (`.ansi.txt` only).
