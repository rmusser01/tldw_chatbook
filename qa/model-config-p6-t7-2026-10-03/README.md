# TASK-33006.7 A view switch opens the new view at its top

Chat settings (Ctrl+O) after TASK-33006.7 and its review round 1, captured on 2026-10-03 from the real app at commit c4fc427c73 (the round-1 code commit; `git status -- tldw_chatbook` was clean at launch) at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes.

## Setup

- **Isolation.** Each session ran in its own tmux server (`-L p6t7fix1c4fc427`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree.
- **Scratch profile.** users_name `verify_p6t7fix1_1ad693f`; `[chat_defaults]` Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048; `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false`; splash off.
- **Two sessions.** 01-06: a dummy `ANTHROPIC_API_KEY` in the environment, so the chat is ready (no request was sent, and the key appears in no capture). 07-10: no key anywhere, so the chat is not ready (key missing).
- **Input.** Ctrl+O with `tmux send-keys`. Everything else is SGR mouse input: a press and release on a disclosure title or a view tab, and 30 wheel-down events on a body row to scroll a view to its end. A click on a view tab gives that tab focus before the switch, as Enter on a focused tab does. 05, 06 and 10 follow a `tmux resize-window` to 235x52.
- **Real profile unchanged.** Before and after: `shasum -a 256 ~/.config/tldw_cli/config.toml` began 15c6cb224a6a51c7 (mtime Sep 26), `ls ~/.local/share/tldw_cli | shasum -a 256` began db7e7faf5bff92d2, and no `verify_p6t7fix1` directory exists there (the profile's data directory was created under the scratch home).
- **Cleanup.** tmux was quit and killed, and the scratch homes deleted. The driver is not committed.

## Captures

Focus on an Input or a Select shows in the `.txt` as the █ cursor. Focus on a Button shows only in the `.ansi.txt`: its label is bold underline (`ESC[1;4m`) on a blue background (`ESC[48;2;25;68;102m`), which no other control in the modal carries (an unfocused button such as Change Alt+M is `ESC[1m` on `ESC[48;2;30;30;30m`).

1. **`01-model-view-scrolled-211x44`**. Sampling and Connection opened with a click on their titles, then the Model view scrolled to its end: Top K is the first body row, and the Model row and the core fields sit above the fold.
2. **`02-context-view-opens-at-model-capacity-211x44`** (click Context and memory). AC#4's capture. Context and memory opens at its top: **Model capacity** with Model window, Max tokens, Safety margin and Safe input ceiling is the first section, and Budget strategy has focus (█). Before the fix it opened at the Model view's offset with Model capacity above the fold (the task's live repro and the mounted test's RED run).
3. **`03-context-view-scrolled-211x44`**. The Context view scrolled down: Safe input ceiling is its first body row. Budget strategy keeps focus (█), since the wheel does not move it.
4. **`04-model-view-opens-at-model-row-211x44`** (click Model and generation). The Model view opens at its top, on the Model row, with Temperature focused (█).
5. **`05-model-view-scrolled-235x52`**. The Model view scrolled at 235x52 (Top K first, Model row hidden).
6. **`06-context-view-opens-at-model-capacity-235x52`** (click Context and memory). Model capacity first, Budget strategy focused (█). At this size the Context view fits, so it cannot be scrolled before a switch back.
7. **`07-missing-key-chat-opens-on-its-fix-211x44`** (Ctrl+O, no key). Opening a not-ready chat: the Model row is first, the tuning rows follow, Connection is open, and Configure credential… has focus (`.ansi.txt` only).
8. **`08-missing-key-context-view-scrolled-211x44`** (click Context and memory, then scroll). The Context view scrolled to its end.
9. **`09-missing-key-switch-lands-as-open-211x44`** (click Model and generation). The switch lands as opening does: 09 is byte-identical to 07 in both the `.txt` and the `.ansi.txt`, including the focus escape on Configure credential…. Before the round-1 fix this switch left the body scrolled 9 rows, with Configure credential… at the top of the viewport and the Model row and tuning rows above the fold (the mounted test's RED run on 1ad693fcb2).
10. **`10-missing-key-switch-lands-as-open-235x52`** (resize, click Context and memory, click Model and generation). The same at 235x52: Model row first, Configure credential… focused (`.ansi.txt` only).

Not this task's: the closed Sampling title wraps to two rows for Anthropic in 04, 07, 09 and 10 (01 and 05 are scrolled past it). That is TASK-33006.2's title as committed at this task's base, and it predates the owner ruling of 2026-10-02 that every closed disclosure title stays one row. This task relies only on which section each view opens at and where focus lands.
