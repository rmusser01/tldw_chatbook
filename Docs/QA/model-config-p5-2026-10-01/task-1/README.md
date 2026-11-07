# TASK-33005.1 live captures (2026-10-01)

Plain-text `tmux capture-pane -p` dumps of the real app, launched from this
worktree in an isolated tmux server at 211x44 (one view re-captured after
resizing the window to 235x52). The profile was disposable: `HOME`,
`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH`
pointed at a scratch directory, `[paths] data_dir` and `users_name` were
scratch values, the keyring backend was the null one, and the real
`~/.config/tldw_cli/config.toml` hashed the same before and after.

The scratch config selected llama.cpp at `http://127.0.0.1:9199`, a port with
nothing listening, so a test is refused.

1. `settings-test-refused-211x44.txt`: Settings > Providers & Models, after
   pressing `t`. The toast and the Endpoint row report the refused connection.
2. `chat-settings-shared-refused-211x44.txt`: Console, Ctrl+O. Chat settings
   ran no test of its own, yet reads "Not ready — endpoint unreachable" and
   "Endpoint · Unreachable — connection refused" from the shared owner.
3. `settings-return-shared-refused-211x44.txt` and
   `settings-return-shared-refused-235x52.txt`: back to Settings (F4), a
   freshly built screen. The Test rows show the refused result instead of
   "Configuration check has not run."

The Console status row still reads "Ready" in these captures: wiring the
Console's own readiness to the shared evidence is TASK-33005.2.
