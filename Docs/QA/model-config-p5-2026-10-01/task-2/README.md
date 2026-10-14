# TASK-33005.2 live captures (2026-10-01)

Plain-text `tmux capture-pane -p` dumps of the real app, launched from this
worktree in an isolated tmux server at 211x44 (the last view after resizing
the window to 235x52). The profile was disposable: `HOME`,
`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH`
pointed at a scratch directory, `[paths] data_dir` and `users_name` were
scratch values, the keyring backend was the null one, and the real
`~/.config/tldw_cli/config.toml` hashed the same before and after.

The scratch config selected llama.cpp at `http://127.0.0.1:9199`, a port with
nothing listening, so a test is refused. For the retry, a throwaway local HTTP
server answered `GET /v1/models` on that port with one model (`model-a`).

Fresh profile (no message ever sent):

0. `0-console-ready-before-test-211x44.txt`: the Console before any test
   (a startup toast covers the header): "Ready — type a message to begin."
1. `1-chat-settings-test-refused-211x44.txt`: Chat settings (Ctrl+O),
   **Test connection & list models**: "Connection test failed: connection
   refused."
2. `2-console-first-run-card-refused-211x44.txt`: Esc back to the Console.
   It read Ready before the test; now the Get started card's first step is
   "Reconnect the provider server" with **Retry connection**.
3. `3-console-retry-still-refused-211x44.txt`: **Retry connection** with the
   server still down: the notice "llama.cpp is still unreachable. Start it,
   then retry." and the card stays.
4. `4-console-retry-restored-211x44.txt`: the server started, one more click
   on **Retry connection**: header Ready, card gone, composer unlocked.

Profile with `[console.onboarding] first_send_completed = true`, Model
section expanded, Inspector open (Alt+I):

5. `5-console-refused-after-settings-t-211x44.txt`: Settings (F4) ▸
   Providers & Models, `t` (refused), then the Console tab: header Blocked,
   Model section "Not ready — endpoint unreachable", **Retry connection** in
   the empty transcript, Inspector "Run: Recovery required", composer "Send
   blocked — retry the connection to continue".
6. `6-console-retry-restored-rails-211x44.txt`: the server started, one
   click on **Retry connection**: Ready everywhere, Send unblocked.
7. `7-console-refused-after-settings-t-235x52.txt`: the same refused state
   at 235x52 after the server was stopped and `t` run again; the Inspector
   shows "Next action: Retry connection".

Fix round 1 (review I-1), a keyed local server. Same isolation, a new
scratch profile (`users_name = "verify_t33005_2r1"`, `first_send_completed`),
vLLM at `http://127.0.0.1:9198` with a saved `api_key`, and a throwaway local
server that answered `GET /v1/models` only for `Authorization: Bearer <its
key>`. Its request log recorded whether each request carried the saved key
(`match`), another key (`other`) or none; it never recorded `none`. Before
this round, the probe sent no key, so this server answered 401 and the
Console read "key rejected" for the right key.

8. `8-console-keyed-vllm-ready-before-test-211x44.txt`: the Console before
   any test, Model section expanded: Ready.
9. `9-chat-settings-keyed-vllm-test-reachable-211x44.txt`: Chat settings
   (Ctrl+O), **Test connection & list models**: "Connection test succeeded;
   1 model listed." (server log: `/v1/models auth=match`).
10. `10-console-keyed-vllm-ready-after-test-211x44.txt`: Esc: the Console
    still reads Ready, with no recovery line.
11. `11-console-keyed-vllm-key-refused-211x44.txt`: the server restarted to
    require a different key, the same test again ("Connection test failed:
    unauthorized.", server log `auth=other`), then Esc: header Blocked, Model
    section "Not ready — vLLM credential was rejected". This verdict is about
    the key that was actually sent.
12. `12-console-keyed-vllm-ready-after-retest-211x44.txt`: the server back on
    the saved key, the same test ("Connection test succeeded", `auth=match`),
    then Esc: Ready again, no restart.

Reading notes for reviewers:

- The status strip's "Agent blocked" (in "Library · Auto off · Agent
  blocked") is the agent's Library access policy chip. It has nothing to do
  with provider readiness, so it reads the same in the Ready and Not ready
  captures.
- Every on-screen string these captures rely on is asserted against the real
  widgets in `Tests/UI/test_console_endpoint_discovery.py`
  (`test_refused_chat_settings_test_blocks_console_until_one_retry`: header
  status Ready/Blocked/Ready, Model section, readiness row, Retry button,
  composer reason) and `test_a_keyed_local_server_is_tested_with_the_key_a_send_uses`.

The readiness words themselves ("Not ready · refused :9099" and the rest of
the spec §5 vocabulary) are TASK-33005.3's; these captures show today's words.
