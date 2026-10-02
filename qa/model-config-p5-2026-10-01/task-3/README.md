# TASK-33005.3 live captures (2026-10-01)

Plain-text `tmux capture-pane -p` dumps (one `-e` ANSI dump) of the real app,
launched from this worktree in an isolated tmux server at 211x44 (capture 9
after resizing the window to 235x52). The profile was disposable: `HOME`,
`XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH`
pointed at a scratch directory, `[paths] data_dir` and
`users_name = "verify_t33005_3"` were scratch values, the keyring backend was
the null one, and the real `~/.config/tldw_cli/config.toml` hashed the same
before and after (as did the `~/.local/share/tldw_cli` listing).

The scratch config selected llama.cpp at `http://127.0.0.1:9199`, a port with
nothing listening, so a test is refused. For the reachable views a throwaway
local HTTP server answered `GET /v1/models` on that port with one model
(`model-a`). Driver scripts and the server lived in the scratchpad and are not
committed.

Profile with `[console.onboarding] first_send_completed = true`, Model section
expanded:

1. `1-console-ready-not-tested-211x44.txt`: the Console before any test. The
   Model section line and the new readiness chip after the Model chip both
   read "Ready · not tested".
2. `2-settings-t-refused-211x44.txt`: Settings (F4) ▸ Providers & Models, `t`:
   the Test result leads with "Readiness   Not ready · refused :9199", then the
   five fact rows.
3. `3-console-refused-after-settings-t-211x44.txt` (and `.ansi.txt`): the
   Console tab: the Model section line and the chip read "Not ready · refused
   :9199". In the ANSI dump only the Model section line carries the readable
   error colour (`38;2;209;126;146`); the word says the same without it.
4. `4-switcher-refused-211x44.txt`: Switch model (Alt+M): llama.cpp's rows
   read "Not ready · refused :9199" (RECENT and NEEDS SETUP).
5. `5-console-retry-reachable-211x44.txt`: the server started, one click on
   **Retry connection**: Model section and chip read "Ready · reachable 17:41"
   (the local time the listing answered).
6. `6-chat-settings-reachable-211x44.txt`: Chat settings (Ctrl+O): the
   readiness block leads with "Ready · reachable 17:41", then "Endpoint ·
   Reachable", "Model · Confirmed", "Generation · Not tested".
7. `7-switcher-reachable-211x44.txt`: Switch model: "Ready · reachable 17:41".
8. `8-settings-return-reachable-211x44.txt`: back to Settings (rebuilt on the
   visit): the Test result adopts the shared result, "Readiness   Ready ·
   reachable 17:41", "Endpoint … model listing reached".
9. `9-console-reachable-235x52.txt`: the Console at 235x52, same words.

Profile with `first_send_completed = false` (a new launch, so nothing tested):

10. `10-console-setup-card-refused-211x44.txt`: Chat settings **Test
    connection & list models** (refused), then Cancel: the Get started card's
    active step reads "1. ● Reconnect the provider server" with "Not ready ·
    refused :9199" on its own line under it.

Reading notes for reviewers:

- "Agent blocked" in the status strip ("Library · Auto off · Agent blocked")
  is the agent's Library access policy chip, unrelated to provider readiness.
- The header word (Ready/Blocked) is the Console run status; the readiness
  word is the chip after "Model: model-a".
- Every string these captures rely on is asserted against the real widgets in
  `Tests/UI/test_console_endpoint_discovery.py`
  (`test_refused_chat_settings_test_blocks_console_until_one_retry`: chip,
  Model section line and its `-blocked` class, Conversation settings row,
  setup card step, switcher word, Chat settings readiness) and
  `Tests/UI/test_settings_provider_test_draft.py`
  (`test_settings_test_result_reaches_chat_settings_for_the_same_connection`:
  Settings Readiness row = Console word = Chat settings word).
