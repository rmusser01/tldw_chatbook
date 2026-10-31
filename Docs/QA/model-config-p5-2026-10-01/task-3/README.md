# TASK-33005.3 live captures (2026-10-01, review round 1)

Plain-text `tmux capture-pane -p` dumps (one `-e` ANSI dump) of the real app,
launched from this worktree in an isolated tmux server at 211x44 (capture 9
after resizing the window to 235x52, capture 11 at 83x30). The profile was
disposable: `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` and
`TLDW_CONFIG_PATH` pointed at a scratch directory, `[paths] data_dir` and
`users_name = "verify_t33005_3r1"` were scratch values, the keyring backend
was the null one, and the real `~/.config/tldw_cli/config.toml` hashed the
same before and after (as did the `~/.local/share/tldw_cli` listing).

The scratch config selected llama.cpp at `http://127.0.0.1:9199`, a port with
nothing listening, so a test is refused. For the reachable views a throwaway
local HTTP server answered `GET /v1/models` on that port with one model
(`model-a`). Driver scripts and the server lived in the scratchpad and are not
committed.

These captures replace the first round's. The readiness word now sits in the
header's status badge (top right), not in a status-strip chip, so the strip
ends with its Context/cost chip again ("Context 0% · Current $0.00 · O…", as
before this task).

Profile with `[console.onboarding] first_send_completed = true`:

1. `1-console-ready-not-tested-211x44.txt`: the Console before any test, on
   its first frame. The header badge and the Model section line both read
   "Ready · not tested".
2. `2-settings-t-refused-211x44.txt`: Settings (F4) ▸ Providers & Models, `t`:
   the Test result leads with "Readiness   Not ready · refused :9199", then the
   five fact rows.
3. `3-console-refused-after-settings-t-211x44.txt` (and `.ansi.txt`): the
   Console tab: the header badge and the Model section line read "Not ready ·
   refused :9199". In the ANSI dump only the Model section line carries the
   error colour (`38;2;209;126;146`); the badge word says the same without it.
4. `4-switcher-refused-211x44.txt`: Switch model (Alt+M): llama.cpp's rows
   read "Not ready · refused :9199" (RECENT and NEEDS SETUP).
5. `5-console-retry-reachable-211x44.txt`: the server started, one click on
   **Retry connection**: header badge and Model section read "Ready ·
   reachable 21:04" (the local time the listing answered).
6. `6-chat-settings-reachable-211x44.txt`: Chat settings (Ctrl+O), scrolled
   to the readiness block: "Ready · reachable 21:04", then "Credential · Not
   required", "Endpoint · Reachable", "Model · Confirmed", "Generation · Not
   tested".
7. `7-switcher-reachable-211x44.txt`: Switch model: "Ready · reachable 21:04".
8. `8-settings-return-reachable-211x44.txt`: back to Settings: the Test result
   reads the shared result, "Readiness   Ready · reachable 21:04", "Endpoint …
   model listing reached".
9. `9-console-reachable-235x52.txt`: the Console at 235x52, same words.
11. `11-console-single-pane-83x30.txt`: the same session at 83 columns
    (single-pane layout). The word does not fit the header row there, so the
    badge keeps the short status, "Ready". Widening back to 211 restored
    "Ready · reachable 21:04".

Profile with `first_send_completed = false` (a new launch, nothing tested):

10. `10-console-setup-card-refused-211x44.txt`: Chat settings **Test
    connection & list models** (refused), then Cancel: the Get started card's
    active step reads "1. ● Reconnect the provider server" with "Not ready ·
    refused :9199" on its own line under it.

Reading notes for reviewers:

- "Agent blocked" in the status strip ("Library · Auto off · Agent blocked")
  is the agent's Library access policy chip, unrelated to provider readiness.
- Every string these captures rely on is asserted against the real widgets in
  `Tests/UI/test_console_endpoint_discovery.py`
  (`test_refused_chat_settings_test_blocks_console_until_one_retry`: header
  badge, Model section line and its `-blocked` class, Conversation settings
  row, setup card step, switcher word, Chat settings readiness),
  `Tests/UI/test_settings_provider_test_draft.py`
  (`test_settings_test_result_reaches_chat_settings_for_the_same_connection`:
  Settings Readiness row = Console word = Chat settings word) and
  `Tests/UI/test_console_narrow_layout.py`
  (`test_console_header_subtitle_yields_width_before_fixed_controls`: the
  word at 140 columns, "Ready" at 60).
