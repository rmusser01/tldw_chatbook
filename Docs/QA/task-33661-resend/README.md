# TASK-33661 live check: Resend a broken Console turn in place

Captured 2026-10-01 on `feat/task-33661-console-resend` (worktree build), full
screen at 211x44 and 235x52 (`tmux capture-pane -p`, plus `-e` for the
`.ansi.txt` copies).

## Procedure

- The app ran from the worktree (`python -m tldw_chatbook.app`, `PYTHONPATH` set
  to the worktree) inside a private tmux server. HOME, USERPROFILE and the XDG
  config, data and cache directories all pointed into a scratch directory. A
  scratch `TLDW_CONFIG_PATH` set a unique `[general] users_name`, a scratch
  `[paths] data_dir`, `first_run` done, the splash screen off, and
  `[chat_defaults] provider = "llama_cpp"` with
  `[api_settings.llama_cpp] api_url = "http://127.0.0.1:9233"`.
- No real provider was contacted. The broken shapes came from a closed port
  (readiness refusal) or a small local llama.cpp-shaped stub on that port,
  switched between "ok", "error" (HTTP 500) and "hang".
- Rows were selected with SGR mouse clicks sent through tmux, or with `j`/`k`.
  Resend was pressed by click or with `r`. The restart shapes quit and
  relaunched the app on the same scratch profile. The dispatch-recovery shape
  killed the app process during a hanging reply and relaunched it.
- Afterwards the real `~/.config/tldw_cli/config.toml` hash and the
  `~/.local/share/tldw_cli` listing matched their pre-run values. The scratch
  profile was deleted. The stub and the driver helpers are not committed.

## Captures

| File | Shape | Shows |
|---|---|---|
| `a1-refused-echo-selected-211x44` | Refused at readiness | User row: `Copy Edit Fork Resend More…`; guide `r Resend`; the "Unsent turn" shelf |
| `a2-refused-echo-selected-again-211x44` | Same, fixed build | Same row after the sync-timer fix |
| `a3-refused-echo-resent-211x44` | `r` on the echo | One user message and its reply; block row, shelf and composer copy gone |
| `a4`/`a5-…-235x52` | Same at 235x52, by click | Same result |
| `b1-provider-error-user-selected-211x44` | Provider HTTP 500 | User row offers Resend |
| `b2-provider-error-assistant-selected-211x44` | Same | Failed reply offers Retry; guide `r Retry` |
| `b3`/`b4-…-235x52` | Click Resend | Reply retried in place; "Agent run failed" row cleared |
| `c1`/`c2-empty-stopped-…-235x52` | Stop before any text, then `r` | Stopped reply and "Response stopped by user." cleared; new reply under the same message |
| `d1`/`d2-restored-failed-…-235x52` | Failed reply after a relaunch ("Response failed.") | Resend offered; re-run in place |
| `e1-dispatch-recovery-user-selected-235x52` | Unresolved dispatch recovery after a crash | No Resend while the card is unresolved |
| `e2`/`e3-dispatch-recovery-…` | Card's Discard after a relaunch | Pre-existing: "That response recovery action is unavailable." (not this change; see the task notes) |
| `z-control-restore-then-send-leaves-duplicate-211x44` | Control: shelf Restore + Enter | The old path leaves the failed echo and block row behind, so the message appears twice |

The first refused-echo run (before the fix) left the transcript frozen on
"Generating…" after the turn completed. The worker started the sync timer
before the normal send path ran, and a timer tick stopped it while the session
still read blocked. The fix leaves that timer to the send path. `a2`–`a5` were
captured on the fixed build, and
`test_resend_worker_leaves_a_refused_echo_sync_timer_to_the_send_path` pins
the fix.
