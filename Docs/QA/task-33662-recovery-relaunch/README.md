# TASK-33662 live check: the recovery card settles after a real relaunch

Captured 2026-10-02 on `fix/task-33662-recovery-after-relaunch` (worktree
build, base `ef8fd5d38a`), full screen at 211x44 (`tmux capture-pane -p`, plus
`-e` for the `.ansi.txt` copies). This re-runs the e2/e3 shape from
`qa/task-33661-resend/`, which showed "That response recovery action is
unavailable." before the fix.

## Procedure

- The app ran from the worktree (`python -m tldw_chatbook.app`, `PYTHONPATH` set
  to the worktree) inside a private tmux server (`tmux -L recovery33662`).
  HOME, USERPROFILE and the XDG config, data and cache directories all pointed
  into a scratch directory. A scratch `TLDW_CONFIG_PATH` set a unique
  `[general] users_name`, a scratch `[paths] data_dir`, first run done, the
  splash screen off, and `llama_cpp` / `model-a` at a local llama.cpp-shaped
  stub on `127.0.0.1:9262`.
- No real provider was contacted. The stub numbers its replies ("Stub reply
  #N") and switches between "ok" and "hang".
- A crash is `kill -9` of the app process while the stub hangs a reply. The
  relaunch uses the same scratch profile, and the conversation is reopened from
  the Chats list.
- Afterwards the real `~/.config/tldw_cli/config.toml` hash and the
  `~/.local/share/tldw_cli` listing matched their pre-run values. The scratch
  profile was deleted. The stub and the driver helpers are not committed.

## Captures

| File | Step | Shows |
|---|---|---|
| `r1-in-flight-before-kill` | Turn two hangs | "Generating…" just before the kill |
| `r2-relaunched-recovery-card` | Relaunch, reopen | Card: "Response delivery status is unknown on the source device.", **Retry anyway** / **Discard**; composer "Send blocked — resolve response recovery first" |
| `r3-retry-anyway-streamed-into-pending-reply` | Click **Retry anyway** | "Stub reply #2" in the same pending reply under "second question"; card gone, Send unblocked |
| `r4-second-crash-recovery-card` | Turn three hangs, kill, relaunch, reopen | The card again |
| `r5-discard-settled-after-relaunch` | Click **Discard** | "Response discarded." under the kept "third question"; card gone, Send unblocked |
| `r6-user-row-offers-resend` | Select "third question" | `Copy  Edit  Fork  Resend  More…`; guide `r Resend` |
| `r7-resend-re-ran-in-place` | Press `r` | "Stub reply #3" directly under "third question"; the discarded row is cleared and the row is healthy again (♻ / Continue) |
| `r8-database-rows-after-resend` | SQLite read of the scratch DB | One live reply under "third question"; the discarded reply is a soft-deleted tombstone; no open dispatch checkpoint |
| `q1-quit-dialog-mid-reply` | Ctrl+Q while a reply hangs | "Quit Chatbook?" with one live agent run |

The `r` series ran before the claim lookup was scoped to the session's own
tree (the second half of the fix). The `f` series re-ran the shapes on the
final build, on the same profile (the stub was restarted, so its numbering
starts again at #1):

| File | Step | Shows |
|---|---|---|
| `f1-final-build-relaunched-card` | "sixth question" hangs, kill, relaunch, reopen | The card |
| `f2-final-build-discarded-user-row-offers-resend` | **Discard**, select "sixth question" | "Response discarded."; `Copy  Edit  Fork  Resend  More…`; guide `r Resend` |
| `f3-final-build-resend-re-ran-in-place` | Press `r` | "Stub reply #1" directly under "sixth question" (database: one live reply, the discarded one tombstoned, no open checkpoint) |
| `f4-final-build-retry-anyway-streamed` | "seventh question" hangs, kill, relaunch, **Retry anyway** | "Stub reply #2" in the pending reply (database: one live complete reply, no open checkpoint) |

## Graceful quit (not a recovery shape)

Confirming **Quit** while a reply hangs settles that reply as `stopped` in the
database and leaves no dispatch checkpoint, so it is not a recovery shape (read
from the scratch database; that profile was not relaunched). The card comes
from a crash, like the killed process above.

Separately, the quit never finished: the process stayed alive after unmount.
In this build it stayed alive more than 2 minutes, including after the stub
was stopped. The merge-base build (`ef8fd5d38a`) did the same: it was still
alive 40 s after **Quit**, with the stub hanging. Both were killed by hand. The
hang predates this change and is outside it.
