# TASK-33005.5 live captures (2026-10-02)

Each file is a `tmux capture-pane -p` dump of the real app (`.ansi.txt`: the
same moment with `-e` colour escapes), launched from this worktree
(`tldw_chatbook.__file__` checked) in an isolated tmux server at 211x44;
capture 7 is after resizing the window to 235x52.

Setup:

- Disposable profile: `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`,
  `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH` pointed at a scratch directory,
  `[paths] data_dir` was scratch, `users_name = "verify_t33005_5"`, null
  keyring backend. The first-run wizard was skipped and the startup "Check
  model lists online?" prompt answered **Don't check**.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7)
  and the `~/.local/share/tldw_cli` listing digest (75c3c25369a989e4) were the
  same before and after.
- The scratch config saved llama.cpp at `http://127.0.0.1:9199` (nothing
  listening: "llama.cpp stopped"; 9099 was taken by another process on this
  machine) as the chat default with `model-a`, and Ollama at
  `http://127.0.0.1:11435`. A real `ollama serve` (OLLAMA_HOST
  127.0.0.1:11435, its one local model `qwen2.5:0.5b`) ran for the session
  and was stopped afterwards. Every other provider kept the shipped defaults.
- Ollama's own request log (`[GIN] ... GET "/v1/models"`) was read to count
  the switcher's probes. No driver script is committed.

Captures:

1. `1-console-before-switcher`: the Console on llama.cpp before any probe.
   The header reads "Ready · not tested".
2. `2-switcher-first-frame`: 0.3 s after Alt+M. The switcher is already
   painted and focused while readiness and the probes run in workers.
3. `3-switcher-llama-refused-ollama-reachable` (and `.ansi.txt`): about 4 s
   later. llama.cpp's rows read "Not ready · refused :9199"; its NEEDS SETUP
   row leads the group and reads "start it; rechecked on open". Ollama reads
   "Ready · reachable 02:27". The shipped `localhost` defaults of Aphrodite,
   TabbyAPI and the two custom slots read "refused :PORT" too (this round's
   `connect_error_is_refused` fix; before it they read "unreachable").
   vLLM stays "Ready · not tested": another process on this machine answers
   404 at its default `localhost:8000`, which is "model listing unavailable",
   not "reachable". The cloud rows still read "no key"; none was contacted.
   **Superseded for the header (final review I-5):** this frame's header,
   beside these rows, still read "Ready · not tested", because the Console
   under the switcher was refreshed only after Esc. The final fix wave
   refreshes it on each probe result; see `../final-fix/` captures 2 and 5.
   The NEEDS SETUP heading has since become "Enter opens the fix or explains
   it".
4. `4-console-status-refused-after-switcher`: Esc. The Console header (this
   chat's status) reads "Not ready · refused :9199" and the composer "Send
   blocked — retry the connection to continue".
5. `5-switcher-on-ollama-reopened-inside-window`: the chat switched to
   `qwen2.5:0.5b` (Find "qwen", Enter). Alt+M opened at 02:28:47 (one Ollama
   GET at 02:28:48), Esc, then Alt+M again at 02:28:50: Ollama logged no
   second GET (the 10 s window). llama.cpp is no longer this chat's provider,
   so its NEEDS SETUP row reads "(any model) llama.cpp Not ready · refused
   :9199 start it; rechecked on open", as in mockup (a).
6. `6-switcher-enter-on-refused-row` (and `.ansi.txt`): Find "llama", the
   llama.cpp NEEDS SETUP row highlighted, Enter. The switcher stays open and
   says "llama.cpp did not answer: start it; rechecked on open." Nothing
   navigated to Settings.
7. `7-switcher-235x52`: the same switcher at 235x52, same words.

Reading note: "Agent blocked" in the status strip is the Library access
policy chip, unrelated to provider readiness.

## Re-run for the request-count claim (review fix round 1, 2026-10-02)

Capture 5's "one GET per window" rested on Ollama's log, which was not
committed. It was re-run with the same setup: a fresh scratch profile
(`users_name = "verify_t33005_5fix"`, `[model_catalog] auto_refresh_enabled
= false`, `[first_run]` marked complete, null keyring), a real `ollama serve`
on `127.0.0.1:11435` as this chat's provider (`qwen2.5:0.5b`), llama.cpp at
`127.0.0.1:9199` with nothing listening, and a separate tmux server at 211x44.
The real `~/.config/tldw_cli/config.toml` (15c6cb224a6a51c7) and the
`~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the same before
and after.

8. `8-ollama-request-log.txt`: every `[GIN]` line Ollama logged in the
   session, unedited. 03:10:55 is a `curl` readiness check made before the
   app started. Alt+M at 03:12:15 produced the GET at 03:12:16. Esc, then
   Alt+M again at 03:12:21, 6 s later and inside the 10 s window, produced
   none. Esc, then Alt+M at 03:12:36, past the window, produced the GET at
   03:12:36. The app sent nothing else to Ollama.
9. `9-switcher-reopened-inside-window-211x44.txt`: the second open
   (03:12:21). The rows read the first probe's result: Ollama "Ready ·
   reachable 03:12" and llama.cpp "Not ready · refused :9199". The other
   Ollama model rows come from the shipped `[providers]` list, not from the
   server.
10. `10-switcher-reopened-after-window-211x44.txt`: the third open
    (03:12:36), after the re-probe. The minute is unchanged, so the words
    are unchanged.
