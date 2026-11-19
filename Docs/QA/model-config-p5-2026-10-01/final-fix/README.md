# Phase 5 final fix wave: live captures (2026-10-02)

Each file is a `tmux capture-pane -p` dump of the real app (`.ansi.txt`: the
same moment with `-e` colour escapes), launched from this worktree after the
rebase onto dev (`tldw_chatbook.__file__` checked) in an isolated tmux server
at 211x44; captures 6 and 7 are after resizing the window to 235x52.

Setup:

- Disposable profile: the app ran under `env -i` with `HOME`,
  `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME` and `TLDW_CONFIG_PATH`
  in a scratch directory, `users_name = "verify_t33005_final"`, the null
  keyring backend, `[first_run] setup_completed = true` and
  `[model_catalog] auto_refresh_enabled = false`. No provider env var was set.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7)
  and the `~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the
  same before and after.
- The scratch config saved llama.cpp at `http://127.0.0.1:9199` (nothing
  listening), Ollama at `http://127.0.0.1:11435` and a fake OpenAI key. A real
  `ollama serve` (OLLAMA_HOST 127.0.0.1:11435, model `qwen2.5:0.5b`) ran while
  stated below. No driver script is committed.

Run A, the task-5 capture 3 scenario (llama.cpp is this chat's provider, the
transcript is empty):

1. `1-console-llama-before-switcher`: header "Ready · not tested".
2. `2-switcher-llama-refused-ollama-reachable`: about 5 s after Alt+M.
   llama.cpp reads "Not ready · refused :9199" on its CURRENT row and leads
   NEEDS SETUP ("Enter opens the fix or explains it"); Ollama reads "Ready ·
   reachable 06:00". The probe result now also refreshes the Console under
   the switcher (final review I-5), so the Console has already turned to its
   blocking Get started card, whose backdrop covers the header: no surface in
   this frame says "Ready · not tested" any more. In task-5 capture 3, taken
   before the fix, the header beside these rows still read it.
3. `3-console-get-started-refused-after-esc`: Esc. The card's step 1 reads
   "Reconnect the provider server · Not ready · refused :9199" with **Retry
   connection**.

Run B, the same check with a transcript, so the header stays visible under
the switcher:

4. `4-console-ollama-chat-before-switcher`: chat default Ollama; one message
   sent and answered (the POST in capture 8). Header "Ready · not tested".
5. `5-switcher-open-header-agrees`: Ollama stopped, then Alt+M. With the
   switcher still open, the header reads "Not ready · refused :11435", the
   same words as the switcher's CURRENT row and its NEEDS SETUP row
   (TASK-30011 AC#6, parent AC#7).
6. `6-settings-t-openai-key-rejected-235x52`: resized to 235x52, F4 ▸
   Providers & Models, OpenAI (draft, not saved) on its shipped endpoint,
   **t**. A real request to `https://api.openai.com/v1/models` with the fake
   key; OpenAI answered 401. "Readiness   Not ready · key rejected".
7. `7-switcher-openai-key-rejected-235x52`: back in the Console, Alt+M, Find
   "gpt-5.6-terra": the OpenAI row reads "Not ready · key rejected · Enter:
   add key in Settings".
8. `8-ollama-request-log.txt`: every request Ollama logged, unedited. 05:59:39
   is a `curl` readiness check before the app started; 06:00:12 is run A's
   switcher probe; 06:02:21-27 is run B's send. Nothing reached Ollama after
   it was stopped.

The "valid cloud key" case still has no real key behind it (see
qa/model-config-p5-2026-10-01/task-4/README.md); that is an owner call,
TASK-33005.6.
