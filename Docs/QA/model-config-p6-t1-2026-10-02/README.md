# TASK-33006.1 Chat settings core-first captures (2026-10-02)

Chat settings (Ctrl+O) after TASK-33006.1, captured at 211x44 and 235x52 from the real app in this worktree. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold text on the focus fill.

## Setup

- The app ran in an isolated tmux server (`-L cap33006t1`) under `env -i`, with `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` in a scratch directory and the null keyring backend.
- Both scratch profiles set `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false` and the splash off.
- Profile A (`verify_cap33006_1`): llama.cpp at `http://127.0.0.1:9199` (nothing listening, never tested), chat default `model-a`, `chat_defaults` Temperature 0.4 and Max tokens 2048.
- Profile B (`verify_cap33006_1b`): chat default OpenAI `gpt-4o` with no key, so the chat is Not ready.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7) and the `~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the same before and after. No provider was contacted. The driver is not committed.

## Captures

1. `01-model-view-open-211x44`: Ctrl+O on a ready chat. The 150x22 frame shows the scope line, the Model row, the CORE rows (Temperature, Max tokens, Streaming On/Off, and the two controls llama.cpp takes), then the closed Sampling, Connection, Request estimate and "Your name in this chat" rows, with the footer, without scrolling. Each row reads label, value, Source word, help line. Focus is on Temperature. The defaults line ("Used by future conversations for llama.cpp.") is the existing footer behaviour that TASK-33006.6 and .5 own.
2. `02-model-view-keyboard-edits-211x44`: Ctrl+A, `0.9`, Tab, Tab, Enter, Down, Enter. Temperature paints 0.9 and Streaming paints Off, each with the Source word "edited *"; the Esc hint reads "Esc close (asks: 2 unsaved)".
3. `03-sampling-opened-from-its-title-211x44`: Tab ×3 to the Sampling title, Enter. The six Sampling rows open in place with the same grammar and columns as the CORE rows; blank fields read "blank = provider default". The body now scrolls, with the fold hint.
4. `04-sampling-opened-235x52`: the same moment after resizing to 235x52. The frame stays 150x22.
5. `05-model-view-open-235x52`: Esc, `d` (discard), Ctrl+O again at 235x52.
6. `06-not-ready-opens-connection-211x44`: profile B, Ctrl+O. The chat is Not ready (no key), so Connection opens by itself and "Configure credential…" has focus (bold on the focus fill in the `.ansi.txt`); Sampling stays closed. The OpenAI choice rows still carry the "Support not verified" note that TASK-33006.2 folds into the help line.
7. `07-not-ready-opens-connection-235x52`: the same at 235x52.

These captures show the modal as TASK-33006.1 left it. Later Phase 6 tasks changed parts of it:

- TASK-33006.4 removed Connection's provider and model pickers.
- TASK-33006.5 renamed the header and the footer actions.
- In the final fix wave, a blank choice Select (a dropdown) shows "default" instead of "Select", and a blank row whose value no layer holds reads "provider" instead of "built-in".

The current layout is in the retaken captures of `qa/model-config-p6-t2-2026-10-02/` through `qa/model-config-p6-t7-2026-10-03/`.
