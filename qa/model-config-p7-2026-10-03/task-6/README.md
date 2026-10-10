# TASK-33007.6: Advanced as one-row disclosures, one frame level (2026-10-04)

Keyboard captures of Settings ▸ Providers & Models with the real stylesheet. All five
show the code at `7adb151708`, Task 6's last code commit, and were taken after it was
made. A first set was taken at `9b7cf1e9d1`, before the scrollbar-gutter fix. Against
that set, 03, 04 and 05 came out byte-identical. 01 and 02 differ only by the reserved
one-column gutter: the Key check's `Test (t)` and one ellipsized help line moved left by
one cell. The `.txt` files are colour-stripped (`tmux capture-pane -p`); the
`.ansi.txt` files keep the colour escapes (`-e`).

## Procedure

- The real app ran in an isolated tmux server (`-L p7t6live`) under `env -i`. `HOME`,
  `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` pointed into a scratch
  directory, and the keyring backend was the null one. `TERM=xterm-256color` and
  `COLORTERM=truecolor` were set.
- The two scratch profiles were written from the shipped template. The template came
  from a process whose own `HOME`/`XDG_*` pointed at a second scratch directory. Both
  profiles set `users_name = "verify_p7t6_live"`, `[first_run] setup_completed = true`,
  the splash off, `[model_catalog] auto_refresh_enabled = false`,
  `[console.onboarding] first_send_completed = true` and `[chat_defaults] temperature = 1.0`:
  - Anthropic: `chat_defaults` = `anthropic` / `claude-sonnet-4-5`, a fake
    `api_settings.anthropic.api_key`, and `model_defaults."claude-sonnet-4-5"` =
    `max_tokens = 8192`, `top_p = 0.9`.
  - llama.cpp: `chat_defaults` = `llama_cpp` / `qwen3-8b-q4`, `api_url =
    "http://127.0.0.1:18769"`, and `model_defaults."qwen3-8b-q4"` = `temperature = 0.6`.
    A loopback server on that port answered `GET /v1/models` with three ids:
    `qwen3-8b-q4`, `qwen3-14b-q4`, `llama-3.2-3b-instruct`.
- Each session starts the same way: wait for the nav bar, then **F4**, **F6**, **Down**,
  **Enter**, **Esc** (Providers & Models). A further **F6** focuses the Provider control,
  which is where the Tab counts below start.
- 03 continues the session that took 02. For every other capture the app was quit with
  **Ctrl+Q** and relaunched on its profile. Both profiles were re-written from the template
  before 01.
- Afterwards the tmux server was killed. A marker file was touched before the first
  launch. `find ~/.config/tldw_cli ~/.local/share/tldw_cli -newer <marker>` listed 0
  entries, and `~/.config/tldw_cli/config.toml` still reads 2026-09-26.

## Captures

- `01-anthropic-connect-through-advanced-closed-211x44` (AC#1, AC#2, AC#8, AC#13). The whole
  card is in the pane without scrolling: Connect, Default model for new chats, Model
  defaults, then the one-row **Advanced** header and five closed one-row titles, in
  order: "▶ Context window · 200,000 tokens · detected, no override", "▶ Saved model list ·
  15 saved in config · none discovered", "▶ Catalog refresh · applies immediately · startup
  refresh Off · per-provider choices not in effect", "▶ Custom endpoints · no named
  endpoints · applies immediately", "▶ Prompt-cache snapshots · llama.cpp only · Off".
  - Prompt-cache snapshots is no longer above Connect.
  - The card draws no frame inside the detail pane. Each section starts with a one-row
    header.
- `02-llamacpp-connect-through-advanced-closed-211x44` (AC#13). llama.cpp the same way.
  Context window reads "unknown · enter the model's documented limit"; Saved model list
  reads "0 saved in config".
- `03-llamacpp-saved-model-list-open-after-discovery-211x44` (AC#3, AC#4, AC#9, AC#12).
  - Keys: **Tab** x15 (to the Saved model list title), **Enter** (opens it), **Tab**,
    **Enter** (Discover models), a 7 s wait, **Tab** x3 (to the list), **Down**, **Space**.
  - The title reads "Saved model list · 0 saved in config · 3 discovered, 3 not saved ·
    1 selected". Each row says in words whether it is selected and whether it is saved:
    "selected · qwen3-8b-q4 · not saved · …" and "not selected · …".
  - The list has the card's control edge, thick while focused, and no box around it.
  - In the colour-stripped `.txt` every row's state is still readable (the `▐X▌` box is
    drawn on every row; only its colour differs).
- `04-llamacpp-catalog-refresh-open-235x52` (AC#5, AC#6, AC#9, AC#12).
  - A fresh 235x52 session. Keys: **Tab** x16 (to the Catalog refresh title), **Enter**.
  - Under the title: the "applies immediately - no Save needed" hint, then "▐X▌ Refresh on
    startup Off" and the line "Refresh on startup is Off, so the per-provider choices
    below are not in effect."
  - "Refresh after (hours) │ 24" is a one-row field. Then one row per provider, 30 in all:
    "▐X▌ OpenAI: refresh On ▐X▌ save to config Off".
  - The whole disclosure fits at 235x52.
- `05-search-opens-prompt-cache-snapshots-211x44` (AC#10).
  - Keys: **/**, type `snapshot keep`, **Enter**.
  - Field search opened the closed Prompt-cache snapshots disclosure and focused Keep
    count (thick edge). The Inspector names it "Focused setting: Snapshot keep count".
  - The snapshot checkbox is a one-row control with no box. Its label and the
    next-launch copy are unchanged.
