# Phase 7 (TASK-33007) full-screen captures (2026-10-06)

The surfaces Phase 7 changed in Settings ▸ Providers & Models and Console Behavior,
captured from the real app at branch head `8a3d3395ec`. Every moment was captured at
211x44. The tmux window was then resized to 235x52, captured again, and resized back.
`.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour
escapes.

**Re-take (2026-10-09).** After review notes 1-8 were fixed (commit `f80fdc05e4`),
`01`, `01b`, `01c`, `02`, `03`, `03b`, `04`, `05a` and `07` were captured again the
same way and overwrote the earlier files. They were taken from the working tree just
before that commit; its last two edits (ruff formatting, and when the
restored-connection read starts) change nothing these profiles show, and a run on the
commit itself showed the same Anthropic card. `05b` and `06a`-`06c` are
still the `8a3d3395ec` captures: Context window and Console Behavior are review notes
9-11, fixed separately. The re-take differs from the procedure below only where
noted: tmux server `-L capA33007`, `users_name = "verify_capA33007_capfix"`, fake keys
of the same kind, and the picker steps of `02`/`03`/`03b` (see those captures). The
same checks held: no provider was contacted, no Save was made (only
`console.rail_state` differed from the seeded profiles), and the real profile's two
fingerprints below were unchanged.

**Second re-take of `04` (2026-10-09).** Temperature's help was cut at 211x44 to
"…higher is more var…". The shared field help is shorter now (`aa479f2482`), and `04`
was captured again at both sizes from the working tree just before that commit and
`87a1ea936d` (the Endpoint guide); the commits change nothing it shows. Same
procedure, with tmux server `-L capD33007`, `users_name = "verify_capD33007_helpfit"`
and fresh fake keys. The session replayed `02`, `03`, **r** and **Discard changes**
before `04`, as above, but not `07`'s wheel steps, so at 211 the Inspector rests a few
rows lower; the card differs from the earlier `04` only in Temperature's help. After
`04` the session reverted the edit, chose Azure OpenAI and tabbed to Endpoint: the
Inspector read "Validation: an http:// or https:// address; required: your resource
host", matching the row. That was reverted too. The same checks held: no provider
contacted, no Save, only `console.rail_state` differed from the seeded profile, and
both real-profile fingerprints were unchanged.

## Procedure

- **Isolation.** The app ran from this worktree in its own tmux server (`-L cap33007p`)
  under `env -i`. `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH`
  pointed into a scratch directory. The keyring backend was the null one, and
  `PYTHONPATH` was this worktree. `TERM=xterm-256color` and `COLORTERM=truecolor` were
  set.
- **Scratch profiles.** Three profiles were written from the shipped template. The
  template came from a process whose own `HOME`/`XDG_*` pointed at a second scratch
  directory. All three set `users_name = "verify_cap33007p_8a3d3395ec"`, the splash off,
  `[model_catalog] auto_refresh_enabled = false`, `[first_run] setup_completed = true`
  and `[console.onboarding] first_send_completed = true`. Each also has a made-up
  `api_key` in `[api_settings.openai]`, `[api_settings.anthropic]` and
  `[api_settings.azure]`. The profiles differ only in `[chat_defaults]`:
  - OpenAI / `gpt-5.6-terra`, the template's own pair;
  - `anthropic` / `claude-sonnet-4-5`;
  - `azure` / `my-gpt-deployment`, with no `api_base_url`, as shipped.
- **Opening the card.** Each session starts the same way: wait for the nav bar, then
  **F4**, **F6**, **Down**, **Enter**, **Esc** (Providers & Models).
- **Input.** Keys came from `tmux send-keys`. Clicks and wheel steps were SGR mouse
  sequences.
- **No provider was contacted.** There was no Test, no discovery and no send. No key
  appears in any capture. No Save was made: after the run, the only keys that differed
  from the seeded profiles were the Console's own `console.rail_state` entries.
- **Real profile unchanged.** Before and after the run, `shasum -a 256
  ~/.config/tldw_cli/config.toml` began `15c6cb224a6a51c7` (mtime Sep 26), and
  `ls ~/.local/share/tldw_cli | shasum -a 256` began `db7e7faf5bff92d2`.
- **Cleanup.** The tmux server was killed and the scratch directory deleted. The driver
  is not committed.

## Captures

### OpenAI session 1 (01-07)

- `01-connect-openai-saved-key`, at rest.
  - Connect for OpenAI with a saved key. Provider's help reads "3 of 60 configured ·
    listed first". API key reads **saved in config**, with **Clear**, the placeholder
    "Paste to replace" whole, and the help "masked · (t) test · (ctrl+l) clear".
  - Env var reads **not set**. Endpoint reads **built-in**. Key check reads "Ready ·
    not tested" with **Test (t)**. There is no restored-connection row: this profile
    restored nothing.
  - Below Connect: Default model, Applies to (one row), Model defaults, then the closed
    Advanced rows. The Inspector shows its Applies to and Next new chat blocks.
- `02-default-model-picker-filtered`. Keys: **F6**, **Tab** x4, then `mini` typed.
  - Tab x4 goes through API key, Env var, Endpoint and Model.
  - The list holds the four ids that match `mini`, under **Saved fallback**, with the
    first, `o4-mini-2025-04-16`, highlighted (in the `.ansi.txt`). The status line
    reads "4 found · Enter picks · Esc cancels".
  - The Inspector names the focused setting, Model.
- `03-applies-to-staged-model`. Keys: **Enter** (it picks the highlighted match),
  **Esc**.
  - Model reads `o4-mini-2025-04-16` **edited \***, and State reads "1 unsaved".
    Provider reads **new-chat default**.
  - Applies to reads "new chats; open chat “Chat 1” is unused and will use OpenAI ·
    o4-mini-2025-04-16." on one row at both sizes.
- `07-inspector-save-reach-staged`. The same staged state, after nine wheel steps up
  over the Inspector. The Inspector shows:
  - **Applies to**: New chats: yes (Ctrl+T, temporary, workspace) · Unused open chats:
    follow the saved default · Chats with work: keep their own; switch there with
    Alt+M · Model defaults: chats that switch to this model pick them up.
  - **Next new chat will use**: still the saved pair, OpenAI · gpt-5.6-terra, and
    "T 0.6 · max 4096 · stream Off".
  - "Unsaved edits apply only after save (s)."
- **r**, then a click on **Discard changes**, reverted the staged model.
- `04-model-defaults-temperature-edited`. A click on the Temperature field, then `0.4`
  typed.
  - The Temperature row reads 0.4 **edited \***, with its help in place of the
    inherits text, whole at both sizes: "Lower keeps replies focused; higher makes
    them varied." Provider and Model keep **new-chat default**.
  - The other rows are unchanged: Max tokens "inherits 4096 · provider"; Streaming,
    Reasoning effort, Reasoning summary and Verbosity are one-row Selects.
  - The closed Sampling row reads "OpenAI does not accept 4 fields (open to list them)"
    at 211 and names the four fields at 235.
- **Esc**, **r**, then a click on **Discard changes**, reverted the edit.
- `05a-advanced-closed`. The **Advanced** header and five one-row closed titles:
  - Context window: unknown;
  - Saved model list: 14 saved in config;
  - Catalog refresh;
  - Custom endpoints;
  - Prompt-cache snapshots.
- `05b-advanced-context-window-open`, after a click on the Context window title.
  - The warning line, then "Context window │ tokens (required when unknown)" and
    **Reset to detected**.
  - The capacity note follows. The other four titles stay closed below it.
- `06a-console-behavior-replay-selects`. A click on the rail's Console Behavior, then a
  click on the "Reasoning replay override" title.
  - Local reasoning history's **Replay** Select ("Automatic (recommended)") is one row.
  - So is the override's **Replay** Select ("Use default"), now open.
- `06b-console-behavior-rail-layout-scope`. Twelve wheel steps down. **Rail layout
  scope** ("Global") is a one-row Select, and so is **Trace viewer** ("Safe").
- `06c-console-behavior-global-fallbacks`. Wheel steps down to Global fallback defaults.
  The 235x52 frame was scrolled again after the resize, because the reflow moved it.
  - The one-row rows: Temperature 0.6 (**Console Behavior**); Max tokens (**provider**,
    "blank = provider default"); Streaming On (**built-in**, "not set here · a
    provider's own setting comes first").
  - Reasoning effort, Reasoning summary, Verbosity and Thinking read "default".
    Thinking budget reads ">= 1024".
  - The closed row reads "▶ Sampling · Top P 0.95 · Min P 0.05 · Top K 50".
  - The Background effects Selects are one row each.

### Anthropic and Azure sessions

- `01b-connect-anthropic-sign-in-with`, at rest.
  - **Sign in with** is one row: the Select ("API key"), **built-in**, and the help
    "bills API credits through your key".
  - Then API key (**saved in config**, the same hint), Env var `ANTHROPIC_API_KEY`,
    Endpoint `https://api.anthropic.com` **built-in**, and Key check "Ready · not tested".
- `01c-connect-azure-saved-key`, at rest, with no base URL.
  - API key reads **saved in config**, with **Clear** and the hint.
  - Env var reads `AZURE_OPENAI_API_KEY` **not set**. Key check reads "Not ready · no URL".
  - The Endpoint row reads placeholder "Enter your resource host", **not set** and
    "required: your resource host".

### OpenAI session 2 (03b)

- `03b-applies-to-chat-with-work`.
  - In the Console: **Ctrl+O**, Temperature changed from 0.6 to 0.9, and Apply to this
    chat (`\e[13;5u`).
  - Then Providers & Models, **F6**, **Tab** x4, `o3-mini` typed ("1 found · Enter
    picks · Esc cancels"), **Enter**, **Esc**: `o3-mini-2025-01-31` is staged.
  - Applies to reads "new chats; open chat “Chat 1” keeps OpenAI · gpt-5.6-terra." on
    one line.

## Review notes (what looks wrong at full screen)

1. **Azure Endpoint row contradicts the Key check** (`01c`). With no base URL, the row
   says **built-in** and "blank uses the provider default", but Azure ships no default
   and the Key check reads "Not ready · no URL". The cause is that `endpoint_row_copy`
   uses the "required" wording only for `API_URL_PROVIDER_KEYS`, the local servers.
   Cloudflare and Databricks are likely to read the same way.
   **Fixed** (`01c`): Azure, Cloudflare and Databricks read **not set**, "required: your
   resource host" (account URL, workspace host) and an "Enter your …" placeholder.
2. **Provider reads "edited \*" when it was not edited** (`03`, `04`). The cause is
   `_resolve_provider_model_for_settings`. It marks Provider and Model as
   `settings_draft` whenever the draft carries a value for them, and the draft carries
   every value. So staging only a Model, or only a Temperature, makes the Provider row
   (and, in `04`, the Model row too) read **edited \***, while State says "1 unsaved".
   **Fixed** (`03`, `04`): each reads **edited \*** only when its own value differs
   from the saved one.
3. **"Review restored OpenAI connection" shows for every OpenAI profile** (`01`, `02`).
   It shows even on a fresh profile with nothing restored. It is a Tab stop between
   Endpoint and Model. While it has focus, the Inspector says "Focused setting: None —
   Tab to a setting". The stop is pinned on purpose (`_OPENAI_STOPS`), and the button
   predates Phase 7. It is still the only row in Connect without the label/control
   grammar.
   **Fixed** (`01`, `02`): it is a Connect row (Connection │ Review │ **restored** │
   help), shown only while a restored connection awaits review, so these profiles have
   no such row or stop. Focused, the Inspector explains it.
4. **Applies to wraps** at both sizes with a dated model id (`03`, `01b`, `01c`). At 211
   it breaks the pair across lines: "Anthropic" ends one line and "· claude-sonnet-4-5."
   starts the next.
   **Fixed** (`03`, `03b`, `01b`, `01c`): one row at both sizes; a row too long for one
   line breaks before the pair, never inside it.
5. **Provider help** is ellipsized at 211 ("configured: Anthropic, Azure OpenAI +1 ·…").
   At 235 the full text, "+1 · 57 more", is ambiguous: one more configured provider,
   then 57 more providers.
   **Fixed** (every Connect capture): "3 of 60 configured · listed first", whole at
   both sizes.
6. **The API key placeholder is cut** to "Local config key saved;" at both sizes.
   **Fixed**: "Paste to replace" (and "Paste API key", "Subscription in use") fit the
   field.
7. **Sign in with breaks the one-row grammar** (`01b`). It has no Source word, and its
   help takes a second row, even at 235, where the help would fit.
   **Fixed** (`01b`): one row with **built-in** and "bills API credits through your
   key"; the long copy is the Inspector's guide.
8. **The picker status line does not count matches** (`02`). With a filter typed it
   still reads "Showing 14 configured models", and no row is highlighted. The Provider
   list says "N found · Enter picks · Esc cancels".
   **Fixed** (`02`, `03b`): "4 found · Enter picks · Esc cancels", the first match
   highlighted, and Enter picks it.
9. **Console Behavior still uses the old layout** (`06a`-`06c`). It keeps its two nested
   frames and a rounded box around the override disclosure. The Providers & Models card
   dropped both. Its Selects also differ in width: the fallback Selects are narrow,
   while Replay, Trace viewer, Rail layout scope and the Background effects Selects
   span the card. Two different rows are both labelled "Replay".
10. **Console Behavior still shows config-key prose** (`06c` at 235, below the frame).
    "Fallback source: [chat_defaults].streaming, temperature, top_p, max_tokens" and
    "Save targets: …" stay in the card. Phase 7 moved this kind of prose out of
    Providers & Models. The list is also incomplete: it leaves out the reasoning,
    thinking, top_k and min_p fallbacks the rows above now show. Also, "Default chat
    display name" is cut to "Default chat display".
11. **Context window's open body** (`05b`) keeps the old layout. It has no Source word
    column, and its prose wraps to two lines. **Reset to detected** is disabled (it is
    dimmed in the `.ansi.txt`), but it still takes a row and still offers "detected"
    when the title says the window is unknown.
