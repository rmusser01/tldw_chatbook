# Phase 7 (TASK-33007) full-screen captures (final set, rebased, 2026-10-09)

The surfaces Phase 7 changed in Settings ▸ Providers & Models and Console Behavior,
plus the Console's Chat settings context fields. **Every capture in this directory was
re-taken together after the rebase onto dev, at `0cd360a642`**: all review notes 1-12,
the checkpoint review's fixes, and dev's "Session summary on quit" (ported into
Advanced as its sixth one-row disclosure) are in. Every moment was captured at
211x44. The tmux window was then resized to 235x52, captured again, and resized back.
`.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour
escapes.

**Final re-take (2026-10-09, `0cd360a642`, rebased onto dev).** One run per profile, under the procedure
below with tmux server `-L capM33007` and `users_name = "verify_capM33007"`.
It shows two checkpoint-review fixes made after the earlier re-takes:
- A card's result line ("Provider settings reverted…") takes no row until a save or
  revert has something to report. At rest the cards no longer open with "… have not
  been saved this session."
- A blank row names its layer once, in its Source word, with only the value in the
  help: **Console Behavior** "inherits 0.6", not "Console Behavior | inherits 0.6 ·
  Console Behavior".

After a revert, the "Settings category changes reverted." toast is dismissed with a
click before the next capture. Same checks: no provider was contacted, no Save was
made, and the real profile's fingerprints were unchanged.

The notes below record how the earlier re-takes went. The files they describe have
since been replaced by the final set.

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
  - The other rows are unchanged: Max tokens **provider** "inherits 4096"; Streaming,
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
- `05b-advanced-context-window-open`, after a click on the Context window title
  (re-taken, see "Re-taken captures" below).
  - One row: "Context window │ tokens", the Source word **not set** in Model defaults'
    Source-word column, and "required for Automatic conversation budgets".
  - No **Reset to detected**: nothing was detected. The other four titles stay closed
    below it.
- `06a-console-behavior-replay-selects` (re-taken). A click on the rail's Console
  Behavior, then a click on the "Reasoning replay override" title.
  - No frame inside the detail pane, and the override disclosure has no box: its title
    is one row, "▼ Reasoning replay override".
  - **Default replay** ("Automatic (recommended)") and the override's **This model's
    replay** ("Use default") are one-row Selects in the card's one control column.
- `06b-console-behavior-rail-layout-scope` (re-taken). Wheel steps down until **Rail
  presentation** sits in the top half. **Trace viewer** ("Safe") and **Rail layout
  scope** ("Global") are one-row Selects in the same column.
- `06c-console-behavior-global-fallbacks` (re-taken). Wheel steps down until **Global
  fallback defaults** sits in the top half; the 235x52 frame was scrolled again after
  the resize, because the reflow moved it.
  - **Chat display name** reads whole. The fallback rows: Temperature 0.6 (**Console
    Behavior**); Max tokens (**provider**, "blank = provider default"); Streaming On
    (**built-in**, "not set · a provider's setting comes first", whole at 211).
  - Reasoning effort, Reasoning summary, Verbosity and Thinking read "default".
    Thinking budget reads ">= 1024".
  - The closed row reads "▶ Sampling · Top P 0.95 · Min P 0.05 · Top K 50".
  - The Background effects Selects are one row each, in the same column as the
    fallback Selects. Nothing follows the card's own status line: no "Composer
    behavior" or "Fallback source" / "Save targets" block.

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

### Context window re-take (08, 08b, 08c; item 12)

Taken 2026-10-09 at branch `p7cf-c` on `f7faf18906` in its own tmux server
(`-L capC33007`), under the same isolation as above (`env -i`, scratch `HOME`,
`XDG_*` and `TLDW_CONFIG_PATH`, null keyring, fake keys) with the OpenAI /
`gpt-5.6-terra` profile. No provider was contacted and nothing was saved: the
only key that differed afterwards was `console.rail_state`. The real profile
hashes were unchanged (`15c6cb224a6a51c7`, `db7e7faf5bff92d2`).

- `08-chat-settings-context`, **Ctrl+O**. The MODEL row reads "Ready · not
  tested · context unknown", and Request estimate "0 / 32,000 tokens (assumed;
  window unknown)". The ▼ of Streaming, Reasoning effort, Reasoning summary
  and Verbosity sit in one column, and so do all six Source words.
- `08b-chat-settings-context-view`, a click on **Context and memory**. Model
  window reads "unknown, 32,000 assumed" and the note "Context window unknown;
  budgets use the assumed size. Enter the model's documented limit in F4
  Settings > Providers & Models." Every Select's ▼ is in one column.
- `08c-advanced-context-window`, **Esc**, then Providers & Models as above.
  The closed title reads "Context window · unknown, 32,000 assumed · enter
  the model's documented limit".

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
9. **Fixed (p7cf-b).** **Console Behavior still uses the old layout** (`06a`-`06c`). It keeps its two nested
   frames and a rounded box around the override disclosure. The Providers & Models card
   dropped both. Its Selects also differ in width: the fallback Selects are narrow,
   while Replay, Trace viewer, Rail layout scope and the Background effects Selects
   span the card. Two different rows are both labelled "Replay".
10. **Fixed (p7cf-b).** **Console Behavior still shows config-key prose** (`06c` at
    235, below the frame).
    "Fallback source: [chat_defaults].streaming, temperature, top_p, max_tokens" and
    "Save targets: …" stay in the card. Phase 7 moved this kind of prose out of
    Providers & Models. The list is also incomplete: it leaves out the reasoning,
    thinking, top_k and min_p fallbacks the rows above now show. Also, "Default chat
    display name" is cut to "Default chat display".
11. **Fixed (p7cf-b).** **Context window's open body** (`05b`) keeps the old layout. It has no Source word
    column, and its prose wraps to two lines. **Reset to detected** is disabled (it is
    dimmed in the `.ansi.txt`), but it still takes a row and still offers "detected"
    when the title says the window is unknown.
12. **Fixed** (`08`, `08b`, `08c`). (a) Chat settings said "~32k context" for
    `gpt-5.6-terra` while Settings ▸ Advanced said "unknown". Both now read the
    one resolver (`resolve_context_window`), and a fallback is "unknown" with the
    size assumed on both. Settings used to drop the provider fallbacks and
    OpenRouter's upstream; it now reads them too. (b) The Chat settings Selects
    were 8, 12 and 13 columns wide, so their ▼ stepped down the column. Each view
    now has one width: 13 in the Model view (numbers included, which also lines
    up the Source words) and 32 in the Context view.

## Re-taken captures (p7cf-b, items 9-11)

`05b`, `06a`, `06b` and `06c` were re-taken at both sizes, `.txt` and `.ansi.txt`, from
branch `p7cf-b` at `18ef163ca6`, after the fixes for items 9-11. The procedure above
held, with these differences:

- The tmux server was `-L capB33007`, and `users_name` was
  `verify_capB33007_18ef163ca6`. Only the OpenAI profile was used.
- `06b` and `06c` scroll until their section header sits in the top half of the pane,
  instead of a fixed count of wheel steps: the card is shorter now.
- Real profile unchanged: before and after, `shasum -a 256 ~/.config/tldw_cli/config.toml`
  began `15c6cb224a6a51c7` (mtime Sep 26), and `ls ~/.local/share/tldw_cli | shasum -a 256`
  began `db7e7faf5bff92d2`. The tmux server was killed and the scratch directory
  deleted.

What changed, by item:

- **9.** The detail pane's border is the only frame: the wrapper, the card, the
  replay-override disclosure and the Permission summaries group draw none. Every
  one-row Input and Select sits in one 32-cell control column, wide enough for the
  longest option ("Memory with latest exchange"). The two replay rows read **Default
  replay** and **This model's replay**. Every prose line starts in the section
  headers' column (the unclassed help lines sat one cell left of them).
- **10.** The read-only "Composer behavior" and "Global fallback defaults" summary that
  followed the card is gone; it restated the card's own rows and the Inspector's
  Override rules. The Inspector's closed **config key** disclosure has a **Fallbacks**
  row naming every `[chat_defaults]` key the group saves: `user_display_name` and all
  14 generation fallbacks. "Default chat display name" is now **Chat display name**,
  which fits its 24-cell label.
- **11.** Context window opens to one row: field (16 cells, Model defaults' column),
  **Reset to detected** in the row while a window is known, a Source word
  (**detected**, **saved in config**, **edited \***, **not set**) and a one-line help.
  The wrapped warning and capacity paragraphs are gone; the focused field guide keeps
  the explanation.
