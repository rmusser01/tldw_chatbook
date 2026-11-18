# TASK-33006.4 Change the model through pick mode captures

Chat settings (Ctrl+O), captured from the real app in this worktree at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold underlined text on the focus fill.

The captures come from two runs:

- **01-07** were retaken on 2026-10-03 in the Phase 6 final fix wave, at commit 5ef85f878d (`git status -- tldw_chatbook` clean). The owner ruling of 2026-10-02 makes the closed Sampling title one row; the earlier 01 and 03 showed it wrapping to two.
- **08-12** are the TASK-33006.4 review-round-1 run (2026-10-03, at that round's tree). The `/endpoint` flow they show is unchanged by the final fix wave. Chat settings behind them predates the fix wave: 08 and 09 still show the old two-row Sampling title ("▶ Sampling ·" / "provider doe…"), and 11 and 12 show a blank dropdown as "Select" beside "built-in … blank = provider default". Both were fixed at 5ef85f878d; see `qa/model-config-33006-captures/` (01, 02, 07) for the current one-row title and blank rows.

## Setup

**01-07 (final fix wave).** Isolated tmux server `-L p6finalt4`, under `env -i`, with a scratch `HOME`/`XDG_*`/`TLDW_CONFIG_PATH`, the null keyring and `PYTHONPATH` set to this worktree. The scratch profile is `verify_p6final_t4`: `[chat_defaults]` Anthropic `claude-sonnet-4-5`, Temperature 0.7, Max tokens 2048, and a llama.cpp endpoint at `http://127.0.0.1:9199` with nothing listening. It sets the same first-run, onboarding, catalog and splash keys as below. A dummy `ANTHROPIC_API_KEY` was set and appears in no capture. Only keys reached the app. The real profile kept 15c6cb224a6a51c7 / db7e7faf5bff92d2 before and after, and the scratch home was deleted. The driver is not committed.

**08-12 (review round 1):**

- **Isolation.** The app ran in an isolated tmux server (`-L p6t4fix1cap7a65db2`) under `env -i`. `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` pointed into a scratch directory, the keyring backend was the null one, and `PYTHONPATH` was this worktree, so the main checkout's code was not used.
- **Scratch profile.** The profile (users_name `verify_p6t4fix1_7a65db2`) set:
  - `[chat_defaults]` to Anthropic `claude-sonnet-4-5`, Temperature 0.7 and Max tokens 2048;
  - a llama.cpp endpoint at `http://127.0.0.1:9199`, with nothing listening;
  - `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true` and `[model_catalog] auto_refresh_enabled = false`;
  - the splash screen off.
- **Dummy key.** A dummy `ANTHROPIC_API_KEY` was in the environment. It appears in no capture.
- **Stub server.** A scratch HTTP server on `127.0.0.1:9187` answered `/v1/models` with two ids, `accounts/fireworks/models/deepseek-v3-0324` and `bartowski/Meta-Llama-3.1-70B-Instruct-GGUF/Meta-Llama-3.1-70B-Instruct-abliterated-Q4_K_M.gguf`. No other provider was contacted, apart from the switcher's own local probes.
- **Input.** Only keys reached the app, sent with `tmux send-keys`: Ctrl+O, Shift+Tab, Tab, Home, Up, End, Backspace, Enter, Esc, `d`, Ctrl+Q, Alt+M as `M-m`, and typed text. The one exception is the template's Create, between captures 9 and 10, pressed with an SGR mouse click (`\e[<0;154;37M`/`m`).
- **Real profile unchanged.** Before and after the run:
  - the real `~/.config/tldw_cli/config.toml` had sha256 prefix 15c6cb224a6a51c7 and mtime Sep 26;
  - `ls ~/.local/share/tldw_cli | shasum -a 256` gave db7e7faf5bff92d2.

  These are the same values the 2026-10-02 run recorded.
- **Cleanup.** tmux was killed and the scratch profile and stub server deleted. The driver is not committed.

## Captures

1. **`01-model-row-211x44`** (Ctrl+O). The MODEL row reads "claude-sonnet-4-5 · Anthropic", then "this chat", "Ready · not tested · ~200k context" and **Change Alt+M**.
   - The pair takes the row's free width.
   - The Source word, the readiness words and Change sit together on the right.
   - The closed Sampling title is one row: "Sampling · Anthropic does not accept 7 fields (open to list them)".
2. **`02-change-opens-pick-mode-211x44`** (Shift+Tab from Temperature to Change, then Enter). Switch model opens over Chat settings in pick mode: "Enter picks · Esc cancel", with no value row and no default actions. The chat's pair is marked ● CURRENT.
3. **`03-esc-returns-to-change-211x44`** (Esc). Chat settings is back unchanged, with focus on Change (`.ansi.txt`).
4. **`04-alt-m-from-temperature-find-opus-211x44`** (Tab to Temperature, Alt+M, type `opus-5`). Pick mode opened from inside a text field. Temperature still reads 0.7.
5. **`05-picked-pair-edited-211x44`** (Enter). The draft is on "claude-opus-5 · Anthropic" with the Source word "edited *". Nothing was applied: the title reads "3 unsaved edits", and the footer reads "Esc close (asks: 3 unsaved)". The picked pair's defaults are staged, so the footer reads **Matches saved defaults**, and no Save as model default is offered. The closed Sampling title counts the 8 fields claude-opus-5 does not accept.
6. **`06-picked-pair-edited-235x52`**. The same draft after a resize to 235x52. The frame stays 150x22.
7. **`07-esc-asks-naming-model-235x52`** (Esc). The unsaved prompt names the pair as one field, "Model". It also names Min P and Streaming, which the rebase onto the new model changed (TASK-33006.16). `d` then discarded the draft.
8. **`08-endpoint-command-template-over-chat-settings-211x44`** (resize to 211x44, type `/endpoint`, Enter twice). The first Enter accepts the completion and the second runs it. "New endpoint from template" opens over the Chat settings this command opened.
9. **`09-endpoint-template-blank-models-211x44`**. The form after these steps:
   - Home, Enter, Up and Enter, which reached the focused Display name field and changed nothing;
   - Shift+Tab to Template, Up to "OpenAI-compatible (blank)", Enter;
   - Tab, then type "Lab box";
   - Tab twice, then End, Backspace ×30 and type `http://127.0.0.1:9187`.

   **Models** is empty.
10. **`10-endpoint-create-pick-mode-lists-served-211x44`** (Create). Pick mode opens over Chat settings with "Lab box " in Find. It lists both models the server serves, although the template named none: the entry was listed before pick mode opened.
11. **`11-long-id-shortened-in-the-middle-211x44`** (type `Q4_K_M`, Enter). The draft lands on the entry and the 94-cell GGUF id, which is wider than the row.
    - The MODEL row reads "bartowski/Meta-Llama-3.1-…t-abliterated-Q4_K_M.gguf · Lab box". The id is shortened in the middle, keeping its quant suffix, and the provider name is whole.
    - Then come "edited *", "Ready · reachable 01:17 · ~32k context" and **Change Alt+M**.
12. **`12-long-id-shortened-in-the-middle-235x52`**. The same draft at 235x52. Esc and `d` then discarded it, and Ctrl+Q quit.

In 08-12 (round 1) the header still reads "Conversation settings" and the footer keeps its old labels, which TASK-33006.5 renamed later. Where their Sampling title shows, it is the Lab box line ("Sampling · hidden for Lab box: Reasoning summary, Verbosity, Thinking, Thinking budget (this provider does not accept them)"): the named form, on one row.
