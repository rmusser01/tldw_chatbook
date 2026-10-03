# TASK-33006.4 Change the model through pick mode captures (2026-10-02)

Chat settings (Ctrl+O) after TASK-33006.4, captured from the real app in this worktree at 211x44 and 235x52. `.txt` is `tmux capture-pane -p`; `.ansi.txt` is the same moment with `-e` colour escapes, where focus shows as bold underlined text on the focus fill.

## Setup

- The app ran in an isolated tmux server (`-L p6t4cap7879189`) under `env -i`, with `HOME`, `XDG_*` and `TLDW_CONFIG_PATH` in a scratch directory and the null keyring backend.
- The scratch profile (`verify_p6t4_7879189`) set `[chat_defaults]` Anthropic `claude-sonnet-4-5`, Temperature 0.7 and Max tokens 2048, a llama.cpp endpoint at `http://127.0.0.1:9199` with nothing listening, `[first_run] setup_completed = true`, `[console.onboarding] first_send_completed = true`, `[model_catalog] auto_refresh_enabled = false` and the splash off. A dummy `ANTHROPIC_API_KEY` was in the environment; it appears in no capture.
- Only real keypresses drove the app (Ctrl+O, Shift+Tab, Tab, Enter, Esc, Alt+M as `M-m`, typed text, `d`). No provider was contacted except the switcher's own local probe of the refused llama.cpp port.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix 15c6cb224a6a51c7, mtime Sep 26) and the `~/.local/share/tldw_cli` name listing (db7e7faf5bff92d2) were the same before and after. tmux was killed and the scratch profile deleted. The driver is not committed.

## Captures

1. `01-model-row-211x44`: Ctrl+O. The MODEL row reads "claude-sonnet-4-5 · Anthropic", the Source word "this chat", "Ready · not tested · ~200k context" and **Change Alt+M**. Connection holds no provider or model picker.
2. `02-change-opens-pick-mode-211x44`: Shift+Tab from Temperature to Change, Enter. Switch model opens over Chat settings in pick mode ("Enter picks · Esc cancel"; no value row or default actions), with the chat's pair marked ● CURRENT. llama.cpp is under NEEDS SETUP ("refused :9199") and cannot be picked.
3. `03-esc-returns-to-change-211x44`: Esc. Chat settings is back unchanged, with focus on Change (bold underlined on the focus fill in the `.ansi.txt`).
4. `04-alt-m-from-temperature-find-opus-211x44`: Tab to Temperature, Alt+M, type `opus-5`. Pick mode opened from inside a text field, and Temperature still reads 0.7 (Alt+M typed nothing).
5. `05-picked-pair-edited-211x44`: Enter. The draft is on "claude-opus-5 · Anthropic" with the Source word "edited *"; the Sampling line now also names Thinking budget, which this model does not take; nothing was applied ("Esc close (asks: 3 unsaved)").
6. `06-picked-pair-edited-235x52`: the same draft after resizing to 235x52; the frame stays 150x22.
7. `07-esc-asks-naming-model-235x52`: Esc. The unsaved prompt names the pair as one field, "Model" (with Min P and Streaming, which the rebase onto the new model changed). `d` then discarded the draft.

The header still reads "Conversation settings" and the footer keeps its old labels: TASK-33006.5 renames them.
