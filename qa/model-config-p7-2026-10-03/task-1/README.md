# TASK-33007.1: the Providers & Models card before and after its move (2026-10-03)

The card's composition moved from `SettingsScreen._render_provider_detail` into
`tldw_chatbook/UI/Settings_Modules/providers_models_card.py` with no behaviour change.
These captures show the card rendering the same before and after the move, at 211x44 with
the real stylesheet.

- `before/` ran from a clean worktree at `origin/dev` `9b28ce1479`, which is the branch base.
  The card code there is the same as at the task's base commit `1fab9f27fd`, which adds only
  a docs file.
- `after/` ran from the task's head.

Every file in `after/` is byte-identical to the file of the same name in `before/`, both the
`.txt` (`tmux capture-pane -p`) and the `.ansi.txt` (`-e`, with colour escapes).

## Procedure

- The real app ran in an isolated tmux server at 211x44, one server per side, under `env -i`.
  `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME` and `TLDW_CONFIG_PATH` pointed into a scratch
  directory, and the keyring backend was the null one.
- Each side had its own fresh scratch profile (`users_name = "verify_p7t1_base"` /
  `"verify_p7t1_head"`). Each profile set `[first_run] setup_completed = true`, the splash
  off, `[model_catalog] auto_refresh_enabled = false` and
  `[console.onboarding] first_send_completed = true`. Everything else came from the shipped
  template, so the provider is OpenAI with no key.
- The key sequence was the same on both sides:
  1. Wait for the nav bar, then F4 (Settings).
  2. F6 (focus the category rail), Down, Enter (Providers & Models).
  3. Capture `01`.
  4. Send eleven mouse-wheel-down events at column 100, row 25 and capture again. Repeat
     until the pane stopped changing (`02`-`08`). The wheel scrolls whichever widget is under
     that cell, so `02` and `03` show the provider list scrolled inside the card.
- The real `~/.config/tldw_cli/config.toml` (sha256 prefix `15c6cb224a6a51c7`) and the
  `~/.local/share/tldw_cli` name listing (`db7e7faf5bff92d2`) were the same before and after.
- The driver script is not committed.

## Captures

- `01-pm`: the card at rest. It shows Prompt-cache snapshots (closed), Connect and the
  provider picker, Model, Endpoint, Credentials, and the start of the Test Provider guidance.
- `02-pm`, `03-pm`: the provider list scrolled to the local providers and then to the legacy
  aliases, then the readiness rows.
- `04-pm`: readiness, Context capacity and Model discovery.
- `05-pm`, `06-pm`: Automatic refresh and the per-provider refresh toggles.
- `07-pm`, `08-pm`: Custom endpoints (two built-in slots), Generation defaults (closed), and
  the catalog and policy rows at the end of the card.
