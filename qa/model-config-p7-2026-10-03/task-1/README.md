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

## Widget tree (AC#2: every id renders, under the same classes)

`widget-tree-211x44-rest.txt` and `widget-tree-211x44-open.txt` list every widget under
`#settings-detail-pane-body` with Providers & Models selected, one row per widget: type, id,
sorted classes, `display`, `disabled`, region, and for text widgets the first 120 characters
of the rendered text. `-rest` is the card as it mounts. `-open` is the card after every
`Collapsible` in it was opened.

- A throwaway pilot test (not committed) produced them. It built `_build_test_app()` inside
  `_SettingsCssHarness` (the real `APP_STYLESHEETS`) at 211x44 under `@private_profile_test`,
  then ran `_settle_settings` and `_click_settings_category(pilot, "providers-models")`. These
  are the same helpers that `Tests/UI/test_settings_providers_models_card_geometry.py` uses.
- It ran once at `9b28ce1479` and once at the task's head, each from its own tree with
  `PYTHONPATH` set to that tree. Each file has 282 rows, and base and head are byte-identical,
  so only one copy is committed. The sha256 values on both sides:
  - rest: `e33377e3234aebaaf4b4463b0051624f240df9343ad7447b7483faef21a2b06b`
  - open: `32da6c2658450fbf274bc807daa10fd36b5c29e20243fab894240501aabec3de`
- The implementer's run and the review fix run produced the same hashes.
