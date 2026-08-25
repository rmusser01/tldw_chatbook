---
id: TASK-21147
title: Quiet startup logging and env-key first-run notice
status: Done
assignee:
  - '@claude'
created_date: '2026-08-25 06:15'
updated_date: '2026-08-25 22:11'
labels:
  - ux
  - app
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
UAT findings G-7, E-1 (findings.md): every cold start prints a wall of DEBUG/WARNING log lines (including 'CRITICAL DEBUG:') to the terminal before the TUI mounts; with a provider env var set, a fresh install boots straight to Console with no acknowledgement and no pointer to the setup wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A default cold start prints no DEBUG/INFO log lines to the terminal before the TUI mounts (WARNING+ only); verbose logging remains available via config or env var
- [x] #2 First run with a provider env key shows a one-time dismissible notice naming the detected key and how to run setup
- [x] #3 The notice never reappears after dismissal
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Quiet pre-TUI stderr sink (WARNING+) at the earliest entry, verbose via env var\n2. Env-key detection helper naming set vars; one-time dismissible notice + persisted flag\n3. Tests; live cold-start capture comparison
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
G-7 had three layers: (1) loguru's package-init full-level stderr sink spewing import-time DEBUG (config.py's 'CRITICAL DEBUG' lines) — new Utils/startup_logging.quiet_startup_stderr() (stdlib+loguru only, no tldw imports) swaps it for a WARNING sink at both entry points (tldw_chatbook.cli before importing app, and a __name__=='__main__' guard atop app.py for the documented python -m tldw_chatbook.app path); (2) _setup_logging's two raw 'INFO: …' prints; (3) the TextualHandler at INFO, which pre-mount falls back to printing on stderr — the DB-migration wall. Both now gate on startup_stderr_is_quiet(); file/RichLog handlers keep their configured levels; TLDW_VERBOSE_STARTUP=1 restores everything (live-verified: 0 noise lines quiet, 30 verbose, TUI healthy).

E-1: env_keys_that_silenced_first_run() (pure, 6 unit tests) names the env vars that made a fresh install skip the wizard offer — suppressed by setup started/completed, a prior notice, or an INLINE config key; the app's offer path shows the one-time toast ('Found OPENAI_API_KEY — you're ready to chat. Run setup any time: …') and persists first_run.env_key_notice_shown via a worker. Live: notice shown once naming the var, flag written, absent on relaunch.

Suites: wizard trio 883 passed + logging buffer test. Files: Utils/startup_logging.py (new), cli.py, app.py, Logging_Config.py, first_run_setup_state.py, Tests/Wizards/test_first_run_setup_state.py.
<!-- SECTION:NOTES:END -->
