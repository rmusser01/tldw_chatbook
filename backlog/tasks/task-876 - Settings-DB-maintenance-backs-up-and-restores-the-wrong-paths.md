---
id: TASK-876
title: >-
  Settings DB maintenance backs up and restores paths that are not the real databases
status: To Do
assignee: []
created_date: '2026-07-27 06:00'
labels:
  - settings
  - bug
  - data-safety
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found while fixing TASK-860 (the Evals DB ignoring the configured profile). The same class of path bug is in the Settings database-maintenance panel, and there it reaches backup and restore.

`Tools_Settings_Window._get_database_path()` builds its path map from `db_config.get("<name>_db_path", "<hardcoded default>")`. Those hardcoded defaults do not match reality in two independent ways:

1. **Wrong filenames.** It claims `tldw_evals_db.db`, `tldw_prompts_db.db`, `tldw_media_db.db`, `tldw_rag_db.db`. The real files are `evals.db`, `tldw_chatbook_prompts.db`, `tldw_chatbook_media_v2.db`.
2. **No profile directory.** It points at `~/.local/share/tldw_cli/<file>`, but every database actually lives under `~/.local/share/tldw_cli/<profile>/<file>`.

The project already has correct resolvers and this panel does not use them. Verified on this machine:

```
config.get_prompts_db_path()  -> ~/.local/share/tldw_cli/default_user/tldw_chatbook_prompts.db
settings UI hardcodes         -> ~/.local/share/tldw_cli/tldw_prompts_db.db
```

There is no `get_evals_db_path()` helper at all, and neither `evals_db_path` nor `rag_db_path` is defined in `config.py`, so for those two the wrong hardcoded fallback always wins. The file also disagrees with itself: `media_db_path` is read with two different defaults (`tldw_cli_media_v2.db` in one place, `tldw_media_db.db` in another), and `prompts_db_path` likewise.

**Why this is data-safety and not cosmetic.** `_get_database_path()` feeds four workers: `_vacuum_single_worker`, `_backup_single_worker`, `_restore_single_worker` and `_check_single_worker`. So a user who opens Settings and backs up a database is backing up a path that does not exist, and a user who restores is writing a backup to a location that is not the live database — while the real one is never touched. Both operations can report success while doing nothing, which is exactly the wrong failure mode for a feature people reach for during recovery.

The blast radius is not limited to Evals: because none of the paths carry the profile segment, every database is affected for any user whose profile is not the literal default, and `evals`/`rag` are wrong for everyone.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] The maintenance panel resolves every database through the project's own path resolvers, not hardcoded literals
- [ ] Paths honour the configured profile, matching where the databases actually are
- [ ] A missing or unresolvable database reports a clear failure instead of silently succeeding
- [ ] Backup followed by restore round-trips against the real file, proven by a test
- [ ] The duplicated, disagreeing per-key defaults inside the file are gone
<!-- AC:END -->
