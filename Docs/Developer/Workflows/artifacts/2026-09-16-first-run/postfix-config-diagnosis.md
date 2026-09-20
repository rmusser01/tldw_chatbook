# Preserved post-fix live assertion failure

Packet `postfix/` is unchanged: all controls passed, final pytest assertion did
not. This is not a production config fix or a retroactive passing test claim.

Before/after raw TOML hashes and full effective paths are in
`postfix/live-110x36-walk.json`; raw files remain under task-owned scratch
`task-6-live-postfix/evidence/live-110x36-walk-{before,after}.toml`.
Both files parse. Exact authority-table semantic differences:

| Table | Removed keys | Changed existing values | Added values |
| --- | --- | --- | --- |
| paths | none | none | none |
| database | none | none | `USER_DB_BASE_DIR="~/.local/share/tldw_cli/"`, `check_integrity_on_startup=false`, `integrity_check_timeout=30` |
| tldw_api | none | none | none |

These three defaults are literally present in `config.py:3970–3974`.
Source cause: `TldwCli._run_blocking_quit_persistence` calls
`persist_cli_config_for_shutdown` (`app.py:19586`). That function reloads via
`_load_cli_config_bootstrap_unlocked(force_reload=True)`, which deep-merges the
user file over `DEFAULT_CONFIG_FROM_TOML` (`config.py:6099,6136`), then persists
that merged config (`config.py:7047–7063`). The snapshot surrounds the entire
walk/quit, so this should not be mislabeled as only a startup rewrite.
Other added sections/default keys and Library rail graduation are reported in
the manifest's changed-section list; whole-file byte equality was never expected.

Effective paths checked before app construction: all `get_*_db_path` results,
user data and model cache are within this disposable profile; permission store is
`data/default_user/mcp_permissions.json`. DB overrides are unchanged afterward.
`get_user_data_dir` selects `[paths].data_dir` before its secured default branch
(`config.py:9034`); custom DB getters read unchanged `[database]` overrides.
`USER_DB_BASE_DIR` is legacy Settings data, not an input to these getters (source
search finds it only in config's template and Settings display/save surfaces).
Effective HOME remains the private per-test HOME, not the host HOME. This packet
did not independently record a second post-quit getter evaluation; a corrected
qualification should do that, rather than implying it already happened here.

No original packet/hash/log was rewritten. Ordinary logs contain no payload
canaries, zero unhandled exceptions, no premature stop, and one intentional stop.
Controls produced three real localhost9099 POSTs and Note
`0fd41344-7caa-4cb9-87f2-16236cb3e59d` with exact accepted content.
