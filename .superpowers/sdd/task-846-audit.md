# TASK-846 — Audit: security checks that name filesystem paths vs. where the app actually writes

**Worktree:** `/Users/macbook-dev/Documents/GitHub/wt-path-hardening` (branch `feat/agent-path-hardening`)
**Mode:** READ-ONLY. Nothing was fixed; no tracked file was modified.
**Interpreter:** `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`
**Date of on-disk comparison:** 2026-07-26, this machine.

## Ground truth resolved on this machine

```
DEFAULT_CONFIG_PATH        : /Users/macbook-dev/.config/tldw_cli/config.toml       exists=True
_get_effective_config_path : /Users/macbook-dev/.config/tldw_cli/config.toml
get_user_data_dir()        : /Users/macbook-dev/.local/share/tldw_cli/default_user  exists=True
get_media_db_path()        : /Users/macbook-dev/.local/share/tldw_cli/default_user/tldw_chatbook_media_v2.db  exists=True
```

Two facts drive most findings:

1. `get_user_data_dir()` = `<base>/<user_folder>`. Every literal of the shape
   `~/.local/share/tldw_cli/<file>` omits the `<user_folder>` segment and is therefore
   **never** the path the app uses.
2. `_get_effective_config_path()` honors `TLDW_CONFIG_PATH`; `DEFAULT_CONFIG_PATH` does not.
   Any security decision built on `DEFAULT_CONFIG_PATH` diverges the moment a profile is active.

---

## Inventory

30 security-relevant checks that name a filesystem path, config key, or database location.
"Match" = does the literal resolve to what the app really uses, on this machine, right now.

| # | Check | Literal it names | Path/key the app really uses | Match |
|---|---|---|---|---|
| 1 | `config.py:4322/4323/4330/4362/4363/4393/4425/4426/4458` — `enable/disable_config_encryption`, `change_encryption_password` | `DEFAULT_CONFIG_PATH` (`config.py:46`) | `_get_effective_config_path()` (`config.py:49`) | **NO** |
| 2 | `Utils/config_encryption.py:245-256` `detect_api_keys` → `should_encrypt_config()` | `["api_key","apikey","api-key","secret","token","password"]` exact-match | real keys are `openai_api_key`, `bing_search_api_key`, `api_token`, `auth_token`, `OPENAI_API_KEY_fallback`, … | **NO** |
| 3 | `Utils/sensitive_paths.py` denylist vs. skill **trust/grant** store | (absent — no entry) | `get_user_data_dir()/skills/trust/{skill_trust_manifest.json, skill_script_grants.json, generation_marker.json, snapshots/}` (`app.py:5181-5194`) | **NO (uncovered)** |
| 4 | `Utils/sensitive_paths.py` denylist vs. `config.toml` **sidecars** | `_get_effective_config_path()` only | app also writes `config.toml.bak` + `config.toml.tmp` (`UI/Screens/settings_screen.py:5596-5605`) | **NO (uncovered)** |
| 5 | `Skills_Interop/skill_trust_store.py:93-96,110` `_validated_trust_file_path(..., base_dir=self.marker_path.parent)` | `base_dir = path.parent` | correct pattern is `base_dir=self.store_dir` (`skill_trust_store.py:491`) | **NO (vacuous)** |
| 6 | `Skills_Interop/local_skills_service.py:1765-1804` `_unsafe_scratch_root` | containment tested root-**inside**-container only | must also reject a root that **contains** the stores | **NO (inverted)** |
| 7 | `MCP/server.py:154` media DB | `get_cli_setting("database","media_db","media_library.db")` | key is `media_db_path`; accessor is `get_media_db_path()` (`config.py:4622`) | **NO** |
| 8 | `MCP/local_store.py:13` `DEFAULT_LOCAL_MCP_STORE_PATH` | `~/.config/tldw_cli/local_mcp_store.json` | `get_user_data_dir()/local_mcp_store.json` (`app.py:5241`) | **NO (latent)** |
| 9 | `MCP/unified_context_store.py:14` | `~/.config/tldw_cli/unified_mcp_context.json` | `get_user_data_dir()/unified_mcp_context.json` (`app.py:5248`) | **NO (latent)** |
| 10 | `MCP/server_target_store.py:13` | `~/.config/tldw_cli/mcp_server_targets.json` | `get_user_data_dir()/mcp_server_targets.json` (`app.py:4028`) | **NO (latent)** |
| 11 | `config.py:4299` `get_detected_api_providers()` | `section_name.startswith("api_settings.")` | tomllib nests → top-level key is `"api_settings"` | **NO** |
| 12 | `config.py:3691` `_is_sensitive_setting_key` (substring) vs `UI/Screens/settings_privacy_security.py:190` (endswith) | identical literal sets, different semantics | — | **DIVERGENT** |
| 13 | `Utils/log_sanitizer.py` (whole module) | 20+ key/pattern literals | zero production importers | **NO** |
| 14 | `Utils/log_sanitizer.py:16` `claude-[a-zA-Z0-9-]+` | Anthropic **key** prefix | `claude-*` is the **model-id** prefix; keys are `sk-ant-…` | **NO (inverted)** |
| 15 | `Event_Handlers/eval_db_operations.py:28` | `~/.config/tldw_cli/evals.db` | `get_user_data_dir()/<user>/evals.db` (`Evals/eval_orchestrator.py:94`) | **NO** |
| 16 | `UI/Tools_Settings_Window.py:6480-6507`, `:6631-6652` DB path map (integrity check / backup / vacuum / chatbook import) | `~/.local/share/tldw_cli/tldw_{media,prompts,evals,rag,subscriptions}_db.db` | `get_*_db_path()` → `get_user_data_dir()/tldw_chatbook_*.db` | **NO** |
| 17 | `Chatbooks/chatbook_importer.py:77-79` extraction root | `~/.local/share/tldw_cli/temp/imports` | `get_user_data_dir()/temp/…` (`chatbook_creator.py:97`) | **NO** |
| 18 | `Chatbooks/local_chatbook_service.py:102-107` | `~/.local/share/tldw_cli/tldw_chatbook_chatbooks.json` | `get_user_data_dir()/…` | **NO** |
| 19 | `Workspaces/registry_service.py:673-707` folder-binding denylist | `Path(resolved.anchor)` and `Path.home()` only | `Utils/sensitive_paths.is_sensitive_path` is never consulted | **NO (incomplete)** |
| 20 | `DB/sql_validation.py:14-24` `VALID_TABLES["chachanotes"]` | 9 names | 35 real tables; `keyword_collections` missing | **NO (over-blocks)** |
| 21 | `DB/sql_validation.py:308` `VALID_COLUMNS` gate | 8 table keys | callers pass `sync_profile_state`, `Transcripts`, `MediaChunks`, `UnvectorizedMediaChunks`, `DocumentVersions` | **NO (no-ops)** |
| 22 | `Utils/path_validation.py:311` `dangerous_patterns` contains `"~/"` | rejects `~/` | 6 `Evals/task_loader.py` sites pass raw user input un-expanded | **NO (over-blocks)** |
| 23 | `Subscriptions/security.py:79,85-89` `BLOCKED_SCHEMES`, `METADATA_ENDPOINTS` | 5 schemes + 3 metadata hosts | real enforcer is `Utils/egress.py:35-44`; these have zero readers | **NO (dead + stale)** |
| 24 | `config.py:3471-3476` `[subscriptions.security]` 5 keys | `enable_ssrf_protection`, `verify_ssl_certificates`, `max_redirects=5`, … | zero readers; real controls are `[web_security] enabled`, per-source `ssl_verify`, `egress.MAX_REDIRECT_HOPS=10` | **NO** |
| 25 | `Tools/file_operation_tools.py:28` `[tools] file_sandbox_root` | key read | key is declared in no default TOML and written by no UI | **read-only/undeclared** |
| 26 | `config.py:3541-3542,3562` `[github] respect_gitignore`, `show_hidden_files`, `profiles_directory` | 3 keys | zero readers; `CodeRepoCopyPasteWindow.py:392-429` uses a hardcoded 10-name list, never parses `.gitignore` | **NO** |
| 27 | `Metrics/logger_config.py:21-22` | `~/.local/tldw_cli/Logs/tldw_app.log` | `get_cli_log_file_path()` → `get_user_data_dir()/tldw_cli_app.log`; `setup_logger` has zero callers | **NO** |
| 28 | `Utils/Utils.py:77-79` `USER_DB_DIR`/`USER_DB_PATH` + `Utils/paths.py:205` `get_user_database_path` | `~/.config/tldw_cli/tldw_cli_Media.db` | `get_media_db_path()` | **NO (dead)** |
| 29 | `UI/Screens/chat_screen.py:15394`, `Event_Handlers/notes_events.py:144`, `note_ingest_events.py:350`, `Subscriptions/website_monitor.py:72` | `Path.home()/".config"/"tldw_cli"/…` | `_get_effective_config_path().parent` | **NO** |
| 30 | `Utils/sensitive_paths.py` — the module this task was cut from | derives config.toml, 3 MCP files, 11 DBs + sidecars, `user_data_dir` file rule | all match on this machine | **YES** |

---

## Findings, in severity order

Severity per the task's rule: **Critical** when the unprotected/mis-protected thing *grants
privilege* (permission stores, trust stores, credentials, config that turns gates off);
**Important** when it only discloses or corrupts data.

---

### CRITICAL-1 — Config encryption encrypts the wrong file and reports success

`tldw_chatbook/config.py:4309-4470`. All three encryption entry points
(`enable_config_encryption`, `disable_config_encryption`, `change_encryption_password`) open
`DEFAULT_CONFIG_PATH` directly instead of `_get_effective_config_path()`. Every other read and
write in the app (`config.py:681`, `:3597`, `:3893`, `:3994`) goes through the accessor.

```python
def enable_config_encryption(password: str) -> bool:          # config.py:4309
        config_data = {}
        if DEFAULT_CONFIG_PATH.exists():                       # :4322
            with open(DEFAULT_CONFIG_PATH, "rb") as f:         # :4323
                config_data = tomllib.load(f)
        encrypted_config = encrypt_api_keys_in_config(config_data, password)
        with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:   # :4330
            toml.dump(encrypted_config, f)
```

Live callers: `UI/Tools_Settings_Window.py:3857`, `:3897`, `:6787`, `:6838`, `:6936`.

**What it fails to protect:** the user's API keys. `TLDW_CONFIG_PATH` is a supported mode
(surfaced as "Override config" at `UI/Screens/settings_screen.py:5035-5058`, described at
`runtime_policy/bootstrap.py:30`, and used throughout this project's own test suite). With a
profile active, "Enable encryption" leaves the live config's keys plaintext, silently rewrites a
*different* config file, and returns `True`. `disable_config_encryption` /
`change_encryption_password` then read `encryption.password_verifier` from the wrong file
(`:4372`, `:4435`) and abort with "No password verifier found" — the user cannot rotate or
remove the password either.

**Verification** (sandboxed `HOME`, real `enable_config_encryption` call, scratchpad `verify4.py`):

```
TLDW_CONFIG_PATH             : …/task846_b0elztkp/profiles/work.toml
_get_effective_config_path() : /private/…/task846_b0elztkp/profiles/work.toml   <- what the app reads/saves
DEFAULT_CONFIG_PATH          : …/task846_b0elztkp/.config/tldw_cli/config.toml  <- what encryption uses
MATCH: False
enable_config_encryption('hunter2hunter2') -> True
ACTIVE profile changed?  False | still plaintext secret present: True
DEFAULT config changed?  True  | plaintext secret still there: False
--> encryption reported success while the ACTIVE config's key stayed plaintext
```

Secondary: `:4330` writes with plain `open(..., "w")` while its two siblings use
`atomic_write_text` (`:4393`, `:4458`) — an interrupted enable truncates the config.

---

### CRITICAL-2 — `should_encrypt_config()` never fires for real provider credentials

`tldw_chatbook/Utils/config_encryption.py:245-256`. `detect_api_keys` matches key names by
**exact equality** against six literals:

```python
                if key.lower() in [
                    "api_key", "apikey", "api-key", "secret", "token", "password",
                ]:
```

The app's real secret-bearing key names are all *prefixed or suffixed*: `openai_api_key`,
`anthropic_api_key`, `cohere_api_key`, `bing_search_api_key`, `tavily_search_api_key`,
`api_token` (github/confluence), `auth_token` (`[tldw_api]`), `OPENAI_API_KEY_fallback`,
`ELEVENLABS_API_KEY_fallback` (`config.py:66,70`, written by
`UI/Tools_Settings_Window.py:5521,5541`). None of them equals any of the six.

**What it fails to protect:** the gate that offers at-rest encryption at all. `config.py:4285`
`should_encrypt_config()` returns `enc_module.detect_api_keys(config)` — so a config full of
plaintext provider keys is reported as having nothing to encrypt, and the user is never
prompted.

**Verification** (scratchpad `verify2.py`):

```
detect_api_keys(config full of REAL provider secrets) -> False
detect_api_keys({'x':{'api_key':'sk-1'}})             -> True
```
(the first dict contains `openai_api_key`, `anthropic_api_key`, `cohere_api_key`,
`bing_search_api_key`, `tavily_search_api_key`, `github.api_token`, both `*_fallback` keys and
`tldw_api.auth_token` — all live-plaintext, all reported clean.)

---

### CRITICAL-3 — The skill **script-grant and trust** store is not on the agent-tool denylist

`tldw_chatbook/app.py:5181-5200` builds the trust store at
`get_user_data_dir()/skills/trust/`. `skill_script_grants.json`
(`Skills_Interop/skill_trust_service.py:49`, joined at `:661`) is the file
`has_script_grant` (`:1452-1465`) consults to authorize **script execution**, and it is
*deliberately* outside the MAC'd manifest (`:653-661`) so it is plain, unauthenticated JSON.

`Utils/sensitive_paths.py` names none of it, and the module's own
`resolved.parent == ctx.user_data_dir` rule structurally cannot reach it: `skills` is an existing
directory, so it is exempted by design (the rule's comment at `sensitive_paths.py:48-51` names
`skills` among the intentionally-reachable containers), and everything nested under it inherits
that exemption.

**What it fails to protect:** the script-execution grant gate and the trust manifest/snapshots
that back trust review — exactly the "one-step gate bypass" class the `mcp_permissions.json`
entry exists to prevent. Reachability still requires a widened `[tools] file_sandbox_root` or a
bound workspace folder root (see IMPORTANT-6), so this is uncovered surface rather than a live
bypass today.

**Verification** (scratchpad `verify3.py`):

```
…/default_user/skills/trust/skill_trust_manifest.json   sensitive=False
…/default_user/skills/trust/skill_script_grants.json    sensitive=False
…/default_user/skills/trust/generation_marker.json      sensitive=False
…/default_user/skills/trust/snapshots                   sensitive=False
…/default_user/skills/tldw_chatbook_skills.json         sensitive=False
```

---

### CRITICAL-4 — `config.toml.bak` (a byte-for-byte copy of the config, API keys included) is not denylisted

`UI/Screens/settings_screen.py:5596-5605` — Advanced config save writes a full backup before
overwriting:

```python
        tmp_path = config_path.with_suffix(config_path.suffix + ".tmp")
        backup_path = config_path.with_suffix(config_path.suffix + ".bak")
        ...
            if config_path.exists():
                backup_path.write_text(
                    config_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
```

The denylist names `_get_effective_config_path()` and nothing else in that directory. `.bak` and
`.tmp` carry identical content — including every plaintext API key.

Note the asymmetry: `sensitive_paths.py` *does* handle the equivalent case for databases
(`_DB_SIDECAR_SUFFIXES` = `-wal`/`-shm`/`-journal`, `_db_sidecar_paths`) and for the MCP
permission store's own `.bak` (`MCP/permission_store.py:232`) via the `user_data_dir` file rule
— but the config file lives in `~/.config/tldw_cli/`, which that rule does not cover.

**What it fails to protect:** the user's API keys, in a file created by a first-party UI action.

**Verification** (scratchpad `verify3.py`; `.bak` absent right now only because Advanced save
has not been used on this machine — the directory does hold two hand-made copies,
`config.toml.bak-1785079350` and `config.toml.pre-lab-cleanup`, both 45 KB, both unprotected):

```
/Users/macbook-dev/.config/tldw_cli/config.toml.bak      exists=False sensitive=False
/Users/macbook-dev/.config/tldw_cli/config.toml.tmp      exists=False sensitive=False
/Users/macbook-dev/.config/tldw_cli/ui_state.toml        exists=True  sensitive=False
/Users/macbook-dev/.config/tldw_cli/runtime_policy.json  exists=True  sensitive=False
/Users/macbook-dev/.config/tldw_cli/config.toml.bak-1785079350  exists=True  sensitive=False
```

---

### CRITICAL-5 — Trust-store path validator's containment test is true by construction

`Skills_Interop/skill_trust_store.py:93-96` and `:110` call the trust-path validator with
`base_dir` set to the candidate's own parent:

```python
        marker_path = _validated_trust_file_path(
            self.marker_path,
            base_dir=self.marker_path.parent,
        )
```

`_validated_trust_file_path` (`:544-554`) rejects when
`get_safe_relative_path(candidate, base) is None`. With `base = candidate.parent` that branch can
never be taken. The correct pattern exists two methods away —
`_validated_manifest_path` (`:490-491`) passes `base_dir=self.store_dir` — and `store_dir` was
available at the marker's construction site (`app.py:5186`).

**What it fails to protect:** confinement of the trust generation marker (the rollback-protection
anchor) to the trust store directory.

**Verification** (scratchpad `verify3.py`, temp dir):

```
base_dir=path.parent  -> ACCEPTED …/totally/elsewhere/generation_marker.json   (containment test is true by construction)
base_dir=real store   -> rejected: unsafe skill trust path  (correct)
```

---

### CRITICAL-6 — Skill scratch-root safety check tests containment in the wrong direction

`Skills_Interop/local_skills_service.py:1765-1804`. The container list is correctly *derived*
(`self.skills_dir`, `trust_store.store_dir`), but the predicate is

```python
        return any(
            get_safe_relative_path(root, container) is not None
            for container in self._unsafe_scratch_root_containers()
        )
```

and `get_safe_relative_path(a, b)` (`Utils/path_validation.py:268-273`) is non-`None` only when
`a` is *inside* `b`. So a scratch root **nested in** the stores is refused, while a scratch root
that **contains** them passes. The docstring at `:1782-1791` claims the opposite guarantee.

**What it fails to protect:** the "a script must never be able to tamper with its own bundle or
the trust store" property. `[skills] script_scratch_root = ~/.local/share/tldw_cli/<user>/skills`
passes while placing the script's cwd one level above both `skills/skills/` (every trusted
bundle) and `skills/trust/` (manifest, snapshots, marker, grants). `self.store_dir` — which holds
the skill index `tldw_chatbook_skills.json` — is not in the container list at all.

**Verification** (scratchpad `verify3.py`, real container paths):

```
candidate=…/default_user/skills/skills/sub  flagged_unsafe=True   [root INSIDE skills_dir  (should be unsafe)]
candidate=…/default_user/skills             flagged_unsafe=False  [root CONTAINS both stores (should be unsafe)]
candidate=…/default_user                    flagged_unsafe=False  [root CONTAINS everything (should be unsafe)]
```

---

### IMPORTANT-1 — MCP server opens a media DB that is on no denylist, via a config key that does not exist

`tldw_chatbook/MCP/server.py:154-155`:

```python
            media_db_path = get_cli_setting("database", "media_db", "media_library.db")
            self.media_db = MediaDatabase(media_db_path)
```

The key is `media_db_path` (`config.py:4624`, declared `config.py:2233`, present in the live
config at line 30). `media_db` does not exist, so the lookup always falls to the relative literal
and the MCP server creates/opens `./media_library.db` in whatever CWD it was launched from. Two
lines above, the same function correctly calls `get_chachanotes_db_path()`.

**What it fails to protect:** `Utils/sensitive_paths.py:98` denylists `get_media_db_path` — the
DB the MCP server does *not* use. Everything the MCP media tools write lands in a CWD-relative
file with no denylist coverage at all.

**Verification** (scratchpad `verify1.py`, then `verify2.py` for the denylist half):

```
get_cli_setting("database","media_db","media_library.db") -> 'media_library.db'
get_cli_setting("database","media_db_path", None)         -> '~/.local/share/tldw_cli/tldw_cli_media_v2.db'
resolved by MCP server  : /Users/macbook-dev/Documents/GitHub/wt-path-hardening/media_library.db exists= False
resolved by app         : /Users/macbook-dev/.local/share/tldw_cli/default_user/tldw_chatbook_media_v2.db
MATCH: False

is_sensitive_path(…/wt-path-hardening/media_library.db) -> False
is_sensitive_path(…/default_user/tldw_chatbook_media_v2.db) -> True
```

---

### IMPORTANT-2 — Three MCP stores default to a directory the app never uses (latent gate-state split)

`MCP/local_store.py:13`, `MCP/unified_context_store.py:14-16`, `MCP/server_target_store.py:13`
all default to `DEFAULT_CONFIG_PATH.parent / <name>` — i.e. `~/.config/tldw_cli/`. The app always
passes `get_user_data_dir() / <name>` explicitly (`app.py:5241`, `:5248`, `:4028`).

This matters more than a normal stale default because the **permission store and execution log
are derived from `store.path`**: `MCP/unified_control_plane_service.py:2430` builds
`Path(store.path).with_name("mcp_permissions.json")` and `:2073` builds
`…with_name("mcp_execution_log.jsonl")`. A `LocalMCPStore()` constructed with no argument
anywhere would place the permission store in `~/.config/tldw_cli/`, where **neither**
`_sensitive_single_file_paths()` (which joins to `get_user_data_dir()`) nor the
`parent == user_data_dir` rule matches.

**What it fails to protect:** nothing today — every construction site passes an explicit path.
Rated Important, not Critical, because it is a latent second location for the permission store
rather than a live bypass.

**Verification** (scratchpad `verify2.py`):

```
local_mcp_store.json
  module default : /Users/macbook-dev/.config/tldw_cli/local_mcp_store.json  exists=False
  app builds     : /Users/macbook-dev/.local/share/tldw_cli/default_user/local_mcp_store.json  exists=True
  MATCH: False
unified_mcp_context.json      module default exists=False ; app-built exists=True ; MATCH: False
mcp_server_targets.json       module default exists=False ; app-built exists=True ; MATCH: False
```
Also verified uncovered: `is_sensitive_path(~/.config/tldw_cli/mcp_permissions.json) -> False`,
`is_sensitive_path(~/.config/tldw_cli/local_mcp_store.json) -> False`.

---

### IMPORTANT-3 — `Utils/log_sanitizer.py` has zero production importers, and its Anthropic rule is inverted

```
$ grep -rn "log_sanitizer" tldw_chatbook/ Tests/ --include="*.py"
Tests/Utils/test_security_enhancements.py:6:from tldw_chatbook.Utils.log_sanitizer import (
```

Nothing in `tldw_chatbook/` imports it. (The 40+ `sanitize_string` hits elsewhere resolve to a
different function in `Utils/input_validation.py`.) Every literal in the module protects nothing
at runtime — only the test file exercises it, so the tests are green and vacuous.

Worse, the rules are wrong in both directions. `log_sanitizer.py:16` names
`claude-[a-zA-Z0-9-]+` as "Anthropic keys" — that is the **model-id** prefix. `:15`'s
`sk-[a-zA-Z0-9]{20,}` character class excludes `-`, so it stops at `sk-ant`/`sk-proj`.

**Verification** (scratchpad `verify2.py`):

```
IN : Using model claude-opus-4-20250514 for chat
OUT: Using model ***ANTHROPIC_KEY*** for chat                 <- model name destroyed
IN : key sk-ant-api03-AbCdEf0123456789AbCdEf0123456789
OUT: key sk-ant-api03-AbCdEf0123456789AbCdEf0123456789        <- real Anthropic key SURVIVES
IN : key sk-proj-AbCdEf0123456789AbCdEf0123456789
OUT: key sk-proj-AbCdEf0123456789AbCdEf0123456789             <- real OpenAI key SURVIVES

sanitize_dict(...) -> {'x-api-key': 'sk-ant-secret',   <- NOT redacted
                       'cohere_api_key': 'abc',        <- NOT redacted
                       'openai_api_key': '***REDACTED***',
                       'api_token': 'ghp_x',           <- NOT redacted
                       'secret_key': 's',              <- NOT redacted
                       'refresh_token': 'r',           <- NOT redacted
                       'auth_token': '***REDACTED***'}
```

If this module is ever wired in as-is, it will redact model names in logs while passing real keys
through. Its `SENSITIVE_FIELDS` also names ten literals that appear nowhere in this codebase
(`aws_access_key_id`, `aws_secret_access_key`, `connection_string`, `api_secret`, `api-secret`,
`credentials`, `pwd`, `passwd`, `api-key`, `private_key`) and misses ~30 real ones.

---

### IMPORTANT-4 — Two "identical" sensitive-key checks disagree in both directions

`config.py:3691` `_is_sensitive_setting_key` (substring `in`) and
`UI/Screens/settings_privacy_security.py:190` `_is_sensitive_config_key` (`endswith`) enumerate
the *same* literal sets with *different* matching semantics, and only the second guards
`_env_var`. A third, strictly smaller copy lives at `Utils/config_encryption.py:250`
(CRITICAL-2), and a fourth duplicates `config.py:3693-3718` verbatim at `config.py:515-546`.

**Verification** (scratchpad `verify2.py`):

```
key                              config._is_sensitive_setting_key   settings_privacy._is_sensitive_config_key
api_key_env_var                  True                               False
max_tokens                       True                               False
OPENAI_API_KEY_fallback          True                               False
ELEVENLABS_API_KEY_fallback      True                               False
search_engine_api_key_bing       True                               False
api_key                          True                               True
openai_api_key                   True                               True
```

Two concrete consequences. **Over-reach:** `config.py` treats `api_key_env_var` (an env-var
*name*, appearing 20× in the default TOML at `config.py:2330+`) and `max_tokens` (30+
occurrences) as secrets — `encrypt_sensitive_fields` (`:554`) encrypts the env-var name into an
`enc:` blob and `_setting_value_for_log` (`:3721`) logs `max_tokens` as `<redacted>`.
**Under-reach:** the Privacy & Security panel's counts (`settings_privacy_security.py:153,160`)
miss the real `[app_tts]` `*_fallback` keys and the `search_engine_api_key_*` keys.

Both lists also name seven literals that are not config keys anywhere (`apikey`, `api-key`,
`secret`, `secret_key`, `access_token`, `refresh_token`, `client_secret`).

---

### IMPORTANT-5 — `get_detected_api_providers()` always returns `[]`

`config.py:4297-4303` iterates `config.items()` looking for `section_name.startswith("api_settings.")`.
`tomllib` parses `[api_settings.openai]` into a *nested* dict, so the top-level key is
`"api_settings"` and the prefix test is never true.

**What it fails to protect:** nothing directly — but four live UI call sites
(`UI/Tools_Settings_Window.py:998, 3161, 3837, 6757`) use it to tell the user which providers
have keys stored, so the encryption/privacy surface always reports zero.

**Verification** (scratchpad `verify1.py`):

```
get_detected_api_providers() -> []
top-level keys containing 'api_settings': ['api_settings']
any key startswith('api_settings.') : False
raw['api_settings'] subkey count    : 28
```

---

### IMPORTANT-6 — Workspace folder-binding denies only `Path.home()`, never consults the denylist

`Workspaces/registry_service.py:673-707`. The docstring (`:666-669`) claims it denies "the
filesystem root, the home directory itself, non-directories, and duplicate/nested roots":

```python
        if resolved == Path(resolved.anchor):                  # :683
        if resolved == Path.home().resolve():                  # :688
```

Two exact-equality literals. `grep -rn sensitive_paths tldw_chatbook/Workspaces/` → nothing.

**What it fails to protect:** binding `~/.config/tldw_cli` (the live config with API keys),
`~/.local/share/tldw_cli` (every app DB), or `~/.ssh` as a workspace folder root all pass. Since
a bound folder root widens what the agent file tools may reach, this is the mechanism by which
CRITICAL-3's uncovered trust store becomes reachable. `is_sensitive_path` still refuses the
individual enumerated files, so this is *scope widening*, not a direct bypass — but it widens the
scope right up to the boundary of the paths the denylist does not enumerate.

---

### IMPORTANT-7 — Evals DB named three different ways; two of them create empty databases

| Source | Path |
|---|---|
| `Event_Handlers/eval_db_operations.py:28` | `Path.home()/".config"/"tldw_cli"/"evals.db"` |
| `Evals/eval_orchestrator.py:90-99` (the real one) | `get_user_data_dir()/<user_id>/"evals.db"` |
| `UI/Tools_Settings_Window.py:6493` | key `evals_db_path`, default `~/.local/share/tldw_cli/tldw_evals_db.db` |

`evals_db_path`, `rag_db_path` and `subscriptions_db_path` are **declared nowhere** in
`config.py`'s `[database]` defaults (`config.py:2226-2246`).

**Verification** (scratchpad `verify4.py`, sandboxed HOME):

```
eval_db_operations.py:28 literal : …/.config/tldw_cli/evals.db
eval_orchestrator.py:94 derived  : …/.local/share/tldw_cli/default_user/evals.db
Tools_Settings_Window:6493 key   : evals_db_path -> '<undeclared>'
```

Same defect class at `UI/Tools_Settings_Window.py:6480-6507` and `:6631-6652`: the DB path map
backing integrity check, backup, vacuum and chatbook import omits the `<user_folder>` segment for
all six DBs *and* uses wrong filenames (`tldw_prompts_db.db` vs. real `tldw_chatbook_prompts.db`;
`tldw_media_db.db` vs. real `tldw_chatbook_media_v2.db`). Those maintenance operations run
against files the app never opens.

---

### IMPORTANT-8 — `[subscriptions.security]` declares five security switches that nothing reads

`config.py:3471-3476`, present in the live config at lines 1142-1148. Per-key grep across
`tldw_chatbook/` returns exactly one occurrence each — the declaration.

| Declared key | What actually governs it |
|---|---|
| `enable_ssrf_protection` | `get_cli_setting("web_security","enabled",True)` — `Utils/egress.py:88` |
| `max_redirects = 5` | `egress.MAX_REDIRECT_HOPS = 10` — hard-coded, `egress.py:44` |
| `verify_ssl_certificates` | per-subscription DB column `ssl_verify` — `Subscriptions/monitoring_engine.py:393,397` |
| `enable_xxe_protection` | unconditional `defusedxml` import — `Subscriptions/security.py:24-33` |
| `request_timeout` | per-call `timeout=` arguments |

**What it fails to protect:** an operator who hardens their config here gets no change in
behavior — the app still follows 10 redirect hops and still fetches with `verify=False` for any
subscription whose `ssl_verify` column is 0. The inverse is worse: `enable_ssrf_protection = false`
reads as a documented escape hatch that silently does nothing.

**Verification** (scratchpad `verify4.py`):

```
get_cli_setting('subscriptions.security','max_redirects')          -> 5
egress.MAX_REDIRECT_HOPS (the real limit)                          = 10
get_cli_setting('subscriptions.security','verify_ssl_certificates') -> True
get_cli_setting('subscriptions.security','enable_ssrf_protection')  -> True
```

---

### IMPORTANT-9 — `Subscriptions/security.py` ships a stale, unread cloud-metadata denylist

`Subscriptions/security.py:79,85-89` defines `BLOCKED_SCHEMES` and `METADATA_ENDPOINTS`. Both
have exactly one occurrence in `tldw_chatbook/` — the definition. The real enforcement is
`Utils/egress.py:35-44`, reached from `security.py:129-135` via `evaluate_url_policy`.

**Verification** (scratchpad `verify4.py`):

```
security.SecurityValidator.METADATA_ENDPOINTS : {'metadata.azure.com','metadata.google.internal','169.254.169.254'}
egress.METADATA_HOSTNAMES                     : frozenset({'metadata.azure.com','metadata.google.internal'})
egress._METADATA_IPS                          : frozenset({169.254.169.254, 100.100.100.200, fd00:ec2::254})
in security's list but NOT enforced by egress : []
enforced by egress but MISSING from security  : ['100.100.100.200', 'fd00:ec2::254']
```

Harmless today (egress is strictly stronger), but it is an authoritative-looking denylist that
enforces nothing and is already two endpoints behind the real one — a trap for the next reader
who extends "the" metadata list.

---

### IMPORTANT-10 — App-state files written to `Path.home()/".config"/"tldw_cli"` ignore `TLDW_CONFIG_PATH`

`UI/Screens/chat_screen.py:15394,15425` (`ui_state.toml`),
`Event_Handlers/notes_events.py:144` and `note_ingest_events.py:350` (`note_templates.json`),
`Subscriptions/website_monitor.py:72` (`feed_cache/`), plus ~25 lower-value sites listed in the
inventory. All should derive from `_get_effective_config_path().parent`.

Separately, ~18 sites use `Path.home()/".local"/"share"/"tldw_cli"/…` and omit the
`<user_folder>` segment (`Chatbooks/chatbook_importer.py:77-79`,
`Chatbooks/local_chatbook_service.py:102-107`, `Character_Chat/Character_Chat_Lib.py:1274,2790,3856`,
`Event_Handlers/conv_char_events.py:4152,4213,4264`, …), so multi-user profiles collide.

**Verification** (scratchpad `verify3.py`):

```
chatbook_importer.py:77 literal : /Users/macbook-dev/.local/share/tldw_cli/temp/imports          exists=True
get_user_data_dir()-derived     : /Users/macbook-dev/.local/share/tldw_cli/default_user/temp/imports  exists=False
chatbook_creator.py:97 derived  : /Users/macbook-dev/.local/share/tldw_cli/default_user/temp/chatbooks
MATCH: False
```
Note the literal directory **exists** while the derived one does not — the importer has been
extracting outside the per-user tree in production.

---

### IMPORTANT-11 — `[tools] file_sandbox_root` is declared in no config and written by no UI

`Tools/file_operation_tools.py:23-28` reads `get_cli_setting("tools", "file_sandbox_root", default_root)`.
`file_sandbox_root` appears **zero times** in `config.py`'s `CONFIG_TOML_CONTENT` and zero times
in the live config. The Settings screen writes only the seven `*_enabled` gate keys
(`UI/Tools_Settings_Window.py:4206` over `Agents/tool_catalog.py:213-238`).

**Verification** (scratchpad `verify3.py`):

```
get_cli_setting('tools','file_sandbox_root', '<default>') -> '<default>'
default the tool falls back to: …/default_user/tool_sandbox exists= True
declared in CONFIG_TOML_CONTENT: False
[tools] section in CONFIG_TOML_CONTENT: False
```

Consequence: the `_sandbox_root_is_hidden` guard (`file_operation_tools.py:674-699`, called at
`:759` and `:915`) that exists to handle a dotted sandbox root is unreachable in practice, and a
user cannot narrow or relocate the sandbox through any surfaced path. Already filed as
`backlog/tasks/task-693`.

---

### Inverse findings — checks that over-block

**INV-1 (verified live) — `sql_validation.VALID_TABLES["chachanotes"]` omits 26 real tables, and one omission breaks a shipped feature.**
`DB/sql_validation.py:14-24` lists 9 tables; the schema has 35. It lists the *link* table
`collection_keywords` but not the *entity* table `keyword_collections`.
`ChaChaNotes_DB.py:9309` `update_keyword_collection` → `_update_generic_item` → `:4312`
`validate_table_name(table_name, "chachanotes")` → `ValueError`.

```
$ python -c "... db.add_keyword_collection('Coll A'); db.update_keyword_collection(1, {'name':'Coll B'}, expected_version=1)"
created collection id: 1
update_keyword_collection -> ValueError: Invalid table name: keyword_collections
```
Full allowlist/schema diff (scratchpad `verify4.py`):
```
listed but NOT a real table : []
real but NOT in allowlist   : ['character_expression_images','chat_dictionaries','conversation_dictionaries',
 'conversation_local_marks','conversation_world_books','db_schema_version','decks','flashcard_assets',
 'flashcard_templates','flashcards','keyword_collections','learning_paths','message_attachments',
 'message_generation_metadata','mindmap_nodes','mindmaps','quiz_attempts','quiz_questions','quizzes',
 'review_history','study_sessions','sync_conflicts','sync_sessions','topics','world_book_entries','world_books']
```

**INV-2 — `sql_validation.py:308` `VALID_COLUMNS` gate silently no-ops for five tables.**
The gate is `if table_name and table_name in VALID_COLUMNS`. Call sites pass
`"sync_profile_state"` (`Sync_Interop/sync_state_repository.py:1875`, immediately before an
`ALTER TABLE … ADD COLUMN` f-string) and `Transcripts`/`MediaChunks`/`UnvectorizedMediaChunks`/
`DocumentVersions` (`DB/Client_Media_DB_v2.py:2950-2952`, `:3180-3182`) — none is a
`VALID_COLUMNS` key, so only the generic `\w+`/reserved-word filter applies. Not exploitable
today (all inputs are in-file literals), but the schema check those call sites' comments claim is
not delivered.

**INV-3 — `validate_path_simple`'s `dangerous_patterns` rejects `~/`.**
`Utils/path_validation.py:311`. Six `Evals/task_loader.py` sites (`:217, :340, :393, :477, :671,
:812`) pass raw user input without `expanduser()` first, so a user typing `~/evals/task.json` is
rejected as a "dangerous pattern". `UI/Screens/library_screen.py:5519,7336,9536,11438,11966` all
expand first — the handling is inconsistent.

**INV-4 — `config.py`'s substring matcher over-encrypts and over-redacts.** See IMPORTANT-4:
`api_key_env_var` (an env-var *name*) is encrypted into the config, and `max_tokens` is logged as
`<redacted>`.

**INV-5 — Two stale comments assert a `validate_path` behavior removed on 2026-07-24.**
`Chatbooks/chatbook_importer.py:544-549` and `app.py:7584-7585` both claim `validate_path`
"rejects ANY resolved path containing a dot component". Commit `bc804b792d` changed it to inspect
only the user portion; `path_validation.py:59-64` now states the opposite explicitly. Both files
hand-roll a replacement whose stated justification is false.

**INV-6 — Dead config keys that read as security controls.**
`[github] respect_gitignore` / `show_hidden_files` / `profiles_directory` (`config.py:3541-3542,3562`)
have zero readers. What actually runs is a hardcoded ten-name list at
`UI/CodeRepoCopyPasteWindow.py:392-406`; no `.gitignore` is parsed anywhere at runtime (`pathspec`
is not a dependency). Two entries in that list (`.venv`, `.env`) are unreachable — the preceding
`not d.startswith(".")` already excluded them. The config comment promises hidden files are shown
"except .gitignore"; `:425` skips every dotfile including `.gitignore` itself.

```
get_cli_setting('github','respect_gitignore')  -> True
get_cli_setting('github','show_hidden_files')  -> False
get_cli_setting('github','profiles_directory') -> '~/.config/tldw_cli/github_profiles'
```

**INV-7 — Dead path constants contradicting the real accessors.**
`Metrics/logger_config.py:21-22` (`~/.local/tldw_cli/Logs/…` — missing `share` *and*
`<user_folder>`, on a `setup_logger` with zero callers) and `Utils/Utils.py:77-79` /
`Utils/paths.py:205-249` (`~/.config/tldw_cli/tldw_cli_Media.db`, zero callers, contradicting
`get_media_db_path()`). `Utils/paths.py` therefore ships two contradictory answers to "where is
the user's data" — its own `get_user_data_dir()` at `:123` correctly delegates to `config`.

---

## Is `Utils/sensitive_paths.py` clean?

**Yes, for everything it enumerates.** Every entry resolves through the app's own accessor and
matches the real on-disk path on this machine. Verified (scratchpad `verify2.py`):

```
ctx.user_data_dir: /Users/macbook-dev/.local/share/tldw_cli/default_user
  file : /Users/macbook-dev/.config/tldw_cli/config.toml                                  exists=True
  file : …/default_user/mcp_permissions.json      exists=False   (not yet created; path is correct)
  file : …/default_user/local_mcp_store.json      exists=True
  file : …/default_user/mcp_execution_log.jsonl   exists=False
  db   : 11 accessors resolved, ALL exist on disk

is_sensitive_path(…/default_user/mcp_permissions.json)      -> True
is_sensitive_path(…/default_user/local_mcp_store.json)      -> True
is_sensitive_path(…/default_user/mcp_execution_log.jsonl)   -> True
is_sensitive_path(/Users/macbook-dev/.config/tldw_cli/config.toml) -> True
is_sensitive_path(…/default_user/tldw_chatbook_media_v2.db) -> True
is_sensitive_path(…/default_user/mcp_permissions.json.bak)  -> True   (user_data_dir file rule)
is_sensitive_path(…/default_user/mcp_execution_log.jsonl.1) -> True   (user_data_dir file rule)
is_sensitive_path(~/.ssh/id_rsa)                            -> True
```

Enforcement is genuinely wired: `Tools/file_operation_tools.py:67, 148, 278, 329, 492, 771, 937`
call `is_sensitive_path` / `resolve_sensitive_context` across `ReadFileTool`, `WriteFileTool`,
`ListDirectoryTool`, `GlobFiles` and `GrepFiles`.

The remaining gaps are **omissions, not drift**: `skills/trust/*` (CRITICAL-3, structurally
excluded by the directory-exemption rule), `config.toml`'s `.bak`/`.tmp` sidecars (CRITICAL-4),
and the rest of `~/.config/tldw_cli/` (`runtime_policy.json`, `ui_state.toml`, and the
`~/.config/tldw_cli/` variants of the MCP stores from IMPORTANT-2). Nothing the module names
points at a path that does not exist.

One residual fragility worth recording: `sensitive_paths.py:209-211` re-spells the three MCP
filenames as its own literals, joined to `get_user_data_dir()`. They agree with
`unified_control_plane_service.py:2430`/`:2073` today only because `store.path`'s parent happens
to be `get_user_data_dir()`. If `app.py:5241` ever moves `local_mcp_store.json` into a
subdirectory, lines 209-211 stop matching **and** the `parent == user_data_dir` fallback stops
firing — Finding 1 reintroduced, one refactor away. `Tests/Utils/test_sensitive_paths.py:73-75`
re-derives from the same `app.py` expression rather than from
`app.unified_mcp_service.permission_store.path`, so it would not catch that variant.

---

## Recommended backlog tasks

1. **Route the three config-encryption entry points through `_get_effective_config_path()`**
   (CRITICAL-1). Regression test must set `TLDW_CONFIG_PATH` and assert the *active* file changed.
   Fold in the non-atomic write at `config.py:4330`.
2. **Replace `detect_api_keys`'s six-literal exact-match with the shared predicate** (CRITICAL-2),
   and collapse the four duplicate sensitive-key lists (`config.py:515`, `config.py:3693`,
   `config_encryption.py:250`, `settings_privacy_security.py:10`) into one module with one
   semantics, keeping the `_env_var` guard and excluding `max_tokens` (CRITICAL-2 + IMPORTANT-4 +
   INV-4). Tests must enumerate real key names read out of `CONFIG_TOML_CONTENT`, not literals.
3. **Cover the skill trust/grant store in `sensitive_paths.py`** (CRITICAL-3) — derive from
   `SkillTrustService`/`SkillTrustStore` attributes, not from a re-spelled `skills/trust` literal,
   and reconcile with the deliberate `skills` directory exemption.
4. **Cover `config.toml`'s `.bak`/`.tmp` sidecars** (CRITICAL-4), mirroring the existing
   `_DB_SIDECAR_SUFFIXES` treatment; consider whether the whole effective-config *directory*
   should get the same file-rule the user data dir has.
5. **Fix the two skills-trust containment checks** (CRITICAL-5 + CRITICAL-6): pass
   `base_dir=store_dir` for the marker, and make `_unsafe_scratch_root` reject roots that
   *contain* the stores as well as roots inside them (add `self.store_dir` to the container list).
6. **Fix `MCP/server.py:154` to call `get_media_db_path()`** (IMPORTANT-1), and grep for any other
   `get_cli_setting("database", ...)` whose key is not one of the declared `*_db_path` names.
7. **Make the three MCP store module defaults derive from `get_user_data_dir()`** or drop the
   defaults entirely and require an explicit path (IMPORTANT-2).
8. **Decide `Utils/log_sanitizer.py`'s fate** (IMPORTANT-3): delete it, or fix the regexes and
   field list and actually wire it in. Shipping it unimported with an inverted Anthropic rule is
   the worst of both.
9. **Have `Workspaces.add_folder_binding` consult `is_sensitive_path`** before accepting a root
   (IMPORTANT-6) — this is the reachability gate for finding 3.
10. **Reconcile the evals-DB and settings DB-path maps with the `get_*_db_path()` accessors**
    (IMPORTANT-7), and declare or delete `evals_db_path` / `rag_db_path` / `subscriptions_db_path`.
11. **Delete or implement `[subscriptions.security]` and `Subscriptions/security.py`'s unread
    denylists** (IMPORTANT-8 + IMPORTANT-9). A config switch named `enable_ssrf_protection` that
    does nothing is worse than no switch.
12. **Add `keyword_collections` (and audit the other 25 omissions) in
    `sql_validation.VALID_TABLES`** (INV-1) — this is a live, reproducible feature break — and
    decide whether `validate_column_name` should fail closed for tables absent from `VALID_COLUMNS`
    (INV-2).
13. **Sweep `Path.home()/".config"/"tldw_cli"` and `Path.home()/".local"/"share"/"tldw_cli"` call
    sites onto the accessors** (IMPORTANT-10); the `chatbook_importer` extraction root is the
    highest-value one since its literal directory demonstrably exists in production.
14. **Cross-cutting test rule (task AC #3):** any test asserting one of these paths must
    re-derive it from the same accessor the app uses. Two existing tests currently re-spell an
    `app.py` literal instead (`Tests/Utils/test_sensitive_paths.py:73-75`,
    `Tests/conftest.py:566-575` + `Tests/Skills/test_skills_library_flow.py:84-106`), so they
    would go vacuous in lockstep with a drift rather than catch it.

### Lower priority (record, don't schedule separately)

- INV-3 `~/` in `dangerous_patterns` vs. `Evals/task_loader.py` (6 sites).
- INV-5 two stale comments citing removed `validate_path` behavior.
- INV-6 dead `[github]` keys + the unreachable `.venv`/`.env` entries.
- INV-7 dead `Metrics/logger_config.py` and `Utils/Utils.py` path constants.
- IMPORTANT-5 `get_detected_api_providers()` always `[]`.
- IMPORTANT-11 `[tools] file_sandbox_root` undeclared (already `task-693`).
