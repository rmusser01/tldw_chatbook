# TASK-851 / TASK-852: config encryption effective-path + detection fix

Branch: `fix/config-encryption-effective-path` (worktree `wt-cfg-crypto`, cut from `origin/dev`).

Together these two defects meant config encryption did not encrypt: the
encryption entry points wrote to the wrong file when a profile was active
(851), and the detector that decides whether to offer encryption never
recognized any of this app's real API-key field names (852).

## TASK-851 — wrong file

### Root cause

`enable_config_encryption`, `disable_config_encryption`, and
`change_encryption_password` in `tldw_chatbook/config.py` all read/wrote
`DEFAULT_CONFIG_PATH` (`Path.home()/.config/tldw_cli/config.toml`) directly,
instead of `_get_effective_config_path()`, which is the only accessor that
honors the `TLDW_CONFIG_PATH` override (the app's "Override config" /
profile mechanism). `enable_config_encryption` also wrote with a plain
`open(path, "w")` + `toml.dump(...)`, unlike its two siblings which already
used `atomic_write_text`.

### Reproduction — before fix

Script (paraphrased; full version was run via `git stash` against the
worktree to get a clean unfixed baseline):

```python
import tldw_chatbook.config as cfg
# ... HOME redirected to a scratch dir; a prior "normal run" already created
# DEFAULT_CONFIG_PATH via load_cli_config_and_ensure_existence() ...

profile_path.write_text(
    '[api_settings.openai]\napi_key = "sk-proj-PLAINTEXT-SENTINEL-abc123"\n'
)
os.environ["TLDW_CONFIG_PATH"] = str(profile_path)

result = cfg.enable_config_encryption("hunter2hunter2")
```

Output:

```
effective path (should be the profile): .../cfg851profile_7ky1asp8/profile_config.toml
enable_config_encryption(...) -> True
---- ACTIVE profile file contents after enable ----
[api_settings.openai]
api_key = "sk-proj-PLAINTEXT-SENTINEL-abc123"

plaintext key STILL present in ACTIVE file: True
DEFAULT_CONFIG_PATH (wrong file) shows an 'encryption' section now: True

BUG REPRODUCED: reported success, but the ACTIVE file's secret is untouched
plaintext, and a completely different file was rewritten instead.
```

### After fix

Same script, same inputs, against the fixed code:

```
enable_config_encryption(...) -> True
---- ACTIVE profile file contents after enable ----
[encryption]
enabled = true
method = "AES-256-GCM-scrypt"
version = 1
password_verifier = "enc:ApDRu5fU7ICrSWiPzI+6DZcc23kDP4VxCyo6H2Pj0GfEQ65gi7pYh4mFuvkKpTvFVuanYQKpuLYiaxaJH6F3fneGifF+5kVVCRSvFGzpYTN4ZpwaTaThXEsP2Z391bAGGXe7wHKYIA3HhoViHOlXva1Cp/MpQzeAXg=="

[api_settings.openai]
api_key = "enc:Ak49BRW40QdCukYtpdMopTiCoCUhTSUDi0xj/cLiL5lny43CheiQX/yGYQwLDABStNz6kL0DJWq+IFmf8EIUC6jubktAGB8ZSg1CVMmPYqP+bX18JdFDaAYjXwzPUw=="

plaintext key STILL present in ACTIVE file: False
DEFAULT_CONFIG_PATH (wrong file) shows an 'encryption' section now: False
```

The active (effective) file is the one that changed; the decoy
`DEFAULT_CONFIG_PATH` was never created.

### Fix

All three entry points now do:

```python
config_path = _get_effective_config_path()
...
if config_path.exists():
    with open(config_path, "rb") as f:
        config_data = tomllib.load(f)
...
atomic_write_text(config_path, toml.dumps(encrypted_config), encoding="utf-8")
```

`atomic_write_text` (`Utils/atomic_file_ops.py`, already used elsewhere in
this codebase, e.g. `apply_settings_mutation_to_cli_config`) writes to a
`tempfile.mkstemp` sibling in the same directory, `fsync`s it, then
`os.replace`s it over the target — so a crash or exception during
serialization never touches the original file. `disable_config_encryption`
and `change_encryption_password` already called this helper (just against
the wrong path); `enable_config_encryption` previously used
`open(path, "w")` (which truncates on `open()`, before any content — or a
raised exception — ever reaches the file) and now uses the same helper.

### Tests — `Tests/test_config_encryption_effective_path.py` (new, 6 tests)

- `test_enable_config_encryption_writes_active_file_not_default`
- `test_disable_config_encryption_reads_and_writes_active_file`
- `test_change_encryption_password_reads_and_writes_active_file`
- `test_enable_disable_roundtrip_with_profile_active` (byte-identical
  restore of the original plaintext config)
- `test_change_password_roundtrip_with_profile_active` (old password
  rejected after rotation, new password works)
- `test_enable_config_encryption_write_is_atomic` (patches both
  `toml.dump` and `toml.dumps` to raise mid-serialization; asserts the
  on-disk file is byte-for-byte unchanged)

All 6 were run against the pre-fix code via `git stash` and failed (5 with
the wrong-file symptom, 1 — the atomicity test — was rewritten once to
isolate purely the atomicity property, independent of the path bug, after
an initial version produced a false pass for the wrong reason). All 6 pass
against the fixed code.

## TASK-852 — detector never fires

### Root cause

`Utils/config_encryption.py`'s `detect_api_keys` matched key names via exact
equality against six literals (`api_key`, `apikey`, `api-key`, `secret`,
`token`, `password`). Every real secret-bearing key in this app is prefixed
or suffixed instead (`bing_search_api_key`, `auth_token`, `api_token`,
`OPENAI_API_KEY_fallback`, ...), so none of them were ever detected, and
`check_encryption_needed()` (`config.py:4270`, the `should_encrypt_config()`
gate referenced by the task) never prompted a user with a config full of
live plaintext secrets to turn encryption on.

There were also three *other* independent copies of "is this key sensitive,"
all disagreeing:

- `config.py:515-546` (inside `encrypt_api_keys_in_config`'s
  `encrypt_sensitive_fields` closure) and `config.py:3691`
  (`_is_sensitive_setting_key`, a near-duplicate) — bare substring
  containment, so `max_tokens` (contains `_token`) and `api_key_env_var`
  (contains `api_key`) were both wrongly flagged.
- `UI/Screens/settings_privacy_security.py:190` (`_is_sensitive_config_key`)
  — `endswith`-only plus an `_env_var` guard, safer than substring but
  missed keys where the sensitive fragment isn't a suffix
  (`search_engine_api_key_baidu`, the `_fallback` family).

### Reproduction — before fix

```python
from tldw_chatbook.Utils.config_encryption import ConfigEncryption
enc = ConfigEncryption()
config = {
    "SearchEngines": {"bing_search_api_key": "bing-real-plaintext-secret"},
    "tldw_api": {"auth_token": "tldw-api-plaintext-secret"},
    "github": {"api_token": "github-plaintext-secret"},
}
enc.detect_api_keys(config)   # -> False (bug)
enc.detect_api_keys({"x": {"api_key": "sk-1"}})  # -> True (only literal names ever worked)
```

### Fix

New module `tldw_chatbook/Utils/sensitive_config_keys.py` — the single
shared predicate, `is_sensitive_config_key(key)`:

1. `_env_var` suffix guard first (env var *names* are never secrets) —
   overrides everything else.
2. Exact-name set: `api_key`, `apikey`, `api-key`, `secret`, `token`,
   `password`, `auth_token`, `api_token`, `access_token`, `secret_key`,
   `refresh_token`, `client_secret`.
3. Containment rule for `api_key` / `apikey` / `api-key` — catches
   provider-prefixed names (`openai_api_key`) and suffix-of-a-suffix names
   (`OPENAI_API_KEY_fallback`, `search_engine_api_key_baidu`) that an
   `endswith`-only rule misses.
4. `endswith` rule (not containment) for `_key`, `_token`, `_secret`,
   `_password` — deliberately `endswith`, not containment: `max_tokens`
   contains `_token` as a bare substring but does not end with it, so it is
   correctly excluded without needing a special case.

All four former call sites now delegate to it:
`config.py` (`encrypt_sensitive_fields` closure; `_is_sensitive_setting_key`
was removed and its two callers — `_setting_value_for_log`,
`_maybe_encrypt_setting_value` — call the shared predicate directly),
`Utils/config_encryption.py::detect_api_keys`, and
`UI/Screens/settings_privacy_security.py` (`_is_sensitive_config_key` and its
two frozensets removed in favor of the shared import).

### After fix

Same repro:

```python
enc.detect_api_keys(config)  # -> True
enc.detect_api_keys({"api_settings": {"openai": {"api_key_env_var": "OPENAI_API_KEY", "max_tokens": 4096}}})  # -> False
```

### Tests

- `Tests/Utils/test_sensitive_config_keys.py` (new, 15 tests) — builds its
  key list by parsing the real `CONFIG_TOML_CONTENT` (via `tomllib.loads`)
  and `DEFAULT_APP_TTS_CONFIG`, not by re-typing literals: real
  `[SearchEngines].*_api_key` names, `[tldw_api].auth_token`,
  `[github].api_token`, the `*_fallback` family, plus explicit exclusions
  (`api_key_env_var`, `api_token_env_var`, `max_tokens`, and the
  `default_keyword_filter` / `extract_keywords` / `max_keywords` /
  `skip_on_keypress` false positives the old substring-matching copy
  produced).
- `Tests/Utils/test_config_encryption.py` — added
  `TestDetectApiKeysRealProviderNames` (3 tests) exercising
  `detect_api_keys` itself with the same real-key-sourcing approach.
- `Tests/test_config_app_config_encryption.py` — added one end-to-end test
  of `check_encryption_needed()` against a live `TLDW_CONFIG_PATH` config
  holding a real plaintext provider key.

All new/updated tests were confirmed to fail against the pre-fix code
(`git stash`) and pass after.

## Where the unified key list lives / semantics chosen

`tldw_chatbook/Utils/sensitive_config_keys.py` — deliberately a small,
dependency-free module (no `Cryptodome` import) so it can be imported
eagerly from `config.py` (which otherwise lazy-loads
`Utils/config_encryption.py` specifically to avoid an eager `Cryptodome`
import at startup) without changing that startup-cost tradeoff.

Semantics: exact-match ∪ containment(`api_key`/`apikey`/`api-key`) ∪
endswith(`_key`/`_token`/`_secret`/`_password`), with an `_env_var`-suffix
guard checked first and winning over everything else. See the module
docstring for the full "why containment here, endswith there" rationale.

## Atomic write

`enable_config_encryption` now serializes with `toml.dumps(...)` first, then
passes the resulting string to `atomic_write_text(path, ...)` — the same
write-temp-then-`os.replace` helper (`Utils/atomic_file_ops.py`) its two
siblings already used. No new helper was written; this was purely routing
`enable_config_encryption` onto the existing one.

## Exact pytest commands run and results

```
# Task-specified starting point (note: pytest collapses "Tests/Utils/ Tests/"
# to just Tests/Utils/'s 476 tests when both paths are given together in
# this environment/pytest version -- observed, not something this task
# fixed; the individual-file runs below cover the top-level Tests/*.py files
# this quirk excludes).
python -m pytest Tests/Utils/ Tests/ -k "config or encrypt or privacy" -q
  -> 83 passed, 393 deselected

# Every file created/edited for this work, run directly:
python -m pytest Tests/Utils/test_config_encryption.py \
                 Tests/Utils/test_sensitive_config_keys.py \
                 Tests/test_config_encryption_effective_path.py \
                 Tests/test_config_app_config_encryption.py \
                 Tests/UI/test_settings_privacy_security.py -q
  -> 71 passed

# Known pre-existing baseline, confirmed unchanged:
python -m pytest Tests/UI/test_tools_settings_window.py -q
  -> 6 failed (test_chat_api_key_*, pre-existing per task brief), 9 passed, 16 skipped

# Broader adjacent-file regression sweep (files not directly touched, but
# config.py-dependent):
python -m pytest Tests/test_config_delete_settings.py Tests/test_config_console_defaults.py \
                 Tests/test_config_library_defaults.py Tests/Utils/test_config_nested_settings.py \
                 Tests/Utils/test_config_import_hygiene.py Tests/UI/test_settings_configuration_hub.py \
                 Tests/UI/test_settings_tools_section.py Tests/UI/test_settings_image_gen_defaults.py -q
  -> 1 failed, 402 passed (0:05:16)
  -- the 1 failure (test_theme_category_opens_without_crashing) is an 8s
     UI-mount timing wait unrelated to config/encryption; re-run in isolation
     passed cleanly (9.53s), confirming pre-existing flakiness under load,
     not a regression from this change.
```

## Files changed

- `tldw_chatbook/config.py` — three encryption entry points routed through
  `_get_effective_config_path()`; `enable_config_encryption` now writes via
  `atomic_write_text`; `encrypt_sensitive_fields` and
  `_setting_value_for_log`/`_maybe_encrypt_setting_value` delegate to the
  shared predicate; `_is_sensitive_setting_key` removed.
- `tldw_chatbook/Utils/config_encryption.py` — `detect_api_keys` delegates
  to the shared predicate.
- `tldw_chatbook/UI/Screens/settings_privacy_security.py` —
  `_is_sensitive_config_key` and its two frozensets removed in favor of the
  shared import.
- `tldw_chatbook/Utils/sensitive_config_keys.py` — new shared predicate
  module.
- `Tests/test_config_encryption_effective_path.py` — new (task-851).
- `Tests/Utils/test_sensitive_config_keys.py` — new (task-852).
- `Tests/Utils/test_config_encryption.py` — extended (task-852).
- `Tests/test_config_app_config_encryption.py` — extended (task-852).

## Out of scope (noted, not fixed)

`UI/Tools_Settings_Window.py::_save_raw_toml_config` (raw TOML editor save,
~line 4248-4260) has the same shape of bug (writes `DEFAULT_CONFIG_PATH`
directly, non-atomically) but is not one of the three named entry points in
task-851's acceptance criteria, so it was left untouched. Worth a follow-up
task if this raw-editor path is meant to also honor `TLDW_CONFIG_PATH`.
