---
id: TASK-35835
title: Quitting after a provider save rewrites config.toml and restores the key env var the save removed
status: Done
assignee:
  - '@claude'
created_date: 2026-10-20 14:53
updated_date: '2026-10-22 20:13'
labels:
- follow-up
- settings
- config
priority: medium
references:
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/config.py
- tldw_chatbook/app_lifecycle.py
- tldw_chatbook/Backup_Recovery/profile_open.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in S1's live verification (2026-10-20). It reproduces on plain dev 1bbc145ffc and on the S1 head, in sandboxed scratch profiles.

**Steps:**
1. In Settings ▸ Providers & Models, choose a keyless local provider (llama.cpp, "No credential required").
2. Choose a model and press s. The save writes `credential_source = "none"` and removes `api_settings.llama_cpp.api_key_env_var` from config.toml.
3. Quit with nothing unsaved (Ctrl+Q).

**What happens.** At the quit, something rewrites the whole file. The sections come back in template order, and `api_key_env_var = "LLAMA_CPP_API_KEY"` is back.

**Measured** (semantic TOML diff, file mtime at the quit's second):
- dev, before the save: `'LLAMA_CPP_API_KEY'`;
- after the save: `None`;
- after the quit: `'LLAMA_CPP_API_KEY'`, with `credential_source` still `'none'`.

**Why it matters:**
- The file the user saved is not the file that remains.
- Each save-and-quit flips the key.
- The quit-time writer puts the template's default back under a key the save deleted.
- Byte-identity checks such as "a quit leaves the file as it was" cannot hold after any provider save in the same session.

**The writer** (found by the review of S1's live-verification commit, and reproduced with a probe on a private profile): `persist_cli_config_for_shutdown()` (config.py ~8406), called at quit from app_lifecycle.py ~2342 after the cache save.
- It reloads the bootstrap config with `force_reload=True`, which is `deep_merge_dicts(DEFAULT_CONFIG_FROM_TOML, file)`.
- It writes that whole merge through `_write_raw_cli_config_unlocked`. So the sections return in template order, and a template key the save deleted returns.
- Its only skip guard, `unchanged_selected_config` (Backup_Recovery/profile_open.py ~73-84), is true only for a recovered or opened profile selection in the current process. On an ordinary launch it rewrites every time.
- The probe: a mutation that deletes `api_settings.llama_cpp.api_key_env_var`, then `persist_cli_config_for_shutdown()`, restores `'LLAMA_CPP_API_KEY'` with `credential_source` still `'none'`.
- A control with no deletion also rewrote a fresh template file, stripping its comments into a plain dump. The rewrite is idempotent only after a first quit has dumped the file once, which is why a boot alone, a draft with quit-Discard, and Discover with quit-Discard each left the live profile byte-identical.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test confirms `persist_cli_config_for_shutdown()` is the writer, and records whether its whole-file rewrite (comments stripped, template order) is intended
- [x] #2 After a provider save, quitting leaves config.toml byte-identical to what the save wrote
- [x] #3 A test pins it: save a keyless provider, quit, and the file keeps the saved credential fields
<!-- AC:END -->

## Implementation Plan

1. Establish intent from history: the pre-TASK-491 quit code, a9ccde5723 (TASK-491 centralisation), b5251e9a6e (TASK-32628 recovery guard, "a defaults-only rewrite"), b04fc23ebc (TASK-33001.7 workaround), and ADR-029.
2. Red first at dev: extend `test_settings_keyless_provider_save_keeps_template_env_name_out` (real Settings save on a private profile, then `persist_cli_config_for_shutdown()`) to require the saved bytes to survive; extend `test_real_app_quit_persists_only_the_effective_config` (the real quit path) to require the shutdown persist to leave the file byte-identical, as the no-save control.
3. Fix in the shared `persist_cli_config_for_shutdown()`: keep its recovery guard and its "file loads safely" gate, drop the template-merge rewrite, and write only when the file itself holds a plaintext secret under unlocked encryption (the one thing it carries), encrypting the file's own contents.
4. Update the pins that recorded the old rewrite (the hub test comments, `test_shutdown_persistence_uses_only_effective_path`), the User Guide Credentials row, and the TASK-33001.7 lesson.
5. Run every test that names the shutdown persist, the recovery guard or the raw writer, plus preflight and the size ratchet.

## Implementation Notes

**Intent (AC#1): the whole-file rewrite was not intended.** Before TASK-491 the quit ran "Save encrypted config if encryption is enabled": it wrote only when encryption was on and unlocked, to encrypt the keys at rest. a9ccde5723 (TASK-491, 2026-07-24) moved that decision into `_config_data_for_persistence` and wrote unconditionally. Its task and plan route "shutdown persistence" through the config owner; neither asks for defaults on disk. b5251e9a6e (TASK-32628) then called it "a defaults-only rewrite" and skipped it for a recovered profile. b04fc23ebc (TASK-33001.7) worked around it. The function force-reloads from disk, so it never carried in-memory state. Loads re-merge the template and migrations persist through their own writer (`_CONFIG_MIGRATIONS` is empty). So all a quit can still owe the disk is encryption of a plaintext secret, such as a hand edit, in an unlocked encrypted file.

**Fix** (in `persist_cli_config_for_shutdown()`, so every caller gets it):
- The recovery guard and the "file loads safely" gate stay; an unreadable or undecryptable file still returns False.
- The template-merge write is gone.
- The function writes only when the file has encryption enabled, a password is set, and it holds a plaintext sensitive value. It then encrypts the file's own contents, so no template keys come back.

**Tests** (private profiles, all through the wrapper):
- `Tests/UI/test_settings_configuration_hub.py::test_settings_keyless_provider_save_keeps_template_env_name_out` (AC#2, AC#3): a real Settings save, then `persist_cli_config_for_shutdown()`, then the bytes must match what the save wrote. Red at dev on `[legacy-env-unset]`: "At index 26452 diff: b'k' != b'u'", which is `api_key_env_var` re-inserted. The other two cases keep every template key, so dev already left them byte-identical.
- `Tests/ProductionApp/test_config_profile_isolation.py::test_real_app_quit_persists_only_the_effective_config` (AC#1, and the no-save control on the real quit path): the real `TldwCli` quit, with the bytes compared around the real persist. Red at dev: "the shutdown persist rewrote the effective config".
  - It cannot compare the whole session. Console's mount saves `console.rail_state` through the normal writer, which also strips comments; this was found by probe, and is listed as a follow-up.
- `Tests/test_config_persistence_owner.py::test_shutdown_persistence_leaves_a_fresh_template_byte_identical` (new control): a fresh commented template survives a quit. Red at dev: the file became a dump.
- `test_shutdown_persistence_uses_only_effective_path` deliberately pinned the old rewrite. It now adds a hand-added plaintext key, so it covers the remaining write: both keys encrypted, no template `api_settings`. Red at dev on the `api_settings` assert. A mutant that skips the encryption write fails it.
- The stale comments in `test_settings_model_save_keeps_the_stored_key_that_resolves` and `..._writes_a_legacy_api_section_key_...` are updated; their assertions are unchanged.

**Runs** (HEAD 74fc6310a7 vs dev 28beeb9ed5, `-n 4`):
- The 10 covering files in one invocation: 246 failed / 363 passed at HEAD, 246 failed / 362 passed at dev, with identical failing-name sets. The failures are mostly `RecoveryRequired` in setup.
- Clean groups at HEAD:
  - `Tests/UI/test_settings_configuration_hub.py` alone: 242 failed, the same names as dev.
  - The config unit files: 86 passed.
  - `Tests/Backup_Recovery/test_config_participant_lifetimes.py`: 3 failed, the same set at dev.
  - The theme and phase6 group: 1 failed, the same as dev (`'>=3.12' == '>=3.11'`). A first run also failed 3 UI-stall timing tests at load average 21; the rerun passed them.
- Preflight passed.
- Size ratchets: 8 failed, all in files this task did not touch, the same set at dev. `config.py` is not ratcheted.

**Files:**
- `tldw_chatbook/config.py`
- `Tests/test_config_persistence_owner.py`
- `Tests/UI/test_settings_configuration_hub.py`
- `Tests/ProductionApp/test_config_profile_isolation.py`
- `Docs/User_Guide/settings.md`: Credentials row; the name returns in **Env var** after a restart, not in the file.
- `backlog/docs/lessons-testing-evidence.md`: an update to the TASK-33001.7 lesson.
