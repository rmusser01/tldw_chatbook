---
id: TASK-35835
title: Quitting after a provider save rewrites config.toml and restores the key env var the save removed
status: In Progress
assignee:
  - '@claude'
created_date: 2026-10-20 14:53
updated_date: '2026-10-22 19:32'
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
- [ ] #1 A test confirms `persist_cli_config_for_shutdown()` is the writer, and records whether its whole-file rewrite (comments stripped, template order) is intended
- [ ] #2 After a provider save, quitting leaves config.toml byte-identical to what the save wrote
- [ ] #3 A test pins it: save a keyless provider, quit, and the file keeps the saved credential fields
<!-- AC:END -->

## Implementation Plan

1. Establish intent from history: the pre-TASK-491 quit code, a9ccde5723 (TASK-491 centralisation), b5251e9a6e (TASK-32628 recovery guard, "a defaults-only rewrite"), b04fc23ebc (TASK-33001.7 workaround), and ADR-029.
2. Red first at dev: extend `test_settings_keyless_provider_save_keeps_template_env_name_out` (real Settings save on a private profile, then `persist_cli_config_for_shutdown()`) to require the saved bytes to survive; extend `test_real_app_quit_persists_only_the_effective_config` (the real quit path) to require the shutdown persist to leave the file byte-identical, as the no-save control.
3. Fix in the shared `persist_cli_config_for_shutdown()`: keep its recovery guard and its "file loads safely" gate, drop the template-merge rewrite, and write only when the file itself holds a plaintext secret under unlocked encryption (the one thing it carries), encrypting the file's own contents.
4. Update the pins that recorded the old rewrite (the hub test comments, `test_shutdown_persistence_uses_only_effective_path`), the User Guide Credentials row, and the TASK-33001.7 lesson.
5. Run every test that names the shutdown persist, the recovery guard or the raw writer, plus preflight and the size ratchet.
