---
id: TASK-35835
title: Quitting after a provider save rewrites config.toml and restores the key env var the save removed
status: To Do
created_date: 2026-10-20 14:53
labels:
- follow-up
- settings
- config
priority: medium
references:
- tldw_chatbook/Chat/provider_setup_persistence.py
- tldw_chatbook/config.py
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
- An unrelated quit-time writer is overwriting a saved setting with the in-memory default.
- Byte-identity checks such as "a quit leaves the file as it was" cannot hold after any provider save in the same session.

The writer is not identified yet. A boot alone, a draft with quit-Discard, and Discover with quit-Discard each leave the file byte-identical.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The code path that writes config.toml at quit is identified and recorded
- [ ] #2 After a provider save, quitting leaves config.toml byte-identical to what the save wrote
- [ ] #3 A test pins it: save a keyless provider, quit, and the file keeps the saved credential fields
<!-- AC:END -->
