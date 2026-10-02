---
id: TASK-33266
title: 'PERF-07: Memoize get_user_data_dir, DB-path accessors and the sensitive-path
  context'
status: To Do
created_date: 2026-09-28 18:02
dependencies:
- TASK-33260
labels:
- performance
- config
- security
- perf-audit-2026-09
priority: high
references:
- qa/perf-structural-audit-2026-09-27/report.md
- qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
get_user_data_dir() (config.py ~9433; 203 call sites, 29 on the boot path) is uncached. It runs 5 admission scopes, the data-root file lock and a secure_private_directory walk: 29-53 ms and 1,000-1,700 open() calls per call. resolve_sensitive_context calls it 19-20 times (about 607 ms per resolution). That runs for every agent file/git/patch tool call, every @-reference, twice in RunLogWriter.bind per send, for the emergency-stop path on every send and on every 30 s scheduler tick, and for every RAGConfig(). Needs owner decision D2: a generation-keyed memo with a per-call full-chain identity re-check (every verified ancestor's lstat identity against the pinned one, or openat through held verified fds) instead of a per-call chain walk. A leaf-only re-check is not sufficient: it would miss an ancestor being re-permissioned or swapped while the leaf is unchanged. Source: the 2026-09-27 structural efficiency audit at dev 840ed2ca58 (qa/perf-structural-audit-2026-09-27/report.md, section 4, PERF-07; every issue with file:line is listed under PERF-07 in qa/perf-structural-audit-2026-09-27/appendix-issues-by-pr.md).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Owner decision D2 is recorded (ADR amendment or task note) before merge
- [x] #2 get_user_data_dir and the get_*_db_path accessors resolve at most once per config generation and data-dir setting, with a documented identity re-check
- [x] #3 resolve_sensitive_context and default_emergency_stop_path are memoized on the same key
- [x] #4 Replacing or re-permissioning the data directory or any verified ancestor is still detected before dependent access (security tests cover both the leaf and an ancestor)
- [x] #5 Main-thread get_user_data_dir calls before _ui_ready drop from about 40 to at most 2
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision D2 is recorded in the approved ADR-126 amendment (PR #2911, "D2 corollary").

**`get_user_data_dir`**
- After the guarded handshake, the resolution is memoized on two things:
  - the config cache object (compared by identity) with its generation and source;
  - `users_name` and the data base, which is either the configured `data_dir` (made absolute, so a relative value follows the working directory) or the HOME-derived default.
- Every call re-observes posture stamps `(dev, ino, type, mode, uid)` for:
  - every component from `/` to each candidate user dir (conventional and fallback, or the configured base);
  - the conventional root, the fallback root and the root lock file, which are the default-root ambiguity inputs.
- Any mismatch runs `_resolve_user_data_dir`, the unmodified body and the only place that creates, hardens or refuses.
- A path that cannot be stamped (not a directory, unsearchable ancestor) makes the memo step aside, so the resolution reports it as before.
- A result is recorded only when stamps taken before and after that resolution are identical, every component of the resolved chain was stamped (it is one of the candidates), and none is a symlink.
- The memo is guarded by a lock (free-threading safe).

**Accessors built on it.** The `get_*_db_path` accessors and `default_emergency_stop_path` derive from `get_user_data_dir()`, so they inherit the memo.

**`resolve_sensitive_context`** memoizes only its raw, config-derived inputs (`_raw_inputs`).
- The key is the config identity, generation and source, the effective config path, the re-verified user data dir, the whole environment (accessors read overrides such as `RAG_PERSIST_DIR`) and the working directory (a relative custom database path is made absolute against it).
- The key is read again after the inputs are built; inputs built while it moved are returned but not kept.
- Every path is still `_resolved()` on every call, and the docstring is amended accordingly.
- A snapshot built while any accessor failed (counted through `_debug`) is never kept, so the deny list cannot keep a gap.

**Measured** (#2919 branch as base, scratch profile)
- Warm `get_user_data_dir`: 12.0 -> 2.2 ms, 351 -> 104 opens; the rest is the guarded handshake (TASK-33560).
- `resolve_sensitive_context`: 237 -> 3.5 ms, 6,738 -> 360 opens.
- Before `_ui_ready`: 40 main-thread calls now make 2 resolutions instead of 40 (AC #5).

**Tests.** `Tests/test_user_data_dir_memo_perf07.py` runs 11 private-profile tests: warm reuse, a re-permissioned leaf, a replaced leaf, a group-writable ancestor (refused as before), a config reload, a working-directory change during resolution, a data dir that cannot be stamped (the resolution's own PrivatePathError still surfaces), an environment override, an override switched while the inputs build, raw-input reuse until the key moves, and a relative database override after a directory change. With the per-call stamp check removed, the three filesystem tests fail; with the raw-input memo hit removed, the reuse test fails.

**Remote worker bundle.** It embeds `sensitive_paths.py`, so it was regenerated. `_raw_inputs` and `_raw_inputs_key` join the laptop-only stub allowlist: it is reached only through `resolve_sensitive_context`, and it fails closed exactly where that function's own `config` import did.

**Not covered.** The bound-profile branch (`verified_user_data_directory`) is unchanged; it returns before the memo and was never the lock-and-walk path.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
