# S25 validation — `Backup_Recovery/`

Validated against worktree HEAD `d0face3ebe` (review baseline `3722a85748`, +25 commits). This package was added to
the slice plan late per the brief, so extra rigor applied below, including two direct re-measurements.

## 1. P1 — Restore publication is O(N²) in `flush_file`/`F_FULLFSYNC` calls (re-measured directly)
- Verdict: CONFIRMED (re-measured, not just traced)
- Site now: `Backup_Recovery/journal.py:1533` `_records`, `:1595` `_flush_record`, `:1626` `_flush_records` (all exact
  line matches to the review); `publication.py:2001` `for item in prepared.artifacts:` loop, `:2706` `_complete_move`,
  `:2727` `_begin_move` (all exact matches).
- Proof — **I reproduced the measurement independently**, without the reviewer's uncommitted `countflush` pytest
  plugin (which is not in the repo). Wrote a one-off `python -` heredoc (no files touched under any protected path)
  that monkeypatches `native_platform.flush_file` at its three import sites (`native_platform`, `journal`,
  `native_files` — each does `from .native_platform import flush_file`, a separate binding per module) with a
  counting/timing wrapper, then calls `pytest.main(["-q", "Tests/Backup_Recovery/test_publication_finalization.py"])`
  in-process so the patch stays live:
  ```
  RESULT rc= 0
  FLUSHCOUNT {'n': 3316, 't': 6.746880225546192} wall= 18.186359332990833
  26 passed
  ```
  `n=3316` is an **exact** match to the review's measured call count (`FLUSHCOUNT ... 'flush_file': 3316`). Wall-clock
  flush time (6.75s) is in the same range as the review's 8.28s (run-to-run disk/cache variance expected, same order
  of magnitude, same test file, same 26 tests).
- Also re-ran the isolated micro-benchmark component: `flush_file` on an open fd vs plain `os.fsync`, 200 iterations
  each: **0.0764 ms/call vs 0.0003 ms/call, 300x** (review: 0.163 ms / 0.0006 ms, 268x — same order of magnitude,
  confirms the Darwin `F_FULLFSYNC` cost is real and dominant).
- `publication.py:1750,1758` `MAX_EVENTS` bound and `Tests/Backup_Recovery/test_large_recovery_records.py:121`
  (1,801 artifacts, one line off the claimed 120) both confirmed present.
- Note: this exceeds "CONFIRMED-BY-TRACE" — a live re-measurement was possible within the stated rules (no app boot,
  no writes outside the validation directory; the instrumentation lived entirely in an ephemeral `python -c` process)
  and reproduced the reviewer's flush-count to the exact integer.

## 2. P2 — `restore_credential_values` is dead code and the only credential-apply path that silently swallows a partial application
- Verdict: CONFIRMED
- Site now: `Backup_Recovery/credentials.py:393-468` — exact match.
- Proof: `grep -rn restore_credential_values tldw_chatbook/ Tests/ backlog/` → the definition plus exactly 5 hits in
  `Tests/Backup_Recovery/test_credentials.py` (import + 4 call sites) and 0 production callers. Read the body: whole
  function wrapped in `try: ... except Exception: return ("credential_scope_apply_unavailable",)` (`:467-468`, exact
  line match), with a per-record apply loop (`set_recovery_secret_if_absent` then `_write(path, ...)`) that has no
  rollback — a failure at record *k* leaves 1..k-1 already mutated. Live path confirmed distinct:
  `replacement.py:1669 _apply_replacement_credentials` (exact line match) journals
  `credential_intended → credential_applied → credentials_completed`.

## 3. P2 — `raw_participants.py` publication is atomic but has zero fsync/flush_directory calls, asymmetric with the rest of the package
- Verdict: CONFIRMED
- Site now: `raw_participants.py:827` `_file`, `:894` `_replace` (exact matches); `os.replace(...)` at `:919` and
  `:926` (exact matches to the review's citation). `mcp_source_participants.py:294` `write_json`, `:325`
  `backup_corrupt` (both within 4 lines of the review's citation).
- Proof: `grep -n "fsync\|flush_file\|flush_directory" tldw_chatbook/Backup_Recovery/raw_participants.py` → **0
  matches**, in a file that writes the user's live `config.toml`. Contrast confirmed:
  `config_binding.py:115,124,125` all call `flush_directory` (exact match to the review's citation). ADR text
  confirmed: `backlog/decisions/126-complete-local-backup-and-recovery.md:558` contains "receipt remain distinct
  from directory durability" (review cited 557-558) and rules only on the TTS outer-backup case.

## 4. P2 — `require_capacity` re-walks every path component per 64 KiB chunk (re-measured directly)
- Verdict: CONFIRMED (re-measured)
- Site now: `space.py:14` `_volume`, `:24` `require_capacity` (exact matches); called from `staging.py:53` (review
  said :52, one line off) and the payload-extraction loop; `archive_reader.py:50` `_space` (review said :47-49, off
  by ~1-3 lines).
- Proof — ran a direct micro-benchmark (realpath'd temp dir, 7 components deep, 300 iterations each):
  ```
  require_capacity: 0.0892 ms/call
  disk_usage:       0.0010 ms/call
  ratio: 89.4x
  projected per GiB (64KiB chunks): 1.46 s
  ```
  Review measured 0.081 ms / 51x / 1.33 s-per-GiB — same order of magnitude on every axis (the ratio varies more
  because bare `shutil.disk_usage` is fast enough to be dominated by timer noise at this N).
- Also confirmed the "dead work in a read-only hashing loop" sub-claim precisely: `archive_reader.py:462`
  `_space(path.parent, total)` sits inside a `for chunk in _member_chunks(...): ... digest.update(chunk)` loop that
  contains **no** `os.write` call anywhere in its body — capacity-checking a loop that writes nothing to disk. (Line
  shifted from the review's :487 to :462, consistent with the 25-commit drift — the write-bearing copy loop is a
  separate call site at `:514`, not this one.)

## 5. P2 — `visual_identity_participants.py` / `persona_visual_participants.py` near-clones with diverged locking
- Verdict: CONFIRMED
- Site now: `visual_identity_participants.py` (1,121 lines) + `persona_visual_participants.py` (835 lines) — `wc -l`
  sums to exactly 1,956, matching the review's total precisely.
- Proof: `persona_visual_participants.py:35` `lock: object = field(default_factory=threading.RLock)` and `:410`
  `while not source.lock.acquire(timeout=0.05):` (both exact line matches) — `visual_identity_participants.py` has no
  equivalent `_Source`-level lock field (its `_lock` references, e.g. `:981`, are a different, candidate-level lock).
  Spot-checked 5 of the claimed shared function names (`_validate_source`, `source_for`, `native_open`, `mkdir`,
  `unlink`) — each defined exactly once in both files.

## 6. P3 — `participants._core_*` is the cross-package admission API for 20 modules, exposed only under underscore names
- Verdict: CONFIRMED
- Site now: `Backup_Recovery/participants.py:392` `_core_access` — exact line match.
- Proof: `grep -rln "_core_access" tldw_chatbook/` → **22** files total, **20** outside `Backup_Recovery/` — both
  numbers exact matches to the finding. The 20 span `DB/*_DB.py` (all ten stores), `Notes/`, `Notifications/`,
  `Kanban_Interop/`, `Scheduling/db/`, `Sync_Interop/`, `Writing_Interop/`, `Research_Interop/`, matching the review's
  named list.

## 7. P3 — `credentials._read_scope` reaches `store._keyring.get_password(...)` past the declared Protocol
- Verdict: CONFIRMED
- Site now: `credentials.py:69` `return store._keyring.get_password(record["service"], record["username"])` — exact
  line match (`_read_scope` def at `:62`).
- Proof: `runtime_policy/server_credentials.py`'s `ServerCredentialStore(Protocol)` declares `set_scoped_secret` /
  `get_scoped_secret` — **no `get_password`**. `InMemoryServerCredentialStore` (`:244`) and
  `UnavailableServerCredentialStore` (`:307`, exact match) have no `_keyring` attribute; only
  `KeyringServerCredentialStore.__init__` (`:412`, exact match) sets `self._keyring`. Re-ran the review's own grep:
  `grep -rn "\._keyring" tldw_chatbook/` surfaces many `self._keyring` uses, but all of them are a class accessing
  its *own* attribute from inside its own defining module (`server_credentials.py`, `Personal_Context/key_protector.py`,
  `Personal_Context/link_key_custody.py`) — `credentials.py:69`'s `store._keyring...` is the only site that reaches
  *another* module's object's private attribute, confirming "the only cross-module reach" as stated.

## 8. P3 — `RecoveryService` never prunes `_futures`/`_states`/`_archives`/`_workspaces`
- Verdict: CONFIRMED
- Site now: `recovery_service.py:256-260` (dict init), `:271` `_start`, guard `if any(not future.done() for future in self._futures.values())` at `:277` (review cited :257-262/271-307, all within range).
- Proof: `grep -n "_futures.pop\|_states.pop\|_archives.pop\|_workspaces.pop\|del self._futures\|del self._states\|del self._archives\|del self._workspaces" recovery_service.py` → **0 hits**. Confirmed unbounded for the process lifetime, and confirmed the review's own "not a defect a user hits" framing (bounded by user actions per session).

## 9. P3 — `native_platform.rename_noreplace` rebuilds `ctypes.CDLL(None)` and re-sets `argtypes`/`restype` every call
- Verdict: CONFIRMED
- Site now: `native_platform.py:36` `def rename_noreplace`, `:45` `libc = ctypes.CDLL(None, use_errno=True)`,
  `fn.argtypes =` / `fn.restype =` set unconditionally on every invocation (no module-level caching).
- Proof: read the function body directly — no lazy singleton, no `@lru_cache`. Call site confirmed once per
  publication unit: `native_files.py:71` `_rename_new` → `rename_noreplace(...)` at `:75`, called from `publish_new`
  (`:101`) via `_rename_new(...)` at `:160` (review cited `:139`; shifted ~21 lines, same call shape, one call per
  `publish_new`).

## Explicit re-check requested by the task: is `Utils/atomic_file_ops.py` really weaker than `Backup_Recovery/native_platform.py`?
- Verdict: CONFIRMED — the review's (and phase4-verification.md's) claim holds under direct inspection.
- Proof: `grep -n "fsync\|os.replace\|dir_fd\|O_DIRECTORY\|chmod" tldw_chatbook/Utils/atomic_file_ops.py` → `os.fsync(f.fileno())` at `:103`/`:178` (the **file**, never a parent-directory fd), `os.replace(...)` at `:110`/`:184`/`:284` (plain path-based, no `dir_fd`), default `mode: int = 0o644`. Zero occurrences of `dir_fd`/`O_DIRECTORY` anywhere in the file — confirms **no parent-directory fsync**. `Backup_Recovery/native_platform.py:12-22 flush_file` does `os.fsync(fd)` **plus** `fcntl.fcntl(fd, fcntl.F_FULLFSYNC)` on Darwin, and `native_platform.py:26 flush_directory` is a separate fd-based directory-fsync primitive that `Backup_Recovery` pairs with every rename (`config_binding.py:115,124,125`) — `atomic_file_ops` has no equivalent call anywhere. `grep -rl atomic_file_ops tldw_chatbook/ --include=*.py | wc -l` → **17** importers, matching the review's count exactly. Also confirmed the secondary claim that adopting `atomic_file_ops` into `credentials.py` would violate its no-exception-interpolation invariant: `atomic_file_ops.py:135` logs `f"Failed to atomically write to {file_path}: {e}"`, directly interpolating the exception.
- Conclusion: **do not consolidate `Backup_Recovery`'s writers down onto `atomic_file_ops`** — they are two durability tiers, and `Backup_Recovery`'s is strictly stronger (parent-directory fsync + Darwin `F_FULLFSYNC`, which `atomic_file_ops` has neither of). The repo-wide question, correctly, runs the other direction: should `atomic_file_ops` gain both barriers for its 17 importers.

TOTALS: confirmed=9 fixed=0 wrong=0 demoted=0 promoted=0

(The "explicit re-check" item above is scored separately from the 9 numbered `###` findings — it validates a
candidate-triage claim the task specifically asked to verify, not a new `###` finding of its own; its result is
CONFIRMED and folds into finding 3's severity reasoning.)
