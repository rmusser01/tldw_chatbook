# S25 — `Backup_Recovery/` (added by this run; the scope table omitted it entirely)

**Coverage:** files read in full: 10 | sampled: 28 | mechanical only: 42 (of 80; 42,955 lines).
Full: `archive_reader.py`, `crypto.py`, `native_files.py`, `native_platform.py`, `space.py`, `limits.py`,
`credentials.py`, `models.py`, `__main__.py`, `__init__.py`. Sampled: 28 (state machines traced through the regions
that matter, structure mapped by `grep -n "^def|^class"`). Mechanical only: 42, swept with targeted greps for
`os.replace`/`fsync`, `except…pass`, `rmtree`, `subprocess`, `os.environ`, `re.compile`, `fetchall`, loguru
interpolation, non-atomic `write_text`/`write_bytes`/`replace`.

> **General note, recorded because it is unusual:** this is the most defensively-written code in the repo.
> `archive_reader.py` independently re-parses the ZIP central directory before handing the stream to `zipfile`,
> bounds decompression per-chunk against the declared size, verifies CRC, rejects symlink/dir/non-deflate members,
> and gates high-compression archives behind an explicit `CompressionReviewRequired`. **Path traversal, symlink
> escape and decompression bombs are all genuinely closed.** The findings below are not in that layer.

## Findings

### P1 [D2] — The restore publication loop re-`fsync`s the entire journal history once per move record, making publication O(N²) in `F_FULLFSYNC` calls
- Where: `Backup_Recovery/journal.py:1626-1633` (`Journal._flush_records`), driven from `publication.py:2001-2050`
  (the `for item in prepared.artifacts:` loop) via `publication.py:2727-2749` (`_begin_move`) and `:2706-2724`
  (`_complete_move`); same shape in `replacement.py:2425-2432` and `rollback_credentials.py:457-459`.
- Evidence: `_flush_records` = `self._records(parent)` (re-reads and re-validates **every** record) + `_flush_record`
  per row, and `_flush_record` ends in `flush_file(fd)` (`journal.py:1620`), which is `os.fsync` **plus
  `fcntl(F_FULLFSYNC)` on Darwin** (`native_platform.py:12-22`). Per artifact the publish loop calls
  `_begin_move`+`_complete_move` twice; the journal grows ~6 records/artifact ⇒ `flush_file` calls ≈ `Σ 4·6i ≈ 12N²`.
  **Measured**, instrumented run (a pytest plugin wrapping `native_platform.flush_file`, `journal.flush_file`,
  `native_files.flush_file`):
  `pytest Tests/Backup_Recovery/test_publication_finalization.py -q -p countflush -s` →
  `FLUSHCOUNT {'flush_file': 3316, 't': 8.278s, 'records': 6262, 'maxjournal': 12}` — **8.28 s of pure fsync with a
  journal only 12 records long**, and 6,262 record re-reads.
  Isolated micro-benchmark on an idle inode: `flush_file` = **0.163 ms**, plain `os.fsync` = 0.0006 ms (268×
  cheaper). Projection at the two measured constants (0.163 ms idle / 2.50 ms under load, `12N²`):
  N=100 → 20 s / 5 min; N=500 → 8 min / 2.1 h; N=1000 → 33 min / 8.4 h. N is bounded only by
  `publication.py:1752` (`required_events = … + 12*len(artifacts) < MAX_EVENTS=100_000`) ⇒ up to ~8,300 units;
  `Tests/Backup_Recovery/test_large_recovery_records.py:120` exercises a single `prepared` record holding
  **1,801 artifacts**.
  Secondary: `Journal._records` is itself O(k²) — it calls `_validate(record.event, record.evidence, records)` per
  record, and `_validate` opens with two O(k) list comprehensions (`journal.py:1533-1559`, `638-652`). `_records` is
  called ~10× per artifact.
- Why it matters: the sweep runs inside `journal._locked(exclusive=True)` with local storage admission paused
  (`publication_started` → `committed`). A restore of a few hundred publication units stops responding for minutes
  to hours with no progress signal, and any interruption lands in `recovery_required`.
- Recommended correction: `Admission._write_new_record` (`admission.py:291-304`) **already calls `flush_file(fd)` on
  every record**, and `_append` follows with `flush_directory(parent)` (`journal.py:1652`) — so every record is
  durable the moment it is written and the full-history re-flush is redundant on the normal path. Track a
  high-water mark of records this `Journal` instance has flushed and re-flush only above it (or only the tail);
  keep the full `_records()` re-validation if the tamper check is wanted, and hoist `_validate`'s
  `events`/`lifecycle` prefix out of the per-record loop.
- Size: M · ADR: no · Confidence: **verified (measured + traced)**
- Pinning test: none. `Tests/Backup_Recovery/test_replacement.py:363-370` and `test_held_sqlite_rollback.py:517-524`
  wrap `_flush_records` to observe *which events are visible* after it — neither asserts that prior records are
  re-flushed, so the behaviour is not pinned as a requirement.
- Already covered: none. TASK-32804's efficiency sub-tasks name Library/RAG/Console/MCP/splash surfaces only.

### P2 [D3] — `restore_credential_values` has zero production callers, and it is the only credential-apply implementation that silently swallows a partial application
- Where: `Backup_Recovery/credentials.py:393-468`.
- Evidence: `grep -rn "restore_credential_values"` → definition + 5 hits in `Tests/Backup_Recovery/test_credentials.py`
  + 2 in a plan doc. No production caller, and no dynamic dispatch (`grep -rn "getattr(.*credentials"` → none). Every
  real importer of the module (17 sites) imports `process_credentials`, `_material`, `plan_credential_scopes`,
  `replacement_credential_records`, `_read`, `_write` — never this. The live path is
  `replacement.py:1669-1751 _apply_replacement_credentials`, which journals
  `credential_intended` → `credential_applied` → `credentials_completed` and is resumable; its helpers
  `raise … from None` rather than returning. The dead function wraps its whole body in
  `except Exception: return ("credential_scope_apply_unavailable",)` (`:467-468`); its apply loop (`:448-464`) calls
  `store.set_recovery_secret_if_absent(...)` then `_write(path, …)` per record, so a failure at record *k* leaves
  records 1..k−1 with fresh keyring secrets created and their staged `targets` files rewritten, and returns a tuple
  indistinguishable from "nothing was applied". The plan document explains the orphan: Step 7 of the 2026-09-07 plan
  specified this as the executor's entry point; the implementation went to the journal-backed `replacement.py` route
  instead and this was left behind, still pinned by four tests.
- Why it matters: a dead, non-resumable, partial-apply-swallowing implementation of the most security-sensitive
  operation in the package, kept green by its own tests — a ready template for anyone wiring "credential restore"
  and looking for the obvious name.
- Size: S · Confidence: verified
- Already covered: **none of TASK-32807's six sub-tasks names `Backup_Recovery`.**

### P2 [D1] — `raw_participants` publication is atomic but never durable: the module contains zero `fsync` calls, asymmetric with every other publication path in the package
- Where: `Backup_Recovery/raw_participants.py:827-891` (`_file`), `:894-937` (`_replace`, the `os.replace` at
  919/926); `mcp_source_participants.py:298-316` (`write_json`, unmanaged branch) and `:325-331` (`backup_corrupt`).
- Evidence: `grep -n "fsync\|flush_file\|flush_directory" raw_participants.py` → **no matches**.
  `_file`'s exit path is `TextIOWrapper.close(text)` then `os.close` — flush to the page cache, no `fsync`;
  `_replace` then calls `os.replace(...)` with no `flush_directory`. Contrast within the same package:
  `admission.py:302, 333, 337, 343`; `config_binding.py:115, 124, 125` (`flush_directory` **before and after** the
  replace); `control_records.py:347, 1075-1076`; `storage_admission.py:2190, 2218`; `native_files.py:64-65, 95-141`.
  `grep -n -i "durab\|crash\|power"` on both files → no comment records the omission as deliberate.
  ADR-126 is aware of the distinction ("A completed rename and its receipt remain distinct from directory
  durability", `126-complete-local-backup-and-recovery.md:557-558`) but rules on it only for the TTS outer-backup
  case, not for raw participants.
- Why it matters: these are the writes to the user's **live `config.toml`**, settings TOML, MCP `targets.json` and
  chat-dictionary files (`raw_participants.py:20-25`). A crash or power loss immediately after the rename can
  publish a name pointing at an uncommitted inode — an empty or truncated live config — and the operation's journal
  holds no evidence of the loss. Filesystem-dependent (ext4 `data=ordered` masks it; APFS and ext4
  `data=writeback`/network mounts do not), which is exactly why the barrier belongs in code.
- Recommended correction: `os.fsync(fd)` before `_close_descriptor` for `mode in {"w","a"}`, and
  `flush_directory(state.pins[destination.parent])` after the rename (both already imported transitively). Mirror in
  `mcp_source_participants.write_json`'s unmanaged branch. **Do NOT route these through `Utils/atomic_file_ops.py`** —
  see the retired candidate below.
- Size: S · ADR: yes (amend `126-complete-local-backup-and-recovery.md`, or record the waiver) · Confidence:
  verified (absence of fsync); consequence inferred (filesystem-dependent)
- Already covered: **TASK-32808.5 [Done] AC#2 is "Every write that claims durability actually fsyncs"** — but its
  Implementation Notes enumerate the audited sites (`ConfigProfileManager._save_one`, `emergency_stop._write`,
  `MCP/permission_store.py`, `Tools/local_tool_impls.py`, `Utils/tls_trust.py`) and **none is in
  `Backup_Recovery`.** The sweep did not reach this package.

### P2 [D2] — `require_capacity` re-walks every path component from `/` once per 64 KiB chunk during restore staging: 1.33 s of pure overhead per GiB copied
- Where: `Backup_Recovery/space.py:14-21` (`_volume` → `pinned_directory`), called from `staging.py:52` (`_copy`'s
  per-chunk loop) and `staging.py:597` (payload extraction loop). Same shape, cheaper constant, in
  `archive_reader.py:47-49 _space`, called per chunk at `:437` (copy) and `:487` (read-only hashing).
- Evidence: `space.py:24-49` — for each requirement `_volume(path)` enters `pinned_directory(selected)`, which opens
  **every path component** with `O_RDONLY|O_DIRECTORY|O_NOFOLLOW` and `fstat`s each (`native_files.py:18-43`), then
  `shutil.disk_usage`. **Measured** (temp dir 7 components deep): `require_capacity` = **0.081 ms/call**; bare
  `shutil.disk_usage` = **0.0016 ms** → 51× overhead. At one call per 64 KiB chunk: **1.33 s per GiB copied**, and
  both `staging._copy` and the payload loop do it. The budget is constant across the loop — `_copy` passes
  `{destination.parent: len(chunk)}` every iteration. And `archive_reader._inspect:487` calls `_space` inside the
  payload **hashing** loop, which writes nothing at all — unconditionally dead work.
- Why it matters: a 20 GiB restore pays ~27 s of pure capacity re-checking, plus ~0.5 s/GiB during read-only archive
  verification. Both on the wall-clock path the user watches.
- Size: S · Confidence: **verified (measured)**

### P2 [D4] — `visual_identity_participants.py` and `persona_visual_participants.py` are near-clones (1,956 lines) whose concurrency discipline has diverged
- Where: `Backup_Recovery/visual_identity_participants.py` (1,121) and `persona_visual_participants.py` (835).
- Evidence: 12 byte-identical functions and ~16 shape-identical ones (`_validate_source`, `source_for`, `request`,
  `_identity`, `current`, `_path`, `_check_path`, `native_open`, `_close`, `native_close`, `mkdir`, `unlink`,
  `rmdir`, `published`, `private_directory`, `directory_created`). Identical module-level state blocks
  (`threading.local()`, two `WeakKeyDictionary`, `_issued`, `_states`, `_publications`). `diff` after normalising the
  two module prefixes: **936 differing lines out of 1,956** — roughly half the combined file is shared.
  **The drift is behavioural:** `persona_visual_participants._Source` carries
  `lock: object = field(default_factory=threading.RLock)` (`:35`) and serialises native operations on it
  (`while not source.lock.acquire(timeout=0.05)` at `:410`, release at `:508`).
  `visual_identity_participants._Source` has **no such field**. The two modules mutate live image files under
  different serialisation rules.
- Recommended correction: extract the shared source/native lifetime into one in-package module parameterised by
  owner id, route table and per-owner hooks. **Settle the lock question explicitly in the same change** — either
  both sources take the RLock or neither does, with the reason recorded.
- Size: L · ADR: no (in-package; use `backlog/docs/library-decomposition-recipe.md`'s shape) · Confidence: verified

### P3 [D3] — `participants._core_*` is the cross-package admission API for 20 modules but is exposed only under underscore names
- Where: `Backup_Recovery/participants.py:392-439` (`_core_access`, `_core_operation`, `_core_transaction`) plus
  `_core_getter`, `_core_cached_connection`, `_register_core_connection`, `_close_settled_core_cache`.
- Evidence: `grep -rln "_core_access" tldw_chatbook/` → 22 files, **20 outside the package**: all ten `DB/*_DB.py`
  stores, `Notes/{file_notes_replica,notes_device_state_store}.py`,
  `Notifications/{event_state_repository,client_notifications_db}.py`,
  `Kanban_Interop/{local_kanban_db,local_kanban_service}.py`, `Scheduling/db/scheduled_tasks_db.py`,
  `Sync_Interop/sync_state_repository.py`, `Writing_Interop/local_writing_service.py`,
  `Research_Interop/local_research_service.py`. Broader census of underscore names crossing this package boundary:
  `_RawDeclaration`(9), `_witnesses`(6), `_core_access`(20 modules), `_tree_member_id`(5), `_read_recovery_file`(4),
  `_repository_participant`(3), `_core_operation`(2), `_Prepared`(2), … — **~30 distinct underscore names.**
  What it guards (read, `:392-417`): under `storage._lock`, it refuses a new cached-handle borrower while the
  participant is retiring, while an operation's provenance does not match, or while local storage is paused —
  raising `RecoveryRequired`. It is the choke point that stops a DB handle being opened mid-backup.
- Why it matters: the underscore says "internal, may change freely"; 20 modules in 8 packages say otherwise. Any
  refactor of `participants.py` that trusts the underscore breaks the admission gate for the entire persistence
  layer.
- Recommended correction: public aliases + `__all__`, then mechanically repoint the 20 importers; keep the
  underscore names as deprecated aliases for one cycle.
- Size: S (mechanical) · Confidence: verified
- Already covered: none. **The brief named one importer (`DB/Library_Collections_DB.py`); the real cluster is 20.**

### P3 [D3] — `credentials._read_scope` reaches `store._keyring.get_password(...)` past the `ServerCredentialStore` Protocol, and the failure is swallowed as "unreadable"
- Where: `Backup_Recovery/credentials.py:62-69`.
- Evidence: `_keyring` is declared only on `KeyringServerCredentialStore.__init__`
  (`runtime_policy/server_credentials.py:412`). The declared Protocol (`:35`) exposes
  `get_scoped_secret`/`set_scoped_secret`/… and **no `get_password`**; `InMemoryServerCredentialStore` (`:244`) and
  `UnavailableServerCredentialStore` (`:307`) have no `_keyring`. `build_default_server_credential_store` always
  returns the keyring store or raises, so production never hits the `AttributeError` — but the only non-server-kind
  callers are in `_capture_record`, whose `except Exception` (`credentials.py:88`) would record
  `status="unreadable"` and emit `credential_unreadable:<id>` instead of failing.
  `grep -rn "\._keyring" tldw_chatbook/` → this is the only cross-module reach.
- Why it matters: generation and citation credentials are captured through a private attribute of one concrete
  store. Substituting any Protocol-conformant store silently degrades those credentials to "unreadable".
- Size: S · Confidence: verified

### P3 [D3] — `RecoveryService` never prunes `_futures`/`_states`/`_archives`/`_workspaces`
- Where: `Backup_Recovery/recovery_service.py:257-262`, `_start` at `:271-307`. The guard is
  `if any(not future.done() for future in self._futures.values())` — an O(N) scan over every operation ever started;
  nothing removes entries, and `_archives[operation]` holds a `SealedArchive` for the process lifetime.
  Bounded by user actions per session (tens), so kilobytes, not a defect a user hits. Listed for completeness.
- Size: S · Confidence: verified

### P3 [D2] — `native_platform.rename_noreplace` builds a fresh `ctypes.CDLL(None)` and re-sets `argtypes`/`restype` on every call
- Where: `Backup_Recovery/native_platform.py:37-68` (also `file_inventory.py:38-57`, `linux_identity` at `:96-106`).
  Called once per `publish_new` (`native_files.py:139`), i.e. once per publication unit — so small, but it is in the
  same loop as the P1 and is a one-line hoist to a module-level lazy singleton.
  *(Lead's note: S01 found the identical shape in `Notes/sync_paths.py`, on a hotter loop — see its P3.)*
- Size: S · Confidence: verified

## Candidate triage
**RETIRED — and this one inverts the obvious recommendation:** `os.replace` in 7 files without
`Utils/atomic_file_ops.py` (TASK-32808.5 Done). **Non-adoption is correct here.** `atomic_file_ops.atomic_write_text`
fsyncs the file but **never the parent directory** (`Utils/atomic_file_ops.py:99-118`), uses path-based `os.replace`
(symlink-racy, no `dir_fd`), defaults to `0o644`, and logs `f"Failed to atomically write to {file_path}: {e}"` at
`:135` — which would break `credentials.py`'s no-exception-interpolation invariant outright. The package's own
`native_platform.flush_file` (fsync + `F_FULLFSYNC` on Darwin, native flush on Windows) + `flush_directory` +
`rename_noreplace` (`renameat2`/`renameatx_np`) + `pinned_directory` fd-based no-follow traversal is **strictly
stronger**. Six of the eight `os.replace` sites already pair with a directory barrier. *(Lead re-verified the
`atomic_file_ops` gap — see `phase4-verification.md`; it is now a repo-wide finding in its own right.)* The two
`raw_participants`/`mcp_source_participants` sites have no barrier at all and are filed as P2.
**RETIRED:** the "backend errors can contain secret values" containment — **it holds.** Every `except` around a store
call in `credentials.py` (88, 350, 359, 467, 1206, 1239, 1256) raises `from None` with a fixed opaque code or sets an
opaque status; no `e`/`error` is interpolated anywhere. The whole 43k-line package has **9 logging call sites**, all
with constant format strings; the only dynamic value is `type(error).__name__` at `local_content_lifetime.py:199`.
`Utils/log_sanitizer.py` is not imported — and does not need to be.
Untrusted-archive parsing (traversal/symlink/bomb/unbounded read) — **all closed**, details in the coverage note.
`age_worker.py` integrity pin — **current**: `shasum -a 256` matches `crypto.py:29` exactly, pinned by
`Tests/Backup_Recovery/test_crypto.py:254`.
`DUP_VERBATIM relocate ×9` / `DUP_SHAPE capture ×11` — **Protocol conformance, not duplication.**
`models.py:131-137` declares `OwnerAdapter(Protocol)` with no base class, so every owner must supply its own; the
`relocate` bodies are 3 lines each with a distinct per-owner semantic comment. Same ruling shape as the lead's
`_perform_safe_cancel` / `_initialize_schema` resolutions.
`native_closed@storage_admission.py:1049` vs `_clear_character_tts_profile_suggestion@personas_screen.py:1524` —
**shape-hash collision between unrelated domains.**
**None in this slice:** god modules >5k lines (largest is `publication.py` at 2,845 — though 4 modules exceed 2,000
with no ratchet row, a gap in 32809.2's hand-picked scope); loguru + stdlib `logging` in one file (4 loguru, 0
stdlib); `get_cli_setting` anywhere; `run_worker(exclusive=True)` without `group=` (the package uses its own
single-worker `ThreadPoolExecutor`); any `_maybe_await` definition or call site.
**CONFIRMED:** `_discard_workspace@recovery_service.py:1308` vs `_discard@later_rollback.py:866` — both
`pinned_directory(work)` → compare `(st_dev, st_ino)` → `shutil.rmtree`. One helper in `native_files.py`.
`unique`/`_unique`/`_unique_object` ×6 — trivial 2-3 line duplicate-detection helpers.
**UNVERIFIED:** `__exit__`/`_executor_finished` (`async_file_participants.py:57,79` vs
`dictionary_source_job.py:104,126`); `_maintenance_resume` (`rag_projection_lifetime.py:249` vs
`RAG_Search/model_recovery.py:441`); `execution_identity@admission_runtime.py:22` vs
`Skills_Interop/skill_trust_service.py:73`.

## D4 observations for repo-wide Phase 3
1. **`OwnerAdapter` is a `Protocol` with no base class** (`Backup_Recovery/models.py:131-137`) with ~12
   implementations across `Notes/`, `Evals/`, `Writing_Interop/`, `Study_Interop/`, `Research_Interop/`,
   `Scheduling/`, `Sync_Interop/`, `Kanban_Interop/`, `TTS/`, `Notifications/`, `DB/recovery_*`, and four in this
   package. This generates the `capture`(11), `relocate`(9), `validate`(2) DUP rows. **Rule it as protocol
   conformance, not duplication** — but note a shared `OwnerAdapterBase` supplying the two default bodies would
   delete ~40 lines across 12 packages at zero behavioural risk.
2. **Two durability tiers, not one cluster.** `Backup_Recovery` writes through a package-local layer strictly
   stronger than `Utils/atomic_file_ops.py`. Do not consolidate downward. The repo-wide question is the reverse:
   should `atomic_file_ops` gain a parent-directory fsync (and the Darwin `F_FULLFSYNC` barrier) for its 17
   importers? **The lead verified the gap; it is filed as a repo-wide finding.**
3. **~30 underscore-prefixed names cross this package's boundary** — the largest private-API-across-packages cluster
   found in the review. One mechanical rename + alias change; worth a single repo-wide task.
4. **`_discard`/`_discard_workspace`** — identity-checked `rmtree`, two verbatim copies in-slice. Check whether
   `Evals/`, `Chunking/` or `Model_Artifacts/` recovery modules carry a third. Home: `Backup_Recovery/native_files.py`.
5. **Per-chunk capacity checking has two implementations** — `space.require_capacity` (component-walking, 0.081 ms)
   and `archive_reader._space` (bare `disk_usage`, 0.0016 ms) — used interchangeably at four per-chunk loop sites.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| Actual `len(prepared.artifacts)` for a real **replace-mode** restore over an existing profile — the N that drives the P1 quadratic. `publication.py:1748` comments "a new tree containing many files is still published with one directory move", and `staging.py:1102-1104` sets `publication_unit` False under a newly-created directory — so a *fresh* restore collapses to ~1 unit while a *replace over an existing profile* does not. The projections assume the latter. | would require running a replace-mode restore against a populated profile; DO-NOT-RUN-THE-APP | `PYTHONPATH=<scratch> pytest Tests/Backup_Recovery/test_f9_replacement_workflow.py -q -p countflush -s` (plugin reports `maxjournal`; N = maxjournal/6). Takes ~3.5 min and failed once under instrumentation — reconcile against an uninstrumented baseline first |
| Whether that `test_f9_replacement_workflow.py` failure is pre-existing or caused by the `Journal._records` wrapper | only ran it instrumented, with `-x`, and the traceback was lost to a `tail` | run it clean, then with `-p countflush`, and diff |
| `later_rollback.execute_rollback`'s `finally: if candidate is None or completed: _discard(work, identity)` (`:2189-2192`) deletes the candidate staging tree *on success*, while its comment says "A prepared candidate may be referenced by a pending new operation" | requires tracing `replacement.replace`'s return contract for the pending/recovery-required case; ran out of budget in `replacement.py` (2,587 lines, sampled) | read `replacement.py:1386-1626` and check whether any returned state names a path under `work`; then `grep -rn "later-rollback-" tldw_chatbook/ Tests/` |
| `credentials.apply_replacement_credential` (`:1228-1246`) catches its own `raise ValueError("credential_scope_changed")` in the enclosing `except Exception` and re-raises it as `credential_scope_apply_unavailable` — a genuine value mismatch reported as a backend outage | did not trace whether the caller distinguishes the two codes in its recovery decision | `grep -rn "credential_scope_apply_unavailable\|credential_scope_changed" Backup_Recovery/replacement.py Backup_Recovery/recovery_service.py` and read `recovery_service.issue_code` (`:166-236`) |
| Whether the `persona_visual` RLock vs `visual_identity` lock-free drift is a known decision | no comment or ADR reference at either site | `grep -rn -i "persona visual\|shared visual identity" backlog/decisions/126-*.md`; both module docstrings cite ruling ranges (70-73 vs 74-78) |
| The 42 mechanical-only files carry no P0/P1 | swept by pattern grep only, not read | read in full, in this order: `config_adapter.py` (1112), `control_records.py` (1087), `inventory.py` (804), `runtime_maintenance.py` (742), `destinations.py` (581), `capture.py` (578) |
