# Throwaway native SQLite close-policy spike

Date: 2026-09-07 (America/Los_Angeles).
Baseline: `4209b06c2`, `codex/canvas-v2-mermaid-design`, worktree
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/canvas-v1`.
Authorization: user approved the proposed throwaway qualification probe after the
recommendation to retain in-process SQLite and test a supported close policy.
This is feasibility evidence, not an approved replacement design or production fix.

## Archive provenance

This historical probe record and the three exact tested sources are archived
beside this file as text, outside test collection:

- [Repository exit probe](2026-09-07-sqlite-native-close-policy-repository.py.txt)
- [Pristine helper-first probe](2026-09-07-sqlite-native-close-policy-helper-first.py.txt)
- [Healthy repository probe](2026-09-07-sqlite-native-close-policy-healthy.py.txt)

Restore these to their original filenames in a fresh owned temporary directory
to reproduce; the healthy probe loads the repository probe by its original name.
The paths below describe the original execution. The later Python 3.12 baseline
approval is recorded in the design amendment, not retroactively in probe results.

## Result

The native `SQLITE_DBCONFIG_NO_CKPT_ON_CLOSE` policy is a viable candidate on this
host. All **25 native-policy cases passed**. Of 25 default-policy controls,
17 passed and eight reproduced the original ordinary-exit foreign-sidecar deletion.
The final combined command therefore exits 1: **42 passed, 8 failed**, one existing
requests dependency warning, 53.09 seconds. The eight failures are real preservation
assertions in default-policy arms, not xfails or assertions requiring the bug.

This supports retaining live SQLite in Chatbook rather than introducing a database
service. It does not finish TASK-31942, Task5 integration, or Canvas qualification.

## Coverage

| Probe | Cases | Native result | Default result |
|---|---:|---|---|
| Preserved actual-repository exit gate, baseline/helper-shaped ownership, ordinary/abrupt exit | 8 | 4 pass | 2 pass; 2 ordinary-exit failures |
| Pristine helper-first ownership; idle/read/write transaction; one/two connections; ordinary/abrupt exit | 24 | 12 pass | 6 pass; 6 ordinary-exit failures |
| Healthy two-connection close; idle/read/write; with/without explicit checkpoint; committed-data recovery and integrity check | 12 | 6 pass | 6 pass |
| Close one helper-backed owner while sibling writer remains active | 2 | 1 pass | 1 pass |
| Actual repository create, close, two reopens; with/without explicit checkpoint | 4 | 2 pass | 2 pass |

Foreign-file observations happen in the external test parent before fixture cleanup.
Both substituted files retain their names, inode identities, link count 1, and exact
sentinel bytes in native-policy arms. Default ordinary exits instead lose both names
and reduce both observer link counts to zero; their bytes remain readable through
the external retained FDs. The cooperative SHARED store lease excludes EXCLUSIVE
acquisition before exit; post-exit EXCLUSIVE acquisition succeeds in both exit modes.

The pristine probe seeds an actual TTS store in a separate, fully exited process.
The live child starts the actual fixed TTS proof helper and acquires its original
pins there. Live SQLite uses the real private factory in the parent; that parent
never opens/closes raw main/WAL/SHM descriptors or an immutable evidence connection.
Both live handles receive and verify the policy before caller SQL. Helpers are
killed/reaped only after external substitution, using captured owned child handles.
Their real retained admission permits remain charged. No SQL or explicit SQLite
close occurs afterward: module-level references survive to interpreter finalization.
These are ownership probes, not the not-yet-implemented Task5 repository wrapper.

Before substitution, an independent SQLite contender returns exactly SQLITE_BUSY
for active write transactions, and succeeds for idle/read controls. Closing one
healthy helper-backed owner also preserves the other connection's write exclusion;
rollback and final cleanup subsequently admit the contender.

Healthy tests commit speed 2.0 and optionally stage uncommitted speed 3.0. Explicit
rollback/close recovers 2.0 and passes integrity_check. Native close leaves legitimate
WAL files; an explicit healthy TRUNCATE checkpoint produces a zero-length WAL.
The tests do not delete that WAL to make recovery pass. Actual repository reopening
also recovers a newly created profile on two successive opens.

## Important integration constraint discovered

The initial coarse child-local instrumentation set the flag for every
`profile_schema.connect_private_sqlite("tts.profile_store", ...)` call, including
schema initialization. All four native arms failed before ready; no foreign files
had yet been substituted. Default ordinary controls still reproduced deletion.

Narrowing the instrumentation to calls made inside `open_exact_current_profile_store`
allowed initialization and all four native exit arms to pass. The schema initializer
uses WAL and the later immutable proof reads the main database, so reliance on
close-time checkpointing is a supported explanation, but the coarse startup failure
was not traced to its precise internal refusal. Do not claim it was fully diagnosed.
This experiment is sufficient to reject a blind global factory toggle as a tested
drop-in fix. Production design must explicitly cover healthy checkpointing,
initialization/migration, and all live owners of the affected store.

One later helper-module import in the throwaway healthy-repository test failed during
collection under pytest importlib mode; explicit loading of the known adjacent
probe module fixed the harness. The final run includes all six originally affected
healthy/sibling cases. Neither development failure is hidden in the final count.

## Scope and limits

- Python 3.12.11, SQLite 3.49.1, macOS 26.5.2 arm64 only. No Linux, Windows,
  alternative SQLite build, or other Python version qualification.
- `setconfig` and constant 1006 are available on this host. The project's declared
  Python >=3.11 requirement is unchanged; a baseline/API compatibility decision is
  still required before adopting the standard-library API.
- No full Textual shutdown, new restart_required mapping, admission latch,
  cancellation/partial publication, restore/tombstone integration, or installed-wheel
  qualification. The old helper-shaped gate keeps its previously documented limits.
- Active read/write transactions were small. Spilled large transactions, live BLOB
  handles, outstanding cursor statements, mixed configured/unconfigured final owners,
  and destructive external truncation of original mapped inodes were not tested.
- Healthy recovery explicitly rolls back active transactions; it does not independently
  establish post-crash recovery of uncommitted work through native finalizers. Foreign
  preservation with active transactions is a separate, passing observation.
- No native syscall trace, exhaustive metadata-mutation audit, or universal guarantee
  against external pathname races. Ownership checks remain necessary.
- A passing probe does not authorize arbitrary close after proof loss or replacement
  proof. Keep production admission/refusal policy until a revised design is approved.
- No product, dependency, Python requirement, ADR, backlog status, or release gate was
  changed. No full suite, host cleanup, user database access, or PR/Git mutation.
  The worktree remains clean. All probe sessions and captured owned child lifetimes
  completed; failed-run containment affects only probe-owned children.

## Reproduction

Run from the baseline worktree above, using the existing shared venv:

```sh
../../.venv/bin/python -m pytest -p Tests.conftest -c pyproject.toml -q \
  /private/tmp/chatbook-native-close-spike.Jnp98p/test_repository_close_policy.py \
  /private/tmp/chatbook-native-close-spike.Jnp98p/test_helper_first_policy.py \
  /private/tmp/chatbook-native-close-spike.Jnp98p/test_repository_healthy_policy.py \
  --basetemp=/private/tmp/chatbook-native-close-spike.Jnp98p/reproduction-NEW \
  --junitxml=/private/tmp/chatbook-native-close-spike.Jnp98p/reproduction-NEW.xml \
  --tb=short
```

Use a fresh owned basetemp name to preserve earlier observations. The repository
test bootstrap isolates configuration and denies network access before product
imports. This source stays throwaway and outside normal test collection.
Final machine-readable evidence: `final-results.xml`.
Ruff check and format check pass for all three source files; git diff check passes.

SHA-256 of final tested source:

- `test_repository_close_policy.py`: `2e8ee6e410bec77bcc45a58ac9d340d453f3992f774ca3b07071612cf5ba7434`
- `test_helper_first_policy.py`: `f9722b7b71a813c58c9f8c77e2268a091c807cdfda220cae884dac53d9f34bfc`
- `test_repository_healthy_policy.py`: `53b8d0bc0f923a18c6f5715d7868916ea2036ba2df0d383e92730227c9a9894f`

## Official references and next decision

[SQLite's documented connection option](https://www.sqlite.org/c3ref/c_dbconfig_defensive.html)
overrides the normal close-time checkpoint behavior. In the tested release's
[pager source](https://raw.githubusercontent.com/sqlite/sqlite/version-3.49.1/src/pager.c),
`SQLITE_NoCkptOnClose` prevents passing the checkpoint buffer to
[sqlite3WalClose](https://raw.githubusercontent.com/sqlite/sqlite/version-3.49.1/src/wal.c),
whose buffer-guarded branch sets the WAL/SHM deletion flag. This source path supports
the hypothesis; it is not a captured native execution trace.
[Python's public setconfig API](https://docs.python.org/3.12/library/sqlite3.html#sqlite3.Connection.setconfig)
was added in 3.12; individual options also depend on the compiled SQLite library.

Recommendation: retain embedded SQLite and the reviewed raw-validation helper
boundary; design narrowly scoped native close policy plus authority-checked healthy
checkpoint behavior. Prefer an explicitly approved Python >=3.12 baseline and a
runtime capability check over private ABI shims. Do not introduce a live SQLite
service on the strength of the earlier retention failure. Written integration/ADR
revision and production verification remain separate approved work.
