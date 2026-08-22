---
id: TASK-19569
title: >-
  Guards that no longer guard — an inert concurrency test, five stale MCP
  monkeypatches, and a severed cause chain
status: Done
assignee: []
created_date: '2026-08-21 20:19'
labels:
  - testing
  - mcp
  - agents
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 5 (test-suite health & guard efficacy)
— its **B1** cluster. Grouped because all three are guards that are red for
reasons unrelated to what they protect, so the thing they protect is currently
unguarded. All re-measured at this branch base.

Worth stating the lane's headline first, because it calibrates this task:
**no hollow guard was found.** 15 of 16 injected defects were caught, several
with real precision (the blocking-I/O guard walks the call graph and reported a
full three-hop chain; the network guard cannot be absorbed by a broad
`except Exception`). These three are the exceptions, and each has a specific,
small cause.

**A — a genuine product defect behind 6 reds.**
`Tests/DB/test_core_sqlite_owner_privacy.py`: **6 failed / 83 passed**, all
six on `media-*` parametrizations; the identical parametrizations pass for
`base`, `chachanotes`, `prompts`, `evals`. Cause confirmed verbatim at
`DB/Client_Media_DB_v2.py:775`:

```python
except (sqlite3.Error, PrivatePathError) as error:      # line 769
    ...
    raise DatabaseError("Failed to connect to media database.") from None   # 775
```

`from None` **severs the cause chain the privacy contract requires** —
`isinstance(err, PrivatePathError)` fails. The log line one statement earlier
even records `error_type=PrivatePathError`, so the code knows what it caught
and throws the information away. This is a product defect, not a test defect.

**B — an inert concurrency guard, and the bug is 100% in the test.**
`Tests/Agents/test_tool_catalog_concurrency.py:49` fails unconditionally with
`assert 2 == 1`. The test installs its counter at line 44 and then **calls
`registry.list_catalog()` itself at line 46**, so it counts its own call.
Production is correct: `Agents/tool_catalog.py:1117-1118` returns one snapshot,
and `invoke_by_name` (1208) → `_owner_record_for_name` (1196-1197) takes
**exactly one**. Hoisting line 46 above line 44 turns it green. Net effect
today: a real name→id→provider TOCTOU guard can no longer detect its own
regression.

**C — five MCP watchlists tests fail on a stale monkeypatch seam.**
`AttributeError: module 'tldw_chatbook.MCP.local_server_tools' has no attribute
'RuntimeSourceStateStore'` at `Tests/MCP/test_local_server_tools.py:376` and
`Tests/MCP/test_gateway_runtime_tools.py:1067, :1171, :1236`. Cause: production
`MCP/local_server_tools.py:53-54` now imports
`load_default_runtime_source_state` from `runtime_policy.bootstrap` and injects
it as `runtime_source_loader=` (line 165); `RuntimeSourceStateStore` is no
longer a module attribute.

**Correction to the review's wording, from this filing's verification:** these
are **in-body failures at the monkeypatch line**, not errors before the bodies
run. And the fifth case is the more insidious one —
`test_local_server_tools.py:207` and `:261` patch with `raising=False`, so the
stale patch **silently installs a never-read attribute** and the test proceeds
to fail downstream with a scrubbed
`ToolResult(ok=False, error='Watchlists tool execution error')`. The
conclusion holds either way: **the watchlists local-server tool contract is
currently unguarded.**

The five tests: `test_watchlists_registration_is_storage_lazy_and_server_mode_never_resolves_path`,
`test_watchlists_lazy_resolver_blocks_replacement_until_failed_close_succeeds`,
`test_real_watchlists_provider_preserves_structured_domain_outcomes`,
`test_real_watchlists_provider_scrubs_unexpected_failures`,
`test_real_watchlists_database_resolution_runs_off_event_loop`.

## Acceptance Criteria

- [x] `Client_Media_DB_v2.py:775` preserves the cause chain (`from error`), so
      a `PrivatePathError` remains identifiable to the privacy contract; the 6
      media reds go green for the right reason, not by relaxing the assertion
- [x] `test_tool_catalog_concurrency` is repaired in the **test** (its own
      `list_catalog()` call moved before the counter is installed) and left
      able to detect a real regression — verified by mutating the production
      snapshot behaviour and seeing it go red
- [x] The five MCP watchlists tests are re-pointed at the current seam
      (`runtime_source_loader` / `load_default_runtime_source_state`) and pass
- [x] `monkeypatch.setattr(..., raising=False)` is removed wherever it is
      masking a renamed or removed attribute in these files — a patch that
      installs a never-read attribute must fail loudly
- [x] Each repaired guard is mutation-checked: it goes red when the behaviour
      it protects is broken. A green test that cannot fail is the defect being
      fixed here
- [x] The watchlists local-server tool contract is demonstrably guarded again

## Implementation Plan

1. Reproduce all three reds at the merge base with a pinned runner (isolated
   HOME/XDG, `tldw_chatbook.__file__` assert) and record exact counts.
2. **A** — restore the cause chain in `Client_Media_DB_v2._get_thread_connection`
   and audit the other DB owners' connect sites for the same `from None` shape.
3. **B** — hoist the test's own `list_catalog()` above the counter install; add
   an explicit `snapshots == []` precondition so the counter can only ever
   observe `invoke_by_name`.
4. **C** — re-point every stale `RuntimeSourceStateStore` patch at the live
   `load_default_runtime_source_state` seam via one shared helper per file; drop
   every `raising=False` in the two files.
5. Mutation-check each repaired guard against the production behaviour it
   protects; Edit-restore and confirm `git diff` is clean.
6. Run `Tests/DB/`, `Tests/Agents/`, `Tests/MCP/`, the related suites, and a
   repo-wide `--collect-only`; baseline anything else red against the merge base.

## Implementation Notes

All three guards repaired and each proven to bite. **These reds are no longer
available as a baseline**: the 6 `test_core_sqlite_owner_privacy` media
failures and the `test_tool_catalog_concurrency` failure have been carried as
"known pre-existing dev reds" by several merged PRs this week, and they are
gone. Any PR that lists them from here on is either stale or has introduced
them.

**A — product fix (the only production change).**
`DB/Client_Media_DB_v2.py:775` now raises `from error` instead of `from None`.
The severing arrived in `a189533c1` ("fix(library): contain media database
diagnostics"), which correctly scrubbed the *message* — the path and driver
text are gone — but also dropped `from e`, and the privacy contract identifies
its boundary failure by walking `__cause__` to a `PrivatePathError`. Chaining
is privacy-safe here: `PrivatePathError.__str__` is `"<status>: <symbolic
reason>"` and every `reason=` in `Utils/private_paths.py` is a fixed token
(`shared_writable_parent`, `missing_parent`, ...), never a path. The scrubbed
message is unchanged.

Audited exhaustively (AST scan of all 65 `PrivatePathResult(...)` constructions
repo-wide — the only way a `PrivatePathError` can be built): 60 pass a string
literal, 4 a ternary between two literals, and 1 is `_failure()` in
`private_sqlite.py`, whose 26 call sites are 25 literals plus one
`type(exc).__name__`. The only non-literal shape anywhere is
`type(exc).__name__` (3 sites), which yields an exception *class name* raised by
a syscall — never a path, filename, or user value.

**Scope caveat on that rationale.** The handler catches `(sqlite3.Error,
PrivatePathError)`, so `from error` chains the `sqlite3.Error` branch too, which
the argument above does not cover. Measured directly by rendering the real
failure through `traceback`, stdlib `logging(exc_info=True)` and loguru: on the
`PrivatePathError` branch chaining adds exactly one line — `PrivatePathError:
unsafe_parent: shared_writable_parent` — and leaks nothing. On the
`sqlite3.Error` branch it does make raw driver text traceback-renderable again,
which `a189533c1` had removed from the *message*. Five realistic connect/PRAGMA
failures were probed and every SQLite message is path-free (`unable to open
database file`, `file is not a database`, ...), so no path or user content can
escape; both app sinks are `diagnose=False` (`Logging_Config.py:438`,
`__init__.py:68`), so no frame locals are dumped either; and the two public read
paths re-sever with `from None`. If tighter containment is wanted later, the
one-line narrowing `from (error if isinstance(error, PrivatePathError) else
None)` satisfies the privacy contract exactly while keeping driver text severed.

**Does the shape recur? No — this was the only affected connect site.** The
four sibling owners (`base`, `chachanotes`, `prompts`, `evals`) either never
catch `PrivatePathError` or already chain with `from e`
(`ChaChaNotes_DB.py:3086`, `Prompts_DB.py:459`), which is why the identical
parametrizations passed for them. Verified by an AST sweep over every
`connect_private_sqlite()` call site in `tldw_chatbook/`: exactly three of the
~46 sites sit in a `try` whose handler can catch a `PrivatePathError` and
re-raises, and after this change all three chain. The other `from None` raises
in `Client_Media_DB_v2.py` (`:2302` media search, `:6758` distinct media types)
catch bare `Exception` — which *does* catch `PrivatePathError`, so the earlier
wording here was wrong. They are nonetheless correctly left alone, for a
stronger reason: `Tests/DB/test_client_media_debug_logging.py:229`
(`test_connection_open_failure_is_wrapped_without_private_diagnostics`)
explicitly asserts `raised.value.__cause__ is None` at those two sites. They
are deliberately, guardedly severed — and they re-contain the inner
`DatabaseError` on both public read paths. `DB/private_sqlite.py`'s many
`from None` raises are the *producers* of `PrivatePathError` — deliberate
`OSError`-detail scrubs where the boundary error is the `PrivatePathError`
itself — and were left alone.

**B — test fix.** The counter install and the test's own `list_catalog()` call
were swapped, plus an `assert snapshots == []` precondition so the counter can
only ever observe `invoke_by_name`. Production was correct throughout.

**C — seam repair.** Each file gained a `_pin_runtime_source(monkeypatch,
source)` helper that patches `local_server_tools.load_default_runtime_source_state`
(the owner-module loader TASK-18609 injects as `runtime_source_loader=`) and
returns a **real `RuntimeSourceState`** rather than the old fakes' bare
`"local"`/`"server"` string — that is production's shape, so the tests now
exercise `WatchlistsToolService._runtime_source`'s attribute branch instead of
its bare-string convenience branch. `source` accepts a callable so
`test_real_watchlists_provider_preserves_structured_domain_outcomes` can still
flip local→server→local between gateway calls. All **12** `raising=False`
arguments were removed (all of them in `test_local_server_tools.py`;
`test_gateway_runtime_tools.py` had none) — not just the 3 sitting on the stale
name: on an injection seam, a rename must fail loudly at the patch line.

**The `raising=False` cases were worse than "fail downstream" — two of them
were GREEN.** Re-measured individually at `da4e828af`:
`test_watchlists_first_local_call_opens_one_read_only_database` (`:261`) and
`test_watchlists_unready_database_is_bounded_and_keeps_other_tools` (`:450`)
both **passed** while patching the vanished `RuntimeSourceStateStore` — the
never-read attribute was installed, the service fell through to the real
loader, and the real loader happens to return `"local"`, which is exactly what
their fakes wanted. Only `:207` failed downstream on a scrubbed `ToolResult`.
So this change repairs **seven** tests, not five: the five reds plus two that
were green while asserting nothing about the seam they patched. The
green-and-hollow pair is the more dangerous half, and neither would ever have
been noticed from a test report.

**What the five tests were actually asserting, and whether that needed
updating.** The runtime-source patch was setup in all five, never the
assertion, so four needed only the re-point: server-mode must short-circuit
before any storage resolution; the lazy resolver must not replace a failed
candidate until its `close()` succeeds; the gateway must pass the provider's
structured domain outcomes through byte-for-byte; database resolution must run
off the event loop. **One did need updating.**
`test_real_watchlists_provider_scrubs_unexpected_failures` asserted no sentinel
in `str(exc)`, capsys, and a loguru sink — but the scrubber it guards
(`WatchlistsToolService._raise_unexpected`) logs through
`logging.getLogger(__name__)`, and none of those three channels sees stdlib
records. Adding `detail=%s` to that log call leaked the sentinel into the
captured log **and the test still passed**. It now also asserts
`sentinel not in caplog.text`, matching its sibling in
`test_local_server_tools.py`, and reds on that mutation.

**Bite-proofs (mutate production → guard reds → Edit-restore → `git diff` clean).**

| # | Mutation | Guard that red |
|---|---|---|
| A | `from error` → `from None` at `Client_Media_DB_v2.py:775` | 6 media parametrizations, `6 failed / 83 passed` |
| B | `_owner_record_for_name` takes two `_ensure_catalog_cache()` snapshots (the historical TOCTOU shape) | `test_invoke_by_name_takes_exactly_one_catalog_snapshot`, `assert 2 == 1` |
| C1 | `WatchlistsToolService._runtime_source` ignores the loaded state (always `"local"`) | `..._server_mode_never_resolves_path` **and** `..._preserves_structured_domain_outcomes` — proving the repaired seam is load-bearing |
| C2 | `_LazyWatchlistsDBResolver` replaces a failed candidate even when `close()` raises | `..._blocks_replacement_until_failed_close_succeeds` |
| C3 | `_raise_unexpected` logs `detail=%s` (the raw exception) | `..._scrubs_unexpected_failures` — **only after** the `caplog` assertion was added; it survived this and two other leak mutations before that |
| C4 | gateway calls the local handler inline instead of `asyncio.to_thread` | `..._database_resolution_runs_off_event_loop` (+ the generic `test_blocking_local_handler_runs_off_event_loop`) |

**Counts (merge base `da4e828af` → this branch).**

| File | Before | After |
|---|---|---|
| `Tests/DB/test_core_sqlite_owner_privacy.py` | 6 failed / 83 passed | **89 passed** |
| `Tests/Agents/test_tool_catalog_concurrency.py` | 1 failed / 2 passed | **3 passed** |
| `Tests/MCP/test_local_server_tools.py` + `test_gateway_runtime_tools.py` | 5 failed / 93 passed | **98 passed** |

Suite gates: `Tests/DB/` 1077 passed / 1 skipped; `Tests/Agents/` 1793 passed;
`Tests/MCP/` 1042 passed; repo-wide `--collect-only -q` 54996 collected, exit 0
(identical to the merge base). `Tests/Media_DB/ Tests/RuntimePolicy/ Tests/Tools/
test_application_state_ownership.py test_remaining_diagnostic_sentinel_matrix.py`
= 2 failed / 1211 passed / 6 errors on **both** this branch and the merge base
(`test_reading_progress_reopens_through_versioned_migration` — a v5→v6
`duplicate column name` migration bug; `test_legacy_server_client_builder_matches
_are_listed_in_migration_audit`; and 6 huggingface-egress fixture errors whose
blamed test ids shuffle run to run). Pre-existing, untouched. `ruff check` clean
on all four changed files; the 5 `ruff check` errors elsewhere under
`Tests/MCP/` are pre-existing at the merge base. **Correction:** `ruff format
--check` is *not* clean on `Client_Media_DB_v2.py` — it reports "would
reformat", but identically at the merge base, and every hunk it wants is in
untouched code (lines 733, 970, 987, 2238, 2289, 3946); the changed region
(769–787) is correctly formatted. The other three files are clean.

`test_reading_progress_reopens_through_versioned_migration` is a **test**
defect, not a shipped migration bug — see the characterisation below.

**Characterisation of the pre-existing media-migration red (for filing).**
`test_reading_progress_reopens_through_versioned_migration` dies with
`DatabaseError: Migration v5->v6 failed: duplicate column name:
chunk_engine_version`, thrown out of `MediaDatabase.__init__`, so the database
cannot be opened at all. **It does not brick a real upgrade.** Probed against
the real `MediaDatabase`: a genuine v2 database upgrades cleanly to v6, and so
does a genuine v5 database (the actual shipped path). The migration is also
atomic — poisoning the version bump after the `ALTER` leaves `version=5` with
the column *absent* and a retry succeeds — so no partial-apply state is
reachable. The failing state is manufactured by the test itself: it fakes a
"v2" database by dropping v3's `ReadingProgress` table and v5's
`transcription_provenance_json` column but **not** v6's
`chunk_engine_version` column, which task-11 added after this test was written.
The consequence is the same class as the three findings above: the media DB's
whole v2→v6 migration chain has been unguarded since v6 landed, because the one
test that walks the chain dies on its last hop. Worth noting separately: any
such schema drift is *unrecoverable* (the second open fails identically, with no
tolerance or repair path), so an idempotent `ADD COLUMN` would be cheap
hardening.

The 6 egress errors are one root cause, not six: something on this path loads
`sentence-transformers/all-MiniLM-L6-v2`, `huggingface_hub` issues a real
`HEAD https://huggingface.co/...` , the network guard blocks it, and
`huggingface_hub` **retries with backoff** — so the blocked-attempt record
drains into whichever test happens to be tearing down. Count is stable (2 in
`Tests/Tools/test_document_expansion_tool.py`, 4 in
`Tests/Tools/test_file_tools_workspace_roots.py`); the blamed ids are not.

**Modified files.** `tldw_chatbook/DB/Client_Media_DB_v2.py`,
`Tests/Agents/test_tool_catalog_concurrency.py`,
`Tests/MCP/test_local_server_tools.py`,
`Tests/MCP/test_gateway_runtime_tools.py`,
`backlog/docs/lessons-testing-evidence.md`.

## Review follow-up (Qodo, PR #1961)

Qodo raised one finding — a rule violation, "Non-contiguous imports in the
gateway test": `Tests/MCP/test_gateway_runtime_tools.py` split its import
section with `gateway = pytest.importorskip("mcp_unified.gateway")` plus five
attribute assignments, so the third-party and local groups were not contiguous
and every local import carried `# noqa: E402`. The new `RuntimeSourceState`
import had been added into that already-split section.

The guard cannot simply be deleted — `mcp-unified` really is optional
(`[mcp]` extra in `pyproject.toml`) and both `mcp_unified.gateway` and
`tldw_chatbook.MCP.gateway_runtime` resolve it at module scope, so the skip has
to run before the imports. It is now a probe-and-skip preamble placed *ahead of*
the three import groups rather than between them:

```python
try:
    import mcp_unified.gateway  # noqa: F401
except ImportError:
    import pytest

    pytest.skip("mcp-unified extra not installed", allow_module_level=True)
```

`try`/`except` lines are exempt from pycodestyle's import-position rule and the
handler body is indented, so nothing below is flagged: the file now has three
contiguous blocks (stdlib / third-party incl. the direct `mcp_unified.gateway`
symbol imports / local, alphabetised) and zero `# noqa: E402`. Confirmed with
`ruff check --isolated --select E402,F401,F811` → "All checks passed!".

**A conftest `collect_ignore` was tried first and rejected on evidence.** With
`Tests/MCP/conftest.py` holding the guard, a directory run skipped the module
correctly, but `pytest Tests/MCP/test_gateway_runtime_tools.py` — a path named
explicitly on the command line — is *not* subject to `collect_ignore` or to
`pytest_ignore_collect`, and produced `ModuleNotFoundError: No module named
'mcp_unified'` → "1 error during collection". That is strictly worse than the
`importorskip` it replaced, so the conftest was removed.

**Guard proven to still guard.** Verified with a `-p` plugin that installs a
`sys.meta_path` finder raising `ModuleNotFoundError` for `mcp_unified`:
explicit-file run → `SKIPPED [1] ...:19: mcp-unified extra not installed`,
`1 skipped`; directory run (`Tests/MCP/`) → same skip line, no collection error.
Every `monkeypatch.setattr` target was re-resolved after the reorder
(`local_server_tools.get_subscriptions_db_path`, `.SubscriptionsDB`,
`.load_default_runtime_source_state`, and `runtime_policy.types.
RuntimeSourceState`), and `import tldw_chatbook.MCP.local_server_tools as
local_server_tools` was rewritten as `from tldw_chatbook.MCP import
local_server_tools` — asserted to bind the identical module object, so the
patches land where they did before.

**Counts (unchanged by the fix).** `Tests/MCP/test_local_server_tools.py` +
`Tests/MCP/test_gateway_runtime_tools.py` → 98 passed;
`Tests/DB/test_core_sqlite_owner_privacy.py` +
`Tests/Agents/test_tool_catalog_concurrency.py` → 92 passed; all four together
→ 190 passed. Repo-wide `--collect-only -q` → 55619 collected, 1 error in
`Tests/UI/test_library_file_notes_workspace.py`
("function uses no argument 'push_phase'") — a pre-existing dev red in a file
this branch does not touch (last changed by `d3833708a`).

Noted, not fixed (out of scope, untouched file): under the same blocker,
`Tests/MCP/test_tools_resources_prompts_real_methods.py` fails rather than skips
— it does a bare in-test `from mcp_unified.gateway import GatewayRequestContext`
with no guard, in two tests.

**Modified files (review follow-up).** `Tests/MCP/test_gateway_runtime_tools.py`.
