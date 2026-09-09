# Buddy PR #2526 — Qodo review corrections

Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/buddy-console-management`
Starting HEAD: `abf47df983a20673efc8de1e0cb3a883e8af2828` (rebased dev integration supplied by root).
No git mutations, nested agents, full-suite runs, paid providers, microphone devices or live profile changes.

## Dispositions

| Qodo comment | Disposition and evidence |
| --- | --- |
| 3962487188, raw Buddy archive paths | Fixed. `read_buddy_archive` lazily calls central `validate_path_simple(..., probe_existing=False)` and passes the returned Path to the existing pinned/no-follow archive reader. Central traversal rejection precedes filesystem opening; selected-file symlinks still fail. The raw-path regression failed before the fix; traversal and no-follow regressions pass. |
| 3962487197, raw management values | Fixed. Lazy `BuddyManagementInput` in central `Utils/input_validation.py` strictly validates bounded strings, booleans, motion and integer dimensions before constructing the domain choice. Dimension normalization reuses `validate_bounded_integer`; target/artwork/Persona membership and edit authority remain explicit UI checks. Overlong archive input previously constructed a choice and now rejects. Lazy import preserves the boot import boundary. |
| 3962487211, synchronous saved hydration | Fixed for collection-size-dependent reads. The real local scope-service tree reader is offloaded while preserving local policy checks and memory-DB behavior. `prepare_console_session_data` preloads unpublished nodes/attachments, cursor, continuation/thinking rows and generation sidecars. Buddy rechecks record/profile/runtime and competing exact session bindings after awaits, then calls ordinary hydration/store publication on the app thread. Worker DB connections close on their own threads. Direct hydration callers remain compatible. See the explicit scope limit below. |
| 3962487220, recorder callback thread | Declined as non-applicable. Installed Textual **8.2.8** `MessagePump.call_later` creates an `events.Callback` and calls `post_message`; `post_message` detects foreign threads and uses `loop.call_soon_threadsafe(self._message_queue.put_nowait, message)`. An actual mounted Textual regression starts the recorder seam in a worker, fires its captured buffer-limit callback in a worker and observes `request_voice` on the app thread. It passes unchanged. Replacing this with blocking `call_from_thread` is unnecessary. |
| 3962487231, unbounded Buddy enumeration | Fixed. `list_buddies(limit=100, offset=0)` validates a maximum page size of 100; `get_buddy` uses a parameterized exact-ID query instead of enumerating the library. Native Previous/Next controls retrieve pages on demand; a separately looked-up current Buddy and any staged selection remain available across pages. Literal Rich Text labels preserve brackets; selections do not silently truncate. A real SQLite test loads 206 installed rows across pages and proves lookup does not call listing. |
| 3962487241, Buddy read transaction boundaries | Fixed. Listing, exact lookup and source-key lookup execute inside `db.transaction()`. SQLite trace regression records each library SELECT inside a transaction. Publication/idempotency behavior remains covered by existing library tests. |
| 3962487249, Workspace schema probe | Fixed narrowly. The v8 `PRAGMA table_info(workspace_records)` detection read uses `WorkspaceDB.transaction()`. Existing initializer/bootstrap semantics and the independent v8 atomic migration runner remain unchanged. Existing fresh/upgrade/restart/failed-version-write tests cover the migration. |
| 3962487256, v69→v70 migration guards | Fixed. Entry and resulting-version reads occur inside the migration transaction, on `cursor.connection`. A forced resulting-version mismatch previously left version70/schema committed; it now rolls schema and version back to69. |
| 3962487265, public library documentation | Fixed. Every public BuddyLibrary method now describes arguments, return values and applicable failure modes, including pagination bounds, unavailable identities, publication source guards and non-mutating selection behavior. |
| 3962487270, Skip priority mismatch | Fixed. Drain and pending Skip share the same question-first selection helper. Paused response→question regression previously spoke the question after Skip and now retains/speaks the response. |
| 3962487279, forged import provenance | Fixed in **both** import paths. Authoring draft metadata and independent Buddy archive snapshots merge archive context first and assign reserved `untrusted-import` last. Tests use attacker-controlled `source_context.provenance="trusted-builtin"`; authoring review and real Buddy publication preserve the system marker while retaining the independent license field. Both tests failed before their respective fixes. |

## Hydration scope and authority

This change does not claim that every SQLite operation leaves the app loop. Existing small per-conversation project/capture/context/speech policy reads, authority-locked dispatch reconciliation and durable repair/write bookkeeping remain in the established store restoration path. Moving those mutations or lock-coupled reconciliation into a worker would change ownership semantics. The potentially large tree, attachment, continuation/thinking, generation-sidecar and cursor reads now finish before publication. `:memory:` databases retain connection-local inline reads; production file-backed databases use worker connections. This is the bounded seam approved by root under ADR-139, not a new execution/runtime owner.

The ten slow-read regressions cover tree, attachments, cursor, continuations and generation, each in normal and mid-read profile-change cases. They failed on the original synchronous reads. They now prove loop progress while a threading.Event holds the read, exactly one off-app-thread bulk read, UI-thread publication only after release, no publication after profile identity changes, preservation of the unrelated active session, and worker-connection cleanup. Additional tests cover before-first cursor reconstruction with and without preloading and a real `:memory:` database.

## Three existing hydration test contracts corrected

Initial broad gate: **220 passed, 3 failed in 96.16s** (log `/tmp/buddy-review-targeted.log`). The failures were:

1. `test_resume_restores_the_complete_versioned_console_settings_snapshot`: legacy row-only wrapper returned base provider/settings instead of restoring the full legacy serialized dataclass.
2. `test_canonical_hydration_makes_persisted_generic_console_forkable`: actual assistant kind `generic` versus expected `None`.
3. `test_settings_replacement_refreshes_the_durable_resume_snapshot`: actual `llama_cpp` versus expected `openai`, because the test used the legacy `console_session_settings` replacement/wrapper instead of canonical safe generation persistence and hydration.

Controlled baseline reproduction: `/tmp/buddy_baseline_plugin.py` executes **HEAD versions of the scope-service, store and hydration modules** in `pytest_configure` after conftest has established isolated test configuration; then runs exactly those three unchanged test functions. All three fail identically (2.43s, `/tmp/buddy-baseline-hydration.log`). This is a controlled module-baseline run, **not** a pristine whole-repository baseline checkout. No source files were swapped during either test run.

Contract evidence: ADR-095 makes safe `console_generation_settings` authoritative, excludes endpoint/identity/prompt/prefill from its payload, and requires current configuration plus row-owned prompt/prefill during resume. The legacy `apply_resume_settings_overrides` docstring explicitly promises only the two row owners. Tests now use `commit_console_settings_live` + `persist_console_settings_commit_serialized` and `hydrate_console_generation_settings` for the round trip, while the wrapper test verifies base settings are preserved. Provider/model/temperature/prompt/prefill assertions remain. ADR-139's explicit None path maps to the current plain `generic/console` store identity; `console_chat_fork` explicitly accepts that identity without Persona/character authority. The fork eligibility assertion remains. All three corrected tests pass (2.73s); no unrelated production settings behavior was changed.

## Verification commands and evidence

All pytest commands use the main venv and worktree imports:

```sh
PYTHONPATH=.:packages/tldw_profile_core/src /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <listed files> -q --no-cov --show-capture=no --tb=short
```

- Initial direct reproductions: Skip mismatch; authoring spoof; path validation bypass; unbounded listing; transaction-free reads; post-commit migration verification; overlong modal path; ten blocking bulk-read cases.
- Small fixed gates: 4 path/provenance/Skip tests; 3 paging/transaction/migration tests; 13 management modal tests; 13 bulk-read/late-identity/concurrent-loader tests; 3 corrected canonical contracts.
- Latest narrow final gate: **13 passed in 10.49s**, covering all ten bulk-read cases, before-first cursor through both paths, and real memory-DB hydration.
- Mounted recorder callback regression passes without production callback changes.
- Final affected integrated gate: **362 passed, 6 warnings in 189.34s**. Durable log: `task-1-final-tests.log` beside this report. The only later source edits moved one TYPE_CHECKING import and wrapped one expression; post-format checks below passed.
- Post-format gate: **20 passed, 76 deselected, 2 warnings in 20.70s**. Actual UI-ready count: **973/973 modules**. Durable log: `task-1-census-tests.log`.
- No unresolved semantic or test failures. Source remains frozen for root's artifact regeneration and independent review.
- Static baseline comparison after both cosmetic corrections: **19 owned changed Python files parse; zero introduced Ruff diagnostics; zero formatter changes intersect modified ranges**. `task-1-static.json` records per-file evidence; inherited lint/format debt is not being rewritten. Source SHA-256 values are recorded in `task-1-frozen-sources.json`.
- Baseline reproduction log and isolated loading plugin are preserved as `task-1-baseline-tests.log` and `task-1-baseline-plugin.py`; import/order limitations remain exactly as described above.
- New Buddy production/test modules pass direct Ruff. `git diff --check` passes.

No new ADR: existing ADR-139 owns the hydration/runtime boundary; ADR-095 supplies the corrected settings-test contract.


Final integrated file set (all from the frozen worktree, main venv):

```text
Tests/Persona_Buddy/test_buddy_library.py
Tests/Persona_Buddy/test_buddy_speech.py
Tests/Persona_Visual/test_persona_visual_importer.py
Tests/Persona_Visual/test_persona_visual_authoring.py
Tests/Persona_Buddy/test_buddy_management_coordinator.py
Tests/UI/test_buddy_management_modal.py
Tests/UI/test_buddy_conversation_modal.py
Tests/UI/test_buddy_management_journey.py
Tests/UI/test_buddy_entry_points.py
Tests/DB/test_workspace_db.py
Tests/Workspaces/test_workspace_db_connection_reuse.py
Tests/Chat/test_chat_conversation_scope_service.py
Tests/Chat/test_console_conversation_hydration.py
Tests/Chat/test_console_provider_continuation.py
Tests/Chat/test_console_thinking_persistence.py
Tests/Chat/test_console_settings_apply_store.py
Tests/Packaging/test_persona_buddy_import_closure.py
Tests/Performance/test_app_import_weight.py
```

Post-format command:

```sh
PYTHONPATH=.:packages/tldw_profile_core/src /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Performance/test_ui_ready_module_census.py Tests/Chat/test_console_conversation_hydration.py Tests/UI/test_buddy_conversation_modal.py -k 'ui_ready or bulk_reads or before_first_cursor or memory_database or legacy_resume_wrapper or canonical_hydration_makes or canonical_settings_apply' -q --no-cov --show-capture=no --tb=short
```

Self-review checked shared-form authority membership, staging across page changes, unchanged no-follow archive checks, both import provenance routes, explicit record/profile revalidation after bulk reads, memory-DB fallback, continuation quarantine fallback, migration failure rollback and absence of off-thread store publication. No new runtime service, dependency, provider permission or tool authority was introduced.

## Rebase and derived checks

Rebased onto dev `80f29a9a1dcd9307662714233c65605e4c517b11`. The only conflict was the testing-evidence lessons document; both independent additions were retained. The feature branch's prior head is preserved by a local recovery ref.

The failed derived-artifact gate exposed two omitted schema allowlist entries (`buddy_profiles`, `buddy_visual_bindings`) and diagnostic inventory drift. Both table names now have matching migrated declarations. Production diagnostic statement changes across all 57 changed production files were reviewed before regeneration: additions contain fixed status messages, with no new interpolated user content or persistent sink topology change.

All six derived checks pass: generated stylesheets; profile-owned paths (48 occurrences / 46 exceptions); diagnostic inventory (593 owners / 12 sink files); unique Backlog IDs; table allowlist (115 tables); and index decisions (282 names / 67 plan pins). Large Buddy-library pagination is documented in the user guide.

## Independent review

A separate source reviewer inspected all eleven dispositions, current fixes, frozen source hashes and existing verification logs without rerunning tests. No Important or Critical findings remained. The reviewer confirmed both import provenance paths, bounded pagination and exact lookup, migration rollback, question-first Skip, worker-connection cleanup and late authority checks, and the independent ADR-095 evidence behind the three corrected settings tests.

## Final dev update

Dev advanced once more to `8aa2211f2b78fcb35c9d1d3db6e15d5f1978da0a` while publication was being verified. A second rebase completed without conflicts; range-diff confirms all three Buddy commits replayed unchanged. The new base adds independent Library structural-wait handling. Fresh Buddy management journeys, entry points, conversation modals and actual startup census passed **45 tests in83.26s**; imports remain973/973. The diagnostic inventory verifies again without regeneration. These checks cover the navigation/startup intersection with the new base rather than claiming another full affected-suite run.
