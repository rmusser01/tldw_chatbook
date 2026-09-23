# S13 — `Canvas/` + `Workspaces/` + `Research_Workspace/`

**Coverage:** files read in full: 6 | sampled: 28 | mechanical only: 19 (of 53).
Out-of-slice files traced as evidence: `scripts/vendor_canvas_mermaid.py`, `scripts/check_canvas_mermaid_assets.py`,
`Notes/notes_scope_service.py`, `MCP/redaction.py`, `Widgets/Console/console_workspace_context.py`.

## Findings

### P1 [D1] — `ShadowRepo._locked()` spins forever, at ~28,000 iterations/second with a `logger.warning` each, when the cross-process lock dir exists but cannot be `rmdir`'d
- Where: `Workspaces/change_tracking.py:322-349` — the two bare `continue` statements at `:331` and `:343`.
- Evidence: **reproduced.** A repro constructs a `ShadowRepo`, creates `lock.d` containing one file, back-dates its
  mtime past `_STALE_LOCK_SECONDS` (300 s), enters `_locked()` →
  **`SPUN 2000 times in 0.072s (deadline was 60.0s, never checked on this branch)`**. The `rmdir()` raises
  `ENOTEMPTY`, is swallowed by `except OSError: pass`, and `continue` re-enters the loop **without
  `time.sleep(_LOCK_RETRY_SECONDS)` and without testing `time.monotonic() > deadline`.** The `stat()`-failure branch
  at `:330-331` has the identical defect.
- Why it matters: **shipped path** — `Chat/console_runtime.py:3534` builds a `ChangeTurnTracker` per agent turn,
  which calls `repo.snapshot(...)` → `_locked()`. The spin holds `_in_process_lock(git_dir)`, so every other
  change-tracking operation on that root then hits its own 60 s timeout; **a CPU core is pegged and the log is
  flooded until the app is killed.** Precondition is an external actor leaving a file in the lock dir (`.DS_Store`,
  a backup/AV agent, an `.nfs*` silly-rename) — not self-inflicted, **which is why this has never been seen, and
  also why nothing recovers from it.**
- Recommended correction: move the deadline test and `time.sleep(_LOCK_RETRY_SECONDS)` to the top of the
  `except FileExistsError` body so every path through the loop is bounded and rate-limited. **Secondary:** `age`
  mixes wall clock (`time.time()` vs `st_mtime`) with a `time.monotonic()` deadline — a forward NTP step makes a
  fresh lock look stale and permits a premature takeover.
- Size: S · Confidence: **verified (reproduced)**

### P1 [D1] — `set_folder_binding_access` performs the same binding-metadata read-modify-write as the exclusion editors but does not take `_BINDING_EXCLUSION_EDIT_LOCK`, so one silently reverts the other
- Where: `Workspaces/registry_service.py:2747-2775` (**unlocked**) vs `:2625` and `:2663` (both **under** the lock,
  defined `:375`); the shared writer is `save_runtime_binding:2486-2546`, whose
  `ON CONFLICT DO UPDATE SET metadata_json = excluded.metadata_json` **replaces the whole blob**.
- Evidence: both are user-driven and provably on different threads — `UI/Screens/settings_screen.py:25468` calls
  `set_folder_binding_access` **synchronously inside an `@on` button handler** (the Textual event loop), while
  `UI/Console_Modules/workspace.py:1105-1133`'s docstring states *"Callers run this off the Textual event loop"*
  before calling the exclusion editors. **The lock's existence is the author's own acknowledgement that this RMW
  races.**
- Why it matters: last writer wins on the whole `metadata` dict. Toggling a folder binding to read-only in Settings
  while the Workspace Files modal commits an exclusion silently drops the other edit — and **both are access-control
  settings over what agent file tools may touch, so a reverted `access: "ro"` re-grants write without telling the
  user.**
- Recommended correction: do the RMW as one `SELECT … ; UPDATE …` inside a single `self.db.transaction()` and
  **delete the module-level lock — a process-local lock cannot serialize a cross-transaction RMW anyway.**
- Size: S/M · ADR: no (`174-workspace-binding-exclusions.md` specifies the feature, not the concurrency) ·
  Confidence: inferred (races traced to code and threads; not reproduced live)

### P2 [D4] — `Workspaces/models.scrub_secret_metadata` is a second, weaker secret-key policy than `MCP/redaction.redact_mapping`, and it is the scrubber on every runtime-binding metadata write
- Where: `Workspaces/models.py:76-84`, `:349-376`, applied at `:232`. Shared helper: `MCP/redaction.py:12, 90, 95,
  138` — 6 importers.
- Evidence, both functions run:
  ```
  workspaces: {'passwd':'hunter2','authorization':'Bearer abc','bearer':'xyz','note':'sk-live-0123…'}
  mcp       : {'passwd':'***','authorization':'***','bearer':'***','private_key':'-----BEGIN','note':'***','api_key':'***'}
  ```
  **Drift is bidirectional:** Workspaces misses `passwd`/`authorization`/`bearer` and has **no value-shape matcher
  at all**; the shared redactor misses `private_key`, which Workspaces catches. Workspaces *drops* the key; the
  shared one *replaces* with `***`.
- Why it matters: this is the boundary between caller-supplied binding metadata and the
  `workspace_runtime_bindings.metadata_json` column on disk. Today's only writers are internal, **so this is
  defence-in-depth with a hole, not a live leak** — but the drop-vs-redact difference also means a legitimate
  `api_key`-named field is **silently lost on round-trip** rather than flagged.
- Size: S · Confidence: verified (empirical)
- Already covered: **TASK-32806.7 (In Progress) — a new instance it does not name.**

### P2 [D1/D3] — `Canvas/` and `Research_Workspace/` have 216 `except` clauses and **zero** log statements, and `Canvas/service.py` re-raises outside the handler so the original exception is discarded
- Evidence:
  ```
  Canvas:             files=18 lines=13884 exceptclauses=167 loggercalls=0
  Research_Workspace: files=14 lines= 7841 exceptclauses= 49 loggercalls=0
  Workspaces:         files=21 lines=13785 exceptclauses=163 loggercalls=38
  ```
  Six near-identical ladders in `Canvas/service.py:114-123, 324-331, 434-441, 482-489, 667-674, 688-695` each set
  `repository_error = CanvasServiceError("operation_failed")` inside `except Exception:` and then
  `raise repository_error` **after** the handler, **so `__context__` is unset**. Also `Canvas/gateway.py:1404,1413`
  `raise RuntimeError(...) from None`, which deletes the aiohttp bind reason.
- Why it matters: a Canvas failure for an unexpected reason surfaces as a generic `operation_failed` and **leaves no
  record anywhere in the process.** A "Canvas won't open" report is unactionable.
- Recommended correction: `backlog/decisions/121-…-browser-sandbox.md:573` is explicit — *"Late worker failures are
  observed without logging arbitrary exception representations"* — **but that decision governs content, not
  observability, and does not require discarding the exception class.** Log `type(exc).__name__` plus the stable
  refusal code; the same-repo precedent is `Workspaces/registry_service.py:719`. Add `from exc` on the re-raises.
- Size: M · ADR: no (ADR-121 constrains the shape, not the existence) · Confidence: verified

### P2 [D2] — `build_console_workspace_state` issues the same unbounded `workspace_memberships` scan twice per rail build, on the Textual event loop, then filters in Python
- `Workspaces/display_state.py:461` and `:508` both call `list_workspace_memberships(...)`, which is
  `SELECT * … WHERE workspace_id = ? ORDER BY …` with `.fetchall()` and **no `LIMIT`**
  (`registry_service.py:1416-1435`). `display_state.py:451` comments *"A pure UI-loop state build…"* — **this runs
  on the loop by design.** The sibling paged accessors `list_workspace_source_memberships` /
  `list_workspace_note_memberships` **prove the table is expected to grow past a page.**
- Size: S · Confidence: verified (by reading; not timed)
- Already covered: TASK-32804 `.11`/`.12` — **a Console-rail instance neither enumerates.**

### P2 [D4] — `Workspaces/models.utc_now_iso` **shadows the sanctioned `Utils/timestamps.utc_now_iso` under the same name** and produces a different, variable-width shape
- `Workspaces/models.py:97-100`, used as the `default_factory` at 8 sites and imported into
  `registry_service.py:46` → `self._now_factory:491`.
  ```
  workspaces: 2026-09-22T04:54:52.894068+00:00   (32 chars)
  canonical : 2026-09-22T04:54:52.894Z           (24 chars)
  workspaces zero-usec: 2026-09-20T12:34:56+00:00 (25 chars — .%f omitted entirely)
  ```
- Why it matters: the canonical shape is fixed-width *"so lexical (`TEXT`) ordering in SQLite equals chronological
  ordering"*; the registry's `ORDER BY created_at ASC` runs over variable-width values. **I checked for a mixed-shape
  column and found none**, so this is not a proven ordering bug today. **The live hazard is the name collision:
  "fixing" the import to the shared helper silently changes every stored shape and *creates* the mixed column.**
- Recommended correction: adopt the shared helper **with** a read-side migration/tolerance (`parse_utc` already
  reads both), **or rename the local one so the collision cannot be resolved by accident.**
- Size: M · ADR: yes (ADR-127/173 already decide it) · Confidence: verified
- Already covered: **TASK-32803.5 (Done) — now insufficient: 9 aware-but-non-canonical writers survive in this slice
  alone, in five distinct shapes** (`Canvas/repository.py:1557`, `Canvas/staging.py:777`, `Workspaces/models.py:100`,
  both `Research_Workspace` adapters, `source_operation_store.py:61`, `change_retention.py:102`).

### P2 [process] — The wired `check_canvas_mermaid_assets` preflight check is a **tautology** for 2 of its 6 declared outputs
- `scripts/vendor_canvas_mermaid.py:349-353` — `build()` does
  `outputs["canvas_runtime_worker_v2.js"] = (STATIC / "canvas_runtime_worker_v2.js").read_bytes()` (and the same for
  `canvas_renderer_v2.js`) and writes them to `output_dir`; `check_canvas_mermaid_assets.py:232` then compares
  `rebuilt` against `committed_dir`, **which defaults to `STATIC` — the directory `build` just read from.**
  `grep -rn` → **no generator anywhere**; the only producers are those two `read_bytes()` calls. **Lead-verified —
  see `phase4-verification.md`.**
- Why it matters: the success line says *"Canvas Mermaid assets reproduce: 6 outputs"*. Four reproduce from
  hash-pinned inputs; **two are checked-in source laundered through a copy** — and they are the largest executables
  in the Canvas runtime closure (85 KB + 59 KB). Tampering is caught only indirectly via a *different* file's
  manifest digest, which a re-run of the documented `reproducible_command` erases.
  **Related:** `Canvas/static/` also holds `canvas_shell.{html,css,js}` (51 KB of browser-side JS **holding the
  session cookie**) and `mermaid-authoring.txt`, none of which is in any manifest `outputs` set — hand-authored
  files living in the generated-assets directory with no integrity check, unlike `canvas_renderer.js` which has one.
- Size: M · ADR: yes (`124-canvas-mermaid-subset-and-immutable-runtime-profiles.md` owns the claim this check backs)
  · Confidence: verified
- Pinning test: `Tests/CI/test_canvas_mermaid_asset_checker.py` pins the comparator but **duplicates
  `EXPECTED_OUTPUTS` verbatim — it cannot notice that two members are self-certifying.**

### P3 [D3] — Six verbatim copies of the same 6-line repository-error ladder in one file
- `Canvas/service.py:114-123, 324-331, 434-441, 482-489, 667-674, 688-695`. One `_repository_call(self, fn, …)`
  carries the P2 logging fix once instead of six times. · Size: S

### P3 [D1] — Startup resume discards every exception from its fan-out
- `Research_Workspace/source_association.py:545-548` and `source_readiness.py:304-307` —
  `await asyncio.gather(…, return_exceptions=True)` **with the returned list unbound**, in modules with zero logger
  calls. A boot-time resume of a partially-committed source operation can fail for every operation and the app
  reports nothing. · Size: S

### P3 [D1] — `remove_`/`has_internal_research_quick_note_owner_proof` skip `_enforce_policy` while every sibling mutator enforces it
- `Notes/notes_scope_service.py:2333-2384`, reached from this slice at `Research_Workspace/local_adapter.py:497,688`.
  **Reachability that matters:** `local_adapter.py:524 _reconcile_quick_note_receipts` resumes leftover receipts at
  startup and can reach `remove_` **without the create gate ever running.** Narrow (the caller must already possess
  the `owner_proof` secret) but an inconsistent enforcement surface on a proof-of-ownership record.
  *(This is the reachability leg of the gap S24's census found.)* · Size: S

### P3 [D4] — Small verbatim clusters inside this slice
- `_page_bounds` (`1 ≤ limit ≤ 100`, `0 ≤ offset ≤ 10 000`) **four times, byte-identical**:
  `Research_Workspace/{server,local}_adapter.py:105/90` and inlined at `registry_service.py:1449-1452, 1493-1496`.
  Home: `Research_Workspace/contracts.py` (already imported by both adapters).
- `_read_bounded` and the `_strict_json`/`object_pairs` hook byte-identical between `Canvas/runtime_assets.py:61,70`
  and `Canvas/profiles.py:190,176` — **two files in the same package.**
- **Drift worth naming:** `Canvas/runtime_assets.py:100 _freeze_json` tests `isinstance(value, dict)` while
  `Canvas/models.py:607 _freeze_json_value` tests `isinstance(value, Mapping)` — **the former leaves a non-`dict`
  Mapping unfrozen.** · Size: S

## Candidate triage
**RETIRED — the lead's dispatch hypotheses, all four:**
- **`Canvas/gateway.py:2898 _canonical_json_wire` datetime/Decimal drift: retired — NO drift.** All four copies ran
  against `datetime`/`Decimal`/`frozenset`/`bytes`/`tuple`/nested: **byte-identical outputs**, every non-container
  passes through unchanged. Both *digest* users use identical `json.dumps` kwargs.
  **task-32862 is a pure mechanical consolidation, not a format decision.** *(This closes the question S14 left open.)*
- **`Canvas/gateway.py:2883 _maybe_await`: retired.** The sync branch returns already-computed authority results.
- **`subprocess.run` without `timeout=` in `Workspaces/`: retired — all four have one**
  (`git_workspace.py:230`, `change_tracking.py:285`, `change_retention.py:58`, `environment_status.py:104`), all
  pass `stdin=DEVNULL` or scrub `GIT_*`, and handle `TimeoutExpired`. **No thirteenth here.**
- **Bracket-containing git error templates: retired for the templates, confirmed for the interpolation.**
  **`PushRefusedError` does not exist anywhere in the repo.** No static template contains a bracket. But
  `git_workspace.py:246-248` and `change_tracking.py:303-305` interpolate **raw git stderr** (capped at 400 chars),
  and `:244` interpolates `str(OSError)` — both routinely carry user paths. **So `Workspaces/` is the producer for
  S18's `notify()`-with-markup finding, and the fix belongs at the consumer (escape), not here.**
- `server_adapter.py` import-closure guard — **green**:
  `pytest Tests/Packaging/test_research_workspace_import_closure.py -q` → **2 passed**.
**RETIRED:** `fetchall_dynamic_sql` `registry_service.py:761` (two constant SQL literals selected on a bool);
`fetchall_no_limit` ×17 — Canvas rows bounded by `MAX_CANVAS_*` constants (SQLite var limit here is 32,766, checked);
`registry_service.py:1402,2472,2815` genuinely unbounded but cold, **only `:1425` is on a hot path** (the P2).
`lock_and_execute` — **retired as filed, re-filed as P1**: both the read and write go through
`db.connection()`/`db.transaction()`; **the real defect is that the lock is incomplete.**
`id_keyed_dict` ×11 (`Canvas/compiler.py`) — function-local dicts keyed on `lxml` elements the enclosing locals keep
alive; **not the recompose-lifecycle pattern.** `re_compile_in_def` — `re._cache` is 512 entries and the tag
vocabulary is a small closed set. `raw_mkdir` ×6, `raw_1024x1024` ×11 — named byte ceilings, all enforced;
`Canvas/archive.py:229` uses `destination.open("xb")` after the mkdir. `except_exception_pass`/`return` ×12 —
folded into the P2 logging finding; the distinctly bad ones are
`change_review_finalization.py:975-977` and the two `gather` sites. `try_import_guard` `change_bounds.py:111` —
the **2-arg** `(section, key, default)` form, not the broken dotted shape. `function_body_import` ×32 — **61
function-body imports of `Backup_Recovery.storage_admission` exist repo-wide**; a repo-wide pattern, not a slice
defect. `registry_service.py` (3,449) / `Canvas/gateway.py` (3,026) as god modules — **both under the ratchet's own
floor (6,760).**
**RETIRED with evidence — symptom real, consumers safe:** `_membership_display_title` emits `[8-hex]` into a Textual
label (`display_state.py:856`), but **both consumers escape** — one goes to `Static(..., markup=False)`, the other
through `_escape_markup` before `Text.from_markup`. The disambiguator survives.
**RETIRED — deliberately stronger:** `Workspaces/file_inspector.py` rolling its own path validation.
`_open_root_descriptor:832`/`_open_target_descriptor:857` traverse with `O_NOFOLLOW|O_DIRECTORY` `openat`
(`dir_fd=`) and pin `st_dev`/`st_ino`, **never path strings. TOCTOU-free; a string validator would be a downgrade.**

## D4 observations for repo-wide Phase 3
1. **`_json_wire`/`_canonical_json_value` (4 copies) — no drift, settled.** Verified byte-identical on
   datetime/Decimal/frozenset/bytes/tuple; both digest users pass identical `json.dumps` kwargs.
   **task-32862 can proceed as a mechanical swap.**
2. **Secret redaction is two policies, not one** — the recursion shape is the boring part, **the leaf policy is the
   finding.**
3. **The pagination contract is inlined 4× in this slice alone**, and the paged/unpaged split inside
   `registry_service` is itself a copy-paste pair.
4. **`Utils/timestamps.py` has 28 importers and zero in this slice**, which owns **9 writers across 5 shapes**.
5. **`storage_admission` is imported inside 61 function bodies repo-wide.** If that is a deliberate boot-weight
   deferral it deserves **one** documented comment at the module, not 61 undocumented repeats.
6. **Package-level logging asymmetry** — `Canvas/` 167 except / 0 log, `Research_Workspace/` 49 / 0, `Workspaces/`
   163 / 38. **The except:log ratio is a good detector for "errors vanish here"** and is worth a repo-wide census.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| The binding-metadata lost update interleaves in the shipped app | needs two screens driven concurrently; app must not be run | live drive: flip a folder binding to read-only in Settings while the Workspace Files modal commits an exclusion on the same binding, then read `metadata_json` — one edit will be absent |
| The double `workspace_memberships` scan is measurable on a real profile | no timing; no large fixture | seed 10,000 memberships and time `build_console_workspace_state` with and without the second read |
| The mermaid check would pass over a tampered `_v2` file **if the manifest is regenerated** | needs network or a pinned offline input dir | one-byte edit to `canvas_runtime_worker_v2.js` **plus** a `vendor_canvas_mermaid.py` regeneration, then run the checker — expect exit 0 |
| Whether any single column mixes `Z` and `+00:00` shapes | confirmed the three producers write to distinct tables; did not enumerate every column | `sqlite3 <profile>/workspaces.db "SELECT created_at FROM workspace_memberships UNION SELECT created_at FROM workspace_runtime_bindings" \| grep -c 'Z$'` |
| `CanvasControlBroker` unauthenticated-connection cap | `asyncio.start_server` (`control_protocol.py:454`) has **no concurrent-connection limit**, each connection holding a ~14 MiB reader high-water mark for the 2 s auth window. **Loopback-only and ADR-121 treats the local parent as semi-trusted, so not filed** | a loopback fuzz opening 1,000 sockets and measuring RSS |
| `Canvas/static/canvas_shell.js` (51 KB, runs in the browser **with the session cookie**) has no integrity check while `canvas_renderer.js` does | read only; it is in neither manifest, and the packaging test only asserts it is *packaged* | `python -c "…print(sorted(m['outputs']))"` on `runtime-manifest.json` → confirm `canvas_shell.js` is absent |
