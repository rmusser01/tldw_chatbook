# TASK-285 Phase 2 — defer tldw_api schema submodule imports off the app-import path

## Goal

Phase 1 (PR #670) made `tldw_api/__init__.py` a lazy PEP 562 re-export layer and deferred every
on-app-path `TLDWAPIClient` import, but honestly reported that `import tldw_chatbook.app` was
*unchanged* (~1.47s) because ~47 schema submodules still loaded eagerly: `Server*Service` /
`Local*Service` modules named their request/response schema types at *module scope*, and the
lazy `__getattr__` layer faithfully served those lookups. Phase 2's job: make the app import load
(near-)zero `tldw_api` schema submodules.

## Method

Not a one-shot static inventory — an iterative measure/fix/re-measure loop, because a
getattr-callsite trace and a plain grep both have blind spots a bare `sys.modules` diff doesn't:

1. **Round 1 (original inventory):** monkeypatch `tldw_api.__getattr__` to record callers,
   `import tldw_chatbook.app` in a scratch-HOME subprocess. Found 48 loaded submodules / 43
   importer files (40 mechanically convertible + 3 hand-fixed edge cases). Wrote an AST classifier
   (`classify_lib.py`) that, per file, buckets every imported schema name into: annotation-only/
   unused (→ `TYPE_CHECKING`), runtime-constructed-in-function (→ function-local import, grouped
   per enclosing `def`), or module/class-scope (→ STOP, allowlist). Wrote a generator
   (`apply_transform.py`) that applies the mechanical cases via precise line-based source surgery
   (not a full-file regenerate — preserves everything except the touched import statements) and
   verified every output re-parses before writing. 48 → 14 loaded submodules.
2. **Round 2:** PEP 562 `__getattr__` caches each name into the package's own `globals()` on first
   resolution, so a second file importing an already-resolved name never re-triggers the recording
   hook — invisible to a getattr trace even though its own module-scope import still forces the
   load on a cold process. Re-running the `sys.modules`-diff inventory (ground truth, caching-order
   independent) surfaced 5 more on-path files. 14 → 3.
3. **Round 3:** a repo-wide grep for `from ..tldw_api.SOMETHING import` (bypassing the package
   entirely) found 12 files importing directly from a schema submodule — something neither the lazy
   package layer nor a getattr trace could ever see. 3 → 2 (only `sync_schemas` from one leftover
   caller).
4. **Round 4:** that leftover caller led to the Sync_Interop domain-adapter cluster: 13 more files
   all importing `SyncV2Envelope` at module scope. Fixed all 13. **Final: 2 loaded submodules**
   (`tldw_api` itself + the one deliberately-allowlisted `kanban_schemas`).

## The one deliberate exception (allowlisted, not forced)

`Kanban_Interop/server_kanban_service.py` keeps a module-scope `KANBAN_OPERATION_SPECS` dict whose
values embed real schema *classes* (`KanbanOperationSpec(..., KanbanBoardCreate, 0)`) for a
30-operation runtime-dispatch table (`__getattr__`-based dynamic method resolution). This is
exactly the "isinstance registry / class-attribute default" case the task said to stop on rather
than force — restructuring it to store class names and `getattr()`-resolve them lazily is a real
architectural change to a dispatch table, not a mechanical import-site move. Left untouched.
`Kanban_Interop/local_kanban_service.py` (which imports the registry) was still converted for its
own separate schema imports, but doesn't change the outcome.

Also left alone (confirmed off-path — zero importers anywhere outside their own package, matching
phase 1's precedent): `Outputs/server_outputs_service.py`, `Sharing/server_sharing_service.py`,
`WebClipper/server_web_clipper_service.py`.

## Annotation-safety edge cases (hand-fixed, not mechanical)

- **`app.py`**: no `from __future__ import annotations` (too large/central to blanket-convert).
  `MCPUnifiedClient` is a *nested* function's return annotation, evaluated eagerly at
  `TldwCli.__init__` time — quoted as a forward-ref string, function-local import added.
- **`Event_Handlers/tldw_api_events.py`**: also lacked future-annotations, and 9/16 names are
  real module-level `def ... -> ProcessVideoRequest:` return annotations. Added the future-import
  (same fix phase 1 used for `Chatbooks/server_chatbook_service.py`); the other 7 (exception
  classes, worker-callback response classes) got function-local imports.
- **`Sync_Interop/envelope_builder.py`, `MCP/server_unified_service.py`,
  `Sync_Interop/domain_adapters/chat.py`**: no existing `if TYPE_CHECKING:` block to merge into —
  created new ones.
- **`Skills_Interop/skill_trust_service.py`**: aliased import
  (`_normalize_skill_name as _normalize_api_skill_name`) — hand-edited (single call site).

## Measurement

**App-import wall time** (interleaved A/B, 6 rounds, scratch HOME/XDG_CONFIG_HOME/TLDW_CONFIG_PATH
each round, `subprocess.run([python, "-c", "import tldw_chatbook.app"])` wall-clock timing).
Baseline = detached worktree at `origin/dev`@21d9e1c0 (current dev, post phase-1 merge + everything
else merged since). Round 0 was a shared cold-start outlier in *both* arms (3,402ms / 1,712ms vs a
steady ~2,100/~1,515ms in rounds 1-5) and is excluded from the headline (including it, the delta is
larger: -776ms mean).

| | baseline (origin/dev) | after (this tree) |
|---|---|---|
| mean (rounds 1-5) | 2,107.5 ms | 1,514.3 ms |
| median (rounds 1-5) | 2,083.0 ms | 1,515.1 ms |

**Delta: -593.3 ms mean / -567.9 ms median (≈ -28.2%)** — in line with the original performance
audit's ~469ms attribution for this cost center. Corroborated by an environment-noise-independent
metric: 48 → 2 loaded `tldw_api.*` submodules.

**`TldwCli()` construction probe** (local mode, no `tldw_server` configured): succeeds in ~160ms;
`sys.modules` diff immediately around the constructor call shows **zero** newly-loaded
`tldw_api.*` submodules — confirms the `from_config`/`LegacyConfigServerClientProvider` fallbacks
(every `Server*Service` constructed with `client=None`) never touch a schema class.

## Tests

New file `Tests/Utils/test_tldw_api_schema_deferral.py` (existing test files are read-only for this
task): a fresh-subprocess check that `import tldw_chatbook.app` only loads `tldw_api.*` submodules
within `{tldw_api, tldw_api.kanban_schemas}`, plus 3 functional smokes — a converted
`Server*Service` constructed with `client=None` that still builds its deferred schema object before
failing with the same pre-existing `ValueError`, the fully-local counterpart succeeding end-to-end
with no client at all, and the allowlisted Kanban module's dynamic-dispatch path still working.

## Files changed

69 source files across `tldw_chatbook/` (5 commits, grouped by discovery round — see git log) +
1 new test file. No existing test file modified (`git diff origin/dev --diff-filter=M --name-only
-- Tests/` is empty).

## Verification status

- `Tests/tldw_api/`: 402 passed
- `Tests/Utils/`: 192 passed (incl. 4 new)
- `Tests/Chat/`: 905 passed, 69 skipped
- Full `Tests/` `--collect-only`: 10,742 tests, zero collection errors
- `python -m py_compile` clean on all 69 changed files
- Broad sweep (32 test dirs matching every touched subsystem — Sync_Interop, Sync_Tests, Skills,
  Kanban, Feedback, Research_Interop, Research, RuntimePolicy, Chatbooks, Character_Chat, MCP,
  MCP_Governance, Audio_Services, Auth_Account, Claims, Companion, Evaluations_Interop,
  External_Connectors, Meetings, Notes, Notifications, Prompt_Management, Prompt_Studio, RAG_Admin,
  Study_Interop, Text2SQL_Interop, Tools_Interop, Translation, Voice_Assistant, Writing_Interop,
  Outputs, Sharing; run foreground in 3 groups): **2,013 passed, 2 skipped, 0 failed**
  (824+1s / 662 / 527+1s)
