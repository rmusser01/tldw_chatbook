---
id: TASK-19569
title: >-
  Guards that no longer guard — an inert concurrency test, five stale MCP
  monkeypatches, and a severed cause chain
status: To Do
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

- [ ] `Client_Media_DB_v2.py:775` preserves the cause chain (`from error`), so
      a `PrivatePathError` remains identifiable to the privacy contract; the 6
      media reds go green for the right reason, not by relaxing the assertion
- [ ] `test_tool_catalog_concurrency` is repaired in the **test** (its own
      `list_catalog()` call moved before the counter is installed) and left
      able to detect a real regression — verified by mutating the production
      snapshot behaviour and seeing it go red
- [ ] The five MCP watchlists tests are re-pointed at the current seam
      (`runtime_source_loader` / `load_default_runtime_source_state`) and pass
- [ ] `monkeypatch.setattr(..., raising=False)` is removed wherever it is
      masking a renamed or removed attribute in these files — a patch that
      installs a never-read attribute must fail loudly
- [ ] Each repaired guard is mutation-checked: it goes red when the behaviour
      it protects is broken. A green test that cannot fail is the defect being
      fixed here
- [ ] The watchlists local-server tool contract is demonstrably guarded again
