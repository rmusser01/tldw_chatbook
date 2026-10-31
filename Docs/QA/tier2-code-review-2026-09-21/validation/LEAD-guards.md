# Lead verification — the three false-green guards (Phase A, re-done on origin/dev d0face3ebe)

These are the report's headline finding and they are repo-wide, so they belong to no single slice.
Re-verified by reproduction today, not carried over from the review baseline.

`./scripts/preflight.sh` is **GREEN** on this SHA. All three defects below are inside that green.

---

## G1 — `check_timestamp_writers.py` does not merely under-match; it whitelists the defect by name

- Verdict: **CONFIRMED, and worse than the report stated.**
- Site: `scripts/check_timestamp_writers.py:80-90` (predicate), `:20` (docstring), `scripts/timestamp_writer_census.tsv` (4 lines, all comments — the census is EMPTY).
- The check matches exactly two kinds: `datetime.utcnow()`, and `<...>.now().isoformat()` where `.now()` is
  naive. Its docstring then states, verbatim:
  `` (``datetime.now(timezone.utc).isoformat()`` does not match and is fine.) ``
- **ADR-173 lists that exact expression as one of the drifting shapes it exists to eliminate**
  (`173-utc-timestamp-storage-format.md:14`: "`datetime.now(timezone.utc).isoformat()` (microseconds, `+00:00`)"),
  and mandates "UTC with a `Z` suffix" (`:46`).
- Measured on this SHA, excluding `Third_Party/`:

  | shape | live sites | ADR-173 |
  |---|---:|---|
  | `now(timezone.utc).isoformat()` | **70** | non-conforming (`+00:00`) — **guard says "fine"** |
  | `now(UTC).isoformat()` | **31** | non-conforming (`+00:00`) — **guard says "fine"** |
  | `isoformat().replace("+00:00","Z")` | 14 | conforming |
  | `utc_now_iso()` (the shared helper) | 72 | conforming |
  | `datetime.utcnow()` | 0 | — guard is correct here |

  **CORRECTION (measured after the repair landed): the AST count is 99, not 101.** My 101 was grep-derived,
  which counts lines rather than call sites and picks up comments and strings. The repaired guard's own
  census pins **117 occurrences across 109 rows — 99 `offset_now_iso` + 18 `strftime_iso`**. The
  conclusion is unchanged and the shape of it is unchanged; only my number was loose.
  Before the repair the check printed `timestamp writers: 0 ... (0 pinned)` / `OK`.
- Reproduction of why the split is a data bug, not a style nit:
  ```
  $ .venv/bin/python -c "from datetime import datetime,timezone; t=datetime(2026,9,21,12,0,0,tzinfo=timezone.utc); a=t.isoformat(); b=a.replace('+00:00','Z'); print(a,b,a==b,a<b,a>=b)"
  2026-09-21T12:00:00+00:00 2026-09-21T12:00:00Z False True False
  ```
  `'+'` is 0x2B, `'Z'` is 0x5A, so for the same instant the `+00:00` row **always** sorts first and
  `WHERE ts >= '<...>Z'` **silently drops it**. That is precisely the failure `task-32803.3`
  ("Fix the media cleanup cutoffs that skip rows") names — and `task-32803.5`
  ("Adopt the shared timestamp helper across the remaining writers") is marked **Done**.
- Fix: widen the predicate to the ADR *contract* (emitted format), not naivety. Then re-pin the census
  with the 101 sites as baseline so it can only shrink. Re-open `32803.5`.

## G2 — `check_textual_worker_contract.py` W002: an `ast.Try` ancestor is treated as protection

- Verdict: **CONFIRMED. Measured 50 hidden sites, not the 59 the report claimed** — correcting my own number.
- Site: `scripts/check_textual_worker_contract.py:218-224`.
  ```python
  cursor = parents.get(node)
  while cursor is not None and cursor is not func:
      if isinstance(cursor, ast.Try):
          guarded = True          # <-- ANY Try ancestor counts
          break
  ```
  A `query_one` in a `finally:` or `except:` body is a **child of the `Try` node** but is **not** covered by
  that `Try`'s own handlers — an exception there propagates straight out. Only `Try.body` is protected.
- Reproduction, same file set / node types / filters as the script itself (`collect_w002` vs the same
  function with the one-line predicate corrected to require `_field(cur, child) == "body"`):
  ```
  current predicate  : 269 site(s)   <- matches preflight's printed "269 post-await DOM lookup(s)"
  corrected predicate: 319 site(s)
  HIDDEN by the bug  : 50 site(s) across 21 newly-visible function(s)
  ```
  The 269 matching preflight exactly is what makes this like-for-like.
- Newly visible includes `UI/Screens/scheduling/schedules_workbench.py::_run_sync` — which is
  independently Batch-1 item 5, found by slice S19 by reading. Two methods, one site: good corroboration.
- Fix: require the child to sit in `Try.body`, keep ascending otherwise (an *outer* try may still protect it).
  Re-pin the census at 319 as baseline, not as 50 new failures.

## G3 — `check_canvas_mermaid_assets.py` is circular

- Verdict: delegated to the S13 validator; folded in when it reports.
- preflight prints `Canvas Mermaid assets reproduce: 6 outputs` on this SHA.

---

## What this means for sequencing

G1 and G2 are **guard repairs that will make existing debt visible**. They must land as their own PR,
with the newly-visible sites **re-pinned as baseline**, BEFORE or INDEPENDENT of the PRs that fix the
underlying writers/lookups. If they land together, the diff stops being reviewable and the census
re-pin becomes indistinguishable from a regression.


---

# Post-repair verification (lead, independent of the implementing agent)

All three repairs landed on `fix/tier2-guards` as `f3bb728761`. I re-checked them rather than accepting the
report:

| check | result |
|---|---|
| `preflight.sh` | GREEN, and now says something true: `6 outputs (4 derived from pinned inputs, 2 vendored digests verified)`, `319 post-await DOM lookup(s)`, `117 pinned` |
| size ratchet | **exactly the baseline 5 failed / 58 passed** — unchanged |
| G1 ratchet actually bites | appended one `now(timezone.utc).isoformat()` to a scanned module -> `99 -> 100`, **exit 1**, message names the site. Restored clean. |
| G2 count | 319 in 157 functions — matches my own pre-repair measurement of 319 exactly |
| G3 pin catches tampering | appended `/*backdoor*/` to `canvas_runtime_worker_v2.js` -> `vendored asset integrity mismatch`, non-zero exit. Restored clean. |
| G3 pin is outside the laundering path | `vendor_canvas_mermaid.py` writes only to `output_dir`/`input_dir`, never `scripts/` — so re-running the documented reproducible command cannot regenerate the pin |

**My first negative-control attempt was invalid** and is worth recording: I injected the test writer into
`Utils/timestamps.py`, which the guard exempts *by design* (`:88`) because ADR-173 makes it the one sanctioned
producer. The guard correctly ignored it. A negative control aimed at an exempt path proves nothing — I
re-ran it against a scanned module, where it failed as it should.

## Two things the implementing agent found that the review did not

1. **The test suite asserted the bug.** `test_does_not_flag_aware_now_isoformat` existed to pin the
   whitelisting of `datetime.now(timezone.utc).isoformat()`. So the guard, its docstring, and its test all
   agreed with each other and all disagreed with ADR-173 — which is why this survived. Same pattern as the
   Confluence false-green test in the security stream. Worth generalising: **a green test is evidence about
   the test, not about the code, until someone checks what it asserts.**
2. **G3 was understated, not overstated.** A *naive* tamper of either self-certifying file was already caught
   as manifest drift. The real hole is the laundering path: `vendor_canvas_mermaid.py --output-dir` defaults
   to `STATIC`, so re-running the documented `reproducible_command` regenerates a self-consistent manifest
   **from the tampered bytes**. Reproduced end to end — the pre-repair checker returned green on an 85 KB
   browser runtime with `/*backdoor*/` appended.
