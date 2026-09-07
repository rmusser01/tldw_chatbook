---
id: TASK-19323
title: security_logger export_events writes unredacted user XML via a checker-invisible sink
status: Done
assignee: ['@claude']
created_date: '2026-08-20'
labels:
  - security
  - diagnostics
  - chunking
  - tooling
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from TASK-19191's independent review; re-verified at dev `a542fd463`.
`tldw_chatbook/Chunking/engine/security_logger.py:199-209` —
`export_events()` writes the whole `self._events` store to disk via plain
`open()` + `json.dump` (`:206-207`). Those events include `xml_sample` —
up to 500 characters of raw user XML captured at `:108` by
`log_xxe_attempt` — plus blocked-pattern details. Two problems compound:

1. **Unredacted export.** The dump ships user document content verbatim to
   an arbitrary caller-chosen file, outside every redaction guarantee the
   TASK-15103/15600 programme established (ADR-029).
2. **Topology blind spot.** `open` is not in
   `scripts/check_persistent_diagnostic_inventory.py`'s `SINK_CALL_NAMES`
   (`:49` — FileHandler variants, `addHandler`,
   `atomic_private_write_bytes`, `open_private_text_append*`), so this
   file-writing sink is invisible to the persistent-sink topology: the
   inventory records `security_logger.py`'s topology as only the
   `SecurityLogger.__init__` loguru `add` sink.

`export_events` has **zero callers** today (whole-tree grep at this base
finds only the def — no production callers, no tests), so the leak is
latent. But latent-plus-invisible is precisely the combination the sink
topology exists to prevent: the first future caller would ship an
unredacted user-content export no gate can see.

Fix shape — pick one repair AND close the blind spot, under the owner's
stability-over-quick-wins ruling (2026-08-11; durable over clever):

- **Repair**: either redact event details at export (drop/redact
  `xml_sample`; keep event type, severity, timestamp, and lengths/counts),
  or retire `export_events` outright as a zero-caller orphan using merged
  TASK-19043's retirement discipline (record any unique validation before
  deleting; there are no tests to retire).
- **Blind spot**: either teach the checker to see bare `open()`-based
  JSON/text dumps of event stores (scope it — a blanket `open` sink rule
  may be noisy), or pin this specific file's export shape so any
  reintroduction or change of a bare-`open()` export here trips the gate.

Knock-on: `security_logger.py` is a TASK-494 owner row (call_count 7,
digest `a872185128dda8da2366`) AND a `persistent_sink_topology` entry in
`Docs/security/production-diagnostic-inventory.json`; regenerate with only
the reviewed delta in the same PR and keep
`scripts/check_persistent_diagnostic_inventory.py` green (the step
TASK-19042/19043 initially missed; 2026-08-20 lesson in
`backlog/docs/lessons-testing-evidence.md`).

**Sub-bar observations recorded here (notes, not separate filings; give
each a disposition — opportunistic fix or explicit defer — do not drop
silently):**

- `tldw_chatbook/Chunking/engine/strategies/json_xml.py:776` and `:843` —
  `logger.error(f"Blocked potential XXE attack: {e}")`: a defusedxml
  exception's `{e}` can carry an entity name from a malicious document.
  Attack metadata rather than innocent user content, so below the repair
  bar, but trimming to `type(e).__name__` is cheap if the file is touched.
- `security_logger.py:209` logs the export destination path at INFO — a
  caller-chosen filesystem path in diagnostics; below-bar alone, and moot
  if `export_events` is retired.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 No code path can write raw user XML (`xml_sample`) or other user-content event details to disk unredacted: `export_events` either redacts details at export or is removed (BOTH: the export is removed AND the capture is trimmed at source — the store now keeps `xml_length`, never the XML)
- [x] #2 The sink-topology blind spot is closed: the inventory checker (or an explicit pin) turns red if `security_logger.py` gains or changes a bare-open() export of the event store (`Tests/Architecture/test_security_logger_write_surface.py`; mutation-proven)
- [x] #3 The persistent diagnostic inventory's `security_logger.py` owner row and persistent_sink_topology entry are regenerated with only the reviewed delta in the same PR and `scripts/check_persistent_diagnostic_inventory.py` passes (owner row 7→6; topology entry byte-identical, verified — the retired call was a diagnostic, not a declared sink)
- [x] #4 If retirement is chosen, TASK-19043's discipline is followed (unique-validation check recorded before deletion); both sub-bar observations receive an explicit disposition in the implementation notes
<!-- AC:END -->

## Implementation Plan

1. Worktree off origin/dev (`0f5cba2f7`); pin venv/PYTHONPATH/cwd with the
   `tldw_chatbook.__file__` assert; baseline the checker, the architecture
   gate, and produce the rebuild-vs-committed inventory diff (dev drift is
   expected per 19191's controller caveat — review any drifted rows per-file
   before absorbing them).
2. Read `security_logger.py` end-to-end and census every consumer of the
   event store's `details` (get_events / export_events / the loguru format).
   Decide redact-at-export vs retire under the stability-over-quick-wins
   ruling; if nothing consumes `details` but the dead export, trim capture at
   the source as well (metadata-only store).
3. Born-red evidence first: a write-surface/retirement/store-content pin
   suite in `Tests/Architecture/` shown failing at base in the honest
   direction, plus a one-off probe demonstrating the actual unredacted
   export at base.
4. Implement the repair; guard suite green; mutation evidence for the new
   guard (Edit-reintroduce a bare-open export, watch it go red,
   Edit-restore).
5. Close the topology blind spot with the simplest mechanism that would have
   caught THIS bug: a module-scoped write-surface pin rather than a global
   `open` sink rule (which would flood the topology).
6. Reconcile the vendoring contract: `security_logger.py` is a vendored
   engine file (`VENDOR_MANIFEST.toml`, spec §5.2), so the repair must ship
   through the sync script's sanctioned deterministic-patch mechanism, not a
   hand edit — extend `Helper_Scripts/sync_chunking_engine.py` with an
   engine-patch table (loud anchors on upstream drift), verify byte-identity
   of patched output vs the shipped file, and keep
   `Tests/Chunking/test_sync_script.py` green.
7. Inventory: per-row review of the full delta, `--write`, JSON diff shows
   only reviewed rows, checker exit 0, invariants probed; restamp the two
   summarization-fixture boundary hashes via the test module's own helpers
   under an isolated HOME.
8. Disposition both sub-bar observations explicitly. Gates: checker exit 0,
   architecture inventory + new guard, summarization privacy suite,
   `Tests/Chunking/`, repo-wide `--collect-only -q`; commit.

## Implementation Notes

Retired the unredacted export **and** trimmed the capture at its source, so
the chunking security-event store is metadata-only by construction; closed
the topology blind spot with a module-scoped write-surface pin. Base:
origin/dev `0f5cba2f7`, branch `task/19323-burn`.

### Part 1 — decision: RETIRE the export AND stop storing the XML

Both halves, not one. The census that settled it: `details` has exactly
three readers in the whole tree — `log_event` (which puts it in `_events`),
`get_events` (in-memory filter, **zero callers**), and the retired
`export_events` (**zero callers**). It never reaches the loguru sink: the
sink's format is `{time}|{level}|{extra[event_type]}|{message}`, and `extra`
carries only `security` + `event_type` (19191's dormant-sink judgment,
re-verified). So nothing consumed `xml_sample` except a dead export.

Alternatives weighed:

- *Redact at export, keep the capture.* Rejected. It leaves ≤500 chars of
  raw user XML sitting in a process-lifetime in-memory list reachable by
  `get_events()`, and makes the safety of the data depend on every future
  reader remembering to redact. That is the "clever but fragile" side of the
  owner's 2026-08-11 stability ruling: one new caller re-opens the hole.
- *Retire the export only.* Rejected as half a fix, for the same reason —
  the liability is the retained content, not just the one writer.
- *Keep the export, gate it behind a flag.* Rejected: speculative surface
  with zero callers; TASK-19043's retirement discipline applies.

The durable fix removes the data itself: `log_xxe_attempt` now records
`xml_length` (an int) instead of `xml_sample`. Detection value is preserved
— event type, severity, timestamp, `source`, `blocked_patterns`, and now the
size — while the reconstructable payload is gone. `log_redos_attempt`'s
`pattern` was left as-is deliberately: a ReDoS pattern is attacker-supplied
*regex*, i.e. attack metadata (same class as the `{e}` ruling below), and
trimming it would cost the only forensic handle on which pattern was
blocked. It is also no longer exportable now that the writer is gone.

**TASK-19043 unique-validation check before deleting** (recorded as required):
whole-tree grep for `export_events` at the base found the definition and
nothing else — no production caller, no test, no dynamic-dispatch shape
(no string literal, no `getattr`, no `action_`/concat fragment). It carried
no validation of its own (no path checks, no redaction) — it was a bare
`open()`+`json.dump`. Nothing unique died with it, so there was nothing to
re-point.

### Part 2 — mechanism: a module-scoped write-surface pin

`Tests/Architecture/test_security_logger_write_surface.py` (4 tests). The
load-bearing one walks the module's AST and asserts it contains **zero**
write-capable calls (`open`, `json.dump`, `write*`, `copy*`, `mkstemp`, …).
No exemption machinery is needed because the module's one legitimate sink is
loguru's `add`, which is not a write-capable *call name* — so the assertion
is a clean "none", and a second test cross-checks the checker's own census
still reports exactly one `loguru_sink` in `SecurityLogger.__init__`.

Scope of that guard, stated precisely (review finding): it is a fixed
name-list AST scan, so its coverage is that list plus `SINK_CALL_NAMES` —
not literally every way to write a file. A write routed through an
arbitrarily-named helper defined in another module escapes it (the reviewer
demonstrated one such escape, alongside two shapes it does catch). Closing
that would need dataflow analysis, which was rejected as the too-clever
option; the reason it is acceptable is the other half of this fix: the event
store is now metadata-only, so a missed export leaks nothing.

Why this over the alternatives in the brief: adding bare `open` to
`SINK_CALL_NAMES` was explicitly ruled out (it would flood the topology with
every file write in the repo); the "flag `json.dump` whose first arg derives
from an event-store attribute" rule is the too-clever option — it would
need dataflow analysis and would still miss `f.write(json.dumps(...))`. A
plain architecture test scoped to the one module that owns a security-event
store is the simplest thing that would have caught THIS bug and catches its
recurrence, and it needed **no change to the checker script** — so no
reclassification risk for any other file (the checker is untouched; the
whole-inventory diff below is code-driven only).

**Mutation evidence.** Re-introduced a bare-open export via Edit — and
deliberately under a *different* name (`dump_events`, not `export_events`),
to prove the pin catches the shape rather than the identifier:

```
AssertionError: security_logger.py gained write-capable call(s) invisible to the
persistent-sink topology; declare and review them instead:
open(output_file, 'w'); json.dump(self._events, f, indent=2, default=str)
1 failed, 3 passed
```

Edit-restored; suite back to 4 passed; `git diff --stat` confirmed the
restore left only the intended change.

### Vendoring: the repair ships through ENGINE_PATCHES, not a hand edit

**The surprise of this task.** `security_logger.py` is a *vendored* file
(`VENDOR_MANIFEST.toml`, spec §5.2: "vendored files are never hand-edited"),
and `Tests/Chunking/test_sync_script.py::test_sync_idempotent_and_rejects_
local_edits` caught my edit exactly as designed:
`FATAL: local modification to vendored file security_logger.py`. The gate
worked; the repair had to change lane.

The spec's sanctioned remedies (shim/subclass, or upstream-then-re-sync)
both fail for a *privacy* repair: a shim cannot unship leaky code — the
bytes still ship, and a vanilla `configure_security_logging()` would
resurrect them. So the fix follows the mechanism the sync script already
uses for ported tests (`TEST_PATCHES`): a new `ENGINE_PATCHES` table applies
the reviewed repair deterministically during sync. The sync stays
idempotent, an *unsanctioned* hand edit still fails loudly, and upstream
drift under a patch anchor dies with a FATAL (probe-verified: feeding the
patcher unrelated text exits with "sync patch anchor not found"). The
canonical vendored state is now upstream-at-pin + rewrite + ENGINE_PATCHES,
and the local-modification check compares against exactly that.

Byte-identity verified: `patch_vendored_file("security_logger.py",
rewrite_imports(<base file>))` == the shipped file, sha256 `dab73f74cbea6cb4`
both sides. `VENDOR_MANIFEST.toml` gains a `[patches]` row recording the
entry so it is upstreamed and dropped at the next re-pin.

**Merge note for the controller:** sibling task-19321 built this same
mechanism independently on its own branch (same `ENGINE_PATCHES` dict, same
integration points). I renamed my helper to its `patch_vendored_file` and
matched its structure, so the two branches should merge as a dict/table
union (entries are disjoint: `chunker.py` there, `security_logger.py` here)
rather than a structural conflict. The `[patches]` manifest tables will
still need their comment blocks reconciled by hand.

### Sub-bar observations — rulings

1. **`strategies/json_xml.py:776/:843` `logger.error(f"Blocked potential
   XXE attack: {e}")` — DEFER, no change.** Ruled acceptable on the merits,
   not just on cost. A defusedxml `EntitiesForbidden`/`DTDForbidden` message
   carries the *entity or DOCTYPE name the attacker chose* — attacker-
   supplied attack metadata, not the victim's prose — which is precisely
   what a security log exists to record, and it is the only handle on which
   construct was blocked. It is also bounded (an entity name, not the
   document). Two further reasons not to touch it here: the file is
   **vendored**, so a cosmetic trim would cost a second ENGINE_PATCHES entry
   and permanent divergence from upstream for no privacy gain; and the
   controller's boundary reserves sibling-owned engine files. If it is ever
   revisited, `type(e).__name__` is the one-line change.
2. **`security_logger.py:209` INFO-logs the export destination path —
   MOOT, resolved by deletion.** It died with `export_events` (it was the
   `logger.info(f"Exported {len(self._events)} security events to
   {output_file}")` line). This is the whole of the owner-row delta below;
   no caller-chosen filesystem path is logged by this module any more.

### Inventory + fixture arithmetic

Checker was **already red at the base** (`0f5cba2f7`, exit 1) from unrelated
dev drift — 19191's documented controller caveat. The full rebuild-vs-
committed diff and per-file review:

- `Chunking/engine/security_logger.py` 7→6, digest `a872185128dda8da2366`
  → `97b2820afc325424cb3e` — **mine**. Call-level diff: exactly one removed
  call, the export's `logger.info(f"Exported … to {output_file}")`. No calls
  added.
- `UI/Console_Modules/workspace.py` 29→30 — **not mine**, pre-existing dev
  drift. Added call diffed at the source-segment level: one constant-message
  `logger.opt(exception=True).debug("Unable to read active workspace during
  Console resume reconcile")` (task-18310's resume reconcile). No fields; safe.
- `UI/Screens/chat_screen.py` 156→157 — **not mine**, same landing: one
  constant-message `logger.opt(exception=True).debug("Unable to reconcile
  Console session with registry-active workspace")`. No fields; safe.

Both drift rows are metadata-only constant messages, so they are absorbed
rather than challenged (per-file review, per 19191's convention).

`persistent_sink_topology`: **unchanged**, deliberately verified rather than
assumed — the file's sink list is byte-identical old vs new (one
`loguru_sink`, digest `7e3c515ca8d99167`, scope `SecurityLogger.__init__`),
because the thing I removed was a *diagnostic* and a bare `open` the
topology never saw. That non-delta is the whole point of the task: the
export was invisible to this census, which is why the pin exists.

Summary: owner_files 517 (unchanged), task_492_calls 1210 (unchanged),
task_494_calls 7128→**7129** (−1 mine, +2 dev drift), persistent_sink_files
7 (unchanged). JSON diff is 7 lines added / 7 removed — the three row pairs
plus the one summary count, nothing else. Invariants probed and printed:
`len(owners)==owner_files`, per-owner sums == summary, `len(topology)==
persistent_sink_files`. Checker exit 0: "517 owners, 1210 TASK-492 calls,
7129 TASK-494 calls, 7 sink files".

**No checker-script change**, so no cross-file reclassification was possible.

Fixture: `Tests/fixtures/summarization_diagnostic_review.json` re-stamped by
the prescribed flow (importlib-load the privacy module as `privmod`,
`owned_paths = set(MODULE_COUNTS)`, `_canonical_sha256(
_normalized_inventory_projection(inv, owned_paths))`) under an **isolated
HOME** — needed for real: an earlier bare probe was observed loading the
live `~/.config/tldw_cli/config.toml`. `c463bf02f9396087…` →
`b3a7ee068f8aeecb73dfa4fcf8b194f558fca3f4296e162374f2d977cd7dab70`,
asserted to appear exactly 2× before replacing and 2× after (old hash
absent). The two summarization owner rows themselves were untouched, which
is the boundary the fixture protects.

### Gates (exact, read from output files)

- `scripts/check_persistent_diagnostic_inventory.py`: exit **0** (red at base).
- `Tests/Architecture/test_persistent_diagnostic_inventory.py` +
  `test_security_logger_write_surface.py`: **69 passed** (65 inventory + 4 new).
- `Tests/LLM_Calls/test_summarization_diagnostic_privacy.py`: **257 passed**.
- `Tests/Chunking/`: **425 passed, 2 failed, 32 skipped, 1 xfailed** (base
  was 424 passed / 3 failed — my +1 is the sync test flipping green). Both
  remaining reds proven pre-existing at pristine base `0f5cba2f7` in a
  pinned-SHA probe worktree: `test_chunker_v2.py::test_process_text_
  tokenizer_override` and `test_golden_parity.py::test_golden_parity[
  tokens-cjk]`, both needing an HF gpt2 tokenizer this machine cannot fetch
  ("couldn't connect to huggingface.co … not in the cached files").
- `Tests/Chunking/test_sync_script.py`: **4 passed, 0 skipped** (verbose-
  confirmed by name, including `test_sync_idempotent_and_rejects_local_edits`),
  run against a local clone of tldw_server at the pin.
- Repo-wide `pytest --collect-only -q`: **52300 collected**, exit 0.
- `ruff check` on the three touched Python files: clean except two
  **pre-existing** findings on `sync_chunking_engine.py:7`
  (`import argparse, hashlib, …` E401/F401), verified identical at base and
  left alone.

### Files changed

- `tldw_chatbook/Chunking/engine/security_logger.py` — metadata-only XXE
  capture (`xml_length`), `export_events` retired, privacy contract in the
  module docstring.
- `Helper_Scripts/sync_chunking_engine.py` — `ENGINE_PATCHES` +
  `patch_vendored_file`, wired into the local-modification check and the
  copy step.
- `tldw_chatbook/Chunking/engine/VENDOR_MANIFEST.toml` — `[patches]` row.
- `Tests/Architecture/test_security_logger_write_surface.py` — new pin (4).
- `Docs/security/production-diagnostic-inventory.json` — regenerated.
- `Tests/fixtures/summarization_diagnostic_review.json` — two boundary hashes.
- This task file.

### Lesson candidate (for the controller to place)

A privacy repair to a **vendored** file has no sanctioned lane in the
§5.2 remedy list: a shim cannot unship leaky bytes. The deterministic
sync-patch table is the lane, and the vendoring gate — not review — is what
forces you into it. Worth recording alongside 19321's identical discovery,
once, rather than twice.
