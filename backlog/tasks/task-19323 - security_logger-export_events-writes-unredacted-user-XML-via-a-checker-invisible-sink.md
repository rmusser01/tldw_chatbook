---
id: TASK-19323
title: security_logger export_events writes unredacted user XML via a checker-invisible sink
status: To Do
assignee: []
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
- [ ] #1 No code path can write raw user XML (`xml_sample`) or other user-content event details to disk unredacted: `export_events` either redacts details at export or is removed
- [ ] #2 The sink-topology blind spot is closed: the inventory checker (or an explicit pin) turns red if `security_logger.py` gains or changes a bare-open() export of the event store
- [ ] #3 The persistent diagnostic inventory's `security_logger.py` owner row and persistent_sink_topology entry are regenerated with only the reviewed delta in the same PR and `scripts/check_persistent_diagnostic_inventory.py` passes
- [ ] #4 If retirement is chosen, TASK-19043's discipline is followed (unique-validation check recorded before deletion); both sub-bar observations receive an explicit disposition in the implementation notes
<!-- AC:END -->
