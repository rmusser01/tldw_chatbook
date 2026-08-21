---
id: TASK-19555
title: >-
  The metadata-only diagnostic guarantee stops at the file sink; the in-app log
  buffer and "Copy all" share path are unfiltered
status: To Do
assignee: []
created_date: '2026-08-21 20:05'
labels:
  - security
  - privacy
  - logging
priority: high
dependencies: []
---

## Description

Source: 2026-08-21 holistic review, Lane 2 (security & privacy) — its **Tier 2
#5**, and one of the two choke-point repairs the lane says covers the most
ground. CONFIRMED; re-verified at this branch base.

**Calibration first, because it matters for scoping:** ADR-029's metadata-only
guarantee is enforced by `PersistentDiagnosticFilter` attached at **exactly two
places, both the rotating FILE handler** (`Logging_Config.py:284,312`). The
file sink is airtight. **Nothing else is.**

The in-app path has no filter anywhere on it:

- `app.py:7836` — `self._log_buffer = deque()  # No maxlen - keep all logs`
- `app.py:7844-7854` — `PersistentLogHandler.emit` appends the formatted record
  straight into the buffer and into `_log_records`. The handler is level
  `NOTSET`, carries **no filter**, and is installed unconditionally.
- `UI/Logs_Window.py:271` — no sanitization on display
- `UI/Logs_Window.py:219 / 515` — a **"Copy all"** control whose
  `_on_copy_all` joins the entire session buffer to the **system clipboard**
  (OSC 52)
- `UI/Logs_Window.py:197-199` — the empty state **tells users to copy and
  share their logs**

So the app collects unfiltered diagnostics without bound, renders them, and
actively invites the user to put them on the clipboard.

The privacy test suite proves the gap by omission:
`test_remaining_diagnostic_sentinel_matrix.py:90-129` asserts only against the
**file** handler; the unfiltered collector is never asserted at all.

Scale, from the lane's AST scan: **872 leaky interpolations**; at INFO and
above, **59 user-content sites and 243 path sites**. Named live examples:
attachment paths, note **titles** on creation, keywords, the message body
prefix (`app.py:8999`, `event.text[:50]`), image prompts, **tool argument
values** (5 sites in `file_operation_tools.py` — a TASK-492 owner class), full
search queries and terms, provider response bodies, note filenames, and 75
ingestion path sites.

**Escalation:** three shipped documents tell users to set `log_level=DEBUG` for
troubleshooting, which admits roughly **800 more** sites including
`sql_logging.preview_params` — 80 characters of *every* SQL string parameter.

TASK-19552 (the Google CSE key at INFO) rides this exact path, which is why
that key does not merely land in a file.

Also worth recording while in this area: `Utils/log_sanitizer.py` has **exactly
one call site in the whole package** — the primitive exists and is essentially
unused. Its own normalization defect is filed separately (TASK-19558).

## Acceptance Criteria

- [ ] The metadata-only guarantee holds on the in-app path, not just the file
      sink — attaching the existing filter at `PersistentLogHandler.emit` is
      the preferred repair, since it is one choke point covering ~800 call
      sites (durable and pragmatic; per-call-site edits are neither)
- [ ] "Copy all" cannot place unredacted user content or credentials on the
      system clipboard
- [ ] `_log_buffer` is bounded — an unbounded session buffer is both a
      memory and a disclosure surface
- [ ] The privacy suite asserts against the **collector**, not only the file
      handler, so the omission that hid this cannot recur
- [ ] The assertion is mutation-checked: removing the filter makes the test
      red
- [ ] The empty-state copy that invites users to share logs is only shown once
      the shared artifact is actually safe to share
- [ ] The `log_level=DEBUG` troubleshooting advice in the three shipped docs is
      revisited in light of what DEBUG admits, and either made safe or made
      honest about what it exposes
