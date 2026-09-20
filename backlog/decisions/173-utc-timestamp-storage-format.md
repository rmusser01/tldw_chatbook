# ADR-173: Canonical UTC timestamp storage format

Status: Proposed
Date: 2026-09-20
Task: TASK-32803.1
Related: TASK-32803 (time work stream)

## Context

The core-runtime review (2026-09-17) found the package has no single contract
for how a timestamp is written or compared. In the wild:

- **~12 distinct UTC string shapes** are produced by ~55 helper copies:
  `datetime.now(timezone.utc).isoformat()` (microseconds, `+00:00`),
  `.isoformat().replace("+00:00","Z")` (microseconds, `Z`),
  `.isoformat(timespec="milliseconds")…Z`, a range of `strftime` shapes
  (`%Y-%m-%d %H:%M:%S`, `%Y-%m-%d %H:%M`, `%Y-%m-%d`), and the deprecated
  `datetime.utcnow()`.
- A **13th shape** comes from SQLite's `CURRENT_TIMESTAMP` default (space
  separator, second precision, no offset: `2026-09-20 12:34:56`), which 31
  tables default to.
- **~100 sites compare these lexically** (SQLite `TEXT` comparison, or Python
  string `<`); only two normalise first.

Because SQLite compares `TEXT` byte-by-byte, a space-separated cutoff sorts
*before* a `T`-separated stored value on the same calendar date (`' '` 0x20 <
`'T'` 0x54). Three user-visible bugs followed from exactly this and were fixed
in TASK-32803.2/.3/.4 (flashcards not coming due until the next day; media
cleanup skipping rows for up to 24h; a marks column receiving two shapes). Those
fixes each converged, independently, on the **same** written shape. This ADR
records that shape as the package contract so the remaining ~50 writers can adopt
one helper (TASK-32803.5) and a guard can enforce it.

## Decision

The canonical stored UTC timestamp shape is **millisecond-precision ISO-8601 in
UTC with a `Z` suffix**:

```
YYYY-MM-DDTHH:MM:SS.mmmZ        e.g. 2026-09-20T12:34:56.789Z
```

Properties that make it the right choice:

- **Fixed width (24 chars).** Every value has the same length, `T` separator,
  and `Z` suffix, so lexical (`TEXT`) ordering equals chronological ordering.
  This is the property whose absence caused the three shipped bugs.
- **Round-trips through the stdlib.** `datetime.fromisoformat` (Python ≥3.11,
  and this project is ≥3.12) both parses it and, given an aware UTC datetime,
  `isoformat(timespec="milliseconds").replace("+00:00","Z")` produces it.
- **Millisecond, not microsecond, precision.** Milliseconds are enough
  resolution for every ordering the app needs, and a fixed 3-digit fraction
  keeps the width constant (a variable microsecond tail does not sort safely
  against a value that happens to have trailing zeros trimmed).

One shared module, `tldw_chatbook/Utils/timestamps.py`, is the only sanctioned
producer and reader:

- `utc_now() -> datetime` — current time, timezone-aware UTC.
- `utc_now_iso() -> str` / `to_utc_iso(dt) -> str` — the canonical shape. A naive
  datetime is interpreted as UTC; an aware one is converted to UTC first.
- `parse_utc(value) -> datetime` — reads **every shape currently on disk**
  (canonical, `+00:00` ISO, `T`- or space-separated, second/millisecond/
  microsecond precision, date-only, and SQLite `CURRENT_TIMESTAMP`) and returns
  an aware UTC datetime; a value with no offset is interpreted as UTC.

**Reading is tolerant; writing is canonical.** New and migrated writers call
`utc_now_iso`/`to_utc_iso`; comparisons that cannot be made lexically safe call
`parse_utc` on both sides. Existing stored values are not rewritten — `parse_utc`
already reads them.

**Naive values are interpreted as UTC.** That is the assumption every existing
lexical comparison already made. Writers that produced naive *local* time via
`datetime.now()` (no `timezone.utc`) are latent bugs; a format guard
(TASK-32803.1 AC#2/#3, to follow this ADR's ratification) will flag
timestamp-bearing writes that do not go through the shared helper.

SQLite's `CURRENT_TIMESTAMP` column defaults are **not** changed by this ADR
(that is a schema-migration question per table); `parse_utc` reads that shape,
and tables whose ordering matters should write the canonical shape from the app
rather than rely on the default.

## Alternatives

- **Microsecond precision (`…%f`, 27 chars).** Also fixed-width if never
  trimmed, but Python's `isoformat()` omits the fraction entirely at whole
  seconds, producing a 19/20-char value that sorts before a fractional one —
  the same class of bug. Milliseconds via explicit `timespec` are always
  present.
- **`+00:00` offset instead of `Z`.** Equivalent semantically and fixed-width,
  but `Z` is shorter, is what the three bug-fixes already emit, and reads
  identically; choosing one avoids a fourteenth shape.
- **Unix epoch integers/reals.** Sorts correctly and is compact, but breaks
  every existing `TEXT` column, human-readability of stored rows, and the ~100
  existing comparison sites; a package-wide column-type migration is far more
  invasive than adopting one string helper.
- **Leave per-module helpers, fix only the comparisons.** The three shipped bugs
  show the divergence recurs whenever a new writer is added; without one home
  and a guard, the next writer reintroduces a shape.

## Consequences

- One helper to import; the ~50 remaining writers adopt it in TASK-32803.5.
- A guard (AC#2/#3) can assert timestamp-bearing writes route through the helper,
  in preflight and the derived-artifacts CI job, so a new divergent shape fails
  before merge.
- No stored data is rewritten and no column type changes; the change is additive
  and backward-compatible via the tolerant parser.
