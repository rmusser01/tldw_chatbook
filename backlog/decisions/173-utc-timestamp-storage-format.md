# ADR-173: Canonical UTC timestamp storage format

Status: Accepted
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
`'T'` 0x54). The three shipped bugs (TASK-32803.2/.3/.4) were all **SQL-level**
comparisons of one writer's `T`-separated `.isoformat()` value against a column
written by SQLite's space-separated `CURRENT_TIMESTAMP` — e.g. flashcard
`next_review` (`ChaChaNotes_DB.py:21570`) compared at `get_due_flashcards`/
`count_due_flashcards` against `DEFAULT CURRENT_TIMESTAMP` columns. Note where
that leaves the fix: a comparison inside a SQL `WHERE`/`ORDER BY` cannot call a
Python helper, and the repo already has the working answer for mixed shapes —
`console_trace_settlement.recover_open_calls` compares three shapes correctly by
normalising both sides through SQLite's `julianday()`. So a canonical **write**
format is necessary but not, by itself, sufficient: it makes *future* data
uniform and Python-side comparisons uniform, but it does not retroactively fix a
column that still holds legacy or `CURRENT_TIMESTAMP` values. This ADR records
the write shape as the package contract (so the remaining ~50 writers adopt one
helper, TASK-32803.5, under a guard) **and** the comparison rule that goes with
it.

## Decision

The canonical stored UTC timestamp shape is **millisecond-precision ISO-8601 in
UTC with a `Z` suffix**:

```
YYYY-MM-DDTHH:MM:SS.mmmZ        e.g. 2026-09-20T12:34:56.789Z
```

Properties that make it the right choice:

- **Fixed width (24 chars** for years 1000–9999**).** Every canonical value has
  the same length, `T` separator, and `Z` suffix, so lexical (`TEXT`) ordering
  equals chronological ordering **for a column all of whose values are canonical**
  — see the guarantee scope below. This is the property whose absence caused the
  three shipped bugs.
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
- `parse_utc(value) -> datetime` — reads every **stored/compared** shape and
  returns an aware UTC datetime (scope and the display-shape exclusion are
  spelled out below).
- `is_canonical_utc(value) -> bool` — True iff `value` is exactly the canonical
  shape; the format guard (AC#2) asserts writes against this.

**Reading is tolerant; writing is canonical.** New and migrated writers call
`utc_now_iso`/`to_utc_iso`. `parse_utc` reads every shape currently **stored or
compared** — the ISO-8601 variants (`T`- or space-separated; second/millisecond/
microsecond precision; `Z` or `+00:00` offset; date-only) and SQLite
`CURRENT_TIMESTAMP` — and returns an aware UTC datetime (a value with no offset is
interpreted as UTC). It does **not** parse human-display `strftime` shapes
(`"September 20, 2026"`, `"HH:MM"`, epoch integers); those are never stored or
compared, and `parse_utc` raises `ValueError` on them, so callers reading a field
that might hold one must handle that. Existing stored values are not rewritten.

**Guarantee scope — read this before relying on lexical ordering.** Free
`TEXT`-order == time-order holds for a column **only if every value in it is
canonical**: it is written *exclusively* through the helper, has *no* legacy
rows, and has *no* `CURRENT_TIMESTAMP` default. A column that mixes shapes still
mis-sorts — an earlier canonical `2026-09-20T11:59:59.000Z` sorts *after* a later
`CURRENT_TIMESTAMP` `2026-09-20 12:00:00` (`'T'` > `' '`), which is the original
bug. So a comparison over any column that may hold pre-existing or
`CURRENT_TIMESTAMP` values **must normalise both sides**: in Python via
`parse_utc`, and **in SQL via `julianday(col)` / `datetime(col)`** (the
`console_trace_settlement.recover_open_calls` precedent) — never a bare `col < ?`.
Adopting the canonical writer on such a column does not make a bare SQL comparison
safe until the column is also migrated (below).

**Naive values are interpreted as UTC.** That is the assumption every existing
lexical comparison already made. Writers that produced naive *local* time via
`datetime.now()` (no `timezone.utc`) are latent bugs; a format guard
(TASK-32803.1 AC#2/#3, to follow this ADR's ratification) will flag
timestamp-bearing writes that do not go through the shared helper.

SQLite's `CURRENT_TIMESTAMP` column defaults are **not** changed by this ADR
(a per-table schema-migration question — 31 tables default to it). Until a given
column's default is changed *and* its legacy rows backfilled, that column holds
mixed shapes and falls squarely under the guarantee-scope rule above: read it
with `parse_utc`, compare it in SQL with `julianday()`/`datetime()`, and do not
assume adopting the canonical writer alone fixed its ordering.

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
- The guarantee is scoped, not global: adopting the writer gives free lexical
  ordering only on all-canonical columns. A column with legacy rows or a
  `CURRENT_TIMESTAMP` default stays mixed until it is separately migrated
  (backfill existing rows to canonical + change the default) or its comparisons
  are normalised in SQL (`julianday()`/`datetime()`). The three shipped fixes
  took the SQL-normalisation path; new all-canonical columns get ordering free.
- No stored data is rewritten and no column type changes; the change is additive
  and backward-compatible via the tolerant parser.
