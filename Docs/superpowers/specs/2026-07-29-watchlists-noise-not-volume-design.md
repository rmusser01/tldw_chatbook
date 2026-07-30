# Watchlists: suppress noise, not changes — design

**Tasks:** TASK-1362 (threshold), with TASK-1361 (snapshot tie-break, already implemented on this branch) landing together.
**Date:** 2026-07-29
**Status:** implemented

## Problem

`change_threshold` defaults to `0.1` and is compared against whole-page character similarity
(`SequenceMatcher` ratio). A small but meaningful edit to a long page — a price, a version number,
one new paragraph — moves that ratio by far less than 0.1, so no item is produced and nothing is
said. Because a below-threshold check does **not** store a new snapshot, drift accumulates against
the last significant baseline: small edits are eventually reported late and lumped together, or (for
a page that changes once and then stays still) never.

Silence is also ambiguous. A check that produces nothing looks identical whether it was a first
check, an unchanged page, or a change that was withheld — the user cannot distinguish "nothing
happened" from "something happened and the app decided not to mention it". This is the same failure
class as the watchlists that never checked at all (TASK-1210).

Meanwhile the control the user actually wants exists and works: `ignore_selectors` is a real DB
column, read by `_fetch_url_content`, applied by the extractor **before** any comparison — named
noise can never trigger a change. But it is reachable from no live UI. The Watchlists source form
exposes Name, URL, Tags; the only screen offering `ignore_selectors` or `change_threshold` is
`UI/SiteConfigSettings.py`, which nothing imports.

**Decided intent (user):** the filter should suppress *noise*, not *changes*. If something a human
wrote changed, report it, however small. The threshold is a volume filter being used as a noise
filter, and no tuning fixes a category error.

## Design

### 1. The threshold is demoted, not removed

Default `change_threshold` becomes **`0.0`** everywhere a default exists:

| Site | Today | Becomes |
|---|---|---|
| `DB/Subscriptions_DB.py:210` column default | `0.1` | `0.0` |
| `monitoring_engine.py:1026` fallback | `0.1` | `0.0` |
| `site_config_manager.py:90` fallback | `0.1` | `0.0` |
| `UI/SiteConfigSettings.py:302` input value (orphan screen) | `"0.1"` | `"0.0"` |

The identical-hash check already short-circuits unchanged pages before the threshold is consulted,
so `0.0` means: any real difference in extracted text — i.e. after noise stripping — produces an
item. The column and comparison stay: a user who wants a volume filter may still raise the value
per source, and the disposition record (§4) tells them what it is withholding. The read must
coerce an explicit NULL: `subscription.get("change_threshold", 0.0)` returns `None` when the column
holds NULL — the key exists — and `pct < None` is a `TypeError` inside a scheduled fetch.

The `[subscriptions.templates.*]` presets in `config.py` (e.g. `0.05` for "Documentation Monitor")
are explicit user-facing template choices, not defaults, and are left alone.

### 2. `ignore_selectors` becomes the visible noise control

The Watchlists source create/edit form gains a multi-line **"Ignore elements (CSS selectors)"**
field, prefilled for new sources with the default set below. Prefilled means *written into the
source's own `ignore_selectors`*, visible and deletable line by line — nothing is ever stripped
invisibly, and future changes to the default set deliberately do not propagate to existing sources
(they own their copy). The prefill lives in the **form**, not in `create_source`: a source created
programmatically gets no silent defaults, because invisible stripping is the thing this design
forbids.

Default set (conservative; each line carries a short trailing comment in the form's placeholder
help, not in the stored value):

```
[class*="cookie-consent"], [class*="cookie-banner"], [id*="cookie-consent"], .cc-banner
[class*="consent-manager"]
.ad, .ads, .advertisement  — ad slots (comma = CSS group, one rule)
.sponsored, [class*="sponsored-"]
.view-count, .views, [class*="viewcount"]
.timestamp               — explicit timestamp classes only
```

Two lines that look obvious are deliberately absent, both verified empirically. CSRF/session
token inputs: `<input value=…>` contributes nothing to extracted text (`get_text()` ignores
attribute values), so a token selector strips nothing and would only teach users that dead lines
are normal. And the broad `[class*="cookie"]`: it matches `class="cookie-recipe-card"` and strips
"Best cookie recipe" — substring selectors are narrowed to consent-banner forms for exactly this
reason.

Deliberately excluded: `time[datetime]` and anything date-*like* beyond the explicit `.timestamp`
class — a release date lives in exactly those elements, and dates are often the payload being
watched.

**Selector semantics, documented rather than changed** (verified empirically): newlines separate
independent rules; a comma *within* a line is a CSS selector group and matches every branch —
`.ad, .timestamp` on one line strips both. There is no format bug to fix, and splitting on commas
would break legitimate selectors (`:is(.a, .b)`, `[data-x="a,b"]`). The form's help text states
both facts.

### 3. Extraction fingerprint: settings changes re-baseline instead of diffing

**Pre-existing bug this design would otherwise amplify:** snapshots store text extracted under the
selectors in force at capture time. Editing `ignore_selectors` (or `extraction_method`) changes
extraction, so the next check's hash differs and a phantom "change" fires whose diff is just the
noise disappearing. Today nobody edits selectors (no UI); after §2 everybody will.

Each snapshot therefore stores an **extraction fingerprint**: a stable hash of the selector list
— normalized by stripping, dropping empties, deduplicating and **sorting**, so cosmetic reordering
does not re-baseline — plus `extraction_method`. The fingerprint comparison runs **before** the
hash comparison: the stored hash was computed under the old settings, so comparing it across a
settings change is meaningless. On mismatch the check **re-baselines** — stores a fresh snapshot, produces no item, and records
disposition `baseline_stored` with reason `extraction_settings_changed` (§4).

Existing snapshots have no fingerprint; absence is treated as a mismatch. This makes the migration
(§5) self-healing: every pre-migration source re-baselines exactly once, silently and visibly at
once — no phantom items, and the Runs pane says why. The cost is honest and bounded: one diff
window is lost per settings change.

### 4. Dispositions: silence stops being ambiguous

`URLMonitor.check_url` currently returns `change_info | None`, and `None` means four different
things. It now returns a disposition alongside any item:

- `baseline_stored` (first check, or fingerprint mismatch — with a `reason`)
- `unchanged` (hash match)
- `withheld_below_threshold` (with the computed percentage, scaled for display like the reader's)
- `changed` (item produced)

Multi-URL sources (`url_list`, `sitemap`) aggregate **counts** per run:
`{"changed": n, "unchanged": n, "withheld": n, "baseline": n}`. The counts are recorded in the
run's existing `stats_json` and rendered as one line in the Runs pane's run detail. Dispositions
apply to **URL checks only**: feeds deduplicate per item and have no baseline, so their runs keep
today's semantics. With the new `0.0`
default `withheld` should be rare; its visibility exists for anyone who raises a threshold, and as
the tell-tale that noise selectors are stripping too much (everything `unchanged` while the page
visibly moves).

### 5. Migration

There are no current Watchlists users (confirmed), so existing rows migrate to the new world rather
than preserving behaviour:

- `change_threshold` → `0.0` for all existing sources.
- Empty `ignore_selectors` → prefilled with the default set. Non-empty values are left untouched.
- The snapshot fingerprint column is added via this DB's actual migration idiom — a
  `PRAGMA table_info` column-presence check with `ALTER TABLE` (the same pattern that added
  `content_kind`). The `schema_version` table exists but is pinned at 1 and never consulted; do
  not invent a versioning scheme around it.

**Corrected (implementation, TASK-1362 Task 2 review):** this design originally asserted the
column-presence `ALTER` and the two data-migration `UPDATE`s "share one transaction," on the theory
that the write would gate the marker structurally. That was false as written. Python's `sqlite3`
module (default isolation, no override anywhere in `BaseDB`/`connect_private_sqlite`) opens an
implicit transaction only before DML (`INSERT`/`UPDATE`/`DELETE`/`REPLACE`), never before DDL, so a
bare `ALTER TABLE` autocommits immediately regardless of what transaction the caller believes it is
in. Proven with a probe: an exception raised between the `ALTER` and the second `UPDATE` left the
fingerprint column present — the one-time gate durably spent — with `change_threshold` moved but
`ignore_selectors` permanently `NULL`, and unrepairable, since a clean re-run sees the column and
skips entirely. What actually shipped instead is an **explicit `BEGIN IMMEDIATE`** transaction
wrapping the `ALTER` and both `UPDATE`s, committed together on success and rolled back together on
any exception (`DB/Subscriptions_DB.py:609-633`), restoring the atomicity the gate depends on.
SQLite's DDL is itself fully transactional; only the `sqlite3` module's implicit-BEGIN policy is
not, and that distinction is what the original wording missed.

Phase D's migration lessons apply mechanically: the corrected values must actually be **written**
(not just returned), the write's success must gate any marker, and `save`-style helpers that signal
failure by returning `False` must have that `False` propagated. The migration is idempotent and its
test performs two loads. Schema-version renumbering must be re-checked at merge — this repo has had
five migration-number collisions.

### 6. The trade-off, named

With `0.0`, a page with churn the default selectors cannot name (view counters with unusual markup,
"3 min read", relative dates, A/B-tested copy) fires an item on **every check**. The failure mode
flips from silence to spam — deliberately, because spam is visible and diagnosable while silence is
neither: the item's diff (TASK-1343) shows exactly which text churned, which names the selector to
add in the now-visible field. That loop — spam → diff names the churn → one selector added — is the
intended workflow, and the form's help text states it.

## Residual visibility gaps (final review, recorded not assumed)

The §3/§1 promise is discharged at *source* granularity: the run row names the source and start
time, the detail line splits `baseline` from `re-baselined (settings changed)` and shows the
largest withheld percentage, and the Save toast warns about the lost window. Still not visible,
deliberately left as scope: which *URL* of a multi-URL source re-baselined (per-URL reasons are
aggregated away and never persisted); the exact bounds of a lost window (the prior run's timestamp
must be read from the table); per-URL withheld percentages (only the max survives); and nothing is
proactive — the user must open the Runs pane. A migration-caused re-baseline also reads "settings
changed" although the migration, not the user, changed them — literally true, potentially
surprising.

Out of scope, filed as follow-up if wanted: automatically flagging a source whose every check
produces an item ("this source looks noisy — inspect its last diff").

## Testing

- **Small-edit-fires**: a one-sentence edit to a long page produces an item under the default
  configuration (TASK-1362 AC#1/#3) — driven through the real producer, and failing under the old
  `0.1` default.
- **Noise-suppression**: a change entirely within a default-ignored element produces `unchanged`,
  not an item.
- **Fingerprint**: editing selectors between checks re-baselines (no item, correct disposition +
  reason); an unchanged fingerprint diffs normally. Mutation: remove the fingerprint comparison —
  the phantom-item test must go red.
- **Dispositions**: each of the four dispositions is recorded where a run can show it;
  `withheld_below_threshold` carries the percentage; multi-URL counts sum correctly.
- **Migration durability**: two loads on one store; thresholds written to `0.0`; empty selectors
  prefilled; non-empty preserved; pre-migration snapshots re-baseline once.
- **Defaults agreement**: a test asserts the DB column default, both code fallbacks, and the orphan
  screen's value agree — so the default cannot drift by path again.
- All selector parsing behaviour pinned as documented (newline rules, comma groups) — semantics
  tests, not parser changes.

## Out of scope

- `baseline_manager.py` adoption or deletion — TASK-1360.
- Appeared/disappeared alert scoping — TASK-1363.
- Auto-flagging noisy sources (possible follow-up).
- The `[subscriptions.templates.*]` preset values.
