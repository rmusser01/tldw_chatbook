# Watchlists Suppress-Noise-Not-Changes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Watchlists change filter suppress noise instead of changes: threshold demoted to `0.0`, `ignore_selectors` surfaced and prefilled visibly, extraction-settings edits re-baseline instead of firing phantom items, and every silent check records why it was silent.

**Architecture:** A new `noise_defaults` module owns the default selector set and the extraction fingerprint. The DB gains a fingerprint column on `url_snapshots` plus a one-time data migration gated on that column's creation. `URLMonitor.check_url` returns `(item, disposition)` and compares fingerprints before hashes. The source create form and a narrow Inspector affordance expose the selectors; the Runs pane renders disposition counts.

**Tech Stack:** Python ≥3.11, SQLite (column-presence migrations, NO version bump), Textual ≥3.3.0, BeautifulSoup/soupsieve selectors.

**Spec:** `Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md` — approved. Do not redesign it.

## Global Constraints

- Worktree `/private/tmp/tldw-1361`, branch `fix/task-1361-snapshot-ordering` (carries the TASK-1361 tie-break; this plan lands TASK-1362 on top).
- `/private/tmp/tldw-1361/.venv/bin/python -m pytest` — never bare `pytest`, **never `-q`** (this repo's config suppresses FAILED summary lines under `-q`).
- **Never `git stash`** (stack shared across 100+ worktrees). **Never `git checkout -- <file>` to revert** — use an editor. **Never run ad-hoc scripts importing real `tldw_chatbook` config/DB modules** — they write on load; use pytest (`conftest.py` isolates `HOME`/`XDG_DATA_HOME`/`TLDW_CONFIG_PATH`). Never write to `~/.config/tldw_cli/` or `~/.local/share/tldw_cli/`.
- Selector semantics are **documented, not changed**: newlines separate rules; commas within a line are CSS groups (verified — `.ad, .timestamp` on one line strips both). No comma-splitting anywhere.
- Migration is the DB's column-presence idiom (`PRAGMA table_info` + `ALTER TABLE`), exactly like the block that added `content_kind` at `DB/Subscriptions_DB.py:562-567`. The `schema_version` table is pinned at 1 and never consulted — do not touch it.
- Every behavioural change carries a mutation check: revert the change with an editor, confirm the named test goes RED, restore.
- Mark new tests `pytest.mark.unit` (module-level `pytestmark`) or CI's `pytest -m unit` silently deselects them.
- Known baseline failures, NOT yours: 2 tree-chevron in `test_destination_visual_parity_correction.py`; an order-dependent `Select`/`Input` mount race that **moves between tests** in `test_watchlists_source_create_form.py` / `test_watchlists_source_frequency_control.py` (both pass in isolation); a collection `ImportError` in `Tests/UI/test_chat_shell_bar.py`.

## File structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Subscriptions/noise_defaults.py` | **Create.** Default selector set, its newline-joined text, the extraction fingerprint. |
| `tldw_chatbook/DB/Subscriptions_DB.py` | **Modify.** Fingerprint column + one-time data migration; column default `0.1` → `0.0`. |
| `tldw_chatbook/Subscriptions/monitoring_engine.py` | **Modify.** Fingerprint write/compare, threshold `0.0` + NULL coercion, `(item, disposition)` return. |
| `tldw_chatbook/Subscriptions/local_watchlists_service.py` | **Modify.** Unpack dispositions in the three URL arms; aggregate counts into `stats`. |
| `tldw_chatbook/Subscriptions/site_config_manager.py:90` | **Modify.** Fallback `0.1` → `0.0`. |
| `tldw_chatbook/UI/SiteConfigSettings.py:302` | **Modify.** Orphan screen input `"0.1"` → `"0.0"` (defaults-agreement only). |
| `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` | **Modify.** Create-form selector field, prefilled. |
| `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py` + screen | **Modify.** Narrow noise-selectors edit affordance for url-type sources. |
| `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py` | **Modify.** Disposition counts line in run detail. |
| `Tests/Subscriptions/test_watchlist_noise_not_volume.py` | **Create.** Tasks 1–4 tests (reuses `_site_source`/`_serve`/`_check` harness from `test_watchlist_content_kind_producer.py`). |

---

### Task 1: `noise_defaults` — the selector set and the fingerprint

**Files:**
- Create: `tldw_chatbook/Subscriptions/noise_defaults.py`
- Test: `Tests/Subscriptions/test_watchlist_noise_not_volume.py` (create; module-level `pytestmark = pytest.mark.unit`)

**Interfaces:**
- Produces: `DEFAULT_IGNORE_SELECTORS: tuple[str, ...]`, `default_ignore_selectors_text() -> str`, `extraction_fingerprint(ignore_selectors: str | None, extraction_method: str | None) -> str`. Tasks 2, 3, 5, 6 consume these names verbatim.

- [ ] **Step 1: Write the failing tests**

```python
"""TASK-1362: suppress noise, not changes.

Spec: Docs/superpowers/specs/2026-07-29-watchlists-noise-not-volume-design.md.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_default_selectors_strip_noise_but_not_cookie_recipes():
    """Every default line must do something; none may eat the payload.

    Proven during spec review: `[class*="cookie"]` matches
    `class="cookie-recipe-card"` and strips a cookie RECIPE, and
    `<input value=...>` never reaches `get_text()` at all. The default set
    was narrowed accordingly; this pins both properties.
    """
    from tldw_chatbook.Subscriptions.monitoring_engine import ContentExtractor
    from tldw_chatbook.Subscriptions.noise_defaults import DEFAULT_IGNORE_SELECTORS

    html = (
        '<div class="cookie-consent-banner">We use cookies</div>'
        '<div class="ad">BUY NOW</div>'
        '<span class="view-count">123 views</span>'
        '<span class="timestamp">12:01</span>'
        '<div class="cookie-recipe-card">Best cookie recipe</div>'
        '<time datetime="2026-07-29">Release date 2026-07-29</time>'
        "<p>real content</p>"
    )
    out = ContentExtractor.extract_text_from_html(
        html, list(DEFAULT_IGNORE_SELECTORS)
    )
    for noise in ("We use cookies", "BUY NOW", "123 views", "12:01"):
        assert noise not in out
    assert "Best cookie recipe" in out
    assert "Release date 2026-07-29" in out
    assert "real content" in out


def test_fingerprint_ignores_cosmetic_selector_edits():
    """Reordering, blank lines and duplicates must not re-baseline."""
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    a = extraction_fingerprint(".ad\n.timestamp\n\n.ad", "auto")
    b = extraction_fingerprint(".timestamp\n.ad", "auto")
    assert a == b


def test_fingerprint_changes_when_extraction_actually_changes():
    from tldw_chatbook.Subscriptions.noise_defaults import extraction_fingerprint

    base = extraction_fingerprint(".ad", "auto")
    assert extraction_fingerprint(".ad\n.sponsored", "auto") != base
    assert extraction_fingerprint(".ad", "raw") != base
    assert extraction_fingerprint(None, "auto") != base
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest Tests/Subscriptions/test_watchlist_noise_not_volume.py -v`
Expected: FAIL — `noise_defaults` does not exist.

- [ ] **Step 3: Implement**

```python
"""The default noise selectors and the extraction fingerprint (TASK-1362).

Selector semantics (verified, documented, NOT changed): newlines separate
independent rules; a comma within a line is a CSS selector group and matches
every branch. Splitting on commas would break `:is(.a, .b)` and
`[data-x="a,b"]`.

Two obvious-looking lines are deliberately absent from the default set.
CSRF/session-token inputs: `<input value=...>` contributes nothing to
`get_text()`, so a token selector strips nothing and only teaches users that
dead lines are normal. The broad `[class*="cookie"]`: it matches
`class="cookie-recipe-card"` and strips a cookie RECIPE -- substring
selectors are narrowed to consent-banner forms for exactly this reason.
Likewise `time[datetime]` is excluded: a release date lives in exactly that
element, and dates are often the payload being watched.
"""

from __future__ import annotations

import hashlib
import json

DEFAULT_IGNORE_SELECTORS: tuple[str, ...] = (
    '[class*="cookie-consent"], [class*="cookie-banner"], '
    '[id*="cookie-consent"], .cc-banner',
    '[class*="consent-manager"]',
    ".ad, .ads, .advertisement",
    '.sponsored, [class*="sponsored-"]',
    '.view-count, .views, [class*="viewcount"]',
    ".timestamp",
)


def default_ignore_selectors_text() -> str:
    """The default set as the newline-joined text a form field holds.

    Returns:
        One rule per line, in the documented order.
    """
    return "\n".join(DEFAULT_IGNORE_SELECTORS)


def extraction_fingerprint(
    ignore_selectors: str | None, extraction_method: str | None
) -> str:
    """A stable hash of the settings that shape extracted text.

    Snapshots store text extracted under the settings in force at capture
    time, so comparing a snapshot across a settings change is meaningless --
    the check must re-baseline instead (spec §3). Normalization: lines are
    stripped, empties dropped, duplicates removed and the result SORTED, so
    cosmetic reordering does not re-baseline.

    Args:
        ignore_selectors: The raw newline-separated selector text, or None.
        extraction_method: The subscription's extraction method, or None
            (normalized to "auto", the code's effective default).

    Returns:
        A hex digest; equal iff extraction behaviour is equal.
    """
    lines = sorted(
        {s.strip() for s in str(ignore_selectors or "").splitlines() if s.strip()}
    )
    payload = {"selectors": lines, "method": (extraction_method or "auto")}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
```

- [ ] **Step 4: Run to verify pass** — same command, expect 3 passed.

- [ ] **Step 5: Mutation checks.** (a) Drop `sorted(...)` → the cosmetic-edits test must go RED. (b) Re-add `input[name="csrf_token"]` to the set and add `<input name="csrf_token" value="TOK">` to the HTML — confirm the first test still passes *without* any assertion about the token, demonstrating why the line is dead weight (record the observation; then remove both again).

- [ ] **Step 6: Commit** — `feat(watchlists): default noise selectors and the extraction fingerprint`

---

### Task 2: DB — fingerprint column and the one-time data migration

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py` — `url_snapshots` DDL (`:262-273`), column default at `:210` (`change_threshold FLOAT DEFAULT 0.1` → `0.0`), and the column-presence migration region (the idiom at `:562-567`).
- Test: `Tests/Subscriptions/test_watchlist_noise_not_volume.py`

**Interfaces:**
- Consumes: `default_ignore_selectors_text` (deferred import inside the migration method, matching how `:1814` defers `item_persist`).
- Produces: `url_snapshots.extraction_fingerprint TEXT` column; migrated rows. Task 3 reads/writes the column.

The one-time gate IS the column check: run the data migration **only inside the branch that adds the column** — column absent means pre-migration DB. Both run in one transaction, so the write gates the marker (Phase D lesson: a migration that returns corrected values without writing them lasts exactly one load).

- [ ] **Step 1: Failing tests**

```python
def _fresh_db(tmp_path=None):
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB

    return SubscriptionsDB(":memory:", "test")


def test_migration_moves_thresholds_and_prefills_empty_selectors():
    """Existing url-family sources move to the new defaults, once.

    Non-empty selectors are preserved; feed sources are untouched; the
    migration must not re-run (its gate is the fingerprint column's absence).
    """
    from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB
    from tldw_chatbook.Subscriptions.noise_defaults import (
        default_ignore_selectors_text,
    )

    db = _fresh_db()
    with db.transaction() as conn:
        # Simulate a pre-migration database: drop the fingerprint column's
        # trace by recreating state the migration expects -- insert rows with
        # the OLD default threshold and empty/custom selectors.
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold)"
            " VALUES ('u1','url','https://a.test',0.1)"
        )
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold,"
            " ignore_selectors) VALUES"
            " ('u2','url','https://b.test',0.1,'.mine')"
        )
        conn.execute(
            "INSERT INTO subscriptions (name, type, source, change_threshold)"
            " VALUES ('f1','rss','https://c.test/feed',0.1)"
        )
        # Remove the column so the migration's gate sees a pre-migration DB.
        conn.execute("ALTER TABLE url_snapshots DROP COLUMN extraction_fingerprint")

    db2 = SubscriptionsDB.__new__(SubscriptionsDB)  # NOT this. See note below.
```

**Note to implementer:** the sketch above shows intent; the mechanics of re-running the migration on an open in-memory DB need the real initializer path — do it by calling the migration method directly (`db._run_column_migrations()`-equivalent, whatever the real name is at the idiom site) after the DROP, not by constructing a second instance (an in-memory DB is per-connection). Assert afterwards:

```python
    rows = {
        r["name"]: dict(r)
        for r in db.conn.execute(
            "SELECT name, change_threshold, ignore_selectors FROM subscriptions"
        ).fetchall()
    }
    assert rows["u1"]["change_threshold"] == 0.0
    assert rows["u1"]["ignore_selectors"] == default_ignore_selectors_text()
    assert rows["u2"]["ignore_selectors"] == ".mine"          # preserved
    assert rows["u2"]["change_threshold"] == 0.0              # still moved
    assert rows["f1"]["ignore_selectors"] in (None, "")       # feeds untouched
    # Idempotence: run the migration again; nothing changes (gate: column now
    # present). Set u1's selectors to '.custom', re-run, assert still '.custom'.
```

Also:

```python
def test_new_db_column_default_is_zero():
    db = _fresh_db()
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO subscriptions (name, type, source) VALUES"
            " ('n','url','https://n.test')"
        )
    row = db.conn.execute(
        "SELECT change_threshold FROM subscriptions WHERE name='n'"
    ).fetchone()
    assert row["change_threshold"] == 0.0


def test_url_snapshots_has_fingerprint_column():
    db = _fresh_db()
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(url_snapshots)")}
    assert "extraction_fingerprint" in cols
```

- [ ] **Step 2: verify FAIL** (no column, default still 0.1).
- [ ] **Step 3: Implement.** DDL gains `extraction_fingerprint TEXT` (nullable); `:210` default becomes `0.0`; migration block, in the idiom's style:

```python
        snapshot_cols = {
            row[1] for row in cursor.execute("PRAGMA table_info(url_snapshots)")
        }
        if "extraction_fingerprint" not in snapshot_cols:
            cursor.execute(
                "ALTER TABLE url_snapshots ADD COLUMN extraction_fingerprint TEXT"
            )
            # One-time TASK-1362 data migration, gated on the column-add so it
            # can never re-run: the ALTER is the marker, and both share this
            # transaction, so the write gates the marker (a migration that
            # returns corrected values without writing them lasts exactly one
            # load -- learned the hard way in Phase D).
            from ..Subscriptions.noise_defaults import default_ignore_selectors_text

            cursor.execute(
                "UPDATE subscriptions SET change_threshold = 0.0"
                " WHERE type IN ('url','url_list','sitemap')"
            )
            cursor.execute(
                "UPDATE subscriptions SET ignore_selectors = ?"
                " WHERE type IN ('url','url_list','sitemap')"
                "   AND (ignore_selectors IS NULL OR TRIM(ignore_selectors) = '')",
                (default_ignore_selectors_text(),),
            )
```

- [ ] **Step 4: verify PASS.**
- [ ] **Step 5: Mutation.** Remove the two `UPDATE`s (keep the ALTER) → the migration test must go RED. Restore with an editor.
- [ ] **Step 6: Commit** — `feat(watchlists): snapshot extraction fingerprint column + one-time threshold/selector migration`

---

### Task 3: `check_url` — fingerprint re-baseline, threshold 0.0, dispositions

**Files:**
- Modify: `tldw_chatbook/Subscriptions/monitoring_engine.py` — `check_url` (threshold read `threshold = subscription.get("change_threshold", 0.1)`, baseline SELECT with the TASK-1361 tie-break, `_store_snapshot(:1174)`).
- Modify: `tldw_chatbook/Subscriptions/local_watchlists_service.py:768-792` — the `url` / `url_list` / `sitemap` arms.
- Test: `Tests/Subscriptions/test_watchlist_noise_not_volume.py`

**Interfaces:**
- Consumes: `extraction_fingerprint` (Task 1), the fingerprint column (Task 2).
- Produces: `check_url(...) -> tuple[Optional[dict], dict]` where the disposition dict is `{"kind": "baseline_stored"|"unchanged"|"withheld_below_threshold"|"changed", "reason": str|None, "withheld_percentage": float|None}` (percentage scaled ×100 like the reader's). `_store_snapshot` gains `fingerprint: str | None = None`. `stats["dispositions"] = {"changed": n, "unchanged": n, "withheld": n, "baseline": n}` on URL-family runs. Tasks 7 consumes the counts; feeds/api arms are untouched (spec: dispositions are URL-only).

Ordering inside `check_url` (spec §3): compute `current_fp = extraction_fingerprint(subscription.get("ignore_selectors"), subscription.get("extraction_method"))` → SELECT now also reads `extraction_fingerprint` → **fingerprint mismatch (including stored NULL) ⇒ re-baseline before any hash comparison** → hash → threshold. Threshold read becomes NULL-safe:

```python
            raw_threshold = subscription.get("change_threshold")
            threshold = 0.0 if raw_threshold is None else float(raw_threshold)
```

- [ ] **Step 1: Failing tests** (reuse `_site_source`/`_serve`/`_check` from `test_watchlist_content_kind_producer.py` — import them):

```python
def test_a_small_edit_to_a_long_page_fires_under_the_default(monkeypatch):
    """TASK-1362 AC#1/#3: one sentence changed on a long page -> item.

    Under the old 0.1 default this exact sequence produced nothing (the edit
    moves whole-page similarity by well under 10%). The mutation check for
    this task restores the 0.1 fallback and this test must go RED.
    """
    # long page: ~40 identical sentences; after: one sentence's version bumped


def test_a_change_entirely_inside_ignored_noise_is_unchanged(monkeypatch):
    """A view-counter tick must not produce an item; disposition 'unchanged'."""
    # source created WITH default selectors text; pages differ only in
    # <span class="view-count">N</span>


def test_editing_selectors_rebaselines_instead_of_phantom_item(monkeypatch):
    """Spec §3: settings edit -> baseline_stored, no item, reason recorded.

    Without the fingerprint comparison this fires a phantom item whose diff
    is just the noise disappearing (the mutation check proves it).
    """
    # check once (baseline), update ignore_selectors via service.update_source,
    # check again with an UNCHANGED page -> no item; run stats dispositions
    # show baseline:1; third check -> unchanged, still no item


def test_withheld_carries_the_scaled_percentage(monkeypatch):
    """Raised threshold: disposition withheld_below_threshold, pct scaled ×100."""
    # change_threshold=0.5, small real edit -> no item;
    # stats dispositions withheld:1; the recorded percentage is >1.0 (scaled),
    # matching the reader's convention from TASK-1343


def test_null_threshold_does_not_typeerror(monkeypatch):
    """An explicit NULL column value must behave as 0.0, not raise."""
    # UPDATE subscriptions SET change_threshold=NULL; a real edit -> item


def test_url_list_aggregates_disposition_counts(monkeypatch):
    """Two URLs, one changed and one unchanged -> counts {changed:1, unchanged:1, ...}."""
```

Write real bodies — the harness makes each ~15 lines. Every assertion must name the disposition it expects; none may pass on a bare "no item".

- [ ] **Step 2: verify FAIL.**
- [ ] **Step 3: Implement** exactly the ordering above. The three service arms unpack the tuple; a tiny helper aggregates:

```python
def _disposition_counts(dispositions: list[dict]) -> dict[str, int]:
    counts = {"changed": 0, "unchanged": 0, "withheld": 0, "baseline": 0}
    keymap = {
        "changed": "changed",
        "unchanged": "unchanged",
        "withheld_below_threshold": "withheld",
        "baseline_stored": "baseline",
    }
    for d in dispositions:
        counts[keymap[str(d.get("kind"))]] += 1
    return counts
```

and each URL-family arm sets `result["stats"]["dispositions"] = _disposition_counts(...)`. `execute_run` already spreads `result["stats"]` into the recorded stats — verify, don't duplicate.

- [ ] **Step 4: verify PASS**, then the whole file plus `Tests/Subscriptions/test_watchlist_content_kind_producer.py` (its e2e tests call the changed arms — update the two `check_url` callers' expectations there if any test touched the old return shape).
- [ ] **Step 5: Mutations, each with an editor revert:** (a) restore `subscription.get("change_threshold", 0.1)` → small-edit test RED; (b) delete the fingerprint comparison → phantom-item test RED; (c) swap the fingerprint check to run *after* the hash check → the selector-edit test must still go RED (this is the ordering the spec mandates — if it stays green the test is not pinning the order).
- [ ] **Step 6: Commit** — `feat(watchlists): fingerprint re-baseline, 0.0 default, check dispositions`

---

### Task 4: defaults agreement — the two remaining `0.1` sites and the tripwire

**Files:**
- Modify: `tldw_chatbook/Subscriptions/site_config_manager.py:90` (`config.get("change_threshold", 0.1)` → `0.0`).
- Modify: `tldw_chatbook/UI/SiteConfigSettings.py:302` (`Input(value="0.1", ...)` → `"0.0"`); this screen is an orphan (nothing imports it) — change the value only, do not wire it.
- Test: `Tests/Subscriptions/test_watchlist_noise_not_volume.py`

- [ ] **Step 1: Failing test**

```python
def test_every_default_threshold_site_agrees_on_zero():
    """The default must not depend on which path created the source.

    Four sites can each impose a default; this pins all of them. The two
    source-text assertions are drift tripwires, honestly labelled: they pin
    the literal in the file, not behaviour, because one site is an orphan
    screen and the other is the DDL.
    """
    from pathlib import Path

    from tldw_chatbook.Subscriptions.site_config_manager import SiteConfigManager

    assert SiteConfigManager({}).change_threshold == 0.0

    root = Path(__file__).resolve().parents[2] / "tldw_chatbook"
    ddl = (root / "DB" / "Subscriptions_DB.py").read_text()
    assert "change_threshold FLOAT DEFAULT 0.0" in ddl
    orphan = (root / "UI" / "SiteConfigSettings.py").read_text()
    assert 'value="0.0", id="change-threshold"' in orphan
    engine = (root / "Subscriptions" / "monitoring_engine.py").read_text()
    assert 'subscription.get("change_threshold", 0.1)' not in engine
```

(Adjust the orphan-file needle to the actual argument order at `:300-303` before asserting; pin what is really there.)

- [ ] **Step 2-4:** FAIL → implement the two one-line edits → PASS.
- [ ] **Step 5: Commit** — `fix(watchlists): remaining change_threshold defaults to 0.0, with a drift tripwire`

---

### Task 5: the create form's visible noise field

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/sources_pane.py` — the create form (`Vertical(id="sources-create-form")`, `:237+`), the id list at `:115-122`, and the submit payload builder.
- Test: `Tests/Watchlists/test_watchlists_sources_pane.py` (extend) and run `Tests/UI/test_watchlists_source_create_form.py`.

**Interfaces:**
- Consumes: `default_ignore_selectors_text()` (Task 1). `create_source` already passes `ignore_selectors` through `_subscription_config_fields` (`local_watchlists_service.py:702`) — no service change.
- Produces: `TextArea(id="sources-create-ignore-selectors")` prefilled; the create payload carries its text verbatim.

**Layout risk, named:** this form sat at zero slack at 160×42 during Phase D (the CONTENT-region regression). The new field is a `TextArea` with `max-height: 4` in `_watchlists.tcss`. After wiring, run `Tests/UI/test_watchlists_source_create_form.py` — if the fits-tests fail at `height: 4`, reduce to 3; if they still fail, **STOP and report** rather than weakening any assertion (that is an adjudication, not a fix).

- [ ] **Step 1: Failing test** — create a source through the pane's real submit path; assert the created row's `ignore_selectors` equals `default_ignore_selectors_text()`; assert a user-cleared field creates with empty selectors (deliberate emptiness is honoured, not re-filled).
- [ ] **Step 2-4:** FAIL → implement (field label **"Ignore elements (CSS selectors — one rule per line; commas group)"**, help text stating the spam→add-selector loop) → PASS, including the create-form fits tests.
- [ ] **Step 5: Mutation:** stop passing the field into the payload → the created-row test must go RED.
- [ ] **Step 6: Commit** — `feat(watchlists): visible, prefilled noise-selector field on the source form`

---

### Task 6: the Inspector's narrow edit affordance

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/inspector_pane.py` (message + field + Save button for url-family sources), `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (handler → `controller.update_source`).
- Test: `Tests/Watchlists/` (extend the inspector tests).

**Why this exists:** there is **no source-edit UI at all** (verified — only alert rules have Edit; `update_source` exists in the service/controller unused by any form). Without an edit path the spec's core loop — spam item → diff names the churn → add one selector — requires deleting and recreating the source. This task adds the *narrowest* edit: one `TextArea` showing the selected url-family source's current `ignore_selectors`, one Save button, posting a message the screen routes to `update_source(item_id, {"ignore_selectors": text})`. No other field becomes editable; a full edit form stays out of scope.

- [ ] **Step 1: Failing test** — select a url source, edit the text, press Save through the real message path; assert the DB row changed. A second test: the affordance does not render for feed sources.
- [ ] **Step 2-4:** FAIL → implement → PASS. Follow the pane conventions (`RecomposeCaptureGuard`, message classes like `IngestRequested`); the Save handler must not `refresh=True`-recompose the screen (Task 5 of Phase D removed exactly that class of bug — patch what needs patching, nothing more). The fingerprint (Task 3) makes the *next check* re-baseline; assert that too: after Save, a check on an unchanged page yields disposition `baseline_stored`, not an item.
- [ ] **Step 5: Mutation:** route Save to a no-op → both the DB-row test and the re-baseline test must go RED.
- [ ] **Step 6: Commit** — `feat(watchlists): edit a source's noise selectors from the Inspector`

---

### Task 7: Runs pane disposition line

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/runs_pane.py:124-135` (`_stats_text`), and the run-row normalizer in `local_watchlists_service.py:1102` if the parsed `stats_json` does not already surface nested keys.
- Test: `Tests/Watchlists/test_watchlists_runs_pane.py` (extend; create if absent).

**Interfaces:** consumes `stats_json` → `dispositions` counts (Task 3's shape).

- [ ] **Step 1: Failing test** — `_stats_text({..., "dispositions": {"changed": 1, "unchanged": 3, "withheld": 0, "baseline": 1}})` contains `"1 changed"`, `"3 unchanged"`, `"1 baseline"`; a run dict *without* the key (feed runs) renders exactly today's text, no empty "Checks:" line. An e2e assertion in `Tests/Subscriptions/test_watchlist_noise_not_volume.py`: after a url run, the stored `stats_json` parses to counts matching the run.
- [ ] **Step 2-4:** FAIL → implement:

```python
        dispositions = run.get("dispositions") or {}
        base = (
            f"Status: {run.get('status', '-')}\n" ...  # existing lines unchanged
        )
        if dispositions:
            base += (
                f"\nChecks: {dispositions.get('changed', 0)} changed | "
                f"{dispositions.get('unchanged', 0)} unchanged | "
                f"{dispositions.get('withheld', 0)} withheld | "
                f"{dispositions.get('baseline', 0)} baseline"
            )
        return base
```

(plus the normalizer surfacing `dispositions` from parsed `stats_json` onto the run dict, mirroring how `found_count` gets there — read `:1102-1110` and match it).
- [ ] **Step 5: Mutation:** drop the normalizer surfacing → the e2e test must go RED while the unit `_stats_text` test stays green (they pin different layers; record both outcomes).
- [ ] **Step 6: Commit** — `feat(watchlists): run detail shows check dispositions`

---

### Task 8: close-out

**Files:** `backlog/tasks/task-1362 - ...md` (ACs, Implementation Notes, status Done), spec `Status: proposed` → `implemented`, and a fresh cross-worktree backlog-ID scan if any new task got filed mid-plan.

- [ ] Run the full affected sweep and record real numbers: `Tests/Subscriptions/ Tests/Scheduling/ Tests/Watchlists/ Tests/DB/ -k subscription` and `Tests/UI/ -k watchlist` (expect only the known movable baseline failures).
- [ ] Check every AC against evidence (the small-edit test satisfies AC#1/#3; the Runs line satisfies AC#2 — "the rule is stated in the UI"; also add one sentence to the form help naming the threshold's role, since AC#2 says UI *or* docs).
- [ ] Update TASK-1362 with notes (approach, the disproven comma bug, the narrowed defaults, the fingerprint) and mark Done. TASK-1361 is already Done on this branch.
- [ ] Commit — `docs: close out TASK-1362`.

## Self-review

**Spec coverage:** §1 → Tasks 3+4 (all four sites + NULL coercion + tripwire). §2 → Tasks 1+5 (+6 for the edit half of "create/edit"). §3 → Tasks 1+2+3 (fingerprint, ordering mutation (c) pins compare-before-hash). §4 → Tasks 3+7 (dispositions, counts, URL-only, feed runs unchanged). §5 → Task 2 (column-presence idiom, gate-is-the-ALTER, idempotence test). §6 → Task 5's help text names the loop. Testing section → mapped 1:1 across task Step 1s; "defaults agreement" → Task 4. Out-of-scope items untouched.

**Placeholders:** Task 3 Step 1 lists test names with docstrings and one-line body sketches rather than full bodies — deliberate: the bodies are ~15-line compositions of an existing harness the implementer must import (`_site_source`/`_serve`/`_check`), and the dispatch must say the stubs are skeletal by design and full bodies are required (the Phase D lesson: say it explicitly or a reviewer correctly flags `...` bodies). Task 2 Step 1 contains an explicit mechanical note where the sketch is intentionally not literal (in-memory DB re-migration mechanics). Everything else carries real code.

**Type consistency:** disposition dict keys (`kind`/`reason`/`withheld_percentage`) and counts keys (`changed`/`unchanged`/`withheld`/`baseline`) match across Tasks 3 and 7; `extraction_fingerprint(str|None, str|None) -> str` matches across 1/2/3; `default_ignore_selectors_text()` across 1/2/5.
