"""Snippet editor and import (PR 3a, Task 5).

Selecting a dataset renders its snippets in the detail pane: a table with
per-row character counts, a highlighted whitespace marker for anomalous
runs, and an exact-duplicate flag (after whitespace normalization). Import
accepts plain text (one snippet per line), CSV (``text`` + optional
``group``), and JSON (round-tripping an export).

**Whitespace validation is this editor's headline feature.** ``"foo"`` and
``"foo "`` measure entirely different next-token distributions -- a user
comparing two snippets where one carries a stray space would read a large
divergence as a finding about the model rather than an editing accident.
The marker must mean something wherever it appears, so a clean snippet must
render **no** marker at all (see
``test_normal_snippet_renders_no_whitespace_marker``).

**Only exact duplicates (after whitespace normalization) are flagged.**
Minimal pairs differing by one word are the instrument the bench measures
with; flagging them would train users to ignore the warning strip where the
whitespace warning also lives (see
``test_minimal_pair_snippets_are_not_flagged_as_duplicates`` -- the
requirement most likely to be implemented backwards).

Mirrors ``test_evals_bench_editor.py``'s harness (bundled CSS, a fake
``app_instance`` exposing ``evaluation_orchestrator.db``) rather than
inventing a second one -- see that file's own module docstring for why the
real stylesheet is loaded, and ``test_evals_screen.py`` for the original
region-based "genuinely rendered, not merely present" lesson this whole
harness shape exists to satisfy.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from textual.app import App

import tldw_chatbook
from tldw_chatbook.DB.Evals_DB import EvalsDB
from tldw_chatbook.Evaluations_Interop.evaluation_normalizers import (
    RESERVED_LOCAL_DATASET_SAMPLES_KEY,
)
from tldw_chatbook.UI.Evals import snippet_editor as snippet_editor_module
from tldw_chatbook.UI.Evals.snippet_editor import (
    SnippetEditor,
    count_warnings,
    find_exact_duplicate_labels,
    import_snippets_into_dataset,
    normalize_snippet_whitespace,
    parse_csv_snippets,
    parse_json_snippets,
    parse_plain_text_snippets,
    render_snippet_cell,
    snippet_whitespace_flag_label,
    whitespace_warning_kinds,
)
from tldw_chatbook.UI.Screens.evals_screen import EvalsScreen

_BUNDLED_CSS_PATH = str(
    Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"
)


class _FakeOrchestrator:
    def __init__(self, db: EvalsDB) -> None:
        self.db = db


class _FakeAppInstance:
    def __init__(self, db: EvalsDB) -> None:
        self.evaluation_orchestrator = _FakeOrchestrator(db)
        self.notifications: list[tuple[str, str]] = []

    def notify(self, message: str, *, severity: str = "information", **kwargs) -> None:
        self.notifications.append((message, severity))


class EvalsHarness(App):
    CSS_PATH = _BUNDLED_CSS_PATH

    def __init__(self, app_instance: _FakeAppInstance) -> None:
        super().__init__()
        self.app_instance = app_instance

    async def on_mount(self) -> None:
        await self.push_screen(EvalsScreen(self.app_instance))


@pytest.fixture
def evals_db() -> EvalsDB:
    return EvalsDB(db_path=":memory:", client_id="test")


@pytest.fixture
def evals_app(evals_db: EvalsDB) -> EvalsHarness:
    return EvalsHarness(_FakeAppInstance(evals_db))


def _make_dataset(
    db: EvalsDB, name: str, snippets: list[dict], **extra_metadata
) -> str:
    metadata = {
        RESERVED_LOCAL_DATASET_SAMPLES_KEY: snippets,
        "sample_count": len(snippets),
        "inline_samples": True,
        **extra_metadata,
    }
    return db.create_dataset(
        name=name, format="custom", source_path=f"inline:{name}", metadata=metadata
    )


def _snip(text: str, *, group: str | None = None, note: str | None = None) -> dict:
    return {"id": str(uuid.uuid4()), "text": text, "group": group, "note": note}


def _row_text(screen, snippet_id: str) -> str:
    widget = screen.query_one(f"#evals-snippet-text-{snippet_id}")
    text = widget.renderable
    return text.plain if hasattr(text, "plain") else str(text)


def _row_meta(screen, snippet_id: str) -> str:
    widget = screen.query_one(f"#evals-snippet-meta-{snippet_id}")
    text = widget.renderable
    return text.plain if hasattr(text, "plain") else str(text)


# ---------------------------------------------------------------------------
# Pure-logic tests: whitespace detection, marker rendering, duplicates
# ---------------------------------------------------------------------------


def test_normal_text_has_no_whitespace_warning_kinds():
    assert whitespace_warning_kinds("The protestors were") == frozenset()
    assert snippet_whitespace_flag_label("The protestors were") is None


def test_leading_whitespace_is_detected():
    assert "leading" in whitespace_warning_kinds(" The protestors were")


def test_trailing_whitespace_is_detected():
    assert "trailing" in whitespace_warning_kinds("The government said ")


def test_interior_whitespace_run_is_detected():
    assert "interior" in whitespace_warning_kinds("The government  said")


def test_single_interior_space_is_not_anomalous():
    """A single space between words is the normal, expected case -- only
    RUNS of 2+ whitespace characters are anomalous mid-string (leading and
    trailing are anomalous at any length, per the design mockup's single
    trailing space example)."""
    assert whitespace_warning_kinds("The protestors were mid sentence") == frozenset()


def test_render_snippet_cell_marks_only_the_anomalous_run():
    rendered = render_snippet_cell("The government said ")
    # The trailing space character itself is REPLACED by the marker glyph
    # (not appended after a literal space) -- character-for-character
    # fidelity with the raw text, one glyph per anomalous character.
    assert rendered.plain == "The government said␣"
    # A style span actually covers the marker glyph -- not merely a glyph
    # swap with no styling applied.
    assert any(span.style for span in rendered.spans)


def test_render_snippet_cell_of_clean_text_carries_no_marker_glyph():
    rendered = render_snippet_cell("The protestors were")
    assert "␣" not in rendered.plain
    assert rendered.plain == "The protestors were"
    assert not rendered.spans


def test_minimal_pair_snippets_are_not_flagged_as_duplicates_pure():
    """The requirement most likely to be implemented backwards: a minimal
    pair differing by one loaded word IS the instrument, not an editing
    mistake."""
    snippets = [_snip("The protestors were"), _snip("The rioters were")]
    assert find_exact_duplicate_labels(snippets) == {}
    assert count_warnings(snippets) == 0


def test_exact_duplicate_after_whitespace_normalization_is_flagged_pure():
    a = _snip("The government said ")  # trailing space
    b = _snip("The government said")  # no trailing space
    labels = find_exact_duplicate_labels([a, b])
    assert a["id"] not in labels
    assert labels[b["id"]] == "exact dup of 1"


def test_normalize_snippet_whitespace_collapses_runs_and_strips_ends():
    assert normalize_snippet_whitespace("  The  government   said ") == (
        "The government said"
    )


# ---------------------------------------------------------------------------
# Import parsers (pure)
# ---------------------------------------------------------------------------


def test_parse_plain_text_snippets_assigns_uuids_and_skips_blank_lines():
    content = "line one\nline two \n\nline three\n"
    snippets = parse_plain_text_snippets(content)
    assert [s["text"] for s in snippets] == ["line one", "line two ", "line three"]
    ids = [s["id"] for s in snippets]
    assert len(set(ids)) == len(ids)
    for snippet_id in ids:
        uuid.UUID(snippet_id)  # does not raise
    assert all(s["group"] is None for s in snippets)


def test_parse_csv_snippets_reads_text_and_optional_group():
    content = "text,group\nThe protestors were,neutral\nThe rioters were,loaded\n"
    snippets = parse_csv_snippets(content)
    assert [s["text"] for s in snippets] == ["The protestors were", "The rioters were"]
    assert [s["group"] for s in snippets] == ["neutral", "loaded"]
    for snippet in snippets:
        uuid.UUID(snippet["id"])


def test_parse_csv_snippets_without_text_column_raises():
    with pytest.raises(ValueError):
        parse_csv_snippets("foo,bar\n1,2\n")


def test_parse_json_snippets_preserves_existing_id_and_mints_one_when_absent():
    fixed_id = str(uuid.uuid4())
    content = json.dumps(
        [
            {"id": fixed_id, "text": "The protestors were", "group": "neutral"},
            {"text": "The rioters were", "group": "loaded"},
        ]
    )
    snippets = parse_json_snippets(content)
    assert snippets[0]["id"] == fixed_id
    uuid.UUID(snippets[1]["id"])
    assert snippets[1]["id"] != fixed_id
    assert [s["group"] for s in snippets] == ["neutral", "loaded"]


def test_parse_json_snippets_rejects_non_list_non_object_payload():
    with pytest.raises(ValueError):
        parse_json_snippets("42")


def test_parse_json_snippets_replaces_an_illegal_id_with_a_fresh_uuid():
    """C2 crash shape 1: `_compose_row` interpolates a snippet's `id`
    directly into Textual widget ids (`evals-snippet-text-<id>` etc); an id
    containing a space is not a legal Textual identifier, so an unvalidated
    JSON export used to reach `_compose_row` unmodified and raise
    `BadIdentifier` at mount time -- well after the (still-successful) DB
    write, leaving the dataset permanently un-openable. Validated at parse
    time now, same fallback a missing id already got."""
    content = json.dumps([{"id": "bad id", "text": "The protestors were"}])
    snippets = parse_json_snippets(content)
    assert snippets[0]["id"] != "bad id"
    uuid.UUID(snippets[0]["id"])  # does not raise


def test_parse_json_snippets_skips_entries_missing_text_or_not_an_object():
    """JSON/CSV per-entry policy asymmetry, settled: both now skip an
    individual bad entry rather than aborting the whole import, mirroring
    `parse_csv_snippets`'s existing blank-row tolerance -- a large,
    otherwise-valid export must not be rejected wholesale over one bad
    entry. Only a structurally invalid payload (not a list/`{"snippets":
    [...]}`  at all, see the rejection test above) still raises."""
    content = json.dumps(
        [
            {"text": "kept one"},
            {"text": ""},
            {"no_text_field": True},
            "not even an object",
            {"text": "kept two"},
        ]
    )
    snippets = parse_json_snippets(content)
    assert [s["text"] for s in snippets] == ["kept one", "kept two"]


def test_import_snippets_into_dataset_dedups_colliding_ids(evals_db):
    """C2 crash shape 2: re-importing the same export twice -- the
    round-trip `parse_json_snippets`'s own docstring advertises -- used to
    append a snippet whose id already existed in the dataset, so the next
    compose() tried to mount two rows sharing one widget id (`MountError`).
    A second import now mints a fresh id for the collision instead."""
    dataset_id = _make_dataset(evals_db, "dedupe-target", [])
    exported = [
        {"id": "fixed-id", "text": "The protestors were", "group": None, "note": None}
    ]
    first = import_snippets_into_dataset(evals_db, dataset_id, exported)
    assert [s["id"] for s in first] == ["fixed-id"]

    second = import_snippets_into_dataset(evals_db, dataset_id, exported)
    ids = [s["id"] for s in second]
    assert len(ids) == len(set(ids)), "duplicate snippet ids after re-import"
    assert ids[0] == "fixed-id"  # the original row's id is untouched
    assert ids[1] != "fixed-id"  # the re-imported row got a fresh id
    uuid.UUID(ids[1])


# ---------------------------------------------------------------------------
# Widget: rendering, region, and "genuinely visible" assertions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_snippet_editor_mounts_with_summary_and_labelled_char_column(
    evals_app, evals_db
):
    dataset_id = _make_dataset(
        evals_db,
        "nouns-12",
        [_snip("The protestors were", group="neutral"), _snip("The rioters were", group="loaded")],
    )
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        editor = screen.query_one("#evals-snippet-editor")
        assert editor.region.width > 0
        assert editor.region.height > 0

        summary = screen.query_one("#evals-snippet-editor-summary")
        summary_text = str(summary.renderable)
        # Two distinct clauses, independently pinned: a regression in
        # len(groups) (e.g. always 1, or double-counting) must not be
        # masked by the snippet-count clause also containing "2".
        assert "2 snippets" in summary_text
        assert "2 groups" in summary_text

        header = screen.query_one("#evals-snippet-table-header")
        header_text = str(header.renderable)
        assert "Chars" in header_text
        assert "token" not in header_text.lower()


@pytest.mark.asyncio
async def test_normal_snippet_renders_no_whitespace_marker(evals_app, evals_db):
    clean = _snip("The protestors were")
    dataset_id = _make_dataset(evals_db, "clean-set", [clean])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        text_widget = screen.query_one(f"#evals-snippet-text-{clean['id']}")
        assert text_widget.region.width > 0
        rendered_text = _row_text(screen, clean["id"])
        assert rendered_text == "The protestors were"
        assert "␣" not in rendered_text

        meta_text = _row_meta(screen, clean["id"])
        assert "␣" not in meta_text
        assert "exact dup" not in meta_text


@pytest.mark.asyncio
async def test_trailing_whitespace_snippet_renders_a_visible_marker(evals_app, evals_db):
    dirty = _snip("The government said ")
    dataset_id = _make_dataset(evals_db, "dirty-set", [dirty])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        text_widget = screen.query_one(f"#evals-snippet-text-{dirty['id']}")
        assert text_widget.region.width > 0
        assert text_widget.region.height > 0
        rendered_text = _row_text(screen, dirty["id"])
        assert rendered_text == "The government said␣"

        # The marker is genuinely styled, not just glyph substitution.
        rich_text = text_widget.renderable
        assert any(span.style for span in rich_text.spans)

        meta_text = _row_meta(screen, dirty["id"])
        assert "trailing" in meta_text
        assert "␣" in meta_text


@pytest.mark.asyncio
async def test_leading_and_interior_whitespace_are_both_detected_on_screen(
    evals_app, evals_db
):
    leading = _snip(" The protestors were")
    interior = _snip("The government  said")
    dataset_id = _make_dataset(evals_db, "assorted-set", [leading, interior])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        leading_text = _row_text(screen, leading["id"])
        assert leading_text.startswith("␣")
        assert "leading" in _row_meta(screen, leading["id"])

        interior_text = _row_text(screen, interior["id"])
        assert "␣␣" in interior_text
        assert "interior" in _row_meta(screen, interior["id"])


@pytest.mark.asyncio
async def test_minimal_pair_snippets_render_with_no_duplicate_warning(evals_app, evals_db):
    """Full-screen counterpart to the pure test above: a genuine minimal
    pair (one loaded noun swapped) must pass with zero warnings end to
    end, not just at the pure-function layer."""
    protestors = _snip("The protestors were")
    rioters = _snip("The rioters were")
    dataset_id = _make_dataset(evals_db, "minimal-pair-set", [protestors, rioters])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        assert "exact dup" not in _row_meta(screen, protestors["id"])
        assert "exact dup" not in _row_meta(screen, rioters["id"])

        warnings_line = str(
            screen.query_one("#evals-snippet-warnings-summary").renderable
        )
        # The `or "0 warnings"` branch was unreachable: `compose()` renders
        # `"No warnings"` for a zero count, never `"0 warnings"` (see
        # `SnippetEditor.compose`'s `if total_warnings else "No warnings"`)
        # -- a bug that started rendering "0 warnings" instead would still
        # have passed this assertion.
        assert warnings_line == "No warnings", warnings_line


@pytest.mark.asyncio
async def test_exact_duplicate_after_normalization_is_flagged_on_screen(
    evals_app, evals_db
):
    dirty = _snip("The government said ")  # row 1: trailing space
    clean = _snip("The government said")  # row 2: exact dup of row 1
    dataset_id = _make_dataset(evals_db, "dup-set", [dirty, clean])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        dirty_meta = _row_meta(screen, dirty["id"])
        assert "trailing" in dirty_meta
        assert "exact dup" not in dirty_meta

        clean_meta = _row_meta(screen, clean["id"])
        assert "exact dup of 1" in clean_meta

        warnings_line = str(
            screen.query_one("#evals-snippet-warnings-summary").renderable
        )
        # Exact count: 1 trailing-whitespace warning + 1 exact-duplicate
        # warning = 2 (see count_warnings' additive footer). A bare "2" in
        # warnings_line" would also pass for "12 warnings" -- anchored to
        # the full rendered token instead.
        assert warnings_line == "2 warnings", warnings_line


@pytest.mark.asyncio
async def test_snippet_table_scrolls_to_reveal_rows_and_import_button_stays_pinned(
    evals_app, evals_db
):
    """I4: panes (and the widgets inside them) never scrolled --
    `Vertical`'s own ``DEFAULT_CSS`` is ``overflow: hidden hidden``, so a
    dataset with more snippets than fit the pane's bounded height (real,
    post-C1) had permanently unreachable rows in an editor whose entire
    job is reviewing a snippet set. ``#evals-snippet-table`` now scrolls
    independently, and -- since it is the ONLY thing that scrolls, not the
    whole ``SnippetEditor`` -- ``#evals-import-snippets`` (a sibling
    AFTER it, never a descendant of it) never moves and needs no scrolling
    to reach, at any scroll position."""
    snippets = [_snip(f"snippet number {i}") for i in range(60)]
    dataset_id = _make_dataset(evals_db, "long-set", snippets)

    async with evals_app.run_test(size=(160, 45)) as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        table = screen.query_one("#evals-snippet-table")
        assert str(table.styles.overflow_y) == "auto"
        assert table.virtual_size.height > table.size.height, (
            "60 rows should overflow the table's bounded height -- if "
            "not, this test isn't actually exercising the overflow case"
        )

        import_button = screen.query_one("#evals-import-snippets")
        button_y_before = import_button.region.y
        assert button_y_before > 0

        last_row_text = screen.query_one(f"#evals-snippet-text-{snippets[-1]['id']}")
        pane = screen.query_one("#evals-detail-pane")
        assert not pane.region.contains_region(last_row_text.region), (
            "the last row should start out-of-view -- otherwise this test "
            "isn't proving scrolling was necessary to reach it"
        )

        table.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        assert table.scroll_offset.y > 0, "the table did not actually scroll"
        last_row_text_after = screen.query_one(
            f"#evals-snippet-text-{snippets[-1]['id']}"
        )
        assert pane.region.contains_region(last_row_text_after.region), (
            "the last row is still unreachable after scrolling to the end"
        )

        # The Import button is OUTSIDE the scrolling region -- its own
        # position must not have moved just because the table scrolled.
        import_button_after = screen.query_one("#evals-import-snippets")
        assert import_button_after.region.y == button_y_before


@pytest.mark.asyncio
async def test_empty_dataset_shows_empty_state_and_import_control(evals_app, evals_db):
    dataset_id = _make_dataset(evals_db, "fresh-set", [])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        screen = evals_app.screen

        empty = screen.query_one("#evals-snippet-empty")
        assert empty.region.width > 0
        assert empty.region.height > 0

        import_button = screen.query_one("#evals-import-snippets")
        assert import_button.region.width > 0
        assert import_button.region.height > 0


# ---------------------------------------------------------------------------
# Import: plain text, CSV, JSON -- UUIDs assigned, group round-trips
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plain_text_import_assigns_uuids_and_persists_inline(
    evals_app, evals_db, tmp_path
):
    dataset_id = _make_dataset(evals_db, "import-target", [])
    txt_path = tmp_path / "snippets.txt"
    txt_path.write_text("line one\nline two\n\nline three\n", encoding="utf-8")

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(txt_path)
        await pilot.pause()

    stored = evals_db.get_dataset(dataset_id)
    samples = stored["metadata"][RESERVED_LOCAL_DATASET_SAMPLES_KEY]
    assert [s["text"] for s in samples] == ["line one", "line two", "line three"]
    for sample in samples:
        uuid.UUID(sample["id"])  # does not raise
    assert stored["metadata"]["sample_count"] == 3


@pytest.mark.asyncio
async def test_csv_import_round_trips_the_group_field(evals_app, evals_db, tmp_path):
    dataset_id = _make_dataset(evals_db, "csv-target", [])
    csv_path = tmp_path / "snippets.csv"
    csv_path.write_text(
        "text,group\nThe protestors were,neutral\nThe rioters were,loaded\n"
        "no group here,\n",
        encoding="utf-8",
    )

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(csv_path)
        await pilot.pause()

    stored = evals_db.get_dataset(dataset_id)
    samples = stored["metadata"][RESERVED_LOCAL_DATASET_SAMPLES_KEY]
    by_text = {s["text"]: s["group"] for s in samples}
    assert by_text["The protestors were"] == "neutral"
    assert by_text["The rioters were"] == "loaded"
    assert by_text["no group here"] is None


@pytest.mark.asyncio
async def test_json_import_round_trips_id_group_and_note(evals_app, evals_db, tmp_path):
    dataset_id = _make_dataset(evals_db, "json-target", [])
    fixed_id = str(uuid.uuid4())
    json_path = tmp_path / "snippets.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "id": fixed_id,
                    "text": "The protestors were",
                    "group": "neutral",
                    "note": "baseline opener",
                },
                {"text": "The rioters were", "group": "loaded"},
            ]
        ),
        encoding="utf-8",
    )

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(json_path)
        await pilot.pause()

    stored = evals_db.get_dataset(dataset_id)
    samples = stored["metadata"][RESERVED_LOCAL_DATASET_SAMPLES_KEY]
    by_id = {s["id"]: s for s in samples}
    assert by_id[fixed_id]["text"] == "The protestors were"
    assert by_id[fixed_id]["group"] == "neutral"
    assert by_id[fixed_id]["note"] == "baseline opener"

    other = [s for s in samples if s["id"] != fixed_id][0]
    uuid.UUID(other["id"])
    assert other["group"] == "loaded"


@pytest.mark.asyncio
async def test_json_import_with_illegal_id_does_not_crash_and_dataset_stays_openable(
    evals_app, evals_db, tmp_path
):
    """End-to-end C2 crash shape 1, through the real mount path (pure-logic
    coverage is in `test_parse_json_snippets_replaces_an_illegal_id_with_a_
    fresh_uuid` above). Before the fix, this raised `BadIdentifier` out of
    `_compose_row`'s `Horizontal(id=f"evals-snippet-row-{snippet_id}")`
    during the post-import `refresh(recompose=True)`, and the write had
    already landed -- so a SECOND selection of the same dataset (the
    "every later selection crashes again" symptom) is the real regression
    check here, not just the import itself surviving."""
    dataset_id = _make_dataset(evals_db, "illegal-id-target", [])
    json_path = tmp_path / "bad_id.json"
    json_path.write_text(
        json.dumps([{"id": "bad id", "text": "The protestors were"}]), encoding="utf-8"
    )
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(json_path)
        await pilot.pause()

        # Re-select the dataset, exactly like a user reopening it after
        # import -- must not crash the detail pane a second time.
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor_again = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        assert editor_again.region.width > 0
        assert editor_again.region.height > 0

    stored = evals_db.get_dataset(dataset_id)
    samples = stored["metadata"][RESERVED_LOCAL_DATASET_SAMPLES_KEY]
    assert len(samples) == 1
    uuid.UUID(samples[0]["id"])  # the illegal id was replaced, not stored verbatim


@pytest.mark.asyncio
async def test_json_import_of_the_same_export_twice_does_not_crash(
    evals_app, evals_db, tmp_path
):
    """End-to-end C2 crash shape 2, through the real mount path: importing
    the same JSON export twice used to append a second snippet sharing the
    first's `id`, and the next `_compose_row` recompose tried to mount two
    rows with the same widget id -- `MountError`, again leaving the
    dataset permanently un-openable afterward."""
    dataset_id = _make_dataset(evals_db, "reimport-target", [])
    json_path = tmp_path / "export.json"
    json_path.write_text(
        json.dumps(
            [{"id": "stable-id", "text": "The protestors were", "group": "neutral"}]
        ),
        encoding="utf-8",
    )
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(json_path)
        await pilot.pause()
        # Re-importing the SAME export -- the round-trip case.
        editor._handle_import_file_selected(json_path)
        await pilot.pause()

        # Selecting the dataset again is the real regression check (see
        # the illegal-id test above for why): the historical crash left
        # every later selection broken too, not just the import itself.
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor_again = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        assert editor_again.region.width > 0
        assert editor_again.region.height > 0

    stored = evals_db.get_dataset(dataset_id)
    samples = stored["metadata"][RESERVED_LOCAL_DATASET_SAMPLES_KEY]
    ids = [s["id"] for s in samples]
    assert len(samples) == 2
    assert len(set(ids)) == 2, "duplicate snippet ids after re-importing the same export"


@pytest.mark.asyncio
async def test_csv_import_without_text_column_notifies_error_and_does_not_persist(
    evals_app, evals_db, tmp_path
):
    dataset_id = _make_dataset(evals_db, "bad-csv-target", [])
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("foo,bar\n1,2\n", encoding="utf-8")

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(csv_path)
        await pilot.pause()

    assert any(
        severity == "error" for _message, severity in evals_app.app_instance.notifications
    )
    stored = evals_db.get_dataset(dataset_id)
    assert stored["metadata"].get("sample_count", 0) == 0


@pytest.mark.asyncio
async def test_non_utf8_import_file_notifies_error_instead_of_crashing(
    evals_app, evals_db, tmp_path
):
    """I1: `read_text(encoding="utf-8")` was wrapped in `except OSError`
    only, but a decode failure raises `UnicodeDecodeError` -- a `ValueError`
    subclass, not an `OSError`. A Latin-1/cp1252 CSV (the most likely
    non-UTF-8 file a real user picks -- it's what Excel exports by default)
    used to propagate straight out of this `push_screen` callback and crash
    the app rather than producing the same notification an unreadable path
    already gets."""
    dataset_id = _make_dataset(evals_db, "cp1252-target", [])
    csv_path = tmp_path / "excel_export.csv"
    # "café" encoded as cp1252/Latin-1 -- 0xE9 for 'é' is not valid UTF-8
    # on its own, so decoding this file as UTF-8 raises UnicodeDecodeError.
    csv_path.write_bytes("text,group\nA caf\xe9 opened,neutral\n".encode("cp1252"))

    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        editor._handle_import_file_selected(csv_path)
        await pilot.pause()

        # The app must still be alive and the screen still responsive --
        # the crash this guards against propagates out of the callback and
        # takes the whole app down with it.
        assert isinstance(evals_app.screen, type(evals_app.screen))
        editor_still_here = evals_app.screen.query_one(
            "#evals-snippet-editor", SnippetEditor
        )
        assert editor_still_here.region.width > 0

    assert any(
        severity == "error" for _message, severity in evals_app.app_instance.notifications
    )
    stored = evals_db.get_dataset(dataset_id)
    assert stored["metadata"].get("sample_count", 0) == 0


@pytest.mark.asyncio
async def test_snippet_editor_never_imports_the_runner_or_capture_client():
    """Import performs local file I/O and DB writes only -- pin that this
    module can never reach a provider, mirroring bench_editor.py's and
    inspector.py's identical guarantee (see test_evals_bench_editor.py)."""
    source = Path(snippet_editor_module.__file__).read_text()
    assert "WordBenchRunner" not in source
    assert "CaptureClientLike" not in source
    assert "capture" + "_client" not in source


# ---------------------------------------------------------------------------
# Screen-seam integration: selecting a dataset mounts SnippetEditor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_evals_screen_dataset_selection_mounts_snippet_editor(evals_app, evals_db):
    dataset_id = _make_dataset(evals_db, "seam-set", [_snip("hello world")])
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id=dataset_id)
        await pilot.pause()
        editor = evals_app.screen.query_one("#evals-snippet-editor", SnippetEditor)
        assert editor.region.width > 0
        assert editor.region.height > 0


@pytest.mark.asyncio
async def test_missing_dataset_still_shows_the_not_found_message(evals_app, evals_db):
    """Pre-existing behaviour from Task 3 must survive this task's rewrite
    of the dataset branch's found-path body."""
    async with evals_app.run_test() as pilot:
        await pilot.pause()
        evals_app.screen.select(kind="dataset", id="does-not-exist")
        await pilot.pause()
        missing = evals_app.screen.query_one("#evals-detail-missing")
        assert "deleted" in str(missing.renderable)
