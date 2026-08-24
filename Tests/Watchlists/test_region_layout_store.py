from unittest.mock import Mock

import pytest

from tldw_chatbook.UI.Watchlists_Modules import region_layout_store
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout

pytestmark = pytest.mark.unit


def _get_from(values):
    return lambda section, key, default=None: values.get(key, default)


def test_missing_layout_uses_first_run_default_and_writes_version(monkeypatch):
    writer = Mock(return_value=True)
    monkeypatch.setattr(region_layout_store, "get_cli_setting", _get_from({}))
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)

    loaded = region_layout_store.load_region_layout()

    assert loaded.collapsed == frozenset({Region.RIGHT_RAIL})
    writer.assert_called_once_with(
        {
            "watchlists": {
                "collapsed_regions": ["right_rail"],
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        },
        delete_keys={"watchlists": ("content_reader_migrated",)},
    )


def test_explicit_empty_layout_means_every_side_pane_is_expanded(monkeypatch):
    writer = Mock()
    monkeypatch.setattr(
        region_layout_store,
        "get_cli_setting",
        _get_from(
            {
                "collapsed_regions": [],
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        ),
    )
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)

    assert region_layout_store.load_region_layout() == RegionLayout()
    writer.assert_not_called()


def test_valid_side_panes_round_trip(monkeypatch):
    saved = {}

    def writer(section_values, *, delete_keys=None):
        saved.update(section_values["watchlists"])
        return True

    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)
    layout = RegionLayout(
        collapsed=frozenset(
            {Region.LEFT_RAIL, Region.ITEMS, Region.RIGHT_RAIL}
        )
    )

    assert region_layout_store.save_region_layout(layout) is True
    monkeypatch.setattr(region_layout_store, "get_cli_setting", _get_from(saved))

    assert region_layout_store.load_region_layout() == layout


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            ["content", "feeds", "unknown", "left_rail"],
            frozenset({Region.LEFT_RAIL}),
        ),
        ("items", frozenset({Region.ITEMS})),
        (42, frozenset()),
    ],
)
def test_load_normalizes_legacy_unknown_singleton_and_non_sequence_values(
    monkeypatch, raw, expected
):
    writer = Mock(return_value=True)
    monkeypatch.setattr(
        region_layout_store,
        "get_cli_setting",
        _get_from(
            {
                "collapsed_regions": raw,
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        ),
    )
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)

    loaded = region_layout_store.load_region_layout()

    assert loaded.collapsed == expected
    normalized_values = [
        region.value
        for region in region_layout_store.COLLAPSIBLE_REGIONS
        if region in expected
    ]
    writer.assert_called_once_with(
        {
            "watchlists": {
                "collapsed_regions": normalized_values,
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        },
        delete_keys={"watchlists": ("content_reader_migrated",)},
    )


def test_stale_version_is_normalized_with_one_atomic_write(monkeypatch):
    writer = Mock(return_value=True)
    monkeypatch.setattr(
        region_layout_store,
        "get_cli_setting",
        _get_from(
            {
                "collapsed_regions": ["items", "right_rail"],
                "layout_version": 1,
            }
        ),
    )
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)

    loaded = region_layout_store.load_region_layout()

    assert loaded.collapsed == frozenset({Region.ITEMS, Region.RIGHT_RAIL})
    writer.assert_called_once_with(
        {
            "watchlists": {
                "collapsed_regions": ["items", "right_rail"],
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        },
        delete_keys={"watchlists": ("content_reader_migrated",)},
    )


@pytest.mark.parametrize("failure", [False, OSError("disk full")])
def test_failed_normalization_applies_safe_layout_and_retries(monkeypatch, failure):
    values = {"collapsed_regions": ["content", "left_rail"], "layout_version": 1}
    writer = Mock(side_effect=failure if isinstance(failure, Exception) else None)
    if failure is False:
        writer.return_value = False
    monkeypatch.setattr(region_layout_store, "get_cli_setting", _get_from(values))
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)

    first = region_layout_store.load_region_layout()
    second = region_layout_store.load_region_layout()

    assert first.collapsed == second.collapsed == frozenset({Region.LEFT_RAIL})
    assert values["layout_version"] == 1
    assert writer.call_count == 2


def test_save_filters_to_side_panes_and_returns_writer_result(monkeypatch):
    writer = Mock(return_value=False)
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)
    layout = RegionLayout(
        collapsed=frozenset({Region.CONTENT, Region.LEFT_RAIL, Region.RIGHT_RAIL})
    )

    assert region_layout_store.save_region_layout(layout) is False
    writer.assert_called_once_with(
        {
            "watchlists": {
                "collapsed_regions": ["left_rail", "right_rail"],
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        },
        delete_keys={"watchlists": ("content_reader_migrated",)},
    )


def test_current_version_with_non_normalized_data_is_rewritten(monkeypatch):
    writer = Mock(return_value=True)
    monkeypatch.setattr(
        region_layout_store,
        "get_cli_setting",
        _get_from(
            {
                "collapsed_regions": [
                    "right_rail",
                    "left_rail",
                    "left_rail",
                    "content",
                ],
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        ),
    )
    monkeypatch.setattr(region_layout_store, "save_settings_to_cli_config", writer)

    loaded = region_layout_store.load_region_layout()

    assert loaded.collapsed == frozenset({Region.LEFT_RAIL, Region.RIGHT_RAIL})
    writer.assert_called_once_with(
        {
            "watchlists": {
                "collapsed_regions": ["left_rail", "right_rail"],
                "layout_version": region_layout_store.LAYOUT_VERSION,
            }
        },
        delete_keys={"watchlists": ("content_reader_migrated",)},
    )
