import pytest

from tldw_chatbook.UI.Watchlists_Modules import region_layout_store
from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region, RegionLayout

pytestmark = pytest.mark.unit


def test_round_trips_collapsed_regions(monkeypatch):
    """Uses FEEDS/RIGHT_RAIL, deliberately not CONTENT: this test is about
    generic round-tripping, and CONTENT now goes through the one-time
    migration in `load_region_layout` (see the dedicated migration tests
    below), which would make a CONTENT-collapsed input diverge from what
    comes back out on the very first load — a different behaviour than the
    one this test checks.
    """
    saved = {}

    def fake_save(section, key, value):
        saved[(section, key)] = value
        return True

    monkeypatch.setattr(region_layout_store, "save_setting_to_cli_config", fake_save)
    region_layout_store.save_region_layout(
        RegionLayout(collapsed=frozenset({Region.FEEDS, Region.RIGHT_RAIL}))
    )

    # Flat section, not "watchlists.layout" — a dotted section silently no-ops.
    assert ("watchlists", "collapsed_regions") in saved
    assert sorted(saved[("watchlists", "collapsed_regions")]) == ["feeds", "right_rail"]

    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: saved.get((section, key), default),
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.FEEDS, Region.RIGHT_RAIL})


def test_load_applies_first_run_default_when_key_was_never_saved(monkeypatch):
    """`get_cli_setting` returns its `default` argument only when the key is
    absent — i.e. this is a genuine "never saved" case, not merely "saved as
    empty." Phase D wires a real reader into CONTENT, so the first-run
    default is now `RegionLayout()` (nothing collapsed) like any other
    region — see `load_region_layout`'s docstring for why the
    never-saved-vs-saved-empty *distinction* still matters even though both
    now resolve to the same value."""
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: default
    )
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config", lambda *a, **k: True
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded == RegionLayout()


def test_load_honors_an_explicit_empty_layout_as_everything_expanded(monkeypatch):
    """A key that was explicitly SAVED as `[]` (the user re-expanded
    everything, including CONTENT, and that was persisted) must be honored
    exactly — not silently overridden back to the first-run default, or a
    user's deliberate choice to keep CONTENT expanded could never survive a
    restart."""
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: []
    )
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config", lambda *a, **k: True
    )
    assert region_layout_store.load_region_layout() == RegionLayout()


def test_load_migrates_a_pre_phase_d_content_collapse_away_on_first_run(monkeypatch):
    """A user who saved a layout before Phase D necessarily has CONTENT in
    `collapsed_regions` unless they specifically expanded it (CONTENT started
    collapsed by default back then), so honoring that membership forever
    would leave the new reader stuck collapsed post-upgrade, looking broken
    rather than shipped. The migration marker is unset (a pre-upgrade config
    never wrote it), so the persisted CONTENT collapse must be dropped, and
    the migration marker must be written so this runs exactly once.

    Fix round 2 (coordinator review, Critical): the first version of this
    test hard-returned `False` for the migration-marker key from `fake_get`
    regardless of what `fake_save` had written, so it measured only ONE
    `load_region_layout()` call and never exercised the write -> read
    feedback loop the whole migration depends on. That let a real bug ship:
    the migration dropped CONTENT from the RETURNED layout and wrote the
    marker, but never rewrote `collapsed_regions` itself -- so the disk
    value stayed `["content", "right_rail"]`, and the very next load (next
    launch, or simply the next remount -- this screen unmounts on
    navigation) saw `migrated == True` and skipped the discard entirely,
    permanently re-collapsing CONTENT for exactly the returning users this
    migration exists to help. `fake_get` now reads back whatever `fake_save`
    actually wrote for EVERY key, and a second `load_region_layout()` call
    against the same fake store asserts the correction survives it.
    """
    saved = {"collapsed_regions": ["content", "right_rail"]}
    migration_writes = []

    def fake_get(section, key, default=None):
        return saved.get(key, default)

    def fake_save(section, key, value):
        saved[key] = value
        if key == "content_reader_migrated":
            migration_writes.append(value)
        return True

    monkeypatch.setattr(region_layout_store, "get_cli_setting", fake_get)
    monkeypatch.setattr(region_layout_store, "save_setting_to_cli_config", fake_save)

    loaded = region_layout_store.load_region_layout()

    assert loaded.collapsed == frozenset({Region.RIGHT_RAIL}), (
        "a stub-era CONTENT collapse must not survive the Phase D upgrade"
    )
    assert migration_writes == [True], "the migration marker must be persisted exactly once"
    assert saved["collapsed_regions"] == ["right_rail"], (
        "the migration must rewrite `collapsed_regions` on disk, not just "
        "the marker -- otherwise the correction is visible for exactly the "
        "one RegionLayout this call returned and reverts on the next load"
    )

    # The write -> read loop, exercised for real: a SECOND load against the
    # same fake store (simulating the next launch, or simply the next
    # remount -- this screen unmounts on navigation) must still see CONTENT
    # expanded. Before the fix this returned `frozenset({Region.CONTENT,
    # Region.RIGHT_RAIL})` -- the marker suppressed the discard, and the
    # stale disk value was all that was left to read.
    reloaded = region_layout_store.load_region_layout()
    assert reloaded.collapsed == frozenset({Region.RIGHT_RAIL}), (
        "the correction must survive a second load, not just the first -- "
        f"got {reloaded.collapsed}"
    )
    assert migration_writes == [True], (
        "the second load must not re-run the migration or write the marker again"
    )


def test_migration_does_not_mark_done_when_the_correcting_write_raises(monkeypatch):
    """Fix round 3 (coordinator review, Critical, carried from round 2):
    marking the migration done regardless of whether the corrected
    `collapsed_regions` write actually landed reproduces round 2's exact bug
    THROUGH the failure path -- disk stays stub-era (`["content", ...]`),
    the marker is now `True`, and the migration can never retry, so CONTENT
    is hidden forever for exactly the returning users this migration exists
    to help.

    This is the "raises" failure mode: `save_setting_to_cli_config` throws.
    `load_region_layout` must not let that escape (a raise here would exit
    the whole app from `on_mount`), must leave the marker unset so the next
    load retries, and must leave disk untouched.
    """
    saved = {"collapsed_regions": ["content", "right_rail"]}

    def fake_get(section, key, default=None):
        return saved.get(key, default)

    def fake_save(section, key, value):
        if key == "collapsed_regions":
            raise OSError("disk full")
        saved[key] = value
        return True

    monkeypatch.setattr(region_layout_store, "get_cli_setting", fake_get)
    monkeypatch.setattr(region_layout_store, "save_setting_to_cli_config", fake_save)

    loaded = region_layout_store.load_region_layout()  # must not raise

    assert loaded.collapsed == frozenset({Region.RIGHT_RAIL}), (
        "the in-memory correction for THIS call must still apply even "
        "though persisting it failed"
    )
    assert "content_reader_migrated" not in saved, (
        "the marker must NOT be set when the correcting write failed -- "
        "setting it anyway stops the migration from ever retrying"
    )
    assert saved["collapsed_regions"] == ["content", "right_rail"], (
        "disk must be unchanged since the write raised"
    )

    # The retry, exercised for real: a second load against the same
    # (still-failing) fake store must still self-correct in memory, not
    # honor the stale disk value now that the marker was correctly left
    # unset.
    reloaded = region_layout_store.load_region_layout()
    assert reloaded.collapsed == frozenset({Region.RIGHT_RAIL}), (
        f"a second load must retry the migration -- got {reloaded.collapsed}"
    )
    assert "content_reader_migrated" not in saved


def test_migration_does_not_mark_done_when_the_correcting_write_returns_false(monkeypatch):
    """The failure mode a raise-only guard misses entirely, and the
    important one: `save_setting_to_cli_config` signals failure by
    RETURNING `False`, not only by raising (see `config.py`). A bare
    `try`/`except Exception` around the write -- exactly what shipped after
    round 2's fix -- does not observe this path at all, so it marked the
    migration done here even though nothing was actually written.
    """
    saved = {"collapsed_regions": ["content", "right_rail"]}

    def fake_get(section, key, default=None):
        return saved.get(key, default)

    def fake_save(section, key, value):
        if key == "collapsed_regions":
            return False  # signals failure WITHOUT raising
        saved[key] = value
        return True

    monkeypatch.setattr(region_layout_store, "get_cli_setting", fake_get)
    monkeypatch.setattr(region_layout_store, "save_setting_to_cli_config", fake_save)

    loaded = region_layout_store.load_region_layout()

    assert loaded.collapsed == frozenset({Region.RIGHT_RAIL}), (
        "the in-memory correction for THIS call must still apply even "
        "though persisting it failed"
    )
    assert "content_reader_migrated" not in saved, (
        "the marker must NOT be set when the correcting write returned "
        "False -- a raise-only guard would wrongly set it here"
    )
    assert saved["collapsed_regions"] == ["content", "right_rail"], (
        "disk must be unchanged since the write reported failure"
    )

    reloaded = region_layout_store.load_region_layout()
    assert reloaded.collapsed == frozenset({Region.RIGHT_RAIL}), (
        f"a second load must retry the migration -- got {reloaded.collapsed}"
    )
    assert "content_reader_migrated" not in saved


def test_load_honors_a_deliberate_content_collapse_once_migrated(monkeypatch):
    """After the one-time migration has run (marker already `True`), a
    CONTENT collapse the user set AFTERWARD — a genuine choice about the
    real reader, not a leftover stub-era default — must be honored like any
    other region, not stripped again on every future load.
    """
    saved = {"collapsed_regions": ["content"], "content_reader_migrated": True}

    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: saved.get(key, default),
    )
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config",
        lambda section, key, value: saved.__setitem__(key, value) or True,
    )

    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.CONTENT})


def test_load_ignores_unknown_region_names(monkeypatch):
    # A config hand-edited or written by a newer version must not crash the screen.
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: ["content", "nonsense", "left_rail"],
    )
    loaded = region_layout_store.load_region_layout()
    assert loaded.collapsed == frozenset({Region.CONTENT, Region.LEFT_RAIL})


def test_load_tolerates_a_non_list_value(monkeypatch):
    monkeypatch.setattr(
        region_layout_store, "get_cli_setting", lambda section, key, default=None: "content"
    )
    assert region_layout_store.load_region_layout().collapsed == frozenset({Region.CONTENT})


def test_save_never_persists_solo(monkeypatch):
    """`solo_region` itself must never round-trip through config.

    NOTE (PR #926 review, Bug 1 fix): before the fix this test asserted
    `["content", "feeds"]` — the solo-DERIVED collapse of the other centre
    panes — as the persisted value. That encoded the bug: restoring that on
    the next launch would have applied a plain (non-solo) collapse of FEEDS
    and CONTENT with no `_pre_solo` baseline left to `Z`-restore from, even
    though the user never configured that layout by hand. The correct
    persisted value while soloed is the PRE-solo baseline, which for a solo
    called directly on a fresh `RegionLayout()` is "nothing collapsed" — see
    `test_save_while_soloed_persists_the_pre_solo_baseline_not_the_solo_derived_collapse`
    below for a case where the baseline is non-empty.
    """
    saved = {}
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config",
        lambda section, key, value: saved.__setitem__((section, key), value) or True,
    )
    region_layout_store.save_region_layout(RegionLayout().solo(Region.ITEMS))
    assert sorted(saved[("watchlists", "collapsed_regions")]) == []
    assert ("watchlists", "solo_region") not in saved


def test_save_while_soloed_persists_the_pre_solo_baseline_not_the_solo_derived_collapse(monkeypatch):
    """Regression test for PR #926 review, Bug 1: saving while soloed must
    persist what the user had BEFORE soloing, not the solo-derived collapse
    of the other centre panes — otherwise a restart strands the user in a
    layout they never configured, with no baseline left to recover from.
    """
    saved = {}
    monkeypatch.setattr(
        region_layout_store, "save_setting_to_cli_config",
        lambda section, key, value: saved.__setitem__((section, key), value) or True,
    )
    pre_solo = RegionLayout().toggle(Region.LEFT_RAIL)
    soloed = pre_solo.solo(Region.ITEMS)

    region_layout_store.save_region_layout(soloed)
    assert sorted(saved[("watchlists", "collapsed_regions")]) == ["left_rail"]

    monkeypatch.setattr(
        region_layout_store, "get_cli_setting",
        lambda section, key, default=None: saved.get((section, key), default),
    )
    reloaded = region_layout_store.load_region_layout()
    assert reloaded.collapsed == pre_solo.collapsed
    assert reloaded != soloed
    assert reloaded.solo_region is None
