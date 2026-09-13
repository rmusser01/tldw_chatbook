"""Future isolated selectors are fenced without creating destination state."""

import pytest

from tldw_chatbook.Backup_Recovery import control_records as records
from tldw_chatbook.Backup_Recovery.bootstrap import startup_permission


@pytest.mark.parametrize('nested', [False, True])
def test_pending_fences_future_selector_without_creating_it(tmp_path, nested):
    destination = tmp_path / 'destination'
    destination.mkdir(mode=0o700)
    selector = destination / ('missing/nested/config.toml' if nested else 'config.toml')
    bootstrap = tmp_path / 'bootstrap'
    records.register_pending(bootstrap, 'op', ('profile',), tmp_path / 'control', (selector,))
    assert list(destination.iterdir()) == []
    assert startup_permission(selector, bootstrap) == (False, 'recovery_pending')
    selector.parent.mkdir(parents=True, exist_ok=True)
    selector.write_bytes(b'broken [')
    assert startup_permission(selector, bootstrap) == (False, 'recovery_pending')
    assert selector.read_bytes() == b'broken ['


@pytest.mark.parametrize('shape', ['leaf_link', 'dangling_leaf', 'ancestor_link',
                                   'existing_ancestor_link', 'dangling_ancestor',
                                   'file_ancestor', 'directory_leaf'])
def test_future_selector_refuses_unverified_paths(tmp_path, shape):
    destination = tmp_path / 'destination'
    destination.mkdir(mode=0o700)
    actual = tmp_path / 'actual'
    actual.mkdir(mode=0o700)
    (actual / 'config.toml').write_bytes(b'original')
    selector = destination / 'config.toml'
    if shape == 'leaf_link':
        selector.symlink_to(actual / 'config.toml')
    elif shape == 'dangling_leaf':
        selector.symlink_to(actual / 'absent')
    elif shape in ('ancestor_link', 'existing_ancestor_link', 'dangling_ancestor'):
        (destination / 'alias').symlink_to(actual / 'absent' if shape == 'dangling_ancestor' else actual)
        selector = destination / 'alias' / ('config.toml' if shape == 'existing_ancestor_link' else 'future/config.toml')
    elif shape == 'file_ancestor':
        (destination / 'file').write_bytes(b'not a directory')
        selector = destination / 'file' / 'config.toml'
    else:
        selector.mkdir(mode=0o700)
    with pytest.raises((OSError, ValueError)):
        records.register_pending(tmp_path / 'bootstrap', 'op', ('profile',),
                                 tmp_path / 'control', (selector,))
    assert not (tmp_path / 'bootstrap').exists()
    assert (actual / 'config.toml').read_bytes() == b'original'


@pytest.mark.parametrize('root_name', ['bootstrap', 'control'])
def test_future_selector_still_refuses_control_overlap(tmp_path, root_name):
    selector = tmp_path / root_name / 'missing' / 'config.toml'
    with pytest.raises(ValueError, match='control_root_overlaps_target'):
        records.register_pending(tmp_path / 'bootstrap', 'op', ('profile',),
                                 tmp_path / 'control', (selector,))
    assert not (tmp_path / root_name).exists()


def test_future_selector_rechecks_absence_under_pinned_ancestor(tmp_path, monkeypatch):
    destination = tmp_path / 'destination'
    destination.mkdir(mode=0o700)
    selector = destination / 'missing' / 'config.toml'
    original = records.pinned_directory
    from contextlib import contextmanager

    @contextmanager
    def changed(path):
        with original(path) as fd:
            if path == destination:
                (destination / 'missing').symlink_to(tmp_path / 'elsewhere')
            yield fd

    monkeypatch.setattr(records, 'pinned_directory', changed)
    with pytest.raises((OSError, ValueError)):
        records.register_pending(tmp_path / 'bootstrap', 'op', ('profile',),
                                 tmp_path / 'control', (selector,))
    assert not (tmp_path / 'bootstrap').exists()
    assert (destination / 'missing').is_symlink()


def test_future_selector_refuses_untrusted_existing_ancestor(tmp_path):
    destination = tmp_path / 'destination'
    destination.mkdir(mode=0o777)
    destination.chmod(0o777)
    try:
        with pytest.raises(OSError):
            records.register_pending(tmp_path / 'bootstrap', 'op', ('profile',),
                                     tmp_path / 'control', (destination / 'config.toml',))
        assert not (tmp_path / 'bootstrap').exists()
    finally:
        destination.chmod(0o700)
