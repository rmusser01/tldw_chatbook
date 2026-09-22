# Tests/Actor_Packs/test_actor_pack_import_hidden_profile_root.py
"""
Regression tests for task-32901: the app-owned actor-pack roots must be
usable under the dotted ADR-127 fallback data root (~/.tldw_cli-data).

`ActorPackImportService.__init__` validates staging_root/profile_root via
`_absolute_path`, which calls `validate_path(path, path.parent)`. The
hidden-base anti-bypass rule rejects any base whose final component is
dotted, so whenever the fallback data root engaged (e.g. a group-writable
~/.local/share on a umask-002 machine) the profile root's immediate parent
was `.tldw_cli-data` and the whole app crashed in `TldwCli.__init__` with
`actor_pack_import_invalid`. User-supplied archive paths must keep the
strict hidden-path rejection.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_chatbook.Actor_Packs.importer import (
    ActorPackImportError,
    _absolute_path,
)

def test_absolute_path_strict_default_still_rejects_dotted_base(
    tmp_path: Path,
) -> None:
    dotted_root = tmp_path / ".tldw_cli-data" / "default_user"
    dotted_root.mkdir(parents=True)
    with pytest.raises(ActorPackImportError, match="actor_pack_import_invalid"):
        _absolute_path(dotted_root)


def test_absolute_path_allows_hidden_for_app_owned_roots(
    tmp_path: Path,
) -> None:
    # validate_path resolves, so compare against the resolved real path.
    dotted_root = (tmp_path / ".tldw_cli-data" / "default_user").resolve()
    dotted_root.mkdir(parents=True)
    assert _absolute_path(dotted_root, allow_hidden=True) == dotted_root
    staging = dotted_root / "actor_pack_imports"
    staging.mkdir()
    assert _absolute_path(staging, allow_hidden=True) == staging


def test_absolute_path_still_rejects_non_absolute_and_relative_noise() -> None:
    with pytest.raises(ActorPackImportError):
        _absolute_path("relative/path", allow_hidden=True)
    with pytest.raises(ActorPackImportError):
        _absolute_path("has\x00nul", allow_hidden=True)


def test_service_constructs_under_dotted_fallback_root(tmp_path: Path) -> None:
    """The wiring boundary, not just the helper, must accept the fallback root.

    app.py passes `staging_root=get_user_data_dir()/actor_pack_imports` and
    `profile_root=get_user_data_dir()`; under the ADR-127 fallback root the
    profile root's immediate parent is `.tldw_cli-data`. Constructing the
    real service (real repository, task-32901's exact crash site) proves
    both constructor call sites keep `allow_hidden=True`.
    """

    from tldw_chatbook.Actor_Packs.importer import ActorPackImportService
    from tldw_chatbook.Actor_Packs.repository import ActorPackRepository
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    dotted_root = (tmp_path / ".tldw_cli-data" / "default_user").resolve()
    dotted_root.mkdir(parents=True)
    db = CharactersRAGDB(str(tmp_path / "profile.db"), client_id="actor-pack-hidden")
    service = ActorPackImportService(
        ActorPackRepository(db),
        staging_root=dotted_root / "actor_pack_imports",
        profile_root=dotted_root,
        local_service=None,
    )
    assert service._profile_root == dotted_root
    assert service._staging_root == dotted_root / "actor_pack_imports"
