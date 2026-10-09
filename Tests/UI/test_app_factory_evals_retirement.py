"""Retire the actual Evals database created by the private app fixture."""

import pytest

from Tests.Backup_Recovery.test_home_citation_retirement import _run
from Tests.UI.test_app_factory_owned_database_retirement import _SCRIPT as _OWNED_SCRIPT

pytestmark = pytest.mark.bootstrap_profile

_SCRIPT = (
    _OWNED_SCRIPT.replace(
        "    original_owners = tuple(owners)",
        "    from tldw_chatbook.DB.Evals_DB import EvalsDB\n"
        "    evaluations = app.evaluation_orchestrator.db\n"
        "    assert type(evaluations) is EvalsDB and not evaluations.is_memory_db\n"
        "    assert evaluations.db_path.is_relative_to(root) or evaluations.db_path.parent == owned_directory\n"
        "    if route != 'borrowed_evals_constructor':\n"
        "        owners.append(evaluations)\n"
        "    original_owners = tuple(owners)",
    )
    .replace(
        "    try:\n        app = app_factory._build_test_app()",
        "    borrowed_evals = None\n"
        "    if route == 'borrowed_evals_constructor':\n"
        "        from tldw_chatbook.DB.Evals_DB import EvalsDB\n"
        "        from tldw_chatbook.Evals import eval_orchestrator\n"
        "        borrowed_evals = EvalsDB(root / 'borrowed-evals.sqlite')\n"
        "        borrowed_evals_connection = borrowed_evals.get_connection()\n"
        "        constructor_patch = patch.object(eval_orchestrator, 'EvalsDB', return_value=borrowed_evals)\n"
        "        constructor_patch.start()\n"
        "    try:\n        app = app_factory._build_test_app()",
    )
    .replace(
        "        assert not owned_directory.exists()",
        "        if borrowed_evals is not None:\n"
        "            assert evaluations is borrowed_evals\n"
        "            assert not physically_closed(borrowed_evals_connection)\n"
        "        assert not owned_directory.exists()",
    )
    .replace(
        "        borrowed.close()",
        "        borrowed.close()\n"
        "        if borrowed_evals is not None:\n"
        "            borrowed_evals.close()",
    )
    .replace(
        "            assert owner.db_path.parent == owned_directory",
        "            assert owner is evaluations or owner.db_path.parent == owned_directory",
    )
    .replace(
        "    worker = None",
        "    if route == 'replaced_evals_field':\n"
        "        app.evaluation_orchestrator = None\n"
        "        app.local_evaluation_service = None\n"
        "    worker = None",
    )
    .replace(
        "            assert all(owner.db_path.parent == owned_directory for owner in original_owners)",
        "            assert all(owner is evaluations or owner.db_path.parent == owned_directory "
        "for owner in original_owners)",
    )
)


@pytest.mark.parametrize(
    "route", ["retained", "replaced_evals_field", "borrowed_evals_constructor"]
)
def test_factory_retires_exact_constructor_evals_before_directory_removal(
    tmp_path, route
):
    _run(tmp_path, route, "evals-owner", script=_SCRIPT)
