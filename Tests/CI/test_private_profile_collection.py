"""Exact-node children prune sibling collectors, not fixtures or isolation."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent
from types import SimpleNamespace

from Tests.private_profile import private_profile_test

_REPO = Path(__file__).resolve().parents[2]


async def test_child_prunes_siblings_and_keeps_package_fixtures_and_fresh_profiles(
    request, tmp_path
):
    with TemporaryDirectory(dir=_REPO) as directory:
        root = Path(directory)
        (root / "conftest.py").write_text(
            dedent("""\
                import json
                import os
                from pathlib import Path
                import pytest

                def pytest_collect_file(file_path, parent):
                    assert file_path.name == "test_selected.py", file_path

                @pytest.fixture(autouse=True)
                def ancestor_fixture():
                    profile = Path(os.environ["TLDW_TEST_CONFIG_ROOT"])
                    receipt = profile / "receipt.json"
                    assert not receipt.exists()
                    yield
                    receipt.write_text(json.dumps({
                        "pid": os.getpid(), "profile": str(profile),
                        "home": os.environ["HOME"],
                    }))
                """)
        )
        package = root / "probe_package"
        package.mkdir()
        (package / "__init__.py").write_text('PACKAGE_VALUE = "initialized"\n')
        (package / "test_sibling.py").write_text(
            'raise AssertionError("unselected sibling imported")\n'
        )
        case = package / "test_selected.py"
        case.write_text(
            dedent("""\
                import pytest
                from . import PACKAGE_VALUE
                from Tests.private_profile import private_profile_test

                class TestSelected:
                    @pytest.mark.asyncio
                    @pytest.mark.parametrize("value", ["selected", "unselected"],
                                             ids=["selected[brackets]", "unselected"])
                    @private_profile_test
                    async def test_selected(self, request, value):
                        assert value == "selected"
                        assert PACKAGE_VALUE == "initialized"
                """)
        )
        node = f"{case.relative_to(_REPO)}::TestSelected::test_selected[selected[brackets]]"

        @private_profile_test
        async def parent_only(request):
            raise AssertionError("the parent must not execute the test body")

        receipts = []
        for attempt in range(2):
            work = tmp_path / str(attempt)
            work.mkdir()
            child_request = SimpleNamespace(
                node=SimpleNamespace(
                    nodeid=node,
                    name="test_selected[selected[brackets]]",
                    get_closest_marker=lambda _: None,
                ),
                config=request.config,
                getfixturevalue=lambda _, work=work: work,
            )
            await parent_only(request=child_request)
            profile = work / "private-profile-process" / "profile"
            receipt = json.loads((profile / "receipt.json").read_text())
            assert receipt["profile"] == str(profile)
            assert receipt["home"] == str(profile / "home")
            assert receipt["pid"] != os.getpid()
            receipts.append(receipt)
        assert receipts[0]["pid"] != receipts[1]["pid"]
        assert receipts[0]["profile"] != receipts[1]["profile"]
