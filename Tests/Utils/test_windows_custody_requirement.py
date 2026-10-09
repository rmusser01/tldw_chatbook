"""An elevated qualification job must exercise its native negative controls."""

import pytest

from Tests.windows_custody import (
    REQUIRED_CUSTODY_ENV,
    elevated_custody_required,
    unavailable_custody,
)


def test_required_custody_fails_instead_of_skipping_missing_capability(monkeypatch):
    monkeypatch.setenv(REQUIRED_CUSTODY_ENV, "1")
    assert elevated_custody_required()
    try:
        unavailable_custody("missing native owner-mutation capability")
    except pytest.fail.Exception as error:
        assert REQUIRED_CUSTODY_ENV in str(error)
        assert "missing native owner-mutation capability" in str(error)
    except pytest.skip.Exception:
        pytest.fail("required native custody was silently skipped")
    else:
        pytest.fail("missing required native custody capability was accepted")


@pytest.mark.parametrize("value", [None, "0"])
def test_ordinary_native_run_retains_capability_skip(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(REQUIRED_CUSTODY_ENV, raising=False)
    else:
        monkeypatch.setenv(REQUIRED_CUSTODY_ENV, value)
    assert not elevated_custody_required()
    with pytest.raises(pytest.skip.Exception, match="missing local capability"):
        unavailable_custody("missing local capability")
