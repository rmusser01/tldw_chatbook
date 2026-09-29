"""Scratch plugin: import tldw_chatbook.app at COLLECTION time, after
Tests/conftest.py has installed its bootstrap sandbox env, so the first
import's module-level config reads do not land inside a per-test redirect
(ADR-126 RecoveryRequired). Same timing fix as lessons-testing-evidence.md
("Tests/UI RecoveryRequired at setup is a profile-selection trip") and the
root conftest's eager tldw_chatbook.Chunking import; no test is edited and
no isolation is disabled."""
import os


def pytest_collection_modifyitems(session, config, items):
    home = os.environ.get("HOME", "")
    assert "tldw_test_config_" in home or "private-profile" in home or os.environ.get("TLDW_TEST_CONFIG_ROOT", "") in home, home
    import tldw_chatbook.app  # noqa: F401
