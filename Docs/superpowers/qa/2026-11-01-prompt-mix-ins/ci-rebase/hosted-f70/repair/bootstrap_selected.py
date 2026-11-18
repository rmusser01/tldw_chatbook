"""Keep the selected legacy app_factory controls on their import-time profile."""

import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        item.add_marker(pytest.mark.bootstrap_profile)
