"""Scratch plugin: apply the repo's own `bootstrap_profile` opt-in marker
(Tests/conftest.py, TASK-32873) to named nodes without editing the tests, so
they keep the sandboxed bootstrap profile (TLDW_TEST_CONFIG_ROOT, a scratch
dir) instead of tripping ADR-126 under the per-test redirect."""
import os

import pytest

TARGETS = set(
    os.environ.get(
        "BOOTSTRAP_TARGETS",
        "test_batch_row_widgets_have_nonzero_geometry_and_do_not_overlap_under_bundled_css",
    ).split(",")
)


def pytest_collection_modifyitems(items):
    for item in items:
        if item.originalname in TARGETS or item.name in TARGETS:
            item.add_marker(pytest.mark.bootstrap_profile)
