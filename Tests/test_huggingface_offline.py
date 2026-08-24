"""No test downloads a model (TASK-21562).

The HuggingFace cache resolves *into* the per-test sandbox
(`.../home/.cache/huggingface/hub`), a directory that never exists, so any code
path that reaches the hub attempts a real download. That is close to invisible
on a developer machine and very visible in CI: one core shard alone recorded
188 egress-blocked errors against `huggingface.co:443` and a CDN address, each
one `huggingface_hub` retrying five times before the network guard's record
failed the test at teardown.

`huggingface_hub` freezes `constants.HF_HUB_OFFLINE` at its own import time, so
an env var written from a fixture arrives too late to be read. The bootstrap in
`Tests/conftest.py` therefore writes it before any import, and an autouse
fixture patches the constant for the case where the module got in first. These
tests pin both halves, because a latch that is only half closed reads exactly
like one that is closed.
"""

from __future__ import annotations

import os
import sys

import pytest

OPT_OUT = "TLDW_TEST_ALLOW_HF_DOWNLOADS"

#: Opting out is a legitimate choice, so these skip rather than fail -- a
#: developer who asked for live fetches should not get a red suite for it. They
#: skip rather than `return`, so the opt-out is visible in the run's summary
#: instead of looking like three tests that passed while asserting nothing.
pytestmark = pytest.mark.skipif(
    os.environ.get(OPT_OUT) == "1",
    reason=f"{OPT_OUT}=1: this session deliberately permits model downloads",
)


def test_the_environment_declares_offline_before_any_import() -> None:
    """The half that does the work: set pre-import, so the constant reads True."""
    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def test_huggingface_hub_actually_reports_itself_offline() -> None:
    """The property, not the mechanism.

    Imports the module directly rather than through `sys.modules`, because this
    test is the one place that should pay that cost: asserting the env var is
    set proves what we wrote, while `is_offline_mode()` proves what the library
    concluded -- and it is the library's conclusion that decides whether a
    request goes out.

    `is_offline_mode` is imported from the package root, which is where
    huggingface_hub 1.12 exports it -- not from `.utils`, where it used to live.
    Deliberately not wrapped in a try/except: if this import breaks on a future
    version, that should be a loud failure telling us to re-check how offline
    mode is decided, not a silently skipped assertion.
    """
    from huggingface_hub import constants, is_offline_mode

    assert constants.HF_HUB_OFFLINE is True
    assert is_offline_mode() is True


def test_the_autouse_fixture_covers_an_already_imported_module() -> None:
    """Once the module is in `sys.modules`, the fixture must have patched it.

    The previous test imports it, so by the time any later test runs the
    fixture's `sys.modules` lookup succeeds. Ordering makes this cheap: nothing
    here imports the hub stack on its own account.
    """
    constants = sys.modules.get("huggingface_hub.constants")
    assert constants is not None, (
        "the previous test imported huggingface_hub, so it must be in sys.modules "
        "-- if it is not, these tests were reordered and this one no longer "
        "checks the already-imported path it exists for"
    )
    assert constants.HF_HUB_OFFLINE is True
