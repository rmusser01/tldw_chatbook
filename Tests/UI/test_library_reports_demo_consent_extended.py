"""120x36 arms of the Library Reports demo-consent pins (TASK-34000.23).

Same harness, helpers and assertions as `test_library_reports_demo_consent.py`;
kept out of the PR-gate lane so that file stays under its 35 s budget (the ten
tests together measured 35.3 s wall). AC#3 names 120x36 explicitly, so the
second size is pinned here rather than dropped.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_library_reports_demo_consent import (
    test_cta_sits_under_the_empty_state_sentence as _cta_under_sentence,
    test_pressing_the_cta_without_confirming_writes_nothing as _press_writes_nothing,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def test_pressing_the_cta_without_confirming_writes_nothing_120x36(
    tmp_path, monkeypatch
):
    await _press_writes_nothing(tmp_path, monkeypatch, (120, 36))


async def test_cta_sits_under_the_empty_state_sentence_120x36(tmp_path, monkeypatch):
    await _cta_under_sentence(tmp_path, monkeypatch, (120, 36))
