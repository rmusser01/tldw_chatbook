"""Run the unchanged durable marker resume control with current source ownership."""

import pytest

from Tests.Chat.test_change_turn_tracking import (
    root as root,  # noqa: PLC0414 - explicit compatibility or pytest-fixture re-export.
)
from Tests.Chat.test_change_turn_tracking import (
    tracker as tracker,  # noqa: PLC0414 - explicit compatibility or pytest-fixture re-export.
)

pytestmark = pytest.mark.bootstrap_profile


def test_original_marker_resume_body_with_current_bootstrap(tmp_path, root, tracker):
    from Tests.Chat.test_change_turn_tracking import (
        test_resume_re_derives_the_summary_row_byte_identical,
    )

    test_resume_re_derives_the_summary_row_byte_identical(tmp_path, root, tracker)
