"""task-31800: the run-at example shown in the reminder/automation forms and
detail panes must always be a FUTURE datetime, never a hardcoded literal that
drifts into the past.

The three sites the UAT flagged (reminder_form placeholder + hint, and
definition_detail example + error copy) all render `example_run_at_text()`, so
pinning the helper pins every caller.
"""
from datetime import datetime, timedelta

from tldw_chatbook.Scheduling.schedule_input_parsing import (
    example_run_at_text,
    parse_forgiving_datetime,
)


def test_example_run_at_text_is_always_in_the_future():
    text = example_run_at_text()
    parsed, assumed_local = parse_forgiving_datetime(text)
    assert parsed is not None, f"the example {text!r} must be a parseable date-time"
    # Strictly future: a user who copies the example verbatim gets a run time
    # that has not already passed.
    assert parsed > datetime.now(parsed.tzinfo), (
        f"example {text!r} parsed to {parsed}, which is not in the future"
    )


def test_example_run_at_text_shape_and_days_ahead():
    # Shared default keeps every caller's shown example identical, and the
    # forgiving 'YYYY-MM-DD 09:00' shape parse_forgiving_datetime accepts.
    text = example_run_at_text(days_ahead=7)
    expected_date = (datetime.now() + timedelta(days=7)).date().isoformat()
    assert text == f"{expected_date} 09:00"


def test_no_literal_past_date_hardcoded_in_the_helper():
    # Guard against a regression back to a fixed literal (the 2026-08-28 09:00
    # the UAT found): the value must change as 'now' advances.
    far = example_run_at_text(days_ahead=3650)
    near = example_run_at_text(days_ahead=1)
    assert far != near
