"""The guard that makes a never-called failure double a test failure.

task-15270 AC#3. `test_send_proceeds_when_auto_retrieve_fails` passed for two
months while its exploding backend was never called (task-15210); the shape
"X still works when Y fails" is only meaningful if Y was attempted. These
tests pin the guard itself -- including that the root conftest's autouse
fixture really does fail the test, not merely warn.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from _pytest.outcomes import Failed

import Tests.conftest as root_conftest
from Tests.fixtures.required_doubles import (
    exploding_double,
    is_exploding_double,
    must_be_called,
    reset_required_doubles,
    uncalled_required_doubles,
)


def _fixture_body(fixture):
    """The undecorated generator behind a `@pytest.fixture`.

    pytest 9 wraps it in a `FixtureFunctionDefinition` (and pytest 8 in a
    `__pytest_wrapped__` holder); `functools.update_wrapper` leaves
    ``__wrapped__`` pointing at the real function in both.
    """
    body = getattr(fixture, "__wrapped__", None)
    if body is None:
        body = getattr(fixture, "_fixture_function", fixture)
    return body


@pytest.fixture(autouse=True)
def _isolate_registry():
    """Keep these tests' own registrations out of the suite-wide guard."""
    reset_required_doubles()
    yield
    reset_required_doubles()


def test_registered_double_that_was_called_reports_nothing():
    double = must_be_called(Mock(), "the retry path must run")

    double()

    assert uncalled_required_doubles() == []


def test_registered_double_that_was_never_called_is_reported():
    must_be_called(Mock(), "the retry path must run")

    reported = uncalled_required_doubles()

    assert len(reported) == 1
    assert "the retry path must run" in reported[0]


@pytest.mark.asyncio
async def test_awaited_double_counts_as_called():
    """An `AsyncMock` records `await_count`, not always `call_count`."""
    double = exploding_double(RuntimeError("backend exploded"), reason="failure path")

    with pytest.raises(RuntimeError):
        await double()

    assert uncalled_required_doubles() == []


def test_exploding_double_registers_itself():
    """The "Y fails" double is guarded by construction, not by remembering."""
    exploding_double(RuntimeError("backend exploded"), reason="failure path")

    assert len(uncalled_required_doubles()) == 1


def test_only_raising_mocks_are_audit_candidates():
    """The audit's classifier: a raising mock, instance or class."""
    assert is_exploding_double(Mock(side_effect=RuntimeError("boom"))) is True
    assert is_exploding_double(AsyncMock(side_effect=RuntimeError)) is True
    assert is_exploding_double(Mock(return_value=3)) is False
    assert is_exploding_double(Mock(side_effect=[1, 2])) is False
    assert is_exploding_double(lambda: None) is False


def test_the_autouse_fixture_fails_the_test_not_just_warns(request):
    """Drive the real conftest fixture body: unmet registration -> Failed.

    Calling the fixture's own function is deliberate -- a guard pinned only
    through its helpers would itself be the vacuous shape this task exists to
    remove.
    """
    fixture_body = _fixture_body(root_conftest.required_doubles_are_called)

    generator = fixture_body(request)
    next(generator)
    must_be_called(Mock(), "the failure path must be attempted")

    with pytest.raises(Failed) as failure:
        next(generator)

    assert "the failure path must be attempted" in str(failure.value)


def test_the_autouse_fixture_passes_when_the_double_was_called(request):
    fixture_body = _fixture_body(root_conftest.required_doubles_are_called)

    generator = fixture_body(request)
    next(generator)
    double = must_be_called(Mock(), "the failure path must be attempted")
    double()

    with pytest.raises(StopIteration):
        next(generator)
