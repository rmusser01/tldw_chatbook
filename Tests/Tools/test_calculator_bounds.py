"""The always-on calculator must not evaluate whatever size it is handed.

TASK-32806.4. `CalculatorTool` is a built-in, it is always on, it inherits
allow, and it evaluates model-supplied arithmetic -- so a prompt injection
is enough to reach it. Its allowed operators included `**` and `*`, and its
allowed constants included `str`, with no bound on either. Measured before
the fix:

    'ab' * 10**7   ->  20 MB string in 1 ms
    7 ** (10**6)   ->  0.37 MB integer in 0.1 s, scaling superlinearly

The enclosing timeout does not help: it abandons the worker thread rather
than stopping it, so a big exponent pins a core for the life of the process.

The bounds are checked against the OPERANDS, so an oversized expression is
refused without ever building the result. These tests assert that refusal is
immediate, not merely eventual.
"""

from __future__ import annotations

import time

import pytest

from tldw_chatbook.Tools.tool_executor import CalculatorTool

# The bounds are imported inside the two tests that need them, so the
# behaviour tests still COLLECT against a tree where they do not exist yet.
# A test file that cannot be collected against the unfixed code proves
# nothing about the fix.


#: Refusal has to be cheap or it is not a fix. The unbounded versions of
#: these took 1 ms to 1.3 s and allocated tens of megabytes; a predicted
#: rejection is microseconds, so this ceiling is loose by three orders of
#: magnitude and still fails if the result is being computed.
IMMEDIATE_SECONDS = 0.25


@pytest.fixture()
def calculator() -> CalculatorTool:
    return CalculatorTool()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4),
        ("2 ** 10", 1024),
        ("abs(-5) * 3", 15),
        ("10 / 4", 2.5),
        ("'ab' * 3", "ababab"),
        ("max(2, 7) ** 2", 49),
    ],
)
async def test_ordinary_arithmetic_still_works(calculator, expression, expected):
    result = await calculator.execute(expression=expression)
    assert result["result"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression",
    [
        "'ab' * 10**7",
        "'x' * 100000",
        "20000 * 'y'",  # operand order must not matter
    ],
)
async def test_oversized_string_repetition_is_refused(calculator, expression):
    started = time.perf_counter()
    with pytest.raises(ValueError):
        await calculator.execute(expression=expression)
    assert time.perf_counter() - started < IMMEDIATE_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "expression",
    [
        "7 ** (10**6)",
        "2 ** 100000",
        "(10**20000) * (10**20000)",
    ],
)
async def test_oversized_exponentiation_is_refused(calculator, expression):
    started = time.perf_counter()
    with pytest.raises(ValueError):
        await calculator.execute(expression=expression)
    assert time.perf_counter() - started < IMMEDIATE_SECONDS


@pytest.mark.asyncio
async def test_the_refusal_says_what_was_wrong(calculator):
    """A refusal the model cannot read is a refusal it will retry blindly."""
    from tldw_chatbook.Tools.tool_executor import (
        MAX_EXPONENT,
        MAX_STRING_RESULT_LENGTH,
    )

    with pytest.raises(ValueError, match="exponent"):
        await calculator.execute(expression=f"2 ** {MAX_EXPONENT + 1}")
    with pytest.raises(ValueError, match="characters"):
        await calculator.execute(expression=f"'ab' * {MAX_STRING_RESULT_LENGTH}")


@pytest.mark.asyncio
async def test_a_huge_expression_is_refused_before_parsing(calculator):
    from tldw_chatbook.Tools.tool_executor import MAX_EXPRESSION_LENGTH

    with pytest.raises(ValueError, match="characters"):
        await calculator.execute(expression="1+" * MAX_EXPRESSION_LENGTH + "1")


@pytest.mark.asyncio
async def test_the_bounds_are_generous_enough_to_be_useful(calculator):
    """Just under each cap must still evaluate, or the fix broke the tool."""
    from tldw_chatbook.Tools.tool_executor import MAX_RESULT_BITS

    result = await calculator.execute(expression="2 ** 4096")
    assert result["result"].bit_length() <= MAX_RESULT_BITS

    result = await calculator.execute(expression="'ab' * 100")
    assert len(result["result"]) == 200
