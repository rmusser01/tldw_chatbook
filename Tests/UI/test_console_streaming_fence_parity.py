"""Incremental streaming fence parity matches the one-shot whole-body scan."""

import pytest

from tldw_chatbook.Widgets.Console.console_transcript import (
    _StreamingFenceParity,
)
from tldw_chatbook.Widgets.Console.console_transcript import (
    _console_markdown_body_ends_in_open_fence as one_shot,
)

pytestmark = pytest.mark.bootstrap_profile

CHUNK_STREAMS = [
    ["```pyth", "on\nprint(", "'hi')\n", "``", "`\nDone ", "text"],
    ["text\n", "```\ncode\n", "```\n", "after"],
    ["~~~\n", "fence\n", "~~~~\n", "outside ", "`` `` ` bad"],
    ["``info with `tick``\n", "x\n", "```\n", "```\n"],
    ["a", "b", "c"],
    ["", "```\n"],
    ["```", "\n", "code\n", "```\ntail"],
]


@pytest.mark.parametrize("chunks", CHUNK_STREAMS)
def test_incremental_verdict_matches_one_shot_at_every_step(chunks):
    parity = _StreamingFenceParity()
    body = ""
    for chunk in chunks:
        body += chunk
        assert parity.ends_in_open_fence(body) == one_shot(body)


def test_complete_lines_fold_once_not_per_tick():
    parity = _StreamingFenceParity()
    parity.ends_in_open_fence("```python\nprint(1)\n")
    assert parity.scanned_upto == len("```python\nprint(1)\n")

    parity.ends_in_open_fence("```python\nprint(1)\nprint(2)\n")
    assert parity.scanned_upto == len("```python\nprint(1)\nprint(2)\n")

    # Repolling without growth never rescan-commits the partial tail.
    body = "```python\nprint(1)\nprint(2)\nprin"
    parity.ends_in_open_fence(body)
    frozen = parity.scanned_upto
    for _ in range(3):
        parity.ends_in_open_fence(body)
    assert parity.scanned_upto == frozen


def test_shrink_rescans_from_scratch():
    parity = _StreamingFenceParity()
    assert parity.ends_in_open_fence("```\ncode") is True
    assert parity.ends_in_open_fence("plain") is False  # shorter: reset path
    assert parity.scanned_upto == 0


def test_partial_tail_is_rejudged_as_it_grows():
    parity = _StreamingFenceParity()
    body = "```python\ncode\n``"
    assert parity.ends_in_open_fence(body) == one_shot(body)
    grown = body + "python extra\n"
    assert parity.ends_in_open_fence(grown) == one_shot(grown)
