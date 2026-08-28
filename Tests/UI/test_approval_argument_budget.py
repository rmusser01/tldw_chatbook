"""TASK-695: a tool call's target must survive argument truncation.

`_summarize_arguments` capped the WHOLE JSON blob at 80 characters, and
`json.dumps` preserves the model's key order. A `write_file` that emits
`content` before `file_path` therefore rendered 67 characters of file body
and truncated the destination out of view -- the card asked "may I write
this?" without showing where.

TASK-1846 gave arguments the full row width but not this: the cap is a text
budget, not a layout one, so the extra cells went unused while the path
stayed hidden.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
    _collapse_pending_calls,
    _summarize_arguments,
    _summarize_row_arguments,
)


@pytest.mark.unit
def test_the_destination_survives_a_huge_leading_content_argument():
    """The reported case: `write_file` with content serialised first."""
    rendered = _summarize_arguments(
        {"content": "x" * 400, "file_path": "~/notes/IMPORTANT.md"}
    )
    assert "IMPORTANT.md" in rendered, (
        "the write destination is truncated out of view; the card asks 'may "
        f"I write this?' without showing where: {rendered!r}"
    )


@pytest.mark.unit
def test_it_holds_through_the_row_renderer_too():
    """The production path is the row renderer, not the helper."""
    entry = _collapse_pending_calls(
        [
            {
                "llm_name": "write_file",
                "arguments": {"content": "y" * 500, "file_path": "/tmp/target.md"},
            }
        ]
    )[0]
    assert "target.md" in _summarize_row_arguments(entry)


@pytest.mark.unit
def test_no_single_argument_may_consume_the_whole_budget():
    """One huge value must not starve every other argument off the line."""
    rendered = _summarize_arguments(
        {"blob": "z" * 1000, "mode": "overwrite", "path": "/etc/hosts"}
    )
    assert "/etc/hosts" in rendered, rendered
    assert "overwrite" in rendered, rendered


@pytest.mark.unit
def test_short_arguments_are_still_rendered_verbatim():
    """The common case must not gain ellipses or reordering noise."""
    assert _summarize_arguments({"path": "~/a.md"}) == '{"path":"~/a.md"}'


@pytest.mark.unit
def test_raw_shell_dedicated_view_does_not_expand_the_generic_summary_budget():
    rendered = _summarize_arguments({"command": "x" * 500})

    assert len(rendered) <= 80
    assert "…" in rendered


@pytest.mark.unit
def test_secret_redaction_still_applies_after_reordering():
    """Redaction parity must survive the new ordering (TASK-1845)."""
    rendered = _summarize_arguments(
        {"api_key": "NOT-A-REAL-KEY-placeholder", "path": "~/a.md"}
    )
    assert "NOT-A-REAL-KEY-placeholder" not in rendered
    assert "***" in rendered


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    [
        "path",
        "file_path",
        "filePath",
        "dest",
        "destination",
        "src",
        "source",
        "output_dir",
        "url",
        "uri",
        "command",
        "target",
        "filename",
        "host",
    ],
)
def test_destination_like_keys_are_recognised(key: str):
    """Args:
    key: An argument name that names WHERE a call acts.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _is_destination_key,
    )

    assert _is_destination_key(key), key


@pytest.mark.unit
@pytest.mark.parametrize(
    "key", ["profile", "filter", "compiled", "urinal", "hostility"]
)
def test_ordinary_keys_are_not_mistaken_for_destinations(key: str):
    """Substring matching treats `profile` as a file and `urinal` as a URL.

    A false positive is only a reordering, not a leak -- but it pushes the
    REAL destination later in a line that is budget-limited, which is the
    defect this task exists to fix.

    Args:
        key: An argument name that merely contains a destination word.
    """
    from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import (
        _is_destination_key,
    )

    assert not _is_destination_key(key), key


@pytest.mark.unit
def test_a_source_path_is_not_reordered_behind_its_destination():
    """`src`/`dest` are both destinations; the call's own order is kept."""
    rendered = _summarize_arguments({"src": "/a/one.md", "dest": "/b/two.md"})
    assert rendered.index('"src"') < rendered.index('"dest"'), rendered


@pytest.mark.unit
def test_the_destination_survives_when_it_is_the_last_of_many_arguments():
    """Hoisting, not just per-value budgets, is what saves this case.

    With a handful of arguments the per-value budget alone keeps the
    destination on screen wherever it sits. With MANY, the shared budget
    shrinks but the total still reaches the line cap, and whatever sits last
    is clipped -- so a call whose path is emitted last loses it again.

    Mutation-checked: removing the hoisting leaves every other test in this
    file green and fails only this one.
    """
    arguments = {f"opt_{n}": f"value-{n}" * 3 for n in range(8)}
    arguments["file_path"] = "~/notes/IMPORTANT.md"

    rendered = _summarize_arguments(arguments)

    assert "IMPORTANT.md" in rendered, (
        f"the destination was clipped off the end of a long argument list: {rendered!r}"
    )


@pytest.mark.unit
def test_a_non_string_argument_key_never_crashes_rendering():
    """`_summarize_arguments` must survive any payload the model emits.

    The old implementation put everything inside one guarded `json.dumps`.
    Hoisting moved key inspection OUT of that guard, and `_snake_case`'s
    `re.sub` raises TypeError on a non-string key -- so a malformed payload
    took down the approval row instead of rendering it. An approval card
    that crashes is an approval the user cannot answer, and the run blocks
    until the auto-deny fires.
    """
    rendered = _summarize_arguments({1: "a", "path": "/x", None: "b"})

    assert "/x" in rendered, rendered
    assert isinstance(rendered, str)
