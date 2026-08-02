"""Deterministic protected-material guards for prompt improvement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_chatbook.Prompt_Management.prompt_preservation import (
    preservation_violations,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "prompt_improvement_cases.json"
CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", CASES["preserves"], ids=lambda case: case["name"])
def test_fixture_corpus_preserves_protected_material(case: dict[str, str]) -> None:
    assert preservation_violations(case["source"], case["result"]) == ()


@pytest.mark.parametrize("case", CASES["vetoes"], ids=lambda case: case["name"])
def test_fixture_corpus_vetoes_changed_protected_material(
    case: dict[str, str],
) -> None:
    violations = preservation_violations(case["source"], case["result"])

    assert case["category"] in violations


@pytest.mark.parametrize("case", CASES["safe_changes"], ids=lambda case: case["name"])
def test_fixture_corpus_avoids_documented_false_positives(
    case: dict[str, str],
) -> None:
    assert preservation_violations(case["source"], case["result"]) == ()


def test_template_placeholder_cardinality_is_preserved_as_a_multiset() -> None:
    source = "{{user}} then {{user}} then {{$QUESTION}}"
    result = "{{user}} then {{$QUESTION}}"

    assert preservation_violations(source, result) == ("template_placeholder",)


@pytest.mark.parametrize(
    "result",
    [
        "~~~python\nprint(1)\n~~~",
        "````python\nprint(1)\n````",
        "```text\nprint(1)\n```",
        "```python\n print(1)\n```",
    ],
)
def test_fenced_code_preserves_delimiter_info_and_exact_body(result: str) -> None:
    source = "```python\nprint(1)\n```"

    assert preservation_violations(source, result) == ("fenced_code",)


def test_fenced_code_order_is_preserved() -> None:
    first = "```python\nprint(1)\n```"
    second = "~~~sql\nselect 1;\n~~~"

    assert preservation_violations(f"{first}\n{second}", f"{second}\n{first}") == (
        "fenced_code",
    )


@pytest.mark.parametrize(
    "result",
    [
        "[[TLDW_PROTECTED:0123456789abcdef0123:0:0123456789abcdef01234567]] twice "
        "[[TLDW_PROTECTED:0123456789abcdef0123:0:0123456789abcdef01234567]]",
        "[[TLDW_PROTECTED:0123456789abcdef0123:1:0123456789abcdef01234567]]",
        "[[TLDW_PROTECTED:0123456789abcdef0123:0:0123456789abcdef01234568]]",
        "",
    ],
)
def test_opaque_placeholder_requires_exact_single_ordered_occurrence(
    result: str,
) -> None:
    source = "[[TLDW_PROTECTED:0123456789abcdef0123:0:0123456789abcdef01234567]]"

    assert preservation_violations(source, result) == ("opaque_placeholder",)


def test_unbalanced_or_prose_angle_brackets_are_not_xml_wrappers() -> None:
    source = "Use <draft only and compare 2 < 3 > 1."
    result = "Compare 2 < 4 > 1 and use <another draft."

    assert preservation_violations(source, result) == ()
