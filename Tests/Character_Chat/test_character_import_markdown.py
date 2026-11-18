"""Regression tests for Markdown-wrapped character card imports.

The second half pins TASK-38226. On dev (29592ecc9b) the parser found
frontmatter and json blocks with two backtracking regexes, cubic and
quadratic on crafted input; ``Character_Chat/card_markdown_scan.py`` replaces
them with linear scans that return what the regexes returned.
"""

from __future__ import annotations

import json
import random
import re
import time

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

from Tests.UI.roleplay_import_fixtures import write_import_fixtures
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    load_character_card_from_string_content,
)


def _v2_card(name: str = "Markdown Hero") -> dict:
    return {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": "Imported from Markdown.",
            "personality": "Careful and direct.",
            "scenario": "A Markdown import test.",
            "first_mes": "Hello from Markdown.",
            "mes_example": "User: hi\nCharacter: hello",
            "creator_notes": "Keep the wrapper format structured.",
            "system_prompt": "Stay in character.",
            "post_history_instructions": "Keep replies brief.",
            "alternate_greetings": ["Second hello."],
            "tags": ["markdown", "test"],
            "creator": "Test Suite",
            "character_version": "1.0",
        },
    }


def test_load_character_card_from_markdown_json_code_block() -> None:
    card = _v2_card("Fenced JSON Hero")
    markdown = f"""# Character Card

```json
{json.dumps(card)}
```
"""

    parsed = load_character_card_from_string_content(markdown)

    assert parsed is not None
    assert parsed["name"] == "Fenced JSON Hero"
    assert parsed["first_message"] == "Hello from Markdown."
    assert parsed["message_example"] == "User: hi\nCharacter: hello"


def test_load_character_card_from_markdown_yaml_frontmatter() -> None:
    markdown = """---
spec: chara_card_v2
spec_version: '2.0'
data:
  name: YAML Hero
  description: Imported from Markdown frontmatter.
  personality: Careful and direct.
  scenario: A Markdown import test.
  first_mes: Hello from YAML frontmatter.
  mes_example: "User: hi\\nCharacter: hello"
  creator_notes: Keep the wrapper format structured.
  system_prompt: Stay in character.
  post_history_instructions: Keep replies brief.
  alternate_greetings:
    - Second hello.
  tags:
    - markdown
    - yaml
  creator: Test Suite
  character_version: '1.0'
---

# YAML Hero
"""

    parsed = load_character_card_from_string_content(markdown)

    assert parsed is not None
    assert parsed["name"] == "YAML Hero"
    assert parsed["first_message"] == "Hello from YAML frontmatter."
    assert parsed["message_example"] == "User: hi\nCharacter: hello"


def test_load_character_card_from_invalid_markdown_returns_none() -> None:
    markdown = """# Not a Character Card

This prose does not contain YAML frontmatter or a fenced JSON character card.
"""

    assert load_character_card_from_string_content(markdown) is None


# --- TASK-38226: a linear scan that reads what dev's regexes read ----------

#: Dev's two regexes, verbatim from ``Character_Chat_Lib.py`` at 29592ecc9b
#: (lines 2613-2615 and 2644-2646): the oracle for the scan that replaced
#: them. They backtrack, so they only ever see short inputs here.
_DEV_FRONTMATTER = re.compile(r"^\s*---\s*\n(.*?)\n\s*---\s*", re.DOTALL)
_DEV_JSON_BLOCK = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE
)


def _dev_frontmatter_body(text: str) -> str | None:
    match = _DEV_FRONTMATTER.match(text)
    return match.group(1) if match else None


def _dev_fenced_json_object(text: str) -> str | None:
    match = _DEV_JSON_BLOCK.search(text)
    return match.group(1) if match else None


def _scan():
    """The linear scan, imported per test so the timing tests below still
    run on dev, where the module does not exist."""
    from tldw_chatbook.Character_Chat import card_markdown_scan

    return card_markdown_scan


#: What the two forms are made of, plus the whitespace the regexes' ``\s``
#: also matches (CR, VT, NEL, LINE SEPARATOR, NBSP) and ``ſ``, which
#: ``re.IGNORECASE`` lets match the ``s`` of the json tag.
_TOKENS = (
    "---", "-", "--", "\n", "\n", " ", "\t", "\r", "\r\n", "\x0b", "\x85",
    "\u2028", "\u00a0", "```", "`", "``", "json", "JSON", "jſon", "js",
    "{", "}", "x", "a: b", '"k": 1',
)  # fmt: skip
_FRONTMATTER_TOKENS = ("---",) * 4 + ("\n",) * 6 + _TOKENS
_FENCE_TOKENS = ("```",) * 4 + ("{", "}") * 3 + _TOKENS


def _token_texts(count: int, seed: int = 38226) -> list[str]:
    """Reproducible token strings: half weighted to frontmatter (and opening
    with ``---`` so ``re.match`` gets past its first step), half to fences."""
    rng = random.Random(seed)
    texts = []
    for index in range(count):
        tokens = _FRONTMATTER_TOKENS if index % 2 == 0 else _FENCE_TOKENS
        body = "".join(rng.choice(tokens) for _ in range(rng.randint(0, 30)))
        texts.append(("---" + body) if index % 4 == 0 else body)
    return texts


def _joined(tokens: tuple[str, ...], max_size: int) -> st.SearchStrategy[str]:
    return st.lists(st.sampled_from(tokens), max_size=max_size).map("".join)


_whitespace = _joined(tuple(token for token in _TOKENS if token.isspace()), 4)
_noise = _joined(_TOKENS, 12)
#: A frontmatter or a json block, with whitespace and noise in every slot the
#: regexes read, so each example tests the step the scan could get wrong.
_frontmatter_shape = st.tuples(
    _whitespace, _whitespace, _noise, _whitespace, _noise
).map(lambda part: f"{part[0]}---{part[1]}\n{part[2]}\n{part[3]}---{part[4]}")
_fence_shape = st.tuples(
    _noise, st.sampled_from(("", "json", "JSON", "jſon", "js")), _whitespace,
    _noise, _whitespace, _noise,
).map(lambda part: f"{part[0]}```{part[1]}{part[2]}{{{part[3]}}}{part[4]}```{part[5]}")  # fmt: skip


@given(
    st.lists(
        st.one_of(_frontmatter_shape, _fence_shape, _noise), min_size=1, max_size=4
    ).map("".join)
)
def test_the_scan_reads_what_devs_regexes_read(text):
    scan = _scan()
    assert scan.frontmatter_body(text) == _dev_frontmatter_body(text)
    assert scan.fenced_json_object(text) == _dev_fenced_json_object(text)


def test_the_scan_reads_what_devs_regexes_read_on_a_seeded_sweep():
    """40,000 reproducible strings, so a run at the dev Hypothesis scale
    still compares thousands of found bodies and blocks with dev's."""
    scan = _scan()
    mismatches = []
    bodies = blocks = 0
    for text in _token_texts(40_000):
        expected = (_dev_frontmatter_body(text), _dev_fenced_json_object(text))
        actual = (scan.frontmatter_body(text), scan.fenced_json_object(text))
        bodies += expected[0] is not None
        blocks += expected[1] is not None
        if actual != expected:
            mismatches.append((text, expected, actual))
    assert mismatches[:5] == []
    # Not vacuous: the sweep finds both forms often (1,822 and 1,114).
    assert bodies >= 1_000 and blocks >= 500, (bodies, blocks)


def _normal_markdown_cards() -> list[tuple[str, str]]:
    """Markdown cards in the shapes people write, plus copies of the
    Markdown card texts the suite already imports (source named)."""
    v2 = _v2_card("Corpus Hero")
    v2["data"]["description"] = "Uses `code`, {braces} and ``` fences."
    v2["data"]["extensions"] = {"depth": {"inner": {"n": [1, {"k": "}"}]}}}
    v2_other = _v2_card("Second Card")
    v1 = {
        "name": "Flat Hero",
        "description": "A V1 card.\nTwo lines.",
        "personality": "Plain.",
        "scenario": "Flat.",
        "first_mes": "Hi.",
        "mes_example": "User: hi",
    }
    v3 = {
        "spec": "chara_card_v3",
        "spec_version": "3.0",
        "data": {**_v2_card("Third Spec")["data"], "nickname": "Tee"},
    }
    unicode_card = _v2_card("Ünïcode Hérø ✨")
    unicode_card["data"]["first_mes"] = "こんにちは — “quoted”"

    jsons = []
    for card in (v2, v1, v3, unicode_card):
        jsons += [
            json.dumps(card),
            json.dumps(card, indent=2),
            json.dumps(card, ensure_ascii=False, indent=4),
        ]
    other = json.dumps(v2_other)
    cards: list[tuple[str, str]] = []
    for index, raw in enumerate(jsons):
        cards += [
            (f"fence-json-{index}", f"```json\n{raw}\n```"),
            (
                f"fence-in-prose-{index}",
                f"# Card\n\nSome prose.\n\n```json\n{raw}\n```\n\nMore prose.\n",
            ),
            (f"fence-bare-{index}", f"```\n{raw}\n```\n"),
            (f"fence-upper-{index}", f"```JSON\n{raw}\n```"),
            (f"fence-inline-{index}", f"```json {raw} ```"),
            (
                f"fence-after-code-{index}",
                f"```python\nprint('x')\n```\n\n```json\n{raw}\n```",
            ),
            (f"fence-two-cards-{index}", f"```json\n{raw}\n```\n```json\n{other}\n```"),
            (f"fence-bom-{index}", f"\ufeff  \n```json\n{raw}\n```"),
            (
                f"fence-braces-before-{index}",
                f"Intro {{not a card}}\n```json\n{raw}\n```",
            ),
        ]
    for index, card in enumerate((v2, v1, v3, unicode_card)):
        dumped = yaml.safe_dump(card, sort_keys=False, allow_unicode=True)
        cards += [
            (f"yaml-{index}", f"---\n{dumped}---\n\n# Body\n"),
            (f"yaml-no-body-{index}", f"---\n{dumped}---\n"),
            (f"yaml-no-newline-{index}", f"---\n{dumped}---"),
            (f"yaml-trailing-spaces-{index}", f"--- \n{dumped}---  \n"),
            (f"yaml-leading-blank-{index}", f"\n\n---\n{dumped}---\n"),
            (f"yaml-inner-blank-{index}", f"---\n\n{dumped}\n---\n"),
            (
                f"yaml-then-fence-{index}",
                f"---\n{dumped}---\n```json\n{other}\n```",
            ),
        ]
    cards += [
        ("yaml-error-then-fence", f"---\nnot: [valid\n---\n```json\n{jsons[0]}\n```"),
        ("yaml-list-then-fence", f"---\n- a list\n---\n```json\n{jsons[0]}\n```"),
        ("yaml-unclosed", "---\nname: Open\ndescription: Never closed.\n"),
        ("fence-unclosed", f"```json\n{jsons[0]}"),
        # A closing fence inside a JSON string ends the block there, as on dev.
        ("fence-closed-inside-string", '```json\n{"name": "Cut", "a": "}\n```"}\n```'),
        ("yaml-empty", "---\n---\n```json\n" + jsons[0] + "\n```"),
        # This file's own two cards.
        (
            "this-file-fenced",
            f"# Character Card\n\n```json\n{json.dumps(_v2_card('Fenced JSON Hero'))}\n```\n",
        ),
        # Tests/UI/test_roleplay_import_sniff.py and test_roleplay_import_app.py.
        ("sniff-bracket", '[Character card]\n\n```json\n{"name": "Tam Reyes"}\n```\n'),
        ("sniff-both", "---\nname: Both\n---\nBP: blood pressure\n---@@@---\nHR: heart rate\n"),
        ("sniff-frontmatter", "---\nname: Tam Reyes\ndescription: A pilot.\n---\n"),
        ("sniff-json-block", '# Card\n\n```json\n{"name": "Tam Reyes"}\n```\n'),
        ("sniff-bare-block", '# Card\n\n```\n{"name": "Tam Reyes"}\n```\n'),
        ("sniff-notes-bare", 'Notes: a pilot.\n\n```\n{"name": "Tam Reyes"}\n```\n'),
        ("app-field-notes", "---\nname: Field Notes\n---\nBP: blood pressure\n---@@@---\nHR: heart rate\n"),
        ("app-both", "---\nname: Both\n---\nBP: blood pressure\n---@@@---\n"),
    ]  # fmt: skip
    return cards + [
        (f"{label}-crlf", text.replace("\n", "\r\n")) for label, text in cards
    ]


_NORMAL_CARDS = _normal_markdown_cards()


def _run_dev_extraction(monkeypatch) -> list[str]:
    """Make the parser extract with dev's regexes; return the calls made."""
    scan = _scan()
    calls: list[str] = []

    def frontmatter(text):
        calls.append("frontmatter")
        return _dev_frontmatter_body(text)

    def json_block(text):
        calls.append("json block")
        return _dev_fenced_json_object(text)

    monkeypatch.setattr(scan, "frontmatter_body", frontmatter)
    monkeypatch.setattr(scan, "fenced_json_object", json_block)
    return calls


def test_every_normal_markdown_card_parses_exactly_as_on_dev(monkeypatch):
    """The whole parser's output, with the scan and with dev's regexes."""
    scanned = {
        label: load_character_card_from_string_content(text)
        for label, text in _NORMAL_CARDS
    }
    calls = _run_dev_extraction(monkeypatch)
    with_dev_regexes = {
        label: load_character_card_from_string_content(text)
        for label, text in _NORMAL_CARDS
    }
    assert len(calls) >= len(_NORMAL_CARDS), "dev's regexes did not run"
    differing = [
        label for label in scanned if scanned[label] != with_dev_regexes[label]
    ]
    assert differing == []


def test_the_normal_card_corpus_holds_real_cards():
    """Not vacuous: most of the corpus imports as a card, in both forms."""
    parsed = {
        label: load_character_card_from_string_content(text)
        for label, text in _NORMAL_CARDS
    }
    cards = [label for label, card in parsed.items() if card is not None]
    assert len(cards) >= 200, len(cards)
    assert any(label.startswith("yaml-") for label in cards)
    assert any(label.startswith("fence-") for label in cards)
    assert parsed["yaml-error-then-fence"]["name"] == "Corpus Hero"


def test_every_text_card_fixture_the_suite_imports_parses_as_on_dev(
    tmp_path, monkeypatch
):
    """Roleplay's import fixtures: every Markdown and text card file."""
    paths = write_import_fixtures(tmp_path)
    texts = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
        if path.suffix in {".md", ".markdown", ".yaml", ".txt"}
    }
    scanned = {
        name: load_character_card_from_string_content(text)
        for name, text in texts.items()
    }
    calls = _run_dev_extraction(monkeypatch)
    with_dev_regexes = {
        name: load_character_card_from_string_content(text)
        for name, text in texts.items()
    }
    assert calls
    assert sum(card is not None for card in scanned.values()) >= 3, scanned
    assert scanned == with_dev_regexes


#: Crafted inputs, 2 to 48 KB, that took seconds in dev's parser (29592ecc9b,
#: this test run there, 2026-11-05).
_DEV_SLOW_INPUTS = {
    # A frontmatter opening, then 2,000 newlines: 2.7 s (cubic).
    "frontmatter-newline-run": "---\n" + "\n" * 2_000,
    # One line, then 48,000 newlines and no closing ---: 2.8 s (quadratic).
    "frontmatter-unclosed-run": "---\nx" + "\n" * 48_000,
    # 12,000 fences, each opening a { that never closes: 2.6 s (quadratic).
    "repeated-open-fences": "```{" * 12_000,
}


@pytest.mark.parametrize("text", _DEV_SLOW_INPUTS.values(), ids=_DEV_SLOW_INPUTS)
def test_a_crafted_markdown_card_parses_well_under_a_fixed_bound(text):
    """Red on dev, where each input took 2.6-2.8 s. The scan takes well
    under a millisecond, so the half-second bound leaves CI more than
    100x room."""
    started = time.perf_counter()
    assert load_character_card_from_string_content(text) is None
    assert time.perf_counter() - started < 0.5


def _adversarial(family: str, size: int) -> str:
    """One crafted input of ``size`` characters, per scan step's worst case."""
    if family == "frontmatter-newline-run":  # the opening's whitespace run
        return "---\n" + "\n" * (size - 4)
    if family == "frontmatter-unclosed-run":  # the closing search's runs
        return "---\nx" + "\n" * (size - 5)
    if family == "frontmatter-dashes":  # a closing-search start per char
        return "---\n" + "-" * (size - 4)
    if family == "repeated-open-fences":  # an opening per fence, no close
        return "```{" * (size // 4)
    if family == "unclosed-braces":  # a closing-search start per char
        return "```{" + "}" * (size - 4)
    if family == "backtick-run":  # an opening-search start per char
        return "`" * size
    raise AssertionError(family)


def _parse_cpu_seconds(text: str) -> float:
    """The parser's CPU time on this thread, the least of three runs."""
    best = float("inf")
    for _ in range(3):
        started = time.thread_time()
        load_character_card_from_string_content(text)
        best = min(best, time.thread_time() - started)
    return best


@pytest.mark.parametrize(
    "family",
    [
        "frontmatter-newline-run",
        "frontmatter-unclosed-run",
        "frontmatter-dashes",
        "repeated-open-fences",
        "unclosed-braces",
        "backtick-run",
    ],
)
def test_a_10_mb_crafted_card_parses_in_linear_time(family):
    """AC#1: four times the input costs about four times the work, not
    sixteen. The ratio of the parser's own thread CPU time is independent
    of the machine's speed and of other processes; linear measures about 4
    here, and the bound of 10 leaves room for cache effects. Dev's parser
    does not finish these sizes in any useful time (2.7 s at 2 KB, cubic):
    test_a_crafted_markdown_card_parses_well_under_a_fixed_bound is the red
    test there."""
    small = _parse_cpu_seconds(_adversarial(family, 2_500_000))
    large = _parse_cpu_seconds(_adversarial(family, 10_000_000))
    # A 2 ms floor keeps timer noise on the fastest families out of the
    # ratio; quadratic work at 10 MB would take hours, far past any floor.
    assert large / max(small, 0.002) < 10, (family, small, large)
