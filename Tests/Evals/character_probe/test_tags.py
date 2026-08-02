import pytest

from tldw_chatbook.Evals.character_probe.tags import (
    BUILTIN_TAGS,
    TAG_KINDS,
    Tag,
)


def test_the_three_kinds_are_exactly_the_specs_three():
    assert TAG_KINDS == ("failure", "notable", "positive")


def test_the_builtin_vocabulary_is_the_ten_the_spec_names():
    assert {t.slug for t in BUILTIN_TAGS} == {
        "broke-character",
        "refused",
        "leaked-prompt",
        "generic-assistant-voice",
        "contradicted-card",
        "ignored-the-question",
        "notable",
        "surprising",
        "in-character",
        "handled-well",
    }


def test_each_builtin_carries_the_kind_the_spec_assigns_it():
    by_slug = {t.slug: t.kind for t in BUILTIN_TAGS}
    assert by_slug["broke-character"] == "failure"
    assert by_slug["refused"] == "failure"
    assert by_slug["leaked-prompt"] == "failure"
    assert by_slug["generic-assistant-voice"] == "failure"
    assert by_slug["contradicted-card"] == "failure"
    assert by_slug["ignored-the-question"] == "failure"
    assert by_slug["notable"] == "notable"
    assert by_slug["surprising"] == "notable"
    assert by_slug["in-character"] == "positive"
    assert by_slug["handled-well"] == "positive"


def test_builtin_slugs_are_unique():
    slugs = [t.slug for t in BUILTIN_TAGS]
    assert len(slugs) == len(set(slugs))


def test_a_tag_without_a_valid_kind_is_rejected_naming_the_kind():
    with pytest.raises(ValueError) as exc:
        Tag(slug="whatever", label="Whatever", kind="bad")
    assert "bad" in str(exc.value)
    assert "failure" in str(exc.value)


def test_a_tag_with_an_empty_kind_is_rejected_rather_than_defaulted():
    """The spec: a guessed kind mis-groups observations; `notable` is not safe."""
    with pytest.raises(ValueError):
        Tag(slug="whatever", label="Whatever", kind="")


def test_a_tag_with_a_non_canonical_slug_is_rejected_naming_the_slug():
    with pytest.raises(ValueError) as exc:
        Tag(slug="Broke Character", label="Broke character", kind="failure")
    assert "Broke Character" in str(exc.value)


def test_a_tag_with_an_empty_label_is_rejected():
    with pytest.raises(ValueError):
        Tag(slug="broke-character", label="", kind="failure")


def test_a_tag_is_frozen():
    tag = BUILTIN_TAGS[0]
    with pytest.raises(Exception):
        tag.slug = "mutated"
