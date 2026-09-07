import pytest

from tldw_chatbook.Evals.character_probe.tags import (
    BUILTIN_TAGS,
    TAG_KINDS,
    Tag,
    canonical_slug,
    resolve_vocabulary,
    tag_by_slug,
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


def test_canonical_slug_lowercases_and_hyphenates():
    assert canonical_slug("Broke Character") == "broke-character"
    assert canonical_slug("  Out Of Character  ") == "out-of-character"
    assert canonical_slug("OOC") == "ooc"


def test_canonical_slug_collapses_runs_and_strips_punctuation():
    assert canonical_slug("broke   character!!") == "broke-character"
    assert canonical_slug("re-broke  --  character") == "re-broke-character"


def test_canonical_slug_rejects_text_with_no_usable_characters():
    with pytest.raises(ValueError):
        canonical_slug("   !!!   ")


def test_resolve_vocabulary_with_no_extras_is_exactly_the_builtins():
    assert resolve_vocabulary(()) == BUILTIN_TAGS


def test_resolve_vocabulary_appends_an_extra_tag():
    vocab = resolve_vocabulary(
        [{"slug": "meta-commentary", "label": "Meta commentary", "kind": "failure"}]
    )
    assert len(vocab) == len(BUILTIN_TAGS) + 1
    assert vocab[-1] == Tag("meta-commentary", "Meta commentary", "failure")


def test_an_extra_tag_may_relabel_a_builtin_in_place():
    vocab = resolve_vocabulary(
        [{"slug": "notable", "label": "Worth a second look", "kind": "notable"}]
    )
    assert len(vocab) == len(BUILTIN_TAGS)
    assert tag_by_slug(vocab, "notable").label == "Worth a second look"


def test_an_extra_tag_may_not_change_a_builtins_kind():
    with pytest.raises(ValueError) as exc:
        resolve_vocabulary(
            [{"slug": "refused", "label": "Refused", "kind": "positive"}]
        )
    assert "refused" in str(exc.value)


def test_an_extra_tag_without_a_kind_is_rejected_naming_the_slug():
    with pytest.raises(ValueError) as exc:
        resolve_vocabulary([{"slug": "meta-commentary", "label": "Meta"}])
    assert "meta-commentary" in str(exc.value)


def test_an_extra_tags_slug_is_canonicalised_rather_than_rejected():
    """A bench author types a label; the stored slug is canonical."""
    vocab = resolve_vocabulary(
        [{"slug": "Meta Commentary", "label": "Meta commentary", "kind": "notable"}]
    )
    assert tag_by_slug(vocab, "meta-commentary").label == "Meta commentary"


def test_an_extra_tag_missing_a_label_falls_back_to_its_slug():
    vocab = resolve_vocabulary([{"slug": "meta-commentary", "kind": "notable"}])
    assert tag_by_slug(vocab, "meta-commentary").label == "meta-commentary"


def test_resolve_vocabulary_accepts_tag_objects_as_well_as_mappings():
    vocab = resolve_vocabulary([Tag("meta-commentary", "Meta", "notable")])
    assert tag_by_slug(vocab, "meta-commentary").kind == "notable"


def test_two_extras_with_the_same_slug_keep_the_last():
    vocab = resolve_vocabulary([
        {"slug": "meta", "label": "First", "kind": "notable"},
        {"slug": "meta", "label": "Second", "kind": "notable"},
    ])
    assert tag_by_slug(vocab, "meta").label == "Second"


def test_tag_by_slug_raises_naming_the_slug_and_the_vocabulary():
    with pytest.raises(KeyError) as exc:
        tag_by_slug(BUILTIN_TAGS, "no-such-tag")
    assert "no-such-tag" in str(exc.value)
    assert "broke-character" in str(exc.value)
