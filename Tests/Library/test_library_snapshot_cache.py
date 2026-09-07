from __future__ import annotations

import pytest

from tldw_chatbook.UI.Library_Modules.library_snapshot_cache import (
    clone_library_source_snapshot,
)


def _snapshot():
    return (
        {
            "notes": ({"id": "n1", "meta": {"tags": ["a"]}},),
            "media": ({"id": "m1"},),
            "conversations": ({"id": "c1"},),
            "prompts": (2, ()),
            "skills": (
                1,
                {
                    "available_skills": [{"name": "alpha", "tags": ["safe"]}],
                    "blocked_skills": [],
                },
            ),
        },
        {"notes": 1, "media": 1, "conversations": 1},
        {"notes": True, "media": True, "conversations": True},
        None,
        None,
        {"study_decks": 1, "flashcards_due": 2, "quizzes": 3},
    )


class _DeepcopyBomb:
    def __deepcopy__(self, memo):
        raise RuntimeError("deepcopy must degrade to a cache miss")


def test_clone_accepts_real_prompt_and_skill_shapes_without_aliasing():
    original = _snapshot()
    cloned = clone_library_source_snapshot(original)
    assert cloned == original
    assert cloned is not original
    assert cloned[0] is not original[0]
    assert cloned[0]["notes"][0] is not original[0]["notes"][0]
    assert cloned[0]["skills"][1] is not original[0]["skills"][1]
    assert (
        cloned[0]["skills"][1]["available_skills"]
        is not original[0]["skills"][1]["available_skills"]
    )


@pytest.mark.parametrize(
    "malformed",
    [None, (), ({},), ({"notes": []}, {}, {}, None, None, {})],
)
def test_clone_rejects_malformed_outer_or_source_shapes(malformed):
    assert clone_library_source_snapshot(malformed) is None


@pytest.mark.parametrize(
    "skills_context",
    [
        {"available_skills": 7, "blocked_skills": []},
        {"available_skills": [{"name": "alpha"}, "not-a-record"], "blocked_skills": []},
        {"available_skills": [], "blocked_skills": [object()]},
    ],
    ids=["available-not-list", "available-mixed-items", "blocked-non-record"],
)
def test_clone_rejects_malformed_nested_skills_payloads(skills_context):
    malformed = _snapshot()
    malformed[0]["skills"] = (1, skills_context)

    assert clone_library_source_snapshot(malformed) is None


def test_clone_treats_deepcopy_failure_as_a_cache_miss():
    malformed = _snapshot()
    malformed[0]["skills"][1]["available_skills"][0]["opaque"] = _DeepcopyBomb()

    assert clone_library_source_snapshot(malformed) is None


def test_clone_prevents_second_clone_mutation_from_reaching_third_clone():
    original = _snapshot()
    second_clone = clone_library_source_snapshot(original)
    assert second_clone is not None
    second_clone[0]["notes"][0]["meta"]["tags"].append("mutated")
    second_clone[0]["skills"][1]["available_skills"][0]["tags"].append("unsafe")

    third_clone = clone_library_source_snapshot(original)
    assert third_clone == original
