"""Validated detached snapshots used to seed a new Library entry."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

LibrarySourceSnapshot = tuple[
    dict[str, tuple[Any, ...]],
    dict[str, int],
    dict[str, bool],
    str | None,
    Any,
    dict[str, int | None],
]

_RECORD_SOURCES = ("notes", "media", "conversations")
_SKILL_RECORD_LISTS = ("available_skills", "blocked_skills")


def _is_valid_library_source_snapshot(snapshot: object) -> bool:
    if not isinstance(snapshot, tuple) or len(snapshot) != 6:
        return False
    records, counts, total_known, lookup_error, _recovery_state, study_counts = (
        snapshot
    )
    if not all(
        isinstance(value, Mapping)
        for value in (records, counts, total_known, study_counts)
    ):
        return False
    if any(not isinstance(records.get(source), tuple) for source in _RECORD_SOURCES):
        return False
    if any(
        not isinstance(record, Mapping)
        for source in _RECORD_SOURCES
        for record in records[source]
    ):
        return False
    if any(not isinstance(counts.get(source), int) for source in _RECORD_SOURCES):
        return False
    if any(
        not isinstance(total_known.get(source), bool) for source in _RECORD_SOURCES
    ):
        return False
    if any(
        study_counts.get(key) is not None
        and not isinstance(study_counts.get(key), int)
        for key in ("study_decks", "flashcards_due", "quizzes")
    ):
        return False
    if lookup_error is not None and not isinstance(lookup_error, str):
        return False
    prompts = records.get("prompts")
    skills = records.get("skills")
    if not isinstance(prompts, tuple) or len(prompts) != 2 or not isinstance(
        prompts[1], tuple
    ):
        return False
    if not isinstance(skills, tuple) or len(skills) != 2 or not isinstance(
        skills[1], Mapping
    ):
        return False
    if prompts[0] is not None and not isinstance(prompts[0], int):
        return False
    if skills[0] is not None and not isinstance(skills[0], int):
        return False
    for key in _SKILL_RECORD_LISTS:
        skill_records = skills[1].get(key, ())
        if not isinstance(skill_records, (list, tuple)):
            return False
        if any(not isinstance(record, Mapping) for record in skill_records):
            return False
    return True


def clone_library_source_snapshot(snapshot: object) -> LibrarySourceSnapshot | None:
    if not _is_valid_library_source_snapshot(snapshot):
        return None
    try:
        cloned = copy.deepcopy(snapshot)
    except Exception:
        return None
    if not _is_valid_library_source_snapshot(cloned):
        return None
    return (
        dict(cloned[0]),
        dict(cloned[1]),
        dict(cloned[2]),
        cloned[3],
        cloned[4],
        dict(cloned[5]),
    )
